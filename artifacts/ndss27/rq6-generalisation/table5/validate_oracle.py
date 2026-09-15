#!/usr/bin/env python3
# ruff: noqa: W505, E501, B905
#   Style only, and suppressed rather than rewritten: this harness is
#   vendored from the campaign that produced the published numbers, and
#   e.g. adding zip(strict=True) would change behaviour where the
#   original silently truncated.  Correctness is gated by
#   validate_oracle.py, not by restyling proven code.
# Vendored verbatim from the soundness-campaign harness; see README.md.
# Experiment script, not library code: see artifacts/ndss27/README.md,
# "Lint and type checking", for why annotations are not required here.
# mypy: disable-error-code="no-untyped-def, no-untyped-call, type-arg, no-any-return, var-annotated, assignment, index, arg-type, union-attr, operator, attr-defined, misc, call-overload, return-value, unreachable"
"""Prove the campaign oracle can still FAIL, before trusting a zero from it.

Four properties, checked against the REAL corpus each ISA runs.  Exits non-zero
if any fails.

  NON-VACUITY   the ground truth must actually find taint.  A GT that is
                identically zero makes `GT & ~MT` unconditionally zero, so no
                under-taint can ever be reported and the campaign scores a
                perfect zero while measuring nothing.  This is not theoretical:
                19,085,696 MIPS cases were scored that way on 2026-09-12,
                because the completion check read MIPS's PC (which Unicorn never
                updates after emu_start), every run raised, the parent swallowed
                the raise, and the empty accumulator came back marked EXACT.

  MUTATION      drop ONE bit the ground truth says is tainted out of the
                engine's answer; the campaign's own test (`gt & ~mt`) must flag
                it.  If it cannot, that ISA's zero means nothing.

  INVALIDATION  a case whose runs cannot complete must come back exact=False,
                which run_campaign counts as `skipped`, NOT as a checked case
                with an empty ground truth.

  NO-PHANTOM    `shrd rbp, rsp` must not report taint in bits the instruction
                cannot depend on.  RBP is both source and destination and is
                untracked, so a leaking oracle reads polarity i-1's result back
                in polarity i; that single form produced 2,810 of the 2,875
                "under-taints" in the 12h campaign of 2026-07-24.

Run:  python validate_oracle.py [--cases N] [--json PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('MICROTAINT_TAINT_IR', '0')
os.environ.setdefault('MICROTAINT_BLOCK', '0')

from corpora import CORPORA
from mt_multiarch import GTSim, _build_isas, mt_batch
from oracle import HardenedGTSim


def popcount(x):
    return bin(x).count('1')


def _forms(isa, key, limit):
    out = []
    for label, asm, srcs, cons in CORPORA[key](isa):
        try:
            code = isa.assemble(asm)
        except Exception:
            try:
                code = bytes.fromhex(asm)      # riscv stores pre-encoded bytes
            except Exception:
                continue
        out.append((label, code, srcs, cons or {}))
        if len(out) >= limit:
            break
    return out


def _state(isa, rng):
    st = {n: rng.getrandbits(isa.bits) for n, _ in isa.gprs}
    for f in isa.flags:
        st[f.name] = rng.getrandbits(1)
    return st


def check_isa(key, isa, forms, cases, rng):
    gt_sim = HardenedGTSim(isa)
    fset = {f.name for f in isa.flags}
    pending = []        # (gt_mask, code, state, taint)
    checked = with_taint = 0

    for _label, code, srcs, _cons in forms:
        for _ in range(cases):
            st = _state(isa, rng)
            tt = {n: 0 for n, _ in isa.gprs}
            for f in isa.flags:
                tt[f.name] = 0
            # SPARSE taint: the exact 2^k oracle only runs when k <= budget
            # (13).  A full-width random mask gives k ~= 32, silently takes the
            # lower-bound path, returns exact=False, and the campaign counts the
            # case as skipped rather than checked.
            for s2 in srcs:
                if s2 not in tt:
                    continue
                width = 1 if s2 in fset else isa.bits
                m = 0
                for _b in range(rng.randint(1, 3)):
                    m |= 1 << rng.randrange(width)
                tt[s2] = m
            if not any(tt.values()):
                continue
            st, tt = isa.canonicalize(st, tt)
            g, ex = gt_sim.taint(code, st, tt)
            if not ex:
                continue
            checked += 1
            if not any(g.values()):
                continue
            with_taint += 1
            pending.append((g, code, st, tt))

    # ONE subprocess for the whole ISA: mt_batch spawns `uv run` per call.
    mutated = caught = 0
    if pending:
        batch = [{'label': 'v', 'asm': 'v', 'bytes': c.hex(),
                  'srcs': list(t), 'state': st, 'taint': t}
                 for _g, c, st, t in pending]
        try:
            mts = mt_batch(isa, batch)
        except Exception as exc:
            print(f'  {key}: mt_batch failed: {exc}')
            mts = []
        for (g, _c, _st, _t), mt in zip(pending, mts):
            live = [(r, b) for r in isa.reg_names for b in range(isa.bits)
                    if (g.get(r, 0) >> b) & 1]
            if not live:
                continue
            r, b = live[rng.randrange(len(live))]
            hurt = dict(mt)
            hurt[r] = (hurt.get(r, 0) or 0) & ~(1 << b)
            mutated += 1
            under = 0
            for rr in isa.reg_names:
                under |= g.get(rr, 0) & ~hurt.get(rr, 0)
            if under:
                caught += 1

    return {
        'checked': checked, 'with_taint': with_taint,
        'mutated': mutated, 'caught': caught,
        'skipped': gt_sim.skipped, 'ran': gt_sim.ran,
    }


def check_no_phantom(isa, rng, cases=40):
    """`shrd rbp, rsp` must not invent taint.  Reported as a leak count."""
    try:
        code = isa.assemble('shrd rbp, rsp, cl')
    except Exception:
        return None
    hard, leaky = HardenedGTSim(isa), GTSim(isa)
    hard_n = leak_n = n = 0
    for _ in range(cases):
        st = _state(isa, rng)
        tt = {n2: 0 for n2, _ in isa.gprs}
        for f in isa.flags:
            tt[f.name] = 0
        # Bit 21 of RCX is NOT in CL, so `shrd rbp, rsp, cl` cannot depend on
        # it.  Any output taint here is phantom, by construction.
        tt['RCX'] = 1 << 21
        n += 1
        g, ex = hard.taint(code, st, tt)
        if ex and any(v for r, v in g.items() if r != 'RCX'):
            hard_n += 1
        try:
            g2, ex2 = leaky.taint(code, st, tt)
            if ex2 and any(v for r, v in g2.items() if r != 'RCX'):
                leak_n += 1
        except Exception:
            pass
    return {'phantom_hardened': hard_n, 'phantom_shipped': leak_n, 'cases': n}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cases', type=int, default=6, help='cases per form')
    ap.add_argument('--forms', type=int, default=12, help='forms per ISA')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--json', metavar='PATH')
    args = ap.parse_args()

    isas = _build_isas('all')
    report: dict = {'experiment': 'campaign-oracle-validation', 'isas': {}}
    failures = []

    for i, key in enumerate(CORPORA):
        isa = isas[key]
        forms = _forms(isa, key, args.forms)
        if not forms:
            print(f'{key:8s} SKIP  no assemblable forms')
            continue
        rng = random.Random(args.seed + i)
        res = check_isa(key, isa, forms, args.cases, rng)

        # A zero is only meaningful if the oracle found taint AND a dropped bit
        # is caught every time.
        vacuous = res['with_taint'] == 0
        mut_ok = res['mutated'] > 0 and res['caught'] == res['mutated']
        ok = not vacuous and mut_ok
        if key == 'x86_64':
            res['no_phantom'] = check_no_phantom(isa, rng)
        if not ok:
            failures.append(key)
        report['isas'][key] = dict(res, ok=ok, vacuous=vacuous)
        print(f'{key:8s} {"OK  " if ok else "FAIL"} '
              f'checked {res["checked"]:5d}  with-taint {res["with_taint"]:5d}  '
              f'mutation {res["caught"]}/{res["mutated"]}  skipped {res["skipped"]}')
        if res.get('no_phantom'):
            np = res['no_phantom']
            print(f'{"":8s}      shrd rbp,rsp phantom taint (RCX bit 21, '
                  f'not in CL): hardened {np["phantom_hardened"]}/{np["cases"]}, '
                  f'shipped GTSim {np["phantom_shipped"]}/{np["cases"]}')

    report['passed'] = not failures
    report['failed_isas'] = failures
    if args.json:
        with open(args.json, 'w') as fh:
            json.dump(report, fh, indent=2)
            fh.write('\n')
        print(f'[json] {args.json}')
    if failures:
        print(f'\nFAILED on {", ".join(failures)}: a zero from those ISAs would '
              f'not mean the engine is sound.')
        return 1
    print('\nPASSED: the ground truth finds taint on every ISA and a dropped '
          'taint bit is always caught.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
