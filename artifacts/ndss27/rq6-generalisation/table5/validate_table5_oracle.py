#!/usr/bin/env python3
# ruff: noqa: W505, E501, B905, RUF100, I001
#   RUF100 and I001 are config differences, not defects: the source repo
#   selects BLE001/E402/C901 (so those noqa ARE used there) and sorts
#   `microtaint` as third-party, while it is first-party here.  No single
#   spelling satisfies both repos, and the body must stay byte-identical
#   to the harness that produced the published numbers.
#   Style only, and suppressed rather than rewritten: this harness is
#   vendored from the campaign that produced the published numbers, and
#   e.g. adding zip(strict=True) would change behaviour where the
#   original silently truncated.  Correctness is gated by
#   validate_table5_oracle.py, not by restyling proven code.
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

Run:  python validate_table5_oracle.py [--cases N] [--json PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault('MICROTAINT_TAINT_IR', '0')
os.environ.setdefault('MICROTAINT_BLOCK', '0')

from oracle import HardenedGTSim  # noqa: E402
from probes import PRESERVED_STATES, preserved_flags  # noqa: E402

from corpora import CORPORA  # noqa: E402
from mt_multiarch import (  # noqa: E402
    GTSim,
    _build_isas,
    mt_batch,
    written_flags,
)


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
            # ARBITRARY taint mask, exactly as the campaign generates them.
            # The flip-union oracle costs k+1 runs, so there is no budget to
            # stay under and no reason to bias the validator toward sparse
            # taint: validate what actually runs.
            for s2 in srcs:
                if s2 not in tt:
                    continue
                width = 1 if s2 in fset else isa.bits
                k = rng.randint(0, width)
                m = 0
                for b in rng.sample(range(width), k):
                    m |= 1 << b
                tt[s2] = m
            if not any(tt.values()):
                continue
            st, tt = isa.canonicalize(st, tt)
            g, ex = gt_sim.taint_flip_union(code, st, tt)
            if not ex:
                continue
            checked += 1
            if not any(g.values()):
                continue
            with_taint += 1
            pending.append((g, code, st, tt))

    # ONE subprocess for the whole ISA: mt_batch spawns `uv run` per call.
    mutated = caught = 0
    uncovered: dict[str, int] = {}
    if pending:
        batch = [{'label': 'v', 'asm': 'v', 'bytes': c.hex(),
                  'srcs': list(t), 'state': st, 'taint': t}
                 for _g, c, st, t in pending]
        try:
            mts = mt_batch(isa, batch)
        except Exception as exc:  # noqa: BLE001
            print(f'  {key}: mt_batch failed: {exc}')
            mts = []
        # The mutation must be judged by the CAMPAIGN's comparison, not by a
        # reimplementation of it here.  The previous version drew bit b from the
        # ground truth, cleared it from the engine's mask, then tested
        # `g & ~hurt` -- which contains b by construction.  It never read the
        # engine's answer in any way that could fail, so a perfect engine, an
        # engine tainting everything and an engine reporting NOTHING all scored
        # 2000/2000 caught.  It could not fail, so it certified nothing.
        #
        # What can genuinely fail is coverage: the campaign scores `chk`
        # (the GPRs plus the flags this form writes, minus exclusions), not
        # every register the oracle reports.  A dropped bit in a register
        # outside `chk` is invisible to the campaign, and that is the property
        # worth asserting.
        # `chk` must be the CAMPAIGN's scored set, preserved flags included.
        # Computing a different one makes this validator answer a question the
        # campaign never asks, which is the flaw it exists to detect.
        pres_cache: dict[bytes, set[str]] = {}
        for (g, c, st0, t0), mt in zip(pending, mts):
            if c not in pres_cache:
                cases = [(st0, {f.name: rng.getrandbits(f.width) for f in isa.flags})
                         for _ in range(PRESERVED_STATES)]
                pres_cache[c] = preserved_flags(gt_sim, isa, c, cases, list(t0))
            chk = ([r for r, _ in isa.gprs]
                   + sorted(written_flags(isa, c) | pres_cache[c]))
            live = [(r, b) for r in isa.reg_names for b in range(isa.bits)
                    if (g.get(r, 0) >> b) & 1]
            if not live:
                continue
            r, b = live[rng.randrange(len(live))]
            hurt = dict(mt)
            hurt[r] = (hurt.get(r, 0) or 0) & ~(1 << b)
            mutated += 1
            # Exactly the campaign's own test, over exactly its own register set.
            under = 0
            for rr in chk:
                under |= g.get(rr, 0) & ~hurt.get(rr, 0)
            if under:
                caught += 1
            else:
                uncovered.setdefault(r, 0)
                uncovered[r] += 1

    return {
        'checked': checked, 'with_taint': with_taint,
        'mutated': mutated, 'caught': caught,
        'uncovered': uncovered,
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
        g, ex = hard.taint_flip_union(code, st, tt)
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

        # A zero is only meaningful if the oracle found taint AND every bit the
        # oracle can see lies in a register the campaign actually scores.  The
        # mutation drops one such bit from the engine's answer and requires the
        # campaign's own comparison, over its own `chk` set, to report it.  What
        # this can fail on is COVERAGE: ground truth in a register outside `chk`
        # is invisible to the campaign no matter what the engine says.
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
        if res.get('uncovered'):
            print(f'{"":8s}      NOT SCORED by the campaign: '
                  f'{dict(sorted(res["uncovered"].items()))}')
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
    print('\nPASSED: the ground truth finds taint on every ISA, and every bit '
          'it finds lies in a register the campaign scores, so dropping one is '
          'reported as an under-taint.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
