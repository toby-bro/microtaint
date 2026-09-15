#!/usr/bin/env python3
"""Prove the RQ6 oracle can still FAIL, before anyone trusts a zero from it.

A soundness campaign reports "0 under-taints".  That sentence is only worth
something if the harness was capable of saying otherwise, and a taint oracle has
several ways to become incapable while still exiting 0:

  * it can return an all-zero ground truth when every run failed, which is
    indistinguishable from "this instruction propagates no taint".  `GT & ~MT`
    is then unconditionally zero and no under-taint can ever be reported.  An
    earlier campaign lost 19,085,696 MIPS cases exactly this way.
  * it can reduce over a SUBSET of the runs when some of them raise.  For the
    lower bound that can only shrink the bound, and the soundness test is
    `lb & ~mt`, so a smaller bound is strictly more lenient: swallowed failures
    hide real under-taints rather than inventing false ones.
  * it can leak state between the polarity runs of one case, which invents taint
    that the instruction cannot carry.  One such defect produced 2,810 of 2,875
    reported under-taints in a 12h campaign, all through `shrd rbp, rsp`.

So this script checks three properties and exits non-zero if any fails:

  EQUIVALENCE   the fast snapshot runner answers identically to the reference
                fresh-Unicorn-per-run implementation, over random states.
  INVALIDATION  a case whose runs cannot complete raises CaseInvalid rather
                than returning a zero mask.
  MUTATION      drop ONE bit that the ground truth says is tainted out of the
                engine's answer, and the campaign's own soundness test must
                flag it.  If it does not, that ISA's zero means nothing.

Run:  uv run python validate_oracle.py [--states N] [--json PATH]
"""
# Experiment script, not library code: see artifacts/ndss27/README.md,
# "Lint and type checking", for why annotations are not required here.
# mypy: disable-error-code="no-untyped-def, no-untyped-call, type-arg"

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('MICROTAINT_TAINT_IR', '0')
os.environ.setdefault('MICROTAINT_BLOCK', '0')

import campaign as C
import multiarch_oracle as O

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.types import ImplicitTaintPolicy


def _engine_provenance() -> dict:
    """Which engine produced this result; never raises."""
    try:
        from microtaint.provenance import engine_provenance
        return dict(engine_provenance())
    except Exception:
        return {}


def _corpus(bench, limit):
    """Forms from the bench the CAMPAIGN actually runs, already assembled.

    Validating `multiarch_oracle.ISAS` instead would silently cover sparc and
    MISS riscv, which is the one ISA whose oracle has been vacuous before.
    Validate the objects under test, not a lookalike.
    """
    return list(bench.entries)[:limit]


def _mt(bench, code, state, taint):
    """The engine's answer, through the campaign's own call path."""
    ctx = EvalContext(input_taint=dict(taint), input_values=dict(state),
                      simulator=CellSimulator(bench.arch),
                      implicit_policy=ImplicitTaintPolicy.IGNORE)
    return C._rule(bench, code).evaluate(ctx)


def _case(bench, srcs, rng):
    """A random (state, taint) pair generated exactly as the campaign does."""
    state = {r: rng.getrandbits(bench.bits) for r in bench.regs}
    taint = C.gen_taint(rng, bench, srcs)
    return bench.canonicalize(state, taint)


def check_equivalence(spec, forms, states, rng):
    """Snapshot runner vs fresh-Uc reference, bit for bit."""
    checked = mismatches = 0
    for _asm, code, _out, _srcs in forms:
        runner = O._Runner(spec, code)
        for _ in range(states):
            st = {r: rng.getrandbits(spec.bits) & spec.mask for r in spec.regs}
            try:
                fast = runner.run(st)
            except O.CaseInvalid:
                continue
            try:
                ref = O._run(spec, code, st)
            except Exception:
                continue
            checked += 1
            if fast != ref:
                mismatches += 1
    return checked, mismatches


def check_invalidation(spec):
    """A case that cannot run must raise, not return zeros."""
    bad = b'\xff\xff\xff\xff'
    st = dict.fromkeys(spec.regs, 1)
    tt = dict.fromkeys(spec.regs, 0)
    tt[spec.regs[0]] = 0b11
    results = {}
    for name, fn in (('lower_bound', O.bitflip_lower_bound), ('exact_gt', O.exact_gt)):
        try:
            got = fn(spec, bad, st, tt)
            results[name] = 'RETURNED %r' % (got,)
        except O.CaseInvalid:
            results[name] = 'CaseInvalid'
        except Exception as exc:
            results[name] = '%s: %s' % (type(exc).__name__, exc)
    ok = all(v == 'CaseInvalid' for v in results.values())
    return ok, results


def check_mutation(spec, forms, states, rng):
    """Drop one truly-tainted bit from the engine's answer; the test must fire.

    This is the property that makes a zero meaningful.  It mirrors the campaign's
    own comparison (`lb & ~mt`) rather than inventing a new one.
    """
    mutated = caught = no_gt = 0
    for _asm, code, _out, srcs in forms:
        for _ in range(states):
            st, tt = _case(spec, srcs, rng)
            try:
                lb = O.bitflip_lower_bound(spec, code, st, tt)
            except O.CaseInvalid:
                continue
            except Exception:
                continue
            live = [(r, b) for r in spec.regs for b in range(spec.bits)
                    if (lb[r] >> b) & 1]
            if not live:
                no_gt += 1
                continue
            try:
                mt = _mt(spec, code, st, tt)
            except Exception:
                continue
            # Remove exactly one bit the ground truth says must be tainted.
            r, b = live[rng.randrange(len(live))]
            hurt = {k: (v or 0) for k, v in mt.items()}
            hurt[r] = (hurt.get(r, 0) or 0) & ~(1 << b)
            mutated += 1
            missed = 0
            for rr in spec.regs:
                missed |= lb[rr] & ~(hurt.get(rr, 0) or 0) & spec.mask
            if missed:
                caught += 1
    return mutated, caught, no_gt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--states', type=int, default=40, help='random states per form')
    ap.add_argument('--forms', type=int, default=6, help='forms per ISA')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--arch', default='all')
    ap.add_argument('--json', metavar='PATH')
    args = ap.parse_args()

    keys = list(C.ALL_ARCHES) if args.arch == 'all' else [args.arch]
    report: dict = {'experiment': 'rq6-oracle-validation',
                    'engine': _engine_provenance(), 'isas': {}}
    failures = []

    for i, key in enumerate(keys):
        try:
            spec = C.build(key)
        except Exception as exc:
            print(f'{key:8s} SKIP  cannot build bench: {exc}')
            continue
        forms = _corpus(spec, args.forms)
        if not forms:
            print(f'{key:8s} SKIP  no assemblable corpus forms')
            continue
        rng = random.Random(args.seed + i)

        ck, mm = check_equivalence(spec, forms, args.states, rng)
        inv_ok, inv = check_invalidation(spec)
        mut, caught, no_gt = check_mutation(spec, forms, args.states, rng)

        equiv_ok = mm == 0 and ck > 0
        mut_ok = mut > 0 and caught == mut
        ok = equiv_ok and inv_ok and mut_ok
        if not ok:
            failures.append(key)

        report['isas'][key] = {
            'label': spec.label, 'forms': len(forms),
            'equivalence': {'checked': ck, 'mismatches': mm, 'ok': equiv_ok},
            'invalidation': {'detail': inv, 'ok': inv_ok},
            'mutation': {'mutated': mut, 'caught': caught,
                         'no_ground_truth': no_gt, 'ok': mut_ok},
            'ok': ok,
        }
        print(f'{key:8s} {"OK  " if ok else "FAIL"} '
              f'equiv {ck - mm}/{ck}   invalidation {"ok" if inv_ok else inv}   '
              f'mutation caught {caught}/{mut} (no-GT {no_gt})')

    report['passed'] = not failures
    report['failed_isas'] = failures
    if args.json:
        with open(args.json, 'w') as fh:
            json.dump(report, fh, indent=2)
            fh.write('\n')
        print(f'[json] {args.json}')

    if failures:
        print(f'\nFAILED on {", ".join(failures)}: the oracle cannot be trusted '
              f'to report a zero on those ISAs.')
        return 1
    print('\nPASSED: the snapshot runner matches the reference, an unrunnable '
          'case invalidates, and a dropped taint bit is caught on every ISA.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
