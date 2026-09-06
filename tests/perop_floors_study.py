"""Full-bank sweep of the per-op-with-floors model vs ground truth + engine oracle.

Standalone (not a pytest gate -- that lives in test_perop_floors.py, bounded).
Reports, per ISA: case count, exact-vs-ground-truth rate, sound-over-approx rate,
shared SLEIGH undefined-flag artifacts, monolithic-fallback rate, and any NEW
under-taints (soundness bugs -- must be zero).

  python -m tests.perop_floors_study [ISA ...] [--n N]
"""
# ruff: noqa: PLC0415
# mypy: disable-error-code="no-untyped-def,no-untyped-call,import-untyped"
from __future__ import annotations

import random
import sys

MASK64 = 0xFFFFFFFFFFFFFFFF


def sweep(isa, n_per=40, seed=7):
    from tests.oracle_harness import (UC_DESCS, build_circuit, classify,
                                      ground_truth, reference_taint)
    from tests.perop_floors import NeedsMonolithic, Unsupported, engine_perop_floors
    from tests.test_perop_floors import _gt_vectors, _under_bits
    from benchmark.instruction_bank import load_bank

    if isa not in UC_DESCS:
        print(f'{isa}: no ground-truth UcDesc; skipping')
        return 0
    ud = UC_DESCS[isa]()
    gp = list(ud.gp)
    keys = gp + list(ud.flags)
    specs = load_bank(isas={isa})
    n_cases = n_exact = n_over = 0
    n_mono = n_unsup = 0
    shared = 0
    new_under = []
    for spec in specs.values():
        regs = spec.regs
        for ins in spec.instructions:
            try:
                circuit = build_circuit(spec.arch, ins.bytes, spec.regs)
            except Exception:  # noqa: BLE001
                continue
            rng = random.Random(f'{seed}:{ins.label}')
            for t, vals in _gt_vectors(gp, rng, n_per):
                in_taint = {r.name: t.get(r.name, 0) for r in regs}
                in_values = {r.name: vals.get(r.name, 0) for r in regs}
                try:
                    got = engine_perop_floors(spec.arch, ins.bytes, regs, in_taint, in_values)
                except NeedsMonolithic:
                    n_mono += 1
                    continue
                except Unsupported:
                    n_unsup += 1
                    continue
                except Exception:  # noqa: BLE001
                    n_unsup += 1
                    continue
                try:
                    gtd = ground_truth(ud, ins.bytes, t, vals)
                    engd = reference_taint(spec.arch, ins.bytes, regs, in_taint,
                                           in_values, circuit=circuit)
                except Exception:  # noqa: BLE001
                    continue
                n_cases += 1
                v = classify(got, gtd, keys)
                if v.exact:
                    n_exact += 1
                elif v.sound:
                    n_over += 1
                u_perop = _under_bits(got, gtd, keys)
                u_eng = _under_bits(engd, gtd, keys)
                for k, m in u_perop.items():
                    extra = m & ~u_eng.get(k, 0)
                    if extra:
                        if len(new_under) < 30:
                            new_under.append((ins.label, k, hex(extra),
                                              {kk: hex(vv) for kk, vv in t.items() if vv}))
                    else:
                        shared += 1
    tot = max(1, n_cases)
    print(f'=== {isa}: per-op-with-floors sweep ===')
    print(f'  handled cases       : {n_cases}')
    print(f'  exact vs GT         : {n_exact} ({100*n_exact/tot:.1f}%)')
    print(f'  sound over-approx   : {n_over} ({100*n_over/tot:.1f}%)')
    print(f'  shared artifacts    : {shared} (engine has them too; undefined flags)')
    print(f'  monolithic fallback : {n_mono} (cmov/rep/opaque -> whole-instr oracle)')
    print(f'  unsupported/err     : {n_unsup} (mem/BE/other, out of study scope)')
    print(f'  NEW under-taints     : {len(new_under)}  <-- MUST be 0')
    for label, k, m, tt in new_under[:30]:
        print(f'    BUG {label}: {k} extra-under={m} taint={tt}')
    return len(new_under)


def main():
    argv = sys.argv[1:]
    n = 40
    if '--n' in argv:
        i = argv.index('--n')
        n = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    isas = [a for a in argv if not a.startswith('--')] or ['AMD64']
    bugs = 0
    for isa in isas:
        bugs += sweep(isa, n_per=n)
        print()
    return 1 if bugs else 0


if __name__ == '__main__':
    raise SystemExit(main())
