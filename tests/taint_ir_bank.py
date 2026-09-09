"""Validate and measure the lowered taint IR over the instruction bank.

The IR is a second implementation of the same taint semantics, so it is checked
two ways: against Unicorn per-bit ground truth (soundness, the only gate that
matters) and against the C per-op composer (agreement between the two
implementations, which catches a lowering slip that ground truth would tolerate
as harmless over-taint).

It also reports the number the whole compaction effort is aimed at: machine
operations to propagate taint through one whole instruction, flags included,
AFTER constant folding and dead-code elimination.  Unlike the interpreted count
this includes the value computation the taint rules depend on, so it is the
honest total a compiled taint circuit would have to execute.
"""
# ruff: noqa: PLC0415
from __future__ import annotations

from microtaint.taint_ir.frompcode import Unsupported, build_ir
from tests.perop_c_bank import Declined, _engine_names

_CACHE: dict = {}
from microtaint.taint_ir.regmap import name_offset, slot_resolver  # noqa: E402,F401


def ir_state(arch, names, values, taints):
    """Register state keyed the way the IR keys it."""
    v, t = {}, {}
    for n in names:
        off = name_offset(arch, n)
        if off is None:
            continue
        for key in ((('reg', off, sz)) for sz in range(1, 9)):
            v[key] = values.get(n, 0)
            t[key] = taints.get(n, 0)
    return v, t


def _prog(arch, code):
    key = (arch.value if hasattr(arch, 'value') else str(arch), code)
    hit = _CACHE.get(key)
    if hit is None:
        try:
            hit = build_ir(arch, code)
        except Unsupported as e:
            hit = ('decline', str(e))
        _CACHE[key] = hit
    return hit


def ir_step(arch, code: bytes, regs, in_taint, in_values):
    """Bank adapter: run the lowered IR for one input state."""
    prog = _prog(arch, code)
    if isinstance(prog, tuple):
        raise Declined(prog[1])
    from microtaint.taint_ir.frompcode import builder_for
    declared = builder_for(arch).declared
    size_of = {nm: sz for nm, sz in declared.values()}
    from microtaint.instrumentation.cell import _build_reg_maps
    size_of.update(_build_reg_maps(arch)[1])

    names = [r.name for r in regs]
    alias = _engine_names(arch, names)
    values, taints = {}, {}
    for n in names:
        en = alias[n]
        off = name_offset(arch, en)
        if off is None:
            continue
        m = (1 << (size_of.get(en, 8) * 8)) - 1
        for sz in range(1, 9):
            values[('reg', off, sz)] = in_values.get(n, 0) & m
            taints[('reg', off, sz)] = in_taint.get(n, 0) & m
    out = prog.run(values, taints)
    res = {n: in_taint.get(n, 0) for n in names}
    for n in names:
        off = name_offset(arch, alias[n])
        if off is None:
            continue
        for sz in range(1, 9):
            if ('reg', off, sz) in out:
                res[n] = out[('reg', off, sz)]
                break
    cost = {'ops': prog.cost(), 'pcode_ops': 0, 'n_route': 0, 'n_diff': 0,
            'n_floor': 0, 'n_cube': 0, 'reads': 0, 'writes': 0, 'forks': 0}
    return res, {}, cost


def main(argv=None):
    import argparse

    from tests.perop_c_bank import run_bank_perop_c

    ap = argparse.ArgumentParser()
    ap.add_argument('--isas', nargs='*', default=['AMD64', 'ARM64', 'RISCV64'])
    ap.add_argument('--sparse', type=int, default=4)
    args = ap.parse_args(argv)
    for isa in args.isas:
        rep = run_bank_perop_c(isas=[isa], n_sparse=args.sparse,
                               ref='ground_truth', step=ir_step)
        print(f'[{isa}] {rep.summary()}')
        print(f'[{isa}] {rep.stats.summary()}')
        print(f'[{isa}] {rep.vs_oracle()}')
        for label, it, _iv, new in rep.new_under_examples[:6]:
            print(f'    NEW UNDER {label}: {[(k, hex(b)) for k, b in new.items()]}')
            print(f'        taint={[(k, hex(x)) for k, x in it.items() if x]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
