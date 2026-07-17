"""M1 regression: the seeded partial-re-execution primitive in cell.pyx.

Intermediate-taint materialization (docs/design/intermediate-taint-materialization.md)
splits an instruction's taint computation at a reused intermediate result: the
arithmetic core is evaluated once, and downstream flag/reuse circuits consume the
materialized value instead of recomputing it.  The load-bearing primitive is the
Cython interpreter's ability to (a) name/read an intermediate `unique` varnode,
(b) seed it and protect it across a re-executed defining op, and (c) run only the
downstream suffix of the op list.

These tests pin the primitive against the existing whole-slice differential on
x86 `cmp rax, rbx`, whose p-code computes every flag from a subtraction whose
result lives in a discarded `unique` varnode.  Two independent guarantees:

  * PROTECTION — running the FULL op list with the intermediate seeded to its own
    per-replica value reproduces the whole-slice differential for EVERY output
    (the protected defining op is a no-op that keeps the seed).
  * SUFFIX — running only the ops AFTER the intermediate's definition, with the
    intermediate seeded, reproduces the whole-slice differential for the
    RESULT-DERIVED outputs (SF/ZF/PF).  CF/OF read the operands directly
    (INT_LESS / INT_SBORROW, before the subtraction) so the cut does not apply
    to them -- the test derives the result-derived set by forward reachability
    rather than hard-coding it.
"""

# mypy: disable-error-code="no-untyped-def,no-untyped-call,attr-defined,union-attr"

from __future__ import annotations

import random

import pytest

from microtaint.instrumentation.cell import PCodeCellEvaluator
from microtaint.sleigh.lifter import get_context
from microtaint.types import Architecture

ARCH = Architecture.AMD64
HEX = '4839d8'  # cmp rax, rbx
MASK64 = (1 << 64) - 1


def _setup():
    ev = PCodeCellEvaluator(ARCH)
    tr = get_context(str(ARCH)).translate(bytes.fromhex(HEX), 0x1000)
    off2name: dict[int, str] = {}
    for name, off in ev._offsets.items():
        off2name.setdefault(int(off), name)

    sub_off = sub_size = None
    flags: list[str] = []
    for op in tr.ops:
        out = op.output
        if out is not None and out.space.name == 'unique' and op.opcode.name == 'INT_SUB':
            sub_off, sub_size = out.offset, out.size
        if out is not None and out.space.name == 'register' and out.size == 1:
            nm = off2name.get(out.offset)
            if nm:
                flags.append(nm)
    assert sub_off is not None, 'no INT_SUB unique in cmp lift'

    # forward reachability from the subtraction result -> result-derived outputs
    reach = {('unique', sub_off)}
    result_derived: set[str] = set()
    for op in tr.ops:
        if any((i.space.name, i.offset) in reach for i in op.inputs) and op.output is not None:
            reach.add((op.output.space.name, op.output.offset))
            if op.output.space.name == 'register' and op.output.size == 1:
                nm = off2name.get(op.output.offset)
                if nm:
                    result_derived.add(nm)
    return ev, sub_off, sub_size, flags, result_derived


def test_result_and_operand_derived_split():
    """The cut applies exactly to SF/ZF/PF; CF/OF are operand-derived."""
    _ev, _off, _sz, flags, result_derived = _setup()
    assert result_derived == {'SF', 'ZF', 'PF'}
    assert set(flags) - result_derived == {'CF', 'OF'}


def test_value_read_of_intermediate():
    """evaluate_uniq_concrete reads the concrete subtraction result a-b."""
    ev, sub_off, sub_size, _flags, _rd = _setup()
    w = sub_size * 8
    rng = random.Random(7)
    for _ in range(500):
        a, b = rng.getrandbits(64), rng.getrandbits(64)
        got = ev.evaluate_uniq_concrete(HEX, {'RAX': a, 'RBX': b}, sub_off, 0, w - 1)
        assert got == ((a - b) & MASK64)


def test_seeded_matches_whole_slice_differential():
    """Protection reproduces the full differential for all flags; a seeded suffix
    run reproduces it for the result-derived flags."""
    ev, sub_off, sub_size, flags, result_derived = _setup()
    w = sub_size * 8
    start_pc = ev.uniq_start_pc(HEX, sub_off)
    assert start_pc > 0
    rng = random.Random(1234)

    for _ in range(4000):
        a, b = rng.getrandbits(64), rng.getrandbits(64)
        ta = rng.getrandbits(64) & rng.getrandbits(64)
        tb = rng.getrandbits(64) & rng.getrandbits(64)
        or_in = {'RAX': (a | ta) & MASK64, 'RBX': (b | tb) & MASK64}
        and_in = {'RAX': (a & ~ta) & MASK64, 'RBX': (b & ~tb) & MASK64}
        t_a = ev.evaluate_uniq_concrete(HEX, or_in, sub_off, 0, w - 1)
        t_b = ev.evaluate_uniq_concrete(HEX, and_in, sub_off, 0, w - 1)
        seeds_a, seeds_b = {sub_off: t_a}, {sub_off: t_b}

        for nm in flags:
            full = ev.evaluate_differential_seeded(HEX, or_in, and_in, {}, {}, nm, 0, 0, 0)
            protect = ev.evaluate_differential_seeded(
                HEX, or_in, and_in, seeds_a, seeds_b, nm, 0, 0, 0,
            )
            assert protect == full, f'protection diverged on {nm}'
            if nm in result_derived:
                suffix = ev.evaluate_differential_seeded(
                    HEX, or_in, and_in, seeds_a, seeds_b, nm, 0, 0, start_pc,
                )
                assert suffix == full, f'seeded suffix diverged on {nm}'


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
