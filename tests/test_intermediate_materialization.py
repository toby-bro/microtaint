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

# mypy: disable-error-code="no-untyped-def,no-untyped-call,attr-defined,union-attr,arg-type"

from __future__ import annotations

import random

import pytest

from microtaint.instrumentation.ast import (
    EvalContext,
    IntermediateTaintExpr,
    IntermediateValueExpr,
    LogicCircuit,
    TaintAssignment,
    TaintOperand,
)
from microtaint.instrumentation.cell import PCodeCellEvaluator
from microtaint.sleigh.lifter import get_context
from microtaint.types import Architecture

ARCH = Architecture.AMD64
HEX = '4839d8'  # cmp rax, rbx
MASK64 = (1 << 64) - 1


class _StubSim:
    """Minimal simulator exposing what the intermediate Exprs read at runtime."""

    def __init__(self):
        self.arch = ARCH
        self._pcode = PCodeCellEvaluator(ARCH)
        self.use_unicorn = False


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


def test_two_phase_circuit_threads_intermediate_and_hides_uniq():
    """M2: a two-phase LogicCircuit materializes an intermediate (value + taint)
    and threads it to downstream assignments; UNIQ_ keys never leak to output.

    Downstream SF is read as bit (W-1) of the materialized taint -- SF is literally
    the result's sign bit -- so it must equal the whole-slice differential T(SF).
    """
    sim = _StubSim()
    ev = sim._pcode
    _e2, sub_off, sub_size, _flags, _rd = _setup()
    w = sub_size * 8
    srcs = ['RAX', 'RBX']
    uniq = f'UNIQ_{sub_off}'

    inter = TaintAssignment(
        target=TaintOperand(uniq, 0, w - 1, True),
        dependencies=[],
        expression=IntermediateTaintExpr(HEX, sub_off, 0, w - 1, srcs),
        is_intermediate=True,
        value_expression=IntermediateValueExpr(HEX, sub_off, 0, w - 1, srcs),
    )
    sf_assign = TaintAssignment(
        target=TaintOperand('SF', 0, 0, True),
        dependencies=[],
        expression=TaintOperand(uniq, w - 1, w - 1, True),
    )
    rcx_assign = TaintAssignment(
        target=TaintOperand('RCX', 0, w - 1, True),
        dependencies=[],
        expression=TaintOperand(uniq, 0, w - 1, True),
    )
    circuit = LogicCircuit([inter, sf_assign, rcx_assign], ARCH, HEX, [])
    assert circuit.has_intermediates
    assert len(circuit.intermediate_assignments) == 1
    assert len(circuit.regular_assignments) == 2

    rng = random.Random(99)
    for _ in range(3000):
        a, b = rng.getrandbits(64), rng.getrandbits(64)
        ta = rng.getrandbits(64) & rng.getrandbits(64)
        tb = rng.getrandbits(64) & rng.getrandbits(64)
        ctx = EvalContext(
            input_taint={'RAX': ta, 'RBX': tb},
            input_values={'RAX': a, 'RBX': b},
            simulator=sim,
        )
        out = circuit.evaluate(ctx)

        or_in = {'RAX': (a | ta) & MASK64, 'RBX': (b | tb) & MASK64}
        and_in = {'RAX': (a & ~ta) & MASK64, 'RBX': (b & ~tb) & MASK64}
        t_t = ev.evaluate_differential_seeded(HEX, or_in, and_in, {}, {}, uniq, 0, w - 1, 0)
        sf_gt = ev.evaluate_differential_seeded(HEX, or_in, and_in, {}, {}, 'SF', 0, 0, 0)

        assert out.get('SF', 0) == sf_gt, 'SF via materialized taint diverged from whole-slice'
        assert out.get('RCX', 0) == t_t, 'full T(t) threading error'
        assert not any(k.startswith('UNIQ_') for k in out), 'UNIQ_ leaked into output'


def test_concrete_seeded_is_the_single_replica_building_block():
    """evaluate_concrete_seeded is the per-replica primitive a segmented differential
    is composed from: XORing its two replicas equals evaluate_differential_seeded, and
    with no seeds reading a UNIQ output it equals evaluate_uniq_concrete."""
    ev = PCodeCellEvaluator(ARCH)
    _e, sub_off, sub_size, _f, _rd = _setup()
    w = sub_size * 8
    spc = ev.uniq_start_pc(HEX, sub_off)
    rng = random.Random(11)
    for _ in range(2000):
        a, b = rng.getrandbits(64), rng.getrandbits(64)
        ta = rng.getrandbits(64) & rng.getrandbits(64)
        tb = rng.getrandbits(64) & rng.getrandbits(64)
        or_in = {'RAX': (a | ta) & MASK64, 'RBX': (b | tb) & MASK64}
        and_in = {'RAX': (a & ~ta) & MASK64, 'RBX': (b & ~tb) & MASK64}
        t_a = ev.evaluate_uniq_concrete(HEX, or_in, sub_off, 0, w - 1)
        t_b = ev.evaluate_uniq_concrete(HEX, and_in, sub_off, 0, w - 1)
        for out in ('SF', 'ZF', 'CF'):
            diff = ev.evaluate_differential_seeded(
                HEX, or_in, and_in, {sub_off: t_a}, {sub_off: t_b}, out, 0, 0, spc,
            )
            c_or = ev.evaluate_concrete_seeded(HEX, or_in, {sub_off: t_a}, out, 0, 0, spc)
            c_and = ev.evaluate_concrete_seeded(HEX, and_in, {sub_off: t_b}, out, 0, 0, spc)
            assert (c_or ^ c_and) == diff

        v = ev.evaluate_concrete_seeded(HEX, {'RAX': a, 'RBX': b}, {}, f'UNIQ_{sub_off}', 0, w - 1, 0)
        assert v == ev.evaluate_uniq_concrete(HEX, {'RAX': a, 'RBX': b}, sub_off, 0, w - 1)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
