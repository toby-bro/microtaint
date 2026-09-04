"""circuit_c must not crash on a wide (>64-bit) circuit.

The compiled LogicCircuit evaluator (microtaint/instrumentation/cell_c/circuit_c.c)
runs on a uint64 stack for speed.  A wide taint value (a 128/256/512-bit vector
slice) overflows it -- previously a segfault in do_evaluate.  A one-time
compile-time sanity gate now routes any assignment with a >64-bit operand or a
>64-bit output to the Python evaluator (arbitrary-width int math), which is
correct if slower.  The uint64 fast path is unchanged, so scalar circuits keep
their speed.

These tests pin: a wide circuit evaluates to the EXACT wide result (no crash, no
truncation), and a narrow circuit still evaluates correctly on the fast path.
"""

from __future__ import annotations

from microtaint.instrumentation.ast import (
    BinaryExpr,
    EvalContext,
    LogicCircuit,
    Op,
    TaintAssignment,
    TaintOperand,
)
from microtaint.types import Architecture, ImplicitTaintPolicy

_FULL64 = (1 << 64) - 1
_FULL128 = (1 << 128) - 1


def _eval(circ: LogicCircuit, taint: dict[str, int]) -> dict[str, int]:
    return circ.evaluate(EvalContext(
        input_taint=taint, input_values={}, simulator=None,
        implicit_policy=ImplicitTaintPolicy.IGNORE,
    ))


def _xor_circuit(bit_end: int) -> LogicCircuit:
    """OUT[0:bit_end] = A_taint ^ B_taint over [0, bit_end]."""
    expr = BinaryExpr(Op.XOR, TaintOperand('A', 0, bit_end, True), TaintOperand('B', 0, bit_end, True))
    target = TaintOperand('OUT', 0, bit_end, True)
    return LogicCircuit([TaintAssignment(target, [], expr)], Architecture.AMD64, '', [])


def test_wide_128bit_circuit_falls_back_and_is_exact() -> None:
    # 128-bit XOR: A all-ones, B low-64 ones -> high 64 set, low 64 cancel.
    out = _eval(_xor_circuit(127), {'A': _FULL128, 'B': _FULL64})
    assert out.get('OUT', 0) == (_FULL128 ^ _FULL64) == (_FULL64 << 64)


def test_wide_256bit_circuit_falls_back_and_is_exact() -> None:
    a = 0xDEADBEEF_00000000_CAFEBABE_11112222_33334444_55556666_77778888_9999AAAA
    b = 0x0F0F0F0F_0F0F0F0F_0F0F0F0F_0F0F0F0F_0F0F0F0F_0F0F0F0F_0F0F0F0F_0F0F0F0F
    out = _eval(_xor_circuit(255), {'A': a, 'B': b})
    assert out.get('OUT', 0) == (a ^ b)


def test_narrow_64bit_circuit_uses_fast_path_and_is_exact() -> None:
    # control: fits uint64 -> compiled fast path, must stay correct.
    out = _eval(_xor_circuit(63), {'A': 0xF0, 'B': 0x0F})
    assert out.get('OUT', 0) == 0xFF
