from __future__ import annotations

import pytest

from microtaint.instrumentation.ast import (
    AvalancheExpr,
    BinaryExpr,
    Constant,
    EvalContext,
    InstructionCellExpr,
    LogicCircuit,
    Op,
    TaintAssignment,
    TaintOperand,
    UnaryExpr,
)
from microtaint.types import Architecture, Register


def test_avalanche_expr() -> None:
    inner = Constant(5, 32)
    av = AvalancheExpr(inner)
    assert str(av) == 'AVALANCHE(0x5)'
    assert av.evaluate(EvalContext({}, {})) == -1

    zero_inner = Constant(0, 32)
    av_zero = AvalancheExpr(zero_inner)
    assert av_zero.evaluate(EvalContext({}, {})) == 0


def test_taint_operand() -> None:
    # Taint target (multi-bit)
    t_op = TaintOperand('RAX', 0, 31, is_taint=True)
    assert str(t_op) == 'T_RAX[31:0]'
    # Concrete target (single bit)
    v_op = TaintOperand('ZF', 0, 0, is_taint=False)
    assert str(v_op) == 'V_ZF[0]'

    # Evaluation
    # Value for V_RAX is 0x12345678, shifted down 0, masked to 32 bits
    assert t_op.evaluate(EvalContext({'RAX': 0xFFFFFFFF}, {})) == 0xFFFFFFFF
    assert v_op.evaluate(EvalContext({}, {'ZF': 1})) == 1

    # Check bit slice
    t_op2 = TaintOperand('RAX', 8, 15, is_taint=True)
    assert t_op2.evaluate(EvalContext({'RAX': 0x12345678}, {})) == 0x56


def test_constant() -> None:
    c = Constant(42, 32)
    assert str(c) == '0x2a'
    assert c.evaluate(EvalContext({}, {})) == 42


def test_unary_expr() -> None:
    op = TaintOperand('RAX', 0, 7, is_taint=False)
    un_expr = UnaryExpr(Op.NOT, op)
    assert str(un_expr) == 'NOT(V_RAX[7:0])'

    assert un_expr.evaluate(EvalContext({'RAX': 0}, {'RAX': 0x0F})) == ~0x0F

    with pytest.raises(NotImplementedError):
        UnaryExpr(Op.AND, op).evaluate(EvalContext({}, {}))


def test_binary_expr() -> None:
    t_op1 = Constant(3, 32)
    t_op2 = Constant(5, 32)

    bin_and = BinaryExpr(Op.AND, t_op1, t_op2)
    assert str(bin_and) == '(0x3 AND 0x5)'
    assert bin_and.evaluate(EvalContext({}, {})) == (3 & 5)

    bin_or = BinaryExpr(Op.OR, t_op1, t_op2)
    assert bin_or.evaluate(EvalContext({}, {})) == (3 | 5)

    bin_xor = BinaryExpr(Op.XOR, t_op1, t_op2)
    assert bin_xor.evaluate(EvalContext({}, {})) == (3 ^ 5)

    with pytest.raises(NotImplementedError):
        BinaryExpr(Op.NOT, t_op1, t_op2).evaluate(EvalContext({}, {}))


def test_taint_assignment() -> None:
    tgt = TaintOperand('RAX', 0, 63, is_taint=True)
    dep1 = TaintOperand('RBX', 0, 63, is_taint=True)

    # 1. No expression
    assign1 = TaintAssignment(tgt, [dep1])
    assert str(assign1) == 'T_RAX[63:0] = T_RBX[63:0]'

    # 2. String expression
    assign2 = TaintAssignment(tgt, [dep1], expression_str='FOO')
    assert str(assign2) == 'T_RAX[63:0] = FOO'

    # 3. Proper expression
    assign3 = TaintAssignment(tgt, [dep1], expression=BinaryExpr(Op.OR, dep1, dep1))
    assert str(assign3) == 'T_RAX[63:0] = (T_RBX[63:0] OR T_RBX[63:0])'


def test_logic_circuit() -> None:
    tgt1 = TaintOperand('RAX', 0, 7, is_taint=True)
    dep1 = TaintOperand('RBX', 0, 7, is_taint=True)
    assign1 = TaintAssignment(tgt1, [dep1], expression=BinaryExpr(Op.AND, dep1, Constant(0xFF, 8)))

    tgt2 = TaintOperand('RCX', 8, 15, is_taint=True)
    dep2 = TaintOperand('RDX', 8, 15, is_taint=True)
    assign2 = TaintAssignment(tgt2, [dep2])  # Default OR evaluation

    circuit = LogicCircuit(
        assignments=[assign1, assign2],
        architecture=Architecture.X86,
        instruction='01d8',
        state_format=[Register('RAX', 8), Register('RBX', 8), Register('RCX', 16), Register('RDX', 16)],
    )

    assert str(assign1) in str(circuit)

    out_taint = circuit.evaluate(EvalContext({'RBX': 0xAB, 'RDX': 0x1234}, {}))
    assert out_taint['RAX'] == 0xAB
    assert out_taint['RCX'] == 0x1200  # RDX[8:15] evaluated is 0x12, pushed up back by bit_start (8) gives 0x1200

    assign3 = TaintAssignment(tgt1, [], expression_str='FOO')
    circuit.assignments = [assign3]
    with pytest.raises(NotImplementedError):
        circuit.evaluate(EvalContext({}, {}))


def test_instruction_cell_expr() -> None:
    cell = InstructionCellExpr(
        architecture=Architecture.AMD64,
        instruction='4801d8',
        out_reg='RAX',
        out_bit_start=0,
        out_bit_end=63,
        inputs={'RAX': TaintOperand('RAX', 0, 63, is_taint=True)},
    )
    assert str(cell) == 'SimulateCell(instr=0x4801d8, out=RAX[63:0], RAX=T_RAX[63:0])'

    with pytest.raises(NotImplementedError):
        cell.evaluate(EvalContext({}, {}))
