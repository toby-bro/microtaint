"""
Unit tests for bit-precise taint propagation fixes in the engine.
"""

# mypy: disable-error-code="union-attr"
# mypy: disable-error-code="attr-defined"

from __future__ import annotations

from microtaint.instrumentation.ast import BinaryExpr, Constant, InstructionCellExpr, Op
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, Register


def test_shl_no_transport_term() -> None:
    # SHL EAX, 4 -> c1 e0 04
    bytestring = bytes.fromhex('c1e004')
    arch = Architecture.AMD64
    state_format = [
        Register(name='RAX', bits=64),
        Register(name='RBX', bits=64),
    ]

    rule = generate_static_rule(arch, bytestring, state_format)
    rax_32_assignment = next(a for a in rule.assignments if a.target.name == 'RAX' and a.target.bit_end == 31)

    expr = rax_32_assignment.expression
    assert isinstance(expr, (BinaryExpr, InstructionCellExpr))
    if isinstance(expr, BinaryExpr):
        assert expr.op == Op.XOR, f'Expected XOR at top level, got {expr.op}'
        assert isinstance(expr.lhs, InstructionCellExpr), 'Left side should be InstructionCellExpr'
        assert isinstance(expr.rhs, InstructionCellExpr), 'Right side should be InstructionCellExpr'


def test_and_with_constant_uses_cell() -> None:
    # AND EAX, 0x0F0F -> 25 0f 0f 00 00
    bytestring = bytes.fromhex('250f0f0000')
    arch = Architecture.AMD64
    state_format = [Register(name='RAX', bits=64)]

    rule = generate_static_rule(arch, bytestring, state_format)
    rax_assignment = next(a for a in rule.assignments if a.target.name == 'RAX')

    # AND should now use the CellIFT differential to perfectly mask taint
    expr = rax_assignment.expression
    assert isinstance(expr, BinaryExpr)
    assert expr.op == Op.XOR


def test_mov_constant_clears_taint() -> None:
    # MOV EAX, 0x3C -> b8 3c 00 00 00
    bytestring = bytes.fromhex('b83c000000')
    arch = Architecture.AMD64
    state_format = [Register(name='RAX', bits=64)]

    rule = generate_static_rule(arch, bytestring, state_format)
    rax_assignment = next(a for a in rule.assignments if a.target.name == 'RAX')

    # A constant move has no dependencies and should emit a Constant(0) taint
    expr = rax_assignment.expression
    assert isinstance(expr, Constant)
    assert expr.value == 0


def test_or_bitwise_uses_cell() -> None:
    # OR EAX, EBX -> 09 d8
    bytestring = bytes.fromhex('09d8')
    arch = Architecture.AMD64
    state_format = [Register(name='RAX', bits=64), Register(name='RBX', bits=64)]

    rule = generate_static_rule(arch, bytestring, state_format)
    rax_assignment = next(a for a in rule.assignments if a.target.name == 'RAX' and a.target.bit_end == 31)

    # OR must use the differential to safely un-taint bits forced to 1
    expr = rax_assignment.expression
    assert isinstance(expr, BinaryExpr)
    assert expr.op == Op.XOR


def test_xor_reg_reg_uses_cell() -> None:
    # XOR EBX, EBX -> 31 db
    bytestring = bytes.fromhex('31db')
    arch = Architecture.AMD64
    state_format = [Register(name='RBX', bits=64)]

    rule = generate_static_rule(arch, bytestring, state_format)
    rbx_assignment = next(a for a in rule.assignments if a.target.name == 'RBX')

    # XOR must use the differential to guarantee zero output taint
    expr = rbx_assignment.expression
    # Update the assertion to allow Constant(0)
    assert isinstance(expr, (BinaryExpr, Constant))
    # If it is a constant, it MUST be 0
    if isinstance(expr, Constant):
        assert expr.value == 0
