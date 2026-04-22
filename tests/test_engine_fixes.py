"""
Unit tests for bit-precise taint propagation fixes in the engine.

These tests verify that:
1. TRANSPORTABLE operations (shifts) don't OR with old taint
2. MAPPED operations with constants (AND, MOV) use cell simulation
3. Taint propagation is bit-precise, not conservative
"""

from __future__ import annotations

from microtaint.instrumentation.ast import BinaryExpr, InstructionCellExpr, Op
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, Register


def test_shl_no_transport_term() -> None:
    """
    Test that SHL (shift left) correctly shifts taint bits without ORing with old taint.

    Fix: Removed transport_term OR in TRANSPORTABLE category.
    Expected: SHL EAX, 4 on taint 0xFFFF should produce 0xFFFF0, not 0xFFFFF.
    """
    # SHL EAX, 4 -> c1 e0 04
    bytestring = bytes.fromhex('c1e004')
    arch = Architecture.AMD64
    state_format = [
        Register(name='RAX', bits=64),
        Register(name='RBX', bits=64),
        Register(name='RCX', bits=64),
        Register(name='RDX', bits=64),
    ]

    rule = generate_static_rule(arch, bytestring, state_format)

    # Should have assignments for RAX
    assert len(rule.assignments) > 0

    # Find the RAX[31:0] assignment (32-bit result)
    rax_32_assignment = None
    for assignment in rule.assignments:
        if assignment.target.name == 'RAX' and assignment.target.bit_end == 31:
            rax_32_assignment = assignment
            break

    assert rax_32_assignment is not None, 'Should have RAX[31:0] assignment'

    # The expression should be a differential (XOR of cells), NOT ORed with transport term
    expr = rax_32_assignment.expression

    # It should be BinaryExpr(XOR, Cell, Cell) without an outer OR
    assert isinstance(expr, BinaryExpr), f'Expected BinaryExpr, got {type(expr)}'
    assert expr.op == Op.XOR, f'Expected XOR at top level, got {expr.op}'
    assert isinstance(expr.lhs, InstructionCellExpr), 'Left side should be InstructionCellExpr'
    assert isinstance(expr.rhs, InstructionCellExpr), 'Right side should be InstructionCellExpr'


def test_and_with_constant_uses_cell() -> None:
    """
    Test that AND with a constant uses cell simulation for bit-precise masking.

    Fix: Added cell simulation for bitwise ops in MAPPED category.
    Expected: AND EAX, 0xF0F should use InstructionCellExpr, not just pass through taint.
    """
    # AND EAX, 0x0F0F -> 25 0f 0f 00 00
    bytestring = bytes.fromhex('250f0f0000')
    arch = Architecture.AMD64
    state_format = [
        Register(name='RAX', bits=64),
        Register(name='RBX', bits=64),
    ]

    rule = generate_static_rule(arch, bytestring, state_format)

    # Should have assignments for RAX
    assert len(rule.assignments) > 0

    # Find the RAX assignment
    rax_assignment = None
    for assignment in rule.assignments:
        if assignment.target.name == 'RAX':
            rax_assignment = assignment
            break

    assert rax_assignment is not None, 'Should have RAX assignment'

    # The expression should be InstructionCellExpr (using cell simulation)
    expr = rax_assignment.expression
    assert isinstance(
        expr, InstructionCellExpr,
    ), f'Expected InstructionCellExpr for AND with constant, got {type(expr)}'


def test_mov_constant_clears_taint() -> None:
    """
    Test that MOV with a constant generates an assignment that clears taint.

    Fix: Instructions with no register dependencies now generate InstructionCellExpr with empty inputs.
    Expected: MOV EAX, 0x3C should produce InstructionCellExpr with no inputs, evaluating to 0 taint.
    """
    # MOV EAX, 0x3C -> b8 3c 00 00 00
    bytestring = bytes.fromhex('b83c000000')
    arch = Architecture.AMD64
    state_format = [
        Register(name='RAX', bits=64),
        Register(name='RCX', bits=64),
    ]

    rule = generate_static_rule(arch, bytestring, state_format)

    # Should generate at least one assignment for RAX
    assert len(rule.assignments) > 0, 'MOV constant should generate assignments'

    # Find the RAX assignment
    rax_assignment = None
    for assignment in rule.assignments:
        if assignment.target.name == 'RAX':
            rax_assignment = assignment
            break

    assert rax_assignment is not None, 'Should have RAX assignment'

    # The expression should be InstructionCellExpr with no inputs
    expr = rax_assignment.expression
    assert isinstance(expr, InstructionCellExpr), f'Expected InstructionCellExpr, got {type(expr)}'
    assert len(expr.inputs) == 0, f'MOV constant should have no register inputs, got {list(expr.inputs.keys())}'


def test_shr_bit_precision() -> None:
    """
    Test that SHR (shift right) correctly shifts taint bits.

    Expected: SHR EAX, 4 on taint 0xFFFF should produce 0x0FFF.
    """
    # SHR EAX, 4 -> c1 e8 04
    bytestring = bytes.fromhex('c1e804')
    arch = Architecture.AMD64
    state_format = [
        Register(name='RAX', bits=64),
    ]

    rule = generate_static_rule(arch, bytestring, state_format)

    # Should have assignments for RAX
    assert len(rule.assignments) > 0

    # Find the RAX assignment
    rax_assignment = None
    for assignment in rule.assignments:
        if assignment.target.name == 'RAX' and assignment.target.bit_end == 31:
            rax_assignment = assignment
            break

    assert rax_assignment is not None, 'Should have RAX assignment'

    # Should be a differential without transport term
    expr = rax_assignment.expression
    assert isinstance(expr, BinaryExpr), f'Expected BinaryExpr, got {type(expr)}'
    assert expr.op == Op.XOR, f'Expected XOR, got {expr.op}'


def test_or_bitwise_uses_cell() -> None:
    """
    Test that bitwise OR uses cell simulation when appropriate.

    Expected: OR EAX, EBX should use cell simulation for bit-precise propagation.
    """
    # OR EAX, EBX -> 09 d8
    bytestring = bytes.fromhex('09d8')
    arch = Architecture.AMD64
    state_format = [
        Register(name='RAX', bits=64),
        Register(name='RBX', bits=64),
    ]

    rule = generate_static_rule(arch, bytestring, state_format)

    # Should have assignments for RAX
    assert len(rule.assignments) > 0

    # Find the RAX assignment
    rax_assignment = None
    for assignment in rule.assignments:
        if assignment.target.name == 'RAX' and assignment.target.bit_end == 31:
            rax_assignment = assignment
            break

    assert rax_assignment is not None, 'Should have RAX assignment'

    # For bitwise OR with registers, should use InstructionCellExpr
    expr = rax_assignment.expression
    assert isinstance(expr, InstructionCellExpr), f'Expected InstructionCellExpr for bitwise OR, got {type(expr)}'


def test_xor_reg_reg_uses_cell() -> None:
    """
    Test that XOR reg, reg uses cell simulation.

    Expected: XOR EBX, EBX should use InstructionCellExpr and produce 0 taint.
    """
    # XOR EBX, EBX -> 31 db
    bytestring = bytes.fromhex('31db')
    arch = Architecture.AMD64
    state_format = [
        Register(name='RBX', bits=64),
    ]

    rule = generate_static_rule(arch, bytestring, state_format)

    # Should have assignments for RBX
    assert len(rule.assignments) > 0

    # Find the RBX assignment
    rbx_assignment = None
    for assignment in rule.assignments:
        if assignment.target.name == 'RBX':
            rbx_assignment = assignment
            break

    assert rbx_assignment is not None, 'Should have RBX assignment'

    # Should use InstructionCellExpr
    expr = rbx_assignment.expression
    assert isinstance(expr, InstructionCellExpr), f'Expected InstructionCellExpr for XOR, got {type(expr)}'
