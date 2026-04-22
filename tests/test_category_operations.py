"""
Bit-Precise Category Operation Tests

These tests validate that the taint engine correctly categorizes instructions
and applies the exact theoretical formulas for Mapped, Monotonic,
Conditionally Transportable, Transportable, and Avalanche categories.
"""

from __future__ import annotations

import pytest

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, Register


@pytest.fixture
def amd64_registers() -> list[Register]:
    return [
        Register(name='RAX', bits=64),
        Register(name='RBX', bits=64),
        Register(name='RCX', bits=64),
        Register(name='RDX', bits=64),
        Register(name='ZF', bits=1),
        Register(name='CF', bits=1),
        Register(name='SF', bits=1),
        Register(name='OF', bits=1),
    ]


def test_category_mapped_negate(amd64_registers: list[Register]) -> None:
    """
    Category: MAPPED (Bitwise NOT)
    Instruction: NOT RAX -> 48 f7 d0

    Rule: Exact 1-to-1 bit mapping. Taint simply passes through directly.
    """
    arch = Architecture.AMD64
    bytestring = bytes.fromhex('48f7d0')

    circuit = generate_static_rule(arch, bytestring, amd64_registers)
    simulator = CellSimulator(arch)

    # Value: 0x0F0F
    # Taint: 0x00FF (Only the lower 8 bits are tainted)
    context = EvalContext(
        input_values={'RAX': 0x0F0F},
        input_taint={'RAX': 0x00FF},
        simulator=simulator,
    )

    output_taint = circuit.evaluate(context)

    # NOT flips the value bits, but the taint strictly stays at the lower 8 bits.
    assert output_taint['RAX'] == 0x00FF, f"Expected 0x00FF, got {hex(output_taint.get('RAX', 0))}"


def test_category_mapped_sext(amd64_registers: list[Register]) -> None:
    """
    Category: MAPPED (Sign Extension)
    Instruction: MOVSXD RAX, EBX -> 48 63 c3

    Rule: Sign extension copies the sign bit to all upper 32 bits.
    If the sign bit is tainted, the upper 32 bits MUST become tainted.
    """
    arch = Architecture.AMD64
    bytestring = bytes.fromhex('4863c3')

    circuit = generate_static_rule(arch, bytestring, amd64_registers)
    simulator = CellSimulator(arch)

    # Value: 0x80000000 (Negative 32-bit integer, sign bit is 1)
    # Taint: 0x80000000 (Only the 31st bit - the sign bit - is tainted)
    context = EvalContext(
        input_values={'RBX': 0x80000000},
        input_taint={'RBX': 0x80000000},
        simulator=simulator,
    )

    output_taint = circuit.evaluate(context)

    # Because RBX is sign-extended into RAX, if bit 31 flips to 0,
    # bits 32-63 will all flip from 1 to 0. Thus, they are all tainted.
    expected = 0xFFFFFFFF80000000
    assert output_taint['RAX'] == expected, f"Expected {hex(expected)}, got {hex(output_taint.get('RAX', 0))}"


def test_category_cond_transportable_equality(amd64_registers: list[Register]) -> None:
    """
    Category: COND_TRANSPORTABLE (Equality checking)
    Instruction: CMP EAX, EBX -> 39 d8 (Checks equality via ZF)

    Rule: Equality is only conditionally transportable if the untainted bits match.
    """
    arch = Architecture.AMD64
    bytestring = bytes.fromhex('39d8')

    circuit = generate_static_rule(arch, bytestring, amd64_registers)
    simulator = CellSimulator(arch)

    # Scenario A: The untainted bits match.
    # Value: EAX=0x10, EBX=0x10 (ZF = 1)
    # Taint: EAX bit 0 is tainted.
    # If EAX bit 0 flips, EAX=0x11, EBX=0x10 (ZF flips to 0). Thus, ZF is tainted.
    ctx_match = EvalContext(
        input_values={'RAX': 0x10, 'RBX': 0x10},
        input_taint={'RAX': 0x01, 'RBX': 0x00},
        simulator=simulator,
    )
    out_match = circuit.evaluate(ctx_match)
    assert out_match['ZF'] == 1, 'ZF should be tainted because untainted bits match.'

    # Scenario B: The untainted bits DO NOT match.
    # Value: EAX=0x10, EBX=0x20 (ZF = 0)
    # Taint: EAX bit 0 is tainted.
    # If EAX bit 0 flips, EAX=0x11, EBX=0x20 (ZF remains 0). Thus, ZF is NOT tainted.
    ctx_no_match = EvalContext(
        input_values={'RAX': 0x10, 'RBX': 0x20},
        input_taint={'RAX': 0x01, 'RBX': 0x00},
        simulator=simulator,
    )
    out_no_match = circuit.evaluate(ctx_no_match)
    assert out_no_match['ZF'] == 0, 'ZF should NOT be tainted because untainted bits differ.'


def test_category_monotonic_less_than(amd64_registers: list[Register]) -> None:
    """
    Category: MONOTONIC (Comparisons for <, >, <=, >=)
    Instruction: CMP EAX, EBX -> 39 d8 (Checks Less-Than via CF)

    Rule: Output taint relies on the monotonic bounds check of the inputs.
    """
    arch = Architecture.AMD64
    bytestring = bytes.fromhex('39d8')

    circuit = generate_static_rule(arch, bytestring, amd64_registers)
    simulator = CellSimulator(arch)

    # Value: EAX=0x10, EBX=0x20 (0x10 < 0x20, so CF = 1)
    # Taint: EAX bit 5 is tainted (0x20).
    # If bit 5 flips, EAX becomes 0x30. (0x30 < 0x20 is FALSE, CF flips to 0).
    context = EvalContext(
        input_values={'RAX': 0x10, 'RBX': 0x20},
        input_taint={'RAX': 0x20, 'RBX': 0x00},
        simulator=simulator,
    )

    output_taint = circuit.evaluate(context)

    # Because flipping the tainted bit changes the less-than truth value, CF is tainted.
    assert output_taint['CF'] == 1, 'CF should be tainted because the monotonic bound was crossed.'


def test_category_transportable_subtraction(amd64_registers: list[Register]) -> None:
    """
    Category: TRANSPORTABLE (Arithmetic Add/Sub)
    Instruction: SUB RAX, RBX -> 48 29 d8

    Rule: Arithmetic carries/borrows cause taint to ripple up the bit chain.
    """
    arch = Architecture.AMD64
    bytestring = bytes.fromhex('4829d8')

    circuit = generate_static_rule(arch, bytestring, amd64_registers)
    simulator = CellSimulator(arch)

    # Value: 0x10 - 0x01 = 0x0F
    # Taint: RBX bit 0 is tainted.
    # If RBX bit 0 flips to 0: 0x10 - 0x00 = 0x10.
    # The output flips from 0x0F (01111 in binary) to 0x10 (10000 in binary).
    context = EvalContext(
        input_values={'RAX': 0x10, 'RBX': 0x01},
        input_taint={'RAX': 0x00, 'RBX': 0x01},
        simulator=simulator,
    )

    output_taint = circuit.evaluate(context)

    # Bits 0, 1, 2, 3, and 4 all flipped. Therefore, all 5 lower bits are tainted!
    expected = 0x1F
    assert output_taint['RAX'] == expected, f"Expected {hex(expected)}, got {hex(output_taint.get('RAX', 0))}"


def test_category_avalanche_imul(amd64_registers: list[Register]) -> None:
    """
    Category: AVALANCHE (Multiplication)
    Instruction: IMUL RAX, RBX -> 48 0f af c3

    Rule: Multiplications are too complex to track bit-precisely. Any tainted
    input bit causes the entire output register to become completely tainted (-1).
    """
    arch = Architecture.AMD64
    bytestring = bytes.fromhex('480fafc3')

    circuit = generate_static_rule(arch, bytestring, amd64_registers)
    simulator = CellSimulator(arch)

    # Value: 0x2 * 0x3 = 0x6
    # Taint: RAX bit 0 is tainted.
    context = EvalContext(
        input_values={'RAX': 0x02, 'RBX': 0x03},
        input_taint={'RAX': 0x01, 'RBX': 0x00},
        simulator=simulator,
    )

    output_taint = circuit.evaluate(context)

    # Because it is categorized as AVALANCHE, the AST applies the `AvalancheExpr`,
    # which evaluates to all 1s (0xFFFFFFFFFFFFFFFF) if any dependency is non-zero.
    expected = 0xFFFFFFFFFFFFFFFF
    # Force 64-bit unsigned comparison
    assert output_taint['RAX'] & 0xFFFFFFFFFFFFFFFF == expected, 'Avalanche should fully taint the 64-bit output.'
