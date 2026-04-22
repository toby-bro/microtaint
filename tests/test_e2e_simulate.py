"""
End-to-End Taint Propagation Tests

These tests validate the full pipeline:
1. Pypcode lifts the instruction and generates the formal LogicCircuit (AST).
2. The AST is evaluated using concrete values and taints.
3. Complex nodes (InstructionCellExpr) dynamically fall back to the CellSimulator.
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
        Register(name='ZF', bits=8),
        Register(name='CF', bits=8),
        Register(name='SF', bits=8),
        Register(name='OF', bits=8),
    ]


def test_e2e_and_constant_clears_taint(amd64_registers: list[Register]) -> None:
    """
    Test exact un-tainting with AND.
    If we AND EAX with 0x0F0F, bits that are 0 in the mask should lose their taint.
    """
    arch = Architecture.AMD64
    # AND EAX, 0x0F0F -> 25 0f 0f 00 00
    bytestring = bytes.fromhex('250f0f0000')

    # 1. Generate Logic Circuit statically via pypcode
    circuit = generate_static_rule(arch, bytestring, amd64_registers)

    # 2. Setup concrete state (Values) and taint state (Taint)
    # Value: 0xFFFF
    # Taint: Fully tainted (0xFFFF)
    input_values = {'RAX': 0xFFFF}
    input_taint = {'RAX': 0xFFFF}

    # 3. Setup Simulator and Context
    simulator = CellSimulator(arch)
    context = EvalContext(input_taint=input_taint, input_values=input_values, simulator=simulator)

    # 4. Evaluate End-to-End
    output_taint = circuit.evaluate(context)

    # 5. Verify the cell simulator successfully masked the taint
    # Only the bits where the mask was 1 (0x0F0F) should remain tainted.
    assert output_taint['RAX'] == 0x0F0F, f"Expected 0x0f0f, got {hex(output_taint.get('RAX', 0))}"


def test_e2e_add_carry_propagation(amd64_registers: list[Register]) -> None:
    """
    Test that ADD correctly propagates taint through the carry chain via the simulator.
    """
    arch = Architecture.AMD64
    # ADD RAX, RBX -> 48 01 d8
    bytestring = bytes.fromhex('4801d8')

    circuit = generate_static_rule(arch, bytestring, amd64_registers)

    # Values: 0xFF + 0x01 = 0x100 (Causes a carry to flip bit 8)
    input_values = {'RAX': 0xFF, 'RBX': 0x01}

    # Taint: Only bit 0 of RAX is tainted
    input_taint = {'RAX': 0x1, 'RBX': 0x0}

    simulator = CellSimulator(arch)
    context = EvalContext(input_taint=input_taint, input_values=input_values, simulator=simulator)

    output_taint = circuit.evaluate(context)

    # Because of the carry chain (0xFF + 1), the single tainted bit 0
    # ripples all the way up to bit 8. Bits 0-8 should be tainted.
    assert output_taint['RAX'] == 0x1FF, f"Expected 0x1ff, got {hex(output_taint.get('RAX', 0))}"


def test_e2e_shift_left(amd64_registers: list[Register]) -> None:
    """
    Test that SHL accurately moves the tainted bits.
    """
    arch = Architecture.AMD64
    # SHL RAX, 4 -> 48 c1 e0 04
    bytestring = bytes.fromhex('48c1e004')

    circuit = generate_static_rule(arch, bytestring, amd64_registers)

    # Values: 0x000F
    input_values = {'RAX': 0x000F}

    # Taint: 0x000F
    input_taint = {'RAX': 0x000F}

    simulator = CellSimulator(arch)
    context = EvalContext(input_taint=input_taint, input_values=input_values, simulator=simulator)

    output_taint = circuit.evaluate(context)

    # The taint should be shifted left by 4 bits exactly like the value
    assert output_taint['RAX'] == 0x00F0, f"Expected 0x00f0, got {hex(output_taint.get('RAX', 0))}"


def test_e2e_xor_clears_taint(amd64_registers: list[Register]) -> None:
    """
    Test that XORing a register with itself clears the taint perfectly.
    """
    arch = Architecture.AMD64
    # XOR RAX, RAX -> 48 31 c0
    bytestring = bytes.fromhex('4831c0')

    circuit = generate_static_rule(arch, bytestring, amd64_registers)

    # Values: Any
    input_values = {'RAX': 0xDEADBEEF}

    # Taint: Fully tainted
    input_taint = {'RAX': 0xFFFFFFFFFFFFFFFF}

    simulator = CellSimulator(arch)
    context = EvalContext(input_taint=input_taint, input_values=input_values, simulator=simulator)

    output_taint = circuit.evaluate(context)

    # XORing a register with itself always yields 0, so output taint MUST be 0.
    assert output_taint['RAX'] == 0x0, f"Expected 0x0, got {hex(output_taint.get('RAX', 0))}"


def test_e2e_and_partial_masking(amd64_registers: list[Register]) -> None:
    """
    Test exact bit-masking with AND.
    Instruction: AND EAX, 0x0F0F0F0F

    If input is fully tainted, only the bits where the constant mask is 1
    should remain tainted, because bits ANDed with 0 become a constant 0.
    """
    arch = Architecture.AMD64
    # AND EAX, 0x0F0F0F0F -> 25 0f 0f 0f 0f
    bytestring = bytes.fromhex('250f0f0f0f')

    circuit = generate_static_rule(arch, bytestring, amd64_registers)

    # Value: Fully populated
    input_values = {'RAX': 0xFFFFFFFF}
    # Taint: Fully tainted
    input_taint = {'RAX': 0xFFFFFFFF}

    simulator = CellSimulator(arch)
    context = EvalContext(input_taint=input_taint, input_values=input_values, simulator=simulator)

    output_taint = circuit.evaluate(context)

    # 0xFFFFFFFF AND 0x0F0F0F0F.
    # Bits 4-7, 12-15, etc. are forced to 0, losing their taint.
    # Bits 0-3, 8-11, etc. pass through, keeping their taint.
    expected_taint = 0x0F0F0F0F
    assert (
        output_taint['RAX'] == expected_taint
    ), f"Expected {hex(expected_taint)}, got {hex(output_taint.get('RAX', 0))}"


def test_e2e_or_partial_masking(amd64_registers: list[Register]) -> None:
    """
    Test exact bit-masking with OR.
    Instruction: OR EAX, 0x0F0F0F0F

    If input is fully tainted, bits ORed with 1 become a constant 1 (losing taint).
    Bits ORed with 0 pass the original value through (keeping taint).
    """
    arch = Architecture.AMD64
    # OR EAX, 0x0F0F0F0F -> 0d 0f 0f 0f 0f
    bytestring = bytes.fromhex('0d0f0f0f0f')

    circuit = generate_static_rule(arch, bytestring, amd64_registers)

    # Value: All zeroes
    input_values = {'RAX': 0x00000000}
    # Taint: Fully tainted
    input_taint = {'RAX': 0xFFFFFFFF}

    simulator = CellSimulator(arch)
    context = EvalContext(input_taint=input_taint, input_values=input_values, simulator=simulator)

    output_taint = circuit.evaluate(context)

    # Let's trace the simulator's math: C(V | T) ^ C(V & ~T)
    # V | T    = 0xFFFFFFFF.   C(0xFFFFFFFF) = 0xFFFFFFFF | 0x0F0F0F0F = 0xFFFFFFFF
    # V & ~T   = 0x00000000.   C(0x00000000) = 0x00000000 | 0x0F0F0F0F = 0x0F0F0F0F
    # 0xFFFFFFFF ^ 0x0F0F0F0F = 0xF0F0F0F0
    #
    # The taint perfectly inverted the mask! Bits forced to 1 lost taint.
    expected_taint = 0xF0F0F0F0
    assert (
        output_taint['RAX'] == expected_taint
    ), f"Expected {hex(expected_taint)}, got {hex(output_taint.get('RAX', 0))}"
