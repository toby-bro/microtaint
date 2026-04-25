"""
Bit-Precise Addition and Carry Ripple Tests.

Verifies that the CellIFT Transportable formula correctly models carry ripples,
carry halting, and prevents taint cancellation when multiple inputs are tainted.
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
    ]


def test_add_no_carry(amd64_registers: list[Register]) -> None:
    """
    Test: 0x10 + 0x01
    No carry occurs. Taint on bit 0 should stay strictly on bit 0.
    """
    arch = Architecture.AMD64
    code = bytes.fromhex('4801d8')  # ADD RAX, RBX
    circuit = generate_static_rule(arch, code, amd64_registers)
    simulator = CellSimulator(arch)

    ctx = EvalContext(
        input_values={'RAX': 0x10, 'RBX': 0x01},
        input_taint={'RAX': 0x00, 'RBX': 0x01},
        simulator=simulator,
    )
    out = circuit.evaluate(ctx)

    # 0x10 + 1 = 0x11. 0x10 + 0 = 0x10. Diff: 0x11 ^ 0x10 = 0x1.
    assert out.get('RAX', 0) == 0x01, 'Taint should not spread without a carry.'


def test_add_one_carry_ripple(amd64_registers: list[Register]) -> None:
    """
    Test: 0x0F + 0x01
    A single tainted bit causes a ripple carry across multiple bits, stopping at bit 4.
    """
    arch = Architecture.AMD64
    code = bytes.fromhex('4801d8')  # ADD RAX, RBX
    circuit = generate_static_rule(arch, code, amd64_registers)
    simulator = CellSimulator(arch)

    ctx = EvalContext(
        input_values={'RAX': 0x0F, 'RBX': 0x01},
        input_taint={'RAX': 0x00, 'RBX': 0x01},
        simulator=simulator,
    )
    out = circuit.evaluate(ctx)

    # V1: 0x0F + 1 = 0x10.
    # V0: 0x0F + 0 = 0x0F.
    # Diff: 0x10 ^ 0x0F = 0x1F (Bits 0, 1, 2, 3, and 4 all flipped).
    assert out.get('RAX', 0) == 0x1F, 'Taint should ripple exactly up to bit 4.'


def test_add_segmented_carries(amd64_registers: list[Register]) -> None:
    """
    Test: 0x080F + 0x0101 (where only the lowest bit of RBX is tainted).
    The ripple from the lowest bit stops at bit 4.
    The unrelated carry happening at bit 8 is NOT affected by the taint and should remain clean.
    """
    arch = Architecture.AMD64
    code = bytes.fromhex('4801d8')  # ADD RAX, RBX
    circuit = generate_static_rule(arch, code, amd64_registers)
    simulator = CellSimulator(arch)

    ctx = EvalContext(
        input_values={'RAX': 0x080F, 'RBX': 0x0101},
        input_taint={'RAX': 0x0000, 'RBX': 0x0001},  # Only the lowest 1 is tainted
        simulator=simulator,
    )
    out = circuit.evaluate(ctx)

    # V1: 0x080F + 0x0101 = 0x0910
    # V0: 0x080F + 0x0100 = 0x090F (Notice the untainted 0x0100 remains constant)
    # Diff: 0x0910 ^ 0x090F = 0x001F.
    # The output taint stops at bit 4. Bits 8-11 are safely untainted because
    # that carry was deterministic and not influenced by the tainted bit.
    assert out.get('RAX', 0) == 0x001F, 'Taint should not jump across separate carry segments.'


def test_add_cancellation_prevention(amd64_registers: list[Register]) -> None:
    """
    Test: 0x01 + 0x01 (Both inputs tainted at bit 0).
    Proves that the Transportable category correctly adds the Transport Term
    (T_RAX | T_RBX) to prevent Taint Cancellation.
    """
    arch = Architecture.AMD64
    code = bytes.fromhex('4801d8')  # ADD RAX, RBX
    circuit = generate_static_rule(arch, code, amd64_registers)
    simulator = CellSimulator(arch)

    ctx = EvalContext(
        input_values={'RAX': 0x01, 'RBX': 0x01},
        input_taint={'RAX': 0x01, 'RBX': 0x01},
        simulator=simulator,
    )
    out = circuit.evaluate(ctx)

    # Differential: (1+1=2) ^ (0+0=0) = 2 (0b10).
    # Transport Term: T_RAX(1) | T_RBX(1) = 1 (0b01).
    # Result: 2 | 1 = 3 (0b11).
    # Both bit 0 (from the inputs) and bit 1 (from the carry) must be tainted.
    assert out.get('RAX', 0) == 0x03, 'Transport term should prevent taint cancellation on bit 0.'
