"""
Bit-Precise Flag Propagation Tests.
Verifies that CF, OF, SF, ZF, and PF are computed perfectly using CellIFT
differentials without over-tainting or python integer bleed.
"""

from __future__ import annotations

import pytest

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, Register


@pytest.fixture
def x86_flags_registers() -> list[Register]:
    return [
        Register(name='EAX', bits=32),
        Register(name='EBX', bits=32),
        Register(name='CF', bits=1),
        Register(name='PF', bits=1),
        Register(name='ZF', bits=1),
        Register(name='SF', bits=1),
        Register(name='OF', bits=1),
    ]


def test_flag_carry_cf(x86_flags_registers: list[Register]) -> None:
    arch = Architecture.X86
    code = bytes.fromhex('01d8')  # ADD EAX, EBX
    circuit = generate_static_rule(arch, code, x86_flags_registers)
    simulator = CellSimulator(arch)

    # POSITIVE: EAX is at max unsigned (0xFFFFFFFF). EBX is 0.
    # If tainted bit 0 flips to 1, an overflow occurs (0xFFFFFFFF + 1 = 0)! CF flips.
    ctx_tainted = EvalContext(
        input_values={'EAX': 0xFFFFFFFF, 'EBX': 0x0},
        input_taint={'EAX': 0x0, 'EBX': 0x1},
        simulator=simulator,
    )
    assert circuit.evaluate(ctx_tainted).get('CF', 0) == 1, 'CF should flip (0 -> 1)'

    # NEGATIVE: EAX is 0x10. EBX is 0.
    # If tainted bit 0 flips to 1, 0x10 + 1 = 0x11. No overflow. CF remains 0.
    ctx_clean = EvalContext(
        input_values={'EAX': 0x10, 'EBX': 0x0},
        input_taint={'EAX': 0x0, 'EBX': 0x1},
        simulator=simulator,
    )
    assert circuit.evaluate(ctx_clean).get('CF', 0) == 0, 'CF cannot flip, taint should be 0'


def test_flag_overflow_of(x86_flags_registers: list[Register]) -> None:
    arch = Architecture.X86
    code = bytes.fromhex('01d8')  # ADD EAX, EBX
    circuit = generate_static_rule(arch, code, x86_flags_registers)
    simulator = CellSimulator(arch)

    # POSITIVE: EAX is max signed positive (0x7FFFFFFF). EBX is 0.
    # If tainted bit 0 flips to 1, signed overflow occurs (becomes negative). OF flips.
    ctx_tainted = EvalContext(
        input_values={'EAX': 0x7FFFFFFF, 'EBX': 0x0},
        input_taint={'EAX': 0x0, 'EBX': 0x1},
        simulator=simulator,
    )
    assert circuit.evaluate(ctx_tainted).get('OF', 0) == 1, 'OF should flip (0 -> 1)'

    # NEGATIVE: Flipping bit 0 of 0x10 doesn't trigger signed overflow.
    ctx_clean = EvalContext(
        input_values={'EAX': 0x10, 'EBX': 0x0},
        input_taint={'EAX': 0x0, 'EBX': 0x1},
        simulator=simulator,
    )
    assert circuit.evaluate(ctx_clean).get('OF', 0) == 0, 'OF cannot flip, taint should be 0'


def test_flag_sign_sf(x86_flags_registers: list[Register]) -> None:
    arch = Architecture.X86
    code = bytes.fromhex('29d8')  # SUB EAX, EBX
    circuit = generate_static_rule(arch, code, x86_flags_registers)
    simulator = CellSimulator(arch)

    # POSITIVE: EAX is 0x0. EBX is 0x0.
    # If tainted bit 0 flips to 1, EAX becomes -1 (0xFFFFFFFF), flipping the sign bit!
    ctx_tainted = EvalContext(
        input_values={'EAX': 0x0, 'EBX': 0x0},
        input_taint={'EAX': 0x0, 'EBX': 0x1},
        simulator=simulator,
    )
    assert circuit.evaluate(ctx_tainted).get('SF', 0) == 1, 'SF should flip (0 -> 1)'

    # NEGATIVE: EAX is 0x10. EBX is 0x0.
    # 0x10 - 1 = 0x0F. Still positive. SF does not flip.
    ctx_clean = EvalContext(
        input_values={'EAX': 0x10, 'EBX': 0x0},
        input_taint={'EAX': 0x0, 'EBX': 0x1},
        simulator=simulator,
    )
    assert circuit.evaluate(ctx_clean).get('SF', 0) == 0, 'SF cannot flip, taint should be 0'


def test_flag_zero_zf_multi_taint(x86_flags_registers: list[Register]) -> None:
    arch = Architecture.X86
    code = bytes.fromhex('39d8')  # CMP EAX, EBX
    circuit = generate_static_rule(arch, code, x86_flags_registers)
    simulator = CellSimulator(arch)

    # POSITIVE: EAX=0x10, EBX=0x10. (ZF is currently 1)
    # If multiple tainted bits flip, they are no longer equal. ZF flips.
    ctx_tainted = EvalContext(
        input_values={'EAX': 0x10, 'EBX': 0x10},
        input_taint={'EAX': 0x0, 'EBX': 0x0F},
        simulator=simulator,
    )
    assert circuit.evaluate(ctx_tainted).get('ZF', 0) == 1, 'ZF should flip (1 -> 0)'

    # NEGATIVE: EAX=0x10, EBX=0x20. (ZF is 0)
    # The tainted bits are 0x0F. EBX can range from 0x20 to 0x2F.
    # It can NEVER equal 0x10. Therefore, ZF CANNOT flip! Taint is 0.
    ctx_clean = EvalContext(
        input_values={'EAX': 0x10, 'EBX': 0x20},
        input_taint={'EAX': 0x0, 'EBX': 0x0F},
        simulator=simulator,
    )
    assert circuit.evaluate(ctx_clean).get('ZF', 0) == 0, 'ZF mathematically cannot flip here'


def test_flag_parity_pf_avalanche(x86_flags_registers: list[Register]) -> None:
    arch = Architecture.X86
    code = bytes.fromhex('85d8')  # TEST EAX, EBX
    circuit = generate_static_rule(arch, code, x86_flags_registers)
    simulator = CellSimulator(arch)

    # Parity uses POPCOUNT (AVALANCHE). ANY taint in operands -> 1-bit taint on PF.
    ctx_tainted = EvalContext(
        input_values={'EAX': 0x0, 'EBX': 0x0},
        input_taint={'EAX': 0x1, 'EBX': 0x0},
        simulator=simulator,
    )

    output = circuit.evaluate(ctx_tainted)
    # The output MUST be strictly clamped to 1 by the mapper fix, not 255.
    assert output.get('PF', 0) == 1, f"PF should be exactly 1, got {output.get('PF', 0)}"
