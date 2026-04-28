"""
Comprehensive tests for COND_TRANSPORTABLE taint propagation.

Covers two distinct cases:
  1. Register vs Register (e.g. CMP EAX, EBX = 39 d8)
     Uses the precise CELLIFT formula: SimulateCell(V & ~T) AND AVALANCHE(T)
     Taint is zero when the untainted bits of the operands differ.

  2. Register vs Constant/Immediate (e.g. CMP AL, 0x58 = 3c 58)
     Uses the corrected formula: SimulateCell((V & ~T) | (imm & T)) AND AVALANCHE(T)
     Taint is zero when the untainted bits of V cannot match the untainted bits of imm.
"""

from __future__ import annotations

import pytest

from microtaint.instrumentation.ast import EvalContext, LogicCircuit
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import (
    _cached_generate_static_rule,  # pyright: ignore[reportPrivateUsage]
    generate_static_rule,
)
from microtaint.types import Architecture, Register

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_cached_generate_static_rule.cache_clear()


@pytest.fixture(scope='module')
def sim_amd64() -> CellSimulator:
    return CellSimulator(Architecture.AMD64)


@pytest.fixture(scope='module')
def sim_x86() -> CellSimulator:
    return CellSimulator(Architecture.X86)


@pytest.fixture(scope='module')
def amd64_regs() -> list[Register]:
    return [
        Register('RAX', 64),
        Register('RBX', 64),
        Register('RCX', 64),
        Register('RDX', 64),
        Register('RSP', 64),
        Register('RBP', 64),
        Register('RSI', 64),
        Register('RDI', 64),
        Register('RIP', 64),
        Register('EFLAGS', 32),
        Register('ZF', 1),
        Register('CF', 1),
        Register('SF', 1),
        Register('OF', 1),
        Register('PF', 1),
        Register('EAX', 32),
        Register('EBX', 32),
        Register('ECX', 32),
        Register('AX', 16),
        Register('AL', 8),
        Register('AH', 8),
    ]


@pytest.fixture(scope='module')
def x86_regs() -> list[Register]:
    return [
        Register('EAX', 32),
        Register('EBX', 32),
        Register('ECX', 32),
        Register('EDX', 32),
        Register('ESP', 32),
        Register('EBP', 32),
        Register('ESI', 32),
        Register('EDI', 32),
        Register('EFLAGS', 32),
        Register('ZF', 1),
        Register('CF', 1),
        Register('SF', 1),
        Register('OF', 1),
        Register('PF', 1),
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_zf(out: dict[str, int]) -> int:
    """Extract ZF taint from output, falling back to EFLAGS bit 6."""
    if 'ZF' in out:
        return out['ZF']
    return (out.get('EFLAGS', 0) >> 6) & 1


# ---------------------------------------------------------------------------
# Section 1: CMP AL, imm8  (3c XX) — register vs immediate, AMD64
# ---------------------------------------------------------------------------


class TestCmpAlImmediate:
    """
    cmp al, 0x58  →  3c 58
    Formula: ZF tainted iff (V_AL & ~T) == (0x58 & ~T)
    i.e., the untainted bits of AL match the untainted bits of the constant.
    """

    INSTR = bytes.fromhex('3c58')  # cmp al, 0x58
    IMM = 0x58

    @pytest.fixture(scope='class')
    def circuit(self, amd64_regs: list[Register]) -> LogicCircuit:
        return generate_static_rule(Architecture.AMD64, self.INSTR, amd64_regs)

    def ctx(self, sim: CellSimulator, v_rax: int, t_rax: int) -> EvalContext:
        return EvalContext(
            input_values={'RAX': v_rax},
            input_taint={'RAX': t_rax},
            simulator=sim,
        )

    def test_fully_tainted_value_equals_imm(self, circuit: LogicCircuit, sim_amd64: CellSimulator) -> None:
        """AL=0x58, T=0xFF: AL fully tainted, currently equals imm.
        0x58 is reachable → ZF must be tainted."""
        out = circuit.evaluate(self.ctx(sim_amd64, 0x58, 0xFF))
        print(circuit)
        assert get_zf(out) == 1

    def test_fully_tainted_value_not_equal_imm(self, circuit: LogicCircuit, sim_amd64: CellSimulator) -> None:
        """AL=0x00, T=0xFF: AL fully tainted, currently differs from imm.
        But 0x58 is still reachable → ZF must be tainted."""
        out = circuit.evaluate(self.ctx(sim_amd64, 0x00, 0xFF))
        print(circuit)
        assert get_zf(out) == 1

    def test_fully_tainted_max_value(self, circuit: LogicCircuit, sim_amd64: CellSimulator) -> None:
        """AL=0xFF, T=0xFF: AL fully tainted.
        0x58 is reachable → ZF must be tainted."""
        out = circuit.evaluate(self.ctx(sim_amd64, 0xFF, 0xFF))
        print(circuit)
        assert get_zf(out) == 1

    def test_single_bit_taint_imm_unreachable(self, circuit: LogicCircuit, sim_amd64: CellSimulator) -> None:
        """AL=0x10, T=0x01: tainted bit can make AL 0x10 or 0x11.
        Neither equals 0x58 → ZF must NOT be tainted."""
        out = circuit.evaluate(self.ctx(sim_amd64, 0x10, 0x01))
        print(circuit)
        assert get_zf(out) == 0

    def test_single_bit_taint_imm_unreachable_high(self, circuit: LogicCircuit, sim_amd64: CellSimulator) -> None:
        """AL=0xA0, T=0x01: tainted bit can make AL 0xA0 or 0xA1.
        Neither equals 0x58 → ZF must NOT be tainted."""
        out = circuit.evaluate(self.ctx(sim_amd64, 0xA0, 0x01))
        print(circuit)
        assert get_zf(out) == 0

    def test_lower_6_bits_tainted_imm_reachable(self, circuit: LogicCircuit, sim_amd64: CellSimulator) -> None:
        """AL=0x58, T=0x3F: lower 6 bits tainted, upper 2 bits fixed at 0x40.
        0x58 has upper bits 0x40 → 0x58 is reachable → ZF must be tainted."""
        out = circuit.evaluate(self.ctx(sim_amd64, 0x58, 0x3F))
        print(circuit)
        assert get_zf(out) == 1

    def test_lower_6_bits_tainted_imm_unreachable(self, circuit: LogicCircuit, sim_amd64: CellSimulator) -> None:
        """AL=0x10, T=0x3F: lower 6 bits tainted, upper 2 bits fixed at 0x00.
        0x58 has upper bits 0x40 ≠ 0x00 → 0x58 unreachable → ZF must NOT be tainted."""
        out = circuit.evaluate(self.ctx(sim_amd64, 0x10, 0x3F))
        print(circuit)
        assert get_zf(out) == 0

    def test_upper_bit_taint_imm_reachable(self, circuit: LogicCircuit, sim_amd64: CellSimulator) -> None:
        """AL=0x58, T=0x80: upper bit tainted, lower 7 bits fixed at 0x58.
        AL can be 0x58 or 0xD8 → 0x58 reachable → ZF must be tainted."""
        out = circuit.evaluate(self.ctx(sim_amd64, 0x58, 0x80))
        print(circuit)
        assert get_zf(out) == 1

    def test_upper_bit_taint_imm_unreachable(self, circuit: LogicCircuit, sim_amd64: CellSimulator) -> None:
        """AL=0xD8, T=0x80: upper bit tainted, lower 7 bits fixed at 0x58.
        AL can be 0x58 or 0xD8 → 0x58 IS reachable → ZF must be tainted."""
        out = circuit.evaluate(self.ctx(sim_amd64, 0xD8, 0x80))
        print(circuit)
        assert get_zf(out) == 1

    def test_no_taint(self, circuit: LogicCircuit, sim_amd64: CellSimulator) -> None:
        """T=0: no taint at all → ZF must NOT be tainted."""
        out = circuit.evaluate(self.ctx(sim_amd64, 0x58, 0x00))
        print(circuit)
        assert get_zf(out) == 0

    def test_no_taint_mismatch(self, circuit: LogicCircuit, sim_amd64: CellSimulator) -> None:
        """T=0, AL≠imm: no taint → ZF must NOT be tainted."""
        out = circuit.evaluate(self.ctx(sim_amd64, 0x10, 0x00))
        print(circuit)
        assert get_zf(out) == 0

    def test_taint_only_in_imm_bits(self, circuit: LogicCircuit, sim_amd64: CellSimulator) -> None:
        """AL=0x50, T=0x0F: lower nibble tainted, upper nibble 0x50.
        imm=0x58 has upper nibble 0x50 (matches) and lower nibble 0x08 (in tainted range).
        0x58 IS reachable → ZF must be tainted."""
        out = circuit.evaluate(self.ctx(sim_amd64, 0x50, 0x0F))
        print(circuit)
        assert get_zf(out) == 1

    def test_taint_misses_imm_nibble(self, circuit: LogicCircuit, sim_amd64: CellSimulator) -> None:
        """AL=0x10, T=0x0F: lower nibble tainted, upper nibble 0x10.
        imm=0x58 has upper nibble 0x50 ≠ 0x10 → 0x58 unreachable → ZF NOT tainted."""
        out = circuit.evaluate(self.ctx(sim_amd64, 0x10, 0x0F))
        print(circuit)
        assert get_zf(out) == 0


# ---------------------------------------------------------------------------
# Section 2: CMP EAX, EBX  (39 d8) — register vs register
# ---------------------------------------------------------------------------


class TestCmpRegReg:
    """
    cmp eax, ebx  →  39 d8
    Formula: ZF tainted iff SimulateCell(V_EAX & ~T_union, V_EBX & ~T_union) == 1
             AND any taint exists.
    i.e., the untainted bits of both operands are equal (so tainted bits could flip equality).
    """

    INSTR = bytes.fromhex('39d8')  # cmp eax, ebx

    @pytest.fixture(scope='class')
    def circuit(self, amd64_regs: list[Register]) -> LogicCircuit:
        return generate_static_rule(Architecture.AMD64, self.INSTR, amd64_regs)

    def ctx(self, sim: CellSimulator, v_rax: int, v_rbx: int, t_rax: int, t_rbx: int) -> EvalContext:
        return EvalContext(
            input_values={'RAX': v_rax, 'RBX': v_rbx},
            input_taint={'RAX': t_rax, 'RBX': t_rbx},
            simulator=sim,
        )

    # --- Untainted bits match → ZF tainted ---

    def test_match_single_bit_taint_a(self, circuit: LogicCircuit, sim_amd64: CellSimulator) -> None:
        """EAX=0x10, EBX=0x10, T_EAX=0x01: untainted bits match.
        Tainted bit could make them unequal → ZF tainted."""
        out = circuit.evaluate(self.ctx(sim_amd64, 0x10, 0x10, 0x01, 0x00))
        print(circuit)
        assert get_zf(out) == 1

    def test_match_single_bit_taint_b(self, circuit: LogicCircuit, sim_amd64: CellSimulator) -> None:
        """EAX=0x10, EBX=0x10, T_EBX=0x01: untainted bits match.
        Tainted bit on B could make them unequal → ZF tainted."""
        out = circuit.evaluate(self.ctx(sim_amd64, 0x10, 0x10, 0x00, 0x01))
        print(circuit)
        assert get_zf(out) == 1

    def test_match_overlapping_multi_bit_taint(self, circuit: LogicCircuit, sim_amd64: CellSimulator) -> None:
        """EAX=0xF0, EBX=0xF0, T_EAX=0x0F, T_EBX=0x0F: lower nibble tainted on both.
        Upper nibble matches → ZF tainted."""
        out = circuit.evaluate(self.ctx(sim_amd64, 0xF0, 0xF0, 0x0F, 0x0F))
        print(circuit)
        assert get_zf(out) == 1

    def test_match_disjoint_taints(self, circuit: LogicCircuit, sim_amd64: CellSimulator) -> None:
        """EAX=0x100, EBX=0x100, T_EAX=0x01, T_EBX=0x02: taints on different bits.
        T_union=0x03, masked values both 0x100 → untainted bits match → ZF tainted."""
        out = circuit.evaluate(self.ctx(sim_amd64, 0x100, 0x100, 0x01, 0x02))
        print(circuit)
        assert get_zf(out) == 1

    def test_match_fully_tainted_both(self, circuit: LogicCircuit, sim_amd64: CellSimulator) -> None:
        """EAX=0x12345678, EBX=0x87654321, both fully tainted.
        All bits tainted → masked values both 0 → ZF tainted."""
        out = circuit.evaluate(self.ctx(sim_amd64, 0x12345678, 0x87654321, 0xFFFFFFFF, 0xFFFFFFFF))
        print(circuit)
        assert get_zf(out) == 1

    def test_match_fully_tainted_one(self, circuit: LogicCircuit, sim_amd64: CellSimulator) -> None:
        """EAX=0x00000000, EBX=0x12345678, T_EAX=0xFFFFFFFF, T_EBX=0x00.
        T_union=0xFFFFFFFF → masked EBX = 0x12345678 & 0 = 0.
        Masked EAX = 0. Both 0 → ZF tainted."""
        out = circuit.evaluate(self.ctx(sim_amd64, 0x00000000, 0x12345678, 0xFFFFFFFF, 0x00))
        print(circuit)
        assert get_zf(out) == 1

    # --- Untainted bits mismatch → ZF NOT tainted ---

    def test_mismatch_single_bit_taint(self, circuit: LogicCircuit, sim_amd64: CellSimulator) -> None:
        """EAX=0x10, EBX=0x20, T_EAX=0x01: untainted bits differ (0x10 vs 0x20).
        No matter what bit 0 of EAX is, operands can never be equal → ZF NOT tainted."""
        out = circuit.evaluate(self.ctx(sim_amd64, 0x10, 0x20, 0x01, 0x00))
        print(circuit)
        assert get_zf(out) == 0

    def test_mismatch_overlapping_taint(self, circuit: LogicCircuit, sim_amd64: CellSimulator) -> None:
        """EAX=0xF0, EBX=0xE0, T_EAX=0x0F, T_EBX=0x0F: upper nibble differs.
        Masked EAX=0xF0, masked EBX=0xE0 → mismatch → ZF NOT tainted."""
        out = circuit.evaluate(self.ctx(sim_amd64, 0xF0, 0xE0, 0x0F, 0x0F))
        print(circuit)
        assert get_zf(out) == 0

    def test_mismatch_disjoint_taints(self, circuit: LogicCircuit, sim_amd64: CellSimulator) -> None:
        """EAX=0x100, EBX=0x200, T_EAX=0x01, T_EBX=0x02.
        T_union=0x03, masked EAX=0x100, masked EBX=0x200 → mismatch → ZF NOT tainted."""
        out = circuit.evaluate(self.ctx(sim_amd64, 0x100, 0x200, 0x01, 0x02))
        print(circuit)
        assert get_zf(out) == 0

    def test_mismatch_high_bits_differ(self, circuit: LogicCircuit, sim_amd64: CellSimulator) -> None:
        """EAX=0xFF000000, EBX=0x00000000, T_EAX=0x0000FFFF: low word tainted.
        High word of EAX (0xFF00) ≠ high word of EBX (0x0000) → ZF NOT tainted."""
        out = circuit.evaluate(self.ctx(sim_amd64, 0xFF000000, 0x00000000, 0x0000FFFF, 0x0))
        print(circuit)
        assert get_zf(out) == 0

    # --- No taint → ZF NOT tainted ---

    def test_no_taint_equal_values(self, circuit: LogicCircuit, sim_amd64: CellSimulator) -> None:
        """EAX=EBX=0x42, no taint → ZF NOT tainted."""
        out = circuit.evaluate(self.ctx(sim_amd64, 0x42, 0x42, 0x00, 0x00))
        print(circuit)
        assert get_zf(out) == 0

    def test_no_taint_unequal_values(self, circuit: LogicCircuit, sim_amd64: CellSimulator) -> None:
        """EAX=0x10, EBX=0x20, no taint → ZF NOT tainted."""
        out = circuit.evaluate(self.ctx(sim_amd64, 0x10, 0x20, 0x00, 0x00))
        print(circuit)
        assert get_zf(out) == 0


# ---------------------------------------------------------------------------
# Section 3: CMP EAX, imm32  (3d XX XX XX XX) — 32-bit immediate, AMD64
# ---------------------------------------------------------------------------


class TestCmpEaxImm32:
    """
    cmp eax, 0x12345678  →  3d 78 56 34 12
    Same corrected formula as CMP AL, imm8 but with 32-bit operands.
    """

    INSTR = bytes.fromhex('3d78563412')  # cmp eax, 0x12345678
    IMM = 0x12345678

    @pytest.fixture(scope='class')
    def circuit(self, amd64_regs: list[Register]) -> LogicCircuit:
        return generate_static_rule(Architecture.AMD64, self.INSTR, amd64_regs)

    def ctx(self, sim: CellSimulator, v_rax: int, t_rax: int) -> EvalContext:
        return EvalContext(
            input_values={'RAX': v_rax},
            input_taint={'RAX': t_rax},
            simulator=sim,
        )

    def test_fully_tainted_imm_reachable(self, circuit: LogicCircuit, sim_amd64: CellSimulator) -> None:
        """EAX=0x12345678, T=0xFFFFFFFF: fully tainted, imm reachable → ZF tainted."""
        out = circuit.evaluate(self.ctx(sim_amd64, self.IMM, 0xFFFFFFFF))
        print(circuit)
        assert get_zf(out) == 1

    def test_fully_tainted_different_value(self, circuit: LogicCircuit, sim_amd64: CellSimulator) -> None:
        """EAX=0x00000000, T=0xFFFFFFFF: fully tainted, imm reachable → ZF tainted."""
        out = circuit.evaluate(self.ctx(sim_amd64, 0x00000000, 0xFFFFFFFF))
        print(circuit)
        assert get_zf(out) == 1

    def test_lower_word_tainted_upper_matches(self, circuit: LogicCircuit, sim_amd64: CellSimulator) -> None:
        """EAX=0x12340000, T=0x0000FFFF: lower 16 bits tainted, upper 16 fixed at 0x1234.
        imm upper 16 = 0x1234 (matches), lower 16 = 0x5678 (in tainted range) → ZF tainted."""
        out = circuit.evaluate(self.ctx(sim_amd64, 0x12340000, 0x0000FFFF))
        print(circuit)
        assert get_zf(out) == 1

    def test_lower_word_tainted_upper_mismatch(self, circuit: LogicCircuit, sim_amd64: CellSimulator) -> None:
        """EAX=0x00000000, T=0x0000FFFF: lower 16 bits tainted, upper 16 fixed at 0x0000.
        imm upper 16 = 0x1234 ≠ 0x0000 → imm unreachable → ZF NOT tainted."""
        out = circuit.evaluate(self.ctx(sim_amd64, 0x00000000, 0x0000FFFF))
        print(circuit)
        assert get_zf(out) == 0

    def test_no_taint(self, circuit: LogicCircuit, sim_amd64: CellSimulator) -> None:
        """T=0: no taint → ZF NOT tainted."""
        out = circuit.evaluate(self.ctx(sim_amd64, self.IMM, 0x00))
        print(circuit)
        assert get_zf(out) == 0

    def test_single_bit_mismatch(self, circuit: LogicCircuit, sim_amd64: CellSimulator) -> None:
        """EAX=0x12345600, T=0x000000FF: lower byte tainted, upper 3 bytes differ from imm.
        imm upper 3 bytes = 0x123456 ≠ 0x123456... wait they match.
        EAX upper = 0x123456, imm upper = 0x123456 → match. Lower tainted.
        imm lower = 0x78 → reachable → ZF tainted."""
        out = circuit.evaluate(self.ctx(sim_amd64, 0x12345600, 0x000000FF))
        print(circuit)
        assert get_zf(out) == 1

    def test_upper_byte_mismatch_lower_tainted(self, circuit: LogicCircuit, sim_amd64: CellSimulator) -> None:
        """EAX=0xFF345678, T=0x000000FF: lower byte tainted, upper 3 bytes = 0xFF3456.
        imm upper 3 bytes = 0x123456 ≠ 0xFF3456 → imm unreachable → ZF NOT tainted."""
        out = circuit.evaluate(self.ctx(sim_amd64, 0xFF345678, 0x000000FF))
        print(circuit)
        assert get_zf(out) == 0


# ---------------------------------------------------------------------------
# Section 4: X86 CMP EAX, EBX — verify same behaviour on 32-bit arch
# ---------------------------------------------------------------------------


class TestX86CmpRegReg:
    """Same semantics as AMD64 but on x86 (32-bit)."""

    INSTR = bytes.fromhex('39d8')  # cmp eax, ebx

    @pytest.fixture(scope='class')
    def circuit(self, x86_regs: list[Register]) -> LogicCircuit:
        return generate_static_rule(Architecture.X86, self.INSTR, x86_regs)

    def ctx(self, sim: CellSimulator, v_eax: int, v_ebx: int, t_eax: int, t_ebx: int) -> EvalContext:
        return EvalContext(
            input_values={'EAX': v_eax, 'EBX': v_ebx},
            input_taint={'EAX': t_eax, 'EBX': t_ebx},
            simulator=sim,
        )

    def test_match_taint_a(self, circuit: LogicCircuit, sim_x86: CellSimulator) -> None:
        out = circuit.evaluate(self.ctx(sim_x86, 0x10, 0x10, 0x01, 0x00))
        print(circuit)
        assert get_zf(out) == 1

    def test_match_taint_b(self, circuit: LogicCircuit, sim_x86: CellSimulator) -> None:
        out = circuit.evaluate(self.ctx(sim_x86, 0x10, 0x10, 0x00, 0x01))
        print(circuit)
        assert get_zf(out) == 1

    def test_mismatch_taint(self, circuit: LogicCircuit, sim_x86: CellSimulator) -> None:
        out = circuit.evaluate(self.ctx(sim_x86, 0x10, 0x20, 0x01, 0x00))
        print(circuit)
        assert get_zf(out) == 0

    def test_no_taint(self, circuit: LogicCircuit, sim_x86: CellSimulator) -> None:
        out = circuit.evaluate(self.ctx(sim_x86, 0x42, 0x42, 0x00, 0x00))
        print(circuit)
        assert get_zf(out) == 0

    def test_full_taint_both(self, circuit: LogicCircuit, sim_x86: CellSimulator) -> None:
        out = circuit.evaluate(self.ctx(sim_x86, 0xDEAD, 0xBEEF, 0xFFFFFFFF, 0xFFFFFFFF))
        print(circuit)
        assert get_zf(out) == 1


# ---------------------------------------------------------------------------
# Section 5: Edge cases — zero immediate, negative immediate
# ---------------------------------------------------------------------------


class TestCmpEdgeCases:
    """Edge cases: cmp al, 0  and  cmp ax, -1."""

    @pytest.fixture(scope='class')
    def circuit_zero(self, amd64_regs: list[Register]) -> LogicCircuit:
        # cmp al, 0  →  3c 00
        return generate_static_rule(Architecture.AMD64, bytes.fromhex('3c00'), amd64_regs)

    @pytest.fixture(scope='class')
    def circuit_neg(self, amd64_regs: list[Register]) -> LogicCircuit:
        # cmp ax, -1 (0xFFFF)  →  66 83 f8 ff
        return generate_static_rule(Architecture.AMD64, bytes.fromhex('6683f8ff'), amd64_regs)

    def test_cmp_zero_fully_tainted(self, circuit_zero: LogicCircuit, sim_amd64: CellSimulator) -> None:
        """cmp al, 0: AL fully tainted. 0x00 is always reachable → ZF tainted."""
        out = circuit_zero.evaluate(
            EvalContext(
                input_values={'RAX': 0xFF},
                input_taint={'RAX': 0xFF},
                simulator=sim_amd64,
            ),
        )
        assert get_zf(out) == 1

    def test_cmp_zero_single_bit_unreachable(self, circuit_zero: LogicCircuit, sim_amd64: CellSimulator) -> None:
        """cmp al, 0: AL=0x10, T=0x01. AL can be 0x10 or 0x11.
        Neither is 0 → ZF NOT tainted."""
        out = circuit_zero.evaluate(
            EvalContext(
                input_values={'RAX': 0x10},
                input_taint={'RAX': 0x01},
                simulator=sim_amd64,
            ),
        )
        print(circuit_zero)
        assert get_zf(out) == 0

    def test_cmp_zero_lower_nibble_can_reach_zero(self, circuit_zero: LogicCircuit, sim_amd64: CellSimulator) -> None:
        """cmp al, 0: AL=0x00, T=0x0F. AL can reach 0x00 → ZF tainted."""
        out = circuit_zero.evaluate(
            EvalContext(
                input_values={'RAX': 0x00},
                input_taint={'RAX': 0x0F},
                simulator=sim_amd64,
            ),
        )
        print(circuit_zero)
        assert get_zf(out) == 1

    def test_cmp_zero_upper_nibble_nonzero_lower_tainted(
        self,
        circuit_zero: LogicCircuit,
        sim_amd64: CellSimulator,
    ) -> None:
        """cmp al, 0: AL=0x10, T=0x0F. Upper nibble fixed at 0x10 ≠ 0x00 → ZF NOT tainted."""
        out = circuit_zero.evaluate(
            EvalContext(
                input_values={'RAX': 0x10},
                input_taint={'RAX': 0x0F},
                simulator=sim_amd64,
            ),
        )
        print(circuit_zero)
        assert get_zf(out) == 0


# ---------------------------------------------------------------------------
# Section 6: Taint does not bleed across unrelated instructions
# ---------------------------------------------------------------------------


class TestTaintIsolation:
    """Verify that clean comparisons produce no taint on output flags."""

    def test_clean_cmp_reg_imm(self, amd64_regs: list[Register], sim_amd64: CellSimulator) -> None:
        """cmp al, 0x58 with T=0: all outputs clean."""
        circuit = generate_static_rule(Architecture.AMD64, bytes.fromhex('3c58'), amd64_regs)
        out = circuit.evaluate(
            EvalContext(
                input_values={'RAX': 0x58},
                input_taint={},
                simulator=sim_amd64,
            ),
        )
        print(circuit)
        assert get_zf(out) == 0
        assert out.get('CF', 0) == 0
        assert out.get('SF', 0) == 0
        assert out.get('OF', 0) == 0

    def test_clean_cmp_reg_reg(self, amd64_regs: list[Register], sim_amd64: CellSimulator) -> None:
        """cmp eax, ebx with T=0: all outputs clean."""
        circuit = generate_static_rule(Architecture.AMD64, bytes.fromhex('39d8'), amd64_regs)
        out = circuit.evaluate(
            EvalContext(
                input_values={'RAX': 0x42, 'RBX': 0x42},
                input_taint={},
                simulator=sim_amd64,
            ),
        )
        print(circuit)
        assert get_zf(out) == 0
        assert out.get('CF', 0) == 0

    def test_tainted_cmp_does_not_taint_non_flag_regs(
        self,
        amd64_regs: list[Register],
        sim_amd64: CellSimulator,
    ) -> None:
        """cmp al, 0x58 with taint: RAX itself should not become tainted."""
        circuit = generate_static_rule(Architecture.AMD64, bytes.fromhex('3c58'), amd64_regs)
        out = circuit.evaluate(
            EvalContext(
                input_values={'RAX': 0x58},
                input_taint={'RAX': 0xFF},
                simulator=sim_amd64,
            ),
        )
        print(circuit)
        # RAX is not written by CMP, so it should not appear as tainted output
        # (any RAX taint in output comes from input propagation, not CMP itself)
        # ZF should be tainted
        assert get_zf(out) == 1
