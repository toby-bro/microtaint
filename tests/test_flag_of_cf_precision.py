"""
test_flag_of_cf_precision.py
=============================

Tests for the OF/CF flag taint precision fix for MONOTONIC 1-bit outputs.

Root cause
----------
INT_SCARRY (OF from add/adc) and INT_SBORROW (OF from sub/sbb/neg) are
classified MONOTONIC.  The MONOTONIC evaluator uses the differential
C1(V|T) XOR C2(V&~T).  When both operands are fully tainted (T=0xFFFF...),
both reps evaluate to the same flag value — rep1 uses (MASK, MASK) and
rep2 uses (0, 0), both of which produce INT_SCARRY=0 for 64-bit arithmetic.
The XOR is 0, so T_OF=0 even though OF genuinely depends on the inputs.

Fix
---
For 1-bit (flag) outputs under MONOTONIC, the expression becomes:
    T_flag = differential OR Aval(OR(all_deps))
meaning: if any input bit is tainted, mark the flag tainted.  The differential
still fires when it can (partial taint, sign-asymmetric inputs) and the
Aval(OR(deps)) acts as the soundness floor for the fully-tainted case.

Test matrix
-----------
For each operation that writes OF or CF:
  1. Both operands fully tainted     → flag must be tainted  (the fixed case)
  2. One operand tainted             → flag must be tainted  (already worked)
  3. Neither operand tainted         → flag must NOT be tainted
  4. Regression: 64-bit RAX output   → must NOT be changed by this fix
"""

# mypy: disable-error-code="no-untyped-def"

from __future__ import annotations

import pytest

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, ImplicitTaintPolicy, Register

AMD64 = Architecture.AMD64
MASK64 = 0xFFFFFFFFFFFFFFFF


@pytest.fixture(scope='module')
def sim() -> CellSimulator:
    return CellSimulator(AMD64)


@pytest.fixture(scope='module')
def regs() -> list[Register]:
    return [
        Register('RAX', 64),
        Register('RBX', 64),
        Register('RCX', 64),
        Register('RDX', 64),
        Register('CF', 1),
        Register('OF', 1),
        Register('ZF', 1),
        Register('SF', 1),
        Register('PF', 1),
    ]


def _ev(
    sim: CellSimulator,
    regs: list[Register],
    code: bytes,
    taint: dict[str, int],
    values: dict[str, int],
) -> dict[str, int]:
    circuit = generate_static_rule(AMD64, code, regs)
    ctx = EvalContext(
        input_taint=taint,
        input_values=values,
        simulator=sim,
        implicit_policy=ImplicitTaintPolicy.IGNORE,
    )
    return circuit.evaluate(ctx)


def _z(regs: list[Register]) -> dict[str, int]:
    return {r.name: 0 for r in regs}


# ---------------------------------------------------------------------------
# ADD — OF via INT_SCARRY, CF via INT_CARRY
# ---------------------------------------------------------------------------


class TestAddFlags:

    def test_add_of_both_fully_tainted(self, sim, regs) -> None:
        """add rax,rbx with T_RAX=T_RBX=MASK → OF must be tainted (fixed case)."""
        out = _ev(
            sim,
            regs,
            bytes.fromhex('4801d8'),
            taint={**_z(regs), 'RAX': MASK64, 'RBX': MASK64},
            values={**_z(regs), 'RAX': 5, 'RBX': 3},
        )
        assert out.get('OF', 0) != 0, f'add: both fully tainted → OF must be tainted; got {out.get("OF", 0):#x}'

    def test_add_cf_both_fully_tainted(self, sim, regs) -> None:
        """add rax,rbx with T_RAX=T_RBX=MASK → CF must be tainted."""
        out = _ev(
            sim,
            regs,
            bytes.fromhex('4801d8'),
            taint={**_z(regs), 'RAX': MASK64, 'RBX': MASK64},
            values={**_z(regs), 'RAX': 5, 'RBX': 3},
        )
        assert out.get('CF', 0) != 0, f'add: both fully tainted → CF must be tainted; got {out.get("CF", 0):#x}'

    def test_add_of_one_operand_tainted(self, sim, regs) -> None:
        """add rax,rbx with only T_RBX tainted → OF must be tainted."""
        out = _ev(
            sim,
            regs,
            bytes.fromhex('4801d8'),
            taint={**_z(regs), 'RBX': 0x4000000000000000},
            values={**_z(regs), 'RAX': 0x4000000000000000, 'RBX': 0x4000000000000000},
        )
        assert out.get('OF', 0) != 0, f'add: T_RBX partial → OF must be tainted; got {out.get("OF", 0):#x}'

    def test_add_of_no_taint(self, sim, regs) -> None:
        """add rax,rbx with no taint → OF must NOT be tainted."""
        out = _ev(sim, regs, bytes.fromhex('4801d8'), taint=_z(regs), values={**_z(regs), 'RAX': 1, 'RBX': 2})
        assert out.get('OF', 0) == 0, f'add: no taint → OF must be clean; got {out.get("OF", 0):#x}'

    def test_add_cf_no_taint(self, sim, regs) -> None:
        """add rax,rbx with no taint → CF must NOT be tainted."""
        out = _ev(sim, regs, bytes.fromhex('4801d8'), taint=_z(regs), values={**_z(regs), 'RAX': 1, 'RBX': 2})
        assert out.get('CF', 0) == 0, f'add: no taint → CF must be clean; got {out.get("CF", 0):#x}'

    def test_add_rax_output_unaffected(self, sim, regs) -> None:
        """Regression: add rax,rbx — 64-bit RAX output must not change."""
        z = _z(regs)
        out = _ev(
            sim,
            regs,
            bytes.fromhex('4801d8'),
            taint={**z, 'RAX': MASK64, 'RBX': MASK64},
            values={**z, 'RAX': 5, 'RBX': 3},
        )
        assert out.get('RAX', 0) != 0, 'add: RAX must still be tainted'
        # RAX is 64-bit, computed via TRANSPORTABLE differential — unaffected
        # by the 1-bit flag fix


# ---------------------------------------------------------------------------
# SUB — OF via INT_SBORROW, CF via INT_LESS
# ---------------------------------------------------------------------------


class TestSubFlags:

    def test_sub_of_both_fully_tainted(self, sim, regs) -> None:
        """sub rax,rbx with T_RAX=T_RBX=MASK → OF must be tainted (fixed case)."""
        out = _ev(
            sim,
            regs,
            bytes.fromhex('4829d8'),
            taint={**_z(regs), 'RAX': MASK64, 'RBX': MASK64},
            values={**_z(regs), 'RAX': 5, 'RBX': 3},
        )
        assert out.get('OF', 0) != 0, f'sub: both fully tainted → OF must be tainted; got {out.get("OF", 0):#x}'

    def test_sub_cf_both_fully_tainted(self, sim, regs) -> None:
        """sub rax,rbx with T_RAX=T_RBX=MASK → CF must be tainted (fixed case)."""
        out = _ev(
            sim,
            regs,
            bytes.fromhex('4829d8'),
            taint={**_z(regs), 'RAX': MASK64, 'RBX': MASK64},
            values={**_z(regs), 'RAX': 5, 'RBX': 3},
        )
        assert out.get('CF', 0) != 0, f'sub: both fully tainted → CF must be tainted; got {out.get("CF", 0):#x}'

    def test_sub_of_no_taint(self, sim, regs) -> None:
        """sub rax,rbx with no taint → OF must NOT be tainted."""
        out = _ev(sim, regs, bytes.fromhex('4829d8'), taint=_z(regs), values={**_z(regs), 'RAX': 5, 'RBX': 3})
        assert out.get('OF', 0) == 0, f'sub: no taint → OF must be clean; got {out.get("OF", 0):#x}'

    def test_sub_cf_no_taint(self, sim, regs) -> None:
        """sub rax,rbx with no taint → CF must NOT be tainted."""
        out = _ev(sim, regs, bytes.fromhex('4829d8'), taint=_z(regs), values={**_z(regs), 'RAX': 5, 'RBX': 3})
        assert out.get('CF', 0) == 0, f'sub: no taint → CF must be clean; got {out.get("CF", 0):#x}'


# ---------------------------------------------------------------------------
# CMP — same as SUB but no destination register
# ---------------------------------------------------------------------------


class TestCmpFlags:

    def test_cmp_of_both_fully_tainted(self, sim, regs) -> None:
        """cmp rax,rbx with T_RAX=T_RBX=MASK → OF must be tainted (fixed case)."""
        out = _ev(
            sim,
            regs,
            bytes.fromhex('4839d8'),
            taint={**_z(regs), 'RAX': MASK64, 'RBX': MASK64},
            values={**_z(regs), 'RAX': 5, 'RBX': 3},
        )
        assert out.get('OF', 0) != 0, f'cmp: both fully tainted → OF must be tainted; got {out.get("OF", 0):#x}'

    def test_cmp_cf_both_fully_tainted(self, sim, regs) -> None:
        """cmp rax,rbx with T_RAX=T_RBX=MASK → CF must be tainted."""
        out = _ev(
            sim,
            regs,
            bytes.fromhex('4839d8'),
            taint={**_z(regs), 'RAX': MASK64, 'RBX': MASK64},
            values={**_z(regs), 'RAX': 5, 'RBX': 3},
        )
        assert out.get('CF', 0) != 0, f'cmp: both fully tainted → CF must be tainted; got {out.get("CF", 0):#x}'

    def test_cmp_zf_both_fully_tainted(self, sim, regs) -> None:
        """cmp rax,rbx with T_RAX=T_RBX=MASK → ZF must be tainted."""
        out = _ev(
            sim,
            regs,
            bytes.fromhex('4839d8'),
            taint={**_z(regs), 'RAX': MASK64, 'RBX': MASK64},
            values={**_z(regs), 'RAX': 5, 'RBX': 3},
        )
        assert out.get('ZF', 0) != 0, f'cmp: both fully tainted → ZF must be tainted; got {out.get("ZF", 0):#x}'

    def test_cmp_sf_both_fully_tainted(self, sim, regs) -> None:
        """cmp rax,rbx with T_RAX=T_RBX=MASK → SF must be tainted."""
        out = _ev(
            sim,
            regs,
            bytes.fromhex('4839d8'),
            taint={**_z(regs), 'RAX': MASK64, 'RBX': MASK64},
            values={**_z(regs), 'RAX': 5, 'RBX': 3},
        )
        assert out.get('SF', 0) != 0, f'cmp: both fully tainted → SF must be tainted; got {out.get("SF", 0):#x}'

    def test_cmp_flags_no_taint(self, sim, regs) -> None:
        """cmp rax,rbx with no taint → no flags tainted."""
        out = _ev(sim, regs, bytes.fromhex('4839d8'), taint=_z(regs), values={**_z(regs), 'RAX': 5, 'RBX': 3})
        for flag in ('CF', 'OF', 'ZF', 'SF', 'PF'):
            assert out.get(flag, 0) == 0, f'cmp no taint: {flag} must be clean; got {out.get(flag, 0):#x}'


# ---------------------------------------------------------------------------
# NEG — OF via INT_SBORROW(0, src)
# ---------------------------------------------------------------------------


class TestNegFlags:

    def test_neg_of_fully_tainted(self, sim, regs) -> None:
        """neg rax with T_RAX=MASK → OF must be tainted (fixed case)."""
        out = _ev(sim, regs, bytes.fromhex('48f7d8'), taint={**_z(regs), 'RAX': MASK64}, values={**_z(regs), 'RAX': 5})
        assert out.get('OF', 0) != 0, f'neg: T_RAX=MASK → OF must be tainted; got {out.get("OF", 0):#x}'

    def test_neg_cf_fully_tainted(self, sim, regs) -> None:
        """neg rax with T_RAX=MASK → CF must be tainted."""
        out = _ev(sim, regs, bytes.fromhex('48f7d8'), taint={**_z(regs), 'RAX': MASK64}, values={**_z(regs), 'RAX': 5})
        assert out.get('CF', 0) != 0, f'neg: T_RAX=MASK → CF must be tainted; got {out.get("CF", 0):#x}'

    def test_neg_of_no_taint(self, sim, regs) -> None:
        """neg rax with no taint → OF must NOT be tainted."""
        out = _ev(sim, regs, bytes.fromhex('48f7d8'), taint=_z(regs), values={**_z(regs), 'RAX': 5})
        assert out.get('OF', 0) == 0, f'neg: no taint → OF must be clean; got {out.get("OF", 0):#x}'


# ---------------------------------------------------------------------------
# ADC / SBB — these also write OF and propagate carry
# ---------------------------------------------------------------------------


class TestAdcSbbFlags:

    def test_adc_of_both_gp_fully_tainted(self, sim, regs) -> None:
        """adc rcx,rdx with T_RCX=T_RDX=MASK → OF must be tainted."""
        out = _ev(
            sim,
            regs,
            bytes.fromhex('4811d1'),
            taint={**_z(regs), 'RCX': MASK64, 'RDX': MASK64},
            values={**_z(regs), 'RCX': 5, 'RDX': 3},
        )
        assert out.get('OF', 0) != 0, f'adc: both GP fully tainted → OF must be tainted; got {out.get("OF", 0):#x}'

    def test_sbb_of_both_gp_fully_tainted(self, sim, regs) -> None:
        """sbb rcx,rdx with T_RCX=T_RDX=MASK → OF must be tainted."""
        out = _ev(
            sim,
            regs,
            bytes.fromhex('4819d1'),
            taint={**_z(regs), 'RCX': MASK64, 'RDX': MASK64},
            values={**_z(regs), 'RCX': 5, 'RDX': 3},
        )
        assert out.get('OF', 0) != 0, f'sbb: both GP fully tainted → OF must be tainted; got {out.get("OF", 0):#x}'

    def test_adc_of_no_taint(self, sim, regs) -> None:
        """adc rcx,rdx with no taint → OF must NOT be tainted."""
        out = _ev(sim, regs, bytes.fromhex('4811d1'), taint=_z(regs), values={**_z(regs), 'RCX': 5, 'RDX': 3, 'CF': 0})
        assert out.get('OF', 0) == 0, f'adc: no taint → OF must be clean; got {out.get("OF", 0):#x}'


# ---------------------------------------------------------------------------
# AND / OR — MONOTONIC but flag output should only taint when input tainted
# ---------------------------------------------------------------------------


class TestAndOrFlags:

    def test_and_sf_both_fully_tainted(self, sim, regs) -> None:
        """and rax,rbx with T_RAX=T_RBX=MASK → SF must be tainted (bit 63 unknown)."""
        out = _ev(
            sim,
            regs,
            bytes.fromhex('4821d8'),
            taint={**_z(regs), 'RAX': MASK64, 'RBX': MASK64},
            values={**_z(regs), 'RAX': 5, 'RBX': 3},
        )
        assert out.get('SF', 0) != 0, f'and: both fully tainted → SF must be tainted; got {out.get("SF", 0):#x}'

    def test_and_zf_both_fully_tainted(self, sim, regs) -> None:
        """and rax,rbx with T_RAX=T_RBX=MASK → ZF must be tainted."""
        out = _ev(
            sim,
            regs,
            bytes.fromhex('4821d8'),
            taint={**_z(regs), 'RAX': MASK64, 'RBX': MASK64},
            values={**_z(regs), 'RAX': 5, 'RBX': 3},
        )
        assert out.get('ZF', 0) != 0, f'and: both fully tainted → ZF must be tainted; got {out.get("ZF", 0):#x}'

    def test_and_flags_no_taint(self, sim, regs) -> None:
        """and rax,rbx with no taint → no flags tainted."""
        out = _ev(sim, regs, bytes.fromhex('4821d8'), taint=_z(regs), values={**_z(regs), 'RAX': 5, 'RBX': 3})
        for flag in ('SF', 'ZF', 'PF', 'CF', 'OF'):
            assert out.get(flag, 0) == 0, f'and no taint: {flag} must be clean; got {out.get(flag, 0):#x}'

    def test_and_flags_no_taint(self, sim, regs) -> None:
        """and rax,rbx with no taint → no flags tainted."""
        out = _ev(sim, regs, bytes.fromhex('4821d8'), taint=_z(regs), values={**_z(regs), 'RAX': 5, 'RBX': 3})
        for flag in ('SF', 'ZF', 'PF', 'CF', 'OF'):
            assert out.get(flag, 0) == 0, f'and no taint: {flag} must be clean; got {out.get(flag, 0):#x}'

    def test_and_result_written_correctly(self, sim, regs) -> None:
        """and rax,rbx — the 64-bit result register is still computed via differential."""
        # With partial taint on RAX, the AND result bits are a subset of RAX taint.
        out = _ev(
            sim,
            regs,
            bytes.fromhex('4821d8'),  # and rax, rbx (48 21 d8)
            taint={**_z(regs), 'RAX': 0xFFFFFFFF00000000},
            values={**_z(regs), 'RAX': 0xFFFFFFFF00000001, 'RBX': 0xFFFFFFFF00000001},
        )
        rax_out = out.get('RAX', 0)
        # Flag floor only applies to 1-bit outputs — 64-bit RAX uses pure differential
        assert rax_out != 0xFFFFFFFFFFFFFFFF, f'and: 64-bit RAX must not be fully tainted by flag floor: {rax_out:#x}'


# ---------------------------------------------------------------------------
# Regression: 64-bit GP registers must not be affected by the 1-bit fix
# ---------------------------------------------------------------------------


class TestRegressionGPRegisters:

    def test_add_rax_differential_unchanged(self, sim, regs) -> None:
        """add rax,rbx — T_RAX 64-bit output must still use pure differential."""
        # With partial taint, the differential gives a precise per-bit mask.
        # The flag fix must not alter the 64-bit RAX output at all.
        taint_rax = 0x00FF00FF00FF00FF  # partial taint
        out = _ev(
            sim,
            regs,
            bytes.fromhex('4801d8'),
            taint={**_z(regs), 'RAX': taint_rax, 'RBX': 0},
            values={**_z(regs), 'RAX': 0x0001000100010001, 'RBX': 0},
        )
        # The 64-bit RAX result should be the differential, NOT all-ones.
        rax_out = out.get('RAX', 0)
        assert rax_out != MASK64, f'add: 64-bit RAX must use differential, not be fully tainted: {rax_out:#x}'
        assert rax_out != 0, f'add: 64-bit RAX with partial taint must be nonzero: {rax_out:#x}'

    def test_sub_rax_differential_unchanged(self, sim, regs) -> None:
        """sub rax,rbx — T_RAX 64-bit output must still use pure differential."""
        taint_rax = 0x0F0F0F0F0F0F0F0F
        out = _ev(
            sim,
            regs,
            bytes.fromhex('4829d8'),
            taint={**_z(regs), 'RAX': taint_rax, 'RBX': 0},
            values={**_z(regs), 'RAX': 0x1111111111111111, 'RBX': 0},
        )
        rax_out = out.get('RAX', 0)
        assert rax_out != MASK64, f'sub: 64-bit RAX must not be fully tainted: {rax_out:#x}'
        assert rax_out != 0, f'sub: 64-bit RAX with partial taint must be nonzero: {rax_out:#x}'
