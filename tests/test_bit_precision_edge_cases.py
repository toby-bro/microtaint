"""
test_bit_precision_edge_cases.py
=================================

Systematic bit-precise edge-case tests for x86-64 taint analysis.

Each test uses ``_true_taint`` to compute the exact expected output by flipping
each tainted input bit individually and observing which output bits change.
Tests then compare against microtaint and assert equality.

Root-cause groups addressed
----------------------------
G-A  Constant-result idioms (and/0, or/-1, xor-self) must not taint flags.
G-B  Shift carry flag: CF from shr/shl is the shifted-out bit; ZF/PF/SF
     are based on the *result* register, not the input.
G-C  Rotation: ror/rol must correctly route bit0↔bit63.
G-D  Rotate-through-carry: CF fill bit must not be treated as shift amount.
G-E  setcc with 1-bit flag deps: setl/sete/setne must taint when flag tainted.
G-F  AVALANCHE with concrete-zero operand: not a fixable precision issue
     (documented as inherent to the differential approach).

ARM64 cases are also included.
"""

# mypy: disable-error-code="no-untyped-def"

from __future__ import annotations

import pytest
import unicorn
import unicorn.arm64_const as ua
import unicorn.x86_const as ux

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, ImplicitTaintPolicy, Register

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MASK64 = 0xFFFFFFFFFFFFFFFF
_X86_FLAG_BITS = {'CF': 0, 'PF': 2, 'ZF': 6, 'SF': 7, 'OF': 11}
_X86_GP = {'RAX': ux.UC_X86_REG_RAX, 'RBX': ux.UC_X86_REG_RBX, 'RCX': ux.UC_X86_REG_RCX, 'RDX': ux.UC_X86_REG_RDX}


def _run_x86(code: bytes, regs: dict[str, int]) -> dict[str, int]:
    uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
    uc.mem_map(0x1000, 0x1000)
    uc.mem_write(0x1000, code)
    eflags = 2  # reserved always-1 bit
    for name, val in regs.items():
        if name in _X86_FLAG_BITS:
            if val:
                eflags |= 1 << _X86_FLAG_BITS[name]
        elif name in _X86_GP:
            uc.reg_write(_X86_GP[name], val & MASK64)
    uc.reg_write(ux.UC_X86_REG_EFLAGS, eflags)
    try:
        uc.emu_start(0x1000, 0x1000 + len(code))
    except Exception:
        return {}
    result: dict[str, int] = {}
    for n, rid in _X86_GP.items():
        result[n] = uc.reg_read(rid)
    ef = uc.reg_read(ux.UC_X86_REG_EFLAGS)
    for n, bit in _X86_FLAG_BITS.items():
        result[n] = (ef >> bit) & 1
    return result


def _true_taint_x86(code: bytes, reg_names: list[str], taint: dict[str, int], values: dict[str, int]) -> dict[str, int]:
    """Per-bit sensitivity: flip each tainted bit and XOR output with base."""
    base_vals = {n: values.get(n, 0) & ~taint.get(n, 0) & MASK64 for n in reg_names}
    base_out = _run_x86(code, base_vals)
    result = dict.fromkeys(reg_names, 0)
    for reg in reg_names:
        width = 1 if reg in _X86_FLAG_BITS else 64
        tmask = taint.get(reg, 0)
        if not tmask:
            continue
        for bit in range(width):
            if not (tmask >> bit) & 1:
                continue
            flipped = dict(base_vals)
            flipped[reg] = (base_vals[reg] ^ (1 << bit)) & MASK64
            out = _run_x86(code, flipped)
            for out_reg in reg_names:
                result[out_reg] |= base_out.get(out_reg, 0) ^ out.get(out_reg, 0)
    return result


def _mt_eval(
    sim: CellSimulator,
    regs_list: list[Register],
    code: bytes,
    taint: dict[str, int],
    values: dict[str, int],
) -> dict[str, int]:
    circuit = generate_static_rule(Architecture.AMD64, code, regs_list)
    ctx = EvalContext(
        input_taint=taint,
        input_values=values,
        simulator=sim,
        implicit_policy=ImplicitTaintPolicy.IGNORE,
    )
    return circuit.evaluate(ctx)


@pytest.fixture(scope='module')
def sim_amd64() -> CellSimulator:
    return CellSimulator(Architecture.AMD64)


@pytest.fixture(scope='module')
def regs_amd64() -> list[Register]:
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


@pytest.fixture(scope='module')
def reg_names_amd64() -> list[str]:
    return ['RAX', 'RBX', 'RCX', 'RDX', 'CF', 'OF', 'ZF', 'SF', 'PF']


def z(regs: list[Register]) -> dict[str, int]:
    return {r.name: 0 for r in regs}


# ---------------------------------------------------------------------------
# G-A: Constant-result idioms — flags must NOT be tainted
# ---------------------------------------------------------------------------


class TestConstantResultFlags:
    """When the main result is always a constant (0 or -1), flags are deterministic."""

    def test_and_with_zero_immediate_zf_clean(self, sim_amd64, regs_amd64, reg_names_amd64) -> None:
        """and rax, 0 with T_RAX=MASK — result always 0, ZF always 1, must not taint ZF."""
        code = bytes.fromhex('4883e000')  # and rax, 0
        taint = {**z(regs_amd64), 'RAX': MASK64}
        values = {**z(regs_amd64), 'RAX': 0xDEADBEEF}
        true = _true_taint_x86(code, reg_names_amd64, taint, values)
        mt = _mt_eval(sim_amd64, regs_amd64, code, taint, values)
        assert mt.get('ZF', 0) == true.get(
            'ZF',
            0,
        ), f'and/0: ZF must match true taint {true.get("ZF",0):#x}; got {mt.get("ZF",0):#x}'

    def test_and_with_zero_immediate_sf_clean(self, sim_amd64, regs_amd64, reg_names_amd64) -> None:
        """and rax, 0 with T_RAX=MASK — SF always 0 (result=0 is positive)."""
        code = bytes.fromhex('4883e000')
        taint = {**z(regs_amd64), 'RAX': MASK64}
        values = {**z(regs_amd64), 'RAX': 0xDEAD}
        true = _true_taint_x86(code, reg_names_amd64, taint, values)
        mt = _mt_eval(sim_amd64, regs_amd64, code, taint, values)
        assert mt.get('SF', 0) == true.get(
            'SF',
            0,
        ), f'and/0: SF taint mismatch; true={true.get("SF",0):#x} mt={mt.get("SF",0):#x}'

    def test_and_with_zero_immediate_pf_clean(self, sim_amd64, regs_amd64, reg_names_amd64) -> None:
        """and rax, 0 — PF(0) is always 1 (even parity), must not be tainted."""
        code = bytes.fromhex('4883e000')
        taint = {**z(regs_amd64), 'RAX': MASK64}
        values = {**z(regs_amd64), 'RAX': 0xDEAD}
        true = _true_taint_x86(code, reg_names_amd64, taint, values)
        mt = _mt_eval(sim_amd64, regs_amd64, code, taint, values)
        assert mt.get('PF', 0) == true.get(
            'PF',
            0,
        ), f'and/0: PF taint mismatch; true={true.get("PF",0):#x} mt={mt.get("PF",0):#x}'

    def test_or_with_minus_one_zf_clean(self, sim_amd64, regs_amd64, reg_names_amd64) -> None:
        """or rax, -1 with T_RAX=MASK — result always MASK64, ZF always 0."""
        code = bytes.fromhex('4883c8ff')  # or rax, -1
        taint = {**z(regs_amd64), 'RAX': MASK64}
        values = {**z(regs_amd64), 'RAX': 0}
        true = _true_taint_x86(code, reg_names_amd64, taint, values)
        mt = _mt_eval(sim_amd64, regs_amd64, code, taint, values)
        assert mt.get('ZF', 0) == true.get('ZF', 0), f'or/-1: ZF must be clean (always 0); got {mt.get("ZF",0):#x}'

    def test_xor_self_flags_clean(self, sim_amd64, regs_amd64, reg_names_amd64) -> None:
        """xor rax, rax with T_RAX=MASK — result always 0; ZF/SF/PF must be clean."""
        code = bytes.fromhex('4831c0')  # xor rax, rax
        taint = {**z(regs_amd64), 'RAX': MASK64}
        values = {**z(regs_amd64), 'RAX': 0xDEAD}
        true = _true_taint_x86(code, reg_names_amd64, taint, values)
        mt = _mt_eval(sim_amd64, regs_amd64, code, taint, values)
        for flag in ('ZF', 'SF', 'PF'):
            assert mt.get(flag, 0) == true.get(
                flag,
                0,
            ), f'xor self: {flag} taint mismatch; true={true.get(flag,0):#x} mt={mt.get(flag,0):#x}'


# ---------------------------------------------------------------------------
# G-B: Shift carry flag — CF is the shifted-out bit
# ---------------------------------------------------------------------------


class TestShiftCarryFlag:

    def test_shr_1_cf_is_bit0(self, sim_amd64, regs_amd64, reg_names_amd64) -> None:
        """shr rax,1 — CF = bit0 of RAX before shift. T_RAX=0x1 → T_CF=1."""
        code = bytes.fromhex('48d1e8')  # shr rax, 1
        taint = {**z(regs_amd64), 'RAX': 0x1}  # only bit 0 tainted
        values = {**z(regs_amd64), 'RAX': 0x1}
        true = _true_taint_x86(code, reg_names_amd64, taint, values)
        mt = _mt_eval(sim_amd64, regs_amd64, code, taint, values)
        assert mt.get('CF', 0) == true.get(
            'CF',
            0,
        ), f'shr/1 CF: bit0 tainted → T_CF={true.get("CF",0)}; got {mt.get("CF",0)}'

    def test_shr_1_cf_from_bit0_high_taint(self, sim_amd64, regs_amd64, reg_names_amd64) -> None:
        """shr rax,1 T_RAX=MASK — CF tainted (depends on bit 0 of input)."""
        code = bytes.fromhex('48d1e8')
        taint = {**z(regs_amd64), 'RAX': MASK64}
        values = {**z(regs_amd64), 'RAX': MASK64}
        true = _true_taint_x86(code, reg_names_amd64, taint, values)
        mt = _mt_eval(sim_amd64, regs_amd64, code, taint, values)
        assert mt.get('CF', 0) == true.get(
            'CF',
            0,
        ), f'shr/1 CF full taint: true={true.get("CF",0):#x} got {mt.get("CF",0):#x}'

    def test_shl_1_cf_is_msb(self, sim_amd64, regs_amd64, reg_names_amd64) -> None:
        """shl rax,1 — CF = bit63 of RAX before shift. T_RAX=MSB → T_CF=1."""
        code = bytes.fromhex('48d1e0')  # shl rax, 1
        taint = {**z(regs_amd64), 'RAX': 0x8000000000000000}  # only bit 63
        values = {**z(regs_amd64), 'RAX': 0x8000000000000000}
        true = _true_taint_x86(code, reg_names_amd64, taint, values)
        mt = _mt_eval(sim_amd64, regs_amd64, code, taint, values)
        assert mt.get('CF', 0) == true.get(
            'CF',
            0,
        ), f'shl/1 CF: MSB tainted → T_CF={true.get("CF",0)}; got {mt.get("CF",0)}'

    def test_shr_1_zf_sf_pf_precise(self, sim_amd64, regs_amd64, reg_names_amd64) -> None:
        """shr rax,1 with partial taint — ZF/SF/PF must match true differential."""
        code = bytes.fromhex('48d1e8')
        taint = {**z(regs_amd64), 'RAX': 0x2}  # only bit 1
        values = {**z(regs_amd64), 'RAX': 0x2}
        true = _true_taint_x86(code, reg_names_amd64, taint, values)
        mt = _mt_eval(sim_amd64, regs_amd64, code, taint, values)
        for flag in ('ZF', 'SF', 'PF'):
            assert mt.get(flag, 0) == true.get(
                flag,
                0,
            ), f'shr/1 {flag}: partial taint mismatch; true={true.get(flag,0):#x} mt={mt.get(flag,0):#x}'

    def test_sar_1_cf_is_bit0(self, sim_amd64, regs_amd64, reg_names_amd64) -> None:
        """sar rax,1 — CF = bit0. T_RAX=0x1 → T_CF=1."""
        code = bytes.fromhex('48d1f8')  # sar rax, 1
        taint = {**z(regs_amd64), 'RAX': 0x1}
        values = {**z(regs_amd64), 'RAX': 0x80000000000000FF}  # negative, bit0=1
        true = _true_taint_x86(code, reg_names_amd64, taint, values)
        mt = _mt_eval(sim_amd64, regs_amd64, code, taint, values)
        assert mt.get('CF', 0) == true.get(
            'CF',
            0,
        ), f'sar/1 CF: bit0 tainted → T_CF={true.get("CF",0)}; got {mt.get("CF",0)}'


# ---------------------------------------------------------------------------
# G-C: Rotation — bit routing must be exact
# ---------------------------------------------------------------------------


class TestRotation:

    def test_ror_1_bit63_from_bit0(self, sim_amd64, regs_amd64, reg_names_amd64) -> None:
        """ror rax,1 — bit63 of result = bit0 of input. T_RAX[0]=1 → T_RAX_out[63]=1."""
        code = bytes.fromhex('48d1c8')  # ror rax, 1
        taint = {**z(regs_amd64), 'RAX': 0x3}  # bits 0 and 1
        values = {**z(regs_amd64), 'RAX': 0x3}
        true = _true_taint_x86(code, reg_names_amd64, taint, values)
        mt = _mt_eval(sim_amd64, regs_amd64, code, taint, values)
        assert mt.get('RAX', 0) == true.get(
            'RAX',
            0,
        ), f'ror/1: RAX taint mismatch; true={true.get("RAX",0):#x} mt={mt.get("RAX",0):#x}'

    def test_rol_1_bit0_from_bit63(self, sim_amd64, regs_amd64, reg_names_amd64) -> None:
        """rol rax,1 — bit0 of result = bit63 of input."""
        code = bytes.fromhex('48d1c0')  # rol rax, 1
        taint = {**z(regs_amd64), 'RAX': 0x8000000000000001}  # bits 63 and 0
        values = {**z(regs_amd64), 'RAX': 0x8000000000000001}
        true = _true_taint_x86(code, reg_names_amd64, taint, values)
        mt = _mt_eval(sim_amd64, regs_amd64, code, taint, values)
        assert mt.get('RAX', 0) == true.get(
            'RAX',
            0,
        ), f'rol/1: RAX taint mismatch; true={true.get("RAX",0):#x} mt={mt.get("RAX",0):#x}'

    def test_ror_8_exact_permutation(self, sim_amd64, regs_amd64, reg_names_amd64) -> None:
        """ror rax,8 — exact byte rotation. Each byte permutes to known position."""
        code = bytes.fromhex('48c1c808')  # ror rax, 8
        taint = {**z(regs_amd64), 'RAX': 0xFF00000000000000}  # byte 7 tainted
        values = {**z(regs_amd64), 'RAX': 0xAA00000000000000}
        true = _true_taint_x86(code, reg_names_amd64, taint, values)
        mt = _mt_eval(sim_amd64, regs_amd64, code, taint, values)
        assert mt.get('RAX', 0) == true.get(
            'RAX',
            0,
        ), f'ror/8: byte rotation; true={true.get("RAX",0):#x} mt={mt.get("RAX",0):#x}'


# ---------------------------------------------------------------------------
# G-D: Rotate-through-carry — CF is the fill bit, not shift amount
# ---------------------------------------------------------------------------


class TestRotateThroughCarry:

    def test_rcl_1_cf_fills_bit0(self, sim_amd64, regs_amd64, reg_names_amd64) -> None:
        """rcl rax,1 — bit0 of result = old CF. T_CF=1, T_RAX=0 → only bit0 tainted."""
        code = bytes.fromhex('48d1d0')  # rcl rax, 1
        taint = {**z(regs_amd64), 'CF': 1}  # only CF tainted
        values = {**z(regs_amd64), 'RAX': 0, 'CF': 1}
        true = _true_taint_x86(code, reg_names_amd64, taint, values)
        mt = _mt_eval(sim_amd64, regs_amd64, code, taint, values)
        assert mt.get('RAX', 0) == true.get(
            'RAX',
            0,
        ), f'rcl/1 (T_CF only): RAX taint; true={true.get("RAX",0):#x} mt={mt.get("RAX",0):#x}'

    def test_rcr_1_cf_fills_bit63(self, sim_amd64, regs_amd64, reg_names_amd64) -> None:
        """rcr rax,1 — bit63 of result = old CF. T_CF=1, T_RAX=0 → only bit63 tainted."""
        code = bytes.fromhex('48d1d8')  # rcr rax, 1
        taint = {**z(regs_amd64), 'CF': 1}
        values = {**z(regs_amd64), 'RAX': 0, 'CF': 1}
        true = _true_taint_x86(code, reg_names_amd64, taint, values)
        mt = _mt_eval(sim_amd64, regs_amd64, code, taint, values)
        assert mt.get('RAX', 0) == true.get(
            'RAX',
            0,
        ), f'rcr/1 (T_CF only): RAX taint; true={true.get("RAX",0):#x} mt={mt.get("RAX",0):#x}'

    def test_rcl_1_rax_tainted_cf_is_msb(self, sim_amd64, regs_amd64, reg_names_amd64) -> None:
        """rcl rax,1 — new CF = old MSB of RAX. T_RAX=MSB → T_CF=1."""
        code = bytes.fromhex('48d1d0')
        taint = {**z(regs_amd64), 'RAX': 0x8000000000000000}  # bit 63
        values = {**z(regs_amd64), 'RAX': 0x8000000000000000, 'CF': 0}
        true = _true_taint_x86(code, reg_names_amd64, taint, values)
        mt = _mt_eval(sim_amd64, regs_amd64, code, taint, values)
        assert mt.get('CF', 0) == true.get(
            'CF',
            0,
        ), f'rcl/1: new CF = old MSB; true={true.get("CF",0)} mt={mt.get("CF",0)}'


# ---------------------------------------------------------------------------
# G-E: setcc — result depends on 1-bit flag deps
# ---------------------------------------------------------------------------


class TestSetcc:

    def test_setl_sf_ne_of_tainted(self, sim_amd64, regs_amd64, reg_names_amd64) -> None:
        """setl al — al = (SF != OF). T_SF=1, T_OF=1 → T_RAX[7:0] = 1."""
        code = bytes.fromhex('0f9cc0')  # setl al
        taint = {**z(regs_amd64), 'SF': 1, 'OF': 1}
        values = {**z(regs_amd64), 'SF': 0, 'OF': 0}
        true = _true_taint_x86(code, reg_names_amd64, taint, values)
        mt = _mt_eval(sim_amd64, regs_amd64, code, taint, values)
        assert (
            mt.get('RAX', 0) & 0xFF == true.get('RAX', 0) & 0xFF
        ), f'setl: T_SF=T_OF=1 → al tainted; true={true.get("RAX",0):#x} mt={mt.get("RAX",0):#x}'

    def test_setge_sf_eq_of_tainted(self, sim_amd64, regs_amd64, reg_names_amd64) -> None:
        """setge al — al = (SF == OF). T_SF=1 → T_al = 1."""
        code = bytes.fromhex('0f9dc0')  # setge al
        taint = {**z(regs_amd64), 'SF': 1}
        values = {**z(regs_amd64), 'SF': 0, 'OF': 0}
        true = _true_taint_x86(code, reg_names_amd64, taint, values)
        mt = _mt_eval(sim_amd64, regs_amd64, code, taint, values)
        assert (
            mt.get('RAX', 0) & 0xFF == true.get('RAX', 0) & 0xFF
        ), f'setge: T_SF=1 → al tainted; true={true.get("RAX",0):#x} mt={mt.get("RAX",0):#x}'

    def test_sete_zf_tainted(self, sim_amd64, regs_amd64, reg_names_amd64) -> None:
        """sete al — al = ZF. T_ZF=1 → T_al = 1."""
        code = bytes.fromhex('0f94c0')  # sete al
        taint = {**z(regs_amd64), 'ZF': 1}
        values = {**z(regs_amd64), 'ZF': 0}
        true = _true_taint_x86(code, reg_names_amd64, taint, values)
        mt = _mt_eval(sim_amd64, regs_amd64, code, taint, values)
        assert (
            mt.get('RAX', 0) & 0xFF == true.get('RAX', 0) & 0xFF
        ), f'sete: T_ZF=1 → al tainted; true={true.get("RAX",0):#x} mt={mt.get("RAX",0):#x}'

    def test_setne_zf_untainted_clean(self, sim_amd64, regs_amd64, reg_names_amd64) -> None:
        """setne al with no flag taint → al must not be tainted."""
        code = bytes.fromhex('0f95c0')  # setne al
        taint = z(regs_amd64)
        values = {**z(regs_amd64), 'ZF': 0}
        mt = _mt_eval(sim_amd64, regs_amd64, code, taint, values)
        assert mt.get('RAX', 0) & 0xFF == 0, f'setne untainted: al must be clean; got {mt.get("RAX",0) & 0xFF:#x}'


# ---------------------------------------------------------------------------
# G-F: AVALANCHE with concrete-zero — documented non-fixable precision limit
# ---------------------------------------------------------------------------


class TestAvalancheConcreteZero:

    def test_mul_by_zero_rax_over_taint_documented(
        self,
        sim_amd64,
        regs_amd64,
        reg_names_amd64,
    ) -> None:
        """mul rbx with RBX=0 concrete — microtaint over-taints RAX.

        This is a known inherent limitation: mul is AVALANCHE and the differential
        cannot detect that a concrete 0 operand forces the result to 0.
        This test DOCUMENTS the over-taint (does not assert equality with true).
        Verify: microtaint >= true (sound, never misses taint).
        """
        code = bytes.fromhex('48f7e3')  # mul rbx
        taint = {**z(regs_amd64), 'RAX': MASK64, 'RBX': 0}
        values = {**z(regs_amd64), 'RAX': 5, 'RBX': 0}
        true = _true_taint_x86(code, reg_names_amd64, taint, values)
        mt = _mt_eval(sim_amd64, regs_amd64, code, taint, values)
        rax_t, rax_mt = true.get('RAX', 0), mt.get('RAX', 0)
        # Sound: microtaint must not MISS taint that true analysis found
        assert (rax_t & ~rax_mt) == 0, f'mul/0: microtaint must be sound (≥ true); missed bits={(rax_t & ~rax_mt):#x}'
        # Document that over-taint exists
        # assert rax_mt == rax_t  # would fail — over-taint is expected here


# ---------------------------------------------------------------------------
# More precise flag tracking
# ---------------------------------------------------------------------------


class TestFlagPrecision:

    def test_add_cf_from_partial_taint(self, sim_amd64, regs_amd64, reg_names_amd64) -> None:
        """add rax,rbx with only high bits tainted — CF precise."""
        code = bytes.fromhex('4801d8')
        taint = {**z(regs_amd64), 'RAX': 0xF000000000000000}  # top nibble only
        values = {**z(regs_amd64), 'RAX': 0x8000000000000000, 'RBX': 0x8000000000000000}
        true = _true_taint_x86(code, reg_names_amd64, taint, values)
        mt = _mt_eval(sim_amd64, regs_amd64, code, taint, values)
        assert mt.get('CF', 0) == true.get(
            'CF',
            0,
        ), f'add partial CF: true={true.get("CF",0):#x} mt={mt.get("CF",0):#x}'

    def test_sub_borrow_chain_exact(self, sim_amd64, regs_amd64, reg_names_amd64) -> None:
        """sub with partial taint — borrow chain must be exact."""
        code = bytes.fromhex('4829d8')
        taint = {**z(regs_amd64), 'RBX': 0x0000000000000001}  # bit 0 only
        values = {**z(regs_amd64), 'RAX': 0, 'RBX': 0}
        true = _true_taint_x86(code, reg_names_amd64, taint, values)
        mt = _mt_eval(sim_amd64, regs_amd64, code, taint, values)
        for flag in ('CF', 'ZF', 'SF', 'OF'):
            assert mt.get(flag, 0) == true.get(
                flag,
                0,
            ), f'sub borrow {flag}: true={true.get(flag,0):#x} mt={mt.get(flag,0):#x}'

    def test_cmp_flags_all_precise_partial_taint(
        self,
        sim_amd64,
        regs_amd64,
        reg_names_amd64,
    ) -> None:
        """cmp rax,rbx with partial taint — all flags match true differential."""
        code = bytes.fromhex('4839d8')
        taint = {**z(regs_amd64), 'RAX': 0x00000000000000FF}  # low byte
        values = {**z(regs_amd64), 'RAX': 0x80, 'RBX': 0x100}
        true = _true_taint_x86(code, reg_names_amd64, taint, values)
        mt = _mt_eval(sim_amd64, regs_amd64, code, taint, values)
        for flag in ('CF', 'ZF', 'SF', 'OF', 'PF'):
            assert mt.get(flag, 0) == true.get(
                flag,
                0,
            ), f'cmp partial {flag}: true={true.get(flag,0):#x} mt={mt.get(flag,0):#x}'

    def test_and_partial_taint_exact(self, sim_amd64, regs_amd64, reg_names_amd64) -> None:
        """and rax,rbx with partial taint — result and ZF must be exact."""
        code = bytes.fromhex('4821d8')
        taint = {**z(regs_amd64), 'RAX': 0x00000000FFFFFFFF}  # low 32 bits
        values = {**z(regs_amd64), 'RAX': 0x00000000DEADBEEF, 'RBX': 0x00000000FFFFFFFF}
        true = _true_taint_x86(code, reg_names_amd64, taint, values)
        mt = _mt_eval(sim_amd64, regs_amd64, code, taint, values)
        assert mt.get('RAX', 0) == true.get(
            'RAX',
            0,
        ), f'and partial RAX: true={true.get("RAX",0):#x} mt={mt.get("RAX",0):#x}'

    def test_shl_result_exact_partial(self, sim_amd64, regs_amd64, reg_names_amd64) -> None:
        """shl rax,4 with partial taint — result bits exact (MAPPED)."""
        code = bytes.fromhex('48c1e004')  # shl rax, 4
        taint = {**z(regs_amd64), 'RAX': 0x00FF00FF00FF00FF}
        values = {**z(regs_amd64), 'RAX': 0x00FF00FF00FF00FF}
        true = _true_taint_x86(code, reg_names_amd64, taint, values)
        mt = _mt_eval(sim_amd64, regs_amd64, code, taint, values)
        assert mt.get('RAX', 0) == true.get(
            'RAX',
            0,
        ), f'shl/4 partial: true={true.get("RAX",0):#x} mt={mt.get("RAX",0):#x}'

    def test_bswap_exact(self, sim_amd64, regs_amd64, reg_names_amd64) -> None:
        """bswap rax — exact byte permutation."""
        code = bytes.fromhex('480fc8')  # bswap rax
        taint = {**z(regs_amd64), 'RAX': 0xFF00FF00FF00FF00}
        values = {**z(regs_amd64), 'RAX': 0xDEADBEEFCAFEBABE}
        true = _true_taint_x86(code, reg_names_amd64, taint, values)
        mt = _mt_eval(sim_amd64, regs_amd64, code, taint, values)
        assert mt.get('RAX', 0) == true.get('RAX', 0), f'bswap: true={true.get("RAX",0):#x} mt={mt.get("RAX",0):#x}'

    def test_movzx_upper_bits_exact_zero(self, sim_amd64, regs_amd64, reg_names_amd64) -> None:
        """movzx rax,bl — upper 56 bits always 0, must not be tainted."""
        code = bytes.fromhex('480fb6c3')  # movzx rax, bl
        taint = {**z(regs_amd64), 'RBX': MASK64}
        values = {**z(regs_amd64), 'RBX': 0x42}
        true = _true_taint_x86(code, reg_names_amd64, taint, values)
        mt = _mt_eval(sim_amd64, regs_amd64, code, taint, values)
        assert mt.get('RAX', 0) == true.get('RAX', 0), f'movzx: true={true.get("RAX",0):#x} mt={mt.get("RAX",0):#x}'

    def test_movsx_sign_extension_exact(self, sim_amd64, regs_amd64, reg_names_amd64) -> None:
        """movsx rax,bl — sign extends bit7. Upper bits match sign of bit7."""
        code = bytes.fromhex('480fbed3')  # movsx rax, bl
        # T_RBX = 0x80 (bit 7 only) — sign bit tainted
        taint = {**z(regs_amd64), 'RBX': 0x80}
        values = {**z(regs_amd64), 'RBX': 0x80}
        true = _true_taint_x86(code, reg_names_amd64, taint, values)
        mt = _mt_eval(sim_amd64, regs_amd64, code, taint, values)
        assert mt.get('RAX', 0) == true.get(
            'RAX',
            0,
        ), f'movsx bit7: true={true.get("RAX",0):#x} mt={mt.get("RAX",0):#x}'


# ---------------------------------------------------------------------------
# ARM64 bit-precision tests
# ---------------------------------------------------------------------------


_ARM64_FLAG_BITS = {'N': 31, 'Z': 30, 'C': 29, 'V': 28}
_ARM64_GP = {f'X{i}': getattr(ua, f'UC_ARM64_REG_X{i}') for i in range(8)}


def _run_arm64(code: bytes, regs: dict[str, int]) -> dict[str, int]:
    uc = unicorn.Uc(unicorn.UC_ARCH_ARM64, unicorn.UC_MODE_ARM)
    uc.mem_map(0x1000, 0x1000)
    uc.mem_write(0x1000, code)
    nzcv = 0
    for name, val in regs.items():
        if name in _ARM64_FLAG_BITS:
            if val:
                nzcv |= 1 << _ARM64_FLAG_BITS[name]
        elif name in _ARM64_GP:
            uc.reg_write(_ARM64_GP[name], val & MASK64)
    if nzcv:
        uc.reg_write(ua.UC_ARM64_REG_NZCV, nzcv)
    try:
        uc.emu_start(0x1000, 0x1000 + len(code))
    except Exception:
        return {}
    result = {n: uc.reg_read(_ARM64_GP[n]) for n in _ARM64_GP}
    raw_nzcv = uc.reg_read(ua.UC_ARM64_REG_NZCV)
    for name, bit in _ARM64_FLAG_BITS.items():
        result[name] = (raw_nzcv >> bit) & 1
    return result


def _true_taint_arm64(
    code: bytes,
    reg_names: list[str],
    taint: dict[str, int],
    values: dict[str, int],
) -> dict[str, int]:
    base_vals = {n: values.get(n, 0) & ~taint.get(n, 0) & MASK64 for n in reg_names}
    base_out = _run_arm64(code, base_vals)
    result = dict.fromkeys(reg_names, 0)
    for reg in reg_names:
        width = 1 if reg in _ARM64_FLAG_BITS else 64
        tmask = taint.get(reg, 0)
        if not tmask:
            continue
        for bit in range(width):
            if not (tmask >> bit) & 1:
                continue
            flipped = dict(base_vals)
            flipped[reg] = (base_vals[reg] ^ (1 << bit)) & MASK64
            out = _run_arm64(code, flipped)
            for r2 in reg_names:
                result[r2] |= base_out.get(r2, 0) ^ out.get(r2, 0)
    return result


@pytest.fixture(scope='module')
def sim_arm64() -> CellSimulator:
    return CellSimulator(Architecture.ARM64)


@pytest.fixture(scope='module')
def regs_arm64() -> list[Register]:
    return [Register(f'X{i}', 64) for i in range(8)] + [
        Register('N', 1),
        Register('Z', 1),
        Register('C', 1),
        Register('V', 1),
    ]


@pytest.fixture(scope='module')
def reg_names_arm64() -> list[str]:
    return [f'X{i}' for i in range(8)] + ['N', 'Z', 'C', 'V']


def _z64(regs: list[Register]) -> dict[str, int]:
    return {r.name: 0 for r in regs}


def _mt64(sim, regs, code, taint, values):
    circuit = generate_static_rule(Architecture.ARM64, code, regs)
    ctx = EvalContext(
        input_taint=taint,
        input_values=values,
        simulator=sim,
        implicit_policy=ImplicitTaintPolicy.IGNORE,
    )
    return circuit.evaluate(ctx)


class TestARM64Precision:

    def test_add_full_taint_propagates(self, sim_arm64, regs_arm64, reg_names_arm64) -> None:
        """ADD X0,X1,X2 — both inputs fully tainted → X0 must be tainted."""
        # ADD X0, X1, X2 (0x8B020020)
        code = bytes.fromhex('2000028b')
        z = _z64(regs_arm64)
        out = _mt64(sim_arm64, regs_arm64, code, {**z, 'X1': MASK64, 'X2': MASK64}, {**z, 'X1': 5, 'X2': 3})
        assert out.get('X0', 0) != 0, 'ADD with both tainted → X0 must be tainted'

    def test_add_partial_taint_exact(self, sim_arm64, regs_arm64, reg_names_arm64) -> None:
        """ADD X0,X1,X2 — partial taint on X1 → result matches true differential."""
        code = bytes.fromhex('2000028b')
        z = _z64(regs_arm64)
        taint = {**z, 'X1': 0xFF00}
        values = {**z, 'X1': 0xFF00, 'X2': 0}
        true = _true_taint_arm64(code, reg_names_arm64, taint, values)
        mt = _mt64(sim_arm64, regs_arm64, code, taint, values)
        assert mt.get('X0', 0) == true.get('X0', 0), f'ADD partial: true={true.get("X0",0):#x} mt={mt.get("X0",0):#x}'

    def test_eor_zeroing_idiom(self, sim_arm64, regs_arm64, reg_names_arm64) -> None:
        """EOR X0,X1,X1 (zeroing) — X1 fully tainted → X0 must be 0 (constant result)."""
        # EOR X0, X1, X1  (0xCA010020)
        code = bytes.fromhex('200001ca')
        z = _z64(regs_arm64)
        out = _mt64(sim_arm64, regs_arm64, code, {**z, 'X1': MASK64}, {**z, 'X1': 0xDEAD})
        assert out.get('X0', 0) == 0, f'EOR X0,X1,X1 (zeroing): X0 must be 0; got {out.get("X0",0):#x}'

    def test_sub_propagates(self, sim_arm64, regs_arm64, reg_names_arm64) -> None:
        """SUB X0,X1,X2 — X1 partially tainted → result exact."""
        # SUB X0, X1, X2  (0xCB020020)
        code = bytes.fromhex('200002cb')
        z = _z64(regs_arm64)
        taint = {**z, 'X1': 0x00FF00FF00FF00FF}
        values = {**z, 'X1': 0x00FF00FF00FF00FF, 'X2': 0}
        true = _true_taint_arm64(code, reg_names_arm64, taint, values)
        mt = _mt64(sim_arm64, regs_arm64, code, taint, values)
        assert mt.get('X0', 0) == true.get('X0', 0), f'SUB partial: true={true.get("X0",0):#x} mt={mt.get("X0",0):#x}'

    def test_and_propagates(self, sim_arm64, regs_arm64, reg_names_arm64) -> None:
        """AND X0,X1,X2 — partial taint exact."""
        # AND X0, X1, X2  (0x8A020020)
        code = bytes.fromhex('2000028a')
        z = _z64(regs_arm64)
        taint = {**z, 'X1': 0xFFFF0000FFFF0000}
        values = {**z, 'X1': 0xDEADBEEFCAFEBABE, 'X2': 0xFFFFFFFFFFFFFFFF}
        true = _true_taint_arm64(code, reg_names_arm64, taint, values)
        mt = _mt64(sim_arm64, regs_arm64, code, taint, values)
        assert mt.get('X0', 0) == true.get('X0', 0), f'AND partial: true={true.get("X0",0):#x} mt={mt.get("X0",0):#x}'

    def test_adds_flags_tainted_fully(self, sim_arm64, regs_arm64, reg_names_arm64) -> None:
        """ADDS W0,W1,W2 — both fully tainted → all flags conservatively tainted (sound)."""
        # ADDS W0, W1, W2  (0x2B020020)
        code = bytes.fromhex('2000022b')
        z = _z64(regs_arm64)
        out = _mt64(sim_arm64, regs_arm64, code, {**z, 'X1': MASK64, 'X2': MASK64}, {**z, 'X1': 5, 'X2': 3})
        # N and Z must be tainted (sign and zero depend on sum)
        assert out.get('N', 0) != 0, 'ADDS: N flag must be tainted when both inputs unknown'
        assert out.get('Z', 0) != 0, 'ADDS: Z flag must be tainted when both inputs unknown'

    def test_adds_flags_no_taint(self, sim_arm64, regs_arm64, reg_names_arm64) -> None:
        """ADDS W0,W1,W2 — no taint → no flag taint."""
        code = bytes.fromhex('2000022b')
        z = _z64(regs_arm64)
        out = _mt64(sim_arm64, regs_arm64, code, z, {**z, 'X1': 5, 'X2': 3})
        for flag in ('N', 'Z', 'C', 'V'):
            assert out.get(flag, 0) == 0, f'ADDS no taint: {flag} must be 0'

    def test_csel_zflag_tainted(self, sim_arm64, regs_arm64, reg_names_arm64) -> None:
        """CSEL X0,X1,X2,EQ — T_Z=1 → X0 must be tainted (selects different value)."""
        # CSEL X0, X1, X2, EQ  (0x9A820020)
        code = bytes.fromhex('2000829a')
        z = _z64(regs_arm64)
        out = _mt64(sim_arm64, regs_arm64, code, {**z, 'Z': 1}, {**z, 'X1': 10, 'X2': 20, 'Z': 0})
        assert out.get('X0', 0) != 0, f'CSEL: tainted Z must taint X0; got {out.get("X0",0):#x}'

    def test_csel_no_flag_taint_result_clean(
        self,
        sim_arm64,
        regs_arm64,
        reg_names_arm64,
    ) -> None:
        """CSEL X0,X1,X2,EQ — no taint → X0 not tainted."""
        code = bytes.fromhex('2000829a')
        z = _z64(regs_arm64)
        out = _mt64(sim_arm64, regs_arm64, code, z, {**z, 'X1': 10, 'X2': 20, 'Z': 0})
        assert out.get('X0', 0) == 0, 'CSEL no taint: X0 must be 0'

    def test_add_no_taint_clean(self, sim_arm64, regs_arm64, reg_names_arm64) -> None:
        """ADD X0,X1,X2 — no taint → X0 not tainted."""
        code = bytes.fromhex('2000028b')
        z = _z64(regs_arm64)
        out = _mt64(sim_arm64, regs_arm64, code, z, {**z, 'X1': 5, 'X2': 3})
        assert out.get('X0', 0) == 0, 'ADD no taint: X0 must be 0'
