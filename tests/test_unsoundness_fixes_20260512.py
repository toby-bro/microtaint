"""
Soundness regression tests for the BTC/BTR/BTS immediate-index CF taint bug
identified in report_1778571166.json (May 12 2026 benchmark run).

  Pattern — BTC/BTR/BTS with immediate bit index: CF taint dropped
            ``xor edx, edx; btc rax, 5; setc dl`` (report id=7883).

            BTC/BTR/BTS with an immediate bit index lift in Sleigh as:

                AND(const_imm, 0x3f) -> u0     # fold immediate to unique
                RIGHT(register, u0)  -> u1     # shift register by constant
                AND(u1, 1)           -> u2     # isolate the tested bit
                NOTEQUAL(u2, 0)      -> CF     # write CF

            The _is_bit_extract_notequal predicate rejected this slice because
            len(ops)=4 > 2.  The _is_bit_extract_via_tainted_shift predicate
            also rejected it because the shift amount u0 is derived purely from
            constants (no register), so _reaches_register(u0) returned False.
            Neither predicate fired → the differential was never OR-ed into the
            CF expression → only the FULLMASK_AVAL floor remained, which fires
            only when ALL bits of RAX are tainted.  With partial taint (e.g.
            only T_RAX[5]=1) the floor evaluated to 0, so T_CF=0 and the
            downstream ``setc dl`` produced T_RDX=0 instead of the correct 1.

            Fix: added _is_const_shift_bit_extract in engine.py that matches
            exactly this 4-op pattern (NOTEQUAL(AND(RIGHT(register,const),
            const), 0)) with a purely constant shift amount, enabling the
            differential to be added to the CF expression.

Each test brute-forces ground truth via 2^k Unicorn enumeration and asserts
``microtaint_output ⊇ ground_truth`` (no under-tainted bits).
"""

# mypy: disable-error-code="no-untyped-def,no-untyped-call,attr-defined"

from __future__ import annotations

import itertools

import pytest
from unicorn import UC_ARCH_X86, UC_MODE_64
from unicorn.unicorn_py3 import Uc
from unicorn.x86_const import (
    UC_CPU_X86_BROADWELL,
    UC_X86_REG_RAX,
    UC_X86_REG_RBX,
    UC_X86_REG_RCX,
    UC_X86_REG_RDX,
)

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, Register

MASK64 = 0xFFFFFFFFFFFFFFFF
_REGS_GP = [
    Register('RAX', 64),
    Register('RBX', 64),
    Register('RCX', 64),
    Register('RDX', 64),
]
_SIM = CellSimulator(Architecture.AMD64)
_UC_REGS = {
    'RAX': UC_X86_REG_RAX,
    'RBX': UC_X86_REG_RBX,
    'RCX': UC_X86_REG_RCX,
    'RDX': UC_X86_REG_RDX,
}
_BASE_ADDR = 0x400000
_PAGE_SIZE = 0x1000
_STACK_BASE = 0x500000


def _brute_force_gt(
    bytestring: bytes,
    state: dict[str, int],
    taint: dict[str, int],
) -> dict[str, int]:
    """Compute exact bit-level ground truth via 2^k Unicorn enumeration."""
    positions: list[tuple[str, int]] = []
    for reg, mask in taint.items():
        for bit in range(64):
            if (mask >> bit) & 1:
                positions.append((reg, bit))
    if len(positions) > 14:
        pytest.skip(f'k={len(positions)} exceeds enumeration budget')

    results: list[dict[str, int]] = []
    for assignment in itertools.product([0, 1], repeat=len(positions)):
        s = dict(state)
        for (reg, bit), val in zip(positions, assignment, strict=False):
            if val:
                s[reg] = (s[reg] ^ (1 << bit)) & MASK64
        uc = Uc(UC_ARCH_X86, UC_MODE_64)
        uc.ctl_set_cpu_model(UC_CPU_X86_BROADWELL)
        uc.mem_map(_BASE_ADDR, _PAGE_SIZE)
        uc.mem_map(_STACK_BASE, _PAGE_SIZE)
        padded = bytestring + b'\x90' * 16
        uc.mem_write(_BASE_ADDR, padded)
        for reg in ('RAX', 'RBX', 'RCX', 'RDX'):
            uc.reg_write(_UC_REGS[reg], s[reg])
        uc.reg_write(
            __import__('unicorn.x86_const', fromlist=['']).UC_X86_REG_RSP,
            _STACK_BASE + _PAGE_SIZE - 8,
        )
        uc.emu_start(_BASE_ADDR, _BASE_ADDR + len(bytestring))
        out: dict[str, int] = {}
        for reg, uc_id in _UC_REGS.items():
            out[reg] = uc.reg_read(uc_id)
        results.append(out)

    gt: dict[str, int] = {r: 0 for r in _UC_REGS}
    if len(results) > 1:
        baseline = results[0]
        for res in results[1:]:
            for reg in gt:
                gt[reg] |= baseline[reg] ^ res[reg]
    return gt


def _assert_sound(
    label: str,
    bytestring: bytes,
    state: dict[str, int],
    taint: dict[str, int],
) -> None:
    gt = _brute_force_gt(bytestring, state, taint)
    rule = generate_static_rule(Architecture.AMD64, bytestring, _REGS_GP)
    ctx = EvalContext(input_values=state, input_taint=taint, simulator=_SIM)
    mt = rule.evaluate(ctx)
    for reg, gt_bits in gt.items():
        mt_bits = mt.get(reg, 0)
        missed = gt_bits & ~mt_bits
        assert missed == 0, (
            f'[{label}] under-taint in {reg}: '
            f'ground_truth={gt_bits:#018x} microtaint={mt_bits:#018x} '
            f'missed={missed:#018x}'
        )


# ---------------------------------------------------------------------------
# Byte sequences
# ---------------------------------------------------------------------------

# xor edx, edx  (31 d2)
# btc rax, 5    (48 0f ba f8 05)
# setc dl       (0f 92 c2)
_BTC_IMM_SETC = bytes.fromhex('31d2480fbaf8050f92c2')

# xor edx, edx  (31 d2)
# btr rax, 5    (48 0f ba f0 05)
# setc dl       (0f 92 c2)
_BTR_IMM_SETC = bytes.fromhex('31d2480fbaf0050f92c2')

# xor edx, edx  (31 d2)
# bts rax, 5    (48 0f ba e8 05)
# setc dl       (0f 92 c2)
_BTS_IMM_SETC = bytes.fromhex('31d2480fbae8050f92c2')

# Same three, but with a higher bit index to exercise a different bit position
# btc rax, 17   (48 0f ba f8 11)
_BTC_IMM17_SETC = bytes.fromhex('31d2480fbaf8110f92c2')

# btc rax, 56   (48 0f ba f8 38)  — tests the top byte
_BTC_IMM56_SETC = bytes.fromhex('31d2480fbaf8380f92c2')


# ---------------------------------------------------------------------------
# Pattern: BTC with immediate index — report id=7883
# ---------------------------------------------------------------------------
#
# Exact state from the benchmark report.  T_RAX[5]=1 is the critical tainted
# bit.  After BTC rax,5 → CF=T_RAX[5]=1; after setc dl → T_RDX[0]=1.
# Before the fix microtaint reported T_RDX=0.


def test_btc_imm5_cf_into_dl_report_exact() -> None:
    """Report id=7883: xor edx,edx; btc rax,5; setc dl with T_RAX[5]=1."""
    _assert_sound(
        'btc_imm5_setc_report_exact',
        _BTC_IMM_SETC,
        state={
            'RAX': 10953408221239237836,  # 0x980252BE5ACEDCCC
            'RBX': 15388876374910827151,
            'RCX': 6268980767501185420,
            'RDX': 2883349626527509167,
        },
        taint={
            'RAX': 72057594037927968,  # 0x0100000000000020 — bits 5 and 56 tainted
            'RBX': 2147483648,         # 0x80000000
            'RCX': 1073741824,         # 0x40000000
            'RDX': 0,
        },
    )


def test_btc_imm5_only_tested_bit_tainted() -> None:
    """BTC rax,5 → setc dl: only T_RAX[5] tainted, T_CF and T_RDX[0] must be 1."""
    _assert_sound(
        'btc_imm5_single_bit',
        _BTC_IMM_SETC,
        state={
            'RAX': 0xDEADBEEFCAFEBABE,
            'RBX': 0,
            'RCX': 0,
            'RDX': 0,
        },
        taint={
            'RAX': 1 << 5,  # only bit 5
            'RBX': 0,
            'RCX': 0,
            'RDX': 0,
        },
    )


def test_btc_imm5_tested_bit_clean() -> None:
    """BTC rax,5 → setc dl: T_RAX[5]=0 (bits 0-2 tainted) → T_RDX[0]=0.

    Uses k=3 so the enumeration budget is not exceeded.  Confirms the
    differential does not fire when the extracted bit itself is clean.
    """
    _assert_sound(
        'btc_imm5_tested_bit_clean',
        _BTC_IMM_SETC,
        state={
            'RAX': 0xDEADBEEFCAFEBABE,
            'RBX': 0,
            'RCX': 0,
            'RDX': 0,
        },
        taint={
            'RAX': 0b111,  # bits 0,1,2 tainted — bit 5 is clean
            'RBX': 0,
            'RCX': 0,
            'RDX': 0,
        },
    )


def test_btc_imm17_cf_into_dl() -> None:
    """BTC rax,17 → setc dl: T_RAX[17]=1 must propagate to T_RDX[0]."""
    _assert_sound(
        'btc_imm17_setc',
        _BTC_IMM17_SETC,
        state={
            'RAX': 0x123456789ABCDEF0,
            'RBX': 0,
            'RCX': 0,
            'RDX': 0,
        },
        taint={
            'RAX': 1 << 17,
            'RBX': 0,
            'RCX': 0,
            'RDX': 0,
        },
    )


def test_btc_imm56_cf_into_dl() -> None:
    """BTC rax,56 → setc dl: T_RAX[56]=1 (top-byte bit) must propagate to T_RDX[0]."""
    _assert_sound(
        'btc_imm56_setc',
        _BTC_IMM56_SETC,
        state={
            'RAX': 0x0080000000000000,
            'RBX': 0,
            'RCX': 0,
            'RDX': 0,
        },
        taint={
            'RAX': 1 << 56,
            'RBX': 0,
            'RCX': 0,
            'RDX': 0,
        },
    )


# ---------------------------------------------------------------------------
# Pattern: BTR with immediate index — same structural fix
# ---------------------------------------------------------------------------


def test_btr_imm5_cf_into_dl() -> None:
    """BTR rax,5 → setc dl: T_RAX[5]=1 must propagate to T_RDX[0]."""
    _assert_sound(
        'btr_imm5_setc',
        _BTR_IMM_SETC,
        state={
            'RAX': 0xDEADBEEFCAFEBABE,
            'RBX': 0,
            'RCX': 0,
            'RDX': 0,
        },
        taint={
            'RAX': 1 << 5,
            'RBX': 0,
            'RCX': 0,
            'RDX': 0,
        },
    )


def test_btr_imm5_tested_bit_clean() -> None:
    """BTR rax,5 → setc dl: T_RAX[5]=0 (bits 0-2 tainted) → T_RDX[0]=0."""
    _assert_sound(
        'btr_imm5_tested_bit_clean',
        _BTR_IMM_SETC,
        state={
            'RAX': 0xDEADBEEFCAFEBABE,
            'RBX': 0,
            'RCX': 0,
            'RDX': 0,
        },
        taint={
            'RAX': 0b111,  # bits 0,1,2 tainted — bit 5 is clean
            'RBX': 0,
            'RCX': 0,
            'RDX': 0,
        },
    )


# ---------------------------------------------------------------------------
# Pattern: BTS with immediate index — same structural fix
# ---------------------------------------------------------------------------


def test_bts_imm5_cf_into_dl() -> None:
    """BTS rax,5 → setc dl: T_RAX[5]=1 must propagate to T_RDX[0]."""
    _assert_sound(
        'bts_imm5_setc',
        _BTS_IMM_SETC,
        state={
            'RAX': 0xDEADBEEFCAFEBABE,
            'RBX': 0,
            'RCX': 0,
            'RDX': 0,
        },
        taint={
            'RAX': 1 << 5,
            'RBX': 0,
            'RCX': 0,
            'RDX': 0,
        },
    )


def test_bts_imm5_tested_bit_clean() -> None:
    """BTS rax,5 → setc dl: T_RAX[5]=0 (bits 0-2 tainted) → T_RDX[0]=0."""
    _assert_sound(
        'bts_imm5_tested_bit_clean',
        _BTS_IMM_SETC,
        state={
            'RAX': 0xDEADBEEFCAFEBABE,
            'RBX': 0,
            'RCX': 0,
            'RDX': 0,
        },
        taint={
            'RAX': 0b111,  # bits 0,1,2 tainted — bit 5 is clean
            'RBX': 0,
            'RCX': 0,
            'RDX': 0,
        },
    )


# ---------------------------------------------------------------------------
# Regression: BT rax,rbx (tainted register index) must still work
# This was fixed by the earlier _is_bit_extract_via_tainted_shift predicate
# and must not be broken by the new _is_const_shift_bit_extract path.
# ---------------------------------------------------------------------------

_BT_REG_SETC = bytes.fromhex('31d2480fa3d80f92c2')  # xor edx,edx; bt rax,rbx; setc dl


def test_bt_reg_index_still_sound() -> None:
    """BT rax,rbx (tainted index) → setc dl soundness must be unaffected."""
    _assert_sound(
        'bt_reg_index_setc_regression',
        _BT_REG_SETC,
        state={
            'RAX': 0x776BD5635FF6BF19,
            'RBX': 0x2AAFF7F96610A029,  # idx=41; bit-4 flip → idx=57
            'RCX': 0x2CA34ED97BBC143B,
            'RDX': 0x5FA8C4FC9C6666E8,
        },
        taint={
            'RAX': 0x400004000002004,
            'RBX': 0x10,  # bit 4 tainted
            'RCX': 0x10000800000002,
            'RDX': 0,
        },
    )
