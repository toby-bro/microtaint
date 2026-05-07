"""
Soundness regression tests for three unsoundness patterns identified in
report_1778107947.json (the May 7, 2026 NDSS-orchestrator run on
benchmark.py with microtaint as the only worker).

  Pattern 1 — SHLD with concrete shift count = 0
              `shld rax, rbx, cl` collapses to a no-op for the destination
              when cl == 0, but microtaint drops the input taint of RAX.
              Report id = 3252.

  Pattern 2 — BEXTR upper-half source bits dropped
              `bextr rax, rbx, rcx` extracts a window of `length` bits
              starting at `start` from rbx.  When start>0 and a tainted
              bit of rbx sits in the upper half of the window, microtaint
              fails to propagate that bit to the corresponding output
              position.  Report id = 5337.

  Pattern 3 — BT with tainted bit-index — CF taint not modelled
              `bt rax, rbx; setc dl` writes bit (rbx mod 64) of rax into
              CF.  When rbx has a tainted bit in its low 6 bits, the
              index varies across runs and CF should be tainted whenever
              the source register has differing bits at the reachable
              positions.  microtaint reports DL clean.  Report id = 8336.

Each test computes the brute-force ground truth by 2^k Unicorn enumeration
and asserts `microtaint_output ⊇ ground_truth` (no under-tainted bits).

We deliberately do NOT cover the `rep stosb` cases (ids 8009, 8337) — the
benchmark suggests microtaint's string-instruction handler has a deeper
shadow-memory issue that's out of scope for this fix.
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
_REGS_GP = [Register('RAX', 64), Register('RBX', 64), Register('RCX', 64), Register('RDX', 64)]
_SIM = CellSimulator(Architecture.AMD64)
_UC_REGS = {
    'RAX': UC_X86_REG_RAX,
    'RBX': UC_X86_REG_RBX,
    'RCX': UC_X86_REG_RCX,
    'RDX': UC_X86_REG_RDX,
}
_BASE_ADDR = 0x400000
_PAGE_SIZE = 0x1000


def _brute_force_gt(
    bytestring: bytes,
    state: dict[str, int],
    taint: dict[str, int],
) -> dict[str, int]:
    """Compute exact bit-level ground truth by 2^k Unicorn enumeration."""
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
        # Map a stack page so the bt-sequence's xor edx,edx and any future
        # tests that touch the stack don't fault.
        stack_base = 0x500000
        uc.mem_map(stack_base, _PAGE_SIZE)
        padded = bytestring + b'\x90' * 16
        uc.mem_write(_BASE_ADDR, padded)
        for reg in ('RAX', 'RBX', 'RCX', 'RDX'):
            uc.reg_write(_UC_REGS[reg], s[reg])
        # RSP at top of stack page so we don't fault on push/pop or
        # implicit stack writes from any tested sequence.
        from unicorn.x86_const import UC_X86_REG_RSP
        uc.reg_write(UC_X86_REG_RSP, stack_base + _PAGE_SIZE - 0x100)
        uc.emu_start(_BASE_ADDR, _BASE_ADDR + len(bytestring))
        results.append({reg: uc.reg_read(_UC_REGS[reg]) for reg in ('RAX', 'RBX', 'RCX', 'RDX')})

    output_taint: dict[str, int] = {}
    for reg in ('RAX', 'RBX', 'RCX', 'RDX'):
        all_or = 0
        all_and = MASK64
        for r in results:
            all_or |= r[reg]
            all_and &= r[reg]
        output_taint[reg] = all_or ^ all_and
    return output_taint


def _eval_microtaint(
    bytestring: bytes,
    state: dict[str, int],
    taint: dict[str, int],
    regs: list[Register] | None = None,
) -> dict[str, int]:
    """Run microtaint's static rule generator + circuit evaluator."""
    regs = regs or _REGS_GP
    circuit = generate_static_rule(Architecture.AMD64, bytestring, regs)
    full_state = {r.name: state.get(r.name, 0) for r in regs}
    full_taint = {r.name: taint.get(r.name, 0) for r in regs}
    ctx = EvalContext(input_values=full_state, input_taint=full_taint, simulator=_SIM)
    raw = circuit.evaluate(ctx)
    return {r.name: (raw.get(r.name, 0) & MASK64 if isinstance(raw.get(r.name, 0), int) else 0) for r in regs}


def _assert_sound(
    label: str,
    bytestring: bytes,
    state: dict[str, int],
    taint: dict[str, int],
    *,
    regs: list[Register] | None = None,
    check_regs: tuple[str, ...] = ('RAX', 'RBX', 'RCX', 'RDX'),
) -> None:
    """Run microtaint and the brute-force GT, then assert no missed bits."""
    gt = _brute_force_gt(bytestring, state, taint)
    mt = _eval_microtaint(bytestring, state, taint, regs=regs)
    misses = []
    for reg in check_regs:
        gv = gt[reg]
        tv = mt.get(reg, 0)
        under = gv & ~tv & MASK64
        if under:
            misses.append(f'{reg}: under=0x{under:016x} (GT=0x{gv:016x}, MT=0x{tv:016x})')
    assert not misses, (
        f'{label}: under-tainted bits — microtaint must not miss any taint that '
        f'GT proves dependent.\n  ' + '\n  '.join(misses)
    )


# =============================================================================
# Pattern 1 — SHLD with concrete shift count == 0
# =============================================================================
#
# `shld rax, rbx, cl`  encoded as 48 0f a5 d8.
# When cl == 0 the instruction leaves RAX unchanged, so the input taint on
# RAX must survive in the output.  Report id 3252 had cl=0 (rcx low-6-bits
# all clear; the only tainted bit of rcx is bit 28, far above cl's range)
# and microtaint dropped RAX's bit-3 taint.

_SHLD_RAX_RBX_CL = bytes.fromhex('480fa5d8')


def test_shld_cl_zero_preserves_rax_taint() -> None:
    """Report id=3252: shld rax, rbx, cl  with concrete cl=0."""
    _assert_sound(
        'shld_rax_rbx_cl_zero',
        _SHLD_RAX_RBX_CL,
        state={
            'RAX': 0x9650280B0F06F0F2,
            'RBX': 0x6C361D4C20B09F3A,
            'RCX': 0xC0B64FE0DEDB3B80,  # low 6 bits == 0 -> cl = 0
            'RDX': 0xA0CDD4730B913907,
        },
        taint={'RAX': 0x8, 'RCX': 0x10000000},
    )


# =============================================================================
# Pattern 2 — BEXTR upper-half source bits dropped
# =============================================================================
#
# `bextr rax, rbx, rcx`  encoded as c4 e2 f0 f7 c3 (VEX.LZ.0F38.W1).
# Semantics: start = rcx[7:0]; length = rcx[15:8];
#            result = (rbx >> start) & ((1 << length) - 1)  (length capped to 64-start).
# Report id 5337: start=15, length=92 -> effective length 49, window covers
# bits 15..63 of rbx.  Tainted rbx bits {28, 44} should map to result bits
# {13, 29}.  microtaint produced bit 13 only — bit 29 was lost.

_BEXTR_RAX_RBX_RCX = bytes.fromhex('c4e2f0f7c3')


def test_bextr_upper_window_bit_propagated() -> None:
    """Report id=5337: bextr rax, rbx, rcx  with upper-window source taint."""
    _assert_sound(
        'bextr_upper_window',
        _BEXTR_RAX_RBX_RCX,
        state={
            'RAX': 0x6F9DFC3BA08C7064,
            'RBX': 0xED7C7E504E0ED847,
            'RCX': 0xA430B7FDDB275C0F,  # start=0x0f=15, length=0x5c=92
            'RDX': 0x14F72BFE59573FA0,
        },
        taint={'RAX': 0x10800000000000, 'RBX': 0x100010000000},
    )


# =============================================================================
# Pattern 3 — BT with tainted bit-index, CF -> setc dl loses taint
# =============================================================================
#
# `xor edx, edx; bt rax, rbx; setc dl`  encoded as 31 d2 48 0f a3 d8 0f 92 c2.
# bt copies bit (rbx & 0x3f) of rax into CF, then setc dl materialises CF in
# dl[0].  Report id 8336: rbx has bit 4 tainted, so the index can flip
# between (concrete_idx) and (concrete_idx ^ 16).  The two source bits at
# those indices in rax differ -> CF varies -> dl[0] must be tainted.
# microtaint reported dl clean.

_BT_SETC = bytes.fromhex('31d2480fa3d80f92c2')


def test_bt_with_tainted_index_taints_cf_into_dl() -> None:
    """Report id=8336: bt rax, rbx with tainted bit-index propagates to dl[0]."""
    _assert_sound(
        'flag_only_bt_cf_setc',
        _BT_SETC,
        state={
            'RAX': 0x776BD5635FF6BF19,
            'RBX': 0x2AAFF7F96610A029,  # idx = 41; with bit-4 flipped -> 57
            'RCX': 0x2CA34ED97BBC143B,
            'RDX': 0x5FA8C4FC9C6666E8,
        },
        taint={
            'RAX': 0x400004000002004,
            'RBX': 0x10,                   # bit 4 of rbx tainted
            'RCX': 0x10000800000002,
        },
    )
