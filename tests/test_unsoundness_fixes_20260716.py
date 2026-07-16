"""Soundness regression tests for the three under-taints found on 2026-07-16 by
the full-corpus single-bit-flip noninterference oracle (benchmark cases 7899,
8094 simd; 8244, 8331 memory; 9835-9845 path_explosion_branching).

Each was a decomposition/classification slip, not a rule-family failure:

  1. Opaque CALLOTHER SIMD intrinsics (pshufb, VEX vpaddb/vpshufb/...) lift to a
     CALLOTHER that was filtered out as an ignored control-flow op, leaving the
     slice mis-classified MAPPED -> bare differential (no soundness floor), which
     cancels on shared tainted lanes and under-taints. Fix: any CALLOTHER slice
     is AVALANCHE (sound over-approximation).

  2. sub/sbb/neg/cmp with a memory operand: the subtractive polarity D^{+-}
     degraded to D^{++} because the LOAD's polarity did not cross the STORE->LOAD
     edge, so borrow bits cancelled and were dropped. Fix: forward the polarity
     across the memory edge.

  3. Intra-sequence forward-skip CBRANCH sequences (e.g. `test rbx,1; jz +3;
     add rax,rbx; shr rbx,1`) were routed to a monolithic circuit that dropped
     the MSB/LSB boundary bits and was non-deterministic. Fix: chain such
     sequences through the existing per-instruction ChainedCircuit.

Tests brute-force ground truth via 2^k Unicorn enumeration and assert
``microtaint_output >= ground_truth`` (no under-tainted bits). Taint configs are
chosen to trigger each bug (they under-taint on the pre-fix engine).
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
    UC_X86_REG_RSP,
)

from microtaint.classifier.categories import InstructionCategory
from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule, get_context
from microtaint.sleigh.mapper import determine_category
from microtaint.types import Architecture, Register

MASK64 = 0xFFFFFFFFFFFFFFFF
_ARCH = Architecture.AMD64
# Include RSP so memory-operand rules (e.g. `sub rax,[rsp-16]`) resolve the
# address the same way the benchmark worker does; GP regs are what we score.
_REGS_GP = [Register('RAX', 64), Register('RBX', 64), Register('RCX', 64), Register('RDX', 64), Register('RSP', 64)]
_SIM = CellSimulator(_ARCH)
_UC_REGS = {'RAX': UC_X86_REG_RAX, 'RBX': UC_X86_REG_RBX, 'RCX': UC_X86_REG_RCX, 'RDX': UC_X86_REG_RDX}
_BASE_ADDR = 0x400000
_PAGE_SIZE = 0x1000
_STACK_BASE = 0x500000
_RSP = _STACK_BASE + _PAGE_SIZE - 256  # leave head-room for [rsp-16] stores


def _brute_force_gt(bytestring: bytes, state: dict[str, int], taint: dict[str, int]) -> dict[str, int]:
    """Exact bit-level ground truth via 2^k Unicorn enumeration."""
    positions = [(reg, bit) for reg, mask in taint.items() for bit in range(64) if (mask >> bit) & 1]
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
        uc.mem_write(_BASE_ADDR, bytestring + b'\x90' * 16)
        for reg in ('RAX', 'RBX', 'RCX', 'RDX'):
            uc.reg_write(_UC_REGS[reg], s[reg])
        uc.reg_write(UC_X86_REG_RSP, _RSP)
        uc.emu_start(_BASE_ADDR, _BASE_ADDR + len(bytestring))
        results.append({reg: uc.reg_read(uc_id) for reg, uc_id in _UC_REGS.items()})
    gt = dict.fromkeys(_UC_REGS, 0)
    baseline = results[0]
    for res in results[1:]:
        for reg in gt:
            gt[reg] |= baseline[reg] ^ res[reg]
    return gt


def _eval(bytestring: bytes, state: dict[str, int], taint: dict[str, int]) -> dict[str, int]:
    rule = generate_static_rule(_ARCH, bytestring, _REGS_GP)
    # RSP must match the ground-truth harness so memory operands ([rsp-16], ...)
    # resolve to the same address; harmless for non-memory instructions.
    ctx = EvalContext(input_values={'RSP': _RSP, **state}, input_taint=taint, simulator=_SIM)
    return rule.evaluate(ctx)


def _assert_sound(label: str, bytestring: bytes, state: dict[str, int], taint: dict[str, int]) -> None:
    gt = _brute_force_gt(bytestring, state, taint)
    mt = _eval(bytestring, state, taint)
    for reg, gt_bits in gt.items():
        missed = gt_bits & ~mt.get(reg, 0)
        assert missed == 0, (
            f'[{label}] under-taint in {reg}: gt={gt_bits:#018x} '
            f'mt={mt.get(reg, 0):#018x} missed={missed:#018x}'
        )


# --------------------------------------------------------------------------- #
# Fix 1 — CALLOTHER intrinsics classify as AVALANCHE                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ('label', 'hexbytes'),
    [
        ('pshufb', '660f3800c1'),        # pshufb xmm0, xmm1
        ('vpaddb', 'c5f1fcc2'),          # vpaddb xmm0, xmm1, xmm2
        ('vpshufb', 'c4e27100c2'),       # vpshufb xmm0, xmm1, xmm2
    ],
)
def test_callother_classified_avalanche(label, hexbytes):
    """Any slice containing a CALLOTHER (opaque intrinsic) must be AVALANCHE,
    never MAPPED (the mis-classification that dropped the soundness floor)."""
    ops = list(get_context(_ARCH).translate(bytes.fromhex(hexbytes), 0x1000).ops)
    assert determine_category(ops) == InstructionCategory.AVALANCHE, label


# NOTE: end-to-end soundness of the *marshalled* pshufb/vpaddb benchmark cases
# (7899, 8094) is verified against the full-corpus bit-flip oracle (both go from
# under-tainting pre-fix to sound); the AVALANCHE rule's own soundness is proved
# in benchmark/soundness/. Here we lock in the classification, which is the fix.


# --------------------------------------------------------------------------- #
# Fix 2 — subtraction borrow-chain through memory                             #
# --------------------------------------------------------------------------- #
def test_sub_borrow_through_memory_sound():
    """`mov [rsp-16],rbx; sub rax,[rsp-16]` with overlapping tainted bits: the
    D^{+-} borrow chain must be captured (pre-fix D^{++} cancels and drops it)."""
    bs = bytes.fromhex('48895c24f0482b4424f0')
    _assert_sound('sub-mem-borrow', bs, {'RAX': 0x10, 'RBX': 0x10, 'RCX': 0, 'RDX': 0},
                  {'RAX': 0x10, 'RBX': 0x10, 'RCX': 0, 'RDX': 0})


# --------------------------------------------------------------------------- #
# Fix 3 — intra-sequence forward-skip branch sequences                        #
# --------------------------------------------------------------------------- #
# xor rax,rax ; [ test rbx,1 ; jz +3 ; add rax,rbx ; shr rbx,1 ] x2
_BRANCH_N2 = bytes.fromhex('4831c048f7c30100000074034801d848d1eb48f7c30100000074034801d848d1eb')


def test_forward_skip_branch_sound():
    """Boundary bits (bit 0 / bit 63) that the monolithic multi-CBRANCH
    differential dropped must now be tainted. RBX bit0=1 keeps every branch
    not-taken so the adds execute."""
    _assert_sound('branch-boundary-bits', _BRANCH_N2,
                  {'RAX': 0, 'RBX': MASK64, 'RCX': 0, 'RDX': 0},
                  {'RAX': 0, 'RBX': 0x8000000000000001, 'RCX': 0, 'RDX': 0})


def test_forward_skip_branch_deterministic():
    """Taint must be a pure function of (bytes,state,taint): evaluating other
    sequences in between must not perturb the branch result (the shared-simulator
    non-determinism the monolithic path exhibited)."""
    state = {'RAX': 0, 'RBX': MASK64, 'RCX': 0, 'RDX': 0}
    taint = {'RAX': 0, 'RBX': MASK64, 'RCX': 0, 'RDX': 0}
    first = _eval(_BRANCH_N2, state, taint)['RAX']
    # pollute the shared simulator with unrelated sequences
    for poll in ('4801d8', '48d1eb48f7c301000000'):  # add rax,rbx ; shr rbx,1; test rbx,1
        _eval(bytes.fromhex(poll), state, taint)
    second = _eval(_BRANCH_N2, state, taint)['RAX']
    assert first == second == MASK64, f'non-deterministic/unsound: {first:#x} vs {second:#x}'


# --------------------------------------------------------------------------- #
# Fix 4 — COND_TRANSPORTABLE flags must OR in the 2-replica differential       #
# --------------------------------------------------------------------------- #
def test_shl_cf_bit_extract_reaches_setc():
    """`shl rax,4; setc dl` — CF is bit 60 of RAX, a monotone bit-copy.

    COND_TRANSPORTABLE derives a flag from a SINGLE masked replica (C_eval on
    V&~T).  Masking the tainted bit 60 to 0 makes CF=0, so `C_eval AND T_any`
    yields T_CF=0 — an under-taint, even though CF *is* exactly that tainted bit.
    The 2-replica differential XOR(C_eval(V|T), C_eval(V&~T)) = 1^0 = 1 is exact,
    and is now OR-ed into every 1-bit flag output.
    """
    bs = bytes.fromhex('48c1e0040f92c2')  # shl rax, 4 ; setc dl
    bit60 = 1 << 60
    state = {'RAX': bit60, 'RBX': 0, 'RCX': 0, 'RDX': 0}
    taint = {'RAX': bit60, 'RBX': 0, 'RCX': 0, 'RDX': 0}
    _assert_sound('shl-cf-bit-extract', bs, state, taint)
    assert _eval(bs, state, taint)['RDX'] & 1, 'CF (= bit 60 of RAX) must reach DL[0]'


# --------------------------------------------------------------------------- #
# Fix 5 — signed overflow (INT_SBORROW): exact sign decomposition              #
# --------------------------------------------------------------------------- #
# Inputs on which the engine's POLARISED 2-replica differential coincidentally
# cancels -- OF(a|Ta, b&~Tb) == OF(a&~Ta, b|Tb) == 0 -- yet flipping an interior
# tainted bit DOES change OF.  Signed overflow is non-monotone, so no 2-corner
# sample can see this.  Only the sign decomposition recovers it:
#     OF  = (a_s ^ b_s) & (b_s ^ Bor),   Bor = [ a[0:63] <u b[0:63] ]  (monotone)
# a_s, b_s and Bor read DISJOINT bits, so they vary independently and the <=2^3
# enumeration is exact.  Machine-checked in benchmark/soundness/prove_signed_overflow.py
# (identity + no-under-taint PROVED for w=2..64).
_SUB_OF_BYTES = bytes.fromhex('4829d80f90c2')  # sub rax, rbx ; seto dl
_SUB_OF_STATE = {'RAX': 2678491694169162878, 'RBX': 9449111174765093383, 'RCX': 0, 'RDX': 0}
_SUB_OF_TAINT = {'RAX': 9225623836668592128, 'RBX': 4611686018427387904, 'RCX': 0, 'RDX': 0}


def test_sub_signed_overflow_sound_when_differential_cancels():
    """OF must be tainted even though both differential corners agree."""
    _assert_sound('sub-of-sign-decomp', _SUB_OF_BYTES, _SUB_OF_STATE, _SUB_OF_TAINT)
    out = _eval(_SUB_OF_BYTES, _SUB_OF_STATE, _SUB_OF_TAINT)
    assert out['RDX'] & 1, 'OF must reach DL[0] on inputs where the differential cancels'


def test_sub_signed_overflow_is_exact_not_a_floor():
    """The sign decomposition is EXACT, not a soundness floor: the taint must
    EQUAL ground truth.  An avalanche/any-taint floor would over-taint here."""
    gt = _brute_force_gt(_SUB_OF_BYTES, _SUB_OF_STATE, _SUB_OF_TAINT)
    mt = _eval(_SUB_OF_BYTES, _SUB_OF_STATE, _SUB_OF_TAINT)
    for reg, gt_bits in gt.items():
        assert mt.get(reg, 0) == gt_bits, (
            f'signed-overflow rule not exact in {reg}: mt={mt.get(reg, 0):#018x} gt={gt_bits:#018x}'
        )
