"""
Soundness regression tests for the four unsoundness patterns identified in
report_1778073337.json.  Each test reproduces the exact case (state, taint,
bytestring) where microtaint produced an under-tainted output, and asserts
that the new implementation is sound.

  Pattern 1 — PEXT (and PDEP) — engine.py software-loop detection
  Pattern 2 — ADCX — slicer.py overlap-aware backward slice
  Pattern 3 — SIMD GP↔XMM roundtrip — worker passes XMM regs in state_format
  Pattern 4 — MULX — slicer.py overlap-aware backward slice (same fix as ADCX)

Each test computes the brute-force ground truth by 2^k Unicorn enumeration and
checks `microtaint_output ⊇ ground_truth` (no under-tainted bits).
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
# SIMD tests need the XMM lane regs in state_format so the engine tracks
# taint flow across the GP↔XMM boundary.  Fix 3 changes the worker (and
# any downstream caller) to include these.
_REGS_GP_XMM = (
    _REGS_GP + [Register(f'XMM{n}_LO', 64) for n in range(8)] + [Register(f'XMM{n}_HI', 64) for n in range(8)]
)
_SIM = CellSimulator(Architecture.AMD64)
_UC_REGS = {
    'RAX': UC_X86_REG_RAX,
    'RBX': UC_X86_REG_RBX,
    'RCX': UC_X86_REG_RCX,
    'RDX': UC_X86_REG_RDX,
}
_BASE_ADDR = 0x400000
_PAGE_SIZE = 0x1000


def _brute_force_gt(  # noqa: C901
    bytestring: bytes,
    state: dict[str, int],
    taint: dict[str, int],
) -> dict[str, int]:
    """Compute exact bit-level ground truth by 2^k Unicorn enumeration.

    For each tainted input bit, run Unicorn with the bit unflipped and flipped
    relative to the base state; bit i of the output is tainted iff its value
    differs across at least two assignments.
    """
    positions: list[tuple[str, int]] = []
    for reg, mask in taint.items():
        for bit in range(64):
            if (mask >> bit) & 1:
                positions.append((reg, bit))
    if len(positions) > 12:
        pytest.skip(f'k={len(positions)} exceeds enumeration budget')

    results: list[dict[str, int]] = []
    for assignment in itertools.product([0, 1], repeat=len(positions)):
        s = dict(state)
        for (reg, bit), val in zip(positions, assignment, strict=False):
            if val:
                s[reg] = (s[reg] ^ (1 << bit)) & MASK64
        # Fresh Unicorn per assignment to avoid state bleed-through.
        # Broadwell CPU model is needed for BMI2 instructions (ADCX, MULX,
        # PEXT, PDEP) — the default model rejects them as invalid.
        uc = Uc(UC_ARCH_X86, UC_MODE_64)
        uc.ctl_set_cpu_model(UC_CPU_X86_BROADWELL)
        uc.mem_map(_BASE_ADDR, _PAGE_SIZE)
        # Pad with NOPs after the bytestring so Unicorn doesn't fault on
        # speculative fetches when we emu_start with end=base+len(bytestring).
        padded = bytestring + b'\x90' * 16
        uc.mem_write(_BASE_ADDR, padded)
        for reg in ('RAX', 'RBX', 'RCX', 'RDX'):
            uc.reg_write(_UC_REGS[reg], s[reg])
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
    # State and taint dicts must contain every register name in `regs`,
    # even if extras (XMM*) are 0.
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
# Pattern 1 — PEXT / PDEP software-loop avalanche
# =============================================================================
#
# `pext rax, rbx, rcx` packs bits of rbx selected by mask rcx into the LSBs
# of rax.  The loop body in Sleigh's PEXT semantics has a backward CBRANCH
# that the slicer never sees (CBRANCH ops have no `output`), so the loop
# structure is invisible at slice level.  Our fix detects backward CBRANCH
# in `translation.ops` and forces full-width AVALANCHE.  This is sound (any
# tainted input bit can affect any output bit position).
#
# All 7 cases come straight from report_1778073337.json IDs 181/478/761/1978/
# 1979/1983/1984.

_PEXT_RAX_RBX_RCX = bytes.fromhex('c4e2e2f5c1')
_PEXT_RCX_RDX_RAX = bytes.fromhex('c4e2eaf5c8')
_PDEP_RAX_RBX_RCX = bytes.fromhex('c4e2e3f5c1')


@pytest.mark.parametrize(
    ('state', 'taint'),
    [
        # id=181
        (
            {
                'RAX': 0x6CECDF821903F9B3,
                'RBX': 0x8B31F4D8E073C1D0,
                'RCX': 0x3063AE7F41FF81D1,
                'RDX': 0x850F1CDC1E2A03F1,
            },
            {'RCX': 0x200000000},
        ),
        # id=478
        (
            {
                'RAX': 0xFE3F566204680A4A,
                'RBX': 0x1BB9D499B6EEA1CF,
                'RCX': 0x72742E763B6BECB7,
                'RDX': 0x902AD0197A0FFA86,
            },
            {'RAX': 0x1000, 'RBX': 0x4000000080000000, 'RDX': 0x8000000120000080},
        ),
        # id=761
        (
            {'RAX': 0x4C442D17D98D2038, 'RBX': 0x54B279CD0C248F00, 'RCX': 0xED50A84E71BFD71B, 'RDX': 0xCD4BA10B4666D99},
            {'RAX': 0x10000000000, 'RBX': 0x20020020000, 'RCX': 0x410080000000000, 'RDX': 0x800},
        ),
        # id=1978
        (
            {'RAX': 0x2298E3D0CB5718CE, 'RBX': 0x1F923C2DA511E769, 'RCX': 0x7B873EEC9FB8785, 'RDX': 0x4169ED13BE0AF8F5},
            {'RAX': 0x4010000800080, 'RCX': 0x100002020, 'RDX': 0x100000000001003},
        ),
        # id=1979
        (
            {
                'RAX': 0x45DC97CB2EF2D364,
                'RBX': 0x2CABBC0BA8B1792F,
                'RCX': 0xF5671BAB96D0398B,
                'RDX': 0xDA7DB85332C8B2B1,
            },
            {'RAX': 0x8400000000000000, 'RCX': 0x1000000000000, 'RDX': 0x20000000000},
        ),
    ],
    ids=['id181', 'id478', 'id761', 'id1978', 'id1979'],
)
def test_pext_rax_sound_under_random_input_taint(state, taint):
    """PEXT must avalanche: any tainted mask/source bit affects all output positions."""
    _assert_sound('pext rax,rbx,rcx', _PEXT_RAX_RBX_RCX, state, taint)


@pytest.mark.parametrize(
    ('state', 'taint'),
    [
        # id=1983
        (
            {'RAX': 0x971D2DA3A5A57309, 'RBX': 0x1797295A996976F6, 'RCX': 0x904DC2698A91420, 'RDX': 0xFAB6819CB53E4FE3},
            {'RAX': 0x1000000000080800, 'RBX': 0x2020000000, 'RCX': 0x88080040, 'RDX': 0x2800000000000},
        ),
        # id=1984
        (
            {
                'RAX': 0x6137D61EE3B33B65,
                'RBX': 0xC25F8C328F6BA093,
                'RCX': 0xD754C81F76149744,
                'RDX': 0xF08A9F63F7131A89,
            },
            {'RAX': 0x200000000000, 'RBX': 0x20000000, 'RCX': 0x20, 'RDX': 0x2},
        ),
    ],
    ids=['id1983', 'id1984'],
)
def test_pext_rcx_rdx_rax_sound(state, taint):
    """The other PEXT operand permutation."""
    _assert_sound('pext rcx,rdx,rax', _PEXT_RCX_RDX_RAX, state, taint)


def test_pdep_avalanche_basic():
    """PDEP is the inverse of PEXT and uses the same software-loop structure.
    Same software-loop detection should apply — any tainted source/mask bit
    must avalanche to the full output mask."""
    state = {'RAX': 0, 'RBX': 0xFF, 'RCX': 0x0123456789ABCDEF, 'RDX': 0}
    # k=8 — within enumeration budget
    taint = {'RBX': 0x0F}
    _assert_sound('pdep rax,rbx,rcx', _PDEP_RAX_RBX_RCX, state, taint)


# =============================================================================
# Pattern 2 — ADCX (BMI2 add-with-carry, integer extended)
# =============================================================================
#
# `adcx rax, rbx` lifts to:
#   ZEXT(rax,9) + ZEXT(rbx,9) + ZEXT(CF,9)  → 9-bit unique
#   COPY low 8 bytes → RAX
#   SUBPIECE high bit → CF
#
# The slicer used to drop the INT_ADDs because the COPY's source size (8)
# didn't match the unique varnode's true size (9).  Our overlap-aware
# slice_backward now includes ops whose output range overlaps any worklist
# entry, restoring the carry chain.
#
# Reproducer: id=151 from report_1778073337.json.

_ADCX_RAX_RBX = bytes.fromhex('66480f38f6c3')


def test_adcx_carry_chain_sound():
    """ADCX must propagate carry through tainted operands soundly."""
    state = {'RAX': 0xE64B9B37AE01A122, 'RBX': 0x16DC5A6EA7890770, 'RCX': 0x4E9D23E9DF162EC3, 'RDX': 0xE3082DDCC9C04B7D}
    taint = {'RAX': 0x1000001008000000, 'RBX': 0x800000000, 'RCX': 0x400000, 'RDX': 0x3200000000000}
    _assert_sound('adcx rax,rbx', _ADCX_RAX_RBX, state, taint)


# =============================================================================
# Pattern 3 — MULX (BMI2 64*64→128, low/high half write)
# =============================================================================
#
# `mulx rax, rbx, rcx` performs RDX * RCX → 128-bit, writes high half to RAX,
# low half to RBX.  The COPY of the low 8 bytes of the 16-byte INT_MULT
# output had the same overlap problem as ADCX's 9-bit ADD: the slicer's
# exact-size match dropped INT_MULT, leaving the slice as just COPY (mapped).
#
# Same overlap-aware slicer fix resolves this.  Once INT_MULT is in the
# slice, AVALANCHE_OPCODES already contains it and the rule fires.
#
# Reproducer: id=1449.

_MULX_RAX_RBX_RCX = bytes.fromhex('c4e2e3f6c1')


def test_mulx_low_half_sound():
    """The 128-bit product's low half (RBX) must avalanche from any tainted
    multiplicand bit, including high bits that affect carries."""
    state = {'RAX': 0x548E2DF49A2F8623, 'RBX': 0x700D5278B081391C, 'RCX': 0xE42390236C597F2E, 'RDX': 0xF0A4F07E43036244}
    taint = {'RBX': 0x20000, 'RCX': 0x5000, 'RDX': 0x800000}
    _assert_sound('mulx rax,rbx,rcx', _MULX_RAX_RBX_RCX, state, taint)


# =============================================================================
# Pattern 4 — SIMD GP↔XMM roundtrip
# =============================================================================
#
# `movq xmm0, rax; movq xmm1, rbx; paddq xmm0, xmm1; movq rax, xmm0`
# Taint flows GP→XMM lane→XMM lane→GP.  The engine already supports XMM via
# StateMapper._synth_xmm_varnode, but only when the caller passes XMM regs
# in state_format.  The previous worker only passed GP regs, so taint on
# the XMM transit was silently dropped (the engine's StateMapper had no
# slot to write it into).
#
# Fix: include XMM<n>_LO/_HI for n=0..7 in the register list.  The engine's
# state-format machinery does the rest.  This regression test verifies the
# fix from the engine side using an explicit register list.
#
# Reproducer: id=902.

_SIMD_PADDQ_ROUNDTRIP = bytes.fromhex('66480f6ec066480f6ecb660fd4c166480f7ec0')


def test_simd_paddq_xmm_roundtrip_sound():
    """Taint MUST survive a GP→XMM→PADDQ→GP roundtrip when XMM regs are
    tracked in state_format."""
    state = {'RAX': 0xE4723A97A7178558, 'RBX': 0x34F5B8EBAEE5B68C, 'RCX': 0xF2128A487A6D9868, 'RDX': 0xF16DA24334DEF7EE}
    taint = {'RBX': 0x800000000041, 'RCX': 0x80000000000000, 'RDX': 0x8000010202000}
    _assert_sound(
        'movq xmm0,rax; movq xmm1,rbx; paddq; movq rax,xmm0',
        _SIMD_PADDQ_ROUNDTRIP,
        state,
        taint,
        regs=_REGS_GP_XMM,
        check_regs=('RAX',),  # only RAX matters for this case
    )


def test_simd_paddq_without_xmm_state_format_documents_limitation():
    """When XMM regs are NOT in state_format the engine cannot track taint
    across the GP↔XMM boundary.  This is a documented limitation of the
    worker contract: callers MUST include XMM regs to get bit-precise
    taint tracking through SIMD.  This test pins that contract — if the
    engine ever changes to widen automatically, this test will break and
    the comment should be updated."""
    state = {'RAX': 0, 'RBX': 0xFF, 'RCX': 0, 'RDX': 0}
    taint = {'RBX': 0xFF}
    mt = _eval_microtaint(_SIMD_PADDQ_ROUNDTRIP, state, taint, regs=_REGS_GP)
    # Without XMM in state_format, RAX is reported clean (under-tainted vs GT).
    # That's the limitation the worker fix addresses.
    assert mt['RAX'] == 0, (
        'Without XMM in state_format the engine reports T_RAX=0 — '
        'callers must include XMM<n>_LO/_HI in state_format to get '
        'sound SIMD taint propagation.'
    )
