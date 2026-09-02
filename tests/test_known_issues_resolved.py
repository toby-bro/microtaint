"""Regression anchors for the three soundness gaps once tracked in KNOWN_ISSUES.md.

All three are now closed (verified continuously by the cross-ISA oracle and fuzzer
in benchmark/generalization/, which report 0 under-taints on all five ISAs). This
file pins the HEADLINE case of each as a fast, deterministic pytest so the specific
fixes cannot silently regress after the tracking doc was removed:

  * Issue  1 -- OF after a multi-bit shift/rotate (SLEIGH preserves, silicon
    recomputes): OF is now conservatively tainted for count != 1, closing the
    latent hardware under-taint (`shl rax,8 ; seto dl`).
  * Issue  0 -- the native LE-only p-code kernel used to return 0 for big-endian
    targets, collapsing every differential to zero: BE arithmetic now propagates
    taint (MIPS64BE / PPC32BE), no zero-collapse.
  * Issue -1 -- a condition flag consumed into a WIDE GPR (ARM64 cset, PPC mfcr)
    used to miss the flag bit because the 1-bit-flag floor was gated to <=8-bit
    outputs: the flag now taints its destination register.
"""

from __future__ import annotations

from keystone import (
    KS_ARCH_ARM64,
    KS_ARCH_MIPS,
    KS_ARCH_PPC,
    KS_ARCH_SPARC,
    KS_ARCH_X86,
    KS_MODE_64,
    KS_MODE_BIG_ENDIAN,
    KS_MODE_LITTLE_ENDIAN,
    KS_MODE_MIPS64,
    KS_MODE_PPC32,
    KS_MODE_SPARC32,
    Ks,
)

import microtaint.sleigh.engine as engine
from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.types import Architecture, ImplicitTaintPolicy, Register

_FULL = 0xFFFFFFFFFFFFFFFF


def _asm(ks: Ks, text: str) -> bytes:
    enc, _ = ks.asm(text, 0x1000)
    return bytes(enc)


def _eval(arch: Architecture, code: bytes, fmt: list[Register],
          in_taint: dict[str, int], in_vals: dict[str, int]) -> dict[str, int]:
    engine._cached_generate_static_rule.cache_clear()
    circ = engine.generate_static_rule(arch, code, fmt)
    zero = {r.name: 0 for r in fmt}
    ctx = EvalContext(
        input_taint={**zero, **in_taint},
        input_values={**zero, **in_vals},
        simulator=CellSimulator(arch),
        implicit_policy=ImplicitTaintPolicy.IGNORE,
    )
    return circ.evaluate(ctx)


# ---------------------------------------------------------------------------
# Issue 1: OF after a multi-bit shift / rotate is conservatively tainted.
# SLEIGH models "undefined OF" as "preserve old OF" (-> taint 0); silicon
# recomputes OF from the tainted result, so `shl rax,8 ; seto dl` would under-
# taint DL on real hardware.  OF must now be tainted when the operand is.
# ---------------------------------------------------------------------------

_KS_X64 = Ks(KS_ARCH_X86, KS_MODE_64)
_X64_FMT = [Register('RAX', 64), Register('RDX', 64)] + [
    Register(f, 1) for f in ('OF', 'CF', 'SF', 'ZF', 'PF')
]


def test_issue1_amd64_multibit_shl_taints_of() -> None:
    out = _eval(Architecture.AMD64, _asm(_KS_X64, 'shl rax, 8'), _X64_FMT,
                {'RAX': _FULL}, {'RAX': 0x00FF00FF00FF00FF})
    assert out.get('OF', 0), 'shl rax,8: OF must be conservatively tainted (silicon recomputes it)'


def test_issue1_amd64_multibit_rol_taints_of() -> None:
    out = _eval(Architecture.AMD64, _asm(_KS_X64, 'rol rax, 8'), _X64_FMT,
                {'RAX': _FULL}, {'RAX': 0x00FF00FF00FF00FF})
    assert out.get('OF', 0), 'rol rax,8: OF must be conservatively tainted (silicon recomputes it)'


def test_issue1_amd64_1bit_shl_taints_of() -> None:
    # Sanity: the DEFINED 1-bit case stays tainted too (OF = MSB ^ CF).
    out = _eval(Architecture.AMD64, _asm(_KS_X64, 'shl rax, 1'), _X64_FMT,
                {'RAX': _FULL}, {'RAX': 0x00FF00FF00FF00FF})
    assert out.get('OF', 0)


# ---------------------------------------------------------------------------
# Issue 0: big-endian arithmetic no longer zero-collapses.  The native LE-only
# kernel returned 0 for BE targets, making every differential-based rule
# under-taint; BE now routes correctly (native-BE-safe or Unicorn).
# ---------------------------------------------------------------------------

_KS_MIPS = Ks(KS_ARCH_MIPS, KS_MODE_MIPS64 | KS_MODE_BIG_ENDIAN)
_MIPS_FMT = [Register(n, 64) for n in ('ZERO', 'AT', 'V0', 'V1', 'A0', 'A1')]


def test_issue0_mips64be_addu_propagates_taint() -> None:
    # addu $2,$4,$5  (V0 = A0 + A1); tainting A0 must taint V0 (was 0-collapse).
    out = _eval(Architecture.MIPS64BE, _asm(_KS_MIPS, 'addu $2, $4, $5'), _MIPS_FMT,
                {'A0': 0x0F0F0F0F}, {'A0': 0x0F0F0F0F, 'A1': 0x0F0F0F0F})
    assert out.get('V0', 0), 'MIPS64BE addu must propagate taint (native BE zero-collapse fixed)'


_KS_PPC = Ks(KS_ARCH_PPC, KS_MODE_PPC32 | KS_MODE_BIG_ENDIAN)
_PPC_FMT = [Register(f'R{i}', 32) for i in range(8)] + [
    Register('XER_SO', 1), Register('XER_OV', 1), Register('XER_CA', 1),
] + [Register(f'CR{i}', 4) for i in range(8)]


def test_issue0_ppc32be_add_propagates_taint() -> None:
    out = _eval(Architecture.PPC32BE, _asm(_KS_PPC, 'add 3, 4, 5'), _PPC_FMT,
                {'R4': 0x0F0F0F0F}, {'R4': 0x0F0F0F0F, 'R5': 0x0F0F0F0F})
    assert out.get('R3', 0), 'PPC32BE add must propagate taint (native BE zero-collapse fixed)'


# ---------------------------------------------------------------------------
# Issue -1: a condition flag consumed into a WIDE (>8-bit) GPR must taint it.
# The 1-bit-flag soundness floor was gated to <=8-bit (x86 setcc) outputs;
# ARM64 cset / PPC mfcr write a flag into a 64/32-bit GPR.
# ---------------------------------------------------------------------------

_KS_ARM = Ks(KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN)
_ARM_FMT = [Register('X0', 64)] + [Register(f, 1) for f in ('N', 'Z', 'C', 'V')]


def test_issuem1_arm64_cset_taints_wide_gpr_from_flag() -> None:
    # cset x0, lt  (x0 = ZEXT(N != V)); tainting N must taint X0 bit 0.
    out = _eval(Architecture.ARM64, _asm(_KS_ARM, 'cset x0, lt'), _ARM_FMT,
                {'N': 1}, {'N': 1, 'V': 0, 'Z': 0, 'C': 1})
    assert out.get('X0', 0) & 1, 'ARM64 cset must taint the wide GPR from its condition flag'


def test_issuem1_ppc32be_mfcr_taints_gpr_from_cr() -> None:
    # mfcr 3  (R3 <- CR); tainting CR0 must taint R3's CR0 field.
    out = _eval(Architecture.PPC32BE, _asm(_KS_PPC, 'mfcr 3'), _PPC_FMT,
                {'CR0': 0xF}, {'CR0': 0xF})
    assert out.get('R3', 0), 'PPC32BE mfcr must taint the GPR from the condition register'


# ---------------------------------------------------------------------------
# Issue -1 / 0 (multi-instruction carry chains): the carry produced by one
# instruction must thread its TAINT into the next (add;adc, subs;sbc,
# subcc;subx, subfc;subfe).  The chained-circuit state format is augmented with
# the sequence's intra-instruction intermediate registers discovered from the
# p-code -- ISA-agnostically -- so ANY ISA's condition/carry register threads,
# not just a hardcoded x86 flag list.  SPARC's icc carry (i_cf) is the case that
# was silently dropped (34% under-taint on dense borrow chains); ARM64/PPC were
# already sound because their flags are enumerated in the state format.
# ---------------------------------------------------------------------------

_KS_SPARC = Ks(KS_ARCH_SPARC, KS_MODE_SPARC32 | KS_MODE_BIG_ENDIAN)
_SPARC_FMT = [Register(n, 32) for n in ('G1', 'G2', 'G3', 'G4', 'G5')]


def test_issuem1_sparc_borrow_chain_threads_carry() -> None:
    """`subcc %g4,%g2,%g1 ; subx %g1,%g2,%g3`: the borrow-out of subcc (icc carry)
    feeds subx, so g3's low bits depend on it.  Exact-2^k-confirmed under-taint
    (missed bits 0xf) before the general carry-threading fix; g3 bits 0-3 must now
    be tainted."""
    code = _asm(_KS_SPARC, 'subcc %g4, %g2, %g1; subx %g1, %g2, %g3')
    # The captured failing case: dense taint on the (post-canonicalisation) inputs.
    out = _eval(
        Architecture.SPARC32BE, code, _SPARC_FMT,
        {'G1': 0x4E003000, 'G2': 0x4E003000},
        {'G1': 0x6AA79987, 'G2': 0xBB91433A, 'G4': 0xD1F6F86C, 'G3': 0x029A7245, 'G5': 0xD340BBCD},
    )
    # Pre-fix: g3 low nibble (carry-dependent) was clean -> 0x...0. Now threaded.
    assert (out.get('G3', 0) & 0xF) == 0xF, (
        'SPARC subcc;subx must thread the icc carry taint into g3 low bits'
    )


# ---------------------------------------------------------------------------
# The ISA-agnostic carry-threading augmentation must thread only SCALAR (<=8-byte)
# intermediates.  A WIDE vector register written-then-read across a two-instruction
# sequence must NOT be added to the chained state format: a >64-bit Register
# corrupts the 64-bit-mask path and segfaults the native cell (paddq roundtrip).
# ---------------------------------------------------------------------------

_X64_VEC_FMT = [Register('RAX', 64)] + [
    Register(f'XMM{n}_{half}', 64) for n in range(3) for half in ('LO', 'HI')
]


def test_chained_augmentation_excludes_wide_vector_regs() -> None:
    """`paddq xmm0,xmm1 ; paddq xmm0,xmm2`: xmm0 is written then read across the
    sequence (a 16-byte varnode).  Building + evaluating the chained rule must not
    crash and must still propagate the vector taint."""
    code = bytes.fromhex('660fd4c1660fd4c2')  # paddq xmm0,xmm1 ; paddq xmm0,xmm2
    # Must not raise (regression: wide XMM in the chain state format segfaulted).
    out = _eval(Architecture.AMD64, code, _X64_VEC_FMT,
                {'XMM1_LO': _FULL}, {})
    assert out.get('XMM0_LO', 0), 'paddq chain must still propagate the vector taint'
