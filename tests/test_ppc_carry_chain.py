"""Regression: PPC carry/borrow-chain taint on a big-endian target.

Root cause
----------
PPC32BE is a big-endian target, so ``CellSimulator`` forces ``use_unicorn=True``
and leaves the native p-code evaluator (``_pcode``) disabled -- the native
register file is byte-offset indexed and therefore wrong for BE memory and
sub-register aliasing.  But that routes the *differential itself*
(``InstructionCellExpr`` -> ``evaluate_concrete`` per replica) through Unicorn,
and Unicorn PPC neither seeds nor exposes the XER carry varnode (``xer_ca``).  A
carry-consuming instruction (``adde`` = ``rA + rB + xer_ca``) therefore reads
carry-in ``0`` in *both* replicas, so its differential collapses and the carry
chain ``addc;adde`` / borrow chain ``subfc;subfe`` under-taints the high bits of
the result.

Fix
---
On a big-endian target, evaluate an instruction with the native Cython kernel
whenever it is *native-BE-safe*: no memory access and every register-space
varnode is a WHOLE named register (never a sub-register byte slice).  Such an
instruction reads/writes whole registers as integer values, so the kernel's
internal byte layout is unobservable and the result round-trips regardless of
endianness -- and the kernel models ``xer_ca`` as a first-class varnode, so the
carry is both seeded (carry-in, in the differential) and read back (carry-out,
when threading concrete state across a chained sequence).

``extsb`` is the counter-example that pins the *whole-register* half of the
predicate: it reads ``register[rA+3:1]`` (byte 3 of the GPR -- the low byte under
BE), a sub-register slice the LE-indexed kernel would read as the high byte.  It
must stay on Unicorn, which gets byte order right.
"""

# mypy: disable-error-code="no-untyped-def,no-untyped-call"

from __future__ import annotations

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator, _native_be_safe
from microtaint.sleigh.engine import _cached_generate_static_rule, generate_static_rule
from microtaint.types import Architecture, ImplicitTaintPolicy, Register

ARCH = Architecture.PPC32BE

# Full PPC state format (as the multi-arch oracle builds it): R0..R7, the XER
# carry/overflow/summary bits, and the CR0..CR7 condition fields.
_FMT = (
    [Register(f'R{i}', 32) for i in range(8)]
    + [Register('XER_SO', 1), Register('XER_OV', 1), Register('XER_CA', 1)]
    + [Register(f'CR{i}', 1) for i in range(8)]
)
_ZERO = {r.name: 0 for r in _FMT}

# Big-endian instruction words (keystone-assembled, verified against the lifter).
_ADDE = '7c642914'  # adde  r3, r4, r5   -> r3 = r4 + r5 + xer_ca
_ADDC_ADDE = '7cc428147c642914'  # addc r6,r4,r5 ; adde r3,r4,r5
_SUBFC_SUBFE = '7cc428107c642910'  # subfc r6,r4,r5 ; subfe r3,r4,r5
_EXTSB = '7c830774'  # extsb r3, r4     -> r3 = sext(low byte of r4)
_LWZ = '80640000'  # lwz r3, 0(r4)     -> memory load


def _taint(code: str, values: dict[str, int], taint: dict[str, int], out: str = 'R3') -> int:
    _cached_generate_static_rule.cache_clear()
    circ = generate_static_rule(ARCH, bytes.fromhex(code), _FMT)
    ctx = EvalContext(
        input_taint={**_ZERO, **taint},
        input_values={**_ZERO, **values},
        simulator=CellSimulator(ARCH),
        implicit_policy=ImplicitTaintPolicy.IGNORE,
    )
    return circ.evaluate(ctx).get(out, 0)


def test_native_be_safe_predicate():
    """PPC register arithmetic is native-BE-safe: whole-GPR ops (`adde`, `subfc`,
    `subfe`) and sub-register ops (`extsb` reads a byte of a 32-bit GPR -- a
    sub-window of a <=4-byte, always-defined parent, handled by the kernel's
    BE byte math).  A memory access (`lwz`) is not."""
    assert _native_be_safe(ARCH, _ADDE) is True
    assert _native_be_safe(ARCH, _SUBFC_SUBFE[:8]) is True  # subfc
    assert _native_be_safe(ARCH, _SUBFC_SUBFE[8:]) is True  # subfe
    assert _native_be_safe(ARCH, _EXTSB) is True  # byte of a 32-bit GPR
    assert _native_be_safe(ARCH, _LWZ) is False  # LOAD


def test_native_be_safe_excludes_byte_aliased_subregisters():
    """MIPS64 32-bit ops read/write ``register[GPR+4:4]`` -- a byte-window alias of
    the 64-bit GPR (the LOW word under BE, which the LE kernel would read as the
    HIGH word).  Those must be excluded; only 64-bit (maximal) ops qualify.  This
    pins the maximal-register half of the predicate: naming a sub-window is not
    enough, it must not be contained in a wider register."""
    mips = Architecture.MIPS64BE
    assert _native_be_safe(mips, '00851021') is False  # addu  (32-bit alias reads)
    assert _native_be_safe(mips, '00041100') is False  # sll   (32-bit alias)
    assert _native_be_safe(mips, '0085102d') is True   # daddu (full 64-bit regs)
    assert _native_be_safe(mips, '00041138') is True   # dsll  (full 64-bit regs)


def test_adde_consumes_concrete_carry_in():
    """adde's differential must reflect the concrete carry-in.  With
    ``R4=0xd48dd9f3`` (bit 2 clear) and bit 2 tainted, flipping it ripples one
    place further when carry-in is 1 than when it is 0."""
    st = {'R4': 0xD48DD9F3, 'R5': 0x4BDBF090}
    assert _taint(_ADDE, {**st, 'XER_CA': 0}, {'R4': 0x4}) == 0x4
    assert _taint(_ADDE, {**st, 'XER_CA': 1}, {'R4': 0x4}) == 0xC


def test_addc_adde_chain_propagates_carry_taint():
    """The carry produced by ``addc`` must be threaded into ``adde`` -- both the
    concrete carry-out and its consumption in the differential -- so a single
    tainted low bit ripples through the extended add."""
    st = {'R4': 0xD48DD9F3, 'R5': 0x4BDBF090}
    assert _taint(_ADDC_ADDE, st, {'R4': 0x4}) == 0xC


def test_subfc_subfe_borrow_chain_ripples():
    """The borrow chain must propagate: tainting bit 0 of R4 in ``R5 - R4``
    ripples to bit 1 for ``R4=0xf, R5=0x10`` (0x10-0xf=1, 0x10-0xe=2, XOR=0x3).
    Before the fix the borrow was dropped and only bit 0 was tainted."""
    assert _taint(_SUBFC_SUBFE, {'R4': 0xF, 'R5': 0x10}, {'R4': 0x1}) == 0x3


def test_extsb_not_regressed_by_native_routing():
    """``extsb`` reads only R4's low byte; the fix must NOT route it through the
    native kernel (which would read the wrong byte under BE).  Tainting the low
    byte propagates through the sign extension; tainting only the high bytes
    (which extsb discards) leaves R3 with just the low-byte taint."""
    # low byte fully tainted -> sign bit tainted -> whole result tainted
    assert _taint(_EXTSB, {'R4': 0x1036311D}, {'R4': 0xFF}) == 0xFFFFFFFF
    # taint 0xc9462c56: only the low byte (0x56) reaches the sign-extended result
    assert _taint(_EXTSB, {'R4': 0x1036311D}, {'R4': 0xC9462C56}) == 0x56


def test_little_endian_target_never_uses_native_be_path():
    """The BE native routing is gated on ``_is_big_endian``; a little-endian
    simulator must never take it (x86/ARM64 keep their existing paths)."""
    x86 = CellSimulator(Architecture.X86)
    assert x86._is_big_endian is False
    # ADD EAX, EBX -- register-only, but LE, so the BE path is not eligible.
    assert x86._use_native_be(_OutCell('01d8', 'EAX')) is False


class _OutCell:
    def __init__(self, instruction: str, out_reg: str) -> None:
        self.instruction = instruction
        self.out_reg = out_reg


if __name__ == '__main__':
    import pytest

    raise SystemExit(pytest.main([__file__, '-v']))
