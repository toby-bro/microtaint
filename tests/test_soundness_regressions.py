"""Soundness regression tests for microtaint.

These tests lock in the soundness fixes for cases where earlier versions
of microtaint UNDER-tainted: i.e. reported a register output as clean (or
narrower than reality) when the per-bit ground-truth shows that taint
genuinely flows.  Under-tainting is the most severe failure mode of a
taint-tracking engine: a downstream consumer can then trust a bit that
should have been marked tainted, and a real exploitable flow can be missed.

The set of cases below was extracted from
``report_1777740360.json`` — the NDSS evaluation report — by intersecting
the 310 register-level disagreements with the per-bit Unicorn ground truth.
At the time of the report 9 cases produced a microtaint output mask that
was a strict subset of the true output mask.  After v0.6.3 (cmov
polarised-destination injection, bit-count cap, gated-passthrough union of
old-dest and source taint), all 9 cases are sound: microtaint either
reports the exact mask or over-approximates.

Each test fixes the assembly bytes, the concrete register state and the
input taint mask, and asserts that microtaint's output mask is a SUPERSET
of the per-bit ground-truth mask (``true & ~got == 0``).  The tests do
not require an exact match because soundness is what matters here:
over-approximation is acceptable, missing taint is not.

All tests share the same ground-truth oracle: per-bit Unicorn sensitivity.
For each tainted input bit we run the instruction (or sequence) twice with
that bit toggled and XOR the outputs; the union over all tainted bits is
the per-bit output sensitivity.  This is the strongest possible ground
truth short of a formal proof.
"""

# ruff: noqa: PLC0415
# mypy: disable-error-code="no-untyped-def,no-untyped-call,import-untyped,attr-defined,no-any-return,import-untyped"

from __future__ import annotations

import pytest
import unicorn
import unicorn.x86_const as ux

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import (
    _cached_generate_static_rule,
    generate_static_rule,
)
from microtaint.types import Architecture, ImplicitTaintPolicy, Register

MASK64 = 0xFFFFFFFFFFFFFFFF
GP_REGS = ('RAX', 'RBX', 'RCX', 'RDX')
_REG_TO_UC = {
    'RAX': ux.UC_X86_REG_RAX,
    'RBX': ux.UC_X86_REG_RBX,
    'RCX': ux.UC_X86_REG_RCX,
    'RDX': ux.UC_X86_REG_RDX,
}


# ── Test infrastructure ─────────────────────────────────────────────────────


@pytest.fixture(scope='module')
def regs() -> list[Register]:
    """64-bit GP registers plus the x86 status flags microtaint tracks."""
    return [Register(n, 64) for n in GP_REGS] + [Register(n, 1) for n in ('CF', 'OF', 'ZF', 'SF', 'PF')]


@pytest.fixture(scope='module')
def sim() -> CellSimulator:
    return CellSimulator(Architecture.AMD64)


def _ground_truth(
    code: bytes,
    state: dict[str, int],
    taint: dict[str, int],
    target: str,
) -> int:
    """Per-bit Unicorn sensitivity oracle.

    For each input bit marked tainted, toggle that bit in the base concrete
    state and XOR the resulting output value with the unflipped output;
    OR the differences across all tainted bits.  The resulting mask has
    bit i set iff there exists at least one tainted input bit whose toggle
    flips bit i of the target output register.  This is the per-bit
    sensitivity, the strongest possible ground truth.
    """
    base = {r: state.get(r, 0) & ~taint.get(r, 0) & MASK64 for r in GP_REGS}

    def run(values: dict[str, int]) -> int | None:
        try:
            uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
            uc.mem_map(0x1000, 0x10000)
            uc.mem_write(0x1000, code)
            uc.reg_write(ux.UC_X86_REG_RSP, 0x8000)
            for r, v in values.items():
                uc.reg_write(_REG_TO_UC[r], v & MASK64)
            uc.emu_start(0x1000, 0x1000 + len(code), timeout=500_000)
            return uc.reg_read(_REG_TO_UC[target])
        except unicorn.UcError:
            return None

    base_out = run(base)
    if base_out is None:
        pytest.fail('Unicorn ground-truth base run failed')

    union = 0
    for src in GP_REGS:
        tmask = taint.get(src, 0)
        if not tmask:
            continue
        for bit in range(64):
            if not (tmask >> bit) & 1:
                continue
            flipped = dict(base)
            flipped[src] = (base[src] ^ (1 << bit)) & MASK64
            out = run(flipped)
            if out is None:
                continue
            union |= base_out ^ out
    return union


def _microtaint_run(
    code: bytes,
    state: dict[str, int],
    taint: dict[str, int],
    regs: list[Register],
    sim: CellSimulator,
) -> dict[str, int]:
    """Compute microtaint's output taint for the given byte sequence."""
    zero = {r.name: 0 for r in regs}
    _cached_generate_static_rule.cache_clear()
    circuit = generate_static_rule(Architecture.AMD64, code, regs)
    ctx = EvalContext(
        input_taint={**zero, **taint},
        input_values={**zero, **state},
        simulator=sim,
        implicit_policy=ImplicitTaintPolicy.IGNORE,
    )
    return circuit.evaluate(ctx)


def _assert_sound(
    code: bytes,
    state: dict[str, int],
    taint: dict[str, int],
    target: str,
    regs: list[Register],
    sim: CellSimulator,
) -> None:
    """Assert that microtaint's output mask is a superset of the ground truth.

    ``true & ~got == 0`` is the formal soundness condition: every bit that
    genuinely depends on a tainted input must be marked tainted by
    microtaint.  Over-approximation (extra bits set) is permitted; missing
    bits is not.
    """
    true_mask = _ground_truth(code, state, taint, target)
    got = _microtaint_run(code, state, taint, regs, sim).get(target, 0)
    under = true_mask & ~got
    assert under == 0, (
        f"UNSOUND: microtaint missed {bin(under).count('1')} bit(s).\n"
        f"  true output taint: {hex(true_mask)}\n"
        f"  microtaint output: {hex(got)}\n"
        f"  missing bits:      {hex(under)}"
    )


# ── BLSI: isolate-lowest-set-bit ────────────────────────────────────────────
#
# Instruction:    blsi rax, rbx
# Encoding:       c4 e2 f8 f3 db
# Semantics:      RAX = RBX & (-RBX) — isolates the lowest set bit of RBX.
#                 If RBX = 0xC,    then -RBX = 0xFFFFFFFFFFFFFFF4,
#                 RAX = 0xC & 0xFF...F4 = 0x4 — the lowest set bit.
#
# Sleigh lift (RAX slice):
#   INT_2COMP   tmp ← RBX                  ; two's complement: -RBX
#   INT_AND     RAX ← tmp & RBX            ; isolate lowest set bit
#
# Why this is a hard soundness case:
#   The slice contains INT_2COMP and INT_AND, which microtaint classifies as
#   TRANSPORTABLE (priority 6 in the category table; INT_2COMP triggers).
#   The two-replica differential C1 ⊕ C2 alone gives:
#     C1 = (V|T) & -(V|T) = lowest_set_bit(V|T)
#     C2 = (V&~T) & -(V&~T) = lowest_set_bit(V&~T)
#   When T = MASK and V is anything, V|T = MASK so C1 = 1 (bit 0); V&~T = 0
#   so C2 = 0; the differential alone is just bit 0.  But the per-bit
#   ground-truth shows that ANY bit position can become the lowest set bit
#   depending on which tainted bit is the lowest 1 — so the true output is
#   MASK64.  Older microtaint reported just 0x1, missing 63 bits.
#
# v0.6.3 fix: TRANSPORTABLE includes the union term ``OR(T_deps)`` as a
# soundness floor.  For blsi the dep is T_RBX, so the floor adds the full
# input taint and the result becomes a superset of MASK64 (= MASK64).


_BLSI_RAX_RBX = bytes.fromhex('c4e2f8f3db')


def test_blsi_all_tainted_full_avalanche(regs, sim):
    """Report case 574 — T = MASK64 in all four registers.

    True output: MASK64 (the lowest set bit of an unknown value can be
    anywhere in 0..63).  Old microtaint reported 0x1.
    """
    state = {
        'RAX': 0x1C66021E1A79AEC1,
        'RBX': 0x988C89DBED5CEA4E,
        'RCX': 0x244F52F41B696031,
        'RDX': 0xE504D1C64312153D,
    }
    taint = dict.fromkeys(GP_REGS, MASK64)
    _assert_sound(_BLSI_RAX_RBX, state, taint, 'RAX', regs, sim)


def test_blsi_only_rbx_fully_tainted(regs, sim):
    """Report case 576 — only RBX is tainted, fully.

    Same diagnosis as the previous test but with a clean RAX in input;
    the soundness floor still has to fire because T_RBX = MASK64.
    """
    state = {
        'RAX': 0x4C7B34E22E07CB63,
        'RBX': 0x80D2370DAD4C20F,
        'RCX': 0x89344B060790EEB4,
        'RDX': 0x78D27E5378A8BA6,
    }
    taint = {'RAX': 0, 'RBX': MASK64, 'RCX': 0, 'RDX': 0}
    _assert_sound(_BLSI_RAX_RBX, state, taint, 'RAX', regs, sim)


def test_blsi_partial_rbx_taint(regs, sim):
    """Report case 577 — RBX partially tainted; small but specific true mask.

    True output: 0xb (bits 0, 1, 3) — only positions where the lowest set
    bit of (V_RBX with some tainted bits flipped) can land.  Old microtaint
    reported 0x9 (missed bit 1).
    """
    state = {
        'RAX': 0x384243B42378FF44,
        'RBX': 0x4480D02D2067902B,
        'RCX': 0xA6E486BF8D2EEA22,
        'RDX': 0x40061ADD316C00BC,
    }
    taint = {
        'RAX': 0,
        'RBX': 0x47F50A35B7A745E3,
        'RCX': 0x9569F8CB1A574CEF,
        'RDX': 0x4FC40B0A966B91E8,
    }
    _assert_sound(_BLSI_RAX_RBX, state, taint, 'RAX', regs, sim)


# ── BLSI followed by ANDN: dependent two-instruction sequence ────────────────
#
# Sequence:       blsi rax, rbx ; andn rcx, rax, rdx
# Bytes:          c4e2f8f3db c4e2f8f2ca
# Semantics:      RAX = blsi(RBX); RCX = (~RAX) & RDX
#
# Sleigh lift (instruction 2, RCX slice):
#   INT_NEGATE    tmp ← ~RAX                ; bitwise NOT
#   INT_AND       RCX ← tmp & RDX
#
# Why this is hard:
#   The chain composes a BLSI (TRANSPORTABLE with soundness floor) with an
#   ANDN (MONOTONIC).  If the first instruction's output taint is
#   under-approximated, the chain inherits the deficit because the second
#   instruction reads RAX as a source.  The fix to BLSI's soundness in turn
#   restores soundness of the chain.

_SEQ_BLSI_ANDN = bytes.fromhex('c4e2f8f3dbc4e2f8f2ca')


def test_blsi_andn_chain_case_204(regs, sim):
    """Report case 204 — chained blsi+andn, target RCX.

    Old microtaint missed 15 bits because BLSI under-tainted RAX, and the
    subsequent ANDN's MONOTONIC formula had no soundness floor to recover
    those bits.
    """
    state = {
        'RAX': 856619935209448460,
        'RBX': 11963170063307501203,
        'RCX': 3206963090649646086,
        'RDX': 6982434297052873721,
    }
    taint = {
        'RAX': 15411684199235726844,
        'RBX': 0,
        'RCX': MASK64,
        'RDX': 9525378169378434294,
    }
    _assert_sound(_SEQ_BLSI_ANDN, state, taint, 'RCX', regs, sim)


def test_blsi_andn_chain_case_237(regs, sim):
    """Report case 237 — same sequence, different state, RCX target.

    Old microtaint missed 14 bits and over-tainted 1.  New version is
    sound (under = 0); over-taint is acceptable.
    """
    state = {
        'RAX': 2879667872895615344,
        'RBX': 4609046372682311681,
        'RCX': 14961678933615522350,
        'RDX': 10523630689843615760,
    }
    taint = {
        'RAX': 13544816196555765797,
        'RBX': 0,
        'RCX': MASK64,
        'RDX': 15231573313693134483,
    }
    _assert_sound(_SEQ_BLSI_ANDN, state, taint, 'RCX', regs, sim)


# ── MOVZX/SUB/MOVSX chain: sub-register width interaction ────────────────────
#
# Sequence:       movzx rcx, al
#                 movzx rdx, bl
#                 sub   rcx, rdx
#                 movsx rax, cl
# Bytes:          480fb6c8 480fb6d3 4829d1 480fbec1
# Semantics:      Read low bytes of RAX/RBX zero-extended into RCX/RDX,
#                 subtract, then sign-extend the low byte of the result back
#                 into RAX.  This is a strcmp-like idiom: the difference
#                 between two bytes propagated as a signed 64-bit value.
#
# Sleigh lift highlights:
#   movzx rcx, al  → INT_ZEXT(RCX:8 ← AL:1)        ; one COPY-like primitive
#   sub   rcx, rdx → INT_SUB(RCX, RDX) (+flags)    ; TRANSPORTABLE
#   movsx rax, cl  → INT_SEXT(RAX:8 ← CL:1)        ; sign-extend low byte
#
# Why this is hard:
#   The intermediate value flows through three width transitions: 8-bit
#   sub-register → 9-bit subtraction result (carry into bit 8) → sign-extend
#   from byte 0 back to 64 bits.  The differential alone correctly tracks
#   the 9 bits produced by the sub, but loses the SIGN-bit propagation
#   through movsx: bit 7 of CL is tainted, so the sign extension may set OR
#   clear all 56 upper bits — every bit of RAX can be tainted.
#
#   Since the test target is RCX, not RAX, the sign-extension issue isn't
#   directly visible — but RCX after sub *should* have its upper bits clean
#   (movzx zero-extends), so the true taint of RCX is constrained.  Old
#   microtaint reported 0x1FF; ground truth is more (because flipping bits
#   of RCX or RDX *as input* produces effects through the chain that the
#   per-instruction analysis misses).

_SEQ_MOVZX_SUB_MOVSX = bytes.fromhex('480fb6c8480fb6d34829d1480fbec1')


def test_movzx_sub_movsx_chain_case_626(regs, sim):
    """Report case 626 — strcmp-like byte-difference chain, target RCX.

    Old microtaint missed 55 bits; the whole upper part of the difference
    was reported clean.  The new version's gated passthrough and chain
    soundness floor cover all the missing bits.
    """
    state = {
        'RAX': 9692544848236511546,
        'RBX': 12871560762356780111,
        'RCX': 7448313025359211831,
        'RDX': 2936346809869392945,
    }
    taint = dict.fromkeys(GP_REGS, MASK64)
    _assert_sound(_SEQ_MOVZX_SUB_MOVSX, state, taint, 'RCX', regs, sim)


# ── SHL x 3 then OR x 3: base64-decode bit-packing ──────────────────────────
#
# Sequence:       shl rax, 18
#                 shl rbx, 12
#                 shl rcx, 6
#                 or  rax, rbx
#                 or  rax, rcx
#                 or  rax, rdx
# Bytes:          48c1e012 48c1e30c 48c1e106 4809d8 4809c8 4809d0
# Semantics:      Pack four 6-bit groups (in RAX..RDX) into a 24-bit word
#                 in RAX.  This is the canonical base64-decode inner step.
#
# Sleigh lift highlights (per shl):
#   INT_AND   masked_count ← shift_count & 0x3F   ; mask shift to 6 bits
#   INT_LEFT  RAX ← RAX << masked_count           ; the actual shift
#   ...flag updates omitted
#
# Why this is hard:
#   Each SHL is MAPPED (constant-shift permutation) for the destination,
#   but the FLAGS set by shl are TRANSLATABLE / COND_TRANSPORTABLE and feed
#   into the next instruction's CBRANCH-free state.  More importantly: with
#   all four input registers fully tainted, the OR's ``T_AND`` formula
#   ``(¬V_a & T_b) | (¬V_b & T_a) | (T_a & T_b)`` interacts with the
#   shifted positions and can leave gaps where a clean bit (constant 0 from
#   the shift) appears to mask a tainted bit.  Old microtaint reported
#   0xfffffffffffc4f8a — 10 bits clean that should be tainted.
#
# v0.6.3 fix: the OR formula is correct on its own, but the chain
# composition needed to ensure that intermediate write-back of taint to RAX
# carries the full 64-bit mask through subsequent OR steps.  After the cmov
# polarised-destination injection (which generalises to any chain that
# writes to a register read in the same instruction), all 64 bits are
# preserved.

_SEQ_BASE64_PACK = bytes.fromhex(
    '48c1e01248c1e30c48c1e1064809d84809c84809d0',
)


def test_base64_pack_chain_case_629(regs, sim):
    """Report case 629 — base64-decode pack, target RAX.

    Old microtaint missed 10 bits in the packed result; new engine is sound.
    """
    state = {
        'RAX': 8291989174624432245,
        'RBX': 10265026935744837952,
        'RCX': 4657709630386508352,
        'RDX': 12522554165043460411,
    }
    taint = dict.fromkeys(GP_REGS, MASK64)
    _assert_sound(_SEQ_BASE64_PACK, state, taint, 'RAX', regs, sim)


# ── BSWAP + AND with constant mask ──────────────────────────────────────────
#
# Sequence:       bswap rax
#                 mov   rbx, 0xff00ff00ff00ff00
#                 and   rax, rbx
# Bytes:          480fc8 48bb00ff00ff00ff00ff 4821d8
# Semantics:      Reverse bytes of RAX, then keep alternate bytes via the
#                 mask.  Common pattern in network-byte-order decoders.
#
# Sleigh lift highlights:
#   bswap rax     → 8 INT_AND/INT_LEFT/INT_RIGHT/INT_OR ops that route
#                   bytes 0→7, 1→6, ..., 7→0.  Each output bit is exactly
#                   one input bit (a bijection) — pure routing.
#   mov   rbx, K  → COPY(RBX ← K)                  ; constant load
#   and   rax, rbx → INT_AND(RAX, RBX)             ; per-bit mask
#
# Why this is hard:
#   The bswap's slice is recognized as a permutation (MAPPED) — exact.
#   The AND uses the per-bit formula
#     T_out = (V_a & T_b) | (V_b & T_a) | (T_a & T_b)
#   Because RBX = 0xff00ff00ff00ff00 is constant after the MOV (its taint
#   is irrelevant in this idiom), the third term collapses and we get
#     T_out = (RBX_const & T_RAX) | (RAX_byte_swapped & T_RBX_after_mov)
#   The chain composition has to thread the byte-swapped taint into the
#   AND's first source correctly.  Old microtaint missed bits in the
#   shuffle when the byte-swap re-arranged tainted positions.

_SEQ_BSWAP_MASK = bytes.fromhex('480fc848bb00ff00ff00ff00ff4821d8')


def test_bswap_mask_chain_case_225(regs, sim):
    """Report case 225 — bswap + masked AND, target RAX.

    Old microtaint missed 6 bits and over-tainted 4.  New engine is sound;
    over-taint is acceptable and unavoidable in the AND formula's third term.
    """
    state = {
        'RAX': 9870832247003503468,
        'RBX': 17009528696640483621,
        'RCX': 13936011587764342463,
        'RDX': 14719852046302904884,
    }
    taint = {
        'RAX': 133148336104993439,
        'RBX': 7595186107965648905,
        'RCX': 7281890570029904882,
        'RDX': MASK64,
    }
    _assert_sound(_SEQ_BSWAP_MASK, state, taint, 'RAX', regs, sim)


def test_bswap_mask_chain_case_244(regs, sim):
    """Report case 244 — same sequence, fully-tainted RBX/RDX inputs."""
    state = {
        'RAX': 9805563300297832242,
        'RBX': 13142736926310483358,
        'RCX': 5624919937413268468,
        'RDX': 338907731038666509,
    }
    taint = {
        'RAX': 11533475702397079476,
        'RBX': MASK64,
        'RCX': 9246457408669399904,
        'RDX': MASK64,
    }
    _assert_sound(_SEQ_BSWAP_MASK, state, taint, 'RAX', regs, sim)


# ── Random-fuzzing soundness: defence against future regressions ────────────
#
# A separate test runs a small randomised sweep over the most error-prone
# instructions (BMI, lea, multi-instruction chains) and asserts soundness
# on every case.  This catches new under-taint bugs that might be
# introduced by future engine changes.


_FUZZ_CASES = [
    # asm_lines, n_random_seeds
    (['blsi rax, rbx'], 8),
    (['blsr rax, rbx'], 8),
    (['andn rax, rbx, rcx'], 8),
    (['lea rax, [rbx + rcx*2]'], 8),
    (['lea rcx, [rax*2 + rdx]'], 8),
    (['blsi rax, rbx', 'andn rcx, rax, rdx'], 6),
    (['movzx rcx, al', 'sub rcx, rdx'], 6),
    (['bswap rax', 'and rax, rbx'], 6),
    (['add rax, rbx', 'add rcx, rax'], 6),
    (['imul rax, rbx', 'add rcx, rax'], 6),
]


def _assemble(asm_lines: list[str]) -> bytes:
    from keystone import KS_ARCH_X86, KS_MODE_64, Ks  # local import

    ks = Ks(KS_ARCH_X86, KS_MODE_64)
    out = b''
    for line in asm_lines:
        enc, _ = ks.asm(line)
        out += bytes(enc)
    return out


@pytest.mark.parametrize(('asm_lines', 'n_seeds'), _FUZZ_CASES)
def test_random_soundness(regs, sim, asm_lines, n_seeds):
    """Random soundness fuzz over BMI / lea / chain patterns.

    For each pattern we draw a small number of random concrete states and
    random partial taint masks, and assert microtaint never under-taints.
    """
    import random

    rng = random.Random(42)
    code = _assemble(asm_lines)
    for seed in range(n_seeds):
        state = {r: rng.randint(0, MASK64) for r in GP_REGS}
        # Avoid divide-by-zero hazards (none in our pattern set, but cheap)
        for line in asm_lines:
            for r in GP_REGS:
                if f'div {r.lower()}' in line and state[r] == 0:
                    state[r] = 1
        # Random taint shape: 30% all-tainted, 70% partial
        if rng.random() < 0.3:
            taint = dict.fromkeys(GP_REGS, MASK64)
        else:
            taint = {r: rng.randint(0, MASK64) for r in GP_REGS}
        for target in GP_REGS:
            true_mask = _ground_truth(code, state, taint, target)
            got = _microtaint_run(code, state, taint, regs, sim).get(target, 0)
            under = true_mask & ~got
            assert under == 0, (
                f"UNSOUND on {asm_lines!r} seed={seed} target={target}\n"
                f"  state = {state}\n"
                f"  taint = {taint}\n"
                f"  true  = {hex(true_mask)}\n"
                f"  got   = {hex(got)}\n"
                f"  under = {hex(under)} ({bin(under).count('1')} bits)"
            )
