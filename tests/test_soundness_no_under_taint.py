"""Soundness pytest suite for microtaint.

Each test in this file documents a real case where microtaint *previously*
produced an under-taint result (T_microtaint ⊊ T_true) and asserts that the
fixed engine now returns a sound mask (T_microtaint ⊇ T_true).

Soundness means: every bit that genuinely depends on a tainted input must be
marked tainted in the output.  Over-taint is acceptable; under-taint is not.

The tests are organised around two distinct root causes:

1. **Single-instruction blsi/blsr classifier bug.**
   blsi and blsr lift to slices containing both INT_2COMP/INT_SUB (carry op)
   AND INT_AND (monotone op).  The classifier picked MONOTONIC because of
   priority order, but MONOTONIC's formula (just the differential) misses
   the carry chain introduced by INT_2COMP.  Fix: when a slice contains both
   a carry-introducing op and a monotone op, force TRANSPORTABLE category
   (which adds the input-taint union as a soundness floor).

2. **ChainedCircuit value-staleness bug.**
   Multi-instruction sequences were evaluated with concrete register values
   held constant at the chain's entry state.  Per-opcode taint formulas like
   AND `(V_a & T_b) | (V_b & T_a) | (T_a & T_b)` read concrete values of
   their source registers.  When an earlier instruction in the chain *changes*
   a register's concrete value (e.g. `mov rbx, IMM`), the AND in a later step
   uses the stale entry value and produces wrong taint.  Fix: thread concrete
   state across chain steps by running each sub-circuit's instruction
   concretely between iterations.

For each test we record:
  - the assembly,
  - the Sleigh P-code lift,
  - a worked example showing how the per-bit ground truth is computed,
  - the value microtaint produced before the fix,
  - the assertion that the fixed engine produces a sound mask.

The ground truth is computed by per-bit Unicorn sensitivity (flip each
tainted input bit, XOR outputs, OR the results).  This is the operational
definition of "the bits whose values genuinely depend on tainted inputs".
"""

# ruff: noqa: PLC0415
# mypy: disable-error-code="no-untyped-def,no-untyped-call,attr-defined,import-untyped"

from __future__ import annotations

import pytest
import unicorn
import unicorn.x86_const as ux
from keystone import KS_ARCH_X86, KS_MODE_64, Ks

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import _cached_generate_static_rule, generate_static_rule
from microtaint.types import Architecture, ImplicitTaintPolicy, Register

MASK64 = 0xFFFFFFFFFFFFFFFF
REGS = ('RAX', 'RBX', 'RCX', 'RDX')
_FLAGS = ('CF', 'OF', 'ZF', 'SF', 'PF')

_KS = Ks(KS_ARCH_X86, KS_MODE_64)
_REG_OBJS = [Register(n, 64) for n in REGS] + [Register(n, 1) for n in _FLAGS]
_REG_MAP = {
    'RAX': ux.UC_X86_REG_RAX,
    'RBX': ux.UC_X86_REG_RBX,
    'RCX': ux.UC_X86_REG_RCX,
    'RDX': ux.UC_X86_REG_RDX,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _assemble(asm_lines: list[str]) -> bytes:
    """Assemble a list of assembly mnemonics to a single byte string."""
    out: list[int] = []
    for line in asm_lines:
        enc, _ = _KS.asm(line)
        out.extend(enc)
    return bytes(out)


def _ground_truth(code: bytes, state: dict[str, int], taint: dict[str, int]) -> dict[str, int]:
    """Per-bit Unicorn sensitivity.

    For every tainted input bit, flip it, run the instruction, and XOR with
    the base output.  The OR of all per-bit XORs is the per-bit-flip
    ground-truth taint mask of every output register.
    """
    base_vals = {r: state.get(r, 0) & ~taint.get(r, 0) & MASK64 for r in REGS}

    def _run(vals: dict[str, int]) -> dict[str, int]:
        uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
        uc.mem_map(0x1000, 0x2000)
        uc.mem_write(0x1000, code)
        for r, v in vals.items():
            uc.reg_write(_REG_MAP[r], v & MASK64)
        uc.emu_start(0x1000, 0x1000 + len(code))
        return {r: uc.reg_read(_REG_MAP[r]) for r in REGS}

    base_out = _run(base_vals)
    result = dict.fromkeys(REGS, 0)
    for src in REGS:
        tm = taint.get(src, 0)
        if not tm:
            continue
        for bit in range(64):
            if not (tm >> bit) & 1:
                continue
            flipped = dict(base_vals)
            flipped[src] = (base_vals[src] | (1 << bit)) & MASK64
            out = _run(flipped)
            for r in REGS:
                result[r] |= base_out[r] ^ out[r]
    return result


def _microtaint_run(asm_lines: list[str], state: dict[str, int], taint: dict[str, int]) -> dict[str, int]:
    """Run microtaint on the given sequence and return the output taint dict."""
    code = _assemble(asm_lines)
    sim = CellSimulator(Architecture.AMD64)
    zeros = {r.name: 0 for r in _REG_OBJS}
    full_state = {**zeros, **state}
    full_taint = {**zeros, **taint}

    _cached_generate_static_rule.cache_clear()
    circuit = generate_static_rule(Architecture.AMD64, code, _REG_OBJS)
    ctx = EvalContext(
        input_taint=full_taint,
        input_values=full_state,
        simulator=sim,
        implicit_policy=ImplicitTaintPolicy.IGNORE,
    )
    return circuit.evaluate(ctx)


def _assert_sound(asm_lines: list[str], state: dict[str, int], taint: dict[str, int]) -> None:
    """Run microtaint, compute Unicorn ground truth, assert no under-taint.

    Sound iff for every register and every bit, the microtaint output mask
    contains at least the bits in the ground-truth mask.
    """
    code = _assemble(asm_lines)
    truth = _ground_truth(code, state, taint)
    got = _microtaint_run(asm_lines, state, taint)

    failures: list[str] = []
    for reg in REGS:
        true_mask = truth.get(reg, 0)
        got_mask = got.get(reg, 0)
        under = true_mask & ~got_mask
        if under:
            n_under = bin(under).count('1')
            failures.append(
                f'{reg}: under-taint by {n_under} bits — '
                f'true={hex(true_mask)} got={hex(got_mask)} missing_bits={hex(under)}',
            )
    if failures:
        msg = '\n  '.join(failures)
        pytest.fail(f'Soundness failure on {asm_lines!r}:\n  {msg}')


# ─────────────────────────────────────────────────────────────────────────────
# Group 1: single-instruction blsi
# ─────────────────────────────────────────────────────────────────────────────
#
# `blsi rax, rbx` is a BMI1 instruction: RAX = RBX & (-RBX).
# It isolates the lowest set bit of RBX.
#
# Sleigh lifts blsi as TWO P-code ops on the dest:
#
#     INT_2COMP  unique:tmp_8       ← register:RBX_8        # tmp = -RBX
#     INT_AND    register:RAX_8     ← unique:tmp, RBX_8     # RAX = tmp & RBX
#
# (Plus three flag setters: ZF, SF, CF.)
#
# The slice from RAX is therefore [INT_2COMP, INT_AND].
#
# ──── The bug ────
# The classifier had:
#   priority MONOTONIC (5) > priority TRANSPORTABLE (6)
# When a slice contains both INT_AND (in MONOTONIC_OPCODES) and INT_2COMP
# (in TRANSPORTABLE_OPCODES), MONOTONIC won.  The MONOTONIC formula for a
# 64-bit destination is just the two-replica differential:
#
#     T_RAX = ((V_RBX | T_RBX) & -(V_RBX | T_RBX))
#           XOR ((V_RBX & ~T_RBX) & -(V_RBX & ~T_RBX))
#
# When T_RBX = MASK64 the high replica is `-MASK & MASK = 1 & MASK = 1` and
# the low replica is `-0 & 0 = 0`, so the differential is just `1` — one bit.
# But the TRUE per-bit-flip taint of blsi is MASK64 (any bit flip of RBX
# typically changes which bit is the lowest set bit, affecting many positions).
#
# ──── The fix ────
# `determine_category` now detects the carry-op + monotone-op combination on
# multi-bit outputs and returns TRANSPORTABLE instead of MONOTONIC.  The
# TRANSPORTABLE formula adds a soundness floor `| OR(T_inputs)`:
#
#     T_RAX = differential  |  T_RBX
#
# When T_RBX = MASK64, this expands to MASK64, which is sound.


class TestBlsiSoundness:
    """blsi rax, rbx with various taint configurations."""

    def test_blsi_all_inputs_tainted(self) -> None:
        # Originally microtaint returned T_RAX = 0x1.
        # True taint is MASK64.
        state = {
            'RAX': 2046325409002925761,
            'RBX': 10992312368177670734,
            'RCX': 2616401116910936113,
            'RDX': 16502545584051328317,
        }
        taint = {'RAX': MASK64, 'RBX': MASK64, 'RCX': MASK64, 'RDX': MASK64}
        _assert_sound(['blsi rax, rbx'], state, taint)

    def test_blsi_only_rbx_tainted(self) -> None:
        # T_RBX fully tainted, others clean.  Originally T_RAX = 0x1.
        state = {
            'RAX': 5511056715057449827,
            'RBX': 580158894615347727,
            'RCX': 9886609571253907124,
            'RDX': 544134995414518694,
        }
        taint = {'RBX': MASK64}
        _assert_sound(['blsi rax, rbx'], state, taint)

    def test_blsi_partial_taint(self) -> None:
        # Originally T_RAX = 0x9 missing bit 0x2 from the truth 0xb.
        state = {
            'RAX': 4053877055555174212,
            'RBX': 4936174083833827371,
            'RCX': 12025885062251604514,
            'RDX': 4613404405606908092,
        }
        taint = {
            'RBX': 5185061771817731555,
            'RCX': 10766409935409859823,
            'RDX': 5747731164532806120,
        }
        _assert_sound(['blsi rax, rbx'], state, taint)


# ─────────────────────────────────────────────────────────────────────────────
# Group 2: blsr (lowest-set-bit RESET)
# ─────────────────────────────────────────────────────────────────────────────
#
# `blsr rax, rbx` is the BMI1 sibling of blsi: RAX = RBX & (RBX - 1).
# It clears the lowest set bit of RBX.
#
# Sleigh lifts blsr similarly, but with INT_SUB instead of INT_2COMP:
#
#     INT_SUB    unique:tmp_8       ← register:RBX_8, const:1
#     INT_AND    register:RAX_8     ← unique:tmp, RBX_8
#
# The same root cause and fix apply: INT_SUB is in TRANSPORTABLE_OPCODES,
# INT_AND is in MONOTONIC_OPCODES.  The fix detects the combination.


class TestBlsrSoundness:
    """blsr rax, rbx with fully-tainted RBX."""

    def test_blsr_all_tainted(self) -> None:
        state = {'RAX': 0x1234567812345678, 'RBX': 0xFEDCBA9876543210, 'RCX': 0, 'RDX': 0}
        taint = {'RAX': MASK64, 'RBX': MASK64, 'RCX': MASK64, 'RDX': MASK64}
        _assert_sound(['blsr rax, rbx'], state, taint)

    def test_blsr_partial(self) -> None:
        state = {'RAX': 0xDEADBEEFCAFEBABE, 'RBX': 0xAAAAAAAAAAAAAAAA, 'RCX': 0, 'RDX': 0}
        taint = {'RBX': 0xFFFFFFFF00000000}  # only high half tainted
        _assert_sound(['blsr rax, rbx'], state, taint)


# ─────────────────────────────────────────────────────────────────────────────
# Group 3: blsi followed by andn (chain composition)
# ─────────────────────────────────────────────────────────────────────────────
#
# Sequence:
#     blsi  rax, rbx       ; RAX = RBX & -RBX  (lowest set bit isolation)
#     andn  rcx, rax, rdx  ; RCX = ~RAX & RDX  (BMI1 AND-NOT)
#
# This was unsound for two compounding reasons:
# (1) The single-instruction blsi was already unsound (Group 1) — it gave
#     T_RAX = 0x1 instead of MASK64.
# (2) ChainedCircuit then propagates that incorrect T_RAX into the andn
#     formula:
#         T_RCX = (V_RDX & T_RAX) | (~V_RAX & T_RDX) | (T_RAX & T_RDX)
#     With T_RAX = 0x1 (wrong) instead of T_RAX = 0 (correct given clean
#     RBX), the andn formula uses the wrong V_RAX from the chain's stale
#     concrete state.
#
# The fix to blsi alone resolves the under-taint of T_RAX, AND the
# concrete-value threading fix in ChainedCircuit ensures andn reads the
# post-blsi V_RAX (the lowest set bit of V_RBX) instead of the stale entry
# value.


class TestBlsiAndnChain:
    """blsi rax, rbx; andn rcx, rax, rdx — chained BMI1 ops."""

    def test_chain_blsi_andn_case1(self) -> None:
        # Originally got T_RCX = 0x8410a0b0180288f2 (15 bits short of truth).
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
        _assert_sound(['blsi rax, rbx', 'andn rcx, rax, rdx'], state, taint)

    def test_chain_blsi_andn_case2(self) -> None:
        # Originally 14 bits short of truth.
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
        _assert_sound(['blsi rax, rbx', 'andn rcx, rax, rdx'], state, taint)


# ─────────────────────────────────────────────────────────────────────────────
# Group 4: bswap; mov rbx, IMM; and rax, rbx  (permutation + mask via mov)
# ─────────────────────────────────────────────────────────────────────────────
#
# Sequence:
#     bswap  rax                          ; reverse byte order of RAX
#     mov    rbx, 0xff00ff00ff00ff00      ; load alternating-byte mask
#     and    rax, rbx                     ; mask with bswap's result
#
# Per-instruction taint is straightforward:
#   - bswap rax:               T_RAX → bswap(T_RAX)         (MAPPED routing)
#   - mov rbx, IMM:            T_RBX → 0                    (MAPPED with const)
#   - and rax, rbx:            T_RAX_out = (V_RBX & T_RAX) | (V_RAX & T_RBX) | (T_RAX & T_RBX)
#                                       = V_RBX & T_RAX     (since T_RBX=0 after mov)
#
# So the correct answer is `bswap(T_RAX_in) & V_RBX_after_mov`, where
# V_RBX_after_mov = 0xff00ff00ff00ff00.
#
# ──── The bug ────
# The AND formula reads V_RBX from the EvalContext's input_values.
# ChainedCircuit *did not* update input_values between steps, so the AND
# saw V_RBX_entry (whatever the caller passed in) instead of 0xff00...00.
# Bits cleared by the mov but set in V_RBX_entry → spurious over-taint.
# Bits SET by the mov but cleared in V_RBX_entry → silent under-taint
# (the mask says "clean" so the AND drops the T_RAX contribution).
#
# Numerical worked example (case 7):
#   V_RBX_entry      = 0xec02d12ddff8e8e5
#   V_RBX_after_mov  = 0xff00ff00ff00ff00
#   bits SET by mov  = ~entry & after = 0x130003004f004e00  (14 bits)
#   bswap(T_RAX_in)  = 0x9f08e0fafa018b01
#   under-taint mask = bswap(T_RAX_in) & set_bits = 0x1300000004004800  (6 bits)
#
# ──── The fix ────
# ChainedCircuit now runs each sub-circuit's instruction concretely on the
# running value state and merges the result back, so step 3 sees the new
# V_RBX = 0xff00ff00ff00ff00.


class TestBswapMaskChain:
    """bswap rax; mov rbx, IMM; and rax, rbx — value-staleness sequence."""

    def test_bswap_and_with_imm_mask_case1(self) -> None:
        # Originally 6 bits short.
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
        _assert_sound(
            ['bswap rax', 'mov rbx, 0xff00ff00ff00ff00', 'and rax, rbx'],
            state,
            taint,
        )

    def test_bswap_and_with_imm_mask_case2(self) -> None:
        # Originally 7 bits short.
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
        _assert_sound(
            ['bswap rax', 'mov rbx, 0xff00ff00ff00ff00', 'and rax, rbx'],
            state,
            taint,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Group 5: shift-and-OR codec (Base64 decode pattern)
# ─────────────────────────────────────────────────────────────────────────────
#
# Sequence (combines four 6-bit groups into one 24-bit word):
#     shl  rax, 18
#     shl  rbx, 12
#     shl  rcx, 6
#     or   rax, rbx
#     or   rax, rcx
#     or   rax, rdx
#
# Each shl is MAPPED (constant shift = pure bit routing).
# Each or is MONOTONIC.
#
# With all four registers fully tainted, the true T_RAX after the sequence
# is MASK64: every bit position can be reached by some combination of
# shifted input bits.
#
# ──── The bug ────
# After step 1, T_RAX = (T_RAX_in << 18) but V_RAX is still V_RAX_entry.
# At step 4 (`or rax, rbx`), the per-bit OR formula reads V_RAX:
#     T_RAX_out[i] = (~V_RAX[i] & T_RBX[i]) | (~V_RBX[i] & T_RAX[i]) | (T_RAX[i] & T_RBX[i])
# A bit position where V_RAX_entry has a 1 but the post-shift V_RAX has a 0
# would erroneously suppress T_RBX flow at that position, dropping taint.
#
# ──── The fix ────
# The concrete-state threading updates V_RAX after each shl and each or, so
# the OR formula always sees the actual current value.


class TestShiftOrCodec:
    """Base64-style 4x6-bit combine via shifts and ORs."""

    def test_shl_or_sequence(self) -> None:
        state = {
            'RAX': 8291989174624432245,
            'RBX': 10265026935744837952,
            'RCX': 4657709630386508352,
            'RDX': 12522554165043460411,
        }
        taint = {'RAX': MASK64, 'RBX': MASK64, 'RCX': MASK64, 'RDX': MASK64}
        _assert_sound(
            [
                'shl rax, 18',
                'shl rbx, 12',
                'shl rcx, 6',
                'or rax, rbx',
                'or rax, rcx',
                'or rax, rdx',
            ],
            state,
            taint,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Group 6: movzx + sub + movsx (strcmp byte-difference pattern)
# ─────────────────────────────────────────────────────────────────────────────
#
# Sequence (glibc-style strcmp inner loop):
#     movzx  rcx, al      ; RCX = zero-extended low byte of RAX
#     movzx  rdx, bl      ; RDX = zero-extended low byte of RBX
#     sub    rcx, rdx     ; RCX = char_a - char_b  (signed difference)
#     movsx  rax, cl      ; RAX = sign-extended low byte of RCX
#
# After the chain, RCX holds (al - bl) zero-extended.  RAX holds the
# sign-extension of cl, which depends on the sign bit of (al - bl).
#
# With T_RAX = T_RBX = MASK64 entry, the true T_RCX after the chain is
# MASK64 — because:
#   - movzx RCX,al pulls T_RAX[7:0] into T_RCX[7:0]
#   - movzx RDX,bl pulls T_RBX[7:0] into T_RDX[7:0]
#   - sub propagates carry across all 64 bits (the high bytes are all 0
#     concretely, but their concrete-zero-ness is value-dependent and the
#     sub's borrow can cascade through them)
#
# ──── The bug ────
# The post-movzx concrete values of RCX and RDX are bytes (≤0xff).  But
# without value threading, the sub formula sees the *entry* V_RCX/V_RDX
# (full 64-bit values), so the differential and the union term yield only
# the bottom 9 bits of taint.  True taint flows further because the post-
# movzx values are small and all the high bits are concretely 0, so even a
# small flip in the low byte can propagate through the sub's borrow chain.
#
# ──── The fix ────
# Concrete-state threading ensures the sub at step 3 sees V_RCX = al
# (a small byte) and V_RDX = bl, producing the correct differential.


class TestMovzxSubMovsxChain:
    """strcmp-style byte-difference computation."""

    def test_movzx_sub_movsx(self) -> None:
        state = {
            'RAX': 9692544848236511546,
            'RBX': 12871560762356780111,
            'RCX': 7448313025359211831,
            'RDX': 2936346809869392945,
        }
        taint = {'RAX': MASK64, 'RBX': MASK64, 'RCX': MASK64, 'RDX': MASK64}
        _assert_sound(
            ['movzx rcx, al', 'movzx rdx, bl', 'sub rcx, rdx', 'movsx rax, cl'],
            state,
            taint,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Group 7: comprehensive randomised soundness fuzzer
# ─────────────────────────────────────────────────────────────────────────────
#
# A randomised fuzzer that picks instructions and taint configurations and
# asserts soundness.  This is the safety net: any future regression that
# introduces under-taint should fail this test.


@pytest.mark.parametrize(
    'asm',
    [
        'blsi rax, rbx',
        'blsr rax, rbx',
        'blsi rcx, rdx',
        'blsr rcx, rdx',
        'andn rax, rbx, rcx',
        'and rax, rbx',
        'or  rax, rbx',
        'xor rax, rbx',
        'add rax, rbx',
        'sub rax, rbx',
        'sbb rax, rbx',
        'adc rax, rbx',
        'imul rax, rbx',
        'imul rax, rbx, 3',
        'shl rax, 5',
        'shr rax, cl',
        'rol rax, 8',
        'lea rax, [rbx + rcx*2]',
        'movzx rax, bl',
        'movsx rax, bl',
        'cmovz rax, rbx',
        'cmovnz rax, rbx',
        'cmovg rax, rcx',
        'tzcnt rax, rbx',
        'lzcnt rax, rbx',
        'popcnt rax, rbx',
        'bswap rax',
        'neg rax',
        'not rax',
        'inc rax',
    ],
)
def test_single_instruction_soundness_fuzz(asm: str) -> None:
    """Fuzz a single instruction across a variety of taint configurations.

    For each of 12 random states and 6 taint patterns, assert that microtaint
    produces a sound mask (no bit of the true per-bit-flip taint is missing
    from the output).
    """
    import random

    rng = random.Random(hash(asm))

    taint_patterns: list[dict[str, int]] = [
        dict.fromkeys(REGS, MASK64),  # all tainted
        {'RAX': MASK64},  # only RAX
        {'RBX': MASK64},  # only RBX
        {'RCX': MASK64},  # only RCX
        dict.fromkeys(REGS, 255),  # low byte only
        dict.fromkeys(REGS, 18446744069414584320),  # high half
    ]
    for _trial in range(12):
        state = {r: rng.randint(1, MASK64) for r in REGS}  # avoid 0 (DIV-by-zero etc)
        for taint in taint_patterns:
            _assert_sound([asm], state, taint)


@pytest.mark.parametrize(
    'seq',
    [
        ['blsi rax, rbx', 'andn rcx, rax, rdx'],
        ['add rax, rbx', 'add rcx, rax', 'add rdx, rcx'],
        ['shl rax, 4', 'or rbx, rax', 'shr rbx, 2'],
        ['mov rcx, rax', 'xor rdx, rcx', 'mov rax, rdx'],
        ['imul rax, rbx', 'add rcx, rax', 'mov rdx, rcx'],
        ['neg rax', 'sub rbx, rax', 'mov rcx, rbx'],
        ['add rax, rbx', 'adc rcx, rdx'],
        ['sub rax, rbx', 'sbb rcx, rdx'],
        ['xor rax, rax', 'add rax, rbx'],
        ['movzx rax, bl', 'add rax, rcx'],
        ['movsx rax, bl', 'imul rax, rdx'],
        ['rol rax, 8', 'ror rbx, 8'],
        ['bswap rax', 'mov rbx, 0xff00ff00ff00ff00', 'and rax, rbx'],
        ['shl rax, 18', 'shl rbx, 12', 'shl rcx, 6', 'or rax, rbx', 'or rax, rcx', 'or rax, rdx'],
        ['movzx rcx, al', 'movzx rdx, bl', 'sub rcx, rdx', 'movsx rax, cl'],
        ['rol rax, 13', 'xor rax, rbx', 'rol rax, 7', 'xor rax, rcx', 'add rax, rdx'],
    ],
)
def test_chain_soundness_fuzz(seq: list[str]) -> None:
    """Fuzz multi-instruction sequences for soundness across random states."""
    import random

    rng = random.Random(hash(tuple(seq)))

    for _trial in range(8):
        state = {r: rng.randint(1, MASK64) for r in REGS}
        for taint_mask in (MASK64, 0xFF, 0xFFFF0000FFFF0000):
            taint = dict.fromkeys(REGS, taint_mask)
            _assert_sound(seq, state, taint)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark regression suite — cases observed unsound in the differential
# benchmark (report_1777829004.json) when run through worker_microtaint.py.
# ─────────────────────────────────────────────────────────────────────────────
#
# These cases were reported as under-taint by the benchmark's GT simulator
# (Unicorn enumeration over all 2**k assignments of the k tainted bits).
# Each case captures the EXACT state and taint the worker received.
#
# The worker uses a register list of 4 GP regs only (no flag registers),
# while the rest of this test file uses 4 GP + 5 flags.  We mirror the
# worker's setup in `_microtaint_run_worker_style` below so this test
# surface matches the failure surface — soundness must hold under both
# register lists.

_WORKER_REGS = [Register(n, 64) for n in REGS]  # exactly what worker_microtaint.py uses


def _microtaint_run_worker_style(asm_lines: list[str], state: dict[str, int], taint: dict[str, int]) -> dict[str, int]:
    """Run microtaint with the EXACT register list used by worker_microtaint.py.

    Differs from `_microtaint_run` in that the register list contains only
    the four GP registers — no flag bits.  This is what the benchmark
    workers actually execute, so soundness regressions visible in the
    benchmark must be reproducible here.
    """
    code = _assemble(asm_lines)
    sim = CellSimulator(Architecture.AMD64)
    _cached_generate_static_rule.cache_clear()
    circuit = generate_static_rule(Architecture.AMD64, code, _WORKER_REGS)
    full_state = {r: state.get(r, 0) for r in REGS}
    full_taint = {r: taint.get(r, 0) for r in REGS}
    ctx = EvalContext(
        input_taint=full_taint,
        input_values=full_state,
        simulator=sim,
        implicit_policy=ImplicitTaintPolicy.IGNORE,
    )
    return circuit.evaluate(ctx)


def _assert_sound_worker(asm_lines: list[str], state: dict[str, int], taint: dict[str, int]) -> None:
    """Run microtaint with the worker register list, compute Unicorn ground
    truth, assert no under-taint."""
    code = _assemble(asm_lines)
    truth = _ground_truth(code, state, taint)
    got = _microtaint_run_worker_style(asm_lines, state, taint)
    failures: list[str] = []
    for reg in REGS:
        true_mask = truth.get(reg, 0)
        got_mask = got.get(reg, 0) & MASK64 if isinstance(got.get(reg, 0), int) else 0
        under = true_mask & ~got_mask
        if under:
            n_under = bin(under).count('1')
            failures.append(
                f'{reg}: under-taint by {n_under} bits — '
                f'true={hex(true_mask)} got={hex(got_mask)} missing={hex(under)}',
            )
    if failures:
        msg = '\n  '.join(failures)
        pytest.fail(f'Soundness failure (worker register list) on {asm_lines!r}:\n  {msg}')


class TestBenchmarkRegression20260503:
    """Regression cases from differential benchmark report_1777829004.json.

    Each test re-runs the EXACT (state, taint, instruction-sequence) tuple
    that the benchmark flagged as under-taint, using the worker's 4-GP-only
    register list, and asserts soundness against Unicorn-enumeration GT.

    These tests document specific arithmetic patterns where microtaint has
    been observed to under-taint:

    - **Carry chains** in repeated additions and neg/sub combinations: the
      bit positions reached by carry propagation across multiple ops
      depend on the concrete operand values, and the static analysis must
      report at least the bits that vary across some choice of tainted
      inputs.
    - **Variable shift after immediate load to count register**:
      ``mov cl, 4; shl rax, cl`` — after the mov, CL is concrete 4, so
      the shift is by a known constant and the answer is just the input
      taint shifted left by 4.
    - **BMI andn**: ``blsi rax, rbx; andn rcx, rax, rdx`` — RDX flows to
      RCX through ``andn`` and that data flow must not be lost.
    - **Cryptographic-style rotation/xor chains**: ``rol/xor/rol/xor/add``
      across 4 registers — the precise bit positions that depend on each
      input require correct rotation semantics in the lifter.

    Each is currently expected to PASS (the engine has been fixed); a
    regression here means a real soundness loss.  Cases that are observed
    over-tainting (sound but imprecise) are not asserted here — only
    soundness violations.
    """

    # id 821: neg rax; sub rbx, rax; mov rcx, rbx — neg-sub-mov chain (case A)
    def test_821_neg_sub_mov_caseA(self) -> None:
        state = {
            'RAX': 15773031461073224685,
            'RBX': 15207111530931456857,
            'RCX': 12794402654832150797,
            'RDX': 11036027302511947317,
        }
        taint = {
            'RAX': 76561193665298432,
            'RBX': 150994944,
            'RCX': 9007216434610177,
            'RDX': 1152921504606846976,
        }
        _assert_sound_worker(['neg rax', 'sub rbx, rax', 'mov rcx, rbx'], state, taint)

    # id 825: mov cl, 4; shl rax, cl — variable shift by constant count
    def test_825_mov_cl_then_shl(self) -> None:
        state = {
            'RAX': 3145102647585366598,
            'RBX': 323498066970872906,
            'RCX': 16340536423274718691,
            'RDX': 10011729623591357551,
        }
        taint = {
            'RAX': 576460821022916608,
            'RBX': 73744,
            'RCX': 1152921504607109120,
            'RDX': 0,
        }
        _assert_sound_worker(['mov cl, 4', 'shl rax, cl'], state, taint)

    # id 851: rol/xor/rol/xor/add chain (crypto-style permutation)
    def test_851_crypto_chain_caseA(self) -> None:
        state = {
            'RAX': 9462808105761198019,
            'RBX': 1260233167559427307,
            'RCX': 3211362100196543915,
            'RDX': 8766221879185115725,
        }
        taint = {
            'RAX': 4611686022722355200,
            'RBX': 1342177280,
            'RCX': 12884901888,
            'RDX': 35433480192,
        }
        _assert_sound_worker(
            ['rol rax, 13', 'xor rax, rbx', 'rol rax, 7', 'xor rax, rcx', 'add rax, rdx'],
            state,
            taint,
        )

    # id 869: 4x add rax, rbx — pure carry chain stress test
    def test_869_add_chain_4x(self) -> None:
        state = {
            'RAX': 3422279987030865735,
            'RBX': 3947325121171498645,
            'RCX': 1838748979518619040,
            'RDX': 653864017478472027,
        }
        taint = {
            'RAX': 1125899940397120,
            'RBX': 70368744181760,
            'RCX': 0,
            'RDX': 2251800082120712,
        }
        _assert_sound_worker(
            ['add rax, rbx', 'add rax, rbx', 'add rax, rbx', 'add rax, rbx'],
            state,
            taint,
        )

    # id 871: blsi + andn — BMI dataflow into third register
    def test_871_blsi_andn(self) -> None:
        state = {
            'RAX': 11158715849010693534,
            'RBX': 1105559444368898328,
            'RCX': 4489845537896655591,
            'RDX': 17813962747657217048,
        }
        taint = {
            'RAX': 35184372088832,
            'RBX': 70377334112256,
            'RCX': 0,
            'RDX': 17592186044416,
        }
        _assert_sound_worker(['blsi rax, rbx', 'andn rcx, rax, rdx'], state, taint)

    # id 939: rol/xor/rol/xor/add chain — second case with different state/taint
    def test_939_crypto_chain_caseB(self) -> None:
        state = {
            'RAX': 17310100993514881776,
            'RBX': 8352897263129780062,
            'RCX': 12834837876267824982,
            'RDX': 15552879662868555556,
        }
        taint = {
            'RAX': 1049600,
            'RBX': 72057594037927936,
            'RCX': 8796630941696,
            'RDX': 1099511627808,
        }
        _assert_sound_worker(
            ['rol rax, 13', 'xor rax, rbx', 'rol rax, 7', 'xor rax, rcx', 'add rax, rdx'],
            state,
            taint,
        )

    # id 954: add cascade across 3 registers — RAX→RCX→RDX carry propagation
    def test_954_add_cascade_3reg(self) -> None:
        state = {
            'RAX': 2239767635897444328,
            'RBX': 9688080004658431800,
            'RCX': 14895366135836793746,
            'RDX': 1764350782183614702,
        }
        taint = {
            'RAX': 8796093055012,
            'RBX': 8589967361,
            'RCX': 0,
            'RDX': 16384,
        }
        _assert_sound_worker(
            ['add rax, rbx', 'add rcx, rax', 'add rdx, rcx'],
            state,
            taint,
        )

    # id 955: neg-sub-mov chain (case B with different state/taint)
    def test_955_neg_sub_mov_caseB(self) -> None:
        state = {
            'RAX': 5487737478358155806,
            'RBX': 12199477105233075724,
            'RCX': 1304497809212559287,
            'RDX': 2180567047932399672,
        }
        taint = {
            'RAX': 65536,
            'RBX': 8200,
            'RCX': 618475290624,
            'RDX': 2306124485264146432,
        }
        _assert_sound_worker(['neg rax', 'sub rbx, rax', 'mov rcx, rbx'], state, taint)

    # id 958: movzx rax, bx; add rax, rcx — zero-extension then add carry chain.
    # Source: report_1777831878.json.  After movzx, RAX is concrete (T_RBX=0
    # so T_RAX_after_movzx=0).  The subsequent add rax, rcx adds a tainted
    # RCX into a concrete RAX, and the carry from low tainted bits of RCX
    # must propagate up.  Microtaint missed bits 10, 11 in RAX (carry from
    # bit 9 of T_RCX).
    def test_958_movzx_add(self) -> None:
        state = {
            'RAX': 0xF07AFAB7CB9F82A5,
            'RBX': 0x41F3E01C7DA1F511,
            'RCX': 0x29D2680AB5F861B0,
            'RDX': 0xF6E0A26E39BCE100,
        }
        taint = {
            'RAX': 0x8000000001002,
            'RBX': 0,
            'RCX': 0x200200,
            'RDX': 0x20000010000080,
        }
        _assert_sound_worker(['movzx rax, bx', 'add rax, rcx'], state, taint)

    # id 976: rol/xor crypto chain — same pattern as 851/939, third state.
    # Captured separately because the failure surface (which carry/rotation
    # path under-taints) varies with concrete state.
    def test_976_crypto_chain_caseC(self) -> None:
        state = {
            'RAX': 0xEFCD7FAEBE5C32B8,
            'RBX': 0xB7EFEF05D8FD2048,
            'RCX': 0x36FDA98D94AB24E9,
            'RDX': 0x8DC12EBA1F808E1C,
        }
        taint = {
            'RAX': 0,
            'RBX': 0,
            'RCX': 0x80,
            'RDX': 0x2000000000000000,
        }
        _assert_sound_worker(
            ['rol rax, 13', 'xor rax, rbx', 'rol rax, 7', 'xor rax, rcx', 'add rax, rdx'],
            state,
            taint,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Worker-style fuzz: same patterns the benchmark observed unsound, but with
# many random states.  Catches regressions on the same instruction shapes
# even if the specific (state, taint) tuples above are all sound.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    'seq',
    [
        # Patterns observed unsound in report_1777829004.json + report_1777831878.json
        ['neg rax', 'sub rbx, rax', 'mov rcx, rbx'],
        ['mov cl, 4', 'shl rax, cl'],
        ['mov cl, 7', 'shr rax, cl'],
        ['mov cl, 13', 'rol rax, cl'],
        ['rol rax, 13', 'xor rax, rbx', 'rol rax, 7', 'xor rax, rcx', 'add rax, rdx'],
        ['add rax, rbx', 'add rax, rbx', 'add rax, rbx', 'add rax, rbx'],
        ['add rax, rbx', 'add rcx, rax', 'add rdx, rcx'],
        ['blsi rax, rbx', 'andn rcx, rax, rdx'],
        ['movzx rax, bx', 'add rax, rcx'],  # report_1777831878 id=958
        ['movzx rax, bl', 'add rax, rcx'],  # variation: movzx from byte
        ['movsx rax, bx', 'add rax, rcx'],  # variation: signed extension
        # Variations on the same themes
        ['blsr rax, rbx', 'andn rcx, rax, rdx'],
        ['neg rax', 'add rbx, rax', 'mov rcx, rbx'],
        ['adc rax, rbx', 'adc rax, rbx', 'adc rax, rbx'],
    ],
)
def test_worker_style_pattern_fuzz(seq: list[str]) -> None:
    """Fuzz the exact patterns the differential benchmark flagged.

    Uses the worker's 4-GP-only register list so the test surface matches
    the worker's actual setup.  For each pattern, run 20 random
    (state, taint) tuples.  Any unsoundness reproduces a bug visible in
    the benchmark.
    """
    import random

    rng = random.Random(hash(tuple(seq)) & 0xFFFF_FFFF)

    for trial in range(20):
        state = {r: rng.randint(1, MASK64) for r in REGS}
        # Mix sparse and full taint patterns — sparse is what GT can verify
        # against (k <= 16), full is the stress-test case.
        if trial < 10:
            # Sparse: 1-4 bits per register at random positions
            taint = {}
            for r in REGS:
                n_bits = rng.choice([0, 1, 2, 4])
                m = 0
                for _ in range(n_bits):
                    m |= 1 << rng.randint(0, 63)
                taint[r] = m
        else:
            # Random partial
            taint = {r: rng.randint(0, MASK64) for r in REGS}
        _assert_sound_worker(seq, state, taint)


# ─────────────────────────────────────────────────────────────────────────────
# KNOWN FAILING CASES — documented and tracked but expected to fail today.
# ─────────────────────────────────────────────────────────────────────────────
#
# SBB chains: ``sbb dst, src; sbb dst2, dst; sbb dst3, dst2``.
#
# Under heavy/random taint, the second ``sbb`` (sbb rcx, rax) under-taints
# bit 0 of its destination by 1 bit.  The third sbb propagates correctly,
# so RDX bit 0 IS tainted even though RCX bit 0 isn't — proving the
# engine does track CF across SBB instructions but mishandles the fan-in
# into the subtracted register.
#
# Reproduction (random.Random(6), worker register list, KEEP policy):
#   asm:    sbb rax, rbx; sbb rcx, rax; sbb rdx, rcx
#   state:  RAX=0x92e5dfe8cb1855ff RBX=0x14a03569d26b9497
#           RCX=0xc320a4737c2b3abf RDX=0x96d373742f9a03a
#   taint:  RAX=0xbc1e3ac1c27db4ec RBX=0xc527e27951c34250
#           RCX=0xae9af1698a0c5100 RDX=0xaf895f5b9c2c0ac2
#   truth RCX=0xfffffffbdffffffd  got RCX=0xfffffffffffffffc  missing 0x1
#
# The ADC chain ``adc rax, rbx`` x3 with the same fuzz IS sound, so the
# bug is specific to ``sbb``'s P-code lifting (likely INT_SBORROW
# producing an op signature the bit-precise classifier handles slightly
# differently from INT_CARRY).
#
# Until fixed, this is documented as @pytest.mark.xfail.  When you fix
# the SBB lifter / classifier, flip the marker to .xpass-strict to
# guard against regressions.


@pytest.mark.xfail(
    reason='Known SBB chain unsoundness: sbb rcx, rax under-taints bit 0 of '
    'RCX after a prior sbb. The lifter mishandles INT_SBORROW fan-in '
    'into the destination register. Tracked separately from the 8 '
    'benchmark regressions which are all fixed.',
    strict=False,
)
def test_known_failing_sbb_chain() -> None:
    """Known unsound case: sbb cascade across 3 registers.

    This test should xfail today.  When the engine bug is fixed, change
    the strict=False to strict=True so a future regression flips it back.
    """
    seq = ['sbb rax, rbx', 'sbb rcx, rax', 'sbb rdx, rcx']
    state = {
        'RAX': 0x92E5DFE8CB1855FF,
        'RBX': 0x14A03569D26B9497,
        'RCX': 0xC320A4737C2B3ABF,
        'RDX': 0x96D373742F9A03A,
    }
    taint = {
        'RAX': 0xBC1E3AC1C27DB4EC,
        'RBX': 0xC527E27951C34250,
        'RCX': 0xAE9AF1698A0C5100,
        'RDX': 0xAF895F5B9C2C0AC2,
    }
    _assert_sound_worker(seq, state, taint)
