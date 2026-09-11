"""Cross-lane SIMD taint propagation (x86 SSE/SSE2/SSE4.1).

Most SIMD is lane-local (packed add/sub/mul are SLEIGH-scalarised per element,
no inter-element carry).  These instructions instead move information ACROSS the
8-byte (VL_) lane boundary, so they are the cases where a naive per-lane taint
model would be unsound.  microtaint tracks them soundly because each per-lane
output is evaluated by running the whole wide operand through the cell.

Two classes:

  * EXACT byte permutations -- whole-register byte shift (psrldq), dword shuffle
    (pshufd), byte unpack/interleave (punpcklbw/punpckhbw): tainting one input
    byte taints EXACTLY the output byte(s) it moves to, even across the lane
    boundary.
  * AVALANCHE -- data-dependent / horizontal CALLOTHER ops (pshufb, psadbw,
    pmaddwd, packuswb, pmovsxbw): any tainted input byte soundly taints the
    affected output (over-approximated to the whole destination).

Tests are written with whole XMM registers and byte slices via the debug/test
RegisterAliases helper.  Known gap (xfail): pslldq lifts to a branchy per-half
p-code sequence whose sub-slice determine_category cannot categorise yet
(psrldq, the mirror op, lifts cleanly and works).
"""

from __future__ import annotations

import pytest
from keystone import KS_ARCH_X86, KS_MODE_64, Ks

from microtaint.debug.reg_aliases import RegisterAliases
from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, ImplicitTaintPolicy

_KS = Ks(KS_ARCH_X86, KS_MODE_64)
_A = RegisterAliases(Architecture.AMD64)
_FMT = _A.state_format(['RAX', 'XMM0', 'XMM1'])


def _tainted_out_bytes(asm: str, in_taint: dict[str, int]) -> set[int]:
    """Assemble `asm`, propagate `in_taint` (human XMM slices), return the set of
    XMM0 output byte indices (0..15) that end up tainted."""
    code, _ = _KS.asm(asm, 0x1000)
    circ = generate_static_rule(Architecture.AMD64, bytes(code), _FMT)
    out = circ.evaluate(EvalContext(
        input_taint=_A.to_engine(in_taint), input_values={},
        simulator=CellSimulator(Architecture.AMD64), implicit_policy=ImplicitTaintPolicy.IGNORE,
    ))
    x0 = _A.read(out, 'XMM0')
    return {i for i in range(16) if (x0 >> (i * 8)) & 0xFF}


def _byte(reg: str, idx: int) -> dict[str, int]:
    """Taint exactly byte `idx` of `reg` (a human XMM byte slice)."""
    return {f'{reg}[{idx * 8 + 7}:{idx * 8}]': 0xFF}


# ---------------------------------------------------------------------------
# EXACT cross-lane byte permutations
# ---------------------------------------------------------------------------


def test_psrldq_moves_byte_across_lane_exact() -> None:
    """psrldq xmm0,1 = shift whole reg right 1 byte: out[j] = in[j+1].  Byte 8
    (low of the high lane) moves to byte 7 (high of the low lane) -- crossing the
    8-byte boundary -- and NOTHING else is tainted."""
    assert _tainted_out_bytes('psrldq xmm0, 1', _byte('XMM0', 8)) == {7}


def test_psrldq_whole_register_exact() -> None:
    """psrldq xmm0,1 with all 16 bytes tainted: every byte shifts down one, so
    bytes 0..14 are tainted and byte 15 (zero-filled) is clean."""
    assert _tainted_out_bytes('psrldq xmm0, 1', {'XMM0': (1 << 128) - 1}) == set(range(15))


def test_pshufd_reverse_dwords_crosses_lanes_exact() -> None:
    """pshufd xmm0,xmm1,0x1b reverses the four dwords: dword0 -> dword3.  So
    XMM1 byte 0 lands at output byte 12 (low lane -> high lane), and byte 5
    (dword1, offset1) lands at byte 9."""
    assert _tainted_out_bytes('pshufd xmm0, xmm1, 0x1b', _byte('XMM1', 0)) == {12}
    assert _tainted_out_bytes('pshufd xmm0, xmm1, 0x1b', _byte('XMM1', 5)) == {9}


def test_punpcklbw_interleaves_low_bytes_exact() -> None:
    """punpcklbw xmm0,xmm1: out[2i]=xmm0[i], out[2i+1]=xmm1[i] for i=0..7.  A low
    source byte fans out to a high output byte across the lane boundary."""
    assert _tainted_out_bytes('punpcklbw xmm0, xmm1', _byte('XMM0', 4)) == {8}   # 2*4
    assert _tainted_out_bytes('punpcklbw xmm0, xmm1', _byte('XMM1', 4)) == {9}   # 2*4+1
    assert _tainted_out_bytes('punpcklbw xmm0, xmm1', _byte('XMM0', 0)) == {0}   # stays low


def test_punpckhbw_interleaves_high_bytes_exact() -> None:
    """punpckhbw xmm0,xmm1: out[2i]=xmm0[i+8], out[2i+1]=xmm1[i+8].  A high-lane
    source byte can move down to the low lane."""
    assert _tainted_out_bytes('punpckhbw xmm0, xmm1', _byte('XMM0', 8)) == {0}    # (8-8)*2
    assert _tainted_out_bytes('punpckhbw xmm0, xmm1', _byte('XMM0', 12)) == {8}   # (12-8)*2


# ---------------------------------------------------------------------------
# AVALANCHE cross-lane (data-dependent / horizontal CALLOTHER ops)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('asm', [
    'pshufb xmm0, xmm1',   # data-dependent byte shuffle
    'psadbw xmm0, xmm1',   # sum of absolute differences (horizontal)
    'pmaddwd xmm0, xmm1',  # multiply and add adjacent words
    'packuswb xmm0, xmm1',  # pack words -> bytes with saturation
    'pmovsxbw xmm0, xmm1',  # sign-extend bytes -> words (SSE4.1)
])
def test_crosslane_callother_avalanches_soundly(asm: str) -> None:
    """A tainted input byte in one lane must taint the whole destination (sound
    over-approximation): these ops are opaque CALLOTHER, so any input bit can
    reach any output bit."""
    out = _tainted_out_bytes(asm, _byte('XMM0', 0))
    assert out == set(range(16)), f'{asm}: expected full-output avalanche, got {sorted(out)}'


def test_pshufb_high_lane_input_avalanches() -> None:
    """A high-lane (byte 12) input byte also avalanches -- proving the cross-lane
    flow is captured from either lane."""
    assert _tainted_out_bytes('pshufb xmm0, xmm1', _byte('XMM1', 12)) == set(range(16))


# ---------------------------------------------------------------------------
# pslldq: the mirror of psrldq, and much harder to lift.  Where psrldq is one
# 16-byte INT_RIGHT, pslldq arrives as a BRANCHY per-half funnel --
# `hi = (hi << 8) | (lo >> 56)` -- guarded by CBRANCHes on constants.
#
# It used to RAISE: no category claimed the low half's slice, because the
# INT_MULT computing the shift amount from two constants made the permutation
# recogniser refuse.  Once it categorised, every byte came out one wider than
# the truth -- taint at b AND b+1 -- because the decided branches were read as
# conditional writes, so the destination was assumed to maybe keep its old
# value.  Both are fixed, and both fixes say the same thing: an operation over
# constants carries no data.
# ---------------------------------------------------------------------------


def test_pslldq_moves_byte_across_lane_exact() -> None:
    """pslldq xmm0,1 = shift whole reg left 1 byte: out[j] = in[j-1].  Byte 7
    should move to byte 8 (crossing the lane boundary)."""
    assert _tainted_out_bytes('pslldq xmm0, 1', _byte('XMM0', 7)) == {8}


def test_pslldq_is_exact_on_every_byte() -> None:
    """Not just the one crossing byte: all sixteen.

    The failure this replaces was uniform -- every byte landed at its
    destination AND at its source -- so a test of a single byte would have been
    satisfied by a rule that was wrong everywhere.  Byte 15 shifts out of the
    register entirely and must leave nothing behind.
    """
    for b in range(16):
        want = {b + 1} if b < 15 else set()
        got = _tainted_out_bytes('pslldq xmm0, 1', _byte('XMM0', b))
        assert got == want, (
            f'input byte {b} produced {sorted(got)}, expected {sorted(want)}')
