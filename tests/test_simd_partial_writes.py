"""
SIMD soundness regression suite for the microtaint C-pcode evaluator.

These tests exercise instructions and chains that hit the partial-register
write/read merge logic in ``cell_c.c::frame_read_reg`` and ``frame_write_reg``
(and the matching Cython helpers in ``cell.pyx``).  They were added after
report_1778076700.json's id=824 (SIMD chain) was shown to be unsound on the C
path despite the engine producing correct output via the chained-circuit
decomposition.

Each test compares the C-pcode evaluator against Unicorn (the gold standard for
x86 semantics).  Mismatch on a single test means a real soundness regression:
the static rule generator builds expressions around concrete simulator outputs,
so wrong concrete outputs translate directly into wrong differential / avalanche
taint masks downstream.

Tests are written with human register names (RAX, and the explicit XMM lane
slices ``XMM0[63:0]`` / ``XMM0[127:64]``) via the debug/test-only
``RegisterAliases`` helper.  They read lanes rather than whole XMM registers
because the comparison is against the Unicorn oracle, whose concrete read path
resolves individual lanes, not a whole-vector operand.
"""

from __future__ import annotations

import pytest

from microtaint.debug.reg_aliases import RegisterAliases
from microtaint.instrumentation.ast import InstructionCellExpr
from microtaint.simulator import CellSimulator, MachineState
from microtaint.types import Architecture

# Module-level sim + alias table — cheap to construct once per test session.
_SIM = CellSimulator(Architecture.AMD64)
_A = RegisterAliases(Architecture.AMD64)


def _eval(
    use_unicorn: bool, bytestring: str, regs_in: dict[str, int], out_reg: str,
    bit_start: int = 0, bit_end: int = 63,
) -> int:
    """Single-cell evaluation on either the C-pcode or Unicorn backend.
    ``out_reg`` is a human name (RAX, XMM0[63:0]); it is translated to the
    engine's geometry name internally."""
    cell = InstructionCellExpr(
        architecture=Architecture.AMD64,
        instruction=bytestring,
        out_reg=_A.to_engine_names(out_reg)[0],
        out_bit_start=bit_start,
        out_bit_end=bit_end,
        inputs={},
    )
    state = MachineState(regs=regs_in, mem={})
    saved = _SIM.use_unicorn
    try:
        _SIM.use_unicorn = use_unicorn
        return _SIM.evaluate_concrete(cell, state)
    finally:
        _SIM.use_unicorn = saved


def _state(overrides: dict[str, int]) -> dict[str, int]:
    """A concrete register state the C evaluator can read: the four GP regs and
    XMM0-7 (all lanes) pre-zeroed, with human-named overrides applied.  Overrides
    use RAX or XMM lane slices (XMM0[63:0], XMM1[127:64], ...)."""
    state = _A.to_engine({f'XMM{n}': 0 for n in range(8)})
    state.update({'RAX': 0, 'RBX': 0, 'RCX': 0, 'RDX': 0})
    state.update(_A.to_engine(overrides))
    return state


def _assert_c_matches_unicorn(
    label: str, bytestring: str, regs_in: dict[str, int], out_reg: str,
    bit_start: int = 0, bit_end: int = 63,
) -> None:
    """C-pcode evaluator must agree bit-for-bit with Unicorn."""
    c = _eval(False, bytestring, regs_in, out_reg, bit_start, bit_end)
    u = _eval(True, bytestring, regs_in, out_reg, bit_start, bit_end)
    assert c == u, (
        f'{label}: C-pcode != Unicorn\n'
        f'  bytes={bytestring}  regs_in={regs_in}\n'
        f'  out={out_reg}[{bit_end}:{bit_start}]\n'
        f'  C       = {hex(c)}\n'
        f'  Unicorn = {hex(u)}'
    )


# =============================================================================
# 1.  Single SIMD instructions -- XMM low lane
# =============================================================================


@pytest.mark.parametrize(
    ('xmm0_lo', 'xmm1_lo', 'label'),
    [
        (0x1, 0x2, 'minimal'),
        (0xFFFFFFFFFFFFFFFF, 0x1, 'overflow byte 0'),
        (0xDEADBEEFCAFEBABE, 0x1234567890ABCDEF, 'random'),
    ],
)
def test_paddq_xmm_lo(xmm0_lo: int, xmm1_lo: int, label: str) -> None:
    """PADDQ -- single 64-bit add per lane.  No partial writes involved;
    serves as a sanity floor before we exercise the byte-fanout cases."""
    _assert_c_matches_unicorn(
        f'paddq xmm0,xmm1 ({label})', '660fd4c1',
        _state({'XMM0[63:0]': xmm0_lo, 'XMM1[63:0]': xmm1_lo}), 'XMM0[63:0]',
    )


@pytest.mark.parametrize(
    ('xmm0_lo', 'xmm1_lo', 'label'),
    [
        (0x0102030405060708, 0x0101010101010101, 'no carry'),
        (0xFFFFFFFFFFFFFFFF, 0x0101010101010101, 'all bytes carry -- but PADDB truncates per byte'),
        (0x80, 0x80, 'single byte saturate'),
    ],
)
def test_paddb_fans_out_to_byte_writes(xmm0_lo: int, xmm1_lo: int, label: str) -> None:
    """PADDB lifts to 16 separate 1-byte INT_ADDs at offsets 0x1200..0x120f.
    The C path must store these as per-byte slots and the read-back at
    XMM0[63:0] must merge them with the original parent value."""
    _assert_c_matches_unicorn(
        f'paddb xmm0,xmm1 ({label})', '660ffcc1',
        _state({'XMM0[63:0]': xmm0_lo, 'XMM1[63:0]': xmm1_lo}), 'XMM0[63:0]',
    )


@pytest.mark.parametrize(
    ('xmm0_lo', 'xmm1_lo'),
    [
        (0xAAAA, 0x5555),  # bit-complement -> all 1s
        (0x0000, 0xFFFF),  # zero ^ ones
        (0xCAFEBABE12345678, 0xDEADBEEF87654321),
    ],
)
def test_pxor_xmm_lo(xmm0_lo: int, xmm1_lo: int) -> None:
    """PXOR is one 16-byte INT_XOR.  The C path's frame_read_reg handles this for
    the low lane (size-8 fits in uint64_t slot)."""
    _assert_c_matches_unicorn(
        'pxor xmm0,xmm1', '660fefc1',
        _state({'XMM0[63:0]': xmm0_lo, 'XMM1[63:0]': xmm1_lo}), 'XMM0[63:0]',
    )


@pytest.mark.parametrize(
    ('xmm0_lo', 'shift_imm', 'encoding'),
    [
        (0xAB, 8, '660f73f008'),  # psllq xmm0, 8
        (0x100, 16, '660f73f010'),  # psllq xmm0, 16
        (0x1, 4, '660f73f004'),  # psllq xmm0, 4
    ],
)
def test_psllq_xmm_lo(xmm0_lo: int, shift_imm: int, encoding: str) -> None:
    """PSLLQ shifts the low lane by an immediate.  Tests the
    write-merge-then-read path for size-8 ops."""
    _assert_c_matches_unicorn(
        f'psllq xmm0, {shift_imm}', encoding,
        _state({'XMM0[63:0]': xmm0_lo}), 'XMM0[63:0]',
    )


# =============================================================================
# 2.  GP <-> XMM lane transitions
# =============================================================================


@pytest.mark.parametrize('rax_in', [0x0, 0xFF, 0xDEADBEEF, 0xCAFEBABE12345678, 0xFFFFFFFFFFFFFFFF])
def test_movq_xmm_from_gp(rax_in: int) -> None:
    """``movq xmm0, rax`` lifts to ``INT_ZEXT register:0x0/8 -> register:0x1200/16``.
    The 16-byte write must populate XMM0's low lane exactly with rax (zero-extended
    into the high 8 bytes of XMM0)."""
    _assert_c_matches_unicorn(
        f'movq xmm0, rax (RAX={hex(rax_in)})', '66480f6ec0',
        _state({'RAX': rax_in}), 'XMM0[63:0]',
    )


@pytest.mark.parametrize('xmm0_lo', [0x0, 0xCAFE, 0xDEADBEEFCAFEBABE])
def test_movq_gp_from_xmm(xmm0_lo: int) -> None:
    """``movq rax, xmm0`` lifts to ``SUBPIECE register:0x1200/16, const:0x0/4 ->
    register:0x0/8``.  The size-16 read must yield the LOW 8 bytes of XMM0."""
    _assert_c_matches_unicorn(
        f'movq rax, xmm0 (XMM0[63:0]={hex(xmm0_lo)})', '66480f7ec0',
        _state({'XMM0[63:0]': xmm0_lo}), 'RAX',
    )


def test_gp_xmm_gp_roundtrip_preserves_value() -> None:
    """movq xmm0, rax; movq rax, xmm0 must be identity for the bottom 8 bytes --
    the simplest GP<->XMM<->GP roundtrip the SIMD-roundtrip soundness fix relies on."""
    rax = 0xDEADBEEFCAFEBABE
    out = _eval(False, '66480f6ec066480f7ec0', _state({'RAX': rax}), 'RAX')
    assert out == rax, f'roundtrip lost data: in={hex(rax)} out={hex(out)}'


# =============================================================================
# 3.  Multi-instruction chains -- interaction of partial writes
# =============================================================================
#
# These chains were the actual reproducer for the id=824 bug.  PADDB fans out to
# 16 per-byte writes; the next instruction (PXOR/PSLLQ) does a wider read/write
# and must NOT see stale per-byte values overlay back onto its result.


def test_paddb_then_pxor_lo_lane() -> None:
    """PADDB (16 byte writes) followed by PXOR (16-byte XOR).  The bug we fixed:
    PXOR's 16-byte read after PADDB was returning 0 because the write-side
    invalidation of stale per-byte sub-writes wasn't running."""
    _assert_c_matches_unicorn(
        'paddb; pxor', '660ffcc1660fefc1',
        _state({'XMM0[63:0]': 0xAB, 'XMM1[63:0]': 0xCD}), 'XMM0[63:0]',
    )


def test_paddb_pxor_psllq_chain() -> None:
    """PADDB; PXOR; PSLLQ -- exercises the write-invalidation logic at every step.
    After PSLLQ writes the wider lane, a final read at XMM0[63:0] must yield the
    shifted value, not be re-overlaid by the stale per-byte PADDB writes."""
    _assert_c_matches_unicorn(
        'paddb; pxor; psllq 8', '660ffcc1660fefc1660f73f008',
        _state({'XMM0[63:0]': 0xAB, 'XMM1[63:0]': 0xCD}), 'XMM0[63:0]',
    )


def test_simd_chain_id_824_simple_state() -> None:
    """The exact instruction sequence from report_1778076700.json id=824, with
    simple state values.  Validates the C path end-to-end."""
    _assert_c_matches_unicorn(
        'id=824 chain (simple state)',
        '66480f6ec066480f6ecb660ffcc1660fefc1660f73f00866480f7ec0',
        _state({'RAX': 0xAB, 'RBX': 0xCD}), 'RAX',
    )


def test_simd_chain_id_824_real_state() -> None:
    """The exact instruction sequence and full register state from
    report_1778076700.json id=824."""
    _assert_c_matches_unicorn(
        'id=824 chain (full state)',
        '66480f6ec066480f6ecb660ffcc1660fefc1660f73f00866480f7ec0',
        _state({
            'RAX': 0x3D75B3CF02C40532,
            'RBX': 0x6D450A09F33CEED,
            'RCX': 0x56CAC07CCFA0DE76,
            'RDX': 0x46B4A1CF43A0520A,
        }),
        'RAX',
    )


# =============================================================================
# 4.  Whole-128-bit ops through the HIGH lane
# =============================================================================
#
# A 16-byte INT_XOR (PXOR over both halves) must XOR the high 8 bytes too.  The
# old uint64_t-per-slot model dropped the high lane; the wide-native cell now
# handles it, so reading XMM0[127:64] after PXOR matches Unicorn.


def test_pxor_full_xmm_hi_lane() -> None:
    """PXOR XMM0, XMM1 XORs both halves: reading the HIGH half must give
    XMM0[127:64] ^ XMM1[127:64], matching Unicorn."""
    _assert_c_matches_unicorn(
        'pxor full lane HI', '660fefc1',
        _state({
            'XMM0[63:0]': 0xAAAA, 'XMM0[127:64]': 0xCCCC,
            'XMM1[63:0]': 0x5555, 'XMM1[127:64]': 0x3333,
        }),
        'XMM0[127:64]',
    )
