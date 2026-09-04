"""Exact-behaviour tests for the debug/test-only register alias helper.

These pin the human <-> engine name translation so tests written in
architectural terms (XMM0, RAX) keep mapping to the geometry names the engine
actually uses.  The helper is derived from pypcode geometry, so the same table
serves every ISA; a few ARM64 cases guard that generality.
"""

from __future__ import annotations

from microtaint.debug.reg_aliases import RegisterAliases
from microtaint.types import Architecture, Register

_FULL64 = 0xFFFFFFFFFFFFFFFF


# ---------------------------------------------------------------------------
# x86-64: XMM lanes and scalars
# ---------------------------------------------------------------------------


def test_xmm_whole_expands_to_both_lanes() -> None:
    a = RegisterAliases(Architecture.AMD64)
    # XMM0 lives at Sleigh offset 0x1200, size 16 -> lanes 0x1200 and 0x1208.
    assert a.to_engine_names('XMM0') == ['VL_0x1200', 'VL_0x1208']
    # XMM1 at 0x1240 (0x1200 + 1*0x40).
    assert a.to_engine_names('XMM1') == ['VL_0x1240', 'VL_0x1248']


def test_xmm_bit_ranges_select_single_lane() -> None:
    a = RegisterAliases(Architecture.AMD64)
    assert a.to_engine_names('XMM0[63:0]') == ['VL_0x1200']
    assert a.to_engine_names('XMM0[127:64]') == ['VL_0x1208']
    assert a.to_engine_names('XMM2[63:0]') == ['VL_0x1280']


def test_xmm_lo_hi_suffix_aliases() -> None:
    a = RegisterAliases(Architecture.AMD64)
    assert a.to_engine_names('XMM0_LO') == ['VL_0x1200']
    assert a.to_engine_names('XMM0_HI') == ['VL_0x1208']
    assert a.to_engine_names('XMM5_HI') == ['VL_0x1348']


def test_scalar_names_pass_through() -> None:
    a = RegisterAliases(Architecture.AMD64)
    assert a.to_engine_names('RAX') == ['RAX']
    assert a.to_engine_names('RDX') == ['RDX']
    # An engine name we do not know is returned unchanged (pass-through).
    assert a.to_engine_names('VL_0x1240') == ['VL_0x1240']


def test_to_human_name_inverts_lanes() -> None:
    a = RegisterAliases(Architecture.AMD64)
    assert a.to_human_name('VL_0x1200') == 'XMM0[63:0]'
    assert a.to_human_name('VL_0x1208') == 'XMM0[127:64]'
    assert a.to_human_name('VL_0x1240') == 'XMM1[63:0]'
    # Scalars come back canonicalised; unknown keys pass through.
    assert a.to_human_name('RAX') == 'RAX'
    assert a.to_human_name('VL_0xdeadbe') == 'VL_0xdeadbe'


def test_name_roundtrip() -> None:
    a = RegisterAliases(Architecture.AMD64)
    for human in ('XMM0[63:0]', 'XMM3[127:64]', 'XMM7[63:0]', 'XMM15[127:64]'):
        (engine,) = a.to_engine_names(human)
        assert a.to_human_name(engine) == human


# ---------------------------------------------------------------------------
# Value / taint splitting across lanes (little-endian x86)
# ---------------------------------------------------------------------------


def test_to_engine_splits_wide_value_le() -> None:
    a = RegisterAliases(Architecture.AMD64)
    val = 0xAAAAAAAAAAAAAAAA_1111111111111111
    got = a.to_engine({'XMM0': val})
    # LE: low 64 bits go to the low-offset lane, high 64 to the high lane.
    assert got == {'VL_0x1200': 0x1111111111111111, 'VL_0x1208': 0xAAAAAAAAAAAAAAAA}


def test_from_engine_recombines_lanes_le() -> None:
    a = RegisterAliases(Architecture.AMD64)
    lanes = {'VL_0x1200': 0x1111111111111111, 'VL_0x1208': 0xAAAAAAAAAAAAAAAA}
    assert a.from_engine(lanes) == {'XMM0': 0xAAAAAAAAAAAAAAAA_1111111111111111}


def test_to_from_engine_roundtrip() -> None:
    a = RegisterAliases(Architecture.AMD64)
    original = {'RAX': _FULL64, 'XMM1': 0xDEADBEEFCAFEBABE_0123456789ABCDEF}
    assert a.from_engine(a.to_engine(original)) == original


def test_single_lane_taint_is_readable() -> None:
    a = RegisterAliases(Architecture.AMD64)
    # A test wants to taint only XMM1's high lane -- one geometry key.
    assert a.to_engine({'XMM1[127:64]': _FULL64}) == {'VL_0x1248': _FULL64}


# ---------------------------------------------------------------------------
# state_format construction
# ---------------------------------------------------------------------------


def test_state_format_expands_vectors_and_keeps_scalars() -> None:
    a = RegisterAliases(Architecture.AMD64)
    fmt = a.state_format([Register('RAX', 64), 'XMM0', 'XMM1'])
    names = [(r.name, r.bits) for r in fmt]
    assert names == [
        ('RAX', 64),
        ('VL_0x1200', 64),
        ('VL_0x1208', 64),
        ('VL_0x1240', 64),
        ('VL_0x1248', 64),
    ]


def test_state_format_no_wide_registers() -> None:
    a = RegisterAliases(Architecture.AMD64)
    fmt = a.state_format(['XMM0', 'XMM1', 'XMM2', 'RAX', 'RBX'])
    # The mask path is 64-bit; nothing wider than a lane may reach the engine.
    assert all(r.bits <= 64 for r in fmt)


# ---------------------------------------------------------------------------
# ISA generality: the same code names ARM64 NEON lanes with no x86 knowledge
# ---------------------------------------------------------------------------


def test_arm64_q_register_lanes() -> None:
    a = RegisterAliases(Architecture.ARM64)
    lanes = a.to_engine_names('Q0')
    assert len(lanes) == 2 and all(k.startswith('VL_') for k in lanes)
    # Round-trip a lane back to a human name mentioning q0 (Sleigh's own case).
    assert a.to_human_name(lanes[0]).upper().startswith('Q0[')


def test_big_endian_lane_bit_mapping_ppc() -> None:
    # On big-endian PPC the lowest-offset lane holds the MOST significant bits,
    # so a wide value splits mirror-imaged relative to little-endian x86.
    a = RegisterAliases(Architecture.PPC32BE)
    vecs = [n for n in ('vs0', 'VS0') if n.upper() in a._vectors]
    assert vecs, 'expected an AltiVec/VSX vector register'
    val = 0xAAAAAAAAAAAAAAAA_1111111111111111
    engine = a.to_engine({'vs0': val})
    # Exactly two lanes, and recombining returns the original wide value.
    assert len(engine) == 2
    assert a.from_engine(engine) == {a._canonical['VS0']: val}
