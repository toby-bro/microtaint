"""SSE/XMM data-movement taint: EXACT 128-bit load and store propagation.

A 128-bit XMM register is tracked as two synthetic 8-byte halves (`XMM<n>_LO` =
bytes 0-7, `XMM<n>_HI` = bytes 8-15).  Wide memory operands used to break this:

  * STORE (`movups [mem], xmm`) emitted a single `MEM_<addr>_16` key whose taint
    mask (a 64-bit shadow value) kept only the low 8 bytes -- an UNSOUND
    under-taint that halved the data flow of every glibc memcpy/memmove.
  * LOAD (`movups xmm, [mem]`) read the whole 16 bytes through one mask, aliasing
    the high 8 bytes onto the low 8 -- a sound but imprecise over-taint.

Both are now byte-EXACT: a wide store/load is split into 8-byte lanes, each wired
to the matching register half, so memory byte j carries exactly the taint of
register byte j (and vice versa) with no over- or under-approximation.  These
tests assert the exact tainted-byte SET in each direction and across a full
memcpy roundtrip, including non-uniform (single-half, single-byte) taint.
"""

import pytest

from microtaint.emulator.shadow import BitPreciseShadowMemory
from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, Register

_SRC = 0x5000
_DST = 0x6000
_FULL = 0xFFFFFFFFFFFFFFFF

_STORE_OPS = {
    'movups [rax],xmm0': '0f1100',
    'movaps [rax],xmm0': '0f2900',
    'movdqu [rax],xmm0': 'f30f7f00',
    'movdqa [rax],xmm0': '660f7f00',
}
_LOAD_OPS = {
    'movups xmm0,[rax]': '0f1000',
    'movaps xmm0,[rax]': '0f2800',
    'movdqu xmm0,[rax]': 'f30f6f00',
    'movdqa xmm0,[rax]': '660f6f00',
}


def _regs() -> list[Register]:
    # XMM0 must be in the state format (as it is in production X64_FORMAT) to be
    # tracked at all; it is split into two 64-bit halves.
    return [Register(name=n, bits=64) for n in ('RAX', 'RSI', 'RDI')] + [
        Register(name='XMM0_LO', bits=64),
        Register(name='XMM0_HI', bits=64),
    ]


def _store_tainted_bytes(opcode_hex: str, in_taint: dict[str, int]) -> set[int]:
    """Store xmm0 (tainted per in_taint) to [RAX]; return the set of destination
    memory byte indices (0..15) that end up tainted."""
    circuit = generate_static_rule(Architecture.AMD64, bytes.fromhex(opcode_hex), _regs())
    shadow = BitPreciseShadowMemory()
    out = circuit.evaluate(EvalContext(
        input_values={'RAX': _DST}, input_taint=in_taint,
        simulator=CellSimulator(Architecture.AMD64), shadow_memory=shadow,
    ))
    for key, mask in out.items():
        if key.startswith('MEM_'):
            body = key[4:]
            last = body.rfind('_')
            shadow.write_mask(int(body[:last], 16), mask, int(body[last + 1:]))
    return {i for i in range(16) if shadow.read_mask(_DST + i, 1)}


def _load_tainted_bytes(opcode_hex: str, tainted_src: set[int]) -> set[int]:
    """Load [RAX] (source bytes in tainted_src are tainted) into xmm0; return the
    set of XMM0 byte indices (0..15) that end up tainted."""
    circuit = generate_static_rule(Architecture.AMD64, bytes.fromhex(opcode_hex), _regs())
    shadow = BitPreciseShadowMemory()
    for i in tainted_src:
        shadow.write_mask(_SRC + i, 0xFF, 1)
    out = circuit.evaluate(EvalContext(
        input_values={'RAX': _SRC}, input_taint={},
        simulator=CellSimulator(Architecture.AMD64), shadow_memory=shadow,
    ))
    lo, hi = out.get('XMM0_LO', 0), out.get('XMM0_HI', 0)
    return (
        {i for i in range(8) if (lo >> (i * 8)) & 0xFF}
        | {8 + i for i in range(8) if (hi >> (i * 8)) & 0xFF}
    )


@pytest.mark.parametrize(('name', 'opcode'), list(_STORE_OPS.items()))
@pytest.mark.parametrize(('in_taint', 'expected'), [
    ({'XMM0_LO': _FULL, 'XMM0_HI': _FULL}, set(range(16))),   # full copy
    ({'XMM0_LO': _FULL}, set(range(8))),                   # low half only
    ({'XMM0_HI': _FULL}, set(range(8, 16))),                  # high half only
    ({'XMM0_LO': 0xFF << 16}, {2}),                           # single byte (lane 0, byte 2)
    ({'XMM0_HI': 0xFF << 24}, {11}),                          # single byte (lane 1, byte 3)
    ({}, set()),                                              # clean input clears the store
])
def test_xmm_store_is_byte_exact(name: str, opcode: str, in_taint: dict[str, int], expected: set[int]) -> None:
    got = _store_tainted_bytes(opcode, in_taint)
    assert got == expected, f'{name} taint={in_taint}: dest bytes {sorted(got)} != {sorted(expected)}'


@pytest.mark.parametrize(('name', 'opcode'), list(_LOAD_OPS.items()))
@pytest.mark.parametrize(('tainted_src', 'expected'), [
    (set(range(16)), set(range(16))),        # full
    (set(range(8)), set(range(8))),    # low half only
    (set(range(8, 16)), set(range(8, 16))),  # high half only
    ({10}, {10}),                            # single byte, high lane
    ({3}, {3}),                              # single byte, low lane
    (set(), set()),                          # clean source
])
def test_xmm_load_is_byte_exact(name: str, opcode: str, tainted_src: set[int], expected: set[int]) -> None:
    got = _load_tainted_bytes(opcode, tainted_src)
    assert got == expected, f'{name} src={sorted(tainted_src)}: xmm bytes {sorted(got)} != {sorted(expected)}'


@pytest.mark.parametrize('tainted_src', [
    set(range(16)),          # whole buffer (attacker data)
    set(range(8, 16)),       # high half only
    {2, 11},                 # one byte in each lane
    {5},                     # single byte
])
def test_sse_memcpy_roundtrip_is_byte_exact(tainted_src: set[int]) -> None:
    """load [rsi] -> xmm0 -> store [rdi]: exactly the tainted source bytes reach
    the destination (the glibc memcpy/memmove data flow), no more, no less."""
    regs = _regs()
    sim = CellSimulator(Architecture.AMD64)
    shadow = BitPreciseShadowMemory()
    for i in tainted_src:
        shadow.write_mask(_SRC + i, 0xFF, 1)
    reg_taint: dict[str, int] = {}

    def _apply(out: dict[str, int]) -> None:
        for key, mask in out.items():
            if key.startswith('MEM_'):
                body = key[4:]
                last = body.rfind('_')
                shadow.write_mask(int(body[:last], 16), mask, int(body[last + 1:]))
            else:
                reg_taint[key] = mask

    ld = generate_static_rule(Architecture.AMD64, bytes.fromhex('0f1006'), regs)  # movups xmm0,[rsi]
    _apply(ld.evaluate(EvalContext(
        input_values={'RSI': _SRC}, input_taint=dict(reg_taint),
        simulator=sim, shadow_memory=shadow)))

    st = generate_static_rule(Architecture.AMD64, bytes.fromhex('0f1107'), regs)  # movups [rdi],xmm0
    _apply(st.evaluate(EvalContext(
        input_values={'RDI': _DST}, input_taint=dict(reg_taint),
        simulator=sim, shadow_memory=shadow)))

    got = {i for i in range(16) if shadow.read_mask(_DST + i, 1)}
    assert got == tainted_src, f'roundtrip: dest bytes {sorted(got)} != source {sorted(tainted_src)}'


def _regs_two_xmm() -> list[Register]:
    return [Register(name=n, bits=64) for n in ('RAX', 'RSI', 'RDI')] + [
        Register(name='XMM0_LO', bits=64), Register(name='XMM0_HI', bits=64),
        Register(name='XMM1_LO', bits=64), Register(name='XMM1_HI', bits=64),
    ]


def _bitwise_out(opcode_hex: str, in_values: dict[str, int], in_taint: dict[str, int]) -> tuple[int, int]:
    circuit = generate_static_rule(Architecture.AMD64, bytes.fromhex(opcode_hex), _regs_two_xmm())
    out = circuit.evaluate(EvalContext(
        input_values=in_values, input_taint=in_taint,
        simulator=CellSimulator(Architecture.AMD64), shadow_memory=BitPreciseShadowMemory(),
    ))
    return out.get('XMM0_LO', 0), out.get('XMM0_HI', 0)


@pytest.mark.parametrize(('name', 'opcode'), [
    ('pxor xmm0,xmm1', '660fefc1'),
    ('por xmm0,xmm1', '660febc1'),
    ('movaps xmm0,xmm1', '0f28c1'),
])
def test_sse2_bitwise_taint_stays_in_lane(name: str, opcode: str) -> None:
    """A 128-bit bitwise/move op must keep taint within its 64-bit lane: tainting
    one input half must not bleed into the other output half (the pre-fix bug)."""
    lo, hi = _bitwise_out(opcode, {}, {'XMM1_HI': _FULL})
    assert (lo, hi) == (0, _FULL), f'{name}: XMM1_HI taint -> (LO={lo:#x}, HI={hi:#x}), expected only HI'
    lo, hi = _bitwise_out(opcode, {}, {'XMM1_LO': _FULL})
    assert (lo, hi) == (_FULL, 0), f'{name}: XMM1_LO taint -> (LO={lo:#x}, HI={hi:#x}), expected only LO'


def test_pand_is_lane_exact_and_value_aware() -> None:
    """pand keeps taint in-lane AND respects masking: a lane ANDed with a concrete
    0 clears its taint; ANDed with all-ones passes it through, in that lane only."""
    ones = {'XMM0_LO': _FULL, 'XMM0_HI': _FULL, 'XMM1_LO': _FULL, 'XMM1_HI': _FULL}
    # mask = all-ones -> taint passes through, in-lane only
    assert _bitwise_out('660fdbc1', ones, {'XMM1_HI': _FULL}) == (0, _FULL)
    assert _bitwise_out('660fdbc1', ones, {'XMM1_LO': _FULL}) == (_FULL, 0)
    # the other operand as a concrete zero mask clears the taint (value-aware AND)
    assert _bitwise_out('660fdbc1', {}, {'XMM1_HI': _FULL}) == (0, 0)
