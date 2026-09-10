"""SSE/XMM data-movement taint: EXACT 128-bit load and store propagation.

Wide memory operands used to break 128-bit XMM tracking:

  * STORE (`movups [mem], xmm`) emitted a single MEM key whose 64-bit mask kept
    only the low 8 bytes -- an UNSOUND under-taint that halved the data flow of
    every glibc memcpy/memmove.
  * LOAD (`movups xmm, [mem]`) read the whole 16 bytes through one mask,
    aliasing the high 8 bytes onto the low 8 -- a sound but imprecise over-taint.

Both are now byte-EXACT: memory byte j carries exactly the taint of register
byte j (and vice versa), with no over- or under-approximation.  These tests
assert the exact tainted-byte SET in each direction and across a full memcpy
roundtrip, including non-uniform (single-half, single-byte) taint.

Tests are written with whole XMM/YMM registers via the debug/test-only
``RegisterAliases`` helper, so they read in architectural terms and do not
depend on the engine's internal lane representation.
"""

import pytest

from microtaint.debug.reg_aliases import RegisterAliases
from microtaint.emulator.shadow import BitPreciseShadowMemory
from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, Register

_A = RegisterAliases(Architecture.AMD64)
_SRC = 0x5000
_DST = 0x6000
_FULL = 0xFFFFFFFFFFFFFFFF
_FULL128 = (1 << 128) - 1

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
    return _A.state_format(['RAX', 'RSI', 'RDI', 'XMM0'])


def _mem_tainted_bytes(out: dict[str, int], base: int, n: int) -> set[int]:
    """Write the MEM_ outputs of a store into a shadow, return tainted byte set."""
    shadow = BitPreciseShadowMemory()
    for key, mask in out.items():
        if key.startswith('MEM_'):
            body = key[4:]
            last = body.rfind('_')
            shadow.write_mask(int(body[:last], 16), mask, int(body[last + 1:]))
    return {i for i in range(n) if shadow.read_mask(base + i, 1)}


def _store_tainted_bytes(opcode_hex: str, in_taint: dict[str, int]) -> set[int]:
    """Store xmm0 (tainted per in_taint) to [RAX]; return the set of destination
    memory byte indices (0..15) that end up tainted."""
    circuit = generate_static_rule(Architecture.AMD64, bytes.fromhex(opcode_hex), _regs())
    out = circuit.evaluate(EvalContext(
        input_values={'RAX': _DST}, input_taint=_A.to_engine(in_taint),
        simulator=CellSimulator(Architecture.AMD64), shadow_memory=BitPreciseShadowMemory(),
    ))
    return _mem_tainted_bytes(out, _DST, 16)


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
    xmm0 = _A.read(out, 'XMM0')
    return {i for i in range(16) if (xmm0 >> (i * 8)) & 0xFF}


@pytest.mark.parametrize(('name', 'opcode'), list(_STORE_OPS.items()))
@pytest.mark.parametrize(('in_taint', 'expected'), [
    ({'XMM0': _FULL128}, set(range(16))),          # full copy
    ({'XMM0[63:0]': _FULL}, set(range(8))),        # low half only
    ({'XMM0[127:64]': _FULL}, set(range(8, 16))),  # high half only
    ({'XMM0[63:0]': 0xFF << 16}, {2}),             # single byte (low half, byte 2)
    ({'XMM0[127:64]': 0xFF << 24}, {11}),          # single byte (high half, byte 3)
    ({}, set()),                                   # clean input clears the store
])
def test_xmm_store_is_byte_exact(name: str, opcode: str, in_taint: dict[str, int], expected: set[int]) -> None:
    got = _store_tainted_bytes(opcode, in_taint)
    assert got == expected, f'{name} taint={in_taint}: dest bytes {sorted(got)} != {sorted(expected)}'


@pytest.mark.parametrize(('name', 'opcode'), list(_LOAD_OPS.items()))
@pytest.mark.parametrize(('tainted_src', 'expected'), [
    (set(range(16)), set(range(16))),        # full
    (set(range(8)), set(range(8))),          # low half only
    (set(range(8, 16)), set(range(8, 16))),  # high half only
    ({10}, {10}),                            # single byte, high half
    ({3}, {3}),                              # single byte, low half
    (set(), set()),                          # clean source
])
def test_xmm_load_is_byte_exact(name: str, opcode: str, tainted_src: set[int], expected: set[int]) -> None:
    got = _load_tainted_bytes(opcode, tainted_src)
    assert got == expected, f'{name} src={sorted(tainted_src)}: xmm bytes {sorted(got)} != {sorted(expected)}'


@pytest.mark.parametrize('tainted_src', [
    set(range(16)),          # whole buffer (attacker data)
    set(range(8, 16)),       # high half only
    {2, 11},                 # one byte in each half
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
    return _A.state_format(['RAX', 'RSI', 'RDI', 'XMM0', 'XMM1'])


def _bitwise_xmm0(opcode_hex: str, in_values: dict[str, int], in_taint: dict[str, int]) -> int:
    """Whole-XMM0 taint (128-bit) after a bitwise op."""
    circuit = generate_static_rule(Architecture.AMD64, bytes.fromhex(opcode_hex), _regs_two_xmm())
    out = circuit.evaluate(EvalContext(
        input_values=_A.to_engine(in_values), input_taint=_A.to_engine(in_taint),
        simulator=CellSimulator(Architecture.AMD64), shadow_memory=BitPreciseShadowMemory(),
    ))
    return _A.read(out, 'XMM0')


@pytest.mark.parametrize(('name', 'opcode'), [
    ('pxor xmm0,xmm1', '660fefc1'),
    ('por xmm0,xmm1', '660febc1'),
    ('movaps xmm0,xmm1', '0f28c1'),
])
def test_sse2_bitwise_taint_stays_in_lane(name: str, opcode: str) -> None:
    """A 128-bit bitwise/move op must keep taint within its 64-bit half: tainting
    one input half must not bleed into the other output half (the pre-fix bug)."""
    # XMM1 high half tainted -> only XMM0 high half tainted.
    assert _bitwise_xmm0(opcode, {}, {'XMM1[127:64]': _FULL}) == (_FULL << 64), f'{name}: high-half bled'
    # XMM1 low half tainted -> only XMM0 low half tainted.
    assert _bitwise_xmm0(opcode, {}, {'XMM1[63:0]': _FULL}) == _FULL, f'{name}: low-half bled'


def test_pand_is_lane_exact_and_value_aware() -> None:
    """pand keeps taint in its 64-bit half AND respects masking: a half ANDed with
    a concrete 0 clears its taint; ANDed with all-ones passes it through."""
    ones = {'XMM0': _FULL128, 'XMM1': _FULL128}
    # mask = all-ones -> taint passes through, in-half only
    assert _bitwise_xmm0('660fdbc1', ones, {'XMM1[127:64]': _FULL}) == (_FULL << 64)
    assert _bitwise_xmm0('660fdbc1', ones, {'XMM1[63:0]': _FULL}) == _FULL
    # the other operand as a concrete zero mask clears the taint (value-aware AND)
    assert _bitwise_xmm0('660fdbc1', {}, {'XMM1[127:64]': _FULL}) == 0


# --- AVX / 256-bit YMM (four 64-bit lanes) ---------------------------------


def _regs_ymm() -> list[Register]:
    return _A.state_format(['RAX', 'RSI', 'RDI', 'YMM0', 'YMM1', 'YMM2'])


def _ymm_load_bytes(opcode_hex: str, tainted_src: set[int]) -> set[int]:
    circuit = generate_static_rule(Architecture.AMD64, bytes.fromhex(opcode_hex), _regs_ymm())
    shadow = BitPreciseShadowMemory()
    for i in tainted_src:
        shadow.write_mask(_SRC + i, 0xFF, 1)
    out = circuit.evaluate(EvalContext(
        input_values={'RAX': _SRC}, input_taint={},
        simulator=CellSimulator(Architecture.AMD64), shadow_memory=shadow))
    ymm0 = _A.read(out, 'YMM0')
    return {i for i in range(32) if (ymm0 >> (i * 8)) & 0xFF}


def _ymm_store_bytes(opcode_hex: str, in_taint: dict[str, int]) -> set[int]:
    circuit = generate_static_rule(Architecture.AMD64, bytes.fromhex(opcode_hex), _regs_ymm())
    out = circuit.evaluate(EvalContext(
        input_values={'RAX': _DST}, input_taint=_A.to_engine(in_taint),
        simulator=CellSimulator(Architecture.AMD64), shadow_memory=BitPreciseShadowMemory()))
    return _mem_tainted_bytes(out, _DST, 32)


@pytest.mark.parametrize(('name', 'opcode'), [
    ('vmovdqu ymm0,[rax]', 'c5fe6f00'),
    ('vmovups ymm0,[rax]', 'c5fc1000'),
])
@pytest.mark.parametrize('rng', [range(8), range(8, 16), range(16, 24), range(24, 32), range(32)])
def test_avx_ymm_load_is_lane_exact(name: str, opcode: str, rng: range) -> None:
    """A 256-bit AVX load must taint exactly the loaded bytes, per 64-bit lane."""
    want = set(rng)
    got = _ymm_load_bytes(opcode, want)
    assert got == want, f'{name} src={sorted(want)}: ymm bytes {sorted(got)} != {sorted(want)}'


@pytest.mark.parametrize(('half', 'expected'), [
    ('YMM0[63:0]', set(range(8))),
    ('YMM0[127:64]', set(range(8, 16))),
    ('YMM0[191:128]', set(range(16, 24))),
    ('YMM0[255:192]', set(range(24, 32))),
])
def test_avx_ymm_store_is_lane_exact(half: str, expected: set[int]) -> None:
    """vmovdqu [rax],ymm0: each tainted 64-bit lane reaches exactly its 8 bytes."""
    got = _ymm_store_bytes('c5fe7f00', {half: _FULL})
    assert got == expected, f'store {half}: dest bytes {sorted(got)} != {sorted(expected)}'


@pytest.mark.parametrize(('src', 'expected'), [
    ('YMM2[191:128]', _FULL << 128),  # ymm2 lane 2 -> ymm0 lane 2
    ('YMM1[127:64]', _FULL << 64),    # ymm1 lane 1 -> ymm0 lane 1
    ('YMM2[63:0]', _FULL),            # ymm2 lane 0 -> ymm0 lane 0
])
def test_avx_vpxor_ymm_is_lane_exact(src: str, expected: int) -> None:
    """vpxor ymm0,ymm1,ymm2 keeps taint in its 64-bit lane across the 256-bit op:
    tainting one source lane taints exactly the same lane of ymm0."""
    circuit = generate_static_rule(Architecture.AMD64, bytes.fromhex('c5f5efc2'), _regs_ymm())
    out = circuit.evaluate(EvalContext(
        input_values={}, input_taint=_A.to_engine({src: _FULL}),
        simulator=CellSimulator(Architecture.AMD64), shadow_memory=BitPreciseShadowMemory()))
    assert _A.read(out, 'YMM0') == expected, f'vpxor {src}: YMM0 taint != {expected:#x}'
