"""A base+index memory operand must resolve to base + index*scale + disp.

`resolve_ptr_with_offset` returns a single (register, constant) pair.  For an
address built from TWO registers it returned the left one and silently dropped
the right:

    if lreg is not None:
        return lreg, loff + roff        # rreg discarded

So `[rdi + rax*1]` resolved to RDI alone, and the shadow was read at the base
instead of the element.  Measured on a guest that copies a tainted buffer into an
mmap'd page: gcc compiled the loop with a biased pointer (`sub rdi, rax`, then
`movzx ecx, [rdi+rax*1]`), the load read RDI -- which is buf-base, an address
holding nothing -- and the copy came out clean.

Same instruction, same final address, same shadow; only the split differs:

    rdi=buf,      rax=0      -> RCX taint 0xff
    rdi=buf-base, rax=base   -> RCX taint 0x00

It stays invisible whenever the index register is TAINTED, because the pointer
avalanche then covers the load: that is base64's table lookup, which over-taints
and so is sound.  The silent under-taint needs an untainted, non-zero index --
an ordinary array walk.
"""
from __future__ import annotations

import pytest

from microtaint.emulator.shadow import BitPreciseShadowMemory
from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, ImplicitTaintPolicy, Register

keystone = pytest.importorskip('keystone')

NAMES = ['RAX', 'RCX', 'RDX', 'RDI', 'CF', 'PF', 'ZF', 'SF', 'OF']
FMT = ([Register(n, 64) for n in NAMES[:4]]
       + [Register(f, 1) for f in NAMES[4:]] + [Register('RSP', 64)])
SIM = CellSimulator(Architecture.AMD64)

#: Where the taint actually lives.  High, like a real stack, to keep the test
#: honest about 64-bit addresses.
TAINTED = 0x0000_8000_0000_DE48


def _load_taint(asm: str, regs: dict[str, int], *, at: int = TAINTED) -> int:
    """RCX taint after `asm`, with one tainted byte in the shadow at `at`."""
    ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
    rule = generate_static_rule(Architecture.AMD64, bytes(ks.asm(asm)[0]), FMT)
    sh = BitPreciseShadowMemory()
    sh.write_mask(at, 0xFF, 1)
    vals = dict.fromkeys(NAMES, 0)
    vals['RSP'] = 0x7FFF_FFF0_0000
    vals.update(regs)
    ctx = EvalContext(input_taint=dict.fromkeys(NAMES, 0), input_values=vals,
                      simulator=SIM, implicit_policy=ImplicitTaintPolicy.IGNORE,
                      shadow_memory=sh)
    return int(rule.evaluate(ctx).get('RCX', 0) or 0)


def test_single_register_base_still_works() -> None:
    """The control: one register, no index."""
    assert _load_taint('movzx ecx, byte ptr [rdi]', {'RDI': TAINTED}) == 0xFF


def test_base_plus_displacement_still_works() -> None:
    """The control: one register plus a constant."""
    assert _load_taint('movzx ecx, byte ptr [rdi+0x10]', {'RDI': TAINTED - 0x10}) == 0xFF


@pytest.mark.parametrize('scale', [1, 2, 4, 8])
def test_base_plus_scaled_index(scale: int) -> None:
    """base + index*scale must read base + index*scale, not base."""
    index = 6
    base = TAINTED - index * scale
    got = _load_taint(f'movzx ecx, byte ptr [rdi+rax*{scale}]',
                      {'RDI': base, 'RAX': index})
    assert got == 0xFF, (
        f'[rdi+rax*{scale}] read the wrong address: the index contributes '
        f'{index * scale:#x} and was dropped, so the shadow was read at the base'
    )


def test_base_plus_index_plus_displacement() -> None:
    """The full form: base + index*scale + disp."""
    got = _load_taint('mov ecx, dword ptr [rdi+rax*4+0x8]',
                      {'RDI': TAINTED - 0x18, 'RAX': 4}, at=TAINTED)
    assert got & 0xFF, 'base + index*4 + 0x8 did not reach the tainted dword'


def test_the_split_between_base_and_index_does_not_matter() -> None:
    """The minimal reproduction: same address, two different register splits.

    This is the shape gcc emitted for the mmap guest, via `sub rdi, rax`.
    """
    direct = _load_taint('movzx ecx, byte ptr [rdi+rax*1]',
                         {'RDI': TAINTED, 'RAX': 0})
    biased = _load_taint('movzx ecx, byte ptr [rdi+rax*1]',
                         {'RDI': TAINTED - 0x7FFF_B7DD_6000, 'RAX': 0x7FFF_B7DD_6000})
    assert direct == biased == 0xFF, (
        f'the same address resolved differently depending on how it was split '
        f'across the two registers: direct={direct:#x} biased={biased:#x}'
    )


def test_a_tainted_index_still_avalanches() -> None:
    """Guard: a tainted index must keep its avalanche.

    The fix must not turn the over-approximating (sound) case into a precise
    read at one address -- the engine does not know which element is read.
    """
    ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
    rule = generate_static_rule(
        Architecture.AMD64, bytes(ks.asm('movzx ecx, byte ptr [rdi+rax*1]')[0]), FMT)
    sh = BitPreciseShadowMemory()
    vals = dict.fromkeys(NAMES, 0)
    vals.update({'RDI': TAINTED, 'RAX': 0, 'RSP': 0x7FFF_FFF0_0000})
    tnt = dict.fromkeys(NAMES, 0)
    tnt['RAX'] = 0xFF                      # the INDEX is secret
    ctx = EvalContext(input_taint=tnt, input_values=vals, simulator=SIM,
                      implicit_policy=ImplicitTaintPolicy.IGNORE, shadow_memory=sh)
    assert int(rule.evaluate(ctx).get('RCX', 0) or 0) != 0, (
        'a load through a tainted index must stay tainted: the engine cannot '
        'know which element was read'
    )


# ---------------------------------------------------------------------------
# Broad sweep: the address split must never change the answer, and an indexed
# operand must never be LESS tainted than the same address without an index.
# ---------------------------------------------------------------------------
#
# The cell-key string protocol (``MEM_<reg>_<off>_<size>``, parsed back as
# ``reg + off`` by cell.pyx) cannot express an index term, so an indexed operand
# does not round-trip through the masked-replica path and falls back to the
# soundness floor: `and ecx,[rdi]` is exact (0x1234) where `and ecx,[rdi+rax*4]`
# is full-width.  That is an over-approximation, which is sound.  These tests pin
# the two properties that must hold whatever the engine does internally, so a
# future change to either the key format or the floor cannot silently reintroduce
# the under-taint.

_SWEEP_REG_MEM = [
    f'{op} {reg}, [rdi{{IDX}}]'
    for op in ('mov', 'add', 'sub', 'and', 'or', 'xor', 'cmp', 'test', 'adc', 'sbb')
    for reg in ('ecx', 'cx', 'cl', 'rcx')
]
_SWEEP_MEM_IMM = [
    f'{op} {pfx} ptr [rdi{{IDX}}], 5'
    for op in ('add', 'sub', 'and', 'or', 'xor', 'cmp', 'shl', 'shr', 'sar')
    for pfx in ('dword', 'qword', 'byte')
]
_SWEEP_MEM_UNARY = [
    f'{op} {pfx} ptr [rdi{{IDX}}]'
    for op in ('inc', 'dec', 'neg', 'not')
    for pfx in ('dword', 'qword', 'byte')
]
_SWEEP_EXT = [
    f'{op} ecx, {pfx} ptr [rdi{{IDX}}]'
    for op in ('movzx', 'movsx')
    for pfx in ('byte', 'word')
] + ['imul ecx, dword ptr [rdi{IDX}]']

_SWEEP = [
    (tmpl, scale)
    for tmpl in _SWEEP_REG_MEM + _SWEEP_MEM_IMM + _SWEEP_MEM_UNARY + _SWEEP_EXT
    for scale in (1, 4)
]


def _sweep_taint(asm: str, regs: dict[str, int]) -> dict[str, int]:
    """Every non-zero output taint, with 8 tainted bytes in the shadow at TAINTED.

    Memory outputs collapse to the key ``MEM``: the RMW forms name their output
    by RESOLVED address, so the three readings agree on the name only if the
    address resolved identically, which is the thing under test elsewhere.
    """
    ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
    rule = generate_static_rule(Architecture.AMD64, bytes(ks.asm(asm)[0]), FMT)
    sh = BitPreciseShadowMemory()
    sh.write_mask(TAINTED, 0xFFFFFFFFFFFFFFFF, 8)
    vals = dict.fromkeys(NAMES, 0)
    vals['RSP'] = 0x7FFF_FFF0_0000
    vals['RCX'] = 0x1234
    vals['RDX'] = 3
    vals.update(regs)
    ctx = EvalContext(input_taint=dict.fromkeys(NAMES, 0), input_values=vals,
                      simulator=SIM, implicit_policy=ImplicitTaintPolicy.IGNORE,
                      shadow_memory=sh)
    out = {}
    for k, v in rule.evaluate(ctx).items():
        iv = int(v or 0)
        if iv:
            out['MEM' if k.startswith('MEM_') else k] = iv
    return out


def _covers(wide: dict[str, int], narrow: dict[str, int]) -> bool:
    """`wide` taints at least every bit `narrow` does."""
    return all((wide.get(k, 0) & v) == v for k, v in narrow.items())


@pytest.mark.parametrize(('tmpl', 'scale'), _SWEEP,
                         ids=[f'{t.replace("{IDX}", "")}-s{s}' for t, s in _SWEEP])
def test_indexed_operand_is_split_invariant_and_sound(tmpl: str, scale: int) -> None:
    """Three readings of the SAME effective address must agree, or over-taint.

    P  ``[rdi]``       rdi=T             the reference: no index at all
    A  ``[rdi+rax*s]`` rdi=T,     rax=0  indexed, the base carries the address
    B  ``[rdi+rax*s]`` rdi=T-s*k, rax=k  indexed, the index carries it

    A must equal B: an answer that depends on how the address was split across
    the two registers is the original bug.  A and B must each cover P: dropping
    to a floor is allowed, losing a bit P had is an under-taint.
    """
    k = 7
    plain = tmpl.replace('{IDX}', '')
    indexed = tmpl.replace('{IDX}', f' + rax*{scale}')

    p = _sweep_taint(plain, {'RDI': TAINTED})
    a = _sweep_taint(indexed, {'RDI': TAINTED, 'RAX': 0})
    b = _sweep_taint(indexed, {'RDI': TAINTED - scale * k, 'RAX': k})

    assert p or a or b, f'{plain!r} produced no taint at all: the check is vacuous'
    assert a == b, (
        f'{indexed!r} answered differently depending on how the address was '
        f'split between base and index:\n  base-carries : {a}\n  index-carries: {b}'
    )
    assert _covers(a, p), (
        f'{indexed!r} under-tainted against the same address without an index, '
        f'with the BASE carrying it:\n  no index: {p}\n  indexed : {a}'
    )
    assert _covers(b, p), (
        f'{indexed!r} under-tainted against the same address without an index, '
        f'with the INDEX carrying it:\n  no index: {p}\n  indexed : {b}'
    )


# ---------------------------------------------------------------------------
# An indexed operand must behave like the same address written any other way.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('tmpl', [
    'and ecx, {M}',
    'add ecx, {M}',
    'xor ecx, {M}',
    'cmp ecx, {M}',
    'mov ecx, {M}',
    'movzx ecx, byte ptr {M}',
    'add dword ptr {M}, 5',
    'inc dword ptr {M}',
])
def test_indexed_matches_the_equivalent_displacement(tmpl: str) -> None:
    """`[rdi+rax*4]` with rax=k must answer exactly as `[rdi+4k]` does.

    Both are a base register plus address arithmetic reaching the same byte, so
    any difference between them is the addressing MODE leaking into the answer
    rather than the address.  That leak is what made `resolve_ptr_with_offset`
    drop the index, and it is what made is_mapped_permutation call
    `and ecx,[rdi+rax*4]` a permutation of RCX while calling `and ecx,[rdi]`
    two sources: with the pointer sliced away as a `unique`, the LOAD counted
    for nothing.
    """
    k, scale = 4, 4
    disp = k * scale
    indexed = tmpl.replace('{M}', f'[rdi+rax*{scale}]')
    displaced = tmpl.replace('{M}', f'[rdi+{disp}]')

    a = _sweep_taint(indexed, {'RDI': TAINTED - disp, 'RAX': k})
    b = _sweep_taint(displaced, {'RDI': TAINTED - disp})

    assert a or b, f'{indexed!r} produced no taint at all: the check is vacuous'
    assert a == b, (
        f'the addressing mode changed the answer for the same address:\n'
        f'  {indexed:<28} {a}\n  {displaced:<28} {b}'
    )
