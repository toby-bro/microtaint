"""`lea r, [base + index*scale + disp]` with DISTINCT base and index under-taints.

`tests/test_lea_scale_taint.py` already covers `lea rax, [rbx + rbx*4]`, where
base and index are the SAME register. That fix did not reach the two-register
form, and the address mode most compilers actually emit, `[base + index*scale]`
with distinct registers, still drops carry bits.

Minimal case, small enough to check by hand and needing no oracle:

    lea rax, [rbx + rcx*4 + 8]    RBX = 0, RCX = 1
    taint RBX = 0xc (bits 2,3), RCX = 0x2 (bit 1)

    base                 = 0 + 1*4 + 8 = 12 = 0b01100
    flip RBX bit 2 -> 4  : 4 + 4 + 8   = 16 = 0b10000   differs in bits 2,3,4
    flip RBX bit 3 -> 8  : 8 + 4 + 8   = 20 = 0b10100   differs in bits 3,4
    flip RCX bit 1 -> 3  : 0 + 12 + 8  = 20 = 0b10100   differs in bits 3,4

so a single tainted bit can move RAX bits 2, 3 and 4, and the truth is 0x1c.
The engine answers 0x2e: it sets bits 1 and 5, which no single flip can move,
and leaves **bit 4 clear**, which three of them do. Bit 4 is a silent
under-taint.

WHERE IT COMES FROM. The lifted p-code is exactly

    unique = RCX * 0x4
    unique = RBX + unique
    RAX    = unique

and the emitted taint expression is a differential OR'd with floors:

    OR( XOR(cell_hi, cell_lo),          the two-corner differential
        T_RCX, T_RBX,                   <-- the carry floor, UNSCALED
        AND(LEFT(T_RCX, 2), ~0), T_RBX) the routing term, correctly scaled

The routing term shifts the index taint by log2(scale), which is right. The
CARRY FLOOR does not: it unions the raw source-register taints at their original
bit positions. For an index that is scaled before the addition those positions
are simply wrong, so the floor covers bits the index cannot reach (the spurious
bits 1 and 5 above) while failing to cover the positions where the carry really
propagates (bit 4). Adding a base is what pulls the floor in: `lea rax, [rcx*4]`
alone has no addition, no floor, and is exact.

The two controls below pin that diagnosis and stop a fix from "passing" by
over-tainting everything: the no-base and no-scale forms are exact today and
must stay exact.
"""
from __future__ import annotations

import pytest

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, ImplicitTaintPolicy, Register

NAMES = ['RAX', 'RBX', 'RCX', 'RDX', 'CF', 'PF', 'ZF', 'SF', 'OF']
FMT = ([Register(n, 64) for n in NAMES[:4]]
       + [Register(f, 1) for f in NAMES[4:]] + [Register('RSP', 64)])
MASK = (1 << 64) - 1


def _engine(asm: str, state: dict[str, int], taint: dict[str, int]) -> dict[str, int]:
    keystone = pytest.importorskip('keystone')
    ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
    rule = generate_static_rule(Architecture.AMD64, bytes(ks.asm(asm)[0]), FMT)
    ctx = EvalContext(input_taint=dict(taint), input_values=dict(state),
                      simulator=CellSimulator(Architecture.AMD64),
                      implicit_policy=ImplicitTaintPolicy.IGNORE)
    return rule.evaluate(ctx)


def _state(**regs: int) -> tuple[dict[str, int], dict[str, int]]:
    state = {'RAX': 0, 'RBX': 0, 'RCX': 0, 'RDX': 0, 'RSP': 0x204000}
    taint = {'RAX': 0, 'RBX': 0, 'RCX': 0, 'RDX': 0}
    for f in NAMES[4:]:
        state[f] = 0
        taint[f] = 0
    for k, v in regs.items():
        if k.startswith('t_'):
            taint[k[2:].upper()] = v
        else:
            state[k.upper()] = v
    return state, taint


def _truth(rbx: int, rcx: int, t_rbx: int, t_rcx: int,
           scale: int, disp: int) -> int:
    """Union of what flipping ONE tainted input bit can move in RAX.

    Pure arithmetic, so the expected value does not depend on an emulator or on
    the engine being tested.
    """
    def f(b: int, c: int) -> int:
        return (b + c * scale + disp) & MASK

    base = f(rbx, rcx)
    moved = 0
    for i in range(64):
        if (t_rbx >> i) & 1:
            moved |= base ^ f(rbx ^ (1 << i), rcx)
        if (t_rcx >> i) & 1:
            moved |= base ^ f(rbx, rcx ^ (1 << i))
    return moved & MASK


#: (asm, RBX, RCX, taint RBX, taint RCX, has base, scale, displacement)
#: Every row is a witness that under-tainted the engine before the waist fix,
#: found by searching small operand values so the arithmetic can be checked by
#: hand.  The truth is recomputed here rather than hardcoded, so a row cannot
#: silently drift into asserting whatever the engine happens to answer.
CASES = [
    ('lea rax, [rcx*4 + 8]', 0x1, 0x5, 0x0, 0xA, False, 4, 8),
    ('lea rax, [rcx*8 - 1]', 0x0, 0x1, 0x0, 0x9, False, 8, -1),
    ('lea rax, [rbx + rcx*2 + 8]', 0x0, 0x4, 0x1D, 0xF, True, 2, 8),
    ('lea rax, [rbx + rcx*4 + 8]', 0x0, 0x7, 0x2C, 0xA, True, 4, 8),
    ('lea rax, [rbx + rcx*4 - 8]', 0x0, 0x1, 0x2C, 0xC, True, 4, -8),
    # base == index: `b + (b << 2)` reads RBX on both sides.  Covered by
    # test_lea_scale_taint.py at 64-bit positions; kept here at small values so
    # the whole family is asserted in one place.
    ('lea rax, [rbx + rbx*4]', 0x1, 0x2, 0x48, 0x0, True, 4, 0),
]


@pytest.mark.parametrize(('asm', 'rbx', 'rcx', 't_rbx', 't_rcx', 'has_base', 'scale', 'disp'),
                         CASES, ids=[c[0] for c in CASES])
def test_address_arithmetic_keeps_its_carry(
    asm: str, rbx: int, rcx: int, t_rbx: int, t_rcx: int,
    has_base: bool, scale: int, disp: int,
) -> None:
    """No address form may drop a bit a single tainted-bit flip can move."""
    state, taint = _state(rbx=rbx, rcx=rcx, t_rbx=t_rbx, t_rcx=t_rcx)
    expected = _truth(rbx if has_base else 0, rcx, t_rbx if has_base else 0,
                      t_rcx, scale=scale, disp=disp)
    got = _engine(asm, state, taint).get('RAX', 0)
    missing = expected & ~got & MASK
    assert not missing, (
        f'{asm}: under-tainted RAX {missing:#x}; a single tainted-bit flip moves '
        f'{expected:#x} and the engine reports {got:#x}'
    )


@pytest.mark.parametrize(('asm', 'scale', 'disp'), [
    ('lea rax, [rcx*4]', 4, 0),      # no addition at all, so no floor is owed
    ('lea rax, [rbx + rcx]', 1, 0),  # addition, but the index is not scaled
])
def test_exact_forms_stay_exact(asm: str, scale: int, disp: int) -> None:
    """Forms that are EXACT today must not be "fixed" by over-tainting.

    A floor that simply unions more bits would make the regressions above pass
    while destroying these, so they are asserted for EQUALITY.  Neither form has
    a waist (the first has no arithmetic, the second no shift), so the fix must
    not touch them at all.
    """
    state, taint = _state(rbx=0x0, rcx=0x3, t_rcx=0x3)
    expected = _truth(0x0, 0x3, 0x0, 0x3, scale=scale, disp=disp)
    got = _engine(asm, state, taint).get('RAX', 0)
    assert got == expected, f'{asm}: expected {expected:#x}, engine {got:#x}'


def test_scaled_index_without_displacement_is_sound() -> None:
    """`lea rax, [rbx + rcx*4]` is sound today, and over-taints.

    Asserted for SOUNDNESS only, not equality.  The engine answers 0xf where the
    truth is 0xc: bits 0 and 1 are the raw index taint unioned at its UNSCALED
    positions, which RCX*4 can never reach.  That over-taint is the same
    misplaced floor as the under-taints above, seen from the other side, so this
    documents it rather than freezing it.  A carry floor is an over-approximation
    by nature, so the assertion that must hold is the soundness one.
    """
    state, taint = _state(rbx=0x0, rcx=0x3, t_rcx=0x3)
    expected = _truth(0x0, 0x3, 0x0, 0x3, scale=4, disp=0)
    got = _engine('lea rax, [rbx + rcx*4]', state, taint).get('RAX', 0)
    assert not (expected & ~got & MASK), (
        f'under-tainted: truth {expected:#x}, engine {got:#x}'
    )


def test_rotate_is_not_split_by_the_waist() -> None:
    """`ror` must keep whole-slice ownership.

    The waist rule relaxed in this fix drops the requirement that the DOWNSTREAM
    read an architectural register.  SLEIGH lifts a rotate's complementary shift
    amount as `INT_SUB(64, k)`, an arithmetic op reading no register, and the
    requirement existed to stop that being mistaken for a fused pair.  The
    protection now rests on the UPSTREAM register read instead, so this asserts
    the rotate is still not split.
    """
    pypcode = pytest.importorskip('pypcode')
    keystone = pytest.importorskip('keystone')
    from microtaint.sleigh.partition import find_waist

    ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
    ctx = pypcode.Context('x86:LE:64:default')
    code = bytes(ks.asm('ror rax, 7', 0x1000)[0])
    ops = [o for o in ctx.translate(code, 0x1000).ops if o.opcode.name != 'IMARK']
    waist = find_waist(ops, ops[-1].output, require_distinct_algebra=True,
                       require_disjoint_inputs=False)
    assert waist is None, f'rotate was split at {waist}'
