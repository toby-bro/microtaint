"""Regression: the carry flag of a >64-bit product needs a floor.

Found by the 2026-09-12 AE soundness campaign: 1 case in 2.4 M, so only deep
sampling surfaces it.  `imul` CF lifts as
``INT_NOTEQUAL(sext(truncated result), full product)`` -- "the product's upper
half is significant".  That predicate is NON-monotone, since a signed product
overflows at BOTH sign extremes, so the 2-corner differential cancels while an
interior value differs.

The floor is gated on a product WIDER than 64 bits AND a single register
value-dep.  Both halves of that gate are asserted here, because both were found
by measurement rather than reasoning:

  * <=32-bit forms multiply into a product the differential handles (0
    under-taints at 46-85% exact across 4,480+ campaign cases each) and must keep
    that exactness;
  * two-dep forms are already covered by the pairwise-avalanche regime -- adding
    the floor left `imul rax, rbx` taint byte-identical while costing 9 -> 23
    circuit nodes.
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


def _run(asm: str, state: dict[str, int],
         taint: dict[str, int]) -> dict[str, int]:
    keystone = pytest.importorskip('keystone')
    ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
    code = bytes(ks.asm(asm)[0])
    rule = generate_static_rule(Architecture.AMD64, code, FMT)
    vals = dict.fromkeys(NAMES, 0)
    vals.update(state)
    vals['RSP'] = 0x204000
    tnt = dict.fromkeys(NAMES, 0)
    tnt.update(taint)
    ctx = EvalContext(input_taint=tnt, input_values=vals,
                      simulator=CellSimulator(Architecture.AMD64),
                      implicit_policy=ImplicitTaintPolicy.IGNORE)
    return {n: int(v or 0) for n, v in rule.evaluate(ctx).items()}


# The recorded campaign witness, not a synthesised state.
_WITNESS_STATE = {'RAX': 0x55B730EF2463D822, 'RBX': 0x497BDEBF3939C8EB,
                  'RCX': 0xBE330AE7FE38F512, 'RDX': 0xB92AC61CD39B34F4,
                  'CF': 1, 'PF': 1, 'ZF': 1, 'SF': 0, 'OF': 0}


def test_imul_immediate_carry_is_tainted() -> None:
    res = _run('imul rax, 3', _WITNESS_STATE, {'RAX': 0xC418000400000100})
    assert res['CF'], (
        'imul CF lost the taint: the product overflows 64 bits, which the 2-corner '
        'differential cannot see, and no floor fired'
    )


@pytest.mark.parametrize('asm', ['imul ax, bx', 'imul eax, ebx', 'mul al', 'mul eax'])
def test_narrow_multiply_keeps_its_exactness(asm: str) -> None:
    """<=32-bit products must NOT be floored: they are sound and precise already.

    A fully-tainted 1-bit flag on every case would mean the floor widened past its
    gate.  Here a single low-bit taint must leave at least one flag clean.
    """
    res = _run(asm, {'RAX': 0x1234, 'RBX': 0x5678, 'RCX': 0, 'RDX': 0,
                     'CF': 0, 'PF': 0, 'ZF': 0, 'SF': 0, 'OF': 0},
               {'RAX': 1})
    assert not all(res.get(f, 0) for f in ('CF', 'OF', 'SF', 'ZF')), (
        f'{asm}: every flag tainted by a single low-bit taint -- the multiply floor '
        'has widened past its >64-bit product gate'
    )


def test_sixteen_bit_imul_carry_is_tainted() -> None:
    """A 16-bit `imul` CF under-taint, which the >64-bit gate above excludes.

    `test_narrow_multiply_keeps_its_exactness` asserts that <=32-bit forms need
    no floor, and its docstring justifies that with campaign evidence: "0
    under-taints at 46-85% exact across 4,480+ campaign cases each".  That
    evidence came from an oracle that could not have produced the counterexample.
    It probed the ground truth by clearing the tainted bits and enumerating from
    that masked base, which destroys the carry structure the overflow depends on.

    Re-running the same corpus against an oracle that flips each tainted bit
    against the REAL state found this in 2 of 320 cases:

        imul ax, 0x7f     RAX = 0xe24b045a3315efbc
        taint RAX = 0xfbbbffffbaeffffd

    Direct execution confirms CF moves when RAX bit 12 is flipped, and the
    engine reports CF clean.  CF for a 16-bit signed multiply says the product
    does not fit in 16 bits, which is exactly the non-monotone predicate the
    module docstring describes, so the width gate is the thing that is wrong,
    not the reasoning behind the floor.

    The values are the campaign witness verbatim rather than a reduced case:
    the predicate is value-conditioned, so shrinking the operand stops
    exercising it.
    """
    res = _run('imul ax, 0x7f',
               {'RAX': 0xE24B045A3315EFBC, 'RBX': 0x6AB00425F1008845,
                'RCX': 0x0A70512BA900B1CA, 'RDX': 0x1432E13251A9AC55,
                'CF': 0, 'PF': 0, 'ZF': 0, 'SF': 0, 'OF': 0},
               {'RAX': 0xFBBBFFFFBAEFFFFD})
    assert res['CF'], (
        'imul CF lost the taint on a 16-bit form: flipping RAX bit 12 changes '
        'CF, so the >64-bit product gate on the floor is too narrow'
    )


# --------------------------------------------------------------------------- #
# mulx: the form that must NOT change when the overflow gate is restructured.
#
# The multiply-overflow floor was gated on "a product wider than 8 bytes", which
# catches `mulx` incidentally.  Replacing that with the structural shape ("a
# 1-bit flag compared against something derived from the product") appears to
# drop `mulx` from the gate, and these pin why that is harmless: `mulx` writes
# NO FLAGS AT ALL, so the floor -- which only ever applies to a 1-bit output --
# could never have fired for it under either gate.
#
# They are written as properties rather than as frozen masks, so they keep
# meaning if the multiply taint is ever made more precise.
# --------------------------------------------------------------------------- #

MULX_FORMS = ['mulx rax, rbx, rcx', 'mulx eax, ebx, ecx']


@pytest.mark.parametrize('asm', MULX_FORMS)
def test_mulx_writes_no_flags(asm: str) -> None:
    """`mulx` is the flag-free multiply; that is its whole point.

    If this ever fails, the overflow floor's gate genuinely does matter for
    `mulx` and restructuring it needs re-examining.
    """
    res = _run(asm, {'RAX': 0x1234, 'RBX': 0, 'RCX': 0x5678, 'RDX': 0xDEAD,
                     'CF': 0, 'PF': 0, 'ZF': 0, 'SF': 0, 'OF': 0},
               {'RDX': 0xFF, 'RCX': 0xFF})
    tainted_flags = {f: res.get(f, 0) for f in ('CF', 'OF', 'SF', 'ZF', 'PF')
                     if res.get(f, 0)}
    assert not tainted_flags, f'{asm} tainted flags {tainted_flags}; mulx sets none'


def test_mulx_keeps_both_halves_sound() -> None:
    """Neither half of the 128-bit product may lose a bit.

    `mulx rax, rbx, rcx` computes RDX * RCX, writing the LOW half to RBX and the
    HIGH half to RAX.  Ground truth is the flip-union over the tainted input
    bits, computed here in Python, so it does not depend on the engine or an
    emulator.
    """
    mask = (1 << 64) - 1

    def truth(rdx: int, rcx: int, t_rdx: int, t_rcx: int) -> tuple[int, int]:
        def f(d: int, c: int) -> int:
            return (d * c) & ((1 << 128) - 1)

        base = f(rdx, rcx)
        lo = hi = 0
        for i in range(64):
            if (t_rdx >> i) & 1:
                d = base ^ f(rdx ^ (1 << i), rcx)
                lo |= d & mask
                hi |= (d >> 64) & mask
            if (t_rcx >> i) & 1:
                d = base ^ f(rdx, rcx ^ (1 << i))
                lo |= d & mask
                hi |= (d >> 64) & mask
        return lo, hi

    # A case whose product genuinely spans both halves, so the HIGH half is not
    # trivially zero and the assertion has something to catch.
    rdx, rcx, t_rdx, t_rcx = 0xDEADBEEF12345678, 0xFEEDFACE87654321, 0xF0, 0x0F
    res = _run('mulx rax, rbx, rcx',
               {'RAX': 0, 'RBX': 0, 'RCX': rcx, 'RDX': rdx,
                'CF': 0, 'PF': 0, 'ZF': 0, 'SF': 0, 'OF': 0},
               {'RDX': t_rdx, 'RCX': t_rcx})
    lo, hi = truth(rdx, rcx, t_rdx, t_rcx)
    assert hi, 'the chosen operands do not reach the high half; pick larger ones'
    assert not (lo & ~res.get('RBX', 0) & mask), (
        f'low half under-tainted: truth {lo:#x}, engine {res.get("RBX", 0):#x}'
    )
    assert not (hi & ~res.get('RAX', 0) & mask), (
        f'high half under-tainted: truth {hi:#x}, engine {res.get("RAX", 0):#x}'
    )
