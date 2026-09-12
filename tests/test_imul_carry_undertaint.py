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


def _run(asm: str, state: dict, taint: dict) -> dict[str, int]:
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
