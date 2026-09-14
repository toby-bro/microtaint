"""Regression: signed overflow needs a floor when a barrel shift feeds the ALU.

Found by the 2026-09-12 AE campaign, and only by the run against the FIXED
engine: 1 case in 2.4 M ARM64 cases, after the baseline had gone 38 M cases
without hitting it.
"""
from __future__ import annotations

from typing import TypedDict

import pytest

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, ImplicitTaintPolicy, Register

NAMES = ['x0', 'x1', 'x2', 'x3', 'NG', 'ZR', 'CY', 'OV']
FMT = ([Register(n, 64) for n in NAMES[:4]]
       + [Register(f, 1) for f in NAMES[4:]] + [Register('sp', 64)])


def _run(asm: str, state: dict[str, int],
         taint: dict[str, int]) -> dict[str, int]:
    keystone = pytest.importorskip('keystone')
    ks = keystone.Ks(keystone.KS_ARCH_ARM64, keystone.KS_MODE_LITTLE_ENDIAN)
    code = bytes(ks.asm(asm)[0])
    rule = generate_static_rule(Architecture.ARM64, code, FMT)
    vals = dict.fromkeys(NAMES, 0)
    vals.update(state)
    vals['sp'] = 0x204000
    tnt = dict.fromkeys(NAMES, 0)
    tnt.update(taint)
    ctx = EvalContext(input_taint=tnt, input_values=vals,
                      simulator=CellSimulator(Architecture.ARM64),
                      implicit_policy=ImplicitTaintPolicy.IGNORE)
    return {n: int(v or 0) for n, v in rule.evaluate(ctx).items()}

# The ARM64 fused-shift overflow, found on the FIXED engine by the parallel
# verification run (1 case in 2.4 M).  `adds x0, x1, x2, asr #n` feeds the barrel
# shifter into the ALU: the exact SignedOverflowTaintExpr declines on the shifted
# operand and the sign floor is wide-output-only, so OV fell to the 2-corner
# differential alone -- which non-monotone signed overflow escapes.
#
# This is the half of e980e36 that restoring only its imul gate left behind.
class Witness(TypedDict):
    """One counter-example: the instruction, the register state it ran from,
    and which input bits were tainted.

    Declared rather than left as a bare dict literal because the values are
    heterogeneous -- a string and two int maps -- so the inferred value type
    collapses to something that cannot be passed on without a complaint at
    every use site.
    """

    asm: str
    state: dict[str, int]
    taint: dict[str, int]

_ADDS_ASR_WITNESS: Witness = {
    'asm': 'adds x0, x1, x2, asr #5',
    'state': {'x0': 0x4CFCDD1CDDAE9408, 'x1': 0xFF9BDE34328EBE2A,
              'x2': 0x1D09DE0297944691, 'x3': 0xB07C5E3A9C89C150,
              'NG': 1, 'ZR': 1, 'CY': 1, 'OV': 0},
    'taint': {'x1': 0x8204000000000000},
}


def test_fused_shift_overflow_is_tainted() -> None:
    res = _run(_ADDS_ASR_WITNESS['asm'], _ADDS_ASR_WITNESS['state'],
               _ADDS_ASR_WITNESS['taint'])
    assert res['OV'], (
        'adds with a shifted operand lost OV: the exact overflow term declines on '
        'the barrel-shifted input and no floor fired'
    )


def test_unshifted_adds_overflow_keeps_its_exact_term() -> None:
    """No shift means no decline, so the floor must not fire and cost precision."""
    res = _run('adds x0, x1, x2',
               {'x0': 0, 'x1': 1, 'x2': 1, 'x3': 0,
                'NG': 0, 'ZR': 0, 'CY': 0, 'OV': 0},
               {'x1': 1})
    assert not res['OV'], (
        'adds x0,x1,x2 with a single low-bit taint should not taint OV -- the '
        'fused-shift floor has widened past its gate'
    )
