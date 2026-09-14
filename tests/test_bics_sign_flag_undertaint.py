"""Regression: the shifted-operand sign floor must root at the producing op.

Found by the 2026-09-12 AE soundness campaign (4 cases in 1.9 M ARM64 cases, so
only deep sampling surfaces it).  `bics x0, x1, x2, lsl #k` lifts as
``(x2<<k) ^ -1`` then ``& x1`` then ``s< 0``.  The floor selected the FIRST
arith/logical op, the intermediate XOR, whose operands are the shifted x2 and the
-1 mask, so x1's sign contribution never reached it.

The tell is that the engine contradicts itself: its own x0 taint has bit 63 set
and matches the ground truth exactly, while NG stays clean -- and for BICS, NG IS
bit 63 of the result.
"""
from __future__ import annotations

import pytest

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, ImplicitTaintPolicy, Register

NAMES = ['x0', 'x1', 'x2', 'x3', 'NG', 'ZR', 'CY', 'OV']
FMT = ([Register(n, 64) for n in NAMES[:4]]
       + [Register(f, 1) for f in NAMES[4:]] + [Register('sp', 64)])

# Campaign witnesses: (bytes, state, taint).  Recorded, not synthesised.

CASES = [
    pytest.param('bics x0, x1, x2, lsl #1',
                 {'x0': 0x3DFB845C05DE24AE, 'x1': 0xAA98FFE4069AFBB5,
                  'x2': 0x98265EFC47533227, 'x3': 0x42DBF16D5159CFD4,
                  'NG': 0, 'ZR': 0, 'CY': 0, 'OV': 0},
                 {'x1': 0x8000000010040000, 'x2': 0x4000000000040000}, id='lsl1'),
    pytest.param('bics x0, x1, x2, lsl #3',
                 {'x0': 0x6CACC424310382EC, 'x1': 0x6DCAEE523400B11E,
                  'x2': 0x2F718ED5D4E9D0E5, 'x3': 0xBCC7B5D23E11A174,
                  'NG': 0, 'ZR': 0, 'CY': 0, 'OV': 1},
                 {'x1': 0xA000020000000000, 'x2': 0x1010000000008000}, id='lsl3'),
    pytest.param('bics x0, x1, x2, lsl #7',
                 {'x0': 0x0C3EBDDB2416C816, 'x1': 0x4A61F7953767A8D9,
                  'x2': 0x7131624D2EBFA6B8, 'x3': 0x583A4BFA5E2383B2,
                  'NG': 1, 'ZR': 1, 'CY': 1, 'OV': 0},
                 {'x2': 0x2100020000000000, 'x1': 0x8004000102000000}, id='lsl7'),
]


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


@pytest.mark.parametrize(('asm', 'state', 'taint'), CASES)
def test_bics_sign_flag_follows_result_sign(
        asm: str, state: dict[str, int], taint: dict[str, int]) -> None:
    res = _run(asm, state, taint)
    assert res['x0'] & (1 << 63), 'precondition: the result sign bit must be tainted'
    assert res['NG'], (
        f'{asm}: the result sign bit is tainted but NG is not -- for BICS, NG IS '
        'that bit, so the sign floor rooted at the wrong op'
    )


def test_unshifted_logical_sign_flag_still_exact() -> None:
    """Single-op slices must be untouched: the producing op IS the first op."""
    res = _run('ands x0, x1, x2',
               {'x0': 0, 'x1': 0xAA98FFE4069AFBB5, 'x2': 0x98265EFC47533227,
                'x3': 0, 'NG': 0, 'ZR': 0, 'CY': 0, 'OV': 0},
               {'x1': 1 << 63})
    assert res['NG'], 'ands NG lost the sign taint'
