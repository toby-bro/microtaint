"""Step 2 of the non-monotone taint theory (docs/design/nonmonotone-taint-theory.md):
EXACT closed-form terms for the two non-orientable primitives -- a comparison bit and
an equality bit -- replacing the avalanche floors on ``cmpw``/``mfcr`` and ``cset``
lt/ge.  Both terms are Z3-proved (benchmark/soundness/prove_comparison_taint.py,
prove_equality_taint.py); here we (1) brute-force the Cython nodes against the true
non-constancy for small widths and (2) check the packed-comparison builder makes the
real instructions exact.
"""

# mypy: disable-error-code="no-untyped-def,no-untyped-call,attr-defined,union-attr"

from __future__ import annotations

import itertools

import microtaint.sleigh.engine as engine
from microtaint.instrumentation.ast import (
    ComparisonTaintExpr,
    Constant,
    EqualityTaintExpr,
    EvalContext,
)
from microtaint.simulator import CellSimulator
from microtaint.types import Architecture, ImplicitTaintPolicy, Register


def _ctx():
    return EvalContext(input_taint={}, input_values={}, simulator=None, implicit_policy=ImplicitTaintPolicy.IGNORE)


def _true_taint(pred, a, ta, b, tb):
    """Non-constancy of pred over the taint cube (a and b vary independently)."""
    abits = [i for i in range(64) if (ta >> i) & 1]
    bbits = [i for i in range(64) if (tb >> i) & 1]
    seen = set()
    for ca in itertools.product([0, 1], repeat=len(abits)):
        aa = a & ~ta
        for bit, v in zip(abits, ca, strict=True):
            aa |= v << bit
        for cb in itertools.product([0, 1], repeat=len(bbits)):
            bb = b & ~tb
            for bit, v in zip(bbits, cb, strict=True):
                bb |= v << bit
            seen.add(pred(aa, bb))
    return 1 if len(seen) > 1 else 0


def test_comparison_taint_expr_exhaustive():
    """ComparisonTaintExpr == true non-constancy of [a OP b], all a,b,Ta,Tb, w=2..4,
    every variant (signed x {<, <=})."""
    for w in (2, 3, 4):
        mask = (1 << w) - 1
        sb = 1 << (w - 1)
        for is_signed in (False, True):
            for or_equal in (False, True):
                def pred(a, b, w=w, is_signed=is_signed, or_equal=or_equal, sb=sb):
                    if is_signed:
                        a = a - (1 << w) if a & sb else a
                        b = b - (1 << w) if b & sb else b
                    return int(a <= b) if or_equal else int(a < b)

                for a in range(mask + 1):
                    for b in range(mask + 1):
                        for ta in range(mask + 1):
                            for tb in range(mask + 1):
                                e = ComparisonTaintExpr(
                                    Constant(a, w), Constant(ta, w), Constant(b, w), Constant(tb, w),
                                    w, is_signed, or_equal,
                                )
                                assert e.evaluate(_ctx()) & 1 == _true_taint(pred, a, ta, b, tb)


def test_equality_taint_expr_exhaustive():
    """EqualityTaintExpr == true non-constancy of [a == b], all a,b,Ta,Tb, w=2..4."""
    for w in (2, 3, 4):
        mask = (1 << w) - 1
        for a in range(mask + 1):
            for b in range(mask + 1):
                for ta in range(mask + 1):
                    for tb in range(mask + 1):
                        e = EqualityTaintExpr(
                            Constant(a, w), Constant(ta, w), Constant(b, w), Constant(tb, w), w,
                        )
                        assert e.evaluate(_ctx()) & 1 == _true_taint(lambda x, y: int(x == y), a, ta, b, tb)


# --- integration: the packed-comparison builder on real instructions ---

_PPC = Architecture.PPC32BE
_PPC_FMT = (
    [Register(f'R{i}', 32) for i in range(8)]
    + [Register('XER_SO', 1), Register('XER_OV', 1), Register('XER_CA', 1)]
    + [Register(f'CR{i}', 4) for i in range(8)]
)
_PPC_ZERO = {r.name: 0 for r in _PPC_FMT}
_CMPW_MFCR = bytes.fromhex('7c0428007c600026')  # cmpw 0,4,5 ; mfcr 3


def _ppc_r3(vals, taint):
    engine._cached_generate_static_rule.cache_clear()
    circ = engine.generate_static_rule(_PPC, _CMPW_MFCR, _PPC_FMT)
    ctx = EvalContext(
        input_taint={**_PPC_ZERO, **taint},
        input_values={**_PPC_ZERO, **vals},
        simulator=CellSimulator(_PPC),
        implicit_policy=ImplicitTaintPolicy.IGNORE,
    )
    return circ.evaluate(ctx).get('R3', 0)


def test_cmpw_mfcr_exact_via_packed_comparison():
    """cmpw packs LT(r4<r5)<<3 | GT(r5<r4)<<2 | EQ<<1 | SO; mfcr moves CR0 into R3[31:28].
    The packed-comparison builder gives each LT/GT/EQ leaf its exact term, so:

    * a DETERMINATE compare (r4 in [0,0xff] always < r5=0x40000000) taints NOTHING in
      R3[31:28] -- the old swapped-comparison floor avalanched the LT/GT bits here;
    * both operands fully tainted taints exactly the LT/GT/EQ bits (31,30,29), not SO."""
    assert _ppc_r3({'R4': 0, 'R5': 0x40000000}, {'R4': 0xFF}) & 0xF0000000 == 0
    assert _ppc_r3({'R4': 0, 'R5': 0}, {'R4': 0xFFFFFFFF, 'R5': 0xFFFFFFFF}) & 0xF0000000 == 0xE0000000
    # a single tainted low bit with a determinate ordering stays untainted
    assert _ppc_r3({'R4': 0x10, 'R5': 0x5678}, {'R4': 0x1}) & 0xF0000000 == 0


def test_cmpw_builder_falls_through_without_xer():
    """Soundness guard: if a leaf operand (SO=xer_so) does not map, the builder must
    return None and fall through to the (sound) differential+floor rather than emit a
    partial term.  With no XER in the state format the CR0 field is still tainted when
    the operands are (soundness preserved), just via the floor."""
    fmt = [Register(f'R{i}', 32) for i in range(8)] + [Register(f'CR{i}', 4) for i in range(8)]
    zero = {r.name: 0 for r in fmt}
    engine._cached_generate_static_rule.cache_clear()
    circ = engine.generate_static_rule(_PPC, _CMPW_MFCR, fmt)
    ctx = EvalContext(
        input_taint={**zero, 'R4': 0xFFFFFFFF, 'R5': 0xFFFFFFFF},
        input_values={**zero, 'R4': 0, 'R5': 0},
        simulator=CellSimulator(_PPC),
        implicit_policy=ImplicitTaintPolicy.IGNORE,
    )
    assert circ.evaluate(ctx).get('R3', 0) & 0xE0000000  # LT/GT/EQ still tainted (sound)


_ARM = Architecture.ARM64
_ARM_FMT = [Register('X0', 64), Register('N', 1), Register('Z', 1), Register('C', 1), Register('V', 1)]
_ARM_ZERO = {r.name: 0 for r in _ARM_FMT}
_CSET_LT = b'\xe0\xa7\x9f\x9a'  # cset x0, lt = ZEXT(N != V)
_CSET_GE = b'\xe0\xb7\x9f\x9a'  # cset x0, ge = ZEXT(N == V)


def _arm_x0(code, vals, taint):
    engine._cached_generate_static_rule.cache_clear()
    circ = engine.generate_static_rule(_ARM, code, _ARM_FMT)
    ctx = EvalContext(
        input_taint={**_ARM_ZERO, **taint},
        input_values={**_ARM_ZERO, **vals},
        simulator=CellSimulator(_ARM, use_unicorn=False, use_c=False),
        implicit_policy=ImplicitTaintPolicy.IGNORE,
    )
    return circ.evaluate(ctx).get('X0', 0)


def test_cset_lt_ge_exact_via_equality_term():
    """cset lt = ZEXT(N!=V), ge = ZEXT(N==V): with N,V tainted DIRECTLY the equality
    term is exact.  Determinate equality must NOT taint (N==N is constant when the
    tainted flags are forced equal by fixing), varying must taint bit 0 only."""
    # lt: N tainted, V=0 fixed -> [N!=0] = N, varies -> tainted
    assert _arm_x0(_CSET_LT, {'N': 1, 'V': 0}, {'N': 1}) == 1
    # lt: BOTH tainted -> N!=V varies -> tainted (bit 0 only)
    assert _arm_x0(_CSET_LT, {'N': 1, 'V': 0}, {'N': 1, 'V': 1}) == 1
    # ge: both tainted -> N==V varies -> tainted
    assert _arm_x0(_CSET_GE, {'N': 1, 'V': 0}, {'N': 1, 'V': 1}) == 1
    # flags not read must not taint
    assert _arm_x0(_CSET_LT, {}, {'Z': 1}) == 0
    assert _arm_x0(_CSET_LT, {}, {'C': 1}) == 0


if __name__ == '__main__':
    import pytest

    raise SystemExit(pytest.main([__file__, '-v']))
