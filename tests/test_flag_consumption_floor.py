"""Regression: the COND_TRANSPORTABLE 1-bit-flag floors must fire when a flag is
consumed into a WIDE register.

A flag read into a 64-bit GPR (ARM64 `cset`/`csel`, e.g. `cset x0, lt` = ZEXT(N!=V))
is still a 0/1 in bit 0.  The two soundness floors were gated on
`out_bit_end - out_bit_start <= 7` (sized for x86 `setcc` BYTE outputs), so a
64-bit consumer got only the masked-single-replica COND term, which under-taints:
masking the tainted flag to 0 collapses `N!=V` to `0`.  This pins:

  * the 2-replica differential floor (single tainted flag), and
  * the FullMaskAvalanche-per-flag floor (BOTH flags tainted -- the non-monotone
    interior the 2-corner differential misses),

both firing regardless of the consuming register width.  Gate-independent
(fixes gate-off), and x86 `setcc` is unchanged (still <= 8 bits).
"""

# mypy: disable-error-code="no-untyped-def,no-untyped-call,attr-defined,union-attr"

from __future__ import annotations

import microtaint.sleigh.engine as engine
from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.types import Architecture, ImplicitTaintPolicy, Register

ARCH = Architecture.ARM64
_FMT = [Register('X0', 64), Register('N', 1), Register('Z', 1), Register('C', 1), Register('V', 1)]
_SIM = CellSimulator(ARCH, use_unicorn=False, use_c=False)
_ZERO = {r.name: 0 for r in _FMT}

# cset x0, lt  (reads N,V);  cset x0, hi (reads Z,C)
_CSET_LT = b'\xe0\xa7\x9f\x9a'
_CSET_HI = b'\xe0\x97\x9f\x9a'


def _x0_taint(code: bytes, taint: dict[str, int]) -> int:
    engine._SEGMENTED = False
    engine._cached_generate_static_rule.cache_clear()
    circ = engine.generate_static_rule(ARCH, code, _FMT)
    ctx = EvalContext(
        input_taint={**_ZERO, **taint},
        input_values={**_ZERO, 'N': 1, 'V': 0, 'Z': 0, 'C': 1},
        simulator=_SIM,
        implicit_policy=ImplicitTaintPolicy.IGNORE,
    )
    return circ.evaluate(ctx).get('X0', 0)


def test_cset_lt_taints_wide_output_from_its_flags():
    """cset x0,lt = ZEXT(N!=V): tainting N, V, or BOTH must taint X0 bit 0."""
    assert _x0_taint(_CSET_LT, {'N': 1}) & 1
    assert _x0_taint(_CSET_LT, {'V': 1}) & 1
    assert _x0_taint(_CSET_LT, {'N': 1, 'V': 1}) & 1  # non-monotone: 2-corner alone misses this
    # flags it does NOT read must not taint the result
    assert _x0_taint(_CSET_LT, {'Z': 1}) == 0
    assert _x0_taint(_CSET_LT, {'C': 1}) == 0
    # taint is confined to bit 0 (no upper-byte over-taint)
    assert _x0_taint(_CSET_LT, {'N': 1}) == 1


def test_cset_hi_taints_wide_output_from_its_flags():
    """cset x0,hi reads Z,C (not N,V)."""
    assert _x0_taint(_CSET_HI, {'Z': 1}) & 1
    assert _x0_taint(_CSET_HI, {'C': 1}) & 1
    assert _x0_taint(_CSET_HI, {'N': 1}) == 0
    assert _x0_taint(_CSET_HI, {'V': 1}) == 0


# csel x0, x1, x2, lt  -- a 2-way select gated by NZCV
_CSEL_LT = b'\x20\xb0\x82\x9a'
_FMT_SEL = [
    Register('X0', 64), Register('X1', 64), Register('X2', 64),
    Register('N', 1), Register('Z', 1), Register('C', 1), Register('V', 1),
]
_ZERO_SEL = {r.name: 0 for r in _FMT_SEL}


def test_csel_tainted_condition_uses_isa_general_passthrough():
    """The cmov/select gated-passthrough resolves its condition flags through the
    mapper (ISA-general), not a hardcoded x86 flag-offset table -- so a tainted
    ARM64 NZCV condition OR-s in the operand union.  With x1==x2 in value the
    differential cancels to 0; only the passthrough (fed by the tainted flag)
    recovers the operand taint."""
    engine._SEGMENTED = False
    engine._cached_generate_static_rule.cache_clear()
    circ = engine.generate_static_rule(ARCH, _CSEL_LT, _FMT_SEL)
    ctx = EvalContext(
        input_taint={**_ZERO_SEL, 'N': 1, 'X1': 0xF},
        input_values={**_ZERO_SEL, 'N': 1, 'V': 0, 'X1': 0xF, 'X2': 0xF},
        simulator=CellSimulator(ARCH, use_unicorn=False, use_c=False),
        implicit_policy=ImplicitTaintPolicy.IGNORE,
    )
    # tainted condition + cancelling operands: differential alone -> 0; the
    # passthrough must recover x1's taint.
    assert circ.evaluate(ctx).get('X0', 0) & 0xF == 0xF


if __name__ == '__main__':
    import pytest

    raise SystemExit(pytest.main([__file__, '-v']))
