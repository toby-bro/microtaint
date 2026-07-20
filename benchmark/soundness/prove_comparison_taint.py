#!/usr/bin/env python3
"""Z3 proof of the cross-corner taint rule for a comparison bit ``[a OP b]``.

A comparison is ANTITONE in its LHS and MONOTONE in its RHS.  MicroTaint's 2-corner
differential picks ONE global polarity per operand, which is exact for a single
comparison but cannot serve a slice where the same operand feeds two comparisons of
OPPOSITE orientation (PPC ``cmpw`` packs ``[a<b]`` and ``[b<a]`` into CR0).  The exact
taint is computed directly from the CROSS corners (ComparisonTaintExpr in ast.pyx)::

    can_be_true = [ min(a) OP max(b) ]     # a smallest, b largest
    always_true = [ max(a) OP min(b) ]     # a largest,  b smallest
    Tr          = can_be_true XOR always_true

with min(x)=V&~T, max(x)=V|T, and a SIGNED compare first XORing the sign bit of both
operands into the unsigned domain (a bijection on the cube).

Theorems discharged, per width w and per variant (signed x {<, <=}):

  no-under-taint  a witness pair in the cube gives different predicate values
                  => the rule fires.  (soundness -- the property the paper claims.)
                  Quantifier-free refutation, so it scales to real widths.
  no-over-taint   the rule fires => a witness pair really exists (predicate is
                  non-constant on the cube).  Needs a ForAll, so Z3 tops out at
                  modest widths.

Refutation verdicts (as in prove_signed_overflow.py):
  PROVED = unsat, CEX = sat (rule WRONG), UNKNOWN = solver gave up.

Run:  uv run --with z3-solver python prove_comparison_taint.py
      [--widths 2,3,4,8,16,32,64] [--precision-max-width 6]
"""

from __future__ import annotations

import argparse
import sys

from z3 import (
    ULE,
    ULT,
    And,
    BitVec,
    BitVecVal,
    ForAll,
    Implies,
    Not,
    Solver,
    Xor,
    unsat,
)


def _verdict(solver: Solver) -> str:
    r = solver.check()
    if r == unsat:
        return 'PROVED'
    return 'CEX' if str(r) == 'sat' else 'UNKNOWN'


def _flip_sign(x, w, is_signed):
    """Signed compare == unsigned compare after flipping the sign bit."""
    if not is_signed:
        return x
    return x ^ BitVecVal(1 << (w - 1), w)


def _rule(a, b, Ta, Tb, w, is_signed, or_equal):
    """ComparisonTaintExpr: cross-corner differential in the (sign-flipped) domain."""
    af = _flip_sign(a, w, is_signed)
    bf = _flip_sign(b, w, is_signed)
    amin, amax = af & ~Ta, af | Ta
    bmin, bmax = bf & ~Tb, bf | Tb
    cmp_op = ULE if or_equal else ULT
    return Xor(cmp_op(amin, bmax), cmp_op(amax, bmin))


def prove_soundness(w: int, is_signed: bool, or_equal: bool) -> str:
    """NO UNDER-TAINT: witness pair in the cube with differing predicate => rule fires.
    Quantifier-free (witnesses existential)."""
    a, b, Ta, Tb = (BitVec(n, w) for n in ('a', 'b', 'Ta', 'Tb'))
    a1, b1, a2, b2 = (BitVec(n, w) for n in ('a1', 'b1', 'a2', 'b2'))
    cmp_op = ULE if or_equal else ULT

    def sp(x, y):
        return cmp_op(_flip_sign(x, w, is_signed), _flip_sign(y, w, is_signed))

    in_cube = And(
        (a1 & ~Ta) == (a & ~Ta), (b1 & ~Tb) == (b & ~Tb),
        (a2 & ~Ta) == (a & ~Ta), (b2 & ~Tb) == (b & ~Tb),
    )
    s = Solver()
    s.add(in_cube, sp(a1, b1) != sp(a2, b2), Not(_rule(a, b, Ta, Tb, w, is_signed, or_equal)))
    return _verdict(s)


def prove_precision(w: int, is_signed: bool, or_equal: bool) -> str:
    """NO OVER-TAINT: rule fires => predicate is non-constant on the cube.  ForAll."""
    a, b, Ta, Tb = (BitVec(n, w) for n in ('a', 'b', 'Ta', 'Tb'))
    ap, bp = BitVec('ap', w), BitVec('bp', w)
    cmp_op = ULE if or_equal else ULT

    def sp(x, y):
        return cmp_op(_flip_sign(x, w, is_signed), _flip_sign(y, w, is_signed))

    # constant on cube: every cube member has the same predicate value as (a,b)
    constant = ForAll(
        [ap, bp],
        Implies(And((ap & ~Ta) == (a & ~Ta), (bp & ~Tb) == (b & ~Tb)), sp(ap, bp) == sp(a, b)),
    )
    s = Solver()
    s.add(_rule(a, b, Ta, Tb, w, is_signed, or_equal), constant)
    return _verdict(s)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--widths', default='2,3,4,5,6,8,16,32,64')
    ap.add_argument('--precision-max-width', type=int, default=6)
    args = ap.parse_args()
    widths = [int(x) for x in args.widths.split(',')]

    ok = True
    for is_signed in (False, True):
        for or_equal in (False, True):
            tag = ('s' if is_signed else 'u') + ('<=' if or_equal else '<')
            print(f'=== variant [{tag}] ===')
            for w in widths:
                snd = prove_soundness(w, is_signed, or_equal)
                prc = prove_precision(w, is_signed, or_equal) if w <= args.precision_max_width else 'skip'
                print(f'  w={w:2}  no-under-taint={snd:7}  no-over-taint={prc}')
                if snd != 'PROVED' or prc == 'CEX':
                    ok = False
    print('\nALL PROVED (soundness all widths; precision <= max-width)' if ok else '\nFAILURES ABOVE')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
