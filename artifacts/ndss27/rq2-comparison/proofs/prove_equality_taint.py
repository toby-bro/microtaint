#!/usr/bin/env python3
"""Z3 proof of the taint rule for an equality bit ``[a == b]`` / ``[a != b]``.

Equality is SYMMETRIC -- non-monotone in BOTH directions -- so it has no polarity
orientation and MicroTaint's 2-corner differential collapses (both extremal corners
land in the equal regime).  The exact taint (EqualityTaintExpr in ast.pyx) asks
whether the equality can VARY over the taint cube::

    equal_achievable   = ((a ^ b) & ~(Ta | Tb)) == 0    # every FIXED bit already agrees
    unequal_achievable = (Ta | Tb) != 0                 # some bit is free to break equality
    Tr = equal_achievable AND unequal_achievable

``[a != b]`` has identical sensitivity (it is the negation of a value that itself
varies iff ``[a==b]`` varies), so the same rule applies.

Theorems discharged, per width w:

  no-under-taint  a witness pair in the cube with different equality value
                  => the rule fires.  Quantifier-free; scales to real widths.
  no-over-taint   the rule fires => equality is non-constant on the cube.  ForAll,
                  so Z3 tops out at modest widths.

PROVED = unsat, CEX = sat (rule WRONG), UNKNOWN = solver gave up.

Run:  uv run --with z3-solver python prove_equality_taint.py
      [--widths 2,3,4,8,16,32,64] [--precision-max-width 8]
"""

from __future__ import annotations

import argparse
import sys

from z3 import (
    And,
    BitVec,
    BitVecVal,
    ForAll,
    Implies,
    Not,
    Solver,
    unsat,
)


def _verdict(solver: Solver) -> str:
    r = solver.check()
    if r == unsat:
        return 'PROVED'
    return 'CEX' if str(r) == 'sat' else 'UNKNOWN'


def _rule(a, b, Ta, Tb, w):
    """EqualityTaintExpr."""
    free = Ta | Tb
    equal_ach = (a ^ b) & ~free == BitVecVal(0, w)
    unequal_ach = free != BitVecVal(0, w)
    return And(equal_ach, unequal_ach)


def prove_soundness(w: int) -> str:
    """NO UNDER-TAINT: a witness pair in the cube with differing equality => rule fires."""
    a, b, Ta, Tb = (BitVec(n, w) for n in ('a', 'b', 'Ta', 'Tb'))
    a1, b1, a2, b2 = (BitVec(n, w) for n in ('a1', 'b1', 'a2', 'b2'))
    in_cube = And(
        (a1 & ~Ta) == (a & ~Ta), (b1 & ~Tb) == (b & ~Tb),
        (a2 & ~Ta) == (a & ~Ta), (b2 & ~Tb) == (b & ~Tb),
    )
    s = Solver()
    s.add(in_cube, (a1 == b1) != (a2 == b2), Not(_rule(a, b, Ta, Tb, w)))
    return _verdict(s)


def prove_precision(w: int) -> str:
    """NO OVER-TAINT: rule fires => equality is non-constant on the cube.  ForAll."""
    a, b, Ta, Tb = (BitVec(n, w) for n in ('a', 'b', 'Ta', 'Tb'))
    ap, bp = BitVec('ap', w), BitVec('bp', w)
    constant = ForAll(
        [ap, bp],
        Implies(And((ap & ~Ta) == (a & ~Ta), (bp & ~Tb) == (b & ~Tb)), (ap == bp) == (a == b)),
    )
    s = Solver()
    s.add(_rule(a, b, Ta, Tb, w), constant)
    return _verdict(s)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--widths', default='2,3,4,5,6,8,16,32,64')
    ap.add_argument('--precision-max-width', type=int, default=8)
    args = ap.parse_args()
    widths = [int(x) for x in args.widths.split(',')]

    ok = True
    for w in widths:
        snd = prove_soundness(w)
        prc = prove_precision(w) if w <= args.precision_max_width else 'skip'
        print(f'w={w:2}  no-under-taint={snd:7}  no-over-taint={prc}')
        if snd != 'PROVED' or prc == 'CEX':
            ok = False
    print('\nALL PROVED (soundness all widths; precision <= max-width)' if ok else '\nFAILURES ABOVE')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
