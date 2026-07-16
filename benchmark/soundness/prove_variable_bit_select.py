#!/usr/bin/env python3
"""Z3 proof of the exact taint rule for variable bit-selection (`bt r, r` -> CF).

`bt rax, rbx` lifts to CF = bit[(rbx & (w-1))] of rax, i.e. a bit SELECTED by a
data-dependent index.  This is NON-MONOTONE in the index, so MicroTaint's
2-replica differential -- which samples only the extremal corners V|T and V&~T --
reads the source at exactly TWO index values and misses every other reachable
index.  That is the `xor edx,edx; bt rax,rbx; setc dl` under-taint class.

Avalanching the flag would be sound but would OVER-taint.  The exact rule instead
enumerates the reachable index set:

    I = { (b & (w-1)) : b in taint-cube(b) }        (|I| <= w, cheap)

    CF is tainted
      <=>  (exists i in I: T_a[i] = 1)                     # selected bit is itself tainted
       or  (exists i,j in I: T_a[i]=T_a[j]=0 and a_i != a_j)  # two reachable CLEAN bits differ

Justification: for a fixed index i, CF = a_i.  Over the cube, the reachable CF
values are  { a_i : i in I, T_a[i]=0 }  union  ({0,1} if some i in I has T_a[i]=1).
CF is tainted exactly when that set has both 0 and 1 -- which is precisely the
disjunction above.  Note the index bits and the source bits are read from
DIFFERENT registers, so they vary independently; and for a tainted index bit the
reachable index set is exactly the sub-cube enumeration, no approximation.

Theorems, per width w in the sweep:

  no-under-taint  a witness in the taint cube flips CF  =>  the rule fires.
                  Refutation query is QUANTIFIER-FREE, so it scales.
  no-over-taint   the rule fires  =>  a witness really exists.  Needs a ForAll,
                  so Z3 tops out at modest widths.

Verdicts are reported honestly:
  PROVED  = unsat   (no counterexample exists)
  CEX     = sat     (a real counterexample: the rule is WRONG)
  UNKNOWN = unknown (Z3 gave up; NOT evidence either way)

Run:  uv run --with z3-solver python prove_variable_bit_select.py
      [--widths 2,4,8,16,32,64] [--precision-max-width 8]
"""

from __future__ import annotations

import argparse
import sys

from z3 import (
    And,
    BitVec,
    BitVecVal,
    Extract,
    ForAll,
    LShR,
    Not,
    Or,
    Solver,
    unsat,
)


def _cf(a, b, w):
    """CF = bit[(b & (w-1))] of a  -- the `bt` semantics."""
    idx = b & BitVecVal(w - 1, w)
    return Extract(0, 0, LShR(a, idx)) == BitVecVal(1, 1)


def _verdict(solver: Solver) -> str:
    """Discharge a refutation query honestly (see module docstring)."""
    r = solver.check()
    if r == unsat:
        return 'PROVED'
    return 'CEX' if str(r) == 'sat' else 'UNKNOWN'


def _reachable_indices(b, Tb, w: int) -> list[int]:
    """Concrete enumeration is impossible symbolically; instead we express the
    rule over ALL indices i in 0..w-1, guarded by "i is reachable".

    i is reachable  <=>  (i & ~Tb_low) == (b & ~Tb_low)   over the low log2(w) bits,
    i.e. i agrees with b on every UNtainted index bit.
    """
    del b, Tb
    return list(range(w))


def _rule(a, b, Ta, Tb, w):
    """Exact taint of CF (see module docstring)."""
    nbits = (w - 1).bit_length()  # log2(w) index bits
    low = BitVecVal((1 << nbits) - 1, w)
    b_idx = b & low
    t_idx = Tb & low

    def reachable(i: int):
        """index i is reachable iff it agrees with b on every untainted index bit"""
        iv = BitVecVal(i, w)
        return (iv & ~t_idx & low) == (b_idx & ~t_idx & low)

    def a_bit(i: int):
        return Extract(0, 0, LShR(a, BitVecVal(i, w))) == BitVecVal(1, 1)

    def ta_bit(i: int):
        return Extract(0, 0, LShR(Ta, BitVecVal(i, w))) == BitVecVal(1, 1)

    idxs = _reachable_indices(b, Tb, w)

    # (a) some reachable index selects a TAINTED source bit
    part_a = Or(*[And(reachable(i), ta_bit(i)) for i in idxs])

    # (b) two reachable indices select CLEAN source bits with DIFFERENT values
    pairs = []
    for i in idxs:
        for j in idxs:
            if j <= i:
                continue
            pairs.append(
                And(
                    reachable(i),
                    reachable(j),
                    Not(ta_bit(i)),
                    Not(ta_bit(j)),
                    a_bit(i) != a_bit(j),
                ),
            )
    part_b = Or(*pairs) if pairs else BitVecVal(0, 1) == BitVecVal(1, 1)
    return Or(part_a, part_b)


def prove_soundness(w: int) -> str:
    """NO UNDER-TAINT: a witness in the cube flips CF => the rule fires."""
    a, b, Ta, Tb = BitVec('a', w), BitVec('b', w), BitVec('Ta', w), BitVec('Tb', w)
    ap, bp = BitVec('ap', w), BitVec('bp', w)
    in_cube = And((ap & ~Ta) == (a & ~Ta), (bp & ~Tb) == (b & ~Tb))
    s = Solver()
    s.add(in_cube, _cf(ap, bp, w) != _cf(a, b, w), Not(_rule(a, b, Ta, Tb, w)))
    return _verdict(s)


def prove_precision(w: int) -> str:
    """NO OVER-TAINT: the rule fires => a witness really exists."""
    a, b, Ta, Tb = BitVec('a', w), BitVec('b', w), BitVec('Ta', w), BitVec('Tb', w)
    ap, bp = BitVec('ap', w), BitVec('bp', w)
    in_cube = And((ap & ~Ta) == (a & ~Ta), (bp & ~Tb) == (b & ~Tb))
    s = Solver()
    s.add(_rule(a, b, Ta, Tb, w))
    s.add(ForAll([ap, bp], Or(Not(in_cube), _cf(ap, bp, w) == _cf(a, b, w))))
    return _verdict(s)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--widths', type=str, default='2,4,8,16,32,64')
    p.add_argument('--precision-max-width', type=int, default=8)
    args = p.parse_args()

    any_cex = False
    print('=== bt: CF = bit[(b & (w-1))] of a  (variable bit-selection) ===')
    print('  width  no-under-taint  no-over-taint')
    for w in [int(x) for x in args.widths.split(',')]:
        t_sound = prove_soundness(w)
        t_prec = prove_precision(w) if w <= args.precision_max_width else 'n/a'
        any_cex |= 'CEX' in (t_sound, t_prec)
        print(f'  w={w:<4d} {t_sound:15s} {t_prec}')

    print()
    print('PROVED  = no counterexample exists (unsat)')
    print('UNKNOWN = Z3 gave up on the quantified query; NOT evidence of a bug')
    print('CEX     = a real counterexample: the rule is WRONG')
    print('n/a     = not attempted (ForAll query is intractable at this width)')
    if any_cex:
        print('\nRESULT: COUNTEREXAMPLE FOUND -- rule is unsound/imprecise')
        return 1
    print('\nRESULT: no counterexample at any tested width.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
