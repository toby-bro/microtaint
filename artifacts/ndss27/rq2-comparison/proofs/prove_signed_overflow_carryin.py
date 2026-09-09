#!/usr/bin/env python3
"""Z3 proof of the signed-overflow taint rule for carry-IN add/subtract.

`adc` / `sbb` / `adcs` / `sbcs` lift their overflow flag as a CHAIN of two
overflow predicates rather than one::

    OF = ovf(a, b)  XOR  ovf(a o b, c)          c = zext(carry flag) in {0,1}

      x86   adc:  scarry(RAX,RBX)  ^^ scarry(RAX+RBX,  zext(CF))
      x86   sbb:  sborrow(RAX,RBX) ^^ sborrow(RAX-RBX, zext(CF))
      ARM64 adcs: scarry(x1,x2)    ^^ scarry(x1+x2,    zext(CY))
      ARM64 sbcs: scarry(x1,~x2)   ^^ scarry(x1+~x2,   zext(CY))

Because the slice then holds TWO overflow ops, the two-operand exact term
declined and OF fell back to the 2-corner differential -- which non-monotone
signed overflow slips through.  This script proves the generalisation.

The claim is that the chain is nothing more exotic than the signed overflow of
the THREE-operand sum a + b + c, and that the existing sign decomposition
extends to it by giving the monotone carry/borrow a third contributor::

    Car3 = [ a[0:w-1] +u b[0:w-1] +u c  >= 2^(w-1) ]      (add family)
    Bor3 = [ a[0:w-1] <u b[0:w-1] +u c ]                  (sub family)
    OF   = ~(a_s ^ b_s) & (b_s ^ Car3)     /    (a_s ^ b_s) & (b_s ^ Bor3)

The sign function is UNCHANGED.  That is the whole point: because c is zext of a
1-bit flag it lies wholly inside the low part and its sign bit is 0, so it can
only shift the carry/borrow INTO the msb.  Exactness rests on the same two facts
as the two-operand rule, both of which survive the extra operand:

  1. a_s (bit w-1 of a), b_s (bit w-1 of b) and Car3/Bor3 (bits 0..w-2 of a and
     b, plus all of c) read DISJOINT input bits, so they vary INDEPENDENTLY and
     enumerating their reachable values is exact rather than a product
     over-approximation.
  2. Car3/Bor3 stays MONOTONE with the third operand (carry is increasing in c;
     borrow is increasing in c and decreasing in a), so its reachable set over
     the taint cube is exactly {polarised-min, polarised-max}.

Theorems discharged, per width w in the sweep:

  chain-equals-sum  the SLEIGH chain ovf(a,b) ^ ovf(a o b, c) equals the
                    widened-arithmetic signed overflow of a o b o c.  This is
                    what licenses treating the pair as one 3-operand flag.
  identity          the decomposition equals that spec.
  monotone-reach    Car3/Bor3 being monotone, the polarised differential XOR
                    equals its true non-constancy (proved both directions).
  no-under-taint    a witness in the taint cube flips OF => the rule fires.
                    Quantifier-free, so it scales to real widths.
  no-over-taint     the rule fires => a witness really exists.  Needs a ForAll,
                    so Z3 tops out at modest widths.

Verdicts are reported honestly:
  PROVED  = unsat   (no counterexample exists)
  CEX     = sat     (a real counterexample: the rule is WRONG)
  UNKNOWN = unknown (Z3 gave up; NOT evidence either way)

Run:  uv run --with z3-solver python prove_signed_overflow_carryin.py
      [--widths 2,3,4,5,6,7,8,16,32,64] [--precision-max-width 5]
"""

from __future__ import annotations

import argparse
import itertools
import sys

from z3 import (
    ULE,
    ULT,
    And,
    BitVec,
    BitVecVal,
    Extract,
    ForAll,
    Not,
    Or,
    SignExt,
    Solver,
    Xor,
    ZeroExt,
    unsat,
)


def _bit(x, i):
    return Extract(i, i, x)


def _is1(x):
    return x == BitVecVal(1, 1)


# --------------------------------------------------------------------------- #
# specs
# --------------------------------------------------------------------------- #
def _ovf2_add(a, b, w):
    r = SignExt(1, a) + SignExt(1, b)
    return _bit(r, w) != _bit(r, w - 1)


def _ovf2_sub(a, b, w):
    r = SignExt(1, a) - SignExt(1, b)
    return _bit(r, w) != _bit(r, w - 1)


def _spec_chain(a, b, c, w, is_sub):
    """What SLEIGH actually computes: two overflow predicates joined by XOR.

    The inner operation wraps at w bits, exactly as the lifted INT_ADD/INT_SUB does.
    """
    if is_sub:
        return Xor(_ovf2_sub(a, b, w), _ovf2_sub(a - b, c, w))
    return Xor(_ovf2_add(a, b, w), _ovf2_add(a + b, c, w))


def _spec_sum(a, b, c, w, is_sub):
    """Signed overflow of the three-operand result, in widened arithmetic."""
    if is_sub:
        r = SignExt(1, a) - SignExt(1, b) - ZeroExt(1, c)
    else:
        r = SignExt(1, a) + SignExt(1, b) + ZeroExt(1, c)
    return _bit(r, w) != _bit(r, w - 1)


# --------------------------------------------------------------------------- #
# decomposition
# --------------------------------------------------------------------------- #
def _low(x, w):
    return Extract(w - 2, 0, x)


def _car3(a, b, c, w):
    """Carry into the msb of a + b + c, computed on the low w-1 bits.

    Widened by two bits so that low(a) + low(b) + c cannot itself wrap.
    """
    la = ZeroExt(2, _low(a, w))
    lb = ZeroExt(2, _low(b, w))
    lc = ZeroExt(2, _low(c, w))
    return ULE(BitVecVal(1, w + 1) << (w - 1), la + lb + lc)


def _bor3(a, b, c, w):
    """Borrow into the msb of a - b - c, on the low w-1 bits."""
    la = ZeroExt(2, _low(a, w))
    lb = ZeroExt(2, _low(b, w))
    lc = ZeroExt(2, _low(c, w))
    return ULT(la, lb + lc)


def _g(x, y, z, is_sub):
    """OF as a function of (a_s, b_s, Bor3|Car3) -- unchanged from two operands."""
    return And(Xor(x, y), Xor(y, z)) if is_sub else And(Not(Xor(x, y)), Xor(y, z))


def _decomp(a, b, c, w, is_sub):
    a_s = _is1(_bit(a, w - 1))
    b_s = _is1(_bit(b, w - 1))
    cc = _bor3(a, b, c, w) if is_sub else _car3(a, b, c, w)
    return _g(a_s, b_s, cc, is_sub)


# --------------------------------------------------------------------------- #
# the taint rule (mirrors SignedOverflowTaintExpr.evaluate)
# --------------------------------------------------------------------------- #
def _rule(a, b, c, Ta, Tb, Tc, w, is_sub):
    a_s = _is1(_bit(a, w - 1))
    b_s = _is1(_bit(b, w - 1))
    Ta_s = _is1(_bit(Ta, w - 1))
    Tb_s = _is1(_bit(Tb, w - 1))

    f = _bor3 if is_sub else _car3
    if is_sub:
        # borrow is decreasing in a, increasing in b and c
        hi = f(a & ~Ta, b | Tb, c | Tc, w)
        lo = f(a | Ta, b & ~Tb, c & ~Tc, w)
    else:
        hi = f(a | Ta, b | Tb, c | Tc, w)
        lo = f(a & ~Ta, b & ~Tb, c & ~Tc, w)
    base_c = f(a, b, c, w)
    T_c = hi != lo

    base = _g(a_s, b_s, base_c, is_sub)
    terms = []
    for da, db, dc in itertools.product([False, True], repeat=3):
        guard = []
        if da:
            guard.append(Ta_s)
        if db:
            guard.append(Tb_s)
        if dc:
            guard.append(T_c)
        x = Xor(a_s, True) if da else a_s
        y = Xor(b_s, True) if db else b_s
        z = Xor(base_c, True) if dc else base_c
        cond = And(*guard) if guard else And(True)
        terms.append(And(cond, _g(x, y, z, is_sub) != base))
    return Or(*terms)


# --------------------------------------------------------------------------- #
# queries
# --------------------------------------------------------------------------- #
def _verdict(solver: Solver) -> str:
    r = solver.check()
    if r == unsat:
        return 'PROVED'
    return 'CEX' if str(r) == 'sat' else 'UNKNOWN'


def _carry_domain(c, w):
    """c is zext of a 1-bit flag, so it is 0 or 1 -- the decomposition's premise."""
    return ULE(c, BitVecVal(1, w))


def prove_chain_equals_sum(w: int, is_sub: bool) -> str:
    a, b, c = BitVec('a', w), BitVec('b', w), BitVec('c', w)
    s = Solver()
    s.add(_carry_domain(c, w))
    s.add(_spec_chain(a, b, c, w, is_sub) != _spec_sum(a, b, c, w, is_sub))
    return _verdict(s)


def prove_identity(w: int, is_sub: bool) -> str:
    a, b, c = BitVec('a', w), BitVec('b', w), BitVec('c', w)
    s = Solver()
    s.add(_carry_domain(c, w))
    s.add(_decomp(a, b, c, w, is_sub) != _spec_sum(a, b, c, w, is_sub))
    return _verdict(s)


def prove_monotone_reachability(w: int, is_sub: bool) -> str:
    """The polarised corners bracket the monotone carry/borrow exactly.

    Both directions: the differential XOR fires iff some point of the cube really
    changes Car3/Bor3.
    """
    a, b, c = BitVec('a', w), BitVec('b', w), BitVec('c', w)
    Ta, Tb, Tc = BitVec('Ta', w), BitVec('Tb', w), BitVec('Tc', w)
    ap, bp, cp = BitVec('ap', w), BitVec('bp', w), BitVec('cp', w)
    f = _bor3 if is_sub else _car3
    if is_sub:
        hi, lo = f(a & ~Ta, b | Tb, c | Tc, w), f(a | Ta, b & ~Tb, c & ~Tc, w)
    else:
        hi, lo = f(a | Ta, b | Tb, c | Tc, w), f(a & ~Ta, b & ~Tb, c & ~Tc, w)
    in_cube = And(
        (ap & ~Ta) == (a & ~Ta),
        (bp & ~Tb) == (b & ~Tb),
        (cp & ~Tc) == (c & ~Tc),
        _carry_domain(cp, w),
    )
    s = Solver()
    s.add(_carry_domain(c, w), ULE(Tc, BitVecVal(1, w)))
    # direction 1: a real change in the cube but the corners agree
    s.add(
        Or(
            And(in_cube, f(ap, bp, cp, w) != f(a, b, c, w), hi == lo),
            # direction 2: corners disagree but nothing in the cube changes it
            And(
                hi != lo,
                ForAll(
                    [ap, bp, cp],
                    Or(Not(in_cube), f(ap, bp, cp, w) == f(a, b, c, w)),
                ),
            ),
        ),
    )
    return _verdict(s)


def prove_soundness(w: int, is_sub: bool) -> str:
    """NO UNDER-TAINT: a witness that flips OF implies the rule fires."""
    a, b, c = BitVec('a', w), BitVec('b', w), BitVec('c', w)
    Ta, Tb, Tc = BitVec('Ta', w), BitVec('Tb', w), BitVec('Tc', w)
    ap, bp, cp = BitVec('ap', w), BitVec('bp', w), BitVec('cp', w)
    in_cube = And(
        (ap & ~Ta) == (a & ~Ta),
        (bp & ~Tb) == (b & ~Tb),
        (cp & ~Tc) == (c & ~Tc),
        _carry_domain(cp, w),
    )
    s = Solver()
    s.add(_carry_domain(c, w), ULE(Tc, BitVecVal(1, w)))
    s.add(in_cube)
    s.add(_spec_sum(ap, bp, cp, w, is_sub) != _spec_sum(a, b, c, w, is_sub))
    s.add(Not(_rule(a, b, c, Ta, Tb, Tc, w, is_sub)))
    return _verdict(s)


def prove_precision(w: int, is_sub: bool) -> str:
    """NO OVER-TAINT: the rule firing implies a witness really exists."""
    a, b, c = BitVec('a', w), BitVec('b', w), BitVec('c', w)
    Ta, Tb, Tc = BitVec('Ta', w), BitVec('Tb', w), BitVec('Tc', w)
    ap, bp, cp = BitVec('ap', w), BitVec('bp', w), BitVec('cp', w)
    in_cube = And(
        (ap & ~Ta) == (a & ~Ta),
        (bp & ~Tb) == (b & ~Tb),
        (cp & ~Tc) == (c & ~Tc),
        _carry_domain(cp, w),
    )
    s = Solver()
    s.add(_carry_domain(c, w), ULE(Tc, BitVecVal(1, w)))
    s.add(_rule(a, b, c, Ta, Tb, Tc, w, is_sub))
    s.add(
        ForAll(
            [ap, bp, cp],
            Or(Not(in_cube), _spec_sum(ap, bp, cp, w, is_sub) == _spec_sum(a, b, c, w, is_sub)),
        ),
    )
    return _verdict(s)


def main() -> int:
    ap_ = argparse.ArgumentParser(description=__doc__)
    ap_.add_argument('--widths', default='2,3,4,5,6,7,8,16,32,64')
    ap_.add_argument('--precision-max-width', type=int, default=5)
    args = ap_.parse_args()
    widths = [int(x) for x in args.widths.split(',')]

    bad = 0
    for is_sub in (False, True):
        fam = 'SBORROW (sbb)' if is_sub else 'SCARRY (adc)'
        print(f'\n=== carry-in signed overflow, {fam} ===')
        print(f'{"w":>4}  {"chain=sum":>10}  {"identity":>9}  {"reach":>8}  {"no-under":>9}  {"no-over":>8}')
        for w in widths:
            chain = prove_chain_equals_sum(w, is_sub)
            ident = prove_identity(w, is_sub)
            reach = prove_monotone_reachability(w, is_sub) if w <= args.precision_max_width else '-'
            sound = prove_soundness(w, is_sub)
            prec = prove_precision(w, is_sub) if w <= args.precision_max_width else '-'
            print(f'{w:>4}  {chain:>10}  {ident:>9}  {reach:>8}  {sound:>9}  {prec:>8}')
            bad += sum(1 for v in (chain, ident, reach, sound, prec) if v == 'CEX')

    print()
    if bad:
        print(f'FAILED: {bad} counterexample(s) -- the rule is WRONG')
        return 1
    print('No counterexamples.  UNKNOWN entries are solver limits, not evidence.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
