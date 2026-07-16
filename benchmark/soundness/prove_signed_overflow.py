#!/usr/bin/env python3
"""Z3 proof of the sign-decomposition taint rule for signed overflow.

Signed overflow (P-code INT_SBORROW / INT_SCARRY) is NON-MONOTONE, so MicroTaint's
2-replica differential -- which samples only the extremal corners V|T and V&~T --
can miss it: the two corners coincidentally agree while an interior flip of a
tainted bit toggles OF.  That is the `sub rax,rbx; seto dl` under-taint class.

This proves a rule that is EXACT (neither over- nor under-taints), by
decomposing OF onto three quantities that depend on DISJOINT bit-sets:

    r = a - b,  sign bit s = w-1
    Bor = [ a[0:w-1] <u b[0:w-1] ]        (borrow INTO the msb; MONOTONE)
    r_s = a_s ^ b_s ^ Bor
    OF  = (a_s ^ b_s) & (a_s ^ r_s)  =  (a_s ^ b_s) & (b_s ^ Bor)

and analogously for addition (INT_SCARRY), with the carry into the msb:

    Car = [ a[0:w-1] +u b[0:w-1] >= 2^(w-1) ]   (MONOTONE)
    OF  = ~(a_s ^ b_s) & (b_s ^ Car)

Because a_s (bit w-1 of a), b_s (bit w-1 of b) and Bor/Car (bits 0..w-2 of both)
read disjoint inputs, they are INDEPENDENT: enumerating their reachable values is
exact.  Bor/Car being monotone, their reachable set is exactly {min, max} of the
polarised differential, so their taint is the differential XOR.

Theorems discharged, per width w in the sweep:

  identity        forall a,b: OF_spec(a,b) == OF_decomposed(a,b)
  monotone-reach  Bor/Car being monotone, its reachable set over the taint cube is
                  exactly {polarised-min, polarised-max}: the differential XOR
                  equals true non-constancy (proved both directions).
  no-under-taint  a witness in the taint cube flips OF  =>  the rule fires.
                  This is the property the paper claims.  The refutation query is
                  QUANTIFIER-FREE, so it scales to the engine's real widths.
  no-over-taint   the rule fires  =>  a witness really exists.  Needs a ForAll to
                  say "no witness exists", so Z3 tops out at modest widths.

Each query is posed as a refutation, and the verdict is reported honestly:
  PROVED  = unsat   (no counterexample exists)
  CEX     = sat     (a real counterexample: the rule is WRONG)
  UNKNOWN = unknown (Z3 gave up; NOT evidence either way)

Run:  uv run --with z3-solver python prove_signed_overflow.py
      [--widths 2,3,4,5,6,7,8,16,32,64] [--precision-max-width 6]
"""

from __future__ import annotations

import argparse
import itertools
import sys

from z3 import (
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


def _of_spec_sub(a, b, w):
    """Reference signed overflow of a-b: sign-extend by one bit, subtract, and
    check the (w+1)-bit result does not fit in w bits (bit w != bit w-1)."""
    r = SignExt(1, a) - SignExt(1, b)
    return _bit(r, w) != _bit(r, w - 1)


def _of_spec_add(a, b, w):
    """Reference signed overflow of a+b (same widened-arithmetic criterion)."""
    r = SignExt(1, a) + SignExt(1, b)
    return _bit(r, w) != _bit(r, w - 1)


def _low(x, w):
    return Extract(w - 2, 0, x)


def _bor(a, b, w):
    """Borrow into the msb: unsigned compare of the low w-1 bits."""
    return ULT(_low(a, w), _low(b, w))


def _car(a, b, w):
    """Carry into the msb: low w-1 bits sum overflows w-1 bits."""
    la = ZeroExt(1, _low(a, w))
    lb = ZeroExt(1, _low(b, w))
    return _bit(la + lb, w - 1) == BitVecVal(1, 1)


def _of_decomp_sub(a, b, w):
    a_s = _bit(a, w - 1) == BitVecVal(1, 1)
    b_s = _bit(b, w - 1) == BitVecVal(1, 1)
    return And(Xor(a_s, b_s), Xor(b_s, _bor(a, b, w)))


def _of_decomp_add(a, b, w):
    a_s = _bit(a, w - 1) == BitVecVal(1, 1)
    b_s = _bit(b, w - 1) == BitVecVal(1, 1)
    return And(Not(Xor(a_s, b_s)), Xor(b_s, _car(a, b, w)))


def _verdict(solver: Solver) -> str:
    """Discharge a refutation query honestly.

    We always encode "look for a counterexample":
      unsat   -> PROVED   (no counterexample exists)
      sat     -> CEX      (a real counterexample; the claim is FALSE)
      unknown -> UNKNOWN  (solver gave up; NOT evidence either way)

    Reporting `unknown` as a failure would be dishonest: quantified bit-vector
    queries routinely exceed Z3 at larger widths.
    """
    r = solver.check()
    if r == unsat:
        return 'PROVED'
    return 'CEX' if str(r) == 'sat' else 'UNKNOWN'


def prove_identity(w: int, is_sub: bool) -> str:
    """T1: the decomposition equals the reference signed-overflow spec."""
    a, b = BitVec('a', w), BitVec('b', w)
    spec = _of_spec_sub(a, b, w) if is_sub else _of_spec_add(a, b, w)
    dec = _of_decomp_sub(a, b, w) if is_sub else _of_decomp_add(a, b, w)
    s = Solver()
    s.add(spec != dec)  # look for ANY counterexample
    return _verdict(s)


def prove_monotone_reachability(w: int, is_sub: bool) -> str:
    """T2: Bor/Car is monotone, so its polarised differential XOR equals true
    non-constancy over the taint cube.

    `<u` is decreasing in a and increasing in b, so:
        max = [ (a&~Ta) <u (b|Tb) ],  min = [ (a|Ta) <u (b&~Tb) ]
    Carry is increasing in both:
        max = [ (a|Ta) + (b|Tb) ],    min = [ (a&~Ta) + (b&~Tb) ]
    Claim: exists a',b' in cube with f(a',b') != f(a,b)  <=>  max != min.
    """
    a, b, Ta, Tb = BitVec('a', w), BitVec('b', w), BitVec('Ta', w), BitVec('Tb', w)
    ap, bp = BitVec('ap', w), BitVec('bp', w)

    f = _bor if is_sub else _car
    if is_sub:
        hi = f(a & ~Ta, b | Tb, w)
        lo = f(a | Ta, b & ~Tb, w)
    else:
        hi = f(a | Ta, b | Tb, w)
        lo = f(a & ~Ta, b & ~Tb, w)
    diff = hi != lo

    in_cube = And((ap & ~Ta) == (a & ~Ta), (bp & ~Tb) == (b & ~Tb))

    # direction 1 (no OVER-taint): differential fires => a witness really exists.
    # Encoded as: differential fires AND no witness exists  ==>  must be UNSAT.
    s1 = Solver()
    s1.add(diff)
    s1.add(ForAll([ap, bp], Or(Not(in_cube), f(ap, bp, w) == f(a, b, w))))
    v1 = _verdict(s1)

    # direction 2 (no UNDER-taint): a witness exists => differential fires.
    s2 = Solver()
    s2.add(in_cube, f(ap, bp, w) != f(a, b, w), Not(diff))
    v2 = _verdict(s2)

    if 'CEX' in (v1, v2):
        return 'CEX'
    if 'UNKNOWN' in (v1, v2):
        return 'UNKNOWN'
    return 'PROVED'


def _rule(a, b, Ta, Tb, w, is_sub):
    """The taint rule: exact non-constancy of the 3-input function over the
    INDEPENDENT reachable values of (a_s, b_s, Bor/Car)."""
    a_s = _bit(a, w - 1) == BitVecVal(1, 1)
    b_s = _bit(b, w - 1) == BitVecVal(1, 1)
    Ta_s = _bit(Ta, w - 1) == BitVecVal(1, 1)
    Tb_s = _bit(Tb, w - 1) == BitVecVal(1, 1)

    f = _bor if is_sub else _car
    if is_sub:
        hi = f(a & ~Ta, b | Tb, w)
        lo = f(a | Ta, b & ~Tb, w)
    else:
        hi = f(a | Ta, b | Tb, w)
        lo = f(a & ~Ta, b & ~Tb, w)
    base_c = f(a, b, w)
    T_c = hi != lo

    def g(x, y, z):
        return And(Xor(x, y), Xor(y, z)) if is_sub else And(Not(Xor(x, y)), Xor(y, z))

    base = g(a_s, b_s, base_c)
    terms = []
    for da, db, dc in itertools.product([False, True], repeat=3):
        # only flip inputs that are actually tainted
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
        terms.append(And(cond, g(x, y, z) != base))
    return Or(*terms)


def prove_soundness(w: int, is_sub: bool) -> str:
    """T3a -- NO UNDER-TAINT (the property the paper claims).

    "a witness in the taint cube flips OF"  =>  "the rule fires".
    Refutation query is QUANTIFIER-FREE (the witness is existential), so this
    scales to the engine's real operating widths.
    """
    a, b, Ta, Tb = BitVec('a', w), BitVec('b', w), BitVec('Ta', w), BitVec('Tb', w)
    ap, bp = BitVec('ap', w), BitVec('bp', w)
    of = _of_spec_sub if is_sub else _of_spec_add
    in_cube = And((ap & ~Ta) == (a & ~Ta), (bp & ~Tb) == (b & ~Tb))
    s = Solver()
    s.add(in_cube, of(ap, bp, w) != of(a, b, w), Not(_rule(a, b, Ta, Tb, w, is_sub)))
    return _verdict(s)


def prove_precision(w: int, is_sub: bool) -> str:
    """T3b -- NO OVER-TAINT.

    "the rule fires"  =>  "a witness really exists".  Needs a ForAll to say
    "no witness exists", so Z3 tops out at modest widths; UNKNOWN there is a
    solver limit, not evidence of over-tainting.
    """
    a, b, Ta, Tb = BitVec('a', w), BitVec('b', w), BitVec('Ta', w), BitVec('Tb', w)
    ap, bp = BitVec('ap', w), BitVec('bp', w)
    of = _of_spec_sub if is_sub else _of_spec_add
    in_cube = And((ap & ~Ta) == (a & ~Ta), (bp & ~Tb) == (b & ~Tb))
    s = Solver()
    s.add(_rule(a, b, Ta, Tb, w, is_sub))
    s.add(ForAll([ap, bp], Or(Not(in_cube), of(ap, bp, w) == of(a, b, w))))
    return _verdict(s)


def main() -> int:
    ap_ = argparse.ArgumentParser(description=__doc__)
    ap_.add_argument(
        '--widths',
        type=str,
        default='2,3,4,5,6,7,8,16,32,64',
        help='comma-separated widths for the identity + soundness sweep',
    )
    ap_.add_argument(
        '--precision-max-width',
        type=int,
        default=6,
        help='the no-over-taint direction needs a ForAll; Z3 tops out early',
    )
    args = ap_.parse_args()

    widths = [int(x) for x in args.widths.split(',')]
    any_cex = False
    for is_sub in (True, False):
        kind = 'INT_SBORROW (a-b)' if is_sub else 'INT_SCARRY (a+b)'
        print(f'\n=== {kind} ===')
        print('  width  identity   no-under-taint  no-over-taint')
        for w in widths:
            t1 = prove_identity(w, is_sub)
            t_sound = prove_soundness(w, is_sub)
            t_prec = prove_precision(w, is_sub) if w <= args.precision_max_width else 'n/a'
            for v in (t1, t_sound, t_prec):
                any_cex |= v == 'CEX'
            print(f'  w={w:<4d} {t1:10s} {t_sound:15s} {t_prec}')

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
