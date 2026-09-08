#!/usr/bin/env python3
"""Z3 proof of the bounded-fill taint rule for multiply with tainted operands.

Multiply genuinely couples every input bit into every higher output bit, so it
cannot be made exact cheaply; it used to avalanche the full output width and was
26.7% of all over-tainted bits in the campaign.  VariableMultiplyTaintExpr
replaces the avalanche with a FILL between two provable bounds on which output
bits can vary.

The 2-corner differential ALONE is NOT sound here, which is the first thing this
script checks.  ``max ^ min`` only sees where the two EXTREME products differ, and
an INTERIOR product can flip a bit BELOW that::

    a in {2,6}, b in {1,3}  ->  products {2, 6, 18}
    max ^ min = 18 ^ 2 = 16 (bit 4 alone), yet bit 2 is tainted by the interior 6

The sound bounds are:

    L = tz_lo(a) + tz_lo(b)     tz_lo(v) = lowest bit that CAN be 1.  Every
                                reachable product is divisible by 2^L, so bits
                                below L are 0 in ALL of them.

    H = highbit(max ^ min)      on the FULL 2w-bit product.  All reachable
                                products lie in [min,max], so they agree on every
                                bit ABOVE the highest bit where the extremes
                                differ.  Computed on the full product and sliced
                                afterwards, because the low word (a*b mod 2^w) is
                                NOT monotone.

    T = ones[L..H], then the returned window is extracted.

Signedness affects only the high half (the low word is bit-identical either way).
A tainted SIGN bit makes the reachable operand a NON-INTERVAL -- a positive block
and a negative block -- so corner min/max is meaningless and the product sign can
flip; the high bits then smear to the top.  With the sign fixed, the signed
corners of the operand box give min/max as usual.

Theorems discharged, per (width, signedness, output window):

  interior-cex     the naive `fill from lowbit(max^min)` IS refutable.  Reported
                   as EXPECTED-CEX: finding a counterexample here CONFIRMS why the
                   tz_lo bound is needed.  (unsat would mean the naive rule was
                   fine and this term is over-engineered.)
  low-bound        every reachable product is divisible by 2^L.
  no-under-taint   two points of the cube whose windows differ at bit i imply the
                   fill covers bit i.  THIS IS THE SOUNDNESS PROPERTY.

No no-over-taint theorem is claimed: the rule is a FILL and deliberately sets
bits inside [L,H] that may not vary.  Its imprecision is measured empirically
(over_bits/gt_bits in the campaign), not proved away.

Verdicts:
  PROVED  = unsat   (no counterexample exists)
  CEX     = sat     (a real counterexample: the rule is WRONG)
  UNKNOWN = unknown (Z3 gave up; NOT evidence either way)

Run:  uv run --with z3-solver python prove_variable_multiply.py
      [--widths 3,4,5,6,8] [--timeout 60]
"""

from __future__ import annotations

import argparse
import sys

from z3 import (
    UGE,
    And,
    BitVec,
    BitVecVal,
    Concat,
    Extract,
    If,
    LShR,
    Or,
    Solver,
    unsat,
)

LOW, HIGH, FULL = 'low', 'high', 'full'


def _widen(v, w):
    """Zero-extend a w-bit vector to 2w bits."""
    return Concat(BitVecVal(0, w), v)


def _swiden(v, w):
    """Sign-extend a w-bit vector to 2w bits."""
    return Concat(If(Extract(w - 1, w - 1, v) == BitVecVal(1, 1),
                     BitVecVal((1 << w) - 1, w), BitVecVal(0, w)), v)


def _product(a, b, w, signed):
    """The full 2w-bit product, as the hardware computes it."""
    if signed:
        return _swiden(a, w) * _swiden(b, w)
    return _widen(a, w) * _widen(b, w)


def _tz_lo(v, t, w):
    """Lowest bit position that CAN be 1, as a 2w-bit count (w if none)."""
    poss = (v & ~t) | t
    acc = BitVecVal(w, 2 * w)
    for i in range(w - 1, -1, -1):
        acc = If(Extract(i, i, poss) == BitVecVal(1, 1), BitVecVal(i, 2 * w), acc)
    return acc


def _highbit_mask(d, w):
    """ones[0 .. highbit(d)] for a 2w-bit d; 0 when d == 0.

    Built by the standard smear-right so it stays a pure bitvector term.
    """
    r = d
    shift = 1
    while shift < 2 * w:
        r = r | LShR(r, shift)
        shift *= 2
    return r


def _fill(a, ta, b, tb, w, signed):
    """The rule: ones[L..H] over the full 2w-bit product."""
    fw = 2 * w
    lo = _tz_lo(a, ta, w) + _tz_lo(b, tb, w)

    sign_tainted = Or(Extract(w - 1, w - 1, ta) == BitVecVal(1, 1),
                      Extract(w - 1, w - 1, tb) == BitVecVal(1, 1))

    if signed:
        alo, ahi = _swiden(a & ~ta, w), _swiden(a | ta, w)
        blo, bhi = _swiden(b & ~tb, w), _swiden(b | tb, w)
        c1, c2, c3, c4 = alo * blo, alo * bhi, ahi * blo, ahi * bhi

        def smin(x, y):
            return If(x <= y, x, y)

        def smax(x, y):
            return If(x <= y, y, x)
        pmin = smin(smin(c1, c2), smin(c3, c4))
        pmax = smax(smax(c1, c2), smax(c3, c4))
    else:
        pmin = _widen(a & ~ta, w) * _widen(b & ~tb, w)
        pmax = _widen(a | ta, w) * _widen(b | tb, w)

    d = pmin ^ pmax
    below_lo = If(lo >= BitVecVal(fw, fw), BitVecVal(-1, fw), (BitVecVal(1, fw) << lo) - 1)
    # ones[0..H] via smear-right of d; all-ones when the sign can flip.
    upto_hi = If(And(sign_tainted, BitVecVal(1 if signed else 0, 1) == BitVecVal(1, 1)),
                 BitVecVal(-1, fw), _highbit_mask(d, w))
    return upto_hi & ~below_lo


def _window(p, w, mode):
    fw = 2 * w
    if mode == LOW:
        return Extract(w - 1, 0, p)
    if mode == HIGH:
        return Extract(fw - 1, w, p)
    return p


def _in_cube(ap, bp, a, ta, b, tb):
    return And((ap & ~ta) == (a & ~ta), (bp & ~tb) == (b & ~tb))


def _verdict(s: Solver, timeout_ms: int) -> str:
    s.set('timeout', timeout_ms)
    r = s.check()
    if r == unsat:
        return 'PROVED'
    return 'CEX' if str(r) == 'sat' else 'UNKNOWN'


def prove_low_bound(w: int, signed: bool, tmo: int) -> str:
    """Every reachable product is divisible by 2^(tz_lo(a)+tz_lo(b))."""
    a, ta, b, tb = BitVec('a', w), BitVec('ta', w), BitVec('b', w), BitVec('tb', w)
    ap, bp = BitVec('ap', w), BitVec('bp', w)
    fw = 2 * w
    lo = _tz_lo(a, ta, w) + _tz_lo(b, tb, w)
    below = If(UGE(lo, BitVecVal(fw, fw)), BitVecVal(-1, fw), (BitVecVal(1, fw) << lo) - 1)
    s = Solver()
    s.add(_in_cube(ap, bp, a, ta, b, tb))
    s.add(_product(ap, bp, w, signed) & below != BitVecVal(0, fw))
    return _verdict(s, tmo)


def prove_soundness(w: int, signed: bool, mode: str, tmo: int) -> str:
    """NO UNDER-TAINT: any two cube points differing at bit i => the fill covers i."""
    a, ta, b, tb = BitVec('a', w), BitVec('ta', w), BitVec('b', w), BitVec('tb', w)
    a1, b1, a2, b2 = BitVec('a1', w), BitVec('b1', w), BitVec('a2', w), BitVec('b2', w)
    r = _window(_fill(a, ta, b, tb, w, signed), w, mode)
    o1 = _window(_product(a1, b1, w, signed), w, mode)
    o2 = _window(_product(a2, b2, w, signed), w, mode)
    s = Solver()
    s.add(_in_cube(a1, b1, a, ta, b, tb), _in_cube(a2, b2, a, ta, b, tb))
    s.add((o1 ^ o2) & ~r != BitVecVal(0, o1.size()))
    return _verdict(s, tmo)


def refute_naive(w: int, mode: str, tmo: int) -> str:
    """The naive `fill from lowbit(max^min)` should be REFUTABLE (sat = expected)."""
    a, ta, b, tb = BitVec('a', w), BitVec('ta', w), BitVec('b', w), BitVec('tb', w)
    a1, b1, a2, b2 = BitVec('a1', w), BitVec('b1', w), BitVec('a2', w), BitVec('b2', w)
    fw = 2 * w
    pmin = _widen(a & ~ta, w) * _widen(b & ~tb, w)
    pmax = _widen(a | ta, w) * _widen(b | tb, w)
    d = pmin ^ pmax
    lowest = d & (~d + BitVecVal(1, fw))  # isolate lowest set bit
    naive = _highbit_mask(d, w) & ~(lowest - BitVecVal(1, fw))
    r = _window(naive, w, mode)
    o1 = _window(_product(a1, b1, w, False), w, mode)
    o2 = _window(_product(a2, b2, w, False), w, mode)
    s = Solver()
    s.add(_in_cube(a1, b1, a, ta, b, tb), _in_cube(a2, b2, a, ta, b, tb))
    s.add(d != BitVecVal(0, fw))
    s.add((o1 ^ o2) & ~r != BitVecVal(0, o1.size()))
    v = _verdict(s, tmo)
    # sat here means "the naive rule under-taints", which is what we want to show.
    return {'CEX': 'REFUTED(good)', 'PROVED': 'SOUND(!!)', 'UNKNOWN': 'UNKNOWN'}[v]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--widths', default='3,4,5,6,8')
    ap.add_argument('--timeout', type=int, default=60, help='per-query seconds')
    args = ap.parse_args()
    widths = [int(x) for x in args.widths.split(',')]
    tmo = args.timeout * 1000

    bad = 0
    print('=== the naive differential fill must be REFUTABLE (this motivates tz_lo) ===')
    print(f'{"w":>4} {"window":>7}  {"verdict":>15}')
    for w in widths:
        for mode in (FULL, HIGH, LOW):
            v = refute_naive(w, mode, tmo)
            print(f'{w:>4} {mode:>7}  {v:>15}', flush=True)
            if v == 'SOUND(!!)':
                bad += 1

    for signed in (False, True):
        print(f'\n=== signed={signed} ===')
        print(f'{"w":>4} {"low-bound":>10}  {"under(low)":>11} {"under(high)":>12} {"under(full)":>12}')
        for w in widths:
            lb = prove_low_bound(w, signed, tmo)
            row = [prove_soundness(w, signed, m, tmo) for m in (LOW, HIGH, FULL)]
            print(f'{w:>4} {lb:>10}  {row[0]:>11} {row[1]:>12} {row[2]:>12}', flush=True)
            bad += sum(1 for v in [lb, *row] if v == 'CEX')

    print()
    if bad:
        print(f'FAILED: {bad} counterexample(s) -- the rule is WRONG')
        return 1
    print('No counterexamples.  UNKNOWN entries are solver limits, not evidence.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
