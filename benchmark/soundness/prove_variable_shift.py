#!/usr/bin/env python3
"""Z3 proof of the reachable-amount subcube rule for shifts by a tainted amount.

A tainted shift amount used to force AvalancheExpr (the whole output width goes
tainted); measured over a 2M-case five-ISA campaign that was 69.8% of ALL
over-tainted bits.  VariableShiftTaintExpr replaces it with

    S = { s0 + sum_{j in T_s} b_j * 2^j },        s0 = V_s & ~T_s

    sm(y, op):  r = y >>> s0
                for j in 0 .. lg-1:
                    if T_s[j]:  r = r op (r >>> 2^j)

    T_r = sm(T_x, OR) | ( sm(x, OR) ^ sm(x, AND) )

The reachable amount set is a SUBCUBE, and log-fold doubling enumerates a subcube
implicitly, so OR and AND over the whole set need no enumeration -- 2*log2(w)
steps, independent of how many bits are tainted or of their values.

The two terms are the two clauses of noninterference:

  * sm(T_x, OR)              some reachable amount brings a TAINTED source bit to
                             this output position;
  * sm(x,OR) ^ sm(x,AND)     the CLEAN value at this position differs across
                             reachable amounts.  Avalanche cannot express this and
                             an interval bound gets it wrong.

Theorems discharged, per (width, amount-mask, shift kind):

  no-under-taint   two points of the input cube whose outputs differ at bit i
                   imply the rule fires at bit i.  THIS IS THE SOUNDNESS PROPERTY.
                   The refutation query is QUANTIFIER-FREE (both witnesses are
                   existential), so it scales to the engine's real widths.
  no-over-taint    the rule firing at bit i implies two such points exist.  Needs
                   a ForAll to say "no witness exists", so Z3 tops out low.

The amount mask matters and is swept: ARM64/x86 mask to w-1 so the amount is
always < w, but PPC masks a 32-bit shift with 0x3f, so the amount can REACH OR
EXCEED the width and the shift saturates.  The implementation folds tainted
amount bits above log2(w) into a full low sweep, and that fold is exactly what
the wide-mask rows here are checking.

Verdicts are reported honestly:
  PROVED  = unsat   (no counterexample exists)
  CEX     = sat     (a real counterexample: the rule is WRONG)
  UNKNOWN = unknown (Z3 gave up; NOT evidence either way)

Run:  uv run --with z3-solver python prove_variable_shift.py
      [--widths 4,8,16,32,64] [--precision-max-width 8]
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
    If,
    LShR,
    Not,
    Or,
    Solver,
    UGE,
    unsat,
)

LEFT, LOGICAL_RIGHT, ARITH_RIGHT = 0, 1, 2
KIND_NAME = {LEFT: 'INT_LEFT', LOGICAL_RIGHT: 'INT_RIGHT', ARITH_RIGHT: 'INT_SRIGHT'}


def _shift(y, s, kind):
    """One shift, matching p-code (and Z3) saturating semantics for s >= width."""
    if kind == LEFT:
        return y << s
    if kind == LOGICAL_RIGHT:
        return LShR(y, s)
    return y >> s  # Z3 >> on BitVec is arithmetic


def _smear(y, ts, s0, kind, w, lg, is_and):
    """OR/AND of `y` shifted by every amount in the reachable subcube.

    Mirrors VariableShiftTaintExpr._smear exactly, including the fold of tainted
    amount bits above log2(w) into a full low sweep.
    """
    # The low sweep covers exactly the NON-saturating amounts; amounts >= w are
    # folded in by the caller (see `rule`).
    low_mask = BitVecVal((1 << lg) - 1, w)
    ts_lo = ts & low_mask

    r = _shift(y, s0, kind)
    for j in range(lg):
        bit = Extract(j, j, ts_lo) == BitVecVal(1, 1)
        shifted = _shift(r, BitVecVal(1 << j, w), kind)
        combined = (r & shifted) if is_and else (r | shifted)
        r = If(bit, combined, r)
    return r


def rule(x, tx, sv, ts, kind, w, amt_mask):
    """T_r as VariableShiftTaintExpr computes it."""
    lg = (w - 1).bit_length()
    m = BitVecVal(amt_mask, w)
    ts_m = ts & m
    s0 = (sv & m) & ~ts_m
    reach = _smear(tx, ts_m, s0, kind, w, lg, False)
    hi = _smear(x, ts_m, s0, kind, w, lg, False)
    lo = _smear(x, ts_m, s0, kind, w, lg, True)

    # Saturating amounts.  The smear reaches s0 .. s0 + 2^lg - 1, i.e. every amount
    # below the width.  When the amount mask is wider than log2(w) (PPC masks a
    # 32-bit shift with 0x3f) an amount >= w is reachable too and p-code saturates
    # it, producing an output the smear never sees.  Fold it in explicitly.
    full = BitVecVal((1 << w) - 1, w)
    zero = BitVecVal(0, w)
    sat = UGE(s0 | ts_m, BitVecVal(w, w))
    if kind == ARITH_RIGHT:
        sign_tainted = Extract(w - 1, w - 1, tx) == BitVecVal(1, 1)
        sign_set = Extract(w - 1, w - 1, x) == BitVecVal(1, 1)
        hi = If(sat, If(Or(sign_tainted, sign_set), hi | full, hi), hi)
        lo = If(sat, If(sign_tainted, zero, If(sign_set, lo, zero)), lo)
        reach = If(And(sat, sign_tainted), reach | full, reach)
    else:
        lo = If(sat, zero, lo)
    return reach | (hi ^ lo)


def _out(x, sv, kind, w, amt_mask):
    """The instruction's actual output: shift by the masked amount."""
    return _shift(x, sv & BitVecVal(amt_mask, w), kind)


def _in_cube(xp, sp, x, tx, sv, ts):
    return And((xp & ~tx) == (x & ~tx), (sp & ~ts) == (sv & ~ts))


def _verdict(s: Solver, timeout_ms: int = 120_000) -> str:
    """Discharge a refutation query, bounded so one hard instance cannot stall the run."""
    s.set('timeout', timeout_ms)
    r = s.check()
    if r == unsat:
        return 'PROVED'
    return 'CEX' if str(r) == 'sat' else 'UNKNOWN'


def prove_soundness(w: int, amt_mask: int, kind: int, timeout_ms: int = 120_000) -> str:
    """NO UNDER-TAINT: any two cube points differing at bit i => rule fires at i."""
    x, tx, sv, ts = BitVec('x', w), BitVec('tx', w), BitVec('s', w), BitVec('ts', w)
    x1, s1, x2, s2 = BitVec('x1', w), BitVec('s1', w), BitVec('x2', w), BitVec('s2', w)
    r = rule(x, tx, sv, ts, kind, w, amt_mask)
    o1 = _out(x1, s1, kind, w, amt_mask)
    o2 = _out(x2, s2, kind, w, amt_mask)
    s = Solver()
    s.add(_in_cube(x1, s1, x, tx, sv, ts), _in_cube(x2, s2, x, tx, sv, ts))
    # some bit differs between the two outputs, yet the rule does not claim it
    s.add((o1 ^ o2) & ~r != BitVecVal(0, w))
    return _verdict(s, timeout_ms)


def prove_precision(w: int, amt_mask: int, kind: int, timeout_ms: int = 120_000) -> str:
    """NO OVER-TAINT: the rule firing at bit i => two cube points differ at i."""
    x, tx, sv, ts = BitVec('x', w), BitVec('tx', w), BitVec('s', w), BitVec('ts', w)
    x1, s1, x2, s2 = BitVec('x1', w), BitVec('s1', w), BitVec('x2', w), BitVec('s2', w)
    r = rule(x, tx, sv, ts, kind, w, amt_mask)
    o1 = _out(x1, s1, kind, w, amt_mask)
    o2 = _out(x2, s2, kind, w, amt_mask)
    s = Solver()
    s.add(r != BitVecVal(0, w))
    # ... yet no pair of cube points realises ANY of the claimed bits
    s.add(
        ForAll(
            [x1, s1, x2, s2],
            Or(
                Not(And(_in_cube(x1, s1, x, tx, sv, ts), _in_cube(x2, s2, x, tx, sv, ts))),
                (o1 ^ o2) & r == BitVecVal(0, w),
            ),
        ),
    )
    return _verdict(s, timeout_ms)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--widths', default='4,8,16,32,64')
    ap.add_argument('--precision-max-width', type=int, default=8)
    ap.add_argument('--timeout', type=int, default=120, help='per-query seconds')
    args = ap.parse_args()
    widths = [int(x) for x in args.widths.split(',')]

    bad = 0
    for kind in (LEFT, LOGICAL_RIGHT, ARITH_RIGHT):
        print(f'\n=== {KIND_NAME[kind]} ===')
        print(f'{"w":>4} {"amt_mask":>10}  {"no-under":>9}  {"no-over":>8}')
        for w in widths:
            # w-1: amount always < width (ARM64, x86, MIPS, RISC-V)
            # 2w-1: amount can reach or exceed the width and saturate (PPC slw/srw)
            for label, mask in (('w-1', w - 1), ('2w-1', 2 * w - 1)):
                tmo = args.timeout * 1000
                sound = prove_soundness(w, mask, kind, tmo)
                prec = prove_precision(w, mask, kind, tmo) if w <= args.precision_max_width else '-'
                print(f'{w:>4} {label:>10}  {sound:>9}  {prec:>8}', flush=True)
                bad += sum(1 for v in (sound, prec) if v == 'CEX')

    print()
    if bad:
        print(f'FAILED: {bad} counterexample(s) -- the rule is WRONG')
        return 1
    print('No counterexamples.  UNKNOWN entries are solver limits, not evidence.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
