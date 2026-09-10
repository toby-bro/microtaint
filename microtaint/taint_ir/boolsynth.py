"""Minimal expressions for boolean functions of at most three variables.

Flag algebra is where a lifter's verbosity concentrates: a condition code is
built from one-bit intermediates combined by AND/OR/XOR/NOT, and both the value
and the taint of that cone are functions of a handful of one-bit leaves.  Once a
cone is reduced to (leaves, truth table), the question of how to emit it is no
longer about the shape the lifter happened to write -- it is "what is the
cheapest expression with this truth table", which for three variables is a
question with 256 answers that can simply be enumerated once.

The payoff is not the table itself but its degenerate cases.  A ten-operation
comparison whose truth table turns out to be `leaf0` collapses to nothing at
all; `x86` SF is exactly that -- the sign bit's taint, arrived at through
corners, minima and maxima that all cancel.

Search is a cost-ordered enumeration over AND/OR/XOR/NOT of previously found
expressions, so the first expression found for a truth table is a cheapest one.
"""
from __future__ import annotations

from typing import Any

#: A synthesised one-bit expression: ('const', v), ('leaf', i),
#: ('not', expr) or (op, lhs, rhs).
BoolExpr = tuple[Any, ...]

# Truth tables are 8-bit: bit i holds f(v0, v1, v2) for i = v0 | v1<<1 | v2<<2.
LEAF_TT = (0b10101010, 0b11001100, 0b11110000)
_ALL = 0xFF


def _search(max_cost: int = 5) -> dict[int, tuple[int, BoolExpr]]:
    """truth table -> (cost, expression tree), cheapest first.

    Expressions are ('leaf', i), ('const', 0|1), or (op, lhs, rhs) with op in
    and/or/xor, plus ('not', e).  Cost counts emitted machine operations.
    """
    best: dict[int, tuple[int, BoolExpr]] = {0: (0, ('const', 0)),
                                             _ALL: (0, ('const', 1))}
    for i, tt in enumerate(LEAF_TT):
        best[tt] = (0, ('leaf', i))
    frontier = dict(best)
    for cost in range(1, max_cost + 1):
        new: dict[int, tuple[int, BoolExpr]] = {}

        def offer(tt: int, expr: BoolExpr) -> None:
            tt &= _ALL
            if tt in best or tt in new:
                return
            new[tt] = (cost, expr)

        for tt, (_c, e) in frontier.items():
            offer(_ALL ^ tt, ('not', e))
        items = list(best.items())
        for tt_a, (_ca, ea) in items:
            for tt_b, (_cb, eb) in items:
                if tt_a >= tt_b:
                    continue
                offer(tt_a & tt_b, ('and', ea, eb))
                offer(tt_a | tt_b, ('or', ea, eb))
                offer(tt_a ^ tt_b, ('xor', ea, eb))
        if not new:
            break
        best.update(new)
        frontier = new
    return best


BEST = _search()


def expr_for(tt: int) -> BoolExpr | None:
    """Cheapest known expression tree for a truth table, or None."""
    hit = BEST.get(tt & _ALL)
    return hit[1] if hit else None


def expr_cost(tt: int) -> int:
    hit = BEST.get(tt & _ALL)
    return hit[0] if hit else 99


def remap(tt: int, old_leaves: tuple[Any, ...], new_leaves: tuple[Any, ...]) -> int:
    """Re-index a truth table from one leaf ordering into a larger one."""
    pos = [new_leaves.index(x) for x in old_leaves]
    out = 0
    for idx in range(8):
        sub = 0
        for k, p in enumerate(pos):
            if (idx >> p) & 1:
                sub |= 1 << k
        if (tt >> sub) & 1:
            out |= 1 << idx
    return out


def combine(op: str, la: tuple[Any, ...], ta: int, lb: tuple[Any, ...], tb: int,
            limit: int = 3) -> tuple[tuple[Any, ...], int] | None:
    """Merge two (leaves, truth table) pairs under a boolean op.

    Returns (leaves, tt) or None when the union needs more leaves than a
    single-word truth table can index.
    """
    leaves = tuple(sorted(set(la) | set(lb)))
    if len(leaves) > limit:
        return None
    ra = remap(ta, la, leaves)
    rb = remap(tb, lb, leaves)
    if op == 'and':
        tt = ra & rb
    elif op == 'or':
        tt = ra | rb
    elif op == 'xor':
        tt = ra ^ rb
    else:
        raise ValueError(op)
    return leaves, tt & _ALL


def negate(leaves: tuple[Any, ...], tt: int) -> tuple[tuple[Any, ...], int]:
    return leaves, (_ALL ^ tt) & _ALL


def select(lc: tuple[Any, ...], tc: int, la: tuple[Any, ...], ta: int,
           lb: tuple[Any, ...], tb: int,
           limit: int = 3) -> tuple[tuple[Any, ...], int] | None:
    """c ? a : b over one-bit values, as (c & a) | (~c & b)."""
    leaves = tuple(sorted(set(lc) | set(la) | set(lb)))
    if len(leaves) > limit:
        return None
    rc = remap(tc, lc, leaves)
    ra = remap(ta, la, leaves)
    rb = remap(tb, lb, leaves)
    return leaves, ((rc & ra) | ((_ALL ^ rc) & rb)) & _ALL
