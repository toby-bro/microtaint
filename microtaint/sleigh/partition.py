"""Waist detection: recognising an instruction that fuses two distinct operations.

Some ISAs encode two independent functional units in one instruction.  The
canonical example is ARM64's shifted-register operand, where the barrel shifter
feeds the ALU::

    sub x0, x1, x2, asr #5
        INT_SRIGHT  u   <- x2, 5        (bit-permutation unit)
        INT_SUB     x0  <- x1, u        (carry-coupled arithmetic unit)

``slice_backward`` flattens this into one op list and ``determine_category``
must label the whole thing with a single category.  The permutation prefix wins
(TRANSLATABLE), and the carry-arithmetic core silently loses the union floor
that makes a bare ``sub`` sound -- the borrow chain is then only covered by the
2-corner differential, which under-taints.

This module detects the fusion *structurally* rather than by naming opcodes, so
the rule is reusable across ISAs (all of which lift to the same p-code).  A
slice is split at a **waist**: a single intermediate varnode ``v`` such that

  (A) ``v`` is the *only* value flowing from the upstream ops to the downstream
      ops, and the two segments read **disjoint** sets of architectural
      registers.  Disjointness is what makes the split lossless: materialising
      the taint of ``v`` discards the joint distribution of the upstream and
      downstream inputs, and that costs nothing exactly when those inputs do not
      overlap.  Where they reconverge (``ror`` -- one register fanning out to two
      shifts and merging at an OR) there is no waist, and the whole-slice
      differential correctly keeps ownership.

  (B) both segments perform real work (at least one non-routing op each), so
      pure extension/copy plumbing is not mistaken for a second operation; and

  (C) the two segments belong to **different taint algebras** -- the coarse
      three-way class below, not the fine-grained ``InstructionCategory``.  The
      coarse class is essential: ``ubfx`` lifts to a rotate followed by two
      constant masks, which have different categories but are both bitwise, and
      splitting a single field-extract primitive would be over-splitting.

Only the class combination "bitwise/permutation upstream, carry-arithmetic
downstream" is acted on today; that is the one whose fusion is known to drop a
soundness floor.  Other combinations are detected but left to the caller.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pypcode.pypcode_native import PcodeOp, Varnode

from microtaint.instrumentation.ast import BinaryExpr, Constant, Expr, Op
from microtaint.sleigh.constfold import const_value, fold_constants

# Coarse taint algebras.  Two ops share an algebra when a single taint-evaluation
# regime can express their composition; they differ when it cannot.
ALG_BITWISE = 'bitwise'  # output bit depends on a fixed set of input bit positions
ALG_ARITH = 'arith'  # carry/borrow couples bit positions together
ALG_COMPARE = 'compare'  # collapses a word to a non-monotone boolean

_ALGEBRA: dict[str, str] = {
    'INT_LEFT': ALG_BITWISE,
    'INT_RIGHT': ALG_BITWISE,
    'INT_SRIGHT': ALG_BITWISE,
    'INT_AND': ALG_BITWISE,
    'INT_OR': ALG_BITWISE,
    'INT_XOR': ALG_BITWISE,
    'INT_NEGATE': ALG_BITWISE,
    'BOOL_NEGATE': ALG_BITWISE,
    'INT_ADD': ALG_ARITH,
    'INT_SUB': ALG_ARITH,
    'INT_MULT': ALG_ARITH,
    'INT_DIV': ALG_ARITH,
    'INT_SDIV': ALG_ARITH,
    'INT_REM': ALG_ARITH,
    'INT_SREM': ALG_ARITH,
    'INT_2COMP': ALG_ARITH,
    'INT_CARRY': ALG_ARITH,
    'INT_SCARRY': ALG_ARITH,
    'INT_SBORROW': ALG_ARITH,
    'INT_EQUAL': ALG_COMPARE,
    'INT_NOTEQUAL': ALG_COMPARE,
    'INT_LESS': ALG_COMPARE,
    'INT_SLESS': ALG_COMPARE,
    'INT_LESSEQUAL': ALG_COMPARE,
    'INT_SLESSEQUAL': ALG_COMPARE,
}

# Value-preserving plumbing: carries no algebra of its own, so it never counts
# as "real work" when deciding whether a segment is a genuine operation.
ROUTING_OPCODES = frozenset({'COPY', 'SUBPIECE', 'PIECE', 'INT_ZEXT', 'INT_SEXT'})


@dataclass(frozen=True)
class Waist:
    """A validated split point inside one instruction's slice."""

    varnode: Varnode  # the intermediate carrying all flow between the segments
    upstream: list[PcodeOp]  # ops computing `varnode`
    downstream: list[PcodeOp]  # ops consuming it to reach the slice target
    upstream_algebra: str
    downstream_algebra: str
    upstream_regs: set[tuple[int, int]]  # (offset, size) of registers read upstream
    downstream_regs: set[tuple[int, int]]


def _key(vn: Varnode) -> tuple[str, int, int]:
    return (vn.space.name, vn.offset, vn.size)


def _overlaps(a: Varnode, b: Varnode) -> bool:
    """True if two varnodes share any byte.

    Exact-key matching is WRONG here, and silently so.  ARM64's extended-register
    operand lifts as ``COPY unique[41984:4] <- w2 ; INT_ZEXT u <- unique[41984:1]``:
    the ZEXT reads ONE byte of the four the COPY wrote.  With exact keys the
    backward cone stops at the ZEXT, never reaches the register read, and the waist
    is rejected for having no upstream register -- so `add x0,x1,w2,uxtb` and its
    siblings silently lost their floor.  slice_backward is overlap-aware for
    precisely this reason; this module has to match it.
    """
    return (
        a.space.name == b.space.name
        and a.offset < b.offset + b.size
        and b.offset < a.offset + a.size
    )


def _drop_constant_ops(ops: list[PcodeOp]) -> list[PcodeOp]:
    """Ops that only compute a constant are routing, not work on data.

    SLEIGH spells constants as arithmetic: `bic x0,x1,x2,lsl #1` materialises its
    mask as ``unique = -0x1`` (INT_2COMP) and a rotate builds its complementary
    amount as ``0x40 - 0xb`` (INT_SUB).  Both are ALG_ARITH by opcode, which made
    an otherwise purely bitwise segment read as MIXED, so _algebra_of declined and
    the waist was rejected.
    """
    folded = fold_constants(ops)
    return [
        op
        for op in ops
        if op.output is None or _key(op.output) not in folded
    ]


def _is_pow2_scale(op: PcodeOp, folded: dict[tuple[str, int, int], int]) -> int | None:
    """log2 of a constant power-of-two multiplier, or None.

    `x * 2^k` IS a left shift by k: a fixed relocation of bit positions, not a
    carry-mixing multiply.  x86 addressing spells its scale that way
    (`lea rax,[rbx+rcx*4+8]` lifts `rcx*4` as INT_MULT), so without this the scaled
    operand is neither recognised as a permutation nor given a floor at the SHIFTED
    positions.
    """
    if op.opcode.name != 'INT_MULT' or len(op.inputs) != 2:
        return None
    for i in (0, 1):
        cv = const_value(op.inputs[i], folded)
        other = const_value(op.inputs[1 - i], folded)
        if cv is not None and other is None and cv > 0 and (cv & (cv - 1)) == 0:
            return cv.bit_length() - 1
    return None


def relocates_bits(ops: list[PcodeOp]) -> bool:
    """True when `ops` MOVE bit positions, rather than merely passing them on.

    A carry floor is owed because a shift has moved the tainted bits out from
    under the differential's coverage.  A shift of ZERO moves nothing, and
    SLEIGH spells an unscaled x86 index as `RCX * 0x1`, which `_is_pow2_scale`
    reports as a shift by 0.  Without this distinction `lea rax,[rbx+rcx]` looks
    like a fused shift-and-add and takes a carry smear it does not need, which
    turned an exact answer into an all-ones mask.
    """
    folded = fold_constants(ops)
    for op in _drop_constant_ops(ops):
        k = _is_pow2_scale(op, folded)
        if k:                       # pow2 multiply by more than 1
            return True
        if op.opcode.name in ('INT_LEFT', 'INT_RIGHT', 'INT_SRIGHT'):
            amt = const_value(op.inputs[1], folded) if len(op.inputs) > 1 else None
            if amt is None or amt != 0:
                return True         # unknown amount: assume it relocates
    return False


def multiply_overflow_flag(ops: list[PcodeOp]) -> bool:
    """True when `ops` compute a flag from "the product did not fit".

    Both spellings of the overflow predicate, recognised by SHAPE rather than by
    width, because the width is incidental and gating on it is what left the
    narrow forms unfloored:

        signed   (imul):  CF = sext(SUBPIECE(P, 0)) != P
        unsigned (mul):   hi = SUBPIECE(P, w);  CF = hi != 0

    Both compare something DERIVED from a multiply's full product, so the test is
    "a comparison one of whose operands traces back to an INT_MULT output through
    truncation, copy or extension".  That is the non-monotone predicate itself,
    which is the thing the 2-corner differential cannot see: a signed product
    overflows at BOTH sign extremes, so the two corners cancel while an interior
    value differs.

    Over-inclusion is a precision cost and under-inclusion is a soundness bug, so
    the closure is deliberately generous about how the product reaches the
    comparison.
    """
    prods = {_key(o.output) for o in ops
             if o.opcode.name == 'INT_MULT' and o.output is not None}
    if not prods:
        return False
    derived = set(prods)
    changed = True
    while changed:
        changed = False
        for o in ops:
            if o.output is None:
                continue
            if o.opcode.name in ('SUBPIECE', 'COPY', 'INT_SEXT', 'INT_ZEXT') and any(
                    _key(i) in derived for i in o.inputs):
                k = _key(o.output)
                if k not in derived:
                    derived.add(k)
                    changed = True
    return any(
        o.opcode.name in ('INT_NOTEQUAL', 'INT_EQUAL')
        and any(_key(i) in derived for i in o.inputs)
        for o in ops
    )


def _algebra_of(ops: list[PcodeOp]) -> str | None:
    """The single algebra a segment belongs to, or None if it is mixed/unknown.

    Conservative by design: a segment we cannot name with one algebra is never
    split on, because condition (C) would be meaningless for it.
    """
    ops = _drop_constant_ops(ops)
    _f = fold_constants(ops)
    classes = {
        (ALG_BITWISE if _is_pow2_scale(op, _f) is not None else _ALGEBRA[op.opcode.name])
        for op in ops
        if op.opcode.name not in ROUTING_OPCODES and op.opcode.name in _ALGEBRA
    }
    unknown = any(
        op.opcode.name not in ROUTING_OPCODES and op.opcode.name not in _ALGEBRA for op in ops
    )
    if unknown or len(classes) != 1:
        return None
    return next(iter(classes))


def _does_work(ops: list[PcodeOp]) -> bool:
    return any(op.opcode.name not in ROUTING_OPCODES for op in _drop_constant_ops(ops))


def _consumes_in_work(ops: list[PcodeOp], vn: Varnode) -> bool:
    """True when a non-routing op in `ops` reads `vn` as a data operand.

    Stands in for the downstream register read in condition (B).  A segment that
    only routes the conduit onward (COPY/SUBPIECE) is plumbing; one that feeds it
    to arithmetic is a second operation even when its other operand is a literal.
    """
    return any(
        op.opcode.name not in ROUTING_OPCODES
        and any(_overlaps(inp, vn) for inp in op.inputs if inp.space.name != 'const')
        for op in _drop_constant_ops(ops)
    )


def _register_reads(ops: list[PcodeOp]) -> set[tuple[int, int]]:
    return {
        (inp.offset, inp.size)
        for op in ops
        for inp in op.inputs
        if inp.space.name == 'register'
    }


def _backward_cone(ops: list[PcodeOp], target: Varnode) -> list[PcodeOp]:
    """Ops in `ops` that contribute to `target`, in original program order.

    Overlap-aware (see `_overlaps`): an op joins the cone when its output shares
    ANY byte with a live varnode, not only when the ranges match exactly.
    """
    live: list[Varnode] = [target]
    cone: list[PcodeOp] = []
    for op in reversed(ops):
        if op.output is None:
            continue
        if not any(_overlaps(op.output, lv) for lv in live):
            continue
        cone.append(op)
        live = [lv for lv in live if not _overlaps(op.output, lv)]
        for inp in op.inputs:
            if inp.space.name != 'const':
                live.append(inp)
    cone.reverse()
    return cone


def _waist_at(
    slice_ops: list[PcodeOp],
    vn: Varnode,
    require_distinct_algebra: bool,
    require_disjoint_inputs: bool,
) -> Waist | None:
    """The waist cut at one candidate conduit, or None if it fails a condition.

    Split out of `find_waist` so each condition reads as its own early
    return rather than as one more branch in a loop that had grown past
    what a reader can hold at once.
    """
    upstream = _backward_cone(slice_ops, vn)
    downstream = [op for op in slice_ops if op not in upstream]
    if not upstream or not downstream:
        return None

    # (A1) `vn` is the sole conduit: no downstream op may read any other
    # value defined upstream, or the cut would not separate the dataflow.
    up_outs = [o.output for o in upstream if o.output is not None]
    leaks = any(
        any(_overlaps(inp, uo) for uo in up_outs) and not _overlaps(inp, vn)
        for op in downstream
        for inp in op.inputs
        if inp.space.name != 'const'
    )
    if leaks:
        return None

    # (A2) disjoint architectural inputs -- the losslessness condition.
    up_regs = _register_reads(upstream)
    down_regs = _register_reads(downstream)
    if require_disjoint_inputs and (up_regs & down_regs):
        return None

    # (B) both sides are genuine operations on data, not plumbing and not
    # constant folding.  The UPSTREAM register-read requirement is what
    # rejects SLEIGH's rotate lifting, where the shift *amount* is computed
    # as INT_SUB(64, 7): that is an arithmetic op reading no register, so
    # without it `ror`/`ubfx`/`extr` and RISC-V's shift-amount masking would
    # all read as a fused arith->bitwise pair and a single rotate primitive
    # would be split in half.
    if not up_regs:
        return None
    # The DOWNSTREAM does not owe a register read.  Requiring one conflated
    # two different things: an op that COMPUTES a constant, which is
    # plumbing and is already removed by _drop_constant_ops, and an op that
    # ADDS a constant to tainted data, which is real work with a real carry
    # chain.  `lea rax,[rcx*4+8]` is the second: its downstream is
    # `0x8 + unique`, reading only the conduit and a constant, so no waist
    # was found, no floor was placed, and the carry out of the shifted bits
    # was covered by nothing but the 2-corner differential.  What the
    # downstream must do is consume the CONDUIT in a non-routing op, which
    # is the property the register read was standing in for.
    if not down_regs and not _consumes_in_work(downstream, vn):
        return None
    if not (_does_work(upstream) and _does_work(downstream)):
        return None

    # (C) different taint algebras -- otherwise one regime composes them.
    up_alg = _algebra_of(upstream)
    down_alg = _algebra_of(downstream)
    if up_alg is None or down_alg is None:
        return None
    if require_distinct_algebra and up_alg == down_alg:
        return None

    return Waist(vn, upstream, downstream, up_alg, down_alg, up_regs, down_regs)


def find_waist(
    slice_ops: list[PcodeOp],
    target: Varnode,
    require_distinct_algebra: bool = True,
    require_disjoint_inputs: bool = True,
) -> Waist | None:
    """Return the waist splitting `slice_ops` into two different operations.

    None when the slice is a single operation -- which is the common case and
    the safe answer, since the caller then keeps today's whole-slice behaviour.

    `require_distinct_algebra` separates two questions that condition (C) had
    conflated.  Deciding whether an instruction encodes TWO DIFFERENT OPERATIONS
    -- the splitting question -- does need the algebras to differ.  Deciding where
    to place a union FLOOR does not: it needs only condition (A), the disjoint-input
    conduit that makes materialising the intermediate lossless.  `bic x0,x1,x2,lsl #1`
    is bitwise on both sides, so it is not two different operations, yet its floor
    still has to be computed at the SHIFTED bit positions.  Callers wanting a floor
    pass False.

    `require_disjoint_inputs` is the same distinction applied to condition (A2).
    Materialising the intermediate is lossless only when the two sides read
    disjoint registers, so SPLITTING needs it.  A floor does not: it is OR-ed into
    the rule, so over-including is a precision cost and under-including is a
    soundness bug.  `lea rax,[rbx+rbx*4]` is the case that matters -- one register
    fills both the base and the index, so `b + (b << 2)` reads RBX on both sides
    and (A2) rejected the waist, leaving the carry between the two shifted copies
    covered by nothing but the 2-corner differential.  With RBX bits 44 and 45
    tainted the product moves bits 44 through 48 and the rule returned 44, 45 and
    48: bits 46 and 47 were a silent under-taint.  Callers wanting a floor pass
    False here too.
    """
    if len(slice_ops) < 2:
        return None

    candidates = [
        op.output
        for op in slice_ops
        if op.output is not None
        and op.output.space.name == 'unique'
        and _key(op.output) != _key(target)
    ]

    for vn in candidates:
        waist = _waist_at(slice_ops, vn, require_distinct_algebra,
                          require_disjoint_inputs)
        if waist is not None:
            return waist

    return None


def _smear_high(expr: Expr, from_bit: int, width: int) -> Expr:
    """Replicate bit `from_bit` of `expr` upward through bit `width - 1`.

    Used for sign-propagating operations, where a tainted sign bit taints every
    replicated copy of itself.  Log-fold doubling keeps the tree shallow.
    """
    fill: Expr = BinaryExpr(Op.AND, expr, Constant(1 << from_bit, 8))
    span = width - 1 - from_bit
    if span <= 0:
        return fill
    step = 1
    while step <= span:
        fill = BinaryExpr(Op.OR, fill, BinaryExpr(Op.LEFT, fill, Constant(step, 8)))
        step *= 2
    return BinaryExpr(Op.AND, fill, Constant(((1 << width) - 1) & ~((1 << from_bit) - 1), 8))


def _smear_down(expr: Expr, from_bit: int, count: int, width: int) -> Expr:
    """Replicate bit `from_bit` of `expr` downward across `count` bit positions.

    An arithmetic right shift by `count` vacates the top `count` bits and fills
    them with copies of the sign bit, so a tainted sign bit taints every copy.
    Note the direction: the fill spreads *down* from the sign bit into the bits
    the shift vacated, which are the highest bits of the result.
    """
    fill: Expr = BinaryExpr(Op.AND, expr, Constant(1 << from_bit, 8))
    if count <= 1:
        return fill
    step = 1
    while step < count:
        fill = BinaryExpr(Op.OR, fill, BinaryExpr(Op.RIGHT, fill, Constant(step, 8)))
        step *= 2
    keep = ((1 << width) - 1) & ~((1 << (from_bit + 1 - count)) - 1)
    return BinaryExpr(Op.AND, fill, Constant(keep, 8))


def varnode_taint_expr(  # noqa: C901
    ops: list[PcodeOp],
    target: Varnode,
    taint_of_register: Callable[[int, int], Expr | None],
) -> Expr | None:
    """Closed-form taint mask of the waist varnode, or None if not expressible.

    `taint_of_register(offset, size)` supplies the taint of an architectural
    register read, and returns None for a register the caller cannot map.

    Only the bitwise/permutation algebra is handled -- by condition (C) that is
    what an upstream segment paired with an arithmetic downstream must be.  Every
    rule below is either exact (shifts, constant masks, extensions) or a sound
    over-approximation (the union fallback for register-register bitwise ops),
    which is the same approximation the TRANSPORTABLE floor already makes.

    P-code is NOT SSA.  SLEIGH freely redefines a varnode, including reading and
    writing it in the same op -- ARM64's extended-register operand lifts as

        unique[b900:8] = unique[a900:8]
        unique[b900:8] = unique[b900:8] << 0x2

    A definition map keyed by varnode keeps only ONE definition (the last), so the
    recursion above met its own output, tripped the cycle guard and returned None:
    `add x0,x1,w2,uxth #2` and its siblings silently lost their floor.  Resolution
    is therefore by REACHING DEFINITION -- the latest definition strictly before
    the point of use -- which is what SSA renaming buys, computed directly rather
    than by materialising new varnodes (pypcode's are immutable natives).
    """
    memo: dict[tuple[tuple[str, int, int], int], Expr | None] = {}
    # Shift amounts and masks are routinely COMPUTED rather than emitted literally
    # (a rotate's complementary amount is `0x40 - 0xb`), so constants must be
    # resolved by folding, not by checking for a `const` varnode.
    _folded = fold_constants(ops)

    def reaching_def(vn: Varnode, before: int) -> tuple[int, PcodeOp] | None:
        """The definition of `vn` in effect at position `before` (overlap-aware)."""
        for i in range(before - 1, -1, -1):
            o = ops[i]
            if o.output is not None and _overlaps(o.output, vn):
                return i, o
        return None

    def const_of(vn: Varnode) -> int | None:
        if vn.space.name == 'const':
            return vn.offset
        return _folded.get(_key(vn))

    def taint(vn: Varnode, at: int, depth: int = 0) -> Expr | None:  # noqa: C901
        if depth > 24:
            return None
        if vn.space.name == 'const':
            return Constant(0, 8)
        if vn.space.name == 'register':
            reg_t: Expr | None = taint_of_register(vn.offset, vn.size)
            return reg_t
        found = reaching_def(vn, at)
        if found is None:
            return None
        idx, op = found
        k = (_key(vn), idx)
        if k in memo:
            return memo[k]
        memo[k] = None  # break any accidental cycle with the sound "unknown"

        name = op.opcode.name
        width = vn.size * 8
        full = (1 << width) - 1
        a = taint(op.inputs[0], idx, depth + 1)
        if a is None:
            return None
        c = const_of(op.inputs[1]) if len(op.inputs) > 1 else None
        res: Expr | None

        if name in ('COPY', 'INT_ZEXT'):
            res = a
        elif name == 'INT_SEXT':
            inner = op.inputs[0].size * 8
            res = BinaryExpr(Op.OR, a, _smear_high(a, inner - 1, width))
        elif name == 'SUBPIECE' and c is not None:
            res = BinaryExpr(Op.AND, BinaryExpr(Op.RIGHT, a, Constant(c * 8, 8)), Constant(full, 8))
        elif name == 'INT_MULT' and _is_pow2_scale(op, _folded) is not None:
            shift_k = _is_pow2_scale(op, _folded)
            data_in = (
                op.inputs[1] if const_value(op.inputs[0], _folded) is not None
                else op.inputs[0]
            )
            base = taint(data_in, idx, depth + 1)
            res = (
                None
                if base is None or shift_k is None
                else BinaryExpr(
                    Op.AND,
                    BinaryExpr(Op.LEFT, base, Constant(shift_k, 8)),
                    Constant(full, 8),
                )
            )
        elif name == 'INT_LEFT' and c is not None:
            res = BinaryExpr(Op.AND, BinaryExpr(Op.LEFT, a, Constant(c, 8)), Constant(full, 8))
        elif name == 'INT_RIGHT' and c is not None:
            res = BinaryExpr(Op.RIGHT, a, Constant(c, 8))
        elif name == 'INT_SRIGHT' and c is not None:
            # Arithmetic shift: the sign bit is replicated into the vacated top,
            # so a tainted sign bit taints all `c` fill bits as well.
            shifted = BinaryExpr(Op.RIGHT, a, Constant(c, 8))
            res = BinaryExpr(Op.OR, shifted, _smear_down(a, width - 1, min(c, width), width))
        elif name in ('INT_NEGATE', 'BOOL_NEGATE'):
            res = a
        elif name == 'INT_AND' and c is not None:
            res = BinaryExpr(Op.AND, a, Constant(c & full, 8))
        elif name == 'INT_OR' and c is not None:
            res = BinaryExpr(Op.AND, a, Constant((~c) & full, 8))
        elif name == 'INT_XOR' and c is not None:
            res = a
        elif name in ('INT_AND', 'INT_OR', 'INT_XOR') and len(op.inputs) > 1:
            b = taint(op.inputs[1], idx, depth + 1)
            res = None if b is None else BinaryExpr(Op.OR, a, b)
        elif name == 'PIECE' and len(op.inputs) > 1:
            b = taint(op.inputs[1], idx, depth + 1)
            low_bits = op.inputs[1].size * 8
            res = None if b is None else BinaryExpr(
                Op.OR, BinaryExpr(Op.LEFT, a, Constant(low_bits, 8)), b,
            )
        else:
            res = None

        memo[k] = res
        return res

    return taint(target, len(ops))


def waist_taint_expr(
    waist: Waist,
    taint_of_register: Callable[[int, int], Expr | None],
) -> Expr | None:
    """Closed-form taint of the waist varnode, or None if not expressible."""
    if waist.upstream_algebra != ALG_BITWISE:
        return None
    return varnode_taint_expr(waist.upstream, waist.varnode, taint_of_register)


__all__ = [
    'ALG_ARITH',
    'ALG_BITWISE',
    'ALG_COMPARE',
    'Waist',
    'find_waist',
    'varnode_taint_expr',
    'waist_taint_expr',
]
