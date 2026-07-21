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


def _algebra_of(ops: list[PcodeOp]) -> str | None:
    """The single algebra a segment belongs to, or None if it is mixed/unknown.

    Conservative by design: a segment we cannot name with one algebra is never
    split on, because condition (C) would be meaningless for it.
    """
    classes = {
        _ALGEBRA[op.opcode.name]
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
    return any(op.opcode.name not in ROUTING_OPCODES for op in ops)


def _register_reads(ops: list[PcodeOp]) -> set[tuple[int, int]]:
    return {
        (inp.offset, inp.size)
        for op in ops
        for inp in op.inputs
        if inp.space.name == 'register'
    }


def _backward_cone(ops: list[PcodeOp], target: Varnode) -> list[PcodeOp]:
    """Ops in `ops` that contribute to `target`, in original program order."""
    live = {_key(target)}
    cone: list[PcodeOp] = []
    for op in reversed(ops):
        if op.output is not None and _key(op.output) in live:
            cone.append(op)
            live.discard(_key(op.output))
            for inp in op.inputs:
                if inp.space.name != 'const':
                    live.add(_key(inp))
    cone.reverse()
    return cone


def find_waist(slice_ops: list[PcodeOp], target: Varnode) -> Waist | None:
    """Return the waist splitting `slice_ops` into two different operations.

    None when the slice is a single operation -- which is the common case and
    the safe answer, since the caller then keeps today's whole-slice behaviour.
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
        upstream = _backward_cone(slice_ops, vn)
        up_keys = {_key(op.output) for op in upstream if op.output is not None}
        downstream = [op for op in slice_ops if op not in upstream]
        if not upstream or not downstream:
            continue

        # (A1) `vn` is the sole conduit: no downstream op may read any other
        # value defined upstream, or the cut would not separate the dataflow.
        conduit = _key(vn)
        leaks = any(
            _key(inp) in up_keys and _key(inp) != conduit
            for op in downstream
            for inp in op.inputs
            if inp.space.name != 'const'
        )
        if leaks:
            continue

        # (A2) disjoint architectural inputs -- the losslessness condition.
        up_regs = _register_reads(upstream)
        down_regs = _register_reads(downstream)
        if up_regs & down_regs:
            continue

        # (B) both sides are genuine operations on data, not plumbing and not
        # constant folding.  The register-read requirement is what rejects
        # SLEIGH's rotate lifting, where the shift *amount* is computed as
        # INT_SUB(64, 7): that is an arithmetic op reading no register, so
        # without this check `ror`/`ubfx`/`extr` and RISC-V's shift-amount
        # masking would all read as a fused arith->bitwise pair and a single
        # rotate primitive would be split in half.
        if not up_regs or not down_regs:
            continue
        if not (_does_work(upstream) and _does_work(downstream)):
            continue

        # (C) different taint algebras -- otherwise one regime composes them.
        up_alg = _algebra_of(upstream)
        down_alg = _algebra_of(downstream)
        if up_alg is None or down_alg is None or up_alg == down_alg:
            continue

        return Waist(vn, upstream, downstream, up_alg, down_alg, up_regs, down_regs)

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


def waist_taint_expr(  # noqa: C901
    waist: Waist,
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
    """
    if waist.upstream_algebra != ALG_BITWISE:
        return None

    defs = {_key(op.output): op for op in waist.upstream if op.output is not None}
    memo: dict[tuple[str, int, int], Expr | None] = {}

    def const_of(vn: Varnode) -> int | None:
        return vn.offset if vn.space.name == 'const' else None

    def taint(vn: Varnode, depth: int = 0) -> Expr | None:  # noqa: C901
        if depth > 24:  # cycles cannot occur in SSA-like p-code, but stay bounded
            return None
        if vn.space.name == 'const':
            return Constant(0, 8)
        if vn.space.name == 'register':
            reg_t: Expr | None = taint_of_register(vn.offset, vn.size)
            return reg_t
        k = _key(vn)
        if k in memo:
            return memo[k]
        memo[k] = None  # break any accidental cycle with the sound "unknown"
        op = defs.get(k)
        if op is None:
            return None

        name = op.opcode.name
        width = vn.size * 8
        full = (1 << width) - 1
        a = taint(op.inputs[0], depth + 1)
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
            b = taint(op.inputs[1], depth + 1)
            res = None if b is None else BinaryExpr(Op.OR, a, b)
        elif name == 'PIECE' and len(op.inputs) > 1:
            b = taint(op.inputs[1], depth + 1)
            low_bits = op.inputs[1].size * 8
            res = None if b is None else BinaryExpr(
                Op.OR, BinaryExpr(Op.LEFT, a, Constant(low_bits, 8)), b,
            )
        else:
            res = None

        memo[k] = res
        return res

    return taint(waist.varnode)


__all__ = [
    'ALG_ARITH',
    'ALG_BITWISE',
    'ALG_COMPARE',
    'Waist',
    'find_waist',
    'waist_taint_expr',
]
