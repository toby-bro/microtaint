"""Closed-form taint for constant-parametrised flags.

After ``slice_simplify`` collapses the constant-count selects of an instruction
like ``shl eax,7``, each flag reduces to a small expression over a single
register operand transformed by constant shifts and masks:

    ZF     = (src << k == 0)             -> EqualityTaintExpr(src << k, 0)
    CF/SF  = MSB(src << k)               -> the taint of that one source bit
    OF     = preserved old flag          -> that flag's taint (a dep passthrough)

This module recognises those shapes on the SIMPLIFIED slice and returns the
EXACT closed-form taint ``Expr`` -- no InstructionCellExpr, so no concrete cell
re-execution.  It is deliberately conservative: it fires ONLY on flag outputs
whose constant-count select actually collapsed, and any shape it is not certain
of returns ``None`` so the caller falls back to the (exact but slow) ICE
differential.  A gap here only costs speed, never soundness; every closed form it
emits is validated against the brute-forced TRUE taint (it equals the true taint,
so it is at worst equal to the ICE differential and tighter where ICE floors).
Register RESULTS are left to the differential for now (the flags dominate the
cell-execution cost).

Handled leaf grammar (everything else declines):
  register leaf; COPY; INT_ZEXT; INT_AND(x, const); INT_LEFT(x, const);
  INT_RIGHT(x, const).  INT_SEXT / INT_SRIGHT (sign-taint replication) are NOT
  handled yet and decline.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

from pypcode.pypcode_native import PcodeOp

from microtaint.instrumentation.ast import (
    BinaryExpr,
    Constant,
    EqualityTaintExpr,
    Expr,
    Op,
    TaintOperand,
)
from microtaint.sleigh.slice_simplify import simplify_slice

if TYPE_CHECKING:
    from microtaint.sleigh.engine import StateMapper


# Structural types shared by pypcode's PcodeOp/Varnode and slice_simplify's
# duck-typed _Op/_Vn, so this module needs no `Any`. Members are read-only
# properties so both pypcode's (read-only) and the shim's (settable) attributes
# satisfy them.
class _Space(Protocol):
    @property
    def name(self) -> str: ...


class _Vn(Protocol):
    @property
    def space(self) -> _Space: ...
    @property
    def offset(self) -> int: ...
    @property
    def size(self) -> int: ...


class _Opcode(Protocol):
    @property
    def name(self) -> str: ...


class _Op(Protocol):
    @property
    def opcode(self) -> _Opcode: ...
    @property
    def output(self) -> _Vn | None: ...
    @property
    def inputs(self) -> Sequence[_Vn]: ...


def _key(vn: _Vn) -> tuple[str, int, int]:
    return (vn.space.name, vn.offset, vn.size)


def _const_of(vn: _Vn) -> int | None:
    return vn.offset if vn.space.name == 'const' else None


class _ChainExprs:
    """(value, taint) exprs for a varnode plus its bit width. `shifted` records
    whether a constant shift was traversed -- the gate for the register-result
    path, so it fires on a shift result but not on a plain mov / and / call."""

    __slots__ = ('shifted', 'taint', 'value', 'width')

    def __init__(self, value: Expr, taint: Expr, width: int, shifted: bool = False) -> None:
        self.value = value
        self.taint = taint
        self.width = width
        self.shifted = shifted


def _mask(expr: Expr, width: int) -> Expr:
    if width >= 64:
        return expr
    return BinaryExpr(Op.AND, expr, Constant((1 << width) - 1, 8))


def _last_write(ops: Sequence[_Op], vn: _Vn, limit: int) -> tuple[_Op | None, int, bool]:
    """(op, index, is_exact) for the LAST write overlapping `vn` at an index
    below `limit`, or (None, -1, False) if none.

    p-code is not SSA: a register is written in place, so a use resolves to the
    most recent write BEFORE it (hence `limit`).  `shl`'s result INT_LEFT reads
    and writes the same varnode, and `ror`/`sar` write the rotated value back
    into the source register -- program order is what disambiguates them."""
    last: _Op | None = None
    last_idx = -1
    last_exact = False
    for i, op in enumerate(ops):
        if i >= limit:
            break
        o = op.output
        if o is None or o.space.name != vn.space.name:
            continue
        if o.offset < vn.offset + vn.size and vn.offset < o.offset + o.size:
            last, last_idx, last_exact = op, i, _key(o) == _key(vn)
    return last, last_idx, last_exact


def _chain(ops: Sequence[_Op], vn: _Vn, mapper: StateMapper, limit: int, depth: int = 0) -> _ChainExprs | None:  # noqa: C901
    """Resolve `vn` (as read at program point `limit`) to (value, taint) exprs
    over a single register operand transformed by constant shifts/masks, or None
    if the shape is not handled."""
    if depth > 24:
        return None
    d, idx, exact = _last_write(ops, vn, limit)
    if d is None:
        # No write before `limit`: a true input leaf (registers only).
        if vn.space.name != 'register':
            return None
        m = mapper.map_to_state(vn.offset, vn.size)
        if m is None or getattr(m, 'name', None) is None or hasattr(m, 'addr_reg'):
            return None  # unmapped or a memory mapping
        w = m.bit_end - m.bit_start + 1
        return _ChainExprs(
            TaintOperand(m.name, m.bit_start, m.bit_end, is_taint=False),
            TaintOperand(m.name, m.bit_start, m.bit_end, is_taint=True),
            w,
        )
    if d.output is None:
        return None
    name = d.opcode.name
    obits = d.output.size * 8
    if not exact:
        # A wider COPY/ZEXT preserves the low bytes it wrote from, so reading a
        # low-aligned sub-range is the sub-range of that op's input.  Anything
        # else (partial/other overlap) is unsafe: decline.
        src = d.inputs[0] if d.inputs else None
        if (name in ('COPY', 'INT_ZEXT') and src is not None
                and d.output.offset == vn.offset and src.offset == vn.offset
                and src.size >= vn.size):
            return _chain(ops, vn, mapper, idx, depth + 1)
        return None
    if name in ('COPY', 'INT_ZEXT'):
        inner = _chain(ops, d.inputs[0], mapper, idx, depth + 1)
        if inner is None:
            return None
        # zext widens with zero taint in the new high bits; value/taint carry.
        return _ChainExprs(inner.value, inner.taint, obits, inner.shifted)
    if name in ('INT_LEFT', 'INT_RIGHT') and len(d.inputs) == 2:
        k = _const_of(d.inputs[1])
        if k is None or k >= obits:
            return None  # shift >= width: bits leave the operand, decline to ICE
        inner = _chain(ops, d.inputs[0], mapper, idx, depth + 1)
        if inner is None:
            return None
        op = Op.LEFT if name == 'INT_LEFT' else Op.RIGHT
        val = _mask(BinaryExpr(op, inner.value, Constant(k, 8)), obits)
        tnt = _mask(BinaryExpr(op, inner.taint, Constant(k, 8)), obits)
        return _ChainExprs(val, tnt, obits, shifted=True)
    if name == 'INT_AND' and len(d.inputs) == 2:
        for i, j in ((0, 1), (1, 0)):
            c = _const_of(d.inputs[i])
            if c is not None:
                inner = _chain(ops, d.inputs[j], mapper, idx, depth + 1)
                if inner is None:
                    return None
                # masking removes the taint of masked-out bits exactly
                val = BinaryExpr(Op.AND, inner.value, Constant(c, 8))
                tnt = BinaryExpr(Op.AND, inner.taint, Constant(c, 8))
                return _ChainExprs(val, tnt, obits, inner.shifted)
    return None


def _select_collapsed(orig: Sequence[_Op], simp: Sequence[_Op]) -> bool:
    """True iff the original slice had a boolean select `INT_OR(INT_AND(c,a),
    INT_AND(!c,b))` whose condition folded to a constant, so ``simplify_slice``
    turned that OR into a COPY.  This is the signature of a constant-parametrised
    conditional (a shift/rotate by an immediate), and it EXCLUDES plain
    AND/OR/test/results, where no select collapses -- exactly the cases the naive
    recogniser wrongly fired on."""
    and_outs = {_key(o.output) for o in orig if o.opcode.name == 'INT_AND' and o.output is not None}
    sel_keys = {
        _key(o.output)
        for o in orig
        if o.opcode.name == 'INT_OR' and len(o.inputs) == 2 and o.output is not None
        and _key(o.inputs[0]) in and_outs and _key(o.inputs[1]) in and_outs
    }
    if not sel_keys:
        return False
    simp_by_out = {_key(o.output): o for o in simp if o.output is not None}
    return any(k in simp_by_out and simp_by_out[k].opcode.name == 'COPY' for k in sel_keys)


def _sign_bit_taint(chain: _ChainExprs) -> Expr:
    """Taint of the MSB (bit width-1) of a value: (taint >> (width-1)) & 1."""
    return BinaryExpr(
        Op.AND,
        BinaryExpr(Op.RIGHT, chain.taint, Constant(chain.width - 1, 8)),
        Constant(1, 8),
    )


def closed_form_taint(  # noqa: C901
    slice_ops: list[PcodeOp],
    mapper: StateMapper,
    out_is_flag: bool,
) -> Expr | None:
    """Return the exact closed-form taint Expr for the slice's output, or None.

    For a 1-bit flag: fires when a constant-count select collapsed (the
    shift-by-immediate signature). For a register RESULT: fires when the value is
    a constant shift of one register (taint = the shifted source taint), which
    removes the shift result's ICE cell too.
    """
    if not slice_ops:
        return None
    simp = simplify_slice(slice_ops)
    if not simp:
        return None

    if not out_is_flag:
        # Register result of a constant shift: taint is the shifted source taint.
        # Gated on `shifted` so it fires on shl/shr but not on mov/and/call/push.
        out = slice_ops[-1].output
        if out is None:
            return None
        idx_by_out = {_key(o.output): i for i, o in enumerate(simp) if o.output is not None}
        limit = idx_by_out.get(_key(out), len(simp)) + 1
        ch = _chain(simp, out, mapper, limit)
        if ch is not None and ch.shifted:
            return _mask(ch.taint, ch.width)
        return None

    if not _select_collapsed(slice_ops, simp):
        return None
    # The output producer is the last op writing the flag varnode; follow COPY
    # chains (the collapsed select becomes COPY(new_flag)) to the real producer.
    if slice_ops[-1].output is None:
        return None
    flag_key = _key(slice_ops[-1].output)
    by_out = {_key(o.output): o for o in simp if o.output is not None}
    cur = by_out.get(flag_key)
    seen: set[tuple[str, int, int]] = set()
    while cur is not None and cur.opcode.name == 'COPY':
        src = cur.inputs[0]
        if src.space.name == 'const':
            return Constant(0, 8)  # constant flag: no taint
        if src.space.name == 'register':  # preserved flag (e.g. OF, count != 1)
            m = mapper.map_to_state(src.offset, src.size)
            if m is not None and getattr(m, 'name', None) is not None and not hasattr(m, 'addr_reg'):
                return TaintOperand(m.name, m.bit_start, m.bit_end, is_taint=True)
            return None
        k = _key(src)
        if k in seen:
            return None
        seen.add(k)
        cur = by_out.get(k)
    if cur is None or cur.output is None:
        return None
    name = cur.opcode.name
    # `cur`'s operands are read at its program point; resolve chains before it.
    idx_by_out = {_key(o.output): i for i, o in enumerate(simp) if o.output is not None}
    limit = idx_by_out.get(_key(cur.output), len(simp))

    # ZF-shape: (x == 0) with x a handled chain.
    if name == 'INT_EQUAL' and len(cur.inputs) == 2:
        for i, j in ((0, 1), (1, 0)):
            if _const_of(cur.inputs[i]) == 0:
                ch = _chain(simp, cur.inputs[j], mapper, limit)
                if ch is not None:
                    return EqualityTaintExpr(ch.value, ch.taint, Constant(0, 8), Constant(0, 8), ch.width)
        return None
    # CF/SF-shape: MSB via signed-less-than-zero, x a handled chain.
    if name == 'INT_SLESS' and len(cur.inputs) == 2 and _const_of(cur.inputs[1]) == 0:
        ch = _chain(simp, cur.inputs[0], mapper, limit)
        if ch is not None:
            return _sign_bit_taint(ch)
    return None


__all__ = ['closed_form_taint']
