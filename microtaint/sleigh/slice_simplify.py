"""Semantics-preserving simplification of a p-code slice.

Many instructions guard a result on a parameter that is CONSTANT for a given
encoding.  The archetype is an x86 shift by an immediate, whose every flag is
lifted as a select on the (constant) shift count::

    ne  = count != 0                      # constant 1 for `shl eax,7`
    new = <compute the new flag>
    CF  = (ne ? new : old_CF)             # INT_OR(INT_AND(ne,new), INT_AND(!ne,old))

For a constant count the select is a dead branch, but the taint pass evaluates
it generically, so the flag slice stays large and routes to the (correct but
slow) ICE differential.  ``simplify_slice`` folds the constants and collapses
the dead branches, exposing the underlying ``new`` computation so a closed-form
recogniser can classify it.

The output is a NEW list of duck-typed p-code ops (``_Op`` / ``_Vn``) that the
existing consumers -- ``fold_constants``, ``determine_category`` and the slicer
helpers -- accept unchanged (they use attribute access, not ``isinstance``).
Every rewrite is a semantics-preserving identity:

  * an op whose output ``fold_constants`` proved constant becomes ``COPY`` of
    that constant;
  * ``AND(x,0)->0``, ``AND(x,~0)->x``, ``AND(x,1)->x`` (only when ``x`` is a
    boolean 0/1 value), ``OR(x,0)->x``, ``OR(x,~0)->~0``, ``XOR(x,0)->x``,
    each emitted as a ``COPY`` (or constant ``COPY``) so the op list stays
    self-contained and re-foldable.

Anything not matching a rule is copied through verbatim (inputs const-substituted
where proven constant), so the transform is always sound: a slice it cannot
simplify is returned semantically identical.
"""

from __future__ import annotations

from typing import NamedTuple

from pypcode import Varnode
from pypcode.pypcode_native import PcodeOp

from microtaint.sleigh.constfold import VarnodeLike, VNKey, fold_constants


class _Space(NamedTuple):
    name: str


class _Opcode(NamedTuple):
    name: str


# Opcodes whose output is a boolean 0/1, so `AND(1, out) == out`.
_BOOLEAN_OPS = frozenset({
    'INT_EQUAL', 'INT_NOTEQUAL', 'INT_LESS', 'INT_LESSEQUAL', 'INT_SLESS',
    'INT_SLESSEQUAL', 'INT_CARRY', 'INT_SCARRY', 'INT_SBORROW',
    'BOOL_AND', 'BOOL_OR', 'BOOL_XOR', 'BOOL_NEGATE',
})


class _Vn:
    """Duck-typed Varnode (space.name / offset / size)."""

    __slots__ = ('offset', 'size', 'space')

    def __init__(self, space_name: str, offset: int, size: int) -> None:
        self.space = _Space(space_name)
        self.offset = offset
        self.size = size


#: An op in a simplified slice: either the lifter's, or one this module
#: synthesised to replace a run of them.  A `type` statement so the two
#: names below can be referenced before they are defined.
type SliceOp = PcodeOp | _Op
#: A varnode in a simplified slice: the lifter's, or a synthesised one.
type SliceVn = Varnode | _Vn


class _Op:
    """Duck-typed PcodeOp (opcode.name / output / inputs)."""

    __slots__ = ('inputs', 'opcode', 'output')

    def __init__(self, name: str, output: SliceVn | None,
                 inputs: list[SliceVn]) -> None:
        self.opcode = _Opcode(name)
        self.output = output
        self.inputs = list(inputs)


def _key(vn: VarnodeLike) -> VNKey:
    return (vn.space.name, vn.offset, vn.size)


def _const(value: int, size: int) -> _Vn:
    return _Vn('const', value & ((1 << (size * 8)) - 1) if size else value, size)


def _algebraic_src(name: str, outp: SliceVn, ins: list[SliceVn],
                   boolean: set[VNKey]) -> SliceVn | None:
    """The varnode `outp` provably equals under an algebraic identity on the
    (const-resolved) inputs `ins`, or None to keep the op.  The boolean-only
    ``AND(1, x) == x`` identity fires only when `x` is a proven 0/1 value."""
    if len(ins) != 2:
        return None
    a, b = ins
    obits = outp.size * 8
    allo = (1 << obits) - 1 if obits else 0
    ca, cb = a.space.name == 'const', b.space.name == 'const'

    def is_bool(v: SliceVn) -> bool:
        return (v.space.name == 'const' and v.offset in (0, 1)) or _key(v) in boolean

    a_bool, b_bool = is_bool(a), is_bool(b)

    if name == 'INT_AND':
        if (ca and a.offset == 0) or (cb and b.offset == 0):
            return _const(0, outp.size)
        if ca and (a.offset == allo or (a.offset == 1 and b_bool)):
            return b
        if cb and (b.offset == allo or (b.offset == 1 and a_bool)):
            return a
    elif name == 'INT_OR':
        if (ca and a.offset == allo) or (cb and b.offset == allo):
            return _const(allo, outp.size)
        if ca and a.offset == 0:
            return b
        if cb and b.offset == 0:
            return a
    elif name == 'INT_XOR':
        if ca and a.offset == 0:
            return b
        if cb and b.offset == 0:
            return a
    return None


def simplify_slice(slice_ops: list[PcodeOp]) -> list[SliceOp]:  # noqa: C901
    """Return a semantics-preserving, constant-folded copy of ``slice_ops``.

    The input list must be in forward program order (as ``slice_backward``
    returns it), so a single sweep sees every definition before its uses.
    """
    folded = fold_constants(slice_ops)
    const_of: dict[VNKey, int] = dict(folded)
    boolean: set[VNKey] = set()
    out: list[SliceOp] = []

    def resolve(vn: Varnode) -> SliceVn:
        if vn.space.name == 'const':
            return vn
        k = _key(vn)
        return _const(const_of[k], vn.size) if k in const_of else vn

    def emit_copy(dst: SliceVn, src: SliceVn) -> None:
        out.append(_Op('COPY', dst, [src]))
        if src.space.name == 'const':
            const_of[_key(dst)] = src.offset
        elif _key(src) in boolean:
            boolean.add(_key(dst))

    for op in slice_ops:
        name = op.opcode.name
        outp = op.output
        if outp is None:
            out.append(_Op(name, None, [resolve(i) for i in op.inputs]))
            continue

        ok = _key(outp)
        if ok in const_of:
            emit_copy(outp, _const(const_of[ok], outp.size))
            continue

        ins = [resolve(i) for i in op.inputs]
        src = _algebraic_src(name, outp, ins, boolean)
        if src is not None:
            emit_copy(outp, src)
            continue

        out.append(_Op(name, outp, ins))
        if name in _BOOLEAN_OPS:
            boolean.add(ok)
        elif name in ('COPY', 'INT_ZEXT') and _key(ins[0]) in boolean:
            boolean.add(ok)

    return out


__all__ = ['simplify_slice']
