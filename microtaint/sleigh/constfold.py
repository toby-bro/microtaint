"""Constant folding over a p-code slice.

SLEIGH routinely computes a constant rather than emitting it.  RISC-V's 64-bit
shifts mask the amount with ``64 - 1`` instead of the literal ``0x3f``::

    INT_SUB   u28416 <- const:64, const:1        # = 63, but not a `const` varnode
    INT_AND   u28928 <- t2, u28416
    INT_LEFT  t0     <- t1, u28928

A recogniser that only accepts a literal `const` operand sees a `unique` there and
declines.  That is why RISC-V `sll`/`srl`/`sra` kept avalanching (6.0x invented
bits) while `sllw`/`srlw`/`sraw`, whose mask IS a literal ``0x1f``, were exact.

The fix belongs here rather than in any one recogniser: ANY operation whose
inputs are all constant yields a constant, so fold the whole slice once and let
callers ask "is this varnode constant, and what is it?".  Folding is also what
tells a caller that an op is mere constant computation rather than real work on
data, which matters when a recogniser insists the slice contain nothing else.

Values are stored masked to their varnode's width, so a caller can use them
directly as masks or shift amounts.
"""

from __future__ import annotations

from pypcode.pypcode_native import PcodeOp, Varnode

VNKey = tuple[str, int, int]


def _key(vn: Varnode) -> VNKey:
    return (vn.space.name, vn.offset, vn.size)


def _sx(v: int, bits: int) -> int:
    """Interpret the low `bits` of v as two's complement."""
    if bits and (v >> (bits - 1)) & 1:
        return v - (1 << bits)
    return v


def _eval(name: str, ins: list[int], in_bits: list[int], out_bits: int) -> int | None:  # noqa: C901
    """Evaluate one p-code op on constant inputs, or None if not modelled."""
    mask = (1 << out_bits) - 1 if out_bits else 0
    a = ins[0] if ins else 0
    b = ins[1] if len(ins) > 1 else 0
    abits = in_bits[0] if in_bits else out_bits
    bbits = in_bits[1] if len(in_bits) > 1 else abits

    if name in ('COPY', 'INT_ZEXT'):
        return a & mask
    if name == 'INT_SEXT':
        return _sx(a, abits) & mask
    if name == 'INT_ADD':
        return (a + b) & mask
    if name == 'INT_SUB':
        return (a - b) & mask
    if name == 'INT_MULT':
        return (a * b) & mask
    if name == 'INT_AND':
        return (a & b) & mask
    if name == 'INT_OR':
        return (a | b) & mask
    if name == 'INT_XOR':
        return (a ^ b) & mask
    if name in ('INT_NEGATE', 'BOOL_NEGATE'):
        return (~a) & mask
    if name == 'INT_2COMP':
        return (-a) & mask
    if name == 'INT_LEFT':
        return (a << b) & mask if b < abits else 0
    if name == 'INT_RIGHT':
        return (a >> b) if b < abits else 0
    if name == 'INT_SRIGHT':
        sa = _sx(a, abits)
        return (sa >> b) & mask if b < abits else ((mask if sa < 0 else 0))
    if name == 'INT_DIV':
        return (a // b) & mask if b else None
    if name == 'INT_REM':
        return (a % b) & mask if b else None
    if name == 'INT_SDIV':
        sa, sb = _sx(a, abits), _sx(b, bbits)
        return int(sa / sb) & mask if sb else None
    if name == 'INT_SREM':
        sa, sb = _sx(a, abits), _sx(b, bbits)
        return (sa - int(sa / sb) * sb) & mask if sb else None
    if name == 'INT_EQUAL':
        return int(a == b)
    if name == 'INT_NOTEQUAL':
        return int(a != b)
    if name == 'INT_LESS':
        return int(a < b)
    if name == 'INT_LESSEQUAL':
        return int(a <= b)
    if name == 'INT_SLESS':
        return int(_sx(a, abits) < _sx(b, bbits))
    if name == 'INT_SLESSEQUAL':
        return int(_sx(a, abits) <= _sx(b, bbits))
    if name == 'INT_CARRY':
        return int(a + b > (1 << out_bits) - 1) if out_bits else int(a + b > mask)
    if name == 'INT_SCARRY':
        s = _sx(a, abits) + _sx(b, bbits)
        return int(not (-(1 << (abits - 1)) <= s < (1 << (abits - 1))))
    if name == 'INT_SBORROW':
        s = _sx(a, abits) - _sx(b, bbits)
        return int(not (-(1 << (abits - 1)) <= s < (1 << (abits - 1))))
    if name == 'BOOL_AND':
        return int(bool(a) and bool(b))
    if name == 'BOOL_OR':
        return int(bool(a) or bool(b))
    if name == 'BOOL_XOR':
        return int(bool(a) != bool(b))
    if name == 'POPCOUNT':
        return int(a).bit_count() & mask
    if name == 'LZCOUNT':
        return (abits - int(a).bit_length()) & mask
    if name == 'SUBPIECE':
        return (a >> (b * 8)) & mask
    if name == 'PIECE':
        return ((a << bbits) | b) & mask
    return None


def fold_constants(ops: list[PcodeOp]) -> dict[VNKey, int]:
    """Every varnode this op list proves constant, mapped to its (masked) value.

    Literal `const` varnodes are not included -- callers already recognise those.
    Only varnodes computed FROM constants appear here, which is exactly the case a
    literal-only check misses.
    """
    known: dict[VNKey, int] = {}
    for op in ops:
        if op.output is None:
            continue
        vals: list[int] = []
        bits: list[int] = []
        ok = True
        for inp in op.inputs:
            bits.append(inp.size * 8)
            if inp.space.name == 'const':
                vals.append(inp.offset)
                continue
            k = _key(inp)
            if k in known:
                vals.append(known[k])
                continue
            ok = False
            break
        if not ok or not vals:
            continue
        res = _eval(op.opcode.name, vals, bits, op.output.size * 8)
        if res is not None:
            known[_key(op.output)] = res
    return known


def const_value(vn: Varnode, folded: dict[VNKey, int]) -> int | None:
    """The constant value of `vn`, whether a literal or computed from literals."""
    if vn.space.name == 'const':
        return vn.offset
    return folded.get(_key(vn))


def is_constant_op(op: PcodeOp, folded: dict[VNKey, int]) -> bool:
    """True if this op only computes a constant -- routing, not work on data."""
    return op.output is not None and _key(op.output) in folded


__all__ = ['const_value', 'fold_constants', 'is_constant_op']
