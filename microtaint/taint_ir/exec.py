"""Compile a lowered taint program for the C evaluator.

The IR is architecture-neutral; this is where it acquires a concrete calling
convention: register state as two flat uint64 arrays indexed by slot, with the
slot baked into every input and output at compile time, so executing an
instruction's taint touches no name, no dict and no Python object.

The same serialized form is what a code generator would consume, so the
interpreter here doubles as the reference a compiled version must match.
"""
# ruff: noqa: PLC0415
from __future__ import annotations

from microtaint.taint_ir import ir as _ir

#: Must match the enum in cell_c/taint_ir_c.c.
_OP_ID = {
    _ir.CONST: 0, _ir.INV: 1, _ir.INT: 2,
    _ir.AND: 3, _ir.OR: 4, _ir.XOR: 5, _ir.ADD: 6, _ir.SUB: 7, _ir.MUL: 8,
    _ir.SHL: 9, _ir.SHR: 10, _ir.SAR: 11, _ir.NOT: 12, _ir.NEG: 13,
    _ir.ULT: 14, _ir.SLT: 15, _ir.EQ: 16, _ir.NEZ: 17, _ir.SEL: 18,
    _ir.POPCNT: 19, _ir.CLZ: 20,
    _ir.UDIV: 21, _ir.UREM: 22, _ir.SDIV: 23, _ir.SREM: 24,
}


def serialize_for_c(prog, slot_of):
    d = prog.serialize(slot_of)
    d['op_ids'] = [_OP_ID[o] for o in d['ops']]
    return d


def compile_program(prog, slot_of):
    """-> (capsule, serialized dict).  Raises KeyError for an unplaceable name."""
    from microtaint.instrumentation.cell_c import taint_ir_c
    d = serialize_for_c(prog, slot_of)
    return taint_ir_c.compile(d), d
