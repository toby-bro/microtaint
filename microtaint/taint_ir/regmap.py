"""Resolving a caller's register names against the IR's byte offsets.

The IR names registers by byte offset because a register file gives one offset
several names -- RAX and EAX and AH, or Ghidra's XMM0_QA where the engine's
geometry says VL_0x1200.  A caller names them however its own taint state does.
Resolving through the offset is what lets the two meet without either side
having to know the other's spelling.
"""
# ruff: noqa: PLC0415
from __future__ import annotations

_OFFSETS: dict = {}


def offsets(arch) -> dict:
    key = arch.value if hasattr(arch, 'value') else str(arch)
    o = _OFFSETS.get(key)
    if o is None:
        from microtaint.instrumentation.cell import _build_reg_maps
        o = dict(_build_reg_maps(arch)[0])
        _OFFSETS[key] = o
    return o


def name_offset(arch, name):
    """Byte offset of a register the caller named, or None.

    Vector lanes reach the taint state as `VL_0x<offset>` -- a name generated
    FROM the geometry rather than taken from the register file -- so the cell
    evaluator's name table does not contain them.  The offset is in the name.
    """
    off = offsets(arch).get(name)
    if off is not None:
        return off
    if name.startswith('VL_0x'):
        try:
            return int(name[3:], 16)
        except ValueError:
            return None
    return None


def slot_resolver(arch, name_to_slot, *, n_reg_slots=None):
    """A `slot_of` for IR keys, given the caller's register-name slot map."""
    by_off: dict = {}
    for name, slot in name_to_slot.items():
        off = name_offset(arch, name)
        if off is not None:
            by_off.setdefault(off, slot)
    base = n_reg_slots
    if base is None:
        base = (max(name_to_slot.values()) + 1) if name_to_slot else 0

    def slot_of(key):
        if isinstance(key, tuple):
            kind = key[0]
            if kind == 'reg':
                return by_off.get(key[1])
            if kind in ('mem', 'addr', 'addrt', 'sttaint'):
                from microtaint.taint_ir.frompcode import access_slot
                return access_slot(base, key[1], kind)
            return None
        return name_to_slot.get(key)

    return slot_of
