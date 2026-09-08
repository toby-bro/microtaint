"""Compiling an instruction's taint program for the live engine.

The engine holds taint as a flat array indexed by slot, one slot per register
name it has interned.  A taint program compiled against that layout can replace
the whole per-output circuit evaluation with a single call, which is what the
hot path does when `program_for` hands it one.

Three conditions have to hold before a program is usable there, and all three
are checked rather than assumed:

  * FEW ENOUGH MEMORY ACCESSES.  Each one occupies four state slots past the
    register file, and the hot path's scratch is fixed-size.
  * EVERY INPUT PLACED.  The engine interns slots lazily, so a register the
    program reads may not have one yet.  Compiling against a missing slot would
    read whatever sits at index -1; the answer is to decline and retry later,
    once the slow path has interned it.
  * NATIVELY EMITTED.  The interpreter is not reachable from the hot path, so a
    program the host emitter declined is no use there.

Everything is off unless MICROTAINT_TAINT_IR is set: this replaces the
evaluator on the hot path, so it is opt-in until the full suite has been run
with it on.
"""
# ruff: noqa: PLC0415
from __future__ import annotations

import os

#: Must match MT_IR_MEM_BASE / MT_IR_MAX_ACC in emulator/fastpath.h.
MEM_SLOT_BASE = 512
MAX_ACCESSES = 8

_CACHE: dict = {}          # (arch, code) -> program | None   (None = permanent no)
_PENDING: set = set()      # (arch, code) awaiting a slot that does not exist yet
_ENABLED = None


def enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.environ.get('MICROTAINT_TAINT_IR', '') not in ('', '0')
    return _ENABLED


def _arch_key(arch):
    return arch.value if hasattr(arch, 'value') else str(arch)


def program_for(arch, code: bytes, name_to_slot: dict):
    """-> (capsule, function address) for the engine's layout, or None.

    The capsule owns the emitted code, so the caller must keep it alive for as
    long as it holds the address.
    """
    if not enabled():
        return None
    key = (_arch_key(arch), bytes(code))
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    if key in _CACHE:
        return None                      # decided against, permanently

    from microtaint.instrumentation.cell_c import taint_ir_c
    from microtaint.taint_ir.exec import compile_program
    from microtaint.taint_ir.frompcode import Unsupported, build_ir
    from microtaint.taint_ir.regmap import slot_resolver

    try:
        prog = build_ir(arch, code)
    except Unsupported:
        _CACHE[key] = None
        return None
    except Exception:  # noqa: BLE001
        _CACHE[key] = None
        return None
    if len(prog.accesses) > MAX_ACCESSES:
        _CACHE[key] = None
        return None
    if len(name_to_slot) >= MEM_SLOT_BASE:
        _CACHE[key] = None            # registers would collide with memory state
        return None

    from microtaint.taint_ir.regmap import name_offset
    # Memory state sits at a FIXED base rather than just past the current slot
    # count: a program is compiled once while the engine goes on interning
    # register slots, and a base that moved would leave already-compiled
    # programs addressing the wrong words.
    slot_of = slot_resolver(arch, name_to_slot, n_reg_slots=MEM_SLOT_BASE)
    touched = {k[1] for (_kind, k) in prog.inputs if isinstance(k, tuple)}
    touched |= {k[1] for k, _n in prog.outputs if isinstance(k, tuple)}
    # Two caller names at one offset would make the write ambiguous: the IR
    # names registers by offset, so only one of them could receive it.
    seen: dict = {}
    for nm in name_to_slot:
        off = name_offset(arch, nm)
        if off in touched:
            if off in seen and seen[off] != name_to_slot[nm]:
                _CACHE[key] = None
                return None
            seen[off] = name_to_slot[nm]
    for (_kind, k) in prog.inputs:
        if slot_of(k) is None:
            _PENDING.add(key)            # a slot may appear later; try again then
            return None
    # An OUTPUT with no slot is a register the engine does not track, so
    # dropping that write is exactly what the circuit path does too
    # (`if (slot >= 0 && slot < n_slots)` in mem_ptr_core's commit loop).
    try:
        cap, _d = compile_program(prog, slot_of)
    except Exception:  # noqa: BLE001
        _CACHE[key] = None
        return None
    if not taint_ir_c.jit(cap):
        _CACHE[key] = None
        return None
    addr = taint_ir_c.fn_addr(cap)
    if not addr:
        _CACHE[key] = None
        return None
    _PENDING.discard(key)
    # Whether the program actually READS the word a load brings in.  It
    # usually does not: a move from memory routes the shadow taint and never
    # looks at the value, and only 32% of the bank's loads have it live (8-12%
    # outside x86).  Resolving a load costs a guest memory read, measured at
    # about 81 ns, so the hot path skips that entirely where the value is dead.
    live_mem = {k[1] for (kind, k), n in prog.inputs.items()
                if kind == 'v' and isinstance(k, tuple) and k[0] == 'mem'
                and prog.live[n]}
    accesses = tuple((0 if a['kind'] == 'load' else 1, a['size'],
                      1 if k in live_mem else 0)
                     for k, a in enumerate(prog.accesses))
    # Whether the program writes the program counter decides whether the hot
    # path has to run it into scratch and apply the implicit-taint policy
    # before committing.
    from microtaint.taint_ir.frompcode import _BUILDERS
    pc_off = _BUILDERS[(_arch_key(arch), 'concrete')].pc_off
    writes_pc = any(isinstance(k, tuple) and k[0] == 'reg' and k[1] == pc_off
                    for k, _n in prog.outputs)
    _CACHE[key] = (cap, addr, accesses, writes_pc)
    return _CACHE[key]


def stats() -> dict:
    compiled = sum(1 for v in _CACHE.values() if v is not None)
    return {'compiled': compiled,
            'declined': len(_CACHE) - compiled,
            'pending_slots': len(_PENDING),
            'enabled': enabled()}
