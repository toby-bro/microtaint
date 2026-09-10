"""Compile a basic block into a plan the C runtime can execute.

This is the COMPILER, and it is the only part of block tainting that is Python.
It runs once per distinct block: lift the block's bytes in one pypcode
translate, cut them into the largest regions that lower as a single program,
emit machine code for each, and hand the C runtime a plan of function pointers.
Everything that happens per block EXECUTION is in `blockpath.h` and never
touches a PyObject.

The slot layout comes from `blockpath_c.layout()` rather than being restated
here, so the two cannot drift: a program compiled against the wrong layout would
address the wrong words and the failure would be silent.
"""
from __future__ import annotations

from typing import Any, Callable

__all__ = ['block_slot_resolver', 'compile_block']

#: An IR key is either a name or a tuple like ('reg', offset) / ('mem', k).
SlotOf = Callable[[Any], 'int | None']


def _layout() -> dict[str, int]:
    from microtaint.emulator import blockpath_c  # noqa: PLC0415

    return dict(blockpath_c.layout())


def block_slot_resolver(arch: Any, name_to_slot: dict[str, int]) -> SlotOf:
    """`slot_of` for a BLOCK program, in the layout blockpath.h runs.

    Register taint sits at the caller's own slot, the register's VALUE at
    `val_base + slot`, and each access takes `per_acc` consecutive slots from
    `mem_base`.  Values and stored values are what separate this from the
    per-instruction layout in fastpath.h; the two are deliberately distinct so
    already-compiled instruction programs keep addressing what they always did.
    """
    from microtaint.taint_ir.regmap import name_offset  # noqa: PLC0415

    lay = _layout()
    acc_off = {'mem': lay['a_mem'], 'addr': lay['a_addr'], 'addrt': lay['a_addrt'],
               'sttaint': lay['a_sttaint'], 'stval': lay['a_stval']}
    by_off: dict[int, int] = {}
    for name, slot in name_to_slot.items():
        off = name_offset(arch, name)  # type: ignore[no-untyped-call]
        if off is not None:
            by_off.setdefault(off, slot)

    def slot_of(key: Any) -> int | None:
        if not isinstance(key, tuple):
            got: int | None = name_to_slot.get(key)
            return got
        kind = key[0]
        if kind == 'reg':
            return by_off.get(key[1])
        if kind == 'regv':
            slot = by_off.get(key[1])
            return None if slot is None else lay['val_base'] + slot
        if kind in acc_off:
            k: int = key[1]
            return lay['mem_base'] + lay['per_acc'] * k + acc_off[kind]
        return None

    return slot_of


def compile_block(arch: Any, code: bytes, base: int, name_to_slot: dict[str, int],
                  *, builder: Any = None,
                  descriptor: Any = None) -> tuple[Any, list[Any], set[int]] | None:
    """-> (plan capsule, [Region], register offsets read), or None.

    None means the caller must send this block down the per-instruction path.
    A block is refused rather than partly handled: a region silently skipped is
    unanalysed code, which is worse than being slow.

    The third element is the set of register byte-offsets whose VALUE the
    block's programs actually read.  Reading the whole register file per block
    instead was 90% of block mode's cost when it was first wired -- 0.392 s of
    0.434 s on bench_untainted -- so the caller turns this into a minimal read
    descriptor and hands it back through `descriptor`.
    """
    from microtaint.emulator import blockpath_c  # noqa: PLC0415
    from microtaint.instrumentation.cell_c import taint_ir_c  # type: ignore[attr-defined]  # noqa: PLC0415
    from microtaint.taint_ir.blocks import plan_block  # noqa: PLC0415
    from microtaint.taint_ir.exec import compile_program  # noqa: PLC0415

    lay = _layout()
    if len(name_to_slot) > lay['val_base']:
        return None                      # registers would collide with values
    slot_of = block_slot_resolver(arch, name_to_slot)
    try:
        regions = plan_block(arch, code, base, builder=builder)
    except Exception:                    # an unliftable block is refused
        return None
    if not regions or any(r.prog is None for r in regions):
        return None

    specs, keep = [], []
    reads: set[int] = set()
    for region in regions:
        prog = region.prog
        accesses = list(getattr(prog, 'accesses', None) or [])
        if len(accesses) > lay['max_acc']:
            return None
        try:
            cap, _ser = compile_program(prog, slot_of)  # type: ignore[no-untyped-call]
        except Exception:                # an unplaceable slot is a refusal
            return None
        if not taint_ir_c.jit(cap):
            return None
        addr = taint_ir_c.fn_addr(cap)
        if not addr:
            return None
        keep.append(cap)
        # Whether the program actually READS the word a load brings in.  A
        # block publishes values, so more loads are live here than on the
        # per-instruction path, but a dead one still costs a guest memory read
        # (~81 ns) for a word nothing looks at.  Same rule engine_glue uses.
        live_mem = {k[1] for (kind, k), n in prog.inputs.items()
                    if kind == 'v' and isinstance(k, tuple) and k[0] == 'mem'
                    and prog.live[n]}
        # Registers whose VALUE survived the dead-code pass.  A taint rule
        # usually routes masks around without ever looking at what the register
        # held, so this set is far smaller than "every register the block
        # mentions" -- and it is the whole difference between a block read and
        # a whole-file read.
        reads |= {k[1] for (kind, k), n in prog.inputs.items()
                  if kind == 'v' and isinstance(k, tuple) and k[0] == 'reg'
                  and prog.live[n]}
        specs.append((addr, region.addr,
                      [(0 if a['kind'] == 'load' else 1, a['size'],
                        1 if k in live_mem else 0)
                       for k, a in enumerate(accesses)]))

    desc = _read_descriptor(descriptor, reads, keep)
    plan = blockpath_c.plan_new(len(code), specs, keep, *desc)
    return plan, regions, reads


def _read_descriptor(descriptor: Any, reads: set[int],
                     keep: list[Any]) -> tuple[int, int, int, int, Any, bool]:
    """The block's own uc_reg_read_batch descriptor, or an empty one.

    Empty means the C runtime reads nothing, which is correct for a block whose
    programs read no register value at all.
    """
    if descriptor is None:
        return (0, 0, 0, 0, None, False)
    got = descriptor(frozenset(reads))
    if got is None:
        return (0, 0, 0, 0, None, False)
    ids, ptrs, vals, n_calls, val_slots, need_flags, hold = got
    keep.append(hold)
    return (ids, ptrs, vals, n_calls, val_slots, need_flags)
