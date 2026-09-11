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

from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:                    # stub-only: opaque PyCapsule handles
    from microtaint.emulator.blockpath_c import _Capsule as BlockPlan
    from microtaint.instrumentation.cell_c.taint_ir_c import (
        _Capsule as CompiledProgram,
    )

from microtaint.taint_ir.blocks import Region
from microtaint.taint_ir.frompcode import Builder
from microtaint.taint_ir.ir import IRKey, IRProg
from microtaint.taint_ir.ir import SlotOf as _SlotOf
from microtaint.types import ArchLike

#: Builds one block's minimal register-read descriptor, or None when the
#: block reads no register value at all.
Descriptor = Callable[[frozenset[int]], 'tuple[int, int, int, int, list[int], bool, object] | None']

__all__ = ['block_slot_resolver', 'compile_block']

#: An IR key is either a name or a tuple like ('reg', offset) / ('mem', k).
#: Re-exported from the IR, which defines what a key is.
SlotOf = _SlotOf


def _layout() -> dict[str, int]:
    from microtaint.emulator import blockpath_c  # noqa: PLC0415

    return dict(blockpath_c.layout())


def block_slot_resolver(arch: ArchLike, name_to_slot: dict[str, int]) -> SlotOf:
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
        off = name_offset(arch, name)
        if off is not None:
            by_off.setdefault(off, slot)

    def slot_of(key: IRKey) -> int | None:
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


#: One region as the C runtime wants it: (function address, guest address,
#: [(kind, size, needs the loaded value)], address-slice function address,
#: program address, address-slice program address, last instruction address).
#: The program addresses are what the runtime interprets when the emitter
#: declined to emit a function; a region with neither is one that did not lower
#: at all.  The last address is what a finding names: the runtime learns the
#: program counter is secret-dependent only once the whole region has run.
RegionSpec = tuple[int, int, list[tuple[int, int, int]], int, int, int, int]


def compile_block(arch: ArchLike, code: bytes, base: int, name_to_slot: dict[str, int],
                  *, builder: Builder | None = None,
                  descriptor: Descriptor | None = None,
                  publish_all_values: bool = False,
                  ) -> tuple[BlockPlan, list[Region], set[int]] | None:
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
    from microtaint.taint_ir.blocks import plan_block  # noqa: PLC0415

    lay = _layout()
    if len(name_to_slot) > lay['val_base']:
        return None                      # registers would collide with values
    slot_of = block_slot_resolver(arch, name_to_slot)
    try:
        # `abs_ram=True`: a block is compiled for the address it runs at, so a
        # PC-relative operand's resolved `ram` address is the live one.
        regions = plan_block(arch, code, base, builder=builder, abs_ram=True,
                             max_acc=lay['max_acc'])
    except Exception:                    # an unliftable block is refused
        return None
    if not regions or any(r.prog is None for r in regions):
        return None

    got = _compile_regions(regions, slot_of, lay, publish_all_values)
    if got is None:
        return None
    specs, keep, reads = got
    desc = _read_descriptor(descriptor, reads, keep)
    plan = blockpath_c.plan_new(len(code), specs, keep, *desc)
    return plan, regions, reads


def _compile_regions(regions: list[Region], slot_of: SlotOf,
                     lay: dict[str, int], publish_all_values: bool,
                     ) -> tuple[list[RegionSpec], list[object], set[int]] | None:
    """Emit each region, and collect what the block reads."""
    from microtaint.instrumentation.cell_c import taint_ir_c  # noqa: PLC0415
    from microtaint.taint_ir.exec import compile_program  # noqa: PLC0415

    specs: list[RegionSpec] = []
    keep: list[object] = []
    reads: set[int] = set()
    # Which register VALUES each region has to publish: only what a LATER
    # region of the same block reads.  The next BLOCK re-reads the register
    # file from the CPU, so the last region need publish nothing at all.
    #
    # This is not cosmetic.  A published value keeps alive whatever computes
    # it, and for a value loaded from memory that means resolving the load --
    # a guest memory read, measured at ~71 ns and the single largest item in
    # computing a block (35 of 98 ns/instr on bench_dense).  Dropping a value
    # nobody reads drops the read that produced it.
    later_reads = _values_read_later(regions)
    if publish_all_values:
        # For a caller that chains regions BY HAND and so needs every value
        # back.  The engine never does: it re-reads the register file from the
        # CPU at each block, so only a later region of the SAME block can want
        # a published value.  Test harnesses that treat one region as one block
        # are the exception.
        later_reads = [None] * len(regions)  # type: ignore[list-item]
    for ri, region in enumerate(regions):
        prog = region.prog
        # compile_block refuses the whole block if any region lowers
        # nowhere, so by here every one of them has a program.
        assert prog is not None
        if later_reads[ri] is not None:
            _keep_only_needed_values(prog, later_reads[ri])
        accesses = list(getattr(prog, 'accesses', None) or [])
        # The lowering was given the limit, so it cut the region rather than
        # producing a program that overflows; a program that got here anyway
        # would address slots the runtime does not have.
        assert len(accesses) <= lay['max_acc'], (
            f'{len(accesses)} accesses in a region, limit {lay["max_acc"]}')
        try:
            cap, _ser = compile_program(prog, slot_of)
        except Exception:                # an unplaceable slot is a refusal
            return None
        # The emitter declines division and count-leading-zeros on purpose and
        # the contract is that the caller keeps the interpreter for those.  So
        # a decline is not a refusal here: the region carries its PROGRAM and
        # the runtime interprets it.  Refusing instead skipped the whole block.
        addr = taint_ir_c.fn_addr(cap) if taint_ir_c.jit(cap) else 0
        prog_addr = taint_ir_c.prog_addr(cap)
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

        # Pass 1 of the two-pass protocol runs only to learn where the LOADS
        # land, and running the whole taint program for that is paying for
        # arithmetic nobody reads: measured, 82.7 ops to produce 1.93
        # addresses.  Re-rooting the SAME program on its address outputs and
        # letting dead-code elimination run again gives the slice that computes
        # just those -- no re-lift, no second lowering, only another emit.
        addr_fn = addr_prog = 0
        if any(a['kind'] == 'load' for a in accesses):
            got = _address_slice(prog, slot_of, taint_ir_c)
            if got is not None:
                addr_cap, addr_fn, addr_prog = got
                keep.append(addr_cap)
        specs.append((addr, region.addr,
                      [(0 if a['kind'] == 'load' else 1, a['size'],
                        1 if k in live_mem else 0)
                       for k, a in enumerate(accesses)],
                      addr_fn, prog_addr, addr_prog,
                      region.last or region.addr))

    return specs, keep, reads


def _values_read_later(regions: list[Region]) -> list[set[int]]:
    """For each region, the register offsets some LATER region reads.

    Read as VALUES: a later region's taint inputs come from the threaded taint,
    not from what this region publishes.
    """
    out: list[set[int]] = [set() for _ in regions]
    acc: set[int] = set()
    for i in range(len(regions) - 1, -1, -1):
        out[i] = set(acc)
        prog = regions[i].prog
        if prog is None:
            continue
        acc |= {k[1] for (kind, k) in prog.inputs
                if kind == 'v' and isinstance(k, tuple) and k[0] == 'reg'}
    return out


def _keep_only_needed_values(prog: IRProg, needed: set[int]) -> None:
    """Drop the value outputs no later region reads, and re-run dead-code
    elimination so whatever computed them goes too."""
    kept = [(k, n) for k, n in prog.outputs
            if not (isinstance(k, tuple) and k[0] == 'regv' and k[1] not in needed)]
    if len(kept) == len(prog.outputs):
        return
    prog.outputs = kept
    prog.live = []
    prog.finish()


def _address_slice(prog: IRProg, slot_of: SlotOf,
                   taint_ir_c: ModuleType,
                   ) -> tuple[CompiledProgram, int, int] | None:
    """The part of `prog` that computes its load addresses, compiled.

    Returns (capsule, function address) or None if it will not compile, in
    which case the caller falls back to running the whole program for pass 1 --
    correct either way, just slower.

    The program is re-rooted in place and restored, because `finish()` only
    marks which nodes are live and `finalize()` builds the compact program from
    that mark.  So the slice costs a dead-code pass and an emit, not another
    lowering.
    """
    from microtaint.taint_ir.exec import compile_program  # noqa: PLC0415

    saved_outputs = list(prog.outputs)
    addr_outputs: list[tuple[IRKey, int]] = [(k, n) for k, n in saved_outputs
                    if isinstance(k, tuple) and k[0] == 'addr']
    if not addr_outputs or len(addr_outputs) == len(saved_outputs):
        return None                      # nothing to prune
    try:
        prog.outputs = addr_outputs
        prog.live = []
        cap, _ser = compile_program(prog, slot_of)
        fn = taint_ir_c.fn_addr(cap) if taint_ir_c.jit(cap) else 0
        return cap, fn, taint_ir_c.prog_addr(cap)
    except Exception:                    # a slice that will not compile is not fatal
        return None
    finally:
        prog.outputs = saved_outputs
        prog.live = []
        prog.finish()


def _read_descriptor(descriptor: Descriptor | None, reads: set[int],
                     keep: list[object],
                     ) -> tuple[int, int, int, int, list[int] | None, bool]:
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
