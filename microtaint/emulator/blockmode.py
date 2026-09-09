"""Compute a basic block's taint in one go, instead of one instruction at a time.

EXPERIMENTAL AND NOT CORRECT YET.  Opt-in with ``MICROTAINT_BLOCK=1``; off by
default, and the per-instruction path is untouched when it is off.  Measured
against that path on the taint-density workloads, it reaches the same final
state on bench_untainted and LOSES taint on the other two:

    bench_untainted   10,540 blocks, 0 unhandled   identical
    bench_sparse      10,701 blocks, 2 unhandled   2 registers and 2 words lost
    bench_dense       20,710 blocks, 0 unhandled   2 words lost, unexplained

bench_sparse is explained by the missing fallback (see below); bench_dense is
not, and finding it is the next thing to do.  Two further gaps: there is no
fallback for a block that cannot be fully handled -- such a block is silently
SKIPPED, which is unanalysed code -- and the plan cache is not invalidated on
self-modifying code.  Do not rely on this for anything yet.

It is also not fast: reading registers through ctypes once per block and
carrying dicts costs more than the per-instruction path saves.  This form exists
to establish that a block's taint EQUALS the per-instruction answer.  Moving it
into the C hot path is what collects the win.

Why.  A per-instruction code hook costs ~18 ns of Unicorn dispatch plus the
engine's own trampoline, measured end to end at +47 ns per instruction on top of
bare emulation, against **0.67 ns per instruction for a block hook**.  Blocks
average 14.7 instructions, so the dispatch very nearly disappears; and lowering a
block as one program rather than fifteen roughly halves the taint arithmetic
too, because dead-code elimination then sees across the block.

How.  The block hook fires BEFORE the block executes, which has two consequences
that shape everything here:

  * the emulator cannot be asked for a register value part-way through a block,
    so the lowering has to compute them.  ``emit='both'`` publishes them, and
    each region hands its values to the next;
  * the taint is SPECULATIVE until the block completes.  A block that faults
    part way would otherwise commit taint for instructions that never ran, which
    is an UNDER-taint: `mov rax, rbx` with a clean rbx clears RAX's taint while
    RAX still holds its old tainted value.  So a block's result is held pending
    and committed when the NEXT block proves it finished (BlockRunner).

This is the correctness form, in Python: it establishes that a block's taint
equals the per-instruction answer on real programs.  It is NOT the fast path --
reading registers through ctypes per block and carrying dicts around costs more
than it saves.  Moving it into the C hot path is what collects the win.
"""
from __future__ import annotations

import os
from typing import Any

from microtaint.emulator import archregs
from microtaint.taint_ir import frompcode
from microtaint.taint_ir.blockrun import BlockRunner, Pending
from microtaint.taint_ir.blocks import plan_block

__all__ = ['BlockMode', 'enabled']

#: Per memory access, the four state slots the two-pass protocol uses; the same
#: layout fastpath.h::mt_ir_mem_step works with.
_ACC = {'mem': 0, 'addr': 1, 'addrt': 2, 'sttaint': 3}
_MAX_ACC = 8
_MASK64 = (1 << 64) - 1


def enabled() -> bool:
    return os.environ.get('MICROTAINT_BLOCK', '') not in ('', '0')


class _Plan:
    """A block's regions, and whether the block can be handled at all."""

    __slots__ = ('handleable', 'regions')

    def __init__(self, regions: list, handleable: bool) -> None:
        self.regions = regions
        self.handleable = handleable


class BlockMode:
    """Drives taint a block at a time, over the wrapper's own state."""

    def __init__(self, wrapper: Any) -> None:
        self.wrapper = wrapper
        self.ql = wrapper.ql
        self.arch = wrapper.arch
        self.regs = archregs.for_arch(self.arch)
        key = self.arch.value if hasattr(self.arch, 'value') else str(self.arch)
        self.builder = frompcode.Builder(self.arch, key.endswith('BE'), 'concrete')
        names = sorted(set(self.builder.name_by_off.values()))
        self.layout = {n: i for i, n in enumerate(names)}
        self.n_reg = len(self.layout)
        self.mem_base = 2 * self.n_reg
        self.width = self.mem_base + 4 * _MAX_ACC
        self._plans: dict[tuple[int, int], _Plan] = {}
        self.runner = BlockRunner(compute=self._compute, apply=self._apply)
        #: Blocks that cannot be handled here.  Counted rather than silently
        #: skipped: an unhandled block is unanalysed code, and the count is what
        #: says whether the fallback is the rarity it is supposed to be.
        self.unhandled = 0
        self.blocks = 0

    # -- installation --------------------------------------------------
    def install(self) -> None:
        from unicorn import UC_HOOK_BLOCK

        self.ql.uc.hook_add(UC_HOOK_BLOCK, self._on_block)

    def _on_block(self, uc: Any, address: int, size: int, user_data: Any) -> None:
        self.blocks += 1
        self.runner.on_block(address, size)

    def finish(self, completed: bool = True) -> None:
        """End of the run.  The last block is committed only if it finished."""
        self.runner.finish(completed)

    # -- the slot mapping ----------------------------------------------
    def _slot_of(self, key: tuple) -> int:
        kind = key[0]
        if kind in ('reg', 'regv'):
            name = self.builder.name_by_off.get(key[1])
            if name is None or name not in self.layout:
                raise KeyError(key)
            return self.layout[name] + (self.n_reg if kind == 'regv' else 0)
        if kind in _ACC:
            return self.mem_base + 4 * key[1] + _ACC[kind]
        raise KeyError(key)

    # -- planning ------------------------------------------------------
    def _plan(self, address: int, size: int) -> _Plan:
        hit = self._plans.get((address, size))
        if hit is not None:
            return hit
        try:
            code = bytes(self.ql.mem.read(address, size))
            regions = plan_block(self.arch, code, builder=self.builder)
        except Exception:            # noqa: BLE001 - an unreadable block is unhandleable
            regions, code = [], b''
        handleable = bool(regions) and all(r.prog is not None for r in regions)
        plan = _Plan(regions, handleable)
        self._plans[(address, size)] = plan
        return plan

    def invalidate(self) -> None:
        """Drop every plan.  A write into the code range makes them stale, in
        exactly the way the decode cache goes stale, and a plan replayed for
        rewritten code answers for instructions that are no longer there."""
        self._plans.clear()

    # -- running -------------------------------------------------------
    def _read_registers(self) -> dict[str, int]:
        """The whole register file, once per block."""
        values: dict[str, int] = {}
        for name, uc_id in zip(self.regs.all_names, self.regs.all_uc_ids, strict=True):
            try:
                values[name] = self.ql.uc.reg_read(uc_id) & _MASK64
            except Exception:        # noqa: BLE001 - a register Unicorn will not give
                values[name] = 0
        self.regs.unpack_flags(values)
        return values

    def _compute(self, address: int, size: int) -> Pending | None:
        from microtaint.instrumentation.cell_c import taint_ir_c
        from microtaint.taint_ir.exec import compile_program

        plan = self._plan(address, size)
        if not plan.handleable:
            self.unhandled += 1
            return None

        values = self._read_registers()
        taint = dict(self.wrapper.register_taint)
        writes: list[tuple[int, int, int]] = []
        reports: list[tuple[str, int, int]] = []
        #: Shadow writes this block has made but not yet committed.  A LATER
        #: REGION of the same block must see them: the lowering refuses a load
        #: after a store within one program, and cutting the region there is
        #: what makes that safe -- but the two halves are still the same block,
        #: and the second half would otherwise read the shadow as it was before
        #: the first half's store.  Measured: without this, bench_dense loses
        #: two tainted words.
        overlay: dict[int, int] = {}

        for region in plan.regions:
            prog = region.prog
            try:
                capsule, _ = compile_program(prog, self._slot_of)
            except KeyError:
                self.unhandled += 1
                return None
            sv = [0] * self.width
            st = [0] * self.width
            for name, slot in self.layout.items():
                sv[slot] = values.get(name, 0) & _MASK64
                st[slot] = taint.get(name, 0)
            accesses = getattr(prog, 'accesses', None) or []
            if len(accesses) > _MAX_ACC:
                self.unhandled += 1
                return None
            if accesses:
                first = taint_ir_c.run(capsule, sv, st)     # pass 1: addresses
                for k, acc in enumerate(accesses):
                    if acc['kind'] != 'load':
                        continue
                    addr = first[self.mem_base + 4 * k + _ACC['addr']]
                    n = acc['size']
                    try:
                        word = int.from_bytes(bytes(self.ql.mem.read(addr, n)), 'little')
                    except Exception:  # noqa: BLE001 - an unreadable load reads zero
                        word = 0
                    sv[self.mem_base + 4 * k] = word
                    st[self.mem_base + 4 * k] = self._shadow_read(overlay, addr, n)
            out = taint_ir_c.run(capsule, sv, st)           # pass 2: the taint
            for k, acc in enumerate(accesses):
                if acc['kind'] == 'load':
                    continue
                addr = out[self.mem_base + 4 * k + _ACC['addr']]
                mask = out[self.mem_base + 4 * k + _ACC['sttaint']]
                writes.append((addr, mask, acc['size']))
                for i in range(acc['size']):
                    overlay[addr + i] = (mask >> (8 * i)) & 0xFF
            # The region's own answer, and the values the next region reads.
            for name, slot in self.layout.items():
                taint[name] = out[slot]
                values[name] = out[self.n_reg + slot] or values.get(name, 0)
            pc_taint = taint.get(self.regs.pc_name, 0)
            if pc_taint:
                reports.append(('implicit', region.addr, pc_taint))
                taint[self.regs.pc_name] = 0

        return Pending(address, {k: v for k, v in taint.items() if v}, writes, reports)

    def _shadow_read(self, overlay: dict[int, int], addr: int, size: int) -> int:
        """The shadow, as this block has left it so far.

        Bytes the block has already stored to come from the overlay; everything
        else from the committed shadow.  Byte-wise because a store and a later
        load need not have the same width or alignment.
        """
        if not overlay:
            return self.wrapper.shadow_mem.read_mask(addr, size)
        mask = 0
        for i in range(size):
            byte = overlay.get(addr + i)
            if byte is None:
                byte = self.wrapper.shadow_mem.read_mask(addr + i, 1)
            mask |= (byte & 0xFF) << (8 * i)
        return mask

    def _apply(self, pending: Pending) -> None:
        """Put a completed block's answer into effect."""
        live = self.wrapper.register_taint
        live.clear()
        live.update(pending.taint)
        for addr, mask, size in pending.writes:
            self.wrapper.shadow_mem.write_mask(addr, mask, size)
        for kind, addr, mask in pending.reports:
            if kind == 'implicit':
                self._report_implicit(addr, mask)

    def _report_implicit(self, address: int, mask: int) -> None:
        """A region made the program counter secret-dependent.

        The report carries the REGION's address, so a finding still names where
        the leak is even though it is emitted a block late.  That lateness is
        deliberate: see blockrun, and the milestones it was decided against.
        """
        reporter = getattr(self.wrapper, 'reporter', None)
        if reporter is None:
            return
        if getattr(self.wrapper, 'check_sc', False):
            reporter.side_channel(address, instruction='', taint_mask=mask)
