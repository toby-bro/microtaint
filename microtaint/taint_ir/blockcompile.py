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

Compiling is also the expensive part, and until `_PLAN_CACHE` existed it ran
again on every run.  Measured on bench_dense, three consecutive runs of the same
binary in one process:

    block mode        run 1  451.8 ms   run 2  428.6 ms   run 3  426.8 ms
      of which here          360.2 ms          354.7 ms          353.3 ms
    per instruction   run 1  659.5 ms   run 2  170.7 ms   run 3  170.1 ms

The per-instruction path has a process-wide rule cache, so its second run is
3.9x faster than its first.  Block mode had none: all 38 blocks were lifted,
lowered and emitted again for every run, which is why it LOST to the slower path
from run 2 onward -- and run 2 onward is the only regime a fuzzer is ever in.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping
from collections.abc import Set as AbstractSet
from types import MappingProxyType, ModuleType
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:                    # stub-only: opaque PyCapsule handles
    from microtaint.emulator.blockpath_c import _Capsule as BlockPlan
    from microtaint.instrumentation.cell_c.taint_ir_c import (
        _Capsule as CompiledProgram,
    )

from microtaint.taint_ir.blocks import Region
from microtaint.taint_ir.frompcode import BUILDER_LOCK, Builder
from microtaint.taint_ir.ir import IRKey, IRProg
from microtaint.taint_ir.ir import SlotOf as _SlotOf
from microtaint.types import ArchLike

#: Builds one block's minimal register-read descriptor, or None when the
#: block reads no register value at all.
Descriptor = Callable[[frozenset[int]], 'tuple[int, int, int, int, list[int], bool, object] | None']

__all__ = ['SlotMap', 'block_slot_resolver', 'cache_clear', 'cache_stats',
           'compile_block', 'freeze_for_reuse']

#: Every distinct slot map this process has seen, and the small integer that
#: stands for it.  Keyed by the map's CONTENT, so two maps that agree get the
#: same token and two that differ never can: the token is exact, not a hash.
_SLOT_TOKENS: dict[frozenset[tuple[str, int]], int] = {}


class SlotMap(Mapping[str, int]):
    """A register-name to slot map that remembers what it is.

    The compiled-block cache is keyed partly on the slot map, because emitted
    code addresses engine slots by index and a program compiled against one
    layout writes its answer to the wrong register under another.  The obvious
    exact key is the set of pairs, and it is what this used to build on every
    lookup.  Measured on a static-glibc guest, per distinct block per run:
    12.5 us to build the frozenset and 31.2 us for the dictionary to hash and
    compare it, against ~76 us for the whole warm path.  Over half the cost of
    a CACHE HIT was identifying the map.

    So the map is asked once, at construction, and carries a token afterwards.
    The token comes from a registry keyed on the content, so it is exact rather
    than a hash: two maps that agree share a token, two that differ cannot.
    Copying the dict is what makes that safe -- a token that outlived an edit
    to the map it stands for would be precisely the silent wrong answer the
    cache key exists to prevent.
    """

    __slots__ = ('_d', '_token')

    def __init__(self, mapping: Mapping[str, int]) -> None:
        self._d: dict[str, int] = dict(mapping)
        content = frozenset(self._d.items())
        tok = _SLOT_TOKENS.get(content)
        if tok is None:
            tok = _SLOT_TOKENS[content] = len(_SLOT_TOKENS)
        self._token = tok

    def __getitem__(self, key: str) -> int:
        return self._d[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    @property
    def token(self) -> int:
        """What the cache key uses instead of the map itself."""
        return self._token


#: An IR key is either a name or a tuple like ('reg', offset) / ('mem', k).
#: Re-exported from the IR, which defines what a key is.
SlotOf = _SlotOf


#: The slot layout, which is a compile-time constant of the C extension and
#: cannot change within a process.  Asking for it cost 4.31 us and it was asked
#: once per `compile_block`, warm path included: 18% of what a cached block
#: costs to hand to a caller.  Read-only by contract, and a proxy rather than a
#: dict so that is enforced rather than asked for.
_LAYOUT: list[MappingProxyType[str, int]] = []


def _layout() -> Mapping[str, int]:
    if not _LAYOUT:
        from microtaint.emulator import blockpath_c  # noqa: PLC0415

        _LAYOUT.append(MappingProxyType(dict(blockpath_c.layout())))
    return _LAYOUT[0]


def block_slot_resolver(arch: ArchLike,
                        name_to_slot: Mapping[str, int]) -> SlotOf:
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
#: The trailing element is the register slots whose VALUE this region
#: publishes.  The runtime seeds and threads only those, instead of copying the
#: whole register file twice per region.
RegionSpec = tuple[int, int, tuple[tuple[int, int, int], ...], int, int, int,
                   int, tuple[int, ...]]


# ---------------------------------------------------------------------------
# The compiled-block cache
# ---------------------------------------------------------------------------
#: How many distinct blocks to keep compiled.  The C runtime's own plan table
#: (BLK_CACHE_CAP in blockpath_c.c) holds 4096 and REFUSES rather than evicting
#: when it fills, so keeping more here than it can hold buys nothing; the two
#: are deliberately the same number.  `MICROTAINT_BLOCK_PLAN_CACHE=0` disables
#: the cache outright -- neither read nor written, so a run under it compiles
#: every block afresh.  That is how a test proves the cache is what is doing the
#: work rather than some other memoisation further down, and it is the switch to
#: reach for when a compiled block is suspected of being wrong.
_CACHE_CAP = 4096


def _cache_cap() -> int:
    raw = os.environ.get('MICROTAINT_BLOCK_PLAN_CACHE', '')
    if not raw:
        return _CACHE_CAP
    try:
        return max(0, int(raw))
    except ValueError:
        return _CACHE_CAP


class _Compiled(NamedTuple):
    """One block, lifted, lowered and emitted, ready to be handed to any
    wrapper in this process.

    Everything here is independent of WHO runs the block.  What is not -- the
    Unicorn register-read buffers -- is deliberately absent: those are the
    caller's, built fresh per plan, so two live wrappers never share the scratch
    the C runtime reads registers into.  Sharing them would be silent: the
    second emulator would read its register file into the first one's buffer.
    """

    size: int                        #: the block's byte length
    specs: tuple[RegionSpec, ...]
    #: The compiled-code capsules the specs' function pointers point into.  A
    #: TUPLE on purpose: `_read_descriptor` appends the caller's keepalive to
    #: the list it is given, and appending to what the cache holds would grow
    #: it without bound and hand wrapper B a reference to wrapper A's buffers.
    #: A tuple makes that mistake an AttributeError instead of a slow leak.
    keep: tuple[object, ...]
    reads: frozenset[int]
    regions: tuple[Region, ...]


#: Keyed by everything a compiled block depends on, and nothing else.  In
#: insertion order, so eviction is FIFO: `dict` guarantees that ordering, and a
#: plain dict needs no lock beyond the GIL, which matters because the compiler
#: is called from the C hook.
_PLAN_CACHE: dict[object, _Compiled | None] = {}
_STATS = {'hits': 0, 'misses': 0, 'refusals': 0, 'evictions': 0}


def _cache_key(arch: ArchLike, code: bytes, base: int,
               name_to_slot: Mapping[str, int],
               publish_all_values: bool) -> object:
    """What two compilations must agree on to be the same compilation.

    Each part earns its place by a way the cache would otherwise be WRONG,
    which is the only kind of cache bug that matters here:

      * `base`, because blocks are compiled with `abs_ram=True`: a PC-relative
        operand's `ram` address is baked into the program, so the same bytes at
        a different address are a different program.  Sharing across addresses
        would resolve a load at the address the block first ran at and read the
        shadow somewhere else entirely -- a clean answer for a tainted word.
      * the CODE, not its address, because that is what makes self-modifying
        code safe: rewritten bytes are a different key and miss, where an
        address-keyed cache would happily run the plan for the bytes that used
        to be there.
      * the slot map, because the emitted code addresses engine slots by index.
        The set of pairs rather than its hash: a hash collision between two slot
        maps would send a program's writes to the wrong register.
      * the architecture, because the same bytes decode differently on each.

    The layout (`blockpath_c.layout()`) is a compile-time constant of the C
    extension, so it cannot differ between two calls in one process.
    """
    arch_key = arch.value if hasattr(arch, 'value') else str(arch)
    # A `SlotMap` answers in O(1); anything else pays for the set of pairs, and
    # is exactly as correct, just slower.  The engine passes a SlotMap.
    slots: object = (name_to_slot.token if isinstance(name_to_slot, SlotMap)
                     else frozenset(name_to_slot.items()))
    return (arch_key, base, code, publish_all_values, slots)


#: A sentinel distinct from a cached `None`, which is itself a real answer.
_MISSING = object()


def _cache_get(key: object) -> tuple[bool, _Compiled | None]:
    """-> (was it there, what it was).  `None` is a real cached answer: a block
    that does not lower costs a full lift to find out, and re-learning that
    every run is exactly the cost this cache exists to remove."""
    got = _PLAN_CACHE.get(key, _MISSING)
    if got is _MISSING:
        _STATS['misses'] += 1
        return False, None
    _STATS['hits'] += 1
    return True, got  # type: ignore[return-value]



def _cache_put(key: object, value: _Compiled | None) -> None:
    cap = _cache_cap()
    if cap <= 0:
        return
    if value is None:
        _STATS['refusals'] += 1
    while len(_PLAN_CACHE) >= cap:
        # FIFO.  Evicting drops this process's reference to the emitted code,
        # but never frees code a LIVE plan still points into: `plan_new` holds
        # its own reference to the capsules through the plan's keepalive.  The
        # worst an eviction costs is a recompile.
        try:
            _PLAN_CACHE.pop(next(iter(_PLAN_CACHE)))
        except (StopIteration, KeyError, RuntimeError):  # changed under us
            break
        _STATS['evictions'] += 1
    _PLAN_CACHE[key] = value


def cache_stats() -> dict[str, int]:
    """What the cache has actually done.  -> hits, misses, refusals, evictions,
    entries, capacity.

    Exported because "the cache is working" is a claim that has to be counted,
    not assumed: a cache whose key is too specific never hits, costs a
    dictionary lookup per block and looks exactly like a cache that works.
    """
    return {**_STATS, 'entries': len(_PLAN_CACHE), 'capacity': _cache_cap()}


def freeze_for_reuse() -> None:
    """Take what is already allocated out of the garbage collector's reach.

    Not a cache operation, and process-wide rather than ours alone, which is
    why it is opt-in and never called by the engine: it is `gc.freeze()`, and
    everything alive when it runs stops being traversed and stops being
    collected.  Call it ONCE, after the engine is set up and before the runs
    begin; calling it per run accumulates a permanent generation that only
    grows.

    It is here because a profile of a real target says so.  On a static-glibc
    guest driven the way a fuzzer drives one -- a fresh emulator per input --
    roughly 14% of cycles go to the collector, much of it traversing structures
    that exist for the life of the process and can never become garbage: the
    compiled blocks in `_PLAN_CACHE`, the emitted code they hold alive, the
    slot map.  Measured over the same workload as INSTRUCTIONS RETIRED, three
    sets, marginal per run:

        garbage collector on      195.3 M
        after freeze_for_reuse()  187.7 M   -3.9%
        collector disabled        182.2 M   -6.7%

    Wall clock first said 13% for the middle row.  That was measured while the
    machine was running something else, and instructions retired is the metric
    that does not move when it is: the honest figure is about 4%.  Run-to-run
    spread is around 3%, so this is a small effect measured carefully rather
    than a large one.

    Disabling the collector outright is worth more and is the caller's decision
    to make, not the engine's; freezing keeps collection working for everything
    allocated afterwards, which is what a long fuzzing run needs.
    """
    import gc  # noqa: PLC0415 - only this function needs it

    gc.freeze()


def cache_clear() -> None:
    """Forget every compiled block, and the counters with them.

    Live plans keep working: each holds its own reference to the code it runs.
    """
    _PLAN_CACHE.clear()
    for k in _STATS:
        _STATS[k] = 0


def compile_block(arch: ArchLike, code: bytes, base: int,
                  name_to_slot: Mapping[str, int],
                  *, builder: Builder | None = None,
                  descriptor: Descriptor | None = None,
                  publish_all_values: bool = False,
                  cache: bool = True,
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

    The lift, the lowering and the emitted code are cached for the process, so
    the second run of a binary compiles nothing; see `_cache_key` for what makes
    two compilations the same one.  The PLAN is still built per call, because
    it points at the caller's own register-read buffers.  `cache=False` forces a
    fresh compilation, which is what a test wants when it is checking the
    compiler rather than the cache.

    On a cache hit the returned `Region`s are the SAME objects an earlier caller
    got, and their `prog` is mutable.  Nothing in the engine touches them; a
    test that wants to take one apart should pass `cache=False`.
    """
    lay = _layout()
    if len(name_to_slot) > lay['val_base']:
        return None                      # registers would collide with values

    # A caller-supplied builder carries its own state, so a block compiled with
    # one is not the same compilation as the same block compiled without it.
    # Rather than try to put a Builder in a cache key, do not cache it.
    key = (None if builder is not None or not cache or _cache_cap() <= 0
           else _cache_key(arch, code, base, name_to_slot, publish_all_values))

    # The lookup happens INSIDE the lock, not before it.  Lowering runs through
    # a shared, stateful Builder and has to be serialised anyway (see
    # `BUILDER_LOCK`), and holding the lock across the lookup is what turns two
    # threads asking for the same block into one compilation and one hit rather
    # than two compilations.  Taking it costs ~50 ns against the ~5 us of
    # building a plan, and this runs once per distinct block, not per execution.
    with BUILDER_LOCK:
        if key is not None:
            seen, hit = _cache_get(key)
            if seen:
                built = hit
            else:
                built = _compile_one(arch, code, base, name_to_slot, lay,
                                     builder, publish_all_values)
                _cache_put(key, built)
        else:
            built = _compile_one(arch, code, base, name_to_slot, lay, builder,
                                 publish_all_values)
    # Outside the lock: this calls back into the caller to build ITS register
    # read buffers, and nothing about that needs the compiler serialised.
    return None if built is None else _plan_from(built, descriptor)


def _compile_one(arch: ArchLike, code: bytes, base: int,
                 name_to_slot: Mapping[str, int], lay: Mapping[str, int],
                 builder: Builder | None,
                 publish_all_values: bool) -> _Compiled | None:
    """Lift, lower and emit one block.  The expensive part, and the only part.

    Measured on a ten-instruction AMD64 block: 6.08 ms, of which 5.13 ms is the
    lift and lowering and 0.64 ms the emit.  Turning the result into a plan
    afterwards costs 1.1 us, which is why that is left to the caller and done
    per wrapper instead of being cached with the rest.
    """
    from microtaint.taint_ir.blocks import plan_block  # noqa: PLC0415

    slot_of = block_slot_resolver(arch, name_to_slot)
    try:
        # `abs_ram=True`: a block is compiled for the address it runs at, so a
        # PC-relative operand's resolved `ram` address is the live one.  It is
        # also why the cache key carries `base`.
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
    return _Compiled(len(code), tuple(specs), tuple(keep), frozenset(reads),
                     tuple(regions))


def _plan_from(built: _Compiled, descriptor: Descriptor | None,
               ) -> tuple[BlockPlan, list[Region], set[int]]:
    """Turn a compiled block into a plan for ONE caller.

    The register-read buffers are the caller's, and are built here rather than
    cached with the code, so two live wrappers never read their register files
    into the same scratch.  `keep` is copied for the same reason: the plan holds
    its own list, and the caller's keepalive goes in that copy.
    """
    from microtaint.emulator import blockpath_c  # noqa: PLC0415

    keep = list(built.keep)
    # `built.reads` is already a frozenset, and it is what the descriptor is
    # keyed on.  Converting it to a set and back again was two passes over it
    # per block per run for nothing.
    desc = _read_descriptor(descriptor, built.reads, keep)
    plan = blockpath_c.plan_new(built.size, list(built.specs), keep, *desc)
    return plan, list(built.regions), set(built.reads)


def _compile_regions(regions: list[Region], slot_of: SlotOf,
                     lay: Mapping[str, int], publish_all_values: bool,
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
        # A TUPLE of accesses, not a list: a spec outlives one caller now that
        # compiled blocks are cached, so nothing it holds should be mutable.
        # The register slots whose VALUE this region publishes.  Only a later
        # region of the same block can want one (`_keep_only_needed_values`
        # has already dropped the rest), so this is a handful of slots where
        # the runtime used to copy all of them: two whole-register-file
        # memcpys per region, one to seed the value half of the output array
        # and one to thread it back.  The last region of a block publishes
        # nothing at all, and then neither copy happens.
        pub = sorted({
            sl - lay['val_base']
            for k, _n in prog.outputs
            if isinstance(k, tuple) and k[0] == 'regv'
            and (sl := slot_of(k)) is not None})
        specs.append((addr, region.addr,
                      tuple((0 if a['kind'] == 'load' else 1, a['size'],
                             1 if k in live_mem else 0)
                            for k, a in enumerate(accesses)),
                      addr_fn, prog_addr, addr_prog,
                      region.last or region.addr, tuple(pub)))

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


def _read_descriptor(descriptor: Descriptor | None, reads: AbstractSet[int],
                     keep: list[object],
                     ) -> tuple[int, int, int, int, list[int] | None, bool]:
    """The block's own uc_reg_read_batch descriptor, or an empty one.

    Empty means the C runtime reads nothing, which is correct for a block whose
    programs read no register value at all.

    `reads` is taken as it comes and only converted when it is not already a
    frozenset: the caller holds one, and it is what the descriptor is keyed on.
    """
    if descriptor is None:
        return (0, 0, 0, 0, None, False)
    got = descriptor(reads if isinstance(reads, frozenset) else frozenset(reads))
    if got is None:
        return (0, 0, 0, 0, None, False)
    ids, ptrs, vals, n_calls, val_slots, need_flags, hold = got
    keep.append(hold)
    return (ids, ptrs, vals, n_calls, val_slots, need_flags)
