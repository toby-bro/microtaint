"""A compiled block is kept for the process, and the key has to be exact.

Compiling a block -- lift, cut into regions, lower, emit -- is 6.08 ms for ten
AMD64 instructions, of which 5.13 ms is the lift and lowering.  Turning the
result into a plan afterwards costs 1.1 us.  Until `_PLAN_CACHE` existed the
whole 6.08 ms ran again on every run, which is why block mode never got faster
the second time a binary ran while the per-instruction path, whose rule cache is
process-wide, got 3.9x faster:

    block mode        run 1  451.8 ms   run 2  428.6 ms   run 3  426.8 ms
    per instruction   run 1  659.5 ms   run 2  170.7 ms   run 3  170.1 ms

A fuzzer only ever lives in run 2 onward.

The danger of a cache is not that it misses.  It is that it HITS when it should
not have, and then a block runs the code compiled for different bytes, a
different address, a different register layout or a different architecture --
silently, with an answer that looks like an answer.  Most of what follows
attacks that one direction, each test naming the specific way the cache would be
wrong if that part of the key were dropped.  The checks are behavioural: the
plan is RUN against a byte-addressed memory and the taint read back, because a
plan that is merely a different object proves nothing.
"""
from __future__ import annotations

import gc
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

from microtaint.emulator import blockpath_c as B
from microtaint.taint_ir import blockcompile as bc
from microtaint.taint_ir import frompcode
from microtaint.taint_ir.blockcompile import (
    Descriptor,
    cache_clear,
    cache_stats,
    compile_block,
)
from microtaint.types import Architecture

if TYPE_CHECKING:                    # stub-only: an opaque PyCapsule handle
    from microtaint.emulator.blockpath_c import _Capsule

_ARCH = Architecture.AMD64
_BASE = 0x401000
_ARENA = 0x7000
_ARENA_LEN = 0x2000
FULL = (1 << 64) - 1

# --- the instructions these tests are built out of -------------------------
_MOV_RAX_RBX = bytes.fromhex('4889d8')          # mov rax, rbx
_MOV_RAX_RCX = bytes.fromhex('4889c8')          # mov rax, rcx
_ADD_RAX_RCX = bytes.fromhex('4801c8')          # add rax, rcx
_RIP_LOAD = bytes.fromhex('488b0510000000')     # mov rax, [rip+0x10]
_LOAD_RBX = bytes.fromhex('488b03')             # mov rax, [rbx]
_STORE_RBX = bytes.fromhex('488903')            # mov [rbx], rax
#: `ud2`.  Refused by the lowering, which is the point: learning that costs a
#: full lift, so the refusal is worth caching too.
_UD2 = bytes.fromhex('0f0b')
#: Four 0xff bytes: not an instruction at all.  The weirdest thing a block can
#: be, and a guest that jumps into its own data produces exactly this.
_GARBAGE = bytes.fromhex('ffffffff')
#: `addi x1, x0, 1` on RISCV64 and `xchg ebx,eax; add [rax],dl` on AMD64.  Both
#: lifters accept it, so the two lowerings can be compared directly rather than
#: inferred -- and one of them touches memory while the other does not.
#:
#: RISCV64 rather than ARM64 because the two register sets have to share ONE
#: slot map for the architecture to be the only thing that differs, and AMD64
#: plus ARM64 is 631 names against the 512 a block layout has room for.
_DUAL_ARCH = bytes.fromhex('93001000')
_DUAL_ARCHES = (Architecture.AMD64, Architecture.RISCV64)


@pytest.fixture(autouse=True)
def _isolated() -> Iterator[None]:
    """Every test starts with an empty cache and zeroed counters.

    The cache is process-wide on purpose, so without this a test would score
    another test's compilations and the counter assertions below would be
    meaningless.
    """
    cache_clear()
    yield
    cache_clear()


@pytest.fixture(scope='module')
def layout() -> dict[str, int]:
    names = sorted(set(frompcode.builder_for(_ARCH).name_by_off.values()))
    return {n: i for i, n in enumerate(names)}


@pytest.fixture(scope='module')
def other_layout(layout: dict[str, int]) -> dict[str, int]:
    """The same registers, every one of them at a different slot.

    Rotating by one guarantees no name keeps its slot, so a program compiled
    against one layout and run under the other writes to the wrong register
    everywhere rather than only where the two happened to differ.
    """
    names = sorted(layout)
    n = len(names)
    return {name: (i + 1) % n for i, name in enumerate(names)}


def _shared_layout() -> dict[str, int]:
    """One slot map both AMD64 and RISCV64 can be lowered against.

    289 names between them, against the 512 a block layout has room for before
    registers would collide with values.
    """
    names: set[str] = set()
    for arch in _DUAL_ARCHES:
        names |= set(frompcode.builder_for(arch).name_by_off.values())
    return {n: i for i, n in enumerate(sorted(names))}


def _arena() -> _Capsule:
    m = B.mem_new(_ARENA, _ARENA_LEN)
    # Distinct bytes everywhere, so a load from the wrong address shows in the
    # VALUE as well as in the taint.
    B.mem_poke(m, _ARENA, bytes((i * 7 + 3) & 0xFF for i in range(_ARENA_LEN)))
    return m


def _run(plan: _Capsule, lay: dict[str, int], address: int, *,
         mem: _Capsule | None = None,
         values: dict[str, int] | None = None,
         taint: dict[str, int] | None = None,
         ) -> tuple[dict[str, int], _Capsule]:
    """Run one plan through the C runtime.  -> (taint by name, the memory)."""
    mem = _arena() if mem is None else mem
    runner = B.runner_new(len(lay), -1, mem)
    if taint:
        seed = [0] * len(lay)
        for name, t in taint.items():
            seed[lay[name]] = t & FULL
        B.runner_seed(runner, seed)
    regs = [0] * len(lay)
    for name, v in (values or {}).items():
        regs[lay[name]] = v & FULL
    assert B.runner_on_block(runner, plan, address, regs) == 0, (
        'the runtime declined this plan, so nothing below is being measured')
    B.runner_finish(runner, True)
    slot_taint = B.runner_taint(runner)
    return {name: slot_taint[s] for name, s in lay.items()}, mem


def _shadow(mem: _Capsule) -> list[int]:
    """Every 8-byte taint mask in the arena, so a store to the wrong address is
    visible rather than merely possible."""
    return [B.mem_mask(mem, _ARENA + off, 8)
            for off in range(0, _ARENA_LEN, 8)]


# ---------------------------------------------------------------------------
# It works at all
# ---------------------------------------------------------------------------

def test_the_second_compilation_of_a_block_compiles_nothing(
        layout: dict[str, int]) -> None:
    """The premise.  Everything below is about the cache being CORRECT; this is
    the one test that says it does anything."""
    first = compile_block(_ARCH, _ADD_RAX_RCX, _BASE, layout)
    assert first is not None
    assert cache_stats()['misses'] == 1
    assert cache_stats()['hits'] == 0

    second = compile_block(_ARCH, _ADD_RAX_RCX, _BASE, layout)
    assert second is not None
    assert cache_stats() | {} == cache_stats()
    assert cache_stats()['hits'] == 1, (
        'the second compilation of the same block missed, so the cache is '
        f'doing nothing: {cache_stats()}')
    assert cache_stats()['misses'] == 1


def test_a_cache_hit_is_a_different_plan_over_the_same_code(
        layout: dict[str, int]) -> None:
    """A plan is per caller; the compiled code behind it is shared.

    The plan carries the addresses of ONE caller's Unicorn register-read
    buffers, so handing the same plan to two callers would have the second
    emulator read its register file into the first one's scratch.
    """
    first = compile_block(_ARCH, _ADD_RAX_RCX, _BASE, layout)
    second = compile_block(_ARCH, _ADD_RAX_RCX, _BASE, layout)
    assert first is not None
    assert second is not None
    assert first[0] is not second[0], 'two callers were handed the same plan'


# ---------------------------------------------------------------------------
# A hit must answer what a fresh compilation answers
# ---------------------------------------------------------------------------

#: (name, instruction bytes, register values, seeded register taint).  Chosen to
#: cover what a compiled block can carry: register-to-register movement, a load
#: whose address is a register, a store, flags, and a region that has to be cut.
_BANK: list[tuple[str, bytes, dict[str, int], dict[str, int]]] = [
    ('mov', _MOV_RAX_RBX, {'RBX': 0x1122334455667788}, {'RBX': 0xFF00}),
    ('add', _ADD_RAX_RCX, {'RAX': 7, 'RCX': 9}, {'RCX': 0xF}),
    ('load', _LOAD_RBX, {'RBX': _ARENA + 0x40}, {}),
    ('store', _STORE_RBX, {'RBX': _ARENA + 0x80, 'RAX': 5}, {'RAX': FULL}),
    ('tainted index', _LOAD_RBX, {'RBX': _ARENA + 0x100}, {'RBX': 0xFF}),
    ('mov;add', _MOV_RAX_RBX + _ADD_RAX_RCX,
     {'RBX': 3, 'RCX': 4}, {'RBX': 0xFF, 'RCX': 0xF0}),
    ('load;store', _LOAD_RBX + _STORE_RBX,
     {'RBX': _ARENA + 0x200}, {'RBX': 0x3}),
    ('rip load', _RIP_LOAD, {}, {}),
]


@pytest.mark.parametrize(('name', 'code', 'values', 'taint'), _BANK,
                         ids=[b[0] for b in _BANK])
def test_a_cached_block_answers_what_a_freshly_compiled_one_answers(
        name: str, code: bytes, values: dict[str, int], taint: dict[str, int],
        layout: dict[str, int]) -> None:
    """The differential this whole file exists to protect.

    A fresh compilation is the reference; the answer off a cache HIT must match
    it in every register slot and in every byte of the memory shadow.  The hit
    is asserted rather than assumed: a key too specific to ever match would make
    this test compare a fresh compilation against another fresh compilation and
    pass on nothing.
    """
    base = _ARENA + 0x400 if name == 'rip load' else _BASE
    ref = compile_block(_ARCH, code, base, layout, cache=False)
    assert ref is not None, f'{name} did not compile at all'
    ref_taint, ref_mem = _run(ref[0], layout, base, values=values, taint=taint)

    compile_block(_ARCH, code, base, layout)          # populate
    before = cache_stats()['hits']
    hit = compile_block(_ARCH, code, base, layout)     # and hit
    assert cache_stats()['hits'] == before + 1, (
        'the second compilation missed, so this compares two fresh '
        'compilations and proves nothing about the cache')
    assert hit is not None
    hit_taint, hit_mem = _run(hit[0], layout, base, values=values, taint=taint)

    assert hit_taint == ref_taint, (
        f'{name}: a cached block computed different register taint from a '
        f'freshly compiled one')
    assert _shadow(hit_mem) == _shadow(ref_mem), (
        f'{name}: a cached block left a different memory shadow')


def test_the_differential_bank_can_tell_taint_from_no_taint(
        layout: dict[str, int]) -> None:
    """The bank has to be able to FAIL.

    A bank of instructions that taint nothing would agree with anything, which
    is how a sweep passes while measuring nothing at all.  So count the cases
    that actually move taint somewhere, and refuse a bank where that is zero.
    """
    moved = 0
    for name, code, values, taint in _BANK:
        base = _ARENA + 0x400 if name == 'rip load' else _BASE
        got = compile_block(_ARCH, code, base, layout, cache=False)
        assert got is not None
        out, mem = _run(got[0], layout, base, values=values, taint=taint)
        if any(out.values()) or any(_shadow(mem)):
            moved += 1
    assert moved >= len(_BANK) // 2, (
        f'only {moved} of {len(_BANK)} bank entries produce any taint at all; '
        f'the differential above is mostly comparing zero against zero')


# ---------------------------------------------------------------------------
# Keys that must not collide.  Each of these is a way the cache would be WRONG.
# ---------------------------------------------------------------------------

def test_the_same_bytes_at_another_address_are_another_block(
        layout: dict[str, int]) -> None:
    """`mov rax, [rip+0x10]` at two addresses reads two different words.

    Blocks are compiled with `abs_ram=True`, so a PC-relative operand's `ram`
    address is resolved at compile time and baked into the program.  Dropping
    `base` from the key would make the second block read the address the FIRST
    one was compiled for: a tainted word read as clean, or a clean word read as
    tainted, with nothing anywhere to say so.

    Only the word the first block reads is tainted, so the second block coming
    back tainted is exactly the collision.
    """
    base_a, base_b = _ARENA + 0x100, _ARENA + 0x200
    target_a = base_a + len(_RIP_LOAD) + 0x10

    def read_at(base: int) -> int:
        mem = _arena()
        B.mem_taint(mem, target_a, FULL, 8)
        got = compile_block(_ARCH, _RIP_LOAD, base, layout)
        assert got is not None
        out, _ = _run(got[0], layout, base, mem=mem)
        return out['RAX']

    assert read_at(base_a) == FULL, (
        'the block compiled for its own address did not read the tainted '
        'word, so this test cannot detect the collision it is here for')
    assert read_at(base_b) == 0, (
        'the same bytes at a different address read the FIRST address: the '
        'cache key is missing the base, and every PC-relative access in a '
        'cached block resolves wherever it first ran')


def test_the_same_bytes_under_another_slot_map_are_another_block(
        layout: dict[str, int], other_layout: dict[str, int]) -> None:
    """Emitted code addresses engine slots by index.

    Two wrappers can disagree about which slot a register is -- a different
    architecture, a register interned in a different order -- and a program
    compiled against one layout writes its answer to the wrong register under
    the other.  Every name sits at a different slot in `other_layout`, so a
    collision shows on the very first register.
    """
    assert layout['RAX'] != other_layout['RAX']

    first = compile_block(_ARCH, _MOV_RAX_RBX, _BASE, layout)
    second = compile_block(_ARCH, _MOV_RAX_RBX, _BASE, other_layout)
    assert first is not None
    assert second is not None
    assert cache_stats()['hits'] == 0, (
        'the two slot maps shared a compilation')

    a, _ = _run(first[0], layout, _BASE,
                values={'RBX': 1}, taint={'RBX': 0xFF})
    b, _ = _run(second[0], other_layout, _BASE,
                values={'RBX': 1}, taint={'RBX': 0xFF})
    assert a['RAX'] == 0xFF, 'the reference layout did not move the taint'
    assert b['RAX'] == 0xFF, (
        'under the second slot map the taint did not reach RAX, so the '
        'program was compiled against the first one')


def test_the_same_bytes_on_another_architecture_are_another_block() -> None:
    """The same bytes, the same address, the same slot map, two lifters.

    `93 00 10 00` is one RISCV64 `addi` that touches no memory, and on AMD64 an
    `xchg` followed by a byte add THROUGH a pointer.  The two share a slot map
    here on purpose: with each architecture's own map the keys would differ on
    the registers alone, and dropping the architecture from the key would go
    unnoticed.
    """
    shared = _shared_layout()
    lowered = {}
    for arch in _DUAL_ARCHES:
        got = compile_block(arch, _DUAL_ARCH, _BASE, shared)
        assert got is not None, (
            f'{arch.name} refused these bytes, so the two lowerings cannot be '
            f'compared')
        lowered[arch] = [(a['kind'], a['size'])
                         for r in got[1] for a in (r.prog.accesses if r.prog else [])]
    assert cache_stats()['hits'] == 0, (
        'the two architectures shared a compilation')

    amd, riscv = (lowered[a] for a in _DUAL_ARCHES)
    assert amd, 'the AMD64 lowering of these bytes touches no memory any more'
    assert not riscv, 'the RISCV64 lowering of these bytes now touches memory'
    assert amd != riscv, (
        f'both architectures produced the same accesses ({amd}), so the key '
        f'ignores the architecture')


def test_the_architecture_is_part_of_the_key() -> None:
    """Said directly, because the behavioural test above needs two lifters that
    both accept one byte string and that is a thin supply."""
    shared = _shared_layout()
    keys = {arch: bc._cache_key(arch, _DUAL_ARCH, _BASE, shared, False)
            for arch in _DUAL_ARCHES}
    assert len(set(keys.values())) == len(_DUAL_ARCHES), keys


def test_rewritten_bytes_are_a_different_block(layout: dict[str, int]) -> None:
    """Self-modifying code, which is what keys the cache on bytes rather than
    on the address they sit at.

    An address-keyed cache runs the plan compiled for the bytes that used to be
    there.  Here the same address holds `mov rax, rbx` and then `mov rax, rcx`:
    with only RBX tainted, the rewritten block must leave RAX clean.
    """
    first = compile_block(_ARCH, _MOV_RAX_RBX, _BASE, layout)
    second = compile_block(_ARCH, _MOV_RAX_RCX, _BASE, layout)
    assert first is not None
    assert second is not None
    assert cache_stats()['hits'] == 0

    a, _ = _run(first[0], layout, _BASE, values={'RBX': 1}, taint={'RBX': 0xFF})
    b, _ = _run(second[0], layout, _BASE, values={'RBX': 1}, taint={'RBX': 0xFF})
    assert a['RAX'] == 0xFF, 'the first block did not move RBX into RAX'
    assert b['RAX'] == 0, (
        'after the bytes were rewritten the block still copied RBX, so it ran '
        'the plan compiled for the code that used to be at this address')


def test_publish_all_values_is_part_of_a_block_s_identity(
        layout: dict[str, int]) -> None:
    """Whether a block publishes its register values changes what it computes.

    A block that publishes nothing has had the code that produced those values
    deleted by the dead-code pass, so handing it to a caller that chains regions
    by hand gives that caller stale registers.
    """
    quiet = compile_block(_ARCH, _MOV_RAX_RBX, _BASE, layout)
    loud = compile_block(_ARCH, _MOV_RAX_RBX, _BASE, layout,
                         publish_all_values=True)
    assert quiet is not None
    assert loud is not None
    assert cache_stats()['hits'] == 0, (
        'publishing values shared a compilation with not publishing them')

    mem = _arena()
    runner = B.runner_new(len(layout), -1, mem)
    regs = [0] * len(layout)
    regs[layout['RBX']] = 0xABCD
    assert B.runner_on_block(runner, loud[0], _BASE, regs) == 0
    B.runner_finish(runner, True)
    assert B.runner_values(runner)[layout['RAX']] == 0xABCD, (
        'the publishing plan did not publish RAX')


@pytest.mark.parametrize(('name', 'code'),
                         [('ud2', _UD2), ('four 0xff bytes', _GARBAGE)])
def test_a_block_that_does_not_lower_is_refused_once(
        name: str, code: bytes, layout: dict[str, int]) -> None:
    """A refusal costs a full lift to discover, so it is cached too.

    `None` is a real answer here, not an absent one: a cache that stored only
    successes would re-lift every unhandleable block on every run, and
    unhandleable blocks are exactly the ones a real binary has most of.
    """
    assert compile_block(_ARCH, code, _BASE, layout) is None, (
        f'{name} lowered, so this test is not exercising a refusal')
    assert cache_stats()['misses'] == 1
    assert compile_block(_ARCH, code, _BASE, layout) is None
    assert cache_stats()['hits'] == 1, (
        f'{name} was lifted a second time to learn the same thing')
    assert cache_stats()['refusals'] == 1


def test_a_caller_supplied_builder_is_never_cached(layout: dict[str, int]) -> None:
    """A Builder carries its own state, and none of it is in the key.

    Rather than try to hash one, a compilation that was given a builder is
    neither read from nor written to the cache.
    """
    builder = frompcode.builder_for(_ARCH)
    assert compile_block(_ARCH, _ADD_RAX_RCX, _BASE, layout,
                         builder=builder) is not None
    assert cache_stats()['entries'] == 0, (
        'a block compiled with a caller-supplied builder was cached')
    assert cache_stats() == {'hits': 0, 'misses': 0, 'refusals': 0,
                             'evictions': 0, 'entries': 0,
                             'capacity': cache_stats()['capacity']}


# ---------------------------------------------------------------------------
# What the cache must NOT keep: one caller's buffers
# ---------------------------------------------------------------------------

def _descriptor(record: list[frozenset[int]]) -> Descriptor:
    """A stand-in for a wrapper's read descriptor.  DATA, not buffers: the
    Unicorn ids of the registers a block reads and the engine slots their
    words land in."""
    def describe(offsets: frozenset[int]) -> tuple[list[int], list[int], bool]:
        record.append(offsets)
        return ([7], [3, -1], False)
    return describe


def test_a_plan_describes_its_read_the_same_way_for_every_caller(
        layout: dict[str, int]) -> None:
    """A compiled block is shared, so what it says about its register read has
    to mean the same thing to whoever runs it.

    The descriptor used to answer with the ADDRESSES of one wrapper's ctypes
    arrays, and that is what made a plan the property of exactly one emulator:
    sharing it would have had two live emulators read their register files into
    the same scratch, and the second answer would have been computed from the
    first one's registers.  Nothing would have raised.  The destination now
    belongs to the hook doing the reading, so what is cached is which registers
    and which slots -- the same answer for everyone.
    """
    record: list[frozenset[int]] = []
    desc = _descriptor(record)
    first = compile_block(_ARCH, _LOAD_RBX, _BASE, layout, descriptor=desc)
    second = compile_block(_ARCH, _LOAD_RBX, _BASE, layout, descriptor=desc)
    assert first is not None
    assert second is not None
    assert cache_stats()['hits'] == 1, 'the second compilation did not hit'
    assert len(record) == 2, (
        f'the descriptor was consulted {len(record)} times for two plans; a '
        f'plan that never asked would read no registers at all')
    assert record[0] == record[1], (
        f'two plans for one block described different reads: {record}')


def test_a_plan_built_with_a_read_descriptor_still_runs(
        layout: dict[str, int]) -> None:
    """The descriptor is parsed by `plan_new`, in C, and a plan that refused it
    would decline every block it was given -- silently, because a declined
    block is skipped rather than reported."""
    got = compile_block(_ARCH, _MOV_RAX_RBX, _BASE, layout,
                        descriptor=_descriptor([]))
    assert got is not None
    out, _ = _run(got[0], layout, _BASE, values={'RBX': 1}, taint={'RBX': 0xFF})
    assert out['RAX'] == 0xFF


def test_the_keepalive_the_cache_holds_cannot_be_appended_to(
        layout: dict[str, int]) -> None:
    """The invariant behind the two tests above, stated where it lives.

    A tuple makes "append the caller's buffers to the shared list" an
    AttributeError instead of a leak nobody notices for a month.
    """
    compile_block(_ARCH, _LOAD_RBX, _BASE, layout)
    entries = [v for v in bc._PLAN_CACHE.values() if v is not None]
    assert entries, 'nothing was cached, so there is no invariant to check'
    for entry in entries:
        assert isinstance(entry.keep, tuple), type(entry.keep)
        assert isinstance(entry.specs, tuple), type(entry.specs)


def test_a_slot_map_token_is_exact_not_a_hash(
        layout: dict[str, int], other_layout: dict[str, int]) -> None:
    """The token stands in for the slot map in the cache key.

    Identifying the map was over half the cost of a cache hit, so a `SlotMap`
    answers with a small integer instead of the set of its pairs.  That is only
    safe while the integer is EXACT: two maps that differ sharing a token would
    hand a caller the plan compiled for the other one's layout, and emitted
    code addresses engine slots by index, so every write would land on the
    wrong register.
    """
    a = bc.SlotMap(layout)
    b = bc.SlotMap(dict(layout))          # same content, different object
    c = bc.SlotMap(other_layout)          # every name at a different slot
    assert a.token == b.token, 'equal maps disagree, so the cache cannot hit'
    assert a.token != c.token, (
        'two DIFFERENT slot maps share a token; a block compiled against one '
        'would be handed back for the other')
    # and one differing in a single slot
    tweaked = dict(layout)
    name = next(iter(tweaked))
    tweaked[name] = tweaked[name] + 1000
    assert bc.SlotMap(tweaked).token != a.token, (
        'a map differing in one slot shares a token with the original')


def test_a_slot_map_does_not_follow_the_dict_it_was_built_from(
        layout: dict[str, int]) -> None:
    """It copies, and that is what makes the token safe.

    A token computed once from a mapping that is later edited would stand for
    something that no longer exists, which is the silent wrong answer the
    cache key is there to prevent.
    """
    src = dict(layout)
    sm = bc.SlotMap(src)
    before = sm.token
    name = next(iter(src))
    src[name] = src[name] + 4242
    assert sm.token == before, 'the token moved when the source dict was edited'
    assert sm[name] == layout[name], 'the SlotMap followed the edit'
    assert len(sm) == len(layout)


def test_a_slot_map_compiles_to_the_same_answer_as_its_dict(
        layout: dict[str, int]) -> None:
    """Behavioural: the token is a faster spelling, not a different map."""
    as_dict = compile_block(_ARCH, _MOV_RAX_RBX, _BASE, layout, cache=False)
    as_map = compile_block(_ARCH, _MOV_RAX_RBX, _BASE, bc.SlotMap(layout),
                           cache=False)
    assert as_dict is not None
    assert as_map is not None
    a, _ = _run(as_dict[0], layout, _BASE, values={'RBX': 1}, taint={'RBX': 0xFF})
    b, _ = _run(as_map[0], layout, _BASE, values={'RBX': 1}, taint={'RBX': 0xFF})
    assert a['RAX'] == 0xFF, 'the dict form stopped moving the taint'
    assert b == a, 'the SlotMap form computed something different'


def test_a_cache_hit_hands_back_the_same_regions(layout: dict[str, int]) -> None:
    """Stated so it is a contract rather than a surprise.

    The `Region`s are shared between callers and their `prog` is mutable, so a
    caller that takes one apart affects what the next caller sees.  Nothing in
    the engine does, and `cache=False` is the way out for a test that wants to.
    """
    a = compile_block(_ARCH, _ADD_RAX_RCX, _BASE, layout)
    b = compile_block(_ARCH, _ADD_RAX_RCX, _BASE, layout)
    fresh = compile_block(_ARCH, _ADD_RAX_RCX, _BASE, layout, cache=False)
    assert a is not None
    assert b is not None
    assert fresh is not None
    assert a[1][0].prog is b[1][0].prog, 'a hit re-lowered the block'
    assert fresh[1][0].prog is not a[1][0].prog, (
        'cache=False returned the cached lowering')


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

def _distinct(n: int) -> list[bytes]:
    """`n` blocks that differ in their BYTES, not only their address, so each
    one is genuinely a separate compilation."""
    # mov rax, imm32 -- one byte of the immediate walks.
    return [bytes.fromhex('48c7c0') + (i & 0xFFFF).to_bytes(4, 'little')
            for i in range(n)]


def test_the_cache_is_bounded(layout: dict[str, int],
                              monkeypatch: pytest.MonkeyPatch) -> None:
    """A long run meets new code forever; the cache must not grow with it."""
    monkeypatch.setenv('MICROTAINT_BLOCK_PLAN_CACHE', '4')
    for code in _distinct(12):
        compile_block(_ARCH, code, _BASE, layout)
    stats = cache_stats()
    assert stats['entries'] <= 4, stats
    assert stats['evictions'] >= 8, (
        f'12 distinct blocks into a cache of 4 evicted {stats["evictions"]}')


def test_an_evicted_block_does_not_break_a_plan_that_is_still_running(
        layout: dict[str, int], monkeypatch: pytest.MonkeyPatch) -> None:
    """Eviction drops the cache's reference to emitted machine code.

    A plan built from that code is still live and its function pointers still
    point into it, so the plan has to own a reference of its own.  If it does
    not, this runs freed executable memory -- which does not raise, it computes
    something.
    """
    monkeypatch.setenv('MICROTAINT_BLOCK_PLAN_CACHE', '2')
    first = compile_block(_ARCH, _MOV_RAX_RBX, _BASE, layout)
    assert first is not None
    for code in _distinct(8):                # push it out several times over
        compile_block(_ARCH, code, _BASE + 0x100, layout)
    assert cache_stats()['evictions'] >= 1, 'nothing was evicted'
    gc.collect()

    out, _ = _run(first[0], layout, _BASE, values={'RBX': 1}, taint={'RBX': 0xFF})
    assert out['RAX'] == 0xFF, (
        'a plan whose compiled block was evicted no longer computes what it '
        'did, so the emitted code was freed while the plan still pointed at it')


def test_the_cache_can_be_turned_off(layout: dict[str, int],
                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """`MICROTAINT_BLOCK_PLAN_CACHE=0` is the switch to reach for when a
    compiled block is suspected of being wrong, so it has to be a real
    bypass: nothing read, nothing written."""
    monkeypatch.setenv('MICROTAINT_BLOCK_PLAN_CACHE', '0')
    a = compile_block(_ARCH, _MOV_RAX_RBX, _BASE, layout)
    b = compile_block(_ARCH, _MOV_RAX_RBX, _BASE, layout)
    assert a is not None
    assert b is not None
    assert cache_stats()['entries'] == 0
    assert cache_stats()['hits'] == 0, 'the cache was still being read'
    out, _ = _run(b[0], layout, _BASE, values={'RBX': 1}, taint={'RBX': 0xFF})
    assert out['RAX'] == 0xFF, 'the uncached compilation is not correct'


def test_lowering_never_runs_without_the_builder_lock(
        layout: dict[str, int], monkeypatch: pytest.MonkeyPatch) -> None:
    """The deterministic half of the thread-safety gate.

    `builder_for` hands every caller the SAME stateful Builder, so two threads
    lowering at once walk over each other's predicate stack and unroll
    bookkeeping.  The stress test below reproduces that, but only about one run
    in three: a race is a poor gate on its own.  This one fails every time the
    lock goes away, which is what makes it the gate.
    """
    seen: list[bool] = []
    real = bc._compile_one

    def watched(*args: object, **kw: object) -> object:
        seen.append(frompcode.BUILDER_LOCK.locked())
        return real(*args, **kw)      # type: ignore[arg-type]

    monkeypatch.setattr(bc, '_compile_one', watched)
    assert compile_block(_ARCH, _ADD_RAX_RCX, _BASE, layout) is not None
    assert compile_block(_ARCH, _UD2, _BASE, layout) is None
    assert compile_block(_ARCH, _ADD_RAX_RCX, _BASE, layout,
                         cache=False) is not None
    assert len(seen) == 3, (
        f'lowering ran {len(seen)} times for three compilations, so this is '
        f'not watching what it thinks it is')
    assert all(seen), (
        'a block was lowered without the builder lock held, so two threads '
        'compiling at once would corrupt each other')


def test_concurrent_compilations_of_one_block_compile_it_once(
        layout: dict[str, int]) -> None:
    """The lookup is inside the lock, so waiting threads find the answer.

    If it were outside, four threads arriving together would all miss, all
    compile, and the last one would win -- correct, but paying the 6 ms four
    times for nothing, which in a threaded fuzzer is exactly the cost this
    cache exists to remove.
    """
    import threading

    codes = [_MOV_RAX_RBX, _ADD_RAX_RCX, _LOAD_RBX, _UD2, *_distinct(4)]
    start = threading.Barrier(4)

    def work() -> None:
        start.wait()
        for code in codes:
            compile_block(_ARCH, code, _BASE, layout)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert cache_stats()['misses'] == len(codes), (
        f'{len(codes)} distinct blocks were compiled '
        f'{cache_stats()["misses"]} times between four threads')
    assert cache_stats()['hits'] == 3 * len(codes)


def test_two_threads_compiling_the_same_block_both_get_a_working_plan(
        layout: dict[str, int], monkeypatch: pytest.MonkeyPatch) -> None:
    """The compiler is called from the C hook, so two emulators in two threads
    reach it at once.  A duplicate compilation is only waste; a plan built from
    a half-inserted entry, or an eviction racing a lookup, is not.

    The cache is deliberately small here so the eviction path runs too.
    """
    import threading

    monkeypatch.setenv('MICROTAINT_BLOCK_PLAN_CACHE', '3')
    codes = [_MOV_RAX_RBX, _ADD_RAX_RCX, _LOAD_RBX, *_distinct(6)]
    errors: list[BaseException] = []
    plans: list[object] = []
    lock = threading.Lock()

    def work() -> None:
        try:
            for _ in range(20):
                for code in codes:
                    got = compile_block(_ARCH, code, _BASE, layout)
                    if got is not None:
                        with lock:
                            plans.append(got[0])
        except BaseException as exc:
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert len(plans) == 4 * 20 * len(codes)

    # And the answer is still right afterwards.
    got = compile_block(_ARCH, _MOV_RAX_RBX, _BASE, layout)
    assert got is not None
    out, _ = _run(got[0], layout, _BASE, values={'RBX': 1}, taint={'RBX': 0xFF})
    assert out['RAX'] == 0xFF


# ---------------------------------------------------------------------------
# What makes the cache hit ACROSS runs
# ---------------------------------------------------------------------------

def test_freezing_for_reuse_keeps_the_cache_working() -> None:
    """`freeze_for_reuse` is `gc.freeze()` with a reason, so the only thing to
    check is that it changes nothing about the answers.

    It exists because a profile of a fuzzer-shaped workload put ~14% of cycles
    in the collector, most of it traversing structures that live for the
    process and can never become garbage.
    """
    import gc

    before = gc.get_freeze_count()
    compile_block(_ARCH, _ADD_RAX_RCX, _BASE, layouts_for_freeze())
    bc.freeze_for_reuse()
    assert gc.get_freeze_count() > before, 'nothing was frozen'
    # the cache still hits, and still answers
    hits = cache_stats()['hits']
    got = compile_block(_ARCH, _ADD_RAX_RCX, _BASE, layouts_for_freeze())
    assert got is not None
    assert cache_stats()['hits'] == hits + 1, 'the cache stopped hitting'


def layouts_for_freeze() -> dict[str, int]:
    names = sorted(set(frompcode.builder_for(_ARCH).name_by_off.values()))
    return {n: i for i, n in enumerate(names)}


def test_the_register_order_a_slot_map_is_built_from_is_stable() -> None:
    """Slots are interned in the order `archregs` lists them, from an empty map.

    That order is the whole reason a second run hits: if it ever became
    non-deterministic -- a set, a dict built from one -- every wrapper would get
    a different slot map, every key would differ, and block mode would silently
    go back to compiling everything on every run with no test failing.
    """
    from microtaint.emulator import archregs

    for arch in (Architecture.AMD64, Architecture.ARM64, Architecture.RISCV64):
        names = list(archregs.for_arch(arch).all_names)
        assert names, f'{arch} lists no registers'
        assert names == list(archregs.for_arch(arch).all_names), (
            f'{arch} lists its registers in a different order each time, so no '
            f'two wrappers would agree on a slot map')
