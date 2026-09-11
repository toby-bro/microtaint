"""A region publishes the register values a later region of its block reads.

Between regions of one block the runtime threads register VALUES, not just
taint: the next region's loads compute their addresses from them.  It used to
do that by copying the whole register file twice per region, once to seed the
output array's value half so a slot the program does not write reads back its
current value, and once to thread the published values back.  That was 210
slots on AMD64 where a handful matter, and the plan has known which handful
since it was compiled.

So the plan now carries them and the runtime touches only those.  The invariant
that makes it safe is narrow and worth stating on its own: **the slots a region
publishes must cover every register value a LATER region of the same block
reads.**  Miss one and that region reads a stale value, computes an address
from it, resolves the load somewhere else, and reports the shadow of the wrong
word: an under-taint that looks like an answer.

`n_pub < 0` keeps the old wholesale copies, so a caller that does not supply the
list is unaffected rather than wrong.
"""
from __future__ import annotations

import pytest

from microtaint.taint_ir import frompcode
from microtaint.taint_ir.blockcompile import compile_block
from microtaint.taint_ir.blocks import Region
from microtaint.types import Architecture

_ARCH = Architecture.AMD64
_BASE = 0x401000

#: `mov rax,[rbx]; mov rcx,[rax]; add rax,rcx`.  The second load's ADDRESS is
#: the first load's value, which is exactly what the two-pass protocol cannot
#: do in one program, so the block is cut and region 0 has to hand RAX over.
_THREADED = bytes.fromhex('488b03488b084801c8')


def _regions(code: bytes) -> list[Region]:
    """The regions AS COMPILED.  `plan_block` alone is not enough: the pruning
    that decides what a region publishes happens in `_compile_regions`, and
    reading the unpruned lowering shows every region publishing everything."""
    layout = {n: i for i, n in enumerate(
        sorted(set(frompcode.builder_for(_ARCH).name_by_off.values())))}
    got = compile_block(_ARCH, code, _BASE, layout, cache=False)
    assert got is not None, 'the block did not lower'
    regions = got[1]
    assert all(r.prog is not None for r in regions)
    return regions


def _published(r: Region) -> set[int]:
    return {k[1] for k, _n in r.prog.outputs           # type: ignore[union-attr]
            if isinstance(k, tuple) and k[0] == 'regv'}


def _values_read(r: Region) -> set[int]:
    return {k[1] for (kind, k) in r.prog.inputs        # type: ignore[union-attr]
            if kind == 'v' and isinstance(k, tuple) and k[0] == 'reg'}


def test_the_example_really_is_cut_into_regions() -> None:
    """The premise.  With one region there is nothing to thread and every
    assertion below would hold of a block that hands over nothing."""
    regions = _regions(_THREADED)
    assert len(regions) >= 2, (
        f'{len(regions)} region(s): a load whose address comes from another '
        f'load must be cut, or this file is testing nothing')


def test_the_pruning_keeps_every_value_a_later_region_reads() -> None:
    """The contract of `_keep_only_needed_values`, stated against the lowering
    it prunes.

    A region does not have to publish every value a later region reads: a slot
    it never WRITES keeps the value it already had in the threading array, and
    that is the whole reason copying only the published slots is sound.  What
    it must publish is every value it DOES write that someone later reads, and
    that is what the pruning decides.

    So compare the two: the unpruned lowering says what the region writes, the
    compiled one says what survived, and the difference must be exactly the
    values nobody reads.
    """
    from microtaint.taint_ir.blocks import plan_block

    raw = plan_block(_ARCH, _THREADED, _BASE, abs_ram=True, max_acc=8)
    compiled = _regions(_THREADED)
    assert len(raw) == len(compiled), (len(raw), len(compiled))

    for i, (r_raw, r_cut) in enumerate(zip(raw, compiled, strict=True)):
        later: set[int] = set()
        for nxt in compiled[i + 1:]:
            later |= _values_read(nxt)
        writes = _published(r_raw)          # before pruning: what it writes
        want = writes & later               # what it writes AND someone reads
        got = _published(r_cut)             # what survived the pruning
        assert got == want, (
            f'region {i} publishes {sorted(got)} where it writes '
            f'{sorted(writes)} and later regions read {sorted(later)}: the '
            f'pruning should have kept exactly {sorted(want)}')


def test_the_last_region_publishes_nothing() -> None:
    """Nothing in the block can read it, and the next BLOCK re-reads the
    register file from the CPU, so the last region hands over nothing and pays
    for neither copy."""
    regions = _regions(_THREADED)
    assert _published(regions[-1]) == set() or len(regions) == 1, (
        f'the last region publishes {sorted(_published(regions[-1]))}, which '
        f'nothing can read')


@pytest.mark.parametrize(('label', 'code'), [
    ('load then index with it', '488b03488b084801c8'),
    ('store then load it back', '488903488b0b4801c8'),
    ('zero then index', '4831c00fb60c03'),
])
def test_the_published_set_is_never_larger_than_what_is_read(
        label: str, code: bytes | str) -> None:
    """Publishing a value keeps alive whatever computed it, and for a value
    loaded from memory that means resolving the load: a guest read for a word
    nobody looks at.  So the set must be tight as well as sufficient."""
    regions = _regions(bytes.fromhex(code) if isinstance(code, str) else code)
    for i, r in enumerate(regions[:-1]):
        later: set[int] = set()
        for nxt in regions[i + 1:]:
            later |= _values_read(nxt)
        extra = _published(r) - later
        assert not extra, (
            f'{label}: region {i} publishes {sorted(extra)} that no later '
            f'region reads, so whatever computes them is kept alive for nothing')


def test_the_runtime_agrees_with_one_instruction_at_a_time() -> None:
    """The behavioural end of it: threading only the published slots must give
    the same answer as running each instruction as its own block, which is the
    reference the block runtime is held to everywhere else."""
    from tests.test_block_runtime_c import _VALUES, _run

    layout = {n: i for i, n in enumerate(
        sorted(set(frompcode.builder_for(_ARCH).name_by_off.values())))}
    # Store a byte, load it back, index with it: the same shape the runtime
    # tests use, and the only one where the second access is guaranteed to
    # land somewhere mapped.  Loading a pointer out of arbitrary arena bytes
    # gives an address the runtime rightly declines.
    seq = [bytes.fromhex(h) for h in ('8855ef', '0fb645ef', '0fb60c03')]
    values = dict(_VALUES)
    taint = {'RDX': 0xFF}
    as_block, _ = _run(layout, seq, values, taint, as_block=True)
    per_instr, _ = _run(layout, seq, values, taint, as_block=False)
    assert any(per_instr.values()), (
        'the reference run produced no taint at all, so this compares nothing')
    assert as_block == per_instr, (
        'threading only the published slots changed the answer')
