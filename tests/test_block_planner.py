# ruff: noqa: PLC0415
"""The block planner: maximal regions that lower as one program, covering all of it.

A block hook fires once per basic block, so the taint of a block wants to be one
compiled program.  A whole block does not always lower -- the two-pass memory
protocol is sound only while no address depends on a value loaded in the same
program, and inside a block a load and an address derived from it often sit
together -- so the block becomes a short sequence of regions, cut where that
fails rather than relaxing the check.

What must hold, whatever the cutting does: the regions cover the block exactly,
in order, with no gap and no overlap.  A gap is silently unanalysed code.

Instruction boundaries come from SLEIGH's IMARKs, so the planner knows no
architecture; the ISA-general half of that is exercised by planning the same
shapes on more than one.
"""
from __future__ import annotations

import pytest

from microtaint.taint_ir.blocks import instruction_starts, plan_block
from microtaint.taint_ir.frompcode import LIFT_BASE, Emit, PointerPolicy
from microtaint.types import Architecture

#: A straight-line run ending in the backward branch every loop body ends in.
_LOOP_BODY = (bytes.fromhex('4883c001')      # add rax, 1
              + bytes.fromhex('4839d8')      # cmp rax, rbx
              + bytes.fromhex('7cf7'))       # jl -9, back to the top

#: A load feeding the ADDRESS of a later load.  The two-pass protocol cannot
#: resolve that in one program, so the planner has to cut here.
_DEPENDENT_ADDRESS = (bytes.fromhex('4883c001')      # add rax, 1
                      + bytes.fromhex('488b4508')    # mov rax, [rbp+8]
                      + bytes.fromhex('488b18')      # mov rbx, [rax]
                      + bytes.fromhex('4801d8'))     # add rax, rbx


def _n_instructions(arch: Architecture, code: bytes) -> int:
    from microtaint.sleigh.lifter import get_context
    key = arch.value if hasattr(arch, 'value') else str(arch)
    return len(instruction_starts(get_context(key).translate(code, LIFT_BASE).ops))


def _assert_covers(regions, arch: Architecture, code: bytes) -> None:
    """In order, contiguous, and covering every instruction exactly once."""
    assert regions, 'the planner returned no regions at all'
    total = _n_instructions(arch, code)
    assert sum(r.count for r in regions) == total, (
        f'regions cover {sum(r.count for r in regions)} of {total} instructions')
    expect_first = 0
    expect_addr = LIFT_BASE
    for r in regions:
        assert r.first == expect_first, f'gap or overlap at instruction {expect_first}'
        assert r.addr == expect_addr, f'gap or overlap at address {expect_addr:#x}'
        assert r.count >= 1
        expect_first += r.count
        expect_addr = r.end
    assert expect_addr == LIFT_BASE + len(code), (
        f'regions end at {expect_addr:#x}, block ends at {LIFT_BASE + len(code):#x}')


def test_a_whole_block_becomes_one_region() -> None:
    regions = plan_block(Architecture.AMD64, _LOOP_BODY)
    _assert_covers(regions, Architecture.AMD64, _LOOP_BODY)
    assert len(regions) == 1, f'expected one region, got {len(regions)}'
    assert regions[0].prog is not None


def test_a_dependent_address_cuts_the_block_without_losing_any_of_it() -> None:
    arch = Architecture.AMD64
    regions = plan_block(arch, _DEPENDENT_ADDRESS)
    _assert_covers(regions, arch, _DEPENDENT_ADDRESS)
    assert len(regions) > 1, (
        "a load feeding a later load's address should have cut the block; "
        'if this stops being true the two-pass protocol has changed and the '
        'reason for cutting needs rechecking')


@pytest.mark.parametrize('code', [_LOOP_BODY, _DEPENDENT_ADDRESS])
def test_every_instruction_is_accounted_for(code: bytes) -> None:
    """The property that matters most: no instruction falls between regions.

    A region with prog=None is the planner saying 'this one instruction lowers
    nowhere, send it down the per-instruction path'.  That is coverage, not a
    gap; an instruction in NO region would be silently unanalysed.
    """
    arch = Architecture.AMD64
    _assert_covers(plan_block(arch, code), arch, code)


def test_regions_are_maximal() -> None:
    """Greedy means each region is as long as it can be: the instruction just
    past a cut must genuinely not fit.  A planner that cut early would still
    cover the block and still be correct, only slower, so this is the test that
    keeps it fast."""
    from microtaint.taint_ir.blocks import _translate
    from microtaint.taint_ir.frompcode import Builder, Unsupported

    arch = Architecture.AMD64
    builder = Builder(arch, False, PointerPolicy.CONCRETE)
    regions = plan_block(arch, _DEPENDENT_ADDRESS, builder=builder)
    ops = _translate(arch, _DEPENDENT_ADDRESS, LIFT_BASE)
    marks = instruction_starts(ops)
    n = len(marks)
    for r in regions[:-1]:
        i, j = r.first, r.first + r.count + 1      # one instruction past the cut
        if j > n:
            continue
        lo = marks[i][0]
        hi = marks[j][0] if j < n else len(ops)
        end = marks[j][1] if j < n else LIFT_BASE + len(_DEPENDENT_ADDRESS)
        with pytest.raises(Unsupported):
            builder.build(ops[lo:hi], end, emit=Emit.BOTH, block=True)


def test_the_planner_is_isa_general() -> None:
    """Nothing in the planner names an architecture; the IMARKs do the work."""
    arm = (bytes.fromhex('20000091')      # add x0, x1, #0
           + bytes.fromhex('4100008b'))   # add x1, x2, x0
    regions = plan_block(Architecture.ARM64, arm)
    _assert_covers(regions, Architecture.ARM64, arm)
    assert any(r.prog is not None for r in regions), 'nothing lowered on ARM64'


@pytest.mark.parametrize('isa', ['AMD64', 'ARM64', 'MIPS64BE', 'PPC32BE', 'RISCV64',
                                 'AMD64_SIMD', 'ARM64_SIMD'])
def test_the_planner_covers_blocks_on_every_bank_isa(isa: str) -> None:
    """Coverage and contiguity on every architecture the bank carries.

    The planner splits by SLEIGH's IMARKs and never names an architecture, so
    the way this stops being true is not a wrong answer on one ISA but a silent
    decline on all of a big-endian one, or an off-by-one in the boundaries that
    only a wider register file exposes.  Blocks here are consecutive bank
    instructions rather than real control flow: not realistic, but it exercises
    exactly what the planner does.
    """
    import sys
    from pathlib import Path

    bench = Path(__file__).resolve().parent.parent / 'benchmark'
    if str(bench) not in sys.path:
        sys.path.insert(0, str(bench))
    from instruction_bank import load_bank  # type: ignore[import-not-found]

    from microtaint.taint_ir.blocks import _translate
    from microtaint.taint_ir.frompcode import Builder

    bank = load_bank()
    if isa not in bank:
        pytest.skip(f'{isa} is not in the bank')
    spec = bank[isa]
    key = spec.arch.value if hasattr(spec.arch, 'value') else str(spec.arch)
    builder = Builder(spec.arch, key.endswith('BE'), PointerPolicy.CONCRETE)

    per_block, planned = 5, 0
    for start in range(0, min(len(spec.instructions), per_block * 12), per_block):
        chunk = spec.instructions[start:start + per_block]
        if len(chunk) < 2:
            break
        code = b''.join(bytes(i.bytes) for i in chunk)
        marks = instruction_starts(_translate(spec.arch, code, LIFT_BASE))
        regions = plan_block(spec.arch, code, builder=builder)

        assert regions, f'{isa}: no regions for a {len(marks)}-instruction block'
        assert sum(r.count for r in regions) == len(marks), (
            f'{isa}: regions cover {sum(r.count for r in regions)} of {len(marks)}')
        pos, addr = 0, LIFT_BASE
        for r in regions:
            assert r.first == pos, f'{isa}: gap or overlap at instruction {pos}'
            assert r.addr == addr, f'{isa}: gap or overlap at address {addr:#x}'
            pos += r.count
            addr = r.end
        assert addr == LIFT_BASE + len(code), (
            f'{isa}: regions end at {addr:#x}, block ends at '
            f'{LIFT_BASE + len(code):#x}')
        assert any(r.prog is not None for r in regions), (
            f'{isa}: nothing in this block lowered at all, which on a '
            f'big-endian or vector bank usually means a silent decline rather '
            f'than a genuinely unlowerable block')
        planned += 1
    assert planned >= 2, f'{isa}: only {planned} blocks planned'
