"""A counting opcode's taint is its reachable RANGE, not "any bit taints all".

SLEIGH lowers the leading-zero and population counts to p-code opcodes
(`LZCOUNT`, `POPCOUNT`) rather than to loops, and the taint rule for both used
to be `span & splat(NEZ(input_taint))`: one tainted input bit marked every bit
of the result.  Sound, and very loose in two specific ways.

A LEADING-ZERO count does not depend on the bits BELOW the highest set one at
all.  Flipping any of them leaves the top bit where it is, so the count cannot
move -- and a random 64-bit value has its top bit a long way up, so taint down
in the low byte moved nothing and was reported anyway.

A POPULATION count that can move by one moves only its bottom bit, never all
seven.

Both are fixed by the codebase's own corner trick: evaluate the count with every
tainted bit CLEARED and again with every tainted bit SET.  Those bracket the
reachable range (popcount is monotone in each bit, a leading-zero count is
antitone in the value), so the bits that can differ are the bits below where the
two endpoints first differ.

This is not a rare path.  `POPCOUNT` is how SLEIGH computes x86's PARITY flag,
so it sits under every arithmetic and logic instruction there, and `LZCOUNT`
carries `lzcnt`, AArch64 `clz`/`cls`, MIPS `clz`/`clo` and PowerPC `cntlzw`.
"""
from __future__ import annotations

import random

import pytest
from instruction_bank import isa_registers

from microtaint.types import Architecture

_ARCH = Architecture.AMD64

#: `lzcnt eax, ebx`, and `popcnt eax, ebx`.
_LZCNT = bytes.fromhex('f30fbdc3')
_POPCNT = bytes.fromhex('f30fb8c3')


def _step(code: bytes, values: dict[str, int],
          taint: dict[str, int]) -> dict[str, int]:
    from tests.taint_ir_bank import ir_step

    regs = list(isa_registers('AMD64'))
    names = [r.name for r in regs]
    v = dict.fromkeys(names, 0) | values
    t = dict.fromkeys(names, 0) | taint
    got, _mem, _cost = ir_step(_ARCH, code, regs, t, v)
    return got


def test_a_leading_zero_count_ignores_bits_below_the_top_one() -> None:
    """The property the old rule got wrong, stated directly.

    EBX has its highest set bit at 27 and the tainted bits are 0..7.  No
    assignment of those can move the top bit, so the count is the same number
    whatever they are, and its taint must be zero.
    """
    got = _step(_LZCNT, {'RBX': 0x0F000000}, {'RBX': 0xFF})
    assert got.get('RAX', 0) == 0, (
        f'lzcnt result tainted {got.get("RAX", 0):#x} by bits that cannot '
        f'reach the count: every tainted bit is below the highest set one')


def test_a_leading_zero_count_follows_a_bit_at_or_above_the_top_one() -> None:
    """And the other half, so the rule is not just returning zero.

    A test that only checked the clean case would pass against a rule that
    dropped the taint altogether, which would be an under-taint and far worse
    than the looseness being fixed.
    """
    above = _step(_LZCNT, {'RBX': 0x0F000000}, {'RBX': 0x10000000})
    assert above.get('RAX', 0) != 0, (
        'a tainted bit ABOVE the highest set one can raise it and change the '
        'count, so the result must be tainted')
    at = _step(_LZCNT, {'RBX': 0x0F000000}, {'RBX': 0x08000000})
    assert at.get('RAX', 0) != 0, (
        'a tainted bit AT the highest set one can clear it and drop the count, '
        'so the result must be tainted')


def test_a_population_count_narrows_to_the_reachable_range() -> None:
    """One tainted bit moves the count by one, so it cannot touch every bit.

    The span for a 32-bit popcount is six bits wide; a count that can only be
    n or n+1 differs in far fewer, and which ones depends on n.
    """
    got = _step(_POPCNT, {'RBX': 0x0000FFFF}, {'RBX': 0x1})
    mask = got.get('RAX', 0)
    assert mask != 0, 'flipping a tainted bit changes the population count'
    assert mask.bit_count() < 6, (
        f'popcount taint {mask:#x} covers {mask.bit_count()} bits for a count '
        f'that can only move by one')


def _sweep(code: bytes, arch: Architecture, isa: str, n: int,
           seed: int) -> tuple[int, int, int]:
    """(compared, under, over) against Unicorn per-bit truth."""
    from tests.oracle_harness import UC_DESCS, ground_truth
    from tests.perop_c_bank import written_registers
    from tests.taint_ir_bank import ir_step

    regs = list(isa_registers(isa))
    names = [r.name for r in regs]
    desc = UC_DESCS[isa]()
    written = written_registers(arch, code)
    rng = random.Random(seed)
    compared = under = over = 0
    for _ in range(n):
        values = {k: rng.getrandbits(64) for k in names}
        taint = dict.fromkeys(names, 0)
        for k in list(desc.gp)[:4]:
            if k in taint:
                taint[k] = rng.getrandbits(8)
        got, _m, _c = ir_step(arch, code, regs, taint, values)
        truth = ground_truth(desc, code, taint, values)
        for k in list(desc.gp) + list(desc.flags):
            if k in desc.flags and written is not None and k not in written:
                continue
            t, g = truth.get(k, 0), got.get(k, 0)
            compared += 1
            if t & ~g:
                under += 1
            elif g & ~t:
                over += 1
    return compared, under, over


def test_a_leading_zero_count_is_exact_against_hardware() -> None:
    """Not merely sound: EXACT, over a sweep, scored against Unicorn.

    Counting what was compared is part of the test; a sweep that compared
    nothing would report zero of both and agree with anything.
    """
    pytest.importorskip('unicorn')
    compared, under, over = _sweep(_LZCNT, _ARCH, 'AMD64', n=16, seed=5)
    assert compared > 50, f'only {compared} outputs compared'
    assert under == 0, f'{under} of {compared} outputs UNDER-tainted'
    assert over == 0, (
        f'{over} of {compared} outputs over-tainted; the corner rule makes a '
        f'leading-zero count exact, so any over-taint means it regressed to an '
        f'approximation')


def test_it_is_not_an_x86_rule() -> None:
    """AArch64 `clz` goes through the same opcode and must be exact too.

    The rule is written against p-code's LZCOUNT rather than against an
    instruction, so an ISA the change was never tested on should get it for
    free.  This is the check that says so.
    """
    pytest.importorskip('unicorn')
    from instruction_bank import load_bank

    spec = load_bank(isas={'ARM64'})['ARM64']
    forms = [i for i in spec.instructions
             if (i.asm or i.label).lower().startswith('clz ')]
    assert forms, 'the ARM64 bank holds no clz form, so this proves nothing'
    for ins in forms:
        compared, under, over = _sweep(bytes(ins.bytes), spec.arch, 'ARM64',
                                       n=8, seed=6)
        assert compared > 20, f'{ins.asm}: only {compared} outputs compared'
        assert under == 0, f'{ins.asm}: {under} UNDER-tainted'
        assert over == 0, f'{ins.asm}: {over} of {compared} over-tainted'
