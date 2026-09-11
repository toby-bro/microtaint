"""A p-code LOOP must be lowered or declined, and never quietly under-taint.

SLEIGH models some instructions as loops rather than as opcodes: a bit scan is
a walk over bit positions, so `bsf` and `bsr` arrive as a backward BRANCH to
the instruction's own IMARK.  The taint IR is straight-line by construction, so
the lowering used to decline them -- and a declined instruction refuses its
whole BLOCK, which the hook then SKIPS, so nothing computes its taint.  On a
static-glibc guest that was `bsf` inside memchr and strlen.

The loop is now UNROLLED, which is exact while it lasts because the predication
machinery already turns each exit test into a select: iteration k's writes land
under "still looping after k", which is the loop's meaning.  Two things have to
hold for that to be sound, and both are tested here:

  * what the unrolling cannot finish must be floored, not dropped;
  * a body that writes MEMORY cannot be floored at all, because the floor would
    have to name every address the remaining iterations might touch.  Those
    decline instead, which is worse than handling them and far better than
    being wrong.

Soundness is scored against Unicorn per-bit ground truth, not against the
engine's other path, and only on the registers the p-code actually WRITES: a
bit scan leaves CF, OF, SF, AF and PF architecturally undefined, the hardware
leaves something in them, and no engine reading that p-code can know what.
Counting the comparison is part of the test, because a scoring loop that
compared nothing would agree with anything.
"""
from __future__ import annotations

import random

import pytest

from benchmark.instruction_bank import isa_registers
from microtaint.taint_ir.frompcode import Unsupported, build_ir
from microtaint.types import Architecture

_ARCH = Architecture.AMD64

#: Bit scans, which SLEIGH lowers as a loop over bit positions.
_BIT_SCANS = {
    'bsf eax, eax': '0fbcc0',
    'bsf rax, rdx': '480fbcc2',
    'bsr eax, eax': '0fbdc0',
    'bsr rax, rdx': '480fbdc2',
}

#: String operations, whose p-code loop WRITES MEMORY an unbounded number of
#: times.  These must decline rather than be floored.
_STRING_OPS = {
    'rep movsb': 'f3a4',
    'rep stosb': 'f3aa',
    'rep movsq': 'f348a5',
}


@pytest.mark.parametrize(('label', 'hexs'), _BIT_SCANS.items(), ids=list(_BIT_SCANS))
def test_a_bit_scan_lowers(label: str, hexs: str) -> None:
    """It lowers at all.  Declining refuses the block, and a refused block is
    skipped, so this is the difference between analysing memchr and not."""
    prog = build_ir(_ARCH, bytes.fromhex(hexs))
    assert prog.cost() > 0, f'{label}: lowered to nothing'


@pytest.mark.parametrize(('label', 'hexs'), _STRING_OPS.items(), ids=list(_STRING_OPS))
def test_a_loop_that_writes_memory_declines(label: str, hexs: str) -> None:
    """The floor cannot cover an unbounded set of addresses.

    Unrolling a `rep` to a fixed limit and flooring the rest would leave the
    stores of every later iteration unmodelled, which is an under-taint.  The
    decline says so in as many words, so the next person reads a reason rather
    than a shrug.
    """
    with pytest.raises(Unsupported) as caught:
        build_ir(_ARCH, bytes.fromhex(hexs))
    assert 'memory' in str(caught.value), (
        f'{label} declined for an unexpected reason: {caught.value}')


def _ground_truth_sweep(hexs: str, n_vectors: int, seed: int) -> tuple[int, int, int]:
    """(compared, under-tainted, over-tainted) against Unicorn per-bit truth.

    Only the registers the p-code WRITES are in scope.  A bit scan leaves five
    flags architecturally undefined; the hardware moves them and SLEIGH never
    computes them, so scoring those would report the ISA as an engine bug.
    """
    from tests.oracle_harness import _uc_desc_amd64, ground_truth
    from tests.perop_c_bank import written_registers
    from tests.taint_ir_bank import ir_step

    code = bytes.fromhex(hexs)
    regs = list(isa_registers('AMD64'))
    names = [r.name for r in regs]
    desc = _uc_desc_amd64()
    try:
        written = written_registers(_ARCH, code)
    except Exception:
        written = None
    rng = random.Random(seed)
    compared = under = over = 0
    for _ in range(n_vectors):
        values = {k: rng.getrandbits(64) for k in names}
        taint = dict.fromkeys(names, 0)
        # A few tainted bits, low down, so the scan's answer really depends on
        # them: taint only the top of a word and the result rarely moves.
        for k in ('RAX', 'RDX', 'RCX'):
            if k in taint:
                taint[k] = rng.getrandbits(9)
        got, _mem, _cost = ir_step(_ARCH, code, regs, taint, values)
        truth = ground_truth(desc, code, taint, values)
        for k in list(desc.gp) + list(desc.flags):
            if written is not None and k in desc.flags and k not in written:
                continue
            t, g = truth.get(k, 0), got.get(k, 0)
            compared += 1
            if t & ~g:
                under += 1
            elif g & ~t:
                over += 1
    return compared, under, over


@pytest.mark.parametrize(('label', 'hexs'), _BIT_SCANS.items(), ids=list(_BIT_SCANS))
def test_a_bit_scan_never_under_taints(label: str, hexs: str) -> None:
    """The property that matters, against hardware rather than against
    ourselves.  Over-tainting is allowed and reported; under-tainting is not."""
    pytest.importorskip('unicorn')
    compared, under, over = _ground_truth_sweep(hexs, n_vectors=16, seed=hash(hexs) & 0xFFFF)
    assert compared > 50, (
        f'{label}: only {compared} outputs compared, which is too few for this '
        f'to mean anything')
    assert under == 0, (
        f'{label}: {under} of {compared} outputs UNDER-tainted (over-tainted '
        f'{over}, which is the acceptable direction)')


def test_the_sweep_can_actually_fail() -> None:
    """The scoring harness is not vacuous.

    An instruction whose taint the engine deliberately over-approximates must
    show OVER-taints through this same sweep; if it reported zero of both for
    everything, the loop above would be comparing nothing that moves.
    """
    pytest.importorskip('unicorn')
    compared, under, over = _ground_truth_sweep('0fbcc0', n_vectors=16, seed=3)
    assert compared > 50
    assert under == 0
    assert over > 0, (
        'the sweep found neither an under-taint nor an over-taint on an '
        'instruction known to over-approximate, so it is not measuring')


def test_an_unrolled_loop_is_not_free() -> None:
    """What unrolling costs, pinned so a change to the limit is visible.

    A bit scan lowers to thousands of operations where an ordinary instruction
    takes tens.  That is the price of not skipping the block, and it is paid
    only by blocks that contain one -- but it should not grow silently.
    """
    prog = build_ir(_ARCH, bytes.fromhex('0fbcc0'))
    assert 500 < prog.cost() < 20000, (
        f'a 32-bit bit scan lowers to {prog.cost()} operations; if the unroll '
        f'limit moved, say so here')


def test_a_narrow_scan_costs_about_half_a_wide_one() -> None:
    """The unrolling is bounded by the OPERAND, not by a flat maximum.

    A loop over bit positions cannot run more times than its widest operand has
    bits, so a 32-bit scan paying for 64 iterations was paying for 32 that can
    never run -- and each one costs around 80 operations, because every write
    in the body becomes a select.  Measured, that halved the 32-bit case with
    no change in precision at all.

    Pinned as a RATIO rather than as two numbers, so it keeps its meaning when
    the lowering gets cheaper for unrelated reasons.
    """
    narrow = build_ir(_ARCH, bytes.fromhex('0fbcc0')).cost()      # bsf eax,eax
    wide = build_ir(_ARCH, bytes.fromhex('480fbcc2')).cost()      # bsf rax,rdx
    assert narrow < wide * 0.75, (
        f'a 32-bit scan costs {narrow} operations against the 64-bit form at '
        f'{wide}; it should be bounded by its own operand width, so well under '
        f'the wide one rather than close to it')
