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


def test_a_recognised_scan_costs_about_what_an_ordinary_instruction_does() -> None:
    """The point of recognising the loop, stated as a number.

    It used to unroll to 2562 operations against 94 for an ordinary
    instruction, and those are paid on every execution of the block containing
    it, not once.  A closed form is the same order as anything else.
    """
    scan = build_ir(_ARCH, bytes.fromhex('0fbcc0')).cost()      # bsf eax,eax
    plain = build_ir(_ARCH, bytes.fromhex('4801d8')).cost()     # add rax,rbx
    assert scan < 6 * plain, (
        f'a bit scan lowers to {scan} operations against {plain} for an '
        f'ordinary instruction; if it stopped being recognised it is back to '
        f'unrolling, which is two orders of magnitude more')


@pytest.mark.parametrize(('label', 'hexs'), [
    ('pext eax,ebx,ecx', 'c4e262f5c1'),
    ('pdep eax,ebx,ecx', 'c4e263f5c1'),
])
def test_a_loop_that_is_not_a_count_is_refused(label: str, hexs: str) -> None:
    """The recogniser must say no to what it cannot justify.

    `pext` and `pdep` lift as loops too, and they are not counting operations:
    the monotonicity argument that lets two corners stand in for the whole taint
    cube says nothing about them.  Refusing is the whole reason the recogniser
    can be trusted on the forms it does accept, so it is worth a test of its
    own rather than being left implicit.
    """
    from microtaint.sleigh.lifter import get_context
    from microtaint.taint_ir.loopform import recognise_loop

    ops = get_context('AMD64').translate(bytes.fromhex(hexs), 0x1000).ops
    loops = [
        (i + (o.inputs[0].offset - (1 << 32)), i)
        for i, o in enumerate(ops)
        if o.opcode.name in ('BRANCH', 'CBRANCH') and o.inputs
        and o.inputs[0].space.name == 'const'
        and (o.inputs[0].offset & 0x80000000)
    ]
    assert loops, f'{label} no longer lifts to a p-code loop, so this proves nothing'
    for target, end in loops:
        assert recognise_loop(ops, target, end, be=False) is None, (
            f'{label}: the recogniser accepted a loop that is not a counting '
            f'operation, so the corner rule it will emit is unjustified')


def _taint_of(code: bytes, values: dict[str, int],
              taint: dict[str, int]) -> dict[str, int]:
    from tests.taint_ir_bank import ir_step

    regs = list(isa_registers('AMD64'))
    names = [r.name for r in regs]
    v = dict.fromkeys(names, 0) | values
    t = dict.fromkeys(names, 0) | taint
    got, _mem, _cost = ir_step(_ARCH, code, regs, t, v)
    return got


@pytest.mark.parametrize(('label', 'hexs'), _BIT_SCANS.items(), ids=list(_BIT_SCANS))
def test_the_closed_form_never_says_less_than_the_unrolling(
        label: str, hexs: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The fast path against the slow path it replaced, over many inputs.

    Unrolling is the reference here because it was the shipped answer and is
    exact while it lasts.  The closed form may be TIGHTER -- and measured, it
    is: `bsf eax,eax` went from 18 over-taints to 5 against hardware -- but it
    may never claim less taint than the unrolling did on any input, because
    that direction is the one that loses a leak.

    Recognition is disabled by making it refuse, which is the same door a loop
    it cannot identify goes through, so this also exercises the fallback.
    """
    import random

    from microtaint.taint_ir import frompcode as fp

    code = bytes.fromhex(hexs)
    regs = list(isa_registers('AMD64'))
    names = [r.name for r in regs]
    rng = random.Random(hash(hexs) & 0xFFFF)
    cases = []
    for _ in range(24):
        values = {k: rng.getrandbits(64) for k in names}
        taint = dict.fromkeys(names, 0)
        for k in ('RAX', 'RDX'):
            taint[k] = rng.getrandbits(12)
        cases.append((values, taint))

    fast = [_taint_of(code, v, t) for v, t in cases]

    # Both the shared Builders and their recognition memo have to go, or the
    # second half of this test re-reads the first half's answer and compares
    # the closed form against itself.
    monkeypatch.setattr(fp, 'recognise_loop', lambda *_a, **_k: None)
    monkeypatch.setattr(fp, '_BUILDERS', {})
    slow = [_taint_of(code, v, t) for v, t in cases]

    compared = 0
    for i, (a, b) in enumerate(zip(fast, slow, strict=True)):
        for k in set(a) | set(b):
            compared += 1
            missing = b.get(k, 0) & ~a.get(k, 0)
            assert not missing, (
                f'{label} case {i}: the closed form dropped {missing:#x} of '
                f'{k} that the unrolling reported '
                f'(unrolled {b.get(k, 0):#x}, closed form {a.get(k, 0):#x})')
    assert compared > 100, (
        f'{label}: only {compared} outputs compared between the two lowerings')


# ---------------------------------------------------------------------------
# A loop's EPILOGUE has to survive the loop.
# ---------------------------------------------------------------------------

#: `pext eax,ebx,ecx` and `pdep eax,ebx,ecx`.  Both lift as loops whose
#: back-edge is a CBRANCH, and both copy their accumulator into the destination
#: AFTER the loop -- which is the shape that went wrong.
_BIT_MOVES = {
    'pext eax,ebx,ecx': 'c4e262f5c1',
    'pdep eax,ebx,ecx': 'c4e263f5c1',
}


@pytest.mark.parametrize(('label', 'hexs'), _BIT_MOVES.items(), ids=list(_BIT_MOVES))
def test_a_loops_result_reaches_its_destination(label: str, hexs: str) -> None:
    """The destination must depend on the source.  It did not.

    A backward CBRANCH means "go round again", and the lowering ANDed that into
    the predicate without ever taking it back out.  The unrolling stops exactly
    when "go round again" folds to FALSE, so everything after the loop -- the
    copy of the accumulator into the destination -- was lowered under a false
    predicate and discarded.  `pext` came out carrying nothing but the previous
    taint of its own destination register.

    `bsf` never showed it: its back-edge is an unconditional BRANCH, which does
    not touch the predicate.

    Asserted as DEPENDENCE rather than against Unicorn on purpose.  Unicorn
    mis-executes these VEX forms -- it does not zero the destination's upper
    half for a 32-bit `pdep` -- so it is not a usable oracle here, while "a
    tainted source must reach the destination" needs no oracle at all and is
    exactly the property that was broken.
    """
    code = bytes.fromhex(hexs)
    clean = _taint_of(code, {'RBX': 0x5BC8FBBC, 'RCX': 0xB0C11FDE}, {})
    assert not clean.get('RAX', 0), (
        f'{label}: RAX is tainted with no tainted input, so this test cannot '
        f'tell a working destination from a broken one')
    for src in ('RBX', 'RCX'):
        got = _taint_of(code, {'RBX': 0x5BC8FBBC, 'RCX': 0xB0C11FDE},
                        {src: 0xFF})
        assert got.get('RAX', 0), (
            f'{label}: {src} is tainted and RAX is not, so the loop result '
            f'never reached the destination register')
