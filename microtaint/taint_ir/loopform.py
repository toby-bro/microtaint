"""Recognise what a p-code LOOP computes, by running it rather than reading it.

SLEIGH models a few instructions as loops over bit positions instead of as
opcodes, and the taint IR is straight-line by construction, so a loop has to be
unrolled.  Unrolling is exact but expensive: measured, `bsf eax,ebx` lowers to
2562 IR operations against 85 for an ordinary instruction, and those operations
are paid on every execution of the block that contains it.

A closed form is two orders of magnitude cheaper.  The obvious way to reach one
is to pattern-match the shape SLEIGH emits, and that is exactly what this module
does NOT do: a pattern is a fact about one instruction set's specification, it
breaks when Ghidra rewrites the spec, and it cannot say whether the match was
right.  This runs the loop CONCRETELY on chosen inputs and asks what function of
its input it computed.  If the answers agree with a known counting form on every
probe, that form is used; if they do not, the caller unrolls exactly as before.

Recognition is therefore semantic and self-checking, and the ISA never comes
into it.  As it happens only x86 needs it -- surveyed across the instruction
bank, AMD64's `bsf`, `bsr` and `tzcnt` are the only looping forms, because every
other instruction set's count lifts to the `LZCOUNT` opcode -- but nothing here
knows that, so an instruction set that later spells one as a loop is recognised
without a change.

WHY A COUNT, specifically.  Each form below is monotone or antitone in every
input bit: setting a bit can only lower a trailing-zero count, only raise a
highest-set-bit index, only raise a population count.  That is what makes the
two corners of the taint cube -- every tainted bit cleared, and every one set --
bracket the whole reachable range, which is what the taint rule needs and what
`frompcode` already does for `POPCOUNT` and `LZCOUNT`.  A loop computing
something NOT monotone would need a different argument, so it is not recognised.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pypcode import PcodeOp, Varnode

from microtaint.sleigh.constfold import _eval

__all__ = ['CounterKind', 'LoopForm', 'VnKey', 'recognise_loop']

#: A varnode's coordinates: space, byte offset, byte size.  Plain data, so a
#: recognition can be cached and reused across translations of the same
#: instruction without holding a lifter object alive.
type VnKey = tuple[str, int, int]

#: Concrete steps one probe may take before the loop is called non-terminating.
#: A bit scan over a 64-bit operand takes at most 64 iterations of a handful of
#: ops; this is far above that and far below a hang.
_MAX_STEPS = 4096


class CounterKind(StrEnum):
    """What a recognised loop computes, over its operand's `width` bits."""

    #: Number of trailing zeros; `width` when the operand is zero.  `bsf`, `tzcnt`.
    CTZ = 'ctz'
    #: Number of leading zeros; `width` when the operand is zero.  `lzcnt`.
    CLZ = 'clz'
    #: Index of the highest set bit.  Undefined at zero.  `bsr`.
    MSB_INDEX = 'msb_index'
    #: Number of set bits.
    POPCOUNT = 'popcount'


@dataclass(frozen=True)
class LoopForm:
    """A loop recognised as `kind` of `src`, left in `dst`.

    `zero_is_special` records that a ZERO operand does not produce the count the
    rest of the form would predict.  It is not a detail: `bsf` leaves 0 where
    the trailing-zero count is the operand width, and that single point is what
    breaks the monotonicity the whole corner argument rests on.  Setting a bit
    can only LOWER a trailing-zero count -- except from zero, where it raises
    it.  The caller must widen to the full span wherever zero is reachable.
    `tzcnt` has no such point, and pays nothing for it.
    """

    kind: CounterKind
    src: VnKey
    dst: VnKey
    width: int
    zero_is_special: bool


def _ref(x: int, kind: CounterKind, width: int) -> int | None:
    """The reference answer, or None where the form leaves it undefined."""
    mask = (1 << width) - 1
    x &= mask
    if kind is CounterKind.CTZ:
        return width if x == 0 else (x & -x).bit_length() - 1
    if kind is CounterKind.CLZ:
        return width if x == 0 else width - x.bit_length()
    if kind is CounterKind.MSB_INDEX:
        return None if x == 0 else x.bit_length() - 1
    return x.bit_count()


#: A compiled varnode access: either a literal, or the byte cells it spans with
#: the shift each contributes.  Byte cells because p-code varnodes OVERLAP --
#: `bsf` writes its answer to `register:0x0:4` and then zero-extends it into
#: `register:0x0:8`, and a state keyed by whole varnodes would hold those as two
#: unrelated values.
type _Access = tuple[int, tuple[tuple[tuple[str, int], int], ...]]

#: One compiled op: opcode, input accesses, output access (or None), input bit
#: widths, output bit width.
type _Step = tuple[str, tuple[_Access, ...], _Access | None,
                   list[int], int]


def _plan(vn: Varnode, be: bool) -> _Access:
    if vn.space.name == 'const':
        return (vn.offset, ())
    cells = tuple(
        (((vn.space.name, vn.offset + i)), 8 * ((vn.size - 1 - i) if be else i))
        for i in range(vn.size)
    )
    return (0, cells)


def _compile(ops: list[PcodeOp], be: bool) -> list[_Step]:
    """Turn the p-code into plain tuples once, so a probe never touches a lifter
    object again.

    Reading `vn.space.name` crosses into the binding on every access, and a
    probe does that thousands of times per input; hoisting it out of the loop is
    most of what makes recognising a 64-bit scan cheaper than unrolling it.
    """
    out: list[_Step] = []
    for o in ops:
        out.append((
            o.opcode.name,
            tuple(_plan(v, be) for v in o.inputs),
            _plan(o.output, be) if o.output is not None else None,
            [v.size * 8 for v in o.inputs],
            o.output.size * 8 if o.output is not None else 0,
        ))
    return out


class _Machine:
    """Byte-addressed concrete state for one instruction's p-code."""

    __slots__ = ('mem',)

    def __init__(self) -> None:
        self.mem: dict[tuple[str, int], int] = {}

    def read(self, acc: _Access) -> int:
        literal, cells = acc
        if not cells:
            return literal
        mem = self.mem
        v = 0
        for key, shift in cells:
            b = mem.get(key)
            if b:
                v |= b << shift
        return v

    def write(self, acc: _Access, value: int) -> None:
        mem = self.mem
        for key, shift in acc[1]:
            mem[key] = (value >> shift) & 0xFF


def _s32(v: int) -> int:
    return v - (1 << 32) if v & 0x80000000 else v


_OPAQUE = frozenset({'LOAD', 'STORE', 'CALLOTHER', 'CALL', 'CALLIND',
                     'BRANCHIND', 'RETURN'})


def _run_to(prog: list[_Step], machine: _Machine, stop_after: int) -> bool:  # noqa: C901
    """Execute from step 0 until control passes beyond `stop_after`.

    Returns False if anything at all was not modelled -- an unknown opcode, a
    branch out of the instruction, a memory access, or a step budget exhausted.
    A probe that cannot be trusted must not be used, so every uncertainty is a
    refusal rather than a guess.
    """
    pc = 0
    steps = 0
    n = len(prog)
    read = machine.read
    while 0 <= pc < n:
        if pc > stop_after:
            return True
        steps += 1
        if steps > _MAX_STEPS:
            return False
        name, ins, outp, bits, out_bits = prog[pc]
        if name == 'IMARK':
            pc += 1
            continue
        if name == 'BRANCH' or name == 'CBRANCH':
            if name == 'CBRANCH' and not read(ins[1]):
                pc += 1
                continue
            literal, cells = ins[0]
            if cells:
                return True              # a ram target: leaves the instruction
            pc += _s32(literal)
            continue
        if name in _OPAQUE:
            return False                 # state this machine does not model
        got = _eval(name, [read(a) for a in ins], bits, out_bits)
        if got is None:
            return False
        if outp is not None:
            machine.write(outp, got)
        pc += 1
    return True


def _loop_io(ops: list[PcodeOp], target: int, end: int,
             ) -> tuple[Varnode, Varnode] | None:
    """(source, destination) for the loop ending at `end`.

    The source is the one REGISTER the instruction reads before writing, over
    everything up to the loop's end -- not just the loop body.  `tzcnt` is why:
    it copies its operand into a temporary BEFORE the loop and the body touches
    only that temporary, so a body-only search finds no register at all and
    refuses an instruction it should recognise.

    The destination is the varnode the BODY writes that something after the loop
    reads.  Each must be unique, or this is not the single-input counter the
    caller is looking for.
    """
    prefix = ops[:end + 1]
    written_before: set[tuple[str, int, int]] = set()
    srcs: dict[tuple[str, int, int], Varnode] = {}
    for o in prefix:
        for v in o.inputs:
            key = (v.space.name, v.offset, v.size)
            if v.space.name == 'register' and key not in written_before:
                srcs[key] = v
        if o.output is not None:
            written_before.add(
                (o.output.space.name, o.output.offset, o.output.size))
    if len(srcs) != 1:
        return None

    body = ops[target:end + 1]

    later_reads: set[tuple[str, int, int]] = set()
    for o in ops[end + 1:]:
        for v in o.inputs:
            later_reads.add((v.space.name, v.offset, v.size))
    dsts = [o.output for o in body
            if o.output is not None
            and (o.output.space.name, o.output.offset, o.output.size) in later_reads]
    if len(dsts) != 1:
        return None
    return next(iter(srcs.values())), dsts[0]


def _probe_values(width: int) -> list[int]:
    """Inputs to compare the loop against the reference forms on.

    Three jobs, and the cost is dominated by the first two because a scan of a
    HIGH single bit is a long loop:

      * single bits pin the direction and the scale (does bit i give i, or
        width-1-i);
      * a few PAIRS separate a trailing-zero count from a highest-set-bit index,
        which agree on every single-bit input and disagree on every other;
      * a deterministic spread of multi-bit values VERIFIES the guess, and does
        the most work per probe: a random word has a lowest and a highest set
        bit in different places, so it separates the forms too, and it costs
        almost nothing because its scan terminates almost immediately.

    Narrow operands take every single bit; wide ones take a spread, because
    sixty-four of them cost four thousand loop iterations to say what twenty say
    just as well, and the multi-bit probes are what would catch a form that
    agreed on all of them anyway.
    """
    mask = (1 << width) - 1
    if width <= 16:
        bits = list(range(width))
    else:
        bits = sorted(
            {0, 1, 2, 3, width // 4, width // 2, 3 * width // 4,
             width - 4, width - 3, width - 2, width - 1}
            | set(range(0, width, 8)),
        )
    out = [1 << i for i in bits]
    out += [
        (1 << a) | (1 << b)
        for a, b in ((0, width - 1), (1, 2), (2, 3), (0, width // 2),
                     (width // 2, width - 1), (width - 2, width - 1))
        if a != b
    ]
    out += [mask, mask >> 1, (mask >> 1) + 1, 3, 6, 0xF0]
    v = 0x9E3779B97F4A7C15
    for _ in range(24):
        v = (v * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        out.append(v & mask)
    return [x for x in out if x]          # zero is asked separately


def recognise_loop(ops: list[PcodeOp], target: int, end: int,  # noqa: C901
                   be: bool) -> LoopForm | None:
    """What `ops[target:end + 1]` computes, or None to unroll it instead.

    `target` and `end` bracket the loop body: the branch at `end` jumps back to
    `target`.  Every refusal below is deliberate -- a loop this cannot identify
    with certainty is one the caller must keep handling the slow, exact way.
    """
    io = _loop_io(ops, target, end)
    if io is None:
        return None
    src, dst = io
    width = src.size * 8
    if width not in (8, 16, 32, 64):
        return None

    prog = _compile(ops, be)
    src_acc, dst_acc = _plan(src, be), _plan(dst, be)

    def observed(x: int, dst_seed: int) -> int | None:
        machine = _Machine()
        machine.write(dst_acc, dst_seed)
        machine.write(src_acc, x)
        if not _run_to(prog, machine, end):
            return None
        return machine.read(dst_acc)

    candidates = set(CounterKind)
    for x in _probe_values(width):
        got = observed(x, 0)
        if got is None:
            return None
        for kind in list(candidates):
            want = _ref(x, kind, width)
            if want is not None and got != want:
                candidates.discard(kind)
        if not candidates:
            return None
    if len(candidates) != 1:
        return None                      # ambiguous: probe harder before trusting it
    kind = next(iter(candidates))

    # Zero, asked on its own.  Seeded with two different destination values so
    # that an instruction which PRESERVES its destination there is refused
    # outright rather than modelled from one arbitrary observation.
    zero_a = observed(0, 0x5A5A5A5A5A5A5A5A)
    zero_b = observed(0, 0xA5A5A5A5A5A5A5A5)
    if zero_a is None or zero_b is None or zero_a != zero_b:
        return None
    return LoopForm(
        kind=kind,
        src=(src.space.name, src.offset, src.size),
        dst=(dst.space.name, dst.offset, dst.size),
        width=width,
        zero_is_special=(zero_a != _ref(0, kind, width)),
    )
