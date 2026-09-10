"""Split a basic block into the largest regions that lower as one program.

A block hook fires once for a whole basic block, at 0.04 ns/instruction against
~18 for a per-instruction code hook, so the taint of a block wants to be one
compiled program rather than fifteen.  Lowering it that way is also cheaper in
itself: dead-code elimination then runs across the block and deletes the flags a
later instruction overwrites, which no per-instruction pass can see.

A whole block does not always lower.  The two-pass memory protocol is sound only
while no address depends on a value loaded in the same program, and inside a
block a load and an address derived from it often sit together.  Where that
happens the region is CUT rather than the check relaxed, and the block becomes a
short sequence of regions instead of one.

Instruction boundaries come from SLEIGH's own IMARKs, so nothing here knows an
architecture: one translate() call covers the block and marks where each
instruction starts.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pypcode import PcodeOp

from microtaint.taint_ir.frompcode import (
    LIFT_BASE,
    Builder,
    Emit,
    Unsupported,
    builder_for,
)
from microtaint.taint_ir.ir import IRProg
from microtaint.types import ArchLike

#: Lower instructions [i, j) of the block: the program, and the index it
#: had to cut at (None when it lowered the whole run).
Lower = Callable[[int, int], tuple[IRProg | None, int | None]]

__all__ = ['Region', 'instruction_starts', 'plan_block']


@dataclass(frozen=True)
class Region:
    """One lowered run of consecutive instructions.

    `first` and `count` index the block's instructions; `addr` and `end` are
    guest addresses, so a caller can map a region back to what it covers.
    """

    first: int
    count: int
    addr: int
    end: int
    prog: IRProg | None          # None if this region lowers nowhere


def _translate(arch: ArchLike, code: bytes, base: int) -> list[PcodeOp]:
    from microtaint.sleigh.lifter import get_context  # noqa: PLC0415 - the lifter is expensive to import eagerly
    key = arch.value if hasattr(arch, 'value') else str(arch)
    return get_context(key).translate(code, base).ops


def instruction_starts(ops: list[PcodeOp]) -> list[tuple[int, int, int]]:
    """(op index, address, size) for each instruction, from the IMARKs."""
    out = []
    for i, op in enumerate(ops):
        if op.opcode.name == 'IMARK' and op.inputs:
            out.append((i, op.inputs[0].offset, op.inputs[0].size))
    return out


def plan_block(arch: ArchLike, code: bytes, base: int = LIFT_BASE, *,
               emit: Emit = Emit.BOTH, abs_ram: bool = False,
               max_acc: int | None = None,
               builder: Builder | None = None) -> list[Region]:
    """Greedy maximal regions covering `code`, in order.

    `abs_ram` says `base` is where the code really runs, so a `ram` varnode --
    what a PC-relative operand lifts to once the displacement has folded into
    the program counter -- names a live guest address and can be lowered as a
    memory access.  It is off by default because the default base is synthetic.

    The whole block is tried first, because that is the common case and it costs
    one lowering; only when it declines is the block walked in shorter runs.
    That first attempt is `_greedy`'s own first step, not a separate fast path:
    when it was written out here as well, every block that declined paid for the
    whole-block lowering TWICE, once here and once inside `_greedy`.

    A single instruction that will not lower at all still gets a Region with
    `prog=None`, so the caller sees complete coverage of the block and can send
    that one instruction down the per-instruction path rather than having to
    work out what is missing.
    """
    if builder is None:
        # The shared one: a Builder costs a register-map build (measured at
        # ~7 ms, 17% of compiling a block), and it holds no state across a
        # `build` call that a later one does not overwrite.
        builder = builder_for(arch)
    ops = _translate(arch, code, base)
    marks = instruction_starts(ops)
    if not marks:
        return []
    n = len(marks)
    end_of = [marks[i + 1][1] if i + 1 < n else base + len(code) for i in range(n)]

    def lower(i: int, j: int) -> tuple[IRProg | None, int | None]:
        """Instructions [i, j) as one program.

        Returns (program, None) on success and (None, cut) on a decline, where
        `cut` is how many of those instructions CAN be taken -- the ordinal the
        decline reported, relative to `i`.  None means the decline named no
        position and the caller has to search.
        """
        lo = marks[i][0]
        hi = marks[j][0] if j < n else len(ops)
        try:
            return builder.build(ops[lo:hi], end_of[j - 1], emit=emit,
                                 block=(j - i > 1), abs_ram=abs_ram,
                                 max_acc=max_acc), None
        except Unsupported as exc:
            return None, getattr(exc, 'cut_at', None)
        except Exception:            # a lifter surprise is a decline, not a crash
            return None, None

    return _greedy(n, marks, end_of, lower)


#: How many instructions to ask for at a region start once the block has been
#: cut at least once.  A block that lowers whole never gets here, and one that
#: does not has short regions after the first: measured over the 61 distinct
#: blocks of the three taint-density guests, 51 of 61 regions are 4
#: instructions or fewer.
_REACH_AFTER_CUT = 4


def _longest(i: int, n: int, reach: int, lower: Lower) -> tuple[int, IRProg | None]:
    """The longest run from `i` that lowers as one program, and that program.

    Asks for `reach` instructions and doubles while they fit, instead of asking
    for the whole remainder every time.  The first probe from a start is the
    expensive one -- it lowers everything it is given and is thrown away on a
    decline -- and probing the remainder was 74% of the planning cost of the
    worst block in bench_dense, which cuts into eight regions.

    Growing is what keeps a region MAXIMAL: a run that fits is accepted only
    once a longer one has been refused.  That is sound because a decline is
    caused by a PAIR of accesses inside the run -- an address that depends on a
    value loaded there, or a load after a store to the same place -- and
    dropping the last instruction cannot create one, so a run that fits implies
    every shorter run fits.
    """
    best_len, best_prog = 0, None
    j = min(n, i + max(1, reach))
    capped = False
    while True:
        prog, cut = lower(i, j)
        if prog is not None:
            best_len, best_prog = j - i, prog
            # A length the decline itself named is a boundary: the instruction
            # after it is the one that broke the run, so there is nothing
            # longer to find.
            if j >= n or capped:
                return best_len, best_prog
            j = min(n, i + 2 * best_len)
            continue
        if cut is not None and best_len < cut < j - i:
            # The decline named a position: everything before it is a
            # candidate, and if THAT declines it names another, so this walks
            # strictly downwards and terminates.
            j, capped = i + cut, True
            continue
        if best_len:
            return best_len, best_prog
        # Nothing has fitted and the decline named no usable position (an
        # unliftable instruction, say): search downwards from the longest.
        for k in range(n, i, -1):
            prog, _c = lower(i, k)
            if prog is not None:
                return k - i, prog
        return 0, None


def _greedy(n: int, marks: list[tuple[int, int, int]], end_of: list[int],
            lower: Lower) -> list[Region]:
    """Longest-first from each cut, so every region is maximal."""
    regions: list[Region] = []
    i = 0
    reach = n                    # the first probe is the whole block
    while i < n:
        taken, prog = _longest(i, n, reach, lower)
        reach = _REACH_AFTER_CUT
        if not taken:
            # This one instruction lowers nowhere.  Emit it as a region with no
            # program so the caller still sees complete coverage of the block
            # and can send it down the per-instruction path.
            regions.append(Region(i, 1, marks[i][1], end_of[i], None))
            i += 1
            continue
        regions.append(Region(i, taken, marks[i][1], end_of[i + taken - 1], prog))
        i += taken
    return regions
