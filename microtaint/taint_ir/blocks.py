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
               builder: Builder | None = None) -> list[Region]:
    """Greedy maximal regions covering `code`, in order.

    `abs_ram` says `base` is where the code really runs, so a `ram` varnode --
    what a PC-relative operand lifts to once the displacement has folded into
    the program counter -- names a live guest address and can be lowered as a
    memory access.  It is off by default because the default base is synthetic.

    The whole block is tried first, because that is the common case and it costs
    one lowering; only when it declines does this walk instruction by
    instruction.  A single instruction that will not lower at all still gets a
    Region with `prog=None`, so the caller sees complete coverage of the block
    and can send that one instruction down the per-instruction path rather than
    having to work out what is missing.
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
                                 block=(j - i > 1), abs_ram=abs_ram), None
        except Unsupported as exc:
            return None, getattr(exc, 'cut_at', None)
        except Exception:            # a lifter surprise is a decline, not a crash
            return None, None

    whole, _cut = lower(0, n)
    if whole is not None:
        return [Region(0, n, marks[0][1], end_of[n - 1], whole)]
    return _greedy(n, marks, end_of, lower)


def _greedy(n: int, marks: list[tuple[int, int, int]], end_of: list[int],
            lower: Lower) -> list[Region]:
    """Longest-first from each cut, so every region is maximal.

    A decline usually says WHICH instruction caused it, and then the boundary is
    known without searching for it: take the instructions before that one and
    start again there.  Searching costs a full lowering per candidate length, and
    a lowering is a SLEIGH translation plus an IR build -- measured at up to
    201 ms for one block.  The search remains as the fallback for a decline that
    names no position (an unliftable instruction, say).
    """
    regions: list[Region] = []
    i = 0
    while i < n:
        taken = 0
        prog = None
        prog, cut = lower(i, n)
        if prog is not None:
            taken = n - i
        else:
            while cut is not None and cut > 0:
                # The decline named a position: everything before it is a
                # candidate, and if THAT declines it names another, so this
                # walks strictly downwards and terminates.
                prog, cut2 = lower(i, i + cut)
                if prog is not None:
                    taken = cut
                    break
                cut = cut2 if cut2 is not None and cut2 < cut else None
            if not taken and prog is None:
                for j in range(n, i, -1):       # no position named: search
                    prog, _c = lower(i, j)
                    if prog is not None:
                        taken = j - i
                        break
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
