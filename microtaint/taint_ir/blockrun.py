"""Run a basic block's taint, and commit it only once the block has completed.

A block hook fires BEFORE the block executes, so the taint it computes is
SPECULATIVE: correct if the block runs to the end, wrong if the block is
abandoned part way.  Measured, that is not hypothetical -- a block hook announces
a 65-byte block whose load faults after six instructions -- and it is not a
harmless over-approximation either.  `mov rax, rbx` with a clean rbx CLEARS
RAX's taint, while RAX in fact still holds its old, tainted value because the
instruction never ran.  An under-taint.

So the taint of a block is held pending, and the proof that a block completed is
that the emulator has reached the NEXT one:

    block A fires  ->  (nothing to commit)      compute A, hold it pending
    block B fires  ->  commit A (it finished)   compute B, hold it pending
    A faults       ->  B never fires, A's pending taint is dropped

Which costs one taint-array copy per block and needs no fault handling at all.

The alternative -- snapshot at entry and restore from a fault hook -- keeps
reports synchronous with the instruction that caused them.  This one defers a
report by a block, which is the right trade for an analysis (a bug finder, a
fuzzer, a concolic engine all want throughput and the right address, not a
synchronous stop) and the wrong one for a mitigation that must stop before the
damage.  The engine is the former.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ['BlockRunner', 'Pending']


@dataclass
class Pending:
    """A block's computed taint, not yet committed.

    `taint` is the whole post-state, `writes` the shadow writes the block's
    stores would make.  Neither is applied until the block is known to have
    completed.
    """

    address: int
    taint: dict[str, int]
    writes: list[tuple[int, int, int]] = field(default_factory=list)
    reports: list[tuple[str, int, int]] = field(default_factory=list)


class BlockRunner:
    """Holds one block's taint pending, and commits it when the next arrives.

    Deliberately ignorant of how a block's taint is computed and of where the
    state lives: `compute` is supplied by the caller and returns a Pending, and
    `apply` puts one into effect.  That keeps the completion rule -- the only
    subtle thing here -- testable without an emulator.
    """

    def __init__(self, compute: Any, apply: Any) -> None:
        self._compute = compute
        self._apply = apply
        self._pending: Pending | None = None
        self.committed = 0
        self.dropped = 0

    @property
    def pending_address(self) -> int | None:
        return self._pending.address if self._pending else None

    def on_block(self, address: int, size: int) -> Pending | None:
        """A block is about to run.  Commit the previous one, compute this one."""
        self.commit_pending()
        self._pending = self._compute(address, size)
        return self._pending

    def commit_pending(self) -> None:
        """Put the held block into effect.  Reaching a new block, or the end of
        the run, is what proves the held one completed."""
        if self._pending is None:
            return
        self._apply(self._pending)
        self._pending = None
        self.committed += 1

    def abandon(self) -> None:
        """The held block did not complete: drop it.

        Nothing of it was applied, so there is nothing to undo.  That is the
        whole reason for holding it.
        """
        if self._pending is not None:
            self._pending = None
            self.dropped += 1

    def finish(self, completed: bool) -> None:
        """End of the run.  `completed` says whether the last block finished.

        The caller knows: emulation that ended by running off the end, or by a
        clean exit syscall, completed its last block; emulation that ended in a
        fault did not.  Getting this wrong in the safe direction (calling it
        abandoned when it completed) loses the last block's taint, so callers
        that cannot tell should say True only when they are sure.
        """
        if completed:
            self.commit_pending()
        else:
            self.abandon()
