"""Return an emulator to a point it has already reached.

A fuzzer builds a new emulator for every input, and on a static-glibc guest
about 87% of what that costs does not depend on the input: Qiling's own setup,
the engine's, and emulating the C library's startup again.  Running to a
checkpoint once and returning to it per input removes all three, and keeps the
translation cache and the process-wide plan table warm as a side effect.

Restoring the guest's MEMORY is the easy part, and Qiling does it.  What this
module exists for is everything else, each item of which was a real divergence
between a restored run and a fresh one before it was handled:

  * regions the guest mapped after the checkpoint are never unmapped by
    Qiling's restore, so a guest that leaks a mapping grows one per iteration;
  * a region that is still mapped but whose PERMISSIONS changed keeps them,
    because Qiling re-maps only what is missing -- a guest that mprotects its
    own data read-only then faults on the next iteration's first store;
  * Unicorn keeps translations of code the guest rewrote, and goes on running
    them after the bytes they came from have been put back;
  * the engine's own plan and decode caches are keyed by address and are
    dropped by a guest WRITE onto code -- and a restore rewrites those bytes
    with no guest write at all;
  * the taint of the previous input has to go, or a restored run reports more
    than a fresh one.

The checkpoint must be taken with the engine already armed.  Hooks arm lazily,
on the first taint, so a checkpoint taken before that would have the first
iteration arm them and every later one start from somewhere else.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple, TypedDict

if TYPE_CHECKING:
    from collections.abc import Mapping


class CheckpointError(RuntimeError):
    """The emulator could not be put back the way it was.

    Raised rather than returned: a restore that half worked produces a run
    whose answer differs from a fresh one for a reason nobody will find.
    """


class MemSnapshot(TypedDict, total=False):
    """What Qiling's `mem.save()` hands back.

    Declared because this module looks INSIDE it -- the read-only-region
    optimisation walks `ram` and decides what has to be rewritten -- and a
    blanket `Any` over the whole checkpoint meant nothing checked that walk.
    `mmio` is Qiling's, opaque here, and its presence is the signal to fall
    back to Qiling's own restore.
    """

    #: (low, high, permissions, label, bytes) per mapped region.
    ram: list[tuple[int, int, int, str, bytes]]
    mmio: object


@dataclass(frozen=True)
class Checkpoint:
    """A point in a guest's execution, and everything needed to return to it."""

    #: Where the guest was.  `resume` starts here.
    address: int
    #: Qiling's own state: registers, memory, file descriptors, OS and loader.
    #: Heterogeneous by construction -- it is Qiling's dict and not ours -- so
    #: the values are `object` and the one key this module looks inside is
    #: narrowed to `MemSnapshot` where it is used.
    state: Mapping[str, object]
    #: (low, high, permissions) for every region that was mapped.  Permissions
    #: are part of the identity: a region still mapped with different ones has
    #: to be unmapped so Qiling's restore re-maps it with the right ones.
    shape: frozenset[tuple[int, int, int]] = field(repr=False)
    #: The engine's self-modifying-code counter as it stood here.  A restore
    #: only has to throw away translations and plans if the guest has rewritten
    #: code since; on everything else that would cost a full re-translation an
    #: iteration for nothing.
    smc_seen: int = 0


class RunOutcome(NamedTuple):
    """How an iteration ended.  Three outcomes, and they are not the same.

    A fuzzer's inputs crash the target, so `faulted` is a result and not an
    error.  `timed_out` is the one that must never be mistaken for anything
    else: `uc.emu_start` does NOT raise when its wall-clock budget expires -- it
    returns normally, having stopped wherever it got to, with every register
    still holding whatever it held.  A harness that read that back as a
    finished run once produced 41 fabricated under-taints against a sound
    engine, none of them real.  So a run is `completed` only when the guest
    reached its own exit, proved by the exit syscall, and never by the absence
    of an exception.
    """

    completed: bool
    faulted: bool
    timed_out: bool
    #: (address, taint mask) for every leak site THIS iteration reached.
    #:
    #: Not the same as what the reporter emitted.  Deduplication describes a
    #: site once for the life of the hook, so an input that re-reaches a known
    #: site produces no finding; this still says it got there.  The runtime
    #: records it as a bit per site and resolves the bitmap once per run, so it
    #: costs no Python call per finding.
    sites: tuple[tuple[int, int], ...] = ()
    #: (site, address, address taint, size, is_store) for every memory access
    #: THIS iteration made through an input-dependent address.
    #:
    #: Not the same as what the reporter emitted: a sink is described once for
    #: the life of the hook, so a known one is silent on every later input,
    #: while this says what was TOUCHED.  A probe that re-runs with a subset of
    #: the input tainted, to learn which bits control an address, needs the
    #: second.
    sinks: tuple[tuple[int, int, int, int, int], ...] = ()
    #: What THIS iteration did.  The hook's own counters run for the life of
    #: the emulator, so in a checkpoint loop they answer a question nobody
    #: asked: a fuzzer wants to know what the input it just ran did, not what
    #: every input since the checkpoint did between them.
    blocks: int = 0
    handled: int = 0
    unhandled: int = 0

    def __bool__(self) -> bool:
        """True when the guest ran to its own exit and nothing else happened."""
        return self.completed and not self.faulted and not self.timed_out


def map_shape(ql: object) -> frozenset[tuple[int, int, int]]:
    """(low, high, permissions) for every mapped region."""
    return frozenset(
        (lo, hi, perms) for lo, hi, perms, *_ in ql.mem.map_info)  # type: ignore[attr-defined]
