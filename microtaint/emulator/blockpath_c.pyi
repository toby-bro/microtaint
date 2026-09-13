"""Type stubs for the block runtime's C surface.

The runtime itself is `blockpath.h` and has no Python in it at all.  Everything
declared here is either the COMPILER handing over a finished plan (once per
distinct block) or a test entry point; none of it runs per block execution.
"""
from collections.abc import Callable, Sequence

class _Capsule: ...

def layout() -> dict[str, int]:
    """The slot layout blockpath.h runs, so Python cannot restate it wrongly."""

def mem_new(base: int, length: int) -> _Capsule: ...
def mem_poke(mem: _Capsule, addr: int, data: bytes) -> None: ...
def mem_taint(mem: _Capsule, addr: int, mask: int, size: int) -> None: ...
def mem_peek(mem: _Capsule, addr: int, size: int) -> int: ...
def mem_mask(mem: _Capsule, addr: int, size: int) -> int: ...

def plan_new(
    size: int,
    regions: Sequence[tuple[int, int, Sequence[tuple[int, int, int]],
                            int, int, int, int, Sequence[int]]],
    keepalive: object,
    uc_ids: list[int] | None = ...,
    val_slots: list[int] | None = ...,
    need_flags: bool = ...,
) -> _Capsule:
    """A block's plan: per region, (function address, guest address, accesses).

    `keepalive` is held for the plan's lifetime; the emitted code the function
    addresses point into belongs to it, and the C runtime does not refcount.

    A region's last element is the register slots whose VALUE it publishes.
    The runtime seeds and threads only those instead of copying the whole
    register file twice per region; omit it and it copies wholesale, which is
    what it always did.

    The trailing arguments are the block's OWN uc_reg_read_batch descriptor:
    the Unicorn register ids it reads and the engine slots their words land in.
    Reading the whole register file instead was 90% of block mode's cost when
    it was first wired.

    WHICH registers, not WHERE: the destination buffer belongs to the hook that
    does the reading, so one plan can be run by two live wrappers.  Holding the
    buffer addresses here is what used to make a plan the property of exactly
    one caller.
    """

def runner_new(n_slots: int, pc_slot: int, mem: _Capsule,
               guest_writes: bool = ...) -> _Capsule: ...
def runner_seed(runner: _Capsule, taint: list[int]) -> None: ...
def runner_taint(runner: _Capsule) -> list[int]: ...
def runner_values(runner: _Capsule) -> list[int]: ...
def runner_on_block(runner: _Capsule, plan: _Capsule, address: int,
                    reg_val: list[int]) -> int: ...
def runner_finish(runner: _Capsule, completed: bool) -> None: ...
def runner_abandon(runner: _Capsule) -> None: ...
def runner_stats(runner: _Capsule) -> dict[str, int]: ...
def hook_invalidate(hook: _Capsule, addr: int = ..., size: int = ...) -> None:
    """Drop every cached plan after a write hit code the hook has planned.

    A plan is keyed by (address, size) alone, so rewritten bytes at the same
    address would otherwise run the plan compiled for what used to be there.

    `addr`/`size` are the guest write itself.  Given them, the block being HELD
    by the deferred commit is abandoned only when the write lands on that
    block's own instructions; without them every invalidation abandons it,
    which is the old behaviour and an under-taint whenever the rewrite was
    somewhere else.  The range the caller guards is one interval, so "somewhere
    else" is the common case as soon as a guest JITs anything.
    """

def hook_code_range(hook: _Capsule) -> tuple[int, int]:
    """(lo, hi) of the bytes this hook has planned, for the mem-write guard."""

def runner_reports(runner: _Capsule) -> list[tuple[int, int]]: ...

# ---------------------------------------------------------------------------
# The live UC_HOOK_BLOCK path.  `hook_ptr` is the C trampoline's address and
# `hook_ud` its user_data, both handed straight to uc_hook_add; the caller MUST
# keep the hook alive for as long as it is registered.
# ---------------------------------------------------------------------------

def hook_new(fastctx: int, compiler: Callable[[int, int], object | None],
             ids: int, ptrs: int, vals: int, n_calls: int,
             reg_slots: list[int], code_lo_addr: int = ...,
             code_hi_addr: int = ..., world: int = ...) -> _Capsule:
    """The block hook's C context.

    `fastctx` is the InstructionHook's C context (from
    `hook_core.c_instruction_hook_ud`), and `ids` / `ptrs` / `vals` are the
    ADDRESSES of the register-read descriptor's ctypes arrays.  A vector
    register is one read call that fills two lanes, so the slot list is per
    name and `n_calls` is counted separately.

    `code_lo_addr` / `code_hi_addr` are the addresses of the instruction hook's
    own `code_lo` and `code_hi`.  Block mode plans blocks that hook never
    decodes, so it widens the same range: it is what the mem-write hook's
    self-modifying-code guard tests, and without it a rewritten block would
    keep running the plan compiled for the bytes that used to be there.

    `world` names the (architecture, slot map) universe this hook's plans
    belong to, from `blockcompile.world_token`.  Plans are kept for the life of
    the PROCESS and a later emulator runs them without entering Python, which
    it may do only within one world: the emitted code addresses engine slots by
    index.  Omit it and this hook shares nothing, which is what every caller
    got before the table existed.
    """

def hook_ptr() -> int:
    """Address of the pure-C UC_HOOK_BLOCK trampoline, for uc_hook_add."""

def hook_ud(hook: _Capsule) -> int:
    """user_data for the trampoline: the hook's C context, not a PyObject."""

def hook_finish(hook: _Capsule, completed: bool) -> None:
    """Commit or abandon the block held back by the deferred-commit protocol.

    A block hook fires for a block that may then fault partway through, so its
    taint is held until the NEXT block proves it completed.
    """

def hook_dedupe(hook: _Capsule, on: bool) -> None:
    """Describe each (address, mask) finding once, or every time it occurs.

    The runtime files a report every time the program counter is found tainted,
    so a secret-dependent branch inside a loop is filed once per iteration.
    With this on, only the first of each pair crosses into Python.

    Keyed on the PAIR and not the address alone, because two findings at one
    address that differ in what leaked are two findings.  `hook_stats` keeps
    counting every occurrence in `reports` and counts the collapsed ones in
    `reports_duplicate`, so nothing the run found is hidden.  When the table
    fills it stops suppressing rather than start dropping.
    """

def hook_reports_forget(hook: _Capsule) -> None:
    """Forget which findings have been described, so they are described again.

    For a caller that wants per-run rather than per-process reporting: a
    checkpoint loop that re-runs the same code otherwise reports a site on the
    first input and never again.
    """

def hook_reports(hook: _Capsule) -> list[tuple[int, int]]:
    """(address, taint mask) for each secret-dependent program counter found,
    draining them so a long run can be drained repeatedly.

    The address is the last instruction of the region that computed it, which
    for a block ending in a branch is the branch.  Reports are held with the
    block's taint and released by the same commit, because a block that faulted
    partway through never happened.
    """

def plans_clear() -> None:
    """Forget every plan the process has kept.

    RETIRES rather than frees them: a live hook's own cache borrows plans from
    this table and cannot be told.  Called by `blockcompile.cache_clear`.
    """

def plans_stats() -> dict[str, int]:
    """What the process-wide plan table holds: entries, retired, capacity.

    `capacity` is `MICROTAINT_BLOCK_PLAN_TABLE`, default 32768.  The table
    never evicts, because a live hook borrows plans from it, so that number is
    the memory bound: each entry holds a block's bytes, its plan, and the
    emitted code the plan runs.  Set it to 0 and nothing is kept, which is the
    behaviour block mode had before the table existed.
    """

def hook_stats(hook: _Capsule) -> dict[str, int]:
    """Blocks seen, handled, and NOT handled.  `unhandled` is unanalysed code,
    so it is counted rather than ignored.

    `planned` is how many blocks this hook asked the Python compiler about and
    `reused` how many it found already planned by the process, without the GIL.
    On a fuzzer, the second run of a binary should be all `reused` and no
    `planned`.  `no_code` counts blocks whose bytes could not be read, which
    can be neither shared nor kept: they are the reason `reused` might be zero
    while everything else looks right.

    `reports` is every finding the runtime made, counted even when the ring
    overflowed, so a drain that returns fewer can be told from a run that found
    fewer.

    `committed_writes` is how many memory writes the runtime has applied to
    the shadow: what one input actually moved, which a checkpoint loop wants
    per iteration.  Measured at 747 an iteration on a static-glibc guest.  It
    is not a cost attribution: suppressing those writes suppresses every taint
    that would have followed from them.

    `invalidations` is how often a write onto code dropped the cached plans;
    `abandoned` is how many of those ALSO threw away the block being held.  The
    two used to be the same number, and the difference between them is exactly
    the taint that used to be lost to a rewrite of some other code."""
