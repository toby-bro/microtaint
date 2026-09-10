"""Type stubs for the block runtime's C surface.

The runtime itself is `blockpath.h` and has no Python in it at all.  Everything
declared here is either the COMPILER handing over a finished plan (once per
distinct block) or a test entry point; none of it runs per block execution.
"""
from collections.abc import Callable

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
    regions: list[tuple[int, int, list[tuple[int, int, int]], int]],
    keepalive: object,
    ids_addr: int = ...,
    ptrs_addr: int = ...,
    vals_addr: int = ...,
    n_calls: int = ...,
    val_slots: list[int] | None = ...,
    need_flags: bool = ...,
) -> _Capsule:
    """A block's plan: per region, (function address, guest address, accesses).

    `keepalive` is held for the plan's lifetime; the emitted code the function
    addresses point into belongs to it, and the C runtime does not refcount.

    The trailing arguments are the block's OWN uc_reg_read_batch descriptor:
    which registers it reads and where they land.  Reading the whole register
    file instead was 90% of block mode's cost when it was first wired.
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
def runner_reports(runner: _Capsule) -> list[tuple[int, int]]: ...

# ---------------------------------------------------------------------------
# The live UC_HOOK_BLOCK path.  `hook_ptr` is the C trampoline's address and
# `hook_ud` its user_data, both handed straight to uc_hook_add; the caller MUST
# keep the hook alive for as long as it is registered.
# ---------------------------------------------------------------------------

def hook_new(fastctx: int, compiler: Callable[[int, int], object | None],
             ids: int, ptrs: int, vals: int, n_calls: int,
             reg_slots: list[int]) -> _Capsule:
    """The block hook's C context.

    `fastctx` is the InstructionHook's C context (from
    `hook_core.c_instruction_hook_ud`), and `ids` / `ptrs` / `vals` are the
    ADDRESSES of the register-read descriptor's ctypes arrays.  A vector
    register is one read call that fills two lanes, so the slot list is per
    name and `n_calls` is counted separately.
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

def hook_stats(hook: _Capsule) -> dict[str, int]:
    """Blocks seen, handled, and NOT handled.  `unhandled` is unanalysed code,
    so it is counted rather than ignored."""
