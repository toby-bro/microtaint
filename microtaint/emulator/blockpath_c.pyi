"""Type stubs for the block runtime's C surface.

The runtime itself is `blockpath.h` and has no Python in it at all.  Everything
declared here is either the COMPILER handing over a finished plan (once per
distinct block) or a test entry point; none of it runs per block execution.
"""
from typing import Any

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
    regions: list[tuple[int, int, list[tuple[int, int, int]]]],
    keepalive: Any,
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
def runner_stats(runner: _Capsule) -> dict[str, Any]: ...
def runner_reports(runner: _Capsule) -> list[tuple[int, int]]: ...
