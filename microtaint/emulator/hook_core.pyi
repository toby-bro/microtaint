"""Type stubs for the hook_core module — Cython-compiled per-instruction
Unicorn callback.

`InstructionHook` is the V5 Cython hot-path hook that replaces the pure
Python `_instruction_evaluator_raw` method on MicrotaintWrapper.  It
holds direct typed references to every wrapper field touched on the
hot path so each callback is a single Python frame followed by typed C
operations.

The hook is registered with Unicorn either via Qiling's wrapper or
directly through ctypes-wrapped uc_hook_add (Tier 4 bypass).  In both
cases Unicorn invokes the instance as `hook(uc, address, size, user_data)`.
"""

import ctypes
from typing import Any, Callable

from microtaint.types import Architecture

class InstructionHook:
    """
    Cython-compiled InstructionHook callable.

    Construct with a MicrotaintWrapper plus a bag of keyword-only
    helpers.  The constructor caches typed references to the wrapper's
    fields (`register_taint`, `_instr_cache`, `_last_tainted_writes`,
    `shadow_mem`, etc.) so the hot path never does Python attribute
    lookups against the wrapper.

    Once registered with Unicorn, every executed instruction in the
    hooked address range invokes `__call__`, which thunks into the
    C-level `_evaluate` routine.
    """

    # ----- wrapper-derived state, captured once at construction -----
    wrapper: Any
    """The owning MicrotaintWrapper. Used only for slow-path callbacks."""

    register_taint: dict[str, int]
    """The wrapper's live register-taint dict; mutated in-place on every callback."""

    last_tainted_writes: set[int]
    """The wrapper's set of guest addresses written this instruction."""

    instr_cache: dict[int, tuple[Any, dict[str, int]]]
    """
    Tier 3 cache: address -> (frozenset(taint.items()), output_state).
    Used as the cold-path lookup when the version-cache misses.
    """

    instr_cache_v: dict[int, tuple[int, int, dict[str, int]]]
    """
    Tier 4b version cache: address -> (input_version, output_version, output_state).
    Hot-path lookup uses a 64-bit version compare instead of a frozenset construction.
    """

    decode_cache: dict[int, tuple[int, bytes, Any]]
    """
    Address-keyed decode cache: address -> (size, instruction_bytes, circuit).
    On a repeat visit the hook reuses this instead of uc_mem_read + the ctypes
    buffer slice + the cached_gen_rule tuplehash (~3.0 us of the cache-hit floor).
    """

    shadow_mem: Any
    """BitPreciseShadowMemory instance (a cdef class from microtaint.emulator.shadow)."""

    sim: Any
    """The CellSimulator used for differential evaluation."""

    policy: Any
    """ImplicitTaintPolicy instance for SC / BOF detection."""

    reporter: Any
    """Findings reporter."""

    ql: Any
    """Qiling instance, used for emu_stop on detected violations."""

    check_bof: bool
    check_sc: bool
    check_aiw: bool
    """Run-time toggles for the four detection categories (BOF, side-channel, AIW)."""

    instr_cache_enabled: bool
    """Tier 3/4b master switch (set from MICROTAINT_DISABLE_INSTR_CACHE env)."""

    taint_version: int
    """
    Monotonic counter incremented each time `register_taint` mutates.
    Cache entries are keyed on this version; equal version <=> identical state.
    """

    code_lo: int
    code_hi: int
    """
    Bounds [code_lo, code_hi) of every address in decode_cache.  The mem-write
    hook invalidates the caches when a write intersects this range (self-
    modifying / JIT'd code).  code_lo > code_hi means "no cached code yet".
    """

    # ----- counters -----
    instr_cache_hits: int
    instr_cache_misses: int

    express_done: int
    """Instructions answered by the express lane.  The C fast path holds this
    counter's ADDRESS (`fctx.express_done`), so it is `cdef public` rather than
    a Python attribute."""

    fast_done: int
    instr_total: int
    """Instructions the C fast path finished without entering Python, and the
    total it was offered.  Coverage is the RATIO of the two, which is why both
    are counted and not only the numerator."""

    prefilter_hits: int
    """Instructions dismissed outright because no input was tainted."""

    fb_pc: int
    fb_pyfall: int
    fb_mem: int
    fb_other: int
    fb_nocircuit: int
    """Why instructions leave the C path, so the biggest reason is a
    measurement rather than a guess."""

    arr_fallbacks: int
    """Instructions the array path had to hand back to the dict path."""

    EXPRESS_MISS_REASONS: tuple[str, ...]
    """Reason names for `express_miss`, in index order (see fastpath.h)."""

    @property
    def express_miss(self) -> tuple[int, ...]:
        """Per-reason express-lane miss counts, paired with
        EXPRESS_MISS_REASONS."""

    @property
    def regs_read(self) -> int:
        """Register reads the fast path performed."""

    @property
    def regs_circuit(self) -> int:
        """Register reads the circuits asked for.  `regs_circuit - regs_read`
        is what the compiled path saved."""

    py_decode_cache: dict[int, tuple[int, bytes, Any]]
    """address -> (size, bytes, circuit); the dict path only."""

    arr_cache: dict[int, Any]
    """address -> (in_snap, out_snap, val_snap, slots)."""

    slots_cache: dict[bytes, list[int]]
    """instruction bytes -> the input slots it reads."""

    taint_ir_progs: dict[bytes, Any]
    """instruction bytes -> compiled taint program.  Holding it here is what
    keeps its emitted code alive."""

    def sync_taint_to_dict(self) -> Any:
        """Flush the slot array into `register_taint` and hand authority back
        to the dict, so an external re-seed is picked up by the next
        instruction."""

    def invalidate_smc(self) -> None:
        """Drop the decode + Tier-3/Tier-4 caches after a write hit cached code."""

    def __init__(
        self,
        wrapper: Any,
        *,
        uc_handle: ctypes.c_void_p,
        uc_mem_read: Callable[..., int],
        uc_reg_read_batch: Callable[..., int],
        mem_buf: Any,
        arch: Architecture,
        cached_gen_rule: Callable[..., Any],
        x64_format_key: Any,
        get_decoded: Callable[..., Any],
        build_offsets_arrs: Callable[..., Any],
        eflags_bits: dict[str, int],
        eval_context_cls: type,
        uc_reg_read_batch_addr: int = ...,
        flag_parent: str = ...,
        pc_reg_name: str = ...,
    ) -> None: ...
    def prepare_block_mode(self, names: Any) -> int:
        """Resolve what the per-instruction path would resolve lazily.

        Block mode never runs that path, so the engine handle, the C memory
        read context and a slot per register are all still unset when the first
        block arrives.  Setup, not the hot path.
        """

    slot_map: dict[str, int]
    """Register name -> slot index in the taint and value arrays."""
    def __call__(
        self,
        uc: Any,
        address: int,
        size: int,
        user_data: Any,
    ) -> None:
        """Unicorn-invoked instruction callback. Hot-path entry point."""

# ---------------------------------------------------------------------------
# Memory hooks (V8) — Cython port of the three Unicorn mem callbacks.
#
# Same design pattern as InstructionHook:
#   - capture wrapper state once at construction
#   - cimport BitPreciseShadowMemory so shadow_mem method calls dispatch at C level
#   - register via direct ctypes-wrapped uc_hook_add (bypassing Qiling's
#     hook_mem_{read,write} and Unicorn's __hook_mem_access_cb wrapper)
# ---------------------------------------------------------------------------

class MemWriteClearHook:
    """
    Unicorn UC_HOOK_MEM_WRITE callback.

    On every guest memory write:
      1. Self-modifying code — if the write intersects the instruction hook's
         cached-code range, invalidate its decode + output caches.
      2. UAF detection — if the target address is poisoned (was munmap'd),
         report and stop.
      3. Taint clearing — addresses outside the wrapper's
         `_last_tainted_writes` set get their shadow taint cleared
         (the program wrote a fresh, untainted value).
    """

    wrapper: Any
    shadow_mem: Any  # BitPreciseShadowMemory (cdef class)
    last_tainted_writes: set[int]
    reporter: Any
    ql: Any
    check_uaf: bool
    instr_hook: Any  # InstructionHook | None — caches to invalidate on SMC

    def __init__(self, wrapper: Any) -> None: ...
    def __call__(
        self,
        uc: Any,
        access: int,
        address: int,
        size: int,
        value: int,
        user_data: Any = ...,
    ) -> None: ...

class MemAccessHook:
    """Unicorn UC_HOOK_MEM_READ callback. Reports UAF on read-after-free."""

    wrapper: Any
    shadow_mem: Any
    reporter: Any
    ql: Any
    check_uaf: bool

    def __init__(self, wrapper: Any) -> None: ...
    def __call__(
        self,
        uc: Any,
        access: int,
        address: int,
        size: int,
        value: int,
        user_data: Any = ...,
    ) -> None: ...

class UafUnmappedWriteHook:
    """
    Unicorn UC_HOOK_MEM_WRITE_UNMAPPED callback.

    Catches the mmap → munmap → write UAF pattern where the target
    page is fully unmapped.  Returns False to terminate the run before
    Unicorn's own crash handler fires.
    """

    wrapper: Any
    shadow_mem: Any
    reporter: Any
    ql: Any

    def __init__(self, wrapper: Any) -> None: ...
    def __call__(
        self,
        uc: Any,
        access: int,
        address: int,
        size: int,
        value: int,
        user_data: Any = ...,
    ) -> bool: ...

class LiveMemReader:
    """
    Cython callable that reads `size` bytes from Unicorn at a given address
    via ctypes-wrapped `uc_mem_read` into a pre-allocated buffer.

    Replaces the wrapper's pure-Python `_read_live_memory` method, which was
    the last Python frame on the per-instruction hot path (~256k calls/run
    in the bench).  Created once in `MicrotaintWrapper._setup_hooks()` and
    passed to circuit_c via `mem_reader=` so OP_PUSH_MEM_VALUE invokes it
    with no Python frame setup.
    """

    wrapper: Any
    uc_mem_read: Callable[..., int]
    mem_buf: Any
    mem_ptrs: dict[int, Any]
    ql: Any

    def __init__(
        self,
        wrapper: Any,
        *,
        uc_mem_read: Callable[..., int],
        mem_buf: Any,
        mem_ptrs: dict[int, Any],
        uc_mem_read_addr: int = ...,
        mem_buf_addr: int = ...,
    ) -> None: ...

    # The read boundary (the Unicorn handle, and the addresses of uc_mem_read
    # and of the ctypes buffer) is plain `cdef`, so it is C-level only and not
    # reachable from Python.  It was declared here as three public ints, which
    # made mypy accept attribute reads that raise at runtime.

    def __call__(self, address: int, size: int) -> int:
        """Read `size` bytes at `address`. Returns 0 on error."""

#: Address of the pure-C UC_HOOK_CODE trampoline, for uc_hook_add.  Register it
#: with the InstructionHook instance as user_data (id(hook)); the caller must
#: keep that instance alive for the hook's lifetime.
def c_instruction_hook_ptr() -> int: ...
