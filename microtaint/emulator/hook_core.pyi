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
from collections.abc import Callable
from typing import TypeAlias

from microtaint.emulator.reporter import Reporter
from microtaint.emulator.shadow import BitPreciseShadowMemory

# The wrapper imports this module, so naming its type here would be a cycle at
# runtime; in a stub the import is only ever read by the checker.
from microtaint.emulator.wrapper import MicrotaintWrapper
from microtaint.instrumentation.ast import LogicCircuit
from microtaint.instrumentation.cell import DecodedOps
from microtaint.simulator import CellSimulator
from microtaint.types import Architecture, ImplicitTaintPolicy

#: The ctypes-wrapped `uc_mem_read(uc, address, buf, size) -> uc_err`.
UcMemRead: TypeAlias = Callable[[ctypes.c_void_p, int, ctypes.Array[ctypes.c_ubyte], int], int]
#: `uc_reg_read_batch(uc, int *regs, void **vals, count) -> uc_err`.
UcRegReadBatch: TypeAlias = Callable[[ctypes.c_void_p, object, object, int], int]
#: The engine's state_format: (name, bits) per register, in slot order.
FormatKey: TypeAlias = tuple[tuple[str, int], ...]

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
    wrapper: MicrotaintWrapper
    """The owning MicrotaintWrapper. Used only for slow-path callbacks."""

    register_taint: dict[str, int]
    """The wrapper's live register-taint dict; mutated in-place on every callback."""

    last_tainted_writes: set[int]
    """The wrapper's set of guest addresses written this instruction."""

    instr_cache: dict[int, tuple[frozenset[tuple[str, int]], dict[str, int]]]
    """
    Tier 3 cache: address -> (frozenset(taint.items()), output_state).
    Used as the cold-path lookup when the version-cache misses.
    """

    instr_cache_v: dict[int, tuple[int, int, dict[str, int]]]
    """
    Tier 4b version cache: address -> (input_version, output_version, output_state).
    Hot-path lookup uses a 64-bit version compare instead of a frozenset construction.
    """

    decode_cache: dict[int, tuple[int, bytes, LogicCircuit]]
    """
    Address-keyed decode cache: address -> (size, instruction_bytes, circuit).
    On a repeat visit the hook reuses this instead of uc_mem_read + the ctypes
    buffer slice + the cached_gen_rule tuplehash (~3.0 us of the cache-hit floor).
    """

    shadow_mem: BitPreciseShadowMemory

    sim: CellSimulator
    """The evaluator used for differential evaluation."""

    policy: ImplicitTaintPolicy
    """Governs the SC / BOF detection."""

    reporter: Reporter

    ql: object
    """The Qiling instance, used for emu_stop on a detected violation.
    Qiling ships no type information, and nothing here reads it: it is
    only ever handed back."""

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

    py_decode_cache: dict[int, tuple[int, bytes, LogicCircuit]]
    """address -> (size, bytes, circuit); the dict path only."""

    arr_cache: dict[int, tuple[int, int, int, list[int]]]
    """address -> (in_snap, out_snap, val_snap, slots)."""

    slots_cache: dict[bytes, list[int]]
    """instruction bytes -> the input slots it reads."""

    taint_ir_progs: dict[bytes, object]
    """instruction bytes -> compiled taint program.  Holding it here is what
    keeps its emitted code alive."""

    def reset_taint(self) -> None:
        """Forget every register's taint: the slot array AND `register_taint`.

        For a harness that restores a checkpoint and drives another input.  In
        block mode the array is authoritative, so clearing the dict alone would
        leave the previous run's taint in place with nothing to say so.
        """

    def sync_taint_to_dict(self) -> None:
        """Flush the slot array into `register_taint` and hand authority back
        to the dict, so an external re-seed is picked up by the next
        instruction."""

    #: Called by `invalidate_smc` when block mode is on, to drop its plan
    #: cache too; set by the wrapper when it installs the block hook.  Takes
    #: the guest write's (address, size), which is what lets block mode tell a
    #: rewrite of the block it is holding from a rewrite of anything else.
    block_invalidate: Callable[[int, int], None] | None

    def code_range_addrs(self) -> tuple[int, int]:
        """Addresses of `code_lo` and `code_hi`, for the block runtime."""

    def invalidate_smc(self, address: int = ..., size: int = ...) -> None:
        """Drop the decode + Tier-3/Tier-4 caches after a write hit cached code.

        `address`/`size` describe the write, and are forwarded to
        `block_invalidate`; they default to nothing said, which keeps the
        blunter behaviour for a caller that has nothing to say."""

    def __init__(
        self,
        wrapper: MicrotaintWrapper,
        *,
        uc_handle: ctypes.c_void_p,
        uc_mem_read: UcMemRead,
        uc_reg_read_batch: UcRegReadBatch,
        mem_buf: ctypes.Array[ctypes.c_ubyte],
        arch: Architecture,
        cached_gen_rule: Callable[[Architecture, bytes, FormatKey], LogicCircuit],
        x64_format_key: FormatKey,
        get_decoded: Callable[[Architecture, bytes], DecodedOps],
        build_offsets_arrs: Callable[[frozenset[int]], object],
        eflags_bits: dict[str, int],
        eval_context_cls: type,
        uc_reg_read_batch_addr: int = ...,
        flag_parent: str = ...,
        pc_reg_name: str = ...,
    ) -> None: ...
    def prepare_block_mode(self, names: list[str]) -> int:
        """Resolve what the per-instruction path would resolve lazily.

        Block mode never runs that path, so the engine handle, the C memory
        read context and a slot per register are all still unset when the first
        block arrives.  Setup, not the hot path.
        """

    slot_map: dict[str, int]
    """Register name -> slot index in the taint and value arrays."""
    def __call__(
        self,
        uc: object,
        address: int,
        size: int,
        user_data: object,
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

    wrapper: MicrotaintWrapper
    shadow_mem: BitPreciseShadowMemory
    last_tainted_writes: set[int]
    reporter: Reporter
    ql: object
    check_uaf: bool
    instr_hook: InstructionHook | None

    def __init__(self, wrapper: MicrotaintWrapper) -> None: ...
    def __call__(
        self,
        uc: object,
        access: int,
        address: int,
        size: int,
        value: int,
        user_data: object = ...,
    ) -> None: ...

class MemAccessHook:
    """Unicorn UC_HOOK_MEM_READ callback. Reports UAF on read-after-free."""

    wrapper: MicrotaintWrapper
    shadow_mem: BitPreciseShadowMemory
    reporter: Reporter
    ql: object
    check_uaf: bool

    def __init__(self, wrapper: MicrotaintWrapper) -> None: ...
    def __call__(
        self,
        uc: object,
        access: int,
        address: int,
        size: int,
        value: int,
        user_data: object = ...,
    ) -> None: ...

class UafUnmappedWriteHook:
    """
    Unicorn UC_HOOK_MEM_WRITE_UNMAPPED callback.

    Catches the mmap → munmap → write UAF pattern where the target
    page is fully unmapped.  Returns False to terminate the run before
    Unicorn's own crash handler fires.
    """

    wrapper: MicrotaintWrapper
    shadow_mem: BitPreciseShadowMemory
    reporter: Reporter
    ql: object

    def __init__(self, wrapper: MicrotaintWrapper) -> None: ...
    def __call__(
        self,
        uc: object,
        access: int,
        address: int,
        size: int,
        value: int,
        user_data: object = ...,
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

    wrapper: MicrotaintWrapper
    uc_mem_read: UcMemRead
    mem_buf: ctypes.Array[ctypes.c_ubyte]
    mem_ptrs: dict[int, object]
    ql: object

    def __init__(
        self,
        wrapper: MicrotaintWrapper,
        *,
        uc_mem_read: UcMemRead,
        mem_buf: ctypes.Array[ctypes.c_ubyte],
        mem_ptrs: dict[int, object],
        uc_mem_read_addr: int = ...,
        mem_buf_addr: int = ...,
    ) -> None: ...

    # The read boundary (the Unicorn handle, and the addresses of uc_mem_read
    # and of the ctypes buffer) is plain `cdef`, so it is C-level only and not
    # reachable from Python.  It was declared here as three public ints, which
    # made mypy accept attribute reads that raise at runtime.

    def __call__(self, address: int, size: int) -> int:
        """Read `size` bytes at `address`. Returns 0 on error."""

# ---------------------------------------------------------------------------
# The pure-C trampolines.  Each pair is (address of the trampoline, its
# user_data), both handed straight to uc_hook_add.  The user_data is the hook's
# C CONTEXT, not the hook object: the callback runs without the GIL, so it
# cannot cast a PyObject to reach the context.  The context carries a borrowed
# pointer back to the hook for the path that does need Python, which is why the
# caller MUST keep the hook alive for as long as it is registered.
# ---------------------------------------------------------------------------

def c_instruction_hook_ptr() -> int:
    """Address of the pure-C UC_HOOK_CODE trampoline."""

def c_instruction_hook_ud(hook: InstructionHook) -> int:
    """user_data for the code trampoline.  Valid while `hook` is alive."""

def c_mem_write_hook_ptr() -> int:
    """Address of the pure-C UC_HOOK_MEM_WRITE trampoline."""

def c_mem_write_hook_ud(hook: MemWriteClearHook) -> int:
    """user_data for the write trampoline.  Valid while `hook` is alive."""

def c_mem_access_hook_ptr() -> int:
    """Address of the pure-C UC_HOOK_MEM_READ trampoline."""

def c_mem_access_hook_ud(hook: MemAccessHook) -> int:
    """user_data for the read trampoline.  Valid while `hook` is alive."""
