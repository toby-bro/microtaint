# ruff: noqa: PYI021
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

    # ----- counters -----
    instr_cache_hits: int
    instr_cache_misses: int

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
    ) -> None: ...

    def __call__(
        self,
        uc: Any,
        address: int,
        size: int,
        user_data: Any,
    ) -> None:
        """Unicorn-invoked instruction callback. Hot-path entry point."""
