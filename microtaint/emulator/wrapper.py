from __future__ import annotations

import ctypes
import logging
import os
from collections.abc import Callable
from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING

import unicorn.unicorn_py3.unicorn as _uu
from qiling import Qiling
from qiling.const import QL_ARCH, QL_INTERCEPT
from unicorn import UC_HOOK_CODE, UC_HOOK_MEM_WRITE_UNMAPPED

from microtaint.emulator import archregs
from microtaint.emulator.hook_core import (
    InstructionHook,
    LiveMemReader,
    MemAccessHook,
    MemWriteClearHook,
    UafUnmappedWriteHook,
    c_instruction_hook_ptr,
    c_instruction_hook_ud,
    c_mem_access_hook_ptr,
    c_mem_access_hook_ud,
    c_mem_write_hook_ptr,
    c_mem_write_hook_ud,
)
from microtaint.emulator.reporter import Reporter
from microtaint.emulator.shadow import BitPreciseShadowMemory
from microtaint.instrumentation.ast import EvalContext
from microtaint.instrumentation.cell import _get_decoded
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import _cached_generate_static_rule
from microtaint.types import Architecture, ImplicitTaintError, ImplicitTaintPolicy

if TYPE_CHECKING:                    # stub-only: an opaque PyCapsule handle
    from microtaint.emulator.blockpath_c import _Capsule

#: The Unicorn UC_HOOK_CODE callback: (uc, address, size, user_data).
#: Both InstructionHook.__call__ and the plain bound method satisfy it.
InstrHook = Callable[[object, int, int, object], None]

#: A Unicorn memory callback: (uc, access, address, size, value, user_data).
#: The invalid-access variants answer False to stop the run.
MemHook = Callable[[object, int, int, int, int, object], None]
InvalidMemHook = Callable[[object, int, int, int, int, object], bool]


#: A Qiling syscall handler.  Each one takes ITS syscall's arguments, so
#: there is no single signature to write; this module only forwards the
#: value to `ql.os.set_syscall`, and Qiling calls it with the right arity.
SyscallHandler = object


logger = logging.getLogger(__name__)

#: x86-64's register format, kept under its historical name for the tests and
#: benchmarks that import it.  Every architecture's format now comes from
#: ``archregs.state_format``; this is what that returns for AMD64.
X64_FORMAT = archregs.state_format(Architecture.AMD64)
_X64_FORMAT_KEY: tuple[tuple[str, int], ...] = tuple((r.name, r.bits) for r in X64_FORMAT)

# Tier 3: empty frozenset for cache key when register_taint is empty.
_EMPTY_FROZENSET: frozenset[tuple[str, int]] = frozenset()

# ---------------------------------------------------------------------------
# ctypes shim for uc_reg_read
#
# Unicorn exposes uc_reg_read(uc_handle, reg_id, *value) as a C function.
# Calling it directly via ctypes bypasses the entire Python binding stack:
#   _select_reg_class -> genexpr -> __seq_tuple -> next() -> __get_reg_read_arg
# which accounts for ~18us per register in the Python binding.
# Direct ctypes cost: ~1us per register.
#
# uclib is the module-level CDLL already loaded by the unicorn package.
# We grab it once at import time so there's zero attribute lookup per call.
# ---------------------------------------------------------------------------
_uclib = _uu.uclib

_UC_HOOK_MEM_WRITE_UNMAPPED = UC_HOOK_MEM_WRITE_UNMAPPED
_uc_reg_read = _uclib.uc_reg_read
_uc_reg_read.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
_uc_reg_read.restype = ctypes.c_int

# uc_reg_read_batch: one C call reads all N registers — ~18x faster than individual calls.
_uc_reg_read_batch = _uclib.uc_reg_read_batch
_uc_reg_read_batch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
_uc_reg_read_batch.restype = ctypes.c_int

# uc_mem_read: bypass Qiling's mem.read stack (~10 us) with direct C call (~2 us).
_uc_mem_read = _uclib.uc_mem_read
_uc_mem_read.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
_uc_mem_read.restype = ctypes.c_int

# uc_hook_add: bypass Qiling's hook_add wrapper AND unicorn's `uccallback`
# / `__hook_code_cb` Python wrappers entirely.  Profile of v6.1 showed
# ~8.6 s out of ~13 s spent in those binding wrappers across 1.19 M
# callbacks.  By going straight to ctypes-wrapped uc_hook_add, we trade
# the two-frame Python wrapper for a single ctypes trampoline frame
# that directly invokes the Cython __call__ (which is itself C).
_uc_hook_add = _uclib.uc_hook_add
_uc_hook_add.argtypes = [
    ctypes.c_void_p,  # uc_engine *
    ctypes.c_void_p,  # uc_hook *
    ctypes.c_int,  # type
    ctypes.c_void_p,  # callback (HOOK_CODE_CFUNC)
    ctypes.c_void_p,  # user_data
    ctypes.c_uint64,  # begin
    ctypes.c_uint64,  # end
]
_uc_hook_add.restype = ctypes.c_int

# Native callback signature for code hooks — same as Unicorn's binding.
_HOOK_CODE_CFUNC = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,  # uc handle
    ctypes.c_uint64,  # address
    ctypes.c_uint32,  # size
    ctypes.c_void_p,  # user_data
)

# Native callback signature for valid mem read/write hooks.
# Same prototype the Unicorn binding uses internally for UC_HOOK_MEM_{READ,WRITE}.
_HOOK_MEM_ACCESS_CFUNC = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,  # uc handle
    ctypes.c_int,  # access type (read/write)
    ctypes.c_uint64,  # address
    ctypes.c_int,  # size
    ctypes.c_int64,  # value
    ctypes.c_void_p,  # user_data
)

# Native callback signature for invalid-mem hooks (returns bool — True keeps
# emulating, False stops).  Used for UC_HOOK_MEM_WRITE_UNMAPPED.
_HOOK_MEM_INVALID_CFUNC = ctypes.CFUNCTYPE(
    ctypes.c_bool,
    ctypes.c_void_p,  # uc handle
    ctypes.c_int,  # access type
    ctypes.c_uint64,  # address
    ctypes.c_int,  # size
    ctypes.c_int64,  # value
    ctypes.c_void_p,  # user_data
)

# UC_HOOK constants we need but that aren't already imported from unicorn.
_UC_HOOK_MEM_READ_CONST = 1024  # UC_HOOK_MEM_READ
_UC_HOOK_MEM_WRITE_CONST = 2048  # UC_HOOK_MEM_WRITE

# Pre-allocated C arrays for the memory-operand reads.
_MEM_BUF_SIZE = 64
_MEM_BUF = (ctypes.c_uint8 * _MEM_BUF_SIZE)()
# Pre-cast typed pointers into _MEM_BUF — direct C dereference, ~2.5x faster than struct.
_MEM_PTRS: dict[int, object] = {
    1: ctypes.cast(_MEM_BUF, ctypes.POINTER(ctypes.c_uint8)),
    2: ctypes.cast(_MEM_BUF, ctypes.POINTER(ctypes.c_uint16)),
    4: ctypes.cast(_MEM_BUF, ctypes.POINTER(ctypes.c_uint32)),
    8: ctypes.cast(_MEM_BUF, ctypes.POINTER(ctypes.c_uint64)),
}

# Vector registers reach the state as geometry-derived VL_<offset> lanes, and
# one Unicorn read of a whole vector fills two of them.  Only x86 is wired that
# way so far: every other architecture's vector file reaches the state through
# the scalar map above where its lanes happen to be nameable (AArch64 `D0` is
# `V0`'s low lane) and through the circuit's own lane synthesis otherwise.
_VECTOR_LANES: dict[Architecture, dict[int, tuple[str, str, int]]] = {
    Architecture.AMD64: {
        0x1200 + i * 0x40: (f'VL_{0x1200 + i * 0x40:#x}',
                            f'VL_{0x1200 + i * 0x40 + 8:#x}', 122 + i)
        for i in range(16)
    },
}


# One architecture's side of the register boundary.
#
# Everything below used to be module-level constants for x86-64, which is why a
# guest of any other architecture read no registers at all: the offset map had
# no entry for a single one of them, the arrays came out empty, and the frame
# was seeded with zeros.  It is per-architecture state now, built once on
# demand from the geometry (see archregs) and held here because the ctypes
# arrays it hands to Unicorn have to stay alive for as long as their addresses
# are cached.
#: What `_RegisterFile._build` returns: the ctypes arrays for one register set
#: plus the raw addresses the C hot path reads them through.  Named because two
#: methods hand it around and an 11-tuple in a signature says nothing.
RegReadDescriptor = tuple[
    # None in all three when the read is empty: n_calls == 0 and there is
    # nothing to allocate.
    ctypes.Array[ctypes.c_int] | None,       # register ids
    ctypes.Array[ctypes.c_uint64] | None,    # the values the read fills
    ctypes.Array[ctypes.c_void_p] | None,    # pointers into the values
    int,                    # how many value slots the read fills
    list[str],              # the register name each slot carries
    bool,                   # the packed flags register is among them
    ctypes.c_int | int,     # n_calls pre-wrapped; a plain 0 when empty
    int, int, int,          # ids / ptrs / vals addresses
    int,                    # n_calls
]

#: What one block's minimal read descriptor is: the three array addresses,
#: the call count, the slot each value lands in, whether the packed flags
#: register is among them, and the arrays themselves so the cache keeps
#: them alive.
BlockReadDescriptor = tuple[int, int, int, int, list[int], bool, object]


class _RegisterFile:
    """The ctypes arrays that carry one guest's registers across the boundary.

    Layout of the per-instruction arrays
    ------------------------------------
    - A scalar register occupies one ctypes ``c_uint64`` slot in ``vals`` and
      one entry in ``names``.
    - A vector register (16 bytes) occupies ONE ``uc_reg_read`` into a 16-byte
      region and TWO consecutive slots; ``names`` gets both lane names but
      ``ids``/``ptrs`` get only one entry, because one read fills both halves.

    The hot path then stores ``vals[i]`` under ``names[i]`` for every slot, so
    the lanes come out as the right two values with no special case there.
    """

    __slots__ = (
        '_cache',
        '_ids',
        '_n_calls',
        '_n_slots',
        '_ptrs',
        '_vals',
        'all_ids',
        'all_names',
        'arch',
        'regs',
        'vectors',
    )

    def __init__(self, arch: Architecture) -> None:
        self.arch = arch
        self.regs = archregs.for_arch(arch)
        self.vectors = _VECTOR_LANES.get(arch, {})
        self._cache: dict[int | frozenset[int], RegReadDescriptor] = {}

        # The whole-file read, for the snapshot fallbacks.  Vector lanes are
        # appended after the scalars so a lane pair stays contiguous.
        names = list(self.regs.all_names)
        ids = list(self.regs.all_uc_ids)
        n_scalar = len(names)
        for _off, (lo, hi, uc_id) in sorted(self.vectors.items()):
            names.extend((lo, hi))
            ids.append(uc_id)
        self.all_names = names
        self.all_ids = ids
        self._n_slots = len(names)
        self._n_calls = len(ids)
        self._ids = (ctypes.c_int * self._n_calls)(*ids)
        self._vals = (ctypes.c_uint64 * self._n_slots)()
        base = ctypes.addressof(self._vals)
        self._ptrs = (ctypes.c_void_p * self._n_calls)(
            *[base + i * 8 for i in range(n_scalar)],
            *[base + (n_scalar + 2 * i) * 8 for i in range(len(self.vectors))],
        )

    # -- per-instruction ------------------------------------------------
    def offsets_arrays(self, offsets: AbstractSet[int]) -> RegReadDescriptor:
        """Build and cache the arrays for one instruction's input offsets.

        Keyed first on ``id(offsets)``, because the caller hands back the same
        set object every time (DecodedOps holds it) and identity is cheaper
        than hashing it.  A plain `set`, despite what this said before: the
        parameter is typed as the abstract kind so both spellings fit.
        """
        oid = id(offsets)
        cached = self._cache.get(oid)
        if cached is not None:
            return cached
        key = frozenset(offsets)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache[oid] = cached
            return cached
        cached = self._cache[key] = self._cache[oid] = self._build(offsets)
        return cached

    def _build(self, offsets: AbstractSet[int]) -> RegReadDescriptor:
        uc_names: list[str] = []      # one per vals slot
        uc_ids: list[int] = []        # one per uc_reg_read call
        call_slots: list[int] = []    # first vals slot each call writes
        needs_flags = False
        seen: set[int] = set()
        next_slot = 0

        offset_to_uc = self.regs.offset_to_uc
        for off in offsets:
            # Vector first: a distinct id space, and one read fills two slots.
            lanes = self.vectors.get(off)
            if lanes is not None:
                lo, hi, uc_id = lanes
                if uc_id in seen:
                    continue
                seen.add(uc_id)
                uc_ids.append(uc_id)
                call_slots.append(next_slot)
                uc_names.append(lo)
                uc_names.append(hi)
                next_slot += 2
                continue
            entry = offset_to_uc.get(off)
            if entry is None:
                continue
            name, uc_id, is_flag = entry
            if uc_id in seen:
                if is_flag:
                    needs_flags = True
                continue
            seen.add(uc_id)
            uc_ids.append(uc_id)
            call_slots.append(next_slot)
            uc_names.append(name)
            next_slot += 1
            if is_flag:
                needs_flags = True

        n_slots = next_slot
        n_calls = len(uc_ids)
        if n_calls == 0:
            return (None, None, None, 0, [], False, 0, 0, 0, 0, 0)

        ids_arr = (ctypes.c_int * n_calls)(*uc_ids)
        vals_arr = (ctypes.c_uint64 * n_slots)()
        ptrs_arr = (ctypes.c_void_p * n_calls)(
            *[ctypes.addressof(vals_arr) + slot * 8 for slot in call_slots],
        )
        # n_calls is pre-wrapped as a ctypes c_int so the per-instruction batch
        # call skips c_int.from_param on every call -- a measured ~2.9% of a
        # taint-heavy trace.  The trailing raw addresses let the Cython hook
        # call uc_reg_read_batch through a C function pointer and read vals_arr
        # via a uint64*, skipping ctypes entirely.  The arrays are cached, so
        # their addresses stay valid.
        return (ids_arr, vals_arr, ptrs_arr, len(uc_names), uc_names, needs_flags,
                ctypes.c_int(n_calls),
                ctypes.addressof(ids_arr), ctypes.addressof(ptrs_arr),
                ctypes.addressof(vals_arr), n_calls)

    # -- whole file -----------------------------------------------------
    def read_all(self, uch: ctypes.c_void_p) -> dict[str, int]:
        """Every register this architecture exposes, in one batch call."""
        try:
            err = _uc_reg_read_batch(uch, self._ids, self._ptrs, self._n_calls)
            if err != 0:
                raise OSError(f'uc_reg_read_batch: {err}')
            out = {self.all_names[i]: int(self._vals[i]) for i in range(self._n_slots)}
            self.regs.unpack_flags(out)
            return out
        except Exception:
            return {}


_REGISTER_FILES: dict[Architecture, _RegisterFile] = {}


def register_file(arch: Architecture) -> _RegisterFile:
    f = _REGISTER_FILES.get(arch)
    if f is None:
        f = _REGISTER_FILES[arch] = _RegisterFile(arch)
    return f


#: Qiling's architecture tag -> the architecture the engine lifts for.  A guest
#: that is not on this list gets a clear failure rather than an x86 lifter
#: pointed at its instructions, which is what used to happen: every byte was
#: decoded as x86-64, no register was read, and the run reported nothing at all.
_QL_ARCH_TO_ARCH: dict[int, Architecture] = {
    QL_ARCH.X86: Architecture.X86,
    QL_ARCH.X8664: Architecture.AMD64,
    QL_ARCH.ARM64: Architecture.ARM64,
    QL_ARCH.RISCV64: Architecture.RISCV64,
    QL_ARCH.PPC: Architecture.PPC32BE,
}


def _guest_architecture(ql: Qiling) -> Architecture:
    try:
        ql_arch = ql.arch.type
    except AttributeError:
        return Architecture.AMD64
    arch = _QL_ARCH_TO_ARCH.get(ql_arch)
    if arch is None:
        raise NotImplementedError(
            f'microtaint has no register geometry for {ql_arch!r}; '
            f'supported guests are '
            f'{", ".join(sorted(a.value for a in _QL_ARCH_TO_ARCH.values()))}')
    return arch


class MicrotaintWrapper:
    def __init__(
        self,
        ql: Qiling,
        check_bof: bool = True,
        check_uaf: bool = True,
        check_sc: bool = True,
        check_aiw: bool = True,
        reporter: Reporter | None = None,
    ) -> None:
        self.ql = ql
        self.check_bof = check_bof
        self.check_uaf = check_uaf
        self.check_sc = check_sc
        self.check_aiw = check_aiw
        self.reporter = reporter or Reporter()

        self.arch = _guest_architecture(ql)
        self._regfile = register_file(self.arch)
        self._regs = self._regfile.regs
        # PC register fed to the circuit, so PC-relative operands resolve
        # against the runtime program counter rather than the translate base.
        self._pc_reg_name = self._regs.pc_name
        self._format = archregs.state_format(self.arch)
        self._format_key: tuple[tuple[str, int], ...] = tuple(
            (r.name, r.bits) for r in self._format)
        self.sim = CellSimulator(self.arch, use_unicorn=False)
        self.shadow_mem = BitPreciseShadowMemory()

        self._register_taint: dict[str, int] = {}
        self._main_bounds: list[tuple[int, int]] = []
        self._main_single: bool = False
        self._main_base: int = 0
        self._main_end: int = 0
        # Pre-instruction snapshots
        self._pre_regs: dict[str, int] = {}
        self._pre_taint: dict[str, int] = {}

        # Tier 3: per-instruction-address memoization cache.
        # Maps address → (taint_signature_tuple, output_state_dict).
        # On a hit, we apply the cached output_state directly without running
        # circuit.evaluate(ectx).  Bench profile: 150 unique addresses for
        # 1.19M callbacks (avg 7948 revisits each) — extremely cache-friendly.
        # Disabled by setting MICROTAINT_DISABLE_INSTR_CACHE=1.
        self._instr_cache_enabled: bool = os.environ.get('MICROTAINT_DISABLE_INSTR_CACHE') != '1'
        self._instr_cache: dict[int, tuple[frozenset[tuple[str, int]], dict[str, int]]] = {}
        self._instr_cache_hits: int = 0
        self._instr_cache_misses: int = 0
        # V5: Cython hot-path hook. Set MICROTAINT_DISABLE_CYTHON_HOOK=1 to
        # fall back to the Python method (for debugging).
        self._disable_cython_hook: bool = os.environ.get('MICROTAINT_DISABLE_CYTHON_HOOK') == '1'
        # V7: register the instruction hook as a PURE-C function pointer (a Cython
        # `with gil` trampoline) instead of a ctypes CFUNCTYPE(python) wrapper, so
        # Unicorn calls it without a per-instruction Python frame (~523ns -> ~140ns
        # per callback; the taint logic in _evaluate is unchanged).  Set
        # MICROTAINT_C_HOOK=0 to fall back to the ctypes CFUNCTYPE path.
        self._use_c_hook: bool = os.environ.get('MICROTAINT_C_HOOK', '1') != '0'
        #: user_data for the C trampoline: the hook's context address,
        #: kept in its ctypes wrapper so Unicorn receives the same object.
        self._instr_hook_ud: ctypes.c_void_p | None = None
        self._instr_hook_obj: object = None
        # Cython mem-hook trampolines (CFUNCTYPE instances + the Cython
        # callables they wrap).  These must be kept alive for the lifetime
        # of the Unicorn instance — Unicorn keeps only the raw function
        # pointer, not the Python object that backs it.
        #: The ctypes callbacks handed to Unicorn, kept alive here because
        #: Unicorn holds only the raw pointer.
        self._mem_cfuncs: list[object] = []

        # Tracks addresses written with nonzero taint by the most recent circuit
        # evaluation. The mem_write hook reads this set to avoid clearing taint
        # that the circuit intentionally set.
        self._last_tainted_writes: set[int] = set()

        # Unicorn C handle — resolved lazily on first instruction hook call
        # because ql.uc may not be valid at __init__ time.
        # Cache ImplicitTaintPolicy — computed once, used on every instruction.
        self._policy = ImplicitTaintPolicy.STOP if (check_sc or check_bof) else ImplicitTaintPolicy.IGNORE
        self._uc_handle: ctypes.c_void_p = None  # type: ignore[assignment]  # set properly in _setup_hooks
        # Cython mem-reader for circuit_c's OP_PUSH_MEM_VALUE.  Built in
        # _setup_hooks() once we have the Unicorn handle.
        self._live_mem_reader: LiveMemReader | None = None
        self._any_taint: bool = False  # set True on first _taint_bytes call
        self._mem_write_hook_registered: bool = False  # set True when hook registered
        # Armed lazily, on the first poison; see _arm_uaf_read_hook.
        self._mem_read_hook: MemAccessHook | None = None
        self._instr_hook_registered: bool = False  # set True when instr hook registered
        #: Block-at-a-time taint (MICROTAINT_BLOCK=1).  Opt-in; the
        #: per-instruction path is untouched when it is off.  The runtime is
        #: pure C (emulator/blockpath.h); this holds the C context alive.
        #: The block hook's C context, an opaque capsule from blockpath_c.
        self._block_ctx: _Capsule | None = None

        self._setup_hooks()

    # ------------------------------------------------------------------
    # Syscall hooks
    # ------------------------------------------------------------------

    def _taint_bytes(self, address: int, n: int) -> None:
        """
        Mark n bytes starting at address as fully tainted.
        Writes in 8-byte chunks to avoid uint64_t overflow in shadow.pyx
        (mask = (1 << (n*8)) - 1 overflows for n > 8).
        """
        self._arm_deferred_hooks()
        FULL_MASK = 0xFFFFFFFFFFFFFFFF
        written = 0
        while written + 8 <= n:
            self.shadow_mem.write_mask(address + written, FULL_MASK, 8)
            written += 8
        if written < n:
            remainder = n - written
            remainder_mask = (1 << (remainder * 8)) - 1
            self.shadow_mem.write_mask(address + written, remainder_mask, remainder)

    # ------------------------------------------------------------------
    # Public bit-precise taint injection API
    # ------------------------------------------------------------------

    def taint_bit(self, address: int, bit_index: int) -> None:
        """
        Mark exactly one bit of memory as tainted, leaving all other bits
        of the shadow byte at `address` untouched.

        `address`   — byte address in emulated memory.
        `bit_index` — which bit within that byte (0 = LSB, 7 = MSB).

        Semantics
        ---------
        OR-into-existing.  Calling taint_bit(addr, 0) then taint_bit(addr, 1)
        leaves shadow[addr] with bits 0 AND 1 set (mask 0x03).  This matches
        the natural reading of "mark this bit as tainted" without disturbing
        other bits' taint state.  Use taint_region(addr, [0x00]) to clear.

        The instruction hook is armed on the first call (same as _taint_bytes).
        """
        if not 0 <= bit_index <= 7:
            raise ValueError(f'bit_index must be 0-7, got {bit_index}')
        self._arm_deferred_hooks()
        # Preserve any existing taint at this byte: read, OR in the new bit, write back.
        existing = self.shadow_mem.read_mask(address, 1)
        self.shadow_mem.write_mask(address, existing | (1 << bit_index), 1)

    def taint_region(self, address: int, mask_bytes: bytes | bytearray) -> None:
        """
        Mark a region of memory with an explicit per-byte, per-bit taint mask.

        `address`    — start address in emulated memory.
        `mask_bytes` — one byte per memory byte; each bit in that byte controls
                       whether the corresponding input bit is considered tainted.
                       0x00 = no taint, 0xFF = fully tainted, 0x0F = low nibble tainted.

        This is the general form of _taint_bytes (which uses 0xFF for every byte).
        """
        self._arm_deferred_hooks()
        for i, mask in enumerate(mask_bytes):
            # Always call write_mask even when mask is 0: this explicitly clears
            # any pre-existing taint at that byte, matching write_mask's documented
            # semantics ("write_mask(addr, 0, n) explicitly clears n bytes of taint").
            self.shadow_mem.write_mask(address + i, mask, 1)

    @property
    def register_taint(self) -> dict[str, int]:
        """Register taint state — the external contract (seeded before a run,
        read during or after it).

        With the default dict hot path this is simply the dict.  Under the
        array-native hook (Phase 1.3c) the authoritative state lives in the hook's
        slot-indexed C array during a run, so reading this property syncs the
        array back into the dict AND hands authority back to the dict; the next
        instruction reloads the array from it, which is what makes an external
        re-seed between runs take effect.
        """
        hook = getattr(self, '_instr_hook_obj', None)
        if hook is not None:
            sync = getattr(hook, 'sync_taint_to_dict', None)
            if sync is not None:
                sync()
        return self._register_taint

    @register_taint.setter
    def register_taint(self, value: dict[str, int]) -> None:
        self._register_taint = value
        hook = getattr(self, '_instr_hook_obj', None)
        if hook is not None:
            try:
                hook.register_taint = value
            except (AttributeError, TypeError):
                pass

    def _make_cython_hook(self) -> InstructionHook | None:
        """Build a Cython-compiled hook callable.  Returns None if hook
        construction itself fails (e.g. some wrapper field isn't ready).
        The hook_core module is a hard import: when it's unavailable,
        microtaint cannot start at all, and that failure surfaces at the
        package import (not here) — exactly the behaviour we want."""
        try:
            return InstructionHook(
                self,
                uc_handle=self._uc_handle,
                uc_mem_read=_uc_mem_read,
                uc_reg_read_batch=_uc_reg_read_batch,
                uc_reg_read_batch_addr=ctypes.cast(_uc_reg_read_batch, ctypes.c_void_p).value or 0,
                mem_buf=_MEM_BUF,
                arch=self.arch,
                cached_gen_rule=_cached_generate_static_rule,
                x64_format_key=self._format_key,
                get_decoded=_get_decoded,
                build_offsets_arrs=self._regfile.offsets_arrays,
                eflags_bits=self._regs.flag_bits,
                flag_parent=self._regs.flag_parent or 'EFLAGS',
                pc_reg_name=self._pc_reg_name,
                eval_context_cls=EvalContext,
            )
        except Exception as exc:
            logger.debug(f'Cython hook construction failed: {exc}')
            return None

    # ------------------------------------------------------------------
    # Block-at-a-time taint
    # ------------------------------------------------------------------
    def _block_mode_enabled(self) -> bool:
        return os.environ.get('MICROTAINT_BLOCK', '') not in ('', '0')

    def _block_read_descriptor(self, offsets: frozenset[int],
                           slot_map: dict[str, int],
                           cache: dict[frozenset[int], BlockReadDescriptor],
                           ) -> BlockReadDescriptor:
        """A minimal uc_reg_read_batch descriptor for one block's reads.

        The whole-file read was 90% of block mode's cost when it was first
        wired, so a block asks only for what its regions read.

        Every call gets TWO 8-byte slots even when it fills one.  Unicorn
        writes a register's natural width, and some are wider than eight
        bytes; the whole-file layout hides that by putting every wide
        register last, but a FILTERED layout interleaves them, and a wide
        read then overwrites the next register's slot.  Measured: with one
        slot per call, asking for more registers made the answer WORSE (one
        diverging word became thirty-six), which is the signature of a
        neighbour being clobbered rather than of a register being missed.
        The padding costs 8 bytes per register and removes the hazard.
        """
        hit = cache.get(offsets)
        if hit is not None:
            return hit
        offset_to_uc = self._regfile.regs.offset_to_uc
        uc_ids: list[int] = []
        slots: list[int] = []
        needs_flags = False
        seen: set[int] = set()
        for off in sorted(offsets):
            lanes = self._regfile.vectors.get(off)
            if lanes is not None:
                lo, hi, uc_id = lanes
                if uc_id in seen:
                    continue
                seen.add(uc_id)
                uc_ids.append(uc_id)
                slots.extend((slot_map.get(lo, -1), slot_map.get(hi, -1)))
                continue
            entry = offset_to_uc.get(off)
            if entry is None:
                continue
            name, uc_id, is_flag = entry
            if is_flag:
                needs_flags = True
            if uc_id in seen:
                continue
            seen.add(uc_id)
            uc_ids.append(uc_id)
            slots.extend((slot_map.get(name, -1), -1))
        n_calls = len(uc_ids)
        got: BlockReadDescriptor
        if n_calls == 0:
            got = (0, 0, 0, 0, [], False, None)
            cache[offsets] = got
            return got
        ids_arr = (ctypes.c_int * n_calls)(*uc_ids)
        vals_arr = (ctypes.c_uint64 * (2 * n_calls))()
        base = ctypes.addressof(vals_arr)
        ptrs_arr = (ctypes.c_void_p * n_calls)(
            *[base + 16 * i for i in range(n_calls)])
        got = (ctypes.addressof(ids_arr), ctypes.addressof(ptrs_arr), base,
               n_calls, slots, needs_flags, (ids_arr, vals_arr, ptrs_arr))
        cache[offsets] = got
        return got

    def _install_block_hook(self, hook: InstructionHook) -> bool:
        """Register the pure-C UC_HOOK_BLOCK trampoline.  -> did it install?

        Block mode OWNS the taint: it computes a whole basic block at a time, so
        the per-instruction hook must NOT also be armed.  Both write the same
        state and would clobber each other, and the result would measure
        whichever ran last.
        """
        from unicorn import UC_HOOK_BLOCK  # noqa: PLC0415

        from microtaint.emulator import blockpath_c  # noqa: PLC0415
        from microtaint.taint_ir.blockcompile import compile_block  # noqa: PLC0415

        regfile = self._regfile
        # Every register needs a slot before the first block compiles: the
        # per-instruction path interns them as instructions mention them, and
        # that path never runs here.
        hook.prepare_block_mode(regfile.all_names)
        slot_map = dict(hook.slot_map)
        reg_slots = [slot_map.get(n, -1) for n in regfile.all_names]

        desc_cache: dict[frozenset[int], BlockReadDescriptor] = {}

        def descriptor(offsets: frozenset[int]) -> BlockReadDescriptor:
            return self._block_read_descriptor(offsets, slot_map, desc_cache)

        def compiler(address: int, size: int) -> _Capsule | None:
            """Plan one block.  Called once per DISTINCT block, with the GIL,
            from the C hook: this is the compiler, not the runtime."""
            try:
                code = bytes(self.ql.mem.read(address, size))
            except Exception:  # an unreadable block is unhandleable
                return None
            got = compile_block(self.arch, code, address, slot_map,
                                descriptor=descriptor)
            return None if got is None else got[0]

        # `_vals` has one slot per NAME and `_ids`/`_ptrs` one per uc_reg_read
        # call: a vector register is one call that fills two lanes.  So the
        # slot list is per name and the call count is separate.
        # The last two are the addresses of the hook's own code_lo/code_hi.
        # Block mode plans blocks the instruction hook never decodes, so it
        # widens that range itself; otherwise the mem-write hook's
        # self-modifying-code guard never fires and a rewritten block keeps
        # running the plan compiled for the bytes that used to be there.
        lo_addr, hi_addr = hook.code_range_addrs()
        block_ctx = blockpath_c.hook_new(
            c_instruction_hook_ud(hook), compiler,
            ctypes.addressof(regfile._ids), ctypes.addressof(regfile._ptrs),
            ctypes.addressof(regfile._vals), regfile._n_calls, reg_slots,
            lo_addr, hi_addr)
        self._block_ctx = block_ctx
        hook.block_invalidate = lambda: blockpath_c.hook_invalidate(block_ctx)
        h = ctypes.c_size_t()
        err = _uc_hook_add(self._uc_handle, ctypes.byref(h), UC_HOOK_BLOCK,
                           ctypes.c_void_p(blockpath_c.hook_ptr()),
                           ctypes.c_void_p(blockpath_c.hook_ud(block_ctx)), 1, 0)
        if err != 0:
            logger.warning(f'block hook registration failed: {err}')
            self._block_ctx = None
            return False
        return True

    def block_mode_finish(self, completed: bool = True) -> None:
        """End of the run: commit the last block, or drop it, and report.

        The caller knows whether the run finished.  A block's taint is held
        until the NEXT block proves it completed, so the last one has nobody to
        prove it and has to be told.  The findings are held by the same rule
        and released here for the same reason, which is why the drain comes
        after the commit rather than before it.
        """
        if self._block_ctx is None:
            return
        from microtaint.emulator import blockpath_c  # noqa: PLC0415

        blockpath_c.hook_finish(self._block_ctx, completed)
        self.block_mode_drain_reports()

    def block_mode_drain_reports(self) -> int:
        """Emit the secret-dependent branches block mode has found.  -> count.

        The C runtime detects them without the GIL and leaves them in a ring;
        turning one into a finding is Python, so it happens here, off the hot
        path.  Until this existed the ring had no reader on the hook path at
        all: block mode ran a real binary, computed that the program counter
        was secret-dependent, and reported nothing.

        Block mode does NOT stop the emulator the way the per-instruction path
        does.  The report is already a block late, so stopping would not
        prevent anything, and these milestones are analyses rather than
        mitigations: the run continues and every later leak is found too.
        """
        if self._block_ctx is None:
            return 0
        from microtaint.emulator import blockpath_c  # noqa: PLC0415

        drained = blockpath_c.hook_reports(self._block_ctx)
        for address, mask in drained:
            self._report_tainted_pc(address, mask)
        return len(drained)

    def _report_tainted_pc(self, address: int, mask: int) -> None:
        """One secret-dependent program counter, as the right KIND of finding.

        The same split the per-instruction path makes: a tainted return or
        indirect jump is control-flow hijack, anything else is a branch whose
        direction leaks.  Read from the instruction rather than from the
        opcode so it is not an x86 rule -- an ISA whose return is spelled
        differently still lands in the right bucket via its own disassembler,
        and one that says nothing at all degrades to the side-channel report
        rather than to silence.
        """
        mnemonic = ''
        asm_str = ''
        try:
            code = bytes(self.ql.mem.read(address, 16))
        except Exception:
            code = b''
        if code:
            mnemonic, asm_str = self._disasm(code, address)
        is_hijack = mnemonic.startswith('ret') or mnemonic in ('jmp', 'call')
        if is_hijack and self.check_bof:
            self.reporter.bof(address, instruction=asm_str)
        elif not is_hijack and self.check_sc:
            self.reporter.side_channel(address, instruction=asm_str,
                                       taint_mask=mask)

    def block_mode_stats(self) -> dict[str, int] | None:
        """Blocks seen, handled, and NOT handled.

        `unhandled` is unanalysed code, so it is counted rather than ignored: a
        block this path cannot lower is skipped, and the only honest thing is
        to say how often that happened.
        """
        if self._block_ctx is None:
            return None
        from microtaint.emulator import blockpath_c  # noqa: PLC0415

        return dict(blockpath_c.hook_stats(self._block_ctx))

    def _arm_deferred_hooks(self) -> None:
        """
        Arm the instruction hook and mem-write hook if not already registered.
        Called by any public taint injection method so that hooks activate the
        first time ANY taint is introduced, regardless of whether it came from
        the read() syscall or a direct taint_bit() / taint_region() call.
        """
        if self._any_taint:
            return
        self._any_taint = True
        if self._block_ctx is None and self._block_mode_enabled():
            block_hook = self._make_cython_hook()
            if block_hook is not None and self._install_block_hook(block_hook):
                self._instr_hook_obj = block_hook          # keep it alive
                self._instr_hook_registered = True         # and do NOT arm it
        if not self._instr_hook_registered:
            # Build the Cython hot-path hook callable. Falls back to the
            # Python method if Cython hook construction fails.
            # `InstrHook` because both InstructionHook.__call__ and the plain
            # bound method _instruction_evaluator_raw satisfy it.
            instr_hook: InstrHook = (
                self._make_cython_hook() or self._instruction_evaluator_raw
                if not self._disable_cython_hook
                else self._instruction_evaluator_raw
            )
            self._instr_hook_obj = instr_hook  # keep alive

            if self._main_single:
                # FAST PATH: bypass Unicorn's Python binding wrappers
                # entirely.  Wrap the Cython callable in a ctypes
                # CFUNCTYPE trampoline (single frame) and register it
                # directly via uc_hook_add — no uccallback/Wrapper frames.
                # Saves ~3-4 us per callback x 1.19M = ~4-5 s on the bench.
                #
                # We must keep the CFUNCTYPE instance alive ourselves
                # (Unicorn won't track it because we're calling uc_hook_add
                # outside of unicorn-py3).  Stash on self._instr_cfunc.
                # Choose the callback: a pure-C trampoline (no per-instruction
                # Python frame) when enabled and the Cython hook is in use, else
                # the ctypes CFUNCTYPE(python) wrapper.  Both call the SAME
                # InstructionHook._evaluate, so taint results are identical.
                cb_ptr: ctypes.c_void_p
                cb_ud: ctypes.c_void_p | None
                if self._use_c_hook and isinstance(instr_hook, InstructionHook):
                    self._instr_cfunc = None  # no CFUNCTYPE needed on the C path
                    # user_data = the hook's C context, not the hook itself: the
                    # trampoline runs without the GIL, so it cannot cast a
                    # PyObject to reach it.  The context carries a pointer back
                    # to the hook for the path that does need Python.  Either
                    # way the hook is kept alive via self._instr_hook_obj.
                    self._instr_hook_ud = ctypes.c_void_p(
                        c_instruction_hook_ud(instr_hook))
                    cb_ptr = ctypes.c_void_p(c_instruction_hook_ptr())
                    cb_ud = self._instr_hook_ud
                else:
                    self._instr_cfunc = _HOOK_CODE_CFUNC(instr_hook)
                    cb_ptr = ctypes.cast(self._instr_cfunc, ctypes.c_void_p)
                    cb_ud = None
                hook_handle = ctypes.c_size_t()
                rc = _uc_hook_add(
                    self._uc_handle,
                    ctypes.byref(hook_handle),
                    UC_HOOK_CODE,
                    cb_ptr,
                    cb_ud,  # user_data (the hook instance on the C path)
                    ctypes.c_uint64(self._main_base),
                    ctypes.c_uint64(self._main_end),
                )
                if rc != 0:
                    # Fall back to the slow path on registration failure.
                    logger.debug(f'uc_hook_add bypass failed (rc={rc}); falling back to ql.uc.hook_add')
                    self.ql.uc.hook_add(
                        UC_HOOK_CODE,
                        instr_hook,
                        begin=self._main_base,
                        end=self._main_end,
                    )
                else:
                    # Stash the Unicorn-internal callback list to keep it
                    # alive (Unicorn frees the function pointer on uc_close
                    # via _callbacks dict; we register our own bookkeeping).
                    self._instr_hook_handle = hook_handle.value
            else:
                self.ql.hook_code(self._instruction_evaluator)
            self._instr_hook_registered = True
        if not self._mem_write_hook_registered:
            # Same Cython hook + direct-bypass registration as in _setup_hooks().
            self._mem_write_hook = MemWriteClearHook(self)
            self._register_cython_mem_hook(
                self._mem_write_hook,
                _UC_HOOK_MEM_WRITE_CONST,
            )
            self._mem_write_hook_registered = True
        # Wire the mem-write hook to the instruction hook so a write onto cached
        # code (self-modifying / JIT) invalidates its decode + output caches.
        # Only the Cython InstructionHook has those caches; the Python fallback
        # re-reads bytes every instruction, so it needs no invalidation.
        if isinstance(self._mem_write_hook, MemWriteClearHook):
            # `self._instr_hook_obj`, not the local: the local is only bound
            # when the instruction hook was actually built, and block mode
            # skips that branch entirely.
            hook_obj = self._instr_hook_obj
            self._mem_write_hook.instr_hook = (
                hook_obj if isinstance(hook_obj, InstructionHook) else None
            )
        # NOTE: block mode caches a PLAN per block, which goes stale the same
        # way a decode cache does when a write lands on cached code.  It is NOT
        # wired to the invalidation yet -- MemWriteClearHook is a cdef class
        # with no __dict__, so it cannot simply carry a reference -- which is
        # one of the reasons block mode is opt-in and not yet trusted on
        # self-modifying code.

    def _syscall_number(self, name: str) -> int | None:
        """The number `name` has on THIS guest architecture, or None.

        Qiling's own id -> name mapper is architecture-specific, so inverting it
        is what makes a syscall hook portable.  Scanned once at setup; the
        ranges cover the usual Linux tables including MIPS' 4000/5000 bases.
        """
        mapper = getattr(self.ql.os, 'syscall_mapper', None)
        if mapper is None:
            return None
        want = f'ql_syscall_{name}'
        for lo, hi in ((0, 600), (4000, 4500), (5000, 5500), (6000, 6500)):
            for num in range(lo, hi):
                try:
                    if mapper(num) == want:
                        return num
                except Exception:  # noqa: S112 - a number this guest does not map
                    continue
        return None

    def _set_syscall_hook(self, name: str, handler: SyscallHandler,
                          intercept: object) -> None:
        """Hook a syscall by name, registered under its number on this guest.

        Registering the NUMBER rather than the name is deliberate.  Qiling
        resolves `hooks[name] or hooks[number]`, so a name-keyed hook silently
        outranks any number-keyed one a caller installed later, and overriding
        our read hook by number is an established way to drive the engine (the
        taint-API tests do exactly that).  Looking the number up per
        architecture keeps that contract and still gets the right syscall.

        The number was previously hard-coded to the x86-64 value, so on every
        other guest architecture this hooked whatever syscall happened to share
        the number: read is 0 on x86-64 but 63 on AArch64 and 5000 on MIPS64.
        The effect was that no taint was ever injected on a non-x86-64 guest and
        the engine reported nothing, quickly.
        """
        num = self._syscall_number(name)
        self.ql.os.set_syscall(num if num is not None else name, handler, intercept)

    def _arm_uaf_read_hook(self) -> None:
        """Register the UAF read callback, now that something is poisoned."""
        if self._mem_read_hook is not None or not self.check_uaf:
            return
        hook = MemAccessHook(self)
        self._mem_read_hook = hook
        self._register_cython_mem_hook(hook, _UC_HOOK_MEM_READ_CONST)
        logger.debug('armed the UAF read hook on first poison')

    def _register_cython_mem_hook(
        self,
        hook_obj: MemHook,
        hook_type: int,
    ) -> None:
        """Register a Cython MemWriteClearHook / MemAccessHook via the
        same direct uc_hook_add bypass used for the instruction hook.

        We trade Qiling's hook_mem_{read,write} wrapper plus Unicorn's
        `uccallback` + `__hook_mem_access_cb` Python frames for a single
        ctypes trampoline frame that calls into Cython's `__call__`.

        When the hook is one of the Cython classes, we go one step further and
        register a pure-C trampoline instead of a CFUNCTYPE, exactly as the
        instruction hook does: a ctypes callback pays libffi closure setup and
        per-argument conversion on EVERY guest load and store, which is one of
        the largest remaining CPython costs in the profile. The hook object is
        passed as user_data and kept alive on `self._mem_cfuncs`.

        The CFUNCTYPE instance (non-C path) must likewise be kept alive for the
        lifetime of the Unicorn instance.
        """
        cb_ud = None
        if self._use_c_hook and isinstance(hook_obj, MemWriteClearHook):
            # keep alive; user_data points INTO it (its C context, because
            # the trampoline runs without the GIL and cannot cast a PyObject)
            self._mem_cfuncs.append(hook_obj)
            cb_ud = ctypes.c_void_p(c_mem_write_hook_ud(hook_obj))
            cb_ptr = ctypes.c_void_p(c_mem_write_hook_ptr())
        elif self._use_c_hook and isinstance(hook_obj, MemAccessHook):
            self._mem_cfuncs.append(hook_obj)
            cb_ud = ctypes.c_void_p(c_mem_access_hook_ud(hook_obj))
            cb_ptr = ctypes.c_void_p(c_mem_access_hook_ptr())
        else:
            cfunc = _HOOK_MEM_ACCESS_CFUNC(hook_obj)
            self._mem_cfuncs.append(cfunc)
            cb_ptr = ctypes.cast(cfunc, ctypes.c_void_p)
        handle = ctypes.c_size_t()
        rc = _uc_hook_add(
            self._uc_handle,
            ctypes.byref(handle),
            hook_type,
            cb_ptr,
            cb_ud,
            ctypes.c_uint64(0),
            ctypes.c_uint64(0xFFFFFFFFFFFFFFFF),
        )
        if rc != 0:
            # Fall back to Qiling's high-level wrapper on registration failure.
            logger.debug(f'uc_hook_add bypass for hook_type {hook_type} failed (rc={rc}); using ql.uc.hook_add')
            self.ql.uc.hook_add(hook_type, hook_obj)

    def _register_cython_invalid_hook(
        self,
        hook_obj: InvalidMemHook,
        hook_type: int,
    ) -> None:
        """Register a Cython invalid-mem hook (e.g. UC_HOOK_MEM_WRITE_UNMAPPED).

        Same bypass as _register_cython_mem_hook but uses HOOK_MEM_INVALID_CFUNC
        (returns bool — False stops emulation).
        """
        cfunc = _HOOK_MEM_INVALID_CFUNC(hook_obj)
        self._mem_cfuncs.append(cfunc)
        handle = ctypes.c_size_t()
        rc = _uc_hook_add(
            self._uc_handle,
            ctypes.byref(handle),
            hook_type,
            ctypes.cast(cfunc, ctypes.c_void_p),
            None,
            ctypes.c_uint64(0),
            ctypes.c_uint64(0xFFFFFFFFFFFFFFFF),
        )
        if rc != 0:
            logger.debug(
                f'uc_hook_add bypass for invalid hook_type {hook_type} failed (rc={rc}); using ql.uc.hook_add',
            )
            self.ql.uc.hook_add(hook_type, hook_obj)

    def _setup_hooks(self) -> None:
        # Cache the raw Unicorn C handle once. ql.uc is a property that
        # does work on every access — caching avoids 407k property calls.
        self._uc_handle = self.ql.uc._uch
        # Build the Cython LiveMemReader once.  Replaces the bound-method
        # _read_live_memory that circuit_c invokes from OP_PUSH_MEM_VALUE
        # — saves ~0.5 us of Python frame setup per call x 256k calls/run.
        # uc_mem_read is also handed over as a raw C function pointer (plus the
        # buffer address) so LiveMemReader can do the guest read without any
        # ctypes marshalling; perf attributed ~14-15% of the taint phase to the
        # ctypes boxing on this path.  MICROTAINT_DISABLE_CMEMREAD=1 keeps the
        # ctypes path (the addresses are simply not passed).
        _mr_addr = 0
        _buf_addr = 0
        if os.environ.get('MICROTAINT_DISABLE_CMEMREAD') != '1':
            try:
                _mr_addr = ctypes.cast(_uc_mem_read, ctypes.c_void_p).value or 0
                _buf_addr = ctypes.addressof(_MEM_BUF)
            except Exception:
                _mr_addr = 0
                _buf_addr = 0
        self._live_mem_reader = LiveMemReader(
            self,
            uc_mem_read=_uc_mem_read,
            mem_buf=_MEM_BUF,
            mem_ptrs=_MEM_PTRS,
            uc_mem_read_addr=_mr_addr,
            mem_buf_addr=_buf_addr,
        )

        self._set_syscall_hook('read', self._sys_read_hook, QL_INTERCEPT.CALL)
        # 334 is rseq on x86-64 only; the stub exists for that specific syscall,
        # and the number means something else elsewhere, so it is registered
        # only where it is the one meant.
        if self.ql.arch.type == QL_ARCH.X8664:
            self.ql.os.set_syscall(334, self._stub_unimplemented_syscall,
                                   QL_INTERCEPT.ENTER)

        if self.check_uaf:
            # UAF detection requires the mem-write hook from the start —
            # writes to poisoned (munmap'd) memory must be intercepted even
            # before any taint is injected (UAF tests pass payload=b'').
            #
            # All three mem hooks are now Cython callables registered via
            # the same direct uc_hook_add bypass we use for the instruction
            # hook (Tier 4).  This eliminates the per-callback Python frame
            # overhead from `__hook_mem_access_cb` / `uccallback`.
            self._mem_write_hook = MemWriteClearHook(self)
            self._uaf_unmapped_hook = UafUnmappedWriteHook(self)
            self._register_cython_mem_hook(
                self._mem_write_hook,
                _UC_HOOK_MEM_WRITE_CONST,
            )
            # The READ callback exists only to catch a read from freed memory,
            # and nothing is freed yet.  Unicorn calls it on every guest load,
            # which measured 97 ns per instruction on bench_dense purely to be
            # registered, so it is armed the first time anything is actually
            # poisoned instead.  The shadow is the single choke point for that
            # (four call sites poison; all of them go through poison()), and it
            # fires this before returning to the guest, so the callback is in
            # place before any read that could be a use-after-free.
            self._mem_read_hook = None
            self.shadow_mem.on_first_poison = self._arm_uaf_read_hook
            self._register_cython_invalid_hook(
                self._uaf_unmapped_hook,
                _UC_HOOK_MEM_WRITE_UNMAPPED,
            )
            self._mem_write_hook_registered = True
            self._set_syscall_hook('munmap', self._munmap_hook, QL_INTERCEPT.ENTER)
        else:
            # For non-UAF modes, defer _mem_write_clear_hook until taint exists.
            # This skips 1.5M hook dispatches before the first read() call.
            self._mem_write_hook_registered = False

        # Register the instruction hook with a C-level address range filter.
        # Unicorn filters instructions in C before any Python is called —
        # eliminates the entire Qiling hook dispatch overhead for libc/loader instructions.
        begin, end = self._get_main_binary_range()
        if begin and end:
            self._main_base = begin
            self._main_end = end
            self._main_single = True
            self._main_bounds = [(begin, end)]
        # Do NOT register the instruction hook here.
        # It is registered lazily in _taint_bytes when the first taint is injected.
        # Unicorn runs 5M+ instructions before taint injection — registering early
        # costs ~6 us/instruction in Unicorn's C->Python overhead even for early-exit.
        # Deferred registration: ~0 cost before taint, normal cost after.
        self._instr_hook_registered = False

    def _sys_read_hook(self, ql: Qiling, fd: int, buf: int, count: int) -> int:
        if fd != 0 or count <= 0:
            try:
                f = ql.os.fd[fd]
                data = f.read(count) if f else b''
            except Exception:
                data = b''
            if data:
                ql.mem.write(buf, data)
                n = len(data)
                self._taint_bytes(buf, n)
                self.reporter.taint_source(buf, n, fd=fd)
            return len(data) if data else -9

        try:
            f = ql.os.stdin
            data = f.read(count)
        except Exception:
            data = b''

        if not data:
            return 0

        n = len(data)
        ql.mem.write(buf, data)
        self._taint_bytes(buf, n)
        self.reporter.taint_source(buf, n, fd=0)
        logger.debug(f'Tainted {n} bytes at 0x{buf:x} from stdin')
        return n

    def _stub_unimplemented_syscall(self, ql: Qiling, *_args: object) -> None:
        ql.arch.regs.write('RAX', 0xFFFFFFFFFFFFFFDA)

    # Labels that Qiling assigns to regions that are NOT user heap data.
    # Munmapping these is glibc/loader cleanup; poisoning them causes
    # false-positive UAFs when libc re-reads them during its own shutdown
    # (observed on glibc >= 2.40 even with syscall(SYS_exit, 0)).
    _SKIP_POISON_LABELS: frozenset[str] = frozenset(
        {
            '[stack]',
            '[GDT]',
            '[vsyscall]',
            '[vdso]',
            '[vvar]',
            '[hook_mem]',
        },
    )

    def _region_label_at(self, ql: Qiling, addr: int, length: int) -> str | None:
        """Return the Qiling memory-map label for the region overlapping addr."""
        end = addr + length
        for start, stop, _perms, label, *_ in ql.mem.map_info:
            if start < end and stop > addr:  # overlap
                return str(label)
        return None

    def _munmap_hook(self, ql: Qiling, addr: int, length: int,
                     *_args: object) -> None:
        if length <= 0:
            return
        label = self._region_label_at(ql, addr, length)
        if label is not None:
            # Skip system / dynamic-library regions.
            if label in self._SKIP_POISON_LABELS:
                logger.debug(f'Skipped poisoning system region {label!r} at 0x{addr:x}')
                return
            # Dynamic libraries end with .so or .so.<N> — libc, ld-linux, etc.
            # These are munmap'd by the dynamic linker during shutdown and
            # re-read by its own cleanup code immediately after.
            if label.endswith(('.so', '.so.2', '.so.1', '.so.0')) or '.so.' in label:
                logger.debug(f'Skipped poisoning shared-lib region {label!r} at 0x{addr:x}')
                return
        self.shadow_mem.poison(addr, length)
        logger.debug(f'Poisoned freed mmap region at 0x{addr:x} ({length}B)')

    def _uaf_unmapped_write_hook(
        self,
        uc: object,  # noqa: ARG002
        access: int,  # noqa: ARG002
        address: int,
        size: int,
        value: int,  # noqa: ARG002
        user_data: object,  # noqa: ARG002
    ) -> bool:
        """Fires when code writes to UNMAPPED memory (UC_HOOK_MEM_WRITE_UNMAPPED).

        Catches the mmap->munmap->write UAF pattern where the page is fully unmapped.
        Returns False so Unicorn terminates the emulation run.
        """
        if self.shadow_mem.is_poisoned(address, size):
            self.reporter.uaf(address, size)
        self.ql.emu_stop()
        return False

    def _mem_write_clear_hook(self, ql: Qiling, _access: int, address: int, size: int, _value: int) -> None:
        if self.check_uaf and self.shadow_mem.is_poisoned(address, size):
            self.reporter.uaf(address, size)
            ql.emu_stop()
            return

        if not self._last_tainted_writes:
            self.shadow_mem.clear(address, size)
        else:
            for i in range(size):
                if address + i not in self._last_tainted_writes:
                    self.shadow_mem.clear(address + i, 1)

    def _mem_access_hook(self, ql: Qiling, _access: int, address: int, size: int, _value: int) -> None:
        if self.check_uaf and self.shadow_mem.is_poisoned(address, size):
            self.reporter.uaf(address, size)
            ql.emu_stop()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_live_registers(self, uch: ctypes.c_void_p) -> dict[str, int]:
        """Every register this guest exposes, in one batch call."""
        vals = self._regfile.read_all(uch)
        return vals if vals else self._get_live_registers_python()

    def _get_live_registers_python(self) -> dict[str, int]:
        """Python-binding fallback for register reads (slower but always correct)."""
        names = self._regfile.all_names
        try:
            raw = self.ql.uc.reg_read_batch(list(self._regfile.all_ids))
            vals = dict(zip(names, raw, strict=False))
        except Exception:
            vals = {}
            for name in names:
                try:
                    vals[name] = self.ql.arch.regs.read(name)
                except Exception:
                    vals[name] = 0
        self._regs.unpack_flags(vals)
        return vals  # return all — caller uses what it needs

    def _get_main_binary_range(self) -> tuple[int, int]:
        """Return (base, end) of the main binary, or (0, 0) if unavailable."""
        try:
            if hasattr(self.ql.loader, 'images') and len(self.ql.loader.images) > 0:
                img = self.ql.loader.images[0]
                return img.base, img.end
        except Exception:
            logger.warning(
                'Failed to get main binary range from Qiling loader — falling back to no range filter',
                exc_info=True,
            )
        return 0, 0

    def _is_main_binary(self, address: int) -> bool:
        if not self._main_bounds:
            if hasattr(self.ql.loader, 'images') and len(self.ql.loader.images) > 0:
                main_image = self.ql.loader.images[0]
                self._main_bounds.append((main_image.base, main_image.end))
                self._main_base = main_image.base
                self._main_end = main_image.end
                self._main_single = True
            else:
                self._main_single = False
                return True
        if self._main_single:
            return self._main_base <= address < self._main_end
        return any(s <= address < e for s, e in self._main_bounds)

    def _disasm(self, instruction_bytes: bytes, address: int) -> tuple[str, str]:
        try:
            md = self.ql.arch.disassembler
            insn = next(md.disasm(instruction_bytes, address))
            return insn.mnemonic.lower(), f'{insn.mnemonic} {insn.op_str}'.strip()
        except Exception:
            return '', ''

    @staticmethod
    def _parse_mem_key(key: str) -> tuple[int, int] | None:
        if not key.startswith('MEM_'):
            return None
        body = key[4:]
        last = body.rfind('_')
        if last < 0:
            return None
        try:
            addr = int(body[:last], 16)
            size = int(body[last + 1 :])
            return addr, size
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Core instruction evaluator
    # ------------------------------------------------------------------

    def _instruction_evaluator_raw(  # noqa: C901
        self,
        uc: object,  # noqa: ARG002
        address: int,
        size: int,
        user_data: object,  # noqa: ARG002
    ) -> None:
        """Direct Unicorn hook — bypasses Qiling's Python dispatch chain (~12 us/instr saved)."""
        if not self.register_taint and not self._any_taint:
            return
        uch = self._uc_handle
        err = _uc_mem_read(uch, address, _MEM_BUF, size)
        instruction_bytes = bytes(_MEM_BUF[:size]) if err == 0 else bytes(self.ql.mem.read(address, size))
        circuit = _cached_generate_static_rule(self.arch, instruction_bytes, self._format_key)

        # Tier 3 fast path: per-address memoization.
        # Check cache BEFORE the expensive register batch read.  On hit
        # we don't need _pre_regs at all — the cached output_state is just
        # a register-taint mapping.
        cache_key = None
        compiled_circuit = circuit._compiled
        if TYPE_CHECKING:
            assert compiled_circuit is not True
        if (
            self._instr_cache_enabled
            and compiled_circuit is not None
            and compiled_circuit is not False
            and not compiled_circuit.has_mem_ops
        ):
            cache_key = frozenset(self.register_taint.items()) if self.register_taint else _EMPTY_FROZENSET
            cached = self._instr_cache.get(address)
            if cached is not None and cached[0] == cache_key:
                # Cache hit: replay the output_state directly.
                output_state = cached[1]
                self._instr_cache_hits += 1
                if self._last_tainted_writes:
                    self._last_tainted_writes.clear()
                if self.register_taint:
                    self.register_taint.clear()
                for key, val in output_state.items():
                    if val > 0:
                        self.register_taint[key] = val
                return
            self._instr_cache_misses += 1

        # Read all live registers for the C1/C2 concrete execution in the pcode evaluator.
        # _load() uses these to populate frame values for both pcode and Unicorn cell paths.
        # Targeted register read: only fetch what this instruction's pcode needs.
        # _get_decoded is lru_cache'd — O(1) dict lookup, no recomputation.
        # input_reg_offsets = exact Sleigh byte offsets of SP_REGISTER pcode inputs.
        try:
            _decoded = _get_decoded(self.arch, instruction_bytes)
            _uc_arrs = _decoded._uc_arrays
            if _uc_arrs is None:
                _uc_arrs = self._regfile.offsets_arrays(_decoded.input_reg_offsets)
                _decoded._uc_arrays = _uc_arrs
            # Inlined _exec_regs_from_arrays — eliminates function call overhead.
            # (This Python fallback keeps the ctypes call path; the trailing raw
            # addresses are for the Cython hook's C-level path only.)
            _ids, _vals, _ptrs, _n, _names, _need_ef, _n_calls = _uc_arrs[:7]
            if _ids is None or _vals is None:
                self._pre_regs = {}
            else:
                # _n_calls = number of uc_reg_read calls (one per UC reg id);
                # _n = number of vals slots = len(_names) (XMM contributes 2).
                _uc_reg_read_batch(uch, _ids, _ptrs, _n_calls)
                self._pre_regs = {_names[_i]: int(_vals[_i]) for _i in range(_n)}
                if _need_ef:
                    self._regs.unpack_flags(self._pre_regs)
        except Exception:
            self._pre_regs = self._get_live_registers(uch)
        # Feed the REAL program counter so PC-relative memory operands (modelled by
        # the circuit as pc-register-relative) resolve against the runtime PC, not
        # the fixed translate base.  `address` is this instruction's own PC.
        self._pre_regs[self._pc_reg_name] = address
        self._pre_taint = dict(self.register_taint)

        ectx = EvalContext(
            input_taint=self._pre_taint,
            input_values=self._pre_regs,
            simulator=self.sim,
            implicit_policy=self._policy,
            shadow_memory=self.shadow_mem,
            mem_reader=self._live_mem_reader,
        )

        try:
            output_state = circuit.evaluate(ectx)

            # Tier 3: cache successful eval result for this address+taint_sig.
            # We only populate the cache for circuits without mem ops (gated
            # by cache_key being non-None).
            if cache_key is not None:
                # Make a private copy of output_state — circuit.evaluate may
                # return ectx.input_taint reference for some Cython paths.
                self._instr_cache[address] = (cache_key, dict(output_state))

            # Single pass over output_state: update shadow_mem, register_taint,
            # _last_tainted_writes, and collect MEM_ entries for AIW check.
            if self._last_tainted_writes:
                self._last_tainted_writes.clear()
            if self.register_taint:
                self.register_taint.clear()
            mem_writes: list[tuple[int, int, int]] = []  # (addr, size, val)

            for key, val in output_state.items():
                if key[:4] == 'MEM_':
                    # key[:4]=='MEM_' is ~2x faster than str.startswith for this hot loop.
                    _body = key[4:]
                    _last = _body.rfind('_')
                    if _last < 0:
                        continue
                    try:
                        mem_addr = int(_body[:_last], 16)
                        mem_size = int(_body[_last + 1 :])
                    except ValueError:
                        continue
                    self.shadow_mem.write_mask(mem_addr, val, mem_size)
                    if val:
                        for i in range(mem_size):
                            if (val >> (i * 8)) & 0xFF:
                                self._last_tainted_writes.add(mem_addr + i)
                        if self.check_aiw:
                            mem_writes.append((mem_addr, mem_size, val))
                elif val > 0:
                    self.register_taint[key] = val

            if self.check_aiw and self.register_taint and mem_writes:
                live_regs = ectx.input_values
                for mem_addr, _, _ in mem_writes:
                    for reg_name, reg_taint in ectx.input_taint.items():
                        if reg_taint == 0:
                            continue
                        reg_val = live_regs.get(reg_name, 0)
                        if reg_val == 0:
                            continue
                        if abs(int(mem_addr) - int(reg_val)) <= 4096:
                            mnemonic, asm_str = self._disasm(instruction_bytes, address)
                            self.reporter.aiw(
                                address,
                                pointer_taint=reg_taint,
                                instruction=asm_str,
                            )
                            self.ql.emu_stop()
                            return

        except ImplicitTaintError as e:
            mnemonic, asm_str = self._disasm(instruction_bytes, address)
            is_hijack = mnemonic.startswith('ret') or mnemonic in ('jmp', 'call')

            if is_hijack and self.check_bof:
                self.reporter.bof(address, instruction=asm_str)
                self.ql.emu_stop()
            elif not is_hijack and self.check_sc:
                taint_mask = 0
                try:
                    for part in str(e).split():
                        if part.startswith('0x'):
                            taint_mask = int(part, 16)
                            break
                except Exception as e:
                    logger.debug(f'Error parsing taint mask from exception message: {e}')
                self.reporter.side_channel(address, instruction=asm_str, taint_mask=taint_mask)
                self.ql.emu_stop()
            else:
                self.ql.emu_stop()

    def _instruction_evaluator(self, ql: Qiling, address: int, size: int) -> None:  # noqa: ARG002
        """Qiling-path fallback — delegates to _instruction_evaluator_raw."""
        self._instruction_evaluator_raw(None, address, size, None)
