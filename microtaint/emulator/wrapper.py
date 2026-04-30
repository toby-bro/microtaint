# mypy: disable-error-code="attr-defined"
from __future__ import annotations

import ctypes
import logging
from typing import Any

import unicorn.unicorn_py3.unicorn as _uu
import unicorn.x86_const as _uc_x86_const
from qiling import Qiling
from qiling.const import QL_INTERCEPT
from unicorn import UC_HOOK_CODE, UC_HOOK_MEM_WRITE_UNMAPPED

from microtaint.emulator.reporter import Reporter
from microtaint.emulator.shadow import BitPreciseShadowMemory
from microtaint.instrumentation.ast import EvalContext, Expr
from microtaint.instrumentation.cell import _get_decoded
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import _cached_generate_static_rule
from microtaint.types import Architecture, ImplicitTaintError, ImplicitTaintPolicy, Register

logger = logging.getLogger(__name__)

X64_FORMAT = [
    Register('RAX', 64),
    Register('RBX', 64),
    Register('RCX', 64),
    Register('RDX', 64),
    Register('RSI', 64),
    Register('RDI', 64),
    Register('RBP', 64),
    Register('RSP', 64),
    Register('R8', 64),
    Register('R9', 64),
    Register('R10', 64),
    Register('R11', 64),
    Register('R12', 64),
    Register('R13', 64),
    Register('R14', 64),
    Register('R15', 64),
    Register('RIP', 64),
    Register('EFLAGS', 32),
    Register('ZF', 1),
    Register('CF', 1),
    Register('SF', 1),
    Register('OF', 1),
    Register('PF', 1),
]
# Pre-computed cache key for X64_FORMAT — avoids rebuilding a 24-element tuple
# via genexpr on every generate_static_rule call (was 2.26s in profiling).
_X64_FORMAT_KEY: tuple[tuple[str, int], ...] = tuple((r.name, r.bits) for r in X64_FORMAT)

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

# Pre-allocated C arrays — defined AFTER _ALL_REG_NAMES below.
_MEM_BUF_SIZE = 64
_MEM_BUF = (ctypes.c_uint8 * _MEM_BUF_SIZE)()
# Pre-cast typed pointers into _MEM_BUF — direct C dereference, ~2.5x faster than struct.
_MEM_PTRS: dict[int, object] = {
    1: ctypes.cast(_MEM_BUF, ctypes.POINTER(ctypes.c_uint8)),
    2: ctypes.cast(_MEM_BUF, ctypes.POINTER(ctypes.c_uint16)),
    4: ctypes.cast(_MEM_BUF, ctypes.POINTER(ctypes.c_uint32)),
    8: ctypes.cast(_MEM_BUF, ctypes.POINTER(ctypes.c_uint64)),
}

# Complete name->UC_X86_REG_* mapping for AMD64 (superset of what we need)
_AMD64_REG_ID: dict[str, int] = {
    'RAX': _uc_x86_const.UC_X86_REG_RAX,
    'RBX': _uc_x86_const.UC_X86_REG_RBX,
    'RCX': _uc_x86_const.UC_X86_REG_RCX,
    'RDX': _uc_x86_const.UC_X86_REG_RDX,
    'RSI': _uc_x86_const.UC_X86_REG_RSI,
    'RDI': _uc_x86_const.UC_X86_REG_RDI,
    'RBP': _uc_x86_const.UC_X86_REG_RBP,
    'RSP': _uc_x86_const.UC_X86_REG_RSP,
    'R8': _uc_x86_const.UC_X86_REG_R8,
    'R9': _uc_x86_const.UC_X86_REG_R9,
    'R10': _uc_x86_const.UC_X86_REG_R10,
    'R11': _uc_x86_const.UC_X86_REG_R11,
    'R12': _uc_x86_const.UC_X86_REG_R12,
    'R13': _uc_x86_const.UC_X86_REG_R13,
    'R14': _uc_x86_const.UC_X86_REG_R14,
    'R15': _uc_x86_const.UC_X86_REG_R15,
    'RIP': _uc_x86_const.UC_X86_REG_RIP,
    'EFLAGS': _uc_x86_const.UC_X86_REG_EFLAGS,
    # 32-bit halves (used by sub-register addressing)
    'EAX': _uc_x86_const.UC_X86_REG_EAX,
    'EBX': _uc_x86_const.UC_X86_REG_EBX,
    'ECX': _uc_x86_const.UC_X86_REG_ECX,
    'EDX': _uc_x86_const.UC_X86_REG_EDX,
    'ESI': _uc_x86_const.UC_X86_REG_ESI,
    'EDI': _uc_x86_const.UC_X86_REG_EDI,
    'EBP': _uc_x86_const.UC_X86_REG_EBP,
    'ESP': _uc_x86_const.UC_X86_REG_ESP,
    'R8D': _uc_x86_const.UC_X86_REG_R8D,
    'R9D': _uc_x86_const.UC_X86_REG_R9D,
    'R10D': _uc_x86_const.UC_X86_REG_R10D,
    'R11D': _uc_x86_const.UC_X86_REG_R11D,
    'R12D': _uc_x86_const.UC_X86_REG_R12D,
    'R13D': _uc_x86_const.UC_X86_REG_R13D,
    'R14D': _uc_x86_const.UC_X86_REG_R14D,
    'R15D': _uc_x86_const.UC_X86_REG_R15D,
}

# Parent register for flag names — we always read EFLAGS and unpack.
_FLAG_PARENTS = {'ZF', 'CF', 'SF', 'OF', 'PF', 'AF', 'DF', 'IF', 'TF'}

# EFLAGS bit positions
_EFLAGS_BITS = {
    'CF': 0,
    'PF': 2,
    'AF': 4,
    'ZF': 6,
    'SF': 7,
    'TF': 8,
    'IF': 9,
    'DF': 10,
    'OF': 11,
}

# Sleigh register-space byte offset -> (uc_reg_name, uc_reg_id, needs_eflags_unpack)
# Covers all GP registers + RIP + flag offsets for AMD64/x86-64.
# Used by _read_regs_by_offsets to translate pcode input_reg_offsets to UC reads.
_SLEIGH_OFFSET_TO_UC: dict[int, tuple[str, int, bool]] = {
    0: ('RAX', 35, False),
    8: ('RCX', 38, False),
    16: ('RDX', 40, False),
    24: ('RBX', 37, False),
    32: ('RSP', 44, False),
    40: ('RBP', 36, False),
    48: ('RSI', 43, False),
    56: ('RDI', 39, False),
    128: ('R8', 106, False),
    136: ('R9', 107, False),
    144: ('R10', 108, False),
    152: ('R11', 109, False),
    160: ('R12', 110, False),
    168: ('R13', 111, False),
    176: ('R14', 112, False),
    184: ('R15', 113, False),
    648: ('RIP', 41, False),
    512: ('EFLAGS', 25, True),  # CF
    514: ('EFLAGS', 25, True),  # PF
    516: ('EFLAGS', 25, True),  # AF
    518: ('EFLAGS', 25, True),  # ZF
    519: ('EFLAGS', 25, True),  # SF
    520: ('EFLAGS', 25, True),
    521: ('EFLAGS', 25, True),
    522: ('EFLAGS', 25, True),  # DF
    523: ('EFLAGS', 25, True),  # OF
}


# Cache of pre-built ctypes arrays per unique offset-set.
# Key: frozenset of Sleigh byte offsets (== DecodedOps.input_reg_offsets).
# Value: (ids_arr, vals_arr, ptrs_arr, uc_names, needs_eflags)
# Built once per unique instruction type, reused on every subsequent call.
_OFFSETS_CACHE: dict[
    int | frozenset[int],
    tuple[object, object, object, int, list[str], bool] | tuple[None, None, None, int, list[str], bool],
] = {}


def _build_offsets_arrays(offsets: frozenset[int]) -> tuple[object, object, object, int, list[str], bool]:
    """Build and cache ctypes arrays. Uses id() fast-path on the hot path."""
    oid = id(offsets)
    cached = _OFFSETS_CACHE.get(oid)
    if cached is not None:
        return cached
    key = frozenset(offsets)
    cached = _OFFSETS_CACHE.get(key)
    if cached is not None:
        _OFFSETS_CACHE[oid] = cached
        return cached

    uc_names: list[str] = []
    uc_ids: list[int] = []
    needs_eflags = False
    seen_ids: set[int] = set()

    for off in offsets:
        entry = _SLEIGH_OFFSET_TO_UC.get(off)
        if entry is None:
            continue
        uc_name, uc_id, is_flag = entry
        if uc_id in seen_ids:
            continue
        seen_ids.add(uc_id)
        uc_names.append(uc_name)
        uc_ids.append(uc_id)
        if is_flag:
            needs_eflags = True

    n = len(uc_ids)
    if n == 0:
        result: tuple[object, object, object, int, list[str], bool] | tuple[None, None, None, int, list[str], bool] = (
            None,
            None,
            None,
            0,
            [],
            False,
        )
        _OFFSETS_CACHE[key] = _OFFSETS_CACHE[oid] = result
        return result

    ids_arr = (ctypes.c_int * n)(*uc_ids)
    vals_arr = (ctypes.c_uint64 * n)()
    ptrs_arr = (ctypes.c_void_p * n)(*[ctypes.addressof(vals_arr) + i * 8 for i in range(n)])

    # Store n = len(uc_names) in the tuple to avoid len() on the hot path
    result = (ids_arr, vals_arr, ptrs_arr, len(uc_names), uc_names, needs_eflags)
    _OFFSETS_CACHE[key] = _OFFSETS_CACHE[oid] = result
    return result


# Pre-built: all 18 "full" register names (for the AIW snapshot fallback)
_ALL_REG_NAMES: list[str] = [
    'RAX',
    'RBX',
    'RCX',
    'RDX',
    'RSI',
    'RDI',
    'RBP',
    'RSP',
    'R8',
    'R9',
    'R10',
    'R11',
    'R12',
    'R13',
    'R14',
    'R15',
    'RIP',
    'EFLAGS',
]
_ALL_REG_IDS: list[int] = [_AMD64_REG_ID[n] for n in _ALL_REG_NAMES]

# Pre-allocated C arrays for uc_reg_read_batch — must come after _ALL_REG_NAMES.
_N_REGS = len(_ALL_REG_NAMES)
_IDS_ARR = (ctypes.c_int * _N_REGS)(*[_AMD64_REG_ID[n] for n in _ALL_REG_NAMES])
_VALS_ARR = (ctypes.c_uint64 * _N_REGS)()
_PTRS_ARR = (ctypes.c_void_p * _N_REGS)(*[ctypes.addressof(_VALS_ARR) + i * 8 for i in range(_N_REGS)])


def _read_regs_ctypes(uch: ctypes.c_void_p, _names_unused: list[str]) -> dict[str, int]:
    """
    Read all 18 AMD64 registers in one uc_reg_read_batch C call (~1.3 us total).
    Falls back gracefully on any error.
    """
    try:
        err = _uc_reg_read_batch(uch, _IDS_ARR, _PTRS_ARR, _N_REGS)
        if err != 0:
            raise OSError(f'uc_reg_read_batch: {err}')
        result: dict[str, int] = {_ALL_REG_NAMES[i]: int(_VALS_ARR[i]) for i in range(_N_REGS)}
        eflags = result.get('EFLAGS', 0)
        for flag, bit in _EFLAGS_BITS.items():
            result[flag] = (eflags >> bit) & 1
        return result
    except Exception:
        return {}


def _collect_expr_reg_names(expr: Expr, names: set[str]) -> None:
    """Recursively walk an Expr tree collecting InstructionCellExpr input keys
    and TaintOperand / ValueOperand names."""
    type_name = type(expr).__name__

    if type_name == 'InstructionCellExpr':
        # inputs is dict[str, Expr] — keys are register names needed as values
        for key in expr.inputs:
            names.add(key)
        # Also recurse into the input Exprs (they may be ValueOperand etc.)
        for sub in expr.inputs.values():
            _collect_expr_reg_names(sub, names)

    elif type_name == 'TaintOperand':
        names.add(expr.name)

    elif type_name == 'BinaryExpr':
        _collect_expr_reg_names(expr.lhs, names)
        _collect_expr_reg_names(expr.rhs, names)

    elif type_name == 'UnaryExpr':
        _collect_expr_reg_names(expr.expr, names)

    elif type_name == 'AvalancheExpr':
        _collect_expr_reg_names(expr.expr, names)

    elif type_name == 'MemoryOperand':
        _collect_expr_reg_names(expr.address_expr, names)

    elif type_name == 'ValueOperand':
        names.add(expr.name)

    # ConstExpr / LiteralExpr have no register deps — stop


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

        self.arch = Architecture.AMD64
        self.sim = CellSimulator(self.arch)
        self.shadow_mem = BitPreciseShadowMemory()

        self.register_taint: dict[str, int] = {}
        self._main_bounds: list[tuple[int, int]] = []
        self._main_single: bool = False
        self._main_base: int = 0
        self._main_end: int = 0
        # Pre-instruction snapshots
        self._pre_regs: dict[str, int] = {}
        self._pre_taint: dict[str, int] = {}

        # Tracks addresses written with nonzero taint by the most recent circuit
        # evaluation. The mem_write hook reads this set to avoid clearing taint
        # that the circuit intentionally set.
        self._last_tainted_writes: set[int] = set()

        # Unicorn C handle — resolved lazily on first instruction hook call
        # because ql.uc may not be valid at __init__ time.
        # Cache ImplicitTaintPolicy — computed once, used on every instruction.
        self._policy = ImplicitTaintPolicy.STOP if (check_sc or check_bof) else ImplicitTaintPolicy.IGNORE
        self._uc_handle: ctypes.c_void_p = None  # type: ignore[assignment]  # set properly in _setup_hooks
        self._any_taint: bool = False  # set True on first _taint_bytes call
        self._mem_write_hook_registered: bool = False  # set True when hook registered
        self._instr_hook_registered: bool = False  # set True when instr hook registered

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
        if not self._any_taint:
            self._any_taint = True
            # Register instruction hook now that taint exists.
            # Before this point, Unicorn ran without any instruction hook —
            # saving ~6 us x N_untainted_instructions in Unicorn's C overhead.
            if not self._instr_hook_registered:
                if self._main_single:
                    self.ql.uc.hook_add(
                        UC_HOOK_CODE,
                        self._instruction_evaluator_raw,
                        begin=self._main_base,
                        end=self._main_end,
                    )
                else:
                    self.ql.hook_code(self._instruction_evaluator)
                self._instr_hook_registered = True
            if not self._mem_write_hook_registered:
                # Register _mem_write_clear_hook now that taint exists.
                self.ql.hook_mem_write(self._mem_write_clear_hook)
                self._mem_write_hook_registered = True
        FULL_MASK = 0xFFFFFFFFFFFFFFFF
        written = 0
        while written + 8 <= n:
            self.shadow_mem.write_mask(address + written, FULL_MASK, 8)
            written += 8
        if written < n:
            remainder = n - written
            remainder_mask = (1 << (remainder * 8)) - 1
            self.shadow_mem.write_mask(address + written, remainder_mask, remainder)

    def _setup_hooks(self) -> None:
        # Cache the raw Unicorn C handle once. ql.uc is a property that
        # does work on every access — caching avoids 407k property calls.
        self._uc_handle = self.ql.uc._uch

        self.ql.os.set_syscall(0, self._sys_read_hook, QL_INTERCEPT.CALL)
        self.ql.os.set_syscall(334, self._stub_unimplemented_syscall, QL_INTERCEPT.ENTER)

        if self.check_uaf:
            # UAF detection requires _mem_write_clear_hook from the start —
            # writes to poisoned (munmap'd) memory must be intercepted even before
            # any taint is injected (UAF tests pass payload=b'').
            self.ql.hook_mem_write(self._mem_write_clear_hook)
            self._mem_write_hook_registered = True
            self.ql.os.set_syscall(11, self._munmap_hook, QL_INTERCEPT.ENTER)
            self.ql.hook_mem_read(self._mem_access_hook)
            # Also catch writes to fully UNMAPPED pages (mmap->munmap->write pattern).
            # UC_HOOK_MEM_WRITE_UNMAPPED fires before Qiling's crash handler.
            self.ql.uc.hook_add(_UC_HOOK_MEM_WRITE_UNMAPPED, self._uaf_unmapped_write_hook)
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

    def _stub_unimplemented_syscall(self, ql: Qiling, *_args: Any) -> None:
        ql.arch.regs.write('RAX', 0xFFFFFFFFFFFFFFDA)

    def _munmap_hook(self, _ql: Qiling, addr: int, length: int, *_args: Any) -> None:
        if length > 0:
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

    def _read_live_memory(self, address: int, size: int) -> int:
        """Read `size` bytes from Unicorn via uc_mem_read + typed ctypes pointer."""
        try:
            err = _uc_mem_read(self._uc_handle, address, _MEM_BUF, size)
            if err == 0:
                ptr = _MEM_PTRS.get(size)
                if ptr is not None:
                    return ptr[0]  # type: ignore[no-any-return,index]
                return int.from_bytes(_MEM_BUF[:size], 'little')
            return int.from_bytes(self.ql.mem.read(address, size), 'little')
        except Exception:
            return 0

    def _get_live_registers(self, uch: ctypes.c_void_p) -> dict[str, int]:
        """Read all 18 AMD64 registers via uc_reg_read_batch, fall back to Python."""
        try:
            return _read_regs_ctypes(uch, _ALL_REG_NAMES)
        except Exception:
            return self._get_live_registers_python()

    def _get_live_registers_python(self) -> dict[str, int]:
        """Python-binding fallback for register reads (slower but always correct)."""
        reg_ids = [_AMD64_REG_ID[n] for n in _ALL_REG_NAMES]
        try:
            raw = self.ql.uc.reg_read_batch(reg_ids)
            vals = dict(zip(_ALL_REG_NAMES, raw, strict=False))
        except Exception:
            vals = {}
            for name in _ALL_REG_NAMES:
                try:
                    vals[name] = self.ql.arch.regs.read(name)
                except Exception:
                    vals[name] = 0
        eflags = vals.get('EFLAGS', 0)
        for flag, bit in _EFLAGS_BITS.items():
            vals[flag] = (eflags >> bit) & 1
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
        circuit = _cached_generate_static_rule(self.arch, instruction_bytes, _X64_FORMAT_KEY)

        # Read all live registers for the C1/C2 concrete execution in the pcode evaluator.
        # _load() uses these to populate frame values for both pcode and Unicorn cell paths.
        # Targeted register read: only fetch what this instruction's pcode needs.
        # _get_decoded is lru_cache'd — O(1) dict lookup, no recomputation.
        # input_reg_offsets = exact Sleigh byte offsets of SP_REGISTER pcode inputs.
        try:
            _decoded = _get_decoded(self.arch, instruction_bytes)
            _uc_arrs = _decoded._uc_arrays
            if _uc_arrs is None:
                _uc_arrs = _build_offsets_arrays(_decoded.input_reg_offsets)
                _decoded._uc_arrays = _uc_arrs
            # Inlined _exec_regs_from_arrays — eliminates function call overhead
            _ids, _vals, _ptrs, _n, _names, _need_ef = _uc_arrs
            if _ids is None:
                self._pre_regs = {}
            else:
                _uc_reg_read_batch(uch, _ids, _ptrs, _n)
                self._pre_regs = {_names[_i]: int(_vals[_i]) for _i in range(_n)}
                if _need_ef:
                    _ef = self._pre_regs.get('EFLAGS', 0)
                    self._pre_regs.update({_f: (_ef >> _b) & 1 for _f, _b in _EFLAGS_BITS.items()})
        except Exception:
            self._pre_regs = self._get_live_registers(uch)
        self._pre_taint = dict(self.register_taint)

        ctx = EvalContext(
            input_taint=self._pre_taint,
            input_values=self._pre_regs,
            simulator=self.sim,
            implicit_policy=self._policy,
            shadow_memory=self.shadow_mem,
            mem_reader=self._read_live_memory,
        )

        try:
            output_state = circuit.evaluate(ctx)

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
                live_regs = ctx.input_values
                for mem_addr, _, _ in mem_writes:
                    for reg_name, reg_taint in ctx.input_taint.items():
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
