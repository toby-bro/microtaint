from __future__ import annotations

import ctypes
import logging
from typing import Any

import unicorn.unicorn_py3.unicorn as _uu
import unicorn.x86_const as _uc_x86_const
from qiling import Qiling
from qiling.const import QL_INTERCEPT

from microtaint.emulator.reporter import Reporter
from microtaint.emulator.shadow import BitPreciseShadowMemory
from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
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

# ---------------------------------------------------------------------------
# ctypes shim for uc_reg_read
#
# Unicorn exposes uc_reg_read(uc_handle, reg_id, *value) as a C function.
# Calling it directly via ctypes bypasses the entire Python binding stack:
#   _select_reg_class → genexpr → __seq_tuple → next() → __get_reg_read_arg
# which accounts for ~18µs per register in the Python binding.
# Direct ctypes cost: ~1µs per register.
#
# uclib is the module-level CDLL already loaded by the unicorn package.
# We grab it once at import time so there's zero attribute lookup per call.
# ---------------------------------------------------------------------------
_uclib = _uu.uclib
_uc_reg_read = _uclib.uc_reg_read  # uc_err uc_reg_read(uc_engine, int, void*)
_uc_reg_read.argtypes = [
    ctypes.c_void_p,  # uc_engine handle
    ctypes.c_int,  # reg id
    ctypes.c_void_p,  # pointer to result buffer
]
_uc_reg_read.restype = ctypes.c_int  # uc_err (0 = ok)

# Pre-allocate one uint64 buffer per register slot to avoid per-call allocation.
# The same buffers are reused across every _read_regs_ctypes call.
_MAX_REGS = 32
_REG_BUFS = (ctypes.c_uint64 * _MAX_REGS)()
_REG_PTRS = [ctypes.byref(_REG_BUFS, i * 8) for i in range(_MAX_REGS)]

# Complete name→UC_X86_REG_* mapping for AMD64 (superset of what we need)
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


def _read_regs_ctypes(uch: ctypes.c_void_p, names: list[str]) -> dict[str, int]:
    """
    Read exactly the named registers from Unicorn via direct ctypes calls.

    Skips Unicorn's Python binding entirely (no _select_reg_class, no __seq_tuple,
    no genexpr).  Cost: ~1 µs per register vs ~18 µs via reg_read_batch.

    Flag names (ZF, CF, SF, OF, PF) cause EFLAGS to be read if not already
    in `names`; individual flag bits are unpacked from EFLAGS in Python.

    Falls back to Python reg_read_batch on any ctypes error to ensure correctness.
    """
    try:
        result: dict[str, int] = {}
        need_eflags = False
        phys: list[tuple[str, int]] = []  # (name, uc_reg_id) pairs to read

        for name in names:
            if name in _FLAG_PARENTS:
                need_eflags = True
            elif name in _AMD64_REG_ID:
                phys.append((name, _AMD64_REG_ID[name]))

        if need_eflags and 'EFLAGS' not in [n for n, _ in phys]:
            phys.append(('EFLAGS', _AMD64_REG_ID['EFLAGS']))

        for i, (name, reg_id) in enumerate(phys):
            err = _uc_reg_read(uch, reg_id, _REG_PTRS[i])
            if err != 0:
                raise OSError(f'uc_reg_read failed: err={err}')
            result[name] = _REG_BUFS[i]

        # Unpack flags from EFLAGS
        if need_eflags or 'EFLAGS' in result:
            eflags = result.get('EFLAGS', 0)
            for flag, bit in _EFLAGS_BITS.items():
                result[flag] = (eflags >> bit) & 1

        return result

    except Exception:
        # ctypes path failed — fall back to Python reg_read_batch
        # This is slower but always correct, and handles environments
        # where the ctypes handle or library binding differs.
        return {}  # unreachable — outer try/except handles fallback


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
        self._uch: ctypes.c_void_p | None = None

        self._setup_hooks()

    def _get_uch(self) -> ctypes.c_void_p:
        """Resolve (and cache) the raw Unicorn C engine handle."""
        if self._uch is None:
            self._uch = self.ql.uc._uch
        return self._uch

    # ------------------------------------------------------------------
    # Syscall hooks
    # ------------------------------------------------------------------

    def _taint_bytes(self, address: int, n: int) -> None:
        """
        Mark n bytes starting at address as fully tainted.
        Writes in 8-byte chunks to avoid uint64_t overflow in shadow.pyx
        (mask = (1 << (n*8)) - 1 overflows for n > 8).
        """
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
        self.ql.os.set_syscall(0, self._sys_read_hook, QL_INTERCEPT.CALL)
        self.ql.os.set_syscall(334, self._stub_unimplemented_syscall, QL_INTERCEPT.ENTER)

        self.ql.hook_mem_write(self._mem_write_clear_hook)

        if self.check_uaf:
            self.ql.os.set_syscall(11, self._munmap_hook, QL_INTERCEPT.ENTER)
            self.ql.hook_mem_read(self._mem_access_hook)

        self.ql.hook_code(self._instruction_evaluator)

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
        try:
            return int.from_bytes(self.ql.mem.read(address, size), 'little')
        except Exception:
            return 0

    def _get_live_registers(self) -> dict[str, int]:
        """
        Read all 18 AMD64 registers from Unicorn via direct ctypes calls.

        Ctypes bypasses the Python binding's per-register dispatch overhead
        (~1 µs/reg vs ~36 µs/reg), giving ~18 µs total vs ~651 µs originally.
        Falls back to Python reg_read_batch if ctypes fails.

        All registers are read unconditionally — selective reading caused
        correctness regressions when the circuit needed registers not predicted
        by the static walker.
        """
        try:
            return _read_regs_ctypes(self._get_uch(), _ALL_REG_NAMES)
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

    def _instruction_evaluator(self, ql: Qiling, address: int, size: int) -> None:  # noqa: C901
        if not self._is_main_binary(address):
            return

        instruction_bytes = bytes(ql.mem.read(address, size))
        circuit = generate_static_rule(self.arch, instruction_bytes, X64_FORMAT)

        self._pre_regs = self._get_live_registers()
        self._pre_taint = dict(self.register_taint)

        policy = ImplicitTaintPolicy.STOP if (self.check_sc or self.check_bof) else ImplicitTaintPolicy.IGNORE

        ctx = EvalContext(
            input_taint=self._pre_taint,
            input_values=self._pre_regs,
            simulator=self.sim,
            implicit_policy=policy,
            shadow_memory=self.shadow_mem,
            mem_reader=self._read_live_memory,
        )

        try:
            output_state = circuit.evaluate(ctx)

            self._last_tainted_writes.clear()
            for key, val in output_state.items():
                if not key.startswith('MEM_') or val == 0:
                    continue
                parsed = self._parse_mem_key(key)
                if parsed is None:
                    continue
                mem_addr, mem_size = parsed
                for i in range(mem_size):
                    byte_taint = (val >> (i * 8)) & 0xFF
                    if byte_taint:
                        self._last_tainted_writes.add(mem_addr + i)

            self.register_taint.clear()

            for key, val in output_state.items():
                if key.startswith('MEM_'):
                    parsed = self._parse_mem_key(key)
                    if parsed is None:
                        continue
                    mem_addr, mem_size = parsed
                    self.shadow_mem.write_mask(mem_addr, val, mem_size)
                elif val > 0:
                    self.register_taint[key] = val

            if self.check_aiw and ctx.input_taint:
                live_regs = ctx.input_values

                for key in output_state:
                    if not key.startswith('MEM_'):
                        continue
                    parsed = self._parse_mem_key(key)
                    if parsed is None:
                        continue
                    mem_addr, _ = parsed

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
                            ql.emu_stop()
                            return

        except ImplicitTaintError as e:
            mnemonic, asm_str = self._disasm(instruction_bytes, address)
            is_hijack = mnemonic.startswith('ret') or mnemonic in ('jmp', 'call')

            if is_hijack and self.check_bof:
                self.reporter.bof(address, instruction=asm_str)
                ql.emu_stop()
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
                ql.emu_stop()
            else:
                ql.emu_stop()
