from __future__ import annotations

import logging

import unicorn
import unicorn.unicorn_py3
from qiling import Qiling
from qiling.const import QL_INTERCEPT

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


class MicrotaintWrapper:
    def __init__(self, ql: Qiling, check_bof: bool = True, check_uaf: bool = True, check_sc: bool = True) -> None:
        self.ql = ql
        self.check_bof = check_bof
        self.check_uaf = check_uaf
        self.check_sc = check_sc

        self.arch = Architecture.AMD64
        self.sim = CellSimulator(self.arch)
        self.shadow_mem = BitPreciseShadowMemory()

        self.register_taint: dict[str, int] = {}
        self._main_bounds: list[tuple[int, int]] = []

        self._setup_hooks()

    def _raw_syscall_hook(self, uc: unicorn.unicorn_py3.Uc, _user_data: None = None) -> None:
        import unicorn.x86_const as x86

        rax = uc.reg_read(x86.UC_X86_REG_RAX)
        rdi = uc.reg_read(x86.UC_X86_REG_RDI)
        rsi = uc.reg_read(x86.UC_X86_REG_RSI)
        rdx = uc.reg_read(x86.UC_X86_REG_RDX)
        rip = uc.reg_read(x86.UC_X86_REG_RIP)
        logger.error(f'[SYSCALL] rip={hex(rip)} rax={hex(rax)} rdi={hex(rdi)} rsi={hex(rsi)} rdx={hex(rdx)}')
        if rax == 0 and rdi == 0 and rdx > 0:
            mask = (1 << (rdx * 8)) - 1
            self.shadow_mem.write_mask(rsi, mask, rdx)
            logger.error(f'[*] Tainted {rdx} bytes at 0x{rsi:x} from STDIN')

    def _setup_hooks(self) -> None:
        # CALL intercept fires before syscall with correct buf address
        # We handle the read ourselves to ensure taint is applied
        self.ql.os.set_syscall(0, self._sys_read_hook, QL_INTERCEPT.CALL)
        self.ql.os.set_syscall(334, self._stub_unimplemented_syscall, QL_INTERCEPT.ENTER)

        if self.check_uaf:
            self.ql.os.set_syscall(11, self._munmap_hook, QL_INTERCEPT.ENTER)
            self.ql.hook_mem_read(self._mem_access_hook)
            self.ql.hook_mem_write(self._mem_access_hook)

        self.ql.hook_code(self._instruction_evaluator)

    def _sys_read_hook(self, ql: Qiling, fd: int, buf: int, count: int) -> int:
        """
        Replaces sys_read entirely via CALL intercept.
        Reads from Qiling's stdin (which may be a pipe, file, or tty),
        writes bytes into emulated memory, and taints the buffer.
        """
        if fd != 0 or count <= 0:
            # For non-stdin reads, fall back to Qiling's default behavior
            # by reading from the fd table
            try:
                f = ql.os.fd[fd]
                data = f.read(count) if f else b''
            except Exception:
                data = b''
            if data:
                ql.mem.write(buf, data)
            return len(data) if data else -9

        # For stdin: read from ql.os.stdin (set via pipe.SimpleInStream or default)
        try:
            f = ql.os.stdin
            data = f.read(count)
        except Exception:
            data = b''

        if not data:
            return 0

        n = len(data)
        ql.mem.write(buf, data)

        # Taint exactly the bytes that were read
        mask = (1 << (n * 8)) - 1
        self.shadow_mem.write_mask(buf, mask, n)
        logger.error(f'[*] Tainted {n} bytes at 0x{buf:x} from STDIN')

        return n

    def _stub_unimplemented_syscall(self, ql: Qiling, *args) -> None:
        # Return ENOSYS so the caller handles it gracefully
        ql.arch.regs.write('RAX', 0xFFFFFFFFFFFFFFDA)  # -38 = ENOSYS

    def _munmap_hook(self, ql: Qiling, addr: int, length: int, *args) -> None:
        if length > 0:
            self.shadow_mem.poison(addr, length)
            logger.error(f'[*] Poisoned freed memory at 0x{addr:x}')

    def _mem_access_hook(self, ql: Qiling, access: int, address: int, size: int, value: int) -> None:
        if self.check_uaf and self.shadow_mem.is_poisoned(address, size):
            logger.error(f'[!] UAF DETECTED: Access to poisoned memory at 0x{address:x}')
            ql.emu_stop()

    def _read_live_memory(self, address: int, size: int) -> int:
        try:
            return int.from_bytes(self.ql.mem.read(address, size), 'little')
        except Exception:
            return 0

    def _get_live_registers(self) -> dict[str, int]:
        vals = {}
        for reg in X64_FORMAT:
            try:
                vals[reg.name] = self.ql.arch.regs.read(reg.name)
            except Exception:
                pass
        return vals

    def _is_main_binary(self, address: int) -> bool:
        if not self._main_bounds:
            if hasattr(self.ql.loader, 'images') and len(self.ql.loader.images) > 0:
                main_image = self.ql.loader.images[0]
                self._main_bounds.append((main_image.base, main_image.end))
            else:
                return True
        return any(s <= address < e for s, e in self._main_bounds)

    def _instruction_evaluator(self, ql: Qiling, address: int, size: int) -> None:
        in_main = self._is_main_binary(address)
        if self._is_main_binary(address):
            logger.error(f'[DBG MAIN] 0x{address:x}')
        if not self._is_main_binary(address):
            return

        # Always propagate taint through all code (including libc/loader)
        # but only report violations when in the main binary
        instruction_bytes = ql.mem.read(address, size)
        circuit = generate_static_rule(self.arch, bytes(instruction_bytes), X64_FORMAT)
        print(f'[DBG] Evaluating instruction at 0x{address:x}, circuit: {circuit}')

        policy = (
            ImplicitTaintPolicy.STOP if (in_main and (self.check_sc or self.check_bof)) else ImplicitTaintPolicy.IGNORE
        )

        ctx = EvalContext(
            input_taint=self.register_taint,
            input_values=self._get_live_registers(),
            simulator=self.sim,
            implicit_policy=policy,
            shadow_memory=self.shadow_mem,
            mem_reader=self._read_live_memory,
        )

        try:
            output_state = circuit.evaluate(ctx)
            self.register_taint.clear()
            for key, val in output_state.items():
                if key.startswith('MEM_'):
                    parts = key.split('_')
                    addr, sz = int(parts[1], 16), int(parts[2])
                    self.shadow_mem.write_mask(addr, val, sz)
                elif val > 0:
                    self.register_taint[key] = val

        except ImplicitTaintError as e:
            # Only reachable when in_main and policy=STOP
            md = ql.arch.disassembler
            try:
                insn = next(md.disasm(instruction_bytes, address))
                mnemonic = insn.mnemonic.lower()
                is_hijack = mnemonic.startswith('ret') or mnemonic in ('jmp', 'call')
            except Exception:
                is_hijack = False

            if is_hijack and self.check_bof:
                logger.error(f'[!] FATAL: Buffer Overflow hijacked RIP at 0x{address:x}')
                ql.emu_stop()
            elif not is_hijack and self.check_sc:
                logger.error(f'[!] CRYPTO SIDE-CHANNEL DETECTED at 0x{address:x}: {e}')
                ql.emu_stop()
            else:
                ql.emu_stop()
