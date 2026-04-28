from __future__ import annotations

import logging

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
        # Tracks addresses written with nonzero taint by the most recent circuit
        # evaluation. The mem_write hook uses this to avoid clearing taint that
        # the circuit intentionally set for this instruction.
        self._last_tainted_writes: set[int] = set()

        self._setup_hooks()

    # ------------------------------------------------------------------
    # Syscall hooks
    # ------------------------------------------------------------------

    def _setup_hooks(self) -> None:
        self.ql.os.set_syscall(0, self._sys_read_hook, QL_INTERCEPT.CALL)
        self.ql.os.set_syscall(334, self._stub_unimplemented_syscall, QL_INTERCEPT.ENTER)

        # mem_write hook clears shadow taint for every store of untainted data.
        # This is necessary because the circuit only produces MEM_ outputs for
        # stores whose pointer addresses map to a known architectural register
        # (direct register-offset addressing). Stores with computed pointers
        # (call, push, mov [rbp-N], ...) don't appear in output_state, so the
        # circuit never gets a chance to clear stale shadow taint at those slots.
        # The Unicorn hook fires AFTER the instruction executes, with the concrete
        # address and value — we use it to clear shadow bytes that are being
        # overwritten with untainted data.
        # NOTE: we only clear if the stored value itself carries no taint
        # (the register_taint check). If a tainted register is stored, the
        # circuit's output_state MEM_ key (when present) already wrote the
        # correct taint mask. We let the circuit own tainted writes; we own
        # the clearing of untainted ones.
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
            return len(data) if data else -9

        try:
            f = ql.os.stdin
            data = f.read(count)
        except Exception:
            data = b''

        if not data:
            # Empty stream / EOF — return 0, introduce no taint
            return 0

        n = len(data)
        ql.mem.write(buf, data)

        # Byte-precise taint: one bit per byte in the mask
        mask = (1 << (n * 8)) - 1
        self.shadow_mem.write_mask(buf, mask, n)
        self.reporter.taint_source(buf, n, fd=0)
        logger.debug(f'Tainted {n} bytes at 0x{buf:x} from stdin')

        return n

    def _stub_unimplemented_syscall(self, ql: Qiling, *args) -> None:
        ql.arch.regs.write('RAX', 0xFFFFFFFFFFFFFFDA)  # -38 = ENOSYS

    def _munmap_hook(self, ql: Qiling, addr: int, length: int, *args) -> None:
        if length > 0:
            self.shadow_mem.poison(addr, length)
            logger.debug(f'Poisoned freed mmap region at 0x{addr:x} ({length}B)')

    def _mem_write_clear_hook(self, ql: Qiling, access: int, address: int, size: int, value: int) -> None:
        """
        Fires after every concrete memory write (Unicorn UC_HOOK_MEM_WRITE).

        With the engine fix in place, the circuit now produces MEM_ output keys
        for all STORE instructions (including push, call, mov [reg+off]).
        Those keys are processed by _instruction_evaluator which calls
        write_mask(addr, val, size) — clearing shadow when val=0 and setting
        it when val>0.

        This hook serves as a safety net for any stores the circuit still misses,
        and handles UAF detection on writes.

        Strategy: clear shadow for any byte that the circuit did NOT explicitly
        write with nonzero taint (tracked in _last_tainted_writes).
        """
        # UAF: detect write to freed memory
        if self.check_uaf and self.shadow_mem.is_poisoned(address, size):
            self.reporter.uaf(address, size)
            ql.emu_stop()
            return

        # Safety-net clearing for any bytes the circuit didn't explicitly taint.
        # After the engine fix this rarely fires, but handles edge cases.
        if not self._last_tainted_writes:
            self.shadow_mem.clear(address, size)
        else:
            for i in range(size):
                if address + i not in self._last_tainted_writes:
                    self.shadow_mem.clear(address + i, 1)

    def _mem_access_hook(self, ql: Qiling, access: int, address: int, size: int, value: int) -> None:
        """UAF detection on memory reads (separate from the write hook)."""
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
        vals: dict[str, int] = {}
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

    def _disasm(self, instruction_bytes: bytes, address: int) -> tuple[str, str]:
        """Return (mnemonic_lower, full_asm_string)."""
        try:
            md = self.ql.arch.disassembler
            insn = next(md.disasm(instruction_bytes, address))
            return insn.mnemonic.lower(), f'{insn.mnemonic} {insn.op_str}'.strip()
        except Exception:
            return '', ''

    @staticmethod
    def _parse_mem_key(key: str) -> tuple[int, int] | None:
        """
        Parse a MEM_ output-state key into (address, size_in_bytes).

        Key format produced by LogicCircuit.evaluate():
            MEM_<hex_address>_<decimal_size_bytes>
        e.g. MEM_0x7fff1000_8  ->  (0x7fff1000, 8)

        Uses rfind('_') to split size off the right so that addresses
        with many hex digits parse cleanly regardless of their magnitude.
        """
        if not key.startswith('MEM_'):
            return None
        body = key[4:]  # strip 'MEM_'
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

    def _instruction_evaluator(self, ql: Qiling, address: int, size: int) -> None:
        if not self._is_main_binary(address):
            return

        instruction_bytes = bytes(ql.mem.read(address, size))
        circuit = generate_static_rule(self.arch, instruction_bytes, X64_FORMAT)

        policy = ImplicitTaintPolicy.STOP if (self.check_sc or self.check_bof) else ImplicitTaintPolicy.IGNORE

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

            # -----------------------------------------------------------
            # Compute which concrete byte addresses the circuit wrote with
            # nonzero taint. The mem_write hook reads this set to avoid
            # clearing taint that the circuit intentionally set.
            # Must be done BEFORE updating shadow memory (the hook fires
            # after Unicorn executes the instruction, which is after we
            # return from hook_code — so by then shadow is already updated
            # and _last_tainted_writes is what the hook sees).
            # -----------------------------------------------------------
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

            # -----------------------------------------------------------
            # Propagate outputs back into live taint state.
            # Registers: clear then selectively re-set tainted ones.
            # Memory: ALWAYS write — val=0 clears stale taint; val>0 sets it.
            # -----------------------------------------------------------
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

            # -----------------------------------------------------------
            # AIW: Arbitrary Indexed Write detection.
            # Use ctx.input_taint (pre-instruction snapshot, already
            # normalized by EvalContext.__init__) because register_taint
            # has been cleared and repopulated above.
            # -----------------------------------------------------------
            if self.check_aiw and ctx.input_taint:
                live_regs = ctx.input_values  # concrete values at insn start

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
                        # Allow ±4096 to cover [reg + small_disp] modes
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
                except Exception:
                    pass
                self.reporter.side_channel(address, instruction=asm_str, taint_mask=taint_mask)
                ql.emu_stop()
            else:
                ql.emu_stop()
