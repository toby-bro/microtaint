from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import unicorn.arm64_const as uc_arm64_const
import unicorn.unicorn_py3 as uc_py3
import unicorn.x86_const as uc_x86_const
from unicorn import (
    UC_ARCH_ARM64,
    UC_ARCH_X86,
    UC_ERR_FETCH_UNMAPPED,
    UC_ERR_MAP,
    UC_HOOK_MEM_UNMAPPED,
    UC_MEM_FETCH_UNMAPPED,
    UC_MODE_32,
    UC_MODE_64,
    UC_MODE_ARM,
)

from microtaint.types import Architecture

logger = logging.getLogger(__name__)

_ARCH_MAP = {
    Architecture.X86: (UC_ARCH_X86, UC_MODE_32),
    Architecture.AMD64: (UC_ARCH_X86, UC_MODE_64),
    Architecture.ARM64: (UC_ARCH_ARM64, UC_MODE_ARM),
}

_UC_REGS = {
    Architecture.X86: {
        'EAX': uc_x86_const.UC_X86_REG_EAX,
        'EBX': uc_x86_const.UC_X86_REG_EBX,
        'ECX': uc_x86_const.UC_X86_REG_ECX,
        'EDX': uc_x86_const.UC_X86_REG_EDX,
        'ESI': uc_x86_const.UC_X86_REG_ESI,
        'EDI': uc_x86_const.UC_X86_REG_EDI,
        'EBP': uc_x86_const.UC_X86_REG_EBP,
        'ESP': uc_x86_const.UC_X86_REG_ESP,
        'EIP': uc_x86_const.UC_X86_REG_EIP,
        'EFLAGS': uc_x86_const.UC_X86_REG_EFLAGS,
    },
    Architecture.AMD64: {
        'RAX': uc_x86_const.UC_X86_REG_RAX,
        'RBX': uc_x86_const.UC_X86_REG_RBX,
        'RCX': uc_x86_const.UC_X86_REG_RCX,
        'RDX': uc_x86_const.UC_X86_REG_RDX,
        'RSI': uc_x86_const.UC_X86_REG_RSI,
        'RDI': uc_x86_const.UC_X86_REG_RDI,
        'RBP': uc_x86_const.UC_X86_REG_RBP,
        'RSP': uc_x86_const.UC_X86_REG_RSP,
        'RIP': uc_x86_const.UC_X86_REG_RIP,
        'R8': uc_x86_const.UC_X86_REG_R8,
        'R9': uc_x86_const.UC_X86_REG_R9,
        'R10': uc_x86_const.UC_X86_REG_R10,
        'R11': uc_x86_const.UC_X86_REG_R11,
        'R12': uc_x86_const.UC_X86_REG_R12,
        'R13': uc_x86_const.UC_X86_REG_R13,
        'R14': uc_x86_const.UC_X86_REG_R14,
        'R15': uc_x86_const.UC_X86_REG_R15,
        'EFLAGS': uc_x86_const.UC_X86_REG_EFLAGS,
    },
    Architecture.ARM64: {
        'X0': uc_arm64_const.UC_ARM64_REG_X0,
        'X1': uc_arm64_const.UC_ARM64_REG_X1,
        'X2': uc_arm64_const.UC_ARM64_REG_X2,
        'X3': uc_arm64_const.UC_ARM64_REG_X3,
        'X4': uc_arm64_const.UC_ARM64_REG_X4,
        'X5': uc_arm64_const.UC_ARM64_REG_X5,
        'X6': uc_arm64_const.UC_ARM64_REG_X6,
        'X7': uc_arm64_const.UC_ARM64_REG_X7,
        'X8': uc_arm64_const.UC_ARM64_REG_X8,
        'X29': uc_arm64_const.UC_ARM64_REG_X29,
        'X30': uc_arm64_const.UC_ARM64_REG_X30,
        'SP': uc_arm64_const.UC_ARM64_REG_SP,
        'PC': uc_arm64_const.UC_ARM64_REG_PC,
        'NZCV': uc_arm64_const.UC_ARM64_REG_NZCV,
    },
}


@dataclass
class MachineState:
    regs: dict[str, int] = field(default_factory=dict[str, int])
    mem: dict[int, int] = field(default_factory=dict[int, int])


class CellSimulator:
    """
    Evaluates microtaint InstructionCellExpr natively by evaluating the instruction via Unicorn
    on V | T and V & ~T, computing the precise logical XOR differential.
    """

    def __init__(self, arch: Architecture) -> None:
        if arch not in _ARCH_MAP:
            raise ValueError(f'Architecture {arch} is not supported by CellSimulator.')
        self.arch = arch
        uc_arch, uc_mode = _ARCH_MAP[arch]
        self.uc = uc_py3.Uc(uc_arch, uc_mode)

        self.CODE_ADDR = 0x400000  # Use high address to avoid conflicts with test data
        self.uc.mem_map(self.CODE_ADDR, 4096)
        self.uc.hook_add(UC_HOOK_MEM_UNMAPPED, self._hook_mem_unmapped)  # pyright: ignore[reportUnknownMemberType]
        self.memory_addrs: set[int] = set()  # Track memory addresses used

    def _hook_mem_unmapped(
        self,
        uc: uc_py3.Uc,
        type_: int,
        address: int,
        _size: int,
        _value: int,
        _user_data: Any,
    ) -> bool:
        if type_ == UC_MEM_FETCH_UNMAPPED:
            return False  # Do not map on instruction fetch unmapped

        page_addr = address & ~0xFFF
        # Map a single page, avoiding exceptions if already mapped
        try:
            uc.mem_map(page_addr, 4096)
            return True
        except uc_py3.UcError as e:
            return e.errno == UC_ERR_MAP

    def _get_uc_reg(self, reg_name: str) -> int:
        mapping = _UC_REGS.get(self.arch, {})
        if reg_name not in mapping:
            raise ValueError(f'Register {reg_name} not mapped for {self.arch}')
        return mapping[reg_name]

    def _read_reg(self, reg_name: str) -> int:  # noqa: C901
        if reg_name.startswith('MEM_'):
            parts = reg_name.split('_')
            addr = int(parts[1], 16)
            size = int(parts[2])

            # Ensure memory page is mapped before reading
            page_addr = addr & ~0xFFF
            try:
                self.uc.mem_map(page_addr, 4096)
            except uc_py3.UcError:
                logger.warning(
                    f'Failed to map memory at {page_addr:#x} during register read. Address may already be mapped.',
                )
                # Already mapped

            mem_data = self.uc.mem_read(addr, size)
            return int.from_bytes(mem_data, 'little')

        # X86 / AMD64 EFLAGS extraction
        if self.arch in (Architecture.X86, Architecture.AMD64):
            if reg_name in ('CF', 'PF', 'ZF', 'SF', 'OF'):
                eflags = int(self.uc.reg_read(uc_x86_const.UC_X86_REG_EFLAGS))
                if reg_name == 'CF':
                    return eflags & 1
                if reg_name == 'PF':
                    return (eflags >> 2) & 1
                if reg_name == 'ZF':
                    return (eflags >> 6) & 1
                if reg_name == 'SF':
                    return (eflags >> 7) & 1
                if reg_name == 'OF':
                    return (eflags >> 11) & 1

        # ARM64 NZCV extraction
        if self.arch == Architecture.ARM64:
            if reg_name in ('N', 'Z', 'C', 'V'):
                nzcv = int(self.uc.reg_read(uc_arm64_const.UC_ARM64_REG_NZCV))
                if reg_name == 'V':
                    return (nzcv >> 28) & 1
                if reg_name == 'C':
                    return (nzcv >> 29) & 1
                if reg_name == 'Z':
                    return (nzcv >> 30) & 1
                if reg_name == 'N':
                    return (nzcv >> 31) & 1

        uc_reg = self._get_uc_reg(reg_name)
        return int(self.uc.reg_read(uc_reg))

    def _execute(
        self,
        bytestring: bytes,
        state: MachineState,
        mem_sizes: dict[int, int] | None = None,
    ) -> None:
        """
        Executes the exact instruction over a given concrete MachineState (writes to registers and memory).
        """
        self.clear_memory_and_registers()
        self.setup_registers_and_memory(state, mem_sizes)

        # Set the Program Counter to the address of the code
        match self.arch:
            case Architecture.X86:
                pc_reg = uc_x86_const.UC_X86_REG_EIP
            case Architecture.AMD64:
                pc_reg = uc_x86_const.UC_X86_REG_RIP
            case Architecture.ARM64:
                pc_reg = uc_arm64_const.UC_ARM64_REG_PC
            case _:
                raise ValueError(f'Unsupported architecture: {self.arch}')

        self.uc.reg_write(pc_reg, self.CODE_ADDR)  # pyright: ignore[reportUnknownMemberType]
        self.uc.mem_write(self.CODE_ADDR, bytestring)

        try:
            self.uc.emu_start(self.CODE_ADDR, self.CODE_ADDR + len(bytestring))
        except uc_py3.UcError as e:
            # If a branch (CALL, JMP, RET) occurs, the PC jumps outside our mapped boundary.
            # Unicorn attempts to fetch the next instruction and throws UC_ERR_FETCH_UNMAPPED.
            # The branch successfully executed, so this is safe to ignore.
            if e.errno == UC_ERR_FETCH_UNMAPPED:
                pass
            else:
                raise

    def setup_registers_and_memory(self, state: MachineState, mem_sizes: dict[int, int] | None) -> None:  # noqa: C901
        for reg_name, val in state.regs.items():
            # Legacy fallback for MEM tuples
            if reg_name.startswith('MEM_'):
                logger.warning(
                    f'Legacy MEM register format detected: {reg_name}. Consider using tuple format for clarity.',
                )
                parts = reg_name.split('_')
                addr = int(parts[1], 16)
                size = int(parts[2])

                self.memory_addrs.add(addr)
                page_addr = addr & ~0xFFF
                try:
                    self.uc.mem_map(page_addr, 4096)
                except uc_py3.UcError:
                    logger.warning(
                        f'Failed to map memory at {page_addr:#x} during state setup. Address may already be mapped.',
                    )

                try:
                    self.uc.mem_write(addr, val.to_bytes(size, 'little'))
                except ValueError:
                    mask = (1 << (size * 8)) - 1
                    self.uc.mem_write(addr, (val & mask).to_bytes(size, 'little'))
                continue

            # Flag writing logic for x86/AMD64
            if reg_name in ('CF', 'PF', 'ZF', 'SF', 'OF') and self.arch in (Architecture.X86, Architecture.AMD64):
                eflags_reg = uc_x86_const.UC_X86_REG_EFLAGS
                eflags = int(self.uc.reg_read(eflags_reg))
                if reg_name == 'CF':
                    eflags = (eflags | 1) if val else (eflags & ~1)
                elif reg_name == 'PF':
                    eflags = (eflags | (1 << 2)) if val else (eflags & ~(1 << 2))
                elif reg_name == 'ZF':
                    eflags = (eflags | (1 << 6)) if val else (eflags & ~(1 << 6))
                elif reg_name == 'SF':
                    eflags = (eflags | (1 << 7)) if val else (eflags & ~(1 << 7))
                elif reg_name == 'OF':
                    eflags = (eflags | (1 << 11)) if val else (eflags & ~(1 << 11))
                self.uc.reg_write(eflags_reg, eflags)  # pyright: ignore[reportUnknownMemberType]
                continue

            # Flag writing logic for ARM64
            if reg_name in ('N', 'Z', 'C', 'V') and self.arch == Architecture.ARM64:
                nzcv_reg = uc_arm64_const.UC_ARM64_REG_NZCV
                nzcv = int(self.uc.reg_read(nzcv_reg))
                if reg_name == 'V':
                    nzcv = (nzcv | (1 << 28)) if val else (nzcv & ~(1 << 28))
                elif reg_name == 'C':
                    nzcv = (nzcv | (1 << 29)) if val else (nzcv & ~(1 << 29))
                elif reg_name == 'Z':
                    nzcv = (nzcv | (1 << 30)) if val else (nzcv & ~(1 << 30))
                elif reg_name == 'N':
                    nzcv = (nzcv | (1 << 31)) if val else (nzcv & ~(1 << 31))
                self.uc.reg_write(nzcv_reg, nzcv)  # pyright: ignore[reportUnknownMemberType]
                continue

            uc_reg = self._get_uc_reg(reg_name)
            self.uc.reg_write(uc_reg, val)  # pyright: ignore[reportUnknownMemberType]

        self.load_memory_state(state, mem_sizes)

    def clear_memory_and_registers(self) -> None:
        # Clear all registers to zero first to avoid state leakage
        defined_registers = _UC_REGS.get(self.arch, {}).keys()

        for reg_name in defined_registers:
            try:
                uc_reg = self._get_uc_reg(reg_name)
                self.uc.reg_write(uc_reg, 0)  # pyright: ignore[reportUnknownMemberType]
            except (ValueError, KeyError):
                pass  # Register not available in this architecture

        # Clear known memory addresses to zero
        for addr in self.memory_addrs:
            try:
                self.uc.mem_write(addr, b'\x00' * 8)
            except uc_py3.UcError:
                logger.warning(f'Failed to clear memory at {addr:#x} during state setup. Address may not be mapped.')

    def load_memory_state(self, state: MachineState, mem_sizes: dict[int, int] | None) -> None:
        for addr, mem_val in state.mem.items():
            if mem_sizes and addr in mem_sizes:
                size = mem_sizes[addr]
            elif mem_val == 0:
                size = 8  # Default to 8 bytes for zero
            else:
                # Infer size, but ensure minimum of 8 bytes for large addresses
                size = max(8, (mem_val.bit_length() + 7) // 8)

            self.memory_addrs.add(addr)
            page_addr = addr & ~0xFFF

            try:
                self.uc.mem_map(page_addr, 4096)
            except uc_py3.UcError:
                logger.warning(
                    f'Failed to map memory at {page_addr:#x} during state setup. Address may already be mapped.',
                )

            try:
                self.uc.mem_write(addr, mem_val.to_bytes(size, 'little'))
            except ValueError:
                mask = (1 << (size * 8)) - 1
                self.uc.mem_write(addr, (mem_val & mask).to_bytes(size, 'little'))

    def evaluate_concrete(self, cell: Any, v_state: MachineState) -> int:
        """
        Tests concrete evaluation natively without taint tracking.
        """
        self._execute(bytes.fromhex(cell.instruction), v_state)
        val = self._read_reg(cell.out_reg)
        mask = (1 << (cell.out_bit_end - cell.out_bit_start + 1)) - 1
        return int((val >> cell.out_bit_start) & mask)

    def evaluate_cell_differential(
        self,
        bytestring: bytes,
        target_reg: str | tuple[str, int, int],
        v_state: MachineState,
        t_state: MachineState,
    ) -> int:
        """
        Computes exactly the cell logic: C_instr(V | T) ^ C_instr(V & ~T)
        """
        # 1. Resolve target format cleanly
        is_reg_slice = False
        if isinstance(target_reg, tuple):
            if target_reg[0] == 'MEM':
                target_reg_str = f'MEM_{target_reg[1]:x}_{target_reg[2]}'
            else:
                # Handle register slice tuple safely: e.g., ('CF', 0, 7)
                target_reg_str = target_reg[0]
                is_reg_slice = True
        else:
            target_reg_str = target_reg

        # 2. Setup unified states
        state_or = MachineState()
        state_and = MachineState()
        mem_sizes: dict[int, int] = {}

        # 3. Process Registers
        all_regs = set(v_state.regs.keys()) | set(t_state.regs.keys())
        for reg in all_regs:
            v = v_state.regs.get(reg, 0)
            t = t_state.regs.get(reg, 0)
            state_or.regs[reg] = v | t
            state_and.regs[reg] = v & ~t

        # 4. Process Memory
        all_addrs = set(v_state.mem.keys()) | set(t_state.mem.keys())
        for addr in all_addrs:
            v_val = v_state.mem.get(addr, 0)
            t_val = t_state.mem.get(addr, 0)

            state_or.mem[addr] = v_val | t_val
            state_and.mem[addr] = v_val & ~t_val

            # Determine size from v_val
            if v_val == 0:
                mem_sizes[addr] = 8
            else:
                mem_sizes[addr] = max(8, (v_val.bit_length() + 7) // 8)

        # 5. Execute using the typed MachineState structures directly
        self._execute(bytestring, state_or, mem_sizes)
        res_or = self._read_reg(target_reg_str)

        self._execute(bytestring, state_and, mem_sizes)
        res_and = self._read_reg(target_reg_str)

        raw_diff = res_or ^ res_and

        # 6. Apply bit slice if it was requested via legacy tuple
        if is_reg_slice and isinstance(target_reg, tuple):
            bit_start = target_reg[1]
            bit_end = target_reg[2]
            mask = (1 << (bit_end - bit_start + 1)) - 1
            return (raw_diff >> bit_start) & mask

        return raw_diff
