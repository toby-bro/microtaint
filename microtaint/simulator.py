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

_UC_REGS: dict[Architecture, dict[str, int]] = {
    Architecture.X86: {
        'EAX': uc_x86_const.UC_X86_REG_EAX,
        'AX': uc_x86_const.UC_X86_REG_AX,
        'AL': uc_x86_const.UC_X86_REG_AL,
        'AH': uc_x86_const.UC_X86_REG_AH,
        'EBX': uc_x86_const.UC_X86_REG_EBX,
        'BX': uc_x86_const.UC_X86_REG_BX,
        'BL': uc_x86_const.UC_X86_REG_BL,
        'BH': uc_x86_const.UC_X86_REG_BH,
        'ECX': uc_x86_const.UC_X86_REG_ECX,
        'CX': uc_x86_const.UC_X86_REG_CX,
        'CL': uc_x86_const.UC_X86_REG_CL,
        'CH': uc_x86_const.UC_X86_REG_CH,
        'EDX': uc_x86_const.UC_X86_REG_EDX,
        'DX': uc_x86_const.UC_X86_REG_DX,
        'DL': uc_x86_const.UC_X86_REG_DL,
        'DH': uc_x86_const.UC_X86_REG_DH,
        'ESI': uc_x86_const.UC_X86_REG_ESI,
        'SI': uc_x86_const.UC_X86_REG_SI,
        'EDI': uc_x86_const.UC_X86_REG_EDI,
        'DI': uc_x86_const.UC_X86_REG_DI,
        'EBP': uc_x86_const.UC_X86_REG_EBP,
        'BP': uc_x86_const.UC_X86_REG_BP,
        'ESP': uc_x86_const.UC_X86_REG_ESP,
        'SP': uc_x86_const.UC_X86_REG_SP,
        'EIP': uc_x86_const.UC_X86_REG_EIP,
        'EFLAGS': uc_x86_const.UC_X86_REG_EFLAGS,
    },
    Architecture.AMD64: {
        'RAX': uc_x86_const.UC_X86_REG_RAX,
        'EAX': uc_x86_const.UC_X86_REG_EAX,
        'AX': uc_x86_const.UC_X86_REG_AX,
        'AL': uc_x86_const.UC_X86_REG_AL,
        'AH': uc_x86_const.UC_X86_REG_AH,
        'RBX': uc_x86_const.UC_X86_REG_RBX,
        'EBX': uc_x86_const.UC_X86_REG_EBX,
        'BX': uc_x86_const.UC_X86_REG_BX,
        'BL': uc_x86_const.UC_X86_REG_BL,
        'BH': uc_x86_const.UC_X86_REG_BH,
        'RCX': uc_x86_const.UC_X86_REG_RCX,
        'ECX': uc_x86_const.UC_X86_REG_ECX,
        'CX': uc_x86_const.UC_X86_REG_CX,
        'CL': uc_x86_const.UC_X86_REG_CL,
        'CH': uc_x86_const.UC_X86_REG_CH,
        'RDX': uc_x86_const.UC_X86_REG_RDX,
        'EDX': uc_x86_const.UC_X86_REG_EDX,
        'DX': uc_x86_const.UC_X86_REG_DX,
        'DL': uc_x86_const.UC_X86_REG_DL,
        'DH': uc_x86_const.UC_X86_REG_DH,
        'RSI': uc_x86_const.UC_X86_REG_RSI,
        'ESI': uc_x86_const.UC_X86_REG_ESI,
        'SI': uc_x86_const.UC_X86_REG_SI,
        'SIL': uc_x86_const.UC_X86_REG_SIL,
        'RDI': uc_x86_const.UC_X86_REG_RDI,
        'EDI': uc_x86_const.UC_X86_REG_EDI,
        'DI': uc_x86_const.UC_X86_REG_DI,
        'DIL': uc_x86_const.UC_X86_REG_DIL,
        'RBP': uc_x86_const.UC_X86_REG_RBP,
        'EBP': uc_x86_const.UC_X86_REG_EBP,
        'BP': uc_x86_const.UC_X86_REG_BP,
        'BPL': uc_x86_const.UC_X86_REG_BPL,
        'RSP': uc_x86_const.UC_X86_REG_RSP,
        'ESP': uc_x86_const.UC_X86_REG_ESP,
        'SP': uc_x86_const.UC_X86_REG_SP,
        'SPL': uc_x86_const.UC_X86_REG_SPL,
        'RIP': uc_x86_const.UC_X86_REG_RIP,
        'R8': uc_x86_const.UC_X86_REG_R8,
        'R8D': uc_x86_const.UC_X86_REG_R8D,
        'R8W': uc_x86_const.UC_X86_REG_R8W,
        'R8B': uc_x86_const.UC_X86_REG_R8B,
        'R9': uc_x86_const.UC_X86_REG_R9,
        'R9D': uc_x86_const.UC_X86_REG_R9D,
        'R9W': uc_x86_const.UC_X86_REG_R9W,
        'R9B': uc_x86_const.UC_X86_REG_R9B,
        'R10': uc_x86_const.UC_X86_REG_R10,
        'R10D': uc_x86_const.UC_X86_REG_R10D,
        'R10W': uc_x86_const.UC_X86_REG_R10W,
        'R10B': uc_x86_const.UC_X86_REG_R10B,
        'R11': uc_x86_const.UC_X86_REG_R11,
        'R11D': uc_x86_const.UC_X86_REG_R11D,
        'R11W': uc_x86_const.UC_X86_REG_R11W,
        'R11B': uc_x86_const.UC_X86_REG_R11B,
        'R12': uc_x86_const.UC_X86_REG_R12,
        'R12D': uc_x86_const.UC_X86_REG_R12D,
        'R12W': uc_x86_const.UC_X86_REG_R12W,
        'R12B': uc_x86_const.UC_X86_REG_R12B,
        'R13': uc_x86_const.UC_X86_REG_R13,
        'R13D': uc_x86_const.UC_X86_REG_R13D,
        'R13W': uc_x86_const.UC_X86_REG_R13W,
        'R13B': uc_x86_const.UC_X86_REG_R13B,
        'R14': uc_x86_const.UC_X86_REG_R14,
        'R14D': uc_x86_const.UC_X86_REG_R14D,
        'R14W': uc_x86_const.UC_X86_REG_R14W,
        'R14B': uc_x86_const.UC_X86_REG_R14B,
        'R15': uc_x86_const.UC_X86_REG_R15,
        'R15D': uc_x86_const.UC_X86_REG_R15D,
        'R15W': uc_x86_const.UC_X86_REG_R15W,
        'R15B': uc_x86_const.UC_X86_REG_R15B,
        'EFLAGS': uc_x86_const.UC_X86_REG_EFLAGS,
    },
    Architecture.ARM64: {
        **{f'X{i}': getattr(uc_arm64_const, f'UC_ARM64_REG_X{i}') for i in range(31)},
        **{f'W{i}': getattr(uc_arm64_const, f'UC_ARM64_REG_W{i}') for i in range(31)},
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

        # Performance optimizations
        self._mapped_pages: set[int] = set()
        self._dirtied_memory: set[int] = set()

        self.CODE_ADDR = 0x400000  # Use high address to avoid conflicts with test data
        self.uc.mem_map(self.CODE_ADDR, 4096)
        self._mapped_pages.add(self.CODE_ADDR)

        self.uc.hook_add(UC_HOOK_MEM_UNMAPPED, self._hook_mem_unmapped)  # pyright: ignore[reportUnknownMemberType]

        # Initialize all CPU registers natively to zero to create the pristine context snapshot
        for reg_name in _UC_REGS.get(self.arch, {}).keys():
            uc_reg = self._get_uc_reg(reg_name)
            if uc_reg is not None:
                self.uc.reg_write(uc_reg, 0)  # pyright: ignore[reportUnknownMemberType]

        self._pristine_context = self.uc.context_save()

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
        if page_addr not in self._mapped_pages:
            uc.mem_map(page_addr, 4096)
            self._mapped_pages.add(page_addr)
            return True
        return False

    def _get_uc_reg(self, reg_name: str) -> int | None:
        mapping = _UC_REGS.get(self.arch, {})
        return mapping.get(reg_name)

    def _read_mem(self, addr: int, size: int) -> int:
        page_addr = addr & ~0xFFF
        if page_addr not in self._mapped_pages:
            self.uc.mem_map(page_addr, 4096)
            self._mapped_pages.add(page_addr)

        mem_data = self.uc.mem_read(addr, size)
        return int.from_bytes(mem_data, 'little')

    def _read_reg(self, reg_name: str) -> int:  # noqa: C901
        if reg_name.startswith('MEM_'):
            parts = reg_name.split('_')
            addr = int(parts[1], 16)
            size = int(parts[2])
            return self._read_mem(addr, size)

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
        if uc_reg is None:
            return 0
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
            # Safe to ignore fetch unmapped occurring post execution branches
            if e.errno == UC_ERR_FETCH_UNMAPPED:
                pass
            else:
                raise

    def setup_registers_and_memory(self, state: MachineState, mem_sizes: dict[int, int] | None) -> None:  # noqa: C901
        for reg_name, val in state.regs.items():
            if reg_name.startswith('MEM_'):
                logger.warning(
                    f'Legacy MEM register format detected: {reg_name}. Consider using tuple format for clarity.',
                )
                parts = reg_name.split('_')
                addr = int(parts[1], 16)
                size = int(parts[2])

                page_addr = addr & ~0xFFF
                if page_addr not in self._mapped_pages:
                    self.uc.mem_map(page_addr, 4096)
                    self._mapped_pages.add(page_addr)

                try:
                    self.uc.mem_write(addr, val.to_bytes(size, 'little'))
                except ValueError:
                    mask = (1 << (size * 8)) - 1
                    self.uc.mem_write(addr, (val & mask).to_bytes(size, 'little'))

                self._dirtied_memory.add(addr)
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
            if uc_reg is not None:
                self.uc.reg_write(uc_reg, val)  # pyright: ignore[reportUnknownMemberType]

        self.load_memory_state(state, mem_sizes)

    def clear_memory_and_registers(self) -> None:
        # Instantly reset all CPU registers natively in C
        self.uc.context_restore(self._pristine_context)

        # Clear ONLY the memory addresses that were written to during the last state setup
        for addr in self._dirtied_memory:
            self.uc.mem_write(addr, b'\x00' * 8)

        self._dirtied_memory.clear()

    def load_memory_state(self, state: MachineState, mem_sizes: dict[int, int] | None) -> None:
        for addr, mem_val in state.mem.items():
            if mem_sizes and addr in mem_sizes:
                size = mem_sizes[addr]
            elif mem_val == 0:
                size = 8  # Default to 8 bytes for zero
            else:
                size = max(8, (mem_val.bit_length() + 7) // 8)

            page_addr = addr & ~0xFFF
            if page_addr not in self._mapped_pages:
                self.uc.mem_map(page_addr, 4096)
                self._mapped_pages.add(page_addr)

            try:
                self.uc.mem_write(addr, mem_val.to_bytes(size, 'little'))
            except ValueError:
                mask = (1 << (size * 8)) - 1
                self.uc.mem_write(addr, (mem_val & mask).to_bytes(size, 'little'))

            self._dirtied_memory.add(addr)

    def evaluate_concrete(self, cell: Any, v_state: MachineState) -> int:
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
        # 1. Resolve target format cleanly
        is_reg_slice = False
        is_mem_target = False
        mem_addr = 0
        mem_size = 0

        if isinstance(target_reg, tuple):
            if target_reg[0] == 'MEM':
                is_mem_target = True
                mem_addr = target_reg[1]
                mem_size = target_reg[2]
                target_reg_str = ''
            else:
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

            if v_val == 0:
                mem_sizes[addr] = 8
            else:
                mem_sizes[addr] = max(8, (v_val.bit_length() + 7) // 8)

        # 5. Execute using the typed MachineState structures directly
        self._execute(bytestring, state_or, mem_sizes)
        res_or = self._read_mem(mem_addr, mem_size) if is_mem_target else self._read_reg(target_reg_str)

        self._execute(bytestring, state_and, mem_sizes)
        res_and = self._read_mem(mem_addr, mem_size) if is_mem_target else self._read_reg(target_reg_str)

        raw_diff = res_or ^ res_and

        # 6. Apply bit slice if it was requested via legacy tuple
        if is_reg_slice and isinstance(target_reg, tuple):
            bit_start = target_reg[1]
            bit_end = target_reg[2]
            mask = (1 << (bit_end - bit_start + 1)) - 1
            return (raw_diff >> bit_start) & mask

        return raw_diff
