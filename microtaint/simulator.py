from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import unicorn
import unicorn.arm64_const
import unicorn.x86_const

from microtaint.types import Architecture

_ARCH_MAP = {
    Architecture.X86: (unicorn.UC_ARCH_X86, unicorn.UC_MODE_32),
    Architecture.AMD64: (unicorn.UC_ARCH_X86, unicorn.UC_MODE_64),
    Architecture.ARM64: (unicorn.UC_ARCH_ARM64, unicorn.UC_MODE_ARM),
}

_UC_REGS = {
    Architecture.X86: {
        'EAX': unicorn.x86_const.UC_X86_REG_EAX,
        'EBX': unicorn.x86_const.UC_X86_REG_EBX,
        'ECX': unicorn.x86_const.UC_X86_REG_ECX,
        'EDX': unicorn.x86_const.UC_X86_REG_EDX,
        'ESI': unicorn.x86_const.UC_X86_REG_ESI,
        'EDI': unicorn.x86_const.UC_X86_REG_EDI,
        'EBP': unicorn.x86_const.UC_X86_REG_EBP,
        'ESP': unicorn.x86_const.UC_X86_REG_ESP,
        'EIP': unicorn.x86_const.UC_X86_REG_EIP,
    },
    Architecture.AMD64: {
        'RAX': unicorn.x86_const.UC_X86_REG_RAX,
        'RBX': unicorn.x86_const.UC_X86_REG_RBX,
        'RCX': unicorn.x86_const.UC_X86_REG_RCX,
        'RDX': unicorn.x86_const.UC_X86_REG_RDX,
        'RSI': unicorn.x86_const.UC_X86_REG_RSI,
        'RDI': unicorn.x86_const.UC_X86_REG_RDI,
        'RBP': unicorn.x86_const.UC_X86_REG_RBP,
        'RSP': unicorn.x86_const.UC_X86_REG_RSP,
        'RIP': unicorn.x86_const.UC_X86_REG_RIP,
        'R8': unicorn.x86_const.UC_X86_REG_R8,
        'R9': unicorn.x86_const.UC_X86_REG_R9,
        'R10': unicorn.x86_const.UC_X86_REG_R10,
        'R11': unicorn.x86_const.UC_X86_REG_R11,
        'R12': unicorn.x86_const.UC_X86_REG_R12,
        'R13': unicorn.x86_const.UC_X86_REG_R13,
        'R14': unicorn.x86_const.UC_X86_REG_R14,
        'R15': unicorn.x86_const.UC_X86_REG_R15,
        'EFLAGS': unicorn.x86_const.UC_X86_REG_EFLAGS,
    },
    Architecture.ARM64: {
        'X0': unicorn.arm64_const.UC_ARM64_REG_X0,
        'X1': unicorn.arm64_const.UC_ARM64_REG_X1,
        'X2': unicorn.arm64_const.UC_ARM64_REG_X2,
        'X3': unicorn.arm64_const.UC_ARM64_REG_X3,
        'X4': unicorn.arm64_const.UC_ARM64_REG_X4,
        'X5': unicorn.arm64_const.UC_ARM64_REG_X5,
        'X6': unicorn.arm64_const.UC_ARM64_REG_X6,
        'X7': unicorn.arm64_const.UC_ARM64_REG_X7,
        'X8': unicorn.arm64_const.UC_ARM64_REG_X8,
        'X29': unicorn.arm64_const.UC_ARM64_REG_X29,
        'X30': unicorn.arm64_const.UC_ARM64_REG_X30,
        'SP': unicorn.arm64_const.UC_ARM64_REG_SP,
        'PC': unicorn.arm64_const.UC_ARM64_REG_PC,
        'NZCV': unicorn.arm64_const.UC_ARM64_REG_NZCV,
    },
}


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
        self.uc = unicorn.Uc(uc_arch, uc_mode)  # type: ignore[attr-defined, no-untyped-call]

        self.CODE_ADDR = 0x400000  # Use high address to avoid conflicts with test data
        self.uc.mem_map(self.CODE_ADDR, 4096)  # type: ignore[no-untyped-call]
        self.uc.hook_add(unicorn.UC_HOOK_MEM_UNMAPPED, self._hook_mem_unmapped)
        self.memory_addrs: set[int] = set()  # Track memory addresses used

    def _hook_mem_unmapped(
        self,
        uc: unicorn.Uc,
        type_: int,
        address: int,
        _size: int,
        _value: int,
        _user_data: Any,
    ) -> bool:
        if type_ == unicorn.UC_MEM_FETCH_UNMAPPED:
            return False  # Do not map on instruction fetch unmapped

        page_addr = address & ~0xFFF
        # Map a single page, avoiding exceptions if already mapped
        try:
            uc.mem_map(page_addr, 4096)
            return True
        except unicorn.UcError as e:
            if e.errno == unicorn.UC_ERR_MAP:
                return True
            return False

    def _get_uc_reg(self, reg_name: str) -> int:
        mapping = _UC_REGS.get(self.arch, {})
        if reg_name not in mapping:
            raise ValueError(f'Register {reg_name} not mapped for {self.arch}')
        return mapping[reg_name]

    def _read_reg(self, reg_name: str) -> int:
        if reg_name.startswith('MEM_'):
            parts = reg_name.split('_')
            addr = int(parts[1], 16)
            size = int(parts[2])

            # Ensure memory page is mapped before reading
            page_addr = addr & ~0xFFF
            try:
                self.uc.mem_map(page_addr, 4096)
            except unicorn.UcError:
                pass  # Already mapped

            mem_data = self.uc.mem_read(addr, size)
            return int.from_bytes(mem_data, 'little')
        if self.arch in (Architecture.X86, Architecture.AMD64):
            if reg_name in ('ZF', 'CF', 'SF', 'OF'):
                eflags = int(self.uc.reg_read(unicorn.x86_const.UC_X86_REG_EFLAGS))  # type: ignore[no-untyped-call]
                if reg_name == 'ZF':
                    return (eflags >> 6) & 1
                if reg_name == 'CF':
                    return eflags & 1
                if reg_name == 'SF':
                    return (eflags >> 7) & 1
                if reg_name == 'OF':
                    return (eflags >> 11) & 1
        uc_reg = self._get_uc_reg(reg_name)
        return int(self.uc.reg_read(uc_reg))  # type: ignore[no-untyped-call]

    def _execute(
        self,
        bytestring: bytes,
        state: Mapping[str, int | dict[int, int]],
        mem_sizes: dict[int, int] | None = None,
    ) -> None:
        """
        Executes the exact instruction over a given concrete state (writes to registers and memory).

        mem_sizes: Optional dict mapping addresses to their expected sizes in bytes.
        """
        # Clear all registers to zero first to avoid state leakage
        reg_names = [
            'RAX',
            'RBX',
            'RCX',
            'RDX',
            'RSI',
            'RDI',
            'R8',
            'R9',
            'R10',
            'R11',
            'R12',
            'R13',
            'R14',
            'R15',
            'RSP',
            'RBP',
        ]
        for reg_name in reg_names:
            try:
                uc_reg = self._get_uc_reg(reg_name)
                self.uc.reg_write(uc_reg, 0)  # type: ignore[no-untyped-call]
            except (ValueError, KeyError):
                pass  # Register not available in this architecture

        # Clear known memory addresses to zero
        for addr in self.memory_addrs:
            try:
                self.uc.mem_write(addr, b'\x00' * 8)
            except unicorn.UcError:
                pass  # Memory not mapped or accessible

        # Load state
        for reg_name, val in state.items():
            if reg_name == 'MEM':
                # Handle dict format: 'MEM': {addr: value, ...}
                if isinstance(val, dict):
                    for addr, mem_val in val.items():
                        # Use provided size or infer from the value
                        if mem_sizes and addr in mem_sizes:
                            size = mem_sizes[addr]
                        elif mem_val == 0:
                            size = 8  # Default to 8 bytes for zero
                        else:
                            # Infer size, but ensure minimum of 8 bytes for large addresses
                            size = max(8, (mem_val.bit_length() + 7) // 8)

                        # Track this address for clearing
                        self.memory_addrs.add(addr)

                        # Ensure memory page is mapped before writing
                        page_addr = addr & ~0xFFF
                        try:
                            self.uc.mem_map(page_addr, 4096)
                        except unicorn.UcError:
                            pass  # Already mapped

                        try:
                            self.uc.mem_write(addr, mem_val.to_bytes(size, 'little'))
                        except ValueError:
                            mask = (1 << (size * 8)) - 1
                            self.uc.mem_write(addr, (mem_val & mask).to_bytes(size, 'little'))
                continue
            if reg_name.startswith('MEM_'):
                parts = reg_name.split('_')
                addr = int(parts[1], 16)
                size = int(parts[2])

                # Track this address for clearing
                self.memory_addrs.add(addr)

                # Ensure memory page is mapped
                page_addr = addr & ~0xFFF
                try:
                    self.uc.mem_map(page_addr, 4096)
                except unicorn.UcError:
                    pass  # Already mapped

                try:
                    self.uc.mem_write(addr, val.to_bytes(size, 'little'))  # type: ignore[union-attr]
                except ValueError:
                    # In python 3.12, to_bytes can take negative values if signed=True,
                    # but memory is 2s complement. So we explicitly mask it.
                    mask = (1 << (size * 8)) - 1
                    self.uc.mem_write(addr, (val & mask).to_bytes(size, 'little'))  # type: ignore[union-attr]
                continue
            uc_reg = self._get_uc_reg(reg_name)
            self.uc.reg_write(uc_reg, val)  # type: ignore[no-untyped-call,arg-type]

        # CRITICAL: Set the Program Counter to the address of the code
        pc_reg = (
            unicorn.x86_const.UC_X86_REG_RIP if self.arch == Architecture.AMD64 else unicorn.x86_const.UC_X86_REG_EIP
        )
        self.uc.reg_write(pc_reg, self.CODE_ADDR)  # type: ignore[no-untyped-call]

        self.uc.mem_write(self.CODE_ADDR, bytestring)  # type: ignore[no-untyped-call]
        self.uc.emu_start(self.CODE_ADDR, self.CODE_ADDR + len(bytestring))  # type: ignore[no-untyped-call]

    def evaluate_concrete(self, cell: Any, v_state: Mapping[str, int]) -> int:
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
        v_state: Mapping[str, int | dict[int, int]],
        t_state: Mapping[str, int | dict[int, int]],
    ) -> int:
        """
        Computes exactly the cell logic: C_instr(V | T) ^ C_instr(V & ~T)
        returning the precise integer bitmask of the output taint for target_reg.

        target_reg can be either a register name string or a tuple ('MEM', address, size).
        """
        # Convert tuple format to string format for memory targets
        if isinstance(target_reg, tuple):
            if target_reg[0] == 'MEM':
                target_reg_str = f'MEM_{target_reg[1]:x}_{target_reg[2]}'
            else:
                raise ValueError(f'Unknown tuple target format: {target_reg}')
        else:
            target_reg_str = target_reg

        state_or: dict[str, int | dict[int, int]] = {}
        state_and: dict[str, int | dict[int, int]] = {}
        mem_sizes: dict[int, int] = {}

        # Collect all registers and memory addresses from both states
        all_keys = set(v_state.keys()) | set(t_state.keys())

        for reg_name in all_keys:
            v = v_state.get(reg_name, 0 if reg_name != 'MEM' else {})
            t = t_state.get(reg_name, 0 if reg_name != 'MEM' else {})

            # Handle memory dict format specially
            if reg_name == 'MEM':
                if not isinstance(v, dict):
                    v = {}
                if not isinstance(t, dict):
                    t = {}
                state_or[reg_name] = {}
                state_and[reg_name] = {}
                all_addrs = set(v.keys()) | set(t.keys())
                for addr in all_addrs:
                    v_val = v.get(addr, 0)
                    t_val = t.get(addr, 0)
                    state_or[reg_name][addr] = v_val | t_val  # type: ignore[index]
                    state_and[reg_name][addr] = v_val & ~t_val  # type: ignore[index]
                    # Determine size from v_val (the actual value, not the taint)
                    if v_val == 0:
                        mem_sizes[addr] = 8  # Default
                    else:
                        mem_sizes[addr] = max(8, (v_val.bit_length() + 7) // 8)
            else:
                # Simulated flip high
                state_or[reg_name] = v | t  # type: ignore[assignment,operator]
                # Simulated flip low
                state_and[reg_name] = v & ~t  # type: ignore[assignment,operator]

        self._execute(bytestring, state_or, mem_sizes)
        res_or = self._read_reg(target_reg_str)

        self._execute(bytestring, state_and, mem_sizes)
        res_and = self._read_reg(target_reg_str)

        return res_or ^ res_and
