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
    UC_ERR_READ_UNMAPPED,
    UC_ERR_WRITE_UNMAPPED,
    UC_HOOK_MEM_UNMAPPED,
    UC_MEM_FETCH_UNMAPPED,
    UC_MODE_32,
    UC_MODE_64,
    UC_MODE_ARM,
)

from microtaint.instrumentation.cell import PCodeCellEvaluator
from microtaint.types import Architecture

logger = logging.getLogger(__name__)


# Lazy import so that simulator.py has no hard dependency on the pcode
# sub-system when use_unicorn=True (the default, unchanged path).
def _get_pcode_evaluator_class() -> type[PCodeCellEvaluator]:

    return PCodeCellEvaluator


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

    def __init__(self, arch: Architecture, use_unicorn: bool = False) -> None:  # noqa: C901
        if arch not in _ARCH_MAP:
            raise ValueError(f'Architecture {arch} is not supported by CellSimulator.')
        self.arch = arch
        self.use_unicorn = use_unicorn
        self._pcode: None | PCodeCellEvaluator = None
        self._pcode_fallback_exc: Any = None

        # Initialise Unicorn FIRST.  pypcode (called inside _get_pcode_evaluator_class)
        # invokes GHIDRA's native runtime which can interfere with Unicorn's allocator
        # if called before uc_open().  Always bring Unicorn up first.
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

        # --- Establish a valid Stack Base ---
        # Prevents UC_ERR_MAP underflows on implicit stack instructions (PUSH/POP/CALL/RET)
        stack_base = 0x80000000
        if self.arch == Architecture.X86:
            self.uc.reg_write(uc_x86_const.UC_X86_REG_ESP, stack_base)  # pyright: ignore[reportUnknownMemberType]
        elif self.arch == Architecture.AMD64:
            self.uc.reg_write(uc_x86_const.UC_X86_REG_RSP, stack_base)  # pyright: ignore[reportUnknownMemberType]
        elif self.arch == Architecture.ARM64:
            self.uc.reg_write(uc_arm64_const.UC_ARM64_REG_SP, stack_base)  # pyright: ignore[reportUnknownMemberType]

        self._pristine_context = self.uc.context_save()

        # P-code native evaluator — created after Unicorn is fully initialised.
        # pypcode's GHIDRA runtime must not start before uc_open() completes.
        if not use_unicorn:
            self._pcode = _get_pcode_evaluator_class()(arch)
            # Cache the fallback exception class — avoids per-call import in hot path.
            from microtaint.instrumentation.cell import PCodeFallbackNeeded as _exc  # noqa: PLC0415

            self._pcode_fallback_exc = _exc

        # Cache PC register ID — avoids a match statement on every _execute call
        if arch == Architecture.X86:
            self._pc_reg: int = uc_x86_const.UC_X86_REG_EIP
        elif arch == Architecture.AMD64:
            self._pc_reg = uc_x86_const.UC_X86_REG_RIP
        else:
            self._pc_reg = uc_arm64_const.UC_ARM64_REG_PC
        # Bytestring cache — skip redundant mem_write when code hasn't changed
        self._last_bytestring: bytes = b''

        # Pre-compute canonical batch register list for this arch (write path only)
        # Excludes sub-registers and flags — those are handled separately below.
        self._batch_reg_pairs: list[tuple[int, str]] = []
        self._pending_reg_writes: dict[int, int] = {}  # reused across calls
        _batch_names = {
            Architecture.AMD64: [
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
            ],
            Architecture.X86: [
                'EAX',
                'EBX',
                'ECX',
                'EDX',
                'ESI',
                'EDI',
                'EBP',
                'ESP',
                'EIP',
                'EFLAGS',
            ],
            Architecture.ARM64: [
                *[f'X{i}' for i in range(31)],
                'SP',
                'PC',
                'NZCV',
            ],
        }.get(arch, [])
        reg_map = _UC_REGS.get(arch, {})
        for name in _batch_names:
            uc_id = reg_map.get(name)
            if uc_id is not None:
                self._batch_reg_pairs.append((uc_id, name))

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
            # parts[0] = 'MEM'
            # parts[1] = hex address OR register name
            # parts[2] = signed offset (when register) OR size (when hex address)
            # parts[3] = size (when register + offset)
            try:
                # Static hex address: MEM_0x7fff1000_8
                addr = int(parts[1], 16)
                size = int(parts[2]) if len(parts) > 2 else 8
            except ValueError:
                # Dynamic register: MEM_RBP_8  or  MEM_RBP_-8_8
                base_reg = parts[1]
                addr = self._read_reg(base_reg)
                if len(parts) > 2:
                    try:
                        # parts[2] is a signed offset, parts[3] (optional) is size
                        offset = int(parts[2])
                        addr = addr + offset
                        size = int(parts[3]) if len(parts) > 3 else 8
                    except ValueError:
                        # parts[2] is size (old format: MEM_RBP_8)
                        size = int(parts[2])
                else:
                    size = 8
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
        Single concrete execution over a MachineState.
        Used by evaluate_concrete. For the differential, use evaluate_cell_differential
        which uses context_save/restore to avoid double setup overhead.
        """
        self.clear_memory_and_registers()
        self.setup_registers_and_memory(state, mem_sizes)
        self.uc.reg_write(self._pc_reg, self.CODE_ADDR)  # pyright: ignore[reportUnknownMemberType]
        if bytestring != self._last_bytestring:
            self.uc.mem_write(self.CODE_ADDR, bytestring)
            self._last_bytestring = bytestring

        try:
            self.uc.emu_start(self.CODE_ADDR, self.CODE_ADDR + len(bytestring))
        except uc_py3.UcError as e:
            # Safe to ignore memory errors that occur when the differential
            # evaluator runs instructions with extreme register values
            # (e.g. V | T where T = 0xFFFF... makes RBP/RSP huge).
            # These errors mean "the concrete value was out of range" — the
            # differential XOR treats the run as producing 0, which is a
            # conservative under-taint for that specific extreme input.
            # All four error codes represent legitimate "address not mapped"
            # conditions that can occur with large synthetic register values:
            #   UC_ERR_FETCH_UNMAPPED : branch/ret to unmapped code address
            #   UC_ERR_READ_UNMAPPED  : load from unmapped address
            #   UC_ERR_WRITE_UNMAPPED : store to unmapped address
            #   UC_ERR_MAP            : mem_map call failed (address space full
            #                          or invalid range from huge register value)
            if e.errno in (UC_ERR_FETCH_UNMAPPED, UC_ERR_MAP, UC_ERR_READ_UNMAPPED, UC_ERR_WRITE_UNMAPPED):
                pass
            else:
                raise

    def setup_registers_and_memory(self, state: MachineState, mem_sizes: dict[int, int] | None) -> None:  # noqa: C901
        for reg_name, val in state.regs.items():

            # Prevent Stack Underflow
            if reg_name in ('RSP', 'ESP', 'SP') and val == 0:
                val = 0x80000000  # noqa: PLW2901

            if reg_name.startswith('MEM_'):
                logger.warning(
                    f'Legacy MEM register format detected: {reg_name}. Consider using tuple format for clarity.',
                )
                parts = reg_name.split('_')

                try:
                    addr = int(parts[1], 16)
                except ValueError:
                    # Dynamically resolve the pointer address from the current state
                    addr = state.regs.get(parts[1], 0)
                    if addr == 0 and parts[1] in ('RSP', 'ESP', 'SP'):
                        addr = 0x80000000  # Fallback to our safe stack base

                size = int(parts[2]) if len(parts) > 2 else 8

                # Map the starting page
                page_addr = addr & ~0xFFF
                if page_addr not in self._mapped_pages:
                    self.uc.mem_map(page_addr, 4096)
                    self._mapped_pages.add(page_addr)

                # Map the ending page in case of a cross-boundary write
                end_page_addr = (addr + size - 1) & ~0xFFF
                if end_page_addr != page_addr and end_page_addr not in self._mapped_pages:
                    self.uc.mem_map(end_page_addr, 4096)
                    self._mapped_pages.add(end_page_addr)

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
                self._pending_reg_writes[uc_reg] = val

        # Flush pending register writes.
        # Use batch only when there are enough registers to amortise the overhead
        # of list construction + ctypes dispatch. Below ~6 registers individual
        # reg_write calls are faster (benchmark-verified).
        if self._pending_reg_writes:
            if len(self._pending_reg_writes) >= 6:
                try:
                    self.uc.reg_write_batch(
                        list(self._pending_reg_writes.items()),
                    )  # pyright: ignore[reportUnknownMemberType]
                except AttributeError:
                    for uc_id, v in self._pending_reg_writes.items():
                        self.uc.reg_write(uc_id, v)  # pyright: ignore[reportUnknownMemberType]
            else:
                for uc_id, v in self._pending_reg_writes.items():
                    self.uc.reg_write(uc_id, v)  # pyright: ignore[reportUnknownMemberType]
            self._pending_reg_writes.clear()

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

            # Map the starting page
            page_addr = addr & ~0xFFF
            if page_addr not in self._mapped_pages:
                self.uc.mem_map(page_addr, 4096)
                self._mapped_pages.add(page_addr)

            # Map the ending page in case of a cross-boundary write
            end_page_addr = (addr + size - 1) & ~0xFFF
            if end_page_addr != page_addr and end_page_addr not in self._mapped_pages:
                self.uc.mem_map(end_page_addr, 4096)
                self._mapped_pages.add(end_page_addr)

            try:
                self.uc.mem_write(addr, mem_val.to_bytes(size, 'little'))
            except ValueError:
                mask = (1 << (size * 8)) - 1
                self.uc.mem_write(addr, (mem_val & mask).to_bytes(size, 'little'))

            self._dirtied_memory.add(addr)

    def evaluate_concrete(self, cell: Any, v_state: MachineState) -> int:
        # --- P-code native path (use_unicorn=False) ---
        # Pass MachineState dicts directly to the pcode evaluator —
        # no flat-dict copy, no 'MEM_<hex>_<size>' key construction per call.
        if not self.use_unicorn:
            assert self._pcode is not None
            try:
                return self._pcode.evaluate_concrete_state(cell, v_state.regs, v_state.mem)
            except self._pcode_fallback_exc:
                self._pcode.fallback_calls += 1
                # Fall through to Unicorn below

        # --- Unicorn path (use_unicorn=True, or pcode fallback) ---
        try:
            self._execute(bytes.fromhex(cell.instruction), v_state)
        except Exception as e:
            logger.error(f'[!] Microtaint Unicorn crash on instruction (hex): {cell.instruction}. Exception: {e}')
            raise

        val = self._read_reg(cell.out_reg)
        mask = (1 << (cell.out_bit_end - cell.out_bit_start + 1)) - 1
        return int((val >> cell.out_bit_start) & mask)

    def evaluate_cell_differential(  # noqa: C901
        self,
        bytestring: bytes,
        target_reg: str | tuple[str, int, int],
        v_state: MachineState,
        t_state: MachineState,
    ) -> int:
        # 1. Resolve target format
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

        # 2. Build OR / AND states
        state_or = MachineState()
        state_and = MachineState()
        mem_sizes: dict[int, int] = {}

        all_regs = set(v_state.regs.keys()) | set(t_state.regs.keys())
        for reg in all_regs:
            v = v_state.regs.get(reg, 0)
            t = t_state.regs.get(reg, 0)
            state_or.regs[reg] = v | t
            state_and.regs[reg] = v & ~t

        all_addrs = set(v_state.mem.keys()) | set(t_state.mem.keys())
        for addr in all_addrs:
            v_val = v_state.mem.get(addr, 0)
            t_val = t_state.mem.get(addr, 0)
            state_or.mem[addr] = v_val | t_val
            state_and.mem[addr] = v_val & ~t_val
            mem_sizes[addr] = 8 if v_val == 0 else max(8, (v_val.bit_length() + 7) // 8)

        # 3. Write code bytes once (cached — skip if unchanged)
        if bytestring != self._last_bytestring:
            self.uc.mem_write(self.CODE_ADDR, bytestring)
            self._last_bytestring = bytestring

        # 4. Run OR: full setup then emu_start.
        # context_save/restore only snapshots CPU registers, NOT memory.
        # For instructions with memory operands both runs need fresh memory writes,
        # so we cannot use context_restore to skip memory setup.
        # We DO use context_save to avoid re-running setup_registers_and_memory
        # for the register portion of the AND run when there is no memory state.
        has_memory = bool(state_or.mem)

        self.clear_memory_and_registers()
        self.setup_registers_and_memory(state_or, mem_sizes)
        self.uc.reg_write(self._pc_reg, self.CODE_ADDR)  # pyright: ignore[reportUnknownMemberType]

        try:
            self.uc.emu_start(self.CODE_ADDR, self.CODE_ADDR + len(bytestring))
        except uc_py3.UcError as e:
            if e.errno not in (UC_ERR_FETCH_UNMAPPED, UC_ERR_MAP, UC_ERR_READ_UNMAPPED, UC_ERR_WRITE_UNMAPPED):
                raise
        res_or = self._read_mem(mem_addr, mem_size) if is_mem_target else self._read_reg(target_reg_str)

        # 5. Run AND.
        if has_memory:
            # Memory present: must do full setup to write correct memory values
            self.clear_memory_and_registers()
            self.setup_registers_and_memory(state_and, mem_sizes)
            self.uc.reg_write(self._pc_reg, self.CODE_ADDR)  # pyright: ignore[reportUnknownMemberType]
        else:
            # No memory: save context after OR setup, build AND register state,
            # save that context too — then restore each before emu_start.
            # This avoids running setup_registers_and_memory a second time from scratch.
            self.uc.context_restore(self._pristine_context)
            self.setup_registers_and_memory(state_and, mem_sizes)
            self.uc.reg_write(self._pc_reg, self.CODE_ADDR)  # pyright: ignore[reportUnknownMemberType]

        try:
            self.uc.emu_start(self.CODE_ADDR, self.CODE_ADDR + len(bytestring))
        except uc_py3.UcError as e:
            if e.errno not in (UC_ERR_FETCH_UNMAPPED, UC_ERR_MAP, UC_ERR_READ_UNMAPPED, UC_ERR_WRITE_UNMAPPED):
                raise
        res_and = self._read_mem(mem_addr, mem_size) if is_mem_target else self._read_reg(target_reg_str)

        raw_diff = res_or ^ res_and

        # 8. Apply bit slice if requested
        if is_reg_slice and isinstance(target_reg, tuple):
            bit_start = target_reg[1]
            bit_end = target_reg[2]
            mask = (1 << (bit_end - bit_start + 1)) - 1
            return (raw_diff >> bit_start) & mask

        return raw_diff
