from __future__ import annotations

import functools
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import unicorn.arm64_const as uc_arm64_const
import unicorn.mips_const as uc_mips_const
import unicorn.ppc_const as uc_ppc_const
import unicorn.riscv_const as uc_riscv_const
import unicorn.sparc_const as uc_sparc_const
import unicorn.unicorn_py3 as uc_py3
import unicorn.x86_const as uc_x86_const
from unicorn import (
    UC_ARCH_ARM64,
    UC_ARCH_MIPS,
    UC_ARCH_PPC,
    UC_ARCH_RISCV,
    UC_ARCH_SPARC,
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
    UC_MODE_BIG_ENDIAN,
    UC_MODE_MIPS64,
    UC_MODE_PPC32,
    UC_MODE_RISCV64,
    UC_MODE_SPARC32,
)

from microtaint.instrumentation.cell import PCodeCellEvaluator, PCodeFallbackNeeded
from microtaint.instrumentation.cell_c.cell_c import PCodeCellEvaluatorC
from microtaint.types import Architecture

logger = logging.getLogger(__name__)


# Lazy import so that simulator.py has no hard dependency on the pcode
# sub-system when use_unicorn=True (the default, unchanged path).
def _get_pcode_evaluator_class() -> type[PCodeCellEvaluator]:

    return PCodeCellEvaluator


_ARCH_MAP: dict[Architecture, tuple[int, int]] = {
    Architecture.X86: (UC_ARCH_X86, UC_MODE_32),
    Architecture.AMD64: (UC_ARCH_X86, UC_MODE_64),
    Architecture.ARM64: (UC_ARCH_ARM64, UC_MODE_ARM),
    Architecture.RISCV64: (UC_ARCH_RISCV, UC_MODE_RISCV64),
    Architecture.MIPS64BE: (UC_ARCH_MIPS, UC_MODE_MIPS64 | UC_MODE_BIG_ENDIAN),
    Architecture.PPC32BE: (UC_ARCH_PPC, UC_MODE_PPC32 | UC_MODE_BIG_ENDIAN),
    Architecture.SPARC32BE: (UC_ARCH_SPARC, UC_MODE_SPARC32 | UC_MODE_BIG_ENDIAN),
}

# SPARC register-window file, in SLEIGH's naming: 8 global + 8 out + 8 local +
# 8 in.  Within a single window these are plain GPRs; save/restore rotates the
# o/l/i banks, which affects sequences, not the data semantics of one
# instruction.  %g0 reads as zero.
_SPARC_GPR: tuple[str, ...] = tuple(
    [f'g{i}' for i in range(8)]
    + [f'o{i}' for i in range(8)]
    + [f'l{i}' for i in range(8)]
    + [f'i{i}' for i in range(8)],
)

# MIPS o32/n64 ABI register names, in SLEIGH's ordering ($0..$31).  SLEIGH's MIPS
# model names them exactly like this, so the state names match the lifter 1:1.
_MIPS_GPR: tuple[str, ...] = (
    'zero', 'at', 'v0', 'v1', 'a0', 'a1', 'a2', 'a3',
    't0', 't1', 't2', 't3', 't4', 't5', 't6', 't7',
    's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7',
    't8', 't9', 'k0', 'k1', 'gp', 'sp', 's8', 'ra',
)

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
    # RISC-V 64-bit (RV64GC).  Both numeric (X0..X31) and ABI aliases
    # (ra/sp/gp/tp/t0-t6/s0-s11/a0-a7) point to the same Unicorn register
    # IDs, so test corpora may use either form.  PC is also exposed.
    Architecture.RISCV64: {
        **{f'X{i}': getattr(uc_riscv_const, f'UC_RISCV_REG_X{i}') for i in range(32)},
        'PC': uc_riscv_const.UC_RISCV_REG_PC,
        # ABI aliases — same uc IDs as their X-numbered counterparts
        'ZERO': uc_riscv_const.UC_RISCV_REG_X0,
        'RA': uc_riscv_const.UC_RISCV_REG_X1,
        'SP': uc_riscv_const.UC_RISCV_REG_X2,
        'GP': uc_riscv_const.UC_RISCV_REG_X3,
        'TP': uc_riscv_const.UC_RISCV_REG_X4,
        'T0': uc_riscv_const.UC_RISCV_REG_X5,
        'T1': uc_riscv_const.UC_RISCV_REG_X6,
        'T2': uc_riscv_const.UC_RISCV_REG_X7,
        'S0': uc_riscv_const.UC_RISCV_REG_X8,
        'FP': uc_riscv_const.UC_RISCV_REG_X8,  # frame pointer alias of s0
        'S1': uc_riscv_const.UC_RISCV_REG_X9,
        'A0': uc_riscv_const.UC_RISCV_REG_X10,
        'A1': uc_riscv_const.UC_RISCV_REG_X11,
        'A2': uc_riscv_const.UC_RISCV_REG_X12,
        'A3': uc_riscv_const.UC_RISCV_REG_X13,
        'A4': uc_riscv_const.UC_RISCV_REG_X14,
        'A5': uc_riscv_const.UC_RISCV_REG_X15,
        'A6': uc_riscv_const.UC_RISCV_REG_X16,
        'A7': uc_riscv_const.UC_RISCV_REG_X17,
        'S2': uc_riscv_const.UC_RISCV_REG_X18,
        'S3': uc_riscv_const.UC_RISCV_REG_X19,
        'S4': uc_riscv_const.UC_RISCV_REG_X20,
        'S5': uc_riscv_const.UC_RISCV_REG_X21,
        'S6': uc_riscv_const.UC_RISCV_REG_X22,
        'S7': uc_riscv_const.UC_RISCV_REG_X23,
        'S8': uc_riscv_const.UC_RISCV_REG_X24,
        'S9': uc_riscv_const.UC_RISCV_REG_X25,
        'S10': uc_riscv_const.UC_RISCV_REG_X26,
        'S11': uc_riscv_const.UC_RISCV_REG_X27,
        'T3': uc_riscv_const.UC_RISCV_REG_X28,
        'T4': uc_riscv_const.UC_RISCV_REG_X29,
        'T5': uc_riscv_const.UC_RISCV_REG_X30,
        'T6': uc_riscv_const.UC_RISCV_REG_X31,
    },
    # MIPS64 big-endian.  Unicorn exposes the GPRs as UC_MIPS_REG_0..31, which are
    # positional; SLEIGH names them by ABI mnemonic, so map mnemonic -> index.
    # No sub-registers and NO condition flags (SLEIGH reports none): compares
    # write GPRs (`slt`), so none of the x86 flag machinery applies here.
    Architecture.MIPS64BE: {
        **{name.upper(): getattr(uc_mips_const, f'UC_MIPS_REG_{i}') for i, name in enumerate(_MIPS_GPR)},
        'PC': uc_mips_const.UC_MIPS_REG_PC,
        'HI': uc_mips_const.UC_MIPS_REG_HI,
        'LO': uc_mips_const.UC_MIPS_REG_LO,
    },
    # PowerPC 32-bit big-endian.  r0..r31 plus the condition/carry registers:
    # unlike x86's flat 1-bit flags, PPC keeps carry/overflow in XER and the
    # condition fields in cr0..cr7.
    Architecture.PPC32BE: {
        **{f'R{i}': getattr(uc_ppc_const, f'UC_PPC_REG_{i}') for i in range(32)},
        'PC': uc_ppc_const.UC_PPC_REG_PC,
    },
    # SPARC 32-bit big-endian.  SLEIGH names the window file g0..g7/o0..o7/
    # l0..l7/i0..i7 and Unicorn exposes UC_SPARC_REG_{G,O,L,I}<n>, so the state
    # names map 1:1.  No sub-register aliasing within a window.
    Architecture.SPARC32BE: {
        **{n.upper(): getattr(uc_sparc_const, f'UC_SPARC_REG_{n.upper()}') for n in _SPARC_GPR},
        'PC': uc_sparc_const.UC_SPARC_REG_PC,
    },
}


@dataclass
@dataclass
class MachineState:
    regs: dict[str, int] = field(default_factory=dict[str, int])
    mem: dict[int, int] = field(default_factory=dict[int, int])


@dataclass(frozen=True, slots=True)
class _OutputCell:
    """Minimal cell for PCodeCellEvaluator.evaluate_concrete to read one instruction
    output register (any width) from a flat input-register dict.  Used by
    CellSimulator._read_reg_concrete."""

    instruction: str
    out_reg: str
    out_bit_start: int = 0
    out_bit_end: int = 0


@functools.lru_cache(maxsize=8192)
def _native_be_safe(arch: Architecture, instruction_hex: str) -> bool:
    """True iff the native p-code kernel evaluates this instruction correctly on a
    big-endian target.  Two conditions:

    (a) No memory access.  The native register file is byte-offset indexed and BE
        memory layout is not yet wired into routing, so LOAD/STORE stay on Unicorn.

    (b) No two ``unique`` varnodes overlap at DIFFERENT base offsets.  The kernel's
        ``uniq_map`` gives each distinct unique offset its own slot, so a byte-window
        of a wider unique (SPARC ``umul`` reads ``unique[p+4:4]`` of the 8-byte
        product ``unique[p:8]``) reads an *unwritten* slot and returns the wrong
        value -- native under-taints there.  (x86 uses SUBPIECE, a value op, so LE
        never hits this.)

    Given (a)+(b) the kernel is byte-correct on any endianness: whole-register and
    sub-register accesses are handled by ``_PCodeFrame._is_big_endian``, and it
    models the condition/carry varnodes Unicorn PPC hides (XER carry, CR fields) --
    which is what makes carry-chain and ``mfcr`` taint sound.  The one residual
    caveat is a sub-window of a wider register whose out-of-window bits are
    architecturally UNDEFINED (MIPS64 32-bit ops on 64-bit GPRs): on a non-canonical
    value native (defined) and Unicorn (arbitrary) diverge, but that is undefined
    behaviour -- real programs and the canonicalising fuzzer keep such registers
    sign-extended, where native OVER-approximates Unicorn (sound), verified 0
    under-taints on the PPC/MIPS/SPARC corpora."""
    from microtaint.sleigh.lifter import get_context  # noqa: PLC0415

    try:
        ops = get_context(str(arch)).translate(bytes.fromhex(instruction_hex), 0x1000).ops
    except Exception:
        return False
    uniq_ranges: list[tuple[int, int]] = []
    for op in ops:
        if op.opcode.name in ('LOAD', 'STORE'):
            return False
        varnodes = list(op.inputs)
        if op.output is not None:
            varnodes.append(op.output)
        for v in varnodes:
            if v.space.name == 'unique':
                uniq_ranges.append((v.offset, v.offset + v.size))
    for i, (a0, a1) in enumerate(uniq_ranges):
        for b0, b1 in uniq_ranges[i + 1:]:
            if a0 != b0 and a0 < b1 and b0 < a1:  # overlap at different base offsets
                return False
    return True


class CellSimulator:
    """
    Evaluates microtaint InstructionCellExpr natively by evaluating the instruction via Unicorn
    on V | T and V & ~T, computing the precise logical XOR differential.
    """

    def __init__(self, arch: Architecture, use_unicorn: bool = False, use_c: bool | None = None) -> None:  # noqa: C901
        if arch not in _ARCH_MAP:
            raise ValueError(f'Architecture {arch} is not supported by CellSimulator.')
        self.arch = arch
        # The native p-code cell evaluators (C and Cython) index the register file
        # by BYTE OFFSET and place byte d of a register at bit d*8 -- correct on
        # little-endian for any register width, but WRONG on big-endian, where byte
        # d of an N-byte register is at bit (N-d-size)*8.  On a BE target they
        # silently read the wrong bytes and the differential collapses to 0
        # (see KNOWN_ISSUES.md).  Until the register model is made width-aware
        # (option B), evaluate BE architectures through Unicorn, which models the
        # full CPU and gets byte order right.
        self._is_big_endian = str(arch).upper().endswith('BE')
        # Big-endian targets now use the NATIVE evaluator, not Unicorn.
        #
        # The comment this replaces claimed the native evaluators "silently read the
        # wrong bytes and the differential collapses to 0" on BE.  That is no longer
        # true of the Cython frame, which threads `_is_big_endian` through register
        # reads, sub-register overlays and both memory paths.  Measured against
        # Unicorn over the BE corpora, the native frame is not merely equal but
        # STRICTLY BETTER on memory: Unicorn returns 0 for a store/load round trip
        # because the address is not mapped in the engine's instance, while the
        # native frame round-trips it exactly.
        #
        #   sw $4,-16($sp); lw $2,-16($sp)   unicorn 0x0   native 0x55667788
        #   sd $4,-16($sp); ld $2,-16($sp)   unicorn 0x0   native 0x1122334455667788
        #
        # That is what made BE memory chains under-taint: MIPS64BE 118 -> 1 and
        # PPC32BE 1 -> 0, at equal or better wall-clock (MIPS 11s -> 10s).
        # MICROTAINT_BE_UNICORN=1 restores the old behaviour.
        if self._is_big_endian and os.environ.get('MICROTAINT_BE_UNICORN') == '1':
            use_unicorn = True
        # Both native evaluators are byte-order aware now (the C kernel's Frame
        # carries `is_big_endian`, mirroring _PCodeFrame), so BE uses the same fast
        # C kernel as LE.  MICROTAINT_DISABLE_C_KERNEL=1 still forces Cython.
        _force_cython_be = False
        self.use_unicorn = use_unicorn
        # Default: use the C kernel when available.  Pass use_c=False to
        # force the Cython evaluator (e.g. for differential testing).  Set
        # MICROTAINT_DISABLE_C_KERNEL=1 in the environment to disable
        # globally without code changes.
        if use_c is None:
            use_c = os.environ.get('MICROTAINT_DISABLE_C_KERNEL') != '1'
        if _force_cython_be:  # retained as an explicit escape hatch
            use_c = False
        self.use_c = use_c
        self._pcode: None | PCodeCellEvaluator | PCodeCellEvaluatorC = None
        self._pcode_fallback_exc: Any = None
        # Native Cython evaluator used on a BIG-ENDIAN target to evaluate
        # register-only instructions, which `_pcode` is disabled for (the native
        # register file is byte-offset indexed and therefore wrong for BE memory
        # and sub-register aliasing -- see the endianness note above).  A
        # register-only instruction reads and writes whole registers as integer
        # values, so the byte layout of the register file is unobservable and the
        # result round-trips correctly regardless of endianness.  The native
        # kernel is also STRICTLY more capable than Unicorn here: it models the
        # PPC XER carry/overflow bits as first-class varnodes, which Unicorn PPC
        # neither seeds nor exposes, so carry/borrow-chain taint (addc;adde,
        # subfc;subfe) is only sound through this path.  Created lazily via
        # `_native_be`; see `_read_reg_concrete` and `evaluate_concrete`.
        self._be_native: PCodeCellEvaluator | None = None

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
        elif self.arch == Architecture.RISCV64:
            self.uc.reg_write(uc_riscv_const.UC_RISCV_REG_X2, stack_base)  # pyright: ignore[reportUnknownMemberType]

        self._pristine_context = self.uc.context_save()

        # P-code native evaluator — created after Unicorn is fully initialised.
        # pypcode's GHIDRA runtime must not start before uc_open() completes.
        if not use_unicorn:
            if use_c:
                # Pure-C evaluator: drop-in replacement for PCodeCellEvaluator.
                self._pcode = PCodeCellEvaluatorC(arch)
            else:
                self._pcode = _get_pcode_evaluator_class()(arch)
            # Cache the fallback exception class — avoids per-call import in hot path.
            self._pcode_fallback_exc = PCodeFallbackNeeded

        # Cache PC register ID — avoids a match statement on every _execute call
        if arch == Architecture.X86:
            self._pc_reg: int = uc_x86_const.UC_X86_REG_EIP
        elif arch == Architecture.AMD64:
            self._pc_reg = uc_x86_const.UC_X86_REG_RIP
        elif arch == Architecture.RISCV64:
            self._pc_reg = uc_riscv_const.UC_RISCV_REG_PC
        elif arch == Architecture.MIPS64BE:
            self._pc_reg = uc_mips_const.UC_MIPS_REG_PC
        elif arch == Architecture.PPC32BE:
            self._pc_reg = uc_ppc_const.UC_PPC_REG_PC
        elif arch == Architecture.SPARC32BE:
            self._pc_reg = uc_sparc_const.UC_SPARC_REG_PC
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
            Architecture.RISCV64: [
                *[f'X{i}' for i in range(32)],
                'PC',
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

    @staticmethod
    def _xmm_uc_id(reg_name: str) -> int | None:
        """Return the UC_X86_REG_XMM<n> id for an XMM<n>_LO / XMM<n>_HI
        state-format name, or None if the name doesn't fit that pattern.

        XMM register tracking splits each 128-bit XMM register into two
        64-bit halves at the state-format level (XMM<n>_LO = bits 0-63,
        XMM<n>_HI = bits 64-127).  Unicorn doesn't have separate ids for
        the halves, so we use the full XMM<n> id and split/combine the
        128-bit value at read/write time.
        """
        if not reg_name.startswith('XMM'):
            return None
        try:
            rest = reg_name[3:]
            num_str, _, half = rest.partition('_')
            n = int(num_str)
        except ValueError:
            return None
        if not (0 <= n < 16) or half not in ('LO', 'HI'):
            return None
        return int(getattr(uc_x86_const, f'UC_X86_REG_XMM{n}'))

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

        # X86 / AMD64 XMM split-half access.  XMM<n>_LO returns the low
        # 64 bits of XMM<n>; XMM<n>_HI returns the high 64 bits.  This
        # mirrors the wrapper.X64_FORMAT layout and lets the differential
        # path round-trip XMM-targeting instructions through Unicorn.
        if self.arch in (Architecture.X86, Architecture.AMD64) and reg_name.startswith('XMM'):
            xmm_id = self._xmm_uc_id(reg_name)
            if xmm_id is not None:
                full = int(self.uc.reg_read(xmm_id))  # 128-bit int
                if reg_name.endswith('_LO'):
                    return full & 0xFFFFFFFFFFFFFFFF
                if reg_name.endswith('_HI'):
                    return (full >> 64) & 0xFFFFFFFFFFFFFFFF

        uc_reg = self._get_uc_reg(reg_name)
        if uc_reg is None:
            return 0
        return int(self.uc.reg_read(uc_reg))

    def _native_be(self) -> PCodeCellEvaluator:
        """Lazily-created native Cython evaluator for big-endian register-only
        instructions (see the `_be_native` field comment)."""
        if self._be_native is None:
            self._be_native = PCodeCellEvaluator(self.arch)
        return self._be_native

    def _use_native_be(self, cell: Any) -> bool:
        """True iff `cell`'s instruction should be evaluated by the native kernel
        on this (big-endian) target: only native-BE-safe instructions qualify, and
        only when Unicorn is the active concrete engine (the native `_pcode` path,
        when enabled, already handles everything)."""
        return (
            self._is_big_endian
            and self.use_unicorn
            and _native_be_safe(self.arch, cell.instruction)
        )

    def _read_reg_concrete(self, instruction_hex: str, input_regs: dict[str, int], reg_name: str, bits: int) -> int:
        """Concrete post-value of output register `reg_name` (width `bits`),
        computed by the native p-code kernel from `input_regs`.  Used on big-endian
        targets to thread concrete state across a chained sequence when Unicorn
        cannot seed/read the XER carry varnodes; correct for native-BE-safe
        instructions (see `_native_be_safe`).  Raises PCodeFallbackNeeded for
        instructions the native kernel cannot decode, so the caller can fall back."""
        # _OutputCell is a structural stand-in for InstructionCellExpr: the native
        # evaluate_concrete reads only .instruction / .out_reg / .out_bit_start /
        # .out_bit_end, which the dataclass supplies.
        cell = _OutputCell(instruction_hex, reg_name, 0, bits - 1)
        return int(self._native_be().evaluate_concrete(cell, input_regs))  # type: ignore[arg-type]

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
            # CRITICAL: invalidate Unicorn's TCG translation cache for the code
            # region we just rewrote.  Unicorn caches JIT-compiled translations
            # of guest basic blocks keyed by guest physical address.  Writing
            # fresh bytes via mem_write does NOT automatically invalidate the
            # cached translation, so the next emu_start runs the OLD code's
            # translation against the new register state — producing nonsense.
            #
            # This bug manifests dramatically on multi-instruction sequences
            # whose last instruction is a memory load (e.g. `mov rax, [rsp-64]`
            # at the end of a `rep stosb` sequence): the load's translation
            # may have been resolved against a different RSP/operand layout
            # in the previous test, and the new sequence loads garbage.
            # Calling ctl_remove_cache forces a fresh translation pass.
            try:
                self.uc.ctl_remove_cache(self.CODE_ADDR, self.CODE_ADDR + len(bytestring))
            except (AttributeError, uc_py3.UcError):
                # Older Unicorn versions don't have ctl_remove_cache; fall back
                # to unmap+remap which also drops cached translations.
                try:
                    self.uc.mem_unmap(self.CODE_ADDR, 4096)
                    self.uc.mem_map(self.CODE_ADDR, 4096)
                    self.uc.mem_write(self.CODE_ADDR, bytestring)
                except uc_py3.UcError:
                    pass

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

                # Byte order must follow the TARGET, not the host.  Seeding a
                # big-endian target's memory little-endian silently feeds the
                # instruction reversed bytes -- the load reads a different value
                # than intended and the differential is quietly wrong.
                _endian = 'big' if self._is_big_endian else 'little'
                try:
                    self.uc.mem_write(addr, val.to_bytes(size, _endian))
                except ValueError:
                    mask = (1 << (size * 8)) - 1
                    self.uc.mem_write(addr, (val & mask).to_bytes(size, _endian))

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

            # XMM<n>_LO / XMM<n>_HI: combine the two halves into one 128-bit
            # value and write once per XMM<n>.  We track which XMM regs were
            # already-written via _pending_reg_writes (keyed by uc_id) so a
            # later half can OR its contribution into the existing entry
            # rather than clobbering it.  When only one half is in state.regs,
            # the other half reads back as the current Unicorn value (which
            # was zeroed by clear_memory_and_registers at _execute start).
            if self.arch in (Architecture.X86, Architecture.AMD64) and reg_name.startswith('XMM'):
                xmm_id = self._xmm_uc_id(reg_name)
                if xmm_id is not None:
                    is_lo = reg_name.endswith('_LO')
                    half_64 = val & 0xFFFFFFFFFFFFFFFF
                    existing = self._pending_reg_writes.get(xmm_id, 0)
                    if is_lo:
                        # Replace low 64 bits, keep upper as-is.
                        new_val = (existing & ~((1 << 64) - 1)) | half_64
                    else:
                        # Replace high 64 bits, keep lower as-is.
                        new_val = (existing & ((1 << 64) - 1)) | (half_64 << 64)
                    self._pending_reg_writes[xmm_id] = new_val
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
                self.uc.mem_write(addr, mem_val.to_bytes(size, 'big' if self._is_big_endian else 'little'))
            except ValueError:
                mask = (1 << (size * 8)) - 1
                self.uc.mem_write(
                    addr,
                    (mem_val & mask).to_bytes(size, 'big' if self._is_big_endian else 'little'),
                )

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

        # --- Native path for big-endian register-only instructions ---
        # Unicorn cannot seed or read the PPC XER carry/overflow varnodes, so a
        # differential over a carry-consuming instruction (adde reading xer_ca)
        # collapses.  The native kernel models those varnodes and is byte-order
        # agnostic for register-only instructions (see `_be_native`).
        if self._use_native_be(cell):
            try:
                return self._native_be().evaluate_concrete_state(cell, v_state.regs, v_state.mem)
            except PCodeFallbackNeeded:
                pass  # native kernel cannot decode -- fall through to Unicorn

        # --- Unicorn path (use_unicorn=True, or pcode fallback) ---
        try:
            self._execute(bytes.fromhex(cell.instruction), v_state)
        except Exception as e:
            logger.error(f'[!] Microtaint Unicorn crash on instruction (hex): {cell.instruction}. Exception: {e}')
            raise

        val = self._read_reg(cell.out_reg)
        mask = (1 << (cell.out_bit_end - cell.out_bit_start + 1)) - 1
        return int((val >> cell.out_bit_start) & mask)

    def evaluate_differential(self, cell: Any, or_inputs: dict[str, int], and_inputs: dict[str, int]) -> int:
        """
        Evaluate the differential of `cell` given two flat input dicts:
        `or_inputs`  is the high-polarity image  (V | T) of every input,
        `and_inputs` is the low-polarity image   (V & ~T).

        Returns ``cell(or_inputs) XOR cell(and_inputs)`` masked to the
        target's bit slice.  Used by ``MemoryDifferentialExpr`` to drive
        the cell.pyx native ``evaluate_differential`` for instructions
        whose ``MachineState``-based path produces wrong addresses (e.g.
        memory inputs with offsets, address-only registers).

        Both input dicts use cell.pyx's flat key format:
          - register name  -> integer value         (e.g. ``'RAX': 0xFF``)
          - ``MEM_<hex>_<size>``  static address    (e.g. ``'MEM_0x1000_8'``)
          - ``MEM_<reg>_<offset>_<size>`` register-relative
            (e.g. ``'MEM_RBP_-16_8'`` resolves at runtime)

        Native p-code path (``use_unicorn=False``) is preferred; falls
        back to two Unicorn emulations if the instruction needs it.
        """
        # --- P-code native path (fast) ---
        if not self.use_unicorn:
            assert self._pcode is not None
            try:
                return self._pcode.evaluate_differential(cell, or_inputs, and_inputs)
            except self._pcode_fallback_exc:
                self._pcode.fallback_calls += 1
                # fall through to Unicorn

        # --- Unicorn fallback path: build MachineStates from flat dicts ---
        # We can't reuse cell.pyx's `_load` here, so we lift the flat dicts
        # into MachineState by calling our existing _build helpers.  The
        # static-and-dynamic MEM key formats are handled by ast.pyx's
        # _build_machine_state which we delegate to.
        from microtaint.instrumentation.ast import EvalContext, _build_machine_state  # noqa: PLC0415

        # We need a context with input_values to resolve dynamic MEM keys.
        # The address registers are present as bare-register entries in
        # both or_inputs and and_inputs (with the same value, since
        # address-only regs are not polarised), so we use either dict.
        ctx = EvalContext(input_taint={}, input_values=or_inputs, simulator=self)

        v_state_or = _build_machine_state(or_inputs, ctx)
        v_state_and = _build_machine_state(and_inputs, ctx)

        try:
            self._execute(bytes.fromhex(cell.instruction), v_state_or)
        except Exception as e:
            logger.error(f'[!] Microtaint Unicorn crash on instruction (hex): {cell.instruction}. Exception: {e}')
            raise
        val_or = self._read_reg(cell.out_reg)

        try:
            self._execute(bytes.fromhex(cell.instruction), v_state_and)
        except Exception as e:
            logger.error(f'[!] Microtaint Unicorn crash on instruction (hex): {cell.instruction}. Exception: {e}')
            raise
        val_and = self._read_reg(cell.out_reg)

        diff = val_or ^ val_and
        mask = (1 << (cell.out_bit_end - cell.out_bit_start + 1)) - 1
        return int((diff >> cell.out_bit_start) & mask)

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
            # See _execute for why ctl_remove_cache is critical here.
            try:
                self.uc.ctl_remove_cache(self.CODE_ADDR, self.CODE_ADDR + len(bytestring))
            except (AttributeError, uc_py3.UcError):
                try:
                    self.uc.mem_unmap(self.CODE_ADDR, 4096)
                    self.uc.mem_map(self.CODE_ADDR, 4096)
                    self.uc.mem_write(self.CODE_ADDR, bytestring)
                except uc_py3.UcError:
                    pass

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
