from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum


class Architecture(StrEnum):
    X86 = 'X86'
    ARM64 = 'ARM64'
    AMD64 = 'AMD64'
    RISCV64 = 'RISCV64'
    # RISC targets. Neither needs an entry in ast.pyx's _ARCH_PARENT_REGS /
    # _ARCH_CHILD_REGS: they have no sub-register aliasing (x86 needs 37 alias
    # rows), and those tables are read with .get(arch, {}).
    #   MIPS64BE -- SLEIGH reports NO condition-flag registers at all: compares
    #     write GPRs (`slt`), so the whole x86 flag apparatus is inapplicable.
    #   PPC32BE  -- carry lives in XER (xer_ca) and conditions in cr0..cr7,
    #     rather than in a flat set of 1-bit flags at fixed offsets.
    # Unicorn 2.1.4 executes PPC32 but NOT PPC64 (UC_ERR_EXCEPTION), so there is
    # no ground-truth oracle for PPC64 -- hence 32-bit here.
    MIPS64BE = 'MIPS64BE'
    PPC32BE = 'PPC32BE'
    #   SPARC32BE -- condition codes live in CCR/icc.  SPARC has REGISTER WINDOWS
    #     (%g/%o/%l/%i, rotated by save/restore) and branch delay slots, but
    #     neither affects the DATA semantics of a single instruction, which is
    #     what rule synthesis operates on: within one window the GPRs are plain
    #     registers with no sub-register aliasing.
    SPARC32BE = 'SPARC32BE'


@dataclass(slots=True)
class Register:
    name: str
    bits: int


class ImplicitTaintPolicy(IntEnum):
    IGNORE = 0
    WARN = 1
    STOP = 2
    KEEP = 3


class ImplicitTaintError(Exception):
    """Raised when an implicit taint dependency is detected and policy is STOP."""
