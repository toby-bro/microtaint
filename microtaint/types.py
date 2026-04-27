from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum


class Architecture(StrEnum):
    X86 = 'X86'
    ARM64 = 'ARM64'
    AMD64 = 'AMD64'


@dataclass(slots=True)
class Register:
    name: str
    bits: int


class ImplicitTaintPolicy(IntEnum):
    IGNORE = 0
    WARN = 1
    STOP = 2


class ImplicitTaintError(Exception):
    """Raised when an implicit taint dependency is detected and policy is STOP."""
