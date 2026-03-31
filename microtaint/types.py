from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Architecture(StrEnum):
    X86 = 'X86'
    ARM64 = 'ARM64'
    AMD64 = 'AMD64'


@dataclass
class Register:
    name: str
    bits: int
