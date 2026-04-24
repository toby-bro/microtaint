from __future__ import annotations

from enum import Enum


class InstructionCategory(Enum):
    NO_DATA_OUTPUTS = 'No Data Outputs'
    MAPPED = 'Mapped'
    MONOTONIC = 'Monotonic'
    TRANSPORTABLE = 'Transportable'
    TRANSLATABLE = 'Translatable'
    COND_TRANSPORTABLE = 'Conditionally Transportable'
    ORABLE = 'Orable'  # for XOR, where we just need to OR the input taints together
    AVALANCHE = 'Avalanche'
    UNKNOWN = 'Unknown'

    def __str__(self) -> str:
        return self.value
