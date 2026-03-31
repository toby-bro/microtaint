from __future__ import annotations

import pypcode

from microtaint.classifier.categories import InstructionCategory

# CellIFT Transportable Operations (Arithmetic, Multiplications, Shifts)
TRANSPORTABLE_OPCODES: set[str] = {
    'INT_ADD',
    'INT_SUB',
    'INT_LEFT',  # Logical Shift Left
    'INT_RIGHT',  # Logical Shift Right
    'INT_SRIGHT',  # Arithmetic Shift Right
    'INT_LESS',  # Math comparisons that calculate bounds
    'INT_SLESS',
    'INT_LESSEQUAL',
    'INT_SLESSEQUAL',
    'INT_EQUAL',
    'INT_NOTEQUAL',
    'INT_CARRY',
    'INT_SCARRY',
    'INT_SBORROW',
}

# CellIFT Mapped Operations (Bitwise logical operations, exact copies)

# CellIFT Avalanche Operations (Multiplications, Divisions)
AVALANCHE_OPCODES: set[str] = {
    'INT_MULT',
    'INT_DIV',
    'INT_SDIV',
    'INT_REM',
    'INT_SREM',
}

MAPPED_OPCODES: set[str] = {
    'COPY',  # Register to register exact move
    'LOAD',  # Memory read
    'STORE',  # Memory write
    'INT_AND',
    'INT_OR',
    'INT_XOR',
    'INT_ZEXT',  # Zero extension
    'INT_SEXT',  # Sign extension
    'SUBPIECE',  # Bit slice extraction
    'PIECE',  # Bit concatenation
    'POPCOUNT',  # Counting bits maps directly linearly
}


def determine_category(slice_ops: list[pypcode.pypcode_native.PcodeOp]) -> InstructionCategory:
    """
    Given a backwards slice of P-Code operations defining an output,
    determine its highest CellIFT category.
    Transportable (arithmetic) supersedes Mapped (bitwise).
    """
    if not slice_ops:
        return InstructionCategory.MAPPED  # Zero ops implies exact copy or identity

    has_mapped = False

    for op in slice_ops:
        op_name = op.opcode.name

        # If any operation in the slice relies on avalanche operations
        if op_name in AVALANCHE_OPCODES:
            return InstructionCategory.AVALANCHE

        # If any operation in the slice relies on arithmetic/carries,
        if op_name in TRANSPORTABLE_OPCODES:
            return InstructionCategory.TRANSPORTABLE

        if op_name in MAPPED_OPCODES:
            has_mapped = True

    if has_mapped:
        return InstructionCategory.MAPPED

    # Default to mapped if unsure, though ideally all ops should be classified
    return InstructionCategory.MAPPED
