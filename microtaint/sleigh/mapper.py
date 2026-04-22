from __future__ import annotations

from pypcode import PcodeOp

from microtaint.classifier.categories import InstructionCategory

# CellIFT Avalanche Operations (Multiplications, Divisions)
AVALANCHE_OPCODES: set[str] = {
    'INT_MULT',
    'INT_DIV',
    'INT_SDIV',
    'INT_REM',
    'INT_SREM',
}

# CellIFT Translatable Operations (Shifts)
TRANSLATABLE_OPCODES: set[str] = {
    'INT_LEFT',  # Logical Shift Left
    'INT_RIGHT',  # Logical Shift Right
    'INT_SRIGHT',  # Arithmetic Shift Right
}

# CellIFT Transportable Operations (Arithmetic Add/Sub)
TRANSPORTABLE_OPCODES: set[str] = {
    'INT_ADD',
    'INT_SUB',
    'INT_CARRY',
    'INT_SCARRY',
    'INT_SBORROW',
}

# CellIFT Conditionally Transportable (Equalities)
COND_TRANSPORTABLE_OPCODES: set[str] = {
    'INT_EQUAL',
    'INT_NOTEQUAL',
}

# CellIFT Monotonic Operations (Comparisons)
MONOTONIC_OPCODES: set[str] = {
    'INT_LESS',
    'INT_SLESS',
    'INT_LESSEQUAL',
    'INT_SLESSEQUAL',
}

# CellIFT Mapped Operations (Bitwise logical operations, exact copies, memory)
MAPPED_OPCODES: set[str] = {
    'COPY',  # Register to register exact move
    'LOAD',  # Memory read
    'STORE',  # Memory write
    'INT_AND',
    'INT_OR',
    'INT_XOR',
    'INT_NEGATE',  # Bitwise NOT
    'INT_ZEXT',  # Zero extension
    'INT_SEXT',  # Sign extension
    'SUBPIECE',  # Bit slice extraction
    'PIECE',  # Bit concatenation
    'POPCOUNT',  # Counting bits maps directly linearly
}


def determine_category(slice_ops: list[PcodeOp]) -> InstructionCategory:  # noqa: C901
    """
    Given a backwards slice of P-Code operations defining an output,
    determine its highest CellIFT category.

    Precedence enforces that the most complex transformation in the slice
    dictates the cell formula used:
    Avalanche > Translatable > Transportable > Cond_Transportable > Monotonic > Mapped
    """
    if not slice_ops:
        return InstructionCategory.MAPPED  # Zero ops implies exact copy or identity

    # 1. Avalanche supersedes everything (destroys bit-precision tracking)
    for op in slice_ops:
        if op.opcode.name in AVALANCHE_OPCODES:
            return InstructionCategory.AVALANCHE

    # 2. Translatable (Shifts)
    for op in slice_ops:
        if op.opcode.name in TRANSLATABLE_OPCODES:
            return InstructionCategory.TRANSLATABLE

    # 3. Transportable (Arithmetic combinations)
    for op in slice_ops:
        if op.opcode.name in TRANSPORTABLE_OPCODES:
            return InstructionCategory.TRANSPORTABLE

    # 4. Conditionally Transportable (Equality checks)
    for op in slice_ops:
        if op.opcode.name in COND_TRANSPORTABLE_OPCODES:
            return InstructionCategory.COND_TRANSPORTABLE

    # 5. Monotonic (Bounds checks / comparisons)
    for op in slice_ops:
        if op.opcode.name in MONOTONIC_OPCODES:
            return InstructionCategory.MONOTONIC

    # 6. Mapped (Direct pass-throughs, bitwise ops, memory mappings)
    for op in slice_ops:
        if op.opcode.name in MAPPED_OPCODES:
            return InstructionCategory.MAPPED

    # Default fallback
    return InstructionCategory.MAPPED
