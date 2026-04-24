from __future__ import annotations

from pypcode import PcodeOp

from microtaint.classifier.categories import InstructionCategory

AVALANCHE_OPCODES: set[str] = {
    'INT_MULT',
    'INT_DIV',
    'INT_SDIV',
    'INT_REM',
    'INT_SREM',
    'POPCOUNT',
}

TRANSLATABLE_OPCODES: set[str] = {
    'INT_LEFT',
    'INT_RIGHT',
    'INT_SRIGHT',
}

TRANSPORTABLE_OPCODES: set[str] = {
    'INT_ADD',
    'INT_SUB',
    'INT_2COMP',
}

COND_TRANSPORTABLE_OPCODES: set[str] = {
    'INT_EQUAL',
    'INT_NOTEQUAL',
}

MONOTONIC_OPCODES: set[str] = {
    'INT_LESS',
    'INT_LESSEQUAL',
    'INT_SLESS',
    'INT_SLESSEQUAL',
    'INT_CARRY',
    'INT_SCARRY',
    'INT_SBORROW',
    'INT_AND',
    'INT_OR',
    'INT_NEGATE',
}

ROUTING_OPCODES: set[str] = {
    'INT_AND',
    'INT_OR',
    'INT_LEFT',
    'INT_RIGHT',
    'INT_SRIGHT',
    'COPY',
    'INT_ZEXT',
    'INT_SEXT',
    'SUBPIECE',
    'PIECE',
    'LOAD',
}

ORABLE_OPCODES: set[str] = {
    'INT_XOR',
}

MAPPED_OPCODES: set[str] = {
    'LOAD',
    'STORE',
}

EXTENSION_OPCODES = {'INT_ZEXT', 'INT_SEXT', 'SUBPIECE', 'PIECE', 'COPY'}


def is_mapped_permutation(slice_ops: list[PcodeOp]) -> bool:
    """
    Heuristic: A true permutation only uses routing/shifting opcodes
    AND relies on only ONE dynamic input (register/memory). All other inputs must be constants.
    """
    dynamic_sources: set[tuple[str, int]] = set()
    has_and_or = False
    has_shift = False

    for op in slice_ops:
        if op.opcode.name not in ROUTING_OPCODES:
            return False

        if op.opcode.name in {'INT_AND', 'INT_OR'}:
            has_and_or = True
        if op.opcode.name in {'INT_LEFT', 'INT_RIGHT', 'INT_SRIGHT'}:
            has_shift = True

        for vn in op.inputs:
            # Ignore constants and temporary microcode registers ('unique')
            if vn.space.name not in ('const', 'unique'):
                # We track the base offset to ensure it's pulling from the same architectural register
                dynamic_sources.add((vn.space.name, vn.offset))

    # If it reads from exactly 1 dynamic architectural source, it's a simple mapped permutation
    if len(dynamic_sources) != 1:
        return False

    # FIX: If it uses AND/OR, it MUST also use a shift to be a spatial permutation (like BSWAP).
    # An isolated AND/OR with a constant is just a bitwise mask and must drop to MONOTONIC.
    if has_and_or and not has_shift:
        return False

    return True


def determine_category(slice_ops: list[PcodeOp]) -> InstructionCategory:  # noqa: C901
    if not slice_ops:
        return InstructionCategory.MAPPED

    # 1. First, check if it fits the perfect single-register permutation heuristic
    if is_mapped_permutation(slice_ops):
        return InstructionCategory.MAPPED

    # 2. Otherwise, strip extensions to find the core ALU operation
    core_ops = [op for op in slice_ops if op.opcode.name not in EXTENSION_OPCODES]

    # If all operations were just simple routing/copies (e.g., pure mov chains)
    if not core_ops:
        return InstructionCategory.MAPPED

    for op in core_ops:
        if op.opcode.name in AVALANCHE_OPCODES:
            return InstructionCategory.AVALANCHE

    for op in core_ops:
        if op.opcode.name in TRANSLATABLE_OPCODES:
            return InstructionCategory.TRANSLATABLE

    for op in core_ops:
        if op.opcode.name in TRANSPORTABLE_OPCODES:
            return InstructionCategory.TRANSPORTABLE

    for op in core_ops:
        if op.opcode.name in COND_TRANSPORTABLE_OPCODES:
            return InstructionCategory.COND_TRANSPORTABLE

    for op in core_ops:
        if op.opcode.name in MONOTONIC_OPCODES:
            return InstructionCategory.MONOTONIC

    for op in core_ops:
        if op.opcode.name in ORABLE_OPCODES:
            return InstructionCategory.ORABLE

    # 3. Fallback for pure memory operations that weren't caught by the heuristic
    for op in core_ops:
        if op.opcode.name in MAPPED_OPCODES:
            return InstructionCategory.MAPPED

    raise ValueError(
        'Unable to determine instruction category for slice_ops: ' + ', '.join(op.opcode.name for op in slice_ops),
    )
