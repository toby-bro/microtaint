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
    'LZCOUNT',
    'FLOAT_ADD',
    'FLOAT_SUB',
    'FLOAT_MULT',
    'FLOAT_DIV',
    'FLOAT_ABS',
    'FLOAT_SQRT',
    'FLOAT_CEIL',
    'FLOAT_FLOOR',
    'FLOAT_ROUND',
    'FLOAT_TRUNC',
    'FLOAT_NEG',
    'FLOAT_INT2FLOAT',
    'FLOAT_FLOAT2FLOAT',
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
    'PTRADD',
    'PTRSUB',
}

COND_TRANSPORTABLE_OPCODES: set[str] = {
    'INT_EQUAL',
    'INT_NOTEQUAL',
    'FLOAT_EQUAL',
    'FLOAT_NOTEQUAL',
    'FLOAT_LESS',
    'FLOAT_LESSEQUAL',
    'FLOAT_NAN',
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
    'BOOL_AND',
    'BOOL_OR',
    'BOOL_NEGATE',
}

ROUTING_OPCODES: set[str] = {
    'INT_LEFT',
    'INT_RIGHT',
    'INT_SRIGHT',
    'COPY',
    'INT_ZEXT',
    'INT_SEXT',
    'SUBPIECE',
    'PIECE',
    'EXTRACT',
    'INSERT',
    'LOAD',
    'INT_AND',
    'INT_OR',
    'BOOL_AND',
    'BOOL_OR',
}

ORABLE_OPCODES: set[str] = {
    'INT_XOR',
    'BOOL_XOR',
}

MAPPED_OPCODES: set[str] = {
    'LOAD',
    'STORE',
}

EXTENSION_OPCODES: set[str] = {
    'INT_ZEXT',
    'INT_SEXT',
    'SUBPIECE',
    'PIECE',
    'COPY',
    'EXTRACT',
    'INSERT',
}

CONTROL_FLOW_OPCODES: set[str] = {
    'BRANCH',
    'BRANCHIND',
    'CBRANCH',
    'CALL',
    'CALLIND',
    'CALLOTHER',
    'RETURN',
}

IGNORED_OPCODES: set[str] = {
    *CONTROL_FLOW_OPCODES,
    'IMARK',
    'INDIRECT',
    'MULTIEQUAL',
    'CPOOLREF',
    'NEW',
    'CAST',
    'SEGMENTOP',
}


def is_mapped_permutation(slice_ops: list[PcodeOp]) -> bool:  # noqa: C901
    """
    Heuristic: A true permutation only uses routing/shifting opcodes
    AND relies on only ONE dynamic input (register/memory). All other inputs must be constants.

    Extended case: exactly 2 dynamic sources are allowed when one is a 1-bit register
    (e.g. CF at offset 0x200) and all ops are routing ops. This handles rotate-through-
    carry instructions (rcl/rcr) where the carry bit IS a genuine external fill source,
    not an intra-instruction intermediate. The differential correctly computes the
    per-bit permutation for these 2-source cases.

    Intra-instruction intermediate registers (written before being read within the slice,
    like CF in ``ror rax,1`` where CF is computed from bit0 of RAX and then used to fill
    bit63) are excluded from the dynamic-source count.
    """
    dynamic_sources: set[tuple[str, int, int]] = set()  # (space, offset, size)
    has_and_or = False
    has_shift = False

    # Track registers written within this slice (intra-instruction intermediates)
    _slice_written: dict[int, int] = {}  # register offset → first write index in slice
    for _i, op in enumerate(slice_ops):
        if op.output is not None and op.output.space.name == 'register':
            if op.output.offset not in _slice_written:
                _slice_written[op.output.offset] = _i

    for _i, op in enumerate(slice_ops):
        # Check whether this op produces an intra-slice intermediate register
        # that is consumed by a later op (like CF in ror rax,1 computed by NOTEQUAL).
        # We skip the routing-op check ONLY for non-routing ops whose output is
        # a DIFFERENT register than the final architectural output AND is written
        # before being consumed by a later op.
        # INT_NOTEQUAL in ror writes CF (offset 0x200, different from RAX at 0) → skip OK.
        # INT_ADD in inc eax writes EAX (offset 0x0, same as RAX output at 0) → NOT skipped.
        #
        # RESTRICTION: only 1-bit registers (flag registers) qualify as intra-intermediates.
        # Wider registers like RSP (64-bit) that are arithmetically modified to compute
        # a memory address are NOT intermediates — they are address operands and their
        # modification must be treated as a non-routing op that breaks the permutation.
        # This prevents push/pop (INT_SUB writes RSP, LOAD reads RSP-8) from being
        # incorrectly classified as a permutation.
        _is_intra_intermediate = False
        if (
            op.opcode.name not in ROUTING_OPCODES
            and op.output is not None
            and op.output.space.name == 'register'
            and op.output.size == 1  # only flag-sized (1-bit) registers qualify
            and op.output.offset in _slice_written
            and _slice_written[op.output.offset] == _i
        ):
            # Get the final output register: the last op that writes to a register in the slice
            _final_out_offsets = {
                o.output.offset
                for o in slice_ops
                if (o.output is not None and o.output.space.name == 'register' and o is slice_ops[-1])
            }
            # Also accept: the output is to a register that is NOT the one we're
            # ultimately computing taint for. We detect this by checking whether
            # any later routing op reads this register as input.
            _consumed_by_routing = any(
                (
                    later_op.opcode.name in ROUTING_OPCODES
                    and any(v.space.name == 'register' and v.offset == op.output.offset for v in later_op.inputs)
                )
                for later_op in slice_ops[_i + 1 :]
            )
            # Only an intermediate if: consumed by routing op AND output register
            # is NOT the same as any routing op's output (i.e. not the main data path).
            if _consumed_by_routing:
                # Verify the intermediate register is not on the main data path
                # (i.e. the final routing output doesn't write to the same offset+size).
                _final_routing_writes = {
                    (o.output.offset, o.output.size)
                    for o in slice_ops
                    if (o.output is not None and o.output.space.name == 'register' and o.opcode.name in ROUTING_OPCODES)
                }
                if (op.output.offset, op.output.size) not in _final_routing_writes:
                    # Also check: is this offset covered by any final routing write
                    # of a different size (e.g. INT_ADD writes EAX offset 0 size 4,
                    # while INT_ZEXT writes RAX offset 0 size 8 — same register family)?
                    _offset_in_routing = any(roff == op.output.offset for roff, _ in _final_routing_writes)
                    if not _offset_in_routing:
                        _is_intra_intermediate = True

        if _is_intra_intermediate:
            # Skip routing check for this op. Collect its external dynamic sources.
            for vn in op.inputs:
                if vn.space.name not in ('const', 'unique'):
                    first_write = _slice_written.get(vn.offset, len(slice_ops))
                    if first_write >= _i:
                        dynamic_sources.add((vn.space.name, vn.offset, vn.size))
            continue

        if op.opcode.name not in ROUTING_OPCODES:
            return False

        if op.opcode.name in {'INT_AND', 'INT_OR', 'BOOL_AND', 'BOOL_OR'}:
            has_and_or = True
        if op.opcode.name in {'INT_LEFT', 'INT_RIGHT', 'INT_SRIGHT'}:
            has_shift = True

        for vn in op.inputs:
            if vn.space.name not in ('const', 'unique'):
                # Exclude intra-instruction intermediates: registers written earlier
                # in this slice (e.g. CF computed by a prior non-routing op).
                first_write = _slice_written.get(vn.offset, len(slice_ops))
                if first_write < _i:
                    continue
                dynamic_sources.add((vn.space.name, vn.offset, vn.size))

    if len(dynamic_sources) == 1:
        # Classic single-source permutation (bswap, shifts by constant, mov, ror, etc.)
        if has_and_or and not has_shift:
            return False
        return True

    if len(dynamic_sources) == 2:
        # Allow 2-source permutation when one source is a *flag* register (e.g. CF
        # at Sleigh offset 0x200) and there IS a shift present (the carry bit is
        # shifted into a bit position).  This covers rcl/rcr where CF fills one bit
        # of the rotated result.
        #
        # Explicitly excluded: shift instructions like `shl rax, cl` where the 1-bit
        # source is CL (sub-byte of RCX, offset 0x8) — that is a shift *amount*, not
        # a fill bit, so it must remain TRANSLATABLE (variable-amount shift → avalanche).
        # We distinguish carry-fill bits from shift-amount bits by checking whether the
        # 1-bit source lives at a known x86 flag register offset.
        _X86_FLAG_OFFSETS = frozenset({0x200, 0x20B, 0x206, 0x207, 0x202, 0x203})
        one_bit_sources = [(sp, off, sz) for sp, off, sz in dynamic_sources if sz == 1]
        multi_bit_sources = [(sp, off, sz) for sp, off, sz in dynamic_sources if sz > 1]
        if (
            len(one_bit_sources) == 1
            and len(multi_bit_sources) == 1
            and has_shift
            and one_bit_sources[0][1] in _X86_FLAG_OFFSETS
        ):
            return True

    return False


def determine_category(slice_ops: list[PcodeOp]) -> InstructionCategory:  # noqa: C901
    if not slice_ops:
        return InstructionCategory.MAPPED

    if is_mapped_permutation(slice_ops):
        return InstructionCategory.MAPPED

    # Safely filter out ignored and extension operations from the core evaluation
    core_ops = [
        op for op in slice_ops if op.opcode.name not in EXTENSION_OPCODES and op.opcode.name not in IGNORED_OPCODES
    ]

    # If all operations were just simple routing, copies, or ignored metadata
    if not core_ops:
        return InstructionCategory.MAPPED

    for op in core_ops:
        if op.opcode.name in AVALANCHE_OPCODES:
            return InstructionCategory.AVALANCHE

    for op in core_ops:
        if op.opcode.name in COND_TRANSPORTABLE_OPCODES:
            return InstructionCategory.COND_TRANSPORTABLE

    # FIX: Translatable MUST be above Monotonic to avoid INT_AND stealing shifts
    for op in core_ops:
        if op.opcode.name in TRANSLATABLE_OPCODES:
            return InstructionCategory.TRANSLATABLE

    for op in core_ops:
        if op.opcode.name in MONOTONIC_OPCODES:
            return InstructionCategory.MONOTONIC

    for op in core_ops:
        if op.opcode.name in TRANSPORTABLE_OPCODES:
            return InstructionCategory.TRANSPORTABLE

    for op in core_ops:
        if op.opcode.name in ORABLE_OPCODES:
            return InstructionCategory.ORABLE

    # Fallback for pure memory operations that weren't caught by the heuristic
    for op in core_ops:
        if op.opcode.name in MAPPED_OPCODES:
            return InstructionCategory.MAPPED

    raise ValueError(
        'Unable to determine instruction category for slice_ops: ' + ', '.join(op.opcode.name for op in slice_ops),
    )
