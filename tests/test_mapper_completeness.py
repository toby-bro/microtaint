import pypcode

from microtaint.sleigh import mapper


def test_all_pcode_ops_are_categorized() -> None:
    """
    Ensures every single OpCode defined in pypcode is explicitly accounted for
    in the taint categorization sets, so the AST builder never crashes on an unknown op.
    """
    # 1. Combine all your defined category sets from mapper.py
    all_categorized = (
        mapper.AVALANCHE_OPCODES
        | mapper.TRANSLATABLE_OPCODES
        | mapper.TRANSPORTABLE_OPCODES
        | mapper.COND_TRANSPORTABLE_OPCODES
        | mapper.MONOTONIC_OPCODES
        | mapper.ROUTING_OPCODES
        | mapper.ORABLE_OPCODES
        | mapper.MAPPED_OPCODES
        | mapper.EXTENSION_OPCODES
    )

    # 2. Add ops that we intentionally IGNORE (Control flow, SSA, etc.)
    # You should add this set to mapper.py!
    IGNORED_OPCODES = {
        'BRANCH',
        'BRANCHIND',
        'CBRANCH',
        'CALL',
        'CALLIND',
        'CALLOTHER',
        'RETURN',
        'IMARK',
        'INDIRECT',
        'MULTIEQUAL',
        'CPOOLREF',
        'NEW',
        'CAST',
        'SEGMENTOP',
    }
    assert (
        IGNORED_OPCODES == mapper.IGNORED_OPCODES
    ), 'Please move IGNORED_OPCODES to mapper.py and keep it updated with any new ignored ops!'

    all_handled = all_categorized | IGNORED_OPCODES

    # 3. Get all opcodes defined by the pypcode library
    all_pypcode_ops: set[str] = {opcode.name for opcode in pypcode.OpCode}  # type: ignore[attr-defined]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportGeneralTypeIssues]  # ty: ignore[not-iterable]

    # 4. Find what is missing
    missing_ops = all_pypcode_ops - all_handled

    assert not missing_ops, f'Missing categorizations for P-Code ops: {missing_ops}'
