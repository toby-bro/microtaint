from __future__ import annotations

from pypcode.pypcode_native import PcodeOp, Varnode


def get_varnode_id(vn: Varnode) -> str:
    """Helper to get a unique identifier for a varnode."""
    if not vn:
        return ''
    return f'{vn.space.name}:{vn.offset}:{vn.size}'


def slice_backward(
    ops: list[PcodeOp],
    target_varnode: Varnode,
) -> list[PcodeOp]:
    """
    Given an ordered list of P-code operations and a target output varnode,
    traverse backward to find all operations that contribute to computing it.
    """
    target_id = get_varnode_id(target_varnode)

    # Track which varnodes we care about parsing backward
    worklist: set[str] = {target_id}
    slice_ops: list[PcodeOp] = []

    for op in reversed(ops):
        if not op.output:
            continue

        out_id = get_varnode_id(op.output)
        if out_id in worklist:
            # This operation contributes to our slice
            slice_ops.append(op)

            # Track all inputs, including LOAD memory pointers to map memory dependencies
            for inp in op.inputs:
                if inp.space.name != 'const':
                    worklist.add(get_varnode_id(inp))

    # Return operations in their original execution (forward) order
    return list(reversed(slice_ops))
