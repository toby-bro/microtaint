from __future__ import annotations

import pypcode


def _get_varnode_id(vn: pypcode.pypcode_native.Varnode) -> str:
    """Helper to get a unique identifier for a varnode."""
    if not vn:
        return ''
    return f'{vn.space.name}:{vn.offset}:{vn.size}'


def slice_backward(
    ops: list[pypcode.pypcode_native.PcodeOp],
    target_varnode: pypcode.pypcode_native.Varnode,
) -> list[pypcode.pypcode_native.PcodeOp]:
    """
    Given an ordered list of P-code operations and a target output varnode,
    traverse backward to find all operations that contribute to computing it.
    """
    target_id = _get_varnode_id(target_varnode)

    # Track which varnodes we care about parsing backward
    worklist: set[str] = {target_id}
    slice_ops: list[pypcode.pypcode_native.PcodeOp] = []

    for op in reversed(ops):
        if not op.output:
            continue

        out_id = _get_varnode_id(op.output)
        if out_id in worklist:
            # This operation contributes to our slice
            slice_ops.append(op)

            # P-Code from Ghidra is almost SSA-like per instruction,
            # but just to be safe, we keep the target_id so we don't drop it.

            # Add all its inputs to the worklist to track them further up
            for inp in op.inputs:
                if inp.space.name != 'const':
                    worklist.add(_get_varnode_id(inp))

    # Return operations in their original execution (forward) order
    return list(reversed(slice_ops))
