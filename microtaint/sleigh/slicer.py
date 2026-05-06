from __future__ import annotations

from pypcode.pypcode_native import PcodeOp, Varnode


def get_varnode_id(vn: Varnode) -> str:
    """Helper to get a unique identifier for a varnode."""
    if not vn:
        return ''
    return f'{vn.space.name}:{vn.offset}:{vn.size}'


def _vn_range(vn: Varnode) -> tuple[str, int, int]:
    """Return (space_name, byte_start, byte_end_exclusive) for a varnode."""
    return (vn.space.name, vn.offset, vn.offset + vn.size)


def slice_backward(
    ops: list[PcodeOp],
    target_varnode: Varnode,
) -> list[PcodeOp]:
    """
    Given an ordered list of P-code operations and a target output varnode,
    traverse backward to find all operations that contribute to computing it.

    Overlap-aware: an op's output is included in the slice if its byte range
    overlaps ANY varnode currently in the worklist within the same address
    space.  This is necessary because pypcode emits SUBPIECE / partial-COPY
    sequences where, e.g., MULX's INT_MULT writes a 16-byte unique varnode
    and the immediately following COPY reads only the low 8 bytes.  The old
    exact-match-by-(space,offset,size) tuple would miss the INT_MULT and
    drop the carry chain entirely (same root cause as ADCX's lifted form
    using a 9-bit INT_ADD whose low 8 bytes feed RAX).
    """
    # Worklist as a list of (space, byte_start, byte_end) ranges.  Lists are
    # cheaper than sets here because the count of live ranges per slice is
    # tiny (≤ ~6 in practice) and overlap checks are pairwise.
    worklist: list[tuple[str, int, int]] = [_vn_range(target_varnode)]
    slice_ops: list[PcodeOp] = []

    def _overlaps_any(out_range: tuple[str, int, int]) -> bool:
        ospace, ostart, oend = out_range
        for wspace, wstart, wend in worklist:
            if wspace == ospace and wstart < oend and ostart < wend:
                return True
        return False

    for op in reversed(ops):
        if not op.output:
            continue

        out_range = _vn_range(op.output)
        if _overlaps_any(out_range):
            # This operation contributes to our slice
            slice_ops.append(op)

            # Track all inputs (including LOAD memory pointers).
            for inp in op.inputs:
                if inp.space.name != 'const':
                    worklist.append(_vn_range(inp))

    # Return operations in their original execution (forward) order
    return list(reversed(slice_ops))
