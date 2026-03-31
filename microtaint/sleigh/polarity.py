from __future__ import annotations

import pypcode

from microtaint.sleigh.slicer import _get_varnode_id


def compute_polarity(  # noqa: C901
    slice_ops: list[pypcode.pypcode_native.PcodeOp],
) -> dict[str, int]:
    """
    Given a backwards slice of P-Code operations defining an output,
    calculate the D-vector polarity mapping for all input varnodes.

    Returns a dictionary mapping varnode_id to its expected D mask (1 or 0).
    A polarity of `1` (D=1) indicates a non-decreasing (positive) dependency like an addition.
    A polarity of `0` (D=0) indicates a non-increasing (negative) dependency like a subtraction.
    """
    if not slice_ops:
        return {}

    polarity_map: dict[str, int] = {}

    # We walk backwards through the ops.
    # Usually inputs start as 1, but operations like INT_SUB can invert the right operand.
    # To properly propagate inversions (e.g., `- (A + B) => -A - B`), we track the
    # expected polarity of intermediate nodes.

    node_polarities = {}
    if slice_ops[-1].output:
        node_polarities[_get_varnode_id(slice_ops[-1].output)] = 1

    for op in reversed(slice_ops):
        if not op.output:
            continue

        out_id = _get_varnode_id(op.output)

        # If this node's output isn't part of tracking, assume D=1
        current_polarity = node_polarities.get(out_id, 1)

        op_name = op.opcode.name  # type: ignore[attr-defined]

        # Mapped bitwise logic functions generally act as 1 (unless NOT is involved)
        # Arithmetic logic passes through the polarity, except subtraction

        if op_name == 'INT_SUB':
            # LHS maintains current polarity
            lhs = _get_varnode_id(op.inputs[0])
            node_polarities[lhs] = current_polarity
            if op.inputs[0].space.name != 'const':
                polarity_map[lhs] = current_polarity

            # RHS inverses the current polarity (1 becomes 0, 0 becomes 1)
            rhs = _get_varnode_id(op.inputs[1])
            inv_polarity = 0 if current_polarity == 1 else 1
            node_polarities[rhs] = inv_polarity
            if op.inputs[1].space.name != 'const':
                polarity_map[rhs] = inv_polarity

        elif op_name in ('INT_MULT', 'INT_ADD', 'INT_ZEXT', 'INT_SEXT', 'INT_AND', 'INT_OR', 'INT_XOR', 'COPY'):
            # Operations where polarity is directly propagated to operands
            for inp in op.inputs:
                if inp.space.name != 'const':
                    inp_id = _get_varnode_id(inp)
                    node_polarities[inp_id] = current_polarity
                    polarity_map[inp_id] = current_polarity

        elif op_name == 'INT_NEGATE':  # Bitwise NOT (often simulated logically as negative)
            for inp in op.inputs:
                if inp.space.name != 'const':
                    inp_id = _get_varnode_id(inp)
                    inv_polarity = 0 if current_polarity == 1 else 1
                    node_polarities[inp_id] = inv_polarity
                    polarity_map[inp_id] = inv_polarity

        else:
            # Default fallback for unhandled or neutral operations
            for inp in op.inputs:
                if inp.space.name != 'const':
                    inp_id = _get_varnode_id(inp)
                    node_polarities[inp_id] = current_polarity
                    polarity_map[inp_id] = current_polarity

    return polarity_map
