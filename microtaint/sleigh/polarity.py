from __future__ import annotations

from pypcode import PcodeOp

from microtaint.sleigh.slicer import get_varnode_id


def compute_polarity(  # noqa: C901
    slice_ops: list[PcodeOp],
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

    node_polarities: dict[str, int] = {}
    if slice_ops[-1].output:
        node_polarities[get_varnode_id(slice_ops[-1].output)] = 1

    for op in reversed(slice_ops):
        if not op.output:
            continue

        out_id = get_varnode_id(op.output)

        # If this node's output isn't part of tracking, assume D=1
        current_polarity = node_polarities.get(out_id, 1)

        op_name = op.opcode.name

        # Mapped bitwise logic functions generally act as 1 (unless NOT is involved)
        # Arithmetic logic passes through the polarity, except subtraction

        # INT_SBORROW(a, b) is the signed overflow of a - b: its RHS is SUBTRACTED,
        # exactly like INT_SUB's.  Without this it fell to the default branch below
        # and marked the RHS positive, CLOBBERING the negative polarity INT_SUB had
        # just assigned to the same varnode (both ops read it -- SBORROW computes OF
        # while SUB computes the result feeding SF).  The loaded value of
        # `cmp rax,[rsp-16]` then looked positive, so the STORE->LOAD forwarding
        # below had nothing to forward and the stored register kept polarity +1 --
        # polarising both comparison operands identically into a lossy D^{++} that
        # cancels.  INT_CARRY/INT_SCARRY are the a + b carries, so both their
        # operands stay positive and they belong with INT_ADD.
        if op_name in ('INT_SUB', 'INT_SBORROW'):
            # LHS maintains current polarity
            lhs = get_varnode_id(op.inputs[0])
            node_polarities[lhs] = current_polarity
            if op.inputs[0].space.name != 'const':
                polarity_map[lhs] = current_polarity

            # RHS inverses the current polarity (1 becomes 0, 0 becomes 1)
            rhs = get_varnode_id(op.inputs[1])
            inv_polarity = 0 if current_polarity == 1 else 1
            node_polarities[rhs] = inv_polarity
            if op.inputs[1].space.name != 'const':
                polarity_map[rhs] = inv_polarity

        elif op_name in (
            'INT_MULT',
            'INT_ADD',
            'INT_CARRY',
            'INT_SCARRY',
            'INT_ZEXT',
            'INT_SEXT',
            'INT_AND',
            'INT_OR',
            'INT_XOR',
            'COPY',
        ):
            # Operations where polarity is directly propagated to operands
            for inp in op.inputs:
                if inp.space.name != 'const':
                    inp_id = get_varnode_id(inp)
                    node_polarities[inp_id] = current_polarity
                    polarity_map[inp_id] = current_polarity

        elif op_name in ('INT_NEGATE', 'BOOL_NEGATE', 'INT_2COMP'):
            # Ops that INVERT the dependency direction: bitwise NOT (~x = -x-1),
            # logical NOT (!x = 1-x), two's-complement negation (-x).  Increasing the
            # operand decreases the output through them, so flip polarity -- exactly
            # like INT_SUB's RHS.  BOOL_NEGATE was previously unhandled (default
            # branch, propagate): PPC `subfe`/`sbc` compute `... - !carry`, i.e.
            # `... - (1 - carry) = ... + carry`, so the borrow-in is EFFECTIVELY a
            # positive operand; leaving it negative polarised it into the wrong
            # differential corner and under-tainted the borrow chain whenever the
            # carry was tainted (subfe: 138/2000 with tainted carry, 0 without).
            for inp in op.inputs:
                if inp.space.name != 'const':
                    inp_id = get_varnode_id(inp)
                    inv_polarity = 0 if current_polarity == 1 else 1
                    node_polarities[inp_id] = inv_polarity
                    polarity_map[inp_id] = inv_polarity

        else:
            # Default fallback for unhandled or neutral operations
            for inp in op.inputs:
                if inp.space.name != 'const':
                    inp_id = get_varnode_id(inp)
                    node_polarities[inp_id] = current_polarity
                    polarity_map[inp_id] = current_polarity

    return polarity_map
