from __future__ import annotations

import pypcode

from microtaint.classifier.categories import InstructionCategory
from microtaint.instrumentation.ast import (
    AvalancheExpr,
    BinaryExpr,
    Expr,
    InstructionCellExpr,
    LogicCircuit,
    MemoryOperand,
    Op,
    TaintAssignment,
    TaintOperand,
    UnaryExpr,
)
from microtaint.sleigh.lifter import get_context
from microtaint.sleigh.mapper import determine_category
from microtaint.sleigh.polarity import compute_polarity
from microtaint.sleigh.slicer import _get_varnode_id, slice_backward
from microtaint.types import Architecture, Register


def _map_sleigh_to_state(
    ctx: pypcode.Context,
    arch: str,
    state_format: list[Register],
    offset: int,
    size: int,
) -> tuple[str, int, int] | None:
    # Handle X86 flags abstract offsets (512-540)
    if 'X86' in arch.upper() and 512 <= offset < 550:
        for sf_reg in state_format:
            if 'FLAGS' in sf_reg.name.upper():
                bit_idx = offset - 512
                # Flags are mapped as 1-bit internally despite Sleigh giving them size=1 (byte)
                return sf_reg.name, bit_idx, bit_idx

    for sf_reg in state_format:
        s_r = ctx.registers.get(sf_reg.name) or ctx.registers.get(sf_reg.name.lower())
        if not s_r:
            continue

        # Check if the requested register falls within this state_format register
        if s_r.offset <= offset and (offset + size) <= (s_r.offset + s_r.size):
            rel_byte = offset - s_r.offset
            bit_start = rel_byte * 8
            bit_end = bit_start + (size * 8) - 1
            return sf_reg.name, bit_start, bit_end

    return None


def generate_static_rule(  # noqa: C901
    arch: Architecture,
    bytestring: bytes,
    state_format: list[Register],
) -> LogicCircuit:
    """
    Statically analyzes an instruction using SLEIGH and generates
    the inferred logic circuit with D-vectors.
    """
    ctx = get_context(arch)
    translation = ctx.translate(bytestring, 0x1000)

    outputs = []
    store_ops = []
    for op in translation.ops:
        if op.output and op.output.space.name == 'register':
            outputs.append(op.output)
        if op.opcode.name == 'STORE':
            store_ops.append(op)

    unique_outputs = {_get_varnode_id(out): out for out in outputs}.values()

    targets_to_evaluate: list[tuple[pypcode.pypcode_native.Varnode, str, int, int]] = []

    mem_targets: list[tuple[pypcode.pypcode_native.Varnode, pypcode.pypcode_native.Varnode, int]] = []
    for out_vn in unique_outputs:
        mapped_out = _map_sleigh_to_state(ctx, arch, state_format, out_vn.offset, out_vn.size)
        if mapped_out:
            targets_to_evaluate.append((out_vn, mapped_out[0], mapped_out[1], mapped_out[2]))

    for store_op in store_ops:
        ptr_vn = store_op.inputs[1]
        val_vn = store_op.inputs[2]
        mem_targets.append((val_vn, ptr_vn, val_vn.size))

    # Add branch implicits (CBRANCH, BRANCHIND, CALLIND) mapping to Program Counter (PC)
    for op in translation.ops:
        op_name = op.opcode.name  # type: ignore[attr-defined]
        if op_name in ('CBRANCH', 'BRANCHIND', 'CALLIND'):
            pc_name = 'EIP' if 'X86' in arch.upper() else 'RIP' if 'AMD64' in arch.upper() else 'PC'
            pc_reg = next((r for r in state_format if r.name.upper() == pc_name), None)
            if not pc_reg:
                continue

            varnode = op.inputs[1] if op_name == 'CBRANCH' else op.inputs[0]

            # Discard if it is strictly a constant dictating the branch
            if varnode.space.name == 'const':
                continue

            targets_to_evaluate.append((varnode, pc_reg.name, 0, pc_reg.bits - 1))

    assignments: list[TaintAssignment] = []

    # Map mem targets explicitly
    for val_vn, ptr_vn, size in mem_targets:
        mapped_addr = _map_sleigh_to_state(ctx, arch, state_format, ptr_vn.offset, ptr_vn.size)
        if mapped_addr:
            fake_target_name = f'MEM_STORE_{ptr_vn.offset}_{size}_{mapped_addr[0]}'
            targets_to_evaluate.append((val_vn, fake_target_name, 0, size * 8 - 1))

    for out_vn, out_name, out_bit_start, out_bit_end in targets_to_evaluate:
        slice_ops = slice_backward(translation.ops, out_vn)
        cat = determine_category(slice_ops)
        polarities = compute_polarity(slice_ops)

        deps: dict[tuple[str, int, int], int] = {}

        # If the output varnode was never produced by any operation in this instruction,
        # it is intrinsically its own direct read dependency (e.g. CBRANCH reading flags).
        if not slice_ops:
            if out_vn.space.name == 'register':
                mapped_dep = _map_sleigh_to_state(ctx, arch, state_format, out_vn.offset, out_vn.size)
                if mapped_dep:
                    deps[mapped_dep] = 1
        else:
            for op in slice_ops:
                if op.opcode.name == 'LOAD':
                    ptr_vn = op.inputs[1]
                    mapped_addr = _map_sleigh_to_state(ctx, arch, state_format, ptr_vn.offset, ptr_vn.size)
                    if mapped_addr:
                        deps[('MEM', ptr_vn.offset, ptr_vn.size, mapped_addr)] = 1

            for vn_id, p in polarities.items():
                parts = vn_id.split(':')
                if len(parts) != 3:
                    continue
                space, st_offset, st_size = parts
                if space == 'register':
                    mapped_dep = _map_sleigh_to_state(ctx, arch, state_format, int(st_offset), int(st_size))
                    if mapped_dep:
                        deps[mapped_dep] = p

        if out_name.startswith('MEM_STORE_'):
            parts = out_name.split('_')
            # offset = int(parts[2])  # Not used yet
            size_b = int(parts[3])
            mapped_addr_name = parts[4]
            # Since Sleigh doesn't tell us bit sizing perfectly for memory yet,
            # assume whole register mapping for pointer
            addr_expr = TaintOperand(mapped_addr_name, 0, 63, is_taint=False)  # Quick hack: 64-bit addr
            target = MemoryOperand(addr_expr, size_b, is_taint=True)
        else:
            target = TaintOperand(out_name, out_bit_start, out_bit_end, is_taint=True)

        # If there are no register dependencies (e.g., MOV reg, constant),
        # create an assignment with an InstructionCellExpr that has no inputs.
        # This will correctly evaluate to 0 taint.
        if not deps:
            in_dict: dict[str, Expr] = {}
            expression = InstructionCellExpr(arch, bytestring.hex(), out_name, out_bit_start, out_bit_end, in_dict)
            assignments.append(TaintAssignment(target=target, dependencies=[], expression=expression))
            continue

        dependencies = []
        dependency_names = []  # Track dep names for cell inputs
        cell_inputs_rep1: dict[str, Expr] = {}
        cell_inputs_rep2: dict[str, Expr] = {}

        for k, p in deps.items():
            if isinstance(k, tuple) and len(k) == 4 and k[0] == 'MEM':
                _, _m_off, m_size, mapped_addr = k
                addr_expr = TaintOperand(mapped_addr[0], mapped_addr[1], mapped_addr[2], is_taint=False)
                T_in = MemoryOperand(addr_expr, m_size, is_taint=True)
                V_in = MemoryOperand(addr_expr, m_size, is_taint=False)
                dep_name = f'MEM_{mapped_addr[0]}'
            else:
                dep_name, r_min, r_max = k
                T_in = TaintOperand(dep_name, r_min, r_max, is_taint=True)
                V_in = TaintOperand(dep_name, r_min, r_max, is_taint=False)
            dependencies.append(T_in)
            dependency_names.append(dep_name)
            v_and_not_t = BinaryExpr(Op.AND, V_in, UnaryExpr(Op.NOT, T_in))

            if p == 1:
                rep1_expr = BinaryExpr(Op.OR, V_in, T_in)
                rep2_expr = v_and_not_t
            else:
                rep1_expr = v_and_not_t
                rep2_expr = BinaryExpr(Op.OR, V_in, T_in)

            # In Cell inputs we just use the name if it is disjoint, but here multiple pieces of same reg could be used
            # We assume disjoint parent registers mapping for cell formulas simplification
            cell_inputs_rep1[dep_name] = rep1_expr
            cell_inputs_rep2[dep_name] = rep2_expr

        if cat == InstructionCategory.MAPPED:
            # For pure bitwise operations (AND, OR, XOR) or operations with constants,
            # use cell simulation to get bit-precise results
            has_bitwise_ops = any(op.opcode.name in ('INT_AND', 'INT_OR', 'INT_XOR') for op in slice_ops)  # type: ignore[attr-defined]

            if has_bitwise_ops or not dependencies:
                # Use cell simulation for bit-precise handling of constants
                in_dict: dict[str, Expr] = {dependency_names[i]: dependencies[i] for i in range(len(dependencies))}
                expr = InstructionCellExpr(arch, bytestring.hex(), out_name, out_bit_start, out_bit_end, in_dict)
            else:
                # For simple register copies, OR the dependencies
                expr = dependencies[0]
                for dep in dependencies[1:]:
                    expr = BinaryExpr(Op.OR, expr, dep)

            # Automatically apply Avalanche to program counter branches
            # Because if a branch condition or address is tainted, the entire PC is entirely tainted.
            if out_name in ('EIP', 'RIP', 'PC'):
                expr = AvalancheExpr(expr)

            assignments.append(TaintAssignment(target=target, dependencies=dependencies, expression=expr))

        elif cat == InstructionCategory.AVALANCHE:
            expression: Expr = dependencies[0]
            for dep in dependencies[1:]:
                expression = BinaryExpr(Op.OR, expression, dep)
            expression = AvalancheExpr(expression)
            assignments.append(TaintAssignment(target=target, dependencies=dependencies, expression=expression))

        elif cat == InstructionCategory.TRANSPORTABLE:
            C1_cell = InstructionCellExpr(
                arch,
                bytestring.hex(),
                out_name,
                out_bit_start,
                out_bit_end,
                cell_inputs_rep1,
            )
            C2_cell = InstructionCellExpr(
                arch,
                bytestring.hex(),
                out_name,
                out_bit_start,
                out_bit_end,
                cell_inputs_rep2,
            )

            # Bit-precise differential: XOR of high and low flip simulations
            expression = BinaryExpr(Op.XOR, C1_cell, C2_cell)

            # Note: Removed transport_term OR for bit-precise taint tracking.
            # The differential alone correctly captures bit-level propagation.

            if out_name in ('EIP', 'RIP', 'PC'):
                expression = AvalancheExpr(expression)
            assignments.append(TaintAssignment(target=target, dependencies=dependencies, expression=expression))
        else:
            in_dict: dict[str, Expr] = {d.name: d for d in dependencies}
            expression = InstructionCellExpr(arch, bytestring.hex(), out_name, out_bit_start, out_bit_end, in_dict)
            if out_name in ('EIP', 'RIP', 'PC'):
                expression = AvalancheExpr(expression)
            assignments.append(TaintAssignment(target=target, dependencies=dependencies, expression=expression))

    return LogicCircuit(
        assignments=assignments,
        architecture=arch,
        instruction=bytestring.hex(),
        state_format=state_format,
    )
