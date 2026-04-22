from __future__ import annotations

from dataclasses import dataclass

from pypcode import Context, PcodeOp, Varnode

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
from microtaint.sleigh.slicer import get_varnode_id, slice_backward
from microtaint.types import Architecture, Register


@dataclass(frozen=True)
class RegMapping:
    name: str
    bit_start: int
    bit_end: int


@dataclass(frozen=True)
class MemMapping:
    offset: int
    size_bytes: int
    addr_reg: RegMapping


@dataclass
class EvalTarget:
    varnode: Varnode
    mapping: RegMapping | MemMapping


def _map_sleigh_to_state(
    ctx: Context,
    arch: str,
    state_format: list[Register],
    offset: int,
    size: int,
) -> RegMapping | None:
    # Handle X86 flags abstract offsets (512-540)
    if 'X86' in arch.upper() and 512 <= offset < 560:
        for sf_reg in state_format:
            if 'FLAGS' in sf_reg.name.upper():
                bit_idx = offset - 512
                # Flags are mapped as 1-bit internally despite Sleigh giving them size=1 (byte)
                return RegMapping(sf_reg.name, bit_idx, bit_idx)

    for sf_reg in state_format:
        s_r = ctx.registers.get(sf_reg.name) or ctx.registers.get(sf_reg.name.lower())
        if not s_r:
            continue

        # Check if the requested register falls within this state_format register
        if s_r.offset <= offset and (offset + size) <= (s_r.offset + s_r.size):
            rel_byte = offset - s_r.offset
            bit_start = rel_byte * 8
            bit_end = bit_start + (size * 8) - 1
            return RegMapping(sf_reg.name, bit_start, bit_end)

    return None


def generate_static_rule(
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

    outputs: list[Varnode] = []
    store_ops: list[PcodeOp] = []
    for op in translation.ops:
        if op.output and op.output.space.name == 'register':
            outputs.append(op.output)
        if op.opcode.name == 'STORE':
            store_ops.append(op)

    unique_outputs = {get_varnode_id(out): out for out in outputs}.values()

    targets_to_evaluate: list[EvalTarget] = []
    mem_targets: list[tuple[Varnode, Varnode, int]] = []

    for out_vn in unique_outputs:
        mapped_out = _map_sleigh_to_state(ctx, arch, state_format, out_vn.offset, out_vn.size)
        if mapped_out:
            targets_to_evaluate.append(EvalTarget(out_vn, mapped_out))

    for store_op in store_ops:
        ptr_vn = store_op.inputs[1]
        val_vn = store_op.inputs[2]
        mem_targets.append((val_vn, ptr_vn, val_vn.size))

    # Add branch implicits (CBRANCH, BRANCHIND, CALLIND) mapping to Program Counter (PC)
    for op in translation.ops:
        op_name = op.opcode.name
        if op_name in ('CBRANCH', 'BRANCHIND', 'CALLIND'):
            pc_name = 'EIP' if 'X86' in arch.upper() else 'RIP' if 'AMD64' in arch.upper() else 'PC'
            pc_reg = next((r for r in state_format if r.name.upper() == pc_name), None)
            if not pc_reg:
                continue

            varnode = op.inputs[1] if op_name == 'CBRANCH' else op.inputs[0]

            # Discard if it is strictly a constant dictating the branch
            if varnode.space.name == 'const':
                continue

            targets_to_evaluate.append(EvalTarget(varnode, RegMapping(pc_reg.name, 0, pc_reg.bits - 1)))

    assignments: list[TaintAssignment] = []

    # Map mem targets explicitly
    for val_vn, ptr_vn, size in mem_targets:
        mapped_addr = _map_sleigh_to_state(ctx, arch, state_format, ptr_vn.offset, ptr_vn.size)
        if mapped_addr:
            mem_map = MemMapping(ptr_vn.offset, size, mapped_addr)
            targets_to_evaluate.append(EvalTarget(val_vn, mem_map))

    for target in targets_to_evaluate:
        out_vn = target.varnode
        mapping = target.mapping

        slice_ops = slice_backward(translation.ops, out_vn)
        cat = determine_category(slice_ops)
        polarities = compute_polarity(slice_ops)

        deps: dict[RegMapping | MemMapping, int] = {}

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
                        deps[MemMapping(ptr_vn.offset, ptr_vn.size, mapped_addr)] = 1

            for vn_id, p in polarities.items():
                parts = vn_id.split(':')
                if len(parts) != 3:
                    continue
                space, st_offset, st_size = parts
                if space == 'register':
                    mapped_dep = _map_sleigh_to_state(ctx, arch, state_format, int(st_offset), int(st_size))
                    if mapped_dep:
                        deps[mapped_dep] = p

        out_target: TaintOperand | MemoryOperand
        if isinstance(mapping, MemMapping):
            addr_expr = TaintOperand(mapping.addr_reg.name, 0, 63, is_taint=False)  # Quick hack: 64-bit addr
            out_target = MemoryOperand(addr_expr, mapping.size_bytes, is_taint=True)
            out_name = f'MEM_{mapping.addr_reg.name}'
            out_bit_start, out_bit_end = 0, (mapping.size_bytes * 8) - 1
        else:
            out_target = TaintOperand(mapping.name, mapping.bit_start, mapping.bit_end, is_taint=True)
            out_name = mapping.name
            out_bit_start, out_bit_end = mapping.bit_start, mapping.bit_end

        # If there are no register dependencies (e.g., MOV reg, constant),
        # create an assignment with an InstructionCellExpr that has no inputs.
        # This will correctly evaluate to 0 taint.
        if not deps:
            empty_in_dict: dict[str, Expr] = {}
            expression = InstructionCellExpr(
                arch,
                bytestring.hex(),
                out_name,
                out_bit_start,
                out_bit_end,
                empty_in_dict,
            )
            assignments.append(TaintAssignment(target=out_target, dependencies=[], expression=expression))
            continue

        dependencies: list[Expr] = []
        dependency_names: list[str] = []
        cell_inputs_rep1: dict[str, Expr] = {}
        cell_inputs_rep2: dict[str, Expr] = {}

        # -- Process Dependencies Cleanly --
        for dep_map, p in deps.items():
            T_in: Expr
            V_in: Expr
            if isinstance(dep_map, MemMapping):
                addr_expr = TaintOperand(
                    dep_map.addr_reg.name,
                    dep_map.addr_reg.bit_start,
                    dep_map.addr_reg.bit_end,
                    is_taint=False,
                )
                T_in = MemoryOperand(addr_expr, dep_map.size_bytes, is_taint=True)
                V_in = MemoryOperand(addr_expr, dep_map.size_bytes, is_taint=False)
                dep_name = f'MEM_{dep_map.addr_reg.name}'
            else:
                T_in = TaintOperand(dep_map.name, dep_map.bit_start, dep_map.bit_end, is_taint=True)
                V_in = TaintOperand(dep_map.name, dep_map.bit_start, dep_map.bit_end, is_taint=False)
                dep_name = dep_map.name

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
            has_bitwise_ops = any(op.opcode.name in ('INT_AND', 'INT_OR', 'INT_XOR') for op in slice_ops)

            if has_bitwise_ops or not dependencies:
                mapped_in_dict = {dependency_names[i]: dependencies[i] for i in range(len(dependencies))}
                expr: Expr = InstructionCellExpr(
                    arch,
                    bytestring.hex(),
                    out_name,
                    out_bit_start,
                    out_bit_end,
                    mapped_in_dict,
                )
            else:
                # For simple register copies, OR the dependencies
                expr = dependencies[0]
                for dep in dependencies[1:]:
                    expr = BinaryExpr(Op.OR, expr, dep)

            # Automatically apply Avalanche to program counter branches
            # Because if a branch condition or address is tainted, the entire PC is entirely tainted.
            if out_name in ('EIP', 'RIP', 'PC'):
                expr = AvalancheExpr(expr)

            assignments.append(TaintAssignment(target=out_target, dependencies=dependencies, expression=expr))

        elif cat == InstructionCategory.AVALANCHE:
            expr = dependencies[0]
            for dep in dependencies[1:]:
                expr = BinaryExpr(Op.OR, expr, dep)
            expr = AvalancheExpr(expr)
            assignments.append(TaintAssignment(target=out_target, dependencies=dependencies, expression=expr))

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
            expr = BinaryExpr(Op.XOR, C1_cell, C2_cell)

            # Should transport term be added back (for transportable instructions) ?

            if out_name in ('EIP', 'RIP', 'PC'):
                expr = AvalancheExpr(expr)
            assignments.append(TaintAssignment(target=out_target, dependencies=dependencies, expression=expr))

        else:
            default_in_dict: dict[str, Expr] = {dependency_names[i]: dependencies[i] for i in range(len(dependencies))}
            expr = InstructionCellExpr(arch, bytestring.hex(), out_name, out_bit_start, out_bit_end, default_in_dict)
            if out_name in ('EIP', 'RIP', 'PC'):
                expr = AvalancheExpr(expr)
            assignments.append(TaintAssignment(target=out_target, dependencies=dependencies, expression=expr))

    return LogicCircuit(
        assignments=assignments,
        architecture=arch,
        instruction=bytestring.hex(),
        state_format=state_format,
    )
