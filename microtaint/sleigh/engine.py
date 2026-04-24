from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from pypcode import Context, PcodeOp, Translation, Varnode

from microtaint.classifier.categories import InstructionCategory
from microtaint.instrumentation.ast import (
    AvalancheExpr,
    BinaryExpr,
    Constant,
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
from microtaint.sleigh.mapper import EXTENSION_OPCODES, TRANSLATABLE_OPCODES, determine_category
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
        bit_idx = offset - 512
        flag_names = {0: 'CF', 2: 'PF', 6: 'ZF', 7: 'SF', 11: 'OF'}

        # 1. Prioritize explicit flag targets if requested (e.g., 'ZF' directly)
        for sf_reg in state_format:
            if sf_reg.name.upper() == flag_names.get(bit_idx):
                return RegMapping(sf_reg.name, 0, 0)

        # 2. Fallback to mapping into the parent EFLAGS register
        for sf_reg in state_format:
            if 'FLAGS' in sf_reg.name.upper():
                return RegMapping(sf_reg.name, bit_idx, bit_idx)

    best_match = None
    for sf_reg in state_format:
        s_r = ctx.registers.get(sf_reg.name) or ctx.registers.get(sf_reg.name.lower())
        if not s_r:
            continue

        # Check if the requested register falls within this state_format register
        if s_r.offset <= offset and (offset + size) <= (s_r.offset + s_r.size):
            rel_byte = offset - s_r.offset
            bit_start = rel_byte * 8
            bit_end = min(bit_start + (size * 8) - 1, sf_reg.bits - 1)
            mapping = RegMapping(sf_reg.name, bit_start, bit_end)

            if s_r.offset == offset and s_r.size == size:
                return mapping

            if not best_match or s_r.size > best_match[1]:
                best_match = (mapping, s_r.size)

    return best_match[0] if best_match else None


def _map_sleigh_to_state_all(
    ctx: Context,
    arch: str,
    state_format: list[Register],
    offset: int,
    size: int,
) -> list[RegMapping]:
    mappings: list[RegMapping] = []
    if 'X86' in arch.upper() and 512 <= offset < 560:
        bit_idx = offset - 512
        flag_names = {0: 'CF', 2: 'PF', 6: 'ZF', 7: 'SF', 11: 'OF'}
        for sf_reg in state_format:
            if 'FLAGS' in sf_reg.name.upper():
                mappings.append(RegMapping(sf_reg.name, bit_idx, bit_idx))
            elif sf_reg.name.upper() == flag_names.get(bit_idx):
                mappings.append(RegMapping(sf_reg.name, 0, 0))
        return mappings

    for sf_reg in state_format:
        s_r = ctx.registers.get(sf_reg.name) or ctx.registers.get(sf_reg.name.lower())
        if not s_r:
            continue

        overlap_start = max(s_r.offset, offset)
        overlap_end = min(s_r.offset + s_r.size, offset + size)

        if overlap_start < overlap_end:
            rel_byte = overlap_start - s_r.offset
            bit_start = rel_byte * 8
            bit_end = min(bit_start + ((overlap_end - overlap_start) * 8) - 1, sf_reg.bits - 1)
            mappings.append(RegMapping(sf_reg.name, bit_start, bit_end))

    return mappings


def apply_sless_msb_split(
    deps: dict[RegMapping | MemMapping, int],
    slice_ops: list[PcodeOp],
    ctx: Context,
    arch: Architecture,
    state_format: list[Register],
) -> dict[RegMapping | MemMapping, int]:
    """
    Isolates the MSB for Signed Comparisons (INT_SLESS, INT_SLESSEQUAL) and
    inverts its polarity. The paper dictates signed comparisons are bitwise
    non-decreasing EXCEPT for the MSB, which is non-increasing.
    """
    sless_ops = [op for op in slice_ops if op.opcode.name in {'INT_SLESS', 'INT_SLESSEQUAL'}]
    if not sless_ops:
        return deps

    new_deps: dict[RegMapping | MemMapping, int] = {}
    msb_mappings: list[tuple[str, int]] = []

    # Identify the exact architectural bit that represents the MSB of the operands
    for op in sless_ops:
        for vn in (op.inputs[0], op.inputs[1]):
            if vn.space.name == 'register':
                m = _map_sleigh_to_state(ctx, arch, state_format, vn.offset, vn.size)
                if m:
                    # Calculate the MSB offset (Size in bytes * 8 - 1)
                    msb_offset_in_reg = m.bit_start + (vn.size * 8) - 1
                    msb_mappings.append((m.name, msb_offset_in_reg))

    # Apply the split to the dependencies
    for dep_map, p in deps.items():
        if isinstance(dep_map, MemMapping):
            new_deps[dep_map] = p
            continue

        matched_msb = next(
            (
                msb
                for (name, msb) in msb_mappings
                if name == dep_map.name and dep_map.bit_start <= msb <= dep_map.bit_end
            ),
            None,
        )

        if matched_msb is not None:
            # 1. Lower bits: Bitwise non-decreasing (polarity = 1)
            if matched_msb > dep_map.bit_start:
                new_deps[RegMapping(dep_map.name, dep_map.bit_start, matched_msb - 1)] = 1

            # 2. MSB: Bitwise non-increasing (polarity = -1)
            new_deps[RegMapping(dep_map.name, matched_msb, matched_msb)] = -1

            # 3. Upper bits (if mapping extends beyond the SLESS operand size)
            if dep_map.bit_end > matched_msb:
                new_deps[RegMapping(dep_map.name, matched_msb + 1, dep_map.bit_end)] = 1
        else:
            new_deps[dep_map] = p

    return new_deps


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

    outputs, store_ops = get_register_outputs_and_stores(translation)

    unique_outputs = {get_varnode_id(out): out for out in outputs}.values()

    targets_to_evaluate, assignments = map_outputs_to_targets(
        arch,
        state_format,
        ctx,
        translation,
        store_ops,
        unique_outputs,
    )

    for target in targets_to_evaluate:
        out_vn = target.varnode
        mapping = target.mapping

        slice_ops = slice_backward(translation.ops, out_vn)
        polarities = compute_polarity(slice_ops)

        # FIX: Pass translation.ops down to completely bypass slicer truncation
        deps = extract_dependencies(arch, state_format, ctx, out_vn, slice_ops, polarities, translation.ops)
        deps = apply_sless_msb_split(deps, slice_ops, ctx, arch, state_format)

        out_target, out_name, out_bit_start, out_bit_end = generate_output_target(mapping)

        generate_taint_assignments(
            arch,
            bytestring,
            assignments,
            slice_ops,
            deps,
            out_target,
            out_name,
            out_bit_start,
            out_bit_end,
            ctx,
            state_format,
        )

    return LogicCircuit(
        assignments=assignments,
        architecture=arch,
        instruction=bytestring.hex(),
        state_format=state_format,
    )


def generate_taint_assignments(  # noqa: C901
    arch: Architecture,
    bytestring: bytes,
    assignments: list[TaintAssignment],
    slice_ops: list[PcodeOp],
    deps: dict[RegMapping | MemMapping, int],
    out_target: TaintOperand | MemoryOperand,
    out_name: str,
    out_bit_start: int,
    out_bit_end: int,
    ctx: Context,
    state_format: list[Register],
) -> None:
    dependencies, dependency_names, cell_inputs_rep1, cell_inputs_rep2 = process_dependencies(deps)

    if not dependencies:
        expr: Expr = Constant(0, out_bit_end - out_bit_start + 1)
        if out_name in ('EIP', 'RIP', 'PC'):
            expr = AvalancheExpr(expr, out_bit_end - out_bit_start + 1)
        assignments.append(TaintAssignment(target=out_target, dependencies=[], expression=expr))
        return

    cat = determine_category(slice_ops)

    def make_differential() -> Expr:
        C1_cell = InstructionCellExpr(arch, bytestring.hex(), out_name, out_bit_start, out_bit_end, cell_inputs_rep1)
        C2_cell = InstructionCellExpr(arch, bytestring.hex(), out_name, out_bit_start, out_bit_end, cell_inputs_rep2)
        return BinaryExpr(Op.XOR, C1_cell, C2_cell)

    if cat == InstructionCategory.AVALANCHE:
        expr = dependencies[0]
        for dep in dependencies[1:]:
            expr = BinaryExpr(Op.OR, expr, dep)
        expr = AvalancheExpr(expr, out_bit_end - out_bit_start + 1)

    elif cat == InstructionCategory.TRANSLATABLE:
        diff_expr = make_differential()

        core_ops = [op for op in slice_ops if op.opcode.name not in EXTENSION_OPCODES]
        shift_op = next((op for op in core_ops if op.opcode.name in TRANSLATABLE_OPCODES), None)

        # Recursive helper to unwind SLEIGH's `unique` micro-registers
        def trace_origins(vn: Varnode, visited: set[int] | None = None) -> set[str]:
            if visited is None:
                visited = set()
            origins: set[str] = set()

            if vn.space.name == 'register':
                m = _map_sleigh_to_state(ctx, arch, state_format, vn.offset, vn.size)
                if m:
                    origins.add(m.name)
            elif vn.space.name == 'unique':
                if vn.offset in visited:
                    return origins
                visited.add(vn.offset)

                # Find the P-code operation that generated this unique varnode
                for op in slice_ops:
                    if op.output and op.output.space.name == 'unique' and op.output.offset == vn.offset:
                        if op.opcode.name == 'LOAD':
                            # If the shift offset came from memory
                            ptr_vn = op.inputs[1]
                            m = _map_sleigh_to_state(ctx, arch, state_format, ptr_vn.offset, ptr_vn.size)
                            if m:
                                origins.add(f'MEM_{m.name}')
                        else:
                            # Recurse through the inputs of the micro-operation (e.g., INT_AND)
                            for inp in op.inputs:
                                origins.update(trace_origins(inp, visited))
                        break
            return origins

        offset_names: set[str] = set()
        # In SLEIGH shifts, inputs[1] is strictly the shift offset amount
        if shift_op and len(shift_op.inputs) > 1:
            offset_names = trace_origins(shift_op.inputs[1])

        # Exclude the primary data source from the fallback avalanche to avoid false positives
        primary_input_name = None
        if shift_op and shift_op.inputs[0].space.name == 'register':
            m = _map_sleigh_to_state(ctx, arch, state_format, shift_op.inputs[0].offset, shift_op.inputs[0].size)
            if m:
                primary_input_name = m.name

        if not offset_names:
            offset_names = {name for name in dependency_names if name not in (out_name, primary_input_name)}

        offset_taints = [dep for dep, name in zip(dependencies, dependency_names) if name in offset_names]

        # If any originating source for the shift offset is tainted, avalanche the output!
        if offset_taints:
            combined_offset = offset_taints[0]
            for t in offset_taints[1:]:
                combined_offset = BinaryExpr(Op.OR, combined_offset, t)

            avalanche_shift = AvalancheExpr(combined_offset, out_bit_end - out_bit_start + 1)
            expr = BinaryExpr(Op.OR, diff_expr, avalanche_shift)
        else:
            expr = diff_expr

    elif cat == InstructionCategory.COND_TRANSPORTABLE:
        # Implements precise Conditional Transportability from the CELLIFT paper:
        # Y^t = C(A \wedge T_AB, B \wedge T_AB) \wedge \bigvee I^t
        # Where T_AB is the mask of bits NOT tainted in ANY input.

        # 1. Create T_union (A^t \vee B^t)
        T_union = dependencies[0]
        for dep in dependencies[1:]:
            T_union = BinaryExpr(Op.OR, T_union, dep)

        masked_inputs: dict[str, Expr] = {}
        # 2. Construct (V_in \wedge ~T_union) for all inputs
        for dep_map in deps.keys():
            if isinstance(dep_map, MemMapping):
                addr_expr = TaintOperand(
                    dep_map.addr_reg.name,
                    dep_map.addr_reg.bit_start,
                    dep_map.addr_reg.bit_end,
                    is_taint=False,
                )
                V_in: Expr = MemoryOperand(addr_expr, dep_map.size_bytes, is_taint=False)
                dep_name = f'MEM_{dep_map.addr_reg.name}'
            else:
                V_in = TaintOperand(dep_map.name, dep_map.bit_start, dep_map.bit_end, is_taint=False)
                dep_name = dep_map.name

            masked_V = BinaryExpr(Op.AND, V_in, UnaryExpr(Op.NOT, T_union))
            masked_inputs[dep_name] = masked_V

        # 3. Evaluate the cell using the masked values: C(A \wedge T_AB, B \wedge T_AB)
        C_eval = InstructionCellExpr(
            arch,
            bytestring.hex(),
            out_name,
            out_bit_start,
            out_bit_end,
            masked_inputs,
        )

        # 4. Avalanche T_union to 1-bit to represent \bigvee I^t (Is ANY bit tainted?)
        T_any = AvalancheExpr(T_union, out_bit_end - out_bit_start + 1)

        # 5. Final Boolean AND
        expr = BinaryExpr(Op.AND, C_eval, T_any)

    elif cat == InstructionCategory.TRANSPORTABLE:
        diff_expr = make_differential()
        # Transportable cells require the transport term (OR of inputs) added back
        # EXCEPTION: 1-bit boolean flags evaluate their conditions perfectly via the differential.
        is_flag = out_bit_end == out_bit_start

        if dependencies and not is_flag:
            transport_term = dependencies[0]
            for dep in dependencies[1:]:
                transport_term = BinaryExpr(Op.OR, transport_term, dep)
            expr = BinaryExpr(Op.OR, diff_expr, transport_term)
        else:
            expr = diff_expr

    elif cat == InstructionCategory.MAPPED:
        # Since the heuristic guarantees exactly ONE dynamic dependency,
        # the taint output is simply the instruction itself applied to the input taint.

        # We use rep1_expr from cell_inputs_rep1 because we want to pass the
        # actual Taint vector directly through the instruction's formula.
        expr = InstructionCellExpr(
            arch,
            bytestring.hex(),
            out_name,
            out_bit_start,
            out_bit_end,
            {dependency_names[0]: dependencies[0]},  # Apply instruction directly to the Taint
        )

    elif cat == InstructionCategory.ORABLE:
        core_ops = [op for op in slice_ops if op.opcode.name not in EXTENSION_OPCODES]
        xor_ops = [op for op in core_ops if op.opcode.name == 'INT_XOR']

        # Detect the XOR zeroing idiom: XOR EAX, EAX
        is_zeroing_idiom = False
        if xor_ops:
            xor_op = xor_ops[0]
            in1, in2 = xor_op.inputs[0], xor_op.inputs[1]
            if in1.space == in2.space and in1.offset == in2.offset and in1.size == in2.size:
                is_zeroing_idiom = True

        if is_zeroing_idiom:
            expr = Constant(0, out_bit_end - out_bit_start + 1)
        elif dependencies:
            expr = dependencies[0]
            for dep in dependencies[1:]:
                expr = BinaryExpr(Op.OR, expr, dep)
        else:
            expr = Constant(0, out_bit_end - out_bit_start + 1)

    elif cat == InstructionCategory.MONOTONIC:
        expr = make_differential()

    else:
        raise ValueError(f'Unsupported instruction category: {cat}')

    if out_name in ('EIP', 'RIP', 'PC'):
        expr = AvalancheExpr(expr, out_bit_end - out_bit_start + 1)

    assignments.append(TaintAssignment(target=out_target, dependencies=dependencies, expression=expr))


def process_dependencies(
    deps: dict[RegMapping | MemMapping, int],
) -> tuple[list[Expr], list[str], dict[str, Expr], dict[str, Expr]]:
    dependencies: list[Expr] = []
    dependency_names: list[str] = []
    cell_inputs_rep1: dict[str, Expr] = {}
    cell_inputs_rep2: dict[str, Expr] = {}

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
    return dependencies, dependency_names, cell_inputs_rep1, cell_inputs_rep2


def generate_output_target(mapping: RegMapping | MemMapping) -> tuple[TaintOperand | MemoryOperand, str, int, int]:
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
    return out_target, out_name, out_bit_start, out_bit_end


def extract_dependencies(
    arch: Architecture,
    state_format: list[Register],
    ctx: Context,
    _out_vn: Varnode,
    _slice_ops: list[PcodeOp],
    polarities: dict[str, int],
    all_ops: list[PcodeOp],
) -> dict[RegMapping | MemMapping, int]:
    deps: dict[RegMapping | MemMapping, int] = {}

    # FIX: Scan ALL ops to completely bypass buggy slicer truncation.
    # Instruction taint is atomic, so any read register contributes to the final state.
    for op in all_ops:
        if op.opcode.name == 'LOAD':
            ptr_vn = op.inputs[1]
            mapped_addr = _map_sleigh_to_state(ctx, arch, state_format, ptr_vn.offset, ptr_vn.size)
            if mapped_addr:
                deps[MemMapping(ptr_vn.offset, ptr_vn.size, mapped_addr)] = 1

        for vn in op.inputs:
            if vn.space.name == 'register':
                mapped_dep = _map_sleigh_to_state(ctx, arch, state_format, vn.offset, vn.size)
                if mapped_dep and mapped_dep not in deps:
                    deps[mapped_dep] = 1  # Default to positive polarity

        # 3. Override with exact, mathematically computed polarities where available
    for vn_id, p in polarities.items():
        parts = vn_id.split(':')
        if len(parts) == 3 and parts[0] == 'register':
            mapped_dep = _map_sleigh_to_state(ctx, arch, state_format, int(parts[1]), int(parts[2]))
            if mapped_dep:
                deps[mapped_dep] = p

    return deps


def map_outputs_to_targets(
    arch: Architecture,
    state_format: list[Register],
    ctx: Context,
    translation: Translation,
    store_ops: list[PcodeOp],
    unique_outputs: Iterable[Varnode],
) -> tuple[list[EvalTarget], list[TaintAssignment]]:
    targets_to_evaluate: list[EvalTarget] = []
    mem_targets: list[tuple[Varnode, Varnode, int]] = []

    for out_vn in unique_outputs:
        mapped_outs = _map_sleigh_to_state_all(ctx, arch, state_format, out_vn.offset, out_vn.size)
        for mapped_out in mapped_outs:
            targets_to_evaluate.append(EvalTarget(out_vn, mapped_out))

    for store_op in store_ops:
        ptr_vn = store_op.inputs[1]
        val_vn = store_op.inputs[2]
        mem_targets.append((val_vn, ptr_vn, val_vn.size))

    for op in translation.ops:
        op_name = op.opcode.name
        if op_name in ('CBRANCH', 'BRANCHIND', 'CALLIND'):
            pc_name = 'EIP' if 'X86' in arch.upper() else 'RIP' if 'AMD64' in arch.upper() else 'PC'
            pc_reg = next((r for r in state_format if r.name.upper() == pc_name), None)
            if not pc_reg:
                continue

            varnode = op.inputs[1] if op_name == 'CBRANCH' else op.inputs[0]
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

    return targets_to_evaluate, assignments


def get_register_outputs_and_stores(translation: Translation) -> tuple[list[Varnode], list[PcodeOp]]:
    outputs: list[Varnode] = []
    store_ops: list[PcodeOp] = []
    for op in translation.ops:
        if op.output and op.output.space.name == 'register':
            outputs.append(op.output)
        if op.opcode.name == 'STORE':
            store_ops.append(op)
    return outputs, store_ops
