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


@dataclass(frozen=True, slots=True)
class RegMapping:
    name: str
    bit_start: int
    bit_end: int


@dataclass(frozen=True, slots=True)
class MemMapping:
    offset: int
    size_bytes: int
    addr_reg: RegMapping


@dataclass(slots=True)
class EvalTarget:
    varnode: Varnode
    mapping: RegMapping | MemMapping


class StateMapper:
    def __init__(self, ctx: Context, arch: str, state_format: list[Register]):
        self.ctx = ctx
        self.arch = arch
        self.state_format = state_format
        self.arm_aliases: dict[str, str] = {'N': 'ng', 'Z': 'zr', 'C': 'cy', 'V': 'ov'}
        self.is_x86 = 'X86' in arch.upper()
        self.is_arm = 'ARM' in arch.upper()

        # Pre-resolve ctx registers to avoid dict lookups in the hot path
        self.sf_resolved: list[tuple[Register, Varnode]] = []
        for sf_reg in state_format:
            s_r = ctx.registers.get(sf_reg.name) or ctx.registers.get(sf_reg.name.lower())
            if not s_r and self.is_arm and sf_reg.name in self.arm_aliases:
                alias = self.arm_aliases[sf_reg.name]
                s_r = ctx.registers.get(alias) or ctx.registers.get(alias.upper())

            if s_r:
                self.sf_resolved.append((sf_reg, s_r))

    def map_to_state(self, offset: int, size: int) -> RegMapping | None:
        if self.is_x86 and 512 <= offset < 560:
            bit_idx = offset - 512
            flag_names = {0: 'CF', 2: 'PF', 6: 'ZF', 7: 'SF', 11: 'OF'}
            requested_flag = flag_names.get(bit_idx)

            for sf_reg in self.state_format:
                if requested_flag and sf_reg.name.upper() == requested_flag:
                    return RegMapping(sf_reg.name, 0, 0)

            for sf_reg in self.state_format:
                if 'FLAGS' in sf_reg.name.upper():
                    return RegMapping(sf_reg.name, bit_idx, bit_idx)

        best_match = None
        end_offset = offset + size
        for sf_reg, s_r in self.sf_resolved:
            if s_r.offset <= offset and end_offset <= (s_r.offset + s_r.size):
                rel_byte = offset - s_r.offset
                bit_start = rel_byte * 8
                bit_end = min(bit_start + (size * 8) - 1, sf_reg.bits - 1)
                mapping = RegMapping(sf_reg.name, bit_start, bit_end)

                if s_r.offset == offset and s_r.size == size:
                    return mapping

                if not best_match or s_r.size < best_match[1]:
                    best_match = (mapping, s_r.size)

        return best_match[0] if best_match else None

    def map_to_state_all(self, offset: int, size: int) -> list[RegMapping]:
        mappings: list[RegMapping] = []
        if self.is_x86 and 512 <= offset < 560:
            bit_idx = offset - 512
            flag_names = {0: 'CF', 2: 'PF', 6: 'ZF', 7: 'SF', 11: 'OF'}
            req_flag = flag_names.get(bit_idx)
            for sf_reg in self.state_format:
                if 'FLAGS' in sf_reg.name.upper():
                    mappings.append(RegMapping(sf_reg.name, bit_idx, bit_idx))
                elif req_flag and sf_reg.name.upper() == req_flag:
                    mappings.append(RegMapping(sf_reg.name, 0, 0))
            return mappings

        end_offset = offset + size
        for sf_reg, s_r in self.sf_resolved:
            overlap_start = max(s_r.offset, offset)
            overlap_end = min(s_r.offset + s_r.size, end_offset)

            if overlap_start < overlap_end:
                rel_byte = overlap_start - s_r.offset
                bit_start = rel_byte * 8
                bit_end = min(bit_start + ((overlap_end - overlap_start) * 8) - 1, sf_reg.bits - 1)
                mappings.append(RegMapping(sf_reg.name, bit_start, bit_end))

        return mappings


def apply_sless_msb_split(
    deps: dict[RegMapping | MemMapping, int],
    slice_ops: list[PcodeOp],
    _ctx: Context,
    _arch: Architecture,
    _state_format: list[Register],
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
        size = op.inputs[0].size
        msb_offset = (size * 8) - 1
        for dep_map in deps.keys():
            if isinstance(dep_map, RegMapping) and dep_map.bit_start <= msb_offset <= dep_map.bit_end:
                msb_mappings.append((dep_map.name, msb_offset))

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

    mapper = StateMapper(ctx, arch, state_format)

    targets_to_evaluate, assignments = map_outputs_to_targets(
        arch,
        state_format,
        translation,
        store_ops,
        unique_outputs,
        mapper,
    )

    for target in targets_to_evaluate:
        out_vn = target.varnode
        mapping = target.mapping

        slice_ops = slice_backward(translation.ops, out_vn)
        polarities = compute_polarity(slice_ops)

        # FIX: Pass translation.ops down to completely bypass slicer truncation
        deps = extract_dependencies(out_vn, slice_ops, polarities, translation.ops, mapper)
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
            mapper,
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
    mapper: StateMapper,
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
        def trace_origins(vn: Varnode, visited: set[int] | None = None) -> set[str]:  # noqa: C901
            if visited is None:
                visited = set()
            origins: set[str] = set()

            if vn.space.name == 'register':
                m = mapper.map_to_state(vn.offset, vn.size)
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
                            m = mapper.map_to_state(ptr_vn.offset, ptr_vn.size)
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
            m = mapper.map_to_state(shift_op.inputs[0].offset, shift_op.inputs[0].size)
            if m:
                primary_input_name = m.name

        if not offset_names:
            offset_names = {name for name in dependency_names if name not in (out_name, primary_input_name)}

        offset_taints = [dep for dep, name in zip(dependencies, dependency_names, strict=True) if name in offset_names]

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
        expr = make_differential()

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


def build_polarized_reg(name: str, slices: list[tuple[int, int, int]], replica_id: int) -> Expr:
    # We construct the full register value by stitching slices
    combined_expr = None
    for s_start, s_end, p in slices:
        V_in = TaintOperand(name, s_start, s_end, is_taint=False)
        T_in = TaintOperand(name, s_start, s_end, is_taint=True)

        # Determine polarization based on p and replica
        # Rep 1: High if p=1, Low if p=-1
        # Rep 2: Low if p=1, High if p=-1
        is_high = (replica_id == 1 and p == 1) or (replica_id == 2 and p <= 0)

        if is_high:
            polarized = BinaryExpr(Op.OR, V_in, T_in)
        else:
            polarized = BinaryExpr(Op.AND, V_in, UnaryExpr(Op.NOT, T_in))
        shifted_polarized = BinaryExpr(Op.LEFT, polarized, Constant(s_start, 8))
        combined_expr = (
            shifted_polarized if combined_expr is None else BinaryExpr(Op.OR, combined_expr, shifted_polarized)
        )
    if combined_expr is None:
        raise ValueError(f'No slices found for register {name}')
    return combined_expr


def process_dependencies(
    deps: dict[RegMapping | MemMapping, int],
) -> tuple[list[Expr], list[str], dict[str, Expr], dict[str, Expr]]:
    """
    Groups dependencies by architectural source and constructs polarized
    replica inputs, correctly stitching split bit-slices (like SLESS MSB).
    """
    dependencies: list[Expr] = []
    dependency_names: list[str] = []

    # Group mappings by their architectural name to handle split slices
    reg_groups: dict[str, list[tuple[int, int, int]]] = {}
    mem_groups: dict[str, list[MemMapping]] = {}

    for dep_map, p in deps.items():
        if isinstance(dep_map, MemMapping):
            key = f'MEM_{dep_map.addr_reg.name}'
            mem_groups.setdefault(key, []).append(dep_map)
        else:
            reg_groups.setdefault(dep_map.name, []).append((dep_map.bit_start, dep_map.bit_end, p))
            # Track unique taints for the Assignment dependencies list
            dependencies.append(TaintOperand(dep_map.name, dep_map.bit_start, dep_map.bit_end, is_taint=True))
            dependency_names.append(dep_map.name)

    cell_inputs_rep1: dict[str, Expr] = {}
    cell_inputs_rep2: dict[str, Expr] = {}

    for name, slices in reg_groups.items():

        cell_inputs_rep1[name] = build_polarized_reg(name, slices, 1)
        cell_inputs_rep2[name] = build_polarized_reg(name, slices, 2)

    # Note: Memory handling remains simple for now as we don't split memory bit-ranges
    for name, mem_list in mem_groups.items():
        m = mem_list[0]
        addr = TaintOperand(m.addr_reg.name, m.addr_reg.bit_start, m.addr_reg.bit_end, is_taint=False)
        T_mem = MemoryOperand(addr, m.size_bytes, is_taint=True)
        V_mem = MemoryOperand(addr, m.size_bytes, is_taint=False)

        # Memory defaults to p=1 for now
        cell_inputs_rep1[name] = BinaryExpr(Op.OR, V_mem, T_mem)
        cell_inputs_rep2[name] = BinaryExpr(Op.AND, V_mem, UnaryExpr(Op.NOT, T_mem))
        dependencies.append(T_mem)
        dependency_names.append(name)

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


def extract_dependencies(  # noqa: C901
    _out_vn: Varnode,
    slice_ops: list[PcodeOp],
    polarities: dict[str, int],
    all_ops: list[PcodeOp],
    mapper: StateMapper,
) -> dict[RegMapping | MemMapping, int]:
    deps: dict[RegMapping | MemMapping, int] = {}

    def resolve_ptr(vn: Varnode, visited: set[int] | None = None) -> RegMapping | None:
        """Recursively unwraps temporary `unique` varnodes back to architectural registers."""
        if visited is None:
            visited = set()
        if vn.space.name == 'register':
            return mapper.map_to_state(vn.offset, vn.size)
        if vn.space.name == 'unique' and vn.offset not in visited:
            visited.add(vn.offset)
            for o in slice_ops:
                if o.output and o.output.offset == vn.offset:
                    for i in o.inputs:
                        res = resolve_ptr(i, visited)
                        if res:
                            return res
        return None

    # FIX: Scan ALL ops to completely bypass buggy slicer truncation.
    # Instruction taint is atomic, so any read register contributes to the final state.
    for op in all_ops:
        if op.opcode.name == 'LOAD':
            ptr_vn = op.inputs[1]
            mapped_addr = resolve_ptr(ptr_vn)
            if mapped_addr:
                deps[MemMapping(ptr_vn.offset, op.output.size if op.output else 8, mapped_addr)] = 1

        for vn in op.inputs:
            if vn.space.name == 'register':
                mapped_dep = mapper.map_to_state(vn.offset, vn.size)
                if mapped_dep and mapped_dep not in deps:
                    deps[mapped_dep] = 1  # Default to positive polarity

        # 3. Override with exact, mathematically computed polarities where available
    for vn_id, p in polarities.items():
        parts = vn_id.split(':')
        if len(parts) == 3 and parts[0] == 'register':
            mapped_dep = mapper.map_to_state(int(parts[1]), int(parts[2]))
            if mapped_dep:
                deps[mapped_dep] = p

    return deps


def map_outputs_to_targets(
    arch: Architecture,
    state_format: list[Register],
    translation: Translation,
    store_ops: list[PcodeOp],
    unique_outputs: Iterable[Varnode],
    mapper: StateMapper,
) -> tuple[list[EvalTarget], list[TaintAssignment]]:
    targets_to_evaluate: list[EvalTarget] = []
    mem_targets: list[tuple[Varnode, Varnode, int]] = []

    for out_vn in unique_outputs:
        mapped_outs = mapper.map_to_state_all(out_vn.offset, out_vn.size)
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
        mapped_addr = mapper.map_to_state(ptr_vn.offset, ptr_vn.size)
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
