from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

from pypcode import Context, PcodeOp, Translation, Varnode

from microtaint.classifier.categories import InstructionCategory
from microtaint.instrumentation.ast import (
    AvalancheExpr,
    BinaryExpr,
    Constant,
    Expr,
    InstructionCellExpr,
    LogicCircuit,
    MemoryDifferentialExpr,
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

_CONST_CACHE: dict[int, Constant] = {}


def _get_zero_constant(size: int) -> Constant:
    if size not in _CONST_CACHE:
        _CONST_CACHE[size] = Constant(0, size)
    return _CONST_CACHE[size]


_OPERAND_CACHE: dict[tuple[str, int, int, bool], TaintOperand] = {}


def _get_taint_operand(name: str, bit_start: int, bit_end: int, is_taint: bool) -> TaintOperand:
    key = (name, bit_start, bit_end, is_taint)
    if key not in _OPERAND_CACHE:
        _OPERAND_CACHE[key] = TaintOperand(name, bit_start, bit_end, is_taint=is_taint)
    return _OPERAND_CACHE[key]


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
    addr_const_offset: int = 0


@dataclass
class EvalTarget:
    varnode: Varnode
    mapping: RegMapping | MemMapping


# ---------------------------------------------------------------------------
# Dependency sets — the core fix lives here.
#
# Previously, extract_dependencies returned a flat dict of all register inputs
# without distinguishing *value* inputs (data being moved) from *address*
# inputs (registers used to compute where to load/store).  This caused:
#
#   push rbp  →  T_MEM = T_RBP | T_RSP   (wrong: RSP is address, not value)
#   mov [rbp-8], rax  →  T_MEM = T_RBP | T_RAX  (wrong: RBP is address)
#
# The fix: extract_dependencies now returns a DependencySet that separates
# value_deps (taint of the data being produced) from addr_deps (taint of
# pointer registers used to compute load/store addresses).
#
# Callers use this to:
#   - Build the value taint expression from value_deps only (for STORE targets)
#   - Build the pointer avalanche expression from addr_deps (for LOAD targets)
#   - Build the differential inputs from value_deps (for arithmetic targets)
# ---------------------------------------------------------------------------


@dataclass
class DependencySet:
    """
    Classified dependencies for a single taint assignment.

    value_deps  : registers/memory that carry the *data* being produced.
                  For a STORE, this is the value being written.
                  For a LOAD, this is the memory content being read.
                  For arithmetic, these are the operands.

    addr_deps   : registers used purely to *address* memory (load pointer,
                  store pointer).  Their taint signals an unknown access
                  target (AIW / tainted-pointer LOAD), not data taint.

    The flat `deps` dict is preserved for the differential and polarity
    machinery that needs the combined view.
    """

    value_deps: dict[RegMapping | MemMapping, int] = field(default_factory=dict[RegMapping | MemMapping, int])
    addr_deps: dict[RegMapping, int] = field(default_factory=dict[RegMapping, int])

    @property
    def all_deps(self) -> dict[RegMapping | MemMapping, int]:
        """Combined view for code that needs both (differential, polarity)."""
        result: dict[RegMapping | MemMapping, int] = {}
        result.update(self.value_deps)
        for k, v in self.addr_deps.items():
            if k not in result:
                result[k] = v
        return result


@dataclass(frozen=True, slots=True)
class _SynthVarnode:
    """Minimal Varnode-shaped object for synthetic state_format entries
    (currently: XMM<n>_LO / XMM<n>_HI).  StateMapper only reads .offset
    and .size from these, never anything else, so the shim is sufficient."""

    offset: int
    size: int


class StateMapper:
    def __init__(self, ctx: Context, arch: str, state_format: list[Register]):
        self.ctx = ctx
        self.arch = arch
        self.state_format = state_format
        self.arm_aliases: dict[str, str] = {'N': 'ng', 'Z': 'zr', 'C': 'cy', 'V': 'ov'}
        # is_x86 covers both "X86" (32-bit) and "AMD64" (64-bit) — they share
        # the Sleigh register space layout (XMM0 at 0x1200, EFLAGS at 0x200, ...).
        arch_upper = str(arch).upper()
        self.is_x86 = 'X86' in arch_upper or 'AMD64' in arch_upper
        self.is_arm = 'ARM' in arch_upper

        self.sf_resolved: list[tuple[Register, Varnode]] = []
        for sf_reg in state_format:
            s_r = ctx.registers.get(sf_reg.name) or ctx.registers.get(sf_reg.name.lower())
            if not s_r and self.is_arm and sf_reg.name in self.arm_aliases:
                alias = self.arm_aliases[sf_reg.name]
                s_r = ctx.registers.get(alias) or ctx.registers.get(alias.upper())
            if not s_r and self.is_x86:
                # Synthetic XMM<n>_LO / XMM<n>_HI: pypcode has a single
                # XMM<n> at offset 0x1200+n*0x40 size 16.  We register
                # _LO at the base offset (low 64 bits) and _HI 8 bytes
                # in (high 64 bits) so Ghidra-emitted varnodes targeting
                # XMM<n>[63:0]  / XMM<n>[127:64] map correctly.
                s_r = self._synth_xmm_varnode(sf_reg.name)  # type: ignore[assignment]
            if s_r:
                self.sf_resolved.append((sf_reg, s_r))

    @staticmethod
    def _synth_xmm_varnode(name: str) -> _SynthVarnode | None:
        """Build a lightweight Varnode-like object for XMM<n>_LO / _HI.

        The StateMapper only reads `.offset` and `.size` from these
        objects, so a tiny shim suffices.  Returns None for any name
        that doesn't match the XMM<n>_LO / XMM<n>_HI pattern.
        """
        if not name.startswith('XMM'):
            return None
        # Parse "XMM<n>_LO" or "XMM<n>_HI"
        try:
            rest = name[3:]
            num_str, _, half = rest.partition('_')
            n = int(num_str)
        except ValueError:
            return None
        if not (0 <= n < 16) or half not in ('LO', 'HI'):
            return None
        # pypcode's XMM<n> sits at 0x1200 + n*0x40, size 16.
        sub_offset = 0 if half == 'LO' else 8
        return _SynthVarnode(offset=0x1200 + n * 0x40 + sub_offset, size=8)

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


def resolve_ptr_with_offset(  # noqa: C901
    vn: Varnode,
    all_ops: list[PcodeOp],
    mapper: StateMapper,
    stop_op_index: int | None = None,
) -> tuple[RegMapping | None, int]:
    """
    Resolves a pointer varnode to (base_register_mapping, signed_const_offset).

    Args:
        vn: The pointer Varnode to resolve.
        all_ops: All PcodeOps for the current instruction translation.
        mapper: StateMapper to resolve registers.
        stop_op_index: Optional limit to stop tracing register definitions.
                       Crucial for preventing post-access register states (like
                       stack pops) from polluting the pointer resolution.
    """
    visited_unique: set[int] = set()
    visited_reg: set[int] = set()

    limit_index = stop_op_index if stop_op_index is not None else len(all_ops)

    def _resolve(current_vn: Varnode) -> tuple[RegMapping | None, int]:  # noqa: C901
        if current_vn.space.name == 'const':
            val = current_vn.offset
            # Handle 64-bit negative offsets
            if val >= (1 << 63):
                val -= 1 << 64
            return None, val

        if current_vn.space.name == 'register':
            reg_off = current_vn.offset
            if reg_off in visited_reg:
                return mapper.map_to_state(current_vn.offset, current_vn.size), 0

            visited_reg.add(reg_off)

            for i, op in enumerate(all_ops):
                if i >= limit_index:
                    break
                if op.opcode.name in ('STORE', 'CALL', 'CALLIND', 'BRANCH', 'BRANCHIND', 'CBRANCH', 'RETURN'):
                    continue

                if (
                    op.output is not None
                    and op.output.space.name == 'register'
                    and op.output.offset == reg_off
                    and op.output.size == current_vn.size
                ):
                    if op.opcode.name in ('COPY', 'INT_ZEXT', 'INT_SEXT'):
                        res = _resolve(op.inputs[0])
                        visited_reg.discard(reg_off)
                        return res
                    if op.opcode.name in ('INT_ADD', 'PTRADD'):
                        lreg, loff = _resolve(op.inputs[0])
                        rreg, roff = _resolve(op.inputs[1])
                        visited_reg.discard(reg_off)
                        if lreg is not None:
                            return lreg, loff + roff
                        if rreg is not None:
                            return rreg, roff + loff
                    elif op.opcode.name == 'INT_SUB':
                        lreg, loff = _resolve(op.inputs[0])
                        _, roff = _resolve(op.inputs[1])
                        visited_reg.discard(reg_off)
                        if lreg is not None:
                            return lreg, loff - roff
                    # Any other defining op: value is computed, use direct mapping.
                    break

            visited_reg.discard(reg_off)
            return mapper.map_to_state(current_vn.offset, current_vn.size), 0

        if current_vn.space.name == 'unique':
            key = current_vn.offset
            if key in visited_unique:
                return None, 0
            visited_unique.add(key)

            for op in all_ops:
                if op.output is not None and op.output.space.name == 'unique' and op.output.offset == key:
                    if op.opcode.name in ('INT_ADD', 'PTRADD'):
                        lreg, loff = _resolve(op.inputs[0])
                        rreg, roff = _resolve(op.inputs[1])
                        if lreg is not None:
                            return lreg, loff + roff
                        if rreg is not None:
                            return rreg, roff + loff
                    elif op.opcode.name == 'INT_SUB':
                        lreg, loff = _resolve(op.inputs[0])
                        _, roff = _resolve(op.inputs[1])
                        if lreg is not None:
                            return lreg, loff - roff
                    elif op.opcode.name in ('COPY', 'INT_ZEXT', 'INT_SEXT'):
                        return _resolve(op.inputs[0])
                    else:
                        for inp in op.inputs:
                            r, o = _resolve(inp)
                            if r is not None:
                                return r, o
                    break

            return None, 0
        return None, 0

    return _resolve(vn)


def apply_sless_msb_split(
    deps: dict[RegMapping | MemMapping, int],
    slice_ops: list[PcodeOp],
    _ctx: Context,
    _arch: Architecture,
    _state_format: list[Register],
) -> dict[RegMapping | MemMapping, int]:
    sless_ops = [op for op in slice_ops if op.opcode.name in {'INT_SLESS', 'INT_SLESSEQUAL'}]
    if not sless_ops:
        return deps

    new_deps: dict[RegMapping | MemMapping, int] = {}
    msb_mappings: list[tuple[str, int]] = []

    for op in sless_ops:
        size = op.inputs[0].size
        msb_offset = (size * 8) - 1
        for dep_map in deps.keys():
            if isinstance(dep_map, RegMapping) and dep_map.bit_start <= msb_offset <= dep_map.bit_end:
                msb_mappings.append((dep_map.name, msb_offset))

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
            if matched_msb > dep_map.bit_start:
                new_deps[RegMapping(dep_map.name, dep_map.bit_start, matched_msb - 1)] = 1
            new_deps[RegMapping(dep_map.name, matched_msb, matched_msb)] = -1
            if dep_map.bit_end > matched_msb:
                new_deps[RegMapping(dep_map.name, matched_msb + 1, dep_map.bit_end)] = 1
        else:
            new_deps[dep_map] = p

    return new_deps


@functools.lru_cache(maxsize=16384)
def _cached_generate_static_rule(
    arch: Architecture,
    bytestring: bytes,
    state_format_tuple: tuple[tuple[str, int], ...],
) -> LogicCircuit:
    state_format = [Register(name=name, bits=bits) for name, bits in state_format_tuple]

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

        dep_set = extract_dependencies(out_vn, slice_ops, polarities, translation.ops, mapper)

        # For STORE targets whose val_vn is a leaf register (no defining op in
        # this instruction), slice_backward returns empty and extract_dependencies
        # finds nothing.  Inject the register directly as the sole value dep.
        if isinstance(mapping, MemMapping) and not dep_set.value_deps and not dep_set.addr_deps:
            if out_vn.space.name == 'register':
                direct_reg = mapper.map_to_state(out_vn.offset, out_vn.size)
                if direct_reg is not None:
                    dep_set.value_deps[direct_reg] = 1
            elif out_vn.space.name == 'const':
                # Stored value is a constant (e.g. call's return address) — always untainted.
                # Leave dep_set empty: generate_taint_assignments will emit zero taint.
                pass

        # Apply polarity split for signed comparisons (value_deps only — addr_deps
        # are not used in the differential so they don't need polarity treatment).
        split_value_deps = apply_sless_msb_split(dep_set.value_deps, slice_ops, ctx, arch, state_format)
        dep_set = DependencySet(value_deps=split_value_deps, addr_deps=dep_set.addr_deps)

        out_target, out_name, out_bit_start, out_bit_end = generate_output_target(mapping)

        generate_taint_assignments(
            arch,
            bytestring,
            assignments,
            slice_ops,
            dep_set,
            out_target,
            out_name,
            out_bit_start,
            out_bit_end,
            mapper,
            mapping,
        )

    return LogicCircuit(
        assignments=assignments,
        architecture=arch,
        instruction=bytestring.hex(),
        state_format=state_format,
    )


def generate_static_rule(
    arch: Architecture,
    bytestring: bytes,
    state_format: list[Register],
) -> LogicCircuit:
    reg_names_tuple = tuple((reg.name, reg.bits) for reg in state_format)
    return _cached_generate_static_rule(arch, bytestring, reg_names_tuple)


def generate_taint_assignments(  # noqa: C901
    arch: Architecture,
    bytestring: bytes,
    assignments: list[TaintAssignment],
    slice_ops: list[PcodeOp],
    dep_set: DependencySet,
    out_target: TaintOperand | MemoryOperand,
    out_name: str,
    out_bit_start: int,
    out_bit_end: int,
    mapper: StateMapper,
    mapping: RegMapping | MemMapping | None = None,
) -> None:
    # -----------------------------------------------------------------------
    # STORE TARGET — memory output of a STORE instruction.
    #
    # The taint of the written memory byte equals the taint of the *value*
    # being stored — never the taint of the address register used to compute
    # the destination.  The address register's taint is an AIW signal
    # (handled separately by the wrapper), not a data-taint signal.
    #
    # We handle this first, before any category dispatch, because STORE
    # targets don't go through the differential machinery at all:
    #   - The differential would XOR two Unicorn runs that both write the
    #     same concrete value, so C1 XOR C2 == 0 always.
    #   - The transport term must use value_deps only.
    # -----------------------------------------------------------------------
    is_store_target = hasattr(out_target, 'address_expr')

    # -----------------------------------------------------------------------
    # RMW DETECTION
    # -----------------------------------------------------------------------
    # An RMW instruction (e.g. `add [rbp-0x10], rax`, `xor [mem], reg`) reads
    # from memory, performs arithmetic with a source register, and writes
    # back.  The output taint depends on BOTH old memory contents AND the
    # source register, including carry/borrow chains.
    #
    # The OR-only path below was correct for pure stores (`mov [mem], reg`),
    # where C1 XOR C2 == 0 — but for RMW the differential is non-zero, and
    # the OR-only path silently drops the carry chain, breaking SipHash-style
    # avalanche.
    #
    # Detection: a STORE target with a LOAD op in the same slice_ops is RMW.
    # Cost: single linear scan over slice_ops (~10 ops max), runs once per
    # unique instruction byte sequence, then cached by _cached_generate_static_rule.
    # Pure stores (the common case) keep the fast OR-only path below.
    # -----------------------------------------------------------------------
    is_rmw = False
    if is_store_target:
        for _op in slice_ops:
            if _op.opcode.name == 'LOAD':
                is_rmw = True
                break

    if is_store_target and not is_rmw:
        value_deps = dep_set.value_deps

        if not value_deps:
            # Stored value is a constant (e.g. call's inst_next) — always untainted.
            expr: Expr = _get_zero_constant(out_bit_end - out_bit_start + 1)
            assignments.append(TaintAssignment(target=out_target, dependencies=[], expression=expr))
            return

        # Build value taint: OR of all value register/memory taints.
        value_dependencies, _, _, _ = process_dependencies(value_deps)

        if value_dependencies:
            expr = value_dependencies[0]
            for t in value_dependencies[1:]:
                expr = BinaryExpr(Op.OR, expr, t)
        else:
            expr = _get_zero_constant(out_bit_end - out_bit_start + 1)

        # Note: addr_deps taint is deliberately excluded here.
        # A tainted store address is an AIW (arbitrary indexed write) — the
        # wrapper detects this via register_taint, not via shadow memory.
        # Putting addr taint into shadow would cause catastrophic false positives:
        # after 'leave' propagates T_RBP→T_RSP, every subsequent push/call would
        # write tainted shadow, poisoning all future return addresses.

        assignments.append(
            TaintAssignment(
                target=out_target,
                dependencies=value_dependencies,
                expression=expr,
            ),
        )
        return

    # -----------------------------------------------------------------------
    # RMW MEMORY TARGET — uses MemoryDifferentialExpr (sleigh.mem_diff) which
    # bypasses the buggy `_build_machine_state` path that drops memory-input
    # offsets and address-only register values.  See sleigh/mem_diff.py for
    # the detailed background on the two underlying bugs in the standard
    # InstructionCellExpr path.
    #
    # The C1 XOR C2 differential captures carry/borrow chains and per-bit
    # dependency structure that the OR-only fast path drops.  We still OR
    # with the explicit value-taint fallback so bits explicitly tainted in
    # inputs are never lost even on simulator failure.
    #
    # addr_deps taint is still excluded here — same security property as
    # the pure-store path: a tainted destination POINTER is an AIW signal,
    # not data taint, and must not poison shadow memory.
    # -----------------------------------------------------------------------
    if is_rmw:
        if not isinstance(mapping, MemMapping):
            raise RuntimeError('RMW target without MemMapping')

        # Collect value-deps as (reg_inputs, mem_inputs) lists for the
        # MemoryDifferentialExpr constructor.  Address-only registers are
        # built from the union of input addresses minus value-deps.
        reg_inputs: list[tuple[str, int, int]] = []
        mem_inputs: list[tuple[str, int, int]] = []
        addr_only_regs_set: set[str] = set()

        # Always add the destination's own address register as address-only.
        addr_only_regs_set.add(mapping.addr_reg.name)

        for dep_map in dep_set.value_deps.keys():
            if isinstance(dep_map, MemMapping):
                mem_inputs.append(
                    (
                        dep_map.addr_reg.name,
                        dep_map.addr_const_offset,
                        dep_map.size_bytes,
                    ),
                )
                addr_only_regs_set.add(dep_map.addr_reg.name)
            else:
                reg_inputs.append((dep_map.name, dep_map.bit_start, dep_map.bit_end))

        # Remove from addr_only_regs any register that is also a value dep.
        value_reg_names = {r[0] for r in reg_inputs}
        addr_only_regs = sorted(addr_only_regs_set - value_reg_names)

        target_spec = (
            'MEM',
            mapping.addr_reg.name,
            mapping.addr_const_offset,
            mapping.size_bytes,
        )

        diff_expr: Expr = MemoryDifferentialExpr(
            bytestring=bytestring,
            target=target_spec,
            reg_inputs=reg_inputs,
            mem_inputs=mem_inputs,
            addr_only_regs=addr_only_regs,
        )

        # Build the explicit value-taint OR fallback (transport term).
        value_dependencies, _, _, _ = process_dependencies(dep_set.value_deps)

        if value_dependencies:
            transport = value_dependencies[0]
            for t in value_dependencies[1:]:
                transport = BinaryExpr(Op.OR, transport, t)
            expr = BinaryExpr(Op.OR, diff_expr, transport)
        else:
            expr = diff_expr

        # addr_deps deliberately excluded — same reason as pure-store path.

        assignments.append(
            TaintAssignment(
                target=out_target,
                dependencies=value_dependencies,
                expression=expr,
            ),
        )
        return

    # -----------------------------------------------------------------------
    # REGISTER / FLAG TARGET — from here on, out_target is a TaintOperand.
    # We work with value_deps for the differential/transport/category logic,
    # and addr_deps for the pointer-avalanche (LOAD pointer taint).
    # -----------------------------------------------------------------------
    dependencies, dependency_names, cell_inputs_rep1, cell_inputs_rep2 = process_dependencies(
        dep_set.value_deps,
    )

    if not dependencies:
        expr = _get_zero_constant(out_bit_end - out_bit_start + 1)
        if out_name in ('EIP', 'RIP', 'PC'):
            expr = AvalancheExpr(expr, out_bit_end - out_bit_start + 1)
        assignments.append(TaintAssignment(target=out_target, dependencies=[], expression=expr))
        return

    cat = determine_category(slice_ops)

    # --- LOAD POINTER TAINT DETECTION ---
    # Identify which value_dep RegMappings are used as *pointer* inputs to
    # LOAD ops.  These drive the pointer-avalanche (unknown read address).
    #
    # Root cause fix #3 (leave/ret): we resolve LOAD pointers by tracing
    # through ALL ops in the instruction (translation.ops), not just slice_ops.
    # This correctly resolves the 'leave' case where the LOAD pointer is a
    # unique varnode defined by 'COPY RSP' where RSP itself was defined by
    # 'COPY RBP' in the same instruction — tracing only slice_ops would stop
    # at RSP and miss that its effective value is old_RBP.
    load_ops = [op for op in slice_ops if op.opcode.name == 'LOAD']

    # --- LOAD POINTER TAINT DETECTION ---
    # addr_deps from DependencySet already contains exactly the registers used
    # as LOAD/STORE pointers, classified correctly by extract_dependencies.
    # We build pointer taint expressions directly from addr_deps — no re-detection.
    #
    # Additionally, we need pointer_reg_names to split 'dependencies' (from
    # value_deps) into mem_taint_exprs vs pointer_taint_exprs for the
    # is_load_like branch.  Note: for LOAD outputs, the pointer register is
    # in addr_deps (not value_deps), so pointer_taint_exprs from value_deps
    # will be empty.  The avalanche is built separately from addr_deps below.

    # Expressions for the pointer avalanche — built from addr_deps, NOT value_deps.
    # This is the correct source: addr_deps holds registers used as memory addresses.
    addr_dep_taint_exprs: list[Expr] = []
    for addr_reg, _ in dep_set.addr_deps.items():
        addr_dep_taint_exprs.append(
            _get_taint_operand(addr_reg.name, addr_reg.bit_start, addr_reg.bit_end, True),
        )

    # Stack pointer names — tainted stack pointers do NOT trigger the avalanche.
    # RSP taint propagates arithmetically through 'leave' (mov rsp, rbp), not
    # because an attacker controls the load address.
    _STACK_POINTER_NAMES: frozenset[str] = frozenset({'RSP', 'ESP', 'SP'})

    non_stack_addr_taint_exprs: list[Expr] = [
        expr
        for expr, reg in zip(addr_dep_taint_exprs, dep_set.addr_deps, strict=False)
        if reg.name not in _STACK_POINTER_NAMES
    ]
    has_tainted_non_stack_pointer = bool(non_stack_addr_taint_exprs)

    # Split value dependencies into mem_taint_exprs and plain data for is_load_like.
    mem_taint_exprs: list[Expr] = []
    for dep_expr, dep_name in zip(dependencies, dependency_names, strict=True):
        if dep_name.startswith('MEM_'):
            mem_taint_exprs.append(dep_expr)

    # --- LOAD-LIKE DETECTION ---
    is_load_like = False
    if load_ops and cat == InstructionCategory.MAPPED:
        for load_op in load_ops:
            if load_op.output is None:
                continue
            if load_op.output.space.name == 'unique':
                is_load_like = True
                break
            if load_op.output.space.name == 'register':
                mapped_load_out = mapper.map_to_state(load_op.output.offset, load_op.output.size)
                if mapped_load_out and mapped_load_out.name == out_name:
                    is_load_like = True
                    break

    # Detect whether this register-target instruction has memory inputs
    # OR address-only registers — both cases need MemoryDifferentialExpr
    # (the standard make_differential() path resolves memory addresses
    # incorrectly because cell_inputs uses the legacy MEM_<reg> key
    # format that drops the offset).  See MemoryDifferentialExpr for
    # the detailed bug background.
    _has_mem_inputs = any(isinstance(d, MemMapping) for d in dep_set.value_deps.keys())
    _value_reg_names = {d.name for d in dep_set.value_deps.keys() if not isinstance(d, MemMapping)}
    _addr_only_regs_set: set[str] = set()
    for d in dep_set.value_deps.keys():
        if isinstance(d, MemMapping) and d.addr_reg.name not in _value_reg_names:
            _addr_only_regs_set.add(d.addr_reg.name)
    _has_addr_only = bool(_addr_only_regs_set)
    _use_mem_diff = _has_mem_inputs or _has_addr_only

    def make_differential() -> Expr:
        if _use_mem_diff:
            # Memory-aware path: route through MemoryDifferentialExpr which
            # builds the simulator state with correct addresses and
            # address-only register values.  Performance: ~2x faster than
            # the BinaryExpr(XOR, C1_cell, C2_cell) path because it shares
            # cell.pyx's _frame_a/_frame_b buffers via evaluate_differential.
            _reg_inputs = []
            _mem_inputs = []
            for d in dep_set.value_deps.keys():
                if isinstance(d, MemMapping):
                    _mem_inputs.append((d.addr_reg.name, d.addr_const_offset, d.size_bytes))
                else:
                    _reg_inputs.append((d.name, d.bit_start, d.bit_end))
            return MemoryDifferentialExpr(
                bytestring=bytestring,
                target=('REG', out_name, out_bit_start, out_bit_end),
                reg_inputs=_reg_inputs,
                mem_inputs=_mem_inputs,
                addr_only_regs=sorted(_addr_only_regs_set),
            )
        # Pure-register fast path: cell.pyx's static-cell evaluation.
        C1_cell = InstructionCellExpr(arch, bytestring.hex(), out_name, out_bit_start, out_bit_end, cell_inputs_rep1)
        C2_cell = InstructionCellExpr(arch, bytestring.hex(), out_name, out_bit_start, out_bit_end, cell_inputs_rep2)
        return BinaryExpr(Op.XOR, C1_cell, C2_cell)

    if is_load_like:
        mem_taint: Expr | None = None
        if mem_taint_exprs:
            mem_taint = mem_taint_exprs[0]
            for t in mem_taint_exprs[1:]:
                mem_taint = BinaryExpr(Op.OR, mem_taint, t)

        # Avalanche only on non-stack pointer taint (addr_deps, excluding RSP/ESP/SP).
        if non_stack_addr_taint_exprs:
            ptr_combined = non_stack_addr_taint_exprs[0]
            for t in non_stack_addr_taint_exprs[1:]:
                ptr_combined = BinaryExpr(Op.OR, ptr_combined, t)
            avalanche_ptr = AvalancheExpr(ptr_combined, out_bit_end - out_bit_start + 1)
            expr = BinaryExpr(Op.OR, avalanche_ptr, mem_taint) if mem_taint is not None else avalanche_ptr
        else:
            expr = mem_taint if mem_taint is not None else _get_zero_constant(out_bit_end - out_bit_start + 1)

    elif cat == InstructionCategory.AVALANCHE:
        expr = dependencies[0]
        for dep in dependencies[1:]:
            expr = BinaryExpr(Op.OR, expr, dep)
        expr = AvalancheExpr(expr, out_bit_end - out_bit_start + 1)

    elif cat == InstructionCategory.TRANSLATABLE:
        diff_expr = make_differential()

        core_ops = [op for op in slice_ops if op.opcode.name not in EXTENSION_OPCODES]
        shift_op = next((op for op in core_ops if op.opcode.name in TRANSLATABLE_OPCODES), None)

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

                for op in slice_ops:
                    if op.output and op.output.space.name == 'unique' and op.output.offset == vn.offset:
                        if op.opcode.name == 'LOAD':
                            ptr_vn = op.inputs[1]
                            m = mapper.map_to_state(ptr_vn.offset, ptr_vn.size)
                            if m:
                                origins.add(f'MEM_{m.name}')
                        else:
                            for inp in op.inputs:
                                origins.update(trace_origins(inp, visited))
                        break
            return origins

        offset_names: set[str] = set()
        if shift_op and len(shift_op.inputs) > 1:
            offset_names = trace_origins(shift_op.inputs[1])

        primary_input_name = None
        if shift_op and shift_op.inputs[0].space.name == 'register':
            m = mapper.map_to_state(shift_op.inputs[0].offset, shift_op.inputs[0].size)
            if m:
                primary_input_name = m.name

        if not offset_names:
            offset_names = {name for name in dependency_names if name not in (out_name, primary_input_name)}

        offset_taints = [dep for dep, name in zip(dependencies, dependency_names, strict=True) if name in offset_names]

        if offset_taints:
            combined_offset = offset_taints[0]
            for t in offset_taints[1:]:
                combined_offset = BinaryExpr(Op.OR, combined_offset, t)

            avalanche_shift = AvalancheExpr(combined_offset, out_bit_end - out_bit_start + 1)
            expr = BinaryExpr(Op.OR, diff_expr, avalanche_shift)
        else:
            expr = diff_expr

    elif cat == InstructionCategory.COND_TRANSPORTABLE:
        T_union = dependencies[0]
        for dep in dependencies[1:]:
            T_union = BinaryExpr(Op.OR, T_union, dep)

        T_any = AvalancheExpr(T_union, out_bit_end - out_bit_start + 1)

        imm_val = None
        for op in slice_ops:
            if op.opcode.name == 'INT_SUB':
                for vn in op.inputs:
                    if vn.space.name == 'const' and vn.offset != 0:
                        raw = vn.offset
                        size_bits = vn.size * 8
                        if size_bits > 0 and raw >= (1 << (size_bits - 1)):
                            raw -= 1 << size_bits
                        imm_val = raw
                        break
            if imm_val is not None:
                break

        has_const_operand = imm_val is not None

        if has_const_operand and len(dep_set.value_deps) == 1:
            dep_map = next(iter(dep_set.value_deps.keys()))
            if TYPE_CHECKING:
                assert imm_val is not None

            if isinstance(dep_map, RegMapping):
                V_masked = cell_inputs_rep2[dep_map.name]
                T_in = _get_taint_operand(dep_map.name, dep_map.bit_start, dep_map.bit_end, True)
                size = dep_map.bit_end - dep_map.bit_start + 1
                imm_expr = Constant(imm_val, size)
                imm_masked = BinaryExpr(Op.AND, imm_expr, T_in)
                corrected = BinaryExpr(Op.OR, V_masked, imm_masked)
                C_eval = InstructionCellExpr(
                    arch,
                    bytestring.hex(),
                    out_name,
                    out_bit_start,
                    out_bit_end,
                    {dep_map.name: corrected},
                )
                expr = BinaryExpr(Op.AND, C_eval, T_any)

            elif isinstance(dep_map, MemMapping):  # pyright: ignore[reportUnnecessaryIsInstance]
                addr_base = _get_taint_operand(
                    dep_map.addr_reg.name,
                    dep_map.addr_reg.bit_start,
                    dep_map.addr_reg.bit_end,
                    False,
                )
                addr_expr: Expr = (
                    BinaryExpr(Op.ADD, addr_base, Constant(dep_map.addr_const_offset, 8))
                    if dep_map.addr_const_offset != 0
                    else addr_base
                )
                T_mem = MemoryOperand(addr_expr, dep_map.size_bytes, is_taint=True)
                V_mem = MemoryOperand(addr_expr, dep_map.size_bytes, is_taint=False)
                imm_expr = Constant(imm_val, dep_map.size_bytes * 8)
                V_masked = BinaryExpr(Op.AND, V_mem, UnaryExpr(Op.NOT, T_mem))
                imm_masked = BinaryExpr(Op.AND, imm_expr, T_mem)
                corrected = BinaryExpr(Op.OR, V_masked, imm_masked)
                dep_name = f'MEM_{dep_map.addr_reg.name}'
                C_eval = InstructionCellExpr(
                    arch,
                    bytestring.hex(),
                    out_name,
                    out_bit_start,
                    out_bit_end,
                    {dep_name: corrected},
                )
                expr = BinaryExpr(Op.AND, C_eval, T_any)

            else:
                raise RuntimeError('Unexpected dependency type in COND_TRANSPORTABLE with const operand')

        else:
            masked_inputs: dict[str, Expr] = {}
            for dep_map in dep_set.value_deps.keys():
                if isinstance(dep_map, MemMapping):
                    addr_base = _get_taint_operand(
                        dep_map.addr_reg.name,
                        dep_map.addr_reg.bit_start,
                        dep_map.addr_reg.bit_end,
                        False,
                    )
                    addr_expr = (
                        BinaryExpr(Op.ADD, addr_base, Constant(dep_map.addr_const_offset, 8))
                        if dep_map.addr_const_offset != 0
                        else addr_base
                    )
                    V_in: Expr = MemoryOperand(addr_expr, dep_map.size_bytes, is_taint=False)
                    dep_name = f'MEM_{dep_map.addr_reg.name}'
                else:
                    V_in = _get_taint_operand(dep_map.name, dep_map.bit_start, dep_map.bit_end, False)
                    dep_name = dep_map.name
                masked_inputs[dep_name] = BinaryExpr(Op.AND, V_in, UnaryExpr(Op.NOT, T_union))

            C_eval = InstructionCellExpr(
                arch,
                bytestring.hex(),
                out_name,
                out_bit_start,
                out_bit_end,
                masked_inputs,
            )
            expr = BinaryExpr(Op.AND, C_eval, T_any)

    elif cat == InstructionCategory.TRANSPORTABLE:
        diff_expr = make_differential()
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

        is_zeroing_idiom = False
        if xor_ops:
            xor_op = xor_ops[0]
            in1, in2 = xor_op.inputs[0], xor_op.inputs[1]
            if in1.space == in2.space and in1.offset == in2.offset and in1.size == in2.size:
                is_zeroing_idiom = True

        if is_zeroing_idiom:
            expr = _get_zero_constant(out_bit_end - out_bit_start + 1)
        elif dependencies:
            expr = dependencies[0]
            for dep in dependencies[1:]:
                expr = BinaryExpr(Op.OR, expr, dep)
        else:
            expr = _get_zero_constant(out_bit_end - out_bit_start + 1)

    elif cat == InstructionCategory.MONOTONIC:
        expr = make_differential()

    else:
        raise ValueError(f'Unsupported instruction category: {cat}')

    # For non-stack LOAD pointers in non-load-like instructions (e.g. ADD RAX, [RBX]),
    # OR in the pointer avalanche. Stack pointer excluded for the same reason.
    if not is_load_like and has_tainted_non_stack_pointer:
        ptr_combined = non_stack_addr_taint_exprs[0]
        for t in non_stack_addr_taint_exprs[1:]:
            ptr_combined = BinaryExpr(Op.OR, ptr_combined, t)
        avalanche_ptr = AvalancheExpr(ptr_combined, out_bit_end - out_bit_start + 1)
        expr = BinaryExpr(Op.OR, expr, avalanche_ptr)

    if out_name in ('EIP', 'RIP', 'PC'):
        expr = AvalancheExpr(expr, out_bit_end - out_bit_start + 1)

    assignments.append(TaintAssignment(target=out_target, dependencies=dependencies, expression=expr))


def build_polarized_reg(name: str, slices: list[tuple[int, int, int]], replica_id: int) -> Expr:
    combined_expr = None
    for s_start, s_end, p in slices:
        V_in = _get_taint_operand(name, s_start, s_end, False)
        T_in = _get_taint_operand(name, s_start, s_end, True)

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
    dependencies: list[Expr] = []
    dependency_names: list[str] = []

    reg_groups: dict[str, list[tuple[int, int, int]]] = {}
    mem_groups: dict[str, list[MemMapping]] = {}

    for dep_map, p in deps.items():
        if isinstance(dep_map, MemMapping):
            key = f'MEM_{dep_map.addr_reg.name}'
            mem_groups.setdefault(key, []).append(dep_map)
        else:
            reg_groups.setdefault(dep_map.name, []).append((dep_map.bit_start, dep_map.bit_end, p))
            dependencies.append(_get_taint_operand(dep_map.name, dep_map.bit_start, dep_map.bit_end, True))
            dependency_names.append(dep_map.name)

    cell_inputs_rep1: dict[str, Expr] = {}
    cell_inputs_rep2: dict[str, Expr] = {}

    for name, slices in reg_groups.items():
        cell_inputs_rep1[name] = build_polarized_reg(name, slices, 1)
        cell_inputs_rep2[name] = build_polarized_reg(name, slices, 2)

    for name, mem_list in mem_groups.items():
        m = mem_list[0]
        addr_base = _get_taint_operand(m.addr_reg.name, m.addr_reg.bit_start, m.addr_reg.bit_end, False)

        if m.addr_const_offset != 0:
            addr_expr: Expr = BinaryExpr(Op.ADD, addr_base, Constant(m.addr_const_offset, 8))
        else:
            addr_expr = addr_base

        T_mem = MemoryOperand(addr_expr, m.size_bytes, is_taint=True)
        V_mem = MemoryOperand(addr_expr, m.size_bytes, is_taint=False)

        cell_inputs_rep1[name] = BinaryExpr(Op.OR, V_mem, T_mem)
        cell_inputs_rep2[name] = BinaryExpr(Op.AND, V_mem, UnaryExpr(Op.NOT, T_mem))
        dependencies.append(T_mem)
        dependency_names.append(name)

    return dependencies, dependency_names, cell_inputs_rep1, cell_inputs_rep2


def generate_output_target(mapping: RegMapping | MemMapping) -> tuple[TaintOperand | MemoryOperand, str, int, int]:
    out_target: TaintOperand | MemoryOperand
    if isinstance(mapping, MemMapping):
        addr_base: Expr = _get_taint_operand(mapping.addr_reg.name, 0, 63, False)
        if mapping.addr_const_offset != 0:
            addr_expr: Expr = BinaryExpr(Op.ADD, addr_base, Constant(mapping.addr_const_offset, 8))
        else:
            addr_expr = addr_base
        out_target = MemoryOperand(addr_expr, mapping.size_bytes, is_taint=True)
        out_name = f'MEM_{mapping.addr_reg.name}'
        out_bit_start, out_bit_end = 0, (mapping.size_bytes * 8) - 1
    else:
        out_target = _get_taint_operand(mapping.name, mapping.bit_start, mapping.bit_end, True)
        out_name = mapping.name
        out_bit_start, out_bit_end = mapping.bit_start, mapping.bit_end
    return out_target, out_name, out_bit_start, out_bit_end


def extract_dependencies(  # noqa: C901
    _out_vn: Varnode,
    _slice_ops: list[PcodeOp],
    polarities: dict[str, int],
    all_ops: list[PcodeOp],
    mapper: StateMapper,
) -> DependencySet:
    """
    Classify all inputs of a taint assignment into value_deps and addr_deps.

    value_deps: registers/memory whose *data content* flows into the output.
                These drive the taint expression (differential, transport, etc.).

    addr_deps:  registers used only to compute a memory *address* (LOAD or
                STORE pointer).  Their taint signals an unknown access target
                (AIW / tainted-pointer load), not a data dependency.  The
                caller uses addr_deps to build the pointer-avalanche when
                appropriate — but NEVER mixes them into the value taint.

    Root cause fix #1: the old implementation returned a flat dict that
    merged both kinds together.  This caused STORE assignments to include
    address-register taint in the stored-value taint, leading to:
      push rbp  →  T_MEM = T_RBP | T_RSP   (T_RSP is address, not value)
    After 'leave' propagates T_RBP→T_RSP, every subsequent push would
    write tainted shadow, causing a cascade of false positive BOFs.
    """
    value_deps: dict[RegMapping | MemMapping, int] = {}
    addr_deps: dict[RegMapping, int] = {}

    # Registers that are used as LOAD/STORE *pointers* within this slice.
    # We identify them first so we can classify register inputs correctly.
    ptr_reg_offsets: set[int] = set()

    def _collect_ptr_offsets(vn: Varnode, visited: set[int] | None = None) -> None:
        """
        Walk a pointer varnode to its ultimate register source(s) and record
        the register offsets in ptr_reg_offsets.

        MUST use all_ops (not slice_ops) to resolve unique temporaries.
        For leaf STORE targets (e.g. 'mov [rbp-8], rax'), slice_ops is empty
        because val_vn=RAX has no defining op — but the STORE pointer chain
        (unique:$u1 = RBP - 8) is only in all_ops.
        """
        if visited is None:
            visited = set()
        if vn.space.name == 'register':
            ptr_reg_offsets.add(vn.offset)
        elif vn.space.name == 'unique':
            if vn.offset in visited:
                return
            visited.add(vn.offset)
            for op in all_ops:  # ← all_ops, not slice_ops
                if op.output is not None and op.output.space.name == 'unique' and op.output.offset == vn.offset:
                    for inp in op.inputs:
                        if inp.space.name != 'const':
                            _collect_ptr_offsets(inp, visited)
                    break

    for op in all_ops:
        if op.opcode.name == 'LOAD':
            _collect_ptr_offsets(op.inputs[1])
        elif op.opcode.name == 'STORE':
            _collect_ptr_offsets(op.inputs[1])

    # Pre-calculate this ONCE instead of inside the resolution loop
    load_op_index = next(
        (i for i, op in enumerate(all_ops) if op.opcode.name == 'LOAD' and op.inputs[1].space.name != 'const'),
        len(all_ops),
    )

    for op in all_ops:
        if op.opcode.name == 'RETURN':
            continue

        if op.opcode.name == 'LOAD':
            ptr_vn = op.inputs[1]
            mapped_addr, const_offset = resolve_ptr_with_offset(ptr_vn, all_ops, mapper, stop_op_index=load_op_index)
            if mapped_addr:
                mem_map = MemMapping(
                    ptr_vn.offset,
                    op.output.size if op.output else 8,
                    mapped_addr,
                    const_offset,
                )
                # LOAD memory content is a *value* dependency.
                value_deps[mem_map] = 1

        for vn in op.inputs:
            if vn.space.name == 'register':
                # Try the singular map first — preserves the existing
                # "smallest covering register" semantics for GPRs and
                # their aliases (RAX → just RAX, not RAX+EAX+AX+AL).
                mapped_dep = mapper.map_to_state(vn.offset, vn.size)
                if mapped_dep is not None:
                    mapped_deps: list[RegMapping] = [mapped_dep]
                else:
                    # No single state_format entry covers the input — fall
                    # back to the multi-mapping form.  This is how XMM
                    # registers (split into XMM<n>_LO + XMM<n>_HI in the
                    # state_format) get all their pieces tracked.
                    mapped_deps = mapper.map_to_state_all(vn.offset, vn.size)
                if not mapped_deps:
                    continue
                for md in mapped_deps:
                    # Classify: is this register used as a pointer, or as data?
                    if vn.offset in ptr_reg_offsets:
                        # Address register — goes into addr_deps.
                        # Do NOT add to value_deps.
                        if md not in addr_deps:
                            addr_deps[md] = 1
                    else:
                        # Data register — goes into value_deps.
                        if md not in value_deps:
                            value_deps[md] = 1

    # Apply polarity annotations to value_deps only (addr_deps don't
    # participate in the differential so polarity is irrelevant for them).
    for vn_id, p in polarities.items():
        parts = vn_id.split(':')
        if len(parts) == 3 and parts[0] == 'register':
            mapped_dep = mapper.map_to_state(int(parts[1]), int(parts[2]))
            if mapped_dep:
                if mapped_dep in value_deps:
                    value_deps[mapped_dep] = p
                # addr_deps polarity left at default 1

    return DependencySet(value_deps=value_deps, addr_deps=addr_deps)


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

    for val_vn, ptr_vn, size in mem_targets:
        base_reg, const_offset = resolve_ptr_with_offset(ptr_vn, translation.ops, mapper)
        if base_reg is None:
            continue
        mem_map = MemMapping(ptr_vn.offset, size, base_reg, const_offset)
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
