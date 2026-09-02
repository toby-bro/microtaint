from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Iterable

from pypcode import Context, PcodeOp, Translation, Varnode

from microtaint.classifier.categories import InstructionCategory
from microtaint.instrumentation.ast import (
    AvalancheExpr,
    BinaryExpr,
    ChainedCircuit,
    ComparisonTaintExpr,
    Constant,
    EqualityTaintExpr,
    Expr,
    FullMaskAvalancheExpr,
    InstructionCellExpr,
    LogicCircuit,
    MemoryDifferentialExpr,
    MemoryOperand,
    Op,
    SignedOverflowTaintExpr,
    TaintAssignment,
    TaintOperand,
    UnaryExpr,
    VariableBitSelectTaintExpr,
    VariableMultiplyTaintExpr,
    VariableShiftTaintExpr,
)
from microtaint.sleigh.constfold import const_value, fold_constants, is_constant_op
from microtaint.sleigh.lifter import get_context
from microtaint.sleigh.mapper import (
    CONTROL_FLOW_OPCODES,
    EXTENSION_OPCODES,
    ROUTING_OPCODES,
    TRANSLATABLE_OPCODES,
    determine_category,
)
from microtaint.sleigh.partition import (
    ALG_ARITH,
    ALG_BITWISE,
    find_waist,
    varnode_taint_expr,
    waist_taint_expr,
)
from microtaint.sleigh.polarity import _is_bitwise_not, compute_polarity
from microtaint.sleigh.slicer import get_varnode_id, slice_backward
from microtaint.types import Architecture, Register

if TYPE_CHECKING:
    from microtaint.simulator import CellSimulator


def _cone_register_taint(
    slice_ops: list[PcodeOp], vn: Varnode, mapper: StateMapper,
) -> Expr | None:
    """Union of the taint of every architectural register `vn` derives from."""
    seen: set[tuple[str, int, int]] = set()
    regs: list[RegMapping] = []

    def walk(v: Varnode, depth: int = 0) -> None:
        if depth > 16 or v.space.name == 'const' or _key_of(v) in seen:
            return
        seen.add(_key_of(v))
        if v.space.name == 'register':
            m = mapper.map_to_state(v.offset, v.size)
            if isinstance(m, RegMapping):
                regs.append(m)
            return
        for o in slice_ops:
            if o.output is not None and _overlaps_vn(o.output, v):
                for i in o.inputs:
                    walk(i, depth + 1)

    walk(vn)
    if not regs:
        return None
    acc: Expr = _get_taint_operand(regs[0].name, regs[0].bit_start, regs[0].bit_end, True)
    for m in regs[1:]:
        acc = BinaryExpr(Op.OR, acc, _get_taint_operand(m.name, m.bit_start, m.bit_end, True))
    return acc


def _reg_taint_for_floor(mapper: StateMapper) -> Callable[[int, int], Expr | None]:
    """Adapter: architectural-register taint lookup for varnode_taint_expr."""

    def _f(offset: int, size: int) -> Expr | None:
        m = mapper.map_to_state(offset, size)
        if not isinstance(m, RegMapping):
            return None
        return _get_taint_operand(m.name, m.bit_start, m.bit_end, True)

    return _f


def _key_of(vn: Varnode) -> tuple[str, int, int]:
    return (vn.space.name, vn.offset, vn.size)


def _overlaps_vn(a: Varnode, b: Varnode) -> bool:
    return (
        a.space.name == b.space.name
        and a.offset < b.offset + b.size
        and b.offset < a.offset + a.size
    )


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


# Marker base for a compile-time-constant memory address (absolute / PC-relative
# operands lifted to a ram-space varnode: the varnode offset IS the address).  A
# MemMapping whose base is this sentinel encodes an absolute address in
# addr_const_offset; downstream it becomes a Format-A key ``MEM_0x<addr>_<size>``
# that both the C kernel (cell.pyx) and the Unicorn path resolve statically.  The
# name has no underscores so that, even on a path that does not special-case it,
# the Format-B key ``MEM_CONSTZERO_<addr>_<size>`` still parses (base resolves to
# 0).  Constant pointer => untainted => plain value dep, no avalanche.
_CONST_ADDR_MARKER = 'CONSTZERO'
_CONST_ADDR_BASE = RegMapping(_CONST_ADDR_MARKER, 0, 63)

# Every instruction is translated at this fixed base (see ctx.translate below).
# A compile-time-constant memory address is therefore expressed relative to it.
_TRANSLATE_BASE = 0x1000


def _pc_reg_mapping(arch: Architecture, state_format: list[Register]) -> RegMapping | None:
    """The program-counter register (RIP / EIP / PC) as a whole-register mapping,
    or None if the state_format has no PC.  Used as the base for PC-relative
    memory so the address resolves against the RUNTIME pc, not the translate base."""
    up = str(arch).upper()
    name = 'RIP' if 'AMD64' in up else 'EIP' if 'X86' in up else 'PC'
    reg = next((r for r in state_format if r.name.upper() == name), None)
    if reg is None:
        reg = next((r for r in state_format if r.name.upper() in ('RIP', 'EIP', 'PC')), None)
    return RegMapping(reg.name, 0, reg.bits - 1) if reg is not None else None


@functools.lru_cache(maxsize=16384)
def _pc_relative_addrs(
    arch: Architecture,
    bytestring: bytes,
    state_format_tuple: tuple[tuple[str, int], ...],
) -> frozenset[int]:
    """Constant memory addresses (as resolved at ``_TRANSLATE_BASE``) whose value
    SHIFTS with the translation base -- i.e. PC-relative operands (x86 RIP-relative,
    ARM64 / PPC literal pools).  Covers both direct ram-space varnodes and
    LOAD/STORE pointers that fold to a constant.  Genuine absolute operands (x86
    ``moffs``) do not shift and are excluded, so they stay baked.

    Detection: translate the SAME bytes at two bases and keep the addresses that
    moved by exactly the base delta.  The two translations share op/varnode order,
    so the extracted address lists align positionally.
    """
    ctx = get_context(arch)
    state_format = [Register(name=n, bits=b) for n, b in state_format_tuple]
    delta = 0x40000

    def _addrs(base: int) -> list[int]:
        ops = ctx.translate(bytestring, base).ops
        mp = StateMapper(ctx, arch, state_format)
        out: list[int] = []
        for op in ops:
            if op.opcode.name in ('LOAD', 'STORE'):
                mapped_addr, const_offset = resolve_ptr_with_offset(op.inputs[1], ops, mp)
                if mapped_addr is None and const_offset != 0:
                    out.append(const_offset)
            for v in (*op.inputs, *([op.output] if op.output is not None else [])):
                if v.space.name == 'ram':
                    out.append(v.offset)
        return out

    a1 = _addrs(_TRANSLATE_BASE)
    a2 = _addrs(_TRANSLATE_BASE + delta)
    # a1 and a2 are the same instruction translated at two bases, so they share
    # op/varnode order and are equal-length by construction -- strict catches a
    # regression instead of silently truncating.
    return frozenset(x for x, y in zip(a1, a2, strict=True) if y - x == delta)


def _const_addr_mem(
    abs_offset: int,
    size: int,
    pc_relative: frozenset[int],
    pc_reg: RegMapping | None,
) -> MemMapping:
    """Build the memory mapping for a compile-time-resolved address.  PC-relative
    addresses use the runtime pc register as base (offset relative to the translate
    base); genuine absolute addresses are baked via the const-zero sentinel."""
    if pc_reg is not None and abs_offset in pc_relative:
        return MemMapping(abs_offset, size, pc_reg, abs_offset - _TRANSLATE_BASE)
    return MemMapping(abs_offset, size, _CONST_ADDR_BASE, abs_offset)


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
    (the XMM<n>/YMM<n> 64-bit lanes).  StateMapper only reads .offset and
    .size from these, never anything else, so the shim is sufficient."""

    offset: int
    size: int


# 64-bit lane -> byte offset within a vector register (ZMM<n> at 0x1200+n*0x40,
# 64 bytes).  The eight lanes cover the whole 512-bit AVX-512 register: XMM is
# bytes 0-15 (LO/HI), YMM adds the upper 128 bits (bytes 16-31), and ZMM_L4..L7
# add the upper 256 bits (bytes 32-63).  The prefix+suffix in a state-register
# name picks the lane, so a Ghidra XMM/YMM/ZMM varnode maps to the right 8 bytes.
_VEC_LANE_OFFSET = {
    ('XMM', 'LO'): 0, ('XMM', 'HI'): 8,
    ('YMM', 'LO'): 16, ('YMM', 'HI'): 24,
    ('ZMM', 'L4'): 32, ('ZMM', 'L5'): 40, ('ZMM', 'L6'): 48, ('ZMM', 'L7'): 56,
}


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
        # Byte->bit arithmetic for SUB-register reads depends on endianness: on a
        # big-endian target byte 0 of a register is its MOST significant one.  The
        # BE architectures name themselves with a `BE` suffix (MIPS64BE, PPC32BE,
        # SPARC32BE); x86/ARM64/RISCV64 are little-endian and keep the LE path.
        self.is_big_endian = arch_upper.endswith('BE')

        self.sf_resolved: list[tuple[Register, Varnode | _SynthVarnode]] = []
        for sf_reg in state_format:
            s_r: Varnode | _SynthVarnode | None = ctx.registers.get(sf_reg.name) or ctx.registers.get(
                sf_reg.name.lower(),
            )
            if not s_r and self.is_arm and sf_reg.name in self.arm_aliases:
                alias = self.arm_aliases[sf_reg.name]
                s_r = ctx.registers.get(alias) or ctx.registers.get(alias.upper())
            if not s_r and self.is_x86:
                # Synthetic XMM<n>_LO / XMM<n>_HI: pypcode has a single
                # XMM<n> at offset 0x1200+n*0x40 size 16.  We register
                # _LO at the base offset (low 64 bits) and _HI 8 bytes
                # in (high 64 bits) so Ghidra-emitted varnodes targeting
                # XMM<n>[63:0]  / XMM<n>[127:64] map correctly.
                s_r = self._synth_xmm_varnode(sf_reg.name)
            if s_r:
                self.sf_resolved.append((sf_reg, s_r))
        # Sleigh byte-offset of each state register, so a wide register output
        # split into halves (XMM<n>_LO/_HI) can locate each half within the
        # parent varnode and feed it the matching memory bytes on a wide LOAD.
        self.name_to_offset: dict[str, int] = {reg.name: sr.offset for reg, sr in self.sf_resolved}

    @staticmethod
    def _synth_xmm_varnode(name: str) -> _SynthVarnode | None:
        """Build a lightweight Varnode-like object for an XMM/YMM/ZMM 64-bit lane.

        The StateMapper only reads `.offset` and `.size` from these objects, so a
        tiny shim suffices.  Returns None for any name that is not a recognised
        vector lane (XMM<n>_LO/_HI, YMM<n>_LO/_HI, ZMM<n>_L4..L7).
        """
        if len(name) < 3:
            return None
        prefix, rest = name[:3], name[3:]
        if prefix not in ('XMM', 'YMM', 'ZMM'):
            return None
        num_str, _, half = rest.partition('_')
        try:
            n = int(num_str)
        except ValueError:
            return None
        lane = _VEC_LANE_OFFSET.get((prefix, half))
        if lane is None or not (0 <= n < 16):
            return None
        return _SynthVarnode(offset=0x1200 + n * 0x40 + lane, size=8)

    def _sub_reg_bit_start(self, rel_byte: int, size: int, reg_bytes: int) -> int:
        """Bit offset, within a state register, of a `size`-byte read at byte
        `rel_byte` from the register's base.

        Endianness-dependent.  On a BIG-ENDIAN target byte 0 of a register is its
        MOST significant one, so byte `rel_byte` of an N-byte register holds bits
        [(N - rel_byte - size)*8 ..].  Applying the little-endian `rel_byte * 8`
        maps a sub-register read to the WRONG bits: SPARC's `sll %g1,%g2,%g3` lifts
        its shift amount as register:0xb:1 -- g2's LAST byte, i.e. its
        LEAST-significant one -- which LE arithmetic reported as bits 24..31 rather
        than 0..7.  The rule then read the wrong taint bits, found none, and
        under-tainted a variable shift by a tainted amount.
        """
        if self.is_big_endian:
            return max((reg_bytes - rel_byte - size) * 8, 0)
        return rel_byte * 8

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
                bit_start = self._sub_reg_bit_start(offset - s_r.offset, size, s_r.size)
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

    initial_limit = stop_op_index if stop_op_index is not None else len(all_ops)

    def _resolve(current_vn: Varnode, limit: int) -> tuple[RegMapping | None, int]:  # noqa: C901
        # `limit` is the exclusive upper bound on op indices that may be
        # considered as definers of the current varnode. When we follow a
        # defining op at index `i`, the recursive resolves of its inputs
        # must use `limit = i`, since those inputs were read at the moment
        # op[i] fired — any later redefinition of those inputs is irrelevant.
        # This is essential for instructions like `rep stosb` where SLEIGH
        # emits a post-store INT_SUB updating RDI within the same op-list:
        # the STORE's address-input was a unique copied from RDI BEFORE that
        # update, and we must not chase that update when resolving the input.
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
                if i >= limit:
                    break
                if op.opcode.name in ('STORE', 'CALL', 'CALLIND', 'BRANCH', 'BRANCHIND', 'CBRANCH', 'RETURN'):
                    continue

                if (
                    op.output is not None
                    and op.output.space.name == 'register'
                    and op.output.offset == reg_off
                    and op.output.size == current_vn.size
                ):
                    # When we recurse into the inputs of op[i], they must be
                    # resolved as they were AT op[i], so the inner limit is i.
                    if op.opcode.name in ('COPY', 'INT_ZEXT', 'INT_SEXT'):
                        res = _resolve(op.inputs[0], i)
                        visited_reg.discard(reg_off)
                        return res
                    if op.opcode.name in ('INT_ADD', 'PTRADD'):
                        lreg, loff = _resolve(op.inputs[0], i)
                        rreg, roff = _resolve(op.inputs[1], i)
                        visited_reg.discard(reg_off)
                        if lreg is not None:
                            return lreg, loff + roff
                        if rreg is not None:
                            return rreg, roff + loff
                    elif op.opcode.name == 'INT_SUB':
                        lreg, loff = _resolve(op.inputs[0], i)
                        _, roff = _resolve(op.inputs[1], i)
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

            for i, op in enumerate(all_ops):
                if i >= limit:
                    break
                if op.output is not None and op.output.space.name == 'unique' and op.output.offset == key:
                    # Same temporal-resolution rule: inputs are read AT op[i],
                    # so recurse with limit = i.
                    if op.opcode.name in ('INT_ADD', 'PTRADD'):
                        lreg, loff = _resolve(op.inputs[0], i)
                        rreg, roff = _resolve(op.inputs[1], i)
                        if lreg is not None:
                            return lreg, loff + roff
                        if rreg is not None:
                            return rreg, roff + loff
                    elif op.opcode.name == 'INT_SUB':
                        lreg, loff = _resolve(op.inputs[0], i)
                        _, roff = _resolve(op.inputs[1], i)
                        if lreg is not None:
                            return lreg, loff - roff
                    elif op.opcode.name in ('COPY', 'INT_ZEXT', 'INT_SEXT'):
                        return _resolve(op.inputs[0], i)
                    else:
                        for inp in op.inputs:
                            r, o = _resolve(inp, i)
                            if r is not None:
                                return r, o
                    break

            return None, 0
        return None, 0

    return _resolve(vn, initial_limit)


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
            # The split is RELATIVE to the operand's own polarity.  A signed compare
            # is monotone only in the sign-biased representation, so the sign bit
            # always carries the OPPOSITE polarity to the magnitude bits -- but for a
            # SUBTRACTED operand the whole pattern flips.  Hardcoding (+1 magnitude,
            # -1 sign) polarised both operands of `cmp rax,[rsp-16]` identically,
            # degrading the differential to a lossy D^{++} that cancels and
            # under-taints the comparison once the memory value is itself tainted.
            # Convention: > 0 positive, <= 0 negative.
            mag_pol = 1 if p > 0 else -1
            sign_pol = -mag_pol
            if matched_msb > dep_map.bit_start:
                new_deps[RegMapping(dep_map.name, dep_map.bit_start, matched_msb - 1)] = mag_pol
            new_deps[RegMapping(dep_map.name, matched_msb, matched_msb)] = sign_pol
            if dep_map.bit_end > matched_msb:
                new_deps[RegMapping(dep_map.name, matched_msb + 1, dep_map.bit_end)] = mag_pol
        else:
            new_deps[dep_map] = p

    return new_deps


@functools.lru_cache(maxsize=16384)
def _cached_generate_static_rule(  # noqa: C901
    arch: Architecture,
    bytestring: bytes,
    state_format_tuple: tuple[tuple[str, int], ...],
) -> LogicCircuit:
    state_format = [Register(name=name, bits=bits) for name, bits in state_format_tuple]

    ctx = get_context(arch)
    translation = ctx.translate(bytestring, 0x1000)

    outputs, store_ops, ram_outputs = get_register_outputs_and_stores(translation)
    unique_outputs = {get_varnode_id(out): out for out in outputs}.values()

    mapper = StateMapper(ctx, arch, state_format)

    # Which compile-time-constant memory addresses are PC-relative (resolve against
    # the runtime pc) vs genuinely absolute (baked).  Shared by outputs and deps.
    pc_relative = _pc_relative_addrs(arch, bytestring, state_format_tuple)
    pc_reg = _pc_reg_mapping(arch, state_format)

    targets_to_evaluate, assignments = map_outputs_to_targets(
        arch,
        state_format,
        translation,
        store_ops,
        unique_outputs,
        mapper,
        ram_outputs,
        pc_relative,
        pc_reg,
    )

    for target in targets_to_evaluate:
        out_vn = target.varnode
        mapping = target.mapping

        slice_ops = slice_backward(translation.ops, out_vn)
        polarities = compute_polarity(slice_ops)

        # When a wide (>8-byte) register output is split into halves (XMM<n>_LO /
        # _HI both mapped from the same 16-byte varnode, as movdqu/movdqa loads
        # emit), tell extract_dependencies which byte range of the output THIS
        # half occupies, so a wide LOAD feeds each half its own memory bytes.
        load_sub: tuple[int, int] | None = None
        if (
            isinstance(mapping, RegMapping)
            and out_vn.space.name == 'register'
            and out_vn.size > 8
        ):
            half_off = mapper.name_to_offset.get(mapping.name, out_vn.offset) - out_vn.offset
            half_size = (mapping.bit_end - mapping.bit_start + 1) // 8
            if 0 <= half_off and half_off + half_size <= out_vn.size:
                load_sub = (half_off, half_size)

        dep_set = extract_dependencies(
            out_vn, slice_ops, polarities, translation.ops, mapper, pc_relative, pc_reg, load_sub,
        )

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

        # Detect whether this instruction has a *forward* CBRANCH — meaning the
        # branch skips over a write to the output register (conditional-move pattern).
        # Backward CBRANCHes are loop bodies (tzcnt, bsf, bsr lift as software loops);
        # the output write is on the loop-exit path and is always executed, so the
        # not-taken passthrough must NOT be applied there.
        #
        # Heuristic: a CBRANCH is "forward" when its target address is greater than
        # the instruction base address (0x1000 in our lifted translation). A backward
        # branch target is smaller (loops back).
        # A SOFTWARE LOOP (tzcnt/bsf/bsr/PEXT) is identified by a BACKWARD branch --
        # the loop back-edge.  Its const-space forward CBRANCH is the zero-check
        # guarding the loop, not a conditional select: the output is written on the
        # loop-exit path and always executes, so the old-dest passthrough must NOT
        # apply (it would leak the destination's previous taint).  cmpxchg, by
        # contrast, is straight-line and has no backward branch.
        _BASE_ADDR = 0x1000
        _has_backward_branch = any(
            op.opcode.name in ('BRANCH', 'CBRANCH')
            and op.inputs
            and op.inputs[0].space.name == 'const'
            and (op.inputs[0].offset & 0x80000000)
            for op in translation.ops
        )

        def _is_forward_cbranch(op: PcodeOp) -> bool:
            """True for a CBRANCH that skips forward over a CONDITIONAL WRITE.

            Two target encodings occur:
              * ram/absolute  -> forward when the target is beyond the instruction
                base (the CMOVcc skip).
              * const/p-code-relative -> a signed op-index delta.  A positive delta
                is a forward skip; cmpxchg lifts its `ZF ? src : dst` select exactly
                this way, and the absolute-only check used to miss it.  Excluded in
                a software loop, where the const CBRANCHes are loop machinery.
            """
            if op.opcode.name != 'CBRANCH' or not op.inputs:
                return False
            tgt = op.inputs[0]
            if tgt.space.name == 'const':
                return not (tgt.offset & 0x80000000) and not _has_backward_branch  # noqa: B023
            return tgt.offset > _BASE_ADDR  # noqa: B023 -- loop-invariant constant

        has_cbranch = any(_is_forward_cbranch(op) for op in translation.ops)

        # Walk back from the CBRANCH condition to find the 1-bit flag registers
        # that determine the branch.  These are needed to gate the cmov old-dest
        # passthrough on whether the condition is concretely known: when none of
        # the flag deps are tainted, the differential C1⊕C2 alone gives the exact
        # answer (in both reps the cmov takes the same path with the same
        # concrete flags), and the old-dest passthrough only adds spurious bits.
        cbranch_flag_deps: list[tuple[int, int]] = []
        cbranch_op = None
        if has_cbranch:
            cbranch_op = next(op for op in translation.ops if _is_forward_cbranch(op))
            _worklist = [cbranch_op.inputs[1]]  # condition input
            _seen: set[tuple[str, int, int]] = set()
            while _worklist:
                _vn = _worklist.pop()
                _key = (_vn.space.name, _vn.offset, _vn.size)
                if _key in _seen:
                    continue
                _seen.add(_key)
                if _vn.space.name == 'register' and _vn.size == 1:
                    cbranch_flag_deps.append((_vn.offset, _vn.size))
                elif _vn.space.name == 'unique':
                    for _prev in translation.ops:
                        if _prev.output is None:
                            continue
                        if _prev.output.space.name == _vn.space.name and _prev.output.offset == _vn.offset:
                            for _inp in _prev.inputs:
                                if _inp.space.name in ('register', 'unique'):
                                    _worklist.append(_inp)

        # A condition flag DEFINED inside this same instruction (cmpxchg computes ZF
        # from RAX vs the destination, then selects on it) is not a tracked state
        # register: its threaded taint is always 0 and cannot gate the passthrough.
        # Detect that so the gate can fall back to the taint of this instruction's
        # own value inputs.  A live-in flag (the chained `cmp; cmovcc` pattern) has
        # no defining op here and keeps the precise flag-register gate.
        cbranch_cond_internal = any(
            any(
                p.output is not None
                and p.output.space.name == 'register'
                and p.output.offset == _foff
                and p.output.size == _fsz
                for p in translation.ops
            )
            for _foff, _fsz in cbranch_flag_deps
        )

        # The old-dest passthrough must fire ONLY when THIS output can actually
        # retain its previous value, i.e. its write is guarded by the CBRANCH.  A
        # conditional write is defined AFTER the CBRANCH in the linear P-code; an
        # unconditional one (rol/ror's data output, whose CBRANCH only guards an
        # undefined OF) precedes it, and injecting old-dest there would over-taint.
        output_cond_written = False
        if has_cbranch:
            _cb_idx = next((i for i, op in enumerate(translation.ops) if _is_forward_cbranch(op)), -1)
            output_cond_written = any(
                i > _cb_idx
                and op.output is not None
                and op.output.space.name == out_vn.space.name
                and op.output.offset == out_vn.offset
                for i, op in enumerate(translation.ops)
            )

        # Detect bit-counting patterns whose result is bounded by the operand
        # width.  For tzcnt/lzcnt/bsf/bsr/popcnt the count fits in
        # ⌈log2(width+1)⌉ bits (e.g. 7 bits for 64-bit operands), regardless of
        # how many input bits are tainted.  We compute is_bit_count here and
        # pass it down so that AVALANCHE/TRANSPORTABLE branches can cap the
        # taint mask to the appropriate width.
        _slice_op_names = [op.opcode.name for op in slice_ops]
        is_bit_count = False
        if 'LZCOUNT' in _slice_op_names:
            is_bit_count = True
        elif 'POPCOUNT' in _slice_op_names:
            # Distinguish a true popcnt instruction from the PF-flag POPCOUNT
            # (which operates on a single masked byte, size < 4).
            for _op in slice_ops:
                if _op.opcode.name == 'POPCOUNT' and _op.inputs[0].size >= 4:
                    is_bit_count = True
                    break
        else:
            # Software-loop bit scans: backward BRANCH + counter step.
            _has_backward = any(
                op.opcode.name == 'BRANCH'
                and op.inputs
                and op.inputs[0].space.name == 'const'
                and (op.inputs[0].offset & 0x80000000)
                for op in translation.ops
            )
            _has_counter_step = any(
                op.opcode.name in ('INT_ADD', 'INT_SUB')
                and len(op.inputs) == 2
                and op.inputs[1].space.name == 'const'
                and op.inputs[1].offset == 1
                for op in slice_ops
            )
            is_bit_count = _has_backward and _has_counter_step

        # Software-loop full-width avalanche.
        # PEXT/PDEP (and similar BMI2 ops) lift to a CONDITIONAL backward
        # branch (CBRANCH offset<0) — distinct from tzcnt/bsf/bsr which use
        # an UNCONDITIONAL backward BRANCH paired with a separate condition.
        # The slicer drops CBRANCH ops (no `output` field) so the loop
        # structure is invisible at slice level: we detect it here, where
        # `translation.ops` is in scope, and force AVALANCHE in
        # generate_taint_assignments.
        # Output width is unbounded (up to register width) — different from
        # is_bit_count which caps to log2(width+1).
        is_software_loop = not is_bit_count and any(
            op.opcode.name == 'CBRANCH'
            and op.inputs
            and op.inputs[0].space.name == 'const'
            and (op.inputs[0].offset & 0x80000000)
            for op in translation.ops
        )

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
            has_cbranch=has_cbranch,
            cbranch_flag_deps=cbranch_flag_deps,
            cbranch_cond_internal=cbranch_cond_internal,
            output_cond_written=output_cond_written,
            is_bit_count=is_bit_count,
            is_software_loop=is_software_loop,
            all_ops=translation.ops,
            cbranch_op=cbranch_op,
        )

    return LogicCircuit(
        assignments=assignments,
        architecture=arch,
        instruction=bytestring.hex(),
        state_format=state_format,
    )


def _branch_forces_monolithic(
    op: PcodeOp,
    cur_off: int,
    instr_offsets: frozenset[int],
    next_instr_addr: int,
) -> bool:
    """True iff this branch op must keep a multi-instruction sequence monolithic.

    Pcode emits CBRANCH for several purposes; only some force the monolithic
    block. ``cur_off`` is the byte offset (relative to BASE=0x1000) of the
    branch's own instruction.

      - CMOVcc skip (target == next_instr_addr) and const-space targets
        (pcode-internal BSF/BSR loops, CMOV internals): chainable.
      - Intra-sequence FORWARD skip (ram target on an instruction boundary
        strictly after this instruction): chaining applies the skipped
        instruction anyway, which only OVER-approximates taint (sound) and stays
        deterministic -> chainable.
      - Backward branch/loop, out-of-sequence jump, or indirect branch: keep the
        monolithic circuit, which preserves the cross-instruction flag->RIP
        dependency the backward slicer needs.
    """
    name = op.opcode.name
    if name == 'BRANCHIND':
        return True  # indirect branch: target is data-dependent, always exits
    if name not in ('BRANCH', 'CBRANCH'):
        return False
    dest = op.inputs[0]
    if dest.space.name == 'const':
        return False  # pcode-internal (BSF loops, CMOVcc uses const offset internally)
    if dest.space.name == 'ram':
        if dest.offset == next_instr_addr:
            return False  # CMOVcc skip to just after the sequence
        tgt = dest.offset - 0x1000
        if tgt in instr_offsets and tgt > cur_off:
            return False  # intra-sequence forward skip: safe to chain
        return True  # backward (loop) or out-of-sequence branch
    return True  # unknown space: conservative, treat as architectural


def _sequence_needs_monolithic(
    ops: list[PcodeOp],
    instr_offsets: frozenset[int],
    next_instr_addr: int,
) -> bool:
    """A multi-instruction sequence must be kept as one monolithic circuit
    (not split into a per-instruction ChainedCircuit) if it contains memory/call
    ops or any branch that is not an intra-sequence forward skip. Forward-skip-
    only and straight-line sequences are chained.
    """
    non_branch_skip = frozenset({'STORE', 'LOAD', 'CALL', 'CALLIND', 'RETURN'})
    cur_off = 0
    for op in ops:
        name = op.opcode.name
        if name == 'IMARK':
            cur_off = op.inputs[0].offset - 0x1000
            continue
        if name in non_branch_skip:
            return True
        if _branch_forces_monolithic(op, cur_off, instr_offsets, next_instr_addr):
            return True
    return False


def generate_static_rule(
    arch: Architecture,
    bytestring: bytes,
    state_format: list[Register],
) -> LogicCircuit:
    reg_names_tuple = tuple((reg.name, reg.bits) for reg in state_format)
    circuit = _cached_generate_static_rule(arch, bytestring, reg_names_tuple)

    # Multi-instruction sequences must be evaluated per-instruction so that
    # intermediate taint state (produced by instruction N) is correctly
    # visible to instruction N+1.  Lifting all instructions into one P-code
    # block and analysing them as a unit loses this: the dep-extraction only
    # sees original-input register names, so an intermediate updated CL after
    # `mov rcx, rdx` is invisible to the subsequent `shr rax, cl`.
    #
    # Detect multiple instructions via IMARK boundaries.  If there is more
    # than one instruction, split the bytestring, generate a circuit per
    # instruction, and wrap the whole thing in a ChainedCircuit that threads
    # the output taint of each step into the input taint of the next.
    #
    # Exceptions — keep as a single monolithic circuit when:
    #   1. Architectural BRANCH/CBRANCH present: a real conditional jump (JNE, JL …)
    #      or indirect branch creates a cross-instruction ZF/SF/… → RIP dependency
    #      that the backward slicer can trace only in the monolithic block.
    #      Splitting loses it.
    #
    #      IMPORTANT: CMOVcc instructions also emit a CBRANCH internally, but their
    #      target is always exactly next_instr_addr (a forward skip to skip the
    #      register write).  That is NOT an architectural branch — RIP always
    #      advances unconditionally — so it must NOT suppress chaining.  Chaining
    #      is precisely what allows flag taint from a preceding CMP to flow into
    #      the CMOV's condition gate (e.g. `cmp rax,rbx; cmovl rdx,rcx` would
    #      otherwise lose T_SF/T_OF between the two instructions).
    #
    #      Const-space CBRANCH targets are pcode-internal (BSF/BSR bit-scan loops,
    #      repeat-string prefixes) and also never exit the sequence — safe to chain.
    #
    #   2. STORE/LOAD present: shadow memory is not threaded between steps,
    #      so memory-taint would be silently lost (push/pop, load-then-store).
    ctx = get_context(arch)
    translation = ctx.translate(bytestring, 0x1000)
    imarks = [(op.inputs[0].offset - 0x1000, op.inputs[0].size) for op in translation.ops if op.opcode.name == 'IMARK']
    if len(imarks) <= 1:
        return circuit  # single instruction — no chaining needed

    _next_instr_addr = 0x1000 + len(bytestring)
    _instr_offsets = frozenset(off for off, _ in imarks)
    if _sequence_needs_monolithic(translation.ops, _instr_offsets, _next_instr_addr):
        return circuit  # monolithic circuit preserves cross-instruction deps

    # Build one sub-circuit per instruction, using a state_format augmented with
    # the sequence's intra-instruction intermediate registers so that a carry/flag
    # produced by one instruction flows into the next (add;adc, subs;sbc,
    # subcc;subx, subfc;subfe, ...).  Without the carry register in the sub-circuit
    # state_format its taint never appears in the assignments and ChainedCircuit
    # cannot thread it forward.
    #
    # The set is discovered from the p-code itself -- ANY register-space varnode
    # that is WRITTEN by one op and READ by another in the sequence is an
    # intra-sequence intermediate -- so the threading is fully ISA-agnostic: x86
    # CF/OF, ARM64 C, PPC xer_ca, SPARC i_cf, and any future ISA's condition
    # register are all picked up by geometry, with no hardcoded per-arch flag list.
    # Over-inclusion is safe: the final output dict is filtered back to the
    # caller's state_format by ChainedCircuit.evaluate.
    _name_by_key: dict[tuple[int, int], str] = {
        (vn.offset, vn.size): nm for nm, vn in ctx.registers.items() if vn.space.name == 'register'
    }
    _written: set[tuple[int, int]] = set()
    _read: set[tuple[int, int]] = set()
    for _op in translation.ops:
        if _op.output is not None and _op.output.space.name == 'register':
            _written.add((_op.output.offset, _op.output.size))
        for _inp in _op.inputs:
            if _inp.space.name == 'register':
                _read.add((_inp.offset, _inp.size))
    _existing_names = {name for name, _ in reg_names_tuple}
    # Only thread scalar intermediates (<=8 bytes: flags, carries, GPRs) -- these
    # are what a carry/condition chain flows through.  Wider register-space
    # varnodes are vector registers, handled by the SIMD lane machinery, not this
    # 64-bit-mask state threading; adding one here builds a >64-bit Register and
    # corrupts the mask path (a 128-bit XMM roundtrip segfaults the native cell).
    _extra_flags = tuple(
        (_name_by_key[_k], _k[1] * 8)
        for _k in sorted(_written & _read)
        if _k[1] <= 8 and _k in _name_by_key and _name_by_key[_k] not in _existing_names
    )
    sub_reg_names_tuple = reg_names_tuple + _extra_flags if _extra_flags else reg_names_tuple

    sub_circuits: list[LogicCircuit] = []
    for addr_offset, length in imarks:
        instr_bytes = bytestring[addr_offset : addr_offset + length]
        sub_circuits.append(_cached_generate_static_rule(arch, instr_bytes, sub_reg_names_tuple))

    return ChainedCircuit(
        sub_circuits=sub_circuits,
        architecture=arch,
        instruction=bytestring.hex(),
        state_format=state_format,
    )


def _build_signed_overflow_taint(  # noqa: C901
    slice_ops: list[PcodeOp],
    mapper: StateMapper,
) -> Expr | None:
    """EXACT taint for a signed-overflow flag (INT_SBORROW / INT_SCARRY), or None.

    Signed overflow is NON-MONOTONE, so the 2-replica differential can miss it: the
    two extremal corners coincidentally agree while an interior flip of a tainted bit
    toggles OF (the `sub rax,rbx; seto dl` under-taint).  The FullMask/pairwise floors
    do not cover it either, since they assume the differential is exact whenever a
    single operand is only PARTIALLY tainted -- false for signed overflow.

    Instead of a floor (which would over-taint), route the flag to the exact sign
    decomposition -- see SignedOverflowTaintExpr and the Z3 proof in
    benchmark/soundness/prove_signed_overflow.py (identity + no-under-taint PROVED for
    w=2..64; no-over-taint PROVED for w<=6).

    Returns None -- so the caller keeps the differential + floor -- whenever the slice
    is not a plain two-operand signed-overflow flag (memory/unique operands, several
    overflow ops, ...).
    """
    ovf = [op for op in slice_ops if op.opcode.name in ('INT_SBORROW', 'INT_SCARRY')]
    if not ovf or len(ovf) > 2 or any(len(o.inputs) != 2 for o in ovf):
        return None
    # Nothing may compute after the overflow structure except value-preserving
    # COPYs, which must be tolerated: ARM64 lifts flags into scratch varnodes and
    # then copies them to the architectural NZCV registers, so the OV slice ends
    # in ``COPY OV <- scratch`` rather than in the INT_SCARRY itself.  Requiring
    # the overflow op to be literally last made the exact term decline on every
    # ARM64 `cmn`/`cmp`.  For the two-op carry-in shape the joining BOOL_XOR is
    # also allowed after them.
    _last_ovf = max(slice_ops.index(o) for o in ovf)
    _tail = [
        o
        for o in slice_ops[_last_ovf + 1 :]
        if not (len(ovf) == 2 and o.opcode.name in ('BOOL_XOR', 'INT_XOR'))
    ]
    if any(
        o.opcode.name != 'COPY' or o.output is None or o.output.size != o.inputs[0].size
        for o in _tail
    ):
        return None

    def _through_copies(vn: Varnode) -> Varnode:
        """Follow value-preserving COPYs from a unique back to its source varnode.

        Several ISAs stage an operand through a temporary before the overflow op:
        x86 `cmp` lifts as ``COPY u <- RAX; INT_SBORROW OF <- u, RBX``, and ARM64
        `cmn` does the same.  Without this, the `u` operand is not a register, the
        exact term declines, and the flag silently falls back to the differential —
        which is precisely what signed overflow is non-monotone enough to escape.
        Only same-size COPY is followed, so no value is reinterpreted.
        """
        for _ in range(8):  # p-code is acyclic; the bound is belt-and-braces
            if vn.space.name != 'unique':
                return vn
            src = next(
                (
                    o.inputs[0]
                    for o in slice_ops
                    if o.opcode.name == 'COPY'
                    and o.output is not None
                    and o.output.space.name == 'unique'
                    and o.output.offset == vn.offset
                    and o.output.size == vn.size
                    and o.inputs[0].size == vn.size
                ),
                None,
            )
            if src is None:
                return vn
            vn = src
        return vn

    def _key_vn(vn: Varnode) -> tuple[str, int, int]:
        return (vn.space.name, vn.offset, vn.size)

    def _defining(vn: Varnode) -> PcodeOp | None:
        if vn.space.name != 'unique':
            return None
        return next(
            (
                o
                for o in slice_ops
                if o.output is not None
                and o.output.space.name == 'unique'
                and o.output.offset == vn.offset
                and o.output.size == vn.size
            ),
            None,
        )

    def _resolve(raw_vn: Varnode) -> tuple[Expr, Expr, int] | None:  # noqa: C901
        """Resolve an operand varnode to (value expr, taint expr, width in bits).

        Beyond plain registers and constants this sees through the two
        value-shaping wrappers ISAs put around a carry operand:
        ``INT_NEGATE`` (ARM64 lifts `sbc` as ``x1 + ~x2 + CY``, so the second
        operand is a bitwise complement -- which relocates no bits, hence carries
        the same taint mask) and ``INT_ZEXT`` of a 1-bit flag (the carry-in
        itself).  Returns None for anything else, so the caller declines rather
        than guessing.
        """
        vn = _through_copies(raw_vn)
        if vn.space.name == 'register':
            m = mapper.map_to_state(vn.offset, vn.size)
            if m is None:
                return None
            return (
                _get_taint_operand(m.name, m.bit_start, m.bit_end, False),
                _get_taint_operand(m.name, m.bit_start, m.bit_end, True),
                vn.size * 8,
            )
        if vn.space.name == 'const':
            return (Constant(vn.offset, vn.size * 8), Constant(0, vn.size * 8), vn.size * 8)
        d = _defining(vn)
        if d is None:
            return None
        if d.opcode.name == 'INT_NEGATE':
            inner = _resolve(d.inputs[0])
            if inner is None:
                return None
            v, t, _ = inner
            return (UnaryExpr(Op.NOT, v), t, vn.size * 8)
        if d.opcode.name == 'INT_ZEXT':
            inner = _resolve(d.inputs[0])
            if inner is None:
                return None
            v, t, _ = inner
            return (v, t, vn.size * 8)
        if d.opcode.name in ('INT_LEFT', 'INT_RIGHT'):
            # A PRE-PROCESSED operand: ARM64's shifted-register compare lifts as
            # `u = x2 << 3 ; tmpOV = sborrow(x1, u)`.  Requiring a bare register
            # here made the exact term decline and the flag fall back to the
            # differential -- which is exactly what non-monotone signed overflow
            # slips through.  A constant-amount logical shift relocates bits
            # without mixing them, so both the value and the taint transform by
            # the same shift and stay closed-form.  INT_SRIGHT is NOT handled: its
            # value needs a sign-replicating fill, so it declines rather than
            # guessing.
            amount = const_value(d.inputs[1], fold_constants(slice_ops))
            inner = _resolve(d.inputs[0])
            if amount is None or inner is None:
                return None
            v, t, _w = inner
            shift_op = Op.LEFT if d.opcode.name == 'INT_LEFT' else Op.RIGHT
            bits = vn.size * 8
            keep = Constant((1 << bits) - 1, 8)
            return (
                BinaryExpr(Op.AND, BinaryExpr(shift_op, v, Constant(amount, 8)), keep),
                BinaryExpr(Op.AND, BinaryExpr(shift_op, t, Constant(amount, 8)), keep),
                bits,
            )
        return None

    # ---- shape 1: a plain two-operand overflow flag -------------------------
    if len(ovf) == 1:
        op = ovf[0]
        if len(op.inputs) != 2:
            return None
        resolved = [_resolve(vn) for vn in op.inputs]
        if any(r is None for r in resolved):
            return None
        (a_val, a_taint, aw), (b_val, b_taint, bw) = resolved  # type: ignore[misc]
        width = max(aw, bw)
        if width < 2:
            return None
        return SignedOverflowTaintExpr(
            a_val,
            a_taint,
            b_val,
            b_taint,
            width,
            op.opcode.name == 'INT_SBORROW',
        )

    # ---- shape 2: a carry-in chain (adc / sbb / adcs / sbcs) ----------------
    #
    # Every ISA checked lifts a carry-in add/subtract's overflow flag the same way::
    #
    #     OF = ovf(A, B)  XOR  ovf(A o B, C)          C = zext(carry flag)
    #
    #   x86   adc:  scarry(RAX,RBX)  ^^ scarry(RAX+RBX,  zext(CF))
    #   x86   sbb:  sborrow(RAX,RBX) ^^ sborrow(RAX-RBX, zext(CF))
    #   ARM64 adcs: scarry(x1,x2)    ^^ scarry(x1+x2,    zext(CY))
    #   ARM64 sbcs: scarry(x1,~x2)   ^^ scarry(x1+~x2,   zext(CY))
    #
    # This is just the signed overflow of the THREE-operand sum A + B + C.  Because
    # C is zext of a 1-bit flag it lies wholly inside the low part and its sign bit
    # is 0, so it shifts only the carry/borrow INTO the msb and leaves the sign
    # function unchanged -- the two-operand rule generalises with a third monotone
    # contributor rather than needing a new decomposition.  Bor/Car stays monotone
    # (increasing in C for both families), and A's sign bit, B's sign bit and
    # Bor/Car still read disjoint inputs, so the enumeration remains exact.
    #
    # Without this the slice has two overflow ops, the exact term declines, and OF
    # falls back to the 2-corner differential -- which non-monotone signed overflow
    # is precisely able to slip through.
    if len(ovf) != 2 or ovf[0].opcode.name != ovf[1].opcode.name:
        return None
    terminal = next(
        (o for o in reversed(slice_ops) if o.opcode.name in ('BOOL_XOR', 'INT_XOR')), None,
    )
    if terminal is None or len(terminal.inputs) != 2:
        return None
    xor_srcs = {_key_vn(_through_copies(v)) for v in terminal.inputs}
    if xor_srcs != {_key_vn(o.output) for o in ovf if o.output is not None}:
        return None

    is_sub = ovf[0].opcode.name == 'INT_SBORROW'
    combiner = 'INT_SUB' if is_sub else 'INT_ADD'
    # Identify which overflow op consumes the partial result A o B.
    outer = None
    for cand in ovf:
        d = _defining(_through_copies(cand.inputs[0]))
        if d is not None and d.opcode.name == combiner:
            outer, partial = cand, d
            break
    if outer is None:
        return None
    inner = ovf[0] if outer is ovf[1] else ovf[1]
    # The partial result must be exactly the inner overflow op's own two operands.
    if [_key_vn(v) for v in partial.inputs] != [_key_vn(v) for v in inner.inputs]:
        return None

    resolved = [_resolve(vn) for vn in inner.inputs]
    c_resolved = _resolve(outer.inputs[1])
    if any(r is None for r in resolved) or c_resolved is None:
        return None
    (a_val, a_taint, aw), (b_val, b_taint, bw) = resolved  # type: ignore[misc]
    c_val, c_taint, _cw = c_resolved
    width = max(aw, bw)
    if width < 2:
        return None
    # The carry-in must be a single bit, or its sign bit could reach the msb and the
    # decomposition's premise (c_s == 0) would not hold.
    zext_src = _defining(_through_copies(outer.inputs[1]))
    if zext_src is None or zext_src.opcode.name != 'INT_ZEXT' or zext_src.inputs[0].size != 1:
        return None

    return SignedOverflowTaintExpr(
        a_val,
        a_taint,
        b_val,
        b_taint,
        width,
        is_sub,
        c_val,
        c_taint,
    )


_SHIFT_KIND = {'INT_LEFT': 0, 'INT_RIGHT': 1, 'INT_SRIGHT': 2}
_SHIFT_PASSTHROUGH = frozenset({'COPY', 'INT_SEXT', 'INT_ZEXT'})
_MUL_PASSTHROUGH = frozenset({'COPY', 'INT_ZEXT', 'INT_SEXT', 'SUBPIECE', 'INT_MULT'})


def _build_variable_multiply_taint(  # noqa: C901
    slice_ops: list[PcodeOp],
    mapper: StateMapper,
    out_width: int,
) -> Expr | None:
    """Sound fill for the taint of a multiply with tainted operands, or None.

    Multiply otherwise falls to AvalancheExpr (full-width taint) and was the
    second-largest source of over-tainted bits in the campaign (26.7%, 0.73x).
    The 2-corner differential alone UNDER-taints -- ``max^min`` misses interior
    products (see VariableMultiplyTaintExpr) -- so this term computes the sound
    fill ``[tz_lo(a)+tz_lo(b) .. highbit(max^min)]`` on the full 2w product and
    extracts the returned word.

    Recognises the three lift shapes, all ISA-independent:

        dst = INT_MULT(a, b)                              low word      (mul, imul)
        dst = SUBPIECE(INT_MULT(ext a, ext b), w/8)       high word (umulh/smulh)
        dst = INT_MULT(ext a, ext b)                      full, widening (umull)

    where ``ext`` is INT_ZEXT (unsigned) or INT_SEXT (signed) -- which is also how
    the signedness of the high half is read.  Declines for anything else, so the
    differential path is unchanged.
    """
    muls = [op for op in slice_ops if op.opcode.name == 'INT_MULT']
    if len(muls) != 1:
        return None
    mul = muls[0]
    if mul.output is None or len(mul.inputs) != 2:
        return None

    def _defining(vn: Varnode) -> PcodeOp | None:
        if vn.space.name != 'unique':
            return None
        return next(
            (
                o
                for o in slice_ops
                if o.output is not None
                and o.output.space.name == 'unique'
                and o.output.offset == vn.offset
                and o.output.size == vn.size
            ),
            None,
        )

    # An operand is either a bare register, or a register widened by zext/sext.
    # The ext type must agree between the two operands (mixed signedness would be a
    # different instruction), and it sets the signed flag.
    signed_flags: set[bool] = set()

    def _operand(vn: Varnode) -> tuple[RegMapping, int] | None:
        d = _defining(vn)
        if d is not None and d.opcode.name in ('INT_ZEXT', 'INT_SEXT'):
            signed_flags.add(d.opcode.name == 'INT_SEXT')
            inner = d.inputs[0]
            width = inner.size * 8
            if inner.space.name != 'register':
                inner_d = _defining(inner)
                if inner_d is None or inner_d.opcode.name != 'COPY':
                    return None
                inner = inner_d.inputs[0]
            m = mapper.map_to_state(inner.offset, inner.size)
            return (m, width) if isinstance(m, RegMapping) else None
        if vn.space.name == 'register':
            m = mapper.map_to_state(vn.offset, vn.size)
            return (m, vn.size * 8) if isinstance(m, RegMapping) else None
        return None

    a_op = _operand(mul.inputs[0])
    b_op = _operand(mul.inputs[1])
    if a_op is None or b_op is None:
        return None
    (a_map, a_w), (b_map, b_w) = a_op, b_op
    if a_w != b_w:
        return None
    in_width = a_w
    if in_width < 2:
        return None
    if len(signed_flags) > 1:
        return None  # mixed zext/sext -> not a plain signed or unsigned multiply
    is_signed = signed_flags == {True}

    # A constant operand is handled exactly by the differential; both must be data.
    if a_map.name == b_map.name and a_map.bit_start == b_map.bit_start:
        # a squared is still a genuine variable multiply -- allow it.
        pass

    # Output window into the 2w-bit product.  This must be read EXACTLY, not
    # guessed: MIPS `mul` lifts as ``v0 = sext(<low 4 bytes of the product>)``, so
    # its output bits 32..63 are replications of product bit 31, NOT the product's
    # own bits 32..63.  Treating the window as [0,64) there under-tainted 553/900
    # cases against ground truth.  Rather than model every truncate-and-widen
    # shape, accept only the two unambiguous ones and decline the rest (falling
    # back to the avalanche, which is sound).
    prod_bits = 2 * in_width
    consumers = [
        o
        for o in slice_ops
        if o is not mul
        and any(
            i.space.name == mul.output.space.name
            and i.offset == mul.output.offset
            and i.size == mul.output.size
            for i in o.inputs
        )
    ]
    # Any op reading only PART of the product varnode (a sub-varnode read, which is
    # not an exact-match consumer) means the result is truncated in a way this term
    # does not model.
    partial_readers = [
        o
        for o in slice_ops
        if o is not mul
        and any(
            i.space.name == mul.output.space.name
            and i.offset != mul.output.offset
            and mul.output.offset <= i.offset < mul.output.offset + mul.output.size
            for i in o.inputs
        )
    ]
    if partial_readers:
        return None

    if not consumers:
        # The multiply writes the architectural output directly: low word, or the
        # full product for a widening multiply.
        out_lo, out_hi = 0, min(out_width, prod_bits)
    elif len(consumers) == 1 and consumers[0].opcode.name == 'SUBPIECE':
        out_lo = consumers[0].inputs[1].offset * 8
        out_hi = min(out_lo + out_width, prod_bits)
    else:
        return None
    if out_hi <= out_lo:
        return None

    # Only the recognised multiply-shaped ops may appear; anything else means the
    # slice does more than this term models.
    for op in slice_ops:
        if op.opcode.name not in _MUL_PASSTHROUGH:
            return None

    return VariableMultiplyTaintExpr(
        _get_taint_operand(a_map.name, a_map.bit_start, a_map.bit_end, False),
        _get_taint_operand(a_map.name, a_map.bit_start, a_map.bit_end, True),
        _get_taint_operand(b_map.name, b_map.bit_start, b_map.bit_end, False),
        _get_taint_operand(b_map.name, b_map.bit_start, b_map.bit_end, True),
        in_width,
        is_signed,
        out_lo,
        out_hi,
    )


def _build_wide_xor_taint(
    slice_ops: list[PcodeOp],
    mapper: StateMapper,
) -> Expr | None:
    """EXACT taint of a wide XOR whose operands are permutations, or None.

    T(a ^ b) is the position-wise union T_a | T_b: an output bit varies iff either
    contributing bit can vary, and unlike AND/OR there is no value-dependent
    masking to make the union pessimistic.  Both operand taints are resolved in
    closed form through varnode_taint_expr, so a shifted or rotated operand
    contributes its taint at the TRANSFORMED positions -- which is the whole point,
    since the raw register taint sits at the wrong bits.

    Declines unless the producing op is an XOR with two resolvable data operands,
    leaving the differential in place.
    """
    term = next(
        (
            o
            for o in reversed(slice_ops)
            if o.opcode.name not in ('COPY', 'SUBPIECE', 'PIECE', 'INT_ZEXT', 'INT_SEXT')
        ),
        None,
    )
    if term is None or term.opcode.name != 'INT_XOR' or len(term.inputs) != 2:
        return None
    if term.output is None or term.output.size * 8 <= 1:
        return None
    folded = fold_constants(slice_ops)
    if _is_bitwise_not(term, folded):
        return None  # a NOT, not a two-operand XOR: polarity handles it
    parts: list[Expr] = []
    for inp in term.inputs:
        if const_value(inp, folded) is not None:
            continue  # a constant contributes no taint
        t = varnode_taint_expr(slice_ops, inp, _reg_taint_for_floor(mapper))
        if t is None:
            return None
        parts.append(t)
    if len(parts) != 2:
        return None
    return BinaryExpr(Op.OR, parts[0], parts[1])


def _build_variable_shift_taint(  # noqa: C901
    slice_ops: list[PcodeOp],
    mapper: StateMapper,
    out_width: int,
) -> Expr | None:
    """EXACT taint for a shift by a DATA-DEPENDENT amount, or None.

    A tainted shift amount otherwise forces AvalancheExpr, tainting the whole
    output width.  Across a 2M-case five-ISA campaign that single behaviour was
    69.8% of all over-tainted bits (4.13x invented bits) from 21 of 175
    instructions, while everything outside shifts and multiply sat at 0.07x.

    Every ISA checked lifts the variable shift identically::

        u   = amount & MASK          (INT_AND with a constant)
        dst = src <shift> u          (INT_LEFT / INT_RIGHT / INT_SRIGHT)

      ARM64 `lslv`  x2 & 0x3f ; x1 << u
      MIPS  `sllv`  a1 & 0x1f ; a0 << u ; sext            (32-bit core, widened)
      PPC   `slw`   r5 & 0x3f ; r4 << u
      RISCV `sll`   likewise

    so the recogniser is structural rather than per-ISA.  The masking INT_AND is
    optional (a bare register amount is accepted, with the mask defaulting to the
    p-code shift's own saturating semantics).

    Declines -- leaving today's behaviour -- unless the slice is exactly that
    shape: one shift op, register source, register (optionally masked) amount, and
    nothing after it but value-preserving widening.  A rotate (`rorv`) reconverges
    two shifts into an OR and is NOT this shape, so it is correctly refused.
    """
    shifts = [op for op in slice_ops if op.opcode.name in _SHIFT_KIND]
    if len(shifts) != 1:
        return None
    sh = shifts[0]
    if sh.output is None or len(sh.inputs) != 2:
        return None

    inner_width = sh.output.size * 8
    if inner_width < 2:
        return None

    def _defining(vn: Varnode) -> PcodeOp | None:
        if vn.space.name != 'unique':
            return None
        return next(
            (
                o
                for o in slice_ops
                if o.output is not None
                and o.output.space.name == 'unique'
                and o.output.offset == vn.offset
                and o.output.size == vn.size
            ),
            None,
        )

    def _reg(vn: Varnode) -> RegMapping | None:
        d = _defining(vn)
        while d is not None and d.opcode.name == 'COPY':
            vn = d.inputs[0]
            d = _defining(vn)
        if vn.space.name != 'register':
            return None
        m = mapper.map_to_state(vn.offset, vn.size)
        return m if isinstance(m, RegMapping) else None

    src = _reg(sh.inputs[0])
    src_const = const_value(sh.inputs[0], fold_constants(slice_ops))
    if src is None and src_const is None:
        return None

    # Amount: either a bare register, or the standard `reg & mask` masking idiom.
    # The mask is resolved through constant folding rather than requiring a literal
    # `const` varnode: RISC-V emits the 64-bit shift mask as ``64 - 1``, so a
    # literal-only check saw a `unique`, declined, and left sll/srl/sra avalanching
    # at 6.0x while the word forms (literal 0x1f) were exact.
    folded = fold_constants(slice_ops)
    amt_mask = (1 << inner_width) - 1
    amt_vn = sh.inputs[1]
    d = _defining(amt_vn)
    if d is not None and d.opcode.name == 'INT_AND':
        # INT_AND is commutative, so the mask may sit on either side.
        for mask_idx, reg_idx in ((1, 0), (0, 1)):
            mv = const_value(d.inputs[mask_idx], folded)
            if mv is not None and const_value(d.inputs[reg_idx], folded) is None:
                amt_mask = mv
                amt_vn = d.inputs[reg_idx]
                break
    amt = _reg(amt_vn)
    if amt is None:
        return None
    if amt_mask == 0:
        return None  # amount forced to 0: a constant shift, not this term's case

    # A constant amount is handled exactly by the existing differential; this term
    # is for the data-dependent case only.  Source and amount aliasing the same
    # register correlates them, which this term's independence assumption forbids.
    if src is not None and amt.name == src.name and amt.bit_start == src.bit_start:
        return None

    # Only value-preserving widening may follow the shift, and every remaining op
    # must belong to the recognised shape -- otherwise the slice does more than
    # this term models.  Ops that merely compute a constant (the folded mask) are
    # not work on data and are allowed regardless of opcode.
    allowed = {id(sh)}
    if d is not None:
        allowed.add(id(d))
    for op in slice_ops:
        if id(op) in allowed or is_constant_op(op, folded):
            continue
        if op.opcode.name not in _SHIFT_PASSTHROUGH:
            return None

    # A CONSTANT source is still a variable shift: `bts rax,rbx` builds its bit mask
    # as `1 << (rbx & 0x3f)`, and the exact reachable-position set is exactly what
    # this term computes (with zero source taint).  Requiring a register source made
    # it decline, so the whole register avalanched -- 14x invented bits.
    _src_val: Expr = (
        Constant(src_const or 0, 8)
        if src is None
        else _get_taint_operand(src.name, src.bit_start, src.bit_end, False)
    )
    _src_taint: Expr = (
        Constant(0, 8)
        if src is None
        else _get_taint_operand(src.name, src.bit_start, src.bit_end, True)
    )
    expr: Expr = VariableShiftTaintExpr(
        _src_val,
        _src_taint,
        _get_taint_operand(amt.name, amt.bit_start, amt.bit_end, False),
        _get_taint_operand(amt.name, amt.bit_start, amt.bit_end, True),
        inner_width,
        _SHIFT_KIND[sh.opcode.name],
        amt_mask,
    )

    # MIPS computes a 32-bit shift and sign-extends it into a 64-bit GPR: the
    # taint of the fill is the inner msb's taint, replicated.
    if out_width > inner_width:
        widening = next(
            (o for o in slice_ops if o.opcode.name in ('INT_SEXT', 'INT_ZEXT')), None,
        )
        if widening is None:
            return None
        if widening.opcode.name == 'INT_SEXT':
            fill: Expr = BinaryExpr(Op.AND, expr, Constant(1 << (inner_width - 1), 8))
            step = 1
            while step < out_width - inner_width + 1:
                fill = BinaryExpr(Op.OR, fill, BinaryExpr(Op.LEFT, fill, Constant(step, 8)))
                step *= 2
            expr = BinaryExpr(
                Op.OR, expr, BinaryExpr(Op.AND, fill, Constant((1 << out_width) - 1, 8)),
            )

    return expr


def _build_variable_bit_select_taint(  # noqa: C901
    slice_ops: list[PcodeOp],
    mapper: StateMapper,
) -> Expr | None:
    """EXACT taint for a bit selected by a data-dependent index, or None.

    `bt rax, rbx` lifts to exactly::

        INT_AND      u0 <- [rbx, w-1]     # mask the bit offset by the operand width
        INT_RIGHT    u1 <- [rax, u0]      # shift the source down by that offset
        INT_AND      u2 <- [u1, 1]        # isolate bit 0
        INT_NOTEQUAL CF <- [u2, 0]

    Selection by a TAINTED index is non-monotone, so the 2-replica differential reads
    the source at only TWO index values and misses every other reachable index --
    the `bt rax,rbx; setc dl` under-taint.  Avalanching CF would be sound but would
    over-taint; VariableBitSelectTaintExpr is exact (see its docstring and
    benchmark/soundness/prove_variable_bit_select.py).

    We match on SEMANTICS, not on a byte pattern: a lone shift whose SOURCE is a
    register and whose AMOUNT is register-derived, feeding the terminal
    INT_NOTEQUAL through an optional bit-isolating AND.  Returns None otherwise, so
    the caller keeps the differential (which is exact for a CONSTANT index, e.g.
    `bt rax, 5`, where the shift amount is const-derived).
    """
    ne_ops = [op for op in slice_ops if op.opcode.name == 'INT_NOTEQUAL']
    shifts = [op for op in slice_ops if op.opcode.name in ('INT_LEFT', 'INT_RIGHT', 'INT_SRIGHT')]
    if len(ne_ops) != 1 or len(shifts) != 1 or ne_ops[0] is not slice_ops[-1]:
        return None
    shift = shifts[0]
    if len(shift.inputs) < 2 or shift.inputs[0].space.name != 'register':
        return None

    def _defining(vn: Varnode) -> PcodeOp | None:
        for op in slice_ops:
            if op.output is not None and op.output.space.name == 'unique' and op.output.offset == vn.offset:
                return op
        return None

    def _sole_register(vn: Varnode) -> Varnode | None:
        """The single register the index derives from, or None if it is
        constant-derived (differential already exact) or more complex."""
        if vn.space.name == 'register':
            return vn
        if vn.space.name == 'const':
            return None
        if vn.space.name != 'unique':
            return None
        d = _defining(vn)
        if d is None:
            return None
        regs = [r for r in (_sole_register(i) for i in d.inputs) if r is not None]
        return regs[0] if len(regs) == 1 else None

    idx_vn = _sole_register(shift.inputs[1])
    if idx_vn is None:
        return None  # constant-derived index: the differential is already exact

    # NOTEQUAL must consume the shift result, optionally via a bit-isolating AND.
    consumed = ne_ops[0].inputs[0]
    if consumed.space.name != 'unique':
        return None
    if shift.output is None or consumed.offset != shift.output.offset:
        d = _defining(consumed)
        if d is None or d.opcode.name != 'INT_AND':
            return None
        if not any(v.space.name == 'const' for v in d.inputs):
            return None
        if not any(
            v.space.name == 'unique' and shift.output is not None and v.offset == shift.output.offset
            for v in d.inputs
        ):
            return None

    src_map = mapper.map_to_state(shift.inputs[0].offset, shift.inputs[0].size)
    idx_map = mapper.map_to_state(idx_vn.offset, idx_vn.size)
    if src_map is None or idx_map is None:
        return None
    width = shift.inputs[0].size * 8
    if width < 2:
        return None
    return VariableBitSelectTaintExpr(
        _get_taint_operand(src_map.name, src_map.bit_start, src_map.bit_end, False),
        _get_taint_operand(src_map.name, src_map.bit_start, src_map.bit_end, True),
        _get_taint_operand(idx_map.name, idx_map.bit_start, idx_map.bit_end, False),
        _get_taint_operand(idx_map.name, idx_map.bit_start, idx_map.bit_end, True),
        width,
    )


_CMP_SIGNED = frozenset({'INT_SLESS', 'INT_SLESSEQUAL'})
_CMP_OR_EQUAL = frozenset({'INT_SLESSEQUAL', 'INT_LESSEQUAL'})
_CMP_OPS = frozenset({'INT_LESS', 'INT_LESSEQUAL', 'INT_SLESS', 'INT_SLESSEQUAL'})
_EQ_OPS = frozenset({'INT_EQUAL', 'INT_NOTEQUAL'})


def _build_packed_comparison_taint(  # noqa: C901
    slice_ops: list[PcodeOp],
    mapper: StateMapper,
) -> Expr | None:
    """EXACT taint for an OR-tree of shifted comparison / equality bits packed into a
    condition field, or None.

    PPC ``cmpw`` writes ``CR0 = LT(a<b)<<3 | GT(b<a)<<2 | EQ(a==b)<<1 | SO`` -- two
    OPPOSITE-orientation signed comparisons plus an equality packed into one field, so no
    single differential polarity is exact and the swapped-comparison floor avalanches the
    whole field.  ARM ``cset lt`` = ``ZEXT(N!=V)`` is the degenerate one-leaf case.

    Give each comparison / equality LEAF its exact closed-form term (ComparisonTaintExpr /
    EqualityTaintExpr, both Z3-proved) and compose them through the PURE-packing grammar --
    OR, left-shift-by-const, const-mask, and routing (COPY / ZEXT / SUBPIECE).  Each of
    those composition steps is EXACT on disjoint bit positions and SOUND in general
    (``taint(x|y) ⊆ Tx|Ty``, shift/mask are bijective/masking), so the result never
    under-taints; it is exact for flag packing, where the leaves read independent
    registers/flags and land in disjoint bits.  Returns None on any op outside the grammar
    (e.g. BOOL_AND/BOOL_OR in ``cset gt``/``le`` -- left to the differential+floor) or when
    the slice has no comparison/equality leaf at all.
    """
    if not slice_ops or slice_ops[-1].output is None:
        return None

    def _defining(vn: Varnode) -> PcodeOp | None:
        for op in slice_ops:
            if (
                op.output is not None
                and op.output.space.name == 'unique'
                and op.output.offset == vn.offset
            ):
                return op
        return None

    def _reg(vn: Varnode) -> RegMapping | None:
        """Resolve a varnode (following COPY) to a register mapping."""
        if vn.space.name == 'register':
            m = mapper.map_to_state(vn.offset, vn.size)
            return m if isinstance(m, RegMapping) else None
        if vn.space.name == 'unique':
            d = _defining(vn)
            if d is not None and d.opcode.name == 'COPY' and len(d.inputs) == 1:
                return _reg(d.inputs[0])
        return None

    def _cmp_leaf(op: PcodeOp) -> Expr | None:
        if len(op.inputs) != 2:
            return None
        a, b = _reg(op.inputs[0]), _reg(op.inputs[1])
        if a is None or b is None:
            return None
        width = op.inputs[0].size * 8
        if width < 1:
            return None
        return ComparisonTaintExpr(
            _get_taint_operand(a.name, a.bit_start, a.bit_end, False),
            _get_taint_operand(a.name, a.bit_start, a.bit_end, True),
            _get_taint_operand(b.name, b.bit_start, b.bit_end, False),
            _get_taint_operand(b.name, b.bit_start, b.bit_end, True),
            width,
            op.opcode.name in _CMP_SIGNED,
            op.opcode.name in _CMP_OR_EQUAL,
        )

    def _eq_leaf(op: PcodeOp) -> Expr | None:
        if len(op.inputs) != 2:
            return None
        a, b = _reg(op.inputs[0]), _reg(op.inputs[1])
        if a is None or b is None:
            return None
        width = op.inputs[0].size * 8
        if width < 1:
            return None
        return EqualityTaintExpr(
            _get_taint_operand(a.name, a.bit_start, a.bit_end, False),
            _get_taint_operand(a.name, a.bit_start, a.bit_end, True),
            _get_taint_operand(b.name, b.bit_start, b.bit_end, False),
            _get_taint_operand(b.name, b.bit_start, b.bit_end, True),
            width,
        )

    leaf_count = [0]

    def _taint(vn: Varnode) -> Expr | None:
        if vn.space.name == 'const':
            return Constant(0, 64)
        if vn.space.name == 'register':
            m = mapper.map_to_state(vn.offset, vn.size)
            if not isinstance(m, RegMapping):
                return None
            return _get_taint_operand(m.name, m.bit_start, m.bit_end, True)
        if vn.space.name != 'unique':
            return None
        d = _defining(vn)
        return _taint_op(d) if d is not None else None

    def _taint_op(op: PcodeOp) -> Expr | None:  # noqa: C901
        name = op.opcode.name
        if name in _EQ_OPS:
            leaf = _eq_leaf(op)
            if leaf is not None:
                leaf_count[0] += 1
            return leaf
        if name in _CMP_OPS:
            leaf = _cmp_leaf(op)
            if leaf is not None:
                leaf_count[0] += 1
            return leaf
        if name == 'INT_OR':
            acc: Expr | None = None
            for inp in op.inputs:
                t = _taint(inp)
                if t is None:
                    return None
                acc = t if acc is None else BinaryExpr(Op.OR, acc, t)
            return acc
        if name == 'INT_LEFT':
            if len(op.inputs) != 2 or op.inputs[1].space.name != 'const':
                return None
            inner = _taint(op.inputs[0])
            if inner is None:
                return None
            return BinaryExpr(Op.LEFT, inner, Constant(op.inputs[1].offset, 8))
        if name == 'INT_AND':
            consts = [i for i in op.inputs if i.space.name == 'const']
            others = [i for i in op.inputs if i.space.name != 'const']
            if len(consts) != 1 or len(others) != 1:
                return None  # AND of two dynamic taints is not a routing mask
            inner = _taint(others[0])
            if inner is None:
                return None
            return BinaryExpr(Op.AND, inner, Constant(consts[0].offset, 64))
        if name in ('COPY', 'INT_ZEXT', 'SUBPIECE') and len(op.inputs) >= 1:
            # routing that preserves bit-0 alignment (SUBPIECE only when it takes the
            # low bytes -- offset operand 0).
            if name == 'SUBPIECE' and (len(op.inputs) < 2 or op.inputs[1].offset != 0):
                return None
            return _taint(op.inputs[0])
        return None

    result = _taint_op(slice_ops[-1])
    if result is None or leaf_count[0] == 0:
        return None
    return result


def _select_branch_expr(  # noqa: C901
    op: PcodeOp,
    ops: list[PcodeOp],
    mapper: StateMapper,
    width: int,
    opnd: Callable[[str, int, int], Expr],
) -> Expr | None:
    """Reconstruct the value expression produced by `op` (a select branch), building each
    input register operand via `opnd(name, bit_start, bit_end)`.  Passing the VALUE, the
    high corner (V|T) or the low corner (V&~T) as `opnd` yields, respectively, the branch's
    concrete value, or the two replicas whose XOR is the branch's own differential taint.
    Grammar: COPY / INT_2COMP / INT_NEGATE / affine (ADD/SUB/AND/OR/XOR/LEFT) of
    registers/consts; None outside it.  Masked to `width` so two's-complement round-trips.
    """
    mask = (1 << width) - 1

    def v(vn: Varnode) -> Expr | None:
        if vn.space.name == 'const':
            return Constant(vn.offset & mask, width)
        if vn.space.name == 'register':
            m = mapper.map_to_state(vn.offset, vn.size)
            if not isinstance(m, RegMapping):
                return None
            operand: Expr = opnd(m.name, m.bit_start, m.bit_end)
            return operand
        if vn.space.name == 'unique':
            d = None
            for o in ops:
                if o.output is not None and o.output.space.name == 'unique' and o.output.offset == vn.offset:
                    d = o
            return _select_branch_expr(d, ops, mapper, width, opnd) if d is not None else None
        return None

    name = op.opcode.name
    ins = op.inputs
    if name == 'COPY' and ins:
        return v(ins[0])
    if name == 'INT_2COMP' and ins:
        a = v(ins[0])
        if a is None:
            return None
        return BinaryExpr(Op.AND, BinaryExpr(Op.SUB, Constant(0, width), a), Constant(mask, width))
    if name == 'INT_NEGATE' and ins:
        a = v(ins[0])
        return BinaryExpr(Op.AND, UnaryExpr(Op.NOT, a), Constant(mask, width)) if a is not None else None
    _bin = {'INT_ADD': Op.ADD, 'INT_SUB': Op.SUB, 'INT_AND': Op.AND, 'INT_OR': Op.OR,
            'INT_XOR': Op.XOR, 'INT_LEFT': Op.LEFT}
    if name in _bin and len(ins) == 2:
        a, b = v(ins[0]), v(ins[1])
        if a is None or b is None:
            return None
        return BinaryExpr(Op.AND, BinaryExpr(_bin[name], a, b), Constant(mask, width))
    return None


def _select_weld(  # noqa: C901
    slice_ops: list[PcodeOp],
    ops: list[PcodeOp],
    cbranch_op: PcodeOp,
    mapper: StateMapper,
    width: int,
) -> Expr | None:
    """Exact taint of a CBRANCH conditional select's INTERIOR -- the WELD of the two
    branch differentials plus their value difference:

        weld = D(A) | D(B) | (val(A) ^ val(B))

    D(A) is branch A's own differential (its two polarity-corner replicas XOR'd), so a
    branch that is `x2+1` or `-x2` gets its true add/negate taint, not a coarse value
    XOR.  This is welded (OR'd) because the reachable output over a varying selector is
    the union of the two branch outcomes.  The caller gates it by the selector taint.

    Handles both spellings of a conditional select:

      * TWO writes, one on each side of the CBRANCH -- the ARM conditional family
        (csel/cneg/cinv/csinc/csneg/csinv); and
      * ONE write, guarded by the CBRANCH -- x86's `cmovcc`, where the not-taken
        path simply falls through and leaves the destination alone.

    The second case has an IMPLICIT else-branch equal to the destination's prior
    value, so it is synthesised rather than declined.  That matters because the
    caller's fallback for it is the old-dest passthrough, a coarse union that ORs
    everything in whenever a condition flag is tainted, whereas the weld's
    `val(A) ^ val(B)` term is exact -- and it is the term that matters: with both
    branch values fully UNTAINTED the output still varies if they DIFFER, because a
    tainted selector chooses between them.

    Returns None for shapes outside the branch grammar.
    """
    if not slice_ops:
        return None
    out_op = slice_ops[-1]
    target = out_op.inputs[0] if out_op.opcode.name == 'COPY' and out_op.inputs else out_op.output
    if target is None:
        return None
    try:
        cbr = ops.index(cbranch_op)
    except ValueError:
        return None
    writes = [i for i, o in enumerate(ops)
              if o.output is not None and o.output.space.name == target.space.name
              and o.output.offset == target.offset]
    before = [i for i in writes if i < cbr]
    after = [i for i in writes if i > cbr]
    if not before and not after:
        return None

    def _val(n: str, bs: int, be: int) -> Expr:
        return _get_taint_operand(n, bs, be, False)

    def _hi(n: str, bs: int, be: int) -> Expr:
        return BinaryExpr(Op.OR, _get_taint_operand(n, bs, be, False), _get_taint_operand(n, bs, be, True))

    def _lo(n: str, bs: int, be: int) -> Expr:
        return BinaryExpr(Op.AND, _get_taint_operand(n, bs, be, False),
                          UnaryExpr(Op.NOT, _get_taint_operand(n, bs, be, True)))

    def branch(op: PcodeOp) -> tuple[Expr, Expr] | None:
        val = _select_branch_expr(op, ops, mapper, width, _val)
        hi = _select_branch_expr(op, ops, mapper, width, _hi)
        lo = _select_branch_expr(op, ops, mapper, width, _lo)
        if val is None or hi is None or lo is None:
            return None
        return val, BinaryExpr(Op.XOR, hi, lo)  # (value, own differential)

    def implicit_dest() -> tuple[Expr, Expr] | None:
        """The branch that performs no write: the destination's prior value."""
        dest = out_op.output
        if dest is None or dest.space.name != 'register':
            return None
        m = mapper.map_to_state(dest.offset, dest.size)
        if not isinstance(m, RegMapping):
            return None
        v = _val(m.name, m.bit_start, m.bit_end)
        h = _hi(m.name, m.bit_start, m.bit_end)
        lo_e = _lo(m.name, m.bit_start, m.bit_end)
        return v, BinaryExpr(Op.XOR, h, lo_e)

    ra = branch(ops[before[-1]]) if before else implicit_dest()
    rb = branch(ops[after[0]]) if after else implicit_dest()
    if ra is None or rb is None:
        return None
    (val_a, d_a), (val_b, d_b) = ra, rb
    val_diff = BinaryExpr(Op.XOR, val_a, val_b)
    return BinaryExpr(Op.OR, BinaryExpr(Op.OR, d_a, d_b), val_diff)


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
    has_cbranch: bool = False,
    cbranch_flag_deps: list[tuple[int, int]] | None = None,
    cbranch_cond_internal: bool = False,
    output_cond_written: bool = False,
    is_bit_count: bool = False,
    is_software_loop: bool = False,
    all_ops: list[PcodeOp] | None = None,
    cbranch_op: PcodeOp | None = None,
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

        # WIDE STORE SPLIT.  A memory target wider than 8 bytes cannot be held
        # in a single 64-bit taint mask: the shadow layer's write_mask takes a
        # uint64, so a single MEM_<addr>_16 key keeps only the low 8 bytes and
        # silently drops the taint of the high bytes.  That is the SSE memcpy
        # under-taint (`movups [rdi], xmm0` copies 16 bytes but taints 8).
        # Split the store into 8-byte chunks, mirroring the XMM_LO/HI register
        # split, so every chunk carries a <=64-bit mask the shadow layer stores
        # exactly.  The value taint is the OR of the value-deps (the same coarse
        # but sound model the wide LOAD side already uses), applied to each
        # chunk, so fully-tainted SIMD data survives the copy intact.
        if isinstance(out_target, MemoryOperand) and out_target.size > 8:
            base_addr = out_target.address_expr
            for k in range(0, out_target.size, 8):
                csize = min(8, out_target.size - k)
                chunk_addr = base_addr if k == 0 else BinaryExpr(Op.ADD, base_addr, Constant(k, 8))
                assignments.append(
                    TaintAssignment(
                        target=MemoryOperand(chunk_addr, csize, is_taint=True),
                        dependencies=value_dependencies,
                        expression=expr,
                    ),
                )
            return

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
        # Subtracted operands must be polarised oppositely (D^{+-}); see the
        # register-target make_differential path for the detailed rationale.
        # Registers are keyed per SLICE (name, bit_start, bit_end); memory keys stay
        # strings.  See MemoryDifferentialExpr: polarity belongs to the dependency
        # slice, not the register name.
        neg_inputs: list[object] = []

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
                if dep_set.value_deps[dep_map] <= 0:
                    neg_inputs.append(
                        f'MEM_{dep_map.addr_reg.name}_{dep_map.addr_const_offset}_{dep_map.size_bytes}',
                    )
            else:
                reg_inputs.append((dep_map.name, dep_map.bit_start, dep_map.bit_end))
                if dep_set.value_deps[dep_map] <= 0:
                    # Per-SLICE key: apply_sless_msb_split gives one register two
                    # slices with opposite polarity (see MemoryDifferentialExpr).
                    neg_inputs.append((dep_map.name, dep_map.bit_start, dep_map.bit_end))

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
            neg_inputs=neg_inputs,
        )

        value_dependencies, _, _, _ = process_dependencies(dep_set.value_deps)
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

    # CMOV / forward-CBRANCH old-destination injection.
    #
    # When an instruction has a forward CBRANCH (typical of cmov/setcc-by-flag
    # patterns), the destination register is implicitly READ on the not-taken
    # path (the register keeps its old value).  But the static slice from the
    # destination's *write* doesn't include any read of the destination, so
    # process_dependencies omits it from cell_inputs.
    #
    # Consequence: when both replicas take the not-taken path (concrete flag
    # FALSE), they each see the same concrete V_dest as the implicit pre-state
    # → C1=C2 → diff=0 → old-dest taint silently dropped.
    #
    # Fix: inject the destination register as a polarised cell_input.  Then:
    #   - condition concretely TRUE  → both reps overwrite dest → diff = T_source
    #   - condition concretely FALSE → both reps preserve dest = V|T or V&~T → diff = T_old_dest
    #   - condition flag tainted     → mixed; the gated passthrough below still applies
    #
    # Excluded: PC/IP/memory outputs (no implicit pre-state read).
    if (
        has_cbranch
        and not isinstance(mapping, MemMapping)
        and out_name not in ('EIP', 'RIP', 'PC')
        and out_name not in cell_inputs_rep1
    ):
        # Only register-typed outputs reach this branch (memory was excluded above).
        # Add a full-width polarised input for the destination register.
        # Use bit_start=0, bit_end=out_bit_end (the full slice of the dest).
        dest_slice = [(out_bit_start, out_bit_end, 0)]
        cell_inputs_rep1[out_name] = build_polarized_reg(out_name, dest_slice, 1)
        cell_inputs_rep2[out_name] = build_polarized_reg(out_name, dest_slice, 2)
        # Add the old-dest taint as a tracked dependency so output_taint is
        # written when only old-dest taint contributes (e.g. cmov not-taken
        # with untainted source — without this dep, the assignment would be
        # eligible for elision).
        old_dest_dep = _get_taint_operand(out_name, out_bit_start, out_bit_end, True)
        if old_dest_dep not in dependencies:
            dependencies.append(old_dest_dep)
            dependency_names.append(out_name)

    if not dependencies:
        expr = _get_zero_constant(out_bit_end - out_bit_start + 1)
        if out_name in ('EIP', 'RIP', 'PC'):
            expr = AvalancheExpr(expr, out_bit_end - out_bit_start + 1)
        assignments.append(TaintAssignment(target=out_target, dependencies=[], expression=expr))
        return

    cat = determine_category(slice_ops, out_width_bits=(out_bit_end - out_bit_start + 1))

    # Software-loop override: PEXT/PDEP and similar BMI2 ops lift to a
    # CBRANCH-driven loop whose body, when linearised by the slicer, gives
    # a wrong straight-line result.  The loop's iteration count and output
    # bit positions both depend on the input bits — full avalanche is the
    # only safe answer.  See engine.py: is_software_loop computation above.
    if is_software_loop:
        cat = InstructionCategory.AVALANCHE

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
            _reg_inputs: list[tuple[str, int, int]] = []
            _mem_inputs: list[tuple[str, int, int]] = []
            # Inputs whose value-dep polarity is negative (subtracted operand)
            # must be polarised oppositely in the differential so it captures the
            # sound D^{+-} borrow chain rather than a lossy D^{++}.
            # Polarity is a property of the dependency SLICE, not the register
            # name: apply_sless_msb_split splits ONE register into two slices with
            # OPPOSITE polarity (sign bit -1, magnitude +1).  Key the negative set
            # per (name, bit_start, bit_end) so MemoryDifferentialExpr can polarise
            # each slice independently, exactly as build_polarized_reg does on the
            # pure-register path.
            _neg_inputs: list[object] = []
            for d in dep_set.value_deps.keys():
                if isinstance(d, MemMapping):
                    _mem_inputs.append((d.addr_reg.name, d.addr_const_offset, d.size_bytes))
                    if dep_set.value_deps[d] <= 0:
                        _neg_inputs.append(f'MEM_{d.addr_reg.name}_{d.addr_const_offset}_{d.size_bytes}')
                else:
                    _reg_inputs.append((d.name, d.bit_start, d.bit_end))
                    if dep_set.value_deps[d] <= 0:
                        _neg_inputs.append((d.name, d.bit_start, d.bit_end))
            return MemoryDifferentialExpr(
                bytestring=bytestring,
                target=('REG', out_name, out_bit_start, out_bit_end),
                reg_inputs=_reg_inputs,
                mem_inputs=_mem_inputs,
                addr_only_regs=sorted(_addr_only_regs_set),
                neg_inputs=_neg_inputs,
            )
        # Pure-register fast path: cell.pyx's static-cell evaluation.
        C1_cell = InstructionCellExpr(arch, bytestring.hex(), out_name, out_bit_start, out_bit_end, cell_inputs_rep1)
        C2_cell = InstructionCellExpr(arch, bytestring.hex(), out_name, out_bit_start, out_bit_end, cell_inputs_rep2)
        return BinaryExpr(Op.XOR, C1_cell, C2_cell)

    def make_mapped_single_call() -> Expr | None:
        """Single-call (linear-part) rule for a MAPPED routing slice.

        A routing slice with a single dynamic input is GF(2)-affine,
        ``f(x, c) = L(x) XOR a(c)``, so the 2-replica differential
        ``f(V|T) XOR f(V&~T)`` collapses to the value-independent
        ``L(T) = f(T) XOR f|_{x->0}``: one instruction execution on the taint mask
        plus the constant part ``a(c) = f|_{x->0}``, folded once at synthesis (vs.
        two executions for the differential). ``a(c)`` is nonzero for bit-injecting
        routing such as OR-with-a-set-constant, so the fold is required, not an
        optimisation. The taint mask placed at the input's register positions is
        exactly ``(V|T) XOR (V&~T)``, reusing the differential's own replica inputs.

        Returns ``None`` -- caller keeps the differential -- unless every
        precondition holds. See ``tests/test_mapped_single_call.py`` for the
        corpus-wide proof that this equals the differential where it fires.

        * pure-register slice (memory / address-only / load-like keep the
          differential -- their dynamic address is a second runtime input);
        * exactly one dynamic register input -- a single routed operand is affine,
          whereas two dynamic inputs into a routing op (e.g. ``a & b``) are not;
        * every slice op is a routing (bit-permuting) op;
        * no control-flow op anywhere in the instruction -- a branch/call taints
          the concrete run with a backend-specific landing address that the
          two-run differential cancels by XOR but a single run cannot (e.g.
          RISC-V ``jalr``'s link-register write is otherwise pure routing).
        """
        if _use_mem_diff or is_load_like:
            return None
        reg_groups: set[str] = set()
        for d in dep_set.value_deps.keys():
            if isinstance(d, MemMapping):
                return None
            reg_groups.add(d.name)
        if len(reg_groups) != 1:
            return None
        if any(op.opcode.name not in ROUTING_OPCODES for op in slice_ops):
            return None
        if any(op.opcode.name in CONTROL_FLOW_OPCODES for op in (all_ops or slice_ops)):
            return None

        name = next(iter(reg_groups))
        hexs = bytestring.hex()
        try:
            from microtaint.simulator import MachineState  # noqa: PLC0415

            # a(c) = f|_{x->0}: the affine constant part (nonzero for OR-with-a-set-
            # constant), the output slice with the dynamic input cleared. One cached
            # concrete evaluation at synthesis; never on the runtime hot path.
            probe = InstructionCellExpr(arch, hexs, out_name, out_bit_start, out_bit_end, {})
            a_c = _synth_simulator(arch).evaluate_concrete(probe, MachineState(regs={name: 0}, mem={}))
        except Exception:
            return None
        # Taint mask placed at the input's register positions: (V|T) XOR (V&~T).
        taint_input = BinaryExpr(Op.XOR, cell_inputs_rep1[name], cell_inputs_rep2[name])
        taint_cell = InstructionCellExpr(arch, hexs, out_name, out_bit_start, out_bit_end, {name: taint_input})
        return BinaryExpr(Op.XOR, taint_cell, Constant(a_c, out_bit_end - out_bit_start + 1))

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
        # Constant-dominating slice: result is always a constant, so any flag
        # computed from it (e.g. PF via POPCOUNT+INT_EQUAL) is deterministic.
        if _slice_has_constant_dominator(slice_ops):
            assignments.append(
                TaintAssignment(
                    target=out_target,
                    dependencies=[],
                    expression=_get_zero_constant(out_bit_end - out_bit_start + 1),
                ),
            )
            return

        # A multiply otherwise avalanches the full output width.  Route it to the
        # sound fill first; it declines (None) for any non-multiply slice, leaving
        # the avalanche path below unchanged.
        _var_mul = _build_variable_multiply_taint(
            slice_ops, mapper, out_bit_end - out_bit_start + 1,
        )
        if _var_mul is not None and not isinstance(mapping, MemMapping):
            assignments.append(
                TaintAssignment(
                    target=out_target,
                    dependencies=dependencies,
                    expression=_var_mul,
                ),
            )
            return

        expr = dependencies[0]
        for dep in dependencies[1:]:
            expr = BinaryExpr(Op.OR, expr, dep)
        out_width = out_bit_end - out_bit_start + 1
        if is_bit_count and out_width >= 8:
            # The result is a count bounded by the SOURCE OPERAND width, not by the
            # output sub-register view width.  For example, BSR RAX,RBX produces a
            # result in 0..63 regardless of whether we are computing taint for RAX
            # (64-bit), AX (16-bit), or AL (8-bit).  Using out_width gives a cap
            # that is too narrow for sub-register views (e.g. cap=0x0f for AL when
            # the correct bound is 0x3f for a 64-bit source operand).
            # The source width is available in dep_set: for all bit-count instructions
            # there is exactly one value dependency whose bit span equals the source
            # operand width.
            _src_width = next(
                (k.bit_end - k.bit_start + 1 for k in dep_set.value_deps if isinstance(k, RegMapping)),
                out_width,  # safe fallback: never reached for well-formed bit-count slices
            )
            count_width = max(1, _src_width.bit_length())  # ⌈log2(src_width)⌉, e.g. 7 for 64-bit
            cap_mask = (1 << count_width) - 1
            avalanche = AvalancheExpr(expr, count_width)
            expr = BinaryExpr(Op.AND, avalanche, Constant(cap_mask, 8))
        else:
            expr = AvalancheExpr(expr, out_width)

    elif cat == InstructionCategory.TRANSLATABLE:
        # A shift by a data-dependent amount otherwise falls to AvalancheExpr and
        # taints the full output width.  Route it to the exact subcube term first;
        # it declines (returning None) for anything that is not exactly a masked
        # variable shift, in which case the differential path below is unchanged.
        # A wide XOR's differential CANCELS: D = Ta ^ Tb is 0 exactly where a bit is
        # tainted in BOTH operands.  `eor x0,x1,x2,ror #11` hits this at the ROTATED
        # positions -- x2's tainted bits {12,17,19,20,50} land on {1,6,8,9,39} and
        # collide with x1's bit 1.  The exact taint of a^b is the position-wise union
        # of the operand taints, so when both resolve in closed form (which the
        # permutation grammar does exactly, rotation included) emit that instead of
        # the differential.  This is EXACT, not a floor: routing the same instruction
        # through ORABLE's union over RAW register taint was tried and cost 53 points
        # of exactness, because the union then sits at the wrong bit positions.
        _xor_term = _build_wide_xor_taint(slice_ops, mapper)
        if _xor_term is not None and not isinstance(mapping, MemMapping):
            assignments.append(
                TaintAssignment(
                    target=out_target, dependencies=dependencies, expression=_xor_term,
                ),
            )
            return

        _var_shift = _build_variable_shift_taint(
            slice_ops, mapper, out_bit_end - out_bit_start + 1,
        )
        if _var_shift is not None and not isinstance(mapping, MemMapping):
            assignments.append(
                TaintAssignment(
                    target=out_target,
                    dependencies=dependencies,
                    expression=_var_shift,
                ),
            )
            return

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

        def _amount_is_const_derived(vn: Varnode, seen: set[int] | None = None) -> bool:
            """True only if the shift amount provably derives from constants ALONE.

            We must PROVE constness, never infer it from an empty trace_origins():
            that returns nothing both for a constant amount AND when the trace
            reaches an operand the mapper cannot resolve.  SPARC's `sll %g1,%g2,%g3`
            lifts its amount as COPY(register:0xb:1) -- G2's low byte -- which
            map_to_state cannot map, so trace_origins comes back empty even though
            the amount is fully data-dependent.  Treating that as "constant" drops
            the avalanche and under-taints a variable shift.
            """
            if seen is None:
                seen = set()
            space = vn.space.name
            if space == 'const':
                return True
            if space != 'unique':
                return False  # register / memory operand: data-dependent
            if vn.offset in seen:
                return False  # cycle: cannot prove constness
            seen.add(vn.offset)
            for op in slice_ops:
                if op.output is not None and op.output.space.name == 'unique' and op.output.offset == vn.offset:
                    if op.opcode.name == 'LOAD':
                        return False
                    return all(_amount_is_const_derived(i, seen) for i in op.inputs)
            return False  # no definition found: cannot prove constness

        offset_names: set[str] = set()
        _offset_is_const_derived = False
        if shift_op and len(shift_op.inputs) > 1:
            offset_names = trace_origins(shift_op.inputs[1])
            _offset_is_const_derived = _amount_is_const_derived(shift_op.inputs[1])

        primary_input_name = None
        if shift_op and shift_op.inputs[0].space.name == 'register':
            m = mapper.map_to_state(shift_op.inputs[0].offset, shift_op.inputs[0].size)
            if m:
                primary_input_name = m.name

        # Conservative fallback for a slice whose shift amount we could not resolve.
        # It must NOT run for a constant amount: extract_dependencies collects the
        # register reads of the WHOLE instruction, so a rotate's flag-preservation
        # ops (`ror` reads the old CF/OF to keep them when count==0) leave CF/OF in
        # the data output's dep list.  Treating those as the "offset" avalanched RAX
        # to all-ones whenever the incoming flags were tainted -- breaking the
        # rol;ror round-trip identity once rol's CF became (correctly) tainted.
        if not offset_names and not _offset_is_const_derived:
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
        # A bit SELECTED by a data-dependent index (`bt rax, rbx` -> CF) is
        # non-monotone in the index: the differential reads the source at only the
        # two extremal index values and misses every other reachable one.  Derive it
        # exactly by enumerating the reachable index set instead of avalanching
        # (which would be sound but would over-taint).  Returns None -- so we fall
        # through -- for a constant-derived index, where the differential is exact.
        _bitsel_expr = _build_variable_bit_select_taint(slice_ops, mapper)
        if _bitsel_expr is not None and not _slice_has_constant_dominator(slice_ops):
            assignments.append(
                TaintAssignment(target=out_target, dependencies=dependencies, expression=_bitsel_expr),
            )
            return

        # EXACT packed comparison/equality taint: a condition field built from an OR-tree
        # of shifted comparison/equality bits (PPC `cmpw` CR0 = LT|GT|EQ|SO; ARM `cset lt`
        # = ZEXT(N!=V)).  cmpw packs two OPPOSITE-orientation signed comparisons, so no
        # single differential polarity is exact and the swapped-comparison floor avalanches
        # the whole field; equality collapses the 2-corner differential outright.  Each leaf
        # gets its Z3-proved closed form, composed through the pure-packing grammar -- exact
        # for flag packing, sound otherwise.  None (fall through to differential+floor) for
        # slices outside the grammar (e.g. `cset gt`/`le`, which BOOL_AND/OR a flag).
        _packed_expr = _build_packed_comparison_taint(slice_ops, mapper)
        if _packed_expr is not None and not _slice_has_constant_dominator(slice_ops):
            assignments.append(
                TaintAssignment(target=out_target, dependencies=dependencies, expression=_packed_expr),
            )
            return

        # Short-circuit: if the backward slice contains a constant-dominating op
        # (AND with 0, OR with -1, XOR-self), the output is always a constant
        # regardless of any tainted input.  T_flag = 0 always.
        if _slice_has_constant_dominator(slice_ops):
            assignments.append(
                TaintAssignment(
                    target=out_target,
                    dependencies=[],
                    expression=_get_zero_constant(out_bit_end - out_bit_start + 1),
                ),
            )
            return

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

        # COND_TRANSPORTABLE derives a flag from a SINGLE masked replica
        # (C_eval = the flag evaluated on V&~T).  That is value-dependent and
        # under-taints whenever masking the tainted input bits happens to zero
        # the flag: e.g. `shl rax, 4` -> CF is bit 60 of RAX, so masking bit 60
        # to 0 yields CF=0 even though CF depends on exactly that bit.
        #
        # The 2-replica differential XOR(C_eval(V|T), C_eval(V&~T)) captures such
        # (monotone) flags EXACTLY, so OR it into every 1-bit flag output.  It is
        # a sound floor: a bit is reported only when the two replicas genuinely
        # differ, so it can never over-taint.
        #
        # This subsumes the former per-shape gates
        # (_is_bit_extract_{notequal,via_tainted_shift,const_shift}), which OR-ed
        # this same differential for three hand-matched patterns only — `shl` CF
        # matched none of them and silently under-tainted.
        # The same reasoning covers setcc-style BYTE outputs: they are the 0/1
        # results of these very conditions.  `cmp rax,[mem]; setl cl` computes
        # CL = [rax <s mem] -- a SIGNED comparison, which apply_sless_msb_split
        # already polarises (the sign bit gets the opposite polarity to the
        # magnitude bits, making the comparison monotone in the biased
        # representation, so the polarised differential is exact).  Its flag is
        # INTERNAL to the monolithic block -- STORE/LOAD forces monolithic -- so no
        # flag-output rule can see it; only the differential on the byte output can.
        # Hence gate on <= 8-bit outputs rather than strictly 1-bit.
        # The 2-replica differential is a SOUND floor: it reports a bit only when the
        # two masked replicas genuinely differ, so it can never over-taint (see the
        # rationale above).  Apply it to EVERY COND_TRANSPORTABLE output, not only x86
        # setcc BYTE outputs -- a flag consumed into a WIDE register (ARM64
        # `cset`/`csel` write the 0/1 condition into a 64-bit GPR) otherwise gets only
        # the masked-single-replica term, which under-taints when masking the flag
        # inputs to 0 collapses the condition (`cset x0,lt` -> N!=V -> 0!=0 -> clean).
        # For a wide output the differential is 0 on the high bits (both replicas 0
        # there) and taints exactly the meaningful low bit, so this never over-taints
        # the upper bytes either.
        expr = BinaryExpr(Op.OR, expr, make_differential())

        # CMOV not-taken passthrough: when the condition is false the destination
        # register keeps its OLD value, so its OLD taint must also survive.
        # The old taint of the destination is simply T_<out_name>[out_bits].
        # We OR it into the expression so that:
        #   - taken path  → (source taint drives output)  OR old_dest_taint
        #   - not-taken   → 0 (C_eval=0 since no write)   OR old_dest_taint
        # This is conservative but sound: in the taken path the old dest taint
        # may be over-counted, but taint propagation must never drop bits.
        # NOTE: The same passthrough is applied generically below for ALL
        # instruction categories that have a CBRANCH in their P-code, which
        # covers MONOTONIC cmovz/cmovs in addition to COND_TRANSPORTABLE cmovl.

    elif cat == InstructionCategory.TRANSPORTABLE:
        diff_expr = make_differential()
        is_flag = out_bit_end == out_bit_start

        if dependencies and not is_flag:
            transport_term = dependencies[0]
            for dep in dependencies[1:]:
                transport_term = BinaryExpr(Op.OR, transport_term, dep)
            out_width = out_bit_end - out_bit_start + 1
            if is_bit_count and out_width >= 8:
                # tzcnt/bsf/bsr lift as software loops with INT_ADD/SUB counter.
                # The output is a bit-index or bit-count bounded by the SOURCE operand
                # width, not by the output sub-register view width.  Using out_width
                # gives an incorrect (too-narrow) cap for sub-register views: e.g.
                # T_AL gets cap=0x0f for BSR RAX,RBX, but the result spans 0..63 (6
                # bits, cap=0x3f).  Instead derive the cap from the source operand
                # width, which is available as the single dep_set entry's bit span.
                _src_width = next(
                    (k.bit_end - k.bit_start + 1 for k in dep_set.value_deps if isinstance(k, RegMapping)),
                    out_width,  # safe fallback: never reached for well-formed bit-count slices
                )
                count_width = max(1, _src_width.bit_length())
                cap_mask = (1 << count_width) - 1
                transport_term = BinaryExpr(
                    Op.AND,
                    AvalancheExpr(transport_term, count_width),
                    Constant(cap_mask, 8),
                )

            # Soundness floor for widening INT_SEXT: the sign extension replicates
            # the inner MSB into every fill bit, so the taint of that one bit must
            # reach ALL fill bits the output slice covers.  The 2-corner differential
            # can miss it entirely -- the sign of a wrapping add is non-monotone, so
            # both polarity corners can share a sign while an interior value flips it.
            #
            # This must fire whenever the slice INTERSECTS the fill region
            # [inner, sext_bits), NOT only when the whole extended value fits in the
            # slice.  A register written by a sext is enumerated as overlapping
            # sub-views -- e.g. a 64-bit V0 as both [63:0] and [63:32] -- and the
            # [63:32] view is ENTIRELY fill.  The old `sext_bits <= out_width` guard
            # skipped it, so that view carried only the raw source union (the sign
            # bit landing at one position instead of the whole fill) and, depending
            # on which view the driver emitted LAST, clobbered the correct [63:0]
            # taint.  MIPS `sw;lw;addu` emits [63:32] last and under-tainted; plain
            # `addu` emits it first and hid the bug.  Firing on intersection makes
            # every sub-view correct and the result order-independent.
            sext_op = next(
                (
                    op
                    for op in slice_ops
                    if op.opcode.name == 'INT_SEXT'
                    and op.output is not None
                    and op.inputs[0].size * 8 < op.output.size * 8
                ),
                None,
            )
            if sext_op is not None and sext_op.output is not None:
                inner = sext_op.inputs[0].size * 8
                sext_bits = sext_op.output.size * 8
                # Fill region intersected with this slice, in EXPR coordinates
                # (expr bit k -> register bit k + out_bit_start).  transport_term is
                # in source coordinates, so the sign bit stays at `inner-1`.
                fill_lo = max(inner, out_bit_start) - out_bit_start
                fill_hi = min(sext_bits - 1, out_bit_end) - out_bit_start
                if fill_hi >= fill_lo:
                    fill_w = fill_hi - fill_lo + 1
                    sign = BinaryExpr(Op.AND, transport_term, Constant(1 << (inner - 1), 8))
                    # Every fill bit equals the sign bit, so avalanche is EXACT here,
                    # not an over-approximation.
                    fill = BinaryExpr(Op.LEFT, AvalancheExpr(sign, fill_w), Constant(fill_lo, 8))
                    transport_term = BinaryExpr(Op.OR, transport_term, fill)

            expr = BinaryExpr(Op.OR, diff_expr, transport_term)
        else:
            expr = diff_expr

    elif cat == InstructionCategory.MAPPED:
        mapped_single = make_mapped_single_call()
        expr = mapped_single if mapped_single is not None else make_differential()

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
        # Signed overflow (INT_SBORROW / INT_SCARRY) is NON-monotone despite living
        # in this category: the differential's two extremal corners can agree while
        # an interior tainted-bit flip toggles OF, and the floors below assume the
        # differential is exact for a single partially-tainted operand -- which is
        # false here.  Route it to the EXACT sign decomposition instead of a floor
        # (proved, and does not over-taint).  Returns None for anything that is not
        # a plain two-operand overflow flag, in which case we fall through.
        _signed_ovf_expr = _build_signed_overflow_taint(slice_ops, mapper)
        if _signed_ovf_expr is not None and not _slice_has_constant_dominator(slice_ops):
            assignments.append(
                TaintAssignment(
                    target=out_target,
                    dependencies=dependencies,
                    expression=_signed_ovf_expr,
                ),
            )
            return

        diff_expr = make_differential()

        # 1-bit flag soundness floor for MONOTONIC.
        #
        # See detailed comment in generate_taint_assignments docstring.
        # Short summary: differential gives 0 for fully-tainted inputs on
        # symmetric comparison ops; FullMaskAvalancheExpr provides the floor.
        #
        # Exception: constant-dominating ops (AND with 0, OR with -1, XOR-self)
        # produce a deterministic result regardless of input, so their flag
        # assignments must NOT get the floor — the differential's 0 is correct.
        is_flag = out_bit_end == out_bit_start
        # Also fire for a WIDE output when every dep is a 1-bit flag: a condition
        # consumed into a large register (ARM64 `cset x0,hi` = C&&!Z -> 64-bit GPR,
        # lifted to BOOL_AND -> MONOTONIC) is still 0/1 in bit 0.  The 2-corner
        # differential misses the non-monotone BOOL_AND when BOTH flags are tainted
        # (corners (1,1)->0 and (0,0)->0; interior (1,0)->1 is missed).  The floor
        # terms below are already bit-0 (Avalanche size 1 / FullMaskAvalanche -> 0/1),
        # so a wide output is tainted only in bit 0 -- no upper-byte over-taint.  This
        # mirrors the COND_TRANSPORTABLE wide-output floor (a9b88bf); MONOTONIC was
        # missed there.
        _mono_all_deps_one_bit = bool(dep_set.value_deps) and all(
            isinstance(dm, RegMapping) and dm.bit_end == dm.bit_start
            for dm in dep_set.value_deps.keys()
        )
        # A WIDE output that is INT_ZEXT of a 1-bit value is a BOOLEAN result (0/1
        # in bit 0): a comparison consumed into a register (MIPS `slt`/`sltu` =
        # zext(a0 < a1), lifted to INT_SLESS/INT_LESS + INT_ZEXT -> MONOTONIC).  Its
        # differential is confined to bit 0, and the symmetric-comparison floor
        # below is bit-0, so firing it taints only bit 0 -- sound, no over-taint of
        # the always-zero upper bits.  This is the wide-operand analog of the
        # 1-bit-flag-dep case above (`cset`), which is also zext-of-1-bit.
        _mono_result_is_boolean = out_bit_end > out_bit_start and any(
            op.opcode.name == 'INT_ZEXT' and op.inputs and op.inputs[0].size == 1
            for op in slice_ops if op.output is not None
        )
        # After correct comparison polarity (compute_polarity now inverts a comparison's
        # LHS), the 2-corner differential is EXACT for every ORIENTABLE MONOTONIC op:
        # add/sub/carry/borrow chains, single-direction comparisons (SLESS/LESS), and
        # mixed-polarity BOOL_AND/BOOL_OR of flags (`cset hi`/`ls`).  The one residual
        # non-monotone op with NO monotone orientation is an EQUALITY (INT_EQUAL/
        # INT_NOTEQUAL -- e.g. `cset lt` = NG!=OV).  So a WIDE boolean output needs the
        # floor ONLY when its slice contains an equality op; firing it for the orientable
        # cases merely re-introduced avalanche over-taint (measured: `cset hi` 20% exact,
        # `slt` 71%).  A 1-bit flag output keeps the floor unconditionally (bit-0 term;
        # x86 setcc is exact via the differential and unaffected).  Signed overflow is
        # already routed to its exact closed form above.  See
        # docs/design/nonmonotone-taint-theory.md.
        _slice_has_equality = any(op.opcode.name in ('INT_EQUAL', 'INT_NOTEQUAL') for op in slice_ops)
        _wide_needs_floor = (_mono_all_deps_one_bit or _mono_result_is_boolean) and _slice_has_equality
        if (is_flag or _wide_needs_floor) and dependencies:
            _is_constant_result = _slice_has_constant_dominator(slice_ops)
            if not _is_constant_result:
                # Symmetric two-operand comparison opcodes can produce
                # coincidental cancellation in the differential when BOTH
                # operands have OVERLAPPING (not just full) taint masks.
                # Example: sub rax, rbx with T_RAX = T_RBX = 0xFFFF0000FFFF0000.
                # The high replica V|T evaluates the comparison with the same
                # bits set in both operands; the low replica V&~T has those
                # bits cleared in both operands.  The two comparisons can
                # coincidentally agree (e.g. both report CF=0) even though
                # individual per-bit flips of one operand alone would change
                # the result.  The FullMaskAvalancheExpr floor only fires
                # when T_j is the FULL mask, so it doesn't cover this case.
                #
                # Fix: when the slice is dominated by one of these symmetric
                # comparison opcodes, use AvalancheExpr (fires on ANY nonzero
                # taint) instead of FullMaskAvalancheExpr.  This is sound
                # (we never miss real taint) at the cost of over-tainting
                # 1-bit flags when the differential alone would have been
                # exact.  For 1-bit outputs this is a minor precision cost.
                _SYMMETRIC_COMPARISON_OPS = {
                    'INT_LESS',
                    'INT_LESSEQUAL',
                    'INT_SLESS',
                    'INT_SLESSEQUAL',
                    'INT_EQUAL',
                    'INT_NOTEQUAL',
                    'INT_CARRY',
                    'INT_SCARRY',
                    'INT_SBORROW',
                }
                _slice_has_symmetric_cmp = any(op.opcode.name in _SYMMETRIC_COMPARISON_OPS for op in slice_ops)
                if _slice_has_symmetric_cmp:
                    # Refined sound floor for symmetric two-operand comparisons.
                    #
                    # The floor must fire in two regimes where the differential
                    # alone can miss real taint:
                    #
                    #   (A) TWO OR MORE deps tainted simultaneously.  The
                    #       high/low replicas saturate every tainted dep
                    #       symmetrically; comparisons can coincidentally agree
                    #       even though per-bit flips of one dep alone would
                    #       change the result.  Detected by the disjunction of
                    #       PAIRWISE conjunctions: AvalancheExpr(d_i) AND
                    #       AvalancheExpr(d_j) for every pair (i,j) — fires iff
                    #       any 2 of the deps have at least one tainted bit.
                    #
                    #       (Earlier versions used a single AND over ALL deps,
                    #       which silently dropped cases where one dep was
                    #       clean: e.g. ``sbb rax,rbx`` after ``sbb rax,rbx``
                    #       has T_CF=0 but T_RAX, T_RBX both partially tainted,
                    #       and INT_LESS can still cancel between high/low
                    #       replicas — observed as the SBB-cascade
                    #       under-tainting in test_known_failing_sbb_chain.)
                    #
                    #   (B) ONE dep with a FULL-MASK taint.  Example: neg rax
                    #       with T_RAX=MASK64.  OF = (RAX == MIN_INT).  The
                    #       differential evaluates INT_EQUAL(MASK, MIN_INT)
                    #       XOR INT_EQUAL(0, MIN_INT) = 0 XOR 0 = 0, missing
                    #       that flipping bit 63 alone changes the equality.
                    #       Detected by FullMaskAvalancheExpr per dep.
                    #
                    # The differential remains exact when EXACTLY ONE dep has
                    # PARTIAL taint (single tainted bit or partial mask short
                    # of the full dep width): in that regime both regimes (A)
                    # and (B) evaluate to 0 and the differential's precision
                    # is preserved.  This is the regime exercised by the
                    # bit-precision tests (`test_flag_carry_cf` etc.).
                    aval_terms: list[Expr] = []
                    fma_terms: list[Expr] = []
                    for dep_map in dep_set.value_deps.keys():
                        if isinstance(dep_map, RegMapping):
                            dep_bits = dep_map.bit_end - dep_map.bit_start + 1
                            dep_expr = _get_taint_operand(dep_map.name, dep_map.bit_start, dep_map.bit_end, True)
                            aval_terms.append(AvalancheExpr(dep_expr, 1))
                            fma_terms.append(FullMaskAvalancheExpr(dep_expr, dep_bits))

                    floor_components: list[Expr] = []
                    # Regime (A): pairwise conjunction over all dep pairs.
                    # Fires iff ANY 2 deps simultaneously have at least one
                    # tainted bit.  This is the right predicate for symmetric
                    # comparison cancellation: the cancellation requires two
                    # operands whose taint can flip independently of each
                    # other.  Using a single AND over ALL deps (the previous
                    # implementation) misfires when one dep is clean — the
                    # other two can still cancel and we'd miss real taint.
                    for i in range(len(aval_terms)):
                        for j in range(i + 1, len(aval_terms)):
                            pair_term: Expr = BinaryExpr(Op.AND, aval_terms[i], aval_terms[j])
                            floor_components.append(pair_term)
                    # Regime (B): disjunction of FMA per dep.
                    floor_components.extend(fma_terms)

                    if floor_components:
                        floor_expr: Expr = floor_components[0]
                        for fc in floor_components[1:]:
                            floor_expr = BinaryExpr(Op.OR, floor_expr, fc)
                        expr = BinaryExpr(Op.OR, diff_expr, floor_expr)
                    else:
                        expr = diff_expr
                else:
                    floor_terms: list[Expr] = []
                    for dep_map in dep_set.value_deps.keys():
                        if isinstance(dep_map, RegMapping):
                            dep_bits = dep_map.bit_end - dep_map.bit_start + 1
                            dep_expr = _get_taint_operand(dep_map.name, dep_map.bit_start, dep_map.bit_end, True)
                            floor_terms.append(FullMaskAvalancheExpr(dep_expr, dep_bits))
                    if floor_terms:
                        floor_expr_2: Expr = floor_terms[0]
                        for ft in floor_terms[1:]:
                            floor_expr_2 = BinaryExpr(Op.OR, floor_expr_2, ft)
                        expr = BinaryExpr(Op.OR, diff_expr, floor_expr_2)
                    else:
                        expr = diff_expr
            else:
                expr = diff_expr
        else:
            expr = diff_expr

    else:
        raise ValueError(f'Unsupported instruction category: {cat}')

    # -----------------------------------------------------------------------
    # FUSED-OPERATION WAIST FLOOR  (see microtaint/sleigh/partition.py)
    #
    # Some ISAs encode two functional units in one instruction -- ARM64's
    # shifted-register operand (`sub x0, x1, x2, asr #5`) runs the barrel
    # shifter into the ALU.  `slice_backward` flattens both into one op list and
    # `determine_category` must pick a single label; the permutation prefix wins
    # (TRANSLATABLE) and the carry-coupled core silently loses the union floor
    # that makes a bare `sub` sound.  The borrow chain is then covered only by
    # the 2-corner differential, which under-taints.
    #
    # `find_waist` detects the fusion structurally (single conduit varnode,
    # disjoint architectural inputs, both sides doing real work, different taint
    # algebras), so the rule holds for any ISA rather than naming opcodes.  The
    # floor is taken over the *materialised waist taint* -- the source taint as
    # transformed by the upstream unit -- not the raw source taint, because the
    # shift has already moved those bits and, for a left shift, discarded some.
    #
    # Restricted to wide outputs: a 1-bit flag fed by a union floor would be
    # tainted by any tainted input bit, which is sound but needlessly coarse.
    # -----------------------------------------------------------------------
    if (
        not is_store_target
        and not isinstance(mapping, MemMapping)
        and out_bit_end > out_bit_start
        and slice_ops
        and slice_ops[-1].output is not None
    ):
        # A floor needs only the disjoint-input conduit, not two distinct algebras:
        # `bic x0,x1,x2,lsl #1` is bitwise on both sides but still owes a floor at the
        # SHIFTED positions, which the raw source union cannot express.
        _waist = find_waist(slice_ops, slice_ops[-1].output, require_distinct_algebra=True)
        if (
            _waist is not None
            and _waist.upstream_algebra == ALG_BITWISE
            and _waist.downstream_algebra == ALG_ARITH
        ):

            def _reg_taint(offset: int, size: int) -> Expr | None:
                m = mapper.map_to_state(offset, size)
                if m is None:
                    return None
                return _get_taint_operand(m.name, m.bit_start, m.bit_end, True)

            _waist_taint = waist_taint_expr(_waist, _reg_taint)
            if _waist_taint is not None:
                _floor_terms: list[Expr] = [_waist_taint]
                for _off, _sz in sorted(_waist.downstream_regs):
                    _dt = _reg_taint(_off, _sz)
                    if _dt is None:  # an unmappable read -> cannot bound the floor
                        _floor_terms = []
                        break
                    _floor_terms.append(_dt)
                if _floor_terms:
                    _fl = _floor_terms[0]
                    for _t in _floor_terms[1:]:
                        _fl = BinaryExpr(Op.OR, _fl, _t)
                    _w = out_bit_end - out_bit_start + 1
                    if _w < 64:
                        _fl = BinaryExpr(Op.AND, _fl, Constant((1 << _w) - 1, 8))
                    expr = BinaryExpr(Op.OR, expr, _fl)

    # -----------------------------------------------------------------------
    # VARIABLE-SHIFT FLAG GATE
    #
    # x86 computes a shift's CF/OF by selecting a bit whose POSITION depends on
    # the shift amount:
    #
    #   shl rax,cl:  u = RAX_old << (amt-1) ; CF = u s< 0     -- bit (w-amt)
    #   shr rax,cl:  u = RAX_old >> (amt-1) ; CF = u & 1      -- bit (amt-1)
    #
    # Selection by a tainted index is NON-MONOTONE, so the 2-corner differential
    # reads the source at only two index values and misses the rest.
    # VariableBitSelectTaintExpr is the exact term for this, but it cannot be used
    # here: it requires the reachable index set to be a SUBCUBE of the index
    # operand's taint cube, and `w - amt` / `amt - 1` are not subcubes of `amt`'s.
    #
    # So the flag gets a gate instead: if the shift AMOUNT carries any taint, the
    # selected position can move and the flag is marked tainted.  On a 1-bit output
    # that costs at most one bit, and it fires only when the amount is genuinely
    # tainted -- a concrete `cl` leaves the differential untouched and exact.
    # -----------------------------------------------------------------------
    if (
        not is_store_target
        and not isinstance(mapping, MemMapping)
        and out_bit_end == out_bit_start
        and not _slice_has_constant_dominator(slice_ops)
    ):
        _folded_amt = fold_constants(slice_ops)

        def _amt_regs(vn: Varnode, depth: int = 0) -> list[RegMapping]:
            """Architectural registers the shift amount derives from."""
            if depth > 12 or vn.space.name == 'const':
                return []
            if _key_of(vn) in _folded_amt:
                return []  # a computed constant: the position cannot move
            if vn.space.name == 'register':
                m = mapper.map_to_state(vn.offset, vn.size)
                return [m] if isinstance(m, RegMapping) else []
            out: list[RegMapping] = []
            for o in slice_ops:
                if o.output is not None and _overlaps_vn(o.output, vn):
                    for i in o.inputs:
                        out.extend(_amt_regs(i, depth + 1))
            return out

        _amt_taints: list[Expr] = []
        for _op in slice_ops:
            if _op.opcode.name in ('INT_LEFT', 'INT_RIGHT', 'INT_SRIGHT') and len(_op.inputs) > 1:
                for _m in _amt_regs(_op.inputs[1]):
                    _amt_taints.append(
                        _get_taint_operand(_m.name, _m.bit_start, _m.bit_end, True),
                    )
        if _amt_taints:
            _acc = _amt_taints[0]
            for _t in _amt_taints[1:]:
                _acc = BinaryExpr(Op.OR, _acc, _t)
            expr = BinaryExpr(Op.OR, expr, BinaryExpr(Op.AND, AvalancheExpr(_acc, 1), Constant(1, 8)))

    # -----------------------------------------------------------------------
    # XOR CANCELLATION ON A 1-BIT FLAG
    #
    # `rcr rax,1` computes OF = MSB(RAX_old) ^ CF_old.  When BOTH of those are
    # tainted the 2-corner differential cancels -- D = Ta ^ Tb is 0 exactly where a
    # bit is tainted in both operands -- so OF read as clean.  Verified by probe:
    # tainting either input alone gives the right answer, only the pair fails.
    #
    # For a ONE-BIT output the union is not merely sound but EXACT (a ^ b varies iff
    # either input can vary), and it costs at most one bit.  That is why this is
    # restricted to 1-bit outputs: the same union on a WIDE xor is a large
    # over-approximation, which is what made the earlier attempt to route `eor
    # x0,x1,x2,ror #11` through ORABLE cost 50+ points of exactness.
    # -----------------------------------------------------------------------
    if (
        not is_store_target
        and not isinstance(mapping, MemMapping)
        and out_bit_end == out_bit_start
        and slice_ops
        and not _slice_has_constant_dominator(slice_ops)
    ):
        _term = next(
            (
                o
                for o in reversed(slice_ops)
                if o.opcode.name not in ('COPY', 'SUBPIECE', 'PIECE', 'INT_ZEXT', 'INT_SEXT')
            ),
            None,
        )
        if _term is not None and _term.opcode.name in ('INT_XOR', 'BOOL_XOR'):
            _xor_terms: list[Expr] = []
            for _inp in _term.inputs:
                if _inp.space.name == 'const':
                    continue
                _ft = varnode_taint_expr(slice_ops, _inp, _reg_taint_for_floor(mapper))
                if _ft is None:
                    # Outside the closed-form grammar (these operands are INT_SLESS
                    # sign tests): fall back to the union of the architectural
                    # registers the operand derives from.  Sound, and on a 1-bit
                    # output the whole term is capped at one bit.
                    _ft = _cone_register_taint(slice_ops, _inp, mapper)
                if _ft is None:
                    _xor_terms = []
                    break
                _xor_terms.append(_ft)
            if len(_xor_terms) == 2:
                _u = BinaryExpr(Op.OR, _xor_terms[0], _xor_terms[1])
                expr = BinaryExpr(
                    Op.OR, expr, BinaryExpr(Op.AND, AvalancheExpr(_u, 1), Constant(1, 8)),
                )

    # 1-bit flag soundness floor for COND_TRANSPORTABLE.
    #
    # Same principle as the MONOTONIC floor: when dep operands are fully
    # tainted, masking forces inputs to 0, making conditional expressions
    # (e.g. INT_NOTEQUAL(0,0)=0) return 0 even though the flag depends on
    # the input.  The floor fires only when dep taint == full mask.
    # Suppressed for constant-dominating slices.
    #
    # Applies to both:
    #   - 1-bit outputs (flag registers: ZF, CF, etc.)
    #   - Small byte outputs (e.g. setcc al = RAX[7:0]) whose ALL deps are
    #     1-bit flag registers.  The floor produces a 1-byte result (0x01 or 0).
    _ct_is_small_output = (out_bit_end - out_bit_start) <= 7  # ≤ 8 bits wide
    _ct_all_deps_one_bit = bool(dep_set.value_deps) and all(
        isinstance(dm, RegMapping) and dm.bit_end == dm.bit_start for dm in dep_set.value_deps.keys()
    )
    # Also fire for a WIDE output when every dep is a 1-bit flag: a condition
    # consumed into a large register (ARM64 `cset`/`csel` -> 64-bit GPR) is still a
    # 0/1 in bit 0, and the 2-corner differential floor misses the non-monotone case
    # where BOTH flags are tainted (`cset x0,lt` = N!=V: the corners (1,1) and (0,0)
    # both give 0, so the interior (0,1)/(1,0) is missed).  FullMaskAvalanche per
    # 1-bit flag dep (masked to bit 0 below) is the sound floor there.
    if (
        cat == InstructionCategory.COND_TRANSPORTABLE
        and (_ct_is_small_output or _ct_all_deps_one_bit)
        and not isinstance(mapping, MemMapping)
        and not _slice_has_constant_dominator(slice_ops)
    ):
        for dep_map in dep_set.value_deps.keys():
            if isinstance(dep_map, RegMapping):
                dep_bits = dep_map.bit_end - dep_map.bit_start + 1
                dep_expr = _get_taint_operand(dep_map.name, dep_map.bit_start, dep_map.bit_end, True)
                _out_width = out_bit_end - out_bit_start + 1
                _floor: Expr = FullMaskAvalancheExpr(dep_expr, dep_bits)
                if _out_width > 1 and _ct_all_deps_one_bit:
                    # setcc-style byte output: result can only be 0x00 or 0x01.
                    # Taint floor = 0x01 (bit 0 only), NOT 0xFF.
                    # AvalancheExpr would give 0xFF (all 8 bits), which is wrong
                    # because bits 7:1 of the result are always 0.
                    _floor = BinaryExpr(Op.AND, _floor, Constant(1, _out_width))
                elif _out_width > 1:
                    _floor = AvalancheExpr(_floor, _out_width)
                expr = BinaryExpr(Op.OR, expr, _floor)

        # Packed OPPOSITE-polarity comparisons.  PPC `cmpw` writes CR0 = LT(r4<r5) |
        # GT(r5<r4) | EQ into one field: a single polarity split (apply_sless_msb_split)
        # can make the differential exact for LT *or* GT but not both, so with both
        # operands sharing a partial taint mask the 2-corner differential under-taints
        # whichever comparison got the wrong polarity (measured: the LT/GT bits of the
        # CR0 field consumed by `mfcr`).  When the slice contains a comparison and its
        # OPERAND-SWAPPED twin -- `c(a,b)` and `c(b,a)` -- OR in the pairwise avalanche
        # (fires when >=2 deps are simultaneously tainted), the same floor the MONOTONIC
        # symmetric-comparison branch uses.  x86 `setcc` reads a SINGLE flag / one
        # comparison direction, so it has no swapped twin and stays bit-exact.
        _cmp_ops = frozenset({
            'INT_LESS', 'INT_LESSEQUAL', 'INT_SLESS', 'INT_SLESSEQUAL',
            'INT_EQUAL', 'INT_NOTEQUAL',
        })
        _cmps = [
            op for op in slice_ops
            if op.opcode.name in _cmp_ops and len(op.inputs) == 2
        ]
        _has_swapped_cmp = any(
            get_varnode_id(a.inputs[0]) == get_varnode_id(b.inputs[1])
            and get_varnode_id(a.inputs[1]) == get_varnode_id(b.inputs[0])
            for i, a in enumerate(_cmps) for b in _cmps[i + 1:]
        )
        if _has_swapped_cmp:
            _pw = [
                AvalancheExpr(_get_taint_operand(dm.name, dm.bit_start, dm.bit_end, True), 1)
                for dm in dep_set.value_deps if isinstance(dm, RegMapping)
            ]
            _ow = out_bit_end - out_bit_start + 1
            for _i in range(len(_pw)):
                for _j in range(_i + 1, len(_pw)):
                    _pair: Expr = BinaryExpr(Op.AND, _pw[_i], _pw[_j])
                    if _ow > 1:
                        _pair = AvalancheExpr(_pair, _ow)
                    expr = BinaryExpr(Op.OR, expr, _pair)

    # For non-stack LOAD pointers in non-load-like instructions (e.g. ADD RAX, [RBX]),
    # OR in the pointer avalanche. Stack pointer excluded for the same reason.
    if not is_load_like and has_tainted_non_stack_pointer:
        ptr_combined = non_stack_addr_taint_exprs[0]
        for t in non_stack_addr_taint_exprs[1:]:
            ptr_combined = BinaryExpr(Op.OR, ptr_combined, t)
        avalanche_ptr = AvalancheExpr(ptr_combined, out_bit_end - out_bit_start + 1)
        expr = BinaryExpr(Op.OR, expr, avalanche_ptr)

    # Conditional-execution gated passthrough (covers CMOV in all categories).
    #
    # The polarised old-dest injection (added in process_dependencies above)
    # makes the differential exact in the two CONCRETE-CONDITION cases:
    #   - condition concretely TAKEN     → both reps overwrite dest →
    #                                       Diff = T_source         ✓
    #   - condition concretely NOT-TAKEN → both reps preserve dest with
    #                                       polarised V|T / V&~T values →
    #                                       Diff = T_old_dest       ✓
    #
    # When the condition flag IS tainted, the high and low replicas can take
    # *different* paths — the XOR may cancel coincidentally-equal bits and
    # underestimate the true taint.  Per the cmov spec:
    #   - condition tainted → T_out = T_old_dest U T_source
    # We OR in this union, gated by the flag-taint mask.  When all flags are
    # concrete (taint=0), the gate evaluates to 0 and the differential alone
    # gives the exact answer.
    #
    # Excluded: PC/IP (branch targets) and memory outputs (no prior value).
    if (
        not isinstance(mapping, MemMapping)
        and out_name not in ('EIP', 'RIP', 'PC')
        and has_cbranch
        and output_cond_written
    ):
        old_dest_taint = _get_taint_operand(out_name, out_bit_start, out_bit_end, True)
        # Resolve the branch-condition flag registers through the mapper rather than
        # a hardcoded x86 flag-offset table: the gated-passthrough must fire on
        # every ISA (ARM64 `csel`/`cset` read NZCV at offsets 256..259, PPC reads
        # CR bits, etc.).  A hardcoded x86 map left those unresolved, so a tainted
        # condition never OR-ed the source/old-dest union in and `csel` under-tainted.
        flag_taint_or: Expr | None = None
        for flag_off, flag_size in cbranch_flag_deps or []:
            _fm = mapper.map_to_state(flag_off, flag_size)
            if _fm is None:
                continue
            flag_taint = _get_taint_operand(_fm.name, _fm.bit_start, _fm.bit_end, True)
            flag_taint_or = flag_taint if flag_taint_or is None else BinaryExpr(Op.OR, flag_taint_or, flag_taint)

        # When the selector flag is computed INSIDE this instruction (cmpxchg's ZF
        # from RAX vs the destination), it is not a tracked register, so its threaded
        # taint is 0 and cannot gate anything.  Gate on the taint of this
        # instruction's own value inputs instead: any tainted input can make the
        # data-dependent select diverge.
        if cbranch_cond_internal:
            for dep_map in dep_set.value_deps.keys():
                if isinstance(dep_map, RegMapping):
                    cond_taint: Expr = _get_taint_operand(
                        dep_map.name,
                        dep_map.bit_start,
                        dep_map.bit_end,
                        True,
                    )
                else:
                    _cond_base = _get_taint_operand(
                        dep_map.addr_reg.name,
                        dep_map.addr_reg.bit_start,
                        dep_map.addr_reg.bit_end,
                        False,
                    )
                    _cond_addr: Expr = (
                        BinaryExpr(Op.ADD, _cond_base, Constant(dep_map.addr_const_offset, 8))
                        if dep_map.addr_const_offset != 0
                        else _cond_base
                    )
                    cond_taint = MemoryOperand(_cond_addr, dep_map.size_bytes, is_taint=True)
                flag_taint_or = cond_taint if flag_taint_or is None else BinaryExpr(Op.OR, flag_taint_or, cond_taint)
        if flag_taint_or is not None:
            out_width = out_bit_end - out_bit_start + 1
            gate = AvalancheExpr(flag_taint_or, out_width)
            # WELDABLE INTERIOR of a two-write conditional select.  When the selector can
            # vary, the reachable output is the UNION of the two branch outcomes, so its
            # taint is the WELD of the branch differentials plus their value difference:
            #     D(A) | D(B) | (val(A) ^ val(B))
            # The extremal 2-corner differential cannot reach this interior -- `cneg`'s
            # selector is `NG==OV`, a non-monotone EQUALITY whose two corners agree, so
            # the branch flip is never sampled and x0 under-tainted.  Taking each branch's
            # OWN differential (not a coarse operand XOR) also makes `csinc`/`csneg` exact,
            # since their branch is `x2+1` / `-x2` with a real add/negate taint.  No
            # old-dest term is needed: one of the two branches always writes the output.
            _weld = (
                _select_weld(slice_ops, all_ops, cbranch_op, mapper, out_width)
                if (cbranch_op is not None and all_ops is not None) else None
            )
            if _weld is not None:
                expr = BinaryExpr(Op.OR, expr, BinaryExpr(Op.AND, _weld, gate))
            else:
                # ONE-write select (x86 `cmov`: the not-taken path keeps the OLD
                # destination) or a branch shape outside the grammar -- fall back to the
                # old-dest + source union plus the pairwise operand value difference.
                source_taint_or: Expr | None = None
                for dep in dependencies:
                    # Skip the injected old-dest dep (out_name with the same slice).
                    if (
                        isinstance(dep, TaintOperand)
                        and dep.name == out_name
                        and dep.bit_start == out_bit_start
                        and dep.bit_end == out_bit_end
                    ):
                        continue
                    source_taint_or = dep if source_taint_or is None else BinaryExpr(Op.OR, source_taint_or, dep)
                if source_taint_or is not None:
                    combined: Expr = BinaryExpr(Op.OR, old_dest_taint, source_taint_or)
                else:
                    combined = old_dest_taint
                _sel_vals: list[Expr] = [
                    _get_taint_operand(dm.name, dm.bit_start, dm.bit_end, False)
                    for dm in dep_set.value_deps
                    if isinstance(dm, RegMapping) and dm.bit_end > dm.bit_start
                ]
                for _i in range(len(_sel_vals)):
                    for _j in range(_i + 1, len(_sel_vals)):
                        _vd: Expr = BinaryExpr(Op.XOR, _sel_vals[_i], _sel_vals[_j])
                        combined = BinaryExpr(Op.OR, combined, _vd)
                expr = BinaryExpr(Op.OR, expr, BinaryExpr(Op.AND, combined, gate))
        else:
            # Fall back to unconditional passthrough when flag deps aren't identified
            # (e.g. CBRANCH on a non-flag predicate). Sound but possibly imprecise.
            expr = BinaryExpr(Op.OR, expr, old_dest_taint)

    # -----------------------------------------------------------------------
    # SIGNED-OVERFLOW FLAG FLOOR, category-independent.
    #
    # A conditional compare (`ccmp`/`ccmn`) sets its overflow flag as
    # `cond ? sborrow(a,b) : bit_of_#nzcv`, so the CBRANCH puts the OV slice in
    # COND_TRANSPORTABLE, where the exact SignedOverflowTaintExpr -- only tried in
    # the MONOTONIC branch -- never runs.  Signed overflow is non-monotone, so the
    # 2-corner differential misses it (measured: `ccmp x1,x2,#0,al` OV).  The exact
    # term is a SOUND floor (it never under-taints), so OR it in whenever the slice
    # is a lone overflow predicate, regardless of category or conditional wrapper.
    if (
        not is_store_target
        and not isinstance(mapping, MemMapping)
        and out_bit_end == out_bit_start
        and any(o.opcode.name in ('INT_SBORROW', 'INT_SCARRY') for o in slice_ops)
    ):
        _ovf_floor = _build_signed_overflow_taint(slice_ops, mapper)
        if _ovf_floor is not None:
            expr = BinaryExpr(Op.OR, expr, BinaryExpr(Op.AND, _ovf_floor, Constant(1, 8)))

    # -----------------------------------------------------------------------
    # SIGN FLAG FLOOR for SHIFTED-operand arithmetic (NG/SF).
    #
    # NG = (a - (b<<k)) s< 0 is the sign bit of the result.  When the shift brings
    # a tainted bit of `b` up to the subtrahend's SIGN position, toggling it is a
    # 2^(w-1) jump that WRAPS the subtract, so the sign becomes non-monotone and the
    # 2-corner differential misses it even with correct polarity (measured: `cmp
    # x1,x2,lsl #3` NG, `adds x0,x1,x2,lsl #2` NG).  The plain (unshifted) sign is
    # monotone and already exact, so this is gated on a shift feeding the
    # arithmetic -- plain `cmp`/`adds` NG is left untouched.
    #
    # The non-monotonicity is precisely a TAINTED SIGN BIT of an operand: toggling
    # it is the 2^(w-1) wrap.  Low-bit taint keeps the sign monotone (a single
    # zero-crossing the differential's polarity corners already bracket), so the
    # floor fires ONLY when a transformed operand's own sign bit is tainted --
    # which keeps it tight (no smear) and leaves the monotone cases exact.
    _sign_term = next(
        (o for o in reversed(slice_ops)
         if o.opcode.name not in ('COPY', 'SUBPIECE', 'PIECE', 'INT_ZEXT', 'INT_SEXT')),
        None,
    )
    _arith = next(
        (o for o in slice_ops
         if o.opcode.name in ('INT_ADD', 'INT_SUB', 'INT_2COMP', 'INT_AND', 'INT_OR', 'INT_XOR')),
        None,
    )
    _has_shift = any(o.opcode.name in ('INT_LEFT', 'INT_RIGHT', 'INT_SRIGHT') for o in slice_ops)
    if (
        not is_store_target
        and not isinstance(mapping, MemMapping)
        and out_bit_end == out_bit_start
        and _sign_term is not None
        and _sign_term.opcode.name in ('INT_SLESS', 'INT_SLESSEQUAL')
        and any(i.space.name == 'const' and i.offset == 0 for i in _sign_term.inputs)
        and _arith is not None
        and _arith.output is not None
        and _has_shift
    ):
        _rt = _reg_taint_for_floor(mapper)
        _parts: list[Expr] = []
        for _inp in _arith.inputs:
            if _inp.space.name == 'const':
                continue
            _pt = varnode_taint_expr(slice_ops, _inp, _rt)
            if _pt is None:
                _pt = _cone_register_taint(slice_ops, _inp, mapper)
            if _pt is not None:
                _parts.append(_pt)
        if _parts:
            _uu: Expr = _parts[0]
            for _p in _parts[1:]:
                _uu = BinaryExpr(Op.OR, _uu, _p)
            _aw = _arith.output.size * 8
            # Fire iff a transformed operand's SIGN bit (bit w-1) is tainted.
            _msb = BinaryExpr(Op.AND, _uu, Constant(1 << (_aw - 1), 8))
            expr = BinaryExpr(Op.OR, expr, BinaryExpr(Op.AND, AvalancheExpr(_msb, 1), Constant(1, 8)))

    # -----------------------------------------------------------------------
    # EQUALITY-TO-ZERO FLAG FLOOR  (ZF and friends)
    #
    # ZF = (a + b == 0) is NON-MONOTONE: the 2-corner differential samples only the
    # polarity extremes, and a wrapping sum can be 0 at BOTH extremes while an
    # interior value is non-zero (measured on `add al,bl` with both sign bits
    # tainted: results {0, 0x80, 0x80, 0}, so both corners give ZF=1 and the
    # interior ZF=0 is missed).  The FullMask floor only fires when a dep is FULLY
    # tainted, so partial taint slips through.
    #
    # The exact-shape floor is EqualityTaintExpr(result, 0): ZF is tainted iff the
    # result can be zero (its bits outside the taint are all 0) AND it carries
    # taint.  The result's concrete value comes from a cell (which re-executes and
    # reads the written register), and its taint is over-approximated by the
    # carry-smear of the source taints -- sound because a carry from a tainted bit
    # only propagates upward, so smearing each source taint up and unioning them
    # covers every bit the sum's taint can reach.  It stays precise on the common
    # case: when the result has a fixed non-zero bit outside the taint (a large
    # `add rax,rbx`), the equal-reachable test is false and the floor does nothing.
    _eq_term = next(
        (o for o in reversed(slice_ops)
         if o.opcode.name not in ('COPY', 'SUBPIECE', 'PIECE', 'INT_ZEXT', 'INT_SEXT')),
        None,
    )
    if (
        not is_store_target
        and not isinstance(mapping, MemMapping)
        and out_bit_end == out_bit_start  # a 1-bit flag
        and _eq_term is not None
        and _eq_term.opcode.name in ('INT_EQUAL', 'INT_NOTEQUAL')
        and len(_eq_term.inputs) == 2
        # only carry/borrow-propagating arithmetic, where smear-up is a sound
        # over-estimate of the result taint
        and any(o.opcode.name in ('INT_ADD', 'INT_SUB', 'INT_2COMP') for o in slice_ops)
        and not _slice_has_constant_dominator(slice_ops)
    ):
        _c_in = next((i for i in _eq_term.inputs if i.space.name == 'const'), None)
        _r_in = next((i for i in _eq_term.inputs if i.space.name == 'register'), None)
        if _r_in is None:
            # xadd/cmpxchg keep the compared arithmetic result in a UNIQUE
            # (`ZF = unique == 0`) while also copying that same unique into an
            # architectural register (`AL = unique`).  The cell can only read
            # registers, so resolve the compared unique to that aliasing register
            # -- the two hold the same value at the comparison point.  This
            # generalises the floor past the register-compared shapes (add/sub/cmp)
            # to every exchange/RMW-with-flags instruction.
            _u_in = next((i for i in _eq_term.inputs if i.space.name == 'unique'), None)
            if _u_in is not None and all_ops is not None:
                for _o in all_ops:
                    if (
                        _o.opcode.name == 'COPY'
                        and _o.output is not None
                        and _o.output.space.name == 'register'
                        and len(_o.inputs) == 1
                        and _o.inputs[0].space.name == 'unique'
                        and _o.inputs[0].offset == _u_in.offset
                        and _o.inputs[0].size == _u_in.size
                    ):
                        _r_in = _o.output
                        break
        _rm = mapper.map_to_state(_r_in.offset, _r_in.size) if _r_in is not None else None
        if _c_in is not None and _r_in is not None and isinstance(_rm, RegMapping):
            _w = _r_in.size * 8
            _src_t: Expr | None = None
            for _dm in dep_set.value_deps:
                if isinstance(_dm, RegMapping):
                    _t = _get_taint_operand(_dm.name, _dm.bit_start, _dm.bit_end, True)
                    _src_t = _t if _src_t is None else BinaryExpr(Op.OR, _src_t, _t)
            if _src_t is not None:
                _smear = _src_t
                _step = 1
                while _step < _w:
                    _smear = BinaryExpr(Op.OR, _smear, BinaryExpr(Op.LEFT, _smear, Constant(_step, 8)))
                    _step *= 2
                _smear = BinaryExpr(Op.AND, _smear, Constant((1 << _w) - 1, 8))
                # The cell re-executes the instruction; it needs the concrete VALUES
                # of the registers it reads (an empty input map evaluates to 0).
                _cell_inputs: dict[str, Expr] = {
                    _dm.name: _get_taint_operand(_dm.name, _dm.bit_start, _dm.bit_end, False)
                    for _dm in dep_set.value_deps
                    if isinstance(_dm, RegMapping)
                }
                _result_val = InstructionCellExpr(
                    arch, bytestring.hex(), _rm.name, _rm.bit_start, _rm.bit_end, _cell_inputs,
                )
                _eq_floor: Expr = EqualityTaintExpr(
                    _result_val, _smear, Constant(_c_in.offset, _w), Constant(0, _w), _w,
                )
                expr = BinaryExpr(Op.OR, expr, BinaryExpr(Op.AND, _eq_floor, Constant(1, 8)))

    # -----------------------------------------------------------------------
    # VARIABLE SHIFT/MASK AMOUNT FLOOR  (bextr / bzhi and shift-composed-with-mask)
    #
    # A shift by a TAINTED amount taints its whole reachable output range -- the
    # shifted bit or the mask boundary can land anywhere the amount reaches.  The
    # exact subcube term (_build_variable_shift_taint) captures this for a LONE
    # masked shift; when it declines because the shift is COMPOSED with a mask
    # (bextr = `(x>>START) & ((1<<LEN)-1)`, bzhi, blsmsk) the amount's
    # non-monotonicity is otherwise lost, under-tainting the exposed bits.  As a
    # sound fallback OR in an avalanche over the output width, gated on the taint of
    # every non-constant shift amount: AvalancheExpr is 0 when the amounts are
    # untainted, so fixed-shift/fixed-mask code stays exact.  Fires only where the
    # exact term already declined, so it never overrides a precise variable shift.
    _var_amt_ops = [
        _o for _o in slice_ops
        if _o.opcode.name in ('INT_LEFT', 'INT_RIGHT', 'INT_SRIGHT')
        and len(_o.inputs) == 2
        and _o.inputs[1].space.name != 'const'
    ]
    if (
        _var_amt_ops
        and not isinstance(mapping, MemMapping)
        and _build_variable_shift_taint(slice_ops, mapper, out_bit_end - out_bit_start + 1) is None
    ):
        _rt2 = _reg_taint_for_floor(mapper)
        _sh_amt_taints: list[Expr] = []
        for _o in _var_amt_ops:
            _at = varnode_taint_expr(slice_ops, _o.inputs[1], _rt2)
            if _at is not None:
                _sh_amt_taints.append(_at)
        if _sh_amt_taints:
            _sh_amt_union: Expr = _sh_amt_taints[0]
            for _a in _sh_amt_taints[1:]:
                _sh_amt_union = BinaryExpr(Op.OR, _sh_amt_union, _a)
            _ow2 = out_bit_end - out_bit_start + 1
            expr = BinaryExpr(Op.OR, expr, AvalancheExpr(_sh_amt_union, _ow2))

    # -----------------------------------------------------------------------
    # NEGATE-THROUGH-SHIFT BORROW FLOOR  (neg/negs with a shifted operand)
    #
    # `-(x << k)` propagates a two's-complement borrow UP from the lowest tainted
    # bit.  Plain `neg`/`sub` are exact under the 2-corner differential, but once a
    # shift relocates the tainted bits the differential misses the borrow and
    # under-taints (the transport union floor sits at the operand's RAW positions,
    # not the shifted ones).  Gated on a negate -- INT_2COMP, or `0 - x` via
    # INT_SUB with a constant-0 minuend (how `negs` lifts) -- *and* a shift, so
    # plain negate and shifted add/sub (all already sound) are untouched.  OR in the
    # borrow-smear of the negated operand's taint read at its post-shift positions.
    _neg_operand = None
    _2comp_op = next((_o for _o in slice_ops if _o.opcode.name == 'INT_2COMP'), None)
    if _2comp_op is not None:
        _neg_operand = _2comp_op.inputs[0]
    else:
        _sub0 = next(
            (_o for _o in slice_ops
             if _o.opcode.name == 'INT_SUB' and len(_o.inputs) == 2
             and _o.inputs[0].space.name == 'const' and _o.inputs[0].offset == 0),
            None,
        )
        if _sub0 is not None:
            _neg_operand = _sub0.inputs[1]
    if (
        _neg_operand is not None
        and any(_o.opcode.name in ('INT_LEFT', 'INT_RIGHT', 'INT_SRIGHT') for _o in slice_ops)
        and not isinstance(mapping, MemMapping)
        and out_bit_end > out_bit_start
    ):
        _ot = varnode_taint_expr(slice_ops, _neg_operand, _reg_taint_for_floor(mapper))
        if _ot is not None:
            _w3 = out_bit_end - out_bit_start + 1
            _sm3 = _ot
            _step3 = 1
            while _step3 < _w3:
                _sm3 = BinaryExpr(Op.OR, _sm3, BinaryExpr(Op.LEFT, _sm3, Constant(_step3, 8)))
                _step3 *= 2
            _sm3 = BinaryExpr(Op.AND, _sm3, Constant((1 << _w3) - 1, 8))
            expr = BinaryExpr(Op.OR, expr, _sm3)

    if out_name in ('EIP', 'RIP', 'PC'):
        expr = AvalancheExpr(expr, out_bit_end - out_bit_start + 1)

    assignments.append(TaintAssignment(target=out_target, dependencies=dependencies, expression=expr))


def _slice_has_constant_dominator(slice_ops: list[PcodeOp]) -> bool:  # noqa: C901
    """Return True if the backward slice contains an operation whose output is
    always a constant regardless of any tainted input.

    This detects three patterns:
      - INT_AND with a constant-0 operand: any_val AND 0 = 0 always.
      - INT_OR  with an all-ones constant:  any_val OR -1 = -1 always.
      - INT_XOR / INT_SUB where both inputs are the same register:
        x XOR x = 0, x SUB x = 0 always (zeroing idioms).

    When any of these are present, the result register (and any flags that
    depend on it) is always a constant, so the FullMaskAvalancheExpr floor
    must NOT fire for their flag assignments.
    """
    for op in slice_ops:
        if op.opcode.name == 'INT_AND':
            for vn in op.inputs:
                if vn.space.name == 'const' and vn.offset == 0:
                    return True
        elif op.opcode.name == 'INT_OR':
            for vn in op.inputs:
                if vn.space.name == 'const':
                    full = (1 << (vn.size * 8)) - 1
                    if vn.offset == full:
                        return True
        elif op.opcode.name in ('INT_XOR', 'INT_SUB'):
            ins = op.inputs
            if (
                len(ins) == 2
                and ins[0].space.name == 'register'
                and ins[1].space.name == 'register'
                and ins[0].offset == ins[1].offset
                and ins[0].size == ins[1].size
            ):
                return True
    return False


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


@functools.lru_cache(maxsize=None)
def _synth_simulator(arch: Architecture) -> CellSimulator:
    """Concrete p-code evaluator used at rule-synthesis time (offline), cached per
    architecture. Used only to fold the single-call MAPPED rule's affine constant
    part ``f|_{x->0}``; never on the runtime hot path.
    """
    from microtaint.simulator import CellSimulator  # noqa: PLC0415

    return CellSimulator(arch, use_unicorn=False, use_c=True)


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


# Lane-independent P-code ops: output byte j depends only on input byte j, at any
# width.  For a >8-byte (vector) value these can be analysed one 64-bit lane at a
# time with no interaction between lanes, which is what lets a single rule keep
# vector taint exact without widening the 64-bit core (see _lane_ok below).  Ops
# NOT in this set (arithmetic with carries, shuffles/CALLOTHER, widening
# INT_ZEXT/SEXT, SUBPIECE/PIECE at non-lane offsets) can move taint across lanes,
# so they are left to the conservative whole-value path.
_LANE_INDEPENDENT_OPS = frozenset({
    'COPY', 'INT_XOR', 'INT_AND', 'INT_OR', 'INT_NEGATE', 'LOAD', 'STORE', 'IMARK',
    # INT_ZEXT is lane-independent for the bytes it copies (byte j <- byte j) and
    # zero-fills the rest; AVX loads/moves lift the upper-lane zeroing of the
    # 256-bit result as a ZEXT into the 512-bit register.  A lane beyond the source
    # gets no dep (untainted), handled where the range is narrowed.
    'INT_ZEXT',
})


def _trace_wide_store_source(
    vn: Varnode, all_ops: list[PcodeOp], seen: set[int] | None = None,
) -> Varnode | None:
    """The single register varnode a STORE value copies verbatim, byte-for-byte,
    or None.  Follows the size-preserving COPY chain SLEIGH emits for a wide
    register store (`movups [mem], xmm`) so that byte j of the stored value is
    byte j of the register.  Any non-copy or size-changing step returns None."""
    if vn.space.name == 'register':
        return vn
    if vn.space.name != 'unique':
        return None
    if seen is None:
        seen = set()
    if vn.offset in seen:
        return None
    seen.add(vn.offset)
    for op in all_ops:
        out = op.output
        if out is not None and out.space.name == 'unique' and out.offset == vn.offset and out.size == vn.size:
            if op.opcode.name == 'COPY' and len(op.inputs) == 1 and op.inputs[0].size == vn.size:
                return _trace_wide_store_source(op.inputs[0], all_ops, seen)
            return None
    return None


def _exact_store_lane_targets(
    src_reg: Varnode,
    ptr_vn: Varnode,
    base_reg: RegMapping | None,
    const_offset: int,
    size: int,
    mapper: StateMapper,
    pc_relative: frozenset[int],
    pc_reg: RegMapping | None,
) -> list[TaintAssignment] | None:
    """One exact 8-byte-lane STORE assignment per register sub-range of a wide
    copy: memory bytes [k, k+8) get EXACTLY the taint of the source register's
    bytes [k, k+8) (e.g. XMM<n>_LO -> bytes 0-7, XMM<n>_HI -> bytes 8-15).  An
    untainted half evaluates to 0 and correctly clears its lane.  Returns None if
    any lane's source does not resolve to a single state register."""
    lanes: list[tuple[int, int, RegMapping]] = []
    for k in range(0, size, 8):
        csize = min(8, size - k)
        src = mapper.map_to_state(src_reg.offset + k, csize)
        if src is None:
            return None
        lanes.append((k, csize, src))
    out_asgs: list[TaintAssignment] = []
    for k, csize, src in lanes:
        if base_reg is None:
            mem_map: MemMapping = _const_addr_mem(const_offset + k, csize, pc_relative, pc_reg)
        else:
            mem_map = MemMapping(ptr_vn.offset, csize, base_reg, const_offset + k)
        mem_target, _n, _bs, _be = generate_output_target(mem_map)
        src_expr: Expr = _get_taint_operand(src.name, src.bit_start, src.bit_end, True)
        out_asgs.append(TaintAssignment(target=mem_target, dependencies=[src_expr], expression=src_expr))
    return out_asgs


def _load_byte_range(
    target: Varnode, load_out: Varnode, all_ops: list[PcodeOp],
) -> tuple[int, int] | None:
    """Byte offset and size, within a wide LOAD's output `load_out`, that feed
    `target`, following the size-preserving COPY chain SLEIGH emits into the XMM
    destination (byte j of the load fills byte j of the register).  None if the
    flow is not a plain copy of a contiguous load sub-range, in which case the
    caller keeps the whole-load dependency (a sound over-approximation)."""

    def walk(space: str, offset: int, sz: int, seen: set[tuple[str, int, int]]) -> tuple[int, int] | None:
        key = (space, offset, sz)
        if key in seen:
            return None
        seen.add(key)
        for op in all_ops:
            out = op.output
            if out is None or out.space.name != space:
                continue
            if not (out.offset <= offset and offset + sz <= out.offset + out.size):
                continue
            sub = offset - out.offset
            if (
                op.opcode.name == 'LOAD'
                and out.space.name == load_out.space.name
                and out.offset == load_out.offset
                and out.size == load_out.size
            ):
                return (sub, sz)
            if op.opcode.name == 'COPY' and len(op.inputs) == 1 and op.inputs[0].size == out.size:
                src = op.inputs[0]
                return walk(src.space.name, src.offset + sub, sz, seen)
            return None
        return None

    return walk(target.space.name, target.offset, target.size, set())


def extract_dependencies(  # noqa: C901
    _out_vn: Varnode,
    _slice_ops: list[PcodeOp],
    polarities: dict[str, int],
    all_ops: list[PcodeOp],
    mapper: StateMapper,
    pc_relative: frozenset[int] = frozenset(),
    pc_reg: RegMapping | None = None,
    load_sub: tuple[int, int] | None = None,
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

    Root cause fix #2 (dest read-back): Ghidra emits flag-update ops that
    read the destination register AFTER the main computation has written it.
    For example, `imul rax,rbx,3` emits `INT_SEXT in=RAX` to compute OF
    after `INT_MULT out=RAX`.  That RAX read should not count as an *input*
    dep.  We track the index of the first write to the output register and
    suppress reads of that same register from ops that come *after* the write.
    """
    value_deps: dict[RegMapping | MemMapping, int] = {}
    addr_deps: dict[RegMapping, int] = {}

    # Lane rule: this target is one 64-bit lane, at byte offset load_sub[0], of a
    # wider (vector) result.  When the whole value-producing slice is
    # lane-independent, output byte j depends only on input byte j, so every input
    # contributes only its bytes at this lane -- the same restriction applied to
    # memory (below) and to register inputs (map_to_state at input.offset + lane
    # offset).  This is what keeps pxor/pand/por and reg-to-reg movaps exact
    # instead of smearing taint across the two 128-bit halves; it needs the guard
    # so shuffles / cross-lane arithmetic stay on the conservative whole-value path.
    _lane_ok: bool = load_sub is not None and bool(_slice_ops) and all(
        _o.opcode.name in _LANE_INDEPENDENT_OPS for _o in _slice_ops
    )

    # Find the index of the first op in all_ops that writes to the output register.
    # Any subsequent op that reads the same register is a read-back of the result
    # (e.g. Ghidra's flag-update patterns) and must NOT be treated as an input dep.
    # We only apply this when the output is a named register (not memory, not unique).
    _out_write_index: int = len(all_ops)  # sentinel: no write found yet
    if _out_vn.space.name == 'register':
        for _i, _op in enumerate(all_ops):
            if (
                _op.output is not None
                and _op.output.space.name == 'register'
                and _op.output.offset == _out_vn.offset
                and _op.output.size == _out_vn.size
            ):
                _out_write_index = _i
                break

    # Registers that are used as LOAD/STORE *pointers* within this slice.
    # We identify them first so we can classify register inputs correctly.
    ptr_reg_offsets: set[int] = set()

    # Compute first-write index for EVERY register written in all_ops.
    # This lets us suppress reads of any register that was written by an
    # earlier op in the same P-code block (intra-instruction intermediate).
    # Example: ror rax,1 writes CF (new-CF = bit0 of RAX) and then reads CF
    # again to place it at bit63. The CF read is not an external input dep.
    _reg_first_write: dict[int, int] = {}  # register offset → first write op index
    for _i, _op in enumerate(all_ops):
        if _op.output is not None and _op.output.space.name == 'register' and _op.output.offset not in _reg_first_write:
            _reg_first_write[_op.output.offset] = _i

    # Byte-precise first-write index: for each register-space byte, the index of
    # the first op that writes it.  A later read whose EVERY byte was written by a
    # strictly-earlier op observes this instruction's own intermediate result
    # (SSA within the instruction), not an external input, so it must not become a
    # value dependency.  This is the byte-granular generalisation of the two
    # coarser (offset-keyed / exact-size) checks below.  It is what suppresses the
    # trailing INT_ZEXT of `mov eax,[mem]` reading the load-written EAX: that read
    # is offset 0 size 4 while the output RAX is offset 0 size 8, so the exact-size
    # check misses it and it leaks in as a spurious old-destination dep -- the root
    # cause of the load strong-update failure (a load from untainted memory then
    # fails to clear the destination register's stale taint).
    _reg_byte_first_write: dict[int, int] = {}
    for _i, _op in enumerate(all_ops):
        if _op.output is not None and _op.output.space.name == 'register':
            for _b in range(_op.output.offset, _op.output.offset + _op.output.size):
                if _b not in _reg_byte_first_write:
                    _reg_byte_first_write[_b] = _i

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

    for _op_idx, op in enumerate(all_ops):
        if op.opcode.name in ('RETURN', 'IMARK'):
            # IMARK is a disassembly marker whose input is the instruction's own
            # bytes (a ram-space varnode at the PC) — not a data dependency.
            continue

        if op.opcode.name == 'LOAD':
            ptr_vn = op.inputs[1]
            mapped_addr, const_offset = resolve_ptr_with_offset(ptr_vn, all_ops, mapper, stop_op_index=load_op_index)
            _load_size = op.output.size if op.output else 8
            _load_off = 0
            _skip_load = False
            if _load_size > 8 and op.output is not None:
                # EXACT wide load: a >8-byte memory taint cannot be held in one
                # 64-bit mask, and the whole-load MemoryOperand aliases the high
                # bytes onto the low 8.  Restrict the dependency to the exact byte
                # range of the load that feeds THIS target, so each XMM half reads
                # only its own 8 bytes.  Non-copy value flows keep the whole load
                # (a sound over-approximation) via _load_byte_range -> None.
                rng = _load_byte_range(_out_vn, op.output, all_ops)
                if rng is not None:
                    _load_off, _load_size = rng
                # Lane rule: when this target is one 64-bit lane of a wide register
                # output (movdqu/movdqa write the whole XMM in one COPY; AVX loads
                # ZEXT the 32-byte load into the 512-bit register), narrow the load
                # window to this lane.  Applies on any lane-independent slice, even
                # when the COPY-chain trace above found nothing (the ZEXT case).
                if _lane_ok and load_sub is not None:
                    sub_off, sub_size = load_sub
                    if sub_off >= _load_size:
                        # Lane lies beyond the loaded bytes (ZEXT zero-fill) -> the
                        # lane is untainted, so contribute no memory dependency.
                        _skip_load = True
                    elif sub_off + sub_size <= _load_size:
                        _load_off += sub_off
                        _load_size = sub_size
            mem_map = None
            if _skip_load:
                mem_map = None
            elif mapped_addr is not None:
                mem_map = MemMapping(ptr_vn.offset, _load_size, mapped_addr, const_offset + _load_off)
            elif const_offset != 0:
                # The pointer folds to a compile-time constant (absolute / PC-
                # relative literal, e.g. ARM64 `ldr w0,#imm`): a constant-address
                # memory value dep instead of a drop.  PC-relative resolves against
                # the runtime pc; absolute is baked.  (const_offset==0 = unresolvable.)
                mem_map = _const_addr_mem(const_offset + _load_off, _load_size, pc_relative, pc_reg)
            if mem_map is not None:
                # LOAD memory content is a *value* dependency, and it carries the
                # polarity compute_polarity derived for the LOADED VALUE -- keyed by
                # the LOAD's own output varnode.  Hardcoding +1 here silently
                # discarded it: in `cmp rax,[rsp-16]` the loaded value is the
                # SUBTRACTED operand, so its polarity is negative.  Losing that
                # polarised both operands identically, degrading the differential to
                # a lossy D^{++} that cancels -- under-tainting the comparison
                # whenever the memory value itself is tainted.
                value_deps[mem_map] = (
                    polarities.get(get_varnode_id(op.output), 1) if op.output is not None else 1
                )

        for vn in op.inputs:
            if vn.space.name == 'register':
                # Byte-precise intra-instruction intermediate suppression: if EVERY
                # byte of this register read was written by a strictly-earlier op in
                # the same instruction, the read observes this instruction's own
                # result, not an external input.  Generalises the exact-size /
                # non-output checks below; in particular it catches a sub-register
                # read of the output written earlier (INT_ZEXT of a load-written
                # EAX), which those two miss.  Reads before the first write, and the
                # writing op's own input reads, are preserved (first_write >= _op_idx).
                if vn.size > 0 and all(
                    _reg_byte_first_write.get(_b, _op_idx) < _op_idx
                    for _b in range(vn.offset, vn.offset + vn.size)
                ):
                    continue

                # Skip read-backs of the destination register that occur AFTER
                # the main computation has already written it.  Ghidra emits
                # flag-update ops (e.g. `INT_SEXT in=RAX` after `INT_MULT out=RAX`
                # for `imul rax,rbx,3`) that reference the destination as an
                # input — but only to compute OF/CF, not as a true source value.
                # We suppress these only when the op comes after _out_write_index.
                # Reads of the destination BEFORE the write (e.g. bswap's source
                # reads of RAX before the final INT_OR writes RAX) are legitimate.
                if (
                    _op_idx > _out_write_index
                    and vn.offset == _out_vn.offset
                    and vn.size == _out_vn.size
                    and _out_vn.space.name == 'register'
                ):
                    continue

                # Also skip intra-instruction intermediates: any non-output register
                # that was WRITTEN earlier in the same P-code block before this read.
                # Example: ror rax,1 writes CF at op 2 (new-CF = bit0 of RAX) then
                # reads CF at op 4 to place it in bit63. The CF read at op 4 is NOT
                # an external input — it is an intermediate computed from RAX.
                # Without this filter, CF would appear as a dep for ror's RAX output,
                # making it COND_TRANSPORTABLE (2 sources) instead of MAPPED (1 source).
                if (
                    vn.offset in _reg_first_write
                    and _reg_first_write[vn.offset] < _op_idx
                    and vn.offset != _out_vn.offset
                ):  # out_vn handled above
                    continue

                # Lane rule: for a lane-independent slice feeding one 64-bit lane
                # of a wider result, this wide (>8-byte, i.e. vector) input
                # contributes only its bytes at the same lane -- input byte j feeds
                # output byte j.  So map just that 8-byte lane (e.g. the XMM_HI
                # target of `pxor` reads only XMM0_HI and XMM1_HI, not the low
                # halves).  Narrow (<=8-byte) inputs are already a single lane.
                _off, _sz = vn.offset, vn.size
                if _lane_ok and vn.size > 8 and vn.offset not in ptr_reg_offsets:
                    _lane_lo, _lane_sz = load_sub  # type: ignore[misc]
                    if _lane_lo >= vn.size:
                        # This output lane is beyond the (narrower) input -- an
                        # INT_ZEXT zero-fill lane -- so the input contributes none.
                        continue
                    _off, _sz = vn.offset + _lane_lo, min(_lane_sz, vn.size - _lane_lo)

                # Try the singular map first — preserves the existing
                # "smallest covering register" semantics for GPRs and
                # their aliases (RAX → just RAX, not RAX+EAX+AX+AL).
                mapped_dep = mapper.map_to_state(_off, _sz)
                if mapped_dep is not None:
                    mapped_deps: list[RegMapping] = [mapped_dep]
                else:
                    # No single state_format entry covers the input — fall
                    # back to the multi-mapping form.  This is how XMM
                    # registers (split into XMM<n>_LO + XMM<n>_HI in the
                    # state_format) get all their pieces tracked.
                    mapped_deps = mapper.map_to_state_all(_off, _sz)
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

            elif vn.space.name == 'ram' and op.opcode.name not in (
                'BRANCH', 'CBRANCH', 'BRANCHIND', 'CALL', 'CALLIND', 'RETURN', 'STORE',
            ):
                # Direct absolute / PC-relative memory DATA operand.  SLEIGH folds
                # a compile-time-known address into a ram-space varnode on ANY data
                # op (COPY for `mov r,[abs]`; INT_ADD/INT_CARRY for `add r,[abs]`;
                # …), not a LOAD.  The pointer is a constant (untainted), so this
                # is a plain VALUE dependency read from the shadow at the absolute
                # address — no pointer avalanche, exactly like a register value.
                # (An RMW that also writes this cell keeps the read: the old value
                # is a genuine input, unlike an intra-instruction register rewrite.)
                # Excluded: control-flow ops, whose ram operand is a branch/call
                # TARGET address (a control destination, not a memory read), and
                # STORE, whose target is handled separately.
                _mem_dep = _const_addr_mem(vn.offset, vn.size, pc_relative, pc_reg)
                if _mem_dep not in value_deps:
                    value_deps[_mem_dep] = 1

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

    # Subtractive-memory soundness: forward a LOAD's negative polarity across
    # the STORE->LOAD (memory) edge.  compute_polarity works on the varnode
    # def-use graph and cannot see a value that transits memory (e.g.
    # `mov [rsp-16], rbx ; sub rax, [rsp-16]`): it correctly marks the LOAD's
    # output varnode as subtracted (polarity 0), but that polarity never reaches
    # the memory value-dep (hardcoded to 1 above) nor the register that supplied
    # the stored value (added to value_deps with default polarity 1).  Left
    # uncorrected, MemoryDifferentialExpr builds a uniform-polarity D^{++}
    # differential instead of the sound D^{+-}, which under-taints the borrow
    # chain.  We recover the polarity by matching LOAD and STORE addresses.
    load_pol: dict[tuple[str, int], int] = {}
    for op in _slice_ops:
        if op.opcode.name == 'LOAD' and op.output is not None:
            base, off = resolve_ptr_with_offset(op.inputs[1], all_ops, mapper, stop_op_index=load_op_index)
            if base is not None:
                key = (base.name, off)
                load_pol[key] = min(load_pol.get(key, 1), polarities.get(get_varnode_id(op.output), 1))
    if load_pol:

        def _trace_store_value(vn: Varnode, seen: set[int] | None = None) -> list[RegMapping]:
            """Trace a STORE value varnode back to its source register(s),
            through the COPY/extension chain SLEIGH emits for `mov [mem], reg`."""
            if seen is None:
                seen = set()
            if vn.space.name == 'register':
                m = mapper.map_to_state(vn.offset, vn.size)
                return [m] if m is not None else []
            if vn.space.name == 'unique' and vn.offset not in seen:
                seen.add(vn.offset)
                for op in all_ops:
                    if op.output is not None and op.output.space.name == 'unique' and op.output.offset == vn.offset:
                        if op.opcode.name in ('COPY', 'INT_ZEXT', 'INT_SEXT', 'SUBPIECE'):
                            traced: list[RegMapping] = []
                            for inp in op.inputs:
                                if inp.space.name != 'const':
                                    traced += _trace_store_value(inp, seen)
                            return traced
                        break
            return []

        # (a) memory value-deps inherit the polarity of the LOAD at their address.
        for dep_map in list(value_deps.keys()):
            if isinstance(dep_map, MemMapping):
                key = (dep_map.addr_reg.name, dep_map.addr_const_offset)
                if key in load_pol:
                    value_deps[dep_map] = min(value_deps[dep_map], load_pol[key])
        # (b) store-forwarding source registers inherit the polarity of the
        #     memory location they feed.
        for op in all_ops:
            if op.opcode.name == 'STORE':
                base, off = resolve_ptr_with_offset(op.inputs[1], all_ops, mapper)
                if base is None:
                    continue
                key = (base.name, off)
                if key not in load_pol:
                    continue
                for md in _trace_store_value(op.inputs[2]):
                    if md in value_deps:
                        value_deps[md] = min(value_deps[md], load_pol[key])

    return DependencySet(value_deps=value_deps, addr_deps=addr_deps)


def map_outputs_to_targets(  # noqa: C901
    arch: Architecture,
    state_format: list[Register],
    translation: Translation,
    store_ops: list[PcodeOp],
    unique_outputs: Iterable[Varnode],
    mapper: StateMapper,
    ram_outputs: list[Varnode] | None = None,
    pc_relative: frozenset[int] = frozenset(),
    pc_reg: RegMapping | None = None,
) -> tuple[list[EvalTarget], list[TaintAssignment]]:
    targets_to_evaluate: list[EvalTarget] = []
    mem_targets: list[tuple[Varnode, Varnode, int, int]] = []

    for out_vn in unique_outputs:
        mapped_outs = mapper.map_to_state_all(out_vn.offset, out_vn.size)
        for mapped_out in mapped_outs:
            targets_to_evaluate.append(EvalTarget(out_vn, mapped_out))

    # Build a map from STORE op id() → index in translation.ops, so we can
    # pass `stop_op_index` to resolve_ptr_with_offset.  This is critical for
    # `rep`-prefixed string instructions (rep stosb, rep movsb, ...) where
    # the SLEIGH-lifted P-code increments RDI/RSI *after* the STORE within
    # the same instruction's op list.  Without stop_op_index, the resolver
    # walks past the STORE and picks up the post-increment INT_SUB defining
    # RDI = old_rdi + 1 - 2*DF, attributing a +1 constant offset to the
    # address that the STORE itself uses.  Result: the static rule writes
    # taint to [RDI+1] instead of [RDI], so for rep stosb the taint at every
    # iteration N lands at byte N+1 instead of byte N.
    store_op_index: dict[int, int] = {id(op): i for i, op in enumerate(translation.ops) if op.opcode.name == 'STORE'}
    for store_op in store_ops:
        ptr_vn = store_op.inputs[1]
        val_vn = store_op.inputs[2]
        mem_targets.append((val_vn, ptr_vn, val_vn.size, store_op_index[id(store_op)]))

    for op in translation.ops:
        op_name = op.opcode.name
        if op_name in ('CBRANCH', 'BRANCHIND', 'CALLIND'):
            pc_name = 'EIP' if 'X86' in arch.upper() else 'RIP' if 'AMD64' in arch.upper() else 'PC'
            # Local Register for the branch target's PC; must NOT shadow the
            # `pc_reg` parameter (a RegMapping used later for const-address
            # stores/ram outputs) -- reusing the name clobbered it for the rest
            # of the function.
            pc_reg_r = next((r for r in state_format if r.name.upper() == pc_name), None)
            if not pc_reg_r:
                continue

            # For CBRANCH:  inputs[0] = branch destination, inputs[1] = condition predicate.
            # For BRANCHIND/CALLIND: inputs[0] = target address.
            #
            # Skip pcode-internal CBRANCH ops: Sleigh emits these as const-space targets
            # to implement multi-exit instructions (BSF/BSR/TZCNT bit-scan loops, CMOVcc
            # skip patterns, REPNE loop exits, etc.).  A const-space target is a relative
            # pcode-PC offset — it can never be an x86 architectural branch and must NOT
            # contribute a T_RIP assignment.  The old code checked inputs[1] (the condition),
            # which is always unique-space, so the check was a no-op and BSF/BSR/TZCNT
            # incorrectly generated T_RIP assignments from their loop CBRANCHes.
            dest = op.inputs[0]
            if op_name == 'CBRANCH' and dest.space.name == 'const':
                continue

            # The varnode whose taint flows into RIP:
            # - CBRANCH: the condition predicate (inputs[1]); when it is tainted, the
            #   taken/not-taken decision is uncertain, so RIP is tainted.
            # - BRANCHIND/CALLIND: the target address (inputs[0]).
            varnode = op.inputs[1] if op_name == 'CBRANCH' else op.inputs[0]
            if varnode.space.name == 'const':
                continue

            targets_to_evaluate.append(EvalTarget(varnode, RegMapping(pc_reg_r.name, 0, pc_reg_r.bits - 1)))

    assignments: list[TaintAssignment] = []

    for val_vn, ptr_vn, size, store_idx in mem_targets:
        # Resolve the address as it stood AT THE STORE — i.e. ignore any
        # register-update ops that come later in translation.ops, since those
        # writes happen after the STORE has already committed its address.
        base_reg, const_offset = resolve_ptr_with_offset(
            ptr_vn,
            translation.ops,
            mapper,
            stop_op_index=store_idx,
        )
        # EXACT wide-store split: a >8-byte store whose value is a verbatim copy
        # of a wide register (the SIMD memcpy pattern, `movups [mem], xmm`) maps
        # byte j of memory to byte j of that register.  Emit one exact 8-byte lane
        # per register half so each lane's taint is EXACTLY its source half (and
        # an untainted half clears its lane).  Non-copy wide stores fall through
        # to the generic path, where the value is split into sound OR-of-deps
        # chunks (see generate_taint_assignments).
        if size > 8:
            src_reg = _trace_wide_store_source(val_vn, translation.ops)
            if src_reg is not None and src_reg.size == size:
                lane_asgs = _exact_store_lane_targets(
                    src_reg, ptr_vn, base_reg, const_offset, size, mapper, pc_relative, pc_reg,
                )
                if lane_asgs is not None:
                    assignments.extend(lane_asgs)
                    continue
        if base_reg is None:
            # Constant / absolute STORE address (e.g. `mov [rip+d], rax`): the
            # pointer folds to a compile-time constant.  Model it as a
            # constant-address memory target (PC-relative -> runtime pc; absolute
            # -> baked) instead of dropping the store.
            mem_map = _const_addr_mem(const_offset, size, pc_relative, pc_reg)
        else:
            mem_map = MemMapping(ptr_vn.offset, size, base_reg, const_offset)
        targets_to_evaluate.append(EvalTarget(val_vn, mem_map))

    # Direct ram-space OUTPUT varnodes (absolute / PC-relative writes lifted as
    # COPY/op out=ram[addr], not STORE): a constant-address memory target.  The
    # output varnode itself is the value whose taint is computed and stored.
    for ram_out in ram_outputs or ():
        mem_map = _const_addr_mem(ram_out.offset, ram_out.size, pc_relative, pc_reg)
        targets_to_evaluate.append(EvalTarget(ram_out, mem_map))

    return targets_to_evaluate, assignments


def get_register_outputs_and_stores(
    translation: Translation,
) -> tuple[list[Varnode], list[PcodeOp], list[Varnode]]:
    outputs: list[Varnode] = []
    store_ops: list[PcodeOp] = []
    ram_outputs: list[Varnode] = []
    _seen_ram: set[tuple[int, int]] = set()
    for op in translation.ops:
        if op.output and op.output.space.name == 'register':
            outputs.append(op.output)
        elif op.output and op.output.space.name == 'ram':
            # Direct absolute / PC-relative memory WRITE: SLEIGH folds a
            # compile-time-known address into a ram-space OUTPUT varnode on any
            # op (e.g. `mov [rip+d],al` lifts as COPY out=ram[addr]), not a
            # STORE.  Record it as a constant-address memory target.
            key = (op.output.offset, op.output.size)
            if key not in _seen_ram:
                _seen_ram.add(key)
                ram_outputs.append(op.output)
        if op.opcode.name == 'STORE':
            store_ops.append(op)
    return outputs, store_ops, ram_outputs
