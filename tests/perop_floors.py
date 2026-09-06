"""Per-op differential WITH FLOORS (Phase 3b precision/soundness study).

This is the GO/NO-GO experiment for the eval-core rework
([[differential-compaction-design]]).  Phase 3a proved the BARE two-corner
differential under-taints (add misses the carry bit + OF; and misses PF) because
two extremal corners coincide on interior bit flips.  The engine's oracle
(circuit.evaluate) is the differential PLUS per-category soundness FLOORS.  This
module implements those floors AT PER-P-CODE-OP granularity and answers:

  * SOUND vs Unicorn per-bit ground truth?  (non-negotiable: zero under-taint)
  * how close to the oracle vs how much over-taint?  (the precision cost of
    per-op compaction; the reconvergence window recovers the rest)

Design (one forward pass over the lifted p-code, carrying (value, taint) per
byte so register aliasing is exact):

  AFFINE ops (operand-const-aware) -> EXACT taint routing (mask/shift/xor
    composition), no differential, no floor.  COPY/ZEXT/SEXT/TRUNC/SUBPIECE/
    PIECE/NEGATE always; XOR always (OR of input taints, exact when inputs
    independent); AND/OR with a CONST operand (masking); LEFT/RIGHT/SRIGHT by
    CONST; MULT by const pow2.

  NON-AFFINE ops -> local two-corner differential over THIS op's concrete inputs
    OR'd with a provably-sound per-op FLOOR:
      TRANSPORTABLE (ADD/SUB/2COMP/PTRADD/PTRSUB): carry-smear -- a tainted input
        bit can carry to every higher output bit, so taint from the lowest
        tainted bit upward.  Exact for add/sub (carry only moves up).
      AVALANCHE (MULT/DIV/REM/POPCOUNT/LZCOUNT/FLOAT_*): any tainted input ->
        whole output tainted.
      var-shift (LEFT/RIGHT/SRIGHT by a variable): tainted count -> whole output;
        else route by the concrete count (affine).
      value-aware AND/OR of two variables: bit i tainted iff a flip of a tainted
        input bit could flip out_i given the other operand's concrete bit.
      1-bit predicates (compares, CARRY/SCARRY/SBORROW, BOOL_*): any tainted
        relevant input bit -> tainted.

The FLOORS here are sound over-approximations of the true per-bit sensitivity;
they are typically == the oracle and never under it vs ground truth.  Where they
differ from the oracle we MEASURE it (tighter-than-oracle is a precision GAIN and
still sound; looser is the documented per-op over-taint recovered by the window).

Register/flag corpus over LE ISAs (AMD64/ARM64/RISCV64), matching the 3a study;
memory + big-endian come with the corpus extension.  Standalone (pypcode only) --
no engine coupling, so this cannot regress the live path.
"""
# ruff: noqa: PLC0415
# mypy: disable-error-code="no-untyped-def,no-untyped-call,attr-defined,import-untyped,var-annotated"
from __future__ import annotations

from microtaint.sleigh.lifter import get_context
from microtaint.types import Architecture

from tests.perop_prototype import Unsupported, _apply, _mask, _signed

MASK64 = 0xFFFFFFFFFFFFFFFF


class NeedsMonolithic(Unsupported):
    """Intra-instruction control flow or reconvergence the per-op window cannot
    resolve soundly (cmov, rep, predicated writes, xor-self cancellation).  The
    design's escape hatch: widen the differential window to the whole instruction
    -- fall back to circuit.evaluate (the oracle, sound by construction).  Raised
    (subclass of Unsupported) so callers route to the monolithic path rather than
    under-taint on a branch's not-taken side."""

_ARCH_LE = {'AMD64': True, 'ARM64': True, 'RISCV64': True,
            'MIPS64BE': False, 'PPC32BE': False}

# Opcodes skipped (no taint effect / handled elsewhere) and control flow.
_SKIP = {'IMARK', 'INDIRECT', 'MULTIEQUAL', 'CPOOLREF', 'NEW', 'CAST', 'SEGMENTOP'}
_CTRL = {'BRANCH', 'BRANCHIND', 'CBRANCH', 'CALL', 'CALLIND', 'RETURN'}

# Always-affine (routing) opcodes regardless of operands.
_ROUTE_ALWAYS = {'COPY', 'INT_ZEXT', 'INT_SEXT', 'INT_TRUNC', 'SUBPIECE',
                 'PIECE', 'INT_NEGATE', 'EXTRACT', 'INSERT'}

# Transportable (carry-coupled) ops: carry-smear floor.
_TRANSPORTABLE = {'INT_ADD', 'INT_SUB', 'INT_2COMP', 'PTRADD', 'PTRSUB'}

# Avalanche ops: any tainted input taints the whole output.
_AVALANCHE = {'INT_MULT', 'INT_DIV', 'INT_SDIV', 'INT_REM', 'INT_SREM',
              'POPCOUNT', 'LZCOUNT'}

# 1-bit predicate ops: any tainted relevant input -> tainted output bit.
_PRED1 = {'INT_EQUAL', 'INT_NOTEQUAL', 'INT_LESS', 'INT_LESSEQUAL',
          'INT_SLESS', 'INT_SLESSEQUAL', 'INT_CARRY', 'INT_SCARRY',
          'INT_SBORROW'}


def _lowest_set(m: int) -> int:
    """Position of the lowest set bit of m, or -1 if m == 0."""
    if m == 0:
        return -1
    return (m & -m).bit_length() - 1


def _smear_up(m: int, om: int) -> int:
    """All bits from the lowest set bit of m upward, capped to output mask om.
    The sound carry-propagation floor: a tainted bit at position p can affect
    every output bit >= p through carry."""
    p = _lowest_set(m)
    if p < 0:
        return 0
    return (~((1 << p) - 1)) & om


def _is_const(vn) -> bool:
    return vn.space.name == 'const'


# Human/bank register name -> pypcode SLEIGH register name, where they differ.
# pypcode is the ISA-general contract, but its register *spelling* is
# Ghidra's (lowercase GP on AArch64, NZCV split as NG/ZR/CY/OV), so the study
# resolves bank names through this alias table + a case-insensitive fallback.
_REG_ALIASES = {
    # AArch64 condition flags (bank N/Z/C/V -> Ghidra NG/ZR/CY/OV)
    'N': 'NG', 'Z': 'ZR', 'C': 'CY', 'V': 'OV',
}


def _resolve_vn(reg_vn, name):
    """Map a bank/human register name to its pypcode varnode, tolerating
    case and the AArch64 flag spelling.  Returns None if unmappable."""
    vn = reg_vn.get(name)
    if vn is not None:
        return vn
    alias = _REG_ALIASES.get(name)
    if alias is not None and alias in reg_vn:
        return reg_vn[alias]
    lo = name.lower()
    if lo in reg_vn:
        return reg_vn[lo]
    up = name.upper()
    if up in reg_vn:
        return reg_vn[up]
    return None


class PerOpFloors:
    """Per-op taint with sound floors.  Byte-granular value+taint stores so
    overlapping registers (AL/AX/EAX/RAX) alias correctly."""

    def __init__(self, ctx, little_endian: bool):
        self.ctx = ctx
        self.le = little_endian
        self.val: dict = {}    # (space, offset) -> concrete byte 0..255
        self.taint: dict = {}  # (space, offset) -> taint byte 0..255

    # -- byte-wise varnode access (mirrors perop_prototype for aliasing) --
    def _rd_val(self, vn) -> int:
        if vn.space.name == 'const':
            return vn.offset & _mask(vn.size)
        return self._gather(self.val, vn)

    def _rd_taint(self, vn) -> int:
        if vn.space.name == 'const':
            return 0
        return self._gather(self.taint, vn)

    def _gather(self, store, vn) -> int:
        v = 0
        for i in range(vn.size):
            byte = store.get((vn.space.name, vn.offset + i), 0)
            sh = i if self.le else (vn.size - 1 - i)
            v |= byte << (8 * sh)
        return v

    def _scatter(self, store, vn, val) -> None:
        for i in range(vn.size):
            sh = i if self.le else (vn.size - 1 - i)
            store[(vn.space.name, vn.offset + i)] = (val >> (8 * sh)) & 0xFF

    def init_reg(self, vn, value: int, taint: int) -> None:
        m = _mask(vn.size)
        self._scatter(self.val, vn, value & m)
        self._scatter(self.taint, vn, taint & m)

    # -- the per-op taint rule -------------------------------------------
    def _op_taint(self, name, op, in_v, in_t) -> int:
        o_sz = op.output.size
        om = _mask(o_sz)
        isz = op.inputs[0].size if op.inputs else o_sz
        a_v = in_v[0] if in_v else 0
        a_t = in_t[0] if in_t else 0
        b_v = in_v[1] if len(in_v) > 1 else 0
        b_t = in_t[1] if len(in_t) > 1 else 0
        b_const = len(op.inputs) > 1 and _is_const(op.inputs[1])

        # ---- always-affine routing (exact) ----
        if name == 'COPY':
            return a_t & om
        if name == 'INT_ZEXT':
            return a_t & om
        if name == 'INT_SEXT':
            # sign-replicate the top taint bit into the extension.
            t = a_t & _mask(isz)
            if (t >> (8 * isz - 1)) & 1:
                t |= (~_mask(isz)) & om
            return t & om
        if name == 'INT_TRUNC':
            return a_t & om
        if name == 'SUBPIECE':
            shift = 8 * (op.inputs[1].offset if b_const else b_v)
            return (a_t >> shift) & om
        if name == 'PIECE':
            lo_bits = 8 * op.inputs[1].size
            return ((a_t << lo_bits) | (b_t & _mask(op.inputs[1].size))) & om
        if name == 'INT_NEGATE':
            return a_t & om
        if name == 'INT_XOR':
            # exact when inputs independent; OR of taints (reconvergence widens).
            return (a_t | b_t) & om
        if name == 'BOOL_XOR':
            return (a_t | b_t) & 1
        if name == 'BOOL_NEGATE':
            return a_t & 1

        # ---- operand-const-aware affine ----
        if name == 'INT_AND':
            if b_const:
                return (a_t & b_v) & om
            if len(op.inputs) > 1 and _is_const(op.inputs[0]):
                return (b_t & a_v) & om
            # two variables: value-aware (bit i tainted iff a flip could flip out)
            return ((a_t & b_v) | (b_t & a_v) | (a_t & b_t)) & om
        if name == 'INT_OR':
            if b_const:
                return (a_t & ~b_v) & om
            if len(op.inputs) > 1 and _is_const(op.inputs[0]):
                return (b_t & ~a_v) & om
            return ((a_t & ~b_v) | (b_t & ~a_v) | (a_t & b_t)) & om
        if name == 'BOOL_AND':
            return ((a_t & b_v) | (b_t & a_v) | (a_t & b_t)) & 1
        if name == 'BOOL_OR':
            return ((a_t & ~b_v) | (b_t & ~a_v) | (a_t & b_t)) & 1

        if name in ('INT_LEFT', 'INT_RIGHT', 'INT_SRIGHT'):
            if b_const or b_t == 0:
                k = b_v
                if name == 'INT_LEFT':
                    return (a_t << k) & om if k < 128 else 0
                if name == 'INT_RIGHT':
                    return (a_t >> k) & om
                # SRIGHT: arithmetic; replicate sign taint into vacated top.
                t = (a_t >> k) if k < 8 * isz else 0
                if (a_t >> (8 * isz - 1)) & 1:  # sign taint fills top k bits
                    fill = (~_mask(isz)) & om  # bits above isz (if widened) -- usually none
                    top = (~((1 << max(0, 8 * isz - k)) - 1)) & _mask(isz)
                    t |= top | fill
                return t & om
            # variable, tainted count -> whole output can change.
            return om

        if name == 'INT_MULT':
            # affine only for a const power-of-two operand (== shift).
            if b_const and b_v != 0 and (b_v & (b_v - 1)) == 0:
                k = b_v.bit_length() - 1
                return (a_t << k) & om
            if (len(op.inputs) > 1 and _is_const(op.inputs[0])
                    and a_v != 0 and (a_v & (a_v - 1)) == 0):
                k = a_v.bit_length() - 1
                return (b_t << k) & om
            # general multiply: avalanche.
            return om if (a_t | b_t) else 0

        # ---- transportable: carry-smear floor | local differential ----
        if name in _TRANSPORTABLE:
            if name == 'INT_2COMP':
                m = a_t
            else:
                m = a_t | b_t
            floor = _smear_up(m, om)
            return (floor | self._local_diff(name, op, in_v, in_t)) & om

        # ---- avalanche ----
        if name in _AVALANCHE:
            return om if any(in_t) else 0

        # ---- 1-bit predicates ----
        if name in _PRED1:
            return 1 if any(in_t) else 0

        # ---- div/rem handled by _AVALANCHE above; anything else opaque ----
        raise Unsupported(name)

    def _local_diff(self, name, op, in_v, in_t) -> int:
        """Bare two-corner differential over THIS op's concrete inputs.  Tightens
        the floor where the two extremal corners actually diverge."""
        try:
            hi = _apply(name, op, [(v | t) for v, t in zip(in_v, in_t)])
            lo = _apply(name, op, [(v & ~t) for v, t in zip(in_v, in_t)])
        except Unsupported:
            return _mask(op.output.size)
        return (hi ^ lo) & _mask(op.output.size)

    def run(self, ops) -> None:
        # Intra-instruction control flow (cmov, rep, string ops) predicates
        # later writes; a straight-line per-op pass would execute the taken side
        # unconditionally and drop the not-taken side's taint (under-taint).
        # Route these to the monolithic window (the sound oracle).
        for op in ops:
            if op.opcode.name in ('CBRANCH', 'BRANCHIND', 'CALLIND', 'CALLOTHER'):
                raise NeedsMonolithic(op.opcode.name)
        for op in ops:
            name = op.opcode.name
            if name in _SKIP or name in _CTRL:
                continue
            if op.output is None:
                if name in ('STORE', 'LOAD', 'CALLOTHER'):
                    raise Unsupported(name)
                continue
            if name in ('LOAD', 'STORE', 'CALLOTHER'):
                raise Unsupported(name)
            in_v = [self._rd_val(v) for v in op.inputs]
            in_t = [self._rd_taint(v) for v in op.inputs]
            out_v = _apply(name, op, in_v)
            out_t = self._op_taint(name, op, in_v, in_t)
            self._scatter(self.val, op.output, out_v)
            self._scatter(self.taint, op.output, out_t)

    def out_taint(self, vn) -> int:
        return self._gather(self.taint, vn) & _mask(vn.size)


def _arch_key(arch) -> str:
    return arch.value if hasattr(arch, 'value') else str(arch)


def perop_floors_taint(arch, code, regs, in_taint, in_values):
    """Run the per-op-with-floors interpreter; return output taint over the
    mappable bank registers.  Raises Unsupported for BE / mem / opaque ops."""
    key = _arch_key(arch)
    if not _ARCH_LE.get(key, False):
        raise Unsupported(f'BE arch {key}')
    ctx = get_context(key)
    reg_vn = ctx.registers
    interp = PerOpFloors(ctx, _ARCH_LE[key])
    mapped = []
    for r in regs:
        vn = _resolve_vn(reg_vn, r.name)
        if vn is None:
            continue
        mapped.append((r.name, vn))
        interp.init_reg(vn, in_values.get(r.name, 0), in_taint.get(r.name, 0))
    ops = ctx.translate(code, 0x1000).ops
    interp.run(ops)
    return {name: interp.out_taint(vn) for name, vn in mapped}


def engine_perop_floors(arch, code, regs, in_taint, in_values, *, circuit=None):
    """Oracle-harness engine adapter for per-op-with-floors."""
    return perop_floors_taint(arch, code, regs, in_taint, in_values)


if __name__ == '__main__':
    from tests.oracle_harness import classify, reference_taint
    from benchmark.instruction_bank import isa_registers
    from microtaint.types import Architecture as A
    regs = list(isa_registers('AMD64'))
    rn = [r.name for r in regs]
    cases = [
        ('add rax, rbx', bytes.fromhex('4801d8')),
        ('sub rax, rbx', bytes.fromhex('4829d8')),
        ('xor rax, rax', bytes.fromhex('4831c0')),
        ('and rax, rbx', bytes.fromhex('4821d8')),
        ('or  rax, rbx', bytes.fromhex('4809d8')),
        ('imul rax, rbx', bytes.fromhex('480fafc3')),
        ('shl rax, 4', bytes.fromhex('48c1e004')),
        ('mov rax, rbx', bytes.fromhex('4889d8')),
    ]
    for label, code in cases:
        it = {r: (MASK64 if r in ('RAX', 'RBX') else 0) for r in rn}
        iv = {r: 0x1234 * (i + 1) for i, r in enumerate(rn)}
        try:
            got = engine_perop_floors(A.AMD64, code, regs, it, iv)
        except Unsupported as e:
            print(f'{label:16s}: UNSUPPORTED {e}'); continue
        ref = reference_taint(A.AMD64, code, regs, it, iv)
        v = classify(got, ref, [r for r in rn if r in got])
        print(f'{label:16s}: exact={v.exact} under={v.under} over={v.over}')
