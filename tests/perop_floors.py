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
    A sound (coarse) carry-propagation floor: a tainted bit at position p can
    affect every output bit >= p through carry.  Kept only as a fallback."""
    p = _lowest_set(m)
    if p < 0:
        return 0
    return (~((1 << p) - 1)) & om


def _add_carry_full(a_v, a_t, b_v, b_t, width, cin_v=0, cin_t=0):
    """EXACT per-bit taint of a + b + carry_in over `width` bits, value-aware.

    Ripples the carry low-to-high carrying (value, taint) for each carry bit.  A
    sum bit is tainted iff one of {a_i, b_i, carry_i} is tainted; a carry-out is
    tainted iff, over the tainted inputs' 0/1 range, the majority can flip.  This
    is exact (matches per-bit ground truth) and O(width) -- `add x0,x1,#0` yields
    exactly x1's taint (no carry to smear); `add rax,rbx` recovers the interior
    carry bits the bare two-corner differential misses.  sub/2comp map on via
    a + ~b + 1.

    Returns (sum_taint, carry_out_taint, carry_into_msb_taint) so the flag ops
    read their exact taint off the SAME ripple: INT_CARRY = carry_out_taint,
    INT_SCARRY (signed overflow = carry_in_msb XOR carry_out_msb) = the OR of the
    two carry taints."""
    out_t = 0
    cv, ct = cin_v & 1, cin_t & 1
    cin_msb_t = 0
    for i in range(width):
        av = (a_v >> i) & 1
        at = (a_t >> i) & 1
        bv = (b_v >> i) & 1
        bt = (b_t >> i) & 1
        if at | bt | ct:
            out_t |= 1 << i
        if i == width - 1:
            cin_msb_t = ct
        lo = (0 if at else av) + (0 if bt else bv) + (0 if ct else cv)
        hi = (1 if at else av) + (1 if bt else bv) + (1 if ct else cv)
        cout_v = 1 if (av + bv + cv) >= 2 else 0
        cout_t = 1 if (1 if lo >= 2 else 0) != (1 if hi >= 2 else 0) else 0
        cv, ct = cout_v, cout_t
    return out_t, ct, cin_msb_t


def _add_carry_taint(a_v, a_t, b_v, b_t, width, cin_v=0, cin_t=0) -> int:
    return _add_carry_full(a_v, a_t, b_v, b_t, width, cin_v, cin_t)[0]


def _is_const(vn) -> bool:
    return vn.space.name == 'const'


#: A varnode's identity: (space name, offset, size).
VId = tuple[str, int, int]


def _vid(vn) -> VId:
    return (vn.space.name, vn.offset, vn.size)


def _reconv_contaminated(ops):
    """Per-VARNODE reconvergence contamination (the tunable window at its natural
    integration granularity: per output slice, not per whole instruction).

    A varnode is contaminated if its dependency cone contains an op whose two
    dynamic inputs share a register source (reconvergence) -- there the per-op
    differential over-taints, so that OUTPUT alone must widen to the monolithic
    differential.  A CLEAN output (cone free of reconvergence) keeps the per-op
    fast path.  Resolved in program order (SSA-style), so add's INT_CARRY reading
    the original RAX is not confused with INT_ADD's written RAX.

    Returns the set of contaminated varnode ids.  (Control flow is handled
    separately: an intra-instruction CBRANCH contaminates everything.)"""
    reg_anc: dict[VId, frozenset[VId]] = {}
    contaminated: set[VId] = set()
    for o in ops:
        in_ancs = []
        in_contam = False
        for i in o.inputs:
            if i.space.name == 'const':
                in_ancs.append(frozenset())
                continue
            vid = _vid(i)
            if vid in reg_anc:
                in_ancs.append(reg_anc[vid])
                if vid in contaminated:
                    in_contam = True
            elif i.space.name == 'register':
                in_ancs.append(frozenset({vid}))
            else:
                in_ancs.append(frozenset())
        dyn = [k for k, i in enumerate(o.inputs) if i.space.name != 'const']
        vids = [_vid(o.inputs[k]) for k in dyn]
        this_reconv = len(vids) != len(set(vids))
        if not this_reconv:
            for a in range(len(dyn)):
                for b in range(a + 1, len(dyn)):
                    if in_ancs[dyn[a]] & in_ancs[dyn[b]]:
                        this_reconv = True
                        break
                if this_reconv:
                    break
        if o.output is not None:
            ovid = _vid(o.output)
            union: set[VId] = set()
            for ia in in_ancs:
                union |= ia
            reg_anc[ovid] = frozenset(union)
            if this_reconv or in_contam:
                contaminated.add(ovid)
            else:
                contaminated.discard(ovid)  # a fresh clean definition clears it
    return contaminated


def _is_reconvergent(ops) -> bool:
    """True if some op's two dynamic inputs share a register ancestor (or repeat).

    That is the reconvergence the design flags: a tainted source reaches one op
    along >=2 paths, so perturbing the intermediate varnodes INDEPENDENTLY (the
    per-op model) explores impossible states and OVER-taints (r=(a+b)-a; xor
    rax,rax; lea/address math).  The whole-instruction differential perturbs the
    ORIGINAL sources once and stays exact, so these route to the monolithic
    window.  Normal add/sub/cmp read DISTINCT sources -> not reconvergent ->
    stay on the per-op fast path.  Conservative (a repeated/shared input into an
    idempotent and/or is safe but still routed to monolithic): never misses an
    over-taint, at a small fallback cost."""
    # Resolve register-leaf ancestors in PROGRAM ORDER (SSA-style): a varnode's
    # ancestors are those of its most recent definition so far; a register with no
    # earlier definition is its own leaf.  (A naive global producer map is wrong:
    # an instruction often writes a register mid-stream while earlier ops read its
    # original value -- e.g. add writes RAX after INT_CARRY/INT_SCARRY read it.)
    reg_anc: dict[VId, frozenset[VId]] = {}   # vid -> register-leaf ancestors
    for o in ops:
        in_ancs = []
        for i in o.inputs:
            if i.space.name == 'const':
                in_ancs.append(frozenset())
                continue
            vid = _vid(i)
            if vid in reg_anc:
                in_ancs.append(reg_anc[vid])
            elif i.space.name == 'register':
                in_ancs.append(frozenset({vid}))
            else:
                in_ancs.append(frozenset())
        dyn = [k for k, i in enumerate(o.inputs) if i.space.name != 'const']
        vids = [_vid(o.inputs[k]) for k in dyn]
        if len(vids) != len(set(vids)):
            return True  # an input used twice (xor rax,rax; sub rax,rax)
        for a in range(len(dyn)):
            for b in range(a + 1, len(dyn)):
                if in_ancs[dyn[a]] & in_ancs[dyn[b]]:
                    return True  # two inputs share a register source
        if o.output is not None:
            union: set[VId] = set()
            for ia in in_ancs:
                union |= ia
            reg_anc[_vid(o.output)] = frozenset(union)
    return False


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
        self.val: dict[tuple[str, int], int] = {}    # (space, offset) -> byte
        self.taint: dict[tuple[str, int], int] = {}  # (space, offset) -> taint byte

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

        # ---- transportable: EXACT value-aware carry ripple ----
        if name == 'INT_ADD':
            return _add_carry_taint(a_v, a_t, b_v, b_t, 8 * o_sz) & om
        if name == 'INT_SUB':
            # a - b = a + ~b + 1 ; ~b has the same per-bit taint as b.
            return _add_carry_taint(a_v, a_t, (~b_v) & om, b_t, 8 * o_sz, 1, 0) & om
        if name == 'INT_2COMP':
            # -a = ~a + 1 = 0 - a
            return _add_carry_taint(0, 0, (~a_v) & om, a_t, 8 * o_sz, 1, 0) & om
        if name in _TRANSPORTABLE:  # PTRADD / PTRSUB (rare): sound carry-smear
            m = a_t | b_t
            return (_smear_up(m, om) | self._local_diff(name, op, in_v, in_t)) & om

        # ---- avalanche ----
        if name == 'POPCOUNT':
            # result in [0, input_bits] -> only the low ceil(log2(bits+1)) bits vary.
            if not any(in_t):
                return 0
            nbits = (8 * isz).bit_length()
            return ((1 << nbits) - 1) & om
        if name in _AVALANCHE:
            return om if any(in_t) else 0

        # ---- carry / signed-overflow flags: EXACT via the same ripple ----
        if name == 'INT_CARRY':          # unsigned carry-out of a + b
            return _add_carry_full(a_v, a_t, b_v, b_t, 8 * isz)[1]
        if name == 'INT_SCARRY':         # signed overflow of a + b
            _o, cout_t, cin_t = _add_carry_full(a_v, a_t, b_v, b_t, 8 * isz)
            return cout_t | cin_t
        if name == 'INT_SBORROW':        # signed overflow of a - b (a + ~b + 1)
            _o, cout_t, cin_t = _add_carry_full(a_v, a_t, (~b_v) & _mask(isz),
                                                b_t, 8 * isz, 1, 0)
            return cout_t | cin_t

        # ---- 1-bit predicates (compares): coarse-but-sound floor ----
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

    def run(self, ops, strict=True) -> None:
        # Intra-instruction control flow (cmov, rep, string ops) predicates
        # later writes; a straight-line per-op pass would execute the taken side
        # unconditionally and drop the not-taken side's taint (under-taint).
        # Route these to the monolithic window (the sound oracle).  This bails
        # even in slice-wise mode: a branch contaminates every later slice.
        for op in ops:
            if op.opcode.name in ('CBRANCH', 'BRANCHIND', 'CALLIND', 'CALLOTHER'):
                raise NeedsMonolithic(op.opcode.name)
        # Whole-instruction window (strict): any reconvergence -> monolithic.
        # Slice-wise callers pass strict=False and instead trust only the outputs
        # that _reconv_contaminated reports clean.
        if strict and _is_reconvergent(ops):
            raise NeedsMonolithic('reconvergence')
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


def perop_floors_slicewise(arch, code, regs, in_taint, in_values):
    """The tunable window at OUTPUT-SLICE granularity: run the per-op pass over
    the whole instruction, then trust the per-op taint only for outputs whose
    dependency cone is reconvergence-free; the rest (contaminated slices) would
    widen to the monolithic differential in the engine.

    Returns (taint_by_reg, clean_regnames): taint_by_reg holds per-op taint for
    every mapped register; clean_regnames is the subset that is per-op-eligible
    (cone free of reconvergence).  Raises NeedsMonolithic only for
    intra-instruction control flow (which contaminates the whole instruction).
    This measures the REAL fast-path coverage: a reconvergent flag no longer
    disqualifies a clean result register."""
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
    interp.run(ops, strict=False)          # may raise NeedsMonolithic on control flow
    contaminated = _reconv_contaminated(ops)
    taint = {name: interp.out_taint(vn) for name, vn in mapped}
    clean = {name for name, vn in mapped if _vid(vn) not in contaminated}
    return taint, clean


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
