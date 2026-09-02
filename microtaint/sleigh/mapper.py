from __future__ import annotations

from pypcode import PcodeOp, Varnode

from microtaint.classifier.categories import InstructionCategory
from microtaint.sleigh.constfold import const_value, fold_constants

AVALANCHE_OPCODES: set[str] = {
    'INT_MULT',
    'INT_DIV',
    'INT_SDIV',
    'INT_REM',
    'INT_SREM',
    'POPCOUNT',
    'LZCOUNT',
    # ZPULL / SPULL: pypcode 4 (Ghidra 11.x) bit-range extract ops (unsigned /
    # signed).  Not emitted by the lifters for any supported ISA yet, so this is a
    # defensive categorisation; avalanche is the sound default (over-taint) until
    # a precise routing model with native cell support is needed.
    'ZPULL',
    'SPULL',
    'FLOAT_ADD',
    'FLOAT_SUB',
    'FLOAT_MULT',
    'FLOAT_DIV',
    'FLOAT_ABS',
    'FLOAT_SQRT',
    'FLOAT_CEIL',
    'FLOAT_FLOOR',
    'FLOAT_ROUND',
    'FLOAT_TRUNC',
    'FLOAT_NEG',
    'FLOAT_INT2FLOAT',
    'FLOAT_FLOAT2FLOAT',
}

TRANSLATABLE_OPCODES: set[str] = {
    'INT_LEFT',
    'INT_RIGHT',
    'INT_SRIGHT',
}

TRANSPORTABLE_OPCODES: set[str] = {
    'INT_ADD',
    'INT_SUB',
    'INT_2COMP',
    'PTRADD',
    'PTRSUB',
}

COND_TRANSPORTABLE_OPCODES: set[str] = {
    'INT_EQUAL',
    'INT_NOTEQUAL',
    'FLOAT_EQUAL',
    'FLOAT_NOTEQUAL',
    'FLOAT_LESS',
    'FLOAT_LESSEQUAL',
    'FLOAT_NAN',
}

MONOTONIC_OPCODES: set[str] = {
    'INT_LESS',
    'INT_LESSEQUAL',
    'INT_SLESS',
    'INT_SLESSEQUAL',
    'INT_CARRY',
    'INT_SCARRY',
    'INT_SBORROW',
    'INT_AND',
    'INT_OR',
    'INT_NEGATE',
    'BOOL_AND',
    'BOOL_OR',
    'BOOL_NEGATE',
}

ROUTING_OPCODES: set[str] = {
    'INT_LEFT',
    'INT_RIGHT',
    'INT_SRIGHT',
    'COPY',
    'INT_ZEXT',
    'INT_SEXT',
    'SUBPIECE',
    'PIECE',
    'EXTRACT',
    'INSERT',
    'LOAD',
    'INT_AND',
    'INT_OR',
    'BOOL_AND',
    'BOOL_OR',
}

ORABLE_OPCODES: set[str] = {
    'INT_XOR',
    'BOOL_XOR',
}

MAPPED_OPCODES: set[str] = {
    'LOAD',
    'STORE',
}

EXTENSION_OPCODES: set[str] = {
    'INT_ZEXT',
    'INT_SEXT',
    'SUBPIECE',
    'PIECE',
    'COPY',
    'EXTRACT',
    'INSERT',
}

CONTROL_FLOW_OPCODES: set[str] = {
    'BRANCH',
    'BRANCHIND',
    'CBRANCH',
    'CALL',
    'CALLIND',
    'RETURN',
}

# Opaque operations SLEIGH cannot decompose into modelled primitives.  P-code
# emits CALLOTHER for two kinds of instruction, neither of which is control
# transfer (real jumps/calls use BRANCH/CALL/RETURN):
#   * opaque vector / crypto ops — pshufb, psadbw, pmaddwd, vpaddb, vpshufb,
#     aesenc, crc32, ...
#   * opaque system ops — syscall, cpuid, rdtsc, int3, ud2, ...
# Their effect on the output is unknown, so any slice touching one must be
# AVALANCHE (taint the whole output whenever any input is tainted) — the sound
# over-approximation, matching the paper's opcode table.  These must NOT be
# ignored by the core_ops filter (that would leave core_ops empty and
# mis-classify the slice as MAPPED -> bare differential -> under-taint on shared
# tainted lanes, e.g. pshufb).
OPAQUE_OPCODES: set[str] = {
    'CALLOTHER',
}

IGNORED_OPCODES: set[str] = {
    *CONTROL_FLOW_OPCODES,
    'IMARK',
    'INDIRECT',
    'MULTIEQUAL',
    'CPOOLREF',
    'NEW',
    'CAST',
    'SEGMENTOP',
}


def _producing_op(core_ops: list[PcodeOp]) -> PcodeOp | None:
    """The op that actually produces the slice's output."""
    for op in reversed(core_ops):
        if op.opcode.name not in ('COPY', 'SUBPIECE', 'PIECE', 'INT_ZEXT', 'INT_SEXT'):
            return op
    return None


def _pow2_mult_shift(op: PcodeOp, folded: dict[tuple[str, int, int], int]) -> int | None:
    """log2 of a constant power-of-two INT_MULT multiplier, or None.

    `x * 2^k` IS a left shift by k: a fixed relocation of bit positions with no
    carry mixing, hence exact taint (T << k), not an avalanche.  x86 spells
    `lea` scales this way (`lea (,%rax,4),%rdx` lifts `rax*4` as INT_MULT).
    Mirrors partition._is_pow2_scale so both classifier and taint-expr agree.
    """
    if op.opcode.name != 'INT_MULT' or len(op.inputs) != 2:
        return None
    for i in (0, 1):
        cv = const_value(op.inputs[i], folded)
        other = const_value(op.inputs[1 - i], folded)
        if cv is not None and other is None and cv > 0 and (cv & (cv - 1)) == 0:
            return cv.bit_length() - 1
    return None


def is_mapped_permutation(slice_ops: list[PcodeOp]) -> bool:  # noqa: C901
    """
    Heuristic: A true permutation only uses routing/shifting opcodes
    AND relies on only ONE dynamic input (register/memory). All other inputs must be constants.

    Extended case: exactly 2 dynamic sources are allowed when one is a 1-bit register
    (e.g. CF at offset 0x200) and all ops are routing ops. This handles rotate-through-
    carry instructions (rcl/rcr) where the carry bit IS a genuine external fill source,
    not an intra-instruction intermediate. The differential correctly computes the
    per-bit permutation for these 2-source cases.

    Intra-instruction intermediate registers (written before being read within the slice,
    like CF in ``ror rax,1`` where CF is computed from bit0 of RAX and then used to fill
    bit63) are excluded from the dynamic-source count.
    """
    # A permutation needs a FIXED bit mapping.  A shift whose AMOUNT is data-derived
    # does not have one -- the mapping moves with the data -- so it is not a mapped
    # permutation however few sources it has.  MIPS `srlv $2,$4,$4` shifts a register
    # by ITSELF: one dynamic source, so this read as MAPPED and got a bare
    # differential, but `x >> (x & 31)` is non-monotone in x and under-tainted
    # 37/200.  Amounts that fold to a constant are still permutations.
    _folded_perm = fold_constants(slice_ops)
    for _op in slice_ops:
        if _op.opcode.name in TRANSLATABLE_OPCODES and len(_op.inputs) > 1:
            _amt = _op.inputs[1]
            if _amt.space.name != 'const' and (
                (_amt.space.name, _amt.offset, _amt.size) not in _folded_perm
            ):
                return False

    dynamic_sources: set[tuple[str, int, int]] = set()  # (space, offset, size)
    has_shift = False

    # Track registers written within this slice (intra-instruction intermediates)
    _slice_written: dict[int, int] = {}  # register offset → first write index in slice
    for _i, op in enumerate(slice_ops):
        if op.output is not None and op.output.space.name == 'register':
            if op.output.offset not in _slice_written:
                _slice_written[op.output.offset] = _i

    for _i, op in enumerate(slice_ops):
        # Check whether this op produces an intra-slice intermediate register
        # that is consumed by a later op (like CF in ror rax,1 computed by NOTEQUAL).
        # We skip the routing-op check ONLY for non-routing ops whose output is
        # a DIFFERENT register than the final architectural output AND is written
        # before being consumed by a later op.
        # INT_NOTEQUAL in ror writes CF (offset 0x200, different from RAX at 0) → skip OK.
        # INT_ADD in inc eax writes EAX (offset 0x0, same as RAX output at 0) → NOT skipped.
        #
        # RESTRICTION: only 1-bit registers (flag registers) qualify as intra-intermediates.
        # Wider registers like RSP (64-bit) that are arithmetically modified to compute
        # a memory address are NOT intermediates — they are address operands and their
        # modification must be treated as a non-routing op that breaks the permutation.
        # This prevents push/pop (INT_SUB writes RSP, LOAD reads RSP-8) from being
        # incorrectly classified as a permutation.
        _is_intra_intermediate = False
        if (
            op.opcode.name not in ROUTING_OPCODES
            and op.output is not None
            and op.output.space.name == 'register'
            and op.output.size == 1  # only flag-sized (1-bit) registers qualify
            and op.output.offset in _slice_written
            and _slice_written[op.output.offset] == _i
        ):
            # Get the final output register: the last op that writes to a register in the slice
            _final_out_offsets = {
                o.output.offset
                for o in slice_ops
                if (o.output is not None and o.output.space.name == 'register' and o is slice_ops[-1])
            }
            # Also accept: the output is to a register that is NOT the one we're
            # ultimately computing taint for. We detect this by checking whether
            # any later routing op reads this register as input.
            _consumed_by_routing = any(
                (
                    later_op.opcode.name in ROUTING_OPCODES
                    and any(v.space.name == 'register' and v.offset == op.output.offset for v in later_op.inputs)
                )
                for later_op in slice_ops[_i + 1 :]
            )
            # Only an intermediate if: consumed by routing op AND output register
            # is NOT the same as any routing op's output (i.e. not the main data path).
            if _consumed_by_routing:
                # Verify the intermediate register is not on the main data path
                # (i.e. the final routing output doesn't write to the same offset+size).
                _final_routing_writes = {
                    (o.output.offset, o.output.size)
                    for o in slice_ops
                    if (o.output is not None and o.output.space.name == 'register' and o.opcode.name in ROUTING_OPCODES)
                }
                if (op.output.offset, op.output.size) not in _final_routing_writes:
                    # Also check: is this offset covered by any final routing write
                    # of a different size (e.g. INT_ADD writes EAX offset 0 size 4,
                    # while INT_ZEXT writes RAX offset 0 size 8 — same register family)?
                    _offset_in_routing = any(roff == op.output.offset for roff, _ in _final_routing_writes)
                    if not _offset_in_routing:
                        _is_intra_intermediate = True

        if _is_intra_intermediate:
            # Skip routing check for this op. Collect its external dynamic sources.
            for vn in op.inputs:
                if vn.space.name not in ('const', 'unique'):
                    first_write = _slice_written.get(vn.offset, len(slice_ops))
                    if first_write >= _i:
                        dynamic_sources.add((vn.space.name, vn.offset, vn.size))
            continue

        if op.opcode.name not in ROUTING_OPCODES:
            return False

        if op.opcode.name in {'INT_LEFT', 'INT_RIGHT', 'INT_SRIGHT'}:
            has_shift = True

        for vn in op.inputs:
            if vn.space.name not in ('const', 'unique'):
                # Exclude intra-instruction intermediates: registers written earlier
                # in this slice (e.g. CF computed by a prior non-routing op).
                first_write = _slice_written.get(vn.offset, len(slice_ops))
                if first_write < _i:
                    continue
                dynamic_sources.add((vn.space.name, vn.offset, vn.size))

    if len(dynamic_sources) == 1:
        return True

    if len(dynamic_sources) == 2:
        # Allow 2-source permutation when one source is a *flag* register (e.g. CF
        # at Sleigh offset 0x200) and there IS a shift present (the carry bit is
        # shifted into a bit position).  This covers rcl/rcr where CF fills one bit
        # of the rotated result.
        #
        # Explicitly excluded: shift instructions like `shl rax, cl` where the 1-bit
        # source is CL (sub-byte of RCX, offset 0x8) — that is a shift *amount*, not
        # a fill bit, so it must remain TRANSLATABLE (variable-amount shift → avalanche).
        # We distinguish carry-fill bits from shift-amount bits by checking whether the
        # 1-bit source lives at a known x86 flag register offset.
        _X86_FLAG_OFFSETS = frozenset({0x200, 0x20B, 0x206, 0x207, 0x202, 0x203})
        one_bit_sources = [(sp, off, sz) for sp, off, sz in dynamic_sources if sz == 1]
        multi_bit_sources = [(sp, off, sz) for sp, off, sz in dynamic_sources if sz > 1]
        if (
            len(one_bit_sources) == 1
            and len(multi_bit_sources) == 1
            and has_shift
            and one_bit_sources[0][1] in _X86_FLAG_OFFSETS
        ):
            return True

    return False


def determine_category(  # noqa: C901
    slice_ops: list[PcodeOp],
    out_width_bits: int = 64,
) -> InstructionCategory:
    if not slice_ops:
        return InstructionCategory.MAPPED

    # Opaque, undecomposable ops (CALLOTHER: pshufb/psadbw/aesenc/syscall/...)
    # dominate the whole slice — see OPAQUE_OPCODES.  Checked first, before the
    # core_ops filter that would otherwise drop them.
    if any(op.opcode.name in OPAQUE_OPCODES for op in slice_ops):
        return InstructionCategory.AVALANCHE

    if is_mapped_permutation(slice_ops):
        return InstructionCategory.MAPPED

    # Safely filter out ignored and extension operations from the core evaluation
    core_ops = [
        op for op in slice_ops if op.opcode.name not in EXTENSION_OPCODES and op.opcode.name not in IGNORED_OPCODES
    ]

    # If all operations were just simple routing, copies, or ignored metadata
    if not core_ops:
        return InstructionCategory.MAPPED

    # AVALANCHE fires on opcode PRESENCE, which is wrong for a multiply by a
    # CONSTANT: `lea rax,[rbx+rcx*4+8]` lifts `rcx*4` as INT_MULT, and that alone
    # avalanched the whole address computation (7.3x invented bits, 1.6% exact)
    # even though its producing op is an INT_ADD.  Multiplying by a constant is
    # affine -- a sum of fixed shifts -- so each output bit still depends on a
    # fixed set of input bit positions and the ordinary carry-coupled regime
    # applies.  Only a data x data multiply is a genuine avalanche.
    # A bare power-of-two scale is a pure linear shift, not a carry-mixing
    # multiply: `lea (,%rax,4),%rdx` lifts as `rdx = rax * 4`, i.e. rax << 2.
    # When that scale is the op that PRODUCES the output, the slice is exactly
    # linear, so MAPPED gives it exact taint (T << k) via the linear projection.
    # (A scale feeding an INT_ADD is handled below: the adder owns the
    # carry-coupled regime and keeps TRANSPORTABLE.)  Non-pow2 constant
    # multiplies carry-mix and fall through to the avalanche guard.  Provably
    # sound: the differential of x << k is exactly T << k.
    _prod = _producing_op(core_ops)
    if _prod is not None and _pow2_mult_shift(_prod, fold_constants(slice_ops)) is not None:
        return InstructionCategory.MAPPED

    _folded_av = fold_constants(slice_ops)
    for op in core_ops:
        if op.opcode.name not in AVALANCHE_OPCODES:
            continue
        if op.opcode.name == 'INT_MULT' and op is not _producing_op(core_ops):
            _data = [
                i
                for i in op.inputs
                if i.space.name != 'const'
                and (i.space.name, i.offset, i.size) not in _folded_av
            ]
            if len(_data) < 2:
                continue  # affine scaling feeding some other producer
        return InstructionCategory.AVALANCHE

    for op in core_ops:
        if op.opcode.name in COND_TRANSPORTABLE_OPCODES:
            return InstructionCategory.COND_TRANSPORTABLE

    # TRANSLATABLE only when the shift IS the operation that PRODUCES the output.
    #
    # This used to fire on the mere PRESENCE of a shift anywhere in the slice, which
    # silently downgraded any slice with a pre-processed operand out of
    # TRANSPORTABLE -- and TRANSPORTABLE is what carries the union floor covering
    # the carry/borrow chain.  The decisive evidence was PPC:
    #
    #   addi  3,4,1365   ->  r3 = r4 + 0x555                  TRANSPORTABLE, sound
    #   addis 3,4,1365   ->  u = 0x555 << 0x10 ; r3 = r4 + u  TRANSLATABLE, 79 under-taints
    #
    # The shift there is on a CONSTANT and contributes nothing to the data flow, yet
    # it changed the classification.  Two things disqualify a shift from owning the
    # slice:
    #   * it only computes a CONSTANT (`addis`, and RISC-V's `64 - 1` shift mask), or
    #   * a CARRY-COUPLED op consumes its result -- ARM64's shifted-register
    #     (`add x0,x1,x2,lsl #3`) and extended-register (`add x0,x1,w2,uxtb`)
    #     operands -- in which case the adder produces the output and owns the
    #     regime; the shifter was an operand pre-processor.
    #
    # Deliberately NOT extended to bitwise consumers (`bic x0,x1,x2,lsl #1`).  Those
    # under-taint too, but a shift feeding an OR is also how a rotate, `extr` and
    # `ubfx` are lifted, and those are exact today -- so that case needs its own
    # discriminator and its own measurement rather than being folded in here.
    _folded = fold_constants(slice_ops)

    def _vkey(vn: Varnode) -> tuple[str, int, int]:
        return (vn.space.name, vn.offset, vn.size)

    _real_shifts = [
        op
        for op in core_ops
        if op.opcode.name in TRANSLATABLE_OPCODES
        and op.output is not None
        and _vkey(op.output) not in _folded
    ]

    def _reaches_shift(vn: Varnode, depth: int = 0) -> bool:
        if depth > 16 or vn.space.name in ('const', 'register'):
            return False
        for o in slice_ops:
            if o.output is None or not (
                o.output.space.name == vn.space.name
                and o.output.offset < vn.offset + vn.size
                and vn.offset < o.output.offset + o.output.size
            ):
                continue
            if o in _real_shifts:
                return True
            if any(_reaches_shift(i, depth + 1) for i in o.inputs):
                return True
        return False

    # Only a CARRY-COUPLED consumer takes ownership.  Ceding to an XOR consumer as
    # well does make `eor x0,x1,x2,ror #11` and `bic x0,x1,x2,lsl #1` sound, but at
    # a precision cost that is not acceptable: ORABLE's union drops them from 96% to
    # 43% and 19% exact.  Those two remain UNSOUND pending a fix that is exact
    # rather than a union -- most likely a closed-form XOR term over the
    # TRANSFORMED operand taint, which keeps the differential's precision.
    # The carry/overflow PREDICATES are carry-coupled consumers too, even though
    # they are not in TRANSPORTABLE_OPCODES: `cmp x1,x2,lsl #3` lifts its OV as
    # `u = x2 << 3 ; tmpOV = sborrow(x1, u)`, and leaving that TRANSLATABLE kept it
    # away from the exact signed-overflow term.
    _carry_consumers = TRANSPORTABLE_OPCODES | {'INT_CARRY', 'INT_SCARRY', 'INT_SBORROW'}
    _carry_consumes_shift = any(
        op.opcode.name in _carry_consumers and any(_reaches_shift(i) for i in op.inputs)
        for op in core_ops
    )
    if _real_shifts and not _carry_consumes_shift:
        return InstructionCategory.TRANSLATABLE

    # XOR makes the monotonic differential UNSOUND.  D^{++} for XOR computes
    #   (a|Ta ^ b|Tb) ^ (a&~Ta ^ b&~Tb) = Ta ^ Tb,
    # which is 0 wherever a bit is tainted in BOTH operands, while the true taint
    # is Ta | Tb.  So a slice whose combining op is XOR must be WELDABLE (the OR
    # of input taints), not monotonic.  SPARC's `xnor` lifts to
    # NEGATE(b); XOR(a, ~b), and the affine NEGATE (a MONOTONIC_OPCODES member)
    # otherwise steals the slice into MONOTONIC and drops the shared-bit taint.
    #
    # Guarded so it only pre-empts MONOTONIC, never the cross-bit categories:
    # a carry op (ADD/SUB/2COMP) must stay TRANSPORTABLE and a shift must stay
    # TRANSLATABLE (both already checked above / excluded here), because weldable
    # would under-taint their bit mixing.  For the remaining position-wise-affine
    # case (XOR with NEGATE/AND/OR/routing) weldable is sound -- it never
    # under-taints -- and exact for plain xor/xnor.
    if any(op.opcode.name in ORABLE_OPCODES for op in core_ops) and not any(
        op.opcode.name in TRANSPORTABLE_OPCODES for op in core_ops
    ):
        return InstructionCategory.ORABLE

    for op in core_ops:
        if op.opcode.name in MONOTONIC_OPCODES:
            # Soundness fix for blsi/blsr-like patterns: if the slice ALSO contains
            # a carry-introducing op (INT_2COMP, INT_ADD, INT_SUB), the MONOTONIC
            # formula (just the differential) can miss carry propagation.  In that
            # case fall through to TRANSPORTABLE, whose `diff | union(T_deps)`
            # formula adds the input-taint union as a soundness floor.
            #
            # IMPORTANT: only apply for multi-bit outputs (≥2 bits).  For 1-bit
            # flag outputs (CF/OF/SF/ZF), MONOTONIC has its own soundness floor
            # via FullMaskAvalanche, while TRANSPORTABLE drops the union term
            # for 1-bit outputs and would lose soundness on flags of cmp/sub/add.
            #
            # Affected instructions (multi-bit case):
            #   blsi rax, rbx  (INT_2COMP + INT_AND)  → would under-taint without floor
            #   blsr rax, rbx  (INT_SUB   + INT_AND)  → same
            # Without this guard, blsi with T_RBX=MASK64 returns T_RAX=1 (the differential
            # of the lowest-set-bit isolation) when the true taint is MASK64.
            if out_width_bits >= 2:
                has_carry_op = any(o.opcode.name in ('INT_2COMP', 'INT_ADD', 'INT_SUB') for o in core_ops)
                if has_carry_op:
                    return InstructionCategory.TRANSPORTABLE
            return InstructionCategory.MONOTONIC

    for op in core_ops:
        if op.opcode.name in TRANSPORTABLE_OPCODES:
            return InstructionCategory.TRANSPORTABLE

    for op in core_ops:
        if op.opcode.name in ORABLE_OPCODES:
            return InstructionCategory.ORABLE

    # Fallback for pure memory operations that weren't caught by the heuristic
    for op in core_ops:
        if op.opcode.name in MAPPED_OPCODES:
            return InstructionCategory.MAPPED

    raise ValueError(
        'Unable to determine instruction category for slice_ops: ' + ', '.join(op.opcode.name for op in slice_ops),
    )
