"""Lower one instruction's p-code into a single branch-free taint program.

One pass over the p-code emits, for every op, both the value expression and the
taint expression, into the same IR.  When the pass ends, each architectural
register's taint is whatever node its slot holds -- result registers and flags
alike, with no per-output program and no second traversal.

Three things make the emitted program small:

  * Constant folding at construction.  A lifter's flag macro is mostly guards
    over operands that are constant once the instruction is decoded
    (`INT_AND(#0xf, #0x3f)`, `(count != 0)`, `(count == 1)`), so most of a
    shift's 38 p-code ops evaluate away entirely.
  * Hash-consing.  The differential corners a flag needs are usually the ones
    the result already computed; sharing them is automatic.
  * Dead-code elimination.  A value is emitted only because some taint rule
    might read it; the ones no rule reads disappear.

Control flow is predicated rather than branched, which keeps the whole
instruction in one basic block: see `_predicated_write`.
"""
# ruff: noqa: PLC0415
from __future__ import annotations

import threading
from enum import StrEnum
from typing import ClassVar

from pypcode import PcodeOp, Varnode

from microtaint.taint_ir.ir import (
    ADD,
    AND,
    CLZ,
    EQ,
    MASK64,
    MUL,
    MULHI,
    NEG,
    NEZ,
    NOT,
    OR,
    POPCNT,
    SAR,
    SDIV,
    SEL,
    SHL,
    SHR,
    SREM,
    SUB,
    UDIV,
    ULT,
    UREM,
    XOR,
    Access,
    IRProg,
)
from microtaint.taint_ir.loopform import CounterKind, LoopForm, recognise_loop
from microtaint.types import ArchLike

LIFT_BASE = 0x1000

#: State slots each memory access occupies, past the register file:
#:   +0  the loaded VALUE (in v[]) and its shadow TAINT (in t[])
#:   +1  the computed ADDRESS            (written to o[])
#:   +2  the taint OF that address       (written to o[])
#:   +3  the taint to STORE at it        (written to o[])
#: Modelling memory as extra slots rather than as a callback keeps the emitted
#: program straight-line and call-free, so every backend runs it unchanged.
MEM_SLOTS_PER_ACCESS = 4


def access_slot(n_reg_slots: int, k: int, what: str) -> int:
    """Slot index for one memory access's state."""
    base = n_reg_slots + MEM_SLOTS_PER_ACCESS * k
    return base + {'mem': 0, 'addr': 1, 'addrt': 2, 'sttaint': 3}[what]


class Unsupported(Exception):
    """This p-code shape is outside the lowering; the caller falls back.

    `cut_at`, when set, is the ordinal (within the lowered region) of the
    instruction that made the shape unlowerable.  A caller splitting a block into
    regions can then take the instructions before it and start a new region
    there, instead of searching for the boundary by lowering repeatedly.
    """

    def __init__(self, *args: object, cut_at: int | None = None) -> None:
        super().__init__(*args)
        self.cut_at = cut_at


def _mask_of(size: int) -> int:
    return MASK64 if size >= 8 else ((1 << (size * 8)) - 1)


def _ops_signature(ops: list[PcodeOp], end: int) -> tuple[object, ...]:
    """A hashable identity for `ops[:end + 1]`, holding no lifter objects.

    Opcode, input coordinates and output coordinates pin the p-code exactly, so
    two instructions with this signature compute the same thing and a
    recognition of one is a recognition of the other.
    """
    return tuple(
        (o.opcode.name,
         tuple((v.space.name, v.offset, v.size) for v in o.inputs),
         (o.output.space.name, o.output.offset, o.output.size)
         if o.output is not None else None)
        for o in ops[:end + 1]
    )


def _smear_right(p: IRProg, n: int, bits: int) -> int:
    """Fill every bit below the highest set one, over a `bits`-wide value.

    Turns "the endpoints of a range differ HERE" into "every bit from there
    down can differ", which is what a taint mask has to say about a counter
    whose value moves within a range.  log2(bits) shift-or steps, so three for
    the seven-bit span a 64-bit count needs.
    """
    sh = 1
    while sh < bits:
        n = p.op(OR, n, p.op(SHR, n, p.const(sh)))
        sh *= 2
    return n


# ── the signed-overflow taint table ──────────────────────────────────
#
# OF = (a XNOR b) AND (a XOR c): the operand sign bits and the carry into the
# sign position.  It is the one flag a two-corner differential cannot answer --
# an XOR of two monotone functions is not monotone, so the corners can agree
# while OF varies and disagree while OF is pinned (a != b forces OF == 0
# however the low bits move).
#
# With only three one-bit inputs, each contributing a value bit and a taint bit,
# the whole question "can OF take both values over this cube" is a function of
# six bits -- so the exact answer is a 64-entry table, and a 64-entry table of
# single bits IS one 64-bit constant.  `(K >> idx) & 1` is exact, branch-free,
# and the same trick applies to any boolean function of at most three one-bit
# inputs, which is what flag algebra is made of.
def _build_of_table() -> int:
    k = 0
    for idx in range(64):
        av, at = idx & 1, (idx >> 1) & 1
        bv, bt = (idx >> 2) & 1, (idx >> 3) & 1
        cv, ct = (idx >> 4) & 1, (idx >> 5) & 1
        seen = set()
        for a in ((0, 1) if at else (av,)):
            for b in ((0, 1) if bt else (bv,)):
                for c in ((0, 1) if ct else (cv,)):
                    seen.add(1 if (a == b and a != c) else 0)
        if len(seen) > 1:
            k |= 1 << idx
    return k


OF_TAINT_TABLE = _build_of_table()


class SymFrame:
    """A register file whose cells hold IR nodes instead of numbers.

    Mirrors the byte-offset addressing of the runtime Frame, including partial
    writes and sub-register overlay, so `mov al, bl` followed by a read of RAX
    composes the same way here as it does at runtime.  Resolving that here, at
    lift time, is what lets the emitted program be flat SSA: no aliasing logic
    survives into the compiled code.

    Values and taint use one class because taint aliases exactly like a value:
    the bits of AL's taint sit in the low byte of RAX's taint.
    """

    def __init__(self, prog: IRProg, declared: dict[int, tuple[str, int]],
                 be: bool, kind: str) -> None:
        self.p = prog
        self.be = be
        self.kind = kind                 # 'v' or 't', for input node naming
        self.declared = declared         # offset -> (name, size)
        self.reg: dict[int, tuple[int, int]] = {}   # offset -> (node, size)
        self.uniq: dict[int, tuple[int, int]] = {}
        #: register offsets this program wrote
        self.touched: set[int] = set()
        #: Offsets a wider write has subsumed.  Their declared input node is
        #: stale from that point on -- writing RAX must be visible through AH --
        #: so a read has to resolve them through the parent instead.
        self.killed: set[tuple[str, int]] = set()

    def _space(self, sp: str) -> dict[int, tuple[int, int]]:
        if sp == 'register':
            return self.reg
        if sp == 'unique':
            return self.uniq
        raise Unsupported(f'space {sp}')

    def _cell(self, sp: str, off: int) -> tuple[int, int] | None:
        """The (node, size) at `off`, materialising a declared register's input
        node the first time it is read."""
        d = self._space(sp)
        c = d.get(off)
        if c is not None:
            return c
        if (sp, off) in self.killed:
            return None
        if sp == 'register':
            decl = self.declared.get(off)
            if decl is not None:
                _name, size = decl
                # Keyed by BYTE OFFSET, not by name: one offset carries several
                # names (Ghidra's XMM0_QA is the engine geometry's VL_0x1200),
                # and choosing one of them here would hand the caller a key its
                # state does not use.  The offset is p-code's own vocabulary.
                key = ('reg', off, size)
                node = (self.p.input_value(key, size * 8) if self.kind == 'v'
                        else self.p.input_taint(key, size * 8))
                c = (node, size)
                d[off] = c
                return c
        return None

    def read(self, sp: str, off: int, size: int) -> int:
        p = self.p
        if size > 8:
            raise Unsupported('wide varnode')
        d = self._space(sp)
        cell = self._cell(sp, off)
        if cell is not None:
            base = cell[0]
        else:
            base = p.const(0)
            # A read that starts inside a wider register (AH inside RAX) takes
            # its bytes from that parent.
            for k in range(off - 1, max(-1, off - 9), -1):
                pc = self._cell(sp, k)
                if pc is not None and k + pc[1] > off:
                    byte_off = off - k
                    shift = ((pc[1] - byte_off - size) * 8 if self.be
                             else byte_off * 8)
                    if 0 <= shift < 64:
                        base = p.op(SHR, pc[0], p.const(shift))
                    break
        # Overlay narrower writes that landed inside this read's range.
        k = off + 1
        end = off + size
        while k < end and k - off < 8:
            sub = d.get(k)
            if sub is not None:
                sub_node, k_sz = sub
                byte_off = k - off
                shift = ((size - byte_off - k_sz) * 8 if self.be
                         else byte_off * 8)
                if 0 <= shift < 64:
                    lane = _mask_of(k_sz) << shift
                    kept = p.op(AND, base, p.const(MASK64 ^ lane))
                    put = p.op(SHL, p.op(AND, sub_node, p.const(_mask_of(k_sz))),
                               p.const(shift))
                    base = p.op(OR, kept, put)
                k += k_sz
            else:
                k += 1
        return p.mask(base, size * 8)

    def write(self, sp: str, off: int, size: int, node: int) -> None:
        p = self.p
        if size > 8:
            raise Unsupported('wide varnode')
        d = self._space(sp)
        node = p.mask(node, size * 8)
        cur = self._cell(sp, off)
        if cur is not None and cur[1] > size:
            # A narrow write into a wider live slot keeps the upper bytes.
            lo = _mask_of(size)
            shift = (cur[1] - size) * 8 if self.be else 0
            lane = lo << shift
            kept = p.op(AND, cur[0], p.const(MASK64 ^ lane))
            put = p.op(SHL, node, p.const(shift)) if shift else node
            d[off] = (p.op(OR, kept, put), cur[1])
        else:
            d[off] = (node, size)
            # A wider write subsumes everything narrower inside it -- both slots
            # already written and declared sub-registers not yet materialised.
            for k in range(off + 1, min(off + size, off + 8)):
                sub = d.get(k)
                if sub is not None and sub[1] < size:
                    del d[k]
                    self.killed.add((sp, k))
                elif sp == 'register':
                    decl = self.declared.get(k)
                    if decl is not None and decl[1] < size:
                        self.killed.add((sp, k))
        if sp == 'register':
            self.touched.add(off)


#: What a load through a secret-dependent address yields.
#:
#:   'avalanche' -- taint the whole loaded word, because WHICH bytes are read is
#:                  then itself secret-dependent.  The sound answer, the one the
#:                  whole-instruction differential gives, and the default.
#:   'concrete'  -- read the shadow at the address the instruction computes and
#:                  make no claim about the address's own taint.  Tighter, and
#:                  UNSOUND unless the caller knows the address is not
#:                  attacker-controlled.  Kept because it is the right answer for
#:                  a caller that has established that, and because the two are
#:                  worth comparing.
#:
#: The default used to be 'concrete', on the stated grounds that it was "what
#: the engine's differential does".  That was wrong, and measurably so:
#:
#:     movzbl (%rax,%rcx,1),%eax   RAX tainted, table clean and public
#:         differential  RAX = 0xffffffffffffffff
#:         compiled      RAX = 0x0
#:
#: so the shipped hot path lost the secret at the first table lookup.  On
#: bench_dense the entire 256-byte state buffer went clean at the first S-box
#: round.  A load through a tainted address is exactly the shape a taint engine
#: exists to follow, so the sound policy is the one that runs.
class PointerPolicy(StrEnum):
    """What a load through a TAINTED address does to the loaded value.

    A StrEnum so the value a caller passes and the member compare equal,
    and so it serialises as itself; the Builder normalises to a member on
    the way in, and every comparison after that is member to member.
    """

    #: A tainted address taints the whole loaded word.  Sound, and the one
    #: that runs: see the note above on what the alternative cost.
    AVALANCHE = 'avalanche'
    #: Read the shadow at the computed address, claiming nothing about the
    #: address's own taint.
    CONCRETE = 'concrete'


POINTER_POLICIES: tuple[PointerPolicy, ...] = tuple(PointerPolicy)


class Emit(StrEnum):
    """What a lowering publishes: taint alone, values alone, or both."""

    TAINT = 'taint'
    VALUE = 'value'
    BOTH = 'both'

#: What every caller gets unless it names something else.  Soundness is not an
#: opt-in.
DEFAULT_POINTER_POLICY: PointerPolicy = PointerPolicy.AVALANCHE

#: The most times a p-code loop is unrolled before the residual case is
#: floored.  A bit scan -- which is what SLEIGH models `bsf` and `bsr` as, a
#: loop over bit positions rather than an opcode -- runs at most as many times
#: as its operand is wide, so 64 covers every width there is.
_UNROLL_LIMIT = 64


def _body_width_bits(ops: list[PcodeOp], lo: int, hi: int) -> int:
    """The widest varnode the loop body touches, in bits.

    A loop over bit positions cannot run more times than its widest operand has
    bits, so this bounds the unrolling far more tightly than the flat maximum
    for anything narrower than a machine word: a 32-bit `bsf` was paying for 64
    iterations of which the last 32 can never run, and each iteration costs
    around 80 IR operations because every write in the body becomes a select.

    Read off varnode SIZES rather than off opcodes, so it is a fact about the
    p-code and not a rule about one instruction set.  Getting it too small
    costs precision and never soundness -- the floor covers whatever the
    unrolling did not reach -- so the widest operand is the safe way to be
    wrong.
    """
    widest = 1
    for o in ops[lo:hi]:
        for v in (*o.inputs, o.output):
            if v is not None and v.size > widest:
                widest = v.size
    return widest * 8


class Builder:
    """Lower one instruction to IR.  `prog.outputs` ends up holding the taint of
    every architectural register, written or passed through."""

    def __init__(self, arch: ArchLike, be: bool,
                 pointer_policy: PointerPolicy = DEFAULT_POINTER_POLICY) -> None:
        # Normalise once, so every comparison below is member to member
        # rather than a string compare, and a bad spelling fails here.
        self.pointer_policy = PointerPolicy(pointer_policy)
        from microtaint.instrumentation.cell import _build_reg_maps
        offsets, sizes = _build_reg_maps(arch)[:2]
        # Keep a NON-OVERLAPPING cover of the register file, widest first.
        #
        # A register file names the same bytes many times over -- RAX and EAX
        # and AH, XMM0_QA and XMM0_DB -- while a caller's taint state holds one
        # slot per architectural lane, not one per name.  Declaring every name
        # would make a sub-register its own input, so reading it would return
        # zero instead of resolving through the lane the caller actually
        # supplied.  Overlapping names are dropped; the symbolic frame already
        # resolves reads and writes inside a wider slot, exactly as the runtime
        # frame does.
        candidates = []
        self.offset_of: dict[str, int] = {}
        for name, off in offsets.items():
            size = sizes.get(name, 8)
            self.offset_of.setdefault(name, off)
            if size > 8 or size <= 0:
                continue
            candidates.append((off, -size, name))
        candidates.sort()
        declared: dict[int, tuple[str, int]] = {}
        self.name_by_off: dict[int, str] = {}
        covered: set[int] = set()
        for off, negsz, name in candidates:
            size = -negsz
            if any(b in covered for b in range(off, off + size)):
                continue
            covered.update(range(off, off + size))
            declared[off] = (name, size)
            self.name_by_off[off] = name
        # The architectural program counter, so a branch can say what it makes
        # secret-dependent.  Ghidra spells it RIP/EIP on x86 and PC everywhere
        # else, which is the whole of the ISA-specific knowledge needed here.
        self.pc_off = None
        self.pc_size = 8
        for off, (nm, size) in declared.items():
            if nm in ('RIP', 'EIP', 'PC'):
                self.pc_off, self.pc_size = off, size
                break
        self.declared = declared
        self.names = {n for n, _s in declared.values()}
        self.be = be
        self.arch = arch
        #: Recognised p-code loops, keyed by the p-code itself.  Lives on the
        #: Builder because `builder_for` shares one per architecture, so the
        #: answer is computed once however many blocks contain the instruction.
        self._loop_cache: dict[tuple[object, ...], LoopForm | None] = {}

    def build(self, ops: list[PcodeOp], end_addr: int, *,  # noqa: C901
              emit: Emit = Emit.TAINT, block: bool = False,
              abs_ram: bool = False, max_acc: int | None = None) -> IRProg:
        """Lower `ops`.  `emit` selects which frame becomes the program's
        outputs: 'taint' (the shipped behaviour) or 'value'.

        The two frames are built side by side and alias identically, so a value
        program is the same lowering read off the other one.  It exists because
        block-level tainting needs the register values BETWEEN the instructions
        of a block, and asking the emulator for them per instruction is the cost
        block tainting is trying to remove.  Publishing them makes them survive
        dead-code elimination, which otherwise deletes every value no taint rule
        reads.

        Nothing on the per-instruction path passes `emit`, so that path is
        bit-identical and no slower.
        """
        if emit not in ('taint', 'value', 'both'):
            raise ValueError(emit)
        # Same normalisation as the policy: a member from here on.
        self.emit = Emit(emit)
        self.block = block
        # `abs_ram` says the ops were translated at the address they really run
        # at, so a `ram` varnode's offset is a live guest address.  Off by
        # default because the per-instruction path lifts every instruction at a
        # synthetic base, where that offset would name the wrong byte -- and a
        # load or store aimed at the wrong byte is an UNDER-taint.
        self.abs_ram = abs_ram
        # The runtime has room for a fixed number of memory accesses per
        # program.  Overflowing it used to refuse the whole BLOCK, which the
        # hook then skipped; declining here instead names the instruction that
        # overflowed, so the planner cuts the region there and every
        # instruction still lands in one.
        self.max_acc = max_acc
        p = IRProg()
        v = SymFrame(p, self.declared, self.be, 'v')
        t = SymFrame(p, self.declared, self.be, 't')
        self.p, self.v, self.t = p, v, t
        #: (opcode, first_node, last_node) per p-code op, so the emitted cost
        #: can be attributed back to the lifter construct that caused it.
        #: Approximate by construction: a hash-consed node is credited to
        #: whichever op created it first.
        self.spans: list[tuple[str, int, int]] = []
        #: One entry per distinct (address, size) the instruction touches.
        #: SLEIGH re-emits the same LOAD once per flag that reads it, so
        #: de-duplicating here turns `add rax, [rbx+16]`'s three identical
        #: loads into one access.
        self.accesses: list[Access] = []
        self.access_map: dict[tuple[str, int, int], int] = {}
        # Predicate stack: (until_pc, saved_pred_value, saved_pred_taint).
        self.pred_v = p.const(1)
        self.pred_t = p.const(0)
        #: value node of a difference -> (av, at, bv, bt, bits) it came from.
        self._diff_of: dict[int, tuple[int, int, int, int, int]] = {}
        pred_stack: list[tuple[int, int, int]] = []

        n = len(ops)
        # SLEIGH names an intra-instruction branch target by ADDRESS, not by
        # p-code index: the end of the instruction for a skip, an IMARK for a
        # jump back into the sequence.  Resolve both to indices up front.
        self.end_addr = end_addr
        self.imark_pc: dict[int, int] = {}
        for i, o in enumerate(ops):
            if o.opcode.name == 'IMARK' and o.inputs:
                self.imark_pc.setdefault(o.inputs[0].offset, i)
        # Which instruction each op belongs to, as the p-code index of that
        # instruction's IMARK.  Only block mode needs it: it is how a branch
        # OUT of the current instruction is told from a branch back into it.
        self.instr_pc = [0] * n
        #: Ordinal of the instruction each op belongs to, counting IMARKs.  A
        #: decline reports this so a caller can cut a region at the instruction
        #: that caused it rather than searching for the boundary.
        self.instr_ord = [0] * n
        cur = 0
        ordinal = -1
        for i, o in enumerate(ops):
            if o.opcode.name == 'IMARK' and o.inputs:
                cur = i
                ordinal += 1
            self.instr_pc[i] = cur
            self.instr_ord[i] = max(ordinal, 0)
        self.cur_instr = 0
        self.n_ops = n
        #: How many times each backward branch has been taken, so a p-code loop
        #: is unrolled a bounded number of times rather than declined.
        unrolled: dict[int, int] = {}
        #: The predicate in force when each p-code loop was first entered, so
        #: the ops after the loop can be lowered under it rather than under
        #: "the loop goes round again", which is false by the time it ends.
        loop_entry_pred: dict[int, int] = {}
        #: Loops whose meaning was RECOGNISED, keyed by the op they start at.
        #: Each is emitted as a closed form and its body skipped entirely.
        loop_forms = self._recognise_loops(ops)
        pc = -1
        while True:
            pc += 1
            if pc >= n:
                break
            while pred_stack and pred_stack[-1][0] == pc:
                _u, self.pred_v, self.pred_t = pred_stack.pop()
            found = loop_forms.get(pc)
            if found is not None:
                form, body_end = found
                self._emit_counter(form)
                pc = body_end        # the increment at the top steps past it
                continue
            op = ops[pc]
            self.cur_instr = self.instr_ord[pc]
            name = op.opcode.name
            if name in ('IMARK', 'INDIRECT', 'MULTIEQUAL', 'CAST', 'CPOOLREF',
                        'NEW', 'SEGMENTOP'):
                continue
            if name == 'CBRANCH':
                target = self._branch_target(op, pc, n)
                if target is None:
                    # A real conditional jump out of the instruction.  Which of
                    # two addresses the counter takes is decided by the
                    # condition, so a tainted condition makes the counter
                    # secret-dependent -- the implicit flow the engine's policy
                    # exists to report.
                    _cv, cond_t = self._read_in(op.inputs[1])
                    self._write_pc(p.op(AND, p.const(_mask_of(self.pc_size)),
                                        p.splat(p.op(NEZ, cond_t))))
                    continue
                if target <= pc:
                    if self.instr_pc[target] != self.instr_pc[pc]:
                        raise Unsupported('backward CBRANCH (p-code loop)')
                    # The predicate as it stood BEFORE the loop, kept because
                    # control reaches the ops AFTER the loop whatever the loop
                    # did, and the predicate below says only whether the loop
                    # goes round AGAIN.  Without this the two are conflated:
                    # the unrolling stops precisely when "go round again" folds
                    # to false, and the rest of the instruction was then lowered
                    # under a FALSE predicate and discarded.
                    #
                    # `pext eax,ebx,ecx` is the case.  Its loop-back is a
                    # CBRANCH, its accumulator is copied into EAX after the
                    # loop, and that copy was thrown away -- so EAX came out
                    # carrying nothing but its own previous taint.  Checked
                    # against Unicorn by hand: eight tainted bits of EBX move
                    # eight bits of the result, and the engine reported zero.
                    # `bsf` never showed it because its loop-back is an
                    # unconditional BRANCH, which does not touch the predicate.
                    if target not in loop_entry_pred:
                        loop_entry_pred[target] = self.pred_v
                    cv, ct = self._read_in(op.inputs[1])
                    # Taking the branch means going round again, so the
                    # iteration that follows runs under `taken`.
                    self.pred_v = p.op(AND, self.pred_v, p.op(NEZ, cv))
                    self.pred_t = p.op(OR, self.pred_t, p.op(NEZ, ct))
                    if self._unroll(ops, target, pc, unrolled):
                        pc = target - 1
                        continue
                    # The loop is over and the instruction continues.  Only the
                    # VALUE predicate is restored: `pred_t` has accumulated the
                    # implicit flow of every exit test, and that is a fact about
                    # the loop the rest of the instruction inherits.
                    self.pred_v = loop_entry_pred.pop(target)
                    continue
                cv, ct = self._read_in(op.inputs[1])
                # The branch SKIPS [pc+1, target); the region therefore runs
                # under `not taken`.
                pred_stack.append((target, self.pred_v, self.pred_t))
                not_c = p.op(XOR, p.op(NEZ, cv), p.const(1))
                self.pred_v = p.op(AND, self.pred_v, not_c)
                self.pred_t = p.op(OR, self.pred_t, p.op(NEZ, ct))
                continue
            if name == 'BRANCH':
                target = self._branch_target(op, pc, n)
                if target is None:
                    self._write_pc(p.const(0))   # a fixed target taints nothing
                    continue
                if target <= pc:
                    if self.instr_pc[target] != self.instr_pc[pc]:
                        # A loop across INSTRUCTIONS is the program's own loop,
                        # which the emulator runs a block at a time; unrolling
                        # it here would model iterations the hook is about to
                        # be called for again.  Only a loop inside ONE
                        # instruction's p-code is this pass's to unroll.
                        raise Unsupported('backward BRANCH (p-code loop)')
                    if self._unroll(ops, target, pc, unrolled):
                        pc = target - 1      # round again; the top re-adds one
                    continue
                # An unconditional forward branch inside the instruction: the
                # skipped region simply never runs under this predicate.
                pred_stack.append((target, self.pred_v, self.pred_t))
                self.pred_v = p.const(0)
                self.pred_t = self.pred_t
                continue
            if name in ('BRANCHIND', 'CALLIND'):
                # An indirect branch: the program counter becomes whatever the
                # operand holds, so it inherits that operand's taint exactly.
                _v, ind_t = self._read_in(op.inputs[0])
                self._write_pc(ind_t)
                continue
            if name == 'CALL':
                # A direct call: the target is fixed, so the counter is not
                # secret-dependent.  The return-address push is an ordinary
                # STORE and was handled as one.
                self._write_pc(p.const(0))
                continue
            if name == 'RETURN':
                # The counter takes whatever RETURN was handed.  On x86 that is
                # the value the preceding LOAD already put there, and on AArch64
                # it is the link register a COPY already moved into it, so this
                # is usually idempotent -- but writing it explicitly does not
                # assume either shape.
                if op.inputs:
                    _v, ret_t = self._read_in(op.inputs[0])
                    self._write_pc(ret_t)
                continue
            if name == 'CALLOTHER':
                self._emit_callother(ops, pc, op)
                continue
            if name == 'LOAD':
                self._emit_load(op)
                continue
            if name == 'STORE':
                self._emit_store(op)
                continue
            if name.startswith('FLOAT_') or name in ('TRUNC', 'INT2FLOAT'):
                raise Unsupported(name)
            if op.output is None:
                raise Unsupported(f'{name} without output')
            start = len(p.nodes)
            self._emit_op(name, op)
            self.spans.append((name, start, len(p.nodes)))

        # Only registers this instruction can have changed are outputs.  An
        # untouched register's taint is its input taint by definition; making
        # the program restate that would cost an operation per architectural
        # register and swamp the instruction's real work.
        # 'both' emits the taint of every register the instruction touched AND
        # its value, under distinct keys, so one program feeds a block: the
        # values it publishes are the inputs of the next instruction in the
        # block, and the emulator is asked for registers once per block rather
        # than once per instruction.  The two frames are hash-consed into one
        # IRProg, so shared subexpressions are computed once.
        want = (('reg', t),) if self.emit is Emit.TAINT else \
               (('regv', v),) if self.emit is Emit.VALUE else \
               (('reg', t), ('regv', v))
        for tag, frame in want:
            dirty_bytes: set[int] = set()
            for off in frame.touched:
                cell = frame.reg.get(off)
                width = cell[1] if cell else 1
                dirty_bytes.update(range(off, off + width))
            for off in sorted(self.declared):
                _nm, size = self.declared[off]
                if not dirty_bytes.intersection(range(off, off + size)):
                    continue
                p.outputs.append(((tag, off, size), frame.read('register', off, size)))
        p.spans = self.spans
        p.accesses = self.accesses
        self._check_address_independence(p)
        if self.block:
            self._check_no_load_after_store()
        return p.finish()

    def _write_pc(self, tnt: int) -> None:
        """Set the program counter's taint.

        Only the TAINT frame is written.  The counter's value at this point is a
        target baked against the lift base, which is not where the instruction
        actually runs; leaving the value alone means a later read sees the
        honest input rather than a fabricated one.  Nothing in these shapes
        reads it -- the branch is the last thing the instruction does.
        """
        if self.pc_off is None:
            raise Unsupported('no program counter in this register file')
        self.t.write('register', self.pc_off, self.pc_size,
                     self.p.mask(tnt, self.pc_size * 8))

    # -- opaque operations ---------------------------------------------
    def _emit_callother(self, ops: list[PcodeOp], pc: int, op: PcodeOp) -> None:
        """An operation p-code does not model (crc32, aes, a fence).

        Its VALUE is unknowable here, so the only safe thing is to make sure
        nothing downstream depends on it: if a later op reads the bytes it
        writes, decline the whole instruction rather than compute with a value
        that was invented.  When nothing reads it, the result is simply an
        opaque function of its inputs, and the sound answer for the register it
        lands in is avalanche -- every output bit tainted if any input bit is.

        A CALLOTHER with no output writes nothing p-code can see, which is what
        the runtime interpreter already assumes; skipping it here keeps the two
        in agreement rather than introducing a new divergence.

        An output WIDER than a word is the same rule written to each of its
        lanes.  Avalanche is the one wide shape that needs no care about which
        lane a bit came from or which end the lanes are numbered from -- every
        lane receives the same all-ones-or-zero mask -- so unlike the exact wide
        shapes above it works on a big-endian target too.

        Refusing these was costing the eight AVX forms in the bank (`vpand`,
        `vpaddd`, `vpunpcklbw`, ...) for no precision at all: Ghidra models them
        as opaque too, so the circuit path answers them with the same avalanche,
        measured at 513 output bits tainted from a single tainted source bit
        against 1 to 6 for the SSE encodings of the same operations.  Lowering
        them changes what they cost, not what they say.
        """
        p = self.p
        if op.output is None:
            return
        osz = self._out(op).size
        if not self._invention_stays_opaque(ops, pc, op.output):
            raise Unsupported('CALLOTHER result is read downstream')
        any_t = p.const(0)
        for vn in op.inputs[1:]:
            if vn.space.name == 'const':
                continue
            if vn.size <= 8:
                any_t = p.op(OR, any_t, self._read_in(vn)[1])
                continue
            for lane in range(0, vn.size, 8):
                any_t = p.op(OR, any_t,
                             self._read_lane(vn, lane, min(8, vn.size - lane))[1])
        avalanche = p.splat(p.op(NEZ, any_t))
        if osz <= 8:
            self._predicated_write(self._out(op), p.const(0),
                                   p.op(AND, p.const(_mask_of(osz)), avalanche))
            return
        for lane in range(0, osz, 8):
            lsz = min(8, osz - lane)
            self._write_lane(self._out(op), lane, lsz, p.const(0),
                             p.op(AND, p.const(_mask_of(lsz)), avalanche))

    #: Opcodes that MOVE bytes without computing on them: their taint rule
    #: reads only taint, never a value.  An invented value can pass through one
    #: of these and stay a lie of exactly the same shape, which is what lets the
    #: opaque-result check follow it instead of refusing at the first reader.
    _MOVEMENT_OPS: ClassVar[frozenset[str]] = frozenset(
        {'COPY', 'INT_ZEXT', 'INT_SEXT', 'SUBPIECE', 'PIECE'})

    def _invention_stays_opaque(self, ops: list[PcodeOp], pc: int,
                                out: Varnode) -> bool:
        """Can this opaque result reach the end of the instruction unread?

        The value written for a CALLOTHER is a fabrication -- p-code does not
        model the operation, so there is nothing else to write -- and a later op
        computing with it would turn that fabrication into a taint answer.  The
        check used to refuse at the first reader of any kind.  But a reader that
        only MOVES the bytes cannot consume the fabrication: its taint rule
        reads taint alone, and zero-extending a zero is still the same zero.  So
        movement propagates the opacity forward instead of ending the analysis,
        and only a reader that computes refuses.

        Which is what the AVX forms need: `vpand xmm0, xmm1, xmm2` lifts to a
        CALLOTHER into a 16-byte temporary and an INT_ZEXT of it into ZMM0, so
        the first reader is always a widening move.

        Nothing here runs under a predicate: a predicated write is value-
        dependent (it XORs the two sides to find the bits they differ on), so an
        instruction with any branch in its p-code is refused rather than
        reasoned about.
        """
        for later in ops[pc + 1:]:
            if later.opcode.name in ('CBRANCH', 'BRANCH', 'BRANCHIND', 'CALLIND'):
                return False
        poisoned = [(out.space.name, out.offset, out.offset + out.size)]

        def reads_poison(vn: Varnode) -> bool:
            return any(vn.space.name == sp and vn.offset < hi
                       and vn.offset + vn.size > lo
                       for sp, lo, hi in poisoned)

        for later in ops[pc + 1:]:
            if not any(reads_poison(vn) for vn in later.inputs):
                continue
            if later.opcode.name not in self._MOVEMENT_OPS or later.output is None:
                return False
            o = later.output
            poisoned.append((o.space.name, o.offset, o.offset + o.size))
        return True

    # -- memory --------------------------------------------------------
    def _access_for(self, kind: str, addr_node: int, size: int) -> int:
        key = (kind, addr_node, size)
        k = self.access_map.get(key)
        if k is None:
            k = len(self.accesses)
            self.access_map[key] = k
            self.accesses.append({'kind': kind, 'size': size, 'addr': addr_node,
                                  'instr': self.cur_instr})
            if self.max_acc is not None and len(self.accesses) > self.max_acc:
                raise Unsupported(
                    f'{len(self.accesses)} memory accesses, the runtime has '
                    f'room for {self.max_acc}', cut_at=self.cur_instr)
            self.p.outputs.append((('addr', k), addr_node))
        return k

    def _emit_load(self, op: PcodeOp) -> None:
        """A load's value and shadow taint enter as INPUTS at a slot the caller
        fills once the address has been computed.

        What a tainted address implies is a policy question, not a lowering
        question -- see POINTER_POLICIES.  Either way the address's own taint is
        published as an output, so a caller can act on it.
        """
        p = self.p
        size = self._out(op).size
        if size > 8:
            self._emit_wide_load(op)
            return
        addr_v, addr_t = self._read_in(op.inputs[1])
        k = self._access_for('load', addr_v, size)
        p.outputs.append((('addrt', k), addr_t))
        val = p.input_value(('mem', k), size * 8)
        tnt = p.input_taint(('mem', k), size * 8)
        if self.pointer_policy is PointerPolicy.AVALANCHE:
            tnt = p.op(OR, tnt,
                       p.op(AND, p.const(_mask_of(size)),
                            p.splat(p.op(NEZ, addr_t))))
        self._predicated_write(self._out(op), val, tnt, proved=True)

    def _lane_taint(self, tnt: int, addr_t: int, lsz: int) -> int:
        """A loaded lane's taint, with the pointer policy applied.

        Under AVALANCHE a tainted address taints the whole word it brought
        back, because which word that is was itself secret.
        """
        p = self.p
        if self.pointer_policy is not PointerPolicy.AVALANCHE:
            return tnt
        return p.op(OR, tnt,
                    p.op(AND, p.const(_mask_of(lsz)),
                         p.splat(p.op(NEZ, addr_t))))

    def _emit_wide_load(self, op: PcodeOp) -> None:
        """A load wider than a machine word, one 8-byte lane at a time.

        A vector load is a byte-for-byte copy, so lane `k` of the destination
        is the 8 bytes at `address + k` whichever way round the target stores
        them, and the runtime reads each lane exactly as it reads any other
        8-byte load.  Splitting is what the per-instruction path already does
        for wide MEMORY, and without it here a block containing one is refused
        whole -- and a refused block is skipped, so nothing computes its taint.
        Measured on a static-glibc guest, wide loads and stores were 41 of the
        58 reasons a block refused, all of them inside the vectorised string
        and memory routines.
        """
        p = self.p
        out = self._out(op)
        addr_v, addr_t = self._read_in(op.inputs[1])
        for lane in range(0, out.size, 8):
            lsz = min(8, out.size - lane)
            a = addr_v if lane == 0 else p.op(ADD, addr_v, p.const(lane))
            k = self._access_for('load', a, lsz)
            p.outputs.append((('addrt', k), addr_t))
            val = p.input_value(('mem', k), lsz * 8)
            tnt = self._lane_taint(p.input_taint(('mem', k), lsz * 8),
                                   addr_t, lsz)
            self._write_lane(out, lane, lsz, val, tnt)

    def _emit_wide_store(self, op: PcodeOp) -> None:
        """The store half of `_emit_wide_load`."""
        p = self.p
        src = op.inputs[2]
        addr_v, addr_t = self._read_in(op.inputs[1])
        for lane in range(0, src.size, 8):
            lsz = min(8, src.size - lane)
            a = addr_v if lane == 0 else p.op(ADD, addr_v, p.const(lane))
            val_v, val_t = self._read_lane(src, lane, lsz)
            self._store_access(a, addr_t, lsz, val_v, val_t)

    def _ram_load(self, vn: Varnode) -> tuple[int, int]:
        """A varnode that IS memory: a load at an address the lifter resolved.

        A PC-relative operand has no address arithmetic left by the time SLEIGH
        is done with it -- the displacement folds into the program counter and
        the operand comes out as a `ram` varnode at an absolute address.  That
        is a load like any other with a constant address and no address taint,
        so it goes through the same access machinery and the caller resolves it
        against guest memory and the shadow in pass one exactly as it does a
        LOAD.

        Refusing these instead is not the safe direction it looks like: the
        block that contains one is refused whole and then SKIPPED, so nothing
        computes its taint at all.  Measured on bench_sparse, that lost RBX,
        R11 and two memory words -- an under-taint, from two refused blocks in
        10,701.
        """
        p = self.p
        if vn.size > 8:
            raise Unsupported('wide ram input')
        k = self._access_for('load', p.const(vn.offset), vn.size)
        p.outputs.append((('addrt', k), p.const(0)))
        return (p.input_value(('mem', k), vn.size * 8),
                p.input_taint(('mem', k), vn.size * 8))

    def _ram_store(self, vn: Varnode, val: int, tnt: int) -> None:
        """The store half of `_ram_load`: a write at a resolved address."""
        p = self.p
        if vn.size > 8:
            raise Unsupported('wide ram output')
        self._store_access(p.const(vn.offset), p.const(0), vn.size, val, tnt)

    def _emit_store(self, op: PcodeOp) -> None:
        """A store contributes the taint to write and the address to write it
        at, both as outputs; committing them is the caller's job.

        A store through a TAINTED address writes somewhere secret-dependent,
        which is a different and much larger obligation than tainting a known
        location -- the address taint is reported so the caller can hand those
        instructions to the slow path rather than silently writing one place.

        Under `emit='both'` (block lowering) the stored VALUE is published too.
        A block that stores to an address and then loads it back has to see its
        own store, and the caller can only resolve a load against guest memory
        as it was before the block ran.  Publishing the taint alone gives the
        later load the right taint and a stale value, which silently sends any
        address derived from it somewhere else: measured on bench_dense, the
        swap loop's `idx` came back one iteration old and `state[idx] = tmp`
        wrote to the previous iteration's index throughout.  Nothing on the
        per-instruction path asks for values, so its slot layout is unchanged.
        """
        size = op.inputs[2].size
        if size > 8:
            self._emit_wide_store(op)
            return
        addr_v, addr_t = self._read_in(op.inputs[1])
        val_v, val_t = self._read_in(op.inputs[2])
        self._store_access(addr_v, addr_t, size, val_v, val_t)

    def _check_address_independence(self, p: IRProg) -> None:
        """An address may not depend on a value this instruction itself loaded.

        The caller runs the program once to obtain the addresses, fills the
        memory slots, then runs it again for the taint -- which is only valid
        while no address is downstream of a memory input.  x86 has no such
        shape, but declining is the honest answer rather than a wrong address.
        """
        mem_nodes = {n for (kind, key), n in p.inputs.items()
                     if isinstance(key, tuple) and key[0] == 'mem' and kind == 'v'}
        mem_nodes |= {n for (kind, key), n in p.inputs.items()
                      if isinstance(key, tuple) and key[0] == 'mem' and kind == 't'}
        if not mem_nodes:
            return
        for acc in self.accesses:
            seen, stack = set(), [acc['addr']]
            while stack:
                n = stack.pop()
                if n in seen:
                    continue
                seen.add(n)
                if n in mem_nodes:
                    raise Unsupported('address depends on a loaded value',
                                      cut_at=acc.get('instr'))
                _o, a, b, c, _imm = p.nodes[n]
                for x in (a, b, c):
                    if x >= 0:
                        stack.append(x)

    def _check_no_load_after_store(self) -> None:
        """A load may not follow a store in the same program.

        The two-pass protocol resolves EVERY load against guest memory and the
        shadow before the program runs, and commits every store after it.
        Within one instruction that ordering is exact, because an instruction
        that both stores and then loads the same place does not exist in the
        lifters we handle.  Across a REGION it is not: `mov [rbp+8], rax`
        followed by `mov rax, [rbp+8]` resolves the load against the shadow as
        it was BEFORE the store, so a tainted value stored and read straight
        back comes out clean.  Measured, that is exactly what happened: RAX
        0xff by instruction, 0x00 as one region.  An under-taint, and the worst
        kind, because storing and reloading is what every compiler does across
        a spill.

        The rule is deliberately blunt: any load after any store declines,
        without asking whether the addresses can alias.  Proving they cannot
        needs the addresses, and the addresses are not known until the program
        has run, which is the same circularity `_check_address_independence`
        refuses.  A caller that wants the region anyway can cut it here and
        lower the two halves, which is what the block planner does.

        Single-instruction programs are unaffected: this only fires on an
        ordering that takes more than one instruction to create.
        """
        stored = False
        for acc in self.accesses:
            if acc['kind'] == 'load' and stored:
                raise Unsupported('load after store (no store-to-load forwarding)',
                                  cut_at=acc.get('instr'))
            if acc['kind'] != 'load':
                stored = True

    # -- helpers -------------------------------------------------------
    def _branch_target(self, op: PcodeOp, pc: int, n: int) -> int | None:
        tgt: int | None
        vn = op.inputs[0]
        if vn.space.name == 'const':
            rel = vn.offset
            bits = vn.size * 8
            if rel >> (bits - 1):
                rel -= 1 << bits
            tgt = pc + rel
            return tgt if 0 <= tgt <= n else None
        if vn.space.name == 'ram':
            if vn.offset == self.end_addr:
                return n                      # skip to the end of the region
            tgt = self.imark_pc.get(vn.offset)
            if tgt is None:
                return None
            if self.block and self.instr_pc[tgt] != self.instr_pc[pc]:
                # Lowering a whole basic block, and this branch leaves the
                # instruction it is in.  Unicorn ends a block at any branch, so
                # such a branch can only be the block's LAST instruction and its
                # target is the next block, whatever the address arithmetic says
                # about direction.  Resolving it to a p-code index inside this
                # region instead reads the loop edge of a `for` as a p-code loop
                # and declines the block: measured, that was 95% of the block
                # executions of bench_untainted and bench_dense.
                #
                # A ram target back into the SAME instruction is still a p-code
                # loop (this is how `rep` lifts) and still declines.
                return None
            return tgt
        return None

    @staticmethod
    def _out(op: PcodeOp) -> Varnode:
        """The op's output varnode.

        `op.output` is None for the ops that write nothing -- STORE,
        BRANCH, CBRANCH, CALL -- and every use of this is on a dispatch
        path that has already matched an opcode which writes.  Stating it
        turns a would-be AttributeError into a DECLINE, which is the sound
        direction: the caller falls back to the differential rather than
        lowering an instruction this does not model.
        """
        out = op.output
        if out is None:
            raise Unsupported(f'{op.opcode.name} writes no output')
        return out

    def _read_in(self, vn: Varnode) -> tuple[int, int]:
        p = self.p
        sp = vn.space.name
        if sp == 'const':
            return p.const(vn.offset & _mask_of(vn.size)), p.const(0)
        if sp in ('register', 'unique'):
            return (self.v.read(sp, vn.offset, vn.size),
                    self.t.read(sp, vn.offset, vn.size))
        if sp == 'ram' and self.abs_ram:
            return self._ram_load(vn)
        raise Unsupported(f'input space {sp}')

    def _predicated_write(self, vn: Varnode, val: int, tnt: int,
                          *, proved: bool = False) -> None:
        """Commit one output under the current predicate.

        `proved` says the VALUE is what p-code semantics computed, so that a
        value which folded to a constant folded because it is constant for
        every input.  It defaults to FALSE, and that polarity is the whole
        point: a caller writing a value it invented gets the safe answer by
        saying nothing, and a future one that forgets loses an optimisation
        rather than under-tainting.  See `_predicated_write_at`.

        With a KNOWN predicate this folds to a plain write (the constant SEL
        disappears).  With an UNKNOWN but untainted predicate it becomes a
        select -- exact, and branch-free, which is the point.

        With a TAINTED predicate it is implicit flow: which value lands is
        itself secret, so the result carries both sides' taint plus every bit on
        which the two sides differ.  That is the exact join, not a floor, and it
        is what makes predicated instructions (cmov, csel, csinc) and a lifter's
        conditional flag macros correct without a fallback.
        """
        sp = vn.space.name
        if sp == 'ram' and self.abs_ram:
            self._ram_store(vn, val, tnt)
            return
        if sp not in ('register', 'unique'):
            raise Unsupported(f'output space {sp}')
        self._predicated_write_at(sp, vn.offset, vn.size, val, tnt, proved=proved)

    def _predicated_write_at(self, sp: str, off: int, size: int,
                             val: int, tnt: int, *, proved: bool = False) -> None:
        """`_predicated_write` addressed by (space, offset, size).

        Split out so a caller holding plain varnode COORDINATES rather than a
        varnode object shares this join rather than restating it.  Restating a
        soundness-critical join is how two of them drift apart.
        """
        p = self.p
        if p.is_const(self.pred_v) and p.const_val(self.pred_v) == 1 \
                and p.is_const(self.pred_t) and p.const_val(self.pred_t) == 0:
            # A value that is the same constant whatever the inputs are cannot
            # carry taint: there is no input bit an attacker can move that
            # changes it.  `xor rax,rax` is the everyday case -- it is how every
            # compiler on every architecture zeroes a register -- and the taint
            # rule for a binary op is `t_a | t_b`, which for two copies of the
            # same node is `t | t = t`, so the taint walked straight through a
            # register that provably holds zero.  The IR had already folded the
            # VALUE to a constant; the two halves simply never consulted each
            # other.  Also covers `sub reg,reg`, `and reg,0`, `or reg,-1`, and
            # the flags those set, which is where most of the over-taint was.
            #
            # `proved` is load-bearing, and is not the same question as
            # `is_const`.  An operation p-code does not model writes an INVENTED
            # value -- `_emit_callother` writes a literal zero because it has to
            # write something -- so `const 0` in the IR means either "proved
            # zero" or "no idea".  Clearing on the second is an under-taint:
            # measured, it lost bit 8 of `crc32 rax,cl` and the ground-truth
            # sweep caught it.  Only the callers that write what p-code
            # semantics actually computed say `proved`.
            if proved and p.is_const(val):
                tnt = p.const(0)
            self.v.write(sp, off, size, val)
            self.t.write(sp, off, size, tnt)
            return
        old_v = self.v.read(sp, off, size)
        old_t = self.t.read(sp, off, size)
        new_v, new_t = self._join_under_predicate(old_v, old_t, val, tnt)
        self.v.write(sp, off, size, new_v)
        self.t.write(sp, off, size, new_t)

    def _join_under_predicate(self, old_v: int, old_t: int,
                              val: int, tnt: int) -> tuple[int, int]:
        """What a write becomes when the predicate is not known-true.

        With an UNKNOWN but untainted predicate it is a select, which is exact
        and branch-free.  With a TAINTED predicate it is implicit flow: WHICH
        value lands is itself secret, so the result carries both sides' taint
        plus every bit on which the two sides differ.  That is the exact join,
        not a floor.

        Shared by the register and the memory paths, because a predicated store
        is the same question about a different place, and two copies of this
        would be two chances for them to disagree.
        """
        p = self.p
        new_v = p.op(SEL, val, old_v, self.pred_v)
        new_t = p.op(SEL, tnt, old_t, self.pred_v)
        leak = p.op(AND, p.splat(self.pred_t),
                    p.op(OR, p.op(OR, tnt, old_t), p.op(XOR, val, old_v)))
        return new_v, p.op(OR, new_t, leak)

    def _store_access(self, addr_v: int, addr_t: int, size: int,
                      val: int, tnt: int) -> None:
        """Commit one store, under the current predicate.

        A store that may not happen is a store of `predicate ? new : old`, so
        the old contents are READ first and joined exactly as a register write
        is.  Refusing instead is not the safe direction: the block is refused
        whole and then skipped, so nothing computes its taint at all.
        """
        p = self.p
        if not (p.is_const(self.pred_v) and p.const_val(self.pred_v) == 1):
            old = self._access_for('load', addr_v, size)
            p.outputs.append((('addrt', old), addr_t))
            val, tnt = self._join_under_predicate(
                p.input_value(('mem', old), size * 8),
                p.input_taint(('mem', old), size * 8), val, tnt)
        k = self._access_for('store', addr_v, size)
        p.outputs.append((('addrt', k), addr_t))
        p.outputs.append((('sttaint', k), p.mask(tnt, size * 8)))
        if self.emit in (Emit.VALUE, Emit.BOTH):
            p.outputs.append((('stval', k), p.mask(val, size * 8)))

    # -- wide varnodes -------------------------------------------------
    #: Opcodes whose every output bit depends only on the input bits in the
    #: same position, so a varnode wider than a machine word can be done one
    #: 8-byte lane at a time with no loss.  Carry-coupled and position-sensitive
    #: opcodes are deliberately absent: splitting those would drop the coupling
    #: between lanes and under-taint.
    _LANE_OPS: ClassVar[frozenset[str]] = frozenset(
        {'COPY', 'INT_ZEXT', 'INT_SEXT', 'INT_NEGATE',
         'INT_AND', 'INT_OR', 'INT_XOR'})

    def _read_lane(self, vn: Varnode, lane: int, lsz: int) -> tuple[int, int]:
        """One 8-byte lane of a varnode, as (value, taint)."""
        p = self.p
        sp = vn.space.name
        if sp == 'const':
            return p.const((vn.offset >> (8 * lane)) & _mask_of(lsz)), p.const(0)
        if sp in ('register', 'unique'):
            return (self.v.read(sp, vn.offset + lane, lsz),
                    self.t.read(sp, vn.offset + lane, lsz))
        if sp == 'ram' and self.abs_ram:
            # A PC-relative operand wider than a machine word: the resolved
            # address is a constant, so lane `k` is simply the load at
            # `address + k`, exactly as for a wide LOAD.
            k = self._access_for('load', p.const(vn.offset + lane), lsz)
            p.outputs.append((('addrt', k), p.const(0)))
            return (p.input_value(('mem', k), lsz * 8),
                    p.input_taint(('mem', k), lsz * 8))
        raise Unsupported(f'wide input space {sp}')

    def _emit_wide_special(self, name: str, op: PcodeOp) -> bool:
        """Wide shapes that are not bit-parallel but are still exact.

        These are what a widening multiply lifts to.  `imul r64, r64` builds a
        128-bit product and then asks whether the low half sign-extends back to
        the whole of it -- that question IS the carry and overflow flags -- so
        without them the instruction has to fall back to re-executing itself in
        SLEIGH, which is the cost this whole path exists to remove.

        Returns True when it handled the op.
        """
        p = self.p
        if self.be:
            raise Unsupported('wide varnode on a big-endian target')
        osz = self._out(op).size
        isz = op.inputs[0].size if op.inputs else osz

        # A divide whose wide dividend has a PROVABLY ZERO high half is a
        # narrow divide.  x86's `div r64` always lifts to a 128-by-64 divide
        # because RDX:RAX is the dividend, and a compiler emitting a 64-bit
        # division clears RDX first (`xor %edx,%edx`, or `cqto` for the signed
        # form), so the high lane folds to a constant zero right here.  Without
        # this the block is refused and then skipped.
        if name in ('INT_DIV', 'INT_REM', 'INT_SDIV', 'INT_SREM') \
                and self._emit_wide_divide(name, op):
            return True

        # A lane-aligned slice of a wide value is just that lane.
        if name == 'SUBPIECE' and isz > 8 and osz <= 8:
            off = op.inputs[1].offset
            if off % 8 or off + osz > isz:
                return False
            av, at = self._read_lane(op.inputs[0], off, osz)
            self._predicated_write(self._out(op), p.mask(av, osz * 8),
                                   p.mask(at, osz * 8), proved=True)
            return True

        # 128-bit product, truncated to 128 bits: the high lane needs the high
        # half of the low-lane product, which is what MULHI is for.
        if (name == 'INT_MULT' and osz == 16 and isz == 16
                and op.inputs[1].size == 16):
            a0, at0 = self._read_lane(op.inputs[0], 0, 8)
            a1, at1 = self._read_lane(op.inputs[0], 8, 8)
            b0, bt0 = self._read_lane(op.inputs[1], 0, 8)
            b1, bt1 = self._read_lane(op.inputs[1], 8, 8)
            lo = p.op(MUL, a0, b0)
            hi = p.op(ADD, p.op(ADD, p.op(MULHI, a0, b0), p.op(MUL, a0, b1)),
                      p.op(MUL, a1, b0))
            any_t = p.op(OR, p.op(OR, at0, at1), p.op(OR, bt0, bt1))
            full = p.splat(p.op(NEZ, any_t))
            self._write_lane(self._out(op), 0, 8, lo, full)
            self._write_lane(self._out(op), 8, 8, hi, full)
            return True

        # Equality over a wide value: the same rule, accumulated across lanes.
        if name in ('INT_EQUAL', 'INT_NOTEQUAL') and isz > 8:
            eq = p.const(1)
            bad = p.const(0)
            anyv = p.const(0)
            for lane in range(0, isz, 8):
                lsz = min(8, isz - lane)
                av, at = self._read_lane(op.inputs[0], lane, lsz)
                bv, bt = self._read_lane(op.inputs[1], lane, lsz)
                eq = p.op(AND, eq, p.op(EQ, av, bv))
                d = p.op(XOR, av, bv)
                tu = p.op(OR, at, bt)
                bad = p.op(OR, bad, p.op(AND, d, p.op(NOT, tu)))
                anyv = p.op(OR, anyv, p.op(OR, d, tu))
            v = eq if name == 'INT_EQUAL' else p.op(XOR, eq, p.const(1))
            can_true = p.op(EQ, bad, p.const(0))
            can_false = p.op(NEZ, anyv)
            self._predicated_write(self._out(op), v,
                                   p.op(AND, can_true, can_false), proved=True)
            return True

        return False

    # -- p-code loops recognised as a closed form -----------------------
    def _recognise_loops(
        self, ops: list[PcodeOp],
    ) -> dict[int, tuple[LoopForm, int]]:
        """Loops whose meaning is known, keyed by the op their body starts at.

        Cheap when there is nothing to find: a scan for a backward branch, and
        on this instruction set that is fewer than ten forms in five hundred.
        Recognition itself runs the loop concretely on about a hundred inputs,
        which is microseconds and happens once per distinct instruction, against
        the tens of milliseconds unrolling the same loop costs.
        """
        out: dict[int, tuple[LoopForm, int]] = {}
        for i, op in enumerate(ops):
            if op.opcode.name not in ('BRANCH', 'CBRANCH') or not op.inputs:
                continue
            tgt = op.inputs[0]
            if tgt.space.name != 'const' or not (tgt.offset & 0x80000000):
                continue
            target = i + (tgt.offset - (1 << 32))
            if not (0 <= target < i) or self.instr_pc[target] != self.instr_pc[i]:
                continue
            # Memoised on the p-code itself.  Recognition runs the loop about a
            # hundred times, and the planner lowers the same instruction many
            # times while it searches for region boundaries, so without this the
            # probe is paid over and over for one answer that cannot change.
            key = (_ops_signature(ops, i), target, i, self.be)
            if key in self._loop_cache:
                form = self._loop_cache[key]
            else:
                form = recognise_loop(ops, target, i, self.be)
                self._loop_cache[key] = form
            if form is not None:
                out[target] = (form, i)
        return out

    def _count_expr(self, kind: CounterKind, v: int, w: int) -> int:
        """The counter's VALUE, as IR, for an operand already masked to `w`."""
        p = self.p
        if kind is CounterKind.POPCOUNT:
            return p.op(POPCNT, v)
        if kind is CounterKind.CLZ:
            return p.op(SUB, p.op(CLZ, v), p.const(64 - w))
        if kind is CounterKind.MSB_INDEX:
            return p.op(SUB, p.const(w - 1),
                        p.op(SUB, p.op(CLZ, v), p.const(64 - w)))
        # Trailing zeros, without a trailing-zero opcode: `v & -v` isolates the
        # lowest set bit, one less than it is every bit below, and their count is
        # the answer.  A zero operand gives an all-ones mask, so the result is
        # the full width -- which is what a trailing-zero count of zero means.
        low = p.op(AND, v, p.op(NEG, v))
        return p.op(POPCNT, p.mask(p.op(SUB, low, p.const(1)), w))

    def _emit_counter(self, form: LoopForm) -> None:
        """Emit a recognised loop as a closed form instead of unrolling it.

        The TAINT comes from the two corners of the taint cube -- every tainted
        bit cleared, and every one set.  They bracket the whole reachable range
        because each recognised form is monotone or antitone in every input bit,
        which is the property `loopform` establishes and the only reason a
        two-point evaluation may stand in for the cube.  The bits that can
        differ are then the bits below where the endpoints first differ.  Same
        rule the lowering already applies to the `POPCOUNT` and `LZCOUNT`
        opcodes; this is the same question asked of a loop.
        """
        p = self.p
        w = form.width
        sv, st = (self.v.read(*form.src), self.t.read(*form.src))
        sv, st = p.mask(sv, w), p.mask(st, w)
        lo_v = p.mask(p.op(AND, sv, p.op(NOT, st)), w)
        hi_v = p.mask(p.op(OR, sv, st), w)

        nb = max(1, w.bit_length())
        span = p.const((1 << nb) - 1)
        varies = _smear_right(
            p, p.op(XOR, self._count_expr(form.kind, lo_v, w),
                    self._count_expr(form.kind, hi_v, w)), nb)
        tnt = p.op(AND, span, varies)

        if form.zero_is_special:
            # The one input where monotonicity fails, so the corners say
            # nothing: `bsf` leaves 0 where the trailing-zero count is the
            # operand width, and setting a bit from there RAISES the answer
            # where everywhere else it lowers it.  Zero is reachable exactly
            # when clearing every tainted bit leaves nothing, and the honest
            # answer there is the whole span.
            reachable = p.op(EQ, lo_v, p.const(0))
            tnt = p.op(OR, tnt, p.op(AND, span, p.splat(reachable)))

        sp, off, size = form.dst
        self._predicated_write_at(sp, off, size,
                                  self._count_expr(form.kind, sv, w), tnt,
                                  proved=True)

    def _unroll(self, ops: list[PcodeOp], target: int, pc: int,
                unrolled: dict[int, int]) -> bool:
        """Go round a p-code loop once more, or close it off soundly.

        SLEIGH models some instructions as LOOPS rather than opcodes -- a bit
        scan is a walk over bit positions, and `bsf` inside glibc's memchr and
        strlen is what this exists for -- and the taint IR is straight-line by
        construction.  Declining instead refused the whole BLOCK, which the
        hook then skipped, so nothing computed its taint at all.

        Unrolling is EXACT while it lasts: the predication machinery already
        turns each exit test into a select, so iteration k's writes land under
        "still looping after k", which is precisely the loop's meaning.

        Returns True to go round again.  When the unrollings run out, the
        residual predicate says whether the loop could still be running.  If it
        folds to a constant zero the unrolling was complete and there is
        nothing left to do.  Otherwise everything the body writes is marked
        fully tainted under that predicate: the VALUE may be wrong there, but
        it is then a tainted value, so a load through it avalanches and a store
        through it is reported -- the engine's existing policies carry it, and
        the answer stays sound rather than silently wrong.
        """
        p = self.p
        limit = min(_UNROLL_LIMIT, _body_width_bits(ops, target, pc + 1))
        unrolled[target] = unrolled.get(target, 0) + 1
        if unrolled[target] <= limit:
            return True
        if p.is_const(self.pred_v) and p.const_val(self.pred_v) == 0:
            return False                 # provably finished: exact
        # A body that WRITES MEMORY cannot be floored this way.  Flooring says
        # "everything this wrote is unknown", and for a register that is a mask
        # we can widen; for memory it would be every address the remaining
        # iterations might have touched, which is unbounded.  A `rep movsb`
        # with a count above the limit would otherwise have its later stores
        # simply not modelled, and that is an under-taint -- the one direction
        # that is never acceptable.  Declining is worse than handling it and
        # better than being wrong.
        for o in ops[target:pc + 1]:
            if o.opcode.name in ('STORE', 'STOREIND'):
                raise Unsupported('p-code loop that writes memory')
            out = o.output
            if out is not None and out.space.name == 'ram':
                raise Unsupported('p-code loop that writes memory')
        guard = p.op(NEZ, self.pred_v)
        for o in ops[target:pc + 1]:
            out = o.output
            if out is None or out.space.name not in ('register', 'unique'):
                continue
            if out.size > 8:
                continue                 # a wide write is lane-addressed
            old_t = self.t.read(out.space.name, out.offset, out.size)
            self.t.write(out.space.name, out.offset, out.size,
                         p.op(OR, old_t,
                              p.mask(p.splat(guard), out.size * 8)))
        return False

    def _emit_wide_shift(self, name: str, op: PcodeOp) -> bool:
        """A shift of a varnode wider than a machine word, exactly.

        `pslldq` / `psrldq` shift a whole vector register, so the p-code is one
        INT_LEFT or INT_RIGHT on a 16-byte varnode.  That is not bit-parallel
        across lanes -- bits cross from one lane into the next -- so the
        lane-by-lane path declines it, and a block containing one is refused
        and then SKIPPED.

        It is still exact rather than approximate, because the shift amount is
        an immediate: lane `i` of the result is a funnel shift of two input
        lanes, and TAINT follows the identical routing, a shift being a pure
        permutation of bit positions.  Returns False when the shape is not one
        this can do exactly, so the caller keeps looking.
        """
        p = self.p
        if self.be:
            return False                 # lane 0 is taken as least significant
        out = self._out(op)
        src = op.inputs[0]
        osz = out.size
        if osz % 8 or src.size != osz:
            return False                 # a ragged top lane is not a clean funnel
        amt_v, amt_t = self._read_in(op.inputs[1])
        if not p.is_const(amt_v) or not p.is_const(amt_t) or p.const_val(amt_t):
            return False                 # a secret or runtime shift distance
        k = p.const_val(amt_v)
        n_lanes = osz // 8
        if k >= osz * 8:                 # shifted out entirely
            for lane in range(n_lanes):
                self._write_lane(out, lane * 8, 8, p.const(0), p.const(0))
            return True
        q, r = divmod(k, 64)
        # Read every lane before writing any: source and destination are the
        # same register here, so a lane written early would be read late.
        lanes = [self._read_lane(src, i * 8, 8) for i in range(n_lanes)]
        zero = (p.const(0), p.const(0))

        def at(i: int) -> tuple[int, int]:
            """Lane `i`, or zero for one shifted in from outside."""
            return lanes[i] if 0 <= i < n_lanes else zero

        for i in range(n_lanes):
            lo = at(i + q) if name == 'INT_RIGHT' else at(i - q)
            hi = at(i + q + 1) if name == 'INT_RIGHT' else at(i - q - 1)
            if r == 0:
                v, t = lo
            elif name == 'INT_RIGHT':
                sh, back = p.const(r), p.const(64 - r)
                v = p.op(OR, p.op(SHR, lo[0], sh), p.op(SHL, hi[0], back))
                t = p.op(OR, p.op(SHR, lo[1], sh), p.op(SHL, hi[1], back))
            else:
                sh, back = p.const(r), p.const(64 - r)
                v = p.op(OR, p.op(SHL, lo[0], sh), p.op(SHR, hi[0], back))
                t = p.op(OR, p.op(SHL, lo[1], sh), p.op(SHR, hi[1], back))
            self._write_lane(out, i * 8, 8, p.mask(v, 64), p.mask(t, 64))
        return True

    def _emit_wide_divide(self, name: str, op: PcodeOp) -> bool:
        """A wide divide that is really a narrow one, exactly.

        Only when every lane above the first has a VALUE that folds to zero on
        both operands: then the quotient and remainder are the 64-bit ones, and
        the lanes above the first are zero too.  Anything else returns False,
        because guessing a value here would put a wrong number in a register a
        later instruction may turn into an address.

        The lane's TAINT is deliberately not required to be zero.  A value that
        folds to a constant is that constant for every input, so nothing about
        it can depend on a secret; a non-zero taint there is the XOR rule's
        over-approximation (`xor %edx,%edx` zeroes the value exactly while
        `OR(at, at)` keeps the taint), and discarding it is a precision gain,
        not a soundness loss.  That idiom is exactly how a compiler sets up a
        64-bit division.
        """
        p = self.p
        out = self._out(op)
        if out.size % 8 or len(op.inputs) < 2:
            return False
        n_lanes = out.size // 8
        if n_lanes < 2:
            return False
        args = []
        for src in op.inputs[:2]:
            if src.size != out.size:
                return False
            lanes = [self._read_lane(src, i * 8, 8) for i in range(n_lanes)]
            for v, _t in lanes[1:]:
                if not p.is_const(v) or p.const_val(v) != 0:
                    return False       # a genuinely wide operand
            args.append(lanes[0])
        (av, at), (bv, bt) = args
        narrow = {'INT_DIV': UDIV, 'INT_REM': UREM,
                  'INT_SDIV': SDIV, 'INT_SREM': SREM}[name]
        v = p.op(narrow, av, bv)
        # Division mixes every bit of both operands, so a tainted bit anywhere
        # in either taints the whole result.  That is what the narrow path
        # does too.
        t = p.splat(p.op(NEZ, p.op(OR, at, bt)))
        self._write_lane(out, 0, 8, v, t)
        for i in range(1, n_lanes):
            self._write_lane(out, i * 8, 8, p.const(0), p.const(0))
        return True

    def _write_lane(self, vn: Varnode, lane: int, lsz: int, val: int, tnt: int) -> None:
        p = self.p
        sp = vn.space.name
        if not (p.is_const(self.pred_v) and p.const_val(self.pred_v) == 1):
            raise Unsupported('predicated wide write')
        if sp == 'ram' and self.abs_ram:
            k = self._access_for('store', p.const(vn.offset + lane), lsz)
            p.outputs.append((('addrt', k), p.const(0)))
            p.outputs.append((('sttaint', k), p.mask(tnt, lsz * 8)))
            if self.emit in (Emit.VALUE, Emit.BOTH):
                p.outputs.append((('stval', k), p.mask(val, lsz * 8)))
            return
        if sp not in ('register', 'unique'):
            raise Unsupported(f'wide output space {sp}')
        self.v.write(sp, vn.offset + lane, lsz, val)
        self.t.write(sp, vn.offset + lane, lsz, tnt)

    def _emit_wide(self, name: str, op: PcodeOp) -> None:
        """Lane-by-lane emission for a bit-parallel op on a wide varnode.

        Vector movement and vector logic are most of what a SIMD lifter emits
        at full register width, and they cost nothing extra here: the symbolic
        frame is byte-addressed, so a lane is just an offset.
        """
        p = self.p
        if self.be:
            # Lane 0 is taken as the least significant; on a big-endian target
            # the lowest byte offset is the MOST significant, so the order would
            # invert.  No big-endian ISA in the corpus lifts a wide varnode, so
            # decline rather than guess.
            raise Unsupported('wide varnode on a big-endian target')
        osz = self._out(op).size
        isz = op.inputs[0].size if op.inputs else osz
        fill_v = fill_t = None
        if name == 'INT_SEXT':
            # The fill comes from the source's top lane and must be read before
            # any output lane is written, in case they alias.
            top_lane = ((isz - 1) // 8) * 8
            top_sz = isz - top_lane
            sv, st = self._read_lane(op.inputs[0], top_lane, top_sz)
            sh = p.const(top_sz * 8 - 1)
            fill_v = p.splat(p.mask(p.op(SHR, sv, sh), 1))
            fill_t = p.splat(p.mask(p.op(SHR, st, sh), 1))
        for lane in range(0, osz, 8):
            lsz = min(8, osz - lane)
            if name in ('INT_ZEXT', 'INT_SEXT') and lane >= isz:
                if name == 'INT_ZEXT':
                    v, tnt = p.const(0), p.const(0)
                else:
                    # INT_SEXT, and only that branch above sets the fill, so
                    # reaching here without one would be a lowering bug rather
                    # than an input the caller can produce.
                    assert fill_v is not None
                    assert fill_t is not None
                    v, tnt = fill_v, fill_t
            else:
                av, at = self._read_lane(op.inputs[0], lane, lsz)
                if name in ('COPY', 'INT_ZEXT', 'INT_SEXT'):
                    v, tnt = av, at
                elif name == 'INT_NEGATE':
                    v, tnt = p.op(NOT, av), at
                else:
                    bv, bt = self._read_lane(op.inputs[1], lane, lsz)
                    if name == 'INT_AND':
                        v = p.op(AND, av, bv)
                        tnt = p.op(OR, p.op(OR, p.op(AND, at, bv),
                                            p.op(AND, bt, av)),
                                   p.op(AND, at, bt))
                    elif name == 'INT_OR':
                        v = p.op(OR, av, bv)
                        tnt = p.op(OR, p.op(OR, p.op(AND, at, p.op(NOT, bv)),
                                            p.op(AND, bt, p.op(NOT, av))),
                                   p.op(AND, at, bt))
                    else:
                        v = p.op(XOR, av, bv)
                        tnt = p.op(OR, at, bt)
            self._write_lane(self._out(op), lane, lsz,
                             p.mask(v, lsz * 8), p.mask(tnt, lsz * 8))

    # -- the per-opcode lowering ---------------------------------------
    def _emit_op(self, name: str, op: PcodeOp) -> None:
        p = self.p
        if any(s > 8 for s in
               [x.size for x in op.inputs] + [self._out(op).size]):
            if name in self._LANE_OPS:
                self._emit_wide(name, op)
                return
            if name in ('INT_LEFT', 'INT_RIGHT') and self._emit_wide_shift(name, op):
                return
            if self._emit_wide_special(name, op):
                return
            raise Unsupported(f'wide varnode ({name})')
        ins = [self._read_in(x) for x in op.inputs]
        av, at = ins[0] if ins else (p.const(0), p.const(0))
        bv, bt = ins[1] if len(ins) > 1 else (p.const(0), p.const(0))
        osz = self._out(op).size
        isz = op.inputs[0].size if op.inputs else osz
        obits = osz * 8
        ibits = isz * 8
        om = _mask_of(osz)

        val, tnt = self._rule(name, op, av, at, bv, bt, isz, obits, ibits, om)
        self._predicated_write(self._out(op), val, tnt, proved=True)

    def _sext(self, node: int, bits: int) -> int:
        p = self.p
        if bits >= 64:
            return node
        sh = p.const(64 - bits)
        return p.op(SAR, p.op(SHL, node, sh), sh)

    def _rule(self, name: str, op: PcodeOp, av: int, at: int,  # noqa: C901
              bv: int, bt: int,
              isz: int, obits: int, ibits: int,
              om: int) -> tuple[int, int]:
        p = self.p
        M = p.const(om)

        # ---- movement and bit-parallel logic: taint routes exactly ----
        if name == 'COPY':
            return p.mask(av, obits), p.mask(at, obits)
        if name == 'INT_ZEXT':
            return p.mask(av, obits), p.mask(at, obits)
        if name == 'INT_SEXT':
            v = p.mask(self._sext(av, ibits), obits)
            # The taint of the replicated sign is the sign bit's own taint.
            sign_t = p.mask(p.op(SHR, at, p.const(ibits - 1)), 1)
            fill = p.op(AND, p.splat(sign_t), p.const(om ^ _mask_of(isz)))
            return v, p.op(OR, p.mask(at, ibits), fill)
        if name == 'INT_NEGATE':
            return p.mask(p.op(NOT, av), obits), p.mask(at, obits)
        if name == 'INT_XOR' or name == 'BOOL_XOR':
            return p.op(XOR, av, bv), p.op(OR, at, bt)
        if name == 'BOOL_NEGATE':
            return p.op(XOR, av, p.const(1)), at
        if name == 'SUBPIECE':
            sh = p.const(op.inputs[1].offset * 8)
            return (p.mask(p.op(SHR, av, sh), obits),
                    p.mask(p.op(SHR, at, sh), obits))
        if name == 'PIECE':
            lo_bits = op.inputs[1].size * 8
            sh = p.const(lo_bits)
            return (p.mask(p.op(OR, p.op(SHL, av, sh), bv), obits),
                    p.mask(p.op(OR, p.op(SHL, at, sh), bt), obits))

        # ---- value-aware boolean: masking is exact per bit ----
        if name in ('INT_AND', 'BOOL_AND'):
            t = p.op(OR, p.op(OR, p.op(AND, at, bv), p.op(AND, bt, av)),
                     p.op(AND, at, bt))
            return p.op(AND, av, bv), p.mask(t, obits)
        if name in ('INT_OR', 'BOOL_OR'):
            t = p.op(OR, p.op(OR, p.op(AND, at, p.op(NOT, bv)),
                              p.op(AND, bt, p.op(NOT, av))),
                     p.op(AND, at, bt))
            return p.op(OR, av, bv), p.mask(t, obits)

        # ---- shifts ----
        if name in ('INT_LEFT', 'INT_RIGHT', 'INT_SRIGHT'):
            return self._shift(name, av, at, bv, bt, ibits, om)

        # ---- carry-coupled: the differential, inlined and exact ----
        if name in ('INT_ADD', 'INT_SUB', 'INT_2COMP'):
            if name == 'INT_2COMP':
                bv, bt, av, at = av, at, p.const(0), p.const(0)
                name = 'INT_SUB'
            lo, hi = self._corners(name, av, at, bv, bt, om)
            v = (p.mask(p.op(ADD, av, bv), obits) if name == 'INT_ADD'
                 else p.mask(p.op(SUB, av, bv), obits))
            t = p.mask(p.op(OR, p.op(XOR, lo, hi), p.op(OR, at, bt)), obits)
            if name == 'INT_SUB':
                # Remember what this difference was a difference OF, so that
                # `(a - b) == 0` can be answered as `a == b`.  See the equality
                # rule below.  Keyed on the VALUE NODE rather than on the
                # varnode it lands in: a node id is a fixed expression, so the
                # entry cannot go stale when a `unique` offset is reused, which
                # pypcode does within and across instructions.
                self._diff_of[v] = (av, at, bv, bt, obits)
            return v, t

        if name == 'INT_CARRY':
            s = p.mask(p.op(ADD, av, bv), ibits)
            v = p.op(ULT, s, p.mask(av, ibits))
            lo, hi = self._corners('INT_ADD', av, at, bv, bt, _mask_of(isz))
            c_lo = p.op(ULT, lo, p.op(AND, av, p.op(NOT, at)))
            c_hi = p.op(ULT, hi, p.op(OR, av, at))
            return v, p.op(XOR, c_lo, c_hi)

        if name in ('INT_SCARRY', 'INT_SBORROW'):
            return self._overflow(name, av, at, bv, bt, ibits)

        # ---- comparisons: exact from the monotone corners ----
        if name in ('INT_EQUAL', 'INT_NOTEQUAL'):
            # `(a - b) == 0` IS `a == b`, exactly, in wrapping arithmetic, and
            # the second form is the one this rule can prove things about.
            #
            # It matters because no lifter writes the first form by choice: a
            # comparison sets the zero flag, and SLEIGH models that as
            # `INT_EQUAL(INT_SUB(a, b), 0)`.  Asked about the DIFFERENCE, the
            # rule below can only prove inequality when some bit of the
            # difference is provably one; asked about the OPERANDS it can prove
            # it whenever they differ in any bit neither side can change, which
            # is a far weaker condition.
            #
            # Measured on `movzx eax,al; cmp eax,-1`, glibc's EOF test: EAX
            # holds a byte so bits 8-31 are clean zero, and -1 has them set, so
            # the operands can never be equal.  As a difference it is `eax + 1`
            # in [1, 0x100], where no single bit is provably one and the rule
            # gives up, and the branch was reported as secret-dependent.
            if p.is_const(bv) and p.const_val(bv) == 0:
                got = self._diff_of.get(av)
                if got is not None and got[4] == ibits:
                    av, at, bv, bt = got[0], got[1], got[2], got[3]
            eq = p.op(EQ, av, bv)
            v = eq if name == 'INT_EQUAL' else p.op(XOR, eq, p.const(1))
            tu = p.op(OR, at, bt)
            can_true = p.op(EQ, p.op(AND, p.op(XOR, av, bv), p.op(NOT, tu)),
                            p.const(0))
            can_false = p.op(NEZ, p.op(OR, tu, p.op(XOR, av, bv)))
            return v, p.op(AND, can_true, can_false)

        if name in ('INT_LESS', 'INT_LESSEQUAL', 'INT_SLESS', 'INT_SLESSEQUAL'):
            return self._compare(name, av, at, bv, bt, ibits)

        # ---- multiply: exact only for a lift-time power of two ----
        if name == 'INT_MULT':
            v = p.mask(p.op(MUL, av, bv), obits)
            if p.is_const(bv) or p.is_const(av):
                k = p.const_val(bv) if p.is_const(bv) else p.const_val(av)
                other_t = bt if p.is_const(av) else at
                if k != 0 and (k & (k - 1)) == 0:
                    sh = k.bit_length() - 1
                    return v, p.mask(p.op(SHL, other_t, p.const(sh)), obits)
            return v, p.op(AND, M, p.splat(p.op(NEZ, p.op(OR, at, bt))))

        if name in ('INT_DIV', 'INT_SDIV', 'INT_REM', 'INT_SREM'):
            v = self._divide(name, av, bv, ibits, obits)
            return v, p.op(AND, M, p.splat(p.op(NEZ, p.op(OR, at, bt))))

        if name in ('POPCOUNT', 'LZCOUNT'):
            # Both count something about the operand, so the answer is a small
            # number and the honest question is which values it can take.  The
            # taint cube has two CORNERS -- every tainted bit cleared, and every
            # one set -- and for a counter they bracket the whole reachable
            # range: popcount is monotone in each bit, and a leading-zero count
            # is antitone in the value.  So evaluate the count at both corners,
            # and the bits that can differ are the bits below where they first
            # differ.  The same trick `_corners` plays for a ripple carry.
            #
            # This replaces "any tainted input bit taints the whole span", which
            # was sound and very loose.  It is loose in two ways that matter:
            #
            #  * a LEADING-ZERO count does not depend on bits below the highest
            #    set one at all, so taint down there moved nothing and was
            #    reported anyway.  Measured on `lzcnt eax,ebx`, 62 of 256
            #    outputs over-tainted; with this, far fewer.
            #  * a POPCOUNT that can move by one or two moves only its bottom
            #    bit or two, never all seven.
            #
            # POPCOUNT is not a rare opcode: it is how SLEIGH computes x86's
            # PARITY flag, so this rule is on the path of every arithmetic and
            # logic instruction there.  LZCOUNT carries `lzcnt`, AArch64
            # `clz`/`cls`, MIPS `clz`/`clo`/`dclz`/`dclo` and PowerPC `cntlzw`.
            x = p.mask(av, ibits)
            t = p.mask(at, ibits)

            def count(node: int) -> int:
                if name == 'POPCOUNT':
                    return p.op(POPCNT, node)
                return p.op(SUB, p.op(CLZ, node), p.const(64 - ibits))

            v = count(x)
            lo = count(p.op(AND, x, p.op(NOT, t)))
            hi = count(p.op(OR, x, t))
            nb = max(1, (ibits).bit_length())
            span = p.const(((1 << nb) - 1) & om)
            return v, p.op(AND, span, _smear_right(p, p.op(XOR, lo, hi), nb))

        raise Unsupported(name)

    def _corners(self, name: str, av: int, at: int, bv: int, bt: int,
                 om: int) -> tuple[int, int]:
        """The two extremal evaluations, masked to the operand width.

        Monotonicity is what makes them sufficient: a ripple carry is monotone
        in every input bit, so a carry can vary over the taint cube exactly when
        it differs between these two.  Subtraction is monotone the other way in
        its right operand, hence the crossed corners.
        """
        p = self.p
        M = p.const(om)
        a_lo = p.op(AND, p.op(AND, av, p.op(NOT, at)), M)
        a_hi = p.op(AND, p.op(OR, av, at), M)
        b_lo = p.op(AND, p.op(AND, bv, p.op(NOT, bt)), M)
        b_hi = p.op(AND, p.op(OR, bv, bt), M)
        if name == 'INT_ADD':
            return (p.op(AND, p.op(ADD, a_lo, b_lo), M),
                    p.op(AND, p.op(ADD, a_hi, b_hi), M))
        return (p.op(AND, p.op(SUB, a_lo, b_hi), M),
                p.op(AND, p.op(SUB, a_hi, b_lo), M))

    def _shift(self, name: str, av: int, at: int, bv: int, bt: int,
               ibits: int, om: int) -> tuple[int, int]:
        p = self.p
        M = p.const(om)
        amt = bv
        wide = p.op(ULT, amt, p.const(64))

        def do(node: int, arith: bool) -> int:
            if arith:
                sh = self._sext(node, ibits)
                r = p.op(SAR, sh, amt)
            else:
                r = (p.op(SHL, node, amt) if name == 'INT_LEFT'
                     else p.op(SHR, p.mask(node, ibits), amt))
            r = p.op(AND, r, M)
            zero = (p.op(AND, self._sext(node, ibits), M) if arith
                    else p.const(0))
            return p.op(SEL, r, zero, wide)

        arith = (name == 'INT_SRIGHT')
        v = do(av, arith)
        # Taint routes through the same shift; for an arithmetic shift the sign
        # bit's taint replicates into the vacated top, which is exactly what
        # shifting the sign-extended taint word does.
        t_routed = do(at, arith)
        if p.is_const(bt) and p.const_val(bt) == 0:
            return v, t_routed
        # A tainted shift amount can move any live source bit anywhere.
        whole = p.op(AND, M, p.splat(p.op(NEZ, p.op(OR, at, av))))
        return v, p.op(SEL, whole, t_routed, p.op(NEZ, bt))

    def _compare(self, name: str, av: int, at: int, bv: int, bt: int,
                 ibits: int) -> tuple[int, int]:
        """`a < b` is monotone -- decreasing in a, increasing in b -- so it can
        be true iff min(a) < max(b) and false iff max(a) >= min(b), and it is
        tainted iff both.  Signed order becomes unsigned by flipping the sign
        bit, which maps the taint box to another box, so the corners survive."""
        p = self.p
        signed = name in ('INT_SLESS', 'INT_SLESSEQUAL')
        or_equal = name in ('INT_LESSEQUAL', 'INT_SLESSEQUAL')
        m = _mask_of(ibits // 8)
        M = p.const(m)
        if signed:
            sb = p.const(1 << (ibits - 1))
            av2 = p.op(AND, p.op(XOR, av, sb), M)
            bv2 = p.op(AND, p.op(XOR, bv, sb), M)
        else:
            av2, bv2 = p.op(AND, av, M), p.op(AND, bv, M)
        a_min = p.op(AND, av2, p.op(NOT, at))
        a_max = p.op(AND, p.op(OR, av2, at), M)
        b_min = p.op(AND, bv2, p.op(NOT, bt))
        b_max = p.op(AND, p.op(OR, bv2, bt), M)
        if or_equal:
            v = p.op(XOR, p.op(ULT, bv2, av2), p.const(1))
            can_true = p.op(XOR, p.op(ULT, b_max, a_min), p.const(1))
            can_false = p.op(ULT, b_min, a_max)
        else:
            v = p.op(ULT, av2, bv2)
            can_true = p.op(ULT, a_min, b_max)
            can_false = p.op(XOR, p.op(ULT, a_max, b_min), p.const(1))
        return v, p.op(AND, can_true, can_false)

    def _overflow(self, name: str, av: int, at: int, bv: int, bt: int,
                  ibits: int) -> tuple[int, int]:
        """Signed overflow, exactly, through the six-bit table.

        The three inputs are the two operand sign bits and the carry into the
        sign position; the carry's own variability comes from the same corner
        sums the addition already computed, recovered as
        `carry_in_msb = sum ^ a ^ b` at the sign bit.  For a subtraction the
        second operand enters complemented, which is the only difference.
        """
        p = self.p
        sub = (name == 'INT_SBORROW')
        m = _mask_of(ibits // 8)
        sh = p.const(ibits - 1)
        one = p.const(1)

        bb = p.op(AND, p.op(NOT, bv), p.const(m)) if sub else bv
        lo, hi = self._corners('INT_SUB' if sub else 'INT_ADD',
                               av, at, bv, bt, m)
        if sub:
            # a - b == a + ~b + 1: recover the same sum sequence so the carry
            # extraction below is the one that belongs to this operand pair.
            b_lo = p.op(AND, p.op(NOT, p.op(AND, p.op(OR, bv, bt), p.const(m))),
                        p.const(m))
            b_hi = p.op(AND, p.op(NOT, p.op(AND, bv, p.op(NOT, bt))),
                        p.const(m))
            a_lo = p.op(AND, p.op(AND, av, p.op(NOT, at)), p.const(m))
            a_hi = p.op(AND, p.op(OR, av, at), p.const(m))
            s_lo = p.op(AND, p.op(ADD, p.op(ADD, a_lo, b_lo), one), p.const(m))
            s_hi = p.op(AND, p.op(ADD, p.op(ADD, a_hi, b_hi), one), p.const(m))
            c_lo = self._carry_in_msb(s_lo, a_lo, b_lo, sh)
            c_hi = self._carry_in_msb(s_hi, a_hi, b_hi, sh)
            val_b = bb
        else:
            a_lo = p.op(AND, p.op(AND, av, p.op(NOT, at)), p.const(m))
            a_hi = p.op(AND, p.op(OR, av, at), p.const(m))
            b_lo = p.op(AND, p.op(AND, bv, p.op(NOT, bt)), p.const(m))
            b_hi = p.op(AND, p.op(OR, bv, bt), p.const(m))
            c_lo = self._carry_in_msb(lo, a_lo, b_lo, sh)
            c_hi = self._carry_in_msb(hi, a_hi, b_hi, sh)
            val_b = bv

        # The concrete flag needs the concrete carry; the CUBE only needs one
        # of its two corners as the representative value, and c_lo is already
        # computed, so nothing extra is emitted for the taint answer.  (Either
        # corner works: when they agree the concrete carry equals them, and when
        # they differ the cube is {0,1} and the value bit is irrelevant.)
        s = p.op(AND, p.op(ADD, p.op(ADD, av, bb), one if sub else p.const(0)),
                 p.const(m))
        c_v = self._carry_in_msb(s, av, bb, sh)
        a_s = p.mask(p.op(SHR, av, sh), 1)
        b_s = p.mask(p.op(SHR, val_b, sh), 1)
        of = p.op(AND, p.op(XOR, p.op(XOR, a_s, b_s), one),
                  p.op(XOR, a_s, c_v))

        a_t = p.mask(p.op(SHR, at, sh), 1)
        b_t = p.mask(p.op(SHR, bt, sh), 1)
        c_t = p.op(XOR, c_lo, c_hi)
        c_v = c_lo
        idx = p.op(OR, p.op(OR, p.op(OR, a_s, p.op(SHL, a_t, p.const(1))),
                            p.op(OR, p.op(SHL, b_s, p.const(2)),
                                 p.op(SHL, b_t, p.const(3)))),
                   p.op(OR, p.op(SHL, c_v, p.const(4)),
                        p.op(SHL, c_t, p.const(5))))
        t = p.op(AND, p.op(SHR, p.const(OF_TAINT_TABLE), idx), one)
        return of, t

    def _carry_in_msb(self, s: int, a: int, b: int, sh: int) -> int:
        """sum_i = a_i ^ b_i ^ carry_i, so the carry into any position is
        recoverable from the sum without a second ripple."""
        p = self.p
        return p.mask(p.op(SHR, p.op(XOR, p.op(XOR, s, a), b), sh), 1)

    def _divide(self, name: str, av: int, bv: int, ibits: int, obits: int) -> int:
        p = self.p
        if name in ('INT_DIV', 'INT_REM'):
            a = p.mask(av, ibits)
            b = p.mask(bv, ibits)
            r = p.op(UDIV if name == 'INT_DIV' else UREM, a, b)
        else:
            a = self._sext(av, ibits)
            b = self._sext(bv, ibits)
            r = p.op(SDIV if name == 'INT_SDIV' else SREM, a, b)
        return p.mask(r, obits)


_BUILDERS: dict[tuple[str, str], Builder] = {}

#: Held for the whole of a lowering.  A Builder is SHARED and stateful -- the
#: predicate stack, the unroll bookkeeping and the loop cache all live on it --
#: so two threads lowering at once walk over each other's state.  Measured: six
#: threads lowering nine short AMD64 blocks raised `IndexError` out of
#: `prog.live[n]` on roughly one run in three, because one thread's
#: `_address_slice` had re-rooted the program and emptied `live` while the other
#: was still reading it.  Nothing crashed and nothing was wrong in a
#: single-threaded run, which is why this went unnoticed: the engine drives one
#: emulator per process today.  Lowering is once per distinct block per process
#: now that compiled blocks are cached, so serialising it costs nothing and
#: removes a whole class of failure from anything that runs several guests at
#: once -- which is what a fuzzer does.
BUILDER_LOCK = threading.Lock()


def builder_for(arch: ArchLike,
                pointer_policy: PointerPolicy = DEFAULT_POINTER_POLICY) -> Builder:
    """The shared Builder for `arch` under `pointer_policy`, made once.

    A Builder costs a register-map build, so callers share one; several of them
    used to reach into `_BUILDERS` with the policy spelled out, which meant that
    changing the default policy raised KeyError in four separate places rather
    than doing the one thing it was supposed to do.  Ask here instead.

    Shared means shared: a caller that LOWERS through the returned Builder must
    hold `BUILDER_LOCK` while it does.
    """
    key = arch.value if hasattr(arch, 'value') else str(arch)
    b = _BUILDERS.get((key, pointer_policy))
    if b is None:
        b = Builder(arch, key.endswith('BE'), pointer_policy)
        _BUILDERS[(key, pointer_policy)] = b
    return b


def build_ir(arch: ArchLike, code: bytes, *,
             pointer_policy: PointerPolicy = DEFAULT_POINTER_POLICY,
             emit: Emit = Emit.TAINT) -> IRProg:
    """Lower one instruction to a taint IR program.  Raises Unsupported."""
    from microtaint.sleigh.lifter import get_context
    key = arch.value if hasattr(arch, 'value') else str(arch)
    b = builder_for(arch, pointer_policy)
    ops = get_context(key).translate(code, LIFT_BASE).ops
    return b.build(ops, LIFT_BASE + len(code), emit=emit)
