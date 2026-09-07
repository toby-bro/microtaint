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
    SLT,
    SREM,
    SUB,
    UDIV,
    ULT,
    UREM,
    XOR,
    IRProg,
)

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
    """This p-code shape is outside the lowering; the caller falls back."""


def _mask_of(size: int) -> int:
    return MASK64 if size >= 8 else ((1 << (size * 8)) - 1)


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

    def __init__(self, prog: IRProg, declared: dict, be: bool, kind: str):
        self.p = prog
        self.be = be
        self.kind = kind                 # 'v' or 't', for input node naming
        self.declared = declared         # offset -> (name, size)
        self.reg: dict = {}              # offset -> (node, size)
        self.uniq: dict = {}
        self.touched: set = set()        # register offsets this program wrote
        #: Offsets a wider write has subsumed.  Their declared input node is
        #: stale from that point on -- writing RAX must be visible through AH --
        #: so a read has to resolve them through the parent instead.
        self.killed: set = set()

    def _space(self, sp):
        if sp == 'register':
            return self.reg
        if sp == 'unique':
            return self.uniq
        raise Unsupported(f'space {sp}')

    def _cell(self, sp, off):
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

    def read(self, sp, off, size):
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

    def write(self, sp, off, size, node):
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


class Builder:
    """Lower one instruction to IR.  `prog.outputs` ends up holding the taint of
    every architectural register, written or passed through."""

    def __init__(self, arch, be: bool):
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
        self.offset_of: dict = {}
        for name, off in offsets.items():
            size = sizes.get(name, 8)
            self.offset_of.setdefault(name, off)
            if size > 8 or size <= 0:
                continue
            candidates.append((off, -size, name))
        candidates.sort()
        declared: dict = {}
        self.name_by_off: dict = {}
        covered: set = set()
        for off, negsz, name in candidates:
            size = -negsz
            if any(b in covered for b in range(off, off + size)):
                continue
            covered.update(range(off, off + size))
            declared[off] = (name, size)
            self.name_by_off[off] = name
        self.declared = declared
        self.names = {n for n, _s in declared.values()}
        self.be = be
        self.arch = arch

    def build(self, ops, end_addr: int) -> IRProg:
        p = IRProg()
        v = SymFrame(p, self.declared, self.be, 'v')
        t = SymFrame(p, self.declared, self.be, 't')
        self.p, self.v, self.t = p, v, t
        #: (opcode, first_node, last_node) per p-code op, so the emitted cost
        #: can be attributed back to the lifter construct that caused it.
        #: Approximate by construction: a hash-consed node is credited to
        #: whichever op created it first.
        self.spans: list = []
        #: One entry per distinct (address, size) the instruction touches.
        #: SLEIGH re-emits the same LOAD once per flag that reads it, so
        #: de-duplicating here turns `add rax, [rbx+16]`'s three identical
        #: loads into one access.
        self.accesses: list = []
        self.access_map: dict = {}
        # Predicate stack: (until_pc, saved_pred_value, saved_pred_taint).
        self.pred_v = p.const(1)
        self.pred_t = p.const(0)
        pred_stack: list = []

        n = len(ops)
        # SLEIGH names an intra-instruction branch target by ADDRESS, not by
        # p-code index: the end of the instruction for a skip, an IMARK for a
        # jump back into the sequence.  Resolve both to indices up front.
        self.end_addr = end_addr
        self.imark_pc = {}
        for i, o in enumerate(ops):
            if o.opcode.name == 'IMARK' and o.inputs:
                self.imark_pc.setdefault(o.inputs[0].offset, i)
        self.n_ops = n
        for pc in range(n):
            while pred_stack and pred_stack[-1][0] == pc:
                _u, self.pred_v, self.pred_t = pred_stack.pop()
            op = ops[pc]
            name = op.opcode.name
            if name in ('IMARK', 'INDIRECT', 'MULTIEQUAL', 'CAST', 'CPOOLREF',
                        'NEW', 'SEGMENTOP'):
                continue
            if name == 'CBRANCH':
                target = self._branch_target(op, pc, n)
                if target is None or target <= pc:
                    raise Unsupported('backward or unresolved CBRANCH')
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
                if target is None or target <= pc:
                    raise Unsupported('backward or unresolved BRANCH')
                # An unconditional forward branch inside the instruction: the
                # skipped region simply never runs under this predicate.
                pred_stack.append((target, self.pred_v, self.pred_t))
                self.pred_v = p.const(0)
                self.pred_t = self.pred_t
                continue
            if name in ('BRANCHIND', 'CALL', 'CALLIND', 'RETURN'):
                raise Unsupported(name)
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
        dirty_bytes = set()
        for off in t.touched:
            cell = t.reg.get(off)
            width = cell[1] if cell else 1
            dirty_bytes.update(range(off, off + width))
        for off in sorted(self.declared):
            _nm, size = self.declared[off]
            if not dirty_bytes.intersection(range(off, off + size)):
                continue
            p.outputs.append((('reg', off, size), t.read('register', off, size)))
        p.spans = self.spans
        p.accesses = self.accesses
        self._check_address_independence(p)
        return p.finish()

    # -- opaque operations ---------------------------------------------
    def _emit_callother(self, ops, pc, op):
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
        """
        p = self.p
        if op.output is None:
            return
        osz = op.output.size
        if osz > 8:
            raise Unsupported('wide CALLOTHER')
        lo, hi = op.output.offset, op.output.offset + osz
        space = op.output.space.name
        for later in ops[pc + 1:]:
            for vn in later.inputs:
                if vn.space.name == space and vn.offset < hi and vn.offset + vn.size > lo:
                    raise Unsupported('CALLOTHER result is read downstream')
        any_t = p.const(0)
        for vn in op.inputs[1:]:
            if vn.space.name == 'const':
                continue
            _v, tt = self._read_in(vn)
            any_t = p.op(OR, any_t, tt)
        tnt = p.op(AND, p.const(_mask_of(osz)), p.splat(p.op(NEZ, any_t)))
        self._predicated_write(op.output, p.const(0), tnt)

    # -- memory --------------------------------------------------------
    def _access_for(self, kind, addr_node, size):
        key = (kind, addr_node, size)
        k = self.access_map.get(key)
        if k is None:
            k = len(self.accesses)
            self.access_map[key] = k
            self.accesses.append({'kind': kind, 'size': size, 'addr': addr_node})
            self.p.outputs.append((('addr', k), addr_node))
        return k

    def _emit_load(self, op):
        """A load's value and shadow taint enter as INPUTS at a slot the caller
        fills once the address has been computed.

        A tainted address is not a decline: which bytes are read is then itself
        secret-dependent, so the loaded word is fully tainted -- sound, and it
        keeps the common clean-pointer case exact.
        """
        p = self.p
        size = op.output.size
        if size > 8:
            raise Unsupported('wide load')
        addr_v, addr_t = self._read_in(op.inputs[1])
        k = self._access_for('load', addr_v, size)
        p.outputs.append((('addrt', k), addr_t))
        val = p.input_value(('mem', k), size * 8)
        mem_t = p.input_taint(('mem', k), size * 8)
        tnt = p.op(OR, mem_t,
                   p.op(AND, p.const(_mask_of(size)),
                        p.splat(p.op(NEZ, addr_t))))
        self._predicated_write(op.output, val, tnt)

    def _emit_store(self, op):
        """A store contributes the taint to write and the address to write it
        at, both as outputs; committing them is the caller's job.

        A store through a TAINTED address writes somewhere secret-dependent,
        which is a different and much larger obligation than tainting a known
        location -- the address taint is reported so the caller can hand those
        instructions to the slow path rather than silently writing one place.
        """
        p = self.p
        size = op.inputs[2].size
        if size > 8:
            raise Unsupported('wide store')
        if not (p.is_const(self.pred_v) and p.const_val(self.pred_v) == 1):
            raise Unsupported('predicated store')
        addr_v, addr_t = self._read_in(op.inputs[1])
        _val_v, val_t = self._read_in(op.inputs[2])
        k = self._access_for('store', addr_v, size)
        p.outputs.append((('addrt', k), addr_t))
        p.outputs.append((('sttaint', k), p.mask(val_t, size * 8)))

    def _check_address_independence(self, p):
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
                    raise Unsupported('address depends on a loaded value')
                _o, a, b, c, _imm = p.nodes[n]
                for x in (a, b, c):
                    if x >= 0:
                        stack.append(x)

    # -- helpers -------------------------------------------------------
    def _branch_target(self, op, pc, n):
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
                return n                      # skip to the end of the instruction
            return self.imark_pc.get(vn.offset)
        return None

    def _read_in(self, vn):
        p = self.p
        sp = vn.space.name
        if sp == 'const':
            return p.const(vn.offset & _mask_of(vn.size)), p.const(0)
        if sp in ('register', 'unique'):
            return (self.v.read(sp, vn.offset, vn.size),
                    self.t.read(sp, vn.offset, vn.size))
        raise Unsupported(f'input space {sp}')

    def _predicated_write(self, vn, val, tnt):
        """Commit one output under the current predicate.

        With a KNOWN predicate this folds to a plain write (the constant SEL
        disappears).  With an UNKNOWN but untainted predicate it becomes a
        select -- exact, and branch-free, which is the point.

        With a TAINTED predicate it is implicit flow: which value lands is
        itself secret, so the result carries both sides' taint plus every bit on
        which the two sides differ.  That is the exact join, not a floor, and it
        is what makes predicated instructions (cmov, csel, csinc) and a lifter's
        conditional flag macros correct without a fallback.
        """
        p = self.p
        sp = vn.space.name
        if sp not in ('register', 'unique'):
            raise Unsupported(f'output space {sp}')
        if p.is_const(self.pred_v) and p.const_val(self.pred_v) == 1 \
                and p.is_const(self.pred_t) and p.const_val(self.pred_t) == 0:
            self.v.write(sp, vn.offset, vn.size, val)
            self.t.write(sp, vn.offset, vn.size, tnt)
            return
        old_v = self.v.read(sp, vn.offset, vn.size)
        old_t = self.t.read(sp, vn.offset, vn.size)
        new_v = p.op(SEL, val, old_v, self.pred_v)
        new_t = p.op(SEL, tnt, old_t, self.pred_t if False else self.pred_v)
        leak = p.op(AND, p.splat(self.pred_t),
                    p.op(OR, p.op(OR, tnt, old_t), p.op(XOR, val, old_v)))
        self.v.write(sp, vn.offset, vn.size, new_v)
        self.t.write(sp, vn.offset, vn.size, p.op(OR, new_t, leak))

    # -- wide varnodes -------------------------------------------------
    #: Opcodes whose every output bit depends only on the input bits in the
    #: same position, so a varnode wider than a machine word can be done one
    #: 8-byte lane at a time with no loss.  Carry-coupled and position-sensitive
    #: opcodes are deliberately absent: splitting those would drop the coupling
    #: between lanes and under-taint.
    _LANE_OPS = {'COPY', 'INT_ZEXT', 'INT_SEXT', 'INT_NEGATE',
                 'INT_AND', 'INT_OR', 'INT_XOR'}

    def _read_lane(self, vn, lane, lsz):
        """One 8-byte lane of a varnode, as (value, taint)."""
        p = self.p
        sp = vn.space.name
        if sp == 'const':
            return p.const((vn.offset >> (8 * lane)) & _mask_of(lsz)), p.const(0)
        if sp in ('register', 'unique'):
            return (self.v.read(sp, vn.offset + lane, lsz),
                    self.t.read(sp, vn.offset + lane, lsz))
        raise Unsupported(f'wide input space {sp}')

    def _emit_wide_special(self, name, op):
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
        osz = op.output.size
        isz = op.inputs[0].size if op.inputs else osz

        # A lane-aligned slice of a wide value is just that lane.
        if name == 'SUBPIECE' and isz > 8 and osz <= 8:
            off = op.inputs[1].offset
            if off % 8 or off + osz > isz:
                return False
            av, at = self._read_lane(op.inputs[0], off, osz)
            self._predicated_write(op.output, p.mask(av, osz * 8),
                                   p.mask(at, osz * 8))
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
            self._write_lane(op.output, 0, 8, lo, full)
            self._write_lane(op.output, 8, 8, hi, full)
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
            self._predicated_write(op.output, v,
                                   p.op(AND, can_true, can_false))
            return True

        return False

    def _write_lane(self, vn, lane, lsz, val, tnt):
        p = self.p
        sp = vn.space.name
        if sp not in ('register', 'unique'):
            raise Unsupported(f'wide output space {sp}')
        if not (p.is_const(self.pred_v) and p.const_val(self.pred_v) == 1):
            raise Unsupported('predicated wide write')
        self.v.write(sp, vn.offset + lane, lsz, val)
        self.t.write(sp, vn.offset + lane, lsz, tnt)

    def _emit_wide(self, name, op):
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
        osz = op.output.size
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
                v, tnt = ((p.const(0), p.const(0)) if name == 'INT_ZEXT'
                          else (fill_v, fill_t))
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
            self._write_lane(op.output, lane, lsz,
                             p.mask(v, lsz * 8), p.mask(tnt, lsz * 8))

    # -- the per-opcode lowering ---------------------------------------
    def _emit_op(self, name, op):
        p = self.p
        if any(s > 8 for s in
               [x.size for x in op.inputs] + [op.output.size]):
            if name in self._LANE_OPS:
                self._emit_wide(name, op)
                return
            if self._emit_wide_special(name, op):
                return
            raise Unsupported(f'wide varnode ({name})')
        ins = [self._read_in(x) for x in op.inputs]
        av, at = ins[0] if ins else (p.const(0), p.const(0))
        bv, bt = ins[1] if len(ins) > 1 else (p.const(0), p.const(0))
        osz = op.output.size
        isz = op.inputs[0].size if op.inputs else osz
        obits = osz * 8
        ibits = isz * 8
        om = _mask_of(osz)

        val, tnt = self._rule(name, op, av, at, bv, bt, osz, isz, obits, ibits, om)
        self._predicated_write(op.output, val, tnt)

    def _sext(self, node, bits):
        p = self.p
        if bits >= 64:
            return node
        sh = p.const(64 - bits)
        return p.op(SAR, p.op(SHL, node, sh), sh)

    def _rule(self, name, op, av, at, bv, bt, osz, isz, obits, ibits, om):
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
            return self._shift(name, av, at, bv, bt, obits, ibits, om)

        # ---- carry-coupled: the differential, inlined and exact ----
        if name in ('INT_ADD', 'INT_SUB', 'INT_2COMP'):
            if name == 'INT_2COMP':
                bv, bt, av, at = av, at, p.const(0), p.const(0)
                name = 'INT_SUB'
            lo, hi = self._corners(name, av, at, bv, bt, om)
            v = (p.mask(p.op(ADD, av, bv), obits) if name == 'INT_ADD'
                 else p.mask(p.op(SUB, av, bv), obits))
            t = p.mask(p.op(OR, p.op(XOR, lo, hi), p.op(OR, at, bt)), obits)
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
            if name == 'POPCOUNT':
                v = p.op(POPCNT, p.mask(av, ibits))
            else:
                v = p.op(SUB, p.op(CLZ, p.mask(av, ibits)), p.const(64 - ibits))
            nb = max(1, (ibits).bit_length())
            span = p.const(((1 << nb) - 1) & om)
            return v, p.op(AND, span, p.splat(p.op(NEZ, at)))

        raise Unsupported(name)

    def _corners(self, name, av, at, bv, bt, om):
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

    def _shift(self, name, av, at, bv, bt, obits, ibits, om):
        p = self.p
        M = p.const(om)
        amt = bv
        wide = p.op(ULT, amt, p.const(64))

        def do(node, arith):
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

    def _compare(self, name, av, at, bv, bt, ibits):
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

    def _overflow(self, name, av, at, bv, bt, ibits):
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

    def _carry_in_msb(self, s, a, b, sh):
        """sum_i = a_i ^ b_i ^ carry_i, so the carry into any position is
        recoverable from the sum without a second ripple."""
        p = self.p
        return p.mask(p.op(SHR, p.op(XOR, p.op(XOR, s, a), b), sh), 1)

    def _divide(self, name, av, bv, ibits, obits):
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


_BUILDERS: dict = {}


def build_ir(arch, code: bytes):
    """Lower one instruction to a taint IR program.  Raises Unsupported."""
    from microtaint.sleigh.lifter import get_context
    key = arch.value if hasattr(arch, 'value') else str(arch)
    b = _BUILDERS.get(key)
    if b is None:
        b = Builder(arch, key.endswith('BE'))
        _BUILDERS[key] = b
    ops = get_context(key).translate(code, LIFT_BASE).ops
    return b.build(ops, LIFT_BASE + len(code))
