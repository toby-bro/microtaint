"""A straight-line 64-bit SSA IR for taint propagation.

Why an IR at all
----------------
The per-op composer in `taint_core.h` decides, at RUN time and per p-code op,
which taint rule applies and how wide the operands are.  Almost none of that
depends on runtime state: an instruction's p-code shape and its constant
operands are fixed the moment it is lifted.  `shl rax, 15` spends most of its
217 operations evaluating flag macros whose guards are `INT_AND(#0xf, #0x3f)`,
`(15 != 0)` and `(15 == 1)` -- all constant, all re-decided on every execution.

Lowering the whole instruction, values and taint together, into one branch-free
SSA program lets those collapse once, at lift time.  What is left is the
irreducible work, and that -- not the interpreted count -- is what a native
compiler for the taint circuit would have to emit.

The instruction set
-------------------
Deliberately small and machine-shaped: every op maps to one or two instructions
on x86-64 and AArch64 alike, so the same IR lowers to either without an
ISA-specific path.  Everything is 64 bits; widths live in explicit masks, which
fold away when they are redundant.

Structural guarantees the builder maintains, because they are what make the IR
compilable:

  * straight-line -- no branches.  P-code's intra-instruction control flow is
    turned into predication (see SEL), so a conditional move and a shift's
    flag macro are both one basic block.
  * SSA -- a node is written once.  Register aliasing is resolved in the
    builder, not at runtime.
  * hash-consed and constant-folded on construction, so building IS the first
    optimisation pass; a later DCE drops whatever no output reads.
"""
from __future__ import annotations

from microtaint.taint_ir import boolsynth as bs

MASK64 = 0xFFFFFFFFFFFFFFFF

# ── opcodes ──────────────────────────────────────────────────────────
CONST = 'const'
INV = 'inv'        # runtime input: a register's VALUE   (imm = slot key)
INT = 'int'        # runtime input: a register's TAINT   (imm = slot key)

AND = 'and'
OR = 'or'
XOR = 'xor'
ADD = 'add'
SUB = 'sub'
MUL = 'mul'
SHL = 'shl'
SHR = 'shr'        # logical right
SAR = 'sar'        # arithmetic right (operates on the full 64-bit word)
NOT = 'not'
NEG = 'neg'        # two's complement; `0 - x`, used to splat a 0/1 flag to a mask
ULT = 'ult'        # unsigned  a < b  -> 0/1
SLT = 'slt'        # signed    a < b  -> 0/1 (full 64-bit signed compare)
EQ = 'eq'          # a == b -> 0/1
NEZ = 'nez'        # a != 0 -> 0/1
SEL = 'sel'        # c ? a : b   (c is 0/1)
POPCNT = 'popcnt'
CLZ = 'clz'
UDIV = 'udiv'
UREM = 'urem'
SDIV = 'sdiv'
SREM = 'srem'
#: A one-bit cone, held as its truth table over at most three one-bit leaves
#: (a, b, c; imm = the table).  Kept symbolic until `finalize`, so the operations
#: a lifter used to build the cone are never emitted at all -- only a cheapest
#: expression for the function they compute.  See boolsynth.
BOOLSYM = 'boolsym'

_BINARY = {AND, OR, XOR, ADD, SUB, MUL, SHL, SHR, SAR, ULT, SLT, EQ,
           UDIV, UREM, SDIV, SREM}
_UNARY = {NOT, NEG, NEZ, POPCNT, CLZ}

#: Ops whose cost is one machine instruction on both x86-64 and AArch64.  SEL is
#: cmov/csel, NEZ is a compare plus a set, so they are charged 2.
_COST = {SEL: 2, NEZ: 2, ULT: 2, SLT: 2, EQ: 2,
         UDIV: 20, UREM: 20, SDIV: 20, SREM: 20, POPCNT: 1, CLZ: 1}


def _s64(v: int) -> int:
    v &= MASK64
    return v - (1 << 64) if v >> 63 else v


def eval_op(op: str, a: int, b: int, c: int, imm: int) -> int:
    """Evaluate one node from its already-evaluated inputs.

    Shared by constant folding and the reference interpreter so the two can
    never disagree about what a node means.
    """
    if op == CONST:
        return imm & MASK64
    if op == BOOLSYM:
        return (imm >> (a | (b << 1) | (c << 2))) & 1
    if op == AND:
        return a & b
    if op == OR:
        return a | b
    if op == XOR:
        return a ^ b
    if op == ADD:
        return (a + b) & MASK64
    if op == SUB:
        return (a - b) & MASK64
    if op == MUL:
        return (a * b) & MASK64
    if op == SHL:
        return (a << b) & MASK64 if b < 64 else 0
    if op == SHR:
        return (a >> b) if b < 64 else 0
    if op == SAR:
        s = _s64(a)
        return (s >> b) & MASK64 if b < 64 else (MASK64 if s < 0 else 0)
    if op == NOT:
        return (~a) & MASK64
    if op == NEG:
        return (-a) & MASK64
    if op == ULT:
        return 1 if a < b else 0
    if op == SLT:
        return 1 if _s64(a) < _s64(b) else 0
    if op == EQ:
        return 1 if a == b else 0
    if op == NEZ:
        return 1 if a else 0
    if op == SEL:
        return a if c else b
    if op == POPCNT:
        return bin(a).count('1')
    if op == CLZ:
        return 64 if a == 0 else 64 - a.bit_length()
    if op == UDIV:
        return 0 if b == 0 else (a // b) & MASK64
    if op == UREM:
        return 0 if b == 0 else (a % b) & MASK64
    if op == SDIV:
        if b == 0:
            return 0
        q = abs(_s64(a)) // abs(_s64(b))
        return (-q if (_s64(a) < 0) != (_s64(b) < 0) else q) & MASK64
    if op == SREM:
        if b == 0:
            return 0
        sa, sb = _s64(a), _s64(b)
        r = abs(sa) % abs(sb)
        return (-r if sa < 0 else r) & MASK64
    raise ValueError(f'unknown IR op {op}')


class IRProg:
    """A hash-consed, constant-folded straight-line program.

    Emission methods return node indices.  Because construction folds and
    hash-conses, the program that comes out of the builder is already free of
    duplicate subexpressions and of everything decidable at lift time; `finish`
    then drops what no output reads.
    """

    __slots__ = ('nodes', '_hc', 'inputs', 'outputs', 'live', 'kbits',
                 'spans', 'uses', 'no_bool')

    def __init__(self):
        self.nodes: list[tuple] = []        # (op, a, b, c, imm)
        self._hc: dict = {}
        self.inputs: dict = {}              # (kind, key) -> node
        self.outputs: list = []             # (key, node) for taint results
        self.live: list = []                # populated by finish()
        #: Upper bound on a node's significant bits.  Tracking it lets a
        #: redundant truncation disappear -- a lifter emits one after almost
        #: every op, and on a value that is already narrow they are pure cost.
        self.kbits: dict = {}
        self.spans: list = []
        #: How many consumers have been wired to each node so far.  Used to
        #: decide whether folding a one-bit cone actually retires its
        #: operands or merely duplicates work they still owe elsewhere.
        self.uses: dict = {}
        #: Set on the program `finalize` produces: one-bit cones have already
        #: been expanded into a cheapest expression there, and re-forming them
        #: would put BOOLSYM back into a program whose whole purpose is to
        #: contain only machine-shaped opcodes.
        self.no_bool = False

    # -- construction --------------------------------------------------
    def _emit(self, op, a=-1, b=-1, c=-1, imm=0) -> int:
        for x in (a, b, c):
            if x >= 0:
                self.uses[x] = self.uses.get(x, 0) + 1
        key = (op, a, b, c, imm)
        n = self._hc.get(key)
        if n is not None:
            return n
        n = len(self.nodes)
        self.nodes.append(key)
        self._hc[key] = n
        return n

    def const(self, v: int) -> int:
        n = self._emit(CONST, imm=v & MASK64)
        self.kbits[n] = (v & MASK64).bit_length()
        return n

    def known_bits(self, n: int) -> int:
        return self.kbits.get(n, 64)

    def _set_bits(self, n: int, op: str, a: int, b: int, c: int) -> None:
        """Propagate the significant-bit bound through the ops where it is
        cheap and exact to do so."""
        kb = self.kbits
        if op in (AND,):
            kb[n] = min(self.known_bits(a), self.known_bits(b))
        elif op in (OR, XOR):
            kb[n] = max(self.known_bits(a), self.known_bits(b))
        elif op == SHR:
            if self.nodes[b][0] == CONST:
                kb[n] = max(0, self.known_bits(a) - self.nodes[b][4])
        elif op == SHL:
            if self.nodes[b][0] == CONST:
                kb[n] = min(64, self.known_bits(a) + self.nodes[b][4])
        elif op in (ULT, SLT, EQ, NEZ):
            kb[n] = 1
        elif op == SEL:
            kb[n] = max(self.known_bits(a), self.known_bits(b))
        elif op == ADD:
            kb[n] = min(64, max(self.known_bits(a), self.known_bits(b)) + 1)
        elif op == POPCNT:
            kb[n] = 7
        elif op == CLZ:
            kb[n] = 7
        elif op == SAR:
            # An arithmetic shift can only widen through sign replication when
            # the source could be negative; a value already known narrow cannot.
            if self.nodes[b][0] == CONST and self.known_bits(a) < 64:
                kb[n] = max(0, self.known_bits(a) - self.nodes[b][4])
        elif op == MUL:
            kb[n] = min(64, self.known_bits(a) + self.known_bits(b))

    def is_const(self, n: int) -> bool:
        return self.nodes[n][0] == CONST

    def const_val(self, n: int) -> int:
        return self.nodes[n][4]

    def input_value(self, key, bits: int = 64) -> int:
        n = self.inputs.get(('v', key))
        if n is None:
            n = self._emit(INV, imm=len(self.inputs))
            self.inputs[('v', key)] = n
            self.kbits[n] = bits
        return n

    def input_taint(self, key, bits: int = 64) -> int:
        n = self.inputs.get(('t', key))
        if n is None:
            n = self._emit(INT, imm=len(self.inputs))
            self.inputs[('t', key)] = n
            self.kbits[n] = bits
        return n

    def op(self, op: str, a: int = -1, b: int = -1, c: int = -1) -> int:
        """Emit `op`, folding when every operand is constant and applying the
        algebraic identities that make the folded program small."""
        nodes = self.nodes
        if op in _BINARY:
            ca, cb = nodes[a][0] == CONST, nodes[b][0] == CONST
            if ca and cb:
                return self.const(eval_op(op, nodes[a][4], nodes[b][4], 0, 0))
            r = self._simplify_binary(op, a, b, ca, cb)
            if r is not None:
                return r
            if op in (AND, OR, XOR):
                r = self._try_bool(op, a, b)
                if r is not None:
                    return r
            n = self._emit(op, a, b)
            self._set_bits(n, op, a, b, -1)
            return n
        if op in _UNARY:
            if nodes[a][0] == CONST:
                return self.const(eval_op(op, nodes[a][4], 0, 0, 0))
            if op == NEZ and (nodes[a][0] in (ULT, SLT, EQ, NEZ, BOOLSYM)
                              or self.known_bits(a) <= 1):
                return a                     # already 0/1
            if op == NOT and nodes[a][0] == NOT:
                return nodes[a][1]
            n = self._emit(op, a)
            self._set_bits(n, op, a, -1, -1)
            return n
        if op == SEL:
            if nodes[c][0] == CONST:
                return a if nodes[c][4] else b
            if a == b:
                return a
            r = self._try_bool_sel(a, b, c)
            if r is not None:
                return r
            n = self._emit(SEL, a, b, c)
            self._set_bits(n, SEL, a, b, c)
            return n
        raise ValueError(f'unknown IR op {op}')

    def _simplify_binary(self, op, a, b, ca, cb):
        """Identities worth having: masks and shifts by lift-time constants are
        everywhere in flag macros, and folding them is most of the win."""
        nodes = self.nodes
        av = nodes[a][4] if ca else None
        bv = nodes[b][4] if cb else None
        if op == AND:
            if bv == 0 or av == 0:
                return self.const(0)
            if bv == MASK64:
                return a
            if av == MASK64:
                return b
            if a == b:
                return a
            # (x & m1) & m2  ->  x & (m1 & m2)
            if cb and nodes[a][0] == AND and nodes[nodes[a][2]][0] == CONST:
                return self.op(AND, nodes[a][1],
                               self.const(nodes[nodes[a][2]][4] & bv))
        elif op == OR:
            if bv == 0:
                return a
            if av == 0:
                return b
            if bv == MASK64 or av == MASK64:
                return self.const(MASK64)
            if a == b:
                return a
        elif op == XOR:
            if bv == 0:
                return a
            if av == 0:
                return b
            if a == b:
                return self.const(0)
        elif op in (ADD, SUB):
            if bv == 0:
                return a
            if op == ADD and av == 0:
                return b
            if op == SUB and a == b:
                return self.const(0)
        elif op == MUL:
            if bv == 1:
                return a
            if av == 1:
                return b
            if bv == 0 or av == 0:
                return self.const(0)
        elif op in (SHL, SHR, SAR):
            if bv == 0:
                return a
            if av == 0:
                return self.const(0)
            # (x << k1) << k2 -> x << (k1+k2), the shape masking produces
            if cb and nodes[a][0] == op and nodes[nodes[a][2]][0] == CONST:
                k = nodes[nodes[a][2]][4] + bv
                if k >= 64:
                    return self.const(MASK64) if op == SAR else self.const(0)
                return self.op(op, nodes[a][1], self.const(k))
        elif op == EQ and a == b:
            return self.const(1)
        elif op == ULT:
            if a == b or bv == 0:
                return self.const(0)
            if cb and self.known_bits(a) < bv.bit_length():
                return self.const(1)          # a is provably below the bound
        elif op == SLT and a == b:
            return self.const(0)
        return None

    # -- the one-bit boolean layer -------------------------------------
    def _sym(self, n: int):
        """(leaves, truth table) for a one-bit node, or None."""
        op, a, b, c, imm = self.nodes[n]
        if op == BOOLSYM:
            leaves = tuple(x for x in (a, b, c) if x >= 0)
            return leaves, imm
        if op == CONST:
            if imm == 0:
                return (), 0x00
            if imm == 1:
                return (), 0xFF
            return None
        if self.known_bits(n) <= 1:
            return (n,), bs.LEAF_TT[0]
        return None

    def _mk_bool(self, leaves, tt):
        """Materialise a one-bit cone as (leaves, truth table).

        Leaves the function does not actually depend on are dropped first: a
        cone that cancels down to a single leaf -- which is what a sign-bit
        comparison does once its corners are pushed to bit level -- must come
        out as that leaf and nothing else, or the whole point is lost.
        """
        tt &= 0xFF
        keep = [i for i in range(len(leaves))
                if any(((tt >> idx) & 1) != ((tt >> (idx | (1 << i))) & 1)
                       for idx in range(8) if not (idx >> i) & 1)]
        if len(keep) != len(leaves):
            new_tt = 0
            for j_idx in range(8):
                full = 0
                for j, i in enumerate(keep):
                    if (j_idx >> j) & 1:
                        full |= 1 << i
                if (tt >> full) & 1:
                    new_tt |= 1 << j_idx
            tt = new_tt
            leaves = tuple(leaves[i] for i in keep)
        if not leaves:
            return self.const(tt & 1)
        if len(leaves) == 1 and tt == bs.LEAF_TT[0]:
            return leaves[0]
        return self._emit_bool(leaves, tt)

    def _emit_bool(self, leaves, tt):
        a = leaves[0] if len(leaves) > 0 else -1
        b = leaves[1] if len(leaves) > 1 else -1
        c = leaves[2] if len(leaves) > 2 else -1
        n = self._emit(BOOLSYM, a, b, c, tt)
        self.kbits[n] = 1
        return n

    def _try_bool(self, op, a, b):
        if self.no_bool:
            return None
        if self.known_bits(a) > 1 or self.known_bits(b) > 1:
            return None
        sa, sb_ = self._sym(a), self._sym(b)
        if sa is None or sb_ is None:
            return None
        r = bs.combine(op, sa[0], sa[1], sb_[0], sb_[1])
        if r is None or not self._bool_pays(r[1], (a, b)):
            return None
        return self._mk_bool(r[0], r[1])

    def _try_bool_sel(self, a, b, c):
        if self.no_bool:
            return None
        if max(self.known_bits(a), self.known_bits(b), self.known_bits(c)) > 1:
            return None
        sa, sb_, sc = self._sym(a), self._sym(b), self._sym(c)
        if sa is None or sb_ is None or sc is None:
            return None
        r = bs.select(sc[0], sc[1], sa[0], sa[1], sb_[0], sb_[1])
        if r is None or not self._bool_pays(r[1], (a, b, c)):
            return None
        return self._mk_bool(r[0], r[1])

    def _bool_pays(self, tt: int, operands) -> bool:
        """Is folding this cone cheaper than emitting the operation directly?

        Re-synthesising from leaves is only a win when it also retires the
        operands.  An operand with another consumer still has to be computed, so
        folding then buys a cheapest expression at the price of duplicating the
        chain that produced it -- which is how a boolean layer makes a program
        BIGGER.  Charge honestly: the direct form costs one operation plus
        whatever operands this use alone keeps alive.
        """
        direct = 1
        for x in operands:
            if self.nodes[x][0] == BOOLSYM and self.uses.get(x, 0) <= 1:
                direct += bs.expr_cost(self.nodes[x][4])
        return bs.expr_cost(tt) <= direct

    # -- convenience ---------------------------------------------------
    def mask(self, n: int, bits: int) -> int:
        """Truncate to `bits`, skipping the AND when it cannot change anything."""
        if bits >= 64 or self.known_bits(n) <= bits:
            return n
        m = (1 << bits) - 1
        return self.op(AND, n, self.const(m))

    def splat(self, flag: int) -> int:
        """A 0/1 flag to an all-ones-or-zero mask (`0 - flag`), the standard way
        to make a predicate usable by bitwise arithmetic without branching."""
        return self.op(NEG, flag)

    # -- finishing -----------------------------------------------------
    def finish(self):
        """Mark the nodes some output actually depends on.

        Everything else is dead: a value computed only because the p-code
        happened to compute it, or a taint term the folding proved constant.
        """
        live = [False] * len(self.nodes)
        stack = [n for _k, n in self.outputs]
        while stack:
            n = stack.pop()
            if live[n]:
                continue
            live[n] = True
            op, a, b, c, _imm = self.nodes[n]
            for x in (a, b, c):
                if x >= 0 and not live[x]:
                    stack.append(x)
        self.live = live
        return self

    def cost(self) -> int:
        """Machine instructions the live program would need.

        Constants and inputs are not counted as work: a constant is an
        immediate operand or a materialised literal, and an input is a load the
        caller has to do regardless of how taint is computed.
        """
        if not self.live:
            self.finish()
        total = 0
        for n, (op, _a, _b, _c, _imm) in enumerate(self.nodes):
            if not self.live[n] or op in (CONST, INV, INT):
                continue
            total += (bs.expr_cost(_imm) if op == BOOLSYM
                      else _COST.get(op, 1))
        return total

    def cost_by_opcode(self) -> dict:
        """Live machine-op cost credited to the p-code opcode that emitted it."""
        if not self.live:
            self.finish()
        out: dict = {}
        for name, lo, hi in self.spans:
            c = 0
            for n in range(lo, hi):
                op = self.nodes[n][0]
                if self.live[n] and op not in (CONST, INV, INT):
                    c += _COST.get(op, 1)
            if c:
                out[name] = out.get(name, 0) + c
        return out

    def n_live(self) -> int:
        if not self.live:
            self.finish()
        return sum(1 for n, (op, *_r) in enumerate(self.nodes)
                   if self.live[n] and op not in (CONST, INV, INT))

    # -- finalisation --------------------------------------------------
    def finalize(self):
        """A compact, contiguous program with every one-bit cone expanded.

        Two things happen here that the backends should not have to know about:
        dead nodes disappear (so node indices are dense, which is what a flat
        scratch array wants), and BOOLSYM becomes real operations via a cheapest
        expression for its truth table.  What comes out uses only the machine-
        shaped opcodes, so an interpreter and a code generator consume the same
        thing.
        """
        if not self.live:
            self.finish()
        out = IRProg()
        out.no_bool = True
        m: dict = {}

        def expand(tree, leaves):
            kind = tree[0]
            if kind == 'leaf':
                return m[leaves[tree[1]]]
            if kind == 'const':
                return out.const(tree[1])
            if kind == 'not':
                return out.op(XOR, expand(tree[1], leaves), out.const(1))
            lhs = expand(tree[1], leaves)
            rhs = expand(tree[2], leaves)
            return out.op({'and': AND, 'or': OR, 'xor': XOR}[kind], lhs, rhs)

        for n, (op, a, b, c, imm) in enumerate(self.nodes):
            if not self.live[n]:
                continue
            if op == CONST:
                m[n] = out.const(imm)
            elif op in (INV, INT):
                key = next(k for (kind, k), node in self.inputs.items()
                           if node == n and kind == ('v' if op == INV else 't'))
                m[n] = (out.input_value(key, self.known_bits(n)) if op == INV
                        else out.input_taint(key, self.known_bits(n)))
            elif op == BOOLSYM:
                leaves = tuple(x for x in (a, b, c) if x >= 0)
                tree = bs.expr_for(imm)
                m[n] = expand(tree, leaves)
                out.kbits[m[n]] = 1
            else:
                m[n] = out.op(op, m[a] if a >= 0 else -1,
                              m[b] if b >= 0 else -1,
                              m[c] if c >= 0 else -1)
        out.outputs = [(k, m[n]) for k, n in self.outputs]
        return out.finish()

    def serialize(self, slot_of):
        """Flat arrays for a C evaluator or a code generator.

        `slot_of(name) -> int` places each register in the caller's value/taint
        arrays; the index is baked in, so running the program involves no name
        lookup at all.
        """
        prog = self.finalize()
        ops, aa, bb, cc, imms = [], [], [], [], []
        for op, a, b, c, imm in prog.nodes:
            ops.append(op)
            aa.append(a)
            bb.append(b)
            cc.append(c)
            imms.append(imm)
        inputs = []
        for (kind, key), n in prog.inputs.items():
            slot = slot_of(key)
            if slot is None:
                raise KeyError(key)
            inputs.append((n, 0 if kind == 'v' else 1, slot))
        outputs = []
        for key, n in prog.outputs:
            slot = slot_of(key)
            if slot is not None:
                outputs.append((slot, n))
        return {'ops': ops, 'a': aa, 'b': bb, 'c': cc, 'imm': imms,
                'inputs': inputs, 'outputs': outputs,
                'n_nodes': len(prog.nodes), 'cost': prog.cost()}

    # -- reference evaluation ------------------------------------------
    def run(self, values: dict, taints: dict) -> dict:
        """Reference interpreter: evaluate the live program for one input state.

        `values`/`taints` are keyed the way the builder keyed its inputs.  Slow
        by design -- this exists to validate a lowering, not to run in anger.
        """
        inv_key = {n: k for (kind, k), n in self.inputs.items() if kind == 'v'}
        int_key = {n: k for (kind, k), n in self.inputs.items() if kind == 't'}
        val = [0] * len(self.nodes)
        for n, (op, a, b, c, imm) in enumerate(self.nodes):
            if op == CONST:
                val[n] = imm
            elif op == INV:
                val[n] = values.get(inv_key[n], 0) & MASK64
            elif op == INT:
                val[n] = taints.get(int_key[n], 0) & MASK64
            else:
                val[n] = eval_op(op, val[a] if a >= 0 else 0,
                                 val[b] if b >= 0 else 0,
                                 val[c] if c >= 0 else 0, imm)
        return {k: val[n] for k, n in self.outputs}

    def dump(self) -> str:
        if not self.live:
            self.finish()
        inv_key = {n: k for (kind, k), n in self.inputs.items() if kind == 'v'}
        int_key = {n: k for (kind, k), n in self.inputs.items() if kind == 't'}
        lines = []
        for n, (op, a, b, c, imm) in enumerate(self.nodes):
            if not self.live[n]:
                continue
            if op == CONST:
                lines.append(f'  %{n} = {imm:#x}')
            elif op == INV:
                lines.append(f'  %{n} = value {inv_key[n]}')
            elif op == INT:
                lines.append(f'  %{n} = taint {int_key[n]}')
            else:
                args = ', '.join(f'%{x}' for x in (a, b, c) if x >= 0)
                lines.append(f'  %{n} = {op} {args}')
        for k, n in self.outputs:
            lines.append(f'  out {k} = %{n}')
        return '\n'.join(lines)
