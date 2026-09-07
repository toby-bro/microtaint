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

    __slots__ = ('nodes', '_hc', 'inputs', 'outputs', 'live', 'kbits', 'spans')

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

    # -- construction --------------------------------------------------
    def _emit(self, op, a=-1, b=-1, c=-1, imm=0) -> int:
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
            n = self._emit(op, a, b)
            self._set_bits(n, op, a, b, -1)
            return n
        if op in _UNARY:
            if nodes[a][0] == CONST:
                return self.const(eval_op(op, nodes[a][4], 0, 0, 0))
            if op == NEZ and nodes[a][0] in (ULT, SLT, EQ, NEZ):
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
            total += _COST.get(op, 1)
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
