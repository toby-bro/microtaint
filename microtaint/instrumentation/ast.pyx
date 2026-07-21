# cython: language_level=3
# cython: profile=False
import os
from enum import Enum
from microtaint.instrumentation.cell_c.circuit_c import compile_circuit
from microtaint.simulator import CellSimulator, MachineState
from microtaint.types import Architecture, Register

def _build_machine_state(dict input_dict, EvalContext context):
    cdef dict regs = {}
    cdef dict mem = {}
    cdef str name, ptr_part, head
    cdef object val
    cdef object addr
    cdef int second_us
    cdef long signed_off
    cdef long size

    for name, val in input_dict.items():
        # Direct slice comparison — faster than str.startswith.
        if name[:4] == 'MEM_':
            ptr_part = name[4:]
            # Format A: MEM_0xHEX  or  MEM_0xHEX_size  (static address)
            if ptr_part[:2] == '0x' or ptr_part[:3] == '-0x':
                addr = int(ptr_part.split('_')[0], 16)
                mem[addr] = val
                continue

            # Format B: MEM_<reg>_<offset>_<size>  (register-relative, signed offset)
            # ptr_part = 'RBP_-16_8' for example.
            second_us = ptr_part.rfind('_')
            if second_us > 0:
                head = ptr_part[:second_us]
                first_us = head.rfind('_')
                if first_us > 0:
                    try:
                        signed_off = int(head[first_us + 1:])
                        addr = context.input_values.get(head[:first_us], 0) + signed_off
                        mem[addr] = val
                        continue
                    except (ValueError, OverflowError):
                        pass

            # Legacy format: MEM_<reg>  (no offset, no size) — kept for
            # backward compatibility with callers that haven't migrated.
            addr = context.input_values.get(ptr_part, 0)
            mem[addr] = val
        else:
            regs[name] = val
    return MachineState(regs=regs, mem=mem)


def _process_output_state(
    dict output_state,
    object shadow_mem,            # BitPreciseShadowMemory (Cython class)
    dict register_taint,          # mutated in place: cleared then refilled
    object last_tainted_writes,   # set, mutated in place
    bint check_aiw,
    list mem_writes,              # caller-allocated list, populated in place
):
    """
    Cython-level post-processing of LogicCircuit.evaluate() output.

    Replaces the Python ``for key, val in output_state.items()`` loop in
    the hook (_instruction_evaluator_raw lines 737–757) which is ~14 s
    out of the 36 s hook tottime in the bench.

    Mutates:
      - register_taint: cleared, then refilled with non-MEM_ entries
      - shadow_mem: write_mask called for each MEM_ entry
      - last_tainted_writes: cleared, then add() for tainted bytes
      - mem_writes: appended (addr, size, val) tuples for AIW check

    Caller is responsible for clearing ``last_tainted_writes`` *before*
    the call (it isn't done here so the caller can decide whether the
    set is empty without reaching here).
    """
    cdef str key
    cdef object val
    cdef str body
    cdef int last
    cdef long mem_addr
    cdef int mem_size
    cdef long val_int
    cdef int i, sb
    cdef object byte_val

    register_taint.clear()

    for key, val in output_state.items():
        if key[:4] == 'MEM_':
            body = key[4:]
            last = body.rfind('_')
            if last < 0:
                continue
            try:
                mem_addr = <long>int(body[:last], 16)
                mem_size = <int>int(body[last + 1:])
            except (ValueError, OverflowError):
                continue
            shadow_mem.write_mask(mem_addr, val, mem_size)
            if val:
                # Use Python int >> for arbitrary-precision safety; mem_size <= 8
                # in practice so the inner loop is tiny.
                val_int = int(val)
                for i in range(mem_size):
                    sb = i << 3
                    if (val_int >> sb) & 0xFF:
                        last_tainted_writes.add(mem_addr + i)
                if check_aiw:
                    mem_writes.append((mem_addr, mem_size, val_int))
        elif val:
            register_taint[key] = val



class Op(str, Enum):
    AND = 'AND'
    OR = 'OR'
    XOR = 'XOR'
    NOT = 'NOT'
    LEFT = 'LEFT'
    RIGHT = 'RIGHT'
    ADD = 'ADD'  # Only for memory offset calculations, not for taint logic
    SUB = 'SUB'  # Only for memory offset calculations, not for taint logic

# Integer opcode constants for fast C-level dispatch in BinaryExpr/UnaryExpr.
# Enum.__eq__ involves Python object machinery; int comparison is ~10x faster.
cdef int _OP_AND   = 0
cdef int _OP_OR    = 1
cdef int _OP_XOR   = 2
cdef int _OP_LEFT  = 3
cdef int _OP_ADD   = 4
cdef int _OP_SUB   = 5
cdef int _OP_NOT   = 6
cdef int _OP_RIGHT = 7

# Mapping from Op enum to int at module init (done once)
_OP_MAP: dict = {}

def _init_op_map():
    global _OP_MAP
    _OP_MAP = {
        Op.AND: _OP_AND, Op.OR: _OP_OR, Op.XOR: _OP_XOR,
        Op.LEFT: _OP_LEFT, Op.ADD: _OP_ADD, Op.SUB: _OP_SUB,
        Op.NOT: _OP_NOT, Op.RIGHT: _OP_RIGHT,
    }
_init_op_map()

# Canonical parent register per architecture
_ARCH_PARENT_REGS: dict[str, dict[str, tuple[str, int]]] = {
    # arch_str -> {child_name: (parent_name, bit_start_in_parent)}
    'AMD64': {
        'AL':  ('RAX', 0),  'AH':  ('RAX', 8),  'AX':  ('RAX', 0),  'EAX': ('RAX', 0),
        'BL':  ('RBX', 0),  'BH':  ('RBX', 8),  'BX':  ('RBX', 0),  'EBX': ('RBX', 0),
        'CL':  ('RCX', 0),  'CH':  ('RCX', 8),  'CX':  ('RCX', 0),  'ECX': ('RCX', 0),
        'DL':  ('RDX', 0),  'DH':  ('RDX', 8),  'DX':  ('RDX', 0),  'EDX': ('RDX', 0),
        'SIL': ('RSI', 0),  'SI':  ('RSI', 0),  'ESI': ('RSI', 0),
        'DIL': ('RDI', 0),  'DI':  ('RDI', 0),  'EDI': ('RDI', 0),
        'BPL': ('RBP', 0),  'BP':  ('RBP', 0),  'EBP': ('RBP', 0),
        'SPL': ('RSP', 0),  'SP':  ('RSP', 0),  'ESP': ('RSP', 0),
        'EIP': ('RIP', 0),
        'R8B': ('R8',  0),  'R8W': ('R8',  0),  'R8D': ('R8',  0),
        'R9B': ('R9',  0),  'R9W': ('R9',  0),  'R9D': ('R9',  0),
        'R10B':('R10', 0),  'R10W':('R10', 0),  'R10D':('R10', 0),
        'R11B':('R11', 0),  'R11W':('R11', 0),  'R11D':('R11', 0),
        'R12B':('R12', 0),  'R12W':('R12', 0),  'R12D':('R12', 0),
        'R13B':('R13', 0),  'R13W':('R13', 0),  'R13D':('R13', 0),
        'R14B':('R14', 0),  'R14W':('R14', 0),  'R14D':('R14', 0),
        'R15B':('R15', 0),  'R15W':('R15', 0),  'R15D':('R15', 0),
    },
    'X86': {
        'AL':  ('EAX', 0),  'AH':  ('EAX', 8),  'AX':  ('EAX', 0),
        'BL':  ('EBX', 0),  'BH':  ('EBX', 8),  'BX':  ('EBX', 0),
        'CL':  ('ECX', 0),  'CH':  ('ECX', 8),  'CX':  ('ECX', 0),
        'DL':  ('EDX', 0),  'DH':  ('EDX', 8),  'DX':  ('EDX', 0),
        'SI':  ('ESI', 0),  'DI':  ('EDI', 0),
        'BP':  ('EBP', 0),  'SP':  ('ESP', 0),
    },
    'ARM64': {},  # ARM64 has W0-W30 as lower 32 bits of X0-X30
}

# Reverse: parent -> [(child_name, bit_start, bit_size)]
_ARCH_CHILD_REGS: dict[str, dict[str, list[tuple[str, int, int]]]] = {
    'AMD64': {
        'RAX': [('EAX',0,32),('AX',0,16),('AH',8,8),('AL',0,8)],
        'RBX': [('EBX',0,32),('BX',0,16),('BH',8,8),('BL',0,8)],
        'RCX': [('ECX',0,32),('CX',0,16),('CH',8,8),('CL',0,8)],
        'RDX': [('EDX',0,32),('DX',0,16),('DH',8,8),('DL',0,8)],
        'RSI': [('ESI',0,32),('SI',0,16),('SIL',0,8)],
        'RDI': [('EDI',0,32),('DI',0,16),('DIL',0,8)],
        'RBP': [('EBP',0,32),('BP',0,16),('BPL',0,8)],
        'RSP': [('ESP',0,32),('SP',0,16),('SPL',0,8)],
        'RIP': [('EIP',0,32)],
        'R8':  [('R8D',0,32),('R8W',0,16),('R8B',0,8)],
        'R9':  [('R9D',0,32),('R9W',0,16),('R9B',0,8)],
        'R10': [('R10D',0,32),('R10W',0,16),('R10B',0,8)],
        'R11': [('R11D',0,32),('R11W',0,16),('R11B',0,8)],
        'R12': [('R12D',0,32),('R12W',0,16),('R12B',0,8)],
        'R13': [('R13D',0,32),('R13W',0,16),('R13B',0,8)],
        'R14': [('R14D',0,32),('R14W',0,16),('R14B',0,8)],
        'R15': [('R15D',0,32),('R15W',0,16),('R15B',0,8)],
    },
    'X86': {
        'EAX': [('AX',0,16),('AH',8,8),('AL',0,8)],
        'EBX': [('BX',0,16),('BH',8,8),('BL',0,8)],
        'ECX': [('CX',0,16),('CH',8,8),('CL',0,8)],
        'EDX': [('DX',0,16),('DH',8,8),('DL',0,8)],
        'ESI': [('SI',0,16)],
        'EDI': [('DI',0,16)],
        'EBP': [('BP',0,16)],
        'ESP': [('SP',0,16)],
    },
    'ARM64': {},
}


def _resolve_register_alias(str name, dict state, object arch) -> object:
    """
    Resolve register value from aliases using the correct architecture hierarchy.
    
    Strategy:
    1. Check if name is a child register — look up parent and extract bits
    2. Check if name is a parent register — look up widest child and promote
    """
    cdef str arch_str = str(arch) if arch is not None else 'AMD64'
    cdef dict parent_map = _ARCH_PARENT_REGS.get(arch_str, {})
    cdef dict child_map = _ARCH_CHILD_REGS.get(arch_str, {})

    # Strategy 1: name is a sub-register, look up its parent
    if name in parent_map:
        parent_name, bit_start = parent_map[name]
        parent_val = state.get(parent_name, None)
        if parent_val is not None:
            return parent_val >> bit_start  # caller applies bit_end mask

    # Strategy 2: name is a parent register, look up children
    if name in child_map:
        for child_name, bit_start, bit_size in child_map[name]:
            child_val = state.get(child_name, None)
            if child_val is not None:
                # Reconstruct: place child bits at their position in parent
                mask = (<object>1 << bit_size) - 1
                return (child_val & mask) << bit_start

    return None

cdef class EvalContext:
    cdef public dict input_taint
    cdef public dict input_values
    cdef public object simulator
    cdef public object implicit_policy
    cdef public object shadow_memory
    cdef public object mem_reader
    cdef public str arch_str  # cached once, avoids str(simulator.arch) per TaintOperand miss

    def __init__(
        self,
        dict input_taint,
        dict input_values,
        object simulator=None,
        object implicit_policy=None,
        object shadow_memory=None,
        object mem_reader=None,
    ):
        cdef str arch_str

        self.simulator = simulator
        self.shadow_memory = shadow_memory
        self.mem_reader = mem_reader

        if implicit_policy is None:
            from microtaint.types import ImplicitTaintPolicy
            self.implicit_policy = ImplicitTaintPolicy.IGNORE
        else:
            self.implicit_policy = implicit_policy

        # Determine architecture for alias resolution
        arch_str = 'AMD64'
        if simulator is not None:
            arch_str = str(simulator.arch)

        self.arch_str = arch_str  # cache for TaintOperand fast path

        # Normalize: always store taint/values under the canonical parent register
        self.input_taint = _normalize_register_dict(input_taint, arch_str)
        self.input_values = _normalize_register_dict(input_values, arch_str)


def _normalize_register_dict(dict d, str arch_str) -> dict:
    """
    Normalize a register dict so all values are stored under the canonical
    parent register name for the given architecture.
    
    e.g. {'AL': 0xFF} -> {'RAX': 0xFF}  (AMD64)
         {'AL': 0xFF} -> {'EAX': 0xFF}  (X86)
    
    If both a parent and child are present, OR them together (union of taints).
    Values that are already under the parent name are kept as-is.
    MEM_ keys are passed through unchanged.
    """
    cdef dict parent_map = _ARCH_PARENT_REGS.get(arch_str, {})
    cdef dict result
    cdef str key
    cdef object val
    cdef str parent_name
    cdef int bit_start
    cdef bint needs_normalize = False

    # Hot-path fast check: if every key is already canonical (not in parent_map)
    # and not MEM_-prefixed-with-aliasing, we can return the dict as-is. This
    # is the common case from the wrapper, which builds _pre_regs from
    # canonical Sleigh names like 'RAX'/'RBX'.
    if parent_map:
        for key in d:
            if key in parent_map:
                needs_normalize = True
                break
    if not needs_normalize:
        # Caller may mutate the result; copy to keep dict ownership clean only
        # when truly needed. EvalContext stores it directly without further
        # mutation in input_taint/input_values, so a shallow alias is fine.
        return d

    result = {}
    for key, val in d.items():
        # Pass through memory and unknown keys unchanged
        if key.startswith('MEM_') or key not in parent_map:
            # Already a parent register or unknown — store as-is
            existing = result.get(key, 0)
            result[key] = existing | val
            continue

        # key is a child register — promote to parent
        parent_name, bit_start = parent_map[key]
        # Shift the child value into its position within the parent
        promoted = val << bit_start
        existing = result.get(parent_name, 0)
        result[parent_name] = existing | promoted

    return result


cdef class Expr:
    cpdef object evaluate(self, EvalContext context):
        raise NotImplementedError('Must implement evaluate in subclasses')


cdef class AvalancheExpr(Expr):
    cdef public Expr expr
    cdef public int size_bits

    def __init__(self, Expr expr, int size_bits):
        self.expr = expr
        self.size_bits = size_bits

    def __str__(self):
        return f'AVALANCHE({self.expr})'

    def __repr__(self):
        return f'AvalancheExpr(expr={repr(self.expr)}, size_bits={self.size_bits})'

    cpdef object evaluate(self, EvalContext context):
        cdef object val = self.expr.evaluate(context)
        if val != 0:
            # FIX: <object>1 forces Python infinite precision shift
            return (<object>1 << self.size_bits) - 1
        return 0


cdef class FullMaskAvalancheExpr(Expr):
    """Flag-floor avalanche: fires only when dep taint == full mask for dep_bits.

    This is used for the MONOTONIC / COND_TRANSPORTABLE 1-bit flag soundness
    floor.  Unlike AvalancheExpr (fires on any nonzero taint), this only fires
    when the dep operand is *fully* tainted — i.e. all dep_bits are unknown.

    With partial taint (e.g. T_EBX = 0x1, single bit), the differential
    already captures the precise flag outcome; we must not override it.
    With full taint (T_EBX = MASK), the differential gives 0 due to equal-
    regime evaluation; this floor provides the soundness safety net.
    """
    cdef public Expr dep
    cdef public object full_mask   # (1 << dep_bits) - 1

    def __init__(self, Expr dep, int dep_bits):
        self.dep = dep
        self.full_mask = (<object>1 << dep_bits) - 1

    def __str__(self):
        return f'FULLMASK_AVAL({self.dep})'

    def __repr__(self):
        return f'FullMaskAvalancheExpr(dep={repr(self.dep)}, full_mask={self.full_mask:#x})'

    cpdef object evaluate(self, EvalContext context):
        cdef object val = self.dep.evaluate(context)
        if val == self.full_mask and val != 0:
            return 1
        return 0


cdef class SignedOverflowTaintExpr(Expr):
    """EXACT taint of signed overflow (P-code INT_SBORROW / INT_SCARRY).

    Signed overflow is NON-MONOTONE, so the 2-replica differential -- which samples
    only the extremal corners V|T and V&~T -- can miss it: the corners coincidentally
    agree while an interior flip of a tainted bit toggles OF.  That is the
    ``sub rax,rbx; seto dl`` under-taint class.

    Rule (sign decomposition).  For width w with sign bit s = w-1::

        Bor = [ a[0:w-1] <u b[0:w-1] ]                 borrow INTO the msb   (a-b)
        Car = [ a[0:w-1] +u b[0:w-1] >= 2^(w-1) ]      carry  INTO the msb   (a+b)
        r_s = a_s ^ b_s ^ Bor
        OF  = (a_s ^ b_s) & (b_s ^ Bor)      (sub)
        OF  = ~(a_s ^ b_s) & (b_s ^ Car)     (add)

    Exactness rests on two facts:

    1. ``a_s`` (bit w-1 of a), ``b_s`` (bit w-1 of b) and ``Bor``/``Car`` (bits
       0..w-2 of both) read DISJOINT input bits, hence vary INDEPENDENTLY -- so
       enumerating their reachable values is exact, not a product over-approximation.
    2. ``Bor``/``Car`` is MONOTONE (``<u`` is decreasing in a, increasing in b; carry
       is increasing in both), so its reachable set over the taint cube is exactly
       {polarised-min, polarised-max} and its taint is the polarised differential
       XOR -- the paper's existing D^{+-} polarity, applied to the sign split.

    T_OF is then the non-constancy of a 3-input boolean function over <= 2^3
    assignments: exact, never a floor.  Machine-checked in
    ``benchmark/soundness/prove_signed_overflow.py``: identity and no-under-taint
    are PROVED for w = 2..64 (complete for the deployed engine); no-over-taint is
    PROVED for w <= 6 (beyond that the ForAll query is intractable for Z3, which is
    a solver limit -- the argument itself is width-independent).
    """
    cdef public Expr a_val
    cdef public Expr a_taint
    cdef public Expr b_val
    cdef public Expr b_taint
    cdef public Expr c_val
    cdef public Expr c_taint
    cdef public int width
    cdef public bint is_sub

    def __init__(self, Expr a_val, Expr a_taint, Expr b_val, Expr b_taint, int width, bint is_sub,
                 Expr c_val=None, Expr c_taint=None):
        self.a_val = a_val
        self.a_taint = a_taint
        self.b_val = b_val
        self.b_taint = b_taint
        # Optional third operand -- the carry/borrow IN of adc/sbb/adcs/sbcs.  It is
        # zext of a 1-bit flag, so it lies entirely inside the low part and its sign
        # bit is 0; it therefore shifts only the carry/borrow INTO the msb, leaving
        # the sign function _g unchanged.  None == the plain two-operand case.
        self.c_val = c_val
        self.c_taint = c_taint
        self.width = width
        self.is_sub = is_sub

    def __str__(self):
        op = 'SBORROW' if self.is_sub else 'SCARRY'
        carry = '' if self.c_val is None else f', +{self.c_val}'
        return f'SIGNED_OVF_TAINT[{op},w={self.width}]({self.a_val}, {self.b_val}{carry})'

    def __repr__(self):
        return (
            f'SignedOverflowTaintExpr(a_val={repr(self.a_val)}, b_val={repr(self.b_val)}, '
            f'width={self.width}, is_sub={self.is_sub})'
        )

    cpdef object evaluate(self, EvalContext context):
        cdef int w = self.width
        cdef object mask = ((<object>1) << w) - 1
        cdef object lowmask = ((<object>1) << (w - 1)) - 1

        cdef object a = self.a_val.evaluate(context) & mask
        cdef object ta = self.a_taint.evaluate(context) & mask
        cdef object b = self.b_val.evaluate(context) & mask
        cdef object tb = self.b_taint.evaluate(context) & mask
        cdef object c = 0
        cdef object tc = 0
        if self.c_val is not None:
            c = self.c_val.evaluate(context) & lowmask
            tc = self.c_taint.evaluate(context) & lowmask

        cdef int a_s = <int>((a >> (w - 1)) & 1)
        cdef int b_s = <int>((b >> (w - 1)) & 1)
        cdef int ta_s = <int>((ta >> (w - 1)) & 1)
        cdef int tb_s = <int>((tb >> (w - 1)) & 1)

        cdef object al = a & lowmask
        cdef object bl = b & lowmask
        cdef object tal = ta & lowmask
        cdef object tbl = tb & lowmask

        cdef int base_c, hi, lo, t_c
        if self.is_sub:
            # [al <u bl] is DECREASING in al, INCREASING in bl -> opposite polarity.
            # [al <u bl + c] is DECREASING in al, INCREASING in bl and c.
            base_c = 1 if al < bl + c else 0
            hi = 1 if (al & ~tal & lowmask) < (bl | tbl) + (c | tc) else 0
            lo = 1 if (al | tal) < (bl & ~tbl & lowmask) + (c & ~tc & lowmask) else 0
        else:
            # carry into msb is INCREASING in all operands -> same polarity.
            base_c = 1 if (al + bl + c) > lowmask else 0
            hi = 1 if ((al | tal) + (bl | tbl) + (c | tc)) > lowmask else 0
            lo = 1 if ((al & ~tal & lowmask) + (bl & ~tbl & lowmask)
                       + (c & ~tc & lowmask)) > lowmask else 0
        t_c = hi ^ lo

        cdef int base = self._g(a_s, b_s, base_c)
        cdef int da, db, dc
        for da in range(2 if ta_s else 1):
            for db in range(2 if tb_s else 1):
                for dc in range(2 if t_c else 1):
                    if self._g(a_s ^ da, b_s ^ db, base_c ^ dc) != base:
                        return 1
        return 0

    cdef int _g(self, int x, int y, int z):
        """OF as a function of (a_s, b_s, Bor|Car)."""
        if self.is_sub:
            return (x ^ y) & (y ^ z)
        return (1 - (x ^ y)) & (y ^ z)


cdef class VariableShiftTaintExpr(Expr):
    """EXACT taint of a shift by a DATA-DEPENDENT amount, in O(log w) steps.

    A tainted shift amount currently forces AvalancheExpr -- sound, but the whole
    output width goes tainted.  Measured over a 2M-case campaign, variable-amount
    shifts were 69.8% of ALL over-tainted bits across five ISAs (4.13x invented
    bits vs ground truth) from just 21 of 175 instructions, while everything
    outside shifts and multiply sat at 0.07x.

    The reachable amount set is a SUBCUBE, not an interval::

        S = { s0 + sum_{j in T_s} b_j * 2^j },      s0 = V_s & ~T_s

    and a subcube is exactly what log-fold doubling enumerates implicitly, so OR
    and AND *over the whole set* need no enumeration::

        sm(y, op):  r = y >>> s0
                    for j in 0 .. log2(w)-1:
                        if T_s[j]:  r = r op (r >>> 2^j)

        T_r = sm(T_x, OR)  |  ( sm(x, OR) ^ sm(x, AND) )

    The two terms are the two clauses of noninterference:

      * ``sm(T_x, OR)`` -- some reachable amount brings a TAINTED source bit to
        this output position;
      * ``sm(x, OR) ^ sm(x, AND)`` -- the CLEAN value at this position differs
        across reachable amounts, so the bit is tainted even though every source
        bit that can land there is clean.  Avalanche cannot express this term and
        an interval bound gets it wrong.

    Cost is 2*log2(w) = 12 shift/or/and steps at w=64, INDEPENDENT of how many
    input bits are tainted and of their values -- unlike VariableBitSelectTaintExpr,
    which enumerates the reachable index set at O(|S|*w).  All w output bits are
    resolved at once.

    Validated bit-for-bit against brute-force 2^k ground truth (0 mismatches, 0
    under-taints over 60000 random cases at w=16, for left / logical-right /
    arithmetic-right).  Machine-checked in
    ``benchmark/soundness/prove_variable_shift.py``.
    """
    cdef public Expr src_val
    cdef public Expr src_taint
    cdef public Expr amt_val
    cdef public Expr amt_taint
    cdef public int width
    cdef public int kind         # 0 = left, 1 = logical right, 2 = arithmetic right
    cdef public int amt_mask

    def __init__(self, Expr src_val, Expr src_taint, Expr amt_val, Expr amt_taint,
                 int width, int kind, int amt_mask):
        self.src_val = src_val
        self.src_taint = src_taint
        self.amt_val = amt_val
        self.amt_taint = amt_taint
        self.width = width
        self.kind = kind
        self.amt_mask = amt_mask

    def __str__(self):
        k = ('<<', '>>', 's>>')[self.kind]
        return f'VAR_SHIFT_TAINT[{k},w={self.width}]({self.src_val}, {self.amt_val})'

    def __repr__(self):
        return (f'VariableShiftTaintExpr(src_val={repr(self.src_val)}, '
                f'amt_val={repr(self.amt_val)}, width={self.width}, kind={self.kind})')

    cdef object _shift(self, object y, int s, object mask, int w):
        """One shift of the accumulator, matching the p-code opcode's semantics."""
        if s >= w:
            # p-code shifts by >= width yield 0 (or the replicated sign for s>>).
            if self.kind == 2:
                return mask if (y >> (w - 1)) & 1 else 0
            return 0
        if self.kind == 0:
            return (y << s) & mask
        if self.kind == 1:
            return (y & mask) >> s
        # arithmetic right: replicate the accumulator's current top bit into the fill
        cdef object r = (y & mask) >> s
        if (y >> (w - 1)) & 1 and s > 0:
            r |= (mask << (w - s)) & mask
        return r

    cdef object _smear(self, object y, object ts, int s0, bint is_and, object mask, int w, int lg):
        """OR/AND of ``y`` shifted by every amount in the reachable subcube."""
        cdef object r = self._shift(y, s0, mask, w)
        cdef int j
        cdef object shifted
        for j in range(lg):
            if (ts >> j) & 1:
                shifted = self._shift(r, 1 << j, mask, w)
                r = (r & shifted) if is_and else (r | shifted)
        return r

    cpdef object evaluate(self, EvalContext context):
        cdef int w = self.width
        cdef object mask = ((<object>1) << w) - 1
        cdef int lg = (w - 1).bit_length()

        cdef object x = self.src_val.evaluate(context) & mask
        cdef object tx = self.src_taint.evaluate(context) & mask
        cdef object ts = self.amt_taint.evaluate(context) & self.amt_mask
        cdef int s0 = <int>((self.amt_val.evaluate(context) & self.amt_mask) & ~ts)

        # Amount bits above log2(w) can only push the shift past the width, which
        # every kind already saturates on; fold them into the low sweep so the
        # step count stays fixed at lg.
        cdef object ts_lo = ts & ((1 << lg) - 1)
        if ts & ~((1 << lg) - 1):
            ts_lo = (1 << lg) - 1

        cdef object reach = self._smear(tx, ts_lo, s0, False, mask, w, lg)
        cdef object hi = self._smear(x, ts_lo, s0, False, mask, w, lg)
        cdef object lo = self._smear(x, ts_lo, s0, True, mask, w, lg)
        return (reach | (hi ^ lo)) & mask


cdef class VariableBitSelectTaintExpr(Expr):
    """EXACT taint of a bit SELECTED by a data-dependent index (`bt r,r` -> CF).

    `bt rax, rbx` lifts to CF = bit[(rbx & (w-1))] of rax.  Selection by a tainted
    index is NON-MONOTONE, so the 2-replica differential -- which samples only the
    corners V|T and V&~T -- reads the source at exactly TWO index values and misses
    every other reachable index: the `bt rax,rbx; setc dl` under-taint.

    Avalanching CF would be sound but would OVER-taint.  Enumerate the reachable
    index set instead::

        I = { i : i agrees with b on every UNTAINTED index bit }      (|I| <= w)

        CF tainted  <=>  (exists i in I: T_a[i] = 1)
                     or  (exists i,j in I: T_a[i]=T_a[j]=0 and a_i != a_j)

    Justification: for a fixed index i, CF = a_i.  Over the cube the reachable CF
    values are { a_i : i in I, T_a[i]=0 } union ({0,1} if some i in I has T_a[i]=1);
    CF is tainted exactly when that set contains both 0 and 1.  Index bits and source
    bits live in different registers, so they vary independently and the enumeration
    is exact -- no approximation.

    Machine-checked in benchmark/soundness/prove_variable_bit_select.py.
    Cost: <= w iterations, short-circuiting on the first tainted reachable bit.
    """
    cdef public Expr src_val
    cdef public Expr src_taint
    cdef public Expr idx_val
    cdef public Expr idx_taint
    cdef public int width

    def __init__(self, Expr src_val, Expr src_taint, Expr idx_val, Expr idx_taint, int width):
        self.src_val = src_val
        self.src_taint = src_taint
        self.idx_val = idx_val
        self.idx_taint = idx_taint
        self.width = width

    def __str__(self):
        return f'VAR_BIT_SELECT_TAINT[w={self.width}]({self.src_val}[{self.idx_val}])'

    def __repr__(self):
        return (
            f'VariableBitSelectTaintExpr(src_val={repr(self.src_val)}, '
            f'idx_val={repr(self.idx_val)}, width={self.width})'
        )

    cpdef object evaluate(self, EvalContext context):
        cdef int w = self.width
        cdef object mask = ((<object>1) << w) - 1
        cdef object a = self.src_val.evaluate(context) & mask
        cdef object ta = self.src_taint.evaluate(context) & mask
        cdef object b = self.idx_val.evaluate(context) & mask
        cdef object tb = self.idx_taint.evaluate(context) & mask

        # index uses log2(w) low bits (bt masks the offset by the operand width)
        cdef int nbits = (w - 1).bit_length()
        cdef object low = ((<object>1) << nbits) - 1
        cdef object b_idx = b & low
        cdef object t_idx = tb & low
        cdef object fixed = b_idx & ~t_idx & low

        cdef int i
        cdef int seen0 = 0
        cdef int seen1 = 0
        for i in range(w):
            # reachable iff i agrees with b on every UNTAINTED index bit
            if (i & ~t_idx & low) != fixed:
                continue
            if (ta >> i) & 1:
                return 1  # (a) a reachable index selects a tainted source bit
            if (a >> i) & 1:
                seen1 = 1
            else:
                seen0 = 1
            if seen0 and seen1:
                return 1  # (b) two reachable clean bits differ
        return 0


cdef class ComparisonTaintExpr(Expr):
    """EXACT taint of a comparison bit ``[a OP b]``, OP in {<, <=}, signed or unsigned.

    A comparison is ANTITONE in its LHS and MONOTONE in its RHS.  The 2-corner
    differential picks ONE global polarity per operand, so it is exact for a single
    comparison but CANNOT serve a slice where the same operand feeds two comparisons of
    OPPOSITE orientation (PPC ``cmpw`` packs ``[a<b]`` and ``[b<a]`` into CR0).  Compute
    the comparison taint directly from the CROSS corners::

        can_be_true = [ min(a) OP max(b) ]     # a smallest, b largest  -> most likely true
        always_true = [ max(a) OP min(b) ]     # a largest,  b smallest -> least likely true
        Tr          = can_be_true XOR always_true

    can_be_true XOR always_true is 1 exactly when the predicate is NON-CONSTANT over the
    taint cube (can be true but not always).  ``min(x)=V&~T``, ``max(x)=V|T``; a SIGNED
    compare first XORs the sign bit of both operands into the unsigned domain (a bijection
    on the cube, so min/max are the signed extremes).  EXACT when a and b are INDEPENDENT
    (distinct operands / disjoint tainted bits) -- the builder gates on that; for ``[a<a]``
    the whole-slice differential is used (already exact).  1-bit result (bit 0).
    Machine-checked in benchmark/soundness/prove_comparison_taint.py.
    """
    cdef public Expr a_val
    cdef public Expr a_taint
    cdef public Expr b_val
    cdef public Expr b_taint
    cdef public int width
    cdef public bint is_signed
    cdef public bint or_equal

    def __init__(self, Expr a_val, Expr a_taint, Expr b_val, Expr b_taint,
                 int width, bint is_signed, bint or_equal):
        self.a_val = a_val
        self.a_taint = a_taint
        self.b_val = b_val
        self.b_taint = b_taint
        self.width = width
        self.is_signed = is_signed
        self.or_equal = or_equal

    def __str__(self):
        op = ('<=' if self.or_equal else '<') + ('s' if self.is_signed else 'u')
        return f'CMP_TAINT[{op},w={self.width}]({self.a_val}, {self.b_val})'

    def __repr__(self):
        return (
            f'ComparisonTaintExpr(a_val={repr(self.a_val)}, b_val={repr(self.b_val)}, '
            f'width={self.width}, is_signed={self.is_signed}, or_equal={self.or_equal})'
        )

    cpdef object evaluate(self, EvalContext context):
        cdef int w = self.width
        cdef object mask = ((<object>1) << w) - 1
        cdef object a = self.a_val.evaluate(context) & mask
        cdef object ta = self.a_taint.evaluate(context) & mask
        cdef object b = self.b_val.evaluate(context) & mask
        cdef object tb = self.b_taint.evaluate(context) & mask
        if self.is_signed:
            # signed compare == unsigned compare after flipping the sign bit
            a = a ^ ((<object>1) << (w - 1))
            b = b ^ ((<object>1) << (w - 1))
        cdef object amin = a & ~ta & mask
        cdef object amax = a | ta
        cdef object bmin = b & ~tb & mask
        cdef object bmax = b | tb
        cdef int can_true, always_true
        if self.or_equal:
            can_true = 1 if amin <= bmax else 0
            always_true = 1 if amax <= bmin else 0
        else:
            can_true = 1 if amin < bmax else 0
            always_true = 1 if amax < bmin else 0
        return can_true ^ always_true


cdef class EqualityTaintExpr(Expr):
    """EXACT taint of an equality bit ``[a == b]`` / ``[a != b]`` (identical sensitivity).

    Equality is SYMMETRIC -- non-monotone in BOTH directions -- so it has no polarity
    orientation and the 2-corner differential collapses (both corners land in the equal
    regime).  Compute directly whether the equality can VARY over the taint cube::

        equal_achievable   = ((a ^ b) & ~(Ta | Tb)) == 0    # every FIXED bit already agrees
        unequal_achievable = (Ta | Tb) != 0                 # some bit is free to break equality
        Tr = equal_achievable AND unequal_achievable

    (equal is reachable iff the bits neither side taints already match, since any free bit
    can be set to match; unequal is reachable iff at least one bit is free to differ.)
    EXACT when a and b are INDEPENDENT (builder-gated); ``[a==a]`` is a constant handled by
    the differential.  1-bit result (bit 0).  Machine-checked in
    benchmark/soundness/prove_equality_taint.py.
    """
    cdef public Expr a_val
    cdef public Expr a_taint
    cdef public Expr b_val
    cdef public Expr b_taint
    cdef public int width

    def __init__(self, Expr a_val, Expr a_taint, Expr b_val, Expr b_taint, int width):
        self.a_val = a_val
        self.a_taint = a_taint
        self.b_val = b_val
        self.b_taint = b_taint
        self.width = width

    def __str__(self):
        return f'EQ_TAINT[w={self.width}]({self.a_val}, {self.b_val})'

    def __repr__(self):
        return (
            f'EqualityTaintExpr(a_val={repr(self.a_val)}, b_val={repr(self.b_val)}, '
            f'width={self.width})'
        )

    cpdef object evaluate(self, EvalContext context):
        cdef int w = self.width
        cdef object mask = ((<object>1) << w) - 1
        cdef object a = self.a_val.evaluate(context) & mask
        cdef object ta = self.a_taint.evaluate(context) & mask
        cdef object b = self.b_val.evaluate(context) & mask
        cdef object tb = self.b_taint.evaluate(context) & mask
        cdef object free = ta | tb
        cdef int equal_ach = 1 if ((a ^ b) & ~free & mask) == 0 else 0
        cdef int unequal_ach = 1 if free != 0 else 0
        return 1 if (equal_ach and unequal_ach) else 0


cdef class TaintOperand(Expr):
    cdef public str name
    cdef public int bit_start
    cdef public int bit_end
    cdef public bint is_taint

    def __init__(self, str name, int bit_start, int bit_end, bint is_taint=True):
        self.name = name
        self.bit_start = bit_start
        self.bit_end = bit_end
        self.is_taint = is_taint

    def __str__(self):
        prefix = 'T' if self.is_taint else 'V'
        if self.bit_start == self.bit_end:
            return f'{prefix}_{self.name}[{self.bit_start}]'
        return f'{prefix}_{self.name}[{self.bit_end}:{self.bit_start}]'

    def __repr__(self):
        return f"TaintOperand(name='{self.name}', bits={self.bit_end}:{self.bit_start}, is_taint={self.is_taint})"


    cpdef object evaluate(self, EvalContext context):
        cdef dict state = context.input_taint if self.is_taint else context.input_values
        cdef object val
        cdef object mask
        cdef str arch_str
        cdef dict parent_map
        cdef object parent_val
        cdef object parent_name
        cdef int bit_start_in_parent

        val = state.get(self.name, None)

        if val is None:
            # State is normalized to parents, so if name not found,
            # it must be a child — look up its parent.
            # arch_str is pre-cached on context — no str(simulator.arch) overhead.
            parent_map = _ARCH_PARENT_REGS.get(context.arch_str, {})
            if self.name in parent_map:
                parent_name, bit_start_in_parent = parent_map[self.name]
                parent_val = state.get(parent_name, None)
                if parent_val is not None:
                    val = parent_val >> bit_start_in_parent
            if val is None:
                val = 0

        mask = (<object>1 << (self.bit_end - self.bit_start + 1)) - 1
        return (val >> self.bit_start) & mask

cdef class MemoryOperand(Expr):
    cdef public Expr address_expr
    cdef public int size
    cdef public bint is_taint

    def __init__(self, Expr address_expr, int size, bint is_taint=True):
        self.address_expr = address_expr
        self.size = size
        self.is_taint = is_taint

    def __str__(self):
        prefix = 'T' if self.is_taint else 'V'
        return f'{prefix}_MEM[{self.address_expr}, size={self.size}]'

    def __repr__(self):
        return f"MemoryOperand(address_expr={repr(self.address_expr)}, size={self.size}, is_taint={self.is_taint})"

    cpdef object evaluate(self, EvalContext context):
        cdef object address = self.address_expr.evaluate(context)
        
        # 1. Native Shadow Memory Integration
        if self.is_taint and context.shadow_memory is not None:
            return context.shadow_memory.read_mask(address, self.size)
            
        # 2. Native Live Memory Reader Integration
        if not self.is_taint and context.mem_reader is not None:
            return context.mem_reader(address, self.size)
            
        # 3. Fallback to dictionary
        cdef str mem_name = f'MEM_{hex(address)}_{self.size}'
        cdef dict state = context.input_taint if self.is_taint else context.input_values
        return state.get(mem_name, 0)


cdef class Constant(Expr):
    cdef public object value
    cdef public int size

    def __init__(self, object value, int size):
        self.value = value
        self.size = size

    def __str__(self):
        return hex(self.value) if isinstance(self.value, int) else str(self.value)

    def __repr__(self):
        return f"Constant(value={hex(self.value) if isinstance(self.value, int) else self.value}, size={self.size})"

    cpdef object evaluate(self, EvalContext context):
        return self.value


cdef class UnaryExpr(Expr):
    cdef public object op
    cdef int _op_int
    cdef public Expr expr

    def __init__(self, object op, Expr expr):
        self.op = op
        self._op_int = _OP_MAP.get(op, -1)
        self.expr = expr

    def __str__(self):
        return f'{self.op.value}({self.expr})'

    def __repr__(self):
        return f"UnaryExpr(op={self.op}, expr={repr(self.expr)})"

    cpdef object evaluate(self, EvalContext context):
        cdef object val = self.expr.evaluate(context)
        if self._op_int == _OP_NOT:
            return ~val
        raise NotImplementedError(f'Unsupported unary op {self.op}')


cdef class BinaryExpr(Expr):
    cdef public object op
    cdef int _op_int  # fast int dispatch
    cdef public Expr lhs
    cdef public Expr rhs

    def __init__(self, object op, Expr lhs, Expr rhs):
        self.op = op
        self._op_int = _OP_MAP.get(op, -1)
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return f'({self.lhs} {self.op.value} {self.rhs})'

    def __repr__(self):
        return f"BinaryExpr(op={self.op}, lhs={repr(self.lhs)}, rhs={repr(self.rhs)})"

    cpdef object evaluate(self, EvalContext context):
        cdef object left = self.lhs.evaluate(context)
        cdef object right = self.rhs.evaluate(context)
        cdef int op = self._op_int
        if op == _OP_AND:   return left & right
        if op == _OP_OR:    return left | right
        if op == _OP_XOR:   return left ^ right
        if op == _OP_LEFT:  return left << right
        if op == _OP_RIGHT: return left >> right
        if op == _OP_ADD:   return left + right
        if op == _OP_SUB:   return left - right
        raise NotImplementedError(f'Unsupported binary op {self.op}')


cdef class TaintAssignment:
    cdef public object target
    cdef public list dependencies
    cdef public Expr expression
    cdef public str expression_str
    cdef public bint is_mem_target  # pre-tagged: True if target has address_expr

    def __init__(self, object target, list dependencies, Expr expression=None, str expression_str=''):
        self.target = target
        self.dependencies = dependencies
        self.expression = expression
        self.expression_str = expression_str
        self.is_mem_target = hasattr(target, 'address_expr')

    def __str__(self):
        cdef str expr_str
        if self.expression is not None:
            expr_str = str(self.expression)
        elif self.expression_str:
            expr_str = self.expression_str
        else:
            expr_str = ' | '.join(str(d) for d in self.dependencies)
        return f'{self.target} = {expr_str}'

    def __repr__(self):
        return f"TaintAssignment(target={repr(self.target)}, expression={repr(self.expression)})"


def _collect_taint_operand_names(expr, result_set):
    """Recursively collect TaintOperand register names from an expression tree."""
    if expr is None:
        return
    if isinstance(expr, TaintOperand):
        if expr.name and not expr.name.startswith('MEM_'):
            result_set.add(expr.name)
    elif isinstance(expr, BinaryExpr):
        _collect_taint_operand_names(expr.lhs, result_set)
        _collect_taint_operand_names(expr.rhs, result_set)
    elif isinstance(expr, UnaryExpr) or isinstance(expr, AvalancheExpr):
        _collect_taint_operand_names(expr.expr, result_set)
    elif isinstance(expr, InstructionCellExpr):
        for _sub_expr in expr.inputs.values():
            _collect_taint_operand_names(_sub_expr, result_set)


cdef class LogicCircuit:
    cdef public list assignments
    cdef public object architecture
    cdef public str instruction
    cdef public list state_format
    cdef public str _pc_target      # pre-computed: 'RIP'/'EIP'/'PC' or None
    cdef public bint has_unicorn_cells  # True if any assignment uses InstructionCellExpr
    cdef public object input_reg_names  # set of register names needed as value inputs
    cdef public object _compiled       # cached CompiledCircuit (or None if compile failed/disabled)

    def __init__(self, list assignments, object architecture, str instruction, list state_format):
        self.assignments = assignments
        self.architecture = architecture
        self.instruction = instruction
        self.state_format = state_format
        # Pre-compute which PC register (if any) is a target — checked every evaluate()
        self._pc_target = None
        self.has_unicorn_cells = False
        self.input_reg_names = set()  # register names needed as VALUE inputs
        self._compiled = None
        for _a in assignments:
            if not _a.is_mem_target and _a.target.name in ('RIP', 'EIP', 'PC'):
                self._pc_target = _a.target.name
            if isinstance(_a.expression, InstructionCellExpr):
                self.has_unicorn_cells = True
            # Collect TaintOperand register names (both taint + value operands)
            _collect_taint_operand_names(_a.expression, self.input_reg_names)

    def __str__(self):
        return '\n'.join(str(a) for a in self.assignments)

    def __repr__(self):
        return f"LogicCircuit(instr={self.instruction}, assignments_count={len(self.assignments)})"

    cpdef dict evaluate(self, EvalContext context):
        # Compiled-bytecode fast path:  if circuit_c is importable and the
        # circuit has a compiled form (or one can be built lazily), use it.
        # Disabled by setting the env var MICROTAINT_DISABLE_COMPILED_CIRCUIT=1
        # or by setting LogicCircuit._compiled to a sentinel.
        #
        # If self.assignments was mutated since the last compile (rare: tests
        # do this; production hot path never does), invalidate the cache.
        cdef int n_live = len(self.assignments)
        if self._compiled is not None and self._compiled is not False:
            try:
                if self._compiled.stats()['n_assignments'] != n_live:
                    self._compiled = None
            except Exception:
                self._compiled = None
        if self._compiled is None:
            # The compiled-circuit fast path is on by default.  Setting
            # MICROTAINT_DISABLE_COMPILED_CIRCUIT=1 forces the Cython AST
            # walker (kept around as a reference / debug fallback).
            if os.environ.get('MICROTAINT_DISABLE_COMPILED_CIRCUIT') == '1':
                self._compiled = False
            else:
                try:
                    # Pass pcode if available — enables CellHandle pre-resolution
                    # so OP_CALL_CELL skips the Python boundary entirely.
                    pcode = None
                    if context.simulator is not None:
                        pcode = getattr(context.simulator, '_pcode', None)
                    if pcode is not None:
                        self._compiled = compile_circuit(self, pcode)
                    else:
                        self._compiled = compile_circuit(self)
                except Exception:
                    # compile_circuit raises if the AST contains an
                    # unsupported expression form (e.g. expression_str=
                    # 'FOO').  Fall back to the Cython AST walker; the
                    # call site below detects _compiled is False and
                    # uses the slow path.
                    self._compiled = False
        if self._compiled is not False and self._compiled is not None:
            return self._compiled.evaluate(context)

        # Cython AST fallback (the original implementation):
        # Cache frequently accessed context fields as C locals — avoids repeated
        # Python property dispatch for each field access in the hot loop.
        cdef dict output_taint = context.input_taint.copy()
        cdef object implicit_policy = context.implicit_policy
        cdef object val
        cdef object address
        cdef str target_name
        cdef int bit_start, bit_end
        cdef object mask
        cdef object current
        cdef Expr dep
        cdef TaintAssignment assignment
        cdef int i
        cdef int num_assignments = len(self.assignments)
        
        # High speed typed C-loop, bypassing Python iterator overhead
        for i in range(num_assignments):
            assignment = <TaintAssignment>self.assignments[i]
            
            # 1. Evaluate the expression or fallback to default OR dependencies
            if assignment.expression is not None:
                val = assignment.expression.evaluate(context)
            elif assignment.expression_str:
                raise NotImplementedError('Arbitrary string expressions not supported.')
            else:
                val = 0
                for dep in assignment.dependencies:
                    val |= dep.evaluate(context)

            # 2. Determine bounds (Memory vs Register) — pre-tagged at construction
            if assignment.is_mem_target:
                address = assignment.target.address_expr.evaluate(context)
                target_name = f'MEM_{hex(address)}_{assignment.target.size}'
                bit_start = 0
                bit_end = assignment.target.size * 8 - 1
            else:
                target_name = assignment.target.name
                bit_start = assignment.target.bit_start
                bit_end = assignment.target.bit_end

            # 3. Apply bit-precise masking to preserve partial registers
            mask = ((<object>1 << (bit_end - bit_start + 1)) - 1) << bit_start
            val = (val << bit_start) & mask

            current = output_taint.get(target_name, 0)
            output_taint[target_name] = (current & ~mask) | val

        # --- THE IMPLICIT TAINT INTERCEPTOR ---
        cdef str pc_reg = None
        if self._pc_target is not None and output_taint.get(self._pc_target, 0) != 0:
            pc_reg = self._pc_target

        if pc_reg is not None:
            from microtaint.types import ImplicitTaintPolicy, ImplicitTaintError
            
            if implicit_policy == ImplicitTaintPolicy.WARN:
                print(
                    f"[Microtaint] Implicit Taint Detected! "
                    f"Control flow ({pc_reg}) depends on tainted data at instruction: {self.instruction}"
                )
            
            elif implicit_policy == ImplicitTaintPolicy.STOP:
                raise ImplicitTaintError(
                    f"\n[!] FATAL: Implicit Taint Detected\n"
                    f"    Instruction (Hex): {self.instruction}\n"
                    f"    Tainted Register : {pc_reg}\n"
                    f"    Taint Mask       : {hex(output_taint[pc_reg])}\n"
                    f"    Reason: The execution of this branch is governed by a tainted condition."
                )
            
            if implicit_policy != ImplicitTaintPolicy.KEEP:
                del output_taint[pc_reg]

        return output_taint


def _run_concrete_step(sub, dict values, sim):
    """Run a sub-circuit's instruction concretely and return updated values.

    Used by ``ChainedCircuit.evaluate`` to thread concrete state across
    sequence steps.  Concrete results are independent of taint, so we use
    the current ``values`` (which start as the chain's entry state).

    Returns a NEW dict; the input ``values`` is not mutated.

    Failures fall back to returning the input dict unchanged — equivalent
    to the pre-fix behaviour (concrete values stale).  This keeps the
    chain evaluator robust against unsupported instructions.
    """
    try:
        bytestring = bytes.fromhex(sub.instruction)
    except (ValueError, AttributeError):
        return values

    # Build a MachineState from the current concrete state.  We only need
    # the registers in state_format; memory and shadow are unchanged.
    from microtaint.simulator import MachineState, _native_be_safe
    regs = {}
    for reg_obj in sub.state_format:
        name = reg_obj.name
        if name in values:
            regs[name] = values[name]

    # On a big-endian target Unicorn cannot seed or read the PPC XER carry/overflow
    # varnodes, so threading concrete state through it silently drops the carry bit
    # (and any GPR whose value depends on a carry-in).  Read the whole post-state
    # back from the native p-code kernel instead, which models those varnodes and
    # is byte-order agnostic for native-BE-safe instructions (see _native_be_safe).
    cdef bint be_native
    try:
        be_native = sim._is_big_endian and _native_be_safe(sim.arch, sub.instruction)
    except Exception:
        be_native = False
    if be_native:
        new_values = dict(values)
        try:
            for reg_obj in sub.state_format:
                new_values[reg_obj.name] = sim._read_reg_concrete(
                    sub.instruction, regs, reg_obj.name, reg_obj.bits,
                )
            return new_values
        except Exception:
            pass  # native kernel cannot decode it -> fall back to Unicorn below

    state = MachineState(regs=regs, mem={})
    try:
        sim._execute(bytestring, state)
    except Exception:
        # Any execution failure (unmapped memory, illegal instruction, etc.)
        # leaves the previous values untouched — sound fallback.
        return values

    new_values = dict(values)
    for reg_obj in sub.state_format:
        name = reg_obj.name
        try:
            new_values[name] = sim._read_reg(name)
        except Exception:
            # Failed to read a register — leave it unchanged.
            pass
    return new_values


cdef class ChainedCircuit:
    """A sequence of LogicCircuits evaluated one-by-one, threading the output
    taint AND the concrete state of each step into the next.

    This is used for multi-instruction sequences.  Lifting all instructions
    into a single P-code block and analysing them as a unit (``LogicCircuit``)
    loses intermediate state: if instruction 1 writes CL and instruction 2
    reads CL, the static rule for the joined block sees the *original* CL dep
    rather than the updated one.  Chaining is the correct compositional fix.

    Concrete-value threading
    ------------------------
    Between steps we run each sub-circuit's instruction concretely on the
    current value state and merge the result back.  This is essential for
    soundness: per-opcode taint formulas like AND
    ``(V_a & T_b) | (V_b & T_a) | (T_a & T_b)`` read the *post-step* concrete
    values of source registers and can under-taint when those values are
    stale.  Example: ``mov rbx, 0xff00...; and rax, rbx`` — without value
    threading, the AND uses the entry V_RBX instead of the post-mov value,
    which can both over- and under-count taint.
    """

    cdef public list sub_circuits
    cdef public object architecture
    cdef public str instruction
    cdef public list state_format

    def __init__(self, list sub_circuits, object architecture, str instruction, list state_format):
        self.sub_circuits = sub_circuits
        self.architecture = architecture
        self.instruction = instruction
        self.state_format = state_format

    def __repr__(self):
        return (f'ChainedCircuit(instr={self.instruction}, '
                f'n_steps={len(self.sub_circuits)})')

    cpdef dict evaluate(self, EvalContext context):
        cdef dict taint = dict(context.input_taint)
        cdef dict values = dict(context.input_values)  # mutable concrete state
        cdef EvalContext step_ctx
        cdef LogicCircuit sub
        cdef object sim = context.simulator

        for sub in self.sub_circuits:
            # Build a new context with the running taint AND running concrete
            # state.  Updating concrete values between steps is essential for
            # soundness: value-dependent taint formulas (notably AND/OR with
            # bits set/cleared by an earlier instruction) read the concrete
            # values of source registers and can under-taint when those
            # values are stale.
            step_ctx = EvalContext(
                input_taint=taint,
                input_values=values,
                simulator=sim,
                implicit_policy=context.implicit_policy,
                shadow_memory=context.shadow_memory,
                mem_reader=context.mem_reader,
            )
            taint = sub.evaluate(step_ctx)

            # Update concrete state by running THIS step's instruction
            # concretely on the running values.  The result is independent of
            # taint (concrete bits don't depend on which bits are tainted),
            # so any valid concrete state works — we use the current `values`.
            #
            # Skipping is safe (correct but possibly imprecise) only if the
            # next step does not read any register written by this step;
            # since we cannot know that statically without re-analysing every
            # sub-circuit, we always run the concrete update.  The cost is
            # one Unicorn execution per chain step; for chains of 2–6 instrs
            # this is negligible compared to the differential evaluation.
            if sim is not None and sub.instruction:
                values = _run_concrete_step(sub, values, sim)

        # Filter to the registers in the caller's state_format.
        # Sub-circuits may have been built with an augmented format that includes
        # flag registers (CF, ZF, etc.) not in the caller's format — those extra
        # keys must not appear in the output dict.
        cdef set caller_names = {reg.name for reg in self.state_format}
        return {k: v for k, v in taint.items() if k in caller_names or k.startswith('MEM_')}

    @property
    def assignments(self) -> list:
        """Flattened list of all assignments across every sub-circuit.

        Provided for structural compatibility with LogicCircuit so that
        code that inspects ``circuit.assignments`` works on both types.
        The list is rebuilt on each access — callers that need it
        repeatedly should cache the result.
        """
        result = []
        for sub in self.sub_circuits:
            result.extend(sub.assignments)
        return result


cdef class InstructionCellExpr(Expr):
    cdef public object architecture
    cdef public str instruction
    cdef public str out_reg
    cdef public int out_bit_start
    cdef public int out_bit_end
    cdef public dict inputs

    def __init__(self, object architecture, str instruction, str out_reg, int out_bit_start, int out_bit_end, dict inputs):
        self.architecture = architecture
        self.instruction = instruction
        self.out_reg = out_reg
        self.out_bit_start = out_bit_start
        self.out_bit_end = out_bit_end
        self.inputs = inputs

    def __str__(self):
        args = ', '.join(f'{k}={v}' for k, v in self.inputs.items())
        return f'SimulateCell(instr=0x{self.instruction}, out={self.out_reg}[{self.out_bit_end}:{self.out_bit_start}], {args})'

    def __repr__(self):
        return f"InstructionCellExpr(instr={self.instruction}, out_reg='{self.out_reg}', inputs={repr(self.inputs)})"

    cpdef object evaluate(self, EvalContext context):
        cdef dict evaluated_inputs = {}
        cdef str name
        cdef Expr expr
        cdef object sim = context.simulator  # cache — avoids repeated __get__ dispatch
        cdef object pcode

        for name, expr in self.inputs.items():
            evaluated_inputs[name] = expr.evaluate(context)

        # Fast path: if the simulator's pcode evaluator implements evaluate_concrete_flat
        # (the C evaluator does), skip MachineState construction entirely.
        # This avoids a Python dataclass alloc + dict-merge per call (~0.4 us savings).
        if sim is not None and not sim.use_unicorn:
            pcode = sim._pcode
            if pcode is not None and hasattr(pcode, 'evaluate_concrete_flat'):
                try:
                    return pcode.evaluate_concrete_flat(self, evaluated_inputs)
                except sim._pcode_fallback_exc:
                    pcode.fallback_calls += 1
                    # Fall through to the standard MachineState path below.

        m_state = _build_machine_state(evaluated_inputs, context)
        return sim.evaluate_concrete(self, m_state)

cdef class MemoryDifferentialExpr(Expr):
    """
    Differential evaluator for instructions whose output is a memory write
    (RMW: read-modify-write) and/or whose value depends on memory inputs
    that the standard ``InstructionCellExpr`` path resolves incorrectly.

    Why this class exists
    ---------------------
    The standard differential is built as
    ``BinaryExpr(XOR, InstructionCellExpr(rep1), InstructionCellExpr(rep2))``,
    where ``InstructionCellExpr.evaluate`` builds a ``MachineState`` via
    ``_build_machine_state``.  For pure register inputs that path is fast
    and correct.  For instructions with memory inputs whose addresses
    involve an offset (e.g. ``[rbp-0x10]``) or whose address register is
    not also a value dep (e.g. ``[rax]`` in ``add rdx, [rax]``), the
    standard path produces a ``MachineState`` with the memory value at
    the wrong address and/or the address register missing from
    ``state.regs``.  Both bugs collapse the differential to
    ``OR-of-input-bits``, breaking SipHash-style avalanche.

    This class bypasses ``_build_machine_state`` by calling the underlying
    ``cell.pyx`` ``evaluate_differential`` with two flat ``MEM_<reg>_<offset>_<size>``
    keyed dicts (``or_inputs`` and ``and_inputs``) — a format that
    ``cell.pyx`` ``_load`` and ``_read_output`` parse natively.

    Performance
    -----------
    - One ``evaluate_differential`` call per instruction execution (the same
      two Unicorn/p-code runs the broken path was already trying to do).
    - Built on top of cell.pyx's existing ``_frame_a`` / ``_frame_b`` shared
      buffers — no extra allocation.
    - Construction (the engine emits this once per unique instruction byte
      sequence) is cached by ``_cached_generate_static_rule``'s LRU.

    Constructor parameters
    ----------------------
    bytestring : bytes
        Raw instruction bytes.
    target : tuple
        ``('MEM', addr_reg_name, addr_const_offset, size_bytes)``  for memory targets,
        or
        ``('REG', name, bit_start, bit_end)``                      for register targets.
    reg_inputs : list[tuple[str, int, int]]
        Register value-deps that contribute to the result, as
        ``(name, bit_start, bit_end)``.  Both V and T are populated for these.
    mem_inputs : list[tuple[str, int, int]]
        Memory value-deps as ``(addr_reg_name, addr_const_offset, size_bytes)``.
    addr_only_regs : list[str]
        Registers that appear ONLY as memory address bases (no value
        contribution).  Their concrete value must be in v_state.regs so the
        simulator can resolve ``[reg+offset]`` correctly; their taint is
        excluded (address-taint is an AIW signal, handled separately).
    """

    cdef public bytes bytestring
    cdef public object target            # tuple, kept as Python object
    cdef public list reg_inputs
    cdef public list mem_inputs
    cdef public list addr_only_regs
    cdef public object neg_inputs        # frozenset of negatively-polarised input keys
    cdef public str _instr_hex           # cached hex form of bytestring
    cdef public str _target_out_reg      # cached out_reg string for cell
    cdef public int _target_bit_start
    cdef public int _target_bit_end

    def __init__(
        self,
        bytes bytestring,
        object target,
        list reg_inputs,
        list mem_inputs,
        list addr_only_regs,
        list neg_inputs=None,
    ):
        cdef str kind, name, addr_reg
        cdef int b_start, b_end
        cdef long offset, size_bytes

        self.bytestring     = bytestring
        self.target         = target
        self.reg_inputs     = reg_inputs
        self.mem_inputs     = mem_inputs
        self.addr_only_regs = addr_only_regs
        # Keys (register names and MEM_<reg>_<off>_<sz> strings) of value-deps
        # that are subtracted operands and must be polarised oppositely so the
        # differential captures the sound D^{+-} borrow chain, not a lossy
        # D^{++}.  Empty -> all inputs positive (legacy behaviour).
        self.neg_inputs     = frozenset(neg_inputs) if neg_inputs else frozenset()
        self._instr_hex     = bytestring.hex()

        # Pre-compute the cell out_reg string.  Both formats use the same
        # parser path in cell.pyx ``_read_output``.
        kind = <str>target[0]
        if kind == 'MEM':
            addr_reg   = <str>target[1]
            offset     = <long>target[2]
            size_bytes = <long>target[3]
            self._target_out_reg   = f'MEM_{addr_reg}_{offset}_{size_bytes}'
            self._target_bit_start = 0
            self._target_bit_end   = <int>(size_bytes * 8 - 1)
        else:  # 'REG'
            name    = <str>target[1]
            b_start = <int>target[2]
            b_end   = <int>target[3]
            self._target_out_reg   = name
            self._target_bit_start = b_start
            self._target_bit_end   = b_end

    def __repr__(self):
        return (
            f'MemoryDifferentialExpr(instr={self._instr_hex}, '
            f'target={self.target}, regs={self.reg_inputs}, '
            f'mem={self.mem_inputs}, addr_only={self.addr_only_regs})'
        )

    def __str__(self):
        return self.__repr__()

    cpdef object evaluate(self, EvalContext context):
        cdef dict or_inputs  = {}
        cdef dict and_inputs = {}
        cdef dict input_values = context.input_values
        cdef dict input_taint  = context.input_taint
        cdef object shadow_memory = context.shadow_memory
        cdef object mem_reader    = context.mem_reader
        cdef object sim           = context.simulator

        cdef str name, addr_reg
        cdef int b_start, b_end
        cdef long offset, size_bytes
        cdef object v, t, slice_mask, t_slice
        cdef object base, addr
        cdef object v_val, t_val
        cdef str mem_key
        cdef bint neg

        if sim is None:
            return 0

        # ---- Register VALUE-deps: per-SLICE polarisation ----
        # A negatively-polarised (subtracted) operand gets the opposite images
        # (or := V&~T, and := V|T) so the differential is the sound D^{+-}.
        #
        # Polarity is a property of the dependency SLICE, not of the register
        # name: apply_sless_msb_split deliberately splits ONE register into two
        # slices with OPPOSITE polarity (sign bit -1, magnitude +1), because a
        # signed comparison is monotone only in the sign-biased representation.
        # So look the polarity up per (name, bit_start, bit_end), and ACCUMULATE
        # each slice into the images -- starting from V and rewriting only that
        # slice's tainted bits.  Keying by name (and overwriting the whole entry
        # per slice) mis-polarised the magnitude half and dropped the first slice,
        # under-tainting `cmp rax,[mem]; setl cl`.  This mirrors what
        # build_polarized_reg already does on the pure-register path.
        for name, b_start, b_end in self.reg_inputs:
            if name not in or_inputs:
                v = input_values.get(name, 0)
                or_inputs[name]  = v
                and_inputs[name] = v

        for name, b_start, b_end in self.reg_inputs:
            t = input_taint.get(name, 0)
            neg = (name, b_start, b_end) in self.neg_inputs
            slice_mask = (((<object>1) << (b_end - b_start + 1)) - 1) << b_start
            t_slice = t & slice_mask
            if neg:
                or_inputs[name]  = or_inputs[name] & ~t_slice
                and_inputs[name] = and_inputs[name] | t_slice
            else:
                or_inputs[name]  = or_inputs[name] | t_slice
                and_inputs[name] = and_inputs[name] & ~t_slice

        # ---- Address-only registers: same value in both runs ----
        for name in self.addr_only_regs:
            if name in or_inputs:
                continue  # already handled as a value-dep
            v = input_values.get(name, 0)
            or_inputs[name]  = v
            and_inputs[name] = v

        # ---- Memory value-deps: read V from mem_reader, T from shadow ----
        for addr_reg, offset, size_bytes in self.mem_inputs:
            base = input_values.get(addr_reg, 0)
            addr = (base + offset) & 0xFFFFFFFFFFFFFFFF
            if mem_reader is not None:
                try:
                    v_val = mem_reader(addr, size_bytes)
                except Exception:
                    v_val = 0
            else:
                v_val = 0
            if shadow_memory is not None:
                try:
                    t_val = shadow_memory.read_mask(addr, size_bytes)
                except Exception:
                    t_val = 0
            else:
                t_val = 0
            mem_key = f'MEM_{addr_reg}_{offset}_{size_bytes}'
            if mem_key in self.neg_inputs:
                or_inputs[mem_key]  = v_val & ~t_val
                and_inputs[mem_key] = v_val | t_val
            else:
                or_inputs[mem_key]  = v_val | t_val
                and_inputs[mem_key] = v_val & ~t_val

        # ---- Run the differential through cell.pyx's native path ----
        # Direct dispatch to the C kernel: skip the
        # ``simulator.evaluate_differential`` Python middleman that just
        # forwards to ``self._pcode.evaluate_differential`` and catches
        # the fallback exception. ~205k calls/run x 0.32 us frame =
        # ~65 ms saved on the bench.  We replicate the middleman's
        # PCodeFallbackNeeded handling locally; if anything else goes
        # wrong, we drop into the OR-of-input-taints fallback below.
        cdef object pcode = sim._pcode if sim is not None else None
        cdef object fallback_exc = sim._pcode_fallback_exc if sim is not None else Exception
        # The native C kernel indexes registers little-endian, so on a big-endian
        # target (sim.use_unicorn is forced True there) it reads the wrong bytes.
        # Skip it and let sim.evaluate_differential run the two replicas through
        # Unicorn, which models byte order correctly.
        cdef bint use_unicorn = sim.use_unicorn if sim is not None else False
        try:
            if pcode is not None and not use_unicorn:
                try:
                    return pcode.evaluate_differential(self, or_inputs, and_inputs)
                except fallback_exc:
                    # Same fallback as simulator.evaluate_differential:
                    # bump the kernel's fallback counter and let the
                    # outer Unicorn-based path run.
                    pcode.fallback_calls += 1
                    return sim.evaluate_differential(
                        self, or_inputs, and_inputs,
                    )
            return sim.evaluate_differential(
                self, or_inputs, and_inputs,
            )
        except Exception:
            # OR-of-input-taints fallback.  Conservative: all explicitly
            # tainted bits are always reported even on simulator failure.
            fallback = 0
            for name, b_start, b_end in self.reg_inputs:
                fallback |= input_taint.get(name, 0)
            for addr_reg, offset, size_bytes in self.mem_inputs:
                if shadow_memory is not None:
                    base = input_values.get(addr_reg, 0)
                    addr = (base + offset) & 0xFFFFFFFFFFFFFFFF
                    try:
                        fallback |= shadow_memory.read_mask(addr, size_bytes)
                    except Exception:
                        pass
            width = self._target_bit_end - self._target_bit_start + 1
            mask = ((<object>1) << width) - 1 if width < 64 else 0xFFFFFFFFFFFFFFFF
            return (fallback >> self._target_bit_start) & mask

    # cell.pyx's evaluate_differential reads `cell.instruction`, `cell.out_reg`,
    # `cell.out_bit_start`, `cell.out_bit_end` — proxy these via Python
    # attribute access using our pre-cached values.
    @property
    def instruction(self):
        return self._instr_hex

    @property
    def out_reg(self):
        return self._target_out_reg

    @property
    def out_bit_start(self):
        return self._target_bit_start

    @property
    def out_bit_end(self):
        return self._target_bit_end