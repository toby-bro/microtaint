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

# Mapping from Op enum to int at module init (done once)
_OP_MAP: dict = {}

def _init_op_map():
    global _OP_MAP
    _OP_MAP = {
        Op.AND: _OP_AND, Op.OR: _OP_OR, Op.XOR: _OP_XOR,
        Op.LEFT: _OP_LEFT, Op.ADD: _OP_ADD, Op.SUB: _OP_SUB,
        Op.NOT: _OP_NOT,
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


cdef class ChainedCircuit:
    """A sequence of LogicCircuits evaluated one-by-one, threading the output
    taint of each step into the input taint of the next.

    This is used for multi-instruction sequences.  Lifting all instructions
    into a single P-code block and analysing them as a unit (``LogicCircuit``)
    loses intermediate state: if instruction 1 writes CL and instruction 2
    reads CL, the static rule for the joined block sees the *original* CL dep
    rather than the updated one.  Chaining is the correct compositional fix.

    The EvalContext is rebuilt before each sub-circuit with the taint dict
    from the previous step.  Concrete ``input_values`` are held constant
    (they reflect the entry state of the whole sequence).  This is sound but
    slightly conservative for value-dependent taint (e.g. `and` with a
    concrete 0 mask) because the concrete values don't update between steps;
    however, the static-rule evaluator already doesn't propagate concrete
    values across steps, so this is no worse than before.
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
        cdef EvalContext step_ctx
        cdef LogicCircuit sub

        for sub in self.sub_circuits:
            # Build a new context with the running taint state but the same
            # concrete values and simulator.  The implicit_policy is propagated
            # so PC-taint detection fires within any step.
            step_ctx = EvalContext(
                input_taint=taint,
                input_values=context.input_values,
                simulator=context.simulator,
                implicit_policy=context.implicit_policy,
                shadow_memory=context.shadow_memory,
                mem_reader=context.mem_reader,
            )
            taint = sub.evaluate(step_ctx)

        return taint

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
    ):
        cdef str kind, name, addr_reg
        cdef int b_start, b_end
        cdef long offset, size_bytes

        self.bytestring     = bytestring
        self.target         = target
        self.reg_inputs     = reg_inputs
        self.mem_inputs     = mem_inputs
        self.addr_only_regs = addr_only_regs
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

        if sim is None:
            return 0

        # ---- Register VALUE-deps: full V|T / V&~T polarisation ----
        for name, b_start, b_end in self.reg_inputs:
            v = input_values.get(name, 0)
            t = input_taint.get(name, 0)
            if b_start == 0 and b_end >= 63:
                or_inputs[name]  = v | t
                and_inputs[name] = v & ~t
            else:
                # Polarise only the slice bits; preserve other bits as V.
                slice_mask = (((<object>1) << (b_end - b_start + 1)) - 1) << b_start
                t_slice = t & slice_mask
                or_inputs[name]  = (v & ~t_slice) | t_slice
                and_inputs[name] = v & ~t_slice

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
        try:
            if pcode is not None:
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