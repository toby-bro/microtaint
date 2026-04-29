# cython: language_level=3
# cython: profile=False
from enum import Enum
from microtaint.simulator import CellSimulator, MachineState
from microtaint.types import Architecture, Register

def _build_machine_state(dict input_dict, EvalContext context):
    cdef dict regs = {}
    cdef dict mem = {}
    cdef str name
    cdef object val
    cdef str ptr_part
    cdef object addr

    for name, val in input_dict.items():
        if name.startswith('MEM_'):
            ptr_part = name[4:]
            if ptr_part.startswith('0x'):
                addr = int(ptr_part.split('_')[0], 16)
            else:
                addr = context.input_values.get(ptr_part, 0)
            mem[addr] = val
        else:
            regs[name] = val
    return MachineState(regs=regs, mem=mem)


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
    cdef dict result = {}
    cdef str key
    cdef object val
    cdef str parent_name
    cdef int bit_start

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


cdef class LogicCircuit:
    cdef public list assignments
    cdef public object architecture
    cdef public str instruction
    cdef public list state_format
    cdef public str _pc_target  # pre-computed: 'RIP'/'EIP'/'PC' or None

    def __init__(self, list assignments, object architecture, str instruction, list state_format):
        self.assignments = assignments
        self.architecture = architecture
        self.instruction = instruction
        self.state_format = state_format
        # Pre-compute which PC register (if any) is a target — checked every evaluate()
        self._pc_target = None
        for _a in assignments:
            if not _a.is_mem_target and _a.target.name in ('RIP', 'EIP', 'PC'):
                self._pc_target = _a.target.name
                break

    def __str__(self):
        return '\n'.join(str(a) for a in self.assignments)

    def __repr__(self):
        return f"LogicCircuit(instr={self.instruction}, assignments_count={len(self.assignments)})"

    cpdef dict evaluate(self, EvalContext context):
        cdef dict output_taint = context.input_taint.copy()
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
            # <object>1 prevents 32-bit overflow crashes when bit-shifting in Cython
            mask = ((<object>1 << (bit_end - bit_start + 1)) - 1) << bit_start
            val = (val << bit_start) & mask

            current = output_taint.get(target_name, 0)
            output_taint[target_name] = (current & ~mask) | val

        # --- THE IMPLICIT TAINT INTERCEPTOR ---
        # _pc_target is pre-computed at circuit build time — no per-call iteration.
        cdef str pc_reg = None
        if self._pc_target is not None and output_taint.get(self._pc_target, 0) != 0:
            pc_reg = self._pc_target

        if pc_reg is not None:
            from microtaint.types import ImplicitTaintPolicy, ImplicitTaintError
            
            if context.implicit_policy == ImplicitTaintPolicy.WARN:
                print(
                    f"[Microtaint] Implicit Taint Detected! "
                    f"Control flow ({pc_reg}) depends on tainted data at instruction: {self.instruction}"
                )
            
            elif context.implicit_policy == ImplicitTaintPolicy.STOP:
                raise ImplicitTaintError(
                    f"\n[!] FATAL: Implicit Taint Detected\n"
                    f"    Instruction (Hex): {self.instruction}\n"
                    f"    Tainted Register : {pc_reg}\n"
                    f"    Taint Mask       : {hex(output_taint[pc_reg])}\n"
                    f"    Reason: The execution of this branch is governed by a tainted condition."
                )
            
            # SAFETY NET: Always drop the PC taint before returning to the user!
            if context.implicit_policy != ImplicitTaintPolicy.KEEP:
                del output_taint[pc_reg]

        return output_taint


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
        assert context.simulator is not None, 'Simulator instance required'
        
        cdef dict evaluated_inputs = {}
        cdef str name
        cdef Expr expr
        
        for name, expr in self.inputs.items():
            evaluated_inputs[name] = expr.evaluate(context)

        m_state = _build_machine_state(evaluated_inputs, context)
        return context.simulator.evaluate_concrete(self, m_state)