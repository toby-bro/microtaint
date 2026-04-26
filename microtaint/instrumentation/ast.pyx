# cython: language_level=3
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
                addr = int(ptr_part, 16)
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


cdef class EvalContext:
    cdef public dict input_taint
    cdef public dict input_values
    cdef public object simulator

    def __init__(self, dict input_taint, dict input_values, object simulator=None):
        self.input_taint = input_taint
        self.input_values = input_values
        self.simulator = simulator


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

    cpdef object evaluate(self, EvalContext context):
        cdef dict state = context.input_taint if self.is_taint else context.input_values
        cdef object val = state.get(self.name, 0)
        # FIX: <object>1 prevents C Undefined Behavior
        cdef object mask = (<object>1 << (self.bit_end - self.bit_start + 1)) - 1
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

    cpdef object evaluate(self, EvalContext context):
        cdef object address = self.address_expr.evaluate(context)
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
        return hex(self.value)

    cpdef object evaluate(self, EvalContext context):
        return self.value


cdef class UnaryExpr(Expr):
    cdef public object op
    cdef public Expr expr

    def __init__(self, object op, Expr expr):
        self.op = op
        self.expr = expr

    def __str__(self):
        return f'{self.op.value}({self.expr})'

    cpdef object evaluate(self, EvalContext context):
        cdef object val = self.expr.evaluate(context)
        if self.op == Op.NOT:
            return ~val
        raise NotImplementedError(f'Unsupported unary op {self.op}')


cdef class BinaryExpr(Expr):
    cdef public object op
    cdef public Expr lhs
    cdef public Expr rhs

    def __init__(self, object op, Expr lhs, Expr rhs):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return f'({self.lhs} {self.op.value} {self.rhs})'

    cpdef object evaluate(self, EvalContext context):
        cdef object left = self.lhs.evaluate(context)
        cdef object right = self.rhs.evaluate(context)
        
        if self.op == Op.AND:
            return left & right
        if self.op == Op.OR:
            return left | right
        if self.op == Op.XOR:
            return left ^ right
        if self.op == Op.LEFT:
            return left << right
        raise NotImplementedError(f'Unsupported binary op {self.op}')


cdef class TaintAssignment:
    cdef public object target
    cdef public list dependencies
    cdef public Expr expression
    cdef public str expression_str

    def __init__(self, object target, list dependencies, Expr expression=None, str expression_str=''):
        self.target = target
        self.dependencies = dependencies
        self.expression = expression
        self.expression_str = expression_str

    def __str__(self):
        cdef str expr_str
        if self.expression is not None:
            expr_str = str(self.expression)
        elif self.expression_str:
            expr_str = self.expression_str
        else:
            expr_str = ' | '.join(str(d) for d in self.dependencies)
        return f'{self.target} = {expr_str}'


cdef class LogicCircuit:
    cdef public list assignments
    cdef public object architecture
    cdef public str instruction
    cdef public list state_format

    def __init__(self, list assignments, object architecture, str instruction, list state_format):
        self.assignments = assignments
        self.architecture = architecture
        self.instruction = instruction
        self.state_format = state_format

    def __str__(self):
        return '\n'.join(str(a) for a in self.assignments)

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
            
            if assignment.expression is not None:
                val = assignment.expression.evaluate(context)
            elif assignment.expression_str:
                raise NotImplementedError('Arbitrary string expressions not supported.')
            else:
                val = 0
                for dep in assignment.dependencies:
                    val |= dep.evaluate(context)

            if isinstance(assignment.target, MemoryOperand):
                address = assignment.target.address_expr.evaluate(context)
                target_name = f'MEM_{hex(address)}_{assignment.target.size}'
                bit_start = 0
                bit_end = assignment.target.size * 8 - 1
            else:
                target_name = assignment.target.name
                bit_start = assignment.target.bit_start
                bit_end = assignment.target.bit_end

            # FIX: <object>1 prevents 32-bit overflow crashes when bit-shifting
            mask = ((<object>1 << (bit_end - bit_start + 1)) - 1) << bit_start
            val = (val << bit_start) & mask

            current = output_taint.get(target_name, 0)
            output_taint[target_name] = (current & ~mask) | val

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

    cpdef object evaluate(self, EvalContext context):
        assert context.simulator is not None, 'Simulator instance required'
        
        cdef dict evaluated_inputs = {}
        cdef str name
        cdef Expr expr
        
        for name, expr in self.inputs.items():
            evaluated_inputs[name] = expr.evaluate(context)

        m_state = _build_machine_state(evaluated_inputs, context)
        return context.simulator.evaluate_concrete(self, m_state)