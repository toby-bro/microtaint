from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from microtaint.simulator import CellSimulator, MachineState
from microtaint.types import Architecture, Register


def _build_machine_state(input_dict: dict[str, int], context: EvalContext) -> MachineState:
    regs: dict[str, int] = {}
    mem: dict[int, int] = {}
    for name, val in input_dict.items():
        if name.startswith('MEM_'):
            # Robust lookup: handle names like MEM_X0 or MEM_0x1000
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
    # Binary operations
    AND = 'AND'
    OR = 'OR'
    XOR = 'XOR'
    # Unary operations
    NOT = 'NOT'
    LEFT = 'LEFT'


@dataclass
class EvalContext:
    input_taint: dict[str, int]
    input_values: dict[str, int]
    simulator: CellSimulator | None = None


@dataclass
class Expr:
    """Base class for AST expressions."""

    def evaluate(self, context: EvalContext) -> int:
        raise NotImplementedError('Must implement evaluate in subclasses')


@dataclass
class AvalancheExpr(Expr):
    """Evaluates to a full bitmask of 1s based on size if the inner expression is non-zero, else 0."""

    expr: Expr
    size_bits: int

    def __str__(self) -> str:
        return f'AVALANCHE({self.expr})'

    def evaluate(self, context: EvalContext) -> int:
        val = self.expr.evaluate(context)
        if val != 0:
            return (1 << self.size_bits) - 1
        return 0


@dataclass
class TaintOperand(Expr):
    """Represents a taint value operand (e.g. the taint status of EAX) or a concrete value operand."""

    name: str
    bit_start: int
    bit_end: int
    is_taint: bool = True  # True if T_x, False if V_x

    def __str__(self) -> str:
        prefix = 'T' if self.is_taint else 'V'
        if self.bit_start == self.bit_end:
            return f'{prefix}_{self.name}[{self.bit_start}]'
        return f'{prefix}_{self.name}[{self.bit_end}:{self.bit_start}]'

    def evaluate(self, context: EvalContext) -> int:
        state = context.input_taint if self.is_taint else context.input_values
        val = state.get(self.name, 0)
        # Extract the bit slice
        mask = (1 << (self.bit_end - self.bit_start + 1)) - 1
        return (val >> self.bit_start) & mask


@dataclass
class MemoryOperand(Expr):
    """Represents a memory operand (dynamically resolved at concrete execution time)."""

    address_expr: Expr
    size: int  # size in bytes
    is_taint: bool = True

    def __str__(self) -> str:
        prefix = 'T' if self.is_taint else 'V'
        return f'{prefix}_MEM[{self.address_expr}, size={self.size}]'

    def evaluate(self, context: EvalContext) -> int:
        address = self.address_expr.evaluate(context)
        mem_name = f'MEM_{hex(address)}_{self.size}'
        state = context.input_taint if self.is_taint else context.input_values
        return state.get(mem_name, 0)


@dataclass
class Constant(Expr):
    """A constant boolean or integer value."""

    value: int
    size: int  # size in bits

    def __str__(self) -> str:
        return hex(self.value)

    def evaluate(self, context: EvalContext) -> int:  # noqa: ARG002
        return self.value


@dataclass
class UnaryExpr(Expr):
    """A unary operator application."""

    op: Op
    expr: Expr

    def __str__(self) -> str:
        return f'{self.op.value}({self.expr})'

    def evaluate(self, context: EvalContext) -> int:
        val = self.expr.evaluate(context)
        if self.op == Op.NOT:
            return ~val
        raise NotImplementedError(f'Unsupported unary op {self.op}')


@dataclass
class BinaryExpr(Expr):
    """A binary operator application."""

    op: Op
    lhs: Expr
    rhs: Expr

    def __str__(self) -> str:
        return f'({self.lhs} {self.op.value} {self.rhs})'

    def evaluate(self, context: EvalContext) -> int:
        left = self.lhs.evaluate(context)
        right = self.rhs.evaluate(context)
        if self.op == Op.AND:
            return left & right
        if self.op == Op.OR:
            return left | right
        if self.op == Op.XOR:
            return left ^ right
        if self.op == Op.LEFT:
            return left << right
        raise NotImplementedError(f'Unsupported binary op {self.op}')


@dataclass
class TaintAssignment:
    """Represents assigning a logic block of taint variables to a target"""

    target: TaintOperand | MemoryOperand
    dependencies: list[Expr]
    expression: Expr | None = None
    expression_str: str = ''

    def __str__(self) -> str:
        if self.expression:
            expr = str(self.expression)
        elif self.expression_str:
            expr = self.expression_str
        else:
            expr = ' | '.join(str(d) for d in self.dependencies)
        return f'{self.target} = {expr}'


@dataclass
class LogicCircuit:
    """Represents the final circuit computing the taint output for an instruction."""

    assignments: list[TaintAssignment]
    architecture: Architecture
    instruction: str
    state_format: list[Register]

    def __str__(self) -> str:
        return '\n'.join(str(a) for a in self.assignments)

    def evaluate(self, context: EvalContext) -> dict[str, int]:
        output_taint: dict[str, int] = context.input_taint.copy()
        for assignment in self.assignments:
            if assignment.expression is not None:
                val = assignment.expression.evaluate(context)
            elif assignment.expression_str:
                raise NotImplementedError('Arbitrary string expressions not supported for evaluation right now.')
            else:
                # For purely mapped operations, it's just an OR of dependencies
                val = 0
                for dep in assignment.dependencies:
                    val |= dep.evaluate(context)

            # Resolve targets securely
            if isinstance(assignment.target, MemoryOperand):
                address = assignment.target.address_expr.evaluate(context)
                target_name = f'MEM_{hex(address)}_{assignment.target.size}'
                bit_start = 0
                bit_end = assignment.target.size * 8 - 1
            else:
                target_name = assignment.target.name
                bit_start = assignment.target.bit_start
                bit_end = assignment.target.bit_end

            # Apply taint slice correctly to overwrite ONLY the targeted bits
            mask = ((1 << (bit_end - bit_start + 1)) - 1) << bit_start
            val = (val << bit_start) & mask

            current = output_taint.get(target_name, 0)
            output_taint[target_name] = (current & ~mask) | val

        return output_taint


@dataclass
class InstructionCellExpr(Expr):
    """Represents evaluating the instruction itself as a logic cell."""

    architecture: Architecture
    instruction: str
    out_reg: str
    out_bit_start: int
    out_bit_end: int
    inputs: dict[str, Expr]

    def __str__(self) -> str:
        args = ', '.join(f'{k}={v}' for k, v in self.inputs.items())
        return f'SimulateCell(instr=0x{self.instruction}, out={self.out_reg}[{self.out_bit_end}:{self.out_bit_start}], {args})'  # noqa: E501

    def evaluate(self, context: EvalContext) -> int:
        assert context.simulator is not None, 'Simulator instance required in context to evaluate instruction cell'

        # 1. Evaluate the exact AST inputs provided to this cell
        # (e.g. C1 evaluates inputs to V | T, while C2 evaluates inputs to V & ~T)
        evaluated_inputs: dict[str, int] = {}
        for name, expr in self.inputs.items():
            evaluated_inputs[name] = expr.evaluate(context)

        # 2. Build the state
        m_state = _build_machine_state(evaluated_inputs, context)

        # 3. Evaluate the instruction concretely ONCE
        # (The simulator's evaluate_concrete already handles the bit slicing via out_bit_start/end)
        return context.simulator.evaluate_concrete(self, m_state)
