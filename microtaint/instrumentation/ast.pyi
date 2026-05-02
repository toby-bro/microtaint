from enum import Enum
from typing import Any, Callable

from microtaint.simulator import CellSimulator, MachineState
from microtaint.types import Architecture, ImplicitTaintPolicy, Register

def _build_machine_state(input_dict: dict[str, int], context: EvalContext) -> MachineState: ...

class Op(str, Enum):
    AND = 'AND'
    OR = 'OR'
    XOR = 'XOR'
    NOT = 'NOT'
    LEFT = 'LEFT'
    ADD = 'ADD'  # Only for memory offset calculations, not for taint logic
    SUB = 'SUB'  # Only for memory offset calculations, not for taint logic

class EvalContext:
    input_taint: dict[str, int]
    input_values: dict[str, int]
    simulator: CellSimulator | None
    implicit_policy: ImplicitTaintPolicy
    shadow_memory: Any | None
    mem_reader: Callable[[int, int], int] | None

    def __init__(
        self,
        input_taint: dict[str, int],
        input_values: dict[str, int],
        simulator: CellSimulator | None = ...,
        implicit_policy: ImplicitTaintPolicy = ...,
        shadow_memory: Any | None = ...,
        mem_reader: Callable[[int, int], int] | None = ...,
    ) -> None: ...

class Expr:
    def evaluate(self, context: EvalContext) -> int: ...

class AvalancheExpr(Expr):
    expr: Expr
    size_bits: int

    def __init__(self, expr: Expr, size_bits: int) -> None: ...
    def evaluate(self, context: EvalContext) -> int: ...

class FullMaskAvalancheExpr(Expr):
    """Fires to 1 only when dep taint value equals the full mask for dep_bits."""

    dep: Expr
    full_mask: int

    def __init__(self, dep: Expr, dep_bits: int) -> None: ...
    def evaluate(self, context: EvalContext) -> int: ...

class TaintOperand(Expr):
    name: str
    bit_start: int
    bit_end: int
    is_taint: bool

    def __init__(self, name: str, bit_start: int, bit_end: int, is_taint: bool = ...) -> None: ...
    def evaluate(self, context: EvalContext) -> int: ...

class MemoryOperand(Expr):
    address_expr: Expr
    size: int
    is_taint: bool

    def __init__(self, address_expr: Expr, size: int, is_taint: bool = ...) -> None: ...
    def evaluate(self, context: EvalContext) -> int: ...

class Constant(Expr):
    value: int
    size: int

    def __init__(self, value: int, size: int) -> None: ...
    def evaluate(self, context: EvalContext) -> int: ...

class UnaryExpr(Expr):
    op: Op
    expr: Expr

    def __init__(self, op: Op, expr: Expr) -> None: ...
    def evaluate(self, context: EvalContext) -> int: ...

class BinaryExpr(Expr):
    op: Op
    lhs: Expr
    rhs: Expr

    def __init__(self, op: Op, lhs: Expr, rhs: Expr) -> None: ...
    def evaluate(self, context: EvalContext) -> int: ...

class TaintAssignment:
    target: TaintOperand | MemoryOperand
    dependencies: list[Expr]
    expression: Expr | None
    expression_str: str

    def __init__(
        self,
        target: TaintOperand | MemoryOperand,
        dependencies: list[Expr],
        expression: Expr | None = ...,
        expression_str: str = ...,
    ) -> None: ...

class LogicCircuit:
    assignments: list[TaintAssignment]
    architecture: Architecture
    instruction: str
    state_format: list[Register]

    def __init__(
        self,
        assignments: list[TaintAssignment],
        architecture: Architecture,
        instruction: str,
        state_format: list[Register],
    ) -> None: ...
    def evaluate(self, context: EvalContext) -> dict[str, int]: ...

class ChainedCircuit(LogicCircuit):
    sub_circuits: list[LogicCircuit]
    architecture: Architecture
    instruction: str
    state_format: list[Register]
    assignments: list[TaintAssignment]  # flattened view across all sub-circuits

    def __init__(
        self,
        sub_circuits: list[LogicCircuit],
        architecture: Architecture,
        instruction: str,
        state_format: list[Register],
    ) -> None: ...
    def evaluate(self, context: EvalContext) -> dict[str, int]: ...

class InstructionCellExpr(Expr):
    architecture: Architecture
    instruction: str
    out_reg: str
    out_bit_start: int
    out_bit_end: int
    inputs: dict[str, Expr]

    def __init__(
        self,
        architecture: Architecture,
        instruction: str,
        out_reg: str,
        out_bit_start: int,
        out_bit_end: int,
        inputs: dict[str, Expr],
    ) -> None: ...
    def evaluate(self, context: EvalContext) -> int: ...

class MemoryDifferentialExpr(Expr):
    bytestring: bytes
    target: tuple[Any, ...]
    reg_inputs: list[tuple[str, int, int]]
    mem_inputs: list[tuple[str, int, int]]
    addr_only_regs: list[str]

    _instr_hex: str
    _target_out_reg: str
    _target_bit_start: int
    _target_bit_end: int

    def __init__(
        self,
        bytestring: bytes,
        target: tuple[str, str, int, int],
        reg_inputs: list[tuple[str, int, int]],
        mem_inputs: list[tuple[str, int, int]],
        addr_only_regs: list[str],
    ) -> None: ...
    def evaluate(self, context: EvalContext) -> int: ...
    @property
    def instruction(self) -> str: ...
    @property
    def out_reg(self) -> str: ...
    @property
    def out_bit_start(self) -> int: ...
    @property
    def out_bit_end(self) -> int: ...
