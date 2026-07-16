from enum import Enum
from typing import Any, Callable

from microtaint.instrumentation.cell_c.circuit_c import CompiledCircuit
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

class SignedOverflowTaintExpr(Expr):
    """EXACT taint of signed overflow (INT_SBORROW / INT_SCARRY) via the sign
    decomposition OF = (a_s ^ b_s) & (b_s ^ Bor)  [sub] / ~(a_s ^ b_s) & (b_s ^ Car)
    [add], where Bor/Car is the monotone borrow/carry into the msb.

    Proved in benchmark/soundness/prove_signed_overflow.py.
    """

    a_val: Expr
    a_taint: Expr
    b_val: Expr
    b_taint: Expr
    width: int
    is_sub: bool

    def __init__(
        self,
        a_val: Expr,
        a_taint: Expr,
        b_val: Expr,
        b_taint: Expr,
        width: int,
        is_sub: bool,
    ) -> None: ...
    def evaluate(self, context: EvalContext) -> int: ...

class VariableBitSelectTaintExpr(Expr):
    """EXACT taint of a bit selected by a data-dependent index (`bt r,r` -> CF).

    Enumerates the reachable index set I (indices agreeing with the index operand on
    every untainted index bit); CF is tainted iff some i in I selects a tainted source
    bit, or two reachable indices select clean source bits with different values.

    Proved in benchmark/soundness/prove_variable_bit_select.py.
    """

    src_val: Expr
    src_taint: Expr
    idx_val: Expr
    idx_taint: Expr
    width: int

    def __init__(
        self,
        src_val: Expr,
        src_taint: Expr,
        idx_val: Expr,
        idx_taint: Expr,
        width: int,
    ) -> None: ...
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
    _compiled: CompiledCircuit | None | bool

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
    neg_inputs: list[str]

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
        neg_inputs: list[str] | None = ...,
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
