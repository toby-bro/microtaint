
from microtaint.instrumentation.ast import InstructionCellExpr
from microtaint.types import Architecture

class PCodeFallbackNeeded(Exception):
    """Raised when the native evaluator encounters an unsupported opcode."""

# Space and Opcode mapping dicts accessible from Python
_SPACE_IDS: dict[str, int]
_OPCODE_ID: dict[str, int]

def _predecode_ops(arch: Architecture, bytestring: bytes) -> tuple[list[tuple[int, ...]], bool]:
    """
    Convert translation ops to compact int tuples.
    Returns (decoded_ops, has_fallback).
    """

def _get_decoded(arch: Architecture, bytestring: bytes) -> tuple[list[tuple[int, ...]], bool]:
    """
    Cached version of _predecode_ops.
    """

class PCodeCellEvaluator:
    """
    Native P-code differential evaluator (Cython, pre-decoded hot path).
    """

    # Publicly accessible C-typed fields
    native_calls: int
    fallback_calls: int
    _offsets: dict[str, int]
    _sizes: dict[str, int]

    def __init__(self, arch: Architecture) -> None: ...
    def evaluate_concrete(self, cell: InstructionCellExpr, flat_inputs: dict[str, int]) -> int:
        """
        Evaluate a single concrete instruction.
        'cell' is expected to have attributes: instruction (hex), out_reg, out_bit_start, out_bit_end.
        """

    def evaluate_differential(
        self,
        cell: InstructionCellExpr,
        or_inputs: dict[str, int],
        and_inputs: dict[str, int],
    ) -> int:
        """
        Evaluate the differential (XOR) of an instruction executed on two input states.
        """

    @property
    def fallback_rate(self) -> float:
        """Percentage of calls that required Unicorn fallback."""

    def stats(self) -> dict[str, int | float]:
        """Return performance statistics."""

    def evaluate_concrete_state(self, cell: InstructionCellExpr, regs: dict[str, int], mem: dict[int, int]) -> int: ...
