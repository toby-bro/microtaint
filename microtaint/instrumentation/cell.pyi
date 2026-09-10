from microtaint.emulator.wrapper import RegReadDescriptor
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

class DecodedOps:
    """Pre-decoded p-code for one instruction, as a C struct array.

    Returned by `_get_decoded`; the fields below are the `cdef public` ones,
    which is everything reachable from Python.  The struct array itself is not,
    except through `get_buf_bytes`.
    """

    n_ops: int
    has_fallback: bool
    #: The instruction carries an opaque data op (CALLOTHER with an output, or
    #: FLOAT) whose taint is avalanched rather than executed concretely.
    avalanche_ok: bool
    next_instr_addr: int
    imark_addr: int
    #: A backward BRANCH to `imark_addr` is present (rep stosb / movsb).
    has_loop: bool
    #: IMARK ram address -> the p-code op index of that IMARK, plus a synthetic
    #: `next_instr_addr -> n_ops` entry so a forward skip to the end resolves.
    imark_to_pc: dict[int, int]
    #: The SP_REGISTER input offsets this instruction reads.
    input_reg_offsets: set[int]
    #: The cached uc_reg_read_batch descriptor for this instruction's
    #: inputs; see wrapper.RegReadDescriptor.
    _uc_arrays: RegReadDescriptor | None

    def get_buf_bytes(self) -> bytes: ...

def _get_decoded(arch: Architecture, bytestring: bytes) -> DecodedOps:
    """
    Cached version of _predecode_ops.
    """

class PCodeCellEvaluator:
    """
    Native P-code differential evaluator (Cython, pre-decoded hot path).
    """

    def reset_frame_cache(self, enable: bool) -> None:
        """Arm (or disarm) per-evaluate frame sharing, emptying the cache and
        pool.  Called once per top-level evaluate."""

    def evaluate_concrete_state_shared(self, cell: InstructionCellExpr,
                                       regs: dict[str, int],
                                       mem: dict[int, int]) -> int:
        """Frame-sharing variant of evaluate_concrete_state: run the whole
        instruction once per distinct (instruction, register inputs) and read
        each output's slice off the cached frame."""

    def evaluate_concrete_all(self, instruction: str, flat_inputs: dict[str, int],
                              out_specs: list[tuple[str, int]]) -> dict[str, int]:
        """Execute `instruction` once, then read every requested output off
        that single frame.  `out_specs` is [(name, bits)]."""

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

#: (offset, size) per SLEIGH register name, keyed by upper-case name.  Declared
#: here because the module is Cython and callers outside it -- the emulator's
#: register file, the block compiler -- read it as a normal import.
def _build_reg_maps(
    arch: Architecture | str,
) -> tuple[dict[str, int], dict[str, int]]: ...

