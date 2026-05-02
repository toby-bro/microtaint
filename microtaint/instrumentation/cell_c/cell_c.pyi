"""Type stubs for the cell_c module — hand-written C extension exposing
the pure-C P-code differential evaluator.

This module is a drop-in replacement for
microtaint.instrumentation.cell.PCodeCellEvaluator.  It bypasses the
Cython-Python boundary and runs the P-code interpreter entirely in C.

The module also exports a `_cell_capi` PyCapsule that other native
extensions (notably circuit_c) import to call into the cell evaluator
without going through the Python C-API at all.
"""

from typing import Any

from microtaint.instrumentation.ast import InstructionCellExpr
from microtaint.types import Architecture

# CellCAPI PyCapsule. Imported by circuit_c at module init via
# PyCapsule_Import("cell_c._cell_capi"). Not intended for Python use;
# typed as Any because PyCapsule has no useful Python interface.
_cell_capi: Any

class PCodeCellEvaluatorC:
    """
    Native (pure-C) P-code differential evaluator.

    API-compatible with PCodeCellEvaluator (the Cython implementation),
    but with all hot paths written in C and the per-cell cache layout
    optimized for Tier-1 CellCAPI fast-call by circuit_c.
    """

    # Public read-only counters (exposed via tp_getset).
    native_calls: int
    fallback_calls: int
    fallback_rate: float
    _offsets: dict[str, int]
    _sizes: dict[str, int]

    def __init__(self, arch: Architecture) -> None: ...
    def evaluate_concrete(
        self,
        cell: InstructionCellExpr,
        flat_inputs: dict[str, int],
    ) -> int:
        """
        Evaluate a single concrete instruction.

        `cell` must have `.instruction` (hex string), `.out_reg`,
        `.out_bit_start`, `.out_bit_end`. `flat_inputs` maps register
        names to integer values.
        """

    def evaluate_concrete_state(
        self,
        cell: InstructionCellExpr,
        regs: dict[str, int],
        mem: dict[int, int],
    ) -> int:
        """Concrete evaluation with explicit (regs, mem) state."""

    def evaluate_concrete_flat(
        self,
        cell: InstructionCellExpr,
        flat_inputs: dict[str, int],
    ) -> int:
        """Flat-input variant of evaluate_concrete (no register splitting)."""

    def evaluate_differential(
        self,
        cell: InstructionCellExpr,
        or_inputs: dict[str, int],
        and_inputs: dict[str, int],
    ) -> int:
        """
        Evaluate the differential (XOR) of an instruction executed on two
        input states. Returns the bit-mask of taint that propagates from
        inputs to the cell's output register slice.
        """

    def make_cell_handle(self, cell: InstructionCellExpr) -> int:
        """
        Pre-resolve a CellHandle for `cell` and return an opaque integer
        token that circuit_c can pass back via OP_CALL_CELL to skip the
        Python boundary entirely. The handle is owned by this evaluator
        and stays valid for its lifetime.
        """

    def stats(self) -> dict[str, int | float]:
        """Return performance counters: native_calls, fallback_calls, fallback_rate."""
