"""Type stubs for the circuit_c module — hand-written C extension that
compiles LogicCircuit ASTs to a compact bytecode and evaluates them
without traversing the Python AST on every instruction.

When cell_c's CellCAPI capsule is available (the common case),
OP_CALL_CELL bytecode dispatches directly to cell_c's C entry point,
bypassing all Python boundaries inside the hot loop.
"""

from typing import Any

from microtaint.instrumentation.ast import EvalContext, LogicCircuit

class CompiledCircuit:
    """
    A LogicCircuit pre-compiled to compact bytecode.

    Created via `compile_circuit(circuit)`.  Two evaluation entry points
    exist:

      * `evaluate(context)` — the standard path, accepting an
        EvalContext; matches LogicCircuit.evaluate's signature for
        drop-in replacement.

      * `evaluate_fast(input_taint, input_values, pcode, ...)` — bypasses
        the EvalContext attribute-extraction step (saves ~1.3 us/call),
        called directly from the wrapper's hot path when the context's
        fields are already in hand.
    """

    # Read-only struct member exposed via tp_members.
    has_mem_ops: int
    """
    1 if any assignment in this circuit reads or writes guest memory.
    The wrapper consults this to decide whether the per-instruction
    Tier-3 taint cache can safely store the output (memory-touching
    circuits would need shadow-memory state in the cache key).
    """

    def evaluate(self, context: EvalContext) -> dict[str, int]:
        """Evaluate the circuit against `context`, returning the new taint state."""

    def evaluate_fast(
        self,
        input_taint: dict[str, int],
        input_values: dict[str, int],
        pcode: Any,
        implicit_policy: Any = ...,
        shadow_memory: Any = ...,
        mem_reader: Any = ...,
    ) -> dict[str, int]:
        """
        Faster variant of evaluate() that takes context fields directly.

        `pcode` is typically the simulator's `_pcode` cell evaluator;
        it's used by OP_CALL_CELL to dispatch into the C kernel via
        CellCAPI when available.
        """

    def stats(self) -> dict[str, int]:
        """
        Return circuit statistics:
          * n_assignments    — total number of LogicCircuit assignments
          * compiled         — number that compiled to bytecode
          * python_fallback  — number that fell back to Python evaluation
        """


def compile_circuit(
    circuit: LogicCircuit,
    pcode: Any = ...,
) -> CompiledCircuit:
    """
    Compile a LogicCircuit AST to bytecode.

    Optional `pcode` (the simulator's cell evaluator) enables Tier-1
    CellHandle pre-resolution: each OP_CALL_CELL bytecode op is bound
    to a CellHandle at compile time, so dispatch into the C kernel
    skips the Python boundary at run time.

    Returns a CompiledCircuit instance.  Raises if the circuit contains
    unsupported expression forms; the caller is expected to fall back
    to the Python AST walker.
    """
