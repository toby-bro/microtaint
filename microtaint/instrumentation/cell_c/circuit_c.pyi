"""Type stubs for the circuit_c module — hand-written C extension that
compiles LogicCircuit ASTs to a compact bytecode and evaluates them
without traversing the Python AST on every instruction.

When cell_c's CellCAPI capsule is available (the common case),
OP_CALL_CELL bytecode dispatches directly to cell_c's C entry point,
bypassing all Python boundaries inside the hot loop.
"""

from typing import Any

from microtaint.instrumentation.ast import EvalContext, LogicCircuit

def cell_capi_loaded() -> bool:
    """True if the cell_c fast-path CAPI is loaded, so OP_CALL_CELL reaches
    `cell_eval_fast` instead of calling back into Python."""

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

    # Read-only struct members exposed via tp_members.
    has_mem_ops: int
    """
    1 if any assignment in this circuit reads or writes guest memory.
    The wrapper consults this to decide whether the per-instruction
    Tier-3 taint cache can safely store the output (memory-touching
    circuits would need shadow-memory state in the cache key).
    """

    value_independent: int
    """
    1 if the taint output is a pure function of the input taints (the circuit
    reads no operand value), so the instruction cache may key on the taint
    signature alone.
    """

    n_assignments: int
    """
    Number of compiled assignments.  A fast attribute so LogicCircuit.evaluate
    can detect a mutated assignment list per evaluate without allocating a
    stats() dict on the hot path.
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

    def evaluate_c_arr(
        self,
        taint_list: list[int],
        val_list: list[int],
        pcode: Any,
        name_to_slot: dict[str, int],
    ) -> Any:
        """Array-gather register taint eval over slot-indexed lists; None when
        the circuit is not c_evaluable."""

    def evaluate_c_arr_ptr(
        self,
        taint_addr: int,
        val_addr: int,
        n_slots: int,
        pcode: Any,
        name_to_slot: dict[str, int],
    ) -> Any:
        """Array-gather register eval over raw uint64 C arrays, given by
        ADDRESS.  Writes the result directly and atomically."""

    def evaluate_c_mem_ptr(
        self,
        taint_addr: int,
        val_addr: int,
        n_slots: int,
        *args: Any,
    ) -> Any:
        """Memory eval over raw uint64 C arrays; atomic.  Returns the memory
        writes as (addr, size, taint)."""

    def evaluate_c(
        self,
        input_taint: dict[str, int],
        input_values: dict[str, int],
        pcode: Any,
    ) -> dict[str, int] | None:
        """
        C-array taint eval for register-only, non-PC circuits.  Bit-identical to
        evaluate(); returns None (caller falls back to evaluate) when the circuit
        is not c_evaluable (memory ops / python fallback / >64-bit constants) or
        any assignment bails.
        """

    def evaluate_c_mem(
        self,
        input_taint: dict[str, int],
        input_values: dict[str, int],
        pcode: Any,
        shadow_memory: Any,
        mem_reader: Any,
    ) -> dict[str, int] | None:
        """
        C-array taint eval for memory circuits (loads/stores/mem-ALU): register
        taints/values come from uint64 arrays, shadow taint is read/written at the
        C level (the shadow C-API capsule), and concrete memory values go through
        `mem_reader` (valid because the caller holds the GIL).  The output dict is
        byte-identical to evaluate()'s (same pass-through + MEM_<hex>_<size> keys).
        Returns None to fall back to evaluate() for PC-writing circuits, >64-bit
        SIMD targets, python fallback, a missing shadow capsule, or any bail.
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


def supported_expr_types() -> list[str]:
    """Expr class names the bytecode compiler emits without Python fallback.

    Kept in sync with compile_expr in circuit_c.c; the expr-coverage guard test
    compares this against every Expr subclass so a newly added Expr the C path
    cannot compile fails CI.
    """
