#ifndef CIRCUIT_BYTECODE_H
#define CIRCUIT_BYTECODE_H

/*
 * Compiled LogicCircuit bytecode
 * ------------------------------
 *
 * A LogicCircuit's expression trees are compiled to a flat bytecode
 * array.  The evaluator is a stack machine over uint64_t with one
 * dispatch loop, no recursion, no virtual calls.
 *
 * Limits: all stack values are <= 64 bits.  Assignments whose
 * AvalancheExpr.size_bits > 64 are NOT compiled — they remain
 * in the Cython AST path, evaluated by the original LogicCircuit.evaluate.
 *
 * Bytecode is a single uint32 stream.  Variable-length args follow each
 * opcode in subsequent uint32 slots.  Strings (register names) are
 * indexes into a per-circuit `string_pool` array.
 */

#include <stdint.h>

/* ── opcodes ─────────────────────────────────────────────────────── */
typedef enum {
    /* Stack pushes */
    OP_PUSH_TAINT = 1,    /* args: name_idx bit_start bit_end */
    OP_PUSH_VALUE,        /* args: name_idx bit_start bit_end */
    OP_PUSH_CONST,        /* args: const_idx */

    /* Memory operands (taint reads via shadow_memory, value reads via
     * mem_reader). The address is whatever is currently on top of stack. */
    OP_PUSH_MEM_TAINT,    /* args: size_bytes — pops addr, pushes shadow_memory.read_mask(addr, size) */
    OP_PUSH_MEM_VALUE,    /* args: size_bytes — pops addr, pushes mem_reader(addr, size) */

    /* Arithmetic / logical (binary: pops 2, pushes 1) */
    OP_AND, OP_OR, OP_XOR,
    OP_ADD, OP_SUB,
    OP_SHL,

    /* Unary */
    OP_NOT,

    /* Avalanche: args size_bits.  Pop 1; if non-zero push (1<<size_bits)-1 else 0 */
    OP_AVALANCHE,         /* args: size_bits  (1..64; >64 not compiled) */

    /* Call into C kernel.
     *   args: cell_idx n_inputs name_idx_0 name_idx_1 ... name_idx_(n-1)
     * Pops n inputs from stack (last pushed is name_idx_(n-1)),
     * builds a flat dict mapping name_idx -> value, calls
     * pcode.evaluate_concrete_flat(cell, dict).  Pushes the result. */
    OP_CALL_CELL,

    /* End of expression bytecode for one assignment.  Pop top-of-stack
     * as the result.  Then the per-assignment epilogue (mask + dict
     * write) is handled by the C evaluator using the assignment's
     * target descriptor. */
    OP_END,

    /* Memory differential — for MemoryDifferentialExpr leaves.  We do
     * not compile these; the bytecode emitter sets the assignment's
     * "needs_python" flag and falls back. */
    OP_HALT_FALLBACK = 255,
} CircuitOp;

/* Stack depth — max observed in measured circuits is < 8.  Bound at 32. */
#define CIRCUIT_STACK_MAX  32

/* Bytecode capacity per assignment — max observed ~30 ops.  Bound 256. */
#define CIRCUIT_BC_MAX     1024

#endif /* CIRCUIT_BYTECODE_H */
