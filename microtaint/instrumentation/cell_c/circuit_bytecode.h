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
    OP_SHR,

    /* Unary */
    OP_NOT,

    /* Avalanche: args size_bits.  Pop 1; if non-zero push (1<<size_bits)-1 else 0 */
    OP_AVALANCHE,         /* args: size_bits  (1..64; >64 not compiled) */

    /* Full-mask avalanche (FullMaskAvalancheExpr): args const_idx(full_mask).
     * Pop dep taint v; push 1 iff v == full_mask && v != 0, else 0. The 1-bit
     * flag soundness floor: fires only when the dep is FULLY tainted. */
    OP_FULLMASK_AVAL,     /* args: const_idx (the full_mask value) */

    /* Equality-bit taint (EqualityTaintExpr): args width. Stack (bottom->top)
     * a_val, a_taint, b_val, b_taint. Push 1 iff equality can BOTH hold and
     * break over the taint cube: ((a^b)&~(ta|tb)&mask)==0 AND (ta|tb)!=0. */
    OP_EQ_TAINT,          /* args: width (1..64) */

    /* Data-dependent bit-select taint (VariableBitSelectTaintExpr, e.g. bt->CF):
     * args width. Stack (bottom->top) src_val, src_taint, idx_val, idx_taint.
     * Push 1 iff the bit selected by the (possibly tainted) index can vary. */
    OP_VAR_BIT_SELECT,    /* args: width (1..64) */

    /* Comparison-bit taint (ComparisonTaintExpr [a<b]/[a<=b], signed/unsigned):
     * args width, flags (bit0=signed, bit1=or_equal). Stack (bottom->top)
     * a_val, a_taint, b_val, b_taint. Push can_be_true XOR always_true over the
     * taint cube: cross corners min(a) OP max(b) vs max(a) OP min(b). */
    OP_CMP_TAINT,         /* args: width, flags */

    /* Signed-overflow taint (SignedOverflowTaintExpr, SCARRY/SBORROW -> OF):
     * args width, flags (bit0=is_sub, bit1=has_carry_in). Stack (bottom->top)
     * a_val, a_taint, b_val, b_taint[, c_val, c_taint]. Sign-decomposition:
     * push non-constancy of OF(a_s, b_s, carry-into-msb) over the taint cube. */
    OP_SIGNED_OVF,        /* args: width, flags */

    /* Multiply taint (VariableMultiplyTaintExpr): args in_width, flags(bit0=signed),
     * out_lo, out_hi. Stack (bottom->top) a_val, a_taint, b_val, b_taint. Sound
     * fill ones[L..H] of the 2w-bit product (128-bit internally), windowed to
     * [out_lo,out_hi). L=tz_lo(a)+tz_lo(b); H=highbit(max^min) (or 2w-1 if a sign
     * bit is tainted). */
    OP_VAR_MUL_TAINT,     /* args: in_width, flags, out_lo, out_hi */

    /* Data-dependent shift taint (VariableShiftTaintExpr): args width, kind
     * (0=left,1=lsr,2=asr), const_idx(amt_mask). Stack (bottom->top) src_val,
     * src_taint, amt_val, amt_taint. Log-fold subcube smear: exact taint of a
     * shift by a tainted amount. */
    OP_VAR_SHIFT,         /* args: width, kind, const_idx */

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
