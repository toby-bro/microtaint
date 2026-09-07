/*
 * circuit_c_api.h — Public C API exported by circuit_c, used by hook_core.
 *
 * circuit_c publishes these function pointers through a PyCapsule named
 * "microtaint.instrumentation.cell_c.circuit_c._circuit_capi"; hook_core imports
 * it once and then calls the array evaluators as plain C functions.
 *
 * Why: calling them as Python methods costs an attribute lookup, a tuple build,
 * PyArg_ParseTuple, and PyLong boxing of the two array addresses on EVERY
 * instruction (perf: object_vacall + vgetargs1_impl + tupledealloc +
 * PyObject_GenericGetAttr + PyLong_FromUnsignedLong).  Through the capsule the
 * arrays are passed as raw uint64_t*, so none of that happens.
 *
 * Two families are exported.  The PyObject-returning ones keep the exact return
 * convention of the Python methods, so callers that already had decline/fallback
 * logic need no change:
 *   - a new reference on success (an int count, or the mem-writes list)
 *   - Py_None  -> the evaluator declined; caller must fall back
 *   - NULL     -> a real error, with the exception set
 * The `_i` ones return a plain int (see MT_EVAL_* below) and write their results
 * into caller-owned C storage, so a steady-state instruction crosses the
 * boundary without allocating, boxing or GC-tracking anything.
 */
#ifndef CIRCUIT_C_API_H
#define CIRCUIT_C_API_H

#include <stdint.h>
#include <Python.h>

/* Optional C-level guest-memory reader.  Returns 0 on success (*out set); any
 * other value means "could not read, use the Python reader for this access".
 * Lets OP_PUSH_MEM_VALUE fetch a value without PyLong-boxing the address, doing
 * a Python call, and boxing the result again. */
typedef int (*mt_mem_read_fn)(void *ctx, uint64_t addr, int size, uint64_t *out);

/* One committed memory write, handed back through a caller-provided C array.
 * The list-of-tuples form allocated a PyList, a tuple and three PyLongs per
 * memory instruction, all of them GC-tracked; this form allocates nothing. */
typedef struct {
    uint64_t addr;
    uint64_t taint;
    int      size;
} MtMemWrite;

/* Result codes for the integer-returning evaluators below.  Kept distinct from
 * a valid count so a decline can never be mistaken for "zero writes". */
#define MT_EVAL_DECLINED (-1)   /* fall back; no exception set */
#define MT_EVAL_ERROR    (-2)   /* exception is set */

/* Bits returned by compiled_flags().  These mirror int fields of CompiledCircuit
 * that the hot path consults on every instruction; reading them through one C
 * call replaces two PyObject_GenericGetAttr lookups per instruction. */
#define MT_CF_C_EVALUABLE     0x01
#define MT_CF_C_MEM_EVALUABLE 0x02
#define MT_CF_HAS_MEM_OPS     0x04
#define MT_CF_VALUE_INDEP     0x08

typedef struct {
    /* `compiled` is a CompiledCircuit (borrowed).  Registers only. */
    PyObject *(*eval_arr_ptr)(PyObject *compiled, uint64_t *taint, uint64_t *val,
                              int n_slots, PyObject *pcode, PyObject *name_to_slot);
    /* Memory circuits: writes register targets into `taint`, memory targets into
     * the shadow, and returns the committed (addr, size, taint) writes. */
    PyObject *(*eval_mem_ptr)(PyObject *compiled, uint64_t *taint, uint64_t *val,
                              int n_slots, PyObject *pcode, PyObject *shadow,
                              PyObject *mem_reader, PyObject *name_to_slot);
    /* As eval_mem_ptr, but installs `mem_fn`/`mem_ctx` as the C reader for the
     * duration of the call (restored afterwards, so nested/other users are
     * unaffected).  `mem_reader` is still required as the fallback for reads the
     * C function declines. */
    PyObject *(*eval_mem_ptr_c)(PyObject *compiled, uint64_t *taint, uint64_t *val,
                                int n_slots, PyObject *pcode, PyObject *shadow,
                                PyObject *mem_reader, PyObject *name_to_slot,
                                mt_mem_read_fn mem_fn, void *mem_ctx);

    /* ---- integer-returning forms: the boundary creates no PyObject ------- */

    /* Registers only.  Returns the number of register targets written, or
     * MT_EVAL_DECLINED / MT_EVAL_ERROR. */
    int (*eval_arr_ptr_i)(PyObject *compiled, uint64_t *taint, uint64_t *val,
                          int n_slots, PyObject *pcode, PyObject *name_to_slot);
    /* Memory circuits, C reader installed as in eval_mem_ptr_c.  Committed
     * writes are stored in `out` (at most `out_cap`); returns how many, or
     * MT_EVAL_DECLINED / MT_EVAL_ERROR.  A circuit with more memory targets
     * than `out_cap` declines rather than truncating, so the caller's fallback
     * still sees a complete result. */
    int (*eval_mem_ptr_ci)(PyObject *compiled, uint64_t *taint, uint64_t *val,
                           int n_slots, PyObject *pcode, PyObject *shadow,
                           PyObject *mem_reader, PyObject *name_to_slot,
                           mt_mem_read_fn mem_fn, void *mem_ctx,
                           MtMemWrite *out, int out_cap);

    /* MT_CF_* bits for a CompiledCircuit, or 0 if `compiled` is not one (which
     * reads as "no capabilities", the safe answer: the caller falls back). */
    int (*compiled_flags)(PyObject *compiled);
} CircuitCAPI;

#endif /* CIRCUIT_C_API_H */
