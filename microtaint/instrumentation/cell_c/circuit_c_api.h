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
 * Return convention is deliberately identical to the Python methods, so the
 * caller's decline/fallback logic is unchanged:
 *   - a new reference on success (an int count, or the mem-writes list)
 *   - Py_None  -> the evaluator declined; caller must fall back
 *   - NULL     -> a real error, with the exception set
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
} CircuitCAPI;

#endif /* CIRCUIT_C_API_H */
