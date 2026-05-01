/*
 * cell_c_api.h — Public C API exported by cell_c, used by circuit_c.
 *
 * Both modules link to this header.  cell_c provides the actual symbols
 * via a PyCapsule named "cell_c._cell_capi"; circuit_c imports it via
 * PyCapsule_Import at module init.
 *
 * Opaque types: the consumer treats EvalC and CellHandle as opaque
 * pointers; only the function-pointer signatures matter.
 */
#ifndef CELL_C_API_H
#define CELL_C_API_H

#include <stdint.h>
#include <Python.h>

/* Forward decls — opaque to circuit_c */
struct _EvalC;
struct _CellHandle;
typedef struct _EvalC      EvalC_API;
typedef struct _CellHandle CellHandle_API;

typedef struct {
    /* Returns 0 on success (out_value populated), 1 on fallback needed
     * (caller should raise PCodeFallbackNeeded), -1 on hard error. */
    int (*cell_eval_fast)(EvalC_API *self, CellHandle_API *h,
                          const uint64_t *input_values, uint64_t *out_value);
    /* Returns the PCodeFallbackNeeded class (borrowed ref). */
    PyObject *(*get_fallback_exc)(EvalC_API *self);
} CellCAPI;

#endif
