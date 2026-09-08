/* The memory callbacks, without the GIL.
 *
 * Unicorn calls UC_HOOK_MEM_READ and UC_HOOK_MEM_WRITE once per guest memory
 * access, and both callbacks were declared `with gil`, so every load and store
 * the program executed acquired and released the GIL.  Measured against a
 * pure-C Unicorn harness, being hooked at all costs about 18 ns per event; the
 * acquire costs several times that, for callbacks whose usual answer is "this
 * address is not poisoned" and "clear the shadow bytes this write covered".
 *
 * Both of those answers are arithmetic over the shadow's page table, which is a
 * plain C open-addressing map with `nogil` accessors.  What kept them behind
 * the GIL was the surrounding bookkeeping -- reaching the hook object, reading
 * a Python set, reporting -- so this header carries the cases that need none of
 * it and hands the rest back.
 *
 * Every function here returns non-zero to mean "I did not handle it", and none
 * of them commits anything in that case, so the GIL path repeats the decision
 * from an unchanged state.
 */
#ifndef MICROTAINT_MEMHOOK_H
#define MICROTAINT_MEMHOOK_H

#include <Python.h>
#include <stdint.h>

#include "../instrumentation/cell_c/circuit_c_api.h"   /* MtMemWrite */

/* Shadow entry points that hold no GIL.  Mirrors the tail of _ShadowCAPI in
 * shadow.pyx; see the note there on why the cast inside them is unchecked. */
typedef void (*MtShadowClearFn)(void *shadow, uint64_t address, int size);
typedef int  (*MtShadowPoisonedFn)(void *shadow, uint64_t address, int size);

/* What the memory callbacks need, resolved once.  The callback's user_data is
 * this struct rather than the hook object, because reaching it must not cost a
 * PyObject cast -- that is the cost being removed. */
typedef struct {
    void               *shadow;        /* BitPreciseShadowMemory, borrowed */
    MtShadowClearFn     clear;
    MtShadowPoisonedFn  is_poisoned;
    PyObject           *owner;         /* the hook, for the GIL path */
    int                *check_uaf;
    /* The instruction hook's claims for the instruction now executing, and the
     * bounds of the code it has decoded.  All borrowed C storage. */
    const MtMemWrite   *ltw;
    const int          *ltw_n;
    PyObject          **lw_set;        /* the Python set the dict path uses */
    const uint64_t     *code_lo;
    const uint64_t     *code_hi;
} MtMemHookUD;

/* UC_HOOK_MEM_READ: report a read from freed memory.  Returns 0 when there is
 * nothing to report, non-zero when the caller must take the GIL and do so. */
static inline int mt_mem_read_nogil(MtMemHookUD *u, uint64_t address, int size) {
    if (!u || !u->check_uaf || !*u->check_uaf) return 0;
    if (!u->is_poisoned || !u->shadow) return 1;
    return u->is_poisoned(u->shadow, address, size) ? 1 : 0;
}

/* UC_HOOK_MEM_WRITE: forget stale taint.  Any byte this write covers that the
 * instruction did not claim as tainted loses its shadow taint -- this is how
 * the engine forgets taint when the program overwrites it with a clean value.
 *
 * Three things send it to the GIL path, and all three are rare:
 *   - the write lands on code the instruction hook has decoded, which has to
 *     invalidate caches held as Python objects;
 *   - UAF checking is on and the target is poisoned, which is a report;
 *   - the dict path's Python set of claims is non-empty, which cannot be
 *     consulted from here.
 */
static inline int mt_mem_write_nogil(MtMemHookUD *u, uint64_t address, int size) {
    if (!u || !u->clear || !u->shadow) return 1;
    /* Self-modifying or JIT'd code.  Two integer compares reject the ordinary
     * data write, which is what almost every write is. */
    if (u->code_lo && u->code_hi && *u->code_hi > *u->code_lo
            && address < *u->code_hi
            && address + (uint64_t)size > *u->code_lo)
        return 1;
    if (u->check_uaf && *u->check_uaf) {
        if (!u->is_poisoned) return 1;
        if (u->is_poisoned(u->shadow, address, size)) return 1;
    }
    if (u->lw_set && *u->lw_set && PySet_GET_SIZE(*u->lw_set) != 0) return 1;

    const int n = u->ltw_n ? *u->ltw_n : 0;
    if (n == 0) {                       /* nothing claimed: clear the lot */
        u->clear(u->shadow, address, size);
        return 0;
    }
    for (int i = 0; i < size; i++) {
        const uint64_t a = address + (uint64_t)i;
        int keep = 0;
        for (int k = 0; k < n; k++) {
            /* unsigned: an address below the range wraps to something large */
            const uint64_t off = a - u->ltw[k].addr;
            if (off < (uint64_t)u->ltw[k].size) {
                /* off >= 8 means the write is wider than its 64-bit mask can
                 * describe.  Keeping the byte over-taints; shifting past the
                 * mask width would drop taint. */
                if (off >= 8 || ((u->ltw[k].taint >> (off * 8)) & 0xFF)) {
                    keep = 1;
                    break;
                }
            }
        }
        if (!keep) u->clear(u->shadow, a, 1);
    }
    return 0;
}

#endif /* MICROTAINT_MEMHOOK_H */
