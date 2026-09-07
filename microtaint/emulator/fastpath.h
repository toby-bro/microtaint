/*
 * fastpath.h - the per-instruction taint hot path, in plain C.
 *
 * Why this is a C file and not more Cython
 * ----------------------------------------
 * Cython compiles to C, so a `cdef` function over C types already emits the C
 * you would write by hand.  What it does NOT do is stop you from touching a
 * Python object: every `object` local costs a refcount pair, every attribute
 * read is a PyObject_GenericGetAttr, and each of those is invisible in the
 * source.  Putting the steady state here makes the absence of Python
 * mechanical rather than a claim: this file has no attribute lookups, no
 * allocations and no boxing, and the only PyObject values it handles are
 * borrowed pointers passed straight through to the evaluator, which needs them.
 *
 * It is a header rather than a separate translation unit so it compiles into
 * hook_core's own object file: no build-system change, and the compiler can
 * still inline across the Cython/C boundary.  Everything is `static`.
 *
 * Division of labour with hook_core.pyx:
 *   - here:  the repeat visit.  Cache probe, register read, evaluator call,
 *            taint commit, snapshot store.  Runs for the overwhelming majority
 *            of executed instructions.
 *   - there: everything that legitimately needs Python.  Decoding an address
 *            for the first time, the dict fallback when an evaluator declines,
 *            implicit-taint reporting, self-modifying-code invalidation.
 * The C side signals those by returning MT_FAST_SLOW; it never guesses.
 */
#ifndef MICROTAINT_FASTPATH_H
#define MICROTAINT_FASTPATH_H

#include <Python.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../instrumentation/cell_c/circuit_c_api.h"

/* ~0 is never a real instruction address, so it can mark an empty slot. */
#define MT_EMPTY_ADDR ((uint64_t)0xFFFFFFFFFFFFFFFFULL)

/* Must match MT_MAX_MEM_WRITES in hook_core.pyx and CMEM_MAX_WRITES in
 * circuit_c.c: the evaluator declines rather than truncate, so these being
 * equal means the C path never declines for want of room. */
#define MT_MAX_MEM_WRITES 32

/* mt_fast_step outcomes. */
#define MT_FAST_DONE  0   /* instruction fully handled */
#define MT_FAST_SLOW  1   /* caller must run the Python path */
#define MT_FAST_ERROR 2   /* a Python exception is set */

/* ------------------------------------------------------------------------ */
/* Per-address cache                                                         */
/*                                                                           */
/* Open-addressing table address -> entry.  The PyObject fields (instruction  */
/* bytes, circuit, the uc_arrays tuple) stay PyObjects because the decoder    */
/* and evaluator genuinely need them, but they are owned pointers here: no    */
/* dict lookup, tuple indexing or boxing on a repeat visit.                   */
/* ------------------------------------------------------------------------ */

typedef struct {
    int        size;
    PyObject  *instr_bytes;       /* owned */
    PyObject  *circuit;           /* owned */
    PyObject  *uc_arrs;           /* owned, may be NULL; keeps ctypes buffers alive */
    int       *slots;             /* input-register slots, malloc'd */
    int        n_in;
    int        have_slots;
    /* The parts of uc_arrs the register read actually needs, unpacked once.
     * uc_arrs is an 11-element Python tuple; unpacking it per instruction cost
     * eleven PyObject fetches and refcount pairs to reach four integers. */
    unsigned long long ids_addr;
    unsigned long long ptrs_addr;
    unsigned long long vals_addr;
    int        n_calls_i;
    int        need_ef;
    int        have_regs;         /* the three addresses above are valid */
    uint64_t  *in_snap;           /* pre-state taint snapshot (snap_n entries) */
    uint64_t  *out_snap;          /* post-state taint snapshot */
    uint64_t  *val_snap;          /* operand values for value-dependent circuits */
    int        snap_n;
    int        have_snap;
    int        have_val;
} MtAddrEntry;

typedef struct {
    uint64_t     *keys;
    MtAddrEntry **vals;
    Py_ssize_t    cap;
    Py_ssize_t    n;
} MtAddrMap;

static int mt_am_init(MtAddrMap *m, Py_ssize_t cap) {
    m->keys = (uint64_t *)malloc((size_t)cap * sizeof(uint64_t));
    if (!m->keys) return -1;
    m->vals = (MtAddrEntry **)calloc((size_t)cap, sizeof(void *));
    if (!m->vals) { free(m->keys); m->keys = NULL; return -1; }
    for (Py_ssize_t i = 0; i < cap; i++) m->keys[i] = MT_EMPTY_ADDR;
    m->cap = cap;
    m->n = 0;
    return 0;
}

static void mt_ae_free(MtAddrEntry *e) {
    if (!e) return;
    Py_XDECREF(e->instr_bytes);
    Py_XDECREF(e->circuit);
    Py_XDECREF(e->uc_arrs);
    free(e->slots);
    free(e->in_snap);
    free(e->out_snap);
    free(e->val_snap);
    free(e);
}

/* Drop every entry but keep the table (used by the self-modifying-code path). */
static void mt_am_clear(MtAddrMap *m) {
    if (!m->vals) return;
    for (Py_ssize_t i = 0; i < m->cap; i++) {
        if (m->keys[i] != MT_EMPTY_ADDR) {
            mt_ae_free(m->vals[i]);
            m->vals[i] = NULL;
            m->keys[i] = MT_EMPTY_ADDR;
        }
    }
    m->n = 0;
}

static void mt_am_free(MtAddrMap *m) {
    mt_am_clear(m);
    free(m->vals); m->vals = NULL;
    free(m->keys); m->keys = NULL;
    m->cap = 0;
    m->n = 0;
}

static inline Py_ssize_t mt_am_slot(MtAddrMap *m, uint64_t key) {
    uint64_t h = key * (uint64_t)0x9E3779B97F4A7C15ULL;
    h ^= h >> 29;
    Py_ssize_t i = (Py_ssize_t)(h & (uint64_t)(m->cap - 1));
    while (m->keys[i] != MT_EMPTY_ADDR && m->keys[i] != key)
        i = (i + 1) & (m->cap - 1);
    return i;
}

static int mt_am_grow(MtAddrMap *m) {
    MtAddrMap nm;
    if (mt_am_init(&nm, m->cap * 2) != 0) return -1;
    for (Py_ssize_t i = 0; i < m->cap; i++) {
        if (m->keys[i] != MT_EMPTY_ADDR) {
            Py_ssize_t j = mt_am_slot(&nm, m->keys[i]);
            nm.keys[j] = m->keys[i];
            nm.vals[j] = m->vals[i];
            nm.n++;
        }
    }
    free(m->keys); free(m->vals);
    m->keys = nm.keys; m->vals = nm.vals; m->cap = nm.cap; m->n = nm.n;
    return 0;
}

static inline MtAddrEntry *mt_am_get(MtAddrMap *m, uint64_t key) {
    if (m->cap == 0) return NULL;
    Py_ssize_t i = mt_am_slot(m, key);
    return (m->keys[i] == key) ? m->vals[i] : NULL;
}

/* Insert (or return the existing) entry for key.  NULL on OOM. */
static MtAddrEntry *mt_am_new(MtAddrMap *m, uint64_t key) {
    if (m->cap == 0 && mt_am_init(m, 256) != 0) return NULL;
    if ((m->n + 1) * 10 >= m->cap * 7 && mt_am_grow(m) != 0) return NULL;
    Py_ssize_t i = mt_am_slot(m, key);
    if (m->keys[i] == key) return m->vals[i];
    MtAddrEntry *e = (MtAddrEntry *)calloc(1, sizeof(MtAddrEntry));
    if (!e) return NULL;
    m->keys[i] = key;
    m->vals[i] = e;
    m->n++;
    return e;
}

/* ------------------------------------------------------------------------ */
/* Hot-path context                                                          */
/*                                                                           */
/* Every field is a C scalar/pointer or a BORROWED PyObject* owned by the     */
/* Cython hook object, so this struct owns nothing and needs no refcounting.  */
/*                                                                           */
/* State the Cython side mutates during a run -- the taint and value arrays   */
/* are reallocated when a new register name is interned, and several slot     */
/* indices are resolved lazily -- is held here as a POINTER TO ITS FIELD in   */
/* the hook object rather than as a copy.  Those addresses are fixed for the  */
/* object's lifetime, so the context can never go stale and there is no       */
/* "refresh the context" step to forget.  A stale copy of g_taint would mean  */
/* writing taint through a freed pointer, which is the one failure here that  */
/* would corrupt memory instead of failing a test.                            */
/* ------------------------------------------------------------------------ */

typedef int (*mt_uc_reg_read_batch_fn)(void *uc, void *ids, void *ptrs, int count);

typedef struct {
    /* taint + value state, slot-indexed (reallocated on slot growth) */
    uint64_t **g_taint;
    uint64_t **g_val;
    int        *n_slots;

    MtAddrMap *map;

    /* Unicorn register-read boundary */
    unsigned long long *uc_handle_addr;   /* resolved on first use */
    mt_uc_reg_read_batch_fn uc_reg_read_batch;

    /* evaluator boundary (borrowed; identities fixed at construction) */
    const CircuitCAPI *capi;
    PyObject *pcode;
    PyObject *shadow;
    PyObject *mem_reader;
    PyObject *slot_map;
    mt_mem_read_fn mem_fn;
    void     *mem_ctx;

    /* flag-register explosion and PC slot, all resolved lazily */
    int  *eflags_slot;
    int **ef_slots;
    int **ef_bits;
    int  *ef_n;
    int  *rip_slot;

    /* results for this instruction */
    MtMemWrite *ltw;
    int        *ltw_n;

    /* config + counters */
    int instr_cache_enabled;
    int use_cregs;
    int use_cmem;
    unsigned long *hits;
    unsigned long *misses;
    /* Instructions this C path completed on its own.  Exposed so coverage is a
     * measurement rather than an assumption: if a change quietly made the path
     * decline, the timing would barely move but this would collapse. */
    unsigned long *fast_done;
    /* Instructions dismissed by the untainted-input exit below. */
    unsigned long *prefilter_hits;
} MtFastCtx;

/* Scatter the flag bits of the packed flags register into their own value
 * slots.  The (slot, bit) pairs were derived once from the arch's flag map. */
static inline void mt_explode_eflags(MtFastCtx *c) {
    uint64_t *g_val = *c->g_val;
    uint64_t ef = g_val[*c->eflags_slot];
    const int *slots = *c->ef_slots, *bits = *c->ef_bits;
    for (int i = 0; i < *c->ef_n; i++) g_val[slots[i]] = (ef >> bits[i]) & 1;
}

/* Read this instruction's live input-register values straight into the
 * slot-indexed value array.  Returns 0 on success, MT_FAST_SLOW when the entry
 * has not been prepared yet (first visit) or the C batch read is unavailable. */
static inline int mt_read_regs(MtFastCtx *c, MtAddrEntry *e, uint64_t address) {
    if (!e->have_regs || !c->use_cregs || !c->uc_reg_read_batch) return MT_FAST_SLOW;
    if (e->n_in > 0 && !e->slots) return MT_FAST_SLOW;
    if (*c->uc_handle_addr == 0) return MT_FAST_SLOW;   /* handle not resolved yet */
    if (e->n_in > 0) {
        c->uc_reg_read_batch((void *)(uintptr_t)*c->uc_handle_addr,
                             (void *)(uintptr_t)e->ids_addr,
                             (void *)(uintptr_t)e->ptrs_addr,
                             e->n_calls_i);
        const uint64_t *vals = (const uint64_t *)(uintptr_t)e->vals_addr;
        uint64_t *g_val = *c->g_val;
        for (int i = 0; i < e->n_in; i++) g_val[e->slots[i]] = vals[i];
    }
    /* The flag and PC slots are interned lazily on the Python side.  Until they
     * exist this instruction must go slow: skipping the flag explosion would
     * leave stale flag values for the evaluator to read. */
    if (e->need_ef) {
        if (*c->eflags_slot < 0) return MT_FAST_SLOW;
        mt_explode_eflags(c);
    }
    if (*c->rip_slot < 0) return MT_FAST_SLOW;
    (*c->g_val)[*c->rip_slot] = address;
    return 0;
}

/* Record which bytes this instruction legitimately tainted, so the memory-write
 * callback does not immediately clear them again.  The (addr, size, mask)
 * triples are kept as-is rather than expanded to one entry per byte. */
static inline void mt_apply_mem_writes(MtFastCtx *c, const MtMemWrite *w, int n) {
    for (int k = 0; k < n; k++) {
        if (w[k].taint == 0) continue;
        if (*c->ltw_n < MT_MAX_MEM_WRITES) c->ltw[(*c->ltw_n)++] = w[k];
    }
}


/* Untainted-input fast exit.
 *
 * The overwhelming majority of instructions in a real program never touch
 * tainted data: loop counters, table setup, pointer arithmetic on clean values.
 * Evaluating the differential for those computes zero the expensive way -- on an
 * all-clean benchmark that was still ~124k SLEIGH cell re-executions.
 *
 * If every taint input of the circuit is zero then f(V|0) XOR f(V&~0) is zero on
 * every target, and every soundness floor is keyed on a tainted input bit, so
 * those contribute zero as well.  The correct result is therefore exactly
 * "clear the bits each assignment writes" -- note CLEAR, not "skip": an
 * instruction overwriting a previously tainted register must drop that taint,
 * and skipping would leave it stale (an over-taint that never decays).
 *
 * Conservative in one direction on purpose: it tests every pool name, including
 * write-only targets, so an instruction whose *output* register is still tainted
 * declines and takes the full path (which clears it correctly).  Cheaper to be
 * conservative here than to track read-vs-written per name.
 *
 * Returns 1 if it handled the instruction.
 */
static inline int mt_untainted_exit(MtFastCtx *c, PyObject *compiled) {
    if (!c->capi->compiled_prefilter) return 0;
    MtPrefilter pf;
    if (c->capi->compiled_prefilter(compiled, c->slot_map, &pf) != 0) return 0;

    uint64_t *g_taint = *c->g_taint;
    const int n_slots = *c->n_slots;
    for (int i = 0; i < pf.n_pool; i++) {
        int slot = pf.pool_slots[i];
        if (slot >= 0 && slot < n_slots && g_taint[slot]) return 0;
    }
    for (int k = 0; k < pf.n_out; k++) {
        int slot = pf.outs[k].slot;
        if (slot >= 0 && slot < n_slots) g_taint[slot] &= ~pf.outs[k].clear_mask;
    }
    *c->ltw_n = 0;
    (*c->prefilter_hits)++;
    (*c->fast_done)++;
    return 1;
}

/*
 * One instruction, start to finish, with no Python operation anywhere.
 *
 * `ent` is this address's cache entry and `compiled` its compiled circuit, both
 * resolved by the caller (which owns them); `cflags` are the MT_CF_* bits.
 *
 * Returns MT_FAST_DONE when the instruction is fully handled, MT_FAST_SLOW when
 * the caller must run the Python path, MT_FAST_ERROR with an exception set.
 *
 * Atomicity: on MT_FAST_SLOW nothing has been committed to the taint array, so
 * the caller's fallback starts from an unmodified state.  Bailing after a
 * partial commit would leave taint the fallback would then compute again.
 */
static int mt_fast_step(MtFastCtx *c, uint64_t address, MtAddrEntry *ent,
                        PyObject *compiled, int cflags) {
    if (!c->capi || !ent) return MT_FAST_SLOW;

    const int has_mem = (cflags & MT_CF_HAS_MEM_OPS) != 0;
    /* Cheapest possible answer first: a handful of loads, before any register
     * read, cache probe or evaluation. */
    if (!has_mem && mt_untainted_exit(c, compiled)) return MT_FAST_DONE;

    const int can_cache = c->instr_cache_enabled && !has_mem;
    const int value_indep = can_cache && (cflags & MT_CF_VALUE_INDEP) != 0;
    const int n_slots = *c->n_slots;
    const size_t nbytes = (size_t)n_slots * sizeof(uint64_t);
    uint64_t *g_taint = *c->g_taint;
    uint64_t *g_val = *c->g_val;

    /* Values are needed for evaluation and for the value-aware cache key of
     * value-dependent circuits.  Value-INDEPENDENT circuits skip the read on a
     * cache hit: their taint output does not depend on operand values. */
    int vals_filled = 0;
    if (!value_indep) {
        if (mt_read_regs(c, ent, address) != 0) return MT_FAST_SLOW;
        vals_filled = 1;
    }

    /* ---- output cache probe -------------------------------------------- */
    if (can_cache && ent->have_snap && ent->snap_n == n_slots) {
        int hit = memcmp(g_taint, ent->in_snap, nbytes) == 0;
        if (hit && ent->have_val) {
            /* A value-dependent circuit must re-check the operand values;
             * n_in == 0 would make that check vacuous, so such entries are
             * never stored with have_val set. */
            for (int i = 0; i < ent->n_in; i++) {
                if (g_val[ent->slots[i]] != ent->val_snap[i]) { hit = 0; break; }
            }
        }
        if (hit) {
            memcpy(g_taint, ent->out_snap, nbytes);
            (*c->hits)++;
            *c->ltw_n = 0;
            (*c->fast_done)++;
            return MT_FAST_DONE;
        }
    }
    if (can_cache) (*c->misses)++;

    if (!vals_filled && mt_read_regs(c, ent, address) != 0) return MT_FAST_SLOW;

    /* ---- snapshot the pre-state, before eval rewrites it in place ------- */
    int want_store = 0;
    if (can_cache) {
        if (!value_indep && (!ent->have_slots || ent->n_in <= 0)) {
            /* no operand-value snapshot possible -> do not cache, else a later
             * hit would ignore changed values */
            want_store = 0;
        } else {
            if (ent->snap_n != n_slots) {
                free(ent->in_snap);
                free(ent->out_snap);
                ent->in_snap = (uint64_t *)malloc(nbytes);
                ent->out_snap = (uint64_t *)malloc(nbytes);
                ent->snap_n = (ent->in_snap && ent->out_snap) ? n_slots : 0;
            }
            if (ent->snap_n == n_slots) {
                memcpy(ent->in_snap, g_taint, nbytes);
                if (value_indep) {
                    ent->have_val = 0;
                } else {
                    if (!ent->val_snap)
                        ent->val_snap = (uint64_t *)malloc((size_t)ent->n_in * sizeof(uint64_t));
                    if (ent->val_snap) {
                        for (int i = 0; i < ent->n_in; i++)
                            ent->val_snap[i] = g_val[ent->slots[i]];
                        ent->have_val = 1;
                    } else {
                        ent->have_val = 0;
                    }
                }
                want_store = value_indep || ent->have_val;
            }
        }
    }

    /* ---- evaluate ------------------------------------------------------- */
    MtMemWrite mw[MT_MAX_MEM_WRITES];
    int rc;
    if (has_mem) {
        if (!c->use_cmem || !c->capi->eval_mem_ptr_ci) return MT_FAST_SLOW;
        rc = c->capi->eval_mem_ptr_ci(compiled, g_taint, g_val, n_slots,
                                      c->pcode, c->shadow, c->mem_reader, c->slot_map,
                                      c->mem_fn, c->mem_ctx, mw, MT_MAX_MEM_WRITES);
    } else {
        if (!c->capi->eval_arr_ptr_i) return MT_FAST_SLOW;
        rc = c->capi->eval_arr_ptr_i(compiled, g_taint, g_val, n_slots,
                                     c->pcode, c->slot_map);
    }
    if (rc == MT_EVAL_ERROR) return MT_FAST_ERROR;
    if (rc == MT_EVAL_DECLINED) return MT_FAST_SLOW;

    /* ---- apply ---------------------------------------------------------- */
    *c->ltw_n = 0;
    if (has_mem) mt_apply_mem_writes(c, mw, rc);

    if (want_store && ent->snap_n == n_slots) {
        memcpy(ent->out_snap, g_taint, nbytes);
        ent->have_snap = 1;
    }
    (*c->fast_done)++;
    return MT_FAST_DONE;
}

#endif /* MICROTAINT_FASTPATH_H */
