/* blockpath_c -- the block-at-a-time taint runtime, and its Python surface.
 *
 * The runtime itself is blockpath.h: pure C, no PyObject, no GIL.  This file is
 * the only place Python appears, and only for things that are NOT on the hot
 * path: building a plan (once per distinct block, from the Python compiler),
 * seeding state, and the test entry points that let the runtime be gated
 * without an emulator.
 *
 * A synthetic byte-addressed memory lives here too, for the same reason: the
 * two-pass protocol, the store-to-load overlay and the deferred commit are the
 * subtle parts, and they are far easier to attack against a memory a test can
 * write to directly than against a live guest.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "taint_ir_c_api.h"

#include "blockpath.h"
#include "fastpath.h"

/* ---------------------------------------------------------------------- */
/* The hook.                                                              */
/*                                                                        */
/* Registered with uc_hook_add as a raw C function pointer, so Unicorn     */
/* calls it with no Python frame and no GIL.  It shares the instruction    */
/* path's MtFastCtx, which already carries everything a block needs: the   */
/* slot-indexed taint and value arrays, the guest memory reader, the       */
/* shadow's C entry points and the batch register read.                    */
/*                                                                        */
/* The ONE place it takes the GIL is a plan-cache miss, which is once per  */
/* distinct block and is the compiler running, not the runtime.            */
/* ---------------------------------------------------------------------- */
#define BLK_CACHE_CAP 4096

typedef struct {
    uint64_t   key;              /* address ^ mixed size; MT_EMPTY_ADDR = free */
    MtBlkPlan *plan;             /* NULL means "planned, and unhandleable" */
    PyObject  *capsule;          /* owned: the plan and its emitted code */
} BlkCacheEnt;

#define BLK_MAX_REP 256

typedef struct {
    MtFastCtx   *fc;             /* borrowed: the instruction path's context */
    MtBlkEnv     env;
    MtBlkPending pend;
    uint64_t    *sv, *st, *so, *cur_val;
    BlkCacheEnt  cache[BLK_CACHE_CAP];
    PyObject    *compiler;       /* owned; called only on a cache miss */
    /* The whole register file, read once per block. */
    unsigned long long ids_addr, ptrs_addr, vals_addr;
    int          n_calls;
    int         *reg_slots;      /* engine slot per read position */
    int          n_regs;
    unsigned long blocks, handled, unhandled, planned;
    /* Why a block was not handled, by reason.  A skipped block is unanalysed
     * code, so "how often and why" has to be a measurement. */
    unsigned long miss[MT_BLK_N_MISS];
    unsigned long no_plan, no_regs, no_slot;
    unsigned long regs_clean;    /* blocks entered with no register tainted */
    /* Findings the runtime computed and nobody had yet collected.  The runtime
     * detects a secret-dependent program counter and leaves it in `pend`;
     * until this existed the HOOK simply dropped it on the next commit, so
     * block mode ran a real binary, found the leak, and reported nothing.  The
     * runner has had this since it was written, which is exactly why no test
     * caught it: the tests drove the runner and the binaries drove the hook.
     *
     * A ring rather than a callback because this runs without the GIL.
     * `rep_total` counts every finding even when the ring is full, so a run
     * that overflows says so instead of quietly reporting fewer leaks. */
    uint64_t rep_addr[BLK_MAX_REP], rep_mask[BLK_MAX_REP];
    int n_rep;
    unsigned long rep_total;
    /* The instruction hook's [code_lo, code_hi): the bytes some cache holds a
     * decode of.  Block mode plans blocks the instruction hook never sees, so
     * without widening this the mem-write hook's self-modifying-code guard
     * never fires and a rewritten block keeps running its old plan. */
    unsigned long long *code_lo, *code_hi;
    unsigned long invalidations;
    /* Of those, how many also threw away the held block.  Counted
     * separately because the two used to be the same number, and the
     * difference is exactly the taint that used to be lost. */
    unsigned long abandoned;
} MtBlkCtx;

/* Diagnostic bisect, the same idea as MICROTAINT_NULL_HOOK on the instruction
 * path: stop the block hook after each stage so the cost of BEING hooked can be
 * told apart from the cost of the work inside.  Purely for measurement -- any
 * value but 0 produces no taint.
 *   1  return as soon as the block is announced
 *   2  ... after the plan-cache lookup
 *   3  ... after the register-file read
 *   4  ... after computing, without committing
 */
static int blk_stop_after = -1;

static int blk_stage(void) {
    if (blk_stop_after < 0) {
        const char *v = getenv("MICROTAINT_BLOCK_STOP");
        blk_stop_after = (v && *v) ? atoi(v) : 0;
    }
    return blk_stop_after;
}

static uint64_t blk_key(uint64_t address, unsigned int size) {
    uint64_t h = address * 0x9E3779B97F4A7C15ULL;
    h ^= (uint64_t)size * 0xC2B2AE3D27D4EB4FULL;
    h ^= h >> 29;
    return h ? h : 1;            /* never collide with the empty marker */
}

static BlkCacheEnt *blk_slot(MtBlkCtx *b, uint64_t key) {
    Py_ssize_t i = (Py_ssize_t)(key & (BLK_CACHE_CAP - 1));
    for (int probe = 0; probe < BLK_CACHE_CAP; probe++) {
        BlkCacheEnt *e = &b->cache[i];
        if (e->key == 0 || e->key == key) return e;
        i = (i + 1) & (BLK_CACHE_CAP - 1);
    }
    return NULL;                 /* full: refuse rather than evict mid-run */
}

/* Read the whole register file into the engine's value array, once per block.
 * The per-instruction path reads only an instruction's live inputs; a block
 * cannot know which registers its later regions will want, and asking Unicorn
 * per region is the cost block mode exists to remove. */
static int blk_read_regs(MtBlkCtx *b, const MtBlkPlan *plan) {
    MtFastCtx *c = b->fc;
    if (!c->use_cregs || !c->uc_reg_read_batch) return -1;
    if (*c->uc_handle_addr == 0) return -1;
    /* MICROTAINT_BLOCK_FULLREGS=1 forces the whole-file read back on.  Kept as
     * an escape hatch: it is the difference between "a register this block
     * needed was not read" and every other kind of divergence, and it found
     * exactly that once. */
    static int full = -1;
    if (full < 0) {
        const char *v = getenv("MICROTAINT_BLOCK_FULLREGS");
        full = (v && *v && *v != '0') ? 1 : 0;
    }
    if (full) {
        c->uc_reg_read_batch((void *)(uintptr_t)*c->uc_handle_addr,
                             (void *)(uintptr_t)b->ids_addr,
                             (void *)(uintptr_t)b->ptrs_addr, b->n_calls);
        const uint64_t *av = (const uint64_t *)(uintptr_t)b->vals_addr;
        uint64_t *gv = *c->g_val;
        const int ns = *c->n_slots;
        for (int i = 0; i < b->n_regs; i++) {
            const int slot = b->reg_slots[i];
            if (slot >= 0 && slot < ns) gv[slot] = av[i];
        }
        if (*c->eflags_slot >= 0) mt_explode_eflags(c);
        return 0;
    }
    /* The BLOCK's own register set, not the whole file.  See MtBlkPlan. */
    if (plan->n_calls <= 0) return 0;         /* reads nothing: nothing to do */
    c->uc_reg_read_batch((void *)(uintptr_t)*c->uc_handle_addr,
                         (void *)(uintptr_t)plan->ids_addr,
                         (void *)(uintptr_t)plan->ptrs_addr, plan->n_calls);
    const uint64_t *vals = (const uint64_t *)(uintptr_t)plan->vals_addr;
    uint64_t *g_val = *c->g_val;
    const int n_slots = *c->n_slots;
    for (int i = 0; i < plan->n_vals; i++) {
        const int slot = plan->val_slots[i];
        if (slot >= 0 && slot < n_slots) g_val[slot] = vals[i];
    }
    if (plan->need_flags && *c->eflags_slot >= 0) mt_explode_eflags(c);
    return 0;
}

/* Adapters: blockpath.h speaks void*, the engine's shadow entry points speak
 * PyObject*.  Neither refcounts and neither takes the GIL. */
static uint64_t blk_shadow_read(void *shadow, uint64_t addr, int size);
static void blk_shadow_write(void *shadow, uint64_t addr, uint64_t mask, int size);
static MtFastCtx *g_shadow_ctx = NULL;   /* set per hook; one engine per process */

static uint64_t blk_shadow_read(void *shadow, uint64_t addr, int size) {
    MtFastCtx *c = (MtFastCtx *)shadow;
    return c->shadow_read_mask ? c->shadow_read_mask(c->shadow, addr, size) : 0;
}

static void blk_shadow_write(void *shadow, uint64_t addr, uint64_t mask, int size) {
    MtFastCtx *c = (MtFastCtx *)shadow;
    if (c->shadow_write_mask) c->shadow_write_mask(c->shadow, addr, mask, size);
}

static int blk_mem_read(void *ctx, uint64_t addr, int size, uint64_t *out) {
    MtFastCtx *c = (MtFastCtx *)ctx;
    if (!c->mem_fn) return -1;
    return c->mem_fn(c->mem_ctx, addr, size, out);
}

/* ----------------------------------------------------------------------- */
/* A flat byte-addressed memory with a shadow beside it, for tests.        */
/* ----------------------------------------------------------------------- */
typedef struct {
    uint64_t base;
    Py_ssize_t len;
    uint8_t *val;
    uint8_t *msk;
} MemArena;

static int arena_in(const MemArena *m, uint64_t addr, int size) {
    return addr >= m->base && (addr + (uint64_t)size) <= m->base + (uint64_t)m->len;
}

static int arena_read(void *ctx, uint64_t addr, int size, uint64_t *out) {
    MemArena *m = (MemArena *)ctx;
    if (!arena_in(m, addr, size)) return -1;      /* what Unicorn refusing looks like */
    uint64_t v = 0;
    for (int i = 0; i < size; i++)
        v |= (uint64_t)m->val[addr - m->base + i] << (8 * i);
    *out = v;
    return 0;
}

static uint64_t arena_read_mask(void *shadow, uint64_t addr, int size) {
    MemArena *m = (MemArena *)shadow;
    if (!arena_in(m, addr, size)) return 0;
    uint64_t t = 0;
    for (int i = 0; i < size; i++)
        t |= (uint64_t)m->msk[addr - m->base + i] << (8 * i);
    return t;
}

/* The guest's own store.  The runtime commits only the SHADOW, because in a
 * real run the CPU has already written the value itself; a synthetic memory has
 * no CPU, so the runner does it here to model one.  Without this a load in a
 * LATER block reads a value that was never stored, which is not a property of
 * the runtime under test. */
static void arena_write_value(MemArena *m, uint64_t addr, uint64_t value, int size) {
    if (!arena_in(m, addr, size)) return;
    for (int i = 0; i < size; i++)
        m->val[addr - m->base + i] = (uint8_t)((value >> (8 * i)) & 0xFF);
}

static void arena_write_mask(void *shadow, uint64_t addr, uint64_t mask, int size) {
    MemArena *m = (MemArena *)shadow;
    if (!arena_in(m, addr, size)) return;
    for (int i = 0; i < size; i++)
        m->msk[addr - m->base + i] = (uint8_t)((mask >> (8 * i)) & 0xFF);
}

static void mem_destroy(PyObject *cap) {
    MemArena *m = (MemArena *)PyCapsule_GetPointer(cap, "microtaint.blockpath.mem");
    if (!m) { PyErr_Clear(); return; }
    free(m->val); free(m->msk); free(m);
}

static PyObject *py_mem_new(PyObject *self, PyObject *args) {
    (void)self;
    unsigned long long base; Py_ssize_t len;
    if (!PyArg_ParseTuple(args, "Kn", &base, &len)) return NULL;
    if (len <= 0) { PyErr_SetString(PyExc_ValueError, "length must be positive"); return NULL; }
    MemArena *m = (MemArena *)calloc(1, sizeof(MemArena));
    if (!m) return PyErr_NoMemory();
    m->base = base; m->len = len;
    m->val = (uint8_t *)calloc((size_t)len, 1);
    m->msk = (uint8_t *)calloc((size_t)len, 1);
    if (!m->val || !m->msk) { free(m->val); free(m->msk); free(m); return PyErr_NoMemory(); }
    return PyCapsule_New(m, "microtaint.blockpath.mem", mem_destroy);
}

static MemArena *mem_of(PyObject *cap) {
    return (MemArena *)PyCapsule_GetPointer(cap, "microtaint.blockpath.mem");
}

static PyObject *py_mem_poke(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; unsigned long long addr; Py_buffer buf;
    if (!PyArg_ParseTuple(args, "OKy*", &cap, &addr, &buf)) return NULL;
    MemArena *m = mem_of(cap);
    if (!m) { PyBuffer_Release(&buf); return NULL; }
    if (!arena_in(m, addr, (int)buf.len)) {
        PyBuffer_Release(&buf);
        PyErr_SetString(PyExc_ValueError, "outside the arena");
        return NULL;
    }
    memcpy(m->val + (addr - m->base), buf.buf, (size_t)buf.len);
    PyBuffer_Release(&buf);
    Py_RETURN_NONE;
}

static PyObject *py_mem_taint(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; unsigned long long addr, mask; int size;
    if (!PyArg_ParseTuple(args, "OKKi", &cap, &addr, &mask, &size)) return NULL;
    MemArena *m = mem_of(cap);
    if (!m) return NULL;
    arena_write_mask(m, addr, mask, size);
    Py_RETURN_NONE;
}

static PyObject *py_mem_peek(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; unsigned long long addr; int size;
    if (!PyArg_ParseTuple(args, "OKi", &cap, &addr, &size)) return NULL;
    MemArena *m = mem_of(cap);
    if (!m) return NULL;
    uint64_t v = 0;
    if (arena_read(m, addr, size, &v) != 0) {
        PyErr_SetString(PyExc_ValueError, "outside the arena");
        return NULL;
    }
    return PyLong_FromUnsignedLongLong(v);
}

static PyObject *py_mem_mask(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; unsigned long long addr; int size;
    if (!PyArg_ParseTuple(args, "OKi", &cap, &addr, &size)) return NULL;
    MemArena *m = mem_of(cap);
    if (!m) return NULL;
    return PyLong_FromUnsignedLongLong(arena_read_mask(m, addr, size));
}

/* ---------------------------------------------------------------------- */
/* Plans                                                                   */
/* ---------------------------------------------------------------------- */
typedef struct {
    MtBlkPlan *plan;
    PyObject  *keepalive;        /* the compiled-code objects the fns point into */
} PlanBox;

static void plan_destroy(PyObject *cap) {
    PlanBox *b = (PlanBox *)PyCapsule_GetPointer(cap, "microtaint.blockpath.plan");
    if (!b) { PyErr_Clear(); return; }
    mt_blk_plan_free(b->plan);
    Py_XDECREF(b->keepalive);
    free(b);
}

/* plan_new(size, [(fn_addr, region_addr, [(kind, size, needval), ...]), ...],
 *          keepalive, ids, ptrs, vals, n_calls, val_slots, need_flags) */
static PyObject *py_plan_new(PyObject *self, PyObject *args) {
    (void)self;
    int size; PyObject *regions, *keepalive, *val_slots = NULL;
    unsigned long long ids = 0, ptrs = 0, vals = 0;
    int n_calls = 0, need_flags = 0;
    if (!PyArg_ParseTuple(args, "iOO|KKKiOp", &size, &regions, &keepalive,
                          &ids, &ptrs, &vals, &n_calls, &val_slots, &need_flags))
        return NULL;
    PyObject *seq = PySequence_Fast(regions, "regions must be a sequence");
    if (!seq) return NULL;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);

    MtBlkPlan *plan = (MtBlkPlan *)calloc(1, sizeof(MtBlkPlan));
    if (!plan) { Py_DECREF(seq); return PyErr_NoMemory(); }
    plan->size = size;
    plan->n_regions = (int)n;
    plan->handleable = 1;
    plan->regions = (MtBlkRegion *)calloc((size_t)(n > 0 ? n : 1), sizeof(MtBlkRegion));
    if (!plan->regions) { free(plan); Py_DECREF(seq); return PyErr_NoMemory(); }

    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        unsigned long long fn_addr, region_addr, addr_fn = 0;
        unsigned long long prog = 0, addr_prog = 0, last_addr = 0;
        PyObject *accs;
        if (!PyArg_ParseTuple(item, "KKO|KKKK", &fn_addr, &region_addr, &accs,
                              &addr_fn, &prog, &addr_prog, &last_addr))
            goto fail;
        MtBlkRegion *r = &plan->regions[i];
        r->fn = (void *)(uintptr_t)fn_addr;
        r->addr_fn = (void *)(uintptr_t)addr_fn;
        r->prog = (void *)(uintptr_t)prog;
        r->addr_prog = (void *)(uintptr_t)addr_prog;
        r->addr = region_addr;
        r->last = last_addr ? last_addr : region_addr;
        /* A region is handleable when it can be RUN, by an emitted function or
         * by the interpreter.  Requiring an emitted one turned a program the
         * emitter merely declined into a skipped block. */
        if (!r->fn && !(r->prog && mt_blk_interp)) plan->handleable = 0;
        PyObject *aseq = PySequence_Fast(accs, "accesses must be a sequence");
        if (!aseq) goto fail;
        const Py_ssize_t na = PySequence_Fast_GET_SIZE(aseq);
        if (na > MT_BLK_MAX_ACC) {
            Py_DECREF(aseq);
            PyErr_SetString(PyExc_ValueError, "too many accesses in one region");
            goto fail;
        }
        r->n_acc = (int)na;
        r->n_load = 0;
        for (Py_ssize_t k = 0; k < na; k++) {
            int kind, asize, needval;
            if (!PyArg_ParseTuple(PySequence_Fast_GET_ITEM(aseq, k), "iii",
                                  &kind, &asize, &needval)) {
                Py_DECREF(aseq); goto fail;
            }
            r->acc_kind[k] = (signed char)kind;
            if (kind == 0) r->n_load++;
            r->acc_size[k] = (signed char)asize;
            r->acc_needval[k] = (signed char)needval;
        }
        Py_DECREF(aseq);
    }
    Py_DECREF(seq);

    plan->ids_addr = ids;
    plan->ptrs_addr = ptrs;
    plan->vals_addr = vals;
    plan->n_calls = n_calls;
    plan->need_flags = need_flags;
    if (val_slots && val_slots != Py_None) {
        PyObject *vs = PySequence_Fast(val_slots, "val_slots must be a sequence");
        if (!vs) { mt_blk_plan_free(plan); return NULL; }
        const Py_ssize_t nv = PySequence_Fast_GET_SIZE(vs);
        plan->val_slots = (int *)calloc((size_t)(nv > 0 ? nv : 1), sizeof(int));
        if (!plan->val_slots) { Py_DECREF(vs); mt_blk_plan_free(plan); return PyErr_NoMemory(); }
        for (Py_ssize_t i = 0; i < nv; i++) {
            long v = PyLong_AsLong(PySequence_Fast_GET_ITEM(vs, i));
            if (v == -1 && PyErr_Occurred()) { Py_DECREF(vs); mt_blk_plan_free(plan); return NULL; }
            plan->val_slots[i] = (int)v;
        }
        plan->n_vals = (int)nv;
        Py_DECREF(vs);
    }

    PlanBox *box = (PlanBox *)calloc(1, sizeof(PlanBox));
    if (!box) { mt_blk_plan_free(plan); return PyErr_NoMemory(); }
    box->plan = plan;
    Py_INCREF(keepalive);
    box->keepalive = keepalive;
    PyObject *cap = PyCapsule_New(box, "microtaint.blockpath.plan", plan_destroy);
    if (!cap) { mt_blk_plan_free(plan); Py_DECREF(keepalive); free(box); return NULL; }
    return cap;

fail:
    Py_DECREF(seq);
    mt_blk_plan_free(plan);
    return NULL;
}

/* ----------------------------------------------------------------------- */
/* The runner: deferred commit, exactly as the hook will drive it.         */
/* ----------------------------------------------------------------------- */
#define RUN_MAX_REP 256

typedef struct {
    int n_slots, pc_slot;
    uint64_t *g_taint;
    uint64_t *sv, *st, *so, *cur_val;
    MtBlkPending pend;
    MtBlkEnv env;
    PyObject *mem_ref;           /* owned */
    int guest_writes;            /* model the CPU's own stores (tests) */
    unsigned long committed, dropped, declined;
    unsigned long miss[MT_BLK_N_MISS];
    uint64_t rep_addr[RUN_MAX_REP], rep_mask[RUN_MAX_REP];
    int n_rep;
} RunnerBox;

static void runner_destroy(PyObject *cap) {
    RunnerBox *r = (RunnerBox *)PyCapsule_GetPointer(cap, "microtaint.blockpath.runner");
    if (!r) { PyErr_Clear(); return; }
    free(r->g_taint); free(r->sv); free(r->st); free(r->so); free(r->cur_val);
    free(r->pend.taint);
    Py_XDECREF(r->mem_ref);
    free(r);
}

static RunnerBox *runner_of(PyObject *cap) {
    return (RunnerBox *)PyCapsule_GetPointer(cap, "microtaint.blockpath.runner");
}

static PyObject *py_runner_new(PyObject *self, PyObject *args) {
    (void)self;
    int n_slots, pc_slot; PyObject *mem_cap; int guest_writes = 1;
    if (!PyArg_ParseTuple(args, "iiO|p", &n_slots, &pc_slot, &mem_cap, &guest_writes))
        return NULL;
    if (n_slots <= 0 || n_slots > MT_BLK_VAL_BASE) {
        PyErr_SetString(PyExc_ValueError, "n_slots outside the block layout");
        return NULL;
    }
    MemArena *m = mem_of(mem_cap);
    if (!m) return NULL;

    RunnerBox *r = (RunnerBox *)calloc(1, sizeof(RunnerBox));
    if (!r) return PyErr_NoMemory();
    r->n_slots = n_slots;
    r->pc_slot = pc_slot;
    r->g_taint = (uint64_t *)calloc((size_t)n_slots, sizeof(uint64_t));
    r->sv = (uint64_t *)calloc(MT_BLK_SLOTS, sizeof(uint64_t));
    r->st = (uint64_t *)calloc(MT_BLK_SLOTS, sizeof(uint64_t));
    r->so = (uint64_t *)calloc(MT_BLK_SLOTS, sizeof(uint64_t));
    r->cur_val = (uint64_t *)calloc(MT_BLK_VAL_BASE, sizeof(uint64_t));
    r->pend.taint = (uint64_t *)calloc((size_t)n_slots, sizeof(uint64_t));
    r->pend.cap_taint = n_slots;
    if (!r->g_taint || !r->sv || !r->st || !r->so || !r->cur_val || !r->pend.taint) {
        free(r->g_taint); free(r->sv); free(r->st); free(r->so); free(r->cur_val);
        free(r->pend.taint); free(r);
        return PyErr_NoMemory();
    }
    r->env.mem_read = arena_read;
    r->env.mem_ctx = m;
    r->env.shadow_read = arena_read_mask;
    r->env.shadow_write = arena_write_mask;
    r->env.shadow = m;
    r->env.sv = r->sv; r->env.st = r->st; r->env.so = r->so;
    r->env.pc_slot = pc_slot;
    r->env.miss = r->miss;
    r->guest_writes = guest_writes;
    Py_INCREF(mem_cap);
    r->mem_ref = mem_cap;
    return PyCapsule_New(r, "microtaint.blockpath.runner", runner_destroy);
}

static int load_u64(PyObject *seq, uint64_t *dst, int n) {
    PyObject *fast = PySequence_Fast(seq, "expected a sequence of ints");
    if (!fast) return -1;
    const Py_ssize_t got = PySequence_Fast_GET_SIZE(fast);
    for (int i = 0; i < n; i++) {
        dst[i] = 0;
        if (i < got) {
            unsigned long long v = PyLong_AsUnsignedLongLong(PySequence_Fast_GET_ITEM(fast, i));
            if (v == (unsigned long long)-1 && PyErr_Occurred()) { Py_DECREF(fast); return -1; }
            dst[i] = v;
        }
    }
    Py_DECREF(fast);
    return 0;
}

static PyObject *py_runner_seed(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap, *taint;
    if (!PyArg_ParseTuple(args, "OO", &cap, &taint)) return NULL;
    RunnerBox *r = runner_of(cap);
    if (!r) return NULL;
    if (load_u64(taint, r->g_taint, r->n_slots) != 0) return NULL;
    Py_RETURN_NONE;
}

static PyObject *py_runner_taint(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    RunnerBox *r = runner_of(cap);
    if (!r) return NULL;
    PyObject *out = PyList_New(r->n_slots);
    if (!out) return NULL;
    for (int i = 0; i < r->n_slots; i++)
        PyList_SET_ITEM(out, i, PyLong_FromUnsignedLongLong(r->g_taint[i]));
    return out;
}

static PyObject *py_runner_values(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    RunnerBox *r = runner_of(cap);
    if (!r) return NULL;
    PyObject *out = PyList_New(r->n_slots);
    if (!out) return NULL;
    /* The values the block left, which live in `sv`: the runtime threads them
     * there in place rather than into a separate array. */
    for (int i = 0; i < r->n_slots; i++)
        PyList_SET_ITEM(out, i, PyLong_FromUnsignedLongLong(r->sv[i]));
    return out;
}

/* Harvest a completed block's findings before the pending is cleared. */
static void runner_take_reports(RunnerBox *r) {
    if (!r->pend.valid) return;
    for (int i = 0; i < r->pend.n_reports && r->n_rep < RUN_MAX_REP; i++) {
        r->rep_addr[r->n_rep] = r->pend.rep_addr[i];
        r->rep_mask[r->n_rep] = r->pend.rep_mask[i];
        r->n_rep++;
    }
    r->pend.n_reports = 0;
}

/* Commit the held block.  Reaching a new block, or the end of the run, is what
 * proves the held one completed. */
static void runner_commit(RunnerBox *r) {
    if (!r->pend.valid) return;
    runner_take_reports(r);
    if (r->guest_writes) {
        MemArena *m = (MemArena *)r->env.shadow;
        for (int i = 0; i < r->pend.n_writes; i++)
            arena_write_value(m, r->pend.writes[i].addr,
                              r->pend.writes[i].value, r->pend.writes[i].size);
    }
    mt_blk_commit(&r->pend, &r->env, r->g_taint, r->n_slots);
    r->committed++;
}

static PyObject *py_runner_on_block(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap, *plan_cap, *reg_val;
    unsigned long long address;
    if (!PyArg_ParseTuple(args, "OOKO", &cap, &plan_cap, &address, &reg_val)) return NULL;
    RunnerBox *r = runner_of(cap);
    if (!r) return NULL;
    PlanBox *pb = (PlanBox *)PyCapsule_GetPointer(plan_cap, "microtaint.blockpath.plan");
    if (!pb) return NULL;

    runner_commit(r);

    uint64_t *vals = (uint64_t *)calloc(MT_BLK_VAL_BASE, sizeof(uint64_t));
    if (!vals) return PyErr_NoMemory();
    if (load_u64(reg_val, vals, r->n_slots) != 0) { free(vals); return NULL; }
    const int rc = mt_blk_compute(pb->plan, &r->env, vals, r->g_taint,
                                  r->n_slots, address, &r->pend);
    free(vals);
    if (rc != MT_BLK_OK) r->declined++;
    return PyLong_FromLong(rc);
}

static PyObject *py_runner_finish(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; int completed;
    if (!PyArg_ParseTuple(args, "Op", &cap, &completed)) return NULL;
    RunnerBox *r = runner_of(cap);
    if (!r) return NULL;
    if (completed) {
        runner_commit(r);
    } else if (r->pend.valid) {
        mt_blk_abandon(&r->pend);
        r->dropped++;
    }
    Py_RETURN_NONE;
}

static PyObject *py_runner_abandon(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    RunnerBox *r = runner_of(cap);
    if (!r) return NULL;
    if (r->pend.valid) { mt_blk_abandon(&r->pend); r->dropped++; }
    Py_RETURN_NONE;
}

static PyObject *py_runner_stats(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    RunnerBox *r = runner_of(cap);
    if (!r) return NULL;
    PyObject *miss = PyList_New(MT_BLK_N_MISS);
    if (!miss) return NULL;
    for (int i = 0; i < MT_BLK_N_MISS; i++)
        PyList_SET_ITEM(miss, i, PyLong_FromUnsignedLong(r->miss[i]));
    return Py_BuildValue("{s:k,s:k,s:k,s:N}",
                         "committed", r->committed,
                         "dropped", r->dropped,
                         "declined", r->declined,
                         "miss", miss);
}

static PyObject *py_runner_reports(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    RunnerBox *r = runner_of(cap);
    if (!r) return NULL;
    PyObject *out = PyList_New(r->n_rep);
    if (!out) return NULL;
    for (int i = 0; i < r->n_rep; i++)
        PyList_SET_ITEM(out, i, Py_BuildValue("(KK)",
                        (unsigned long long)r->rep_addr[i],
                        (unsigned long long)r->rep_mask[i]));
    return out;
}

/* Take a completed block's findings before the pending is cleared.
 *
 * Called at exactly the two places the block's taint is committed, because a
 * finding and the taint it came from are the same claim: a block that faults
 * partway through never happened, so neither its taint nor its report may
 * stand.  Counting is unconditional; storing stops at the ring's end. */
static void blkctx_take_reports(MtBlkCtx *b) {
    if (!b->pend.valid) return;
    for (int i = 0; i < b->pend.n_reports; i++) {
        b->rep_total++;
        if (b->n_rep >= BLK_MAX_REP) continue;
        b->rep_addr[b->n_rep] = b->pend.rep_addr[i];
        b->rep_mask[b->n_rep] = b->pend.rep_mask[i];
        b->n_rep++;
    }
    b->pend.n_reports = 0;
}

/* The trampoline.  No GIL except on a plan-cache miss. */
static void mt_blk_hook(void *uc, uint64_t address, uint32_t size, void *user_data) {
    (void)uc;
    MtBlkCtx *b = (MtBlkCtx *)user_data;
    MtFastCtx *c = b->fc;
    b->blocks++;
    const int stage = blk_stage();
    if (stage == 1) return;

    const uint64_t key = blk_key(address, size);
    BlkCacheEnt *ent = blk_slot(b, key);
    if (!ent) { b->unhandled++; b->no_slot++; return; }

    if (ent->key == 0) {
        /* First sight of this block: the compiler runs, once, with the GIL.
         * Everything after this point in the block's life is C. */
        PyGILState_STATE gil = PyGILState_Ensure();
        ent->key = key;
        ent->plan = NULL;
        ent->capsule = NULL;
        b->planned++;
        PyObject *res = PyObject_CallFunction(b->compiler, "KI",
                                              (unsigned long long)address,
                                              (unsigned int)size);
        if (!res) {
            PyErr_Clear();
        } else if (res != Py_None) {
            PlanBox *box = (PlanBox *)PyCapsule_GetPointer(
                res, "microtaint.blockpath.plan");
            if (box) {
                ent->plan = box->plan;
                Py_INCREF(res);
                ent->capsule = res;
            } else {
                PyErr_Clear();
            }
        }
        Py_XDECREF(res);
        PyGILState_Release(gil);
        /* This block's bytes are now behind a cached plan, so a write into
         * them has to invalidate it.  Widened whether or not the plan
         * compiled: a REFUSED block is cached as a refusal, and rewritten
         * bytes at the same address must not inherit it. */
        if (b->code_lo && b->code_hi) {
            if (*b->code_hi <= *b->code_lo) {   /* empty: seed both ends */
                *b->code_lo = address;
                *b->code_hi = address + (uint64_t)size;
            } else {
                if (address < *b->code_lo) *b->code_lo = address;
                if (address + (uint64_t)size > *b->code_hi)
                    *b->code_hi = address + (uint64_t)size;
            }
        }
    }

    if (stage == 2) return;

    /* Reaching a new block proves the held one completed. */
    if (b->pend.valid) {
        blkctx_take_reports(b);
        mt_blk_commit(&b->pend, &b->env, *c->g_taint, *c->n_slots);
    }

    /* How often is the whole register state clean at a block boundary?  That
     * is the cheap sound condition for skipping a block entirely: nothing to
     * clear, nothing that can spread, so taint can only arrive through a load
     * off tainted memory.  Counted before acting on it, and counted HERE
     * because the Python-visible register_taint is not a live view of this
     * array -- a probe that read it reported 100% clean on every workload,
     * which is plainly wrong. */
    {
        const uint64_t *gt = *c->g_taint;
        const int ns = *c->n_slots;
        uint64_t any = 0;
        for (int i = 0; i < ns; i++) any |= gt[i];
        b->regs_clean += (any == 0);
    }

    if (!ent->plan) { b->unhandled++; b->no_plan++; return; }
    if (blk_read_regs(b, ent->plan) != 0) { b->unhandled++; b->no_regs++; return; }
    if (stage == 3) return;
    b->env.pc_slot = *c->rip_slot;
    b->env.stop_after = stage;
    if (mt_blk_compute(ent->plan, &b->env, *c->g_val, *c->g_taint,
                       *c->n_slots, address, &b->pend) != MT_BLK_OK) {
        b->unhandled++;
        return;
    }
    if (stage == 4) mt_blk_abandon(&b->pend);
    b->handled++;
}

static void blkctx_destroy(PyObject *cap) {
    MtBlkCtx *b = (MtBlkCtx *)PyCapsule_GetPointer(cap, "microtaint.blockpath.hook");
    if (!b) { PyErr_Clear(); return; }
    for (int i = 0; i < BLK_CACHE_CAP; i++) Py_XDECREF(b->cache[i].capsule);
    Py_XDECREF(b->compiler);
    free(b->reg_slots); free(b->sv); free(b->st); free(b->so); free(b->cur_val);
    free(b->pend.taint);
    free(b);
}

/* hook_new(fastctx_addr, compiler, ids, ptrs, vals, n_calls, reg_slots) */
static PyObject *py_hook_new(PyObject *self, PyObject *args) {
    (void)self;
    unsigned long long fc_addr, ids, ptrs, vals;
    unsigned long long code_lo_addr = 0, code_hi_addr = 0;
    PyObject *compiler, *slots;
    int n_calls;
    if (!PyArg_ParseTuple(args, "KOKKKiO|KK", &fc_addr, &compiler, &ids, &ptrs,
                          &vals, &n_calls, &slots,
                          &code_lo_addr, &code_hi_addr)) return NULL;
    if (!PyCallable_Check(compiler)) {
        PyErr_SetString(PyExc_TypeError, "compiler must be callable");
        return NULL;
    }
    MtFastCtx *c = (MtFastCtx *)(uintptr_t)fc_addr;
    if (!c) { PyErr_SetString(PyExc_ValueError, "no engine context"); return NULL; }

    PyObject *fast = PySequence_Fast(slots, "reg_slots must be a sequence");
    if (!fast) return NULL;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);

    MtBlkCtx *b = (MtBlkCtx *)calloc(1, sizeof(MtBlkCtx));
    if (!b) { Py_DECREF(fast); return PyErr_NoMemory(); }
    b->reg_slots = (int *)calloc((size_t)(n > 0 ? n : 1), sizeof(int));
    b->sv = (uint64_t *)calloc(MT_BLK_SLOTS, sizeof(uint64_t));
    b->st = (uint64_t *)calloc(MT_BLK_SLOTS, sizeof(uint64_t));
    b->so = (uint64_t *)calloc(MT_BLK_SLOTS, sizeof(uint64_t));
    b->cur_val = (uint64_t *)calloc(MT_BLK_VAL_BASE, sizeof(uint64_t));
    b->pend.taint = (uint64_t *)calloc(MT_BLK_VAL_BASE, sizeof(uint64_t));
    b->pend.cap_taint = MT_BLK_VAL_BASE;
    if (!b->reg_slots || !b->sv || !b->st || !b->so || !b->cur_val || !b->pend.taint) {
        Py_DECREF(fast);
        free(b->reg_slots); free(b->sv); free(b->st); free(b->so);
        free(b->cur_val); free(b->pend.taint); free(b);
        return PyErr_NoMemory();
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        long v = PyLong_AsLong(PySequence_Fast_GET_ITEM(fast, i));
        if (v == -1 && PyErr_Occurred()) { Py_DECREF(fast); free(b); return NULL; }
        b->reg_slots[i] = (int)v;
    }
    Py_DECREF(fast);
    b->n_regs = (int)n;
    b->fc = c;
    b->code_lo = (unsigned long long *)(uintptr_t)code_lo_addr;
    b->code_hi = (unsigned long long *)(uintptr_t)code_hi_addr;
    b->ids_addr = ids; b->ptrs_addr = ptrs; b->vals_addr = vals;
    b->n_calls = n_calls;
    Py_INCREF(compiler);
    b->compiler = compiler;
    b->env.mem_read = blk_mem_read;
    b->env.mem_ctx = c;
    b->env.shadow_read = blk_shadow_read;
    b->env.shadow_write = blk_shadow_write;
    b->env.shadow = c;
    b->env.sv = b->sv; b->env.st = b->st; b->env.so = b->so;
    b->env.pc_slot = -1;
    b->env.miss = b->miss;
    (void)g_shadow_ctx;
    return PyCapsule_New(b, "microtaint.blockpath.hook", blkctx_destroy);
}

static MtBlkCtx *blkctx_of(PyObject *cap) {
    return (MtBlkCtx *)PyCapsule_GetPointer(cap, "microtaint.blockpath.hook");
}

static PyObject *py_hook_ptr(PyObject *self, PyObject *args) {
    (void)self; (void)args;
    return PyLong_FromUnsignedLongLong((unsigned long long)(uintptr_t)&mt_blk_hook);
}

static PyObject *py_hook_ud(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    MtBlkCtx *b = blkctx_of(cap);
    if (!b) return NULL;
    return PyLong_FromUnsignedLongLong((unsigned long long)(uintptr_t)b);
}

static PyObject *py_hook_finish(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; int completed;
    if (!PyArg_ParseTuple(args, "Op", &cap, &completed)) return NULL;
    MtBlkCtx *b = blkctx_of(cap);
    if (!b) return NULL;
    if (completed) {
        if (b->pend.valid) {
            blkctx_take_reports(b);
            mt_blk_commit(&b->pend, &b->env, *b->fc->g_taint, *b->fc->n_slots);
        }
    } else {
        mt_blk_abandon(&b->pend);
    }
    Py_RETURN_NONE;
}

/* hook_invalidate(hook): drop every cached plan after a write hit cached code.
 *
 * Called from the mem-write hook's self-modifying-code branch, which the block
 * hook arms by widening [code_lo, code_hi) above.  A plan is keyed by (address,
 * size) alone, so without this the rewritten bytes would run the plan compiled
 * for what used to be there -- taint computed for instructions that are gone.
 * Rare, so a full clear and a lazy re-plan is the right shape. */
static PyObject *py_hook_invalidate(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    /* The guest write that brought us here.  Optional, and a caller that omits
     * it gets the old, indiscriminate behaviour: every plan dropped AND the
     * held block abandoned. */
    unsigned long long addr = 0, wsize = 0;
    int have_range = 0;
    if (!PyArg_ParseTuple(args, "O|KK", &cap, &addr, &wsize)) return NULL;
    have_range = (wsize > 0);
    MtBlkCtx *b = blkctx_of(cap);
    if (!b) return NULL;
    for (int i = 0; i < BLK_CACHE_CAP; i++) {
        Py_XDECREF(b->cache[i].capsule);
        b->cache[i].capsule = NULL;
        b->cache[i].plan = NULL;
        b->cache[i].key = 0;
    }
    /* Abandoning the HELD block is a much bigger claim than dropping the
     * plans, and it is only true when the write landed on that block's own
     * instructions: then it was planned from bytes it did not run, and its
     * taint -- and its findings, which are as much its answer as its taint is
     * -- are not to be trusted.  A write anywhere else leaves it correct, and
     * throwing it away is an under-taint.  Measured on a guest that rewrites a
     * four-byte function in its own text: abandoning unconditionally lost the
     * taint of the value computed just before the rewrite, and with it the
     * secret-dependent branch that value reached. */
    if (!have_range || mt_blk_pending_hit(&b->pend, addr, wsize)) {
        b->pend.n_reports = 0;
        mt_blk_abandon(&b->pend);
        b->abandoned++;
    }
    b->invalidations++;
    if (b->code_lo && b->code_hi) { *b->code_lo = ~(unsigned long long)0; *b->code_hi = 0; }
    Py_RETURN_NONE;
}

/* hook_code_range(hook) -> (lo, hi), the bytes it has planned. */
static PyObject *py_hook_code_range(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    MtBlkCtx *b = blkctx_of(cap);
    if (!b) return NULL;
    return Py_BuildValue("(KK)",
                         b->code_lo ? *b->code_lo : (unsigned long long)0,
                         b->code_hi ? *b->code_hi : (unsigned long long)0);
}

/* hook_reports(hook) -> [(address, taint mask), ...], and forget them.
 *
 * Draining is Python and therefore cold: the caller does it at the end of the
 * run, or whenever it wants to emit what has been found so far.  Forgetting on
 * read is what lets a long run be drained repeatedly without the ring
 * overflowing. */
static PyObject *py_hook_reports(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    MtBlkCtx *b = blkctx_of(cap);
    if (!b) return NULL;
    PyObject *out = PyList_New(b->n_rep);
    if (!out) return NULL;
    for (int i = 0; i < b->n_rep; i++) {
        PyObject *it = Py_BuildValue("(KK)", (unsigned long long)b->rep_addr[i],
                                     (unsigned long long)b->rep_mask[i]);
        if (!it) { Py_DECREF(out); return NULL; }
        PyList_SET_ITEM(out, i, it);
    }
    b->n_rep = 0;
    return out;
}

static PyObject *py_hook_stats(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    MtBlkCtx *b = blkctx_of(cap);
    if (!b) return NULL;
    PyObject *miss = PyList_New(MT_BLK_N_MISS);
    if (!miss) return NULL;
    for (int i = 0; i < MT_BLK_N_MISS; i++)
        PyList_SET_ITEM(miss, i, PyLong_FromUnsignedLong(b->miss[i]));
    /* One `s:` pair per key, and the count is NOT checked: a format string
     * one pair short silently drops the LAST key, which is how adding
     * `abandoned` here made `reports_pending` disappear. */
    return Py_BuildValue("{s:k,s:k,s:k,s:k,s:k,s:k,s:k,s:k,s:N,s:k,s:k,s:K,s:l,s:k,s:i}",
                         "blocks", b->blocks, "handled", b->handled,
                         "unhandled", b->unhandled, "planned", b->planned,
                         "no_plan", b->no_plan, "no_regs", b->no_regs,
                         "cache_full", b->no_slot, "regs_clean", b->regs_clean, "miss", miss,
                         "invalidations", b->invalidations,
                         "abandoned", b->abandoned,
                         "last_bad_addr", (unsigned long long)b->env.last_bad_addr,
                         "last_bad_size", (long)b->env.last_bad_size,
                         "reports", b->rep_total, "reports_pending", b->n_rep);
}

static PyObject *py_layout(PyObject *self, PyObject *args) {
    (void)self; (void)args;
    return Py_BuildValue("{s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i}",
                         "val_base", MT_BLK_VAL_BASE,
                         "mem_base", MT_BLK_MEM_BASE,
                         "per_acc", MT_BLK_PER_ACC,
                         "max_acc", MT_BLK_MAX_ACC,
                         "max_writes", MT_BLK_MAX_WRITES,
                         "slots", MT_BLK_SLOTS,
                         "a_mem", MT_BLK_A_MEM,
                         "a_addr", MT_BLK_A_ADDR,
                         "a_addrt", MT_BLK_A_ADDRT,
                         "a_sttaint", MT_BLK_A_STTAINT,
                         "a_stval", MT_BLK_A_STVAL);
}

static PyMethodDef Methods[] = {
    {"mem_new", py_mem_new, METH_VARARGS, "mem_new(base, length) -> memory"},
    {"mem_poke", py_mem_poke, METH_VARARGS, "mem_poke(mem, addr, bytes)"},
    {"mem_taint", py_mem_taint, METH_VARARGS, "mem_taint(mem, addr, mask, size)"},
    {"mem_peek", py_mem_peek, METH_VARARGS, "mem_peek(mem, addr, size) -> int"},
    {"mem_mask", py_mem_mask, METH_VARARGS, "mem_mask(mem, addr, size) -> int"},
    {"plan_new", py_plan_new, METH_VARARGS, "plan_new(size, regions, keepalive) -> plan"},
    {"runner_new", py_runner_new, METH_VARARGS, "runner_new(n_slots, pc_slot, mem) -> runner"},
    {"runner_seed", py_runner_seed, METH_VARARGS, "runner_seed(runner, taint)"},
    {"runner_taint", py_runner_taint, METH_VARARGS, "runner_taint(runner) -> [int]"},
    {"runner_values", py_runner_values, METH_VARARGS,
     "runner_values(runner) -> [int], the register values the last block left"},
    {"runner_on_block", py_runner_on_block, METH_VARARGS,
     "runner_on_block(runner, plan, address, reg_val) -> rc"},
    {"runner_finish", py_runner_finish, METH_VARARGS, "runner_finish(runner, completed)"},
    {"runner_abandon", py_runner_abandon, METH_VARARGS, "runner_abandon(runner)"},
    {"runner_stats", py_runner_stats, METH_VARARGS, "runner_stats(runner) -> dict"},
    {"runner_reports", py_runner_reports, METH_VARARGS, "runner_reports(runner) -> [(addr, mask)]"},
    {"hook_new", py_hook_new, METH_VARARGS,
     "hook_new(fastctx, compiler, ids, ptrs, vals, n_calls, reg_slots"
     "[, code_lo_addr, code_hi_addr]) -> hook"},
    {"hook_ptr", py_hook_ptr, METH_NOARGS, "hook_ptr() -> the C trampoline's address"},
    {"hook_ud", py_hook_ud, METH_VARARGS, "hook_ud(hook) -> its user_data address"},
    {"hook_finish", py_hook_finish, METH_VARARGS, "hook_finish(hook, completed)"},
    {"hook_reports", py_hook_reports, METH_VARARGS,
     "hook_reports(hook) -> [(address, mask)], draining them"},
    {"hook_stats", py_hook_stats, METH_VARARGS, "hook_stats(hook) -> dict"},
    {"hook_invalidate", py_hook_invalidate, METH_VARARGS,
     "hook_invalidate(hook) -- drop every cached plan (self-modifying code)"},
    {"hook_code_range", py_hook_code_range, METH_VARARGS,
     "hook_code_range(hook) -> (lo, hi) of the bytes it has planned"},
    {"layout", py_layout, METH_NOARGS, "layout() -> the block slot layout"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef Module = {
    PyModuleDef_HEAD_INIT, "blockpath_c",
    "Block-at-a-time taint: the C runtime, and the surface that gates it.",
    -1, Methods, NULL, NULL, NULL, NULL,
};

PyMODINIT_FUNC PyInit_blockpath_c(void) {
    PyObject *m = PyModule_Create(&Module);
    if (!m) return NULL;
    /* The taint-IR interpreter, for a region whose program the host emitter
     * declined.  Without it such a region cannot run and the block is skipped,
     * so this is imported by the full dotted name too: PyCapsule_Import needs
     * the module importable as a TOP-LEVEL one, which only happens when its
     * directory is on sys.path (tests do that; the emulator does not). */
    TaintIrCAPI *api = (TaintIrCAPI *)PyCapsule_Import(
        "taint_ir_c._taint_ir_capi", 0);
    if (!api) {
        PyErr_Clear();
        PyObject *mod = PyImport_ImportModule(
            "microtaint.instrumentation.cell_c.taint_ir_c");
        if (mod) {
            PyObject *cap = PyObject_GetAttrString(mod, "_taint_ir_capi");
            if (cap) {
                api = (TaintIrCAPI *)PyCapsule_GetPointer(
                    cap, "taint_ir_c._taint_ir_capi");
                Py_DECREF(cap);
            }
            Py_DECREF(mod);
        }
        if (!api) PyErr_Clear();
    }
    if (api) mt_blk_interp = api->run;
    return m;
}
