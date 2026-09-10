/*
 * circuit_c.c — Compiled LogicCircuit evaluator
 *
 * Public API:
 *   compile_circuit(logic_circuit) -> CompiledCircuit
 *       One-time compile: walk the LogicCircuit's TaintAssignments,
 *       emit per-assignment bytecode for those that compile cleanly,
 *       fall back to a Python ref to the original Cython AST eval for
 *       the rest.
 *
 *   compiled.evaluate(eval_context) -> dict
 *       Drop-in replacement for LogicCircuit.evaluate.  Runs the
 *       per-assignment bytecode in a tight C loop, with calls into the
 *       cell_c.PCodeCellEvaluatorC for InstructionCellExpr leaves.
 *
 * The compile is cached per LogicCircuit by Python (the LRU on
 * generate_static_rule already caches the LogicCircuit; we attach the
 * compiled form as an attribute on the LogicCircuit at first eval).
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>
#include <stdint.h>
#include <string.h>
#include "circuit_bytecode.h"
#include "cell_c_api.h"
#include "circuit_c_api.h"

/* Global CAPI pointer — populated at module init via PyCapsule_Import. */
static CellCAPI *g_cell_capi = NULL;

/* Optional C-level guest-memory reader, installed for the duration of one
 * eval_mem_ptr_c call and then restored (so multiple wrappers in one process
 * stay independent).  Declared here because OP_PUSH_MEM_VALUE, far above the
 * capsule shims, is what consults it. */
static mt_mem_read_fn g_mem_fn = NULL;
static void *g_mem_ctx = NULL;

/* Shadow-memory C API — exported by microtaint.emulator.shadow as the
 * "_shadow_capi" capsule (a struct of function pointers).  Lets the C taint
 * eval path (evaluate_c_mem) read/write shadow taint at the C level, skipping
 * the PyObject_CallMethod(shadow, "read_mask", ...) attribute-resolution churn.
 * The layout must match _ShadowCAPI in shadow.pyx: field order and signatures.
 * Only the prefix this file uses is declared -- entries appended after these
 * (the GIL-free clear/is_poisoned the memory hooks call) do not change where
 * these two sit, so declaring the prefix stays correct.
 * Imported lazily (shadow is guaranteed loaded by the first memory instruction,
 * long after both modules' inits), so circuit_c never depends on shadow at its
 * own module init. */
typedef struct {
    uint64_t (*read_mask)(PyObject *shadow, uint64_t address, int size);
    void     (*write_mask)(PyObject *shadow, uint64_t address, uint64_t mask, int size);
} ShadowCAPI;

static ShadowCAPI *g_shadow_capi = NULL;
static int g_shadow_capi_tried = 0;

static ShadowCAPI *ensure_shadow_capi(void) {
    if (g_shadow_capi || g_shadow_capi_tried) return g_shadow_capi;
    g_shadow_capi_tried = 1;
    PyObject *mod = PyImport_ImportModule("microtaint.emulator.shadow");
    if (mod) {
        PyObject *cap = PyObject_GetAttrString(mod, "_shadow_capi");
        if (cap && PyCapsule_CheckExact(cap)) {
            g_shadow_capi = (ShadowCAPI *)PyCapsule_GetPointer(
                cap, "microtaint.emulator.shadow._shadow_capi");
        }
        Py_XDECREF(cap);
        Py_DECREF(mod);
    }
    if (!g_shadow_capi) PyErr_Clear();  /* not fatal — fall back to PyObject calls */
    return g_shadow_capi;
}

/* ──────────────────────────────────────────────────────────────────
 * Per-assignment compiled program
 * ────────────────────────────────────────────────────────────────── */

typedef enum {
    TGT_REG = 0,
    TGT_MEM_STATIC,    /* address known at compile time */
    TGT_MEM_DYNAMIC    /* address depends on tainted regs — must compute at run time */
} TargetKind;

typedef struct {
    /* Bytecode for the rhs expression */
    uint32_t  *bc;
    int        bc_len;

    /* Target descriptor */
    int        target_kind;        /* TargetKind */
    int        target_name_idx;    /* into string_pool, REG name or MEM_<hex>_<sz> */
    int        target_bit_start;
    int        target_bit_end;
    int        target_size_bytes;  /* for MEM targets */

    /* For TGT_MEM_DYNAMIC: separate tiny bytecode that computes the address */
    uint32_t  *addr_bc;
    int        addr_bc_len;

    /* If non-NULL, this assignment was not compilable.  Fall back to
     * the original Cython TaintAssignment. */
    PyObject  *python_assignment;
} AssignmentProg;

/* Per-circuit compiled form */
/* One memory READ's address program (see mem_reads on CompiledCircuit). */
typedef struct {
    uint32_t *bc;
    int       bc_len;
    int       size_bytes;
} MemReadSite;

typedef struct {
    PyObject_HEAD

    /* Owning Python objects we keep alive */
    PyObject       *python_circuit;        /* the LogicCircuit (for fallback) */
    PyObject       *cells;                 /* list of InstructionCellExpr (one per OP_CALL_CELL) */
    PyObject       *cell_handles;          /* list of capsules wrapping CellHandle*  (one per cell) */
    PyObject       *constants;             /* list of int constants */
    PyObject       *string_pool;           /* list of str — names referenced by bytecode */
    PyObject       *string_pool_dict;      /* dict for compile-time str-> idx lookup */
    PyObject       *arch_str;              /* str: 'AMD64' / 'X86' / 'ARM64' */

    /* Per-assignment bytecode programs */
    AssignmentProg *progs;
    int             n_progs;

    /* Cached PC target string (string_pool index, or -1) */
    int             pc_target_idx;

    /* Whether ANY assignment fell back to Python.  If true, run-time
     * uses a hybrid path: each assignment is dispatched individually. */
    int             has_python_fallback;

    /* Tier 3: whether this circuit reads memory (any OP_PUSH_MEM_*
     * or TGT_MEM_* in the bytecode).  When false, the wrapper can
     * safely cache (input_taint → output_state) without including
     * shadow-memory state in the cache key. */
    int             has_mem_ops;

    /* Whether the taint output is value-INDEPENDENT: the program reads no
     * operand VALUE (no OP_PUSH_VALUE / OP_PUSH_MEM_VALUE), so its taint is a
     * pure function of the input taints (e.g. mov/xor/not/zext).  When true the
     * instruction cache can key on the taint signature alone; when false the
     * taint is value-dependent (and/or/add/...) and the cache must also key on
     * operand values.  Default 1 (value-independent), cleared on the first
     * value push.  Sound-by-construction: a missed clear cannot happen because
     * every value input goes through those two opcodes. */
    int             value_independent;

    /* Precomputed for the GIL-free do_evaluate_c path.  const_u64 mirrors
     * `constants` as uint64 (two's-complement); c_evaluable is set when the whole
     * circuit can run with NO PyObject access: no python fallback, no memory ops,
     * and every constant fits 64 bits. */
    uint64_t       *const_u64;
    int             n_const_u64;
    int             c_evaluable;

    /* Resolved CellHandle capsule pointers, mirroring `cell_handles`.
     * OP_CALL_CELL used to re-do PyList_GET_SIZE + PyList_GET_ITEM +
     * PyCapsule_CheckExact + PyCapsule_GetPointer on EVERY cell call, and a
     * single AMD64 instruction makes ~5 of those, purely to arrive at a pointer
     * that never changes.  Resolved once here; a NULL entry means "that slot is
     * not a usable capsule", which is exactly the condition the old inline
     * checks were testing for.  Borrowed: the capsules are owned by
     * cell_handles, which this object owns and which is not replaced after
     * compilation. */
    CellHandle_API **cell_h;
    int             n_cell_h;

    /* Cached target (slot, mask) pairs for the untainted-input fast exit; see
     * MtOutSlot.  Built on first use, invalidated with pool_to_slot because the
     * slots come from the same name_to_slot mapping.  out_ok is 0 when the
     * circuit is not eligible, so ineligibility is decided once, not per call. */
    /* The circuit READS memory somewhere (a load, or a mem-ALU operand), as
     * opposed to merely writing it.  A store-only circuit cannot pull taint in
     * from memory, so when its register inputs are clean its result is clean --
     * which is what lets the untainted-input exit cover stores.  A circuit that
     * reads memory needs the effective address before that can be decided, and
     * the address is computed inline in the bytecode, so those still evaluate. */
    int             has_mem_source;

    /* One standalone address program per memory READ, captured at compile time.
     * A load's address is computed inline in the main bytecode, so there is no
     * way to ask "is this load's source tainted?" without it.  Recompiling the
     * same address expression into its own program lets the untainted-input exit
     * evaluate just the address (values only, no cells) and consult the shadow.
     * mem_reads_ok is cleared if ANY read site could not be captured faithfully,
     * because a partial list would check some loads and silently skip others. */
    MemReadSite    *mem_reads;
    int             n_mem_reads;
    int             mem_reads_ok;

    MtOutSlot      *out_slots;
    int             n_out_slots;
    int             out_built;
    int             out_ok;

    /* Like c_evaluable but ALSO admits memory circuits: set when every constant
     * fits 64 bits and there is no python fallback (memory ops ARE allowed).
     * evaluate_c_mem runs the C-array interior for these, reading/writing shadow
     * taint at the C level; loads/stores/mem-ALU that would otherwise take the
     * PyObject-heavy do_evaluate path.  PC-writing circuits are excluded at call
     * time (they need do_evaluate's implicit-taint policy check). */
    int             c_mem_evaluable;

    /* Persistent per-circuit scratch buffers for the C-array eval paths
     * (evaluate_c / evaluate_c_mem), sized to the string pool.  Allocated lazily
     * and reused across calls so the hot path does no per-instruction
     * calloc/free (the inventory's malloc/free churn).  in_t/in_v/out_t are
     * fully overwritten every call (the fill loop sets all npool slots), so no
     * per-call zeroing is needed. */
    uint64_t       *scratch_t;   /* input taint  (npool) */
    uint64_t       *scratch_v;   /* input values (npool) */
    uint64_t       *scratch_o;   /* output taint (npool, evaluate_c only) */
    int             scratch_cap; /* current capacity (in uint64 elements) */

    /* string_pool index -> global register SLOT, for the array-gather eval path
     * (evaluate_c_arr).  Built lazily from a name->slot map on first call and
     * cached (string_pool is fixed per circuit).  -1 = name not in the slot map
     * (treated as untainted / zero-value).  string_pool holds canonical PARENT
     * register names and OP_PUSH_* slices by bit_start/bit_end, so a slot holds
     * the full parent taint and no normalization is needed. */
    int            *pool_to_slot;
    int             pool_to_slot_n;   /* npool at build time (0 = not built) */
    /* WHICH name->slot mapping the cache above was built from.  The hook interns
     * register names LAZILY, so that mapping GROWS while this circuit lives: a
     * name absent on the first visit maps to -1, and keying the cache on npool
     * alone (which never changes, string_pool being fixed per circuit) pinned
     * that -1 for good.  The circuit then read 0 for an input the hook was by
     * then tracking, and dropped the taint of every output it had not yet
     * interned -- an UNDER-taint, not a miss.  Holding a strong reference makes
     * the identity test safe (a freed dict could otherwise be replaced at the
     * same address by an unrelated one), and the entry count is a sound
     * generation counter because a slot map only ever grows. */
    PyObject       *pool_map_src;     /* the name_to_slot it was built from */
    Py_ssize_t      pool_map_n;       /* len(name_to_slot) at build time */
} CompiledCircuit;

/* ──────────── string_pool helpers ──────────── */

static int strpool_intern(CompiledCircuit *cc, const char *s) {
    PyObject *key = PyUnicode_FromString(s);
    if (!key) return -1;
    PyObject *idx_obj = PyDict_GetItem(cc->string_pool_dict, key);
    if (idx_obj) {
        int idx = (int)PyLong_AsLong(idx_obj);
        Py_DECREF(key);
        return idx;
    }
    int idx = (int)PyList_GET_SIZE(cc->string_pool);
    PyList_Append(cc->string_pool, key);
    PyObject *iv = PyLong_FromLong(idx);
    PyDict_SetItem(cc->string_pool_dict, key, iv);
    Py_DECREF(iv);
    Py_DECREF(key);
    return idx;
}

/* Resolve a child register name to its canonical parent (and bit offset within
 * the parent) for the given architecture.  Mirrors _ARCH_PARENT_REGS in ast.pyx.
 *
 * Returns 1 if name is a child whose parent we resolved into *out_parent /
 * *out_offset; 0 if name is already canonical / unknown — caller uses name as-is. */
static int resolve_parent_for_arch(PyObject *arch_str_obj,
                                    const char *name,
                                    PyObject **out_parent_str,
                                    int *out_bit_offset) {
    static PyObject *parent_regs_dict = NULL;
    if (!parent_regs_dict) {
        PyObject *mod = PyImport_ImportModule("microtaint.instrumentation.ast");
        if (!mod) return 0;
        parent_regs_dict = PyObject_GetAttrString(mod, "_ARCH_PARENT_REGS");
        Py_DECREF(mod);
        if (!parent_regs_dict) return 0;
    }
    PyObject *arch_map = PyDict_GetItem(parent_regs_dict, arch_str_obj);
    if (!arch_map) return 0;
    PyObject *name_obj = PyUnicode_FromString(name);
    if (!name_obj) return 0;
    PyObject *info = PyDict_GetItem(arch_map, name_obj);
    Py_DECREF(name_obj);
    if (!info || !PyTuple_Check(info) || PyTuple_GET_SIZE(info) < 2) return 0;
    *out_parent_str = PyTuple_GET_ITEM(info, 0);
    Py_INCREF(*out_parent_str);
    *out_bit_offset = (int)PyLong_AsLong(PyTuple_GET_ITEM(info, 1));
    return 1;
}

/* ──────────── Bytecode emitter ──────────── */

typedef struct {
    uint32_t  buf[CIRCUIT_BC_MAX];
    int       len;
    int       overflow;     /* set if buf overflowed */
    int       fallback;     /* set if uncompilable opcode encountered */
    /* Counted so a captured address subprogram can be vetted for the things
     * that would make evaluating it standalone unsound: a cell call (its value
     * would be missing) or a memory read (the address would depend on memory we
     * have not proven clean). */
    int       n_cells;
    int       n_mem;
} BCEmit;

static void emit(BCEmit *e, uint32_t v) {
    if (e->len >= CIRCUIT_BC_MAX) { e->overflow = 1; return; }
    e->buf[e->len++] = v;
}

/* Forward decl */
static void compile_expr(CompiledCircuit *cc, BCEmit *e, PyObject *expr);

/* Emit an OP_CALL_CELL. `cell_obj` is the InstructionCellExpr. */
static void emit_call_cell(CompiledCircuit *cc, BCEmit *e, PyObject *cell_obj) {
    /* Get .inputs dict and walk in dict-iteration order, emitting
     * per-input expressions, then a final OP_CALL_CELL with the count
     * and per-input name-idx args. */
    PyObject *inputs = PyObject_GetAttrString(cell_obj, "inputs");
    if (!inputs || !PyDict_Check(inputs)) {
        Py_XDECREF(inputs);
        e->fallback = 1;
        return;
    }

    /* Memory-keyed cell inputs (MEM_<...>) ARE handled by the fast cell-call
     * path: CellHandle parses the key (static MEM_0x<addr>_<size> or
     * register-relative MEM_<reg>_<off>_<size>) and cell_eval_fast writes the
     * pushed value to the frame's memory (two-pass load, mirroring load_flat).
     * We only need has_mem_ops set so the wrapper's per-instruction cache stays
     * sound; the input value-expr's MemoryOperand leaves normally set it, but
     * mark it here too since expr_tree_reads_memory does not descend into an
     * InstructionCellExpr's inputs dict. */
    {
        PyObject *k, *v;
        Py_ssize_t sp = 0;
        while (PyDict_Next(inputs, &sp, &k, &v)) {
            const char *kn = PyUnicode_AsUTF8(k);
            if (!kn) { PyErr_Clear(); continue; }
            if (strncmp(kn, "MEM_", 4) == 0) { cc->has_mem_ops = 1; break; }
        }
    }

    /* Add cell to cells list */
    int cell_idx = (int)PyList_GET_SIZE(cc->cells);
    PyList_Append(cc->cells, cell_obj);

    Py_ssize_t n = PyDict_Size(inputs);

    /* Pre-emit each input expression in order, then collect names */
    int name_idxs[64];
    if (n > 64) {
        Py_DECREF(inputs);
        e->fallback = 1;
        return;
    }
    PyObject *key, *val;
    Py_ssize_t pos = 0;
    int i = 0;
    while (PyDict_Next(inputs, &pos, &key, &val)) {
        const char *name = PyUnicode_AsUTF8(key);
        if (!name) { Py_DECREF(inputs); e->fallback = 1; return; }
        compile_expr(cc, e, val);
        if (e->fallback) { Py_DECREF(inputs); return; }
        name_idxs[i++] = strpool_intern(cc, name);
    }
    Py_DECREF(inputs);

    e->n_cells++;
    emit(e, OP_CALL_CELL);
    emit(e, (uint32_t)cell_idx);
    emit(e, (uint32_t)n);
    for (i = 0; i < (int)n; i++) emit(e, (uint32_t)name_idxs[i]);
}

/* Walk a Python Expr tree looking for memory-reading leaves.  Returns 1
 * iff any subexpression's class name is in MEMORY_READING_EXPR_CLASSES.
 *
 * Used to fix the `has_mem_ops` contract: when an assignment's expression
 * contains an expression class that `compile_expr` cannot emit (and the
 * whole assignment becomes a Python fallback), we still need to know if
 * the *Python* evaluation will read shadow memory.  If it will, the
 * wrapper's per-instruction cache must not treat the circuit as
 * cache-eligible — otherwise stale taint replay across iterations.
 *
 * The walk is conservative: any class name match anywhere in the tree
 * sets has_mem_ops.  Refcounting: borrowed references only; we never
 * own anything past the function boundary.
 *
 * Recursion depth is bounded by the depth of the AST, which is small
 * for static rules — typical ~10 levels, hard cap below as safety.
 */
static int expr_tree_reads_memory(PyObject *expr, int depth) {
    if (depth > 64) return 0;   /* safety: pathological depth */
    if (!expr || expr == Py_None) return 0;
    PyObject *cls = PyObject_GetAttrString(expr, "__class__");
    if (!cls) { PyErr_Clear(); return 0; }
    PyObject *name_obj = PyObject_GetAttrString(cls, "__name__");
    Py_DECREF(cls);
    if (!name_obj) { PyErr_Clear(); return 0; }
    const char *name = PyUnicode_AsUTF8(name_obj);
    int hit = 0;
    if (name) {
        if (strcmp(name, "MemoryOperand") == 0) {
            hit = 1;
        }
    }
    Py_DECREF(name_obj);
    if (hit) return 1;

    /* Recurse through common child slots.  We don't know the exact set of
     * Expr shapes here, but each known kind uses one of: lhs+rhs (BinaryExpr),
     * expr (UnaryExpr / AvalancheExpr), address_expr (MemoryOperand — already
     * matched above), or no children (TaintOperand / Constant). */
    static const char *child_attrs[] = {"lhs", "rhs", "expr", "address_expr", NULL};
    for (int i = 0; child_attrs[i]; i++) {
        if (!PyObject_HasAttrString(expr, child_attrs[i])) continue;
        PyObject *child = PyObject_GetAttrString(expr, child_attrs[i]);
        if (!child) { PyErr_Clear(); continue; }
        int sub = expr_tree_reads_memory(child, depth + 1);
        Py_DECREF(child);
        if (sub) return 1;
    }
    return 0;
}

/* Convert a Python int to a 64-bit value with two's-complement semantics.
 * Accepts any value representable in 64 bits SIGNED or UNSIGNED
 * ([INT64_MIN, UINT64_MAX]); a negative int becomes its 2's-complement uint64
 * (e.g. -8 -> 0xFFFFFFFFFFFFFFF8), which is exactly how a negative address
 * offset or immediate must land on the uint64 bytecode stack.  Returns 1 on
 * success (*out set), 0 if the magnitude needs more than 64 bits (the caller
 * then falls back to the Python evaluator, which is arbitrary-precision). */
static int pylong_to_u64(PyObject *v, uint64_t *out) {
    unsigned long long uv = PyLong_AsUnsignedLongLong(v);
    if (!PyErr_Occurred()) { *out = (uint64_t)uv; return 1; }
    PyErr_Clear();
    long long sv = PyLong_AsLongLong(v);
    if (!PyErr_Occurred()) { *out = (uint64_t)sv; return 1; }
    PyErr_Clear();
    return 0;
}

/* Compile one Expr subtree into bytecode (post-order walk: emit operands first). */

/* Capture a standalone program computing THIS memory read's address.
 *
 * The address is otherwise only computed inline in the main bytecode, mixed
 * into the surrounding taint computation, so there is no way to ask "is this
 * load's source tainted?" without re-deriving it.  Recompiling the same address
 * expression into its own program gives the untainted-input exit something it
 * can evaluate on values alone.
 *
 * Refuses (and disables the whole mechanism for this circuit) whenever the
 * address could not be reproduced faithfully:
 *   - a cell call in the address: its concrete value would be missing, and a
 *     wrong address means consulting the shadow for the wrong bytes, which
 *     would report clean for a tainted load.  That is a silent UNDER-taint, so
 *     it is refused rather than approximated.
 *   - a memory read in the address (a load whose pointer itself comes from
 *     memory): the pointer's own taint is not established at that point.
 *   - a width the 64-bit shadow mask cannot describe in one read.
 * Refusal costs a fast path, never correctness.
 *
 * compile_expr appends any cell it meets to cc->cells, so the second compile is
 * rolled back to the prior length; otherwise a rejected capture would leave a
 * duplicate cell behind. */
static void record_mem_read(CompiledCircuit *cc, PyObject *addr_e, int size_bytes) {
    if (!cc->mem_reads_ok) return;
    if (size_bytes <= 0 || size_bytes > 8) { cc->mem_reads_ok = 0; return; }

    Py_ssize_t cells_before = PyList_GET_SIZE(cc->cells);
    BCEmit ae = {{0}, 0, 0, 0, 0, 0};
    compile_expr(cc, &ae, addr_e);
    if (PyList_GET_SIZE(cc->cells) > cells_before)
        PyList_SetSlice(cc->cells, cells_before, PyList_GET_SIZE(cc->cells), NULL);

    if (ae.fallback || ae.overflow || ae.n_cells || ae.n_mem) { cc->mem_reads_ok = 0; return; }
    emit(&ae, OP_END);
    if (ae.overflow) { cc->mem_reads_ok = 0; return; }

    MemReadSite *grown = (MemReadSite *)realloc(
        cc->mem_reads, (size_t)(cc->n_mem_reads + 1) * sizeof(MemReadSite));
    if (!grown) { cc->mem_reads_ok = 0; return; }
    cc->mem_reads = grown;
    MemReadSite *site = &cc->mem_reads[cc->n_mem_reads];
    site->bc = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)ae.len);
    if (!site->bc) { cc->mem_reads_ok = 0; return; }
    memcpy(site->bc, ae.buf, sizeof(uint32_t) * (size_t)ae.len);
    site->bc_len = ae.len;
    site->size_bytes = size_bytes;
    cc->n_mem_reads++;
}


static void compile_expr(CompiledCircuit *cc, BCEmit *e, PyObject *expr) {
    if (e->fallback || e->overflow) return;
    if (!expr || expr == Py_None) {
        /* Treat as Constant 0 */
        int ci = (int)PyList_GET_SIZE(cc->constants);
        PyObject *zero = PyLong_FromLong(0);
        PyList_Append(cc->constants, zero);
        Py_DECREF(zero);
        emit(e, OP_PUSH_CONST);
        emit(e, (uint32_t)ci);
        return;
    }
    PyObject *cls = (PyObject *)Py_TYPE(expr);
    PyObject *cls_name = PyObject_GetAttrString(cls, "__name__");
    const char *cn = PyUnicode_AsUTF8(cls_name);

    if (strcmp(cn, "TaintOperand") == 0) {
        Py_DECREF(cls_name);
        PyObject *name = PyObject_GetAttrString(expr, "name");
        PyObject *bs = PyObject_GetAttrString(expr, "bit_start");
        PyObject *be = PyObject_GetAttrString(expr, "bit_end");
        PyObject *it = PyObject_GetAttrString(expr, "is_taint");
        if (!name || !bs || !be || !it) goto err_taintop;
        const char *sname = PyUnicode_AsUTF8(name);
        if (!sname) goto err_taintop;
        int bit_start = (int)PyLong_AsLong(bs);
        int bit_end   = (int)PyLong_AsLong(be);
        int is_taint  = PyObject_IsTrue(it);

        /* Sanity gate: a wide (>64-bit) operand cannot fit the uint64 bytecode
         * stack (mask_range is uint64, PUSH_* truncates).  Bail to the Python
         * evaluator (arbitrary width) instead of feeding do_evaluate a value it
         * would mangle.  One compile-time check per operand; the uint64 fast path
         * is unchanged. */
        if (bit_end - bit_start + 1 > 64) goto err_taintop;

        /* Resolve to canonical parent register at compile time, mirroring
         * what TaintOperand.evaluate does at runtime. */
        PyObject *parent_str = NULL;
        int parent_bit_off = 0;
        const char *canonical_name = sname;
        if (resolve_parent_for_arch(cc->arch_str, sname, &parent_str, &parent_bit_off)) {
            canonical_name = PyUnicode_AsUTF8(parent_str);
            bit_start += parent_bit_off;
            bit_end   += parent_bit_off;
        }
        int idx = strpool_intern(cc, canonical_name);
        Py_XDECREF(parent_str);

        Py_DECREF(name); Py_DECREF(bs); Py_DECREF(be); Py_DECREF(it);
        emit(e, is_taint ? OP_PUSH_TAINT : OP_PUSH_VALUE);
        emit(e, (uint32_t)idx);
        emit(e, (uint32_t)bit_start);
        emit(e, (uint32_t)bit_end);
        return;
    err_taintop:
        Py_XDECREF(name); Py_XDECREF(bs); Py_XDECREF(be); Py_XDECREF(it);
        e->fallback = 1;
        return;
    }
    if (strcmp(cn, "Constant") == 0) {
        Py_DECREF(cls_name);
        PyObject *value = PyObject_GetAttrString(expr, "value");
        if (!value) { e->fallback = 1; return; }
        /* Compile if the value fits in 64 bits (signed or unsigned two's
         * complement).  Negative address offsets and immediates lower to
         * negative Constants; pylong_to_u64 accepts them and only bails on a
         * genuinely >64-bit magnitude. */
        if (!PyLong_Check(value)) { Py_DECREF(value); e->fallback = 1; return; }
        uint64_t uv;
        if (!pylong_to_u64(value, &uv)) { Py_DECREF(value); e->fallback = 1; return; }
        int ci = (int)PyList_GET_SIZE(cc->constants);
        PyList_Append(cc->constants, value);
        Py_DECREF(value);
        emit(e, OP_PUSH_CONST);
        emit(e, (uint32_t)ci);
        (void)uv;
        return;
    }
    if (strcmp(cn, "BinaryExpr") == 0) {
        Py_DECREF(cls_name);
        PyObject *op = PyObject_GetAttrString(expr, "op");
        PyObject *lhs = PyObject_GetAttrString(expr, "lhs");
        PyObject *rhs = PyObject_GetAttrString(expr, "rhs");
        if (!op || !lhs || !rhs) {
            Py_XDECREF(op); Py_XDECREF(lhs); Py_XDECREF(rhs);
            e->fallback = 1; return;
        }
        PyObject *opv = PyObject_GetAttrString(op, "value");
        const char *opname = PyUnicode_AsUTF8(opv);
        compile_expr(cc, e, lhs);
        compile_expr(cc, e, rhs);
        Py_DECREF(lhs); Py_DECREF(rhs);
        if (e->fallback) { Py_DECREF(op); Py_DECREF(opv); return; }
        if      (strcmp(opname,"AND") == 0) emit(e, OP_AND);
        else if (strcmp(opname,"OR")  == 0) emit(e, OP_OR);
        else if (strcmp(opname,"XOR") == 0) emit(e, OP_XOR);
        else if (strcmp(opname,"ADD") == 0) emit(e, OP_ADD);
        else if (strcmp(opname,"SUB") == 0) emit(e, OP_SUB);
        else if (strcmp(opname,"LEFT")== 0) emit(e, OP_SHL);
        else if (strcmp(opname,"RIGHT")==0) emit(e, OP_SHR);
        else { e->fallback = 1; }
        Py_DECREF(op); Py_DECREF(opv);
        return;
    }
    if (strcmp(cn, "UnaryExpr") == 0) {
        Py_DECREF(cls_name);
        PyObject *op = PyObject_GetAttrString(expr, "op");
        PyObject *sub = PyObject_GetAttrString(expr, "expr");
        if (!op || !sub) { Py_XDECREF(op); Py_XDECREF(sub); e->fallback = 1; return; }
        PyObject *opv = PyObject_GetAttrString(op, "value");
        const char *opname = PyUnicode_AsUTF8(opv);
        compile_expr(cc, e, sub);
        Py_DECREF(sub);
        if (e->fallback) { Py_DECREF(op); Py_DECREF(opv); return; }
        if (strcmp(opname,"NOT") == 0) emit(e, OP_NOT);
        else e->fallback = 1;
        Py_DECREF(op); Py_DECREF(opv);
        return;
    }
    if (strcmp(cn, "AvalancheExpr") == 0) {
        Py_DECREF(cls_name);
        PyObject *sub = PyObject_GetAttrString(expr, "expr");
        PyObject *sb  = PyObject_GetAttrString(expr, "size_bits");
        if (!sub || !sb) { Py_XDECREF(sub); Py_XDECREF(sb); e->fallback = 1; return; }
        int size_bits = (int)PyLong_AsLong(sb);
        Py_DECREF(sb);
        if (size_bits <= 0 || size_bits > 64) {
            /* Can't fit in uint64 — fall back */
            Py_DECREF(sub);
            e->fallback = 1;
            return;
        }
        compile_expr(cc, e, sub);
        Py_DECREF(sub);
        if (e->fallback) return;
        emit(e, OP_AVALANCHE);
        emit(e, (uint32_t)size_bits);
        return;
    }
    if (strcmp(cn, "FullMaskAvalancheExpr") == 0) {
        Py_DECREF(cls_name);
        PyObject *sub = PyObject_GetAttrString(expr, "dep");
        PyObject *fm  = PyObject_GetAttrString(expr, "full_mask");
        if (!sub || !fm) { Py_XDECREF(sub); Py_XDECREF(fm); e->fallback = 1; return; }
        if (!PyLong_Check(fm)) { Py_DECREF(sub); Py_DECREF(fm); e->fallback = 1; return; }
        /* full_mask > 64 bits can't fit the uint64 stack -> Python fallback. */
        unsigned long long fmv = PyLong_AsUnsignedLongLong(fm);
        (void)fmv;
        if (PyErr_Occurred()) { PyErr_Clear(); Py_DECREF(sub); Py_DECREF(fm); e->fallback = 1; return; }
        compile_expr(cc, e, sub);
        Py_DECREF(sub);
        if (e->fallback) { Py_DECREF(fm); return; }
        int ci = (int)PyList_GET_SIZE(cc->constants);
        PyList_Append(cc->constants, fm);
        Py_DECREF(fm);
        emit(e, OP_FULLMASK_AVAL);
        emit(e, (uint32_t)ci);
        return;
    }
    if (strcmp(cn, "EqualityTaintExpr") == 0) {
        Py_DECREF(cls_name);
        PyObject *av = PyObject_GetAttrString(expr, "a_val");
        PyObject *at = PyObject_GetAttrString(expr, "a_taint");
        PyObject *bv = PyObject_GetAttrString(expr, "b_val");
        PyObject *bt = PyObject_GetAttrString(expr, "b_taint");
        PyObject *w  = PyObject_GetAttrString(expr, "width");
        if (!av || !at || !bv || !bt || !w) {
            Py_XDECREF(av); Py_XDECREF(at); Py_XDECREF(bv); Py_XDECREF(bt); Py_XDECREF(w);
            e->fallback = 1; return;
        }
        int width = (int)PyLong_AsLong(w);
        Py_DECREF(w);
        if (width <= 0 || width > 64) {  /* mask can't fit uint64 */
            Py_DECREF(av); Py_DECREF(at); Py_DECREF(bv); Py_DECREF(bt);
            e->fallback = 1; return;
        }
        compile_expr(cc, e, av);   /* stack bottom->top: a_val, a_taint, b_val, b_taint */
        compile_expr(cc, e, at);
        compile_expr(cc, e, bv);
        compile_expr(cc, e, bt);
        Py_DECREF(av); Py_DECREF(at); Py_DECREF(bv); Py_DECREF(bt);
        if (e->fallback) return;
        emit(e, OP_EQ_TAINT);
        emit(e, (uint32_t)width);
        return;
    }
    if (strcmp(cn, "VariableBitSelectTaintExpr") == 0) {
        Py_DECREF(cls_name);
        PyObject *sv = PyObject_GetAttrString(expr, "src_val");
        PyObject *st = PyObject_GetAttrString(expr, "src_taint");
        PyObject *iv = PyObject_GetAttrString(expr, "idx_val");
        PyObject *it = PyObject_GetAttrString(expr, "idx_taint");
        PyObject *w  = PyObject_GetAttrString(expr, "width");
        if (!sv || !st || !iv || !it || !w) {
            Py_XDECREF(sv); Py_XDECREF(st); Py_XDECREF(iv); Py_XDECREF(it); Py_XDECREF(w);
            e->fallback = 1; return;
        }
        int width = (int)PyLong_AsLong(w);
        Py_DECREF(w);
        if (width <= 0 || width > 64) {
            Py_DECREF(sv); Py_DECREF(st); Py_DECREF(iv); Py_DECREF(it);
            e->fallback = 1; return;
        }
        compile_expr(cc, e, sv);   /* stack: src_val, src_taint, idx_val, idx_taint */
        compile_expr(cc, e, st);
        compile_expr(cc, e, iv);
        compile_expr(cc, e, it);
        Py_DECREF(sv); Py_DECREF(st); Py_DECREF(iv); Py_DECREF(it);
        if (e->fallback) return;
        emit(e, OP_VAR_BIT_SELECT);
        emit(e, (uint32_t)width);
        return;
    }
    if (strcmp(cn, "ComparisonTaintExpr") == 0) {
        Py_DECREF(cls_name);
        PyObject *av = PyObject_GetAttrString(expr, "a_val");
        PyObject *at = PyObject_GetAttrString(expr, "a_taint");
        PyObject *bv = PyObject_GetAttrString(expr, "b_val");
        PyObject *bt = PyObject_GetAttrString(expr, "b_taint");
        PyObject *w  = PyObject_GetAttrString(expr, "width");
        PyObject *sg = PyObject_GetAttrString(expr, "is_signed");
        PyObject *oe = PyObject_GetAttrString(expr, "or_equal");
        if (!av || !at || !bv || !bt || !w || !sg || !oe) {
            Py_XDECREF(av); Py_XDECREF(at); Py_XDECREF(bv); Py_XDECREF(bt);
            Py_XDECREF(w); Py_XDECREF(sg); Py_XDECREF(oe);
            e->fallback = 1; return;
        }
        int width = (int)PyLong_AsLong(w);
        int flags = (PyObject_IsTrue(sg) ? 1 : 0) | (PyObject_IsTrue(oe) ? 2 : 0);
        Py_DECREF(w); Py_DECREF(sg); Py_DECREF(oe);
        if (width <= 0 || width > 64) {
            Py_DECREF(av); Py_DECREF(at); Py_DECREF(bv); Py_DECREF(bt);
            e->fallback = 1; return;
        }
        compile_expr(cc, e, av);   /* stack: a_val, a_taint, b_val, b_taint */
        compile_expr(cc, e, at);
        compile_expr(cc, e, bv);
        compile_expr(cc, e, bt);
        Py_DECREF(av); Py_DECREF(at); Py_DECREF(bv); Py_DECREF(bt);
        if (e->fallback) return;
        emit(e, OP_CMP_TAINT);
        emit(e, (uint32_t)width);
        emit(e, (uint32_t)flags);
        return;
    }
    if (strcmp(cn, "SignedOverflowTaintExpr") == 0) {
        Py_DECREF(cls_name);
        PyObject *av = PyObject_GetAttrString(expr, "a_val");
        PyObject *at = PyObject_GetAttrString(expr, "a_taint");
        PyObject *bv = PyObject_GetAttrString(expr, "b_val");
        PyObject *bt = PyObject_GetAttrString(expr, "b_taint");
        PyObject *cv = PyObject_GetAttrString(expr, "c_val");
        PyObject *ct = PyObject_GetAttrString(expr, "c_taint");
        PyObject *w  = PyObject_GetAttrString(expr, "width");
        PyObject *sub = PyObject_GetAttrString(expr, "is_sub");
        if (!av || !at || !bv || !bt || !w || !sub) {
            Py_XDECREF(av); Py_XDECREF(at); Py_XDECREF(bv); Py_XDECREF(bt);
            Py_XDECREF(cv); Py_XDECREF(ct); Py_XDECREF(w); Py_XDECREF(sub);
            e->fallback = 1; return;
        }
        int width = (int)PyLong_AsLong(w);
        int is_sub = PyObject_IsTrue(sub);
        int has_c = (cv && cv != Py_None && ct && ct != Py_None);
        Py_DECREF(w); Py_DECREF(sub);
        if (width < 2 || width > 64) {   /* need a sign bit; >64 can't fit uint64 */
            Py_DECREF(av); Py_DECREF(at); Py_DECREF(bv); Py_DECREF(bt);
            Py_XDECREF(cv); Py_XDECREF(ct);
            e->fallback = 1; return;
        }
        compile_expr(cc, e, av);   /* stack: a_val,a_taint,b_val,b_taint[,c_val,c_taint] */
        compile_expr(cc, e, at);
        compile_expr(cc, e, bv);
        compile_expr(cc, e, bt);
        if (has_c) { compile_expr(cc, e, cv); compile_expr(cc, e, ct); }
        Py_DECREF(av); Py_DECREF(at); Py_DECREF(bv); Py_DECREF(bt);
        Py_XDECREF(cv); Py_XDECREF(ct);
        if (e->fallback) return;
        emit(e, OP_SIGNED_OVF);
        emit(e, (uint32_t)width);
        emit(e, (uint32_t)(is_sub | (has_c ? 2 : 0)));
        return;
    }
    if (strcmp(cn, "VariableMultiplyTaintExpr") == 0) {
        Py_DECREF(cls_name);
        PyObject *av = PyObject_GetAttrString(expr, "a_val");
        PyObject *at = PyObject_GetAttrString(expr, "a_taint");
        PyObject *bv = PyObject_GetAttrString(expr, "b_val");
        PyObject *bt = PyObject_GetAttrString(expr, "b_taint");
        PyObject *iw = PyObject_GetAttrString(expr, "in_width");
        PyObject *sg = PyObject_GetAttrString(expr, "is_signed");
        PyObject *ol = PyObject_GetAttrString(expr, "out_lo");
        PyObject *oh = PyObject_GetAttrString(expr, "out_hi");
        if (!av || !at || !bv || !bt || !iw || !sg || !ol || !oh) {
            Py_XDECREF(av); Py_XDECREF(at); Py_XDECREF(bv); Py_XDECREF(bt);
            Py_XDECREF(iw); Py_XDECREF(sg); Py_XDECREF(ol); Py_XDECREF(oh);
            e->fallback = 1; return;
        }
        int in_width = (int)PyLong_AsLong(iw);
        int is_signed = PyObject_IsTrue(sg);
        int out_lo = (int)PyLong_AsLong(ol);
        int out_hi = (int)PyLong_AsLong(oh);
        Py_DECREF(iw); Py_DECREF(sg); Py_DECREF(ol); Py_DECREF(oh);
        /* operands must fit the uint64 stack; product is 2w<=128 (handled via
         * __int128); output window must fit uint64. */
        if (in_width < 1 || in_width > 64 || out_hi - out_lo > 64 || out_hi - out_lo <= 0) {
            Py_DECREF(av); Py_DECREF(at); Py_DECREF(bv); Py_DECREF(bt);
            e->fallback = 1; return;
        }
        compile_expr(cc, e, av);   /* stack: a_val, a_taint, b_val, b_taint */
        compile_expr(cc, e, at);
        compile_expr(cc, e, bv);
        compile_expr(cc, e, bt);
        Py_DECREF(av); Py_DECREF(at); Py_DECREF(bv); Py_DECREF(bt);
        if (e->fallback) return;
        emit(e, OP_VAR_MUL_TAINT);
        emit(e, (uint32_t)in_width);
        emit(e, (uint32_t)(is_signed ? 1 : 0));
        emit(e, (uint32_t)out_lo);
        emit(e, (uint32_t)out_hi);
        return;
    }
    if (strcmp(cn, "VariableShiftTaintExpr") == 0) {
        Py_DECREF(cls_name);
        PyObject *sv = PyObject_GetAttrString(expr, "src_val");
        PyObject *st = PyObject_GetAttrString(expr, "src_taint");
        PyObject *mv = PyObject_GetAttrString(expr, "amt_val");
        PyObject *mt = PyObject_GetAttrString(expr, "amt_taint");
        PyObject *w  = PyObject_GetAttrString(expr, "width");
        PyObject *kd = PyObject_GetAttrString(expr, "kind");
        PyObject *am = PyObject_GetAttrString(expr, "amt_mask");
        if (!sv || !st || !mv || !mt || !w || !kd || !am) {
            Py_XDECREF(sv); Py_XDECREF(st); Py_XDECREF(mv); Py_XDECREF(mt);
            Py_XDECREF(w); Py_XDECREF(kd); Py_XDECREF(am);
            e->fallback = 1; return;
        }
        int width = (int)PyLong_AsLong(w);
        int kind = (int)PyLong_AsLong(kd);
        Py_DECREF(w); Py_DECREF(kd);
        /* amt_mask must fit uint64 (folded MIPS 64-bit masks are exactly 64-bit). */
        if (!PyLong_Check(am)) { Py_DECREF(am); goto vsh_fb; }
        (void)PyLong_AsUnsignedLongLong(am);
        if (PyErr_Occurred()) { PyErr_Clear(); Py_DECREF(am); goto vsh_fb; }
        if (width <= 0 || width > 64) { Py_DECREF(am); goto vsh_fb; }
        compile_expr(cc, e, sv);   /* stack: src_val, src_taint, amt_val, amt_taint */
        compile_expr(cc, e, st);
        compile_expr(cc, e, mv);
        compile_expr(cc, e, mt);
        Py_DECREF(sv); Py_DECREF(st); Py_DECREF(mv); Py_DECREF(mt);
        if (e->fallback) { Py_DECREF(am); return; }
        int amci = (int)PyList_GET_SIZE(cc->constants);
        PyList_Append(cc->constants, am);
        Py_DECREF(am);
        emit(e, OP_VAR_SHIFT);
        emit(e, (uint32_t)width);
        emit(e, (uint32_t)kind);
        emit(e, (uint32_t)amci);
        return;
    vsh_fb:
        Py_XDECREF(sv); Py_XDECREF(st); Py_XDECREF(mv); Py_XDECREF(mt);
        e->fallback = 1; return;
    }
    if (strcmp(cn, "InstructionCellExpr") == 0) {
        Py_DECREF(cls_name);
        emit_call_cell(cc, e, expr);
        return;
    }
    if (strcmp(cn, "MemoryOperand") == 0) {
        Py_DECREF(cls_name);
        PyObject *addr_e = PyObject_GetAttrString(expr, "address_expr");
        PyObject *sz     = PyObject_GetAttrString(expr, "size");
        PyObject *it     = PyObject_GetAttrString(expr, "is_taint");
        if (!addr_e || !sz || !it) {
            Py_XDECREF(addr_e); Py_XDECREF(sz); Py_XDECREF(it);
            e->fallback = 1; return;
        }
        int size_bytes = (int)PyLong_AsLong(sz);
        int is_taint = PyObject_IsTrue(it);
        Py_DECREF(sz); Py_DECREF(it);
        compile_expr(cc, e, addr_e);
        if (e->fallback) { Py_DECREF(addr_e); return; }
        record_mem_read(cc, addr_e, size_bytes);
        Py_DECREF(addr_e);
        e->n_mem++;
        emit(e, is_taint ? OP_PUSH_MEM_TAINT : OP_PUSH_MEM_VALUE);
        emit(e, (uint32_t)size_bytes);
        cc->has_mem_ops = 1;
        cc->has_mem_source = 1;   /* reads memory: see has_mem_source */
        return;
    }
    /* Unknown expression form — fall back to the Python evaluator */
    Py_DECREF(cls_name);
    e->fallback = 1;
}

/* ──────────── CompiledCircuit lifecycle ──────────── */

static PyTypeObject CompiledCircuitType;

static PyObject *CompiledCircuit_new(PyTypeObject *t, PyObject *a, PyObject *k) {
    (void)a; (void)k;
    CompiledCircuit *self = (CompiledCircuit *)t->tp_alloc(t, 0);
    if (!self) return NULL;
    self->python_circuit = NULL;
    self->cells = PyList_New(0);
    self->cell_handles = PyList_New(0);
    self->constants = PyList_New(0);
    self->string_pool = PyList_New(0);
    self->string_pool_dict = PyDict_New();
    self->arch_str = NULL;
    self->progs = NULL;
    self->n_progs = 0;
    self->pc_target_idx = -1;
    self->has_mem_source = 0;
    self->mem_reads = NULL;
    self->n_mem_reads = 0;
    self->mem_reads_ok = 1;
    self->has_python_fallback = 0;
    self->value_independent = 0;  /* set from the LogicCircuit in compile_circuit */
    self->c_mem_evaluable = 0;    /* set in compile_circuit; 0 => fall back to do_evaluate */
    self->scratch_t = NULL;
    self->scratch_v = NULL;
    self->scratch_o = NULL;
    self->scratch_cap = 0;
    self->pool_to_slot = NULL;
    self->pool_to_slot_n = 0;
    self->pool_map_src = NULL;
    self->pool_map_n = -1;
    self->out_slots = NULL;
    self->n_out_slots = 0;
    self->out_built = 0;
    self->out_ok = 0;
    return (PyObject *)self;
}

static void CompiledCircuit_dealloc(CompiledCircuit *self) {
    Py_XDECREF(self->python_circuit);
    Py_XDECREF(self->cells);
    Py_XDECREF(self->cell_handles);
    Py_XDECREF(self->constants);
    Py_XDECREF(self->string_pool);
    Py_XDECREF(self->string_pool_dict);
    Py_XDECREF(self->arch_str);
    if (self->progs) {
        for (int i = 0; i < self->n_progs; i++) {
            free(self->progs[i].bc);
            free(self->progs[i].addr_bc);
            Py_XDECREF(self->progs[i].python_assignment);
        }
        free(self->progs);
    }
    free(self->const_u64);
    free(self->cell_h);   /* borrowed pointers; the capsules belong to cell_handles */
    free(self->out_slots);
    if (self->mem_reads) {
        for (int i = 0; i < self->n_mem_reads; i++) free(self->mem_reads[i].bc);
        free(self->mem_reads);
    }
    free(self->scratch_t);
    free(self->scratch_v);
    free(self->scratch_o);
    free(self->pool_to_slot);
    Py_CLEAR(self->pool_map_src);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/* compile_circuit(logic_circuit) — entry point */
static PyObject *py_compile_circuit(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *circuit;
    PyObject *pcode_arg = NULL;
    if (!PyArg_ParseTuple(args, "O|O", &circuit, &pcode_arg)) return NULL;

    CompiledCircuit *cc = (CompiledCircuit *)CompiledCircuit_new(&CompiledCircuitType, NULL, NULL);
    if (!cc) return NULL;
    cc->python_circuit = circuit; Py_INCREF(circuit);

    /* Value-independence of the taint transfer is a property of the original
     * p-code ops, computed by the circuit builder and stored on the LogicCircuit
     * (absent -> 0/value-dependent, the safe default).  It is ANDed with the
     * C-evaluability gate below (a Python fallback could read values outside the
     * bytecode). */
    {
        PyObject *vi = PyObject_GetAttrString(circuit, "value_independent");
        cc->value_independent = (vi && PyObject_IsTrue(vi)) ? 1 : 0;
        Py_XDECREF(vi);
        PyErr_Clear();
    }

    /* Extract arch_str = str(circuit.architecture) for parent-register resolution */
    PyObject *arch_obj = PyObject_GetAttrString(circuit, "architecture");
    if (arch_obj) {
        cc->arch_str = PyObject_Str(arch_obj);
        Py_DECREF(arch_obj);
    }
    if (!cc->arch_str) {
        cc->arch_str = PyUnicode_FromString("AMD64");
    }

    PyObject *assignments = PyObject_GetAttrString(circuit, "assignments");
    if (!assignments || !PyList_Check(assignments)) {
        Py_XDECREF(assignments);
        Py_DECREF(cc);
        PyErr_SetString(PyExc_TypeError, "circuit.assignments must be a list");
        return NULL;
    }
    Py_ssize_t n = PyList_GET_SIZE(assignments);
    cc->n_progs = (int)n;
    cc->progs = (AssignmentProg *)calloc(n, sizeof(AssignmentProg));
    if (!cc->progs) { Py_DECREF(assignments); Py_DECREF(cc); PyErr_NoMemory(); return NULL; }

    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *a = PyList_GET_ITEM(assignments, i);
        AssignmentProg *p = &cc->progs[i];

        /* Target descriptor */
        PyObject *target = PyObject_GetAttrString(a, "target");
        PyObject *is_mem = PyObject_GetAttrString(a, "is_mem_target");
        if (!target || !is_mem) { Py_XDECREF(target); Py_XDECREF(is_mem); goto fb; }
        int is_mem_target = PyObject_IsTrue(is_mem);
        Py_DECREF(is_mem);
        /* t_name/t_bs/t_be kept at outer scope (NULL-init) so the `fb:` cleanup
         * is valid on either target branch; the memory branch leaves them NULL. */
        PyObject *t_name = NULL, *t_bs = NULL, *t_be = NULL;
        if (is_mem_target) {
            /* Memory target: compile the VALUE expr (shared block below) plus a
             * separate tiny bytecode for the ADDRESS expr, evaluated at run time
             * to form the MEM_0x<addr>_<size> output key.  Symmetric to the
             * memory INPUT handling in cell_c's cell_eval_fast. */
            PyObject *addr_e = PyObject_GetAttrString(target, "address_expr");
            PyObject *szo    = PyObject_GetAttrString(target, "size");
            Py_DECREF(target);
            if (!addr_e || !szo) { Py_XDECREF(addr_e); Py_XDECREF(szo); goto fb; }
            int sz_int = (int)PyLong_AsLong(szo);
            Py_DECREF(szo);
            /* >8-byte store cannot fit a uint64 taint mask.  The engine already
             * splits wide stores into <=8-byte chunks, so this is a defensive
             * guard; keep such a target on the Python path. */
            if (sz_int <= 0 || sz_int > 8) {
                Py_DECREF(addr_e);
                p->python_assignment = a; Py_INCREF(a);
                cc->has_python_fallback = 1; cc->has_mem_ops = 1;
                continue;
            }
            BCEmit addr_emit = {{0}, 0, 0, 0, 0, 0};
            compile_expr(cc, &addr_emit, addr_e);
            Py_DECREF(addr_e);
            if (addr_emit.fallback || addr_emit.overflow) {
                p->python_assignment = a; Py_INCREF(a);
                cc->has_python_fallback = 1; cc->has_mem_ops = 1;
                continue;
            }
            emit(&addr_emit, OP_END);
            p->addr_bc = (uint32_t *)malloc(sizeof(uint32_t) * addr_emit.len);
            if (!p->addr_bc) {
                p->python_assignment = a; Py_INCREF(a);
                cc->has_python_fallback = 1; cc->has_mem_ops = 1;
                continue;
            }
            memcpy(p->addr_bc, addr_emit.buf, sizeof(uint32_t) * addr_emit.len);
            p->addr_bc_len = addr_emit.len;
            p->target_kind = TGT_MEM_DYNAMIC;
            p->target_size_bytes = sz_int;
            p->target_bit_start = 0;
            p->target_bit_end = sz_int * 8 - 1;
            p->target_name_idx = -1;  /* built at run time from the address */
            cc->has_mem_ops = 1;
            /* fall through to the shared value-expr compile */
        } else {
            /* Register target. */
            t_name = PyObject_GetAttrString(target, "name");
            t_bs   = PyObject_GetAttrString(target, "bit_start");
            t_be   = PyObject_GetAttrString(target, "bit_end");
            Py_DECREF(target);
            if (!t_name || !t_bs || !t_be) { Py_XDECREF(t_name); t_name = NULL; Py_XDECREF(t_bs); t_bs = NULL; Py_XDECREF(t_be); t_be = NULL; goto fb; }
            const char *tn = PyUnicode_AsUTF8(t_name);
            if (!tn) { Py_DECREF(t_name); t_name = NULL; Py_DECREF(t_bs); t_bs = NULL; Py_DECREF(t_be); t_be = NULL; goto fb; }
            p->target_kind = TGT_REG;
            p->target_name_idx = strpool_intern(cc, tn);
            p->target_bit_start = (int)PyLong_AsLong(t_bs);
            p->target_bit_end   = (int)PyLong_AsLong(t_be);
            p->target_size_bytes = 0;
            Py_DECREF(t_name); t_name = NULL;
            Py_DECREF(t_bs);   t_bs   = NULL;
            Py_DECREF(t_be);   t_be   = NULL;
        }

        /* Sanity gate: a wide (>64-bit) output cannot fit the uint64 bytecode
         * stack / result mask.  Route the whole assignment to the Python
         * evaluator (arbitrary width) rather than mangle it. */
        if (p->target_bit_end - p->target_bit_start + 1 > 64) {
            p->python_assignment = a; Py_INCREF(a);
            cc->has_python_fallback = 1;
            continue;
        }

        /* Compile the rhs expression */
        PyObject *expr = PyObject_GetAttrString(a, "expression");
        if (!expr || expr == Py_None) {
            Py_XDECREF(expr);
            /* Dependencies fallback — uncommon, keep Python.
             * has_mem_ops contract: walk the *assignment*'s expression
             * (even if None we also check dependencies through it via
             * the python_assignment path at runtime).  The expr is None
             * here so we can only conservatively trust the existing
             * has_mem_ops; the deps will be evaluated in Python and
             * could reference memory, so re-check via target/deps. */
            p->python_assignment = a; Py_INCREF(a);
            cc->has_python_fallback = 1;
            /* Pessimistic: dependency-only fallback assignments may read
             * memory through MemoryOperand deps even with expression=None. */
            PyObject *deps = PyObject_GetAttrString(a, "dependencies");
            if (deps) {
                if (PyList_Check(deps) || PyTuple_Check(deps)) {
                    Py_ssize_t n = PySequence_Size(deps);
                    for (Py_ssize_t i = 0; i < n; i++) {
                        PyObject *d = PySequence_GetItem(deps, i);
                        if (d) {
                            if (expr_tree_reads_memory(d, 0)) {
                                cc->has_mem_ops = 1;
                            }
                            Py_DECREF(d);
                        }
                    }
                }
                Py_DECREF(deps);
            } else {
                PyErr_Clear();
            }
            continue;
        }
        BCEmit emit_buf = {{0}, 0, 0, 0};
        compile_expr(cc, &emit_buf, expr);
        if (emit_buf.fallback || emit_buf.overflow) {
            /* Bytecode compile failed.  Conservatively check whether the
             * *Python* fallback evaluation will read shadow memory.  If it
             * will, `has_mem_ops` must reflect that, otherwise the
             * wrapper's per-instruction cache will replay stale taint. */
            if (expr_tree_reads_memory(expr, 0)) {
                cc->has_mem_ops = 1;
            }
            Py_DECREF(expr);
            p->python_assignment = a; Py_INCREF(a);
            cc->has_python_fallback = 1;
            continue;
        }
        Py_DECREF(expr);
        emit(&emit_buf, OP_END);
        p->bc_len = emit_buf.len;
        p->bc = (uint32_t *)malloc(sizeof(uint32_t) * emit_buf.len);
        if (!p->bc) goto fb;
        memcpy(p->bc, emit_buf.buf, sizeof(uint32_t) * emit_buf.len);
        continue;

    fb:
        Py_XDECREF(t_name); Py_XDECREF(t_bs); Py_XDECREF(t_be);
        p->python_assignment = a; Py_INCREF(a);
        cc->has_python_fallback = 1;
    }

    /* Cache PC target */
    PyObject *pc = PyObject_GetAttrString(circuit, "_pc_target");
    if (pc && pc != Py_None) {
        const char *pcn = PyUnicode_AsUTF8(pc);
        if (pcn) cc->pc_target_idx = strpool_intern(cc, pcn);
    }
    Py_XDECREF(pc);

    /* Build cell handles via pcode.make_cell_handle if supported (only on
     * the C kernel).  These pre-resolve out_reg / out_bit_start/end and
     * input register offsets so the OP_CALL_CELL hot path skips all
     * GetAttr calls and dict lookups. */
    if (pcode_arg && pcode_arg != Py_None) {
        PyObject *make_fn = PyObject_GetAttrString(pcode_arg, "make_cell_handle");
        if (make_fn && PyCallable_Check(make_fn)) {
            Py_ssize_t n_cells = PyList_GET_SIZE(cc->cells);
            for (Py_ssize_t i = 0; i < n_cells; i++) {
                PyObject *cell = PyList_GET_ITEM(cc->cells, i);
                /* Get the input names list from the cell's inputs dict, in
                 * the same order they appear in bytecode.  The bytecode
                 * emitter walks PyDict_Next, which preserves insertion
                 * order in CPython >= 3.7. */
                PyObject *inputs = PyObject_GetAttrString(cell, "inputs");
                if (!inputs) { PyErr_Clear(); PyList_Append(cc->cell_handles, Py_None); continue; }
                PyObject *names_list = PyList_New(0);
                PyObject *k, *v;
                Py_ssize_t pos = 0;
                while (PyDict_Next(inputs, &pos, &k, &v)) {
                    PyList_Append(names_list, k);
                }
                Py_DECREF(inputs);
                PyObject *handle = PyObject_CallFunctionObjArgs(make_fn, cell, names_list, NULL);
                Py_DECREF(names_list);
                if (!handle) {
                    PyErr_Clear();
                    Py_INCREF(Py_None);
                    PyList_Append(cc->cell_handles, Py_None);
                } else {
                    PyList_Append(cc->cell_handles, handle);
                    Py_DECREF(handle);
                }
            }
            Py_DECREF(make_fn);
        } else {
            PyErr_Clear();
            Py_XDECREF(make_fn);
        }
    }

    Py_DECREF(assignments);

    /* Precompute the GIL-free do_evaluate_c data: a uint64 mirror of `constants`
     * and the c_evaluable gate.  A constant wider than 64 bits (or any python
     * fallback / memory op) disqualifies the whole circuit from the C path. */
    {
        int const_ok = 1;
        Py_ssize_t nc = PyList_GET_SIZE(cc->constants);
        if (nc > 0) {
            cc->const_u64 = (uint64_t *)calloc((size_t)nc, sizeof(uint64_t));
            if (!cc->const_u64) { const_ok = 0; }
            else {
                cc->n_const_u64 = (int)nc;
                for (Py_ssize_t ci = 0; ci < nc; ci++) {
                    uint64_t uv;
                    if (!pylong_to_u64(PyList_GET_ITEM(cc->constants, ci), &uv)) { const_ok = 0; break; }
                    cc->const_u64[ci] = uv;
                }
            }
        }
        cc->c_evaluable = const_ok && !cc->has_python_fallback && !cc->has_mem_ops;
        cc->c_mem_evaluable = const_ok && !cc->has_python_fallback;
    }

    /* Resolve the cell capsules once (see cell_h in the struct).  Slots that are
     * not usable capsules stay NULL and OP_CALL_CELL falls back exactly as it
     * did when it tested them inline. */
    if (cc->cell_handles && PyList_Check(cc->cell_handles)) {
        Py_ssize_t nh = PyList_GET_SIZE(cc->cell_handles);
        if (nh > 0) {
            cc->cell_h = (CellHandle_API **)calloc((size_t)nh, sizeof(CellHandle_API *));
            if (cc->cell_h) {
                cc->n_cell_h = (int)nh;
                for (Py_ssize_t hi = 0; hi < nh; hi++) {
                    PyObject *cap = PyList_GET_ITEM(cc->cell_handles, hi);
                    if (cap && cap != Py_None && PyCapsule_CheckExact(cap)) {
                        cc->cell_h[hi] = (CellHandle_API *)PyCapsule_GetPointer(cap, "CellHandle");
                        if (!cc->cell_h[hi]) PyErr_Clear();
                    }
                }
            }
        }
    }
    /* Airtight value-independence: only trust it when the whole circuit is
     * C-evaluated (no Python fallback that could read operand values outside
     * the OP_PUSH_VALUE opcodes) and reads no memory. */
    cc->value_independent = cc->value_independent
                            && !cc->has_python_fallback && !cc->has_mem_ops;
    return (PyObject *)cc;
}

/* ──────────── Bytecode evaluator ──────────── */

/* uint64 stack with Python-int spill for values that exceed 64 bits.
 * Since we only compile assignments where AvalancheExpr.size_bits <= 64,
 * and BinaryExpr.OR over multiple uint64 stays within uint64, we don't
 * need spill in the compiled set.  Pure uint64 stack. */

static uint64_t mask_range(int width) {
    if (width >= 64) return UINT64_MAX;
    return (((uint64_t)1) << width) - 1;
}

/* OF as a function of (a_s, b_s, borrow|carry-into-msb) -- mirrors
 * SignedOverflowTaintExpr._g. x,y,z are 0/1. */
static inline int sof_g(int is_sub, int x, int y, int z) {
    return is_sub ? ((x ^ y) & (y ^ z)) : ((1 - (x ^ y)) & (y ^ z));
}

/* ---- VariableMultiplyTaintExpr helpers (128-bit product for w up to 64) ---- */
/* Lowest bit that can be 1 (untainted-set or tainted); w if none. w<=64. */
static inline int vmul_tz_lo(uint64_t v, uint64_t t, int w) {
    uint64_t poss = (v & ~t) | t;
    if (w < 64) poss &= (((uint64_t)1 << w) - 1);
    if (poss == 0) return w;
    return __builtin_ctzll(poss);
}
static inline unsigned __int128 mask128(int n) {
    if (n <= 0) return 0;
    if (n >= 128) return ~(unsigned __int128)0;
    return ((unsigned __int128)1 << n) - 1;
}
static inline int bitlen128(unsigned __int128 x) {
    if (x == 0) return 0;
    uint64_t hi = (uint64_t)(x >> 64);
    if (hi) return 128 - __builtin_clzll(hi);
    return 64 - __builtin_clzll((uint64_t)x);
}
/* Interpret low w bits of v (v<2^w) as two's-complement, as a signed __int128. */
static inline __int128 vmul_sx(uint64_t v, int w) {
    if ((v >> (w - 1)) & 1) return (__int128)v - ((__int128)1 << w);
    return (__int128)v;
}

/* ---- VariableShiftTaintExpr helpers (mirror _shift / _smear) ---- */
/* kind: 0=left, 1=logical right, 2=arithmetic right. w<=64. */
static inline uint64_t vsh_one(int kind, uint64_t y, int s, uint64_t mask, int w) {
    if (s >= w) {
        if (kind == 2) return ((y >> (w - 1)) & 1) ? mask : 0;
        return 0;
    }
    if (kind == 0) return (y << s) & mask;
    if (kind == 1) return (y & mask) >> s;
    uint64_t r = (y & mask) >> s;                 /* arithmetic right */
    if (((y >> (w - 1)) & 1) && s > 0) r |= (mask << (w - s)) & mask;
    return r;
}
static inline uint64_t vsh_smear(int kind, uint64_t y, uint64_t ts, int s0,
                                 int is_and, uint64_t mask, int w, int lg) {
    uint64_t r = vsh_one(kind, y, s0, mask, w);
    for (int j = 0; j < lg; j++) {
        if ((ts >> j) & 1) {
            uint64_t shifted = vsh_one(kind, r, 1 << j, mask, w);
            r = is_and ? (r & shifted) : (r | shifted);
        }
    }
    return r;
}

/* Per-evaluate memo of resolved input operand values, indexed by string_pool
 * index.  OP_PUSH_TAINT / OP_PUSH_VALUE look up input_taint / input_values (a
 * Python dict) and convert to uint64 per operand OCCURRENCE; a large flag
 * differential references the same register many times.  This caches the RAW
 * per-name value on first use so repeats read an array slot (the per-op bit
 * slice/mask is still applied).  Lazy: an unused name costs nothing; the first
 * use costs exactly the old lookup, so this is never slower than the dict path.
 * A >64-bit taint value is never memoed (it still bails to the Python evaluator,
 * unchanged).  memo->n == 0 disables it (pool too large / not set up). */
#define PUSH_MEMO_CAP 512
typedef struct {
    uint64_t *tv;   /* raw input_taint[name]  when rt[i] */
    uint64_t *vv;   /* raw input_values[name] when rv[i] */
    uint8_t  *rt;   /* taint slot resolved */
    uint8_t  *rv;   /* value slot resolved */
    int       n;    /* string_pool length covered (0 => disabled) */
} PushMemo;

/* Evaluate a single bytecode program for one assignment.
 * Returns the result as Python int (PyLong) or NULL on error. */
/* When c_out != NULL, eval_program runs in GIL-FREE C mode: OP_PUSH_TAINT/VALUE
 * read the caller's uint64 arrays (indexed by string_pool idx) instead of the
 * Python input dicts, OP_PUSH_CONST reads cc->const_u64, OP_END writes *c_out
 * and returns a non-NULL sentinel, and any op needing Python (OP_PUSH_MEM_*, an
 * OP_CALL_CELL that cannot take the C fast path) returns NULL to bail.  With
 * c_out == NULL the function is byte-identical to before (the Python path). */
static PyObject *eval_program(CompiledCircuit *cc,
                              uint32_t *bc, int bc_len,
                              PyObject *context,
                              PyObject *input_taint, PyObject *input_values,
                              PyObject *shadow_memory, PyObject *mem_reader,
                              PyObject *pcode_eval, PushMemo *memo,
                              const uint64_t *c_in_taint, const uint64_t *c_in_values,
                              uint64_t *c_out) {
    uint64_t stack[CIRCUIT_STACK_MAX];
    int sp = 0;
    int pc = 0;

    while (pc < bc_len) {
        uint32_t op = bc[pc++];
        switch (op) {
        case OP_PUSH_TAINT: {
            int name_idx  = (int)bc[pc++];
            int bit_start = (int)bc[pc++];
            int bit_end   = (int)bc[pc++];
            uint64_t v;
            if (c_out) {
                v = c_in_taint[name_idx];   /* C mode: array read, no PyObject */
            } else if (memo && name_idx < memo->n && memo->rt[name_idx]) {
                v = memo->tv[name_idx];   /* memo hit: raw value, no dict lookup */
            } else {
                PyObject *name = PyList_GET_ITEM(cc->string_pool, name_idx);
                PyObject *val = PyDict_GetItem(input_taint, name);
                v = 0;
                if (val) {
                    v = (uint64_t)PyLong_AsUnsignedLongLong(val);
                    if (PyErr_Occurred()) {
                        PyErr_Clear();
                        /* Value too big for u64; bail to the Python evaluator.
                         * Not memoed, so behaviour is unchanged. */
                        return Py_BuildValue("");  /* sentinel; handled below */
                    }
                }
                if (memo && name_idx < memo->n) { memo->tv[name_idx] = v; memo->rt[name_idx] = 1; }
            }
            int width = bit_end - bit_start + 1;
            uint64_t m = mask_range(width);
            stack[sp++] = (v >> bit_start) & m;
            break;
        }
        case OP_PUSH_VALUE: {
            int name_idx  = (int)bc[pc++];
            int bit_start = (int)bc[pc++];
            int bit_end   = (int)bc[pc++];
            uint64_t v;
            if (c_out) {
                v = c_in_values[name_idx];   /* C mode: array read, no PyObject */
            } else if (memo && name_idx < memo->n && memo->rv[name_idx]) {
                v = memo->vv[name_idx];
            } else {
                PyObject *name = PyList_GET_ITEM(cc->string_pool, name_idx);
                PyObject *val = PyDict_GetItem(input_values, name);
                v = 0;
                if (val) {
                    v = (uint64_t)PyLong_AsUnsignedLongLong(val);
                    if (PyErr_Occurred()) PyErr_Clear();
                }
                if (memo && name_idx < memo->n) { memo->vv[name_idx] = v; memo->rv[name_idx] = 1; }
            }
            int width = bit_end - bit_start + 1;
            uint64_t m = mask_range(width);
            stack[sp++] = (v >> bit_start) & m;
            break;
        }
        case OP_PUSH_CONST: {
            int ci = (int)bc[pc++];
            if (c_out) {
                stack[sp++] = (ci >= 0 && ci < cc->n_const_u64) ? cc->const_u64[ci] : 0;
                break;
            }
            PyObject *cv = PyList_GET_ITEM(cc->constants, ci);
            /* Mask variant: two's-complement low-64 bits, so a negative constant
             * (e.g. -8 -> 0xFFFFFFFFFFFFFFF8) loads correctly and never raises.
             * Validated to fit 64 bits at compile time (pylong_to_u64). */
            uint64_t v = (uint64_t)PyLong_AsUnsignedLongLongMask(cv);
            if (PyErr_Occurred()) PyErr_Clear();
            stack[sp++] = v;
            break;
        }
        case OP_AND:  if (sp < 2) goto err; sp--; stack[sp-1] &=  stack[sp]; break;
        case OP_OR:   if (sp < 2) goto err; sp--; stack[sp-1] |=  stack[sp]; break;
        case OP_XOR:  if (sp < 2) goto err; sp--; stack[sp-1] ^=  stack[sp]; break;
        case OP_ADD:  if (sp < 2) goto err; sp--; stack[sp-1] +=  stack[sp]; break;
        case OP_SUB:  if (sp < 2) goto err; sp--; stack[sp-1] -=  stack[sp]; break;
        case OP_SHL: {
            if (sp < 2) goto err;
            sp--;
            uint64_t shift = stack[sp] & 63;
            stack[sp-1] = stack[sp-1] << shift;
            break;
        }
        case OP_SHR: {
            if (sp < 2) goto err;
            sp--;
            uint64_t shift = stack[sp] & 63;
            stack[sp-1] = stack[sp-1] >> shift;
            break;
        }
        case OP_NOT:  if (sp < 1) goto err; stack[sp-1] = ~stack[sp-1]; break;
        case OP_AVALANCHE: {
            int size_bits = (int)bc[pc++];
            if (sp < 1) goto err;
            uint64_t v = stack[sp-1];
            stack[sp-1] = (v != 0) ? mask_range(size_bits) : 0;
            break;
        }
        case OP_FULLMASK_AVAL: {
            int ci = (int)bc[pc++];
            if (sp < 1) goto err;
            PyObject *cv = PyList_GET_ITEM(cc->constants, ci);
            uint64_t fm = (uint64_t)PyLong_AsUnsignedLongLong(cv);
            if (PyErr_Occurred()) PyErr_Clear();
            uint64_t v = stack[sp-1];
            stack[sp-1] = (v == fm && v != 0) ? 1 : 0;
            break;
        }
        case OP_EQ_TAINT: {
            int w = (int)bc[pc++];
            if (sp < 4) goto err;
            uint64_t tb = stack[--sp], b = stack[--sp], ta = stack[--sp], a = stack[--sp];
            uint64_t mask = mask_range(w);
            a &= mask; ta &= mask; b &= mask; tb &= mask;
            uint64_t free = ta | tb;
            int equal_ach   = (((a ^ b) & ~free & mask) == 0);
            int unequal_ach = (free != 0);
            stack[sp++] = (equal_ach && unequal_ach) ? 1 : 0;
            break;
        }
        case OP_VAR_SHIFT: {
            int w = (int)bc[pc++];
            int kind = (int)bc[pc++];
            int amci = (int)bc[pc++];
            if (sp < 4) goto err;
            uint64_t tamt = stack[--sp], vamt = stack[--sp], tx = stack[--sp], x = stack[--sp];
            uint64_t mask = mask_range(w);
            PyObject *amcv = PyList_GET_ITEM(cc->constants, amci);
            uint64_t amt_mask = (uint64_t)PyLong_AsUnsignedLongLong(amcv);
            if (PyErr_Occurred()) PyErr_Clear();
            int lg = 0; { unsigned int y = (unsigned int)(w - 1); while (y) { lg++; y >>= 1; } }
            x &= mask; tx &= mask;
            uint64_t ts = tamt & amt_mask;
            uint64_t s0_full = (vamt & amt_mask) & ~ts;
            int s0 = (s0_full < (uint64_t)w) ? (int)s0_full : w;
            uint64_t ts_lo = ts & mask_range(lg);
            uint64_t reach = vsh_smear(kind, tx, ts_lo, s0, 0, mask, w, lg);
            uint64_t hi = vsh_smear(kind, x, ts_lo, s0, 0, mask, w, lg);
            uint64_t lo = vsh_smear(kind, x, ts_lo, s0, 1, mask, w, lg);
            if ((((uint64_t)s0) | ts) >= (uint64_t)w) {   /* saturating amounts reachable */
                if (kind == 2) {
                    if ((tx >> (w - 1)) & 1) { hi |= mask; lo = 0; reach |= mask; }
                    else if ((x >> (w - 1)) & 1) { hi |= mask; lo &= mask; }
                    else { lo = 0; }
                } else {
                    lo = 0;
                }
            }
            stack[sp++] = (reach | (hi ^ lo)) & mask;
            break;
        }
        case OP_VAR_MUL_TAINT: {
            int w      = (int)bc[pc++];
            int flags  = (int)bc[pc++];
            int out_lo = (int)bc[pc++];
            int out_hi = (int)bc[pc++];
            int is_signed = flags & 1;
            if (sp < 4) goto err;
            uint64_t tb = stack[--sp], vb = stack[--sp], ta = stack[--sp], va = stack[--sp];
            uint64_t mask = mask_range(w);
            va &= mask; ta &= mask; vb &= mask; tb &= mask;
            int lo = vmul_tz_lo(va, ta, w) + vmul_tz_lo(vb, tb, w);
            int fbits = 2 * w;
            unsigned __int128 fm = mask128(fbits);
            int signed_hi = is_signed && (out_hi > w);
            int hi;
            if (!signed_hi) {
                unsigned __int128 pmin = (unsigned __int128)(va & ~ta) * (unsigned __int128)(vb & ~tb);
                unsigned __int128 pmax = (unsigned __int128)(va | ta) * (unsigned __int128)(vb | tb);
                unsigned __int128 d = (pmin ^ pmax) & fm;
                hi = d ? bitlen128(d) - 1 : -1;
            } else if (((ta >> (w - 1)) & 1) || ((tb >> (w - 1)) & 1)) {
                hi = 2 * w - 1;
            } else {
                __int128 alo = vmul_sx(va & ~ta, w), ahi = vmul_sx(va | ta, w);
                __int128 blo = vmul_sx(vb & ~tb, w), bhi = vmul_sx(vb | tb, w);
                __int128 p[4] = { alo * blo, alo * bhi, ahi * blo, ahi * bhi };
                __int128 mn = p[0], mx = p[0];
                for (int i = 1; i < 4; i++) { if (p[i] < mn) mn = p[i]; if (p[i] > mx) mx = p[i]; }
                unsigned __int128 pmin = (unsigned __int128)mn & fm;
                unsigned __int128 pmax = (unsigned __int128)mx & fm;
                unsigned __int128 d = (pmin ^ pmax) & fm;
                hi = d ? bitlen128(d) - 1 : -1;
            }
            unsigned __int128 full = 0;
            if (hi >= lo) full = mask128(hi + 1) ^ mask128(lo);
            int ow = out_hi - out_lo;
            unsigned __int128 windowed = (full >> out_lo) & mask128(ow);
            stack[sp++] = (uint64_t)windowed;
            break;
        }
        case OP_SIGNED_OVF: {
            int w = (int)bc[pc++];
            int flags = (int)bc[pc++];
            int is_sub = flags & 1, has_c = (flags & 2) ? 1 : 0;
            int need = has_c ? 6 : 4;
            if (sp < need) goto err;
            uint64_t c = 0, tc = 0;
            if (has_c) { tc = stack[--sp]; c = stack[--sp]; }
            uint64_t tb = stack[--sp], b = stack[--sp], ta = stack[--sp], a = stack[--sp];
            uint64_t mask = mask_range(w), lowmask = mask_range(w - 1);
            a &= mask; ta &= mask; b &= mask; tb &= mask; c &= lowmask; tc &= lowmask;
            int a_s = (int)((a >> (w - 1)) & 1), b_s = (int)((b >> (w - 1)) & 1);
            int ta_s = (int)((ta >> (w - 1)) & 1), tb_s = (int)((tb >> (w - 1)) & 1);
            uint64_t al = a & lowmask, bl = b & lowmask, tal = ta & lowmask, tbl = tb & lowmask;
            int base_c, hi, lo;
            if (is_sub) {
                base_c = (al < bl + c);
                hi = ((al & ~tal & lowmask) < (bl | tbl) + (c | tc));
                lo = ((al | tal) < (bl & ~tbl & lowmask) + (c & ~tc & lowmask));
            } else {
                base_c = ((al + bl + c) > lowmask);
                hi = (((al | tal) + (bl | tbl) + (c | tc)) > lowmask);
                lo = (((al & ~tal & lowmask) + (bl & ~tbl & lowmask) + (c & ~tc & lowmask)) > lowmask);
            }
            int t_c = hi ^ lo;
            int base = sof_g(is_sub, a_s, b_s, base_c);
            int result = 0;
            for (int da = 0; da < (ta_s ? 2 : 1) && !result; da++)
                for (int db = 0; db < (tb_s ? 2 : 1) && !result; db++)
                    for (int dc = 0; dc < (t_c ? 2 : 1) && !result; dc++)
                        if (sof_g(is_sub, a_s ^ da, b_s ^ db, base_c ^ dc) != base) result = 1;
            stack[sp++] = (uint64_t)result;
            break;
        }
        case OP_CMP_TAINT: {
            int w = (int)bc[pc++];
            int flags = (int)bc[pc++];
            if (sp < 4) goto err;
            uint64_t tb = stack[--sp], b = stack[--sp], ta = stack[--sp], a = stack[--sp];
            uint64_t mask = mask_range(w);
            a &= mask; ta &= mask; b &= mask; tb &= mask;
            if (flags & 1) {   /* signed: flip sign bit into the unsigned domain */
                uint64_t sb = (uint64_t)1 << (w - 1);
                a ^= sb; b ^= sb;
            }
            uint64_t amin = a & ~ta & mask, amax = a | ta;
            uint64_t bmin = b & ~tb & mask, bmax = b | tb;
            int can_true, always_true;
            if (flags & 2) { can_true = (amin <= bmax); always_true = (amax <= bmin); }
            else           { can_true = (amin <  bmax); always_true = (amax <  bmin); }
            stack[sp++] = (uint64_t)(can_true ^ always_true);
            break;
        }
        case OP_VAR_BIT_SELECT: {
            int w = (int)bc[pc++];
            if (sp < 4) goto err;
            uint64_t tb = stack[--sp], b = stack[--sp], ta = stack[--sp], a = stack[--sp];
            uint64_t mask = mask_range(w);
            a &= mask; ta &= mask; b &= mask; tb &= mask;
            int nbits = 0; { unsigned int x = (unsigned int)(w - 1); while (x) { nbits++; x >>= 1; } }
            uint64_t low = mask_range(nbits);   /* nbits==0 -> 0 */
            uint64_t t_idx = tb & low;
            uint64_t fixed = b & ~t_idx & low;
            int seen0 = 0, seen1 = 0, result = 0;
            for (int i = 0; i < w; i++) {
                if (((uint64_t)i & ~t_idx & low) != fixed) continue;
                if ((ta >> i) & 1) { result = 1; break; }
                if ((a >> i) & 1) seen1 = 1; else seen0 = 1;
                if (seen0 && seen1) { result = 1; break; }
            }
            stack[sp++] = (uint64_t)result;
            break;
        }
        case OP_PUSH_MEM_TAINT: {
            int size = (int)bc[pc++];
            if (sp < 1) goto err;
            uint64_t addr = stack[--sp];
            uint64_t v = 0;
            if (c_out) {
                /* C-array mode: read shadow taint at the C level via the shadow
                 * capsule (the caller holds the GIL).  Requires a live shadow +
                 * the capsule; otherwise bail so the caller drops to the Python
                 * evaluator.  This is the hot LOAD path — no PyObject churn. */
                if (shadow_memory == NULL || shadow_memory == Py_None || !g_shadow_capi) goto err;
                stack[sp++] = g_shadow_capi->read_mask(shadow_memory, addr, size);
                break;
            }
            if (shadow_memory != Py_None && shadow_memory != NULL) {
                PyObject *addr_obj = PyLong_FromUnsignedLongLong(addr);
                PyObject *sz_obj   = PyLong_FromLong(size);
                PyObject *r = PyObject_CallMethod(shadow_memory, "read_mask", "OO", addr_obj, sz_obj);
                Py_DECREF(addr_obj); Py_DECREF(sz_obj);
                if (!r) return NULL;
                v = (uint64_t)PyLong_AsUnsignedLongLong(r);
                if (PyErr_Occurred()) PyErr_Clear();
                Py_DECREF(r);
            } else {
                /* Dict fallback: state.get(f'MEM_{hex(addr)}_{size}', 0) */
                char mem_key[64];
                snprintf(mem_key, sizeof(mem_key), "MEM_0x%llx_%d", (unsigned long long)addr, size);
                PyObject *k = PyUnicode_FromString(mem_key);
                PyObject *val = PyDict_GetItem(input_taint, k);
                Py_DECREF(k);
                if (val) {
                    v = (uint64_t)PyLong_AsUnsignedLongLong(val);
                    if (PyErr_Occurred()) PyErr_Clear();
                }
            }
            stack[sp++] = v;
            break;
        }
        case OP_PUSH_MEM_VALUE: {
            int size = (int)bc[pc++];
            if (sp < 1) goto err;
            uint64_t addr = stack[--sp];
            uint64_t v = 0;
            /* C-array mode reads the concrete value through mem_reader (a Python
             * callable, valid because the caller holds the GIL); the dict
             * fallback needs input_values, which C mode does not supply, so bail
             * if there is no reader. */
            if (c_out && (mem_reader == NULL || mem_reader == Py_None)
                      && g_mem_fn == NULL) goto err;
            /* C reader first: no PyLong for the address, no Python call, no
             * PyLong for the result.  Falls through to the Python reader for any
             * access it declines (short read, unmapped, width > 8). */
            if (g_mem_fn != NULL && g_mem_fn(g_mem_ctx, addr, size, &v) == 0) {
                /* value already in v */
            } else if (mem_reader != Py_None && mem_reader != NULL) {
                PyObject *addr_obj = PyLong_FromUnsignedLongLong(addr);
                PyObject *sz_obj   = PyLong_FromLong(size);
                PyObject *r = PyObject_CallFunctionObjArgs(mem_reader, addr_obj, sz_obj, NULL);
                Py_DECREF(addr_obj); Py_DECREF(sz_obj);
                if (!r) return NULL;
                v = (uint64_t)PyLong_AsUnsignedLongLong(r);
                if (PyErr_Occurred()) PyErr_Clear();
                Py_DECREF(r);
            } else {
                char mem_key[64];
                snprintf(mem_key, sizeof(mem_key), "MEM_0x%llx_%d", (unsigned long long)addr, size);
                PyObject *k = PyUnicode_FromString(mem_key);
                PyObject *val = PyDict_GetItem(input_values, k);
                Py_DECREF(k);
                if (val) {
                    v = (uint64_t)PyLong_AsUnsignedLongLong(val);
                    if (PyErr_Occurred()) PyErr_Clear();
                }
            }
            stack[sp++] = v;
            break;
        }
        case OP_CALL_CELL: {
            int cell_idx = (int)bc[pc++];
            int n_inputs = (int)bc[pc++];
            if (sp < n_inputs) goto err;

            /* Fast path: if cell_handles[cell_idx] is a CellHandle capsule
             * AND we have the CellCAPI loaded AND pcode is the C kernel,
             * call cell_eval_fast directly with no Python boundary. */
            int fast_path_taken = 0;
            uint64_t fast_v = 0;
            /* cell_h was resolved at compile time, so reaching the kernel costs
             * one bounds check and one load rather than four Python API calls. */
            CellHandle_API *h = (cell_idx >= 0 && cell_idx < cc->n_cell_h)
                                ? cc->cell_h[cell_idx] : NULL;
            if (g_cell_capi && h) {
                /* Stack input slots: stack[sp - n_inputs .. sp - 1] */
                uint64_t inp_vals[16];
                for (int i = 0; i < n_inputs && i < 16; i++) {
                    inp_vals[i] = stack[sp - n_inputs + i];
                }
                int rc = g_cell_capi->cell_eval_fast(
                    (EvalC_API *)pcode_eval, h, inp_vals, &fast_v);
                if (rc == 0) {
                    fast_path_taken = 1;
                } else if (rc == 1) {
                    /* Fallback needed — raise PCodeFallbackNeeded
                     * just like Python path would. */
                    PyObject *exc = g_cell_capi->get_fallback_exc((EvalC_API *)pcode_eval);
                    PyErr_SetString(exc, "instruction requires Unicorn");
                    return NULL;
                }
                /* rc < 0: hard error, fall through to Python path */
            }

            if (fast_path_taken) {
                pc += n_inputs;       /* skip per-input name idxs */
                sp -= n_inputs;
                stack[sp++] = fast_v;
                break;
            }

            if (c_out) goto err;   /* C mode: the Python slow cell path needs the GIL; bail */
            /* Slow path: build a Python dict and call evaluate_concrete.
             * Used for non-C-kernel pcode (Cython fallback) or when a
             * handle wasn't pre-resolved. */
            PyObject *cell = PyList_GET_ITEM(cc->cells, cell_idx);
            PyObject *inputs_dict = PyDict_New();
            if (!inputs_dict) return NULL;
            for (int i = 0; i < n_inputs; i++) {
                int name_idx = (int)bc[pc + i];
                PyObject *name = PyList_GET_ITEM(cc->string_pool, name_idx);
                uint64_t v = stack[sp - n_inputs + i];
                PyObject *vobj = PyLong_FromUnsignedLongLong(v);
                PyDict_SetItem(inputs_dict, name, vobj);
                Py_DECREF(vobj);
            }
            pc += n_inputs;
            sp -= n_inputs;

            PyObject *r = PyObject_CallMethod(pcode_eval, "evaluate_concrete",
                                              "OO", cell, inputs_dict);
            Py_DECREF(inputs_dict);
            if (!r) {
                return NULL;
            }
            uint64_t v2 = (uint64_t)PyLong_AsUnsignedLongLong(r);
            Py_DECREF(r);
            if (PyErr_Occurred()) PyErr_Clear();
            stack[sp++] = v2;
            break;
        }
        case OP_END: {
            if (sp != 1) goto err;
            if (c_out) { *c_out = stack[0]; return (PyObject *)cc; /* non-NULL C-mode sentinel */ }
            return PyLong_FromUnsignedLongLong(stack[0]);
        }
        default:
            goto err;
        }
        if (sp >= CIRCUIT_STACK_MAX) goto err;
    }
err:
    PyErr_SetString(PyExc_RuntimeError, "circuit bytecode overflow / corrupt");
    return NULL;
}

/* Normalize a register dict: promote child register names (e.g., AL) to
 * their canonical parents (e.g., EAX), shifting the value into position.
 * Mirrors microtaint.instrumentation.ast._normalize_register_dict.
 *
 * Always returns a NEW dict (caller owns).  Even in the no-op case,
 * we copy — this keeps refcount handling unambiguous in the caller. */
static PyObject *normalize_register_dict(PyObject *arch_str, PyObject *input_dict) {
    if (!input_dict || !PyDict_Check(input_dict))
        return NULL;

    static PyObject *parent_regs_dict = NULL;
    if (!parent_regs_dict) {
        PyObject *mod = PyImport_ImportModule("microtaint.instrumentation.ast");
        if (!mod) return PyDict_Copy(input_dict);
        parent_regs_dict = PyObject_GetAttrString(mod, "_ARCH_PARENT_REGS");
        Py_DECREF(mod);
        if (!parent_regs_dict) return PyDict_Copy(input_dict);
    }
    PyObject *arch_map = PyDict_GetItem(parent_regs_dict, arch_str);
    if (!arch_map) return PyDict_Copy(input_dict);

    /* Hot-path fast check: if no key is a known child, no promotion is needed,
     * so return a borrowed ref (INCREF'd -> still an owned ref, per the "always
     * fresh refs" contract) instead of a full copy.  Both callers (do_evaluate,
     * evaluate_c) treat the normalized dict as READ-ONLY -- do_evaluate copies it
     * into output_taint before mutating, evaluate_c only reads it and copies for
     * output -- so sharing the input dict is safe and saves a per-instruction
     * dict copy on the common (full-register) path. */
    int needs_norm = 0;
    PyObject *key, *val;
    Py_ssize_t pos = 0;
    while (PyDict_Next(input_dict, &pos, &key, &val)) {
        if (PyDict_Contains(arch_map, key) == 1) {
            needs_norm = 1;
            break;
        }
    }
    if (!needs_norm) { Py_INCREF(input_dict); return input_dict; }

    /* Slow path: build a new dict with parent promotion. */
    PyObject *result = PyDict_New();
    if (!result) return NULL;

    pos = 0;
    while (PyDict_Next(input_dict, &pos, &key, &val)) {
        PyObject *info = PyDict_GetItem(arch_map, key);
        if (info && PyTuple_Check(info) && PyTuple_GET_SIZE(info) == 2) {
            PyObject *parent_name = PyTuple_GET_ITEM(info, 0);
            int bit_start = (int)PyLong_AsLong(PyTuple_GET_ITEM(info, 1));
            PyObject *bs_obj = PyLong_FromLong(bit_start);
            PyObject *promoted = PyNumber_Lshift(val, bs_obj);
            Py_DECREF(bs_obj);
            if (!promoted) { Py_DECREF(result); return NULL; }
            PyObject *existing = PyDict_GetItem(result, parent_name);
            PyObject *new_val;
            if (existing) {
                new_val = PyNumber_Or(existing, promoted);
                Py_DECREF(promoted);
            } else {
                new_val = promoted;
            }
            if (!new_val) { Py_DECREF(result); return NULL; }
            PyDict_SetItem(result, parent_name, new_val);
            Py_DECREF(new_val);
        } else {
            PyObject *existing = PyDict_GetItem(result, key);
            if (existing) {
                PyObject *new_val = PyNumber_Or(existing, val);
                if (!new_val) { Py_DECREF(result); return NULL; }
                PyDict_SetItem(result, key, new_val);
                Py_DECREF(new_val);
            } else {
                PyDict_SetItem(result, key, val);
            }
        }
    }
    return result;
}

/* Diagnose why an assignment failed to produce an integer.
 *
 * The evaluator returns None when it cannot compute a mask, and the caller then
 * has nothing useful to say.  The overwhelmingly common cause is an input mask
 * wider than the 64-bit mask path -- a taint mask built by SUMMING bit
 * positions carries into bit 64 the moment two draws collide -- so name that
 * input rather than leaving a bare TypeError from the arithmetic below.
 * Called only on the failure path, so it costs nothing when things work. */
static void set_non_integer_result_error(PyObject *input_taint)
{
    PyObject *key, *val;
    Py_ssize_t pos = 0;
    if (input_taint && PyDict_Check(input_taint)) {
        while (PyDict_Next(input_taint, &pos, &key, &val)) {
            if (!PyLong_Check(val)) continue;
            size_t nbits = _PyLong_NumBits(val);
            if (nbits != (size_t)-1 && nbits > 64) {
                /* Replace whatever the arithmetic raised: "unsupported operand
                 * type(s) for <<: 'NoneType' and 'int'" says nothing about the
                 * caller's mistake, and this says everything. */
                PyErr_Clear();
                PyErr_Format(PyExc_ValueError,
                             "input taint mask for %S is %zu bits wide; the taint path is "
                             "64-bit, so a bit above the register width cannot be represented",
                             key, nbits);
                return;
            }
        }
    }
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "taint evaluation produced a non-integer result");
    }
}


/* The main entry point: CompiledCircuit.evaluate(context). */
/* Internal: do the actual evaluation given pre-extracted context fields.
 * Steals no references; caller owns them all. */
static PyObject *do_evaluate(CompiledCircuit *self,
                              PyObject *context,
                              PyObject *input_taint,
                              PyObject *input_values,
                              PyObject *implicit_policy,
                              PyObject *shadow_memory,
                              PyObject *mem_reader,
                              PyObject *pcode_eval) {
    /* Normalize register names (AL → EAX, EAX → RAX, etc.) to canonical
     * parents.  Always returns fresh refs (Option A), so we own them. */
    PyObject *taint_norm  = NULL;
    PyObject *values_norm = NULL;
    if (context == NULL) {
        taint_norm  = normalize_register_dict(self->arch_str, input_taint);
        values_norm = normalize_register_dict(self->arch_str, input_values);
    } else {
        /* Context-provided dicts have already been normalized inside
         * EvalContext.__init__, so just take a fresh ref. */
        Py_INCREF(input_taint);
        Py_INCREF(input_values);
        taint_norm = input_taint;
        values_norm = input_values;
    }
    if (!taint_norm || !values_norm) {
        Py_XDECREF(taint_norm); Py_XDECREF(values_norm);
        return NULL;
    }

    /* output_taint = taint_norm.copy() */
    PyObject *output_taint = PyDict_Copy(taint_norm);
    if (!output_taint) {
        Py_DECREF(taint_norm); Py_DECREF(values_norm);
        return NULL;
    }

    /* frame-recycle: arm the cell's per-evaluate frame cache so an instruction's result
     * and flags share one execution across the per-assignment eval_program calls
     * below. Only when the C kernel is in use (a CellHandle capsule proves it);
     * reset_frame_cache honors MICROTAINT_RECYCLE_FRAMES (on by default). */
    if (g_cell_capi && g_cell_capi->reset_frame_cache
            && pcode_eval && pcode_eval != Py_None
            && self->cell_handles && PyList_Check(self->cell_handles)) {
        Py_ssize_t nh = PyList_GET_SIZE(self->cell_handles);
        for (Py_ssize_t hi = 0; hi < nh; hi++) {
            PyObject *hc = PyList_GET_ITEM(self->cell_handles, hi);
            if (hc && hc != Py_None && PyCapsule_CheckExact(hc)) {
                g_cell_capi->reset_frame_cache((EvalC_API *)pcode_eval, 1);
                break;
            }
        }
    }

    /* Per-evaluate operand-value memo (see PushMemo), shared across all the
     * per-assignment eval_program calls (input dicts are constant per evaluate).
     * Stack-allocated; disabled for an unusually large string pool. */
    PushMemo memo;
    uint64_t memo_tv[PUSH_MEMO_CAP], memo_vv[PUSH_MEMO_CAP];
    uint8_t  memo_rt[PUSH_MEMO_CAP], memo_rv[PUSH_MEMO_CAP];
    int pool_n = self->string_pool ? (int)PyList_GET_SIZE(self->string_pool) : 0;
    if (pool_n > 0 && pool_n <= PUSH_MEMO_CAP) {
        memo.tv = memo_tv; memo.vv = memo_vv; memo.rt = memo_rt; memo.rv = memo_rv;
        memo.n = pool_n;
        memset(memo_rt, 0, (size_t)pool_n);
        memset(memo_rv, 0, (size_t)pool_n);
    } else {
        memo.tv = NULL; memo.vv = NULL; memo.rt = NULL; memo.rv = NULL; memo.n = 0;
    }

    /* For each assignment: compile if compiled, else fall back to Python. */
    for (int i = 0; i < self->n_progs; i++) {
        AssignmentProg *prog = &self->progs[i];

        if (prog->python_assignment != NULL) {
            /* Need a real EvalContext for the AST fallback path.  If the
             * caller didn't pass one (evaluate_fast path), build one lazily.
             * Most circuits have 0 fallback assignments so this is rare. */
            PyObject *ctx_for_py = context;
            PyObject *built_ctx = NULL;
            if (ctx_for_py == NULL) {
                PyObject *ast_mod = PyImport_ImportModule("microtaint.instrumentation.ast");
                if (!ast_mod) { Py_DECREF(output_taint); return NULL; }
                PyObject *ec_cls = PyObject_GetAttrString(ast_mod, "EvalContext");
                Py_DECREF(ast_mod);
                if (!ec_cls) { Py_DECREF(output_taint); return NULL; }
                PyObject *kw = PyDict_New();
                PyDict_SetItemString(kw, "input_taint", input_taint);
                PyDict_SetItemString(kw, "input_values", input_values);
                /* simulator we don't have directly here; use NULL via a dummy */
                if (implicit_policy) PyDict_SetItemString(kw, "implicit_policy", implicit_policy);
                if (shadow_memory) PyDict_SetItemString(kw, "shadow_memory", shadow_memory);
                if (mem_reader) PyDict_SetItemString(kw, "mem_reader", mem_reader);
                PyObject *empty = PyTuple_New(0);
                built_ctx = PyObject_Call(ec_cls, empty, kw);
                Py_DECREF(empty); Py_DECREF(kw); Py_DECREF(ec_cls);
                if (!built_ctx) { Py_DECREF(output_taint); return NULL; }
                ctx_for_py = built_ctx;
            }
            PyObject *expr = PyObject_GetAttrString(prog->python_assignment, "expression");
            PyObject *target = PyObject_GetAttrString(prog->python_assignment, "target");
            PyObject *is_mem = PyObject_GetAttrString(prog->python_assignment, "is_mem_target");
            if (!expr || !target || !is_mem) {
                Py_XDECREF(expr); Py_XDECREF(target); Py_XDECREF(is_mem);
                Py_XDECREF(built_ctx); Py_DECREF(output_taint); return NULL;
            }
            int is_mem_target = PyObject_IsTrue(is_mem);
            Py_DECREF(is_mem);

            PyObject *val;
            if (expr != Py_None) {
                val = PyObject_CallMethod(expr, "evaluate", "O", ctx_for_py);
            } else {
                /* Match Cython AST behavior: an assignment with neither
                 * expression nor empty dependencies but with expression_str
                 * set is unsupported (e.g., 'FOO' literal). */
                PyObject *expr_str = PyObject_GetAttrString(prog->python_assignment, "expression_str");
                if (expr_str && PyUnicode_Check(expr_str) && PyUnicode_GET_LENGTH(expr_str) > 0) {
                    Py_DECREF(expr_str);
                    Py_DECREF(expr); Py_DECREF(target); Py_XDECREF(built_ctx); Py_DECREF(output_taint);
                    PyErr_SetString(PyExc_NotImplementedError,
                                    "Arbitrary string expressions not supported.");
                    return NULL;
                }
                Py_XDECREF(expr_str);
                PyObject *deps = PyObject_GetAttrString(prog->python_assignment, "dependencies");
                if (!deps) { Py_DECREF(expr); Py_DECREF(target); Py_XDECREF(built_ctx); Py_DECREF(output_taint); return NULL; }
                val = PyLong_FromLong(0);
                Py_ssize_t nd = PyList_Size(deps);
                for (Py_ssize_t di = 0; di < nd; di++) {
                    PyObject *d = PyList_GetItem(deps, di);
                    PyObject *dv = PyObject_CallMethod(d, "evaluate", "O", ctx_for_py);
                    if (!dv) { Py_DECREF(val); Py_DECREF(deps); Py_DECREF(expr); Py_DECREF(target); Py_XDECREF(built_ctx); Py_DECREF(output_taint); return NULL; }
                    PyObject *nv = PyNumber_Or(val, dv);
                    Py_DECREF(val); Py_DECREF(dv);
                    val = nv;
                }
                Py_DECREF(deps);
            }
            Py_DECREF(expr);
            if (!val) { Py_DECREF(target); Py_XDECREF(built_ctx); Py_DECREF(output_taint); return NULL; }

            PyObject *target_name_obj = NULL;
            int bit_start, bit_end;
            if (is_mem_target) {
                PyObject *addr_e = PyObject_GetAttrString(target, "address_expr");
                PyObject *sz     = PyObject_GetAttrString(target, "size");
                PyObject *addr   = PyObject_CallMethod(addr_e, "evaluate", "O", ctx_for_py);
                Py_DECREF(addr_e);
                if (!addr || !sz) { Py_XDECREF(addr); Py_XDECREF(sz); Py_DECREF(val); Py_DECREF(target); Py_XDECREF(built_ctx); Py_DECREF(output_taint); return NULL; }
                PyObject *hex_addr = PyObject_CallFunction(PyDict_GetItemString(PyEval_GetBuiltins(), "hex"), "O", addr);
                int sz_int = (int)PyLong_AsLong(sz);
                target_name_obj = PyUnicode_FromFormat("MEM_%U_%d", hex_addr, sz_int);
                Py_DECREF(addr); Py_DECREF(sz); Py_DECREF(hex_addr);
                bit_start = 0;
                bit_end = sz_int * 8 - 1;
            } else {
                target_name_obj = PyObject_GetAttrString(target, "name");
                PyObject *bs = PyObject_GetAttrString(target, "bit_start");
                PyObject *be = PyObject_GetAttrString(target, "bit_end");
                bit_start = (int)PyLong_AsLong(bs);
                bit_end   = (int)PyLong_AsLong(be);
                Py_DECREF(bs); Py_DECREF(be);
            }
            Py_DECREF(target);

            int width = bit_end - bit_start + 1;
            PyObject *one   = PyLong_FromLong(1);
            PyObject *width_obj = PyLong_FromLong(width);
            PyObject *bs_obj    = PyLong_FromLong(bit_start);
            PyObject *one_shl = PyNumber_Lshift(one, width_obj);
            PyObject *m1      = PyLong_FromLong(1);
            PyObject *mask_unshifted = PyNumber_Subtract(one_shl, m1);
            PyObject *mask = PyNumber_Lshift(mask_unshifted, bs_obj);
            Py_DECREF(one); Py_DECREF(width_obj); Py_DECREF(bs_obj);
            Py_DECREF(one_shl); Py_DECREF(m1); Py_DECREF(mask_unshifted);

            /* Same contract as the compiled path below: `val` is whatever the
             * expression evaluated to, so every step here is checked rather
             * than assumed.  The shift amount used to leak a PyLong per
             * assignment as well. */
            PyObject *bs_shift = PyLong_FromLong(bit_start);
            PyObject *val_shifted = bs_shift ? PyNumber_Lshift(val, bs_shift) : NULL;
            Py_XDECREF(bs_shift);
            Py_DECREF(val);
            PyObject *val_masked  = (val_shifted && mask) ? PyNumber_And(val_shifted, mask) : NULL;
            Py_XDECREF(val_shifted);

            PyObject *current = val_masked ? PyDict_GetItem(output_taint, target_name_obj) : NULL;
            if (val_masked && !current) current = PyLong_FromLong(0);
            else Py_XINCREF(current);
            PyObject *neg_one = current ? PyLong_FromLong(-1) : NULL;
            PyObject *not_mask = neg_one ? PyNumber_Xor(mask, neg_one) : NULL;
            Py_XDECREF(neg_one);
            PyObject *current_clear = not_mask ? PyNumber_And(current, not_mask) : NULL;
            Py_XDECREF(current); Py_XDECREF(not_mask);
            PyObject *new_val = current_clear ? PyNumber_Or(current_clear, val_masked) : NULL;
            Py_XDECREF(current_clear); Py_XDECREF(val_masked); Py_XDECREF(mask);
            if (!new_val) {
                Py_XDECREF(target_name_obj);
                Py_XDECREF(built_ctx);
                Py_DECREF(output_taint);
                set_non_integer_result_error(input_taint);
                return NULL;
            }

            PyDict_SetItem(output_taint, target_name_obj, new_val);
            Py_DECREF(new_val);
            Py_DECREF(target_name_obj);
            Py_XDECREF(built_ctx);
            continue;
        }

        /* Compiled fast path */
        PyObject *result = eval_program(self, prog->bc, prog->bc_len, context,
                                        taint_norm, values_norm,
                                        shadow_memory, mem_reader, pcode_eval, &memo, NULL, NULL, NULL);
        if (!result) {
            if (PyErr_Occurred()) PyErr_Clear();
            PyObject *assignments = PyObject_GetAttrString(self->python_circuit, "assignments");
            PyObject *a = PyList_GetItem(assignments, i);
            PyObject *expr = PyObject_GetAttrString(a, "expression");
            PyObject *val;
            if (context) {
                val = PyObject_CallMethod(expr, "evaluate", "O", context);
            } else {
                val = PyLong_FromLong(0);
            }
            Py_DECREF(expr);
            Py_DECREF(assignments);
            if (!val) { Py_DECREF(output_taint); return NULL; }
            result = val;
        }

        int bit_start = prog->target_bit_start;
        int bit_end   = prog->target_bit_end;

        /* Resolve the output key.  Register target: the interned name.  Memory
         * target: evaluate the address bytecode and build MEM_0x<addr>_<size>
         * exactly as the Cython path does (f'MEM_{hex(address)}_{size}'). */
        PyObject *target_name = NULL;
        if (prog->target_kind == TGT_REG) {
            target_name = PyList_GET_ITEM(self->string_pool, prog->target_name_idx);
            Py_INCREF(target_name);
        } else {
            PyObject *addr_obj = eval_program(self, prog->addr_bc, prog->addr_bc_len,
                                              context, taint_norm, values_norm,
                                              shadow_memory, mem_reader, pcode_eval, &memo, NULL, NULL, NULL);
            if (!addr_obj || !PyLong_Check(addr_obj)) {
                /* Address did not resolve in C; recompute via Python when a real
                 * context is available (mirrors the value-fallback above). */
                Py_XDECREF(addr_obj); addr_obj = NULL;
                if (context) {
                    PyObject *asgs = PyObject_GetAttrString(self->python_circuit, "assignments");
                    if (asgs) {
                        PyObject *a2 = PyList_GetItem(asgs, i);
                        PyObject *tgt2 = a2 ? PyObject_GetAttrString(a2, "target") : NULL;
                        PyObject *ae2  = tgt2 ? PyObject_GetAttrString(tgt2, "address_expr") : NULL;
                        if (ae2) addr_obj = PyObject_CallMethod(ae2, "evaluate", "O", context);
                        Py_XDECREF(ae2); Py_XDECREF(tgt2); Py_DECREF(asgs);
                    }
                }
                if (!addr_obj) { PyErr_Clear(); Py_DECREF(result); Py_DECREF(output_taint); return NULL; }
            }
            PyObject *hexfn = PyDict_GetItemString(PyEval_GetBuiltins(), "hex");
            PyObject *hexb = hexfn ? PyObject_CallFunctionObjArgs(hexfn, addr_obj, NULL) : NULL;
            Py_DECREF(addr_obj);
            if (!hexb) { Py_DECREF(result); Py_DECREF(output_taint); return NULL; }
            target_name = PyUnicode_FromFormat("MEM_%U_%d", hexb, prog->target_size_bytes);
            Py_DECREF(hexb);
            if (!target_name) { Py_DECREF(result); Py_DECREF(output_taint); return NULL; }
        }

        int width = bit_end - bit_start + 1;

        /* Fast path: the whole slice + existing value fit u64 (always true for a
         * compiled assignment: width<=64 and bit_start+width<=64), so the merge
         * (current & ~mask) | ((result << bit_start) & mask) is one uint64
         * expression and one PyLong allocation, versus ~6 arbitrary-precision
         * PyNumber_* temporaries.  Falls through to the PyLong path only if
         * `result` or the existing entry is a >64-bit (wide SIMD) value. */
        if (bit_start + width <= 64) {
            int ok = 1;
            unsigned long long rw = PyLong_AsUnsignedLongLong(result);
            if (PyErr_Occurred()) { PyErr_Clear(); ok = 0; }
            uint64_t cur = 0;
            if (ok) {
                PyObject *cur_obj = PyDict_GetItem(output_taint, target_name);  /* borrowed */
                if (cur_obj) {
                    cur = (uint64_t)PyLong_AsUnsignedLongLong(cur_obj);
                    if (PyErr_Occurred()) { PyErr_Clear(); ok = 0; }
                }
            }
            if (ok) {
                uint64_t mask = ((width >= 64) ? ~(uint64_t)0
                                               : (((uint64_t)1 << width) - 1)) << bit_start;
                uint64_t nv = (cur & ~mask) | (((uint64_t)rw << bit_start) & mask);
                PyObject *new_val = PyLong_FromUnsignedLongLong(nv);
                Py_DECREF(result);
                if (!new_val) { Py_DECREF(output_taint); Py_DECREF(target_name); return NULL; }
                PyDict_SetItem(output_taint, target_name, new_val);
                Py_DECREF(new_val);
                Py_DECREF(target_name);
                continue;
            }
            /* fall through: `result` still owned */
        }

        PyObject *one   = PyLong_FromLong(1);
        PyObject *width_obj = PyLong_FromLong(width);
        PyObject *bs_obj    = PyLong_FromLong(bit_start);
        PyObject *one_shl = PyNumber_Lshift(one, width_obj);
        PyObject *m1      = PyLong_FromLong(1);
        PyObject *mask_unshifted = PyNumber_Subtract(one_shl, m1);
        PyObject *mask = PyNumber_Lshift(mask_unshifted, bs_obj);
        Py_DECREF(one); Py_DECREF(width_obj); Py_DECREF(bs_obj);
        Py_DECREF(one_shl); Py_DECREF(m1); Py_DECREF(mask_unshifted);

        PyObject *bs_obj2 = PyLong_FromLong(bit_start);
        PyObject *val_shifted = bs_obj2 ? PyNumber_Lshift(result, bs_obj2) : NULL;
        Py_XDECREF(bs_obj2);
        Py_DECREF(result);
        /* `result` is whatever the evaluator returned, and a fallback path can
         * hand back None: shifting that raises, and the unchecked chain below
         * then passed NULL to PyNumber_And and took the interpreter down.  A
         * taint mask one bit wider than its register reached here that way. */
        PyObject *val_masked  = (val_shifted && mask) ? PyNumber_And(val_shifted, mask) : NULL;
        Py_XDECREF(val_shifted);
        if (!val_masked) {
            Py_XDECREF(mask);
            Py_DECREF(output_taint);
            Py_DECREF(target_name);
            set_non_integer_result_error(input_taint);
            return NULL;
        }

        PyObject *current = PyDict_GetItem(output_taint, target_name);
        if (!current) current = PyLong_FromLong(0);
        else Py_INCREF(current);
        PyObject *neg_one = PyLong_FromLong(-1);
        PyObject *not_mask = neg_one ? PyNumber_Xor(mask, neg_one) : NULL;
        Py_XDECREF(neg_one);
        PyObject *current_clear = (current && not_mask) ? PyNumber_And(current, not_mask) : NULL;
        Py_XDECREF(current); Py_XDECREF(not_mask);
        PyObject *new_val = current_clear ? PyNumber_Or(current_clear, val_masked) : NULL;
        Py_XDECREF(current_clear); Py_DECREF(val_masked); Py_DECREF(mask);
        if (!new_val) {
            Py_DECREF(output_taint);
            Py_DECREF(target_name);
            return NULL;
        }

        PyDict_SetItem(output_taint, target_name, new_val);
        Py_DECREF(new_val);
        Py_DECREF(target_name);
    }

    /* PC implicit-taint check */
    if (self->pc_target_idx >= 0) {
        PyObject *pc_name = PyList_GET_ITEM(self->string_pool, self->pc_target_idx);
        PyObject *pc_taint = PyDict_GetItem(output_taint, pc_name);
        if (pc_taint) {
            int nz = PyObject_IsTrue(pc_taint);
            if (nz) {
                PyObject *pmod = PyImport_ImportModule("microtaint.types");
                if (pmod) {
                    PyObject *policy_cls = PyObject_GetAttrString(pmod, "ImplicitTaintPolicy");
                    PyObject *KEEP = PyObject_GetAttrString(policy_cls, "KEEP");
                    PyObject *WARN = PyObject_GetAttrString(policy_cls, "WARN");
                    PyObject *STOP = PyObject_GetAttrString(policy_cls, "STOP");
                    int is_keep = PyObject_RichCompareBool(implicit_policy, KEEP, Py_EQ);
                    int is_warn = PyObject_RichCompareBool(implicit_policy, WARN, Py_EQ);
                    int is_stop = PyObject_RichCompareBool(implicit_policy, STOP, Py_EQ);
                    if (is_warn == 1) {
                        PyObject *instr = PyObject_GetAttrString(self->python_circuit, "instruction");
                        if (instr) {
                            PySys_WriteStdout("[Microtaint] Implicit Taint Detected! "
                                              "Control flow (%s) depends on tainted data at instruction: %s\n",
                                              PyUnicode_AsUTF8(pc_name), PyUnicode_AsUTF8(instr));
                            Py_DECREF(instr);
                        }
                    } else if (is_stop == 1) {
                        PyObject *err_cls = PyObject_GetAttrString(pmod, "ImplicitTaintError");
                        PyObject *instr = PyObject_GetAttrString(self->python_circuit, "instruction");
                        const char *instr_s = (instr && PyUnicode_Check(instr)) ? PyUnicode_AsUTF8(instr) : "";
                        const char *pc_s = PyUnicode_AsUTF8(pc_name);
                        PyObject *taint_repr = PyObject_CallMethod(pc_taint, "__index__", NULL);
                        unsigned long long taint_val = taint_repr ? PyLong_AsUnsignedLongLong(taint_repr) : 0;
                        Py_XDECREF(taint_repr);
                        if (PyErr_Occurred()) PyErr_Clear();
                        PyErr_Format(err_cls,
                            "\n[!] FATAL: Implicit Taint Detected\n"
                            "    Instruction (Hex): %s\n"
                            "    Tainted Register : %s\n"
                            "    Taint Mask       : 0x%llx\n"
                            "    Reason: The execution of this branch is governed by a tainted condition.",
                            instr_s, pc_s, taint_val);
                        Py_XDECREF(instr);
                        Py_DECREF(err_cls);
                        Py_DECREF(KEEP); Py_DECREF(WARN); Py_DECREF(STOP);
                        Py_DECREF(policy_cls); Py_DECREF(pmod);
                        Py_DECREF(output_taint);
                        return NULL;
                    }
                    if (is_keep != 1) {
                        PyDict_DelItem(output_taint, pc_name);
                    }
                    Py_DECREF(KEEP); Py_DECREF(WARN); Py_DECREF(STOP);
                    Py_DECREF(policy_cls); Py_DECREF(pmod);
                }
            }
        }
    }

    Py_DECREF(taint_norm);
    Py_DECREF(values_norm);
    return output_taint;
}
/* Interned EvalContext / simulator attribute names.  PyObject_GetAttrString
 * rebuilds and rehashes a temporary PyUnicode on every call; interning once and
 * using PyObject_GetAttr skips that per-evaluate string churn on the hot path. */
static PyObject *g_ctx_attr[7];  /* taint,values,policy,shadow,reader,sim,_pcode */
static int intern_ctx_attrs(void) {
    static const char *names[7] = {
        "input_taint", "input_values", "implicit_policy",
        "shadow_memory", "mem_reader", "simulator", "_pcode",
    };
    for (int i = 0; i < 7; i++) {
        if (!g_ctx_attr[i]) {
            g_ctx_attr[i] = PyUnicode_InternFromString(names[i]);
            if (!g_ctx_attr[i]) return -1;
        }
    }
    return 0;
}

static PyObject *CompiledCircuit_evaluate(CompiledCircuit *self, PyObject *args) {
    PyObject *context;
    if (!PyArg_ParseTuple(args, "O", &context)) return NULL;
    if (intern_ctx_attrs() < 0) return NULL;

    PyObject *input_taint = PyObject_GetAttr(context, g_ctx_attr[0]);
    PyObject *input_values = PyObject_GetAttr(context, g_ctx_attr[1]);
    PyObject *implicit_policy = PyObject_GetAttr(context, g_ctx_attr[2]);
    PyObject *shadow_memory = PyObject_GetAttr(context, g_ctx_attr[3]);
    PyObject *mem_reader = PyObject_GetAttr(context, g_ctx_attr[4]);
    PyObject *simulator = PyObject_GetAttr(context, g_ctx_attr[5]);
    if (!input_taint || !input_values || !implicit_policy || !simulator) {
        Py_XDECREF(input_taint); Py_XDECREF(input_values);
        Py_XDECREF(implicit_policy); Py_XDECREF(shadow_memory);
        Py_XDECREF(mem_reader); Py_XDECREF(simulator);
        return NULL;
    }
    /* simulator may be None (tests sometimes skip it).  In that case
     * pcode is also None — the slow path fallback will run with pcode=None,
     * which only matters if the circuit has cells that need evaluation. */
    PyObject *pcode = NULL;
    if (simulator != Py_None) {
        pcode = PyObject_GetAttr(simulator, g_ctx_attr[6]);
        if (!pcode) {
            /* Some simulators may not have _pcode; clear the error and
             * proceed with pcode=None.  do_evaluate handles NULL pcode
             * by falling through to the Python evaluation path. */
            PyErr_Clear();
        }
    }

    PyObject *result = do_evaluate(self, context, input_taint, input_values,
                                    implicit_policy, shadow_memory, mem_reader,
                                    pcode ? pcode : Py_None);
    Py_DECREF(input_taint); Py_DECREF(input_values);
    Py_DECREF(implicit_policy); Py_XDECREF(shadow_memory);
    Py_XDECREF(mem_reader); Py_DECREF(simulator); Py_XDECREF(pcode);
    return result;
}

/* Fast direct entry: CompiledCircuit.evaluate_fast(input_taint, input_values,
 *                                                   pcode, implicit_policy,
 *                                                   shadow_memory, mem_reader)
 *
 * Skips EvalContext construction.  The hook calls this with already-extracted
 * fields.  Save ~1.3 us/call across ~1.2M calls = ~1.5 s on the bench. */
static PyObject *CompiledCircuit_evaluate_fast(CompiledCircuit *self, PyObject *args) {
    PyObject *input_taint, *input_values, *pcode;
    PyObject *implicit_policy = Py_None;
    PyObject *shadow_memory = Py_None;
    PyObject *mem_reader = Py_None;
    if (!PyArg_ParseTuple(args, "OOO|OOO", &input_taint, &input_values, &pcode,
                          &implicit_policy, &shadow_memory, &mem_reader))
        return NULL;
    return do_evaluate(self, NULL, input_taint, input_values,
                       implicit_policy,
                       (shadow_memory == Py_None) ? NULL : shadow_memory,
                       (mem_reader == Py_None) ? NULL : mem_reader,
                       pcode);
}


/* Stats: how many assignments are compiled vs python-fallback */
static PyObject *CompiledCircuit_stats(CompiledCircuit *self, PyObject *_unused) {
    (void)_unused;
    int compiled = 0, fallback = 0;
    for (int i = 0; i < self->n_progs; i++) {
        if (self->progs[i].python_assignment) fallback++;
        else compiled++;
    }
    return Py_BuildValue("{s:i,s:i,s:i}",
        "n_assignments", self->n_progs,
        "compiled", compiled,
        "python_fallback", fallback);
}

/* evaluate_c(input_taint, input_values) -> output_taint dict, computed on the
 * GIL-FREE C path (register-only circuits only).  Returns None if the circuit is
 * not c_evaluable (has mem ops / python fallback / >64-bit constants) or if any
 * assignment bails, so the caller falls back to the normal evaluate().  This is
 * the pure-C taint evaluator (given the compiled circuit) -- the input/output
 * marshalling here still touches Python for the test/standalone entry, but the
 * evaluation loop itself uses no PyObject; a future in-engine caller can shuttle
 * the uint64 arrays directly with the GIL released. */
/* Ensure the persistent scratch buffers hold at least `npool` uint64 each.
 * Returns 0 on success, -1 on allocation failure (caller falls back). */
static int ensure_scratch(CompiledCircuit *self, int npool) {
    if (npool <= self->scratch_cap && self->scratch_t) return 0;
    int cap = npool > 0 ? npool : 1;
    uint64_t *t = (uint64_t *)realloc(self->scratch_t, (size_t)cap * sizeof(uint64_t));
    uint64_t *v = (uint64_t *)realloc(self->scratch_v, (size_t)cap * sizeof(uint64_t));
    uint64_t *o = (uint64_t *)realloc(self->scratch_o, (size_t)cap * sizeof(uint64_t));
    if (t) self->scratch_t = t;
    if (v) self->scratch_v = v;
    if (o) self->scratch_o = o;
    if (!t || !v || !o) return -1;
    self->scratch_cap = cap;
    return 0;
}

static PyObject *CompiledCircuit_evaluate_c(CompiledCircuit *self, PyObject *args) {
    PyObject *input_taint, *input_values, *pcode;
    if (!PyArg_ParseTuple(args, "OOO", &input_taint, &input_values, &pcode)) return NULL;
    if (!self->c_evaluable) { Py_RETURN_NONE; }
    /* A circuit that writes PC needs do_evaluate's implicit-taint (SC/BOF) policy
     * check (do_evaluate guards it on pc_target_idx >= 0); evaluate_c does not do
     * that check, so it must refuse such circuits and let the caller use the full
     * evaluate().  For pc_target_idx < 0 no implicit-taint check fires, so the C
     * path is complete. */
    if (self->pc_target_idx >= 0) { Py_RETURN_NONE; }
    /* Match do_evaluate: normalize child registers to their canonical parents. */
    PyObject *taint_norm  = normalize_register_dict(self->arch_str, input_taint);
    PyObject *values_norm = normalize_register_dict(self->arch_str, input_values);
    if (!taint_norm || !values_norm) { Py_XDECREF(taint_norm); Py_XDECREF(values_norm); Py_RETURN_NONE; }

    int npool = (int)PyList_GET_SIZE(self->string_pool);
    if (ensure_scratch(self, npool) != 0) {
        Py_DECREF(taint_norm); Py_DECREF(values_norm); return PyErr_NoMemory();
    }
    uint64_t *in_t = self->scratch_t;
    uint64_t *in_v = self->scratch_v;
    uint64_t *out_t = self->scratch_o;
    for (int i = 0; i < npool; i++) {
        PyObject *name = PyList_GET_ITEM(self->string_pool, i);
        PyObject *tv = PyDict_GetItem(taint_norm, name);
        in_t[i] = tv ? (uint64_t)PyLong_AsUnsignedLongLong(tv) : 0;
        if (tv && PyErr_Occurred()) PyErr_Clear();
        PyObject *vv = PyDict_GetItem(values_norm, name);
        in_v[i] = vv ? (uint64_t)PyLong_AsUnsignedLongLong(vv) : 0;
        if (vv && PyErr_Occurred()) PyErr_Clear();
        out_t[i] = in_t[i];   /* output starts as the input taint (pass-through), like taint_norm.copy() */
    }
    int ok = 1;
    for (int p = 0; p < self->n_progs && ok; p++) {
        AssignmentProg *pr = &self->progs[p];
        if (pr->target_kind != TGT_REG || pr->python_assignment) { ok = 0; break; }
        uint64_t result = 0;
        PyObject *rc = eval_program(self, pr->bc, pr->bc_len, NULL, NULL, NULL,
                                    NULL, NULL, pcode, NULL, in_t, in_v, &result);
        if (!rc) { ok = 0; break; }   /* bailed (mem / non-fast cell / error) */
        int bs = pr->target_bit_start, be = pr->target_bit_end;
        int width = be - bs + 1;
        uint64_t mask = ((width >= 64) ? ~(uint64_t)0 : (((uint64_t)1 << width) - 1)) << bs;
        int idx = pr->target_name_idx;
        out_t[idx] = (out_t[idx] & ~mask) | ((result << bs) & mask);
    }
    PyObject *outd = NULL;
    if (ok) {
        outd = PyDict_Copy(taint_norm);   /* pass-through all input taint, then overlay targets */
        if (outd) {
            for (int p = 0; p < self->n_progs; p++) {
                int idx = self->progs[p].target_name_idx;
                PyObject *name = PyList_GET_ITEM(self->string_pool, idx);
                PyObject *val = PyLong_FromUnsignedLongLong(out_t[idx]);
                if (val) { PyDict_SetItem(outd, name, val); Py_DECREF(val); }
            }
        }
    }
    Py_DECREF(taint_norm); Py_DECREF(values_norm);
    if (!ok) { Py_RETURN_NONE; }
    return outd;
}

static int ensure_pool_to_slot(CompiledCircuit *self, PyObject *name_to_slot, int npool);

/* evaluate_c_arr(taint_by_slot, values_by_slot, pcode, name_to_slot) -> dict of
 * TARGET register taints, or None to fall back.
 *
 * The array-gather sibling of evaluate_c (Phase 1): register taint/values arrive
 * as slot-indexed Python lists instead of dicts, so the per-op input fill is an
 * array gather (PyList_GET_ITEM at pool_to_slot[i]) rather than a PyDict_GetItem
 * hash lookup.  string_pool holds canonical PARENT register names and OP_PUSH_*
 * slices by bit range, so no normalization is needed.  Returns ONLY the computed
 * target taints (pass-through of untouched registers is the identity on the
 * input arrays); the caller reconstructs the full state.  This validates the
 * gather bit-exactly vs evaluate_c BEFORE it is wired into the hook's C-array
 * state (the live path is unchanged until then). */
static PyObject *CompiledCircuit_evaluate_c_arr(CompiledCircuit *self, PyObject *args) {
    PyObject *taint_list, *val_list, *pcode, *name_to_slot;
    if (!PyArg_ParseTuple(args, "OOOO", &taint_list, &val_list, &pcode, &name_to_slot))
        return NULL;
    if (!self->c_evaluable) { Py_RETURN_NONE; }
    if (self->pc_target_idx >= 0) { Py_RETURN_NONE; }

    int npool = (int)PyList_GET_SIZE(self->string_pool);
    /* Lazily build + cache pool_to_slot (string_pool idx -> global slot). */
    if (ensure_pool_to_slot(self, name_to_slot, npool) != 0) return PyErr_NoMemory();
    if (ensure_scratch(self, npool) != 0) return PyErr_NoMemory();
    uint64_t *in_t = self->scratch_t, *in_v = self->scratch_v, *out_t = self->scratch_o;
    Py_ssize_t nlist = PyList_GET_SIZE(taint_list);
    for (int i = 0; i < npool; i++) {
        int slot = self->pool_to_slot[i];
        uint64_t tv = 0, vv = 0;
        if (slot >= 0 && slot < nlist) {
            tv = (uint64_t)PyLong_AsUnsignedLongLong(PyList_GET_ITEM(taint_list, slot));
            if (PyErr_Occurred()) PyErr_Clear();
            vv = (uint64_t)PyLong_AsUnsignedLongLong(PyList_GET_ITEM(val_list, slot));
            if (PyErr_Occurred()) PyErr_Clear();
        }
        in_t[i] = tv; in_v[i] = vv; out_t[i] = tv;
    }
    int ok = 1;
    for (int p = 0; p < self->n_progs && ok; p++) {
        AssignmentProg *pr = &self->progs[p];
        if (pr->target_kind != TGT_REG || pr->python_assignment) { ok = 0; break; }
        uint64_t result = 0;
        PyObject *rc = eval_program(self, pr->bc, pr->bc_len, NULL, NULL, NULL,
                                    NULL, NULL, pcode, NULL, in_t, in_v, &result);
        if (!rc) { ok = 0; break; }
        int bs = pr->target_bit_start, be = pr->target_bit_end, width = be - bs + 1;
        uint64_t mask = ((width >= 64) ? ~(uint64_t)0 : (((uint64_t)1 << width) - 1)) << bs;
        int idx = pr->target_name_idx;
        out_t[idx] = (out_t[idx] & ~mask) | ((result << bs) & mask);
    }
    if (!ok) { Py_RETURN_NONE; }
    PyObject *outd = PyDict_New();
    if (!outd) return NULL;
    for (int p = 0; p < self->n_progs; p++) {
        int idx = self->progs[p].target_name_idx;
        PyObject *nm = PyList_GET_ITEM(self->string_pool, idx);
        PyObject *val = PyLong_FromUnsignedLongLong(out_t[idx]);
        if (val) { PyDict_SetItem(outd, nm, val); Py_DECREF(val); }
    }
    return outd;
}

/* evaluate_c_arr_ptr(taint_addr, val_addr, n_slots, pcode, name_to_slot) -> int
 *   number of register targets written, or None to fall back.
 *
 * The live-usable form of evaluate_c_arr: register taint/values live in raw
 * uint64 C arrays owned by the hook (addresses passed as ints), indexed by
 * global slot.  Inputs are gathered into scratch (via the cached pool_to_slot),
 * the differential runs, and -- ATOMICALLY -- the target slots of the taint
 * array are overwritten ONLY if every assignment succeeded (a bail leaves the
 * array untouched, so state never half-updates).  No dict is built on input or
 * output: this is the whole point of the C-native interface.  Pass-through
 * (untouched slots) is the identity on the live array. */
/* string_pool index -> global taint slot, cached.  Rebuilt when the pool size
 * changes; an unmapped name stays -1, which every reader treats as "no taint",
 * matching what the evaluator does for a name the hook has not interned. */
static int ensure_pool_to_slot(CompiledCircuit *self, PyObject *name_to_slot, int npool) {
    /* npool alone is NOT a cache key: it is fixed for the life of the circuit,
     * while name_to_slot grows as the hook interns names lazily.  Rebuild
     * whenever a different mapping is passed or the one we built from has
     * gained entries; both tests are O(1). */
    Py_ssize_t nmap = PyDict_Size(name_to_slot);
    if (self->pool_to_slot && self->pool_to_slot_n == npool
        && self->pool_map_src == name_to_slot && self->pool_map_n == nmap) return 0;
    free(self->pool_to_slot);
    self->pool_to_slot = (int *)malloc((size_t)(npool > 0 ? npool : 1) * sizeof(int));
    if (!self->pool_to_slot) return -1;
    for (int i = 0; i < npool; i++) {
        PyObject *nm = PyList_GET_ITEM(self->string_pool, i);
        PyObject *sv = PyDict_GetItem(name_to_slot, nm);  /* borrowed */
        self->pool_to_slot[i] = (sv && PyLong_Check(sv)) ? (int)PyLong_AsLong(sv) : -1;
    }
    self->pool_to_slot_n = npool;
    Py_INCREF(name_to_slot);
    Py_XSETREF(self->pool_map_src, name_to_slot);
    self->pool_map_n = nmap;
    self->out_built = 0;   /* target slots came from this mapping */
    return 0;
}

/* The implicit-taint interceptor, in C.
 *
 * Mirrors LogicCircuit.evaluate exactly: when the circuit writes PC and the
 * computed PC taint is nonzero, WARN prints and clears, STOP raises, and both
 * IGNORE and (implicitly) WARN clear the PC taint afterwards while KEEP retains
 * it.  Only the printing and raising need Python, and those are the rare cases,
 * so they are handed back with MT_EVAL_PC_REPORT and nothing is committed.
 *
 * Returns 0 to continue committing, MT_EVAL_PC_REPORT to defer to Python.
 * `pc_taint_out` is cleared in place for the policies that drop it. */
static inline int mt_check_implicit(CompiledCircuit *self, uint64_t *out_t, int policy) {
    if (self->pc_target_idx < 0) return 0;
    if (out_t[self->pc_target_idx] == 0) return 0;   /* no leak: nothing to decide */
    if (policy == MT_POLICY_WARN || policy == MT_POLICY_STOP) return MT_EVAL_PC_REPORT;
    if (policy != MT_POLICY_KEEP) out_t[self->pc_target_idx] = 0;
    return 0;
}

/* Register-only array evaluator.  Returns the number of register targets
 * written, MT_EVAL_DECLINED (caller falls back) or MT_EVAL_ERROR.  The
 * PyObject-returning arr_ptr_core below is a thin wrapper on this, so both the
 * Python method and the C API run exactly the same code. */
static int arr_ptr_core_i(CompiledCircuit *self, uint64_t *g_t, uint64_t *g_v,
                          int n_slots, PyObject *pcode, PyObject *name_to_slot,
                          int policy) {
    if (!self->c_evaluable) { return MT_EVAL_DECLINED; }
    /* PC-writing circuits are handled here now: the implicit-taint check is a
     * test on the computed PC taint, done below once out_t exists. */

    int npool = (int)PyList_GET_SIZE(self->string_pool);
    if (ensure_pool_to_slot(self, name_to_slot, npool) != 0) { PyErr_NoMemory(); return MT_EVAL_ERROR; }
    if (ensure_scratch(self, npool) != 0) { PyErr_NoMemory(); return MT_EVAL_ERROR; }
    uint64_t *in_t = self->scratch_t, *in_v = self->scratch_v, *out_t = self->scratch_o;
    for (int i = 0; i < npool; i++) {
        int slot = self->pool_to_slot[i];
        uint64_t tv = (slot >= 0 && slot < n_slots) ? g_t[slot] : 0;
        uint64_t vv = (slot >= 0 && slot < n_slots) ? g_v[slot] : 0;
        in_t[i] = tv; in_v[i] = vv; out_t[i] = tv;
    }
    int ok = 1;
    for (int p = 0; p < self->n_progs && ok; p++) {
        AssignmentProg *pr = &self->progs[p];
        if (pr->target_kind != TGT_REG || pr->python_assignment) { ok = 0; break; }
        uint64_t result = 0;
        PyObject *rc = eval_program(self, pr->bc, pr->bc_len, NULL, NULL, NULL,
                                    NULL, NULL, pcode, NULL, in_t, in_v, &result);
        if (!rc) { ok = 0; break; }
        int bs = pr->target_bit_start, be = pr->target_bit_end, width = be - bs + 1;
        uint64_t mask = ((width >= 64) ? ~(uint64_t)0 : (((uint64_t)1 << width) - 1)) << bs;
        int idx = pr->target_name_idx;
        out_t[idx] = (out_t[idx] & ~mask) | ((result << bs) & mask);
    }
    if (!ok) { return MT_EVAL_DECLINED; }   /* g_t untouched: atomic */
    int pcrc = mt_check_implicit(self, out_t, policy);
    if (pcrc) return pcrc;                  /* still nothing committed */
    int n_written = 0;
    for (int p = 0; p < self->n_progs; p++) {
        int idx = self->progs[p].target_name_idx;
        int slot = self->pool_to_slot[idx];
        if (slot >= 0 && slot < n_slots) { g_t[slot] = out_t[idx]; n_written++; }
    }
    return n_written;
}

static PyObject *arr_ptr_core(CompiledCircuit *self, uint64_t *g_t, uint64_t *g_v,
                              int n_slots, PyObject *pcode, PyObject *name_to_slot) {
    /* The Python-facing method keeps its old contract: PC-writing circuits fall
     * back so do_evaluate applies the policy, hence MT_POLICY_WARN here (any
     * policy whose action is not decidable in C would do). */
    int r = arr_ptr_core_i(self, g_t, g_v, n_slots, pcode, name_to_slot, MT_POLICY_WARN);
    if (r == MT_EVAL_DECLINED || r == MT_EVAL_PC_REPORT) { Py_RETURN_NONE; }
    if (r == MT_EVAL_ERROR) return NULL;
    return PyLong_FromLong(r);
}

/* evaluate_c_mem(input_taint, input_values, pcode, shadow_memory, mem_reader)
 *   -> output_taint dict, or None to fall back to the full evaluate().
 *
 * The memory-capable sibling of evaluate_c: runs the C-array interior for
 * circuits that read/write memory (loads, stores, mem-ALU), so they skip the
 * PyObject-heavy do_evaluate path.  Register taints/values come from uint64
 * arrays (no per-op dict lookups); shadow taint is read at the C level via the
 * shadow capsule; concrete memory values (mem-ALU cells) go through mem_reader,
 * valid because the caller holds the GIL.  The output dict is byte-identical to
 * do_evaluate's (same pass-through copy + MEM_<hex>_<size> keys built the same
 * way), so the hook's post-processing is unchanged.  Returns None (caller falls
 * back) when the circuit is not c_mem_evaluable, writes PC (needs the
 * implicit-taint policy check), the shadow capsule is unavailable, a target is
 * wider than 64 bits, or any assignment bails in C mode -- so every hard case
 * still gets do_evaluate's exact answer. */
static PyObject *CompiledCircuit_evaluate_c_mem(CompiledCircuit *self, PyObject *args) {
    PyObject *input_taint, *input_values, *pcode, *shadow_memory, *mem_reader;
    if (!PyArg_ParseTuple(args, "OOOOO", &input_taint, &input_values, &pcode,
                          &shadow_memory, &mem_reader)) return NULL;
    if (!self->c_mem_evaluable) { Py_RETURN_NONE; }
    if (self->pc_target_idx >= 0) { Py_RETURN_NONE; }   /* needs do_evaluate's implicit-taint check */
    if (!ensure_shadow_capi()) { Py_RETURN_NONE; }      /* no C-level shadow access -> fall back */

    PyObject *taint_norm  = normalize_register_dict(self->arch_str, input_taint);
    PyObject *values_norm = normalize_register_dict(self->arch_str, input_values);
    if (!taint_norm || !values_norm) { Py_XDECREF(taint_norm); Py_XDECREF(values_norm); Py_RETURN_NONE; }

    int npool = (int)PyList_GET_SIZE(self->string_pool);
    if (ensure_scratch(self, npool) != 0) {
        Py_DECREF(taint_norm); Py_DECREF(values_norm); return PyErr_NoMemory();
    }
    uint64_t *in_t = self->scratch_t;
    uint64_t *in_v = self->scratch_v;
    for (int i = 0; i < npool; i++) {
        PyObject *name = PyList_GET_ITEM(self->string_pool, i);
        PyObject *tv = PyDict_GetItem(taint_norm, name);
        in_t[i] = tv ? (uint64_t)PyLong_AsUnsignedLongLong(tv) : 0;
        if (tv && PyErr_Occurred()) PyErr_Clear();
        PyObject *vv = PyDict_GetItem(values_norm, name);
        in_v[i] = vv ? (uint64_t)PyLong_AsUnsignedLongLong(vv) : 0;
        if (vv && PyErr_Occurred()) PyErr_Clear();
    }

    /* output starts as a copy of the input taint (pass-through), then each
     * assignment overlays its target -- exactly like do_evaluate. */
    PyObject *output_taint = PyDict_Copy(taint_norm);
    int ok = (output_taint != NULL);

    /* frame-recycle: arm the per-evaluate cell frame cache identically to
     * do_evaluate, so an instruction's result + flags share one cell execution
     * and pool_idx/recycle_count reset per instruction (this restores the exact
     * reset cadence the memory path had when it ran through do_evaluate). */
    if (ok && g_cell_capi && g_cell_capi->reset_frame_cache
            && pcode && pcode != Py_None
            && self->cell_handles && PyList_Check(self->cell_handles)) {
        Py_ssize_t nh = PyList_GET_SIZE(self->cell_handles);
        for (Py_ssize_t hi = 0; hi < nh; hi++) {
            PyObject *hc = PyList_GET_ITEM(self->cell_handles, hi);
            if (hc && hc != Py_None && PyCapsule_CheckExact(hc)) {
                g_cell_capi->reset_frame_cache((EvalC_API *)pcode, 1);
                break;
            }
        }
    }

    for (int i = 0; ok && i < self->n_progs; i++) {
        AssignmentProg *prog = &self->progs[i];
        if (prog->python_assignment) { ok = 0; break; }   /* excluded by gate, but be safe */

        uint64_t result = 0;
        PyObject *rc = eval_program(self, prog->bc, prog->bc_len, NULL, NULL, NULL,
                                    shadow_memory, mem_reader, pcode, NULL, in_t, in_v, &result);
        if (rc != (PyObject *)self) {           /* NULL (bail) or the too-wide None sentinel */
            if (rc && rc != (PyObject *)self) Py_DECREF(rc);
            if (PyErr_Occurred()) PyErr_Clear();
            ok = 0; break;
        }

        int bit_start = prog->target_bit_start;
        int bit_end   = prog->target_bit_end;
        int width = bit_end - bit_start + 1;
        if (bit_start + width > 64) { ok = 0; break; }   /* >64-bit target: use do_evaluate */

        PyObject *target_name = NULL;
        if (prog->target_kind == TGT_REG) {
            target_name = PyList_GET_ITEM(self->string_pool, prog->target_name_idx);
            Py_INCREF(target_name);
        } else {
            uint64_t addr = 0;
            PyObject *arc = eval_program(self, prog->addr_bc, prog->addr_bc_len, NULL, NULL, NULL,
                                         shadow_memory, mem_reader, pcode, NULL, in_t, in_v, &addr);
            if (arc != (PyObject *)self) {
                if (arc && arc != (PyObject *)self) Py_DECREF(arc);
                if (PyErr_Occurred()) PyErr_Clear();
                ok = 0; break;
            }
            /* MEM_%U_%d with %U = hex(addr) -- byte-identical to do_evaluate. */
            PyObject *addr_obj = PyLong_FromUnsignedLongLong(addr);
            PyObject *hexfn = PyDict_GetItemString(PyEval_GetBuiltins(), "hex");
            PyObject *hexb = (hexfn && addr_obj) ? PyObject_CallFunctionObjArgs(hexfn, addr_obj, NULL) : NULL;
            Py_XDECREF(addr_obj);
            if (!hexb) { if (PyErr_Occurred()) PyErr_Clear(); ok = 0; break; }
            target_name = PyUnicode_FromFormat("MEM_%U_%d", hexb, prog->target_size_bytes);
            Py_DECREF(hexb);
            if (!target_name) { if (PyErr_Occurred()) PyErr_Clear(); ok = 0; break; }
        }

        /* Merge into output_taint[target_name] with do_evaluate's u64 fast path:
         * (current & ~mask) | ((result << bit_start) & mask).  Bail if the
         * existing entry is a >64-bit value (SIMD), so do_evaluate handles it. */
        uint64_t mask = ((width >= 64) ? ~(uint64_t)0 : (((uint64_t)1 << width) - 1)) << bit_start;
        uint64_t cur = 0;
        PyObject *cur_obj = PyDict_GetItem(output_taint, target_name);   /* borrowed */
        if (cur_obj) {
            cur = (uint64_t)PyLong_AsUnsignedLongLong(cur_obj);
            if (PyErr_Occurred()) { PyErr_Clear(); Py_DECREF(target_name); ok = 0; break; }
        }
        uint64_t nv = (cur & ~mask) | ((result << bit_start) & mask);
        PyObject *new_val = PyLong_FromUnsignedLongLong(nv);
        if (!new_val) { Py_DECREF(target_name); ok = 0; break; }
        PyDict_SetItem(output_taint, target_name, new_val);
        Py_DECREF(new_val);
        Py_DECREF(target_name);
    }

    Py_DECREF(taint_norm); Py_DECREF(values_norm);
    if (!ok) { Py_XDECREF(output_taint); Py_RETURN_NONE; }
    return output_taint;
}

/* evaluate_c_mem_ptr(taint_addr, val_addr, n_slots, pcode, shadow, mem_reader,
 *                    name_to_slot) -> list[(addr,size,taint)] for mem writes,
 *                    or None to fall back.
 *
 * The memory counterpart of evaluate_c_arr_ptr, so BOTH eval paths share the
 * hook's g_taint/g_val C arrays (the atomicity finding: register + memory instrs
 * share the taint state, so they must convert together).  Register target taint
 * is written to g_taint; memory target taint is written to the shadow via the
 * capsule.  All writes are DEFERRED to a single commit at the end so a bail
 * never half-updates state (atomic).  Returns the committed mem writes so the
 * hook can update last_tainted_writes + run the AIW check exactly as it does
 * from the MEM_ dict keys today -- no MEM_ string round-trip is built. */
#define CMEM_MAX_WRITES 32
/* Memory-capable array evaluator.  Committed writes are stored in `out` (at most
 * `out_cap` of them); returns how many, or MT_EVAL_DECLINED / MT_EVAL_ERROR.
 * `out` may be NULL with out_cap 0, in which case a circuit that commits any
 * memory write declines: the caller would otherwise lose writes it must see. */
static int mem_ptr_core_i(CompiledCircuit *self, uint64_t *g_t, uint64_t *g_v,
                          int n_slots, PyObject *pcode, PyObject *shadow_memory,
                          PyObject *mem_reader, PyObject *name_to_slot,
                          MtMemWrite *out, int out_cap, int policy) {
    if (!self->c_mem_evaluable) { return MT_EVAL_DECLINED; }
    if (!ensure_shadow_capi()) { return MT_EVAL_DECLINED; }

    int npool = (int)PyList_GET_SIZE(self->string_pool);
    if (ensure_pool_to_slot(self, name_to_slot, npool) != 0) { PyErr_NoMemory(); return MT_EVAL_ERROR; }
    if (ensure_scratch(self, npool) != 0) { PyErr_NoMemory(); return MT_EVAL_ERROR; }
    uint64_t *in_t = self->scratch_t, *in_v = self->scratch_v, *out_t = self->scratch_o;
    for (int i = 0; i < npool; i++) {
        int slot = self->pool_to_slot[i];
        uint64_t tv = (slot >= 0 && slot < n_slots) ? g_t[slot] : 0;
        uint64_t vv = (slot >= 0 && slot < n_slots) ? g_v[slot] : 0;
        in_t[i] = tv; in_v[i] = vv; out_t[i] = tv;
    }

    /* arm frame-recycle identically to evaluate_c_mem / do_evaluate */
    if (g_cell_capi && g_cell_capi->reset_frame_cache && pcode && pcode != Py_None
            && self->cell_handles && PyList_Check(self->cell_handles)) {
        Py_ssize_t nh = PyList_GET_SIZE(self->cell_handles);
        for (Py_ssize_t hi = 0; hi < nh; hi++) {
            PyObject *hc = PyList_GET_ITEM(self->cell_handles, hi);
            if (hc && hc != Py_None && PyCapsule_CheckExact(hc)) {
                g_cell_capi->reset_frame_cache((EvalC_API *)pcode, 1);
                break;
            }
        }
    }

    /* Deferred mem-write accumulation (addr,size)->taint, so multiple slices to
     * one address merge and nothing is committed on a bail. */
    uint64_t mw_addr[CMEM_MAX_WRITES];
    int      mw_size[CMEM_MAX_WRITES];
    uint64_t mw_taint[CMEM_MAX_WRITES];
    int      n_mw = 0;
    int ok = 1;
    for (int i = 0; ok && i < self->n_progs; i++) {
        AssignmentProg *prog = &self->progs[i];
        if (prog->python_assignment) { ok = 0; break; }
        uint64_t result = 0;
        PyObject *rc = eval_program(self, prog->bc, prog->bc_len, NULL, NULL, NULL,
                                    shadow_memory, mem_reader, pcode, NULL, in_t, in_v, &result);
        if (rc != (PyObject *)self) {
            if (rc && rc != (PyObject *)self) Py_DECREF(rc);
            if (PyErr_Occurred()) PyErr_Clear();
            ok = 0; break;
        }
        int bs = prog->target_bit_start, be = prog->target_bit_end, width = be - bs + 1;
        if (bs + width > 64) { ok = 0; break; }
        uint64_t mask = ((width >= 64) ? ~(uint64_t)0 : (((uint64_t)1 << width) - 1)) << bs;
        if (prog->target_kind == TGT_REG) {
            int idx = prog->target_name_idx;
            out_t[idx] = (out_t[idx] & ~mask) | ((result << bs) & mask);
        } else {
            uint64_t addr = 0;
            PyObject *arc = eval_program(self, prog->addr_bc, prog->addr_bc_len, NULL, NULL, NULL,
                                         shadow_memory, mem_reader, pcode, NULL, in_t, in_v, &addr);
            if (arc != (PyObject *)self) {
                if (arc && arc != (PyObject *)self) Py_DECREF(arc);
                if (PyErr_Occurred()) PyErr_Clear();
                ok = 0; break;
            }
            int sz = prog->target_size_bytes;
            /* find existing accumulation for (addr,sz) or add one */
            int slot = -1;
            for (int k = 0; k < n_mw; k++) {
                if (mw_addr[k] == addr && mw_size[k] == sz) { slot = k; break; }
            }
            if (slot < 0) {
                if (n_mw >= CMEM_MAX_WRITES) { ok = 0; break; }
                slot = n_mw++; mw_addr[slot] = addr; mw_size[slot] = sz; mw_taint[slot] = 0;
            }
            mw_taint[slot] = (mw_taint[slot] & ~mask) | ((result << bs) & mask);
        }
    }
    if (!ok) { return MT_EVAL_DECLINED; }   /* nothing committed: atomic */
    /* Reporting the writes is part of the contract (taint clearing and the AIW
     * check both need them), so a result that would not fit declines instead of
     * committing a partial answer. */
    if (n_mw > out_cap) { return MT_EVAL_DECLINED; }
    int pcrc = mt_check_implicit(self, out_t, policy);
    if (pcrc) return pcrc;                  /* nothing committed */

    /* Commit: register targets -> g_taint, memory targets -> shadow. */
    for (int p = 0; p < self->n_progs; p++) {
        if (self->progs[p].target_kind != TGT_REG) continue;
        int idx = self->progs[p].target_name_idx;
        int slot = self->pool_to_slot[idx];
        if (slot >= 0 && slot < n_slots) g_t[slot] = out_t[idx];
    }
    for (int k = 0; k < n_mw; k++) {
        g_shadow_capi->write_mask(shadow_memory, mw_addr[k], mw_taint[k], mw_size[k]);
        out[k].addr = mw_addr[k];
        out[k].taint = mw_taint[k];
        out[k].size = mw_size[k];
    }
    return n_mw;
}

static PyObject *mem_ptr_core(CompiledCircuit *self, uint64_t *g_t, uint64_t *g_v,
                              int n_slots, PyObject *pcode, PyObject *shadow_memory,
                              PyObject *mem_reader, PyObject *name_to_slot) {
    MtMemWrite mw[CMEM_MAX_WRITES];
    int n = mem_ptr_core_i(self, g_t, g_v, n_slots, pcode, shadow_memory,
                           mem_reader, name_to_slot, mw, CMEM_MAX_WRITES,
                           MT_POLICY_WARN);   /* see arr_ptr_core */
    if (n == MT_EVAL_DECLINED || n == MT_EVAL_PC_REPORT) { Py_RETURN_NONE; }
    if (n == MT_EVAL_ERROR) return NULL;
    PyObject *writes = PyList_New(0);
    if (!writes) return NULL;
    for (int k = 0; k < n; k++) {
        PyObject *tup = Py_BuildValue("(KiK)", mw[k].addr, mw[k].size, mw[k].taint);
        if (tup) { PyList_Append(writes, tup); Py_DECREF(tup); }
    }
    return writes;
}

/* ---- Python-level wrappers (unchanged signatures) ---------------------- */

static PyObject *CompiledCircuit_evaluate_c_arr_ptr(CompiledCircuit *self, PyObject *args) {
    unsigned long long taint_addr = 0, val_addr = 0;
    int n_slots = 0;
    PyObject *pcode, *name_to_slot;
    if (!PyArg_ParseTuple(args, "KKiOO", &taint_addr, &val_addr, &n_slots, &pcode, &name_to_slot))
        return NULL;
    return arr_ptr_core(self, (uint64_t *)(uintptr_t)taint_addr,
                        (uint64_t *)(uintptr_t)val_addr, n_slots, pcode, name_to_slot);
}

static PyObject *CompiledCircuit_evaluate_c_mem_ptr(CompiledCircuit *self, PyObject *args) {
    unsigned long long taint_addr = 0, val_addr = 0;
    int n_slots = 0;
    PyObject *pcode, *shadow_memory, *mem_reader, *name_to_slot;
    if (!PyArg_ParseTuple(args, "KKiOOOO", &taint_addr, &val_addr, &n_slots,
                          &pcode, &shadow_memory, &mem_reader, &name_to_slot))
        return NULL;
    return mem_ptr_core(self, (uint64_t *)(uintptr_t)taint_addr,
                        (uint64_t *)(uintptr_t)val_addr, n_slots, pcode,
                        shadow_memory, mem_reader, name_to_slot);
}

/* ---- C API (see circuit_c_api.h): same cores, no Python call machinery ---- */

static PyObject *capi_eval_arr_ptr(PyObject *compiled, uint64_t *taint, uint64_t *val,
                                   int n_slots, PyObject *pcode, PyObject *name_to_slot) {
    /* The cast below is unchecked, so verify the type here: a Python method call
     * would have raised AttributeError on a wrong object, but a bad cast would
     * corrupt memory.  One pointer compare per instruction. */
    if (!PyObject_TypeCheck(compiled, &CompiledCircuitType)) { Py_RETURN_NONE; }
    return arr_ptr_core((CompiledCircuit *)compiled, taint, val, n_slots, pcode, name_to_slot);
}

static PyObject *capi_eval_mem_ptr(PyObject *compiled, uint64_t *taint, uint64_t *val,
                                   int n_slots, PyObject *pcode, PyObject *shadow,
                                   PyObject *mem_reader, PyObject *name_to_slot) {
    if (!PyObject_TypeCheck(compiled, &CompiledCircuitType)) { Py_RETURN_NONE; }
    return mem_ptr_core((CompiledCircuit *)compiled, taint, val, n_slots, pcode,
                        shadow, mem_reader, name_to_slot);
}

static PyObject *capi_eval_mem_ptr_c(PyObject *compiled, uint64_t *taint, uint64_t *val,
                                     int n_slots, PyObject *pcode, PyObject *shadow,
                                     PyObject *mem_reader, PyObject *name_to_slot,
                                     mt_mem_read_fn mem_fn, void *mem_ctx) {
    if (!PyObject_TypeCheck(compiled, &CompiledCircuitType)) { Py_RETURN_NONE; }
    mt_mem_read_fn prev_fn = g_mem_fn;
    void *prev_ctx = g_mem_ctx;
    g_mem_fn = mem_fn; g_mem_ctx = mem_ctx;
    PyObject *r = mem_ptr_core((CompiledCircuit *)compiled, taint, val, n_slots, pcode,
                               shadow, mem_reader, name_to_slot);
    g_mem_fn = prev_fn; g_mem_ctx = prev_ctx;
    return r;
}

static int capi_eval_arr_ptr_i(PyObject *compiled, uint64_t *taint, uint64_t *val,
                               int n_slots, PyObject *pcode, PyObject *name_to_slot,
                               int policy) {
    if (!PyObject_TypeCheck(compiled, &CompiledCircuitType)) return MT_EVAL_DECLINED;
    return arr_ptr_core_i((CompiledCircuit *)compiled, taint, val, n_slots, pcode,
                          name_to_slot, policy);
}

static int capi_eval_mem_ptr_ci(PyObject *compiled, uint64_t *taint, uint64_t *val,
                                int n_slots, PyObject *pcode, PyObject *shadow,
                                PyObject *mem_reader, PyObject *name_to_slot,
                                mt_mem_read_fn mem_fn, void *mem_ctx,
                                MtMemWrite *out, int out_cap, int policy) {
    if (!PyObject_TypeCheck(compiled, &CompiledCircuitType)) return MT_EVAL_DECLINED;
    mt_mem_read_fn prev_fn = g_mem_fn;
    void *prev_ctx = g_mem_ctx;
    g_mem_fn = mem_fn; g_mem_ctx = mem_ctx;
    int r = mem_ptr_core_i((CompiledCircuit *)compiled, taint, val, n_slots, pcode,
                           shadow, mem_reader, name_to_slot, out, out_cap, policy);
    g_mem_fn = prev_fn; g_mem_ctx = prev_ctx;
    return r;
}

static int capi_compiled_flags(PyObject *compiled) {
    if (!compiled || !PyObject_TypeCheck(compiled, &CompiledCircuitType)) return 0;
    CompiledCircuit *cc = (CompiledCircuit *)compiled;
    return (cc->c_evaluable     ? MT_CF_C_EVALUABLE     : 0)
         | (cc->c_mem_evaluable ? MT_CF_C_MEM_EVALUABLE : 0)
         | (cc->has_mem_ops     ? MT_CF_HAS_MEM_OPS     : 0)
         | (cc->value_independent ? MT_CF_VALUE_INDEP   : 0)
         | (cc->pc_target_idx >= 0 ? MT_CF_PC_TARGET     : 0)
         | (cc->has_python_fallback ? MT_CF_PY_FALLBACK  : 0);
}


/* Resolve what the untainted-input fast exit needs.  Eligible circuits are
 * register-only with no python fallback and no PC target (a PC target needs the
 * implicit-taint policy check, which is a decision, not a computation).
 *
 * Soundness of the exit this feeds: with every input taint zero the differential
 * is f(V) XOR f(V) = 0 on every target, and every soundness floor is keyed on a
 * tainted input bit, so all of them contribute zero too.  The result is
 * therefore exactly "clear each target's written bits", with no evaluation. */
static int capi_compiled_prefilter(PyObject *compiled, PyObject *name_to_slot,
                                   MtPrefilter *pf) {
    if (!compiled || !PyObject_TypeCheck(compiled, &CompiledCircuitType)) return -1;
    CompiledCircuit *cc = (CompiledCircuit *)compiled;
    if (cc->pc_target_idx >= 0 || cc->has_python_fallback) return -1;
    /* Register-only circuits, plus memory circuits that only WRITE memory: a
     * store cannot bring taint in, and the shadow bytes it dirties are cleared
     * by the UC_HOOK_MEM_WRITE callback, which clears any written address this
     * instruction did not claim as tainted (and with clean inputs it claims
     * none).  A circuit that reads memory is not eligible: deciding whether the
     * source is tainted needs the effective address. */
    if (cc->has_mem_source && !(cc->mem_reads_ok && cc->n_mem_reads > 0)) return -1;
    if (!cc->c_evaluable && !cc->c_mem_evaluable) return -1;

    int npool = (int)PyList_GET_SIZE(cc->string_pool);
    if (ensure_pool_to_slot(cc, name_to_slot, npool) != 0) { PyErr_Clear(); return -1; }

    if (!cc->out_built) {
        cc->out_built = 1;
        cc->out_ok = 0;
        free(cc->out_slots);
        cc->out_slots = NULL;
        cc->n_out_slots = 0;
        int n = cc->n_progs;
        if (n > 0) {
            cc->out_slots = (MtOutSlot *)calloc((size_t)n, sizeof(MtOutSlot));
            if (!cc->out_slots) return -1;
            int ok = 1;
            int n_out = 0;   /* only register targets land here */
            for (int i = 0; i < n; i++) {
                AssignmentProg *pr = &cc->progs[i];
                if (pr->python_assignment) { ok = 0; break; }
                if (pr->target_kind != TGT_REG) continue;   /* store: nothing in g_t to clear */
                int bs = pr->target_bit_start, be = pr->target_bit_end;
                int width = be - bs + 1;
                if (bs < 0 || width <= 0 || bs + width > 64) { ok = 0; break; }
                int idx = pr->target_name_idx;
                if (idx < 0 || idx >= npool) { ok = 0; break; }
                cc->out_slots[n_out].slot = cc->pool_to_slot[idx];
                cc->out_slots[n_out].clear_mask =
                    ((width >= 64) ? ~(uint64_t)0 : (((uint64_t)1 << width) - 1)) << bs;
                n_out++;
            }
            if (!ok) { free(cc->out_slots); cc->out_slots = NULL; return -1; }
            cc->n_out_slots = n_out;
        }
        cc->out_ok = 1;
    }
    if (!cc->out_ok) return -1;

    pf->pool_slots = cc->pool_to_slot;
    pf->n_pool = npool;
    pf->outs = cc->out_slots;
    pf->n_out = cc->n_out_slots;
    pf->reads_memory = cc->has_mem_source;
    return 0;
}


/* See mem_reads_clean in circuit_c_api.h.
 *
 * Evaluates each captured load-address program on the CURRENT operand values
 * and asks the shadow whether the bytes at that address carry taint.  The
 * address programs were vetted at capture time to contain no cell call and no
 * memory read, so evaluating them here needs values only and cannot itself
 * consult memory.
 *
 * Conservative at every exit: anything unproven returns 0 (not clean), which
 * costs a fast path.  The failure that must not happen is the opposite --
 * returning 1 for a load whose source is tainted -- so every uncertainty is
 * resolved against the exit. */
static int capi_mem_reads_clean(PyObject *compiled, uint64_t *g_t, uint64_t *g_v,
                                int n_slots, PyObject *pcode, PyObject *shadow,
                                PyObject *name_to_slot) {
    if (!compiled || !PyObject_TypeCheck(compiled, &CompiledCircuitType)) return 0;
    CompiledCircuit *cc = (CompiledCircuit *)compiled;
    if (!cc->mem_reads_ok || cc->n_mem_reads <= 0) return 0;
    if (!ensure_shadow_capi()) return 0;

    int npool = (int)PyList_GET_SIZE(cc->string_pool);
    if (ensure_pool_to_slot(cc, name_to_slot, npool) != 0) { PyErr_Clear(); return 0; }
    if (ensure_scratch(cc, npool) != 0) { PyErr_Clear(); return 0; }

    uint64_t *in_t = cc->scratch_t, *in_v = cc->scratch_v;
    for (int i = 0; i < npool; i++) {
        int slot = cc->pool_to_slot[i];
        int ok = (slot >= 0 && slot < n_slots);
        in_t[i] = ok ? g_t[slot] : 0;
        in_v[i] = ok ? g_v[slot] : 0;
    }

    for (int r = 0; r < cc->n_mem_reads; r++) {
        MemReadSite *site = &cc->mem_reads[r];
        uint64_t addr = 0;
        PyObject *rc = eval_program(cc, site->bc, site->bc_len, NULL, NULL, NULL,
                                    shadow, NULL, pcode, NULL, in_t, in_v, &addr);
        if (rc != (PyObject *)cc) {           /* bailed: address not established */
            if (rc && rc != (PyObject *)cc) Py_DECREF(rc);
            if (PyErr_Occurred()) PyErr_Clear();
            return 0;
        }
        if (g_shadow_capi->read_mask(shadow, addr, site->size_bytes) != 0) return 0;
    }
    return 1;
}

static CircuitCAPI g_circuit_capi_struct = { capi_eval_arr_ptr, capi_eval_mem_ptr,
                                             capi_eval_mem_ptr_c,
                                             capi_eval_arr_ptr_i, capi_eval_mem_ptr_ci,
                                             capi_compiled_flags, capi_compiled_prefilter,
                                             capi_mem_reads_clean };

static PyMethodDef CompiledCircuit_methods[] = {
    {"evaluate",      (PyCFunction)CompiledCircuit_evaluate,      METH_VARARGS, NULL},
    {"evaluate_fast", (PyCFunction)CompiledCircuit_evaluate_fast, METH_VARARGS, NULL},
    {"evaluate_c",    (PyCFunction)CompiledCircuit_evaluate_c,    METH_VARARGS,
     "GIL-free C-path taint eval (register-only circuits); None if not c_evaluable."},
    {"evaluate_c_arr", (PyCFunction)CompiledCircuit_evaluate_c_arr, METH_VARARGS,
     "Array-gather register taint eval (slot-indexed lists); None if not c_evaluable."},
    {"evaluate_c_arr_ptr", (PyCFunction)CompiledCircuit_evaluate_c_arr_ptr, METH_VARARGS,
     "Array-gather register eval over raw uint64 C arrays (addresses); atomic direct write."},
    {"evaluate_c_mem", (PyCFunction)CompiledCircuit_evaluate_c_mem, METH_VARARGS,
     "C-array taint eval for memory circuits (loads/stores/mem-ALU); None to fall back."},
    {"evaluate_c_mem_ptr", (PyCFunction)CompiledCircuit_evaluate_c_mem_ptr, METH_VARARGS,
     "Memory eval over raw uint64 C arrays; atomic; returns mem writes as (addr,size,taint)."},
    {"stats",         (PyCFunction)CompiledCircuit_stats,         METH_NOARGS,  NULL},
    {NULL}
};

static PyMemberDef CompiledCircuit_members[] = {
    {"has_mem_ops", T_INT, offsetof(CompiledCircuit, has_mem_ops), READONLY,
     "True if this circuit reads or writes memory (disables wrapper-level Tier 3 cache)."},
    {"value_independent", T_INT, offsetof(CompiledCircuit, value_independent), READONLY,
     "True if the taint output is a pure function of the input taints (reads no "
     "operand value); the instruction cache may then key on the taint signature alone."},
    {"n_assignments", T_INT, offsetof(CompiledCircuit, n_progs), READONLY,
     "Number of assignments compiled.  A fast member so the wrapper can detect a "
     "mutated assignment list per evaluate without allocating a stats() dict."},
    {NULL}
};

static PyTypeObject CompiledCircuitType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "circuit_c.CompiledCircuit",
    .tp_basicsize = sizeof(CompiledCircuit),
    .tp_dealloc   = (destructor)CompiledCircuit_dealloc,
    .tp_flags     = Py_TPFLAGS_DEFAULT,
    .tp_methods   = CompiledCircuit_methods,
    .tp_members   = CompiledCircuit_members,
    .tp_new       = CompiledCircuit_new,
};

/* Module */
/* Introspection: the Expr class names compile_expr() can emit to bytecode
 * WITHOUT falling back to the Python evaluator. MUST be kept in exact sync with
 * the strcmp dispatch in compile_expr (and emit_call_cell / MemoryOperand). The
 * expr-coverage guard test compares this against every Expr subclass so a newly
 * added Expr that the C compiler does not handle fails CI (and would silently
 * recross the Python boundary). */
static const char *SUPPORTED_EXPR_TYPES[] = {
    "TaintOperand", "Constant", "BinaryExpr", "UnaryExpr",
    "AvalancheExpr", "InstructionCellExpr", "MemoryOperand",
    "FullMaskAvalancheExpr", "EqualityTaintExpr", "VariableBitSelectTaintExpr",
    "ComparisonTaintExpr", "SignedOverflowTaintExpr", "VariableMultiplyTaintExpr",
    "VariableShiftTaintExpr",
};

static PyObject *py_supported_expr_types(PyObject *self, PyObject *args) {
    (void)self; (void)args;
    int n = (int)(sizeof(SUPPORTED_EXPR_TYPES) / sizeof(SUPPORTED_EXPR_TYPES[0]));
    PyObject *list = PyList_New(n);
    if (!list) return NULL;
    for (int i = 0; i < n; i++) {
        PyObject *s = PyUnicode_FromString(SUPPORTED_EXPR_TYPES[i]);
        if (!s) { Py_DECREF(list); return NULL; }
        PyList_SET_ITEM(list, i, s);
    }
    return list;
}

static PyObject *py_cell_capi_loaded(PyObject *self, PyObject *args) {
    (void)self; (void)args;
    return PyBool_FromLong(g_cell_capi != NULL);
}

static PyMethodDef module_methods[] = {
    {"compile_circuit", py_compile_circuit, METH_VARARGS, "Compile a LogicCircuit to bytecode."},
    {"supported_expr_types", py_supported_expr_types, METH_NOARGS,
     "List of Expr class names the bytecode compiler handles without Python fallback."},
    {"cell_capi_loaded", py_cell_capi_loaded, METH_NOARGS,
     "True if the cell_c fast-path CAPI is loaded (OP_CALL_CELL uses cell_eval_fast)."},
    {NULL}
};
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "circuit_c", NULL, -1, module_methods
};

PyMODINIT_FUNC PyInit_circuit_c(void) {
    if (PyType_Ready(&CompiledCircuitType) < 0) return NULL;
    PyObject *m = PyModule_Create(&moduledef);
    if (!m) return NULL;
    Py_INCREF(&CompiledCircuitType);
    PyModule_AddObject(m, "CompiledCircuit", (PyObject *)&CompiledCircuitType);

    /* Publish the C API so hook_core can call the array evaluators directly,
     * without the per-instruction attribute lookup / tuple / PyArg_ParseTuple /
     * PyLong boxing that a Python method call costs. */
    {
        PyObject *cap = PyCapsule_New((void *)&g_circuit_capi_struct,
                                      "microtaint.instrumentation.cell_c.circuit_c._circuit_capi",
                                      NULL);
        if (cap) PyModule_AddObject(m, "_circuit_capi", cap);
        else PyErr_Clear();  /* not fatal: caller falls back to the methods */
    }

    /* Import cell_c's CellCAPI capsule.  If cell_c isn't available or
     * the capsule isn't there, g_cell_capi stays NULL and we use the
     * Python slow path for OP_CALL_CELL. */
    g_cell_capi = (CellCAPI *)PyCapsule_Import("cell_c._cell_capi", 0);
    if (!g_cell_capi) {
        PyErr_Clear();
        /* PyCapsule_Import needs cell_c importable as a TOP-LEVEL module, which
         * only happens when its directory is on sys.path (e.g. tests/conftest.py).
         * In normal use (the emulator) it is not, so g_cell_capi would stay NULL
         * and every OP_CALL_CELL would take the ~3x-slower Python evaluate_concrete
         * path.  Fall back to importing cell_c by its full dotted name and reading
         * the same capsule attribute -- no sys.path dependency. */
        PyObject *cmod = PyImport_ImportModule("microtaint.instrumentation.cell_c.cell_c");
        if (cmod) {
            PyObject *cap = PyObject_GetAttrString(cmod, "_cell_capi");
            if (cap && PyCapsule_CheckExact(cap)) {
                g_cell_capi = (CellCAPI *)PyCapsule_GetPointer(cap, "cell_c._cell_capi");
            }
            Py_XDECREF(cap);
            Py_DECREF(cmod);
        }
        if (!g_cell_capi) PyErr_Clear();   /* not fatal — Cython kernel users still work */
    }

    return m;
}
