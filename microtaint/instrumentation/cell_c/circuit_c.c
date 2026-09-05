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

/* Global CAPI pointer — populated at module init via PyCapsule_Import. */
static CellCAPI *g_cell_capi = NULL;

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
        Py_DECREF(addr_e);
        if (e->fallback) return;
        emit(e, is_taint ? OP_PUSH_MEM_TAINT : OP_PUSH_MEM_VALUE);
        emit(e, (uint32_t)size_bytes);
        cc->has_mem_ops = 1;
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
    self->has_python_fallback = 0;
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
        if (is_mem_target) {
            /* Memory target — keep the original Python TaintAssignment for now.
             * Compiling memory targets requires evaluating the address expr at
             * runtime (which the evaluator could do), but it's the rare case;
             * skip for now. */
            Py_DECREF(target);
            p->python_assignment = a; Py_INCREF(a);
            cc->has_python_fallback = 1;
            cc->has_mem_ops = 1;   /* memory write — disables Tier 3 cache */
            continue;
        }
        /* Register target.
         * Initialize to NULL so the `fb:` cleanup is safe regardless of
         * which path we take (each Py_DECREF is paired with a clear).
         * This also silences -Wmaybe-uninitialized warnings. */
        PyObject *t_name = NULL, *t_bs = NULL, *t_be = NULL;
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

/* Evaluate a single bytecode program for one assignment.
 * Returns the result as Python int (PyLong) or NULL on error. */
static PyObject *eval_program(CompiledCircuit *cc,
                              AssignmentProg *prog,
                              PyObject *context,
                              PyObject *input_taint, PyObject *input_values,
                              PyObject *shadow_memory, PyObject *mem_reader,
                              PyObject *pcode_eval) {
    uint64_t stack[CIRCUIT_STACK_MAX];
    int sp = 0;
    int pc = 0;
    uint32_t *bc = prog->bc;
    int bc_len = prog->bc_len;

    while (pc < bc_len) {
        uint32_t op = bc[pc++];
        switch (op) {
        case OP_PUSH_TAINT: {
            int name_idx  = (int)bc[pc++];
            int bit_start = (int)bc[pc++];
            int bit_end   = (int)bc[pc++];
            PyObject *name = PyList_GET_ITEM(cc->string_pool, name_idx);
            PyObject *val = PyDict_GetItem(input_taint, name);
            uint64_t v = 0;
            if (val) {
                v = (uint64_t)PyLong_AsUnsignedLongLong(val);
                if (PyErr_Occurred()) {
                    PyErr_Clear();
                    /* Value too big for u64; fall back to Python int math */
                    /* Punt: bail out by raising — caller uses Python fallback */
                    return Py_BuildValue("");  /* sentinel; handled below */
                }
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
            PyObject *name = PyList_GET_ITEM(cc->string_pool, name_idx);
            PyObject *val = PyDict_GetItem(input_values, name);
            uint64_t v = 0;
            if (val) {
                v = (uint64_t)PyLong_AsUnsignedLongLong(val);
                if (PyErr_Occurred()) PyErr_Clear();
            }
            int width = bit_end - bit_start + 1;
            uint64_t m = mask_range(width);
            stack[sp++] = (v >> bit_start) & m;
            break;
        }
        case OP_PUSH_CONST: {
            int ci = (int)bc[pc++];
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
            if (mem_reader != Py_None && mem_reader != NULL) {
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
            if (g_cell_capi && cell_idx < (int)PyList_GET_SIZE(cc->cell_handles)) {
                PyObject *handle_cap = PyList_GET_ITEM(cc->cell_handles, cell_idx);
                if (handle_cap && handle_cap != Py_None
                    && PyCapsule_CheckExact(handle_cap)) {
                    CellHandle_API *h = (CellHandle_API *)PyCapsule_GetPointer(handle_cap, "CellHandle");
                    if (h) {
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
                }
            }

            if (fast_path_taken) {
                pc += n_inputs;       /* skip per-input name idxs */
                sp -= n_inputs;
                stack[sp++] = fast_v;
                break;
            }

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

    /* Hot-path fast check: if no key is a known child, just copy. */
    int needs_norm = 0;
    PyObject *key, *val;
    Py_ssize_t pos = 0;
    while (PyDict_Next(input_dict, &pos, &key, &val)) {
        if (PyDict_Contains(arch_map, key) == 1) {
            needs_norm = 1;
            break;
        }
    }
    if (!needs_norm) return PyDict_Copy(input_dict);

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

            PyObject *val_shifted = PyNumber_Lshift(val, PyLong_FromLong(bit_start));
            Py_DECREF(val);
            PyObject *val_masked  = PyNumber_And(val_shifted, mask);
            Py_DECREF(val_shifted);

            PyObject *current = PyDict_GetItem(output_taint, target_name_obj);
            if (!current) current = PyLong_FromLong(0);
            else Py_INCREF(current);
            PyObject *neg_one = PyLong_FromLong(-1);
            PyObject *not_mask = PyNumber_Xor(mask, neg_one);
            Py_DECREF(neg_one);
            PyObject *current_clear = PyNumber_And(current, not_mask);
            Py_DECREF(current); Py_DECREF(not_mask);
            PyObject *new_val = PyNumber_Or(current_clear, val_masked);
            Py_DECREF(current_clear); Py_DECREF(val_masked); Py_DECREF(mask);

            PyDict_SetItem(output_taint, target_name_obj, new_val);
            Py_DECREF(new_val);
            Py_DECREF(target_name_obj);
            Py_XDECREF(built_ctx);
            continue;
        }

        /* Compiled fast path */
        PyObject *result = eval_program(self, prog, context,
                                        taint_norm, values_norm,
                                        shadow_memory, mem_reader, pcode_eval);
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

        PyObject *bs_obj2 = PyLong_FromLong(bit_start);
        PyObject *val_shifted = PyNumber_Lshift(result, bs_obj2);
        Py_DECREF(bs_obj2);
        Py_DECREF(result);
        PyObject *val_masked  = PyNumber_And(val_shifted, mask);
        Py_DECREF(val_shifted);

        PyObject *target_name = PyList_GET_ITEM(self->string_pool, prog->target_name_idx);
        Py_INCREF(target_name);

        PyObject *current = PyDict_GetItem(output_taint, target_name);
        if (!current) current = PyLong_FromLong(0);
        else Py_INCREF(current);
        PyObject *neg_one = PyLong_FromLong(-1);
        PyObject *not_mask = PyNumber_Xor(mask, neg_one);
        Py_DECREF(neg_one);
        PyObject *current_clear = PyNumber_And(current, not_mask);
        Py_DECREF(current); Py_DECREF(not_mask);
        PyObject *new_val = PyNumber_Or(current_clear, val_masked);
        Py_DECREF(current_clear); Py_DECREF(val_masked); Py_DECREF(mask);

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
static PyObject *CompiledCircuit_evaluate(CompiledCircuit *self, PyObject *args) {
    PyObject *context;
    if (!PyArg_ParseTuple(args, "O", &context)) return NULL;

    PyObject *input_taint = PyObject_GetAttrString(context, "input_taint");
    PyObject *input_values = PyObject_GetAttrString(context, "input_values");
    PyObject *implicit_policy = PyObject_GetAttrString(context, "implicit_policy");
    PyObject *shadow_memory = PyObject_GetAttrString(context, "shadow_memory");
    PyObject *mem_reader = PyObject_GetAttrString(context, "mem_reader");
    PyObject *simulator = PyObject_GetAttrString(context, "simulator");
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
        pcode = PyObject_GetAttrString(simulator, "_pcode");
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

static PyMethodDef CompiledCircuit_methods[] = {
    {"evaluate",      (PyCFunction)CompiledCircuit_evaluate,      METH_VARARGS, NULL},
    {"evaluate_fast", (PyCFunction)CompiledCircuit_evaluate_fast, METH_VARARGS, NULL},
    {"stats",         (PyCFunction)CompiledCircuit_stats,         METH_NOARGS,  NULL},
    {NULL}
};

static PyMemberDef CompiledCircuit_members[] = {
    {"has_mem_ops", T_INT, offsetof(CompiledCircuit, has_mem_ops), READONLY,
     "True if this circuit reads or writes memory (disables wrapper-level Tier 3 cache)."},
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
 * the strcmp dispatch in compile_expr. The expr-coverage guard test compares
 * this against every Expr subclass so a newly added Expr the C compiler cannot
 * handle fails CI (and would silently recross the Python boundary). */
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

static PyMethodDef module_methods[] = {
    {"compile_circuit", py_compile_circuit, METH_VARARGS, "Compile a LogicCircuit to bytecode."},
    {"supported_expr_types", py_supported_expr_types, METH_NOARGS,
     "List of Expr class names the bytecode compiler handles without Python fallback."},
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

    /* Import cell_c's CellCAPI capsule.  If cell_c isn't available or
     * the capsule isn't there, g_cell_capi stays NULL and we use the
     * Python slow path for OP_CALL_CELL. */
    g_cell_capi = (CellCAPI *)PyCapsule_Import("cell_c._cell_capi", 0);
    if (!g_cell_capi) {
        PyErr_Clear();   /* not fatal — Cython kernel users still work */
    }

    return m;
}
