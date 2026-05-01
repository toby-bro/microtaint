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
#include <stdint.h>
#include <string.h>
#include "circuit_bytecode.h"

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
    /* Add cell to cells list */
    int cell_idx = (int)PyList_GET_SIZE(cc->cells);
    PyList_Append(cc->cells, cell_obj);

    /* Get .inputs dict and walk in dict-iteration order, emitting
     * per-input expressions, then a final OP_CALL_CELL with the count
     * and per-input name-idx args. */
    PyObject *inputs = PyObject_GetAttrString(cell_obj, "inputs");
    if (!inputs || !PyDict_Check(inputs)) {
        Py_XDECREF(inputs);
        e->fallback = 1;
        return;
    }
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
        /* Only compile if it fits in uint64 */
        if (!PyLong_Check(value)) { Py_DECREF(value); e->fallback = 1; return; }
        unsigned long long uv = PyLong_AsUnsignedLongLong(value);
        if (PyErr_Occurred()) { PyErr_Clear(); Py_DECREF(value); e->fallback = 1; return; }
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
        return;
    }
    /* Unknown / MemoryDifferentialExpr / etc — fall back */
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
    if (!PyArg_ParseTuple(args, "O", &circuit)) return NULL;

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
            continue;
        }
        /* Register target */
        PyObject *t_name = PyObject_GetAttrString(target, "name");
        PyObject *t_bs   = PyObject_GetAttrString(target, "bit_start");
        PyObject *t_be   = PyObject_GetAttrString(target, "bit_end");
        Py_DECREF(target);
        if (!t_name || !t_bs || !t_be) { Py_XDECREF(t_name); Py_XDECREF(t_bs); Py_XDECREF(t_be); goto fb; }
        const char *tn = PyUnicode_AsUTF8(t_name);
        if (!tn) { Py_DECREF(t_name); Py_DECREF(t_bs); Py_DECREF(t_be); goto fb; }
        p->target_kind = TGT_REG;
        p->target_name_idx = strpool_intern(cc, tn);
        p->target_bit_start = (int)PyLong_AsLong(t_bs);
        p->target_bit_end   = (int)PyLong_AsLong(t_be);
        p->target_size_bytes = 0;
        Py_DECREF(t_name); Py_DECREF(t_bs); Py_DECREF(t_be);

        /* Compile the rhs expression */
        PyObject *expr = PyObject_GetAttrString(a, "expression");
        if (!expr || expr == Py_None) {
            Py_XDECREF(expr);
            /* Dependencies fallback — uncommon, keep Python */
            p->python_assignment = a; Py_INCREF(a);
            cc->has_python_fallback = 1;
            continue;
        }
        BCEmit emit_buf = {{0}, 0, 0, 0};
        compile_expr(cc, &emit_buf, expr);
        Py_DECREF(expr);
        if (emit_buf.fallback || emit_buf.overflow) {
            p->python_assignment = a; Py_INCREF(a);
            cc->has_python_fallback = 1;
            continue;
        }
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
            uint64_t v = (uint64_t)PyLong_AsUnsignedLongLong(cv);
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
        case OP_NOT:  if (sp < 1) goto err; stack[sp-1] = ~stack[sp-1]; break;
        case OP_AVALANCHE: {
            int size_bits = (int)bc[pc++];
            if (sp < 1) goto err;
            uint64_t v = stack[sp-1];
            stack[sp-1] = (v != 0) ? mask_range(size_bits) : 0;
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
            PyObject *cell = PyList_GET_ITEM(cc->cells, cell_idx);
            /* Build inputs dict {name: value_python_int} */
            PyObject *inputs_dict = PyDict_New();
            if (!inputs_dict) return NULL;
            for (int i = 0; i < n_inputs; i++) {
                int name_idx = (int)bc[pc + i];
                PyObject *name = PyList_GET_ITEM(cc->string_pool, name_idx);
                /* Stack order: input 0 was pushed first, sits deepest */
                uint64_t v = stack[sp - n_inputs + i];
                PyObject *vobj = PyLong_FromUnsignedLongLong(v);
                PyDict_SetItem(inputs_dict, name, vobj);
                Py_DECREF(vobj);
            }
            pc += n_inputs;
            sp -= n_inputs;

            /* Call pcode.evaluate_concrete(cell, inputs_dict) — works on both
             * the Cython PCodeCellEvaluator and the C PCodeCellEvaluatorC since
             * both implement this signature with a flat-dict input format.
             * (evaluate_concrete_flat exists only on the C kernel; using
             * evaluate_concrete keeps us kernel-agnostic.) */
            PyObject *r = PyObject_CallMethod(pcode_eval, "evaluate_concrete",
                                              "OO", cell, inputs_dict);
            Py_DECREF(inputs_dict);
            if (!r) {
                /* Could be PCodeFallbackNeeded — let caller handle by falling
                 * back to Python evaluator for this assignment. */
                return NULL;
            }
            uint64_t v = (uint64_t)PyLong_AsUnsignedLongLong(r);
            Py_DECREF(r);
            if (PyErr_Occurred()) PyErr_Clear();
            stack[sp++] = v;
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

/* The main entry point: CompiledCircuit.evaluate(context). */
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
    PyObject *pcode = PyObject_GetAttrString(simulator, "_pcode");
    if (!pcode) {
        Py_DECREF(input_taint); Py_DECREF(input_values);
        Py_DECREF(implicit_policy); Py_XDECREF(shadow_memory);
        Py_XDECREF(mem_reader); Py_DECREF(simulator);
        return NULL;
    }

    /* output_taint = input_taint.copy() */
    PyObject *output_taint = PyDict_Copy(input_taint);
    if (!output_taint) goto out_err;

    /* For each assignment: compile if compiled, else fall back to Python. */
    for (int i = 0; i < self->n_progs; i++) {
        AssignmentProg *prog = &self->progs[i];

        if (prog->python_assignment != NULL) {
            /* Fall back: evaluate this single TaintAssignment via Python.
             * Easiest: build a one-assignment LogicCircuit and evaluate it.
             * Cheaper: invoke the same code as Cython does inline.  Use the
             * original circuit's evaluate when ALL assignments are Python (rare);
             * here, do it per-assignment. */
            PyObject *expr = PyObject_GetAttrString(prog->python_assignment, "expression");
            PyObject *target = PyObject_GetAttrString(prog->python_assignment, "target");
            PyObject *is_mem = PyObject_GetAttrString(prog->python_assignment, "is_mem_target");
            if (!expr || !target || !is_mem) {
                Py_XDECREF(expr); Py_XDECREF(target); Py_XDECREF(is_mem);
                goto out_err;
            }
            int is_mem_target = PyObject_IsTrue(is_mem);
            Py_DECREF(is_mem);

            PyObject *val;
            if (expr != Py_None) {
                val = PyObject_CallMethod(expr, "evaluate", "O", context);
            } else {
                /* dependencies fallback: OR */
                PyObject *deps = PyObject_GetAttrString(prog->python_assignment, "dependencies");
                if (!deps) { Py_DECREF(expr); Py_DECREF(target); goto out_err; }
                val = PyLong_FromLong(0);
                Py_ssize_t nd = PyList_Size(deps);
                for (Py_ssize_t di = 0; di < nd; di++) {
                    PyObject *d = PyList_GetItem(deps, di);
                    PyObject *dv = PyObject_CallMethod(d, "evaluate", "O", context);
                    if (!dv) { Py_DECREF(val); Py_DECREF(deps); Py_DECREF(expr); Py_DECREF(target); goto out_err; }
                    PyObject *nv = PyNumber_Or(val, dv);
                    Py_DECREF(val); Py_DECREF(dv);
                    val = nv;
                }
                Py_DECREF(deps);
            }
            Py_DECREF(expr);
            if (!val) { Py_DECREF(target); goto out_err; }

            /* Compute target bounds */
            PyObject *target_name_obj = NULL;
            int bit_start, bit_end;
            if (is_mem_target) {
                PyObject *addr_e = PyObject_GetAttrString(target, "address_expr");
                PyObject *sz     = PyObject_GetAttrString(target, "size");
                PyObject *addr   = PyObject_CallMethod(addr_e, "evaluate", "O", context);
                Py_DECREF(addr_e);
                if (!addr || !sz) { Py_XDECREF(addr); Py_XDECREF(sz); Py_DECREF(val); Py_DECREF(target); goto out_err; }
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

            /* Apply mask: mask = ((1 << width) - 1) << bit_start */
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

            /* current = output_taint.get(target_name, 0) */
            PyObject *current = PyDict_GetItem(output_taint, target_name_obj);
            if (!current) current = PyLong_FromLong(0);
            else Py_INCREF(current);
            /* not_mask = ~mask  */
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
            continue;
        }

        /* Compiled fast path */
        PyObject *result = eval_program(self, prog, context,
                                        input_taint, input_values,
                                        shadow_memory, mem_reader, pcode);
        if (!result) {
            /* Bytecode hit something it couldn't handle.  Could be a real
             * Python exception (e.g. PCodeFallbackNeeded from cell call).
             * Fall back to Python AST eval for this assignment. */
            if (PyErr_Occurred()) PyErr_Clear();
            /* Use the python_assignment if we kept it, else build one from
             * the original circuit's assignments[i]. */
            PyObject *assignments = PyObject_GetAttrString(self->python_circuit, "assignments");
            PyObject *a = PyList_GetItem(assignments, i);
            PyObject *expr = PyObject_GetAttrString(a, "expression");
            PyObject *val = PyObject_CallMethod(expr, "evaluate", "O", context);
            Py_DECREF(expr);
            Py_DECREF(assignments);
            if (!val) goto out_err;
            result = val;
        }

        /* Result is a Python int.  Apply mask + write to output_taint. */
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

    /* PC implicit-taint check (mirror of Cython evaluate) */
    if (self->pc_target_idx >= 0) {
        PyObject *pc_name = PyList_GET_ITEM(self->string_pool, self->pc_target_idx);
        PyObject *pc_taint = PyDict_GetItem(output_taint, pc_name);
        if (pc_taint) {
            int nz = PyObject_IsTrue(pc_taint);
            if (nz) {
                /* Check policy.  Import ImplicitTaintPolicy lazily. */
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
                        goto out_err;
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

    Py_DECREF(input_taint); Py_DECREF(input_values);
    Py_DECREF(implicit_policy); Py_XDECREF(shadow_memory);
    Py_XDECREF(mem_reader); Py_DECREF(simulator); Py_DECREF(pcode);
    return output_taint;

out_err:
    Py_XDECREF(input_taint); Py_XDECREF(input_values);
    Py_XDECREF(implicit_policy); Py_XDECREF(shadow_memory);
    Py_XDECREF(mem_reader); Py_XDECREF(simulator); Py_XDECREF(pcode);
    return NULL;
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
    {"evaluate", (PyCFunction)CompiledCircuit_evaluate, METH_VARARGS, NULL},
    {"stats",    (PyCFunction)CompiledCircuit_stats,    METH_NOARGS,  NULL},
    {NULL}
};

static PyTypeObject CompiledCircuitType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "circuit_c.CompiledCircuit",
    .tp_basicsize = sizeof(CompiledCircuit),
    .tp_dealloc   = (destructor)CompiledCircuit_dealloc,
    .tp_flags     = Py_TPFLAGS_DEFAULT,
    .tp_methods   = CompiledCircuit_methods,
    .tp_new       = CompiledCircuit_new,
};

/* Module */
static PyMethodDef module_methods[] = {
    {"compile_circuit", py_compile_circuit, METH_VARARGS, "Compile a LogicCircuit to bytecode."},
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
    return m;
}
