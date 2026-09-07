/*
 * taint_ir_c — execute a lowered taint program.
 * ============================================
 *
 * microtaint/taint_ir lowers one instruction's whole taint propagation, flags
 * included, to a straight-line SSA program over 64-bit words.  This module runs
 * that program: a flat array of nodes, one scratch slot each, one dispatch
 * loop, no branches in the program itself and no allocation per execution.
 *
 * The point of the flat form is that it is the same input a code generator
 * wants.  Interpreting it is the portable path and the correctness reference;
 * lowering it to machine code is the fast path, and both consume this layout,
 * so a program compiled once can be executed either way without a second
 * representation.
 *
 * Register state crosses the boundary as two plain uint64 arrays indexed by
 * slot, with the slot baked into the program at compile time -- so running an
 * instruction's taint involves no name lookup, no dict, and no Python object.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>

/* Opcodes.  Order is arbitrary but must match _OP_ID in taint_ir/exec.py. */
enum {
    IR_CONST = 0, IR_INV, IR_INT,
    IR_AND, IR_OR, IR_XOR, IR_ADD, IR_SUB, IR_MUL,
    IR_SHL, IR_SHR, IR_SAR, IR_NOT, IR_NEG,
    IR_ULT, IR_SLT, IR_EQ, IR_NEZ, IR_SEL,
    IR_POPCNT, IR_CLZ, IR_UDIV, IR_UREM, IR_SDIV, IR_SREM,
    IR_MULHI,
    IR_NOPS
};

typedef struct {
    int32_t node;
    int32_t kind;      /* 0 = value, 1 = taint */
    int32_t slot;
} IRInput;

typedef struct {
    int32_t slot;
    int32_t node;
} IROutput;

struct IRProgC;

typedef struct {
    int       n_nodes;
    uint8_t  *op;
    int32_t  *a, *b, *c;
    uint64_t *imm;
    int       n_inputs;
    IRInput  *inputs;
    int       n_outputs;
    IROutput *outputs;
    uint64_t *scratch;   /* n_nodes wide, owned; reused across executions */
    /* Native code for this program, when the host has an emitter and the
     * program is within what it handles.  NULL means "interpret". */
    void     *jit_code;
    size_t    jit_size;
    void    (*jit_fn)(const uint64_t *, const uint64_t *, uint64_t *);
} IRProgC;

#if defined(__x86_64__)
#  define MT_HAVE_JIT 1
#  include "taint_jit_x64.h"
#endif

static void irprog_free(IRProgC *p) {
    if (!p) return;
#ifdef MT_HAVE_JIT
    if (p->jit_code) munmap(p->jit_code, p->jit_size);
#endif
    free(p->op); free(p->a); free(p->b); free(p->c); free(p->imm);
    free(p->inputs); free(p->outputs); free(p->scratch);
    free(p);
}

static void irprog_destructor(PyObject *cap) {
    irprog_free((IRProgC *)PyCapsule_GetPointer(cap, "microtaint.taint_ir.prog"));
}

/* ── the evaluator ───────────────────────────────────────────────────
 *
 * One pass, in program order.  The IR is SSA and topologically ordered by
 * construction, so a node's operands are always already in scratch and no
 * dependence tracking is needed at run time.
 */
static void ir_run(const IRProgC *p, const uint64_t *values,
                   const uint64_t *taints, uint64_t *out_taints) {
    uint64_t *s = p->scratch;
    for (int i = 0; i < p->n_inputs; i++) {
        const IRInput *in = &p->inputs[i];
        s[in->node] = in->kind ? taints[in->slot] : values[in->slot];
    }
    const uint8_t *op = p->op;
    const int32_t *ai = p->a, *bi = p->b, *ci = p->c;
    const uint64_t *imm = p->imm;
    for (int n = 0; n < p->n_nodes; n++) {
        uint64_t a = ai[n] >= 0 ? s[ai[n]] : 0;
        uint64_t b = bi[n] >= 0 ? s[bi[n]] : 0;
        switch (op[n]) {
        case IR_CONST: s[n] = imm[n]; break;
        case IR_INV: case IR_INT: break;             /* preloaded above */
        case IR_AND: s[n] = a & b; break;
        case IR_OR:  s[n] = a | b; break;
        case IR_XOR: s[n] = a ^ b; break;
        case IR_ADD: s[n] = a + b; break;
        case IR_SUB: s[n] = a - b; break;
        case IR_MUL: s[n] = a * b; break;
        case IR_SHL: s[n] = (b < 64) ? (a << b) : 0; break;
        case IR_SHR: s[n] = (b < 64) ? (a >> b) : 0; break;
        case IR_SAR: s[n] = (b < 64) ? (uint64_t)(((int64_t)a) >> b)
                                     : (uint64_t)(((int64_t)a) >> 63); break;
        case IR_NOT: s[n] = ~a; break;
        case IR_NEG: s[n] = (uint64_t)(-(int64_t)a); break;
        case IR_ULT: s[n] = (a < b) ? 1 : 0; break;
        case IR_SLT: s[n] = ((int64_t)a < (int64_t)b) ? 1 : 0; break;
        case IR_EQ:  s[n] = (a == b) ? 1 : 0; break;
        case IR_NEZ: s[n] = a ? 1 : 0; break;
        case IR_SEL: s[n] = (ci[n] >= 0 && s[ci[n]]) ? a : b; break;
        case IR_POPCNT: s[n] = (uint64_t)__builtin_popcountll(a); break;
        case IR_CLZ: s[n] = a ? (uint64_t)__builtin_clzll(a) : 64; break;
        case IR_UDIV: s[n] = b ? a / b : 0; break;
        case IR_UREM: s[n] = b ? a % b : 0; break;
        case IR_SDIV: s[n] = b ? (uint64_t)((int64_t)a / (int64_t)b) : 0; break;
        case IR_SREM: s[n] = b ? (uint64_t)((int64_t)a % (int64_t)b) : 0; break;
        case IR_MULHI:
            s[n] = (uint64_t)(((unsigned __int128)a * (unsigned __int128)b) >> 64);
            break;
        default: s[n] = 0; break;
        }
    }
    for (int i = 0; i < p->n_outputs; i++)
        out_taints[p->outputs[i].slot] = s[p->outputs[i].node];
}

/* ── Python surface ──────────────────────────────────────────────────── */

static int fill_i32(int32_t **dst, PyObject *seq, Py_ssize_t n) {
    *dst = (int32_t *)malloc(sizeof(int32_t) * (size_t)(n ? n : 1));
    if (!*dst) return -1;
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *it = PySequence_GetItem(seq, i);
        if (!it) return -1;
        (*dst)[i] = (int32_t)PyLong_AsLong(it);
        Py_DECREF(it);
        if (PyErr_Occurred()) return -1;
    }
    return 0;
}

static PyObject *py_compile(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *d;
    if (!PyArg_ParseTuple(args, "O", &d)) return NULL;
    if (!PyDict_Check(d)) {
        PyErr_SetString(PyExc_TypeError, "expected the serialize() dict");
        return NULL;
    }
    PyObject *ops = PyDict_GetItemString(d, "op_ids");
    PyObject *a = PyDict_GetItemString(d, "a");
    PyObject *b = PyDict_GetItemString(d, "b");
    PyObject *c = PyDict_GetItemString(d, "c");
    PyObject *imm = PyDict_GetItemString(d, "imm");
    PyObject *ins = PyDict_GetItemString(d, "inputs");
    PyObject *outs = PyDict_GetItemString(d, "outputs");
    if (!ops || !a || !b || !c || !imm || !ins || !outs) {
        PyErr_SetString(PyExc_KeyError, "incomplete program dict");
        return NULL;
    }
    Py_ssize_t n = PySequence_Size(ops);
    IRProgC *p = (IRProgC *)calloc(1, sizeof(IRProgC));
    if (!p) return PyErr_NoMemory();
    p->n_nodes = (int)n;
    p->op = (uint8_t *)malloc((size_t)(n ? n : 1));
    p->imm = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)(n ? n : 1));
    p->scratch = (uint64_t *)calloc((size_t)(n ? n : 1), sizeof(uint64_t));
    if (!p->op || !p->imm || !p->scratch) { irprog_free(p); return PyErr_NoMemory(); }
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *it = PySequence_GetItem(ops, i);
        if (!it) { irprog_free(p); return NULL; }
        p->op[i] = (uint8_t)PyLong_AsLong(it);
        Py_DECREF(it);
        it = PySequence_GetItem(imm, i);
        if (!it) { irprog_free(p); return NULL; }
        p->imm[i] = (uint64_t)PyLong_AsUnsignedLongLongMask(it);
        Py_DECREF(it);
        if (PyErr_Occurred()) { irprog_free(p); return NULL; }
    }
    if (fill_i32(&p->a, a, n) < 0 || fill_i32(&p->b, b, n) < 0
            || fill_i32(&p->c, c, n) < 0) { irprog_free(p); return NULL; }

    Py_ssize_t ni = PySequence_Size(ins);
    p->n_inputs = (int)ni;
    p->inputs = (IRInput *)malloc(sizeof(IRInput) * (size_t)(ni ? ni : 1));
    if (!p->inputs) { irprog_free(p); return PyErr_NoMemory(); }
    for (Py_ssize_t i = 0; i < ni; i++) {
        PyObject *t = PySequence_GetItem(ins, i);
        if (!t) { irprog_free(p); return NULL; }
        int node, kind, slot;
        if (!PyArg_ParseTuple(t, "iii", &node, &kind, &slot)) {
            Py_DECREF(t); irprog_free(p); return NULL;
        }
        Py_DECREF(t);
        p->inputs[i].node = node; p->inputs[i].kind = kind;
        p->inputs[i].slot = slot;
    }
    Py_ssize_t no = PySequence_Size(outs);
    p->n_outputs = (int)no;
    p->outputs = (IROutput *)malloc(sizeof(IROutput) * (size_t)(no ? no : 1));
    if (!p->outputs) { irprog_free(p); return PyErr_NoMemory(); }
    for (Py_ssize_t i = 0; i < no; i++) {
        PyObject *t = PySequence_GetItem(outs, i);
        if (!t) { irprog_free(p); return NULL; }
        int slot, node;
        if (!PyArg_ParseTuple(t, "ii", &slot, &node)) {
            Py_DECREF(t); irprog_free(p); return NULL;
        }
        Py_DECREF(t);
        p->outputs[i].slot = slot; p->outputs[i].node = node;
    }
    return PyCapsule_New(p, "microtaint.taint_ir.prog", irprog_destructor);
}

static int load_slots(PyObject *seq, uint64_t *dst, int n) {
    for (int i = 0; i < n; i++) {
        PyObject *it = PySequence_GetItem(seq, i);
        if (!it) return -1;
        dst[i] = (uint64_t)PyLong_AsUnsignedLongLongMask(it);
        Py_DECREF(it);
        if (PyErr_Occurred()) return -1;
    }
    return 0;
}

static PyObject *py_run(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap, *vals, *tnts;
    if (!PyArg_ParseTuple(args, "OOO", &cap, &vals, &tnts)) return NULL;
    IRProgC *p = (IRProgC *)PyCapsule_GetPointer(cap, "microtaint.taint_ir.prog");
    if (!p) return NULL;
    int n = (int)PySequence_Size(vals);
    if (n < 0 || PySequence_Size(tnts) != n) {
        PyErr_SetString(PyExc_ValueError, "values and taints must be same length");
        return NULL;
    }
    uint64_t *v = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)(n ? n : 1));
    uint64_t *t = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)(n ? n : 1));
    if (!v || !t) { free(v); free(t); return PyErr_NoMemory(); }
    if (load_slots(vals, v, n) < 0 || load_slots(tnts, t, n) < 0) {
        free(v); free(t); return NULL;
    }
    if (p->jit_fn) p->jit_fn(v, t, t); else ir_run(p, v, t, t);
    PyObject *out = PyList_New(n);
    if (out) {
        for (int i = 0; i < n; i++)
            PyList_SET_ITEM(out, i, PyLong_FromUnsignedLongLong(t[i]));
    }
    free(v); free(t);
    return out;
}

/* Time the program itself: the state arrays are prepared once, so what is
 * measured is the taint propagation and nothing around it. */
static PyObject *py_bench(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap, *vals, *tnts;
    long iters = 10000;
    if (!PyArg_ParseTuple(args, "OOO|l", &cap, &vals, &tnts, &iters)) return NULL;
    IRProgC *p = (IRProgC *)PyCapsule_GetPointer(cap, "microtaint.taint_ir.prog");
    if (!p) return NULL;
    int n = (int)PySequence_Size(vals);
    uint64_t *v = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)(n ? n : 1));
    uint64_t *t = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)(n ? n : 1));
    uint64_t *o = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)(n ? n : 1));
    if (!v || !t || !o) { free(v); free(t); free(o); return PyErr_NoMemory(); }
    if (load_slots(vals, v, n) < 0 || load_slots(tnts, t, n) < 0) {
        free(v); free(t); free(o); return NULL;
    }
    struct timespec t0, t1;
    Py_BEGIN_ALLOW_THREADS
    clock_gettime(CLOCK_MONOTONIC, &t0);
    /* The output array is seeded once, not per iteration: what is being timed
     * is the taint program, and a per-iteration copy of the register file would
     * add several nanoseconds of unrelated work to a program that may itself
     * take three. */
    memcpy(o, t, sizeof(uint64_t) * (size_t)n);
    if (p->jit_fn) {
        for (long i = 0; i < iters; i++) p->jit_fn(v, t, o);
    } else {
        for (long i = 0; i < iters; i++) ir_run(p, v, t, o);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    Py_END_ALLOW_THREADS
    double ns = ((double)(t1.tv_sec - t0.tv_sec) * 1e9
                 + (double)(t1.tv_nsec - t0.tv_nsec)) / (double)iters;
    uint64_t sink = 0;
    for (int i = 0; i < n; i++) sink ^= o[i];
    free(v); free(t); free(o);
    return Py_BuildValue("(dK)", ns, (unsigned long long)sink);
}

/* Compile the program to native code.  Returns True when the emitter took it;
 * False is not an error -- the interpreter stays correct and is used instead. */
static PyObject *py_jit(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    IRProgC *p = (IRProgC *)PyCapsule_GetPointer(cap, "microtaint.taint_ir.prog");
    if (!p) return NULL;
#ifdef MT_HAVE_JIT
    if (p->jit_fn) Py_RETURN_TRUE;
    void *code = NULL; size_t sz = 0;
    mt_taint_fn fn = mt_jit_compile(p, &code, &sz);
    if (!fn) Py_RETURN_FALSE;
    p->jit_fn = fn; p->jit_code = code; p->jit_size = sz;
    Py_RETURN_TRUE;
#else
    Py_RETURN_FALSE;
#endif
}

static PyObject *py_jit_size(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    IRProgC *p = (IRProgC *)PyCapsule_GetPointer(cap, "microtaint.taint_ir.prog");
    if (!p) return NULL;
    return PyLong_FromLong(p->jit_fn ? (long)p->jit_size : 0L);
}

/* Address of the emitted function, for a caller that will invoke it directly
 * from its own C code.  Zero when the program was not emitted. */
static PyObject *py_fn_addr(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    IRProgC *p = (IRProgC *)PyCapsule_GetPointer(cap, "microtaint.taint_ir.prog");
    if (!p) return NULL;
    return PyLong_FromVoidPtr((void *)p->jit_fn);
}

static PyObject *py_n_nodes(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    IRProgC *p = (IRProgC *)PyCapsule_GetPointer(cap, "microtaint.taint_ir.prog");
    if (!p) return NULL;
    return PyLong_FromLong(p->n_nodes);
}

static PyMethodDef methods[] = {
    {"compile", py_compile, METH_VARARGS, "compile a serialized IR program"},
    {"run", py_run, METH_VARARGS, "run once; returns the updated taint slots"},
    {"bench", py_bench, METH_VARARGS, "time the program; returns (ns, sink)"},
    {"n_nodes", py_n_nodes, METH_VARARGS, "node count of a compiled program"},
    {"jit", py_jit, METH_VARARGS, "emit native code; True if the host emitter took it"},
    {"jit_size", py_jit_size, METH_VARARGS, "bytes of native code, or 0"},
    {"fn_addr", py_fn_addr, METH_VARARGS, "address of the emitted function, or 0"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "taint_ir_c", NULL, -1, methods
};

PyMODINIT_FUNC PyInit_taint_ir_c(void) {
    return PyModule_Create(&moduledef);
}
