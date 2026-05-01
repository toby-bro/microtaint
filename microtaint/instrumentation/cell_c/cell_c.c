/*
 * cell_c.c — pure-C drop-in replacement for PCodeCellEvaluator.
 *
 * Public API matches microtaint.instrumentation.cell.PCodeCellEvaluator:
 *   evaluate_concrete(cell, flat_inputs)        -> int  (or raises PCodeFallbackNeeded)
 *   evaluate_concrete_state(cell, regs, mem)    -> int
 *   evaluate_differential(cell, or_in, and_in)  -> int
 *   stats() -> {'native_calls', 'fallback_calls', 'fallback_rate'}
 *   .native_calls / .fallback_calls properties (read/write)
 *
 * Architecture-aware: register table is built per-instance from
 * microtaint.instrumentation.cell._build_reg_maps(arch).
 *
 * x86 EFLAGS quirk: the Sleigh spec writes individual flag registers
 * (CF@512, PF@514, ZF@518, SF@519, DF@522, OF@523).  When the user
 * reads EFLAGS (offset 640, size 4) we reconstruct it from those, mirroring
 * cell.pyx _read_output exactly.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>
#include "cell_core.h"

/* ──────────── Per-architecture register table ──────────── */

typedef struct {
    char name[16];
    int  off;
    int  sz;
} RegEntry;

#define MAX_REGS 2048

/* Open-addressing hash for register names → index in reg_table.
 * MAX_REGS=2048, hash capacity 4096 for ~50% load factor. */
#define REG_HASH_CAP   4096
#define REG_HASH_MASK  (REG_HASH_CAP - 1)
#define REG_HASH_EMPTY -1

/* FNV-1a 32-bit string hash */
static uint32_t name_hash(const char *s) {
    uint32_t h = 2166136261u;
    while (*s) { h ^= (uint8_t)*s++; h *= 16777619u; }
    return h;
}

static int reg_offset_in(const RegEntry *t, int n, const char *name) {
    /* Kept for fallback paths; hot path uses reg_lookup() below. */
    for (int i = 0; i < n; i++) if (strcmp(t[i].name, name) == 0) return t[i].off;
    return -1;
}
static int reg_size_in(const RegEntry *t, int n, const char *name) {
    for (int i = 0; i < n; i++) if (strcmp(t[i].name, name) == 0) return t[i].sz;
    return 8;
}

/* Uppercase a string into a small buffer (returns true if it fit) */
static int upper_into(char *dst, int dst_sz, const char *src) {
    int i;
    for (i = 0; i < dst_sz - 1 && src[i]; i++) {
        char c = src[i];
        dst[i] = (c >= 'a' && c <= 'z') ? c - 32 : c;
    }
    dst[i] = 0;
    return src[i] == 0;
}

/* ──────────── Object type ──────────── */

typedef struct {
    PyObject_HEAD
    Frame    frame_a;
    Frame    frame_b;
    PyObject *arch;
    PyObject *get_decoded_fn;
    PyObject *fallback_exc;
    PyObject *bundle_cache;     /* dict: bytes -> capsule */
    int      native_calls;
    int      fallback_calls;
    RegEntry reg_table[MAX_REGS];
    int      n_regs;
    /* Hash table: maps name → index in reg_table (-1 = empty slot) */
    int      reg_hash[REG_HASH_CAP];
    int      pc_off;
    int      pc_sz;
    /* Cached EFLAGS reconstruction info for fast x86 path */
    int      eflags_off;        /* -1 if not x86 */
    int      cf_off, pf_off, zf_off, sf_off, df_off, of_off;
} EvalC;

/* O(1) hash-table register lookup. Returns index in reg_table or -1. */
static inline int reg_lookup(const EvalC *self, const char *name) {
    uint32_t h = name_hash(name) & REG_HASH_MASK;
    for (int probes = 0; probes < REG_HASH_CAP; probes++) {
        int idx = self->reg_hash[h];
        if (idx == REG_HASH_EMPTY) return -1;
        if (strcmp(self->reg_table[idx].name, name) == 0) return idx;
        h = (h + 1) & REG_HASH_MASK;
    }
    return -1;
}

/* Get (offset, size) from name; returns 1 on success, 0 on miss. */
static inline int reg_off_size(const EvalC *self, const char *name, int *off, int *sz) {
    int idx = reg_lookup(self, name);
    if (idx < 0) return 0;
    *off = self->reg_table[idx].off;
    *sz  = self->reg_table[idx].sz;
    return 1;
}

/* Insert a name → index mapping into the hash table */
static void reg_hash_insert(EvalC *self, const char *name, int idx) {
    uint32_t h = name_hash(name) & REG_HASH_MASK;
    for (int probes = 0; probes < REG_HASH_CAP; probes++) {
        if (self->reg_hash[h] == REG_HASH_EMPTY) {
            self->reg_hash[h] = idx;
            return;
        }
        h = (h + 1) & REG_HASH_MASK;
    }
}

static PyTypeObject EvalCType;

/* ──────────── Bundle cache ──────────── */

static void bundle_destructor(PyObject *cap) {
    free(PyCapsule_GetPointer(cap, "DecodedBundle"));
}

static DecodedBundle *get_bundle(EvalC *self, PyObject *bytestring) {
    PyObject *cap = PyDict_GetItem(self->bundle_cache, bytestring);
    if (cap) return (DecodedBundle *)PyCapsule_GetPointer(cap, "DecodedBundle");

    PyObject *decoded_obj = PyObject_CallFunctionObjArgs(
        self->get_decoded_fn, self->arch, bytestring, NULL);
    if (!decoded_obj) return NULL;

    PyObject *buf_bytes = PyObject_CallMethod(decoded_obj, "get_buf_bytes", NULL);
    if (!buf_bytes) { Py_DECREF(decoded_obj); return NULL; }

    DecodedBundle *bundle = (DecodedBundle *)malloc(sizeof(DecodedBundle));
    if (!bundle) { Py_DECREF(buf_bytes); Py_DECREF(decoded_obj); PyErr_NoMemory(); return NULL; }

    PyObject *tmp;
    tmp = PyObject_GetAttrString(decoded_obj, "n_ops");
    bundle->n_ops = (int)PyLong_AsLong(tmp); Py_DECREF(tmp);
    tmp = PyObject_GetAttrString(decoded_obj, "has_fallback");
    bundle->has_fallback = PyObject_IsTrue(tmp); Py_DECREF(tmp);
    tmp = PyObject_GetAttrString(decoded_obj, "next_instr_addr");
    bundle->next_instr_addr = (uint64_t)PyLong_AsUnsignedLongLong(tmp); Py_DECREF(tmp);

    Py_buffer view;
    if (PyObject_GetBuffer(buf_bytes, &view, PyBUF_SIMPLE) == 0) {
        Py_ssize_t copy_sz = view.len < (Py_ssize_t)sizeof(bundle->buf)
                             ? view.len : (Py_ssize_t)sizeof(bundle->buf);
        memcpy(bundle->buf, view.buf, copy_sz);
        PyBuffer_Release(&view);
    }
    Py_DECREF(buf_bytes);
    Py_DECREF(decoded_obj);

    PyObject *new_cap = PyCapsule_New(bundle, "DecodedBundle", bundle_destructor);
    if (!new_cap) { free(bundle); return NULL; }
    PyDict_SetItem(self->bundle_cache, bytestring, new_cap);
    Py_DECREF(new_cap);
    return bundle;
}

/* ──────────── Frame loaders ──────────── */

static void load_regs_state(EvalC *self, Frame *f, PyObject *regs_dict) {
    PyObject *key, *val;
    Py_ssize_t pos = 0;
    while (PyDict_Next(regs_dict, &pos, &key, &val)) {
        if (!PyUnicode_Check(key)) continue;
        const char *name = PyUnicode_AsUTF8(key);
        if (!name) continue;
        char up[16]; if (!upper_into(up, sizeof(up), name)) continue;
        int off, sz;
        if (!reg_off_size(self, up, &off, &sz)) continue;
        uint64_t v = (uint64_t)(PyLong_AsUnsignedLongLong(val) & 0xFFFFFFFFFFFFFFFFULL);
        if (PyErr_Occurred()) PyErr_Clear();
        frame_write_reg(f, off, sz, mask64(v, sz));
    }
}

static void load_mem_state(EvalC *self, Frame *f, PyObject *mem_dict) {
    PyObject *addr_obj, *val_obj;
    Py_ssize_t pos = 0;
    while (PyDict_Next(mem_dict, &pos, &addr_obj, &val_obj)) {
        uint64_t addr = (uint64_t)PyLong_AsUnsignedLongLong(addr_obj);
        if (PyErr_Occurred()) PyErr_Clear();
        uint64_t mval = (uint64_t)(PyLong_AsUnsignedLongLong(val_obj) & 0xFFFFFFFFFFFFFFFFULL);
        if (PyErr_Occurred()) PyErr_Clear();
        int msz;
        if (mval == 0) {
            msz = 8;
        } else {
            int bl = 64 - __builtin_clzll(mval);
            msz = (bl + 7) / 8;
            if (msz < 1) msz = 1;
            if (msz > 8) msz = 8;
        }
        mem_write(&f->mem, addr, mval, msz);
    }
}

static int load_flat(EvalC *self, Frame *f, PyObject *inputs_dict) {
    PyObject *key, *val;
    Py_ssize_t pos = 0;
    PyObject *deferred = PyList_New(0);
    if (!deferred) return -1;

    /* Pass 1: registers */
    while (PyDict_Next(inputs_dict, &pos, &key, &val)) {
        if (!PyUnicode_Check(key)) continue;
        const char *name = PyUnicode_AsUTF8(key);
        if (!name) continue;
        if (strncmp(name, "MEM_", 4) == 0) {
            PyList_Append(deferred, key);
            continue;
        }
        char up[16]; if (!upper_into(up, sizeof(up), name)) continue;
        int off, sz;
        if (!reg_off_size(self, up, &off, &sz)) continue;
        uint64_t v = (uint64_t)(PyLong_AsUnsignedLongLong(val) & 0xFFFFFFFFFFFFFFFFULL);
        if (PyErr_Occurred()) PyErr_Clear();
        frame_write_reg(f, off, sz, mask64(v, sz));
    }

    /* Pass 2: MEM_ entries */
    Py_ssize_t n = PyList_GET_SIZE(deferred);
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *k = PyList_GET_ITEM(deferred, i);
        PyObject *vobj = PyDict_GetItem(inputs_dict, k);
        if (!vobj) continue;
        uint64_t v = (uint64_t)(PyLong_AsUnsignedLongLong(vobj) & 0xFFFFFFFFFFFFFFFFULL);
        if (PyErr_Occurred()) PyErr_Clear();
        const char *name = PyUnicode_AsUTF8(k);
        if (!name) continue;
        const char *body = name + 4;
        const char *last_us = strrchr(body, '_');
        if (!last_us) continue;
        int size = atoi(last_us + 1);

        char head[128];
        Py_ssize_t hlen = last_us - body;
        if (hlen <= 0 || hlen >= (Py_ssize_t)sizeof(head)) continue;
        memcpy(head, body, hlen); head[hlen] = 0;

        uint64_t addr;
        if (head[0] == '0' && head[1] == 'x') {
            addr = (uint64_t)strtoull(head, NULL, 16);
        } else if (head[0] == '-' && head[1] == '0' && head[2] == 'x') {
            addr = (uint64_t)(int64_t)strtoll(head, NULL, 16);
        } else {
            const char *inner = strrchr(head, '_');
            if (!inner) continue;
            char regname[64];
            Py_ssize_t rlen = inner - head;
            if (rlen <= 0 || rlen >= (Py_ssize_t)sizeof(regname)) continue;
            for (int j = 0; j < rlen; j++) {
                char c = head[j];
                regname[j] = (c >= 'a' && c <= 'z') ? c - 32 : c;
            }
            regname[rlen] = 0;
            int64_t offset = (int64_t)atoll(inner + 1);
            int roff, rsz;
            if (!reg_off_size(self, regname, &roff, &rsz)) continue;
            uint64_t base = frame_read_reg(f, roff, rsz);
            addr = base + (uint64_t)offset;
        }
        mem_write(&f->mem, addr, v, size);
    }
    Py_DECREF(deferred);
    return 0;
}

/* Read output: handles register, MEM_<hex>, MEM_<reg>_<off>, plus x86
 * EFLAGS reconstruction from individual flag registers. */
static uint64_t read_output_full(EvalC *self, Frame *f,
                                  const char *out_reg, int bit_start, int bit_end) {
    int width = bit_end - bit_start + 1;
    uint64_t val, m;

    if (strncmp(out_reg, "MEM_", 4) == 0) {
        const char *body = out_reg + 4;
        const char *last_us = strrchr(body, '_');
        if (!last_us) return 0;
        int size = atoi(last_us + 1);
        char head[128];
        Py_ssize_t hlen = last_us - body;
        if (hlen <= 0 || hlen >= (Py_ssize_t)sizeof(head)) return 0;
        memcpy(head, body, hlen); head[hlen] = 0;

        uint64_t addr;
        if (head[0] == '0' && head[1] == 'x') {
            addr = (uint64_t)strtoull(head, NULL, 16);
        } else if (head[0] == '-' && head[1] == '0' && head[2] == 'x') {
            addr = (uint64_t)(int64_t)strtoll(head, NULL, 16);
        } else {
            const char *inner = strrchr(head, '_');
            if (!inner) return 0;
            char regname[64];
            Py_ssize_t rlen = inner - head;
            if (rlen <= 0 || rlen >= (Py_ssize_t)sizeof(regname)) return 0;
            for (int j = 0; j < rlen; j++) {
                char c = head[j];
                regname[j] = (c >= 'a' && c <= 'z') ? c - 32 : c;
            }
            regname[rlen] = 0;
            int64_t offset = (int64_t)atoll(inner + 1);
            int roff, rsz;
            if (!reg_off_size(self, regname, &roff, &rsz)) return 0;
            uint64_t base = frame_read_reg(f, roff, rsz);
            addr = base + (uint64_t)offset;
        }
        val = mem_read(&f->mem, addr, size);
        if (width >= 64) return val >> bit_start;
        m = ((uint64_t)1 << width) - 1;
        return (val >> bit_start) & m;
    }

    char up[16];
    if (!upper_into(up, sizeof(up), out_reg)) return 0;
    int off, sz;
    if (!reg_off_size(self, up, &off, &sz)) return 0;
    val = frame_read_reg(f, off, sz);

    /* x86 EFLAGS reconstruction (mirrors cell.pyx _read_output exactly):
     * EFLAGS@640 size 4 is never written by Sleigh; rebuild from per-flag regs. */
    if (val == 0 && self->eflags_off >= 0 && off == self->eflags_off && sz == 4) {
        val = 0;
        if (self->cf_off >= 0) val |= frame_read_reg(f, self->cf_off, 1) <<  0;
        if (self->pf_off >= 0) val |= frame_read_reg(f, self->pf_off, 1) <<  2;
        if (self->zf_off >= 0) val |= frame_read_reg(f, self->zf_off, 1) <<  6;
        if (self->sf_off >= 0) val |= frame_read_reg(f, self->sf_off, 1) <<  7;
        if (self->df_off >= 0) val |= frame_read_reg(f, self->df_off, 1) << 10;
        if (self->of_off >= 0) val |= frame_read_reg(f, self->of_off, 1) << 11;
    }

    if (width >= 64) return val >> bit_start;
    m = ((uint64_t)1 << width) - 1;
    return (val >> bit_start) & m;
}

/* ──────────── Type lifecycle ──────────── */

static PyObject *EvalC_new(PyTypeObject *type, PyObject *args, PyObject *kw) {
    (void)args; (void)kw;
    EvalC *self = (EvalC *)type->tp_alloc(type, 0);
    if (!self) return NULL;
    memset(&self->frame_a, 0, sizeof(Frame));
    memset(&self->frame_b, 0, sizeof(Frame));
    mem_clear(&self->frame_a.mem);
    mem_clear(&self->frame_b.mem);
    self->arch = NULL;
    self->get_decoded_fn = NULL;
    self->fallback_exc = NULL;
    self->bundle_cache = PyDict_New();
    self->native_calls = 0;
    self->fallback_calls = 0;
    self->n_regs = 0;
    for (int i = 0; i < REG_HASH_CAP; i++) self->reg_hash[i] = REG_HASH_EMPTY;
    self->pc_off = 0;
    self->pc_sz = 0;
    self->eflags_off = -1;
    self->cf_off = self->pf_off = self->zf_off = self->sf_off = self->df_off = self->of_off = -1;
    return (PyObject *)self;
}

static int EvalC_init(EvalC *self, PyObject *args, PyObject *kw) {
    (void)kw;
    PyObject *arch;
    if (!PyArg_ParseTuple(args, "O", &arch)) return -1;
    Py_XDECREF(self->arch);
    self->arch = arch; Py_INCREF(arch);

    PyObject *cell_mod = PyImport_ImportModule("microtaint.instrumentation.cell");
    if (!cell_mod) return -1;
    self->get_decoded_fn = PyObject_GetAttrString(cell_mod, "_get_decoded");
    self->fallback_exc   = PyObject_GetAttrString(cell_mod, "PCodeFallbackNeeded");
    PyObject *build_fn   = PyObject_GetAttrString(cell_mod, "_build_reg_maps");
    Py_DECREF(cell_mod);
    if (!self->get_decoded_fn || !self->fallback_exc || !build_fn) {
        Py_XDECREF(build_fn); return -1;
    }

    PyObject *result = PyObject_CallFunctionObjArgs(build_fn, arch, NULL);
    Py_DECREF(build_fn);
    if (!result || !PyTuple_Check(result) || PyTuple_GET_SIZE(result) < 2) {
        Py_XDECREF(result); return -1;
    }
    PyObject *offsets = PyTuple_GET_ITEM(result, 0);
    PyObject *sizes   = PyTuple_GET_ITEM(result, 1);

    PyObject *key, *val;
    Py_ssize_t pos = 0;
    self->n_regs = 0;
    while (PyDict_Next(offsets, &pos, &key, &val) && self->n_regs < MAX_REGS) {
        if (!PyUnicode_Check(key)) continue;
        const char *name = PyUnicode_AsUTF8(key);
        if (!name) continue;
        int off = (int)PyLong_AsLong(val);
        if (PyErr_Occurred()) { PyErr_Clear(); continue; }
        PyObject *sz_obj = PyDict_GetItem(sizes, key);
        int sz = sz_obj ? (int)PyLong_AsLong(sz_obj) : 8;
        if (PyErr_Occurred()) { PyErr_Clear(); sz = 8; }
        RegEntry *e = &self->reg_table[self->n_regs++];
        size_t nlen = strlen(name);
        if (nlen >= sizeof(e->name)) nlen = sizeof(e->name) - 1;
        memcpy(e->name, name, nlen);
        e->name[nlen] = 0;
        for (size_t i = 0; i < nlen; i++)
            if (e->name[i] >= 'a' && e->name[i] <= 'z') e->name[i] -= 32;
        e->off = off;
        e->sz  = sz;
        reg_hash_insert(self, e->name, self->n_regs - 1);
    }
    Py_DECREF(result);

    /* PC: try RIP, EIP, PC */
    static const char *PC_NAMES[] = {"RIP","EIP","PC",NULL};
    for (int i = 0; PC_NAMES[i]; i++) {
        int off = reg_offset_in(self->reg_table, self->n_regs, PC_NAMES[i]);
        if (off >= 0) {
            self->pc_off = off;
            self->pc_sz  = reg_size_in(self->reg_table, self->n_regs, PC_NAMES[i]);
            self->frame_a.arch_pc_off = off;
            self->frame_a.arch_pc_sz  = self->pc_sz;
            self->frame_b.arch_pc_off = off;
            self->frame_b.arch_pc_sz  = self->pc_sz;
            break;
        }
    }

    /* x86 flag register offsets (set if EFLAGS exists in the table) */
    self->eflags_off = reg_offset_in(self->reg_table, self->n_regs, "EFLAGS");
    if (self->eflags_off >= 0) {
        self->cf_off = reg_offset_in(self->reg_table, self->n_regs, "CF");
        self->pf_off = reg_offset_in(self->reg_table, self->n_regs, "PF");
        self->zf_off = reg_offset_in(self->reg_table, self->n_regs, "ZF");
        self->sf_off = reg_offset_in(self->reg_table, self->n_regs, "SF");
        self->df_off = reg_offset_in(self->reg_table, self->n_regs, "DF");
        self->of_off = reg_offset_in(self->reg_table, self->n_regs, "OF");
    }

    return 0;
}

static void EvalC_dealloc(EvalC *self) {
    Py_XDECREF(self->arch);
    Py_XDECREF(self->get_decoded_fn);
    Py_XDECREF(self->fallback_exc);
    Py_XDECREF(self->bundle_cache);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/* ──────────── Common: hex → bytes for instruction key ──────────── */

static PyObject *hex_to_bytes(const char *hex) {
    Py_ssize_t hlen = strlen(hex);
    Py_ssize_t blen = hlen / 2;
    if (blen > 32) blen = 32;
    uint8_t buf[32];
    for (Py_ssize_t i = 0; i < blen; i++) {
        char p[3] = {hex[i*2], hex[i*2+1], 0};
        buf[i] = (uint8_t)strtoul(p, NULL, 16);
    }
    return PyBytes_FromStringAndSize((char*)buf, blen);
}

/* Read cell.instruction (hex string) → bytes object, NULL on error */
static PyObject *cell_instr_bytes(PyObject *cell_obj) {
    PyObject *hex = PyObject_GetAttrString(cell_obj, "instruction");
    if (!hex) return NULL;
    const char *s = PyUnicode_AsUTF8(hex);
    if (!s) { Py_DECREF(hex); return NULL; }
    PyObject *r = hex_to_bytes(s);
    Py_DECREF(hex);
    return r;
}

/* Read cell.out_reg / out_bit_start / out_bit_end into out params.
 * Returns newly-allocated copy of out_reg in *out_reg_copy (caller frees).
 * On error returns -1 and sets exception. */
static int cell_output_info(PyObject *cell_obj, char **out_reg_copy,
                             int *bit_start, int *bit_end) {
    *out_reg_copy = NULL;
    PyObject *r = PyObject_GetAttrString(cell_obj, "out_reg");
    PyObject *bs = PyObject_GetAttrString(cell_obj, "out_bit_start");
    PyObject *be = PyObject_GetAttrString(cell_obj, "out_bit_end");
    if (!r || !bs || !be) {
        Py_XDECREF(r); Py_XDECREF(bs); Py_XDECREF(be);
        return -1;
    }
    const char *s = PyUnicode_AsUTF8(r);
    if (!s) { Py_DECREF(r); Py_DECREF(bs); Py_DECREF(be); return -1; }
    *out_reg_copy = strdup(s);
    *bit_start = (int)PyLong_AsLong(bs);
    *bit_end   = (int)PyLong_AsLong(be);
    Py_DECREF(r); Py_DECREF(bs); Py_DECREF(be);
    return 0;
}

/* ──────────── Methods ──────────── */

static PyObject *EvalC_evaluate_concrete(EvalC *self, PyObject *args) {
    PyObject *cell_obj, *flat_inputs;
    if (!PyArg_ParseTuple(args, "OO", &cell_obj, &flat_inputs)) return NULL;

    PyObject *ib = cell_instr_bytes(cell_obj);
    if (!ib) return NULL;
    DecodedBundle *bundle = get_bundle(self, ib);
    Py_DECREF(ib);
    if (!bundle) { self->fallback_calls++; return NULL; }
    if (bundle->has_fallback) {
        self->fallback_calls++;
        PyErr_SetString(self->fallback_exc, "instruction requires Unicorn");
        return NULL;
    }

    char *out_reg = NULL; int bs, be;
    if (cell_output_info(cell_obj, &out_reg, &bs, &be) < 0) return NULL;

    Frame *f = &self->frame_a;
    frame_clear(f);
    if (load_flat(self, f, flat_inputs) < 0) { free(out_reg); return NULL; }

    if (execute_decoded(f, bundle) == EXEC_FALLBACK) {
        free(out_reg);
        self->fallback_calls++;
        PyErr_SetString(self->fallback_exc, "execution requires Unicorn");
        return NULL;
    }
    uint64_t out = read_output_full(self, f, out_reg, bs, be);
    free(out_reg);
    self->native_calls++;
    return PyLong_FromUnsignedLongLong(out);
}

static PyObject *EvalC_evaluate_concrete_state(EvalC *self, PyObject *args) {
    PyObject *cell_obj, *regs_dict, *mem_dict;
    if (!PyArg_ParseTuple(args, "OOO", &cell_obj, &regs_dict, &mem_dict)) return NULL;

    PyObject *ib = cell_instr_bytes(cell_obj);
    if (!ib) return NULL;
    DecodedBundle *bundle = get_bundle(self, ib);
    Py_DECREF(ib);
    if (!bundle) { self->fallback_calls++; return NULL; }
    if (bundle->has_fallback) {
        self->fallback_calls++;
        PyErr_SetString(self->fallback_exc, "instruction requires Unicorn");
        return NULL;
    }

    char *out_reg = NULL; int bs, be;
    if (cell_output_info(cell_obj, &out_reg, &bs, &be) < 0) return NULL;

    Frame *f = &self->frame_a;
    frame_clear(f);
    load_regs_state(self, f, regs_dict);
    load_mem_state(self, f, mem_dict);

    if (execute_decoded(f, bundle) == EXEC_FALLBACK) {
        free(out_reg);
        self->fallback_calls++;
        PyErr_SetString(self->fallback_exc, "execution requires Unicorn");
        return NULL;
    }
    uint64_t out = read_output_full(self, f, out_reg, bs, be);
    free(out_reg);
    self->native_calls++;
    return PyLong_FromUnsignedLongLong(out);
}

static PyObject *EvalC_evaluate_differential(EvalC *self, PyObject *args) {
    PyObject *cell_obj, *or_inputs, *and_inputs;
    if (!PyArg_ParseTuple(args, "OOO", &cell_obj, &or_inputs, &and_inputs)) return NULL;

    PyObject *ib = cell_instr_bytes(cell_obj);
    if (!ib) return NULL;
    DecodedBundle *bundle = get_bundle(self, ib);
    Py_DECREF(ib);
    if (!bundle) { self->fallback_calls++; return NULL; }
    if (bundle->has_fallback) {
        self->fallback_calls++;
        PyErr_SetString(self->fallback_exc, "instruction requires Unicorn");
        return NULL;
    }

    char *out_reg = NULL; int bs, be;
    if (cell_output_info(cell_obj, &out_reg, &bs, &be) < 0) return NULL;

    Frame *fa = &self->frame_a;
    Frame *fb = &self->frame_b;
    frame_clear(fa);
    frame_clear(fb);
    if (load_flat(self, fa, or_inputs)  < 0) { free(out_reg); return NULL; }
    if (load_flat(self, fb, and_inputs) < 0) { free(out_reg); return NULL; }

    int rc_a = execute_decoded(fa, bundle);
    int rc_b = execute_decoded(fb, bundle);
    if (rc_a == EXEC_FALLBACK || rc_b == EXEC_FALLBACK) {
        free(out_reg);
        self->fallback_calls++;
        PyErr_SetString(self->fallback_exc, "execution requires Unicorn");
        return NULL;
    }
    uint64_t va = read_output_full(self, fa, out_reg, bs, be);
    uint64_t vb = read_output_full(self, fb, out_reg, bs, be);
    free(out_reg);
    self->native_calls++;
    return PyLong_FromUnsignedLongLong(va ^ vb);
}

/* evaluate_concrete_flat(cell, flat_inputs, ctx_input_values=None) -> int
 *
 * Same as evaluate_concrete but takes a flat dict in cell.pyx format directly,
 * skipping the MachineState construction in ast.pyx _build_machine_state.
 *
 * The optional ctx_input_values is consulted only for resolving MEM_<reg>_<off>_<size>
 * keys when no equivalent MEM_ key is in flat_inputs (currently load_flat handles
 * this from the dict itself, so ctx_input_values is reserved for future use).
 *
 * Caller invariant: flat_inputs uses the same key format as evaluate_concrete:
 *   "REG_NAME" -> int        (uppercase or lowercase OK)
 *   "MEM_<hex>_<size>"       static address
 *   "MEM_<reg>_<signed_off>_<size>"  register-relative
 */
static PyObject *EvalC_evaluate_concrete_flat(EvalC *self, PyObject *args) {
    PyObject *cell_obj, *flat_inputs;
    PyObject *_unused = NULL;
    if (!PyArg_ParseTuple(args, "OO|O", &cell_obj, &flat_inputs, &_unused)) return NULL;

    PyObject *ib = cell_instr_bytes(cell_obj);
    if (!ib) return NULL;
    DecodedBundle *bundle = get_bundle(self, ib);
    Py_DECREF(ib);
    if (!bundle) { self->fallback_calls++; return NULL; }
    if (bundle->has_fallback) {
        self->fallback_calls++;
        PyErr_SetString(self->fallback_exc, "instruction requires Unicorn");
        return NULL;
    }

    char *out_reg = NULL; int bs, be;
    if (cell_output_info(cell_obj, &out_reg, &bs, &be) < 0) return NULL;

    Frame *f = &self->frame_a;
    frame_clear(f);
    if (load_flat(self, f, flat_inputs) < 0) { free(out_reg); return NULL; }

    if (execute_decoded(f, bundle) == EXEC_FALLBACK) {
        free(out_reg);
        self->fallback_calls++;
        PyErr_SetString(self->fallback_exc, "execution requires Unicorn");
        return NULL;
    }
    uint64_t out = read_output_full(self, f, out_reg, bs, be);
    free(out_reg);
    self->native_calls++;
    return PyLong_FromUnsignedLongLong(out);
}

static PyObject *EvalC_stats(EvalC *self, PyObject *_unused) {
    (void)_unused;
    double total = self->native_calls + self->fallback_calls;
    double rate  = total > 0 ? self->fallback_calls / total : 0.0;
    return Py_BuildValue("{s:i,s:i,s:d}",
        "native_calls",   self->native_calls,
        "fallback_calls", self->fallback_calls,
        "fallback_rate",  rate);
}

static PyObject *EvalC_get_native_calls(EvalC *self, void *_) { (void)_; return PyLong_FromLong(self->native_calls); }
static PyObject *EvalC_get_fallback_calls(EvalC *self, void *_) { (void)_; return PyLong_FromLong(self->fallback_calls); }
static int EvalC_set_native_calls(EvalC *self, PyObject *v, void *_) { (void)_; self->native_calls = (int)PyLong_AsLong(v); return 0; }
static int EvalC_set_fallback_calls(EvalC *self, PyObject *v, void *_) { (void)_; self->fallback_calls = (int)PyLong_AsLong(v); return 0; }
static PyObject *EvalC_get_fallback_rate(EvalC *self, void *_) {
    (void)_;
    double total = self->native_calls + self->fallback_calls;
    return PyFloat_FromDouble(total > 0 ? self->fallback_calls / total : 0.0);
}
/* _offsets / _sizes: Python dicts built from the reg_table for compat with
 * tests and external callers that read them.  Built lazily on access. */
static PyObject *EvalC_build_offsets(EvalC *self, int want_sizes) {
    PyObject *d = PyDict_New();
    if (!d) return NULL;
    for (int i = 0; i < self->n_regs; i++) {
        const RegEntry *e = &self->reg_table[i];
        PyObject *k = PyUnicode_FromString(e->name);
        PyObject *v = PyLong_FromLong(want_sizes ? e->sz : e->off);
        if (!k || !v) { Py_XDECREF(k); Py_XDECREF(v); Py_DECREF(d); return NULL; }
        PyDict_SetItem(d, k, v);
        Py_DECREF(k); Py_DECREF(v);
    }
    return d;
}
static PyObject *EvalC_get_offsets(EvalC *self, void *_) { (void)_; return EvalC_build_offsets(self, 0); }
static PyObject *EvalC_get_sizes  (EvalC *self, void *_) { (void)_; return EvalC_build_offsets(self, 1); }

static PyGetSetDef EvalC_getset[] = {
    {"native_calls",   (getter)EvalC_get_native_calls,   (setter)EvalC_set_native_calls,   NULL, NULL},
    {"fallback_calls", (getter)EvalC_get_fallback_calls, (setter)EvalC_set_fallback_calls, NULL, NULL},
    {"fallback_rate",  (getter)EvalC_get_fallback_rate,  NULL, NULL, NULL},
    {"_offsets",       (getter)EvalC_get_offsets,        NULL, NULL, NULL},
    {"_sizes",         (getter)EvalC_get_sizes,          NULL, NULL, NULL},
    {NULL}
};

static PyMethodDef EvalC_methods[] = {
    {"evaluate_differential",   (PyCFunction)EvalC_evaluate_differential,   METH_VARARGS, NULL},
    {"evaluate_concrete",       (PyCFunction)EvalC_evaluate_concrete,       METH_VARARGS, NULL},
    {"evaluate_concrete_state", (PyCFunction)EvalC_evaluate_concrete_state, METH_VARARGS, NULL},
    {"evaluate_concrete_flat",  (PyCFunction)EvalC_evaluate_concrete_flat,  METH_VARARGS, NULL},
    {"stats",                   (PyCFunction)EvalC_stats,                   METH_NOARGS,  NULL},
    {NULL}
};

static PyTypeObject EvalCType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "cell_c.PCodeCellEvaluatorC",
    .tp_basicsize = sizeof(EvalC),
    .tp_dealloc   = (destructor)EvalC_dealloc,
    .tp_flags     = Py_TPFLAGS_DEFAULT,
    .tp_methods   = EvalC_methods,
    .tp_getset    = EvalC_getset,
    .tp_new       = EvalC_new,
    .tp_init      = (initproc)EvalC_init,
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "cell_c", NULL, -1, NULL
};

PyMODINIT_FUNC PyInit_cell_c(void) {
    if (PyType_Ready(&EvalCType) < 0) return NULL;
    PyObject *m = PyModule_Create(&moduledef);
    if (!m) return NULL;
    Py_INCREF(&EvalCType);
    PyModule_AddObject(m, "PCodeCellEvaluatorC", (PyObject *)&EvalCType);
    return m;
}
