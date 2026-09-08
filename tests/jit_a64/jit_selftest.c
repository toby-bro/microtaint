/*
 * Differential self-test for a taint-IR host backend.
 *
 * Builds random straight-line programs in the same shape the lowering
 * produces, runs each one through the interpreter and through the emitter, and
 * compares every output word.  It is the only way to check an emitter for a
 * host this machine is not: cross-compile it, run it under qemu-user, and the
 * comparison is still exact.
 *
 *   aarch64-linux-gnu-gcc -O2 -I<cell_c> -o jit_selftest jit_selftest.c
 *   qemu-aarch64 -L /usr/aarch64-linux-gnu ./jit_selftest
 *
 * Building it for the machine you are on tests the x86-64 backend the same
 * way, which is what makes a disagreement attributable to the backend rather
 * than to the harness.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The pieces of taint_ir_c.c the emitter needs, without Python. */
enum {
    IR_CONST = 0, IR_INV, IR_INT,
    IR_AND, IR_OR, IR_XOR, IR_ADD, IR_SUB, IR_MUL,
    IR_SHL, IR_SHR, IR_SAR, IR_NOT, IR_NEG,
    IR_ULT, IR_SLT, IR_EQ, IR_NEZ, IR_SEL,
    IR_POPCNT, IR_CLZ, IR_UDIV, IR_UREM, IR_SDIV, IR_SREM,
    IR_MULHI,
    IR_NOPS
};

typedef struct { int32_t node, kind, slot; } IRInput;
typedef struct { int32_t slot, node; } IROutput;

typedef struct IRProgC {
    int       n_nodes;
    uint8_t  *op;
    int32_t  *a, *b, *c;
    uint64_t *imm;
    int       n_inputs;
    IRInput  *inputs;
    int       n_outputs;
    IROutput *outputs;
    uint64_t *scratch;
    void     *jit_code;
    size_t    jit_size;
    void    (*jit_fn)(const uint64_t *, const uint64_t *, uint64_t *);
} IRProgC;

#if defined(__x86_64__)
#  include "taint_jit_x64.h"
#  define BACKEND "x86-64"
#elif defined(__aarch64__)
#  include "taint_jit_a64.h"
#  define BACKEND "aarch64"
#else
#  error "no backend for this host"
#endif

/* The interpreter, copied verbatim from taint_ir_c.c: it is the oracle, so it
 * has to be the same code and not a paraphrase of it. */
static void ir_run(const IRProgC *p, const uint64_t *in_values,
                   const uint64_t *in_taints, uint64_t *out_taints) {
    uint64_t *s = p->scratch;
    const int32_t *ai = p->a, *bi = p->b, *ci = p->c;
    for (int i = 0; i < p->n_inputs; i++) {
        const IRInput *in = &p->inputs[i];
        s[in->node] = in->kind ? in_taints[in->slot] : in_values[in->slot];
    }
    for (int n = 0; n < p->n_nodes; n++) {
        uint8_t op = p->op[n];
        if (op == IR_INV || op == IR_INT) continue;
        uint64_t a = (ai[n] >= 0) ? s[ai[n]] : 0;
        uint64_t b = (bi[n] >= 0) ? s[bi[n]] : 0;
        switch (op) {
        case IR_CONST: s[n] = p->imm[n]; break;
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
        case IR_MULHI:
            s[n] = (uint64_t)(((unsigned __int128)a * (unsigned __int128)b) >> 64);
            break;
        default: s[n] = 0; break;
        }
    }
    for (int i = 0; i < p->n_outputs; i++)
        out_taints[p->outputs[i].slot] = s[p->outputs[i].node];
}

/* ── random programs ─────────────────────────────────────────────────── */

static uint64_t rng_state = 0x2545F4914F6CDD1Dull;
static uint64_t rnd(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

#define N_SLOTS 64

/* Opcodes the emitter is expected to take.  Division and clz are declined by
 * both backends, so a program containing one proves nothing about either. */
static const uint8_t OPS[] = {
    IR_AND, IR_OR, IR_XOR, IR_ADD, IR_SUB, IR_MUL,
    IR_SHL, IR_SHR, IR_SAR, IR_NOT, IR_NEG,
    IR_ULT, IR_SLT, IR_EQ, IR_NEZ, IR_SEL, IR_POPCNT, IR_MULHI,
};
#define N_OPS ((int)(sizeof(OPS) / sizeof(OPS[0])))

static int arity(uint8_t op) {
    switch (op) {
    case IR_NOT: case IR_NEG: case IR_NEZ: case IR_POPCNT: return 1;
    case IR_SEL: return 3;
    default: return 2;
    }
}

static IRProgC *make_program(int n_nodes, int n_outputs) {
    IRProgC *p = (IRProgC *)calloc(1, sizeof(IRProgC));
    p->n_nodes = n_nodes;
    p->op = (uint8_t *)calloc((size_t)n_nodes, 1);
    p->a = (int32_t *)malloc(sizeof(int32_t) * (size_t)n_nodes);
    p->b = (int32_t *)malloc(sizeof(int32_t) * (size_t)n_nodes);
    p->c = (int32_t *)malloc(sizeof(int32_t) * (size_t)n_nodes);
    p->imm = (uint64_t *)calloc((size_t)n_nodes, sizeof(uint64_t));
    p->scratch = (uint64_t *)calloc((size_t)n_nodes, sizeof(uint64_t));
    p->inputs = (IRInput *)calloc((size_t)n_nodes, sizeof(IRInput));
    p->outputs = (IROutput *)calloc((size_t)n_outputs, sizeof(IROutput));
    for (int i = 0; i < n_nodes; i++) { p->a[i] = p->b[i] = p->c[i] = -1; }

    for (int i = 0; i < n_nodes; i++) {
        uint64_t r = rnd() % 100;
        if (i < 3 || r < 20) {
            /* An input, from the value array or the taint array. */
            p->op[i] = (r & 1) ? IR_INV : IR_INT;
            p->inputs[p->n_inputs].node = i;
            p->inputs[p->n_inputs].kind = (p->op[i] == IR_INT) ? 1 : 0;
            p->inputs[p->n_inputs].slot = (int32_t)(rnd() % N_SLOTS);
            p->n_inputs++;
        } else if (r < 35) {
            p->op[i] = IR_CONST;
            /* A mix of the shapes that show up: small counts, single bits,
             * masks, and values that need all four moves to materialise. */
            switch (rnd() % 5) {
            case 0: p->imm[i] = rnd() % 70; break;            /* shift counts */
            case 1: p->imm[i] = 1ull << (rnd() % 64); break;
            case 2: p->imm[i] = (1ull << (rnd() % 64)) - 1; break;
            case 3: p->imm[i] = rnd() % 4096; break;          /* imm12 range */
            default: p->imm[i] = rnd(); break;
            }
        } else {
            uint8_t op = OPS[rnd() % N_OPS];
            p->op[i] = op;
            int n_ar = arity(op);
            p->a[i] = (int32_t)(rnd() % (uint64_t)i);
            if (n_ar >= 2) p->b[i] = (int32_t)(rnd() % (uint64_t)i);
            if (n_ar >= 3) p->c[i] = (int32_t)(rnd() % (uint64_t)i);
        }
    }
    p->n_outputs = n_outputs;
    for (int i = 0; i < n_outputs; i++) {
        p->outputs[i].slot = (int32_t)(rnd() % N_SLOTS);
        p->outputs[i].node = (int32_t)(rnd() % (uint64_t)n_nodes);
    }
    return p;
}

static void free_program(IRProgC *p) {
    free(p->op); free(p->a); free(p->b); free(p->c); free(p->imm);
    free(p->scratch); free(p->inputs); free(p->outputs); free(p);
}

static const char *OPNAME[] = {
    "CONST", "INV", "INT", "AND", "OR", "XOR", "ADD", "SUB", "MUL",
    "SHL", "SHR", "SAR", "NOT", "NEG", "ULT", "SLT", "EQ", "NEZ", "SEL",
    "POPCNT", "CLZ", "UDIV", "UREM", "SDIV", "SREM", "MULHI",
};

/* A failing program, in a form that can be read back and shrunk by hand.  A
 * mismatch is worth nothing without the program that produced it. */
static void dump_program(const IRProgC *p) {
    printf("  program: %d nodes, %d inputs, %d outputs\n",
           p->n_nodes, p->n_inputs, p->n_outputs);
    for (int i = 0; i < p->n_nodes; i++) {
        printf("    n%-3d %-7s", i, OPNAME[p->op[i]]);
        if (p->op[i] == IR_CONST)
            printf(" imm=%016llx", (unsigned long long)p->imm[i]);
        else if (p->op[i] == IR_INV || p->op[i] == IR_INT) {
            for (int k = 0; k < p->n_inputs; k++)
                if (p->inputs[k].node == i) printf(" slot=%d", p->inputs[k].slot);
        } else {
            if (p->a[i] >= 0) printf(" a=n%d", p->a[i]);
            if (p->b[i] >= 0) printf(" b=n%d", p->b[i]);
            if (p->c[i] >= 0) printf(" c=n%d", p->c[i]);
        }
        printf("\n");
    }
    for (int i = 0; i < p->n_outputs; i++)
        printf("    out slot=%d <- n%d\n", p->outputs[i].slot, p->outputs[i].node);
}

/* One opcode at a time, over the values its rule turns on.  A random program
 * says only that something disagrees; this says which encoding is wrong. */
static const uint64_t EDGE[] = {
    0, 1, 2, 63, 64, 65, 0x7FFFFFFFFFFFFFFFull, 0x8000000000000000ull,
    ~0ull, 0x0123456789ABCDEFull, 0xFFFFFFFF00000000ull, 0x00000000FFFFFFFFull,
};
#define N_EDGE ((int)(sizeof(EDGE) / sizeof(EDGE[0])))

static int directed(void) {
    int bad = 0;
    for (int oi = 0; oi < N_OPS; oi++) {
        uint8_t op = OPS[oi];
        int n_ar = arity(op);
        /* n0, n1, n2 are inputs; n3 is the operation; one output. */
        IRProgC *p = (IRProgC *)calloc(1, sizeof(IRProgC));
        p->n_nodes = 4;
        p->op = (uint8_t *)calloc(4, 1);
        p->a = (int32_t *)malloc(sizeof(int32_t) * 4);
        p->b = (int32_t *)malloc(sizeof(int32_t) * 4);
        p->c = (int32_t *)malloc(sizeof(int32_t) * 4);
        p->imm = (uint64_t *)calloc(4, sizeof(uint64_t));
        p->scratch = (uint64_t *)calloc(4, sizeof(uint64_t));
        p->inputs = (IRInput *)calloc(3, sizeof(IRInput));
        p->outputs = (IROutput *)calloc(1, sizeof(IROutput));
        for (int i = 0; i < 4; i++) { p->a[i] = p->b[i] = p->c[i] = -1; }
        for (int i = 0; i < 3; i++) {
            p->op[i] = IR_INV;
            p->inputs[i].node = i; p->inputs[i].kind = 0; p->inputs[i].slot = i;
        }
        p->n_inputs = 3;
        p->op[3] = op;
        p->a[3] = 0;
        if (n_ar >= 2) p->b[3] = 1;
        if (n_ar >= 3) p->c[3] = 2;
        p->n_outputs = 1;
        p->outputs[0].slot = 8; p->outputs[0].node = 3;

        void *code = NULL; size_t sz = 0;
        mt_taint_fn fn = mt_jit_compile(p, &code, &sz);
        if (!fn) { printf("  %-7s DECLINED\n", OPNAME[op]); free_program(p); continue; }
        int op_bad = 0;
        for (int x = 0; x < N_EDGE; x++)
        for (int y = 0; y < N_EDGE; y++)
        for (int z = 0; z < 2; z++) {
            uint64_t vals[N_SLOTS] = {0}, taints[N_SLOTS] = {0};
            uint64_t want[N_SLOTS] = {0}, got[N_SLOTS] = {0};
            vals[0] = EDGE[x]; vals[1] = EDGE[y]; vals[2] = (uint64_t)z;
            ir_run(p, vals, taints, want);
            fn(vals, taints, got);
            if (want[8] != got[8]) {
                if (!op_bad)
                    printf("  %-7s WRONG: a=%016llx b=%016llx c=%d "
                           "interp=%016llx jit=%016llx\n", OPNAME[op],
                           (unsigned long long)vals[0], (unsigned long long)vals[1],
                           z, (unsigned long long)want[8],
                           (unsigned long long)got[8]);
                op_bad++;
            }
        }
        if (op_bad) { bad++; printf("  %-7s %d wrong of %d\n", OPNAME[op],
                                    op_bad, N_EDGE * N_EDGE * 2); }
        free_program(p);
    }
    return bad;
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "directed") == 0) {
        printf("%s backend, one opcode at a time:\n", BACKEND);
        int bad = directed();
        printf("%s directed: %d opcodes wrong\n", BACKEND, bad);
        return bad ? 1 : 0;
    }
    int iters = (argc > 1) ? atoi(argv[1]) : 4000;
    int failures = 0, compiled = 0, declined = 0;

    for (int it = 0; it < iters; it++) {
        /* Sized to cross the register file: ten nodes fit in registers on both
         * hosts, forty do not, so the spill paths are exercised too. */
        int n_nodes = 4 + (int)(rnd() % 60);
        int n_out = 1 + (int)(rnd() % 4);
        IRProgC *p = make_program(n_nodes, n_out);

        void *code = NULL; size_t sz = 0;
        mt_taint_fn fn = mt_jit_compile(p, &code, &sz);
        if (!fn) { declined++; free_program(p); continue; }
        compiled++;

        for (int trial = 0; trial < 4; trial++) {
            uint64_t vals[N_SLOTS], taints[N_SLOTS];
            uint64_t want[N_SLOTS], got[N_SLOTS];
            for (int i = 0; i < N_SLOTS; i++) {
                /* Extremes as well as noise: the shift and compare rules turn
                 * on 0, 1, 63, 64 and the sign bit. */
                switch (rnd() % 6) {
                case 0: vals[i] = 0; break;
                case 1: vals[i] = ~0ull; break;
                case 2: vals[i] = 1ull << 63; break;
                case 3: vals[i] = rnd() % 65; break;
                default: vals[i] = rnd(); break;
                }
                taints[i] = (rnd() % 4) ? rnd() : 0;
                want[i] = got[i] = 0xDEADBEEFCAFEBABEull;
            }
            ir_run(p, vals, taints, want);
            fn(vals, taints, got);
            for (int i = 0; i < N_SLOTS; i++) {
                if (want[i] != got[i]) {
                    if (failures == 0) {
                        printf("MISMATCH iter %d trial %d slot %d: "
                               "interp %016llx jit %016llx\n",
                               it, trial, i,
                               (unsigned long long)want[i],
                               (unsigned long long)got[i]);
                        dump_program(p);
                        const char *dumpf = getenv("JIT_DUMP");
                        if (dumpf) {
                            FILE *f = fopen(dumpf, "wb");
                            if (f) { fwrite(code, 1, sz, f); fclose(f);
                                     printf("  code -> %s (%zu bytes mapped)\n",
                                            dumpf, sz); }
                        }
                    }
                    failures++;
                    break;
                }
            }
        }
        free_program(p);
    }

    printf("%s backend: %d programs compiled, %d declined, %d mismatches\n",
           BACKEND, compiled, declined, failures);
#ifdef MT_JIT_DEBUG
    { extern int mt_dbg_spill, mt_dbg_takereg, mt_dbg_default;
      printf("  declines: spill-slots %d, no-register %d, no-rule %d\n",
             mt_dbg_spill, mt_dbg_takereg, mt_dbg_default); }
#endif
    return failures ? 1 : 0;
}
