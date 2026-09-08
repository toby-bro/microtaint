/* What does it cost to BE hooked, with no Python anywhere?
 *
 * The engine's null-hook measurement puts a per-instruction C hook at about
 * 356 ns above bare Qiling, but that figure contains a GIL acquire the Cython
 * callback declaration implies.  This separates the two: same emulation, same
 * workload, hooks that are genuinely empty C functions.
 *
 *   no hook      the denominator
 *   code hook    UC_HOOK_CODE, empty callback -- Unicorn's per-instruction
 *                dispatch and nothing else
 *   block hook   UC_HOOK_BLOCK, empty callback -- the same, once per block
 *   code+count   the callback increments a counter, to price a real body
 *   code+regs    the callback reads four guest registers through
 *                uc_reg_read_batch, which is what the engine's fast path does
 *                on every instruction whose inputs are not provably clean.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unicorn/unicorn.h>

#define CODE 0x1000000
#define STACK 0x2000000
static unsigned long long g_calls;

static void cb_empty(uc_engine *uc, uint64_t a, uint32_t s, void *d) { (void)uc;(void)a;(void)s;(void)d; }
static void cb_count(uc_engine *uc, uint64_t a, uint32_t s, void *d) { (void)uc;(void)a;(void)s;(void)d; g_calls++; }
static void cb_blk(uc_engine *uc, uint64_t a, uint32_t s, void *d) { (void)uc;(void)a;(void)s;(void)d; g_calls++; }

/* Four registers, read the way the engine reads them. */
static int g_nregs = 4;
static int g_ids[8] = {UC_X86_REG_RAX, UC_X86_REG_RBX, UC_X86_REG_RDX, UC_X86_REG_RCX,
                       UC_X86_REG_RSI, UC_X86_REG_RDI, UC_X86_REG_R8, UC_X86_REG_R9};
static uint64_t g_vals[8];
static void *g_ptrs[8];
static void cb_regs(uc_engine *uc, uint64_t a, uint32_t s, void *d) {
    (void)a;(void)s;(void)d;
    uc_reg_read_batch(uc, g_ids, g_ptrs, g_nregs);
    g_calls++;
}

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* body_n straight-line ALU instructions, then dec/jnz back to the top. */
static size_t build(unsigned char *p, int body_n) {
    static const unsigned char ops[][3] = {
        {0x48,0x01,0xD8}, {0x48,0x31,0xD0}, {0x48,0x21,0xD8}, {0x48,0x09,0xC3},
        {0x48,0x29,0xD8}, {0x48,0x89,0xC3}, {0x48,0x01,0xC3}, {0x48,0x31,0xD8},
    };
    size_t n = 0;
    for (int i = 0; i < body_n; i++) { memcpy(p+n, ops[i % 8], 3); n += 3; }
    p[n++] = 0x48; p[n++] = 0xFF; p[n++] = 0xC9;             /* dec rcx      */
    p[n++] = 0x0F; p[n++] = 0x85;                            /* jnz rel32    */
    int32_t rel = -(int32_t)(n + 4);
    memcpy(p+n, &rel, 4); n += 4;
    return n;
}

typedef enum { M_NONE, M_CODE, M_BLOCK, M_CODE_COUNT, M_CODE_REGS } hmode_t;

static double run_one(hmode_t mode, int body_n, uint64_t iters,
                      uint64_t *instrs, uint64_t *calls) {
    unsigned char code[4096];
    size_t len = build(code, body_n);
    uc_engine *uc; uc_hook h;
    if (uc_open(UC_ARCH_X86, UC_MODE_64, &uc)) { fprintf(stderr, "uc_open\n"); exit(1); }
    uc_mem_map(uc, CODE, 0x10000, UC_PROT_ALL);
    uc_mem_map(uc, STACK, 0x10000, UC_PROT_ALL);
    uc_mem_write(uc, CODE, code, len);
    uint64_t rcx = iters, rsp = STACK + 0x8000, rax = 3, rbx = 5, rdx = 7;
    uc_reg_write(uc, UC_X86_REG_RCX, &rcx); uc_reg_write(uc, UC_X86_REG_RSP, &rsp);
    uc_reg_write(uc, UC_X86_REG_RAX, &rax); uc_reg_write(uc, UC_X86_REG_RBX, &rbx);
    uc_reg_write(uc, UC_X86_REG_RDX, &rdx);
    g_calls = 0;
    switch (mode) {
    case M_CODE:       uc_hook_add(uc, &h, UC_HOOK_CODE,  (void*)cb_empty, NULL, 1, 0); break;
    case M_CODE_COUNT: uc_hook_add(uc, &h, UC_HOOK_CODE,  (void*)cb_count, NULL, 1, 0); break;
    case M_BLOCK:      uc_hook_add(uc, &h, UC_HOOK_BLOCK, (void*)cb_blk,   NULL, 1, 0); break;
    case M_CODE_REGS:  uc_hook_add(uc, &h, UC_HOOK_CODE,  (void*)cb_regs,  NULL, 1, 0); break;
    default: break;
    }
    double t0 = now();
    uc_err e = uc_emu_start(uc, CODE, CODE + len, 0, 0);
    double dt = now() - t0;
    if (e != UC_ERR_OK && e != UC_ERR_FETCH_UNMAPPED)
        fprintf(stderr, "  (emu %s)\n", uc_strerror(e));
    *instrs = iters * (uint64_t)(body_n + 2);
    *calls = g_calls;
    uc_close(uc);
    return dt;
}

int main(int argc, char **argv) {
    for (int i = 0; i < 8; i++) g_ptrs[i] = &g_vals[i];
    int body_n = argc > 1 ? atoi(argv[1]) : 16;
    uint64_t iters = argc > 2 ? strtoull(argv[2], NULL, 0) : 2000000;
    int reps = argc > 3 ? atoi(argv[3]) : 3;
    if (argc > 4) g_nregs = atoi(argv[4]);
    const char *names[] = {"no hook", "code hook", "block hook",
                          "code hook + counter", "code hook + N reg reads"};
    hmode_t modes[] = {M_NONE, M_CODE, M_BLOCK, M_CODE_COUNT, M_CODE_REGS};
    enum { NMODE = 5 };
    double best[NMODE]; uint64_t ins = 0, calls[NMODE];
    for (int m = 0; m < NMODE; m++) {
        best[m] = 1e30;
        for (int r = 0; r < reps; r++) {
            uint64_t c; double dt = run_one(modes[m], body_n, iters, &ins, &c);
            if (dt < best[m]) { best[m] = dt; calls[m] = c; }
        }
    }
    printf("%d instructions per block, %llu blocks, %llu instructions, best of %d\n",
           body_n + 2, (unsigned long long)iters, (unsigned long long)ins, reps);
    for (int m = 0; m < NMODE; m++)
        printf("  %-20s %8.1f ms   %7.2f ns/instruction   %+7.2f vs bare   (%llu calls)\n",
               names[m], best[m] * 1e3, best[m] * 1e9 / ins,
               (best[m] - best[0]) * 1e9 / ins, (unsigned long long)calls[m]);
    return 0;
}
