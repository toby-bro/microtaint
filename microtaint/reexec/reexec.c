/* AMD64 native re-execution harness: copies the reexec trampoline template into
 * an RWX buffer, patches one instruction per call, executes it on the host CPU
 * with a seeded register+flags state, and captures the result -- signal-guarded.
 *
 * Build (self-test):
 *   cc -O2 -o reexec reexec.c reexec_amd64.S && ./reexec
 * Build (shared lib for ctypes):
 *   cc -O2 -shared -fPIC -o reexec.so reexec.c reexec_amd64.S
 */
#define _GNU_SOURCE
#include <setjmp.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>

#include "reexec.h"

extern char reexec_run[];
extern char reexec_tmpl_hole[];
extern char reexec_tmpl_end[];

static unsigned char *g_buf = NULL;
static long g_hole_off = 0, g_tmpl_size = 0;
static sigjmp_buf g_env;
static volatile int g_fault_sig = 0;
/* Set only while a trampoline call is in flight.  A fault OUTSIDE that window
 * (e.g. a genuine crash elsewhere in the process, or Python/Unicorn's own
 * handling) must NOT be diverted into our siglongjmp with a stale g_env. */
static volatile sig_atomic_t g_in_reexec = 0;
static struct sigaction g_saved[4];  /* pre-arm dispositions: SEGV,ILL,FPE,BUS */

static int _sig_index(int sig) {
    switch (sig) {
    case SIGSEGV: return 0;
    case SIGILL:  return 1;
    case SIGFPE:  return 2;
    case SIGBUS:  return 3;
    default:      return -1;
    }
}

static void fault_handler(int sig) {
    if (g_in_reexec) { g_fault_sig = sig; siglongjmp(g_env, sig); }
    /* Not our fault: restore the previous disposition and return so the
     * faulting instruction re-executes under the original handler. */
    int idx = _sig_index(sig);
    if (idx >= 0) sigaction(sig, &g_saved[idx], NULL);
}

static char g_altstack[64 * 1024];

int reexec_init(void) {
    g_hole_off = (long)(reexec_tmpl_hole - reexec_run);
    g_tmpl_size = (long)(reexec_tmpl_end - reexec_run);
    g_buf = mmap(NULL, (size_t)g_tmpl_size, PROT_READ | PROT_WRITE | PROT_EXEC,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (g_buf == MAP_FAILED) { g_buf = NULL; return -1; }
    memcpy(g_buf, reexec_run, (size_t)g_tmpl_size);
    /* A faulting instruction runs with the TARGET rsp (possibly garbage), so the
     * signal frame must be delivered on a dedicated stack or we double-fault. */
    stack_t ss; memset(&ss, 0, sizeof(ss));
    ss.ss_sp = g_altstack; ss.ss_size = sizeof(g_altstack); ss.ss_flags = 0;
    if (sigaltstack(&ss, NULL) != 0) return -2;
    return 0;
}

/* Only these signals are trapped during execution. */
static void install_handlers(struct sigaction old[4]) {
    struct sigaction sa; memset(&sa, 0, sizeof(sa));
    sa.sa_handler = fault_handler; sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_NODEFER | SA_ONSTACK;
    sigaction(SIGSEGV, &sa, &old[0]);
    sigaction(SIGILL,  &sa, &old[1]);
    sigaction(SIGFPE,  &sa, &old[2]);
    sigaction(SIGBUS,  &sa, &old[3]);
}
static void restore_handlers(struct sigaction old[4]) {
    sigaction(SIGSEGV, &old[0], NULL);
    sigaction(SIGILL,  &old[1], NULL);
    sigaction(SIGFPE,  &old[2], NULL);
    sigaction(SIGBUS,  &old[3], NULL);
}

/* Returns 0 on success (state updated in place), or the trapped signal number
 * (>0) on fault, or -1 if not initialised, -2 if instruction too long. */
int reexec_run_one(cpu_state_t *state, const unsigned char *instr, int len) {
    if (!g_buf) return -1;
    if (len < 1 || len > 15) return -2;
    memcpy(g_buf + g_hole_off, instr, (size_t)len);
    memset(g_buf + g_hole_off + len, 0x90, (size_t)(16 - len)); /* nop pad */
    __builtin___clear_cache((char *)g_buf, (char *)g_buf + g_tmpl_size);

    struct sigaction old[4];
    install_handlers(old);
    g_fault_sig = 0;
    int rc = 0;
    if (sigsetjmp(g_env, 0) == 0) {
        g_in_reexec = 1;
        ((void (*)(cpu_state_t *))g_buf)(state);
        g_in_reexec = 0;
    } else {
        g_in_reexec = 0;
        rc = g_fault_sig;
    }
    restore_handlers(old);
    return rc;
}

/* --- Amortised API: install handlers once, patch once, call many. --- */
int reexec_arm(void) { if (!g_buf) return -1; install_handlers(g_saved); return 0; }
void reexec_disarm(void) { restore_handlers(g_saved); }

int reexec_set_instr(const unsigned char *instr, int len) {
    if (!g_buf) return -1;
    if (len < 1 || len > 15) return -2;
    memcpy(g_buf + g_hole_off, instr, (size_t)len);
    memset(g_buf + g_hole_off + len, 0x90, (size_t)(16 - len));
    __builtin___clear_cache((char *)g_buf, (char *)g_buf + g_tmpl_size);
    return 0;
}

/* Call the already-patched instruction with handlers already armed. */
int reexec_call(cpu_state_t *state) {
    g_fault_sig = 0;
    if (sigsetjmp(g_env, 0) == 0) {
        g_in_reexec = 1;
        ((void (*)(cpu_state_t *))g_buf)(state);
        g_in_reexec = 0;
        return 0;
    }
    g_in_reexec = 0;
    return g_fault_sig;
}

#ifdef REEXEC_BENCH
#include <time.h>
static double now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}
int main(void) {
    if (reexec_init() != 0) { printf("init failed\n"); return 2; }
    reexec_arm();
    unsigned char add[] = {0x48, 0x01, 0xd8};  /* add rax, rbx */
    reexec_set_instr(add, 3);
    cpu_state_t s; memset(&s, 0, sizeof(s)); s.gpr[0] = 5; s.gpr[3] = 7; s.rflags = 0x202;
    long N = 5000000;
    /* (a) safe path: sigsetjmp + call per iteration */
    double t0 = now_ns();
    for (long i = 0; i < N; i++) { s.gpr[0] = 5; reexec_call(&s); }
    double a = (now_ns() - t0) / N;
    /* (b) raw trampoline call, no sigsetjmp (handlers still armed) */
    void (*fn)(cpu_state_t *) = (void (*)(cpu_state_t *))g_buf;
    t0 = now_ns();
    for (long i = 0; i < N; i++) { s.gpr[0] = 5; fn(&s); }
    double b = (now_ns() - t0) / N;
    reexec_disarm();
    printf("reexec add rax,rbx: safe(sigsetjmp+call)=%.1f ns   raw(call only)=%.1f ns   (result rax=%lu)\n",
           a, b, s.gpr[0]);
    return 0;
}
#endif

#ifdef REEXEC_SELFTEST
static int check(const char *name, const unsigned char *code, int len,
                 cpu_state_t *in, const cpu_state_t *want, uint64_t flag_mask) {
    cpu_state_t s = *in;
    int rc = reexec_run_one(&s, code, len);
    if (rc != 0) { printf("  %-16s FAULT sig=%d\n", name, rc); return 1; }
    int bad = 0;
    for (int i = 0; i < 16; i++) {
        if (i == 4) continue; /* rsp: harness/self-test ignores */
        if (s.gpr[i] != want->gpr[i]) {
            printf("  %-16s gpr[%d]=%#lx want %#lx\n", name, i, s.gpr[i], want->gpr[i]);
            bad = 1;
        }
    }
    if (((s.rflags ^ want->rflags) & flag_mask) != 0) {
        printf("  %-16s rflags=%#lx want %#lx (mask %#lx)\n", name, s.rflags, want->rflags, flag_mask);
        bad = 1;
    }
    if (!bad) printf("  %-16s OK  (rax=%#lx rflags=%#lx)\n", name, s.gpr[0], s.rflags);
    return bad;
}

/* CF=0x1 PF=0x4 ZF=0x40 SF=0x80 OF=0x800 */
#define OSZAPC 0x8D5
int main(void) {
    if (reexec_init() != 0) { printf("init failed\n"); return 2; }
    int fails = 0;
    cpu_state_t in; memset(&in, 0, sizeof(in));

    /* add rax, rbx : 48 01 d8 ; rax=5, rbx=7 -> rax=12 */
    in.gpr[0] = 5; in.gpr[3] = 7; in.rflags = 0x202;
    cpu_state_t w = in; w.gpr[0] = 12; w.rflags = 0x206; /* PF set: 0x0C has even parity */
    fails += check("add rax,rbx", (unsigned char[]){0x48,0x01,0xd8}, 3, &in, &w, OSZAPC);

    /* xor rcx, rcx : 48 31 c9 ; -> rcx=0, ZF=1 PF=1 */
    memset(&in, 0, sizeof(in)); in.gpr[1] = 0x1234; in.rflags = 0x202;
    w = in; w.gpr[1] = 0; w.rflags = 0x202 | 0x40 | 0x4; /* ZF|PF */
    fails += check("xor rcx,rcx", (unsigned char[]){0x48,0x31,0xc9}, 3, &in, &w, OSZAPC);

    /* shl rax, 1 : 48 d1 e0 ; rax=0x8000000000000000 -> 0, CF=1 OF=? */
    memset(&in, 0, sizeof(in)); in.gpr[0] = 0x8000000000000000ULL; in.rflags = 0x202;
    w = in; w.gpr[0] = 0; w.rflags = 0x202 | 0x1 | 0x40 | 0x4; /* CF|ZF|PF (OF set too but we mask it out) */
    fails += check("shl rax,1", (unsigned char[]){0x48,0xd1,0xe0}, 3, &in, &w, 0x40 | 0x4 | 0x1);

    /* inc r15 : 49 ff c7 ; r15=41 -> 42 */
    memset(&in, 0, sizeof(in)); in.gpr[15] = 41; in.rflags = 0x202;
    w = in; w.gpr[15] = 42; w.rflags = 0x202;
    fails += check("inc r15", (unsigned char[]){0x49,0xff,0xc7}, 3, &in, &w, 0x40 | 0x80);

    printf(fails ? "\nFAILS=%d\n" : "\nALL OK\n", fails);
    return fails ? 1 : 0;
}
#endif
