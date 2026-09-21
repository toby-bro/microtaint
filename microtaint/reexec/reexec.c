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
#include <stdint.h>

#include "reexec.h"

#if !MICROTAINT_REEXEC_SUPPORTED
/* No trampoline on this host: see the comment on MICROTAINT_REEXEC_SUPPORTED
 * in reexec.h.  The API is still provided so cell_c links and runs unchanged;
 * every entry point reports "not available" and the engine takes the SLEIGH
 * path, which is exactly what it does on a host with no trampoline.  This is
 * the same fallback, expressed in the source rather than in the build script,
 * so the sources compile on Windows and macOS instead of the wheel failing. */

int  reexec_init(void)                    { return -1; }
int  reexec_arm(void)                     { return -1; }
void reexec_disarm(void)                  { }
int  reexec_set_instr(const unsigned char *i, int n) { (void)i; (void)n; return -1; }
int  reexec_call(cpu_state_t *s)          { (void)s; return -1; }
int  reexec_run_one(cpu_state_t *s, const unsigned char *i, int n)
                                          { (void)s; (void)i; (void)n; return -1; }
int  reexec_arch_reg_count(const char *a) { (void)a; return 0; }
const char *reexec_arch_reg_name(const char *a, int i) { (void)a; (void)i; return 0; }
int  reexec_run_regs(const char *a, const unsigned char *i, int n,
                     uint64_t *v, int c)
{ (void)a; (void)i; (void)n; (void)v; (void)c; return -1; }

#else  /* MICROTAINT_REEXEC_SUPPORTED */

#include <setjmp.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>

extern char reexec_run[];
extern char reexec_tmpl_hole[];
extern char reexec_tmpl_holeend[];
extern char reexec_tmpl_end[];

static unsigned char *g_buf = NULL;
static long g_hole_off = 0, g_tmpl_size = 0, g_hole_size = 0;
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
    g_hole_size = (long)(reexec_tmpl_holeend - reexec_tmpl_hole);  /* 16 amd64 / 4 arm64 */
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
    if (len < 1 || len > g_hole_size) return -2;
    memcpy(g_buf + g_hole_off, instr, (size_t)len);
    memset(g_buf + g_hole_off + len, 0x90, (size_t)(g_hole_size - len)); /* nop pad */
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
    if (len < 1 || len > g_hole_size) return -2;
    memcpy(g_buf + g_hole_off, instr, (size_t)len);
    memset(g_buf + g_hole_off + len, 0x90, (size_t)(g_hole_size - len));
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

/* ---------------- ISA-general register-shuttle API ----------------
 * All ISA specifics (which register names exist, and how each maps into the
 * host cpu_state) live HERE, compiled per-HOST.  A register role:
 *   0 = GPR at cpu_state index `arg`
 *   1 = status flag at rflags bit `arg`   (amd64)
 *   2 = stack pointer (cpu_state.sp)       (arm64)
 *   3 = NZCV word     (cpu_state.nzcv)     (arm64)
 */
typedef struct { const char *name; int role; int arg; } ReExecRegDef;

#if defined(__aarch64__)
static const char *const _HOST_ARCH = "ARM64";
static const char *const _HOST_ARCH2 = "AARCH64";
static const ReExecRegDef _REGS[] = {
    {"X0",0,0},{"X1",0,1},{"X2",0,2},{"X3",0,3},{"X4",0,4},{"X5",0,5},{"X6",0,6},{"X7",0,7},
    {"X8",0,8},{"X9",0,9},{"X10",0,10},{"X11",0,11},{"X12",0,12},{"X13",0,13},{"X14",0,14},{"X15",0,15},
    {"X16",0,16},{"X17",0,17},{"X18",0,18},{"X19",0,19},{"X20",0,20},{"X21",0,21},{"X22",0,22},{"X23",0,23},
    {"X24",0,24},{"X25",0,25},{"X26",0,26},{"X27",0,27},{"X28",0,28},{"X29",0,29},{"X30",0,30},
    {"SP",2,0},{"NZCV",3,0},
};
#else
static const char *const _HOST_ARCH = "AMD64";
static const char *const _HOST_ARCH2 = "X86_64";
static const ReExecRegDef _REGS[] = {
    {"RAX",0,0},{"RCX",0,1},{"RDX",0,2},{"RBX",0,3},{"RSP",0,4},{"RBP",0,5},{"RSI",0,6},{"RDI",0,7},
    {"R8",0,8},{"R9",0,9},{"R10",0,10},{"R11",0,11},{"R12",0,12},{"R13",0,13},{"R14",0,14},{"R15",0,15},
    {"CF",1,0},{"PF",1,2},{"AF",1,4},{"ZF",1,6},{"SF",1,7},{"OF",1,11},{"DF",1,10},
};
#endif
#define _N_REGS ((int)(sizeof(_REGS)/sizeof(_REGS[0])))

static int _arch_is_host(const char *arch_name) {
    if (!arch_name) return 0;
    return (strstr(arch_name, _HOST_ARCH) != NULL) || (strstr(arch_name, _HOST_ARCH2) != NULL);
}

int reexec_arch_reg_count(const char *arch_name) {
    if (!g_buf && reexec_init() != 0) return 0;   /* also lazily inits the trampoline */
    return _arch_is_host(arch_name) ? _N_REGS : 0;
}

const char *reexec_arch_reg_name(const char *arch_name, int i) {
    if (!_arch_is_host(arch_name) || i < 0 || i >= _N_REGS) return NULL;
    return _REGS[i].name;
}

/* Amortised across calls: handlers armed once, the instruction re-patched only
 * when its bytes change (loops / the two differential corners reuse it). */
static int g_armed = 0;
static unsigned char g_last_instr[16];
static int g_last_len = -1;

int reexec_run_regs(const char *arch_name, const unsigned char *instr, int len,
                    uint64_t *reg_vals, int n) {
    if (!_arch_is_host(arch_name) || n != _N_REGS) return -1;
    if (!g_buf && reexec_init() != 0) return -1;
    cpu_state_t st;
    memset(&st, 0, sizeof(st));
    /* build cpu_state from reg_vals (arch-specific roles) */
    for (int i = 0; i < n; i++) {
        const ReExecRegDef *r = &_REGS[i];
#if defined(__aarch64__)
        if (r->role == 0) st.x[r->arg] = reg_vals[i];
        else if (r->role == 2) st.sp = reg_vals[i];
        else if (r->role == 3) st.nzcv = reg_vals[i];
#else
        if (r->role == 0) st.gpr[r->arg] = reg_vals[i];
        else if (r->role == 1) st.rflags |= (reg_vals[i] & 1) << r->arg;
#endif
    }
#if !defined(__aarch64__)
    st.rflags |= 0x2;  /* reserved bit 1 */
#endif
    /* amortised run */
    if (!g_armed) { if (reexec_arm() != 0) return -1; g_armed = 1; }
    if (g_last_len != len || memcmp(g_last_instr, instr, (size_t)len) != 0) {
        if (reexec_set_instr(instr, len) != 0) return -2;
        memcpy(g_last_instr, instr, (size_t)len);
        g_last_len = len;
    }
    int rc = reexec_call(&st);
    if (rc != 0) { g_last_len = -1; return rc; }
    /* split cpu_state back into reg_vals */
    for (int i = 0; i < n; i++) {
        const ReExecRegDef *r = &_REGS[i];
#if defined(__aarch64__)
        if (r->role == 0) reg_vals[i] = st.x[r->arg];
        else if (r->role == 2) reg_vals[i] = st.sp;
        else if (r->role == 3) reg_vals[i] = st.nzcv;
#else
        if (r->role == 0) reg_vals[i] = st.gpr[r->arg];
        else if (r->role == 1) reg_vals[i] = (st.rflags >> r->arg) & 1;
#endif
    }
    return 0;
}

#ifdef REEXEC_BENCH
#include <time.h>
static double now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}
#if defined(__aarch64__)
int main(void) {
    if (reexec_init() != 0) { printf("init failed\n"); return 2; }
    reexec_arm();
    unsigned char add[] = {0x00, 0x00, 0x01, 0x8b};  /* add x0, x0, x1 (LE) */
    reexec_set_instr(add, 4);
    cpu_state_t s; memset(&s, 0, sizeof(s)); s.x[0] = 5; s.x[1] = 7;
    long N = 5000000;
    double t0 = now_ns();
    for (long i = 0; i < N; i++) { s.x[0] = 5; reexec_call(&s); }
    double a = (now_ns() - t0) / N;
    reexec_disarm();
    printf("reexec add x0,x0,x1: safe(sigsetjmp+call)=%.1f ns   (result x0=%lu)\n", a, s.x[0]);
    return 0;
}
#else
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
#endif  /* arch */
#endif  /* REEXEC_BENCH */

#ifdef REEXEC_SELFTEST
#if defined(__aarch64__)
/* AArch64 self-test: a few register-only instructions with known results. */
int main(void) {
    if (reexec_init() != 0) { printf("init failed\n"); return 2; }
    int fails = 0;
    cpu_state_t s;
    /* add x0, x0, x1 : x0=5, x1=7 -> 12 */
    memset(&s, 0, sizeof(s)); s.x[0] = 5; s.x[1] = 7;
    if (reexec_run_one(&s, (unsigned char[]){0x00,0x00,0x01,0x8b}, 4) != 0 || s.x[0] != 12) {
        printf("  add x0,x0,x1  FAIL x0=%lu\n", s.x[0]); fails++;
    } else printf("  add x0,x0,x1  OK x0=%lu\n", s.x[0]);
    /* sub x2, x2, x2 -> 0 */
    memset(&s, 0, sizeof(s)); s.x[2] = 0x1234;
    if (reexec_run_one(&s, (unsigned char[]){0x42,0x00,0x02,0xcb}, 4) != 0 || s.x[2] != 0) {
        printf("  sub x2,x2,x2  FAIL x2=%lu\n", s.x[2]); fails++;
    } else printf("  sub x2,x2,x2  OK x2=%lu\n", s.x[2]);
    /* movz x3, #0x1234 -> 0x1234 */
    memset(&s, 0, sizeof(s));
    if (reexec_run_one(&s, (unsigned char[]){0x83,0x46,0x82,0xd2}, 4) != 0 || s.x[3] != 0x1234) {
        printf("  movz x3,#0x1234 FAIL x3=%#lx\n", s.x[3]); fails++;
    } else printf("  movz x3,#0x1234 OK x3=%#lx\n", s.x[3]);
    /* adds x4, x4, x5 with x4=x5=0 -> Z set (nzcv bit30) */
    memset(&s, 0, sizeof(s));
    if (reexec_run_one(&s, (unsigned char[]){0x84,0x00,0x05,0xab}, 4) != 0 || !(s.nzcv & (1u<<30))) {
        printf("  adds x4,x4,x5  FAIL nzcv=%#lx\n", s.nzcv); fails++;
    } else printf("  adds x4,x4,x5  OK nzcv=%#lx (Z set)\n", s.nzcv);
    printf(fails ? "\nFAILS=%d\n" : "\nALL OK\n", fails);
    return fails ? 1 : 0;
}
#else
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
#endif  /* arch */
#endif  /* REEXEC_SELFTEST */

#endif  /* MICROTAINT_REEXEC_SUPPORTED */
