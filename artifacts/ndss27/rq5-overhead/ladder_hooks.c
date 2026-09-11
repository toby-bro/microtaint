/* Pure-C UC_HOOK_CODE callbacks, for the overhead ladder.
 *
 * The Python rungs in overhead_ladder.py measure per-instruction control THROUGH
 * CPython: GIL acquire, frame construction, argument marshalling.  These rungs
 * remove all of that and leave only what Unicorn itself pays to call out on every
 * instruction, so the ladder can separate "getting control per instruction" from
 * "getting control per instruction in Python".
 *
 * No unicorn.h needed: the two entry points we use are taken as function pointers
 * from the caller, which already has them loaded (unicorn's own ctypes CDLL).
 * That also guarantees we call into the SAME libunicorn the emulator is running,
 * rather than whichever one a linker happened to find.
 */
#include <stdint.h>
#include <stddef.h>

typedef int (*uc_hook_add_t)(void *, size_t *, int, void *, void *, uint64_t, uint64_t);
typedef int (*uc_reg_read_t)(void *, int, void *);
typedef int (*uc_mem_read_t)(void *, uint64_t, void *, size_t);

static uc_hook_add_t g_hook_add;
static uc_reg_read_t g_reg_read;
static uc_mem_read_t g_mem_read;
static int           g_regs[4];
static size_t        g_handle;

/* Incremented by every callback.  Two jobs: it stops the compiler treating the
 * empty callback as dead, and it lets the harness PROVE the hook fired.  A C hook
 * that silently never fires would otherwise read as "free", which is exactly the
 * kind of fast wrong number this ladder exists to avoid. */
static volatile uint64_t g_count;
static volatile uint64_t g_mem_count;
static size_t            g_mem_handle;

void ladder_init(void *hook_add, void *reg_read, void *mem_read,
                 int r0, int r1, int r2, int r3) {
    g_hook_add = (uc_hook_add_t)hook_add;
    g_reg_read = (uc_reg_read_t)reg_read;
    g_mem_read = (uc_mem_read_t)mem_read;
    g_regs[0] = r0; g_regs[1] = r1; g_regs[2] = r2; g_regs[3] = r3;
    g_count = 0;
    g_mem_count = 0;
}

static void cb_empty(void *uc, uint64_t addr, uint32_t size, void *ud) {
    (void)uc; (void)addr; (void)size; (void)ud;
    g_count++;
}

static void cb_regs(void *uc, uint64_t addr, uint32_t size, void *ud) {
    (void)addr; (void)size; (void)ud;
    uint64_t v;
    for (int i = 0; i < 4; i++) {
        g_reg_read(uc, g_regs[i], &v);
    }
    g_count++;
}

/* Reads the four registers AND eight bytes of guest memory, to price a memory
 * read against a register read.  Unicorn serves the two through different paths
 * -- a register read can be answered from the CPU state, a memory read goes
 * through the address-space lookup -- and a taint engine does both on roughly
 * half the instructions it sees, so assuming they cost the same is an
 * assumption worth not making.
 *
 * The address read is the instruction's own, which is certainly mapped: this
 * rung prices the READ, and a fault would price error handling instead. */
static void cb_regs_mem(void *uc, uint64_t addr, uint32_t size, void *ud) {
    (void)size; (void)ud;
    uint64_t v;
    for (int i = 0; i < 4; i++) {
        g_reg_read(uc, g_regs[i], &v);
    }
    uint64_t buf;
    g_mem_read(uc, addr, &buf, sizeof(buf));
    g_count++;
}

/* `which`: 0 = empty callback, 1 = read four registers, 2 = + a memory read.
 * `hook_type` is passed in rather than hardcoded so the constant comes from the
 * same unicorn build as everything else. */
int ladder_install(void *uc, int hook_type, int which) {
    void *cb = (which == 0) ? (void *)cb_empty
             : (which == 1) ? (void *)cb_regs
             : (void *)cb_regs_mem;
    return g_hook_add(uc, &g_handle, hook_type, cb, NULL, 1, 0);
}

/* A UC_HOOK_MEM_READ|WRITE callback.  Different signature from a code hook, and
 * different cost: it fires per memory ACCESS rather than per instruction, and it
 * is what a use-after-free detector needs in order to see every load and store.
 * microtaint registers one, so the ladder should price one. */
static void cb_mem(void *uc, int type, uint64_t address, int size,
                   int64_t value, void *ud) {
    (void)uc; (void)type; (void)address; (void)size; (void)value; (void)ud;
    g_mem_count++;
}

/* Register the memory hook alongside whatever code hook is already installed.
 * `hook_type` is the OR of the UC_HOOK_MEM_* bits, passed in so the constants
 * come from the same unicorn build as everything else. */
int ladder_install_mem(void *uc, int hook_type) {
    return g_hook_add(uc, &g_mem_handle, hook_type, (void *)cb_mem, NULL, 1, 0);
}

uint64_t ladder_count(void) { return g_count; }
uint64_t ladder_mem_count(void) { return g_mem_count; }
