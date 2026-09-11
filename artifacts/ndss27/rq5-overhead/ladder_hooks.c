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

static uc_hook_add_t g_hook_add;
static uc_reg_read_t g_reg_read;
static int           g_regs[4];
static size_t        g_handle;

/* Incremented by every callback.  Two jobs: it stops the compiler treating the
 * empty callback as dead, and it lets the harness PROVE the hook fired.  A C hook
 * that silently never fires would otherwise read as "free", which is exactly the
 * kind of fast wrong number this ladder exists to avoid. */
static volatile uint64_t g_count;

void ladder_init(void *hook_add, void *reg_read, int r0, int r1, int r2, int r3) {
    g_hook_add = (uc_hook_add_t)hook_add;
    g_reg_read = (uc_reg_read_t)reg_read;
    g_regs[0] = r0; g_regs[1] = r1; g_regs[2] = r2; g_regs[3] = r3;
    g_count = 0;
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

/* `which`: 0 = empty callback, 1 = read four registers.
 * `hook_type` is passed in rather than hardcoded so the constant comes from the
 * same unicorn build as everything else. */
int ladder_install(void *uc, int hook_type, int which) {
    void *cb = (which == 0) ? (void *)cb_empty : (void *)cb_regs;
    return g_hook_add(uc, &g_handle, hook_type, cb, NULL, 1, 0);
}

uint64_t ladder_count(void) { return g_count; }
