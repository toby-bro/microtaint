/* Native re-execution API shared by reexec.c (the implementation) and the
 * cell kernel (cell_c.c), which calls it as the 4th concrete-execution path.
 *
 * Only meaningful on x86_64 hosts; the build hook compiles reexec.c +
 * reexec_amd64.S into cell_c ONLY there and defines MICROTAINT_HAVE_REEXEC, so
 * cell_c gates all reexec use on that macro and falls back to SLEIGH elsewhere.
 */
#ifndef MICROTAINT_REEXEC_H
#define MICROTAINT_REEXEC_H

#include <stdint.h>

/* cpu_state_t.gpr[16] in x86-64 encoding order:
 *   0 rax 1 rcx 2 rdx 3 rbx 4 rsp 5 rbp 6 rsi 7 rdi
 *   8 r8  9 r9  10 r10 11 r11 12 r12 13 r13 14 r14 15 r15
 */
typedef struct { uint64_t gpr[16]; uint64_t rflags; } cpu_state_t;

/* Initialise the trampoline (mmap RWX + copy template + altstack). 0 on success. */
int  reexec_init(void);
/* Install fault handlers persistently (call once); disarm at teardown. */
int  reexec_arm(void);
void reexec_disarm(void);
/* Patch the current instruction (amortised across many reexec_call). */
int  reexec_set_instr(const unsigned char *instr, int len);
/* Execute the already-patched instruction on *state (handlers must be armed).
 * Returns 0 on success (state updated), or the trapped signal number on fault. */
int  reexec_call(cpu_state_t *state);
/* All-in-one: patch + install handlers + call + restore. 0 / signal / -1 / -2. */
int  reexec_run_one(cpu_state_t *state, const unsigned char *instr, int len);

#endif /* MICROTAINT_REEXEC_H */
