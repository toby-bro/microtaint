/* Native re-execution API shared by reexec.c (the implementation) and the
 * cell kernel (cell_c.c), which calls it as the 4th concrete-execution path.
 *
 * cell_c gates all reexec use on MICROTAINT_HAVE_REEXEC and falls back to
 * SLEIGH without it.
 */
#ifndef MICROTAINT_REEXEC_H
#define MICROTAINT_REEXEC_H

#include <stdint.h>

/* Where the trampoline can actually be built.  It needs three things, and
 * having the right CPU is only the first:
 *
 *   - an x86-64 or AArch64 host, for which a trampoline exists at all;
 *   - POSIX mmap and sigaction, to map the template executable and to catch
 *     the fault when the re-executed instruction misbehaves.  Windows has
 *     neither: <sys/mman.h> is simply absent there;
 *   - an ELF assembler.  The .S files end in `.section .note.GNU-stack`,
 *     which is ELF syntax, and their `.globl` names carry no leading
 *     underscore, which is what Mach-O expects of a C symbol.
 *
 * Deciding that here rather than in the build script means the sources
 * compile everywhere and simply provide no trampoline where they cannot:
 * reexec.c becomes stubs that report failure, the .S files assemble to
 * nothing, and the engine takes the SLEIGH path it already takes on a host
 * with no trampoline at all.
 */
#if defined(MICROTAINT_REEXEC_SUPPORTED)
  /* honour an explicit override from the build */
#elif (defined(__x86_64__) || defined(__aarch64__)) \
      && defined(__ELF__) && !defined(_WIN32)
#  define MICROTAINT_REEXEC_SUPPORTED 1
#else
#  define MICROTAINT_REEXEC_SUPPORTED 0
#endif

/* Host-native CPU state shuttled to/from the trampoline.  Arch-conditional so
 * the same harness (reexec.c) drives either trampoline; the field layout MUST
 * match the corresponding reexec_<arch>.S offsets exactly.
 *
 * AMD64: gpr[16] in x86-64 encoding order
 *   0 rax 1 rcx 2 rdx 3 rbx 4 rsp 5 rbp 6 rsi 7 rdi
 *   8 r8  9 r9  10 r10 11 r11 12 r12 13 r13 14 r14 15 r15
 * AArch64: x[31] = x0..x30, then sp, then nzcv.
 */
#if defined(__aarch64__)
typedef struct { uint64_t x[31]; uint64_t sp; uint64_t nzcv; } cpu_state_t;
#else
typedef struct { uint64_t gpr[16]; uint64_t rflags; } cpu_state_t;
#endif

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

/* ---- ISA-general register-shuttle API (keeps all ISA specifics in reexec) ----
 *
 * cell_c (and any caller) stays ISA-agnostic: it asks reexec for the ordered
 * list of register NAMES this arch shuttles, resolves each to its own storage,
 * and passes their values as a flat array.  reexec owns the name<->cpu_state
 * mapping (GPR index / flag bit / SP / NZCV) and the trampoline.
 *
 * reexec_arch_reg_count returns 0 when native re-exec is unavailable for
 * arch_name on THIS host (wrong ISA, or unsupported) -- the caller then falls
 * back to SLEIGH.  arch_name is matched by substring (e.g. "AMD64", "ARM64"). */
int         reexec_arch_reg_count(const char *arch_name);
const char *reexec_arch_reg_name(const char *arch_name, int i);
/* Run instr with reg_vals[i] = value of reg i (reexec_arch_reg_name order) on
 * input; reg_vals is updated in place with the outputs.  n must equal
 * reexec_arch_reg_count(arch_name).  Returns 0 on success, else fault/-1/-2. */
int         reexec_run_regs(const char *arch_name, const unsigned char *instr,
                            int len, uint64_t *reg_vals, int n);

#endif /* MICROTAINT_REEXEC_H */
