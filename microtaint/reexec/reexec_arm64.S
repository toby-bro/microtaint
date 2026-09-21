/* Assembles to nothing where the trampoline cannot exist: see
 * MICROTAINT_REEXEC_SUPPORTED in reexec.h.  This file uses ELF syntax
 * (.section .note.GNU-stack, and .globl names with no leading
 * underscore), which a Mach-O assembler rejects outright. */
#if (defined(__x86_64__) || defined(__aarch64__)) \
    && defined(__ELF__) && !defined(_WIN32)
/* AArch64 native re-execution trampoline template (analogue of reexec_amd64.S).
 *
 * reexec_run(cpu_state_t *state /x0/) loads the target GPRs (x0-x30), SP and
 * NZCV from *state, executes ONE instruction patched into the 4-byte hole at
 * reexec_tmpl_hole, then writes the resulting GPRs + SP + NZCV back.  The whole
 * [reexec_run, reexec_tmpl_end) blob is copied verbatim into an RWX buffer and
 * run there; PC-relative refs to the inline scratch are preserved by the copy.
 *
 * x18 holds the state pointer throughout (caller-saved on AAPCS64, so free to
 * clobber); the caller (reexec) must not run an instruction that references x18
 * -- reexec's eligibility check excludes those and falls back to SLEIGH.
 *
 * cpu_state_t (arm64): uint64 x[31] at 0..240, uint64 sp at 248, uint64 nzcv at 256.
 */
    .text
    .globl reexec_run
    .globl reexec_tmpl_hole
    .globl reexec_tmpl_holeend
    .globl reexec_tmpl_end
    .align 4
reexec_run:
    mov     x18, x0                 /* x18 = &state (persistent base) */
    /* save host callee-saved (x19-x30) + host sp into inline slots (x17 base) */
    adr     x17, host_save
    stp     x19, x20, [x17, #0]
    stp     x21, x22, [x17, #16]
    stp     x23, x24, [x17, #32]
    stp     x25, x26, [x17, #48]
    stp     x27, x28, [x17, #64]
    stp     x29, x30, [x17, #80]
    mov     x16, sp
    str     x16, [x17, #96]

    /* load target NZCV */
    ldr     x16, [x18, #256]
    msr     nzcv, x16
    /* load target SP first (x16 temp), before x16/x17 get their target values */
    ldr     x16, [x18, #248]
    mov     sp, x16
    /* load x0..x15 */
    ldr     x0,  [x18, #0]
    ldr     x1,  [x18, #8]
    ldr     x2,  [x18, #16]
    ldr     x3,  [x18, #24]
    ldr     x4,  [x18, #32]
    ldr     x5,  [x18, #40]
    ldr     x6,  [x18, #48]
    ldr     x7,  [x18, #56]
    ldr     x8,  [x18, #64]
    ldr     x9,  [x18, #72]
    ldr     x10, [x18, #80]
    ldr     x11, [x18, #88]
    ldr     x12, [x18, #96]
    ldr     x13, [x18, #104]
    ldr     x14, [x18, #112]
    ldr     x15, [x18, #120]
    /* load x19..x30 */
    ldr     x19, [x18, #152]
    ldr     x20, [x18, #160]
    ldr     x21, [x18, #168]
    ldr     x22, [x18, #176]
    ldr     x23, [x18, #184]
    ldr     x24, [x18, #192]
    ldr     x25, [x18, #200]
    ldr     x26, [x18, #208]
    ldr     x27, [x18, #216]
    ldr     x28, [x18, #224]
    ldr     x29, [x18, #232]
    ldr     x30, [x18, #240]
    /* load x16, x17 LAST (were temps) */
    ldr     x17, [x18, #136]
    ldr     x16, [x18, #128]

reexec_tmpl_hole:
    .word   0xd503201f              /* NOP; patched per call (arm64 instr = 4 bytes) */
reexec_tmpl_holeend:

    /* save target state (x18 still = &state; instruction excluded from using x18) */
    str     x2,  [x18, #16]
    str     x3,  [x18, #24]
    str     x4,  [x18, #32]
    str     x5,  [x18, #40]
    str     x6,  [x18, #48]
    str     x7,  [x18, #56]
    str     x8,  [x18, #64]
    str     x9,  [x18, #72]
    str     x10, [x18, #80]
    str     x11, [x18, #88]
    str     x12, [x18, #96]
    str     x13, [x18, #104]
    str     x14, [x18, #112]
    str     x15, [x18, #120]
    str     x16, [x18, #128]
    str     x17, [x18, #136]
    str     x19, [x18, #152]
    str     x20, [x18, #160]
    str     x21, [x18, #168]
    str     x22, [x18, #176]
    str     x23, [x18, #184]
    str     x24, [x18, #192]
    str     x25, [x18, #200]
    str     x26, [x18, #208]
    str     x27, [x18, #216]
    str     x28, [x18, #224]
    str     x29, [x18, #232]
    str     x30, [x18, #240]
    str     x0,  [x18, #0]
    str     x1,  [x18, #8]
    /* sp + nzcv (x2 free now) */
    mov     x2, sp
    str     x2,  [x18, #248]
    mrs     x2, nzcv
    str     x2,  [x18, #256]

    /* restore host callee-saved + host sp */
    adr     x17, host_save
    ldp     x19, x20, [x17, #0]
    ldp     x21, x22, [x17, #16]
    ldp     x23, x24, [x17, #32]
    ldp     x25, x26, [x17, #48]
    ldp     x27, x28, [x17, #64]
    ldp     x29, x30, [x17, #80]
    ldr     x16, [x17, #96]
    mov     sp, x16
    ret

    /* inline scratch/host-save data (copied with the code); page-aligned off the
     * code page so writes don't cause self-modifying-code pipeline flushes. */
    .p2align 12
host_save:
    .space 128
reexec_tmpl_end:
    .section .note.GNU-stack,"",@progbits

#endif  /* ELF host with a trampoline */
