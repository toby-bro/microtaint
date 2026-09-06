/* AMD64 native re-execution trampoline template.
 *
 * A copyable code+data blob: reexec_run(cpu_state_t *state /rdi/) loads the
 * target GPRs+RFLAGS from *state, executes ONE instruction (patched into the
 * 16-byte hole at reexec_tmpl_hole), then writes the resulting GPRs+RFLAGS back
 * into *state.  All scratch/host-save slots live INSIDE the blob and are reached
 * RIP-relatively, so the whole [reexec_run, reexec_tmpl_end) region can be
 * memcpy'd into an RWX buffer and called there (intra-blob displacements are
 * preserved).  Single-threaded use only (shared inline scratch slots).
 *
 * cpu_state_t layout: uint64 gpr[16] at offsets 0..120 in encoding order
 *   0 rax 1 rcx 2 rdx 3 rbx 4 rsp 5 rbp 6 rsi 7 rdi
 *   8 r8  9 r9  10 r10 11 r11 12 r12 13 r13 14 r14 15 r15
 * then uint64 rflags at offset 128.
 */
    .text
    .globl reexec_run
    .globl reexec_tmpl_hole
    .globl reexec_tmpl_holeend
    .globl reexec_tmpl_end
    .align 16
reexec_run:
    /* --- save host callee-saved + host rsp into inline slots (rip-rel) --- */
    mov %rbx, host_rbx(%rip)
    mov %rbp, host_rbp(%rip)
    mov %r12, host_r12(%rip)
    mov %r13, host_r13(%rip)
    mov %r14, host_r14(%rip)
    mov %r15, host_r15(%rip)
    mov %rsp, host_rsp(%rip)
    mov %rdi, stateptr(%rip)      /* rdi = &state */
    mov %rdi, %r11               /* r11 = &state (working base) */

    /* --- load target RFLAGS (still on host stack) --- */
    mov 128(%r11), %rax
    push %rax
    popfq

    /* --- load target GPRs (r11 is base; load r11 and rsp last) --- */
    mov 0(%r11), %rax
    mov 8(%r11), %rcx
    mov 16(%r11), %rdx
    mov 24(%r11), %rbx
    mov 40(%r11), %rbp
    mov 48(%r11), %rsi
    mov 56(%r11), %rdi
    mov 64(%r11), %r8
    mov 72(%r11), %r9
    mov 80(%r11), %r10
    mov 96(%r11), %r12
    mov 104(%r11), %r13
    mov 112(%r11), %r14
    mov 120(%r11), %r15
    mov 32(%r11), %rsp           /* target rsp (now on target stack) */
    mov 88(%r11), %r11           /* target r11 LAST */

reexec_tmpl_hole:
    /* 16-byte instruction hole (nop-filled; patched per call) */
    .byte 0x90,0x90,0x90,0x90,0x90,0x90,0x90,0x90
    .byte 0x90,0x90,0x90,0x90,0x90,0x90,0x90,0x90
reexec_tmpl_holeend:

    /* --- save target state --- */
    mov %rax, save_rax(%rip)     /* stash target rax */
    mov %r11, save_r11(%rip)     /* stash target r11 */
    mov %rsp, save_rsp(%rip)     /* stash target rsp */
    mov host_rsp(%rip), %rsp     /* back to host stack (mov: no flag change) */
    pushfq
    pop %rax                     /* rax = target rflags */
    mov %rax, save_flags(%rip)
    mov stateptr(%rip), %r11     /* r11 = &state */
    mov %rcx, 8(%r11)
    mov %rdx, 16(%r11)
    mov %rbx, 24(%r11)
    mov %rbp, 40(%r11)
    mov %rsi, 48(%r11)
    mov %rdi, 56(%r11)
    mov %r8, 64(%r11)
    mov %r9, 72(%r11)
    mov %r10, 80(%r11)
    mov %r12, 96(%r11)
    mov %r13, 104(%r11)
    mov %r14, 112(%r11)
    mov %r15, 120(%r11)
    mov save_rax(%rip), %rax
    mov %rax, 0(%r11)
    mov save_rsp(%rip), %rax
    mov %rax, 32(%r11)
    mov save_r11(%rip), %rax
    mov %rax, 88(%r11)
    mov save_flags(%rip), %rax
    mov %rax, 128(%r11)
    /* --- restore host callee-saved --- */
    mov host_rbx(%rip), %rbx
    mov host_rbp(%rip), %rbp
    mov host_r12(%rip), %r12
    mov host_r13(%rip), %r13
    mov host_r14(%rip), %r14
    mov host_r15(%rip), %r15
    ret

    /* --- inline scratch/host-save data (copied with the code) --- */
    /* Page-align OFF the executing code page: the trampoline WRITES these slots
     * every call; keeping them in the code's cache lines triggers self-modifying
     * -code pipeline flushes.  RIP-relative displacement is still preserved when
     * the whole [reexec_run, reexec_tmpl_end) blob is copied verbatim. */
    .p2align 12
host_rbx:   .quad 0
host_rbp:   .quad 0
host_r12:   .quad 0
host_r13:   .quad 0
host_r14:   .quad 0
host_r15:   .quad 0
host_rsp:   .quad 0
stateptr:   .quad 0
save_rax:   .quad 0
save_r11:   .quad 0
save_rsp:   .quad 0
save_flags: .quad 0
reexec_tmpl_end:
    .section .note.GNU-stack,"",@progbits
