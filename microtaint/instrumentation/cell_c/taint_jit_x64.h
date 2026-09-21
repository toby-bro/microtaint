#ifndef TAINT_JIT_X64_H
#define TAINT_JIT_X64_H

/*
 * taint_jit_x64 — emit machine code for a lowered taint program.
 * =============================================================
 *
 * The taint IR is straight-line 64-bit integer code with no control flow and no
 * memory traffic beyond its input and output arrays, which is about the most
 * compilable shape there is.  Handing it to clang produces excellent code but
 * costs milliseconds per program; this emitter produces slightly worse code in
 * microseconds, which is the trade a runtime JIT wants.
 *
 * Nothing here is target-specific in the sense that matters: the IR comes from
 * p-code and is ISA-general, and this file is a backend for the HOST the
 * analysis runs on.  A second backend for another host would consume exactly
 * the same program.
 *
 * Register discipline
 * -------------------
 *   r13, r14, r15   the three argument pointers (values, taint, out), moved out
 *                   of the SysV argument registers at entry so those become
 *                   allocatable.
 *   rcx             reserved.  Variable shifts need their count in cl, and
 *                   reserving one register outright is far simpler -- and no
 *                   slower in practice -- than evicting whatever happens to be
 *                   in rcx at every shift.
 *   the other ten   allocated by linear scan over the SSA order, with the
 *                   furthest-next-use victim spilled to the frame.
 *
 * Programs using division or a count-leading-zeros are DECLINED rather than
 * emitted: both need fixed-register or CPU-feature handling out of proportion
 * to how rarely a taint rule reaches them, and declining costs only a fall back
 * to the interpreter.
 */

#include <stdint.h>
#include <string.h>

#include "mt_codebuf.h"

/* Which x86-64 calling convention the emitted function must obey.
 *
 * This file emits a function that C code CALLS, so its entry sequence has to
 * agree with whatever the host compiler believes about argument registers and
 * about which registers a callee may clobber.  x86-64 has two answers:
 *
 *              args 1..3        callee-saved, of the ones this emitter uses
 *   SysV       rdi, rsi, rdx    rbx, rbp, r12-r15
 *   Win64      rcx, rdx, r8     rbx, rbp, rsi, rdi, r12-r15
 *
 * The difference that bites is rsi and rdi: the allocator has them in its
 * pool, and on Windows a function that clobbers them and returns has
 * corrupted its caller.  So Win64 saves two more registers rather than
 * shrinking the pool, which keeps the allocator's decisions -- and therefore
 * its spill paths -- identical on both, so the differential harness covers
 * the same code on either.
 *
 * MT_JIT_FORCE_WIN64 exists for that harness: it builds this emitter in Win64
 * mode on a Linux host and calls the result through an `ms_abi` pointer, so
 * the Windows entry sequence is EXECUTED and diffed against the interpreter
 * rather than merely compiled.
 */
#if defined(MT_JIT_FORCE_WIN64)
#  define MT_JIT_WIN64 1
#elif defined(_WIN32)
#  define MT_JIT_WIN64 1
#else
#  define MT_JIT_WIN64 0
#endif

#define JR_RAX 0
#define JR_RCX 1
#define JR_RDX 2
#define JR_RBX 3
#define JR_RSP 4
#define JR_RBP 5
#define JR_RSI 6
#define JR_RDI 7
#define JR_R8  8
#define JR_R12 12
#define JR_R13 13
#define JR_R14 14
#define JR_R15 15

/* Allocation pool: everything but rsp/rbp, the three pointer registers, and the
 * reserved shift register. */
static const int JIT_POOL[] = {0, 2, 3, 6, 7, 8, 9, 10, 11, 12};
#define JIT_NPOOL ((int)(sizeof(JIT_POOL) / sizeof(JIT_POOL[0])))

#define JIT_SPILL_BASE 100
#define JIT_MAX_SPILL  256
/* A constant that has not been materialised.  Most nodes that feed a taint rule
 * are masks and shift counts fixed at lift time, and x86 takes an immediate
 * operand directly -- so giving each one a register and a `mov` would spend a
 * scarce resource on something the instruction encoding can carry for free. */
#define JIT_CONST_LOC  (JIT_SPILL_BASE + JIT_MAX_SPILL)

typedef struct {
    uint8_t *buf;
    size_t   len, cap;
    int      overflow;
} JitBuf;

static void jb_byte(JitBuf *b, uint8_t v) {
    if (b->len >= b->cap) { b->overflow = 1; return; }
    b->buf[b->len++] = v;
}
static void jb_bytes(JitBuf *b, const uint8_t *v, int n) {
    for (int i = 0; i < n; i++) jb_byte(b, v[i]);
}
static void jb_u32(JitBuf *b, uint32_t v) {
    jb_byte(b, (uint8_t)v); jb_byte(b, (uint8_t)(v >> 8));
    jb_byte(b, (uint8_t)(v >> 16)); jb_byte(b, (uint8_t)(v >> 24));
}
static void jb_u64(JitBuf *b, uint64_t v) {
    jb_u32(b, (uint32_t)v); jb_u32(b, (uint32_t)(v >> 32));
}

/* REX.W with the extension bits for reg / index / rm. */
static void rex_w(JitBuf *b, int reg, int rm) {
    jb_byte(b, (uint8_t)(0x48 | ((reg >> 3) & 1) << 2 | ((rm >> 3) & 1)));
}
static void modrm_rr(JitBuf *b, int reg, int rm) {
    jb_byte(b, (uint8_t)(0xC0 | ((reg & 7) << 3) | (rm & 7)));
}
/* [base + disp32]; base is never rsp-with-index here except rsp itself, which
 * needs a SIB byte. */
static void modrm_mem(JitBuf *b, int reg, int base, int disp) {
    jb_byte(b, (uint8_t)(0x80 | ((reg & 7) << 3) | (base & 7)));
    if ((base & 7) == 4) jb_byte(b, 0x24);
    jb_u32(b, (uint32_t)disp);
}

static void j_mov_rr(JitBuf *b, int dst, int src) {
    if (dst == src) return;
    rex_w(b, src, dst); jb_byte(b, 0x89); modrm_rr(b, src, dst);
}
static void j_mov_ri(JitBuf *b, int dst, uint64_t imm) {
    if (imm == 0) {                       /* xor r, r */
        rex_w(b, dst, dst); jb_byte(b, 0x31); modrm_rr(b, dst, dst); return;
    }
    if (imm <= 0xFFFFFFFFULL) {           /* mov r32, imm32 zero-extends */
        if (dst >= 8) jb_byte(b, 0x41);
        jb_byte(b, (uint8_t)(0xB8 + (dst & 7)));
        jb_u32(b, (uint32_t)imm);
        return;
    }
    rex_w(b, 0, dst); jb_byte(b, (uint8_t)(0xB8 + (dst & 7))); jb_u64(b, imm);
}
static void j_load(JitBuf *b, int dst, int base, int disp) {
    rex_w(b, dst, base); jb_byte(b, 0x8B); modrm_mem(b, dst, base, disp);
}
static void j_store(JitBuf *b, int base, int disp, int src) {
    rex_w(b, src, base); jb_byte(b, 0x89); modrm_mem(b, src, base, disp);
}
/* add/or/and/sub/xor/cmp, register to register (dst op= src). */
static void j_alu(JitBuf *b, uint8_t opc, int dst, int src) {
    rex_w(b, src, dst); jb_byte(b, opc); modrm_rr(b, src, dst);
}
#define J_ADD 0x01
#define J_OR  0x09
#define J_AND 0x21
#define J_SUB 0x29
#define J_XOR 0x31
#define J_CMP 0x39
#define J_TEST 0x85

static void j_imul(JitBuf *b, int dst, int src) {
    rex_w(b, dst, src); jb_byte(b, 0x0F); jb_byte(b, 0xAF); modrm_rr(b, dst, src);
}
static void j_unary(JitBuf *b, int ext, int dst) {   /* /2 = not, /3 = neg */
    rex_w(b, 0, dst); jb_byte(b, 0xF7); jb_byte(b, (uint8_t)(0xC0 | (ext << 3) | (dst & 7)));
}
static void j_shift_imm(JitBuf *b, int ext, int dst, int imm) {
    rex_w(b, 0, dst); jb_byte(b, 0xC1);
    jb_byte(b, (uint8_t)(0xC0 | (ext << 3) | (dst & 7)));
    jb_byte(b, (uint8_t)imm);
}
static void j_shift_cl(JitBuf *b, int ext, int dst) {
    rex_w(b, 0, dst); jb_byte(b, 0xD3);
    jb_byte(b, (uint8_t)(0xC0 | (ext << 3) | (dst & 7)));
}
#define J_SHL 4
#define J_SHR 5
#define J_SAR 7
#define J_NOT 2
#define J_NEG 3

/* setcc into the low byte; REX is emitted unconditionally so rsi/rdi/rbp/rsp
 * address sil/dil/bpl/spl rather than the legacy high-byte registers. */
static void j_setcc(JitBuf *b, uint8_t cc, int dst) {
    jb_byte(b, (uint8_t)(0x40 | ((dst >> 3) & 1)));
    jb_byte(b, 0x0F); jb_byte(b, cc); jb_byte(b, (uint8_t)(0xC0 | (dst & 7)));
}
static void j_movzx8(JitBuf *b, int dst, int src) {
    rex_w(b, dst, src); jb_byte(b, 0x0F); jb_byte(b, 0xB6); modrm_rr(b, dst, src);
}
static void j_cmovcc(JitBuf *b, uint8_t cc, int dst, int src) {
    rex_w(b, dst, src); jb_byte(b, 0x0F); jb_byte(b, cc); modrm_rr(b, dst, src);
}
/* mul r/m64: RDX:RAX = RAX * r/m64.  The only instruction here that dictates
 * its own registers, which is why the high half of a product needs a small
 * dance around rax and rdx rather than falling out of the allocator. */
static void j_mul_rm(JitBuf *b, int src) {
    rex_w(b, 0, src); jb_byte(b, 0xF7);
    jb_byte(b, (uint8_t)(0xC0 | (4 << 3) | (src & 7)));
}

static void j_popcnt(JitBuf *b, int dst, int src) {
    jb_byte(b, 0xF3); rex_w(b, dst, src);
    jb_byte(b, 0x0F); jb_byte(b, 0xB8); modrm_rr(b, dst, src);
}
static void j_cmp_ri(JitBuf *b, int r, int imm) {   /* cmp r64, imm8 */
    rex_w(b, 0, r); jb_byte(b, 0x83);
    jb_byte(b, (uint8_t)(0xC0 | (7 << 3) | (r & 7))); jb_byte(b, (uint8_t)imm);
}

/* group-1 ALU with an immediate: add=0 or=1 and=4 sub=5 xor=6 cmp=7. */
#define JI_ADD 0
#define JI_OR  1
#define JI_AND 4
#define JI_SUB 5
#define JI_XOR 6
#define JI_CMP 7
static void j_alu_ri(JitBuf *b, int ext, int dst, int64_t imm) {
    rex_w(b, 0, dst);
    if (imm >= -128 && imm <= 127) {
        jb_byte(b, 0x83);
        jb_byte(b, (uint8_t)(0xC0 | (ext << 3) | (dst & 7)));
        jb_byte(b, (uint8_t)imm);
    } else {
        jb_byte(b, 0x81);
        jb_byte(b, (uint8_t)(0xC0 | (ext << 3) | (dst & 7)));
        jb_u32(b, (uint32_t)(int32_t)imm);
    }
}

/* An immediate is usable when it survives the sign extension x86 applies to
 * the 32-bit encoding. */
static int j_fits_imm32(uint64_t v) {
    int64_t s = (int64_t)v;
    return s >= -2147483648LL && s <= 2147483647LL;
}
static void j_push(JitBuf *b, int r) {
    if (r >= 8) jb_byte(b, 0x41);
    jb_byte(b, (uint8_t)(0x50 + (r & 7)));
}
static void j_pop(JitBuf *b, int r) {
    if (r >= 8) jb_byte(b, 0x41);
    jb_byte(b, (uint8_t)(0x58 + (r & 7)));
}
static void j_addsub_rsp(JitBuf *b, int add, int imm) {
    rex_w(b, 0, JR_RSP); jb_byte(b, 0x81);
    jb_byte(b, (uint8_t)(0xC0 | ((add ? 0 : 5) << 3) | JR_RSP));
    jb_u32(b, (uint32_t)imm);
}

/* ── the allocator ───────────────────────────────────────────────────── */

typedef struct {
    JitBuf   b;
    int     *loc;        /* node -> register, or JIT_SPILL_BASE + slot */
    int     *last;       /* node -> index of its last use */
    int      occupant[16];   /* physreg -> node, or -1 */
    uint8_t  spill_used[JIT_MAX_SPILL];
    const uint64_t *imm;     /* node -> its constant, when loc is JIT_CONST_LOC */
    int      n_spill;
    int      cur;        /* node being emitted, for last-use decisions */
    int      ok;
} JitCtx;

static int jit_spill_slot(JitCtx *c) {
    for (int i = 0; i < JIT_MAX_SPILL; i++)
        if (!c->spill_used[i]) { c->spill_used[i] = 1;
                                 if (i + 1 > c->n_spill) c->n_spill = i + 1;
                                 return i; }
    c->ok = 0;
    return 0;
}

static void jit_spill(JitCtx *c, int reg) {
    int n = c->occupant[reg];
    if (n < 0) return;
    /* A value nothing will read again is dropped rather than stored.  `last`
     * is -1 exactly when no node takes it as an operand and no output names
     * it, so there is no reader left to disappoint -- and a dead value is the
     * best victim there is, since evicting it costs no instruction at all. */
    if (c->last[n] < 0) { c->loc[n] = -1; c->occupant[reg] = -1; return; }
    int slot = jit_spill_slot(c);
    j_store(&c->b, JR_RSP, slot * 8, reg);
    c->loc[n] = JIT_SPILL_BASE + slot;
    c->occupant[reg] = -1;
}

/* Take a register, evicting the value whose next use is furthest away.
 *
 * The excluded registers are excluded ABSOLUTELY, not merely preferred against.
 * An operand that dies at this node has already been released, so its register
 * reads as free -- which is what makes in-place operation possible -- but its
 * VALUE is still needed until the instruction that consumes it is emitted.
 * Handing that register out as a scratch, or as the destination of an operation
 * that writes before it reads, silently computes with the wrong operand.
 *
 * The node being emitted is never a candidate.  Its register is marked occupied
 * before the operation is emitted so nothing else lands on it, but it holds no
 * value yet -- spilling it would store garbage and, worse, would move `loc` for
 * that node to a spill slot the emission is about to contradict by writing the
 * result into the register.  (That is exactly what `mulhi` did: its scratch
 * could evict its own destination, and the output then read the slot.)
 */
static int jit_take_reg_ex(JitCtx *c, int e0, int e1, int e2) {
    for (int i = 0; i < JIT_NPOOL; i++) {
        int r = JIT_POOL[i];
        if (r == e0 || r == e1 || r == e2) continue;
        if (c->occupant[r] == c->cur) continue;
        if (c->occupant[r] < 0) return r;
    }
    /* -2, not -1: a register holding a value with no further use has
     * `last` == -1, and starting the search at -1 would refuse to pick it --
     * so a program with a dead node could exhaust the pool and be declined
     * even though the ideal victim was sitting right there. */
    int best = -1, best_use = -2;
    for (int i = 0; i < JIT_NPOOL; i++) {
        int r = JIT_POOL[i];
        if (r == e0 || r == e1 || r == e2) continue;
        if (c->occupant[r] == c->cur) continue;
        int n = c->occupant[r];
        int u = (n >= 0) ? c->last[n] : -1;
        if (u > best_use) { best_use = u; best = r; }
    }
    if (best < 0) { c->ok = 0; return JIT_POOL[0]; }
    jit_spill(c, best);
    return best;
}

static int jit_take_reg(JitCtx *c, int avoid_a, int avoid_b) {
    (void)avoid_a; (void)avoid_b;
    return jit_take_reg_ex(c, -1, -1, -1);
}

/* Ensure node `n` is in a register and return it. */
static int jit_in_reg(JitCtx *c, int n, int avoid_a, int avoid_b) {
    int l = c->loc[n];
    if (l < JIT_SPILL_BASE) return l;
    if (l == JIT_CONST_LOC) {
        int r = jit_take_reg_ex(c, avoid_a, avoid_b, -1);
        j_mov_ri(&c->b, r, c->imm[n]);
        c->loc[n] = r;
        c->occupant[r] = n;
        return r;
    }
    int slot = l - JIT_SPILL_BASE;
    int r = jit_take_reg_ex(c, avoid_a, avoid_b, -1);
    j_load(&c->b, r, JR_RSP, slot * 8);
    c->spill_used[slot] = 0;
    c->loc[n] = r;
    c->occupant[r] = n;
    return r;
}

static void jit_release(JitCtx *c, int n) {
    int l = c->loc[n];
    if (l < JIT_SPILL_BASE) c->occupant[l] = -1;
    else if (l != JIT_CONST_LOC) c->spill_used[l - JIT_SPILL_BASE] = 0;
    c->loc[n] = -1;
}


/* ── emission ────────────────────────────────────────────────────────── */

/* dst = ra <op> rb, honouring x86's two-address form.  The only awkward case is
 * a non-commutative op whose destination register already holds the RIGHT
 * operand: moving the left operand in would destroy it, so the right one is
 * parked in the reserved scratch first. */
static void jit_binop(JitCtx *c, uint8_t opc, int commutative,
                      int dst, int ra, int rb) {
    if (dst == rb && ra != rb) {
        if (commutative) { j_alu(&c->b, opc, dst, ra); return; }
        j_mov_rr(&c->b, JR_RCX, rb);
        j_mov_rr(&c->b, dst, ra);
        j_alu(&c->b, opc, dst, JR_RCX);
        return;
    }
    j_mov_rr(&c->b, dst, ra);
    j_alu(&c->b, opc, dst, rb);
}

static void jit_imul(JitCtx *c, int dst, int ra, int rb) {
    if (dst == rb) { j_imul(&c->b, dst, ra); return; }
    j_mov_rr(&c->b, dst, ra);
    j_imul(&c->b, dst, rb);
}

static void jit_cmp_set(JitCtx *c, uint8_t cc, int dst, int ra, int rb) {
    j_alu(&c->b, J_CMP, ra, rb);       /* cmp ra, rb */
    j_setcc(&c->b, cc, dst);
    j_movzx8(&c->b, dst, dst);
}

/* A shift whose count is not known until run time.  x86 takes the count modulo
 * the operand width, while the IR defines a shift of 64 or more as producing
 * zero (or the sign, for an arithmetic shift), so the out-of-range case is
 * selected explicitly rather than left to the hardware's masking. */
static void jit_shift_var(JitCtx *c, int kind, int dst, int ra, int rb,
                          int tmp) {
    j_mov_rr(&c->b, JR_RCX, rb);
    if (kind == J_SAR) {
        j_mov_rr(&c->b, tmp, ra);
        j_shift_imm(&c->b, J_SAR, tmp, 63);      /* the saturating result */
    } else {
        j_alu(&c->b, J_XOR, tmp, tmp);
    }
    j_mov_rr(&c->b, dst, ra);
    j_shift_cl(&c->b, kind, dst);
    j_cmp_ri(&c->b, JR_RCX, 64);
    j_cmovcc(&c->b, 0x43, dst, tmp);             /* cmovae: count >= 64 */
}

/* On a Linux host built in Win64 mode (the harness), the pointer must be
 * declared ms_abi or the compiler would emit a SysV call to Win64 code. */
#if MT_JIT_WIN64 && !defined(_WIN32)
typedef void (__attribute__((ms_abi)) *mt_taint_fn)(
    const uint64_t *, const uint64_t *, uint64_t *);
#else
typedef void (*mt_taint_fn)(const uint64_t *, const uint64_t *, uint64_t *);
#endif

/* True when every opcode in the program has an emission rule here.  Division
 * and count-leading-zeros are the exceptions: both want fixed registers or a
 * CPU feature check out of proportion to how rarely a taint rule reaches them,
 * and declining just falls back to the interpreter. */
static int jit_supported(const IRProgC *p) {
    for (int i = 0; i < p->n_nodes; i++) {
        switch (p->op[i]) {
        case IR_UDIV: case IR_UREM: case IR_SDIV: case IR_SREM:
        case IR_CLZ:
            return 0;
        default: break;
        }
    }
    return 1;
}

/* Compile `p` into executable memory.  Returns NULL if the program is outside
 * what this emitter handles or if anything ran out of room; the caller keeps
 * the interpreter for those. */
static mt_taint_fn mt_jit_compile(const IRProgC *p, void **code_out,
                                  size_t *size_out) {
    if (!jit_supported(p)) return NULL;
    int n = p->n_nodes;
    size_t cap = 64 + (size_t)n * 48;
    JitCtx c;
    memset(&c, 0, sizeof(c));
    c.ok = 1;
    c.b.buf = (uint8_t *)malloc(cap);
    c.b.cap = cap;
    c.loc = (int *)malloc(sizeof(int) * (size_t)(n ? n : 1));
    c.last = (int *)malloc(sizeof(int) * (size_t)(n ? n : 1));
    if (!c.b.buf || !c.loc || !c.last) {
        free(c.b.buf); free(c.loc); free(c.last); return NULL;
    }
    c.imm = p->imm;
    for (int i = 0; i < 16; i++) c.occupant[i] = -1;
    for (int i = 0; i < n; i++) { c.loc[i] = -1; c.last[i] = -1; }
    for (int i = 0; i < n; i++) {
        if (p->a[i] >= 0) c.last[p->a[i]] = i;
        if (p->b[i] >= 0) c.last[p->b[i]] = i;
        if (p->c[i] >= 0) c.last[p->c[i]] = i;
    }
    /* Outputs are stored after the last node, matching the interpreter's
     * contract that the out array may alias the taint array. */
    for (int i = 0; i < p->n_outputs; i++) c.last[p->outputs[i].node] = n;

    /* Where each input lands, so a node can be materialised on first use. */
    int8_t *in_kind = (int8_t *)calloc((size_t)(n ? n : 1), 1);
    int32_t *in_slot = (int32_t *)calloc((size_t)(n ? n : 1), sizeof(int32_t));
    if (!in_kind || !in_slot) {
        free(c.b.buf); free(c.loc); free(c.last); free(in_kind); free(in_slot);
        return NULL;
    }
    for (int i = 0; i < p->n_inputs; i++) {
        in_kind[p->inputs[i].node] = (int8_t)(1 + p->inputs[i].kind);
        in_slot[p->inputs[i].node] = p->inputs[i].slot;
    }

    /* Prologue.  The frame size is patched once the spill high-water mark is
     * known, so it is emitted with a fixed-width immediate. */
    j_push(&c.b, JR_RBX); j_push(&c.b, JR_R12);
    j_push(&c.b, JR_R13); j_push(&c.b, JR_R14); j_push(&c.b, JR_R15);
#if MT_JIT_WIN64
    /* Callee-saved on Windows and in the pool, so they must be restored. */
    j_push(&c.b, JR_RSI); j_push(&c.b, JR_RDI);
#endif
    /* Either way the pushes leave rsp 16-byte aligned: entry is 8 mod 16, and
     * 5 pushes make 48 while 7 make 64.  The frame below is rounded to 16, so
     * alignment holds all the way to the body. */
    size_t frame_patch = c.b.len + 3;
    j_addsub_rsp(&c.b, 0, 0);
#if MT_JIT_WIN64
    j_mov_rr(&c.b, JR_R13, JR_RCX);
    j_mov_rr(&c.b, JR_R14, JR_RDX);
    j_mov_rr(&c.b, JR_R15, JR_R8);
#else
    j_mov_rr(&c.b, JR_R13, JR_RDI);
    j_mov_rr(&c.b, JR_R14, JR_RSI);
    j_mov_rr(&c.b, JR_R15, JR_RDX);
#endif

    for (int i = 0; i < n && c.ok; i++) {
        c.cur = i;
        uint8_t op = p->op[i];
        int an = p->a[i], bn = p->b[i], cn = p->c[i];

        /* A constant costs nothing until something needs it in a register. */
        if (op == IR_CONST) { c.loc[i] = JIT_CONST_LOC; continue; }

        /* Can this op take its right operand as an immediate? */
        int b_imm = 0;
        int64_t imm_v = 0;
        if (bn >= 0 && c.loc[bn] == JIT_CONST_LOC) {
            switch (op) {
            case IR_AND: case IR_OR: case IR_XOR: case IR_ADD: case IR_SUB:
            case IR_ULT: case IR_SLT: case IR_EQ:
                if (j_fits_imm32(p->imm[bn])) {
                    b_imm = 1; imm_v = (int64_t)p->imm[bn];
                }
                break;
            case IR_SHL: case IR_SHR: case IR_SAR:
                b_imm = 1; imm_v = (int64_t)p->imm[bn];
                break;
            default: break;
            }
        }

        int ra = -1, rb = -1, rc = -1;
        if (an >= 0) ra = jit_in_reg(&c, an, -1, -1);
        if (bn >= 0 && !b_imm) rb = jit_in_reg(&c, bn, ra, -1);
        if (cn >= 0) rc = jit_in_reg(&c, cn, ra, rb);
        /* Operands dying here free their registers before the destination is
         * chosen, which is what makes the common case in-place. */
        if (an >= 0 && c.last[an] == i) jit_release(&c, an);
        if (bn >= 0 && c.last[bn] == i && bn != an) jit_release(&c, bn);
        if (cn >= 0 && c.last[cn] == i && cn != an && cn != bn) jit_release(&c, cn);

        /* A select reads its condition AFTER writing the destination, so the
         * destination may not land on the condition's register.  Every other op
         * reads all operands before its first write, so reuse is safe there.
         *
         * Prefer the left operand's register when it just died: x86's
         * two-address form then needs no move at all, which is most of what
         * separates emitted code from compiled code on this kind of program. */
        int excl = (op == IR_SEL) ? rc : -1;
        int dst;
        if (op == IR_MULHI) {
            /* `mul` writes rdx:rax, so the destination has to be elsewhere. */
            dst = jit_take_reg_ex(&c, JR_RAX, JR_RDX, -1);
        } else if (ra >= 0 && ra != excl && c.occupant[ra] < 0) {
            dst = ra;
        } else {
            dst = jit_take_reg_ex(&c, excl, -1, -1);
        }
        c.occupant[dst] = i;
        c.loc[i] = dst;

        switch (op) {
        case IR_CONST: j_mov_ri(&c.b, dst, p->imm[i]); break;
        case IR_INV: j_load(&c.b, dst, JR_R13, in_slot[i] * 8); break;
        case IR_INT: j_load(&c.b, dst, JR_R14, in_slot[i] * 8); break;
        case IR_AND:
            if (b_imm) { j_mov_rr(&c.b, dst, ra); j_alu_ri(&c.b, JI_AND, dst, imm_v); }
            else jit_binop(&c, J_AND, 1, dst, ra, rb);
            break;
        case IR_OR:
            if (b_imm) { j_mov_rr(&c.b, dst, ra); j_alu_ri(&c.b, JI_OR, dst, imm_v); }
            else jit_binop(&c, J_OR, 1, dst, ra, rb);
            break;
        case IR_XOR:
            if (b_imm) { j_mov_rr(&c.b, dst, ra); j_alu_ri(&c.b, JI_XOR, dst, imm_v); }
            else jit_binop(&c, J_XOR, 1, dst, ra, rb);
            break;
        case IR_ADD:
            if (b_imm) { j_mov_rr(&c.b, dst, ra); j_alu_ri(&c.b, JI_ADD, dst, imm_v); }
            else jit_binop(&c, J_ADD, 1, dst, ra, rb);
            break;
        case IR_SUB:
            if (b_imm) { j_mov_rr(&c.b, dst, ra); j_alu_ri(&c.b, JI_SUB, dst, imm_v); }
            else jit_binop(&c, J_SUB, 0, dst, ra, rb);
            break;
        case IR_MUL: jit_imul(&c, dst, ra, rb); break;
        case IR_NOT: j_mov_rr(&c.b, dst, ra); j_unary(&c.b, J_NOT, dst); break;
        case IR_NEG: j_mov_rr(&c.b, dst, ra); j_unary(&c.b, J_NEG, dst); break;
        case IR_ULT: case IR_SLT: case IR_EQ: {
            uint8_t cc = (op == IR_ULT) ? 0x92 : (op == IR_SLT) ? 0x9C : 0x94;
            if (b_imm) {
                j_alu_ri(&c.b, JI_CMP, ra, imm_v);
                j_setcc(&c.b, cc, dst);
                j_movzx8(&c.b, dst, dst);
            } else {
                jit_cmp_set(&c, cc, dst, ra, rb);
            }
            break;
        }
        case IR_NEZ:
            j_alu(&c.b, J_TEST, ra, ra);
            j_setcc(&c.b, 0x95, dst);                             /* setne */
            j_movzx8(&c.b, dst, dst);
            break;
        case IR_SEL:
            if (dst == ra) {
                j_alu(&c.b, J_TEST, rc, rc);
                j_cmovcc(&c.b, 0x44, dst, rb);                    /* cmove  */
            } else if (dst == rb) {
                j_alu(&c.b, J_TEST, rc, rc);
                j_cmovcc(&c.b, 0x45, dst, ra);                    /* cmovne */
            } else {
                j_mov_rr(&c.b, dst, rb);
                j_alu(&c.b, J_TEST, rc, rc);
                j_cmovcc(&c.b, 0x45, dst, ra);
            }
            break;
        case IR_POPCNT: j_popcnt(&c.b, dst, ra); break;
        case IR_MULHI: {
            /* The right operand must survive `mov rax, <left>`, so move it out
             * of rax/rdx first; then free rax and rdx, in that order, so that a
             * left operand living in rdx is read before rdx is spilled. */
            int rb2 = rb;
            if (rb2 == JR_RAX || rb2 == JR_RDX) {
                rb2 = jit_take_reg_ex(&c, JR_RAX, JR_RDX, ra);
                j_mov_rr(&c.b, rb2, rb);
                c.occupant[rb2] = -1;
            }
            if (c.occupant[JR_RAX] >= 0) jit_spill(&c, JR_RAX);
            j_mov_rr(&c.b, JR_RAX, ra);
            if (c.occupant[JR_RDX] >= 0) jit_spill(&c, JR_RDX);
            j_mul_rm(&c.b, rb2);
            j_mov_rr(&c.b, dst, JR_RDX);
            break;
        }
        case IR_SHL: case IR_SHR: case IR_SAR: {
            int kind = (op == IR_SHL) ? J_SHL : (op == IR_SHR) ? J_SHR : J_SAR;
            if (b_imm) {
                uint64_t k = (uint64_t)imm_v;
                if (k >= 64) {
                    if (kind == J_SAR) {
                        j_mov_rr(&c.b, dst, ra);
                        j_shift_imm(&c.b, J_SAR, dst, 63);
                    } else {
                        j_mov_ri(&c.b, dst, 0);
                    }
                } else if (k == 0) {
                    j_mov_rr(&c.b, dst, ra);
                } else {
                    j_mov_rr(&c.b, dst, ra);
                    j_shift_imm(&c.b, kind, dst, (int)k);
                }
            } else {
                /* The scratch holds the out-of-range result and is written
                 * before `ra` is copied into the destination, so it may not be
                 * ra, the destination, or the count. */
                int tmp = jit_take_reg_ex(&c, ra, dst, rb);
                c.occupant[tmp] = -1;           /* scratch, live for one op */
                jit_shift_var(&c, kind, dst, ra, rb, tmp);
            }
            break;
        }
        default: c.ok = 0; break;
        }
    }

    if (c.ok) {
        for (int i = 0; i < p->n_outputs; i++) {
            int node = p->outputs[i].node;
            int r = jit_in_reg(&c, node, -1, -1);
            j_store(&c.b, JR_R15, p->outputs[i].slot * 8, r);
        }
        int frame = ((c.n_spill * 8) + 15) & ~15;
        c.b.buf[frame_patch + 0] = (uint8_t)frame;
        c.b.buf[frame_patch + 1] = (uint8_t)(frame >> 8);
        c.b.buf[frame_patch + 2] = (uint8_t)(frame >> 16);
        c.b.buf[frame_patch + 3] = (uint8_t)(frame >> 24);
        j_addsub_rsp(&c.b, 1, frame);
#if MT_JIT_WIN64
        j_pop(&c.b, JR_RDI); j_pop(&c.b, JR_RSI);
#endif
        j_pop(&c.b, JR_R15); j_pop(&c.b, JR_R14); j_pop(&c.b, JR_R13);
        j_pop(&c.b, JR_R12); j_pop(&c.b, JR_RBX);
        jb_byte(&c.b, 0xC3);
    }

    mt_taint_fn fn = NULL;
    if (c.ok && !c.b.overflow) {
        size_t sz = (c.b.len + 4095) & ~(size_t)4095;
        void *mem = mt_code_alloc(sz);
        if (mem) {
            memcpy(mem, c.b.buf, c.b.len);
            if (mt_code_protect_exec(mem, sz, c.b.len) == 0) {
                fn = (mt_taint_fn)mem;
                if (code_out) *code_out = mem;
                if (size_out) *size_out = sz;
            } else {
                mt_code_free(mem, sz);
            }
        }
    }
    free(c.b.buf); free(c.loc); free(c.last); free(in_kind); free(in_slot);
    return fn;
}

#endif /* TAINT_JIT_X64_H */
