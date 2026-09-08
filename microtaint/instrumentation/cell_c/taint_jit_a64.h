#ifndef TAINT_JIT_A64_H
#define TAINT_JIT_A64_H

/*
 * taint_jit_a64 — emit machine code for a lowered taint program, AArch64 host.
 * ===========================================================================
 *
 * The sibling of taint_jit_x64.h, consuming exactly the same program: the IR
 * comes from p-code and is ISA-general, and a backend here is about the HOST
 * the analysis runs on, not about the guest it analyses.  Every guest
 * architecture the engine lifts therefore reaches native taint code on either
 * host, which is the whole point of having two.
 *
 * Where it differs from the x86-64 backend, and why
 * ------------------------------------------------
 *   THREE-ADDRESS.  `and x5, x6, x7` needs no move to set up a destination, so
 *   the x64 backend's care about landing the destination on the left operand's
 *   register buys nothing here and is gone.
 *
 *   TWENTY-FIVE ALLOCATABLE REGISTERS against x86-64's ten.  Spilling is
 *   therefore rare rather than routine, but the same linear scan is kept: a
 *   taint program is bounded only by the guest instruction that produced it,
 *   and a wide SIMD rule can still outrun the register file.
 *
 *   NO IMMEDIATE FORMS for the bitwise operations.  AArch64's logical
 *   immediates are the bitmask encoding, which most of the masks a taint rule
 *   uses do not fit; add, subtract, compare and the shifts do take one
 *   directly, and those are taken.  Everything else materialises the constant,
 *   which the wide register file can afford.
 *
 *   POPCOUNT LIVES IN THE VECTOR UNIT.  `cnt` has no scalar form, so a
 *   population count round-trips through v0 rather than declining: the parity
 *   flag of every x86 arithmetic instruction reaches it, so declining would
 *   cost most of the x86 bank.
 *
 * Division and count-leading-zeros are DECLINED, as on x86-64, and for the same
 * reason: a taint rule reaches them rarely enough that the interpreter is the
 * cheaper answer.
 *
 * Register discipline
 * -------------------
 *   x0, x1, x2      the argument pointers (values, taint, out).  AAPCS64 hands
 *                   them over in exactly the registers we want them in, so
 *                   unlike the x86-64 backend nothing has to be moved at entry.
 *   x3 - x17        allocatable, and caller-saved: using these and no others
 *                   means the emitted function needs no prologue beyond its
 *                   spill frame and no epilogue beyond `ret`.
 *   x19 - x28       allocatable, callee-saved, so they are saved on entry and
 *                   restored on exit -- but only the ones actually taken.
 *   x18             never used: it is the platform register, reserved by some
 *                   ABIs and free on others, and there are twenty-five others.
 *   x29, x30, sp    frame pointer, link register, stack.
 */

#include <stdint.h>
#include <string.h>
#include <sys/mman.h>

#define JA_SP    31           /* also XZR, depending on the instruction */
#define JA_ZR    31
#define JA_VALS  0
#define JA_TAINT 1
#define JA_OUT   2

/* Allocation pool: the caller-saved scratch first, so a short program never
 * touches a callee-saved register and never needs to save one. */
static const int JIT_POOL[] = {
     3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
    19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
};
#define JIT_NPOOL ((int)(sizeof(JIT_POOL) / sizeof(JIT_POOL[0])))
#define JA_NREGS 32

#define JIT_SPILL_BASE 100
#define JIT_MAX_SPILL  256
/* A constant that has not been materialised.  Unlike x86-64 this backend
 * usually has to put it in a register anyway, but deferring still pays: a
 * constant feeding only an add, a shift or a compare never becomes one. */
#define JIT_CONST_LOC  (JIT_SPILL_BASE + JIT_MAX_SPILL)

/* Condition codes.  Inverting one is a flip of the low bit, which is what
 * `cset` relies on. */
#define JA_EQ 0x0
#define JA_NE 0x1
#define JA_HS 0x2
#define JA_LO 0x3
#define JA_LT 0xB

typedef struct {
    uint8_t *buf;
    size_t   len, cap;
    int      overflow;
} JitBuf;

static void jb_word(JitBuf *b, uint32_t insn) {
    if (b->len + 4 > b->cap) { b->overflow = 1; return; }
    b->buf[b->len++] = (uint8_t)insn;
    b->buf[b->len++] = (uint8_t)(insn >> 8);
    b->buf[b->len++] = (uint8_t)(insn >> 16);
    b->buf[b->len++] = (uint8_t)(insn >> 24);
}

static void jb_patch(JitBuf *b, size_t at, uint32_t insn) {
    if (at + 4 > b->len) { b->overflow = 1; return; }
    b->buf[at + 0] = (uint8_t)insn;
    b->buf[at + 1] = (uint8_t)(insn >> 8);
    b->buf[at + 2] = (uint8_t)(insn >> 16);
    b->buf[at + 3] = (uint8_t)(insn >> 24);
}

/* ── instruction encodings ───────────────────────────────────────────── */

/* ldr/str Xt, [Xn, #off] -- unsigned offset, scaled by 8, so it reaches 32760
 * bytes.  A slot past that is refused rather than wrapped: the mask would
 * silently address the wrong word, which is a wrong taint. */
static int j_off12(JitBuf *b, int byte_off) {
    uint32_t imm12 = (uint32_t)byte_off >> 3;
    if (byte_off < 0 || (byte_off & 7) || imm12 > 0xFFF) { b->overflow = 1; return 0; }
    return (int)imm12;
}
static void j_load(JitBuf *b, int rt, int rn, int byte_off) {
    uint32_t imm12 = (uint32_t)j_off12(b, byte_off);
    jb_word(b, 0xF9400000u | (imm12 << 10) | ((uint32_t)rn << 5) | (uint32_t)rt);
}
static void j_store(JitBuf *b, int rn, int byte_off, int rt) {
    uint32_t imm12 = (uint32_t)j_off12(b, byte_off);
    jb_word(b, 0xF9000000u | (imm12 << 10) | ((uint32_t)rn << 5) | (uint32_t)rt);
}

/* Shifted-register data processing, shift 0.  `base` carries the opcode. */
static void j_dp3(JitBuf *b, uint32_t base, int rd, int rn, int rm) {
    jb_word(b, base | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rd);
}

#define JA_AND 0x8A000000u
#define JA_ORR 0xAA000000u
#define JA_EOR 0xCA000000u
#define JA_ADD 0x8B000000u
#define JA_SUB 0xCB000000u
#define JA_ORN 0xAA200000u
#define JA_SUBS 0xEB000000u          /* cmp Xn, Xm  == subs xzr, Xn, Xm */
#define JA_LSLV 0x9AC02000u
#define JA_LSRV 0x9AC02400u
#define JA_ASRV 0x9AC02800u
#define JA_UMULH 0x9BC07C00u

static void j_mov_rr(JitBuf *b, int rd, int rm) {
    if (rd == rm) return;
    j_dp3(b, JA_ORR, rd, JA_ZR, rm);              /* orr Xd, xzr, Xm */
}

/* add/sub Xd, Xn, #imm12 (no shift).  `sub` when `is_sub`. */
static void j_addsub_imm(JitBuf *b, int is_sub, int rd, int rn, uint32_t imm12) {
    uint32_t base = is_sub ? 0xD1000000u : 0x91000000u;
    jb_word(b, base | ((imm12 & 0xFFF) << 10) | ((uint32_t)rn << 5) | (uint32_t)rd);
}

/* cmp Xn, #imm12  == subs xzr, Xn, #imm12 */
static void j_cmp_imm(JitBuf *b, int rn, uint32_t imm12) {
    jb_word(b, 0xF1000000u | ((imm12 & 0xFFF) << 10)
               | ((uint32_t)rn << 5) | (uint32_t)JA_ZR);
}

static void j_cmp_rr(JitBuf *b, int rn, int rm) {
    j_dp3(b, JA_SUBS, JA_ZR, rn, rm);
}

/* csel Xd, Xn, Xm, cond  ->  Xd = cond ? Xn : Xm */
static void j_csel(JitBuf *b, int rd, int rn, int rm, uint32_t cond) {
    jb_word(b, 0x9A800000u | ((uint32_t)rm << 16) | (cond << 12)
               | ((uint32_t)rn << 5) | (uint32_t)rd);
}

/* cset Xd, cond  ==  csinc Xd, xzr, xzr, invert(cond) */
static void j_cset(JitBuf *b, int rd, uint32_t cond) {
    jb_word(b, 0x9A800400u | ((uint32_t)JA_ZR << 16) | ((cond ^ 1u) << 12)
               | ((uint32_t)JA_ZR << 5) | (uint32_t)rd);
}

/* mul Xd, Xn, Xm  ==  madd Xd, Xn, Xm, xzr */
static void j_mul(JitBuf *b, int rd, int rn, int rm) {
    jb_word(b, 0x9B000000u | ((uint32_t)rm << 16) | ((uint32_t)JA_ZR << 10)
               | ((uint32_t)rn << 5) | (uint32_t)rd);
}

/* Shifts by a constant, as the bitfield moves they really are. */
static void j_lsl_imm(JitBuf *b, int rd, int rn, int k) {
    uint32_t immr = (uint32_t)((64 - k) & 63), imms = (uint32_t)(63 - k);
    jb_word(b, 0xD3400000u | (immr << 16) | (imms << 10)
               | ((uint32_t)rn << 5) | (uint32_t)rd);
}
static void j_lsr_imm(JitBuf *b, int rd, int rn, int k) {
    jb_word(b, 0xD3400000u | ((uint32_t)k << 16) | (63u << 10)
               | ((uint32_t)rn << 5) | (uint32_t)rd);
}
static void j_asr_imm(JitBuf *b, int rd, int rn, int k) {
    jb_word(b, 0x93400000u | ((uint32_t)k << 16) | (63u << 10)
               | ((uint32_t)rn << 5) | (uint32_t)rd);
}

/* A 64-bit constant, in as few moves as it takes: the 16-bit chunks that are
 * already zero (or already all-ones, starting from `movn`) cost nothing. */
static void j_mov_ri(JitBuf *b, int rd, uint64_t v) {
    int zero_chunks = 0, ones_chunks = 0;
    for (int i = 0; i < 4; i++) {
        uint16_t chunk = (uint16_t)(v >> (16 * i));
        if (chunk == 0x0000) zero_chunks++;
        if (chunk == 0xFFFF) ones_chunks++;
    }
    if (ones_chunks > zero_chunks) {
        /* movn seeds every OTHER bit to one, so the inverted value's zero
         * chunks are the ones that then need no movk. */
        uint64_t inv = ~v;
        int first = -1;
        for (int i = 0; i < 4; i++)
            if ((uint16_t)(inv >> (16 * i)) != 0) { first = i; break; }
        if (first < 0) first = 0;
        jb_word(b, 0x92800000u | ((uint32_t)first << 21)
                   | ((uint32_t)((uint16_t)(inv >> (16 * first))) << 5)
                   | (uint32_t)rd);
        for (int i = 0; i < 4; i++) {
            uint16_t chunk = (uint16_t)(v >> (16 * i));
            if (i == first || chunk == 0xFFFF) continue;
            jb_word(b, 0xF2800000u | ((uint32_t)i << 21)
                       | ((uint32_t)chunk << 5) | (uint32_t)rd);
        }
        return;
    }
    int first = -1;
    for (int i = 0; i < 4; i++)
        if ((uint16_t)(v >> (16 * i)) != 0) { first = i; break; }
    if (first < 0) {                       /* the constant is zero */
        jb_word(b, 0xD2800000u | (uint32_t)rd);
        return;
    }
    jb_word(b, 0xD2800000u | ((uint32_t)first << 21)
               | ((uint32_t)((uint16_t)(v >> (16 * first))) << 5) | (uint32_t)rd);
    for (int i = first + 1; i < 4; i++) {
        uint16_t chunk = (uint16_t)(v >> (16 * i));
        if (chunk == 0) continue;
        jb_word(b, 0xF2800000u | ((uint32_t)i << 21)
                   | ((uint32_t)chunk << 5) | (uint32_t)rd);
    }
}

/* Population count, through v0: fmov / cnt.8b / addv.8b / fmov back. */
static void j_popcnt(JitBuf *b, int rd, int rn) {
    jb_word(b, 0x9E670000u | ((uint32_t)rn << 5) | 0u);     /* fmov d0, Xn   */
    jb_word(b, 0x0E205800u);                                 /* cnt  v0.8b    */
    jb_word(b, 0x0E31B800u);                                 /* addv b0, v0.8b*/
    jb_word(b, 0x9E660000u | (0u << 5) | (uint32_t)rd);      /* fmov Xd, d0   */
}

static void j_ret(JitBuf *b) { jb_word(b, 0xD65F03C0u); }

/* sub/add sp, sp, #imm12 */
static void j_sp_adjust(JitBuf *b, int is_sub, uint32_t imm12) {
    uint32_t base = is_sub ? 0xD1000000u : 0x91000000u;
    jb_word(b, base | ((imm12 & 0xFFF) << 10)
               | ((uint32_t)JA_SP << 5) | (uint32_t)JA_SP);
}

static int j_fits_imm12(uint64_t v) { return v <= 0xFFFu; }

/* ── the allocator ───────────────────────────────────────────────────── */

typedef struct {
    JitBuf   b;
    int     *loc;                     /* node -> register, or SPILL_BASE+slot */
    int     *last;                    /* node -> index of its last use */
    int      occupant[JA_NREGS];      /* physreg -> node, or -1 */
    uint8_t  spill_used[JIT_MAX_SPILL];
    uint8_t  callee_used[JA_NREGS];   /* which x19..x28 were ever taken */
    const uint64_t *imm;
    int      n_spill;
    int      spill_base;              /* byte offset of the spill area in the frame */
    int      cur;
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
    j_store(&c->b, JA_SP, c->spill_base + slot * 8, reg);
    c->loc[n] = JIT_SPILL_BASE + slot;
    c->occupant[reg] = -1;
}

/* Take a register, evicting the value whose next use is furthest away.
 *
 * The excluded registers are excluded ABSOLUTELY, not merely preferred against:
 * an operand that dies at this node has already been released, so its register
 * reads as free, but its VALUE is still needed until the instruction consuming
 * it is emitted.  Handing that register out as a scratch would silently compute
 * with the wrong operand.
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
        if (c->occupant[r] < 0) { c->callee_used[r] = 1; return r; }
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
    c->callee_used[best] = 1;
    return best;
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
    j_load(&c->b, r, JA_SP, c->spill_base + slot * 8);
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

/* dst = ra <shift> rb, saturating to zero (or to the sign, for an arithmetic
 * shift) when the count is 64 or more.  AArch64's variable shifts read only the
 * low six bits of the count, exactly as x86's do, so the out-of-range case has
 * to be selected in either way.
 *
 * The count is compared BEFORE the shift, not after.  The destination may be
 * the count's own register -- three-address operations let it, and the
 * allocator takes the offer -- and a comparison placed after the shift would
 * then read the shifted result instead of the count.  Nothing between the two
 * touches the flags. */
static void jit_shift_var(JitCtx *c, uint32_t base, int arith,
                          int dst, int ra, int rb, int tmp) {
    if (arith) j_asr_imm(&c->b, tmp, ra, 63);
    else       j_mov_ri(&c->b, tmp, 0);
    j_cmp_imm(&c->b, rb, 64);
    j_dp3(&c->b, base, dst, ra, rb);
    j_csel(&c->b, dst, tmp, dst, JA_HS);
}

typedef void (*mt_taint_fn)(const uint64_t *, const uint64_t *, uint64_t *);

/* True when every opcode in the program has an emission rule here.  Division
 * and count-leading-zeros are the exceptions, as on x86-64: a taint rule
 * reaches them rarely enough that falling back to the interpreter is cheaper
 * than the fixed-register and feature handling they would need. */
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
    /* Six words is the worst case for one node (a four-word constant plus its
     * operation), and the frame and the stores are the rest. */
    size_t cap = 256 + (size_t)n * 32 + (size_t)p->n_outputs * 8;
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
    for (int i = 0; i < JA_NREGS; i++) c.occupant[i] = -1;
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

    /* The frame is one `sub sp` whose size is patched once the spill
     * high-water mark and the set of callee-saved registers taken are known.
     * The ten callee-saved registers are saved at the bottom of it; the spills
     * live above them.  Which ones were taken is not known yet either, so the
     * saves are emitted as a fixed block of ten stores and rewritten to `nop`
     * for the registers that stayed untouched. */
    size_t frame_patch = c.b.len;
    j_sp_adjust(&c.b, 1, 0);
    size_t save_patch = c.b.len;
    for (int i = 0; i < 10; i++) j_store(&c.b, JA_SP, i * 8, 19 + i);
    c.spill_base = 10 * 8;

    for (int i = 0; i < n && c.ok; i++) {
        c.cur = i;
        uint8_t op = p->op[i];
        int an = p->a[i], bn = p->b[i], cn = p->c[i];

        /* A constant costs nothing until something needs it in a register. */
        if (op == IR_CONST) { c.loc[i] = JIT_CONST_LOC; continue; }

        /* Can this op take its right operand as an immediate?  Only the ones
         * whose AArch64 form has an immediate field wide enough to be worth
         * it: the logical operations use the bitmask encoding, which most
         * taint masks do not fit, so they materialise instead. */
        int b_imm = 0;
        uint64_t imm_v = 0;
        if (bn >= 0 && c.loc[bn] == JIT_CONST_LOC) {
            switch (op) {
            case IR_ADD: case IR_SUB: case IR_ULT: case IR_SLT: case IR_EQ:
                if (j_fits_imm12(p->imm[bn])) { b_imm = 1; imm_v = p->imm[bn]; }
                break;
            case IR_SHL: case IR_SHR: case IR_SAR:
                b_imm = 1; imm_v = p->imm[bn];
                break;
            default: break;
            }
        }

        int ra = -1, rb = -1, rc = -1;
        if (an >= 0) ra = jit_in_reg(&c, an, -1, -1);
        if (bn >= 0 && !b_imm) rb = jit_in_reg(&c, bn, ra, -1);
        if (cn >= 0) rc = jit_in_reg(&c, cn, ra, rb);
        /* Operands dying here free their registers before the destination is
         * chosen, so the destination usually lands on one of them.  Unlike
         * x86-64 that is a convenience rather than a saved move: every
         * operation below is three-address. */
        if (an >= 0 && c.last[an] == i) jit_release(&c, an);
        if (bn >= 0 && c.last[bn] == i && bn != an) jit_release(&c, bn);
        if (cn >= 0 && c.last[cn] == i && cn != an && cn != bn) jit_release(&c, cn);

        int dst = jit_take_reg_ex(&c, -1, -1, -1);
        c.occupant[dst] = i;
        c.loc[i] = dst;

        switch (op) {
        case IR_INV: j_load(&c.b, dst, JA_VALS, in_slot[i] * 8); break;
        case IR_INT: j_load(&c.b, dst, JA_TAINT, in_slot[i] * 8); break;
        case IR_AND: j_dp3(&c.b, JA_AND, dst, ra, rb); break;
        case IR_OR:  j_dp3(&c.b, JA_ORR, dst, ra, rb); break;
        case IR_XOR: j_dp3(&c.b, JA_EOR, dst, ra, rb); break;
        case IR_ADD:
            if (b_imm) j_addsub_imm(&c.b, 0, dst, ra, (uint32_t)imm_v);
            else       j_dp3(&c.b, JA_ADD, dst, ra, rb);
            break;
        case IR_SUB:
            if (b_imm) j_addsub_imm(&c.b, 1, dst, ra, (uint32_t)imm_v);
            else       j_dp3(&c.b, JA_SUB, dst, ra, rb);
            break;
        case IR_MUL: j_mul(&c.b, dst, ra, rb); break;
        case IR_MULHI: j_dp3(&c.b, JA_UMULH, dst, ra, rb); break;
        case IR_NOT: j_dp3(&c.b, JA_ORN, dst, JA_ZR, ra); break;
        case IR_NEG: j_dp3(&c.b, JA_SUB, dst, JA_ZR, ra); break;
        case IR_ULT: case IR_SLT: case IR_EQ: {
            uint32_t cond = (op == IR_ULT) ? JA_LO : (op == IR_SLT) ? JA_LT : JA_EQ;
            if (b_imm) j_cmp_imm(&c.b, ra, (uint32_t)imm_v);
            else       j_cmp_rr(&c.b, ra, rb);
            j_cset(&c.b, dst, cond);
            break;
        }
        case IR_NEZ:
            j_cmp_imm(&c.b, ra, 0);
            j_cset(&c.b, dst, JA_NE);
            break;
        case IR_SEL:
            /* dst = c ? a : b, the same order the interpreter uses. */
            j_cmp_imm(&c.b, rc, 0);
            j_csel(&c.b, dst, ra, rb, JA_NE);
            break;
        case IR_POPCNT: j_popcnt(&c.b, dst, ra); break;
        case IR_SHL: case IR_SHR: case IR_SAR: {
            int arith = (op == IR_SAR);
            uint32_t base = (op == IR_SHL) ? JA_LSLV
                          : (op == IR_SHR) ? JA_LSRV : JA_ASRV;
            if (b_imm) {
                if (imm_v >= 64) {
                    if (arith) j_asr_imm(&c.b, dst, ra, 63);
                    else       j_mov_ri(&c.b, dst, 0);
                } else if (imm_v == 0) {
                    j_mov_rr(&c.b, dst, ra);
                } else if (op == IR_SHL) {
                    j_lsl_imm(&c.b, dst, ra, (int)imm_v);
                } else if (op == IR_SHR) {
                    j_lsr_imm(&c.b, dst, ra, (int)imm_v);
                } else {
                    j_asr_imm(&c.b, dst, ra, (int)imm_v);
                }
            } else {
                /* The scratch holds the out-of-range result and must survive
                 * until the select, so it may not be the destination, the
                 * shifted value, or the count. */
                int tmp = jit_take_reg_ex(&c, ra, dst, rb);
                c.occupant[tmp] = -1;           /* scratch, live for one op */
                jit_shift_var(&c, base, arith, dst, ra, rb, tmp);
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
            j_store(&c.b, JA_OUT, p->outputs[i].slot * 8, r);
        }
        /* Restore only what was saved, then unwind the frame. */
        for (int i = 0; i < 10; i++)
            if (c.callee_used[19 + i]) j_load(&c.b, 19 + i, JA_SP, i * 8);
        int frame = (c.spill_base + c.n_spill * 8 + 15) & ~15;
        if (frame > 0xFFF) c.ok = 0;          /* one `sub sp` cannot reach it */
        jb_patch(&c.b, frame_patch,
                 0xD1000000u | ((uint32_t)frame << 10)
                 | ((uint32_t)JA_SP << 5) | (uint32_t)JA_SP);
        for (int i = 0; i < 10; i++) {
            if (c.callee_used[19 + i]) continue;
            jb_patch(&c.b, save_patch + (size_t)i * 4, 0xD503201Fu);  /* nop */
        }
        j_sp_adjust(&c.b, 0, (uint32_t)frame);
        j_ret(&c.b);
    }

    mt_taint_fn fn = NULL;
    if (c.ok && !c.b.overflow) {
        size_t sz = (c.b.len + 4095) & ~(size_t)4095;
        void *mem = mmap(NULL, sz, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mem != MAP_FAILED) {
            memcpy(mem, c.b.buf, c.b.len);
            if (mprotect(mem, sz, PROT_READ | PROT_EXEC) == 0) {
                __builtin___clear_cache((char *)mem, (char *)mem + c.b.len);
                fn = (mt_taint_fn)mem;
                if (code_out) *code_out = mem;
                if (size_out) *size_out = sz;
            } else {
                munmap(mem, sz);
            }
        }
    }
    free(c.b.buf); free(c.loc); free(c.last); free(in_kind); free(in_slot);
    return fn;
}

#endif /* TAINT_JIT_A64_H */
