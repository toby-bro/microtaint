#ifndef TAINT_CORE_H
#define TAINT_CORE_H

/*
 * taint_core.h — per-p-code-op taint composition, one pass, all outputs.
 * ======================================================================
 *
 * The engine's historical model evaluates ONE compiled expression per output
 * SLICE: `add rax, rbx` builds seven programs (RAX, CF, OF, ZF, SF, PF, AF),
 * each of which re-derives the answer from RAX/RBX, and most of which get there
 * by RE-EXECUTING the instruction in SLEIGH twice (the two differential
 * corners, ~776 ns per cell).  That is the cost this header removes.
 *
 * Here taint is composed the way GLIFT/CellIFT compose shadow logic: one rule
 * per p-code opcode, applied in a SINGLE forward pass over the lifted p-code,
 * carrying a taint word beside every varnode.  Every output of the instruction
 * — result registers and flags alike — is simply read off the taint frame when
 * the pass ends.  There is no flag special case, no waiting on an intermediate,
 * and no second pass.
 *
 * The rules
 * ---------
 * Three kinds, and the point of splitting them is that only the third costs
 * anything:
 *
 *   ROUTE   the output taint is a permutation/mask of the input taint, so it is
 *           EXACT and free: COPY, ZEXT, SEXT, TRUNC, SUBPIECE, PIECE, NEGATE,
 *           XOR, BOOL_XOR, BOOL_NEGATE, AND/OR (value-aware), shifts by a known
 *           amount, MULT by a known power of two.  1-4 machine ops.
 *
 *   DIFF    the output taint follows from the two extremal corners
 *               lo = f(v & ~t)     hi = f(v | t)
 *           evaluated INLINE, in machine arithmetic, with no re-execution.
 *           This is exact — not a floor — wherever f is monotone in every input
 *           bit, which covers the whole carry-coupled family: ADD, SUB, 2COMP,
 *           INT_CARRY, and the unsigned/signed comparisons.  See
 *           `mt_add_taint` for the monotonicity argument; the direct-bit OR
 *           `| ta | tb` is what repairs the classic two-corner under-taint.
 *
 *   FLOOR   a provably sound over-approximation, for the ops where no cheap
 *           exact form exists: MULT/DIV/REM by a variable, POPCOUNT, shifts by
 *           a tainted amount, FLOAT_*.  Avalanche over the output width.
 *
 * And one op is genuinely harder than a differential, which is why the
 * classification matters: INT_SCARRY / INT_SBORROW (signed overflow) is
 * c_{w-1} XOR c_w, an XOR of two monotone functions and therefore NOT monotone
 * itself — the two corners can agree while the flag varies in between, and they
 * can disagree while the flag is constant.  `mt_scarry_taint` enumerates the
 * (msb_a, msb_b, carry-into-msb) cube instead, at most eight combinations, and
 * is exact.  Same story, one level down, for INT_EQUAL.
 *
 * Cost accounting
 * ---------------
 * Every rule charges its primitive-op count to `MtTaintCost`, so "how many
 * operations does propagating taint through this instruction actually take,
 * flags included" is a measured number rather than an estimate.  The counts are
 * the ops the rule performs on taint/value words; frame reads and writes are
 * counted separately so the two can be told apart.
 *
 * Soundness contract
 * ------------------
 * Under-taint is never acceptable.  Every rule here is either exact or an
 * over-approximation of the true per-bit sensitivity, and `MT_TAINT_DECLINE`
 * exists for the cases this pass will not answer (opaque CALLOTHER, a p-code
 * shape outside the modelled set): the caller must then fall back to the
 * monolithic differential rather than trust a partial answer.
 */

#include <stdint.h>
#include "pcode_defs.h"

/* ── cost model ───────────────────────────────────────────────────────
 * One counter per rule class plus a primitive-op total.  `ops` is what the
 * user-facing "operations per instruction" figure reports. */
typedef struct {
    uint32_t pcode_ops;   /* p-code ops the pass visited */
    uint32_t ops;         /* primitive machine ops charged to taint */
    uint32_t n_route;     /* ops answered by exact routing (free) */
    uint32_t n_diff;      /* ops answered by an inlined differential */
    uint32_t n_floor;     /* ops answered by a soundness floor */
    uint32_t n_cube;      /* ops answered by cube enumeration (SCARRY family) */
    uint32_t reads;       /* frame reads (value + taint) */
    uint32_t writes;      /* frame writes */
    uint32_t forks;       /* tainted-conditional forks taken */
} MtTaintCost;

/* Return codes for the taint pass. */
#define MT_TAINT_OK        0
#define MT_TAINT_DECLINE   1   /* shape outside the modelled set: fall back */

/* ── small helpers ───────────────────────────────────────────────────── */

static inline uint64_t mt_mask(int sz) {
    static const uint64_t MT[9] = {
        0, 0xFFULL, 0xFFFFULL, 0xFFFFFFULL, 0xFFFFFFFFULL,
        0xFFFFFFFFFFULL, 0xFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL
    };
    return (sz <= 0) ? 0 : (sz >= 8 ? MT[8] : MT[sz]);
}

/* Sign-extend the low `sz` bytes of v to 64 bits. */
static inline uint64_t mt_sext(uint64_t v, int sz) {
    if (sz <= 0 || sz >= 8) return v;
    uint64_t m = mt_mask(sz);
    uint64_t sb = 1ULL << (sz * 8 - 1);
    v &= m;
    return (v & sb) ? (v | ~m) : v;
}

/* ── ADD / SUB: the inlined differential, and why it is EXACT ──────────
 *
 * lo = f(v & ~t) and hi = f(v | t) are the extremal corners.  A ripple carry
 * c_i is a monotone function of every input bit (c_{i+1} = maj(a_i,b_i,c_i),
 * monotone in all three, composed by induction), so c_i can vary over the taint
 * cube IFF it differs between the two corners:
 *
 *     c_i varies  <=>  c_i(lo) != c_i(hi)
 *
 * and since sum_i = a_i ^ b_i ^ c_i,
 *
 *     (lo ^ hi)_i = (ta_i ^ tb_i) ^ (c_i(lo) ^ c_i(hi))
 *
 * Where m_i = (ta|tb)_i is 0 the first term vanishes and (lo^hi)_i is exactly
 * the carry-variability bit; where m_i is 1 the bit is tainted anyway.  Hence
 *
 *     sum_taint = (lo ^ hi) | ta | tb          — exact, 6 machine ops
 *
 * This is the repair the bare two-corner differential was missing: without
 * `| ta | tb`, a bit whose two inputs both flip cancels in the XOR and the
 * carry it generates is reported clean (the historical `add` under-taint).
 */
static inline uint64_t mt_add_taint(uint64_t a_v, uint64_t a_t,
                                    uint64_t b_v, uint64_t b_t,
                                    int sz, MtTaintCost *c) {
    uint64_t m = mt_mask(sz);
    uint64_t lo = ((a_v & ~a_t) + (b_v & ~b_t)) & m;
    uint64_t hi = ((a_v | a_t) + (b_v | b_t)) & m;
    if (c) { c->ops += 8; c->n_diff++; }
    return ((lo ^ hi) | a_t | b_t) & m;
}

/* a - b, via the same argument: subtraction is monotone increasing in a and
 * monotone decreasing in b, so the corners are (min a, max b) and (max a,
 * min b). */
static inline uint64_t mt_sub_taint(uint64_t a_v, uint64_t a_t,
                                    uint64_t b_v, uint64_t b_t,
                                    int sz, MtTaintCost *c) {
    uint64_t m = mt_mask(sz);
    uint64_t lo = ((a_v & ~a_t) - (b_v | b_t)) & m;
    uint64_t hi = ((a_v | a_t) - (b_v & ~b_t)) & m;
    if (c) { c->ops += 8; c->n_diff++; }
    return ((lo ^ hi) | a_t | b_t) & m;
}

/* Unsigned carry-out of a + b over `sz` bytes.  Monotone in every input bit,
 * so the corners settle it exactly.  Computed at 65-bit width by comparing the
 * truncated sum against an operand (the standard carry test). */
static inline uint64_t mt_carry_taint(uint64_t a_v, uint64_t a_t,
                                      uint64_t b_v, uint64_t b_t,
                                      int sz, MtTaintCost *c) {
    uint64_t m = mt_mask(sz);
    uint64_t alo = a_v & ~a_t & m, blo = b_v & ~b_t & m;
    uint64_t ahi = (a_v | a_t) & m,  bhi = (b_v | b_t) & m;
    uint64_t clo = (((alo + blo) & m) < alo) ? 1 : 0;
    uint64_t chi = (((ahi + bhi) & m) < ahi) ? 1 : 0;
    if (c) { c->ops += 10; c->n_diff++; }
    return clo ^ chi;
}

/* Carry INTO the msb of a + b, at the two corners.  Used by the signed-overflow
 * cube below: it is the third free variable of OF = c_{w-1} ^ c_w. */
static inline void mt_carry_in_msb(uint64_t a_v, uint64_t a_t,
                                   uint64_t b_v, uint64_t b_t,
                                   int sz, int *lo_out, int *hi_out) {
    int bits = sz * 8;
    uint64_t sub = (bits >= 64) ? ~0ULL >> 1 : ((1ULL << (bits - 1)) - 1);
    uint64_t alo = a_v & ~a_t & sub, blo = b_v & ~b_t & sub;
    uint64_t ahi = (a_v | a_t) & sub, bhi = (b_v | b_t) & sub;
    *lo_out = (int)(((alo + blo) >> (bits - 1)) & 1);
    *hi_out = (int)(((ahi + bhi) >> (bits - 1)) & 1);
}

/* ── signed overflow: the op that a differential cannot answer ─────────
 *
 * OF = c_{w-1} XOR c_w = c XOR maj(a_msb, b_msb, c).  Each of c_{w-1} and c_w
 * is monotone, but their XOR is not, so the two extremal corners are neither
 * sufficient (they can agree while OF varies) nor necessary (they can differ
 * while OF is pinned — e.g. a_msb != b_msb forces c_w == c_{w-1}, so OF == 0
 * no matter how the low bits move).
 *
 * The three arguments are independent: a_msb and b_msb come from the operands,
 * c_{w-1} depends only on the bits below.  So enumerate their reachable sets —
 * at most 2x2x2 — and report the flag tainted iff OF takes both values.  Exact,
 * and cheap because the cube is tiny and usually degenerate.
 */
static inline uint64_t mt_scarry_taint(uint64_t a_v, uint64_t a_t,
                                       uint64_t b_v, uint64_t b_t,
                                       int sz, MtTaintCost *c) {
    int bits = sz * 8;
    int sh = bits - 1;
    int am_v = (int)((a_v >> sh) & 1), am_t = (int)((a_t >> sh) & 1);
    int bm_v = (int)((b_v >> sh) & 1), bm_t = (int)((b_t >> sh) & 1);
    int c_lo, c_hi;
    mt_carry_in_msb(a_v, a_t, b_v, b_t, sz, &c_lo, &c_hi);

    int saw0 = 0, saw1 = 0;
    for (int ai = 0; ai < 2; ai++) {
        int av = am_t ? ai : am_v;
        if (!am_t && ai) break;
        for (int bi = 0; bi < 2; bi++) {
            int bv = bm_t ? bi : bm_v;
            if (!bm_t && bi) break;
            for (int ci = 0; ci < 2; ci++) {
                int cv = ci ? c_hi : c_lo;
                if (ci && c_hi == c_lo) break;
                int cout = (av + bv + cv) >= 2 ? 1 : 0;
                if (cv ^ cout) saw1 = 1; else saw0 = 1;
            }
        }
    }
    if (c) { c->ops += 18; c->n_cube++; }
    return (saw0 && saw1) ? 1 : 0;
}

/* Signed borrow: overflow of a - b, i.e. of a + ~b + 1.  ~b carries b's taint
 * bit-for-bit, and the +1 rides in as the carry-in, so the same cube applies
 * with b replaced by its complement.  The carry-into-msb corners must be taken
 * on that same complemented operand, which `mt_carry_in_msb` does once the
 * caller hands it (~b, tb) — the +1 is folded by adding it to the low half. */
static inline uint64_t mt_sborrow_taint(uint64_t a_v, uint64_t a_t,
                                        uint64_t b_v, uint64_t b_t,
                                        int sz, MtTaintCost *c) {
    int bits = sz * 8;
    int sh = bits - 1;
    uint64_t nb_v = ~b_v;
    int am_v = (int)((a_v >> sh) & 1), am_t = (int)((a_t >> sh) & 1);
    int bm_v = (int)((nb_v >> sh) & 1), bm_t = (int)((b_t >> sh) & 1);

    /* carry into the msb of a + ~b + 1, at both corners.  ~b is monotone
     * DECREASING in b, so the low corner of the sum pairs min(a) with min(~b)
     * = ~max(b), and the high corner pairs max(a) with ~min(b). */
    uint64_t sub = (bits >= 64) ? (~0ULL >> 1) : ((1ULL << (bits - 1)) - 1);
    uint64_t alo = a_v & ~a_t & sub, ahi = (a_v | a_t) & sub;
    uint64_t nblo = (~(b_v | b_t)) & sub, nbhi = (~(b_v & ~b_t)) & sub;
    int c_lo = (int)(((alo + nblo + 1) >> (bits - 1)) & 1);
    int c_hi = (int)(((ahi + nbhi + 1) >> (bits - 1)) & 1);

    int saw0 = 0, saw1 = 0;
    for (int ai = 0; ai < 2; ai++) {
        int av = am_t ? ai : am_v;
        if (!am_t && ai) break;
        for (int bi = 0; bi < 2; bi++) {
            int bv = bm_t ? bi : bm_v;
            if (!bm_t && bi) break;
            for (int ci = 0; ci < 2; ci++) {
                int cv = ci ? c_hi : c_lo;
                if (ci && c_hi == c_lo) break;
                int cout = (av + bv + cv) >= 2 ? 1 : 0;
                if (cv ^ cout) saw1 = 1; else saw0 = 1;
            }
        }
    }
    if (c) { c->ops += 20; c->n_cube++; }
    return (saw0 && saw1) ? 1 : 0;
}

/* ── comparisons ──────────────────────────────────────────────────────
 * `a < b` is monotone (decreasing in a, increasing in b), so the flag can be
 * true iff min(a) < max(b) and false iff max(a) >= min(b); it is tainted iff
 * both hold.  Exact, four comparisons. */
static inline uint64_t mt_cmp_taint(uint64_t a_v, uint64_t a_t,
                                    uint64_t b_v, uint64_t b_t,
                                    int sz, int is_signed, int or_equal,
                                    MtTaintCost *c) {
    uint64_t m = mt_mask(sz);
    uint64_t amin = a_v & ~a_t & m, amax = (a_v | a_t) & m;
    uint64_t bmin = b_v & ~b_t & m, bmax = (b_v | b_t) & m;
    int can_true, can_false;
    if (is_signed) {
        /* Flipping the sign bit maps the signed order onto the unsigned one.
         * The taint cube is a box in bit space and the map is a per-bit XOR, so
         * the image is still a box: its corners are (v^sb)&~t and (v^sb)|t. */
        uint64_t sb = 1ULL << (sz * 8 - 1);
        uint64_t av2 = (a_v ^ sb) & m, bv2 = (b_v ^ sb) & m;
        amin = av2 & ~a_t;   amax = (av2 | a_t) & m;
        bmin = bv2 & ~b_t;   bmax = (bv2 | b_t) & m;
    }
    if (or_equal) {
        can_true  = (amin <= bmax);
        can_false = (amax >  bmin);
    } else {
        can_true  = (amin <  bmax);
        can_false = (amax >= bmin);
    }
    if (c) { c->ops += 12; c->n_diff++; }
    return (can_true && can_false) ? 1 : 0;
}

/* Equality is not monotone either, but it is decided directly: it can hold iff
 * every KNOWN bit agrees, and it can fail iff some bit is free or a known bit
 * already differs. */
static inline uint64_t mt_eq_taint(uint64_t a_v, uint64_t a_t,
                                   uint64_t b_v, uint64_t b_t,
                                   int sz, MtTaintCost *c) {
    uint64_t m = mt_mask(sz);
    uint64_t t = (a_t | b_t) & m;
    uint64_t diff = (a_v ^ b_v) & m;
    int can_true  = ((diff & ~t) == 0);
    int can_false = (t != 0) || (diff != 0);
    if (c) { c->ops += 8; c->n_cube++; }
    return (can_true && can_false) ? 1 : 0;
}

/* ── the per-op rule ─────────────────────────────────────────────────
 * `v[]`/`t[]` hold the input values and taints, read before the op executed so
 * an output that aliases an input is still correct.  Returns the output taint
 * masked to the output width; sets *decline when the op is outside the model.
 */
static inline uint64_t mt_op_taint(const PCOp *op, const uint64_t *v,
                                   const uint64_t *t, MtTaintCost *c,
                                   int *decline) {
    int oid = op->oid;
    int osz = op->o_sz;
    int isz = op->n_ins > 0 ? op->i0_sz : osz;
    uint64_t om = mt_mask(osz);
    uint64_t a_v = v[0], a_t = t[0], b_v = v[1], b_t = t[1];
    int b_const = (op->n_ins > 1 && op->i1_sp == SP_CONST);
    int a_const = (op->n_ins > 0 && op->i0_sp == SP_CONST);

    switch (oid) {
    /* ---- exact routing: the taint is the input taint, reshaped ---- */
    case OP_COPY: case OP_CAST: case OP_INT_ZEXT: case OP_INT_TRUNC:
        if (c) { c->ops += 1; c->n_route++; }
        return a_t & om;

    case OP_INT_NEGATE:
        if (c) { c->ops += 1; c->n_route++; }
        return a_t & om;

    case OP_INT_SEXT: {
        uint64_t im = mt_mask(isz);
        uint64_t r = a_t & im;
        if ((r >> (isz * 8 - 1)) & 1) r |= ~im;   /* sign taint fills the top */
        if (c) { c->ops += 4; c->n_route++; }
        return r & om;
    }

    case OP_SUBPIECE: {
        int shift = 8 * (int)(b_const ? op->i1_off : b_v);
        if (c) { c->ops += 2; c->n_route++; }
        return (shift >= 64) ? 0 : ((a_t >> shift) & om);
    }

    case OP_PIECE: {
        int lo_bits = 8 * op->i1_sz;
        uint64_t r = (lo_bits >= 64) ? 0 : (a_t << lo_bits);
        if (c) { c->ops += 3; c->n_route++; }
        return (r | (b_t & mt_mask(op->i1_sz))) & om;
    }

    case OP_INT_XOR: case OP_BOOL_XOR:
        /* Exact when the operands are independent; a shared source is caught by
         * the reconvergence check, which routes that output to the monolithic
         * differential rather than trusting this OR. */
        if (c) { c->ops += 2; c->n_route++; }
        return (a_t | b_t) & om;

    case OP_BOOL_NEGATE:
        if (c) { c->ops += 1; c->n_route++; }
        return a_t & om;

    /* ---- value-aware boolean: masking is exact per bit ---- */
    case OP_INT_AND: case OP_BOOL_AND:
        /* out_i flips iff a tainted input bit can move it: a's bit matters when
         * b's is 1 (or itself free), and symmetrically. */
        if (c) { c->ops += 6; c->n_route++; }
        return ((a_t & b_v) | (b_t & a_v) | (a_t & b_t)) & om;

    case OP_INT_OR: case OP_BOOL_OR:
        if (c) { c->ops += 7; c->n_route++; }
        return ((a_t & ~b_v) | (b_t & ~a_v) | (a_t & b_t)) & om;

    /* ---- shifts ---- */
    case OP_INT_LEFT: case OP_INT_RIGHT: case OP_INT_SRIGHT: {
        if (b_const || b_t == 0) {
            uint64_t k = b_v;
            if (c) { c->ops += 4; c->n_route++; }
            if (k >= 64) {
                if (oid != OP_INT_SRIGHT) return 0;
                k = 63;
            }
            if (oid == OP_INT_LEFT)  return (a_t << k) & om;
            if (oid == OP_INT_RIGHT) return (a_t >> k) & om;
            /* SRIGHT: the sign bit's taint replicates into the vacated top. */
            uint64_t im = mt_mask(isz);
            uint64_t r = (a_t & im) >> k;
            if ((a_t >> (isz * 8 - 1)) & 1) {
                int keep = isz * 8 - (int)k;
                r |= (keep <= 0) ? im : (~((1ULL << keep) - 1) & im);
            }
            return r & om;
        }
        /* A tainted shift amount can place any source bit anywhere. */
        if (c) { c->ops += 2; c->n_floor++; }
        return ((a_t | a_v) != 0) ? om : 0;
    }

    /* ---- carry-coupled: inlined differential, exact ---- */
    case OP_INT_ADD:
        return mt_add_taint(a_v, a_t, b_v, b_t, osz, c);
    case OP_INT_SUB:
        return mt_sub_taint(a_v, a_t, b_v, b_t, osz, c);
    case OP_INT_2COMP:
        return mt_sub_taint(0, 0, a_v, a_t, osz, c);

    case OP_INT_CARRY:
        return mt_carry_taint(a_v, a_t, b_v, b_t, isz, c);
    case OP_INT_SCARRY:
        return mt_scarry_taint(a_v, a_t, b_v, b_t, isz, c);
    case OP_INT_SBORROW:
        return mt_sborrow_taint(a_v, a_t, b_v, b_t, isz, c);

    /* ---- comparisons: exact from the monotone corners ---- */
    case OP_INT_EQUAL: case OP_INT_NOTEQUAL:
        return mt_eq_taint(a_v, a_t, b_v, b_t, isz, c);
    case OP_INT_LESS:
        return mt_cmp_taint(a_v, a_t, b_v, b_t, isz, 0, 0, c);
    case OP_INT_LESSEQUAL:
        return mt_cmp_taint(a_v, a_t, b_v, b_t, isz, 0, 1, c);
    case OP_INT_SLESS:
        return mt_cmp_taint(a_v, a_t, b_v, b_t, isz, 1, 0, c);
    case OP_INT_SLESSEQUAL:
        return mt_cmp_taint(a_v, a_t, b_v, b_t, isz, 1, 1, c);

    /* ---- multiply: exact only for a known power of two ---- */
    case OP_INT_MULT: {
        if (b_const && b_v != 0 && (b_v & (b_v - 1)) == 0) {
            int k = 63 - __builtin_clzll(b_v);
            if (c) { c->ops += 4; c->n_route++; }
            return (k >= 64) ? 0 : ((a_t << k) & om);
        }
        if (a_const && a_v != 0 && (a_v & (a_v - 1)) == 0) {
            int k = 63 - __builtin_clzll(a_v);
            if (c) { c->ops += 4; c->n_route++; }
            return (k >= 64) ? 0 : ((b_t << k) & om);
        }
        if (c) { c->ops += 2; c->n_floor++; }
        return (a_t | b_t) ? om : 0;
    }

    /* ---- avalanche floors ---- */
    case OP_INT_DIV: case OP_INT_SDIV: case OP_INT_REM: case OP_INT_SREM:
        if (c) { c->ops += 2; c->n_floor++; }
        return (a_t | b_t) ? om : 0;

    case OP_POPCOUNT: case OP_LZCOUNT: {
        /* The result cannot exceed the input width, so only the low
         * ceil(log2(bits+1)) bits of the output can move. */
        if (!(a_t)) { if (c) { c->ops += 1; c->n_route++; } return 0; }
        int bits = isz * 8;
        int nb = 0;
        while ((1 << nb) <= bits) nb++;
        if (c) { c->ops += 4; c->n_floor++; }
        return ((1ULL << nb) - 1) & om;
    }

    default:
        *decline = 1;
        return om;
    }
}

#endif /* TAINT_CORE_H */
