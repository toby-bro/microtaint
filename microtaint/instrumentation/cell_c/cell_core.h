#ifndef CELL_CORE_H
#define CELL_CORE_H

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/* Constants mirroring cell.pyx */
/* REGS_ARR_SIZE: large enough to cover both x86/AMD64 (offsets up to ~1100)
 * and ARM64 (offsets up to ~16640 for X-registers, vector regs higher).
 * Cython falls back to a Python dict for offsets above its 1104 limit; we
 * use a flat array sized to cover the common architectures' GP register files
 * AND their vector files (x86 XMM/YMM/ZMM 0x1200, PPC vs 0x4200, ARM q/z up to
 * 0x6040), so the width-native wide-vector path can address them by lane.
 * frame_clear is dirty-tracked, so a larger array costs no per-instruction time,
 * only memory. */
#define REGS_ARR_SIZE  24832
#define MAX_PCODE_OPS  96
#define MAX_UNIQ       64
#define MAX_DIRTY      128

#include "pcode_defs.h"
#include "taint_core.h"

/* Open-addressing hash for memory (single-byte values) */
#define MEM_CAP    256
#define MEM_MASK   (MEM_CAP - 1)
#define MEM_EMPTY  UINT64_MAX

typedef struct {
    uint64_t keys[MEM_CAP];
    uint64_t vals[MEM_CAP];
    int      dirty;   /* 1 if any byte was written since the last clear */
} MemMap;

static inline void mem_clear(MemMap *m) {
    memset(m->keys, 0xFF, sizeof(m->keys));
    m->dirty = 0;
}
static inline void mem_write_byte(MemMap *m, uint64_t addr, uint8_t b) {
    uint32_t s = (uint32_t)(addr * 2654435761UL) & MEM_MASK;
    while (m->keys[s] != MEM_EMPTY && m->keys[s] != addr)
        s = (s + 1) & MEM_MASK;
    m->keys[s] = addr;
    m->vals[s] = b;
    m->dirty = 1;
}
static inline uint8_t mem_read_byte(const MemMap *m, uint64_t addr) {
    uint32_t s = (uint32_t)(addr * 2654435761UL) & MEM_MASK;
    while (m->keys[s] != MEM_EMPTY && m->keys[s] != addr)
        s = (s + 1) & MEM_MASK;
    return (m->keys[s] == addr) ? (uint8_t)m->vals[s] : 0;
}
static inline void mem_write(MemMap *m, uint64_t addr, uint64_t val, int size, int be) {
    uint64_t mask = (size >= 8) ? UINT64_MAX : (((uint64_t)1 << (size*8)) - 1);
    val &= mask;
    for (int i = 0; i < size; i++) {
        /* BE stores the most-significant byte at the lowest address. */
        int shift = be ? (size - 1 - i) * 8 : i * 8;
        mem_write_byte(m, addr+i, (uint8_t)(val >> shift));
    }
}
static inline uint64_t mem_read(const MemMap *m, uint64_t addr, int size, int be) {
    uint64_t r = 0;
    for (int i = 0; i < size; i++) {
        int shift = be ? (size - 1 - i) * 8 : i * 8;
        r |= ((uint64_t)mem_read_byte(m, addr+i)) << shift;
    }
    return r;
}

/* Frame */
typedef struct {
    uint64_t regs_arr[REGS_ARR_SIZE];
    uint8_t  regs_sz [REGS_ARR_SIZE];
    uint8_t  regs_set[REGS_ARR_SIZE];
    int      dirty[MAX_DIRTY];
    int      dirty_count;
    uint64_t uniq_arr[MAX_UNIQ];
    uint8_t  uniq_set[MAX_UNIQ];
    MemMap   mem;
    uint64_t arch_pc_off;
    int      arch_pc_sz;
    /* Target byte order.  The register file is indexed by BYTE OFFSET, so on a
     * big-endian target byte d of an N-byte register sits at bit (N-d-size)*8,
     * not d*8.  Getting this wrong does not fail loudly -- it reads the wrong
     * bytes and the differential silently collapses toward 0.  Mirrors
     * _PCodeFrame._is_big_endian in cell.pyx. */
    int      is_big_endian;
} Frame;

static inline void frame_clear(Frame *f) {
    for (int i = 0; i < f->dirty_count; i++) f->regs_set[f->dirty[i]] = 0;
    f->dirty_count = 0;
    for (int i = 0; i < MAX_UNIQ; i++) f->uniq_set[i] = 0;
    /* The mem hash is 2KB; only pay the memset when a store actually dirtied it
     * (most instructions touch no memory). Bit-exact: an untouched map is still
     * all-empty from the previous clear. */
    if (f->mem.dirty) mem_clear(&f->mem);
}

static inline uint64_t mask64(uint64_t val, int sz) {
    static const uint64_t MT[9] = {
        0, 0xFF, 0xFFFF, 0xFFFFFF, 0xFFFFFFFF,
        0xFFFFFFFFFFULL, 0xFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL
    };
    if (sz <= 0 || sz > 8) return val;
    return val & MT[sz];
}

static inline int64_t signed64(uint64_t val, int sz) {
    static const uint64_t SE[9] = {
        0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFF00ULL,
        0xFFFFFFFFFFFF0000ULL, 0xFFFFFFFFFF000000ULL,
        0xFFFFFFFF00000000ULL, 0xFFFFFF0000000000ULL,
        0xFFFF000000000000ULL, 0xFF00000000000000ULL,
        0x0000000000000000ULL
    };
    val = mask64(val, sz);
    uint64_t msb = (uint64_t)1 << (sz*8 - 1);
    if (val & msb) return (int64_t)(val | SE[sz]);
    return (int64_t)val;
}

static inline void frame_write_reg(Frame *f, long off, int sz, uint64_t val) {
    val = mask64(val, sz);
    if (off >= 0 && off < REGS_ARR_SIZE) {
        if (!f->regs_set[off] && f->dirty_count < MAX_DIRTY)
            f->dirty[f->dirty_count++] = (int)off;
        /* Same-offset narrower write: e.g. ``mov al, bl`` writes byte 0
         * (size 1) of RAX while the slot already holds the full 8-byte
         * RAX value.  We MUST preserve the upper bytes — clobbering the
         * size-8 entry with a size-1 entry would silently lose them and
         * any later read of RAX would return only the new low byte
         * (the bug observed in test_misc_partial_writes).  Merge by
         * keeping the wider stored size and overlaying the narrow new
         * value onto the low ``sz`` bytes. */
        if (f->regs_set[off] && (int)f->regs_sz[off] > sz) {
            uint64_t lo_mask = (sz >= 8)
                ? 0xFFFFFFFFFFFFFFFFULL
                : (((uint64_t)1 << (sz * 8)) - 1);
            /* A narrower write sharing the base offset targets the LOW bytes
             * under LE but the HIGH bytes under BE, because the base offset is
             * the most-significant byte there. */
            if (f->is_big_endian) {
                int be_wshift = ((int)f->regs_sz[off] - sz) * 8;
                if (be_wshift >= 0 && be_wshift < 64) {
                    f->regs_arr[off] = (f->regs_arr[off] & ~(lo_mask << be_wshift))
                                     | ((val & lo_mask) << be_wshift);
                }
            } else {
                f->regs_arr[off] = (f->regs_arr[off] & ~lo_mask) | (val & lo_mask);
            }
            /* regs_sz stays at the wider size — the slot still represents
             * the full architectural register. */
        } else {
            f->regs_arr[off] = val;
            f->regs_sz [off] = (uint8_t)sz;
            /* Wider/equal write — invalidate any per-byte sub-writes that
             * were overlaid in earlier ops and now sit inside our range.
             * Without this, a later wider read would re-merge those stale
             * sub-byte values onto the freshly-written wider value and
             * silently revert the overlay (observed: ``paddb`` fans out
             * 16 byte writes, then ``psllq`` writes a wider lane that
             * should logically subsume them, but the read after the
             * shift saw the stale 0-byte sub-writes).  Limit to the same
             * 8-byte window as the read-side guard. */
            long invalidate_end = off + sz;
            if (invalidate_end > off + 8) invalidate_end = off + 8;
            for (long k = off + 1; k < invalidate_end && k < REGS_ARR_SIZE; k++) {
                if (f->regs_set[k] && (int)f->regs_sz[k] < sz) {
                    f->regs_set[k] = 0;
                }
            }
        }
        f->regs_set[off] = 1;
    }
}

static inline uint64_t frame_read_reg(const Frame *f, long off, int sz) {
    if (off >= 0 && off < REGS_ARR_SIZE) {
        /* Step 1 — establish the base value of this register slot.
         * If this exact slot was written, use it.  Otherwise look
         * backwards for a parent register that contains this offset
         * (e.g. reading AH after writing only RAX).  Otherwise base = 0. */
        uint64_t base = 0;
        if (f->regs_set[off]) {
            base = f->regs_arr[off];
        } else {
            for (long k = off-1; k >= 0 && off-k <= 8; k--) {
                if (f->regs_set[k] && k + (long)f->regs_sz[k] > off) {
                    long byte_off = off - k;
                    if (f->is_big_endian) {
                        /* BE: byte `byte_off` from the MSB of a parent of
                         * regs_sz[k] bytes; the sub-value's low bit sits at
                         * (parent_sz - byte_off - sz)*8. */
                        long be_shift = ((long)f->regs_sz[k] - byte_off - sz) * 8;
                        base = (be_shift >= 0 && be_shift < 64)
                             ? (f->regs_arr[k] >> be_shift) : 0;
                    } else {
                        base = f->regs_arr[k] >> (byte_off * 8);
                    }
                    break;
                }
            }
        }

        /* Step 2 — overlay any sub-register writes that fall INSIDE our
         * read range.  Critical for x86 partial-register writes like
         * `mov ah, bh`: after the COPY writes byte 1 (AH) we read RAX
         * (offset 0, size 8) and must merge the written AH byte over
         * the original RAX value.  Without this overlay the read
         * returns the pre-write parent value alone and the partial
         * write is silently lost.
         *
         * NOTE: we only overlay sub-writes whose start offset is within
         * 8 bytes of `off` — beyond that, the overlay would shift past
         * the 64-bit width of `base` and (on x86-64) the SHL by ≥64
         * masks the count modulo 64, producing a wrong lane_mask that
         * zeroes the low bits.  This guard limits the overlay to the
         * representable low-8-byte window, which is the only case our
         * GP-register partial-write fix actually needs.  Wider XMM/YMM
         * reads (size > 8) are handled in their own slots (XMM<n>_LO at
         * offset 0x1200, XMM<n>_HI at offset 0x1208) by the engine. */
        long end_off = off + sz;
        long k = off + 1;
        while (k < end_off && k < REGS_ARR_SIZE && k - off < 8) {
            if (f->regs_set[k]) {
                int k_sz = (int)f->regs_sz[k];
                if (k_sz <= 0) { k++; continue; }
                long byte_off = k - off;
                uint64_t sub_mask = (k_sz >= 8)
                    ? 0xFFFFFFFFFFFFFFFFULL
                    : (((uint64_t)1 << (k_sz * 8)) - 1);
                uint64_t sub_val = f->regs_arr[k] & sub_mask;
                /* BE: the sub-slot at `byte_off` from the MSB of this sz-byte
                 * read occupies bits (sz - byte_off - k_sz)*8 upward. */
                long be_shift = f->is_big_endian
                    ? ((long)sz - byte_off - k_sz) * 8
                    : (byte_off * 8);
                if (be_shift >= 0 && be_shift < 64) {
                    uint64_t lane_mask = sub_mask << be_shift;
                    base = (base & ~lane_mask) | (sub_val << be_shift);
                }
                k += k_sz;
            } else {
                k++;
            }
        }
        return mask64(base, sz);
    }
    return 0;
}

static inline uint64_t frame_read_d(const Frame *f, int sp, uint64_t off, int sz) {
    if (sp == SP_CONST)    return mask64((uint64_t)off, sz);
    if (sp == SP_REGISTER) return frame_read_reg(f, (long)off, sz);
    if (sp == SP_UNIQUE)   return (off < MAX_UNIQ && f->uniq_set[off]) ? mask64(f->uniq_arr[off], sz) : 0;
    if (sp == SP_RAM)      return mem_read(&f->mem, (uint64_t)off, sz, f->is_big_endian);
    return 0;
}

static inline void frame_write_d(Frame *f, int sp, uint64_t off, int sz, uint64_t val) {
    val = mask64(val, sz);
    if (sp == SP_REGISTER) { frame_write_reg(f, (long)off, sz, val); return; }
    if (sp == SP_UNIQUE)   { if (off < MAX_UNIQ) { f->uniq_arr[off]=val; f->uniq_set[off]=1; } return; }
    if (sp == SP_RAM)      { mem_write(&f->mem, (uint64_t)off, val, sz, f->is_big_endian); }
}

/* Maps an in-sequence x86 instruction address to its pcode op index.
 * Populated from DecodedOps.imark_to_pc on bundle creation.  Used by
 * BRANCH/CBRANCH to translate a ram-space target back to a pcode pc. */
typedef struct {
    uint64_t addr;
    int      pc;
} ImarkEntry;

typedef struct {
    PCOp     buf[MAX_PCODE_OPS];
    int      n_ops;
    int      has_fallback;
    /* Opaque data op (CALLOTHER-with-output / FLOAT): avalanche its taint at the
     * differential level instead of executing it via Unicorn.  Mirrors
     * cell.pyx DecodedOps.avalanche_ok. */
    int      avalanche_ok;
    /* 1 if this bundle is a SINGLE register-only instruction with no memory,
     * control-flow, or opaque ops -- eligible for native re-execution (the C
     * kernel's 4th path).  Computed once at bundle creation. */
    int      reexec_ok;
    uint64_t next_instr_addr;
    /* IMARK address → pcode pc.  Linear scan; n_imarks ≤ MAX_PCODE_OPS. */
    int          n_imarks;
    ImarkEntry   imarks[MAX_PCODE_OPS];
} DecodedBundle;

#define EXEC_OK          0
#define EXEC_FALLBACK    1
/* The taint pass met a p-code shape it does not model (wide vector, memory,
 * opaque op, tainted branch condition).  The VALUE state is undefined at that
 * point, so the caller must re-run without the taint frame or fall back to the
 * monolithic differential -- it must never read a partial taint answer. */
#define EXEC_TAINT_DECLINE 2

/* Linear-scan lookup of an IMARK address in the bundle's imark table.
 * Returns the pcode pc for that address, or -1 if not present.  Typical
 * sequence has ≤ 8 IMARKs so the linear scan is fine. */
/* Read a 128-bit value from a unique slot pair (low at off, high at off+8), the
 * same layout the splittable-op block writes.  For <=8-byte or non-unique sources
 * this is just the zero-extended 64-bit value. */
static inline unsigned __int128 frame_read_u128(const Frame *f, int sp,
                                                uint64_t off, int sz) {
    if (sz <= 8)
        return (unsigned __int128) frame_read_d(f, sp, off, sz);
    /* Read the low and high 8-byte lanes via frame_read_d so this works for a
     * unique slot pair (off / off+8) AND a register vector (sleigh off / off+8),
     * not just unique -- a whole-register 128-bit shift reads from a register. */
    unsigned __int128 lo = frame_read_d(f, sp, off, 8);
    unsigned __int128 hi = frame_read_d(f, sp, off + 8, sz - 8);
    return lo | (hi << 64);
}
static inline void frame_write_u128(Frame *f, int sp, uint64_t off, int sz,
                                    unsigned __int128 val) {
    if (sz > 16) sz = 16;
    if (sz < 16 && sz > 0) {
        unsigned __int128 m = (((unsigned __int128)1) << (sz*8)) - 1;
        val &= m;
    }
    frame_write_d(f, sp, off, 8, (uint64_t)val);
    if (sz > 8) frame_write_d(f, sp, off + 8, sz - 8, (uint64_t)(val >> 64));
}

static inline int bundle_lookup_imark(const DecodedBundle *d, uint64_t addr) {
    for (int i = 0; i < d->n_imarks; i++) {
        if (d->imarks[i].addr == addr) return d->imarks[i].pc;
    }
    return -1;
}

/* Sign-extend a uint64 ``v`` interpreted as a signed integer of ``sz``
 * bytes (1, 2, 4, or 8) to int64. */
static inline int64_t sign_extend_n(uint64_t v, int sz) {
    if (sz <= 0 || sz >= 8) return (int64_t)v;
    int shift = (8 - sz) * 8;
    return ((int64_t)(v << shift)) >> shift;
}

#define EXEC_LOOP_BUDGET 256

/* Execute the decoded p-code.
 *
 * With `tf == NULL` this is the value-only interpreter it has always been.
 * With `tf` supplied it is the ONE-PASS taint composer: `tf` is a second Frame
 * holding a taint word beside every varnode, and each op's output taint is
 * derived from its inputs' (value, taint) by the per-opcode rules in
 * taint_core.h.  Because taint aliases registers exactly the way values do
 * (AL inside RAX, a flag byte inside a flag block), the same Frame machinery --
 * partial writes, sub-register overlay, endianness -- serves both, so there is
 * no separate taint state model to keep in sync.
 *
 * Every output of the instruction, result registers and flags alike, is simply
 * read off `tf` when the pass returns.  No per-output program, no second pass,
 * no waiting on an intermediate.
 */
static inline int execute_decoded_t(Frame *f, Frame *tf, const DecodedBundle *d,
                                    MtTaintCost *tc) {
    int skip = 0;
    uint64_t next_addr = d->next_instr_addr;
    int loop_iters = 0;
    int pc = 0;

    while (pc < d->n_ops) {
        const PCOp *op = &d->buf[pc];
        int oid = op->oid;
        if (skip && oid != OP_IMARK) { pc++; continue; }

        /* ── taint pass: capture inputs, then gate ────────────────────
         * Inputs are read BEFORE the op executes: SLEIGH freely writes a
         * varnode it also reads (add writes RAX after INT_CARRY read it), so
         * reading them afterwards would compose the taint of the wrong value.
         *
         * The gate is deliberately narrow.  Anything outside the modelled set
         * declines the whole instruction rather than contributing a partial
         * answer, because an unmodelled op silently producing 0 taint is an
         * under-taint, and under-taint is never acceptable. */
        uint64_t t_iv[3] = {0, 0, 0}, t_it[3] = {0, 0, 0};
        if (tf) {
            if (op->o_sz > 8 || op->i0_sz > 8 || op->i1_sz > 8 || op->i2_sz > 8)
                return EXEC_TAINT_DECLINE;
            switch (oid) {
            case OP_LOAD: case OP_STORE: case OP_CALLOTHER:
            case OP_FLOAT_ANY: case OP_TRUNC_FLOAT: case OP_UNKNOWN:
            case OP_PTRADD: case OP_PTRSUB: case OP_MULTIEQUAL:
            case OP_INDIRECT: case OP_CPOOLREF: case OP_NEW: case OP_SEGMENT:
            case OP_INSERT: case OP_EXTRACT: case OP_UNIMPLEMENTED:
            case OP_BRANCHIND: case OP_CALLIND: case OP_CALL: case OP_RETURN:
                return EXEC_TAINT_DECLINE;
            default: break;
            }
            if (op->n_ins > 0 && op->i0_sp != SP_CONST) {
                t_iv[0] = frame_read_d(f,  op->i0_sp, op->i0_off, op->i0_sz);
                t_it[0] = frame_read_d(tf, op->i0_sp, op->i0_off, op->i0_sz);
            } else if (op->n_ins > 0) {
                t_iv[0] = mask64((uint64_t)op->i0_off, op->i0_sz);
            }
            if (op->n_ins > 1 && op->i1_sp != SP_CONST) {
                t_iv[1] = frame_read_d(f,  op->i1_sp, op->i1_off, op->i1_sz);
                t_it[1] = frame_read_d(tf, op->i1_sp, op->i1_off, op->i1_sz);
            } else if (op->n_ins > 1) {
                t_iv[1] = mask64((uint64_t)op->i1_off, op->i1_sz);
            }
            if (op->n_ins > 2 && op->i2_sp != SP_CONST) {
                t_iv[2] = frame_read_d(f,  op->i2_sp, op->i2_off, op->i2_sz);
                t_it[2] = frame_read_d(tf, op->i2_sp, op->i2_off, op->i2_sz);
            } else if (op->n_ins > 2) {
                t_iv[2] = mask64((uint64_t)op->i2_off, op->i2_sz);
            }
            if (tc) { tc->pcode_ops++; tc->reads += 2 * (op->n_ins > 3 ? 3 : op->n_ins); }
            /* A tainted branch condition is implicit flow: the not-taken side's
             * writes are still observable through the taken side's absence.  A
             * straight-line pass would follow one side and drop the other, so
             * decline until the fork-and-merge path lands. */
            if (oid == OP_CBRANCH && op->n_ins > 1 && t_it[1] != 0)
                return EXEC_TAINT_DECLINE;
        }

        uint64_t a, b, result;
        int64_t  sa, sb;
        int      sz;

        /* Wide-op decomposition for 128-bit SIMD bitwise ops.
         *
         * Background: register slots in ``Frame`` are uint64_t (8 bytes).
         * SLEIGH lifts PXOR/PAND/POR/MOVDQA over xmm registers as a
         * SINGLE 16-byte ``INT_XOR`` / ``INT_AND`` / ``INT_OR`` /
         * ``COPY`` op at offset 0x1200 (XMM0_LO base).  The default
         * dispatch reads only the low 8 bytes via ``frame_read_d`` (which
         * truncates to uint64_t), so the high 8 bytes (XMM_HI at offset
         * 0x1208) are silently dropped — XMM_HI keeps its original value.
         *
         * Fix: when the op is bit-independent (XOR/AND/OR/COPY/ZEXT) and
         * the output is wider than 8 bytes, split into two 8-byte sub-ops
         * — one at the original offset, one at offset+8.  This is exact
         * for bit-independent ops because each output byte depends only
         * on the corresponding input byte(s).
         *
         * Other 16-byte ops (e.g. INT_ADD on a hypothetical 128-bit add)
         * are NOT split — carry propagation crosses the 8-byte boundary,
         * so naive splitting would be unsound.  In practice SLEIGH does
         * not emit such ops for SSE/AVX (PADDQ/B/W/D all decompose at
         * the lifter level into per-lane sub-ops ≤ 8 bytes).  PIECE and
         * SUBPIECE are also handled here when their output is wide.
         *
         * Sound tightening: only the lane-independent opcodes listed
         * below are eligible.  Anything else with output > 8 bytes
         * triggers the unicorn fallback by NOT being split here — the
         * existing dispatch will read the truncated value and the
         * eventual mismatch with the static rule's symbolic expression
         * (which uses Unicorn for full-instruction simulation) will
         * simply be a precision loss in the C path's concrete value,
         * not an unsound taint result.
         */
        /* Wide vector STORE: STORE(space_const, addr, value) with a >8-byte
         * value (input 2).  STORE has no output, so it is handled before the
         * output-gated wide block.  Write the full width to memory lane-by-lane
         * (mem_write is endian-aware). */
        if (oid == OP_STORE && op->i2_sz > 8) {
            uint64_t addr = frame_read_d(f, op->i1_sp, op->i1_off, op->i1_sz);
            for (int lane = 0; lane < op->i2_sz; lane += 8) {
                int lsz = (op->i2_sz - lane < 8) ? (op->i2_sz - lane) : 8;
                uint64_t v = frame_read_d(f, op->i2_sp, op->i2_off + lane, lsz);
                mem_write(&f->mem, addr + lane, v, lsz, f->is_big_endian);
            }
            pc++;
            continue;
        }
        if (op->o_sp != NO_OUT_SPACE && op->o_sz > 8) {
            int splittable = (oid == OP_INT_XOR || oid == OP_INT_AND
                              || oid == OP_INT_OR || oid == OP_COPY
                              || oid == OP_INT_ZEXT
                              || oid == OP_INT_SEXT
                              || oid == OP_INT_NEGATE);
            if (splittable) {
                /* Split into 8-byte lanes and apply the op lane-by-lane.  Byte-
                 * parallel bitwise / movement is width-agnostic; ZEXT/SEXT zero-
                 * or sign-fill the lanes above the source.  For total==16 this is
                 * exactly the old low+high pair (register lanes at o_off/o_off+8,
                 * the 128-bit unique embryo's uniq_arr[off]/uniq_arr[off+8]); for
                 * 32/64-byte YMM/ZMM it continues to the higher register lanes.
                 * A register o_off is a sleigh byte offset so o_off+lane is the
                 * right lane. */
                int total_sz = op->o_sz;
                uint64_t sext_fill = 0;
                if (oid == OP_INT_SEXT) {
                    int msb_lane = ((op->i0_sz - 1) / 8) * 8;
                    int msb_within = op->i0_sz - msb_lane;
                    uint64_t src_hi = frame_read_d(f, op->i0_sp,
                                                   op->i0_off + msb_lane, msb_within);
                    sext_fill = ((src_hi >> (msb_within * 8 - 1)) & 1)
                              ? UINT64_MAX : 0;
                }
                for (int lane = 0; lane < total_sz; lane += 8) {
                    int lsz = (total_sz - lane < 8) ? (total_sz - lane) : 8;
                    uint64_t v0, v1 = 0, r;
                    if ((oid == OP_INT_ZEXT || oid == OP_INT_SEXT)
                            && lane >= op->i0_sz) {
                        v0 = (oid == OP_INT_SEXT) ? sext_fill : 0;
                    } else {
                        v0 = frame_read_d(f, op->i0_sp, op->i0_off + lane, lsz);
                    }
                    if (op->n_ins >= 2 && lane < op->i1_sz) {
                        int l1 = (op->i1_sz - lane < lsz) ? (op->i1_sz - lane) : lsz;
                        v1 = frame_read_d(f, op->i1_sp, op->i1_off + lane, l1);
                    }
                    switch (oid) {
                        case OP_INT_XOR: r = v0 ^ v1; break;
                        case OP_INT_AND: r = v0 & v1; break;
                        case OP_INT_OR:  r = v0 | v1; break;
                        case OP_INT_NEGATE: r = ~v0; break;
                        default:         r = v0; break;  /* COPY/ZEXT/SEXT */
                    }
                    frame_write_d(f, op->o_sp, op->o_off + lane, lsz, r);
                }
                pc++;
                continue;  /* skip the regular dispatch for this op */
            }
            /* 128-bit multiply (widening MUL/IMUL): not lane-independent, so the
             * split block above skips it.  Read both operands as full 128-bit and
             * multiply.  Without this the low halves are multiplied truncated and
             * the high half reads 0 -- which is exactly what made the imul CF flag
             * (`sext(RAX) != full_product`) under-taint. */
            if (oid == OP_INT_MULT) {
                unsigned __int128 p = frame_read_u128(f, op->i0_sp, op->i0_off, op->i0_sz)
                                    * frame_read_u128(f, op->i1_sp, op->i1_off, op->i1_sz);
                frame_write_u128(f, op->o_sp, op->o_off, op->o_sz, p);
                pc++;
                continue;
            }
            /* 128-bit left / logical-right shift (x86 pslldq / psrldq lift to a
             * single whole-register INT_LEFT / INT_RIGHT).  Position-sensitive,
             * so run on the full 128-bit value read via the (now register-aware)
             * frame_read_u128. */
            if (oid == OP_INT_LEFT) {
                unsigned __int128 v = frame_read_u128(f, op->i0_sp, op->i0_off, op->i0_sz);
                uint64_t sh = frame_read_d(f, op->i1_sp, op->i1_off, op->i1_sz);
                unsigned __int128 r = (sh >= (uint64_t)(op->o_sz*8)) ? 0 : (v << sh);
                frame_write_u128(f, op->o_sp, op->o_off, op->o_sz, r);
                pc++;
                continue;
            }
            if (oid == OP_INT_RIGHT) {
                unsigned __int128 v = frame_read_u128(f, op->i0_sp, op->i0_off, op->i0_sz);
                uint64_t sh = frame_read_d(f, op->i1_sp, op->i1_off, op->i1_sz);
                unsigned __int128 r = (sh >= (uint64_t)(op->o_sz*8)) ? 0 : (v >> sh);
                frame_write_u128(f, op->o_sp, op->o_off, op->o_sz, r);
                pc++;
                continue;
            }
            /* Wide vector LOAD: reg/unique = LOAD(space_const, addr).  Read o_sz
             * bytes from memory lane-by-lane (mem_read is endian-aware) and store
             * the full width, so movdqu / ldr q / lvx keeps every byte's taint. */
            if (oid == OP_LOAD) {
                uint64_t addr = frame_read_d(f, op->i1_sp, op->i1_off, op->i1_sz);
                for (int lane = 0; lane < op->o_sz; lane += 8) {
                    int lsz = (op->o_sz - lane < 8) ? (op->o_sz - lane) : 8;
                    uint64_t v = mem_read(&f->mem, addr + lane, lsz, f->is_big_endian);
                    frame_write_d(f, op->o_sp, op->o_off + lane, lsz, v);
                }
                pc++;
                continue;
            }
        }

        /* 128-bit / 64-bit divide and remainder (DIV/IDIV read a 128-bit
         * dividend, produce a <=8-byte quotient/remainder). */
        if ((oid == OP_INT_DIV || oid == OP_INT_REM || oid == OP_INT_SDIV
             || oid == OP_INT_SREM) && op->i0_sz > 8) {
            unsigned __int128 num = frame_read_u128(f, op->i0_sp, op->i0_off, op->i0_sz);
            uint64_t den = frame_read_d(f, op->i1_sp, op->i1_off, op->i1_sz);
            uint64_t res;
            if (den == 0) {
                res = 0;
            } else if (oid == OP_INT_DIV) {
                res = (uint64_t)(num / den);
            } else if (oid == OP_INT_REM) {
                res = (uint64_t)(num % den);
            } else {
                int bits = op->i0_sz * 8;
                __int128 snum = (num & (((unsigned __int128)1) << (bits-1)))
                    ? (__int128)(num - (((unsigned __int128)1) << bits)) : (__int128)num;
                __int128 sden = (int64_t)den;
                res = (oid == OP_INT_SDIV)
                    ? (uint64_t)(snum / sden) : (uint64_t)(snum % sden);
            }
            if (op->o_sp != NO_OUT_SPACE)
                frame_write_d(f, op->o_sp, op->o_off, op->o_sz, res);
            pc++;
            continue;
        }

        /* Comparison of 128-bit operands.  The output is 1 byte, so the o_sz>8
         * block above never sees it; reading only the low half via the 64-bit
         * dispatch compares the wrong thing (and for the imul CF idiom the low
         * halves are always equal, yielding a silent 0). */
        if ((oid == OP_INT_EQUAL || oid == OP_INT_NOTEQUAL || oid == OP_INT_LESS
             || oid == OP_INT_SLESS || oid == OP_INT_LESSEQUAL
             || oid == OP_INT_SLESSEQUAL)
            && (op->i0_sz > 8 || op->i1_sz > 8)) {
            unsigned __int128 w0 = frame_read_u128(f, op->i0_sp, op->i0_off, op->i0_sz);
            unsigned __int128 w1 = frame_read_u128(f, op->i1_sp, op->i1_off, op->i1_sz);
            uint64_t res;
            if (oid == OP_INT_EQUAL)         res = (w0 == w1);
            else if (oid == OP_INT_NOTEQUAL) res = (w0 != w1);
            else if (oid == OP_INT_LESS)     res = (w0 <  w1);
            else if (oid == OP_INT_LESSEQUAL) res = (w0 <= w1);
            else {
                int bits = (op->i0_sz > op->i1_sz ? op->i0_sz : op->i1_sz) * 8;
                unsigned __int128 sbit = ((unsigned __int128)1) << (bits - 1);
                __int128 s0 = (w0 & sbit) ? (__int128)(w0 - (((unsigned __int128)1) << bits)) : (__int128)w0;
                __int128 s1 = (w1 & sbit) ? (__int128)(w1 - (((unsigned __int128)1) << bits)) : (__int128)w1;
                res = (oid == OP_INT_SLESS) ? (s0 < s1) : (s0 <= s1);
            }
            if (op->o_sp != NO_OUT_SPACE)
                frame_write_d(f, op->o_sp, op->o_off, op->o_sz, res);
            pc++;
            continue;
        }

        switch (oid) {
        case OP_IMARK: case OP_RETURN: case OP_CALL: break;

        case OP_INT_XOR:
            if (op->o_sp != NO_OUT_SPACE) frame_write_d(f, op->o_sp, op->o_off, op->o_sz,
                frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz) ^
                frame_read_d(f,op->i1_sp,op->i1_off,op->i1_sz));
            break;
        case OP_INT_AND:
            if (op->o_sp != NO_OUT_SPACE) frame_write_d(f, op->o_sp, op->o_off, op->o_sz,
                frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz) &
                frame_read_d(f,op->i1_sp,op->i1_off,op->i1_sz));
            break;
        case OP_BOOL_AND:
            if (op->o_sp != NO_OUT_SPACE) frame_write_d(f, op->o_sp, op->o_off, op->o_sz,
                (frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz) &&
                 frame_read_d(f,op->i1_sp,op->i1_off,op->i1_sz)) ? 1 : 0);
            break;
        case OP_INT_OR:
            if (op->o_sp != NO_OUT_SPACE) frame_write_d(f, op->o_sp, op->o_off, op->o_sz,
                frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz) |
                frame_read_d(f,op->i1_sp,op->i1_off,op->i1_sz));
            break;
        case OP_BOOL_OR:
            if (op->o_sp != NO_OUT_SPACE) frame_write_d(f, op->o_sp, op->o_off, op->o_sz,
                (frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz) ||
                 frame_read_d(f,op->i1_sp,op->i1_off,op->i1_sz)) ? 1 : 0);
            break;
        case OP_INT_ADD: case OP_PTRADD:
            if (op->o_sp != NO_OUT_SPACE) frame_write_d(f, op->o_sp, op->o_off, op->o_sz,
                frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz) +
                frame_read_d(f,op->i1_sp,op->i1_off,op->i1_sz));
            break;
        case OP_INT_SUB: case OP_PTRSUB:
            if (op->o_sp != NO_OUT_SPACE) frame_write_d(f, op->o_sp, op->o_off, op->o_sz,
                frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz) -
                frame_read_d(f,op->i1_sp,op->i1_off,op->i1_sz));
            break;
        case OP_INT_MULT:
            if (op->o_sp != NO_OUT_SPACE) frame_write_d(f, op->o_sp, op->o_off, op->o_sz,
                frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz) *
                frame_read_d(f,op->i1_sp,op->i1_off,op->i1_sz));
            break;
        case OP_INT_DIV:
            if (op->o_sp != NO_OUT_SPACE) {
                b = frame_read_d(f,op->i1_sp,op->i1_off,op->i1_sz);
                frame_write_d(f, op->o_sp, op->o_off, op->o_sz,
                    b ? frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz)/b : 0);
            } break;
        case OP_INT_SDIV:
            if (op->o_sp != NO_OUT_SPACE) {
                sz = op->i0_sz;
                sa = signed64(frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz), sz);
                sb = signed64(frame_read_d(f,op->i1_sp,op->i1_off,op->i1_sz), sz);
                if (!sb) { frame_write_d(f,op->o_sp,op->o_off,op->o_sz,0); break; }
                int64_t q = sa/sb;
                if ((sa^sb)<0 && q*sb!=sa) q++;
                frame_write_d(f, op->o_sp, op->o_off, op->o_sz, (uint64_t)q);
            } break;
        case OP_INT_REM:
            if (op->o_sp != NO_OUT_SPACE) {
                b = frame_read_d(f,op->i1_sp,op->i1_off,op->i1_sz);
                frame_write_d(f, op->o_sp, op->o_off, op->o_sz,
                    b ? frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz)%b : 0);
            } break;
        case OP_INT_SREM:
            if (op->o_sp != NO_OUT_SPACE) {
                sz = op->i0_sz;
                sa = signed64(frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz), sz);
                sb = signed64(frame_read_d(f,op->i1_sp,op->i1_off,op->i1_sz), sz);
                frame_write_d(f, op->o_sp, op->o_off, op->o_sz,
                    sb ? (uint64_t)(sa - sb*(sa/sb)) : 0);
            } break;
        case OP_INT_2COMP:
            if (op->o_sp != NO_OUT_SPACE) frame_write_d(f, op->o_sp, op->o_off, op->o_sz,
                (uint64_t)(-(int64_t)frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz)));
            break;
        case OP_INT_NEGATE:
            if (op->o_sp != NO_OUT_SPACE) frame_write_d(f, op->o_sp, op->o_off, op->o_sz,
                ~frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz));
            break;
        case OP_BOOL_NEGATE:
            if (op->o_sp != NO_OUT_SPACE) frame_write_d(f, op->o_sp, op->o_off, op->o_sz,
                frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz) ? 0 : 1);
            break;
        case OP_INT_LEFT:
            /* P-code spec: if input1 is >= number of bits in output, result is 0.
             * (Ghidra pcodedescription.html.)  See cell.pyx for the SHLD/BEXTR
             * cases (ids 3252, 5337) that motivated this fix.  C undefined-
             * behaviour for shifts >= type-width also forces the explicit guard. */
            if (op->o_sp != NO_OUT_SPACE) {
                b = frame_read_d(f,op->i1_sp,op->i1_off,op->i1_sz);
                if (b >= (uint64_t)(op->o_sz * 8)) {
                    frame_write_d(f, op->o_sp, op->o_off, op->o_sz, 0);
                } else {
                    frame_write_d(f, op->o_sp, op->o_off, op->o_sz,
                        frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz) << b);
                }
            } break;
        case OP_INT_RIGHT:
            if (op->o_sp != NO_OUT_SPACE) {
                b = frame_read_d(f,op->i1_sp,op->i1_off,op->i1_sz);
                if (b >= (uint64_t)(op->o_sz * 8)) {
                    frame_write_d(f, op->o_sp, op->o_off, op->o_sz, 0);
                } else {
                    frame_write_d(f, op->o_sp, op->o_off, op->o_sz,
                        frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz) >> b);
                }
            } break;
        case OP_INT_SRIGHT:
            /* P-code spec: if input1 is >= bits-in-output, result is 0 if input0
             * is non-negative, all 1-bits (within output width) if negative. */
            if (op->o_sp != NO_OUT_SPACE) {
                uint64_t result_sright;
                sz = op->i0_sz;
                sa = signed64(frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz), sz);
                b  = frame_read_d(f,op->i1_sp,op->i1_off,op->i1_sz);
                if (b >= (uint64_t)(op->o_sz * 8)) {
                    if (sa < 0) {
                        if (op->o_sz >= 8) result_sright = (uint64_t)0xFFFFFFFFFFFFFFFFULL;
                        else               result_sright = ((uint64_t)1 << (op->o_sz * 8)) - 1;
                    } else {
                        result_sright = 0;
                    }
                } else {
                    result_sright = (uint64_t)(sa >> b);
                }
                frame_write_d(f, op->o_sp, op->o_off, op->o_sz, result_sright);
            } break;
        case OP_INT_EQUAL:
            if (op->o_sp != NO_OUT_SPACE) frame_write_d(f, op->o_sp, op->o_off, op->o_sz,
                frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz) ==
                frame_read_d(f,op->i1_sp,op->i1_off,op->i1_sz) ? 1 : 0);
            break;
        case OP_INT_NOTEQUAL:
            if (op->o_sp != NO_OUT_SPACE) frame_write_d(f, op->o_sp, op->o_off, op->o_sz,
                frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz) !=
                frame_read_d(f,op->i1_sp,op->i1_off,op->i1_sz) ? 1 : 0);
            break;
        case OP_BOOL_XOR: {
            uint64_t ba = frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz) ? 1 : 0;
            uint64_t bb = frame_read_d(f,op->i1_sp,op->i1_off,op->i1_sz) ? 1 : 0;
            if (op->o_sp != NO_OUT_SPACE) frame_write_d(f, op->o_sp, op->o_off, op->o_sz, ba ^ bb);
            } break;
        case OP_INT_LESS:
            if (op->o_sp != NO_OUT_SPACE) frame_write_d(f, op->o_sp, op->o_off, op->o_sz,
                frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz) <
                frame_read_d(f,op->i1_sp,op->i1_off,op->i1_sz) ? 1 : 0);
            break;
        case OP_INT_LESSEQUAL:
            if (op->o_sp != NO_OUT_SPACE) frame_write_d(f, op->o_sp, op->o_off, op->o_sz,
                frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz) <=
                frame_read_d(f,op->i1_sp,op->i1_off,op->i1_sz) ? 1 : 0);
            break;
        case OP_INT_SLESS:
            if (op->o_sp != NO_OUT_SPACE) {
                sz = op->i0_sz;
                frame_write_d(f, op->o_sp, op->o_off, op->o_sz,
                    signed64(frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz),sz) <
                    signed64(frame_read_d(f,op->i1_sp,op->i1_off,op->i1_sz),sz) ? 1 : 0);
            } break;
        case OP_INT_SLESSEQUAL:
            if (op->o_sp != NO_OUT_SPACE) {
                sz = op->i0_sz;
                frame_write_d(f, op->o_sp, op->o_off, op->o_sz,
                    signed64(frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz),sz) <=
                    signed64(frame_read_d(f,op->i1_sp,op->i1_off,op->i1_sz),sz) ? 1 : 0);
            } break;
        case OP_INT_CARRY:
            if (op->o_sp != NO_OUT_SPACE) {
                sz = op->i0_sz;
                a = frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz);
                b = frame_read_d(f,op->i1_sp,op->i1_off,op->i1_sz);
                result = mask64(a+b, sz);
                frame_write_d(f, op->o_sp, op->o_off, op->o_sz, result < mask64(a,sz) ? 1 : 0);
            } break;
        case OP_INT_SCARRY:
            if (op->o_sp != NO_OUT_SPACE) {
                sz = op->i0_sz;
                sa = signed64(frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz),sz);
                sb = signed64(frame_read_d(f,op->i1_sp,op->i1_off,op->i1_sz),sz);
                int64_t sr = sa+sb;
                int64_t msb = (int64_t)1 << (sz*8-1);
                frame_write_d(f, op->o_sp, op->o_off, op->o_sz,
                    ((sa^sr) & (sb^sr) & msb) ? 1 : 0);
            } break;
        case OP_INT_SBORROW:
            if (op->o_sp != NO_OUT_SPACE) {
                sz = op->i0_sz;
                sa = signed64(frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz),sz);
                sb = signed64(frame_read_d(f,op->i1_sp,op->i1_off,op->i1_sz),sz);
                int64_t sr = sa-sb;
                int64_t msb = (int64_t)1 << (sz*8-1);
                frame_write_d(f, op->o_sp, op->o_off, op->o_sz,
                    ((sa^sb) & (sa^sr) & msb) ? 1 : 0);
            } break;
        case OP_INT_ZEXT: case OP_COPY: case OP_INT_TRUNC: case OP_CAST:
            if (op->o_sp != NO_OUT_SPACE) frame_write_d(f, op->o_sp, op->o_off, op->o_sz,
                frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz));
            break;
        case OP_INT_SEXT:
            if (op->o_sp != NO_OUT_SPACE) frame_write_d(f, op->o_sp, op->o_off, op->o_sz,
                (uint64_t)signed64(frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz), op->i0_sz));
            break;
        case OP_PIECE:
            if (op->o_sp != NO_OUT_SPACE) {
                a = frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz);
                b = frame_read_d(f,op->i1_sp,op->i1_off,op->i1_sz);
                frame_write_d(f, op->o_sp, op->o_off, op->o_sz, (a << (op->i1_sz*8)) | b);
            } break;
        case OP_SUBPIECE:
            if (op->o_sp != NO_OUT_SPACE) {
                b = frame_read_d(f,op->i1_sp,op->i1_off,op->i1_sz); /* byte offset */
                if (op->i0_sz > 8 && b >= 8) {
                    /* Source is a 128-bit unique slot written by the splittable block.
                     * The splittable block stores the high 8 bytes at compact index
                     * i0_off+8 (it calls frame_write_d with offset+8, so uniq_arr[i0_off+8]
                     * holds the high half).  Read from there and shift by (b-8) bytes. */
                    uint64_t hi_slot = op->i0_off + 8;
                    uint64_t hi_val = (hi_slot < MAX_UNIQ && f->uniq_set[hi_slot])
                                      ? f->uniq_arr[hi_slot] : 0;
                    frame_write_d(f, op->o_sp, op->o_off, op->o_sz,
                                  hi_val >> ((b - 8) * 8));
                } else {
                    a = frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz);
                    frame_write_d(f, op->o_sp, op->o_off, op->o_sz, a >> (b*8));
                }
            } break;
        case OP_POPCOUNT:
            if (op->o_sp != NO_OUT_SPACE) {
                a = frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz);
                a = a - ((a>>1) & 0x5555555555555555ULL);
                a = (a & 0x3333333333333333ULL) + ((a>>2) & 0x3333333333333333ULL);
                a = (a + (a>>4)) & 0x0F0F0F0F0F0F0F0FULL;
                a = (a * 0x0101010101010101ULL) >> 56;
                frame_write_d(f, op->o_sp, op->o_off, op->o_sz, a);
            } break;
        case OP_LZCOUNT:
            if (op->o_sp != NO_OUT_SPACE) {
                a = frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz);
                /* __builtin_clzll counts over 64 bits, but the varnode is
                 * i0_sz bytes wide -- subtract the padding above it, or a 32-bit
                 * `cntlzw` reports 32 too many.  Latent until big-endian targets
                 * (PPC32's 4-byte registers) reached the C kernel. */
                frame_write_d(f, op->o_sp, op->o_off, op->o_sz,
                    a ? (uint64_t)(__builtin_clzll(a) - (64 - op->i0_sz*8))
                      : (uint64_t)(op->i0_sz*8));
            } break;
        case OP_LOAD:
            if (op->o_sp != NO_OUT_SPACE) {
                a = frame_read_d(f,op->i1_sp,op->i1_off,op->i1_sz);
                frame_write_d(f, op->o_sp, op->o_off, op->o_sz, mem_read(&f->mem, a, op->o_sz, f->is_big_endian));
            } break;
        case OP_STORE:
            a = frame_read_d(f,op->i1_sp,op->i1_off,op->i1_sz);
            b = frame_read_d(f,op->i2_sp,op->i2_off,op->i2_sz);
            mem_write(&f->mem, a, b, op->i2_sz, f->is_big_endian);
            break;
        case OP_MULTIEQUAL: case OP_INDIRECT:
            if (op->o_sp != NO_OUT_SPACE && op->n_ins > 0)
                frame_write_d(f, op->o_sp, op->o_off, op->o_sz,
                    frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz));
            break;
        case OP_CBRANCH: {
            uint64_t cond = frame_read_d(f,op->i1_sp,op->i1_off,op->i1_sz);
            /* CBRANCH semantics — accepted dest patterns:
             *   (1) const-space: pcode-relative signed offset.
             *   (2) ram-space == next_instr_addr: forward skip-to-end.
             *   (3) ram-space == any in-sequence IMARK: in-cell jump.
             *   (4) ram-space, other: real x86 conditional jump out of cell. */
            if (op->i0_sp == SP_CONST) {
                if (cond) {
                    int64_t rel = sign_extend_n((uint64_t)op->i0_off, op->i0_sz);
                    int new_pc = pc + (int)rel;
                    if (new_pc < 0 || new_pc > d->n_ops) return EXEC_FALLBACK;
                    if (rel <= 0) {
                        loop_iters++;
                        if (loop_iters > EXEC_LOOP_BUDGET) return EXEC_FALLBACK;
                    }
                    pc = new_pc;
                    continue;
                }
                /* not taken: fall through to pc++ at end */
                break;
            }
            if (op->i0_sp == SP_RAM) {
                uint64_t dest = (uint64_t)op->i0_off;
                if (dest == next_addr) {
                    if (cond) skip = 1;
                } else {
                    int target_pc = bundle_lookup_imark(d, dest);
                    if (target_pc >= 0) {
                        if (cond) {
                            if (target_pc <= pc) {
                                loop_iters++;
                                if (loop_iters > EXEC_LOOP_BUDGET) return EXEC_FALLBACK;
                            }
                            pc = target_pc;
                            continue;
                        }
                        /* not taken: fall through */
                    } else if (f->arch_pc_off && f->arch_pc_sz) {
                        /* Pattern (4): real x86 conditional jump out of cell. */
                        frame_write_reg(f, (long)f->arch_pc_off, f->arch_pc_sz,
                            cond ? dest : dest + 1);
                    }
                }
                break;
            }
            return EXEC_FALLBACK;
            }
        case OP_BRANCH: {
            /* BRANCH semantics — accepted dest patterns:
             *   (1) const-space: pcode-relative signed offset.
             *   (2) ram-space == any in-sequence IMARK: jump to that op.
             *   (3) ram-space == next_instr_addr: forward skip to end. */
            if (op->i0_sp == SP_CONST) {
                int64_t rel = sign_extend_n((uint64_t)op->i0_off, op->i0_sz);
                int new_pc = pc + (int)rel;
                if (new_pc < 0 || new_pc > d->n_ops) return EXEC_FALLBACK;
                if (rel <= 0) {
                    loop_iters++;
                    if (loop_iters > EXEC_LOOP_BUDGET) return EXEC_FALLBACK;
                }
                pc = new_pc;
                continue;
            }
            if (op->i0_sp == SP_RAM) {
                uint64_t dest = (uint64_t)op->i0_off;
                if (dest == next_addr) {
                    pc = d->n_ops;  /* exit the cell */
                    continue;
                }
                int target_pc = bundle_lookup_imark(d, dest);
                if (target_pc >= 0) {
                    if (target_pc <= pc) {
                        loop_iters++;
                        if (loop_iters > EXEC_LOOP_BUDGET) return EXEC_FALLBACK;
                    }
                    pc = target_pc;
                    continue;
                }
                return EXEC_FALLBACK;
            }
            return EXEC_FALLBACK;
            }
        case OP_BRANCHIND: case OP_CALLIND:
            /* Write the jump target into the architectural PC register so that
             * SimCell read-back produces the correct concrete value: SimH and
             * SimL will differ exactly when the target depends on a tainted
             * register, matching the AVALANCHE expression the engine generates.
             *
             * Without this write PC stays 0 (pristine), making SimH == SimL
             * and collapsing the differential to 0.  All other register writes
             * for the instruction (link-register save, RSP adjustment, etc.)
             * were already handled by the preceding pcode ops. */
            if (f->arch_pc_off && f->arch_pc_sz) {
                frame_write_reg(f, (long)f->arch_pc_off, f->arch_pc_sz,
                    frame_read_d(f, op->i0_sp, op->i0_off, op->i0_sz));
            }
            break;
        case OP_CALLOTHER: if (op->callother_out) return EXEC_FALLBACK; break;
        case OP_FLOAT_ANY: case OP_TRUNC_FLOAT: case OP_UNKNOWN: return EXEC_FALLBACK;
        default: break;
        }
        if (tf && op->o_sp != NO_OUT_SPACE) {
            int decline = 0;
            uint64_t ot = mt_op_taint(op, t_iv, t_it, tc, &decline);
            if (decline) return EXEC_TAINT_DECLINE;
            frame_write_d(tf, op->o_sp, op->o_off, op->o_sz, ot);
            if (tc) tc->writes++;
        }
        pc++;
    }
    /* Note: pcode-native does NOT write RIP at end-of-cell for in-cell
     * loops (rep stosb/movsb, BSF/BSR).  Unicorn's apparent RIP taint
     * for these instructions in the test_pcode_matches_unicorn parity
     * tests comes from a different layer of the rule generator, not
     * from per-cell evaluation, and is over-conservative.  Pcode-native
     * is the more precise answer and the parity test should be relaxed
     * for these cases. */
    return EXEC_OK;
}

/* Value-only execution: the historical entry point, unchanged in behaviour. */
static inline int execute_decoded(Frame *f, const DecodedBundle *d) {
    return execute_decoded_t(f, NULL, d, NULL);
}

#endif
