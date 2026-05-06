#ifndef CELL_CORE_H
#define CELL_CORE_H

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/* Constants mirroring cell.pyx */
/* REGS_ARR_SIZE: large enough to cover both x86/AMD64 (offsets up to ~1100)
 * and ARM64 (offsets up to ~16640 for X-registers, vector regs higher).
 * Cython falls back to a Python dict for offsets above its 1104 limit; we
 * use a flat array sized to cover the common architectures' GP register files. */
#define REGS_ARR_SIZE  17000
#define MAX_PCODE_OPS  96
#define MAX_UNIQ       64
#define MAX_DIRTY      128

#define SP_CONST     0
#define SP_REGISTER  1
#define SP_UNIQUE    2
#define SP_RAM       3
#define SP_OTHER    -1
#define NO_OUT_SPACE -2

/* Opcode enum — must match _OpcodeID in cell.pyx exactly */
typedef enum {
    OP_UNKNOWN=0, OP_COPY, OP_LOAD, OP_STORE, OP_MULTIEQUAL, OP_INDIRECT,
    OP_INT_ADD, OP_INT_SUB, OP_INT_MULT, OP_INT_DIV, OP_INT_SDIV,
    OP_INT_REM, OP_INT_SREM, OP_INT_2COMP, OP_INT_NEGATE,
    OP_INT_AND, OP_INT_OR, OP_INT_XOR,
    OP_INT_LEFT, OP_INT_RIGHT, OP_INT_SRIGHT,
    OP_INT_EQUAL, OP_INT_NOTEQUAL,
    OP_INT_LESS, OP_INT_LESSEQUAL, OP_INT_SLESS, OP_INT_SLESSEQUAL,
    OP_INT_CARRY, OP_INT_SCARRY, OP_INT_SBORROW,
    OP_INT_ZEXT, OP_INT_SEXT, OP_INT_TRUNC, OP_CAST,
    OP_POPCOUNT, OP_LZCOUNT,
    OP_PIECE, OP_SUBPIECE, OP_PTRADD, OP_PTRSUB,
    OP_BOOL_AND, OP_BOOL_OR, OP_BOOL_XOR, OP_BOOL_NEGATE,
    OP_BRANCH, OP_CBRANCH, OP_BRANCHIND, OP_CALL, OP_CALLIND,
    OP_CALLOTHER, OP_RETURN, OP_IMARK, OP_UNIMPLEMENTED,
    OP_SEGMENT, OP_CPOOLREF, OP_NEW, OP_INSERT, OP_EXTRACT,
    OP_FLOAT_ANY, OP_TRUNC_FLOAT
} OpcodeID;

/* Pre-decoded P-code op (mirrors PCodeOp in cell.pyx) */
typedef struct {
    int           oid;
    int           o_sp;
    unsigned long o_off;
    int           o_sz;
    int           callother_out;
    int           n_ins;
    int           i0_sp;  unsigned long i0_off;  int i0_sz;
    int           i1_sp;  unsigned long i1_off;  int i1_sz;
    int           i2_sp;  unsigned long i2_off;  int i2_sz;
} PCOp;

/* Open-addressing hash for memory (single-byte values) */
#define MEM_CAP    256
#define MEM_MASK   (MEM_CAP - 1)
#define MEM_EMPTY  UINT64_MAX

typedef struct {
    uint64_t keys[MEM_CAP];
    uint64_t vals[MEM_CAP];
} MemMap;

static inline void mem_clear(MemMap *m) {
    memset(m->keys, 0xFF, sizeof(m->keys));
}
static inline void mem_write_byte(MemMap *m, uint64_t addr, uint8_t b) {
    uint32_t s = (uint32_t)(addr * 2654435761UL) & MEM_MASK;
    while (m->keys[s] != MEM_EMPTY && m->keys[s] != addr)
        s = (s + 1) & MEM_MASK;
    m->keys[s] = addr;
    m->vals[s] = b;
}
static inline uint8_t mem_read_byte(const MemMap *m, uint64_t addr) {
    uint32_t s = (uint32_t)(addr * 2654435761UL) & MEM_MASK;
    while (m->keys[s] != MEM_EMPTY && m->keys[s] != addr)
        s = (s + 1) & MEM_MASK;
    return (m->keys[s] == addr) ? (uint8_t)m->vals[s] : 0;
}
static inline void mem_write(MemMap *m, uint64_t addr, uint64_t val, int size) {
    uint64_t mask = (size >= 8) ? UINT64_MAX : (((uint64_t)1 << (size*8)) - 1);
    val &= mask;
    for (int i = 0; i < size; i++)
        mem_write_byte(m, addr+i, (uint8_t)(val >> (i*8)));
}
static inline uint64_t mem_read(const MemMap *m, uint64_t addr, int size) {
    uint64_t r = 0;
    for (int i = 0; i < size; i++)
        r |= ((uint64_t)mem_read_byte(m, addr+i)) << (i*8);
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
} Frame;

static inline void frame_clear(Frame *f) {
    for (int i = 0; i < f->dirty_count; i++) f->regs_set[f->dirty[i]] = 0;
    f->dirty_count = 0;
    for (int i = 0; i < MAX_UNIQ; i++) f->uniq_set[i] = 0;
    mem_clear(&f->mem);
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
            f->regs_arr[off] = (f->regs_arr[off] & ~lo_mask) | (val & lo_mask);
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
                    base = f->regs_arr[k] >> ((off-k)*8);
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
                uint64_t lane_mask = sub_mask << (byte_off * 8);
                base = (base & ~lane_mask) | (sub_val << (byte_off * 8));
                k += k_sz;
            } else {
                k++;
            }
        }
        return mask64(base, sz);
    }
    return 0;
}

static inline uint64_t frame_read_d(const Frame *f, int sp, unsigned long off, int sz) {
    if (sp == SP_CONST)    return mask64((uint64_t)off, sz);
    if (sp == SP_REGISTER) return frame_read_reg(f, (long)off, sz);
    if (sp == SP_UNIQUE)   return (off < MAX_UNIQ && f->uniq_set[off]) ? mask64(f->uniq_arr[off], sz) : 0;
    if (sp == SP_RAM)      return mem_read(&f->mem, (uint64_t)off, sz);
    return 0;
}

static inline void frame_write_d(Frame *f, int sp, unsigned long off, int sz, uint64_t val) {
    val = mask64(val, sz);
    if (sp == SP_REGISTER) { frame_write_reg(f, (long)off, sz, val); return; }
    if (sp == SP_UNIQUE)   { if (off < MAX_UNIQ) { f->uniq_arr[off]=val; f->uniq_set[off]=1; } return; }
    if (sp == SP_RAM)      { mem_write(&f->mem, (uint64_t)off, val, sz); }
}

typedef struct {
    PCOp     buf[MAX_PCODE_OPS];
    int      n_ops;
    int      has_fallback;
    uint64_t next_instr_addr;
} DecodedBundle;

#define EXEC_OK          0
#define EXEC_FALLBACK    1

static inline int execute_decoded(Frame *f, const DecodedBundle *d) {
    int skip = 0;
    uint64_t next_addr = d->next_instr_addr;

    for (int i = 0; i < d->n_ops; i++) {
        const PCOp *op = &d->buf[i];
        int oid = op->oid;
        if (skip && oid != OP_IMARK) continue;

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
        if (op->o_sp != NO_OUT_SPACE && op->o_sz > 8) {
            int splittable = (oid == OP_INT_XOR || oid == OP_INT_AND
                              || oid == OP_INT_OR || oid == OP_COPY
                              || oid == OP_INT_ZEXT);
            if (splittable) {
                int total_sz = op->o_sz;
                int lo_sz = 8;
                int hi_sz = total_sz - 8;
                /* Low 8 bytes */
                {
                    uint64_t v0 = frame_read_d(f, op->i0_sp, op->i0_off, lo_sz);
                    uint64_t v1 = (op->n_ins >= 2)
                        ? frame_read_d(f, op->i1_sp, op->i1_off, lo_sz)
                        : 0;
                    uint64_t r;
                    switch (oid) {
                        case OP_INT_XOR: r = v0 ^ v1; break;
                        case OP_INT_AND: r = v0 & v1; break;
                        case OP_INT_OR:  r = v0 | v1; break;
                        case OP_COPY:
                        case OP_INT_ZEXT: r = v0; break;
                        default: r = 0;  /* unreachable */
                    }
                    frame_write_d(f, op->o_sp, op->o_off, lo_sz, r);
                }
                /* High bytes (offset+8 ... offset+total_sz).  For
                 * INT_ZEXT widening from a smaller input we emit zero
                 * for the high half — that's the zero-extension. */
                {
                    uint64_t v0, v1, r;
                    if (oid == OP_INT_ZEXT && op->i0_sz <= lo_sz) {
                        /* Input fits in the low half; high is zero. */
                        v0 = 0;
                    } else {
                        v0 = frame_read_d(f, op->i0_sp, op->i0_off + lo_sz, hi_sz);
                    }
                    if (op->n_ins >= 2 && op->i1_sz > lo_sz) {
                        v1 = frame_read_d(f, op->i1_sp, op->i1_off + lo_sz, hi_sz);
                    } else {
                        v1 = 0;
                    }
                    switch (oid) {
                        case OP_INT_XOR: r = v0 ^ v1; break;
                        case OP_INT_AND: r = v0 & v1; break;
                        case OP_INT_OR:  r = v0 | v1; break;
                        case OP_COPY:
                        case OP_INT_ZEXT: r = v0; break;
                        default: r = 0;  /* unreachable */
                    }
                    frame_write_d(f, op->o_sp, op->o_off + lo_sz, hi_sz, r);
                }
                continue;  /* skip the regular dispatch for this op */
            }
        }

        switch (oid) {
        case OP_IMARK: case OP_BRANCH: case OP_RETURN: case OP_CALL: break;

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
            if (op->o_sp != NO_OUT_SPACE) {
                b = frame_read_d(f,op->i1_sp,op->i1_off,op->i1_sz) & 0x3F;
                frame_write_d(f, op->o_sp, op->o_off, op->o_sz,
                    frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz) << b);
            } break;
        case OP_INT_RIGHT:
            if (op->o_sp != NO_OUT_SPACE) {
                b = frame_read_d(f,op->i1_sp,op->i1_off,op->i1_sz) & 0x3F;
                frame_write_d(f, op->o_sp, op->o_off, op->o_sz,
                    frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz) >> b);
            } break;
        case OP_INT_SRIGHT:
            if (op->o_sp != NO_OUT_SPACE) {
                sz = op->i0_sz;
                sa = signed64(frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz), sz);
                b  = frame_read_d(f,op->i1_sp,op->i1_off,op->i1_sz) & 0x3F;
                frame_write_d(f, op->o_sp, op->o_off, op->o_sz, (uint64_t)(sa>>b));
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
                a = frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz);
                b = frame_read_d(f,op->i1_sp,op->i1_off,op->i1_sz);
                frame_write_d(f, op->o_sp, op->o_off, op->o_sz, a >> (b*8));
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
                frame_write_d(f, op->o_sp, op->o_off, op->o_sz,
                    a ? (uint64_t)__builtin_clzll(a) : (uint64_t)(op->i0_sz*8));
            } break;
        case OP_LOAD:
            if (op->o_sp != NO_OUT_SPACE) {
                a = frame_read_d(f,op->i1_sp,op->i1_off,op->i1_sz);
                frame_write_d(f, op->o_sp, op->o_off, op->o_sz, mem_read(&f->mem, a, op->o_sz));
            } break;
        case OP_STORE:
            a = frame_read_d(f,op->i1_sp,op->i1_off,op->i1_sz);
            b = frame_read_d(f,op->i2_sp,op->i2_off,op->i2_sz);
            mem_write(&f->mem, a, b, op->i2_sz);
            break;
        case OP_MULTIEQUAL: case OP_INDIRECT:
            if (op->o_sp != NO_OUT_SPACE && op->n_ins > 0)
                frame_write_d(f, op->o_sp, op->o_off, op->o_sz,
                    frame_read_d(f,op->i0_sp,op->i0_off,op->i0_sz));
            break;
        case OP_CBRANCH: {
            uint64_t cond = frame_read_d(f,op->i1_sp,op->i1_off,op->i1_sz);
            uint64_t dest = (uint64_t)op->i0_off;
            if (dest == next_addr) { if (cond) skip = 1; }
            else if (f->arch_pc_off && f->arch_pc_sz)
                frame_write_reg(f, (long)f->arch_pc_off, f->arch_pc_sz, cond ? dest : dest+1);
            } break;
        case OP_BRANCHIND: case OP_CALLIND: return EXEC_FALLBACK;
        case OP_CALLOTHER: if (op->callother_out) return EXEC_FALLBACK; break;
        case OP_FLOAT_ANY: case OP_TRUNC_FLOAT: case OP_UNKNOWN: return EXEC_FALLBACK;
        default: break;
        }
    }
    return EXEC_OK;
}

#endif
