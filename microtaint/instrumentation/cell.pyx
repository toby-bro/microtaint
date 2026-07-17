# cython: language_level=3
# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: infer_types=True
# cython: cdivision=True
"""
microtaint.instrumentation.cell  (Cython)
==========================================
Native P-code differential evaluator — fast Cython port of cell.py.

Performance architecture
------------------------
The bottleneck in a pure Python pcode evaluator is Python attribute access
inside the hot loop: `vn.space.name`, `vn.offset`, `vn.size` and
`op.opcode.name` are all Python attr lookups, costing ~50 ns each.
For a typical instruction with 12 pcode ops and 3 varnodes per op, that is
~1800 ns of pure attribute-access overhead before any arithmetic is done.

This file eliminates that overhead with a **pre-decode** step:
  _predecode_ops(arch, bytestring) → list[tuple]
  Each tuple is (oid, out_space, out_off, out_sz, inputs)
  where inputs = ((space0, off0, sz0), (space1, off1, sz1), …)
  All fields are C ints pre-extracted from the pypcode objects.
  This list is cached with the same LRU as the translation itself.

The inner loop in execute_decoded() unpacks plain Python tuples into typed
Cython `cdef int` locals — zero attribute access, zero Python object creation
for the common arithmetic operations.

Frame storage uses Python dicts (identical to cell.py) so the AH/BH
sub-register fallback is preserved exactly.
"""

from libc.stdint cimport uint64_t, int64_t, uint8_t

import functools
import logging

from microtaint.sleigh.lifter import get_context
from microtaint.types import Architecture

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentinel
# ---------------------------------------------------------------------------

class PCodeFallbackNeeded(Exception):
    """Raised when the native evaluator encounters an unsupported opcode."""


# ---------------------------------------------------------------------------
# Space ID constants  (avoids 'register'/'unique'/'const' string comparison)
# ---------------------------------------------------------------------------

cdef int SP_CONST    = 0
cdef int SP_REGISTER = 1
cdef int SP_UNIQUE   = 2
cdef int SP_RAM      = 3
cdef int SP_OTHER    = -1

cdef int NO_OUT_SPACE = -2   # sentinel for "no output varnode"

_SPACE_IDS = {
    'const':    SP_CONST,
    'register': SP_REGISTER,
    'unique':   SP_UNIQUE,
    'ram':      SP_RAM,
}


# ---------------------------------------------------------------------------
# Mask helpers (pure C, no GIL)
# ---------------------------------------------------------------------------

cdef uint64_t _MASK_TABLE[9]
_MASK_TABLE[0] = 0
_MASK_TABLE[1] = 0xFF
_MASK_TABLE[2] = 0xFFFF
_MASK_TABLE[3] = 0xFFFFFF
_MASK_TABLE[4] = 0xFFFFFFFF
_MASK_TABLE[5] = 0xFFFFFFFFFF
_MASK_TABLE[6] = 0xFFFFFFFFFFFF
_MASK_TABLE[7] = 0xFFFFFFFFFFFFFF
_MASK_TABLE[8] = 0xFFFFFFFFFFFFFFFF

cdef uint64_t _SEXT_TABLE[9]
_SEXT_TABLE[0] = 0xFFFFFFFFFFFFFFFF
_SEXT_TABLE[1] = 0xFFFFFFFFFFFFFF00
_SEXT_TABLE[2] = 0xFFFFFFFFFFFF0000
_SEXT_TABLE[3] = 0xFFFFFFFFFF000000
_SEXT_TABLE[4] = 0xFFFFFFFF00000000
_SEXT_TABLE[5] = 0xFFFFFF0000000000
_SEXT_TABLE[6] = 0xFFFF000000000000
_SEXT_TABLE[7] = 0xFF00000000000000
_SEXT_TABLE[8] = 0x0000000000000000


cdef inline uint64_t _mask64(uint64_t val, int sz) noexcept nogil:
    if sz <= 0 or sz > 8:
        return val
    return val & _MASK_TABLE[sz]


cdef inline int64_t _signed64(uint64_t val, int sz) noexcept nogil:
    cdef uint64_t msb
    val = _mask64(val, sz)
    msb = <uint64_t>1 << (sz * 8 - 1)
    if val & msb:
        return <int64_t>(val | _SEXT_TABLE[sz])
    return <int64_t>val


# ---------------------------------------------------------------------------
# Opcode integer IDs
# ---------------------------------------------------------------------------

cdef enum _OpcodeID:
    OP_UNKNOWN = 0
    OP_COPY
    OP_LOAD
    OP_STORE
    OP_MULTIEQUAL
    OP_INDIRECT
    OP_INT_ADD
    OP_INT_SUB
    OP_INT_MULT
    OP_INT_DIV
    OP_INT_SDIV
    OP_INT_REM
    OP_INT_SREM
    OP_INT_2COMP
    OP_INT_NEGATE
    OP_INT_AND
    OP_INT_OR
    OP_INT_XOR
    OP_INT_LEFT
    OP_INT_RIGHT
    OP_INT_SRIGHT
    OP_INT_EQUAL
    OP_INT_NOTEQUAL
    OP_INT_LESS
    OP_INT_LESSEQUAL
    OP_INT_SLESS
    OP_INT_SLESSEQUAL
    OP_INT_CARRY
    OP_INT_SCARRY
    OP_INT_SBORROW
    OP_INT_ZEXT
    OP_INT_SEXT
    OP_INT_TRUNC
    OP_CAST
    OP_POPCOUNT
    OP_LZCOUNT
    OP_PIECE
    OP_SUBPIECE
    OP_PTRADD
    OP_PTRSUB
    OP_BOOL_AND
    OP_BOOL_OR
    OP_BOOL_XOR
    OP_BOOL_NEGATE
    OP_BRANCH
    OP_CBRANCH
    OP_BRANCHIND
    OP_CALL
    OP_CALLIND
    OP_CALLOTHER
    OP_RETURN
    OP_IMARK
    OP_UNIMPLEMENTED
    OP_SEGMENT
    OP_CPOOLREF
    OP_NEW
    OP_INSERT
    OP_EXTRACT
    OP_FLOAT_ANY
    OP_TRUNC_FLOAT


_OPCODE_ID = {
    'COPY': OP_COPY, 'LOAD': OP_LOAD, 'STORE': OP_STORE,
    'MULTIEQUAL': OP_MULTIEQUAL, 'INDIRECT': OP_INDIRECT,
    'INT_ADD': OP_INT_ADD, 'INT_SUB': OP_INT_SUB, 'INT_MULT': OP_INT_MULT,
    'INT_DIV': OP_INT_DIV, 'INT_SDIV': OP_INT_SDIV,
    'INT_REM': OP_INT_REM, 'INT_SREM': OP_INT_SREM,
    'INT_2COMP': OP_INT_2COMP, 'INT_NEGATE': OP_INT_NEGATE,
    'INT_AND': OP_INT_AND, 'INT_OR': OP_INT_OR, 'INT_XOR': OP_INT_XOR,
    'INT_LEFT': OP_INT_LEFT, 'INT_RIGHT': OP_INT_RIGHT, 'INT_SRIGHT': OP_INT_SRIGHT,
    'INT_EQUAL': OP_INT_EQUAL, 'INT_NOTEQUAL': OP_INT_NOTEQUAL,
    'INT_LESS': OP_INT_LESS, 'INT_LESSEQUAL': OP_INT_LESSEQUAL,
    'INT_SLESS': OP_INT_SLESS, 'INT_SLESSEQUAL': OP_INT_SLESSEQUAL,
    'INT_CARRY': OP_INT_CARRY, 'INT_SCARRY': OP_INT_SCARRY, 'INT_SBORROW': OP_INT_SBORROW,
    'INT_ZEXT': OP_INT_ZEXT, 'INT_SEXT': OP_INT_SEXT, 'INT_TRUNC': OP_INT_TRUNC,
    'CAST': OP_CAST, 'POPCOUNT': OP_POPCOUNT, 'LZCOUNT': OP_LZCOUNT,
    'PIECE': OP_PIECE, 'SUBPIECE': OP_SUBPIECE,
    'PTRADD': OP_PTRADD, 'PTRSUB': OP_PTRSUB,
    'BOOL_AND': OP_BOOL_AND, 'BOOL_OR': OP_BOOL_OR,
    'BOOL_XOR': OP_BOOL_XOR, 'BOOL_NEGATE': OP_BOOL_NEGATE,
    'BRANCH': OP_BRANCH, 'CBRANCH': OP_CBRANCH, 'BRANCHIND': OP_BRANCHIND,
    'CALL': OP_CALL, 'CALLIND': OP_CALLIND, 'CALLOTHER': OP_CALLOTHER, 'RETURN': OP_RETURN,
    'IMARK': OP_IMARK, 'UNIMPLEMENTED': OP_UNIMPLEMENTED,
    'SEGMENT': OP_SEGMENT, 'CPOOLREF': OP_CPOOLREF, 'NEW': OP_NEW,
    'INSERT': OP_INSERT, 'EXTRACT': OP_EXTRACT,
    'TRUNC': OP_TRUNC_FLOAT, 'CEIL': OP_TRUNC_FLOAT,
    'FLOOR': OP_TRUNC_FLOAT, 'ROUND': OP_TRUNC_FLOAT,
}


cdef inline int _opcode_id(object name) noexcept:
    cdef object r = _OPCODE_ID.get(name)
    if r is None:
        if (<str>name).startswith('FLOAT_'):
            return OP_FLOAT_ANY
        return OP_UNKNOWN
    return <int>r


# ---------------------------------------------------------------------------
# Pre-decode: convert PcodeOp list to plain int tuples (cached)
#
# Each decoded op is a Python tuple:
#   (oid, out_space, out_off, out_sz, has_callother_output,
#    n_inputs, i0_sp, i0_off, i0_sz, i1_sp, i1_off, i1_sz,
#                                    i2_sp, i2_off, i2_sz)
# Fixed-width 15-int tuple so Cython can unpack with typed locals.
# i2_* is (0,0,0) when n_inputs < 3.
# has_callother_output is 1 when oid==OP_CALLOTHER and output is not None.
# ---------------------------------------------------------------------------

cdef int _sp_id(object vn):
    cdef object s = _SPACE_IDS.get(vn.space.name)
    return <int>s if s is not None else SP_OTHER



# Covers all normal x86-64 Sleigh register offsets (0 … 1103).
# Exotic registers (segment descriptors, BND, …) fall back to the dict.
DEF REGS_ARR_SIZE = 1104

# Max pcode ops per instruction (empirically: BT ~44, SHR ~38, typical ~16).
DEF MAX_PCODE_OPS = 96

ctypedef struct PCodeOp:
    int           oid
    int           o_sp
    unsigned long o_off
    int           o_sz
    int           callother_out
    int           n_ins
    int           i0_sp
    unsigned long i0_off
    int           i0_sz
    int           i1_sp
    unsigned long i1_off
    int           i1_sz
    int           i2_sp
    unsigned long i2_off
    int           i2_sz


cdef class DecodedOps:
    """Cached pre-decoded pcode ops in a C struct array — no Python tuple overhead."""
    cdef PCodeOp buf[MAX_PCODE_OPS]
    cdef public int       n_ops
    cdef public bint      has_fallback
    cdef public uint64_t  next_instr_addr  # 0x1000 + len(bytestring); used by CBRANCH to detect CMOVxx skip pattern
    cdef public uint64_t  imark_addr       # 0x1000 — base address used by IMARK; loop-back BRANCH targets this
    cdef public bint      has_loop         # True iff a backward BRANCH to imark_addr is present (rep stosb / movsb pattern)
    # Map from in-sequence x86 instruction address (= IMARK ram address) to
    # the pcode-op index where that IMARK lives.  Populated during predecode.
    # Used by BRANCH / CBRANCH evaluation to translate a ram-space target
    # back to a pcode pc.  Also includes ``next_instr_addr`` -> n_ops as a
    # synthetic "after the last op" entry so forward-skip-to-end works.
    cdef public dict      imark_to_pc
    cdef public object    input_reg_offsets  # set of SP_REGISTER input offsets
    cdef public object    _uc_arrays         # cached (ids_arr,vals_arr,ptrs_arr,names,needs_eflags)
    # Intermediate-taint materialization (see docs/design/intermediate-taint-materialization.md):
    #   uniq_map    — raw SP_UNIQUE Sleigh offset -> compact slot index 0..31.  The compact
    #                 index is what the op buffer stores; this recovers the slot from the raw
    #                 offset the engine knows, so a caller can seed/read an intermediate.
    #   uniq_def_pc — compact slot -> pc index of its FIRST defining op.  A seeded-partial run
    #                 starts just past this so the arithmetic core is not recomputed.
    cdef public dict      uniq_map
    cdef public dict      uniq_def_pc

    def get_buf_bytes(self):
        """Return the raw PCodeOp struct array as a bytes object.
        Used by cell_c.c to copy the pre-decoded ops into its own DecodedBundle."""
        return bytes((<unsigned char*>self.buf)[:sizeof(PCodeOp) * MAX_PCODE_OPS])


def _predecode_ops(arch, bytestring):
    """
    Translate bytestring into a DecodedOps C-struct buffer (cached by _get_decoded).
    Returns a DecodedOps instance with has_fallback, n_ops, and the struct array filled.
    All fields are C-typed — no Python tuples in the execution hot loop.
    """
    ctx         = get_context(arch)
    translation = ctx.translate(bytestring, 0x1000)
    ops         = translation.ops
    cdef DecodedOps result = DecodedOps()
    result.n_ops = 0
    result.has_fallback = False
    # Translation always uses base 0x1000; next x86 instruction is right after the bytes.
    # CMOVxx CBRANCH targets exactly this address; a JL/JE targets a different address.
    result.next_instr_addr = <uint64_t>(0x1000 + len(bytestring))
    result.imark_addr = <uint64_t>0x1000
    result.has_loop = False
    result.imark_to_pc = {}
    result.input_reg_offsets = set()
    result.uniq_def_pc = {}
    has_fallback = False
    # Compact unique-space mapping: raw offset → sequential index 0,1,2,...
    cdef dict uniq_map = {}
    cdef int  uniq_next = 0
    cdef PCodeOp* op_ptr
    cdef int _ii, _oid, _cb_total
    cdef bint _ok
    cdef uint64_t _cb_dest

    for op in ops:
        oid    = _opcode_id(op.opcode.name)
        out    = op.output
        ins    = op.inputs

        # Check for fallback conditions
        if oid == OP_BRANCH:
            has_fallback = True  # may be cleared after full decode
        elif oid == OP_CBRANCH:
            has_fallback = True  # may be cleared after full decode
        elif oid == OP_CALLOTHER and out is not None:
            has_fallback = True
        elif oid == OP_FLOAT_ANY or oid == OP_TRUNC_FLOAT:
            has_fallback = True
        elif oid == OP_UNKNOWN:
            has_fallback = True

        # Encode output
        if out is None:
            o_sp, o_off, o_sz = NO_OUT_SPACE, 0, 0
        else:
            o_sp, o_off, o_sz = _sp_id(out), out.offset, out.size
            if o_sp == SP_UNIQUE:
                if o_off not in uniq_map:
                    uniq_map[o_off] = uniq_next; uniq_next += 1
                o_off = uniq_map[o_off]

        # Encode inputs (up to 3)
        n = len(ins)
        if n >= 1:
            i0_sp, i0_off, i0_sz = _sp_id(ins[0]), ins[0].offset, ins[0].size
            if i0_sp == SP_UNIQUE:
                if i0_off not in uniq_map:
                    uniq_map[i0_off] = uniq_next; uniq_next += 1
                i0_off = uniq_map[i0_off]
        else:
            i0_sp, i0_off, i0_sz = 0, 0, 0
        if n >= 2:
            i1_sp, i1_off, i1_sz = _sp_id(ins[1]), ins[1].offset, ins[1].size
            if i1_sp == SP_UNIQUE:
                if i1_off not in uniq_map:
                    uniq_map[i1_off] = uniq_next; uniq_next += 1
                i1_off = uniq_map[i1_off]
        else:
            i1_sp, i1_off, i1_sz = 0, 0, 0
        if n >= 3:
            i2_sp, i2_off, i2_sz = _sp_id(ins[2]), ins[2].offset, ins[2].size
            if i2_sp == SP_UNIQUE:
                if i2_off not in uniq_map:
                    uniq_map[i2_off] = uniq_next; uniq_next += 1
                i2_off = uniq_map[i2_off]
        else:
            i2_sp, i2_off, i2_sz = 0, 0, 0

        # has_callother_output flag (needed to raise the right fallback)
        callother_out = 1 if (oid == OP_CALLOTHER and out is not None) else 0

        if result.n_ops < MAX_PCODE_OPS:
            op_ptr = &result.buf[result.n_ops]
            op_ptr.oid = oid
            op_ptr.o_sp = o_sp; op_ptr.o_off = o_off; op_ptr.o_sz = o_sz
            op_ptr.callother_out = callother_out; op_ptr.n_ins = n
            op_ptr.i0_sp = i0_sp; op_ptr.i0_off = i0_off; op_ptr.i0_sz = i0_sz
            op_ptr.i1_sp = i1_sp; op_ptr.i1_off = i1_off; op_ptr.i1_sz = i1_sz
            op_ptr.i2_sp = i2_sp; op_ptr.i2_off = i2_off; op_ptr.i2_sz = i2_sz
            # Track IMARK ram-address → pc index so BRANCH/CBRANCH can map a
            # ram destination back to a pcode op.  IMARK input is the x86
            # instruction's start address in ram space.
            if oid == OP_IMARK and i0_sp == SP_RAM:
                result.imark_to_pc[<uint64_t>i0_off] = result.n_ops
            # Record the first defining op index for each unique compact slot
            # (o_off is already the compact index after the remap above).
            if o_sp == SP_UNIQUE and o_off not in result.uniq_def_pc:
                result.uniq_def_pc[o_off] = result.n_ops
            result.n_ops += 1

    # Expose the raw-offset -> compact-slot map so callers can name an
    # intermediate unique varnode for seeding / reading (materialization).
    result.uniq_map = uniq_map

    # ── Decide whether this instruction can run in the pcode evaluator ──
    # without falling back to Unicorn.  The supported control-flow patterns:
    #
    #   (a) JL / JE / JNE etc.  — exactly one CBRANCH, the last op,
    #                             dest in ram space, dest != next_instr_addr.
    #                             The evaluator writes RIP = dest or dest+1 so
    #                             the differential XOR detects tainted control
    #                             flow as RIP taint.
    #   (b) CMOVZ / CMOVNZ etc. — CBRANCH (anywhere) with dest == next_instr_addr.
    #                             ``skip_remaining`` is set when the condition
    #                             is true; subsequent ops become no-ops.
    #   (c) BSF / BSR / RCL etc. — internal pcode-relative loops.  BRANCH and
    #                             CBRANCH with dest in const space — interpreter
    #                             follows them as signed relative jumps (capped
    #                             to a small iteration budget for safety).
    #   (d) REP STOSB / MOVSB    — a CBRANCH to next_instr_addr (loop exit) plus
    #                             a backward BRANCH to imark_addr (loop top).
    #                             Iteration count is bounded by the per-cell
    #                             iteration budget.
    #
    # Anything else stays has_fallback=True:
    #   - CALLOTHER with output, FLOAT, UNKNOWN (already flagged earlier)
    #   - BRANCH to ram[X] where X is neither imark_addr nor next_instr_addr.
    #
    # Note: BRANCHIND and CALLIND are NOT in this list.  They have no output
    # varnode and are treated as no-ops in execute_decoded (same as RETURN):
    # all register writes produced by the instruction appear in pcode ops that
    # precede the terminal CALLIND/BRANCHIND, and T_PC is handled by the
    # engine's map_outputs_to_targets → AVALANCHE floor path.  Routing to
    # Unicorn for these ops is never necessary and always slower (SimH with
    # extreme polarity inputs causes an immediate FETCH_UNMAPPED fault anyway).
    if has_fallback:
        _ok = 1
        _cb_total = 0
        for _ii in range(result.n_ops):
            _oid = result.buf[_ii].oid
            if _oid == OP_BRANCH:
                # const-space target: pcode-relative signed offset.  Always
                # supported — interpreter follows it as a relative jump.
                if result.buf[_ii].i0_sp == SP_CONST:
                    continue
                # ram-space target: valid iff it lands on an IMARK address
                # already in this sequence (loop-back or in-sequence jump),
                # or equals next_instr_addr (forward skip to the end).
                # Anything else is a real x86 jump out of the cell, which we
                # don't model — fall back.
                if result.buf[_ii].i0_sp == SP_RAM:
                    _cb_dest = <uint64_t>result.buf[_ii].i0_off
                    if _cb_dest == result.next_instr_addr:
                        continue
                    if _cb_dest in result.imark_to_pc:
                        # Backward jump => loop; flag for the iteration cap.
                        if (<int>result.imark_to_pc[_cb_dest]) <= _ii:
                            result.has_loop = True
                        continue
                _ok = 0; break
            if _oid == OP_CBRANCH:
                _cb_total += 1
                if result.buf[_ii].i0_sp == SP_CONST:
                    # const-space pcode-relative CBRANCH (BSF/BSR pattern) —
                    # interpreter follows it as a signed relative jump.
                    continue
                if result.buf[_ii].i0_sp != SP_RAM:
                    _ok = 0; break
                _cb_dest = <uint64_t>result.buf[_ii].i0_off
                # Forward skip to end of sequence (CMOVxx, rep-loop exit).
                if _cb_dest == result.next_instr_addr:
                    continue
                # Any in-sequence IMARK address — a forward or backward
                # conditional jump within this cell is fine.
                if _cb_dest in result.imark_to_pc:
                    if (<int>result.imark_to_pc[_cb_dest]) <= _ii:
                        result.has_loop = True
                    continue
                # Pattern (a): real x86 conditional branch out of cell —
                # only valid if this is the last op (exit pattern).
                if _ii != result.n_ops - 1:
                    _ok = 0; break
        if _ok:
            has_fallback = False

    for _ii in range(result.n_ops):
        if result.buf[_ii].n_ins >= 1 and result.buf[_ii].i0_sp == SP_REGISTER:
            result.input_reg_offsets.add(result.buf[_ii].i0_off)
        if result.buf[_ii].n_ins >= 2 and result.buf[_ii].i1_sp == SP_REGISTER:
            result.input_reg_offsets.add(result.buf[_ii].i1_off)
        if result.buf[_ii].n_ins >= 3 and result.buf[_ii].i2_sp == SP_REGISTER:
            result.input_reg_offsets.add(result.buf[_ii].i2_off)

    result.has_fallback = has_fallback
    return result


@functools.lru_cache(maxsize=16384)
def _get_decoded(arch, bytestring):
    # Returns a DecodedOps object (C struct array, has_fallback, input_reg_offsets)
    return _predecode_ops(arch, bytestring)


# ---------------------------------------------------------------------------
# Frame  — Python-dict storage (identical semantics to cell.py)
# ---------------------------------------------------------------------------

cdef class _PCodeFrame:
    # Hot-path registers: C arrays, no boxing
    cdef uint64_t regs_arr[REGS_ARR_SIZE]
    cdef uint8_t  regs_sz [REGS_ARR_SIZE]
    cdef uint8_t  regs_set[REGS_ARR_SIZE]
    # Dirty-slot tracker: only zero written slots in clear() instead of scanning all 1104
    cdef int dirty[48]   # offsets of written regs_arr slots (48 > max flags+regs written)
    cdef int dirty_count
    # Compact unique-space array: indices 0..31 for ≤8-byte unique varnodes.
    cdef uint64_t uniq_arr[32]
    cdef uint8_t  uniq_set[32]  # which slots are written
    # High-half companion for 16-byte (128-bit) unique varnodes.
    # uniq_arr[slot] holds the low 64 bits; uniq_hi[slot] holds bits 64..127.
    # Only populated for instructions with 16-byte pcode intermediates
    # (CQO/INT_SEXT, MUL/IMUL widening, DIV/IDIV 128-bit dividend).
    cdef uint64_t uniq_hi[32]
    cdef uint8_t  uniq_hi_set[32]
    # Seeded/protected unique slots (intermediate-taint materialization): a
    # protected slot holds a caller-supplied value that a re-executed defining
    # op must NOT clobber (see seed_uniq / write_d).
    cdef uint8_t  uniq_protected[32]
    # Cold fallback for offsets >= REGS_ARR_SIZE
    cdef public dict regs
    cdef public dict reg_sizes
    # Fallback uniq dict (unused after compact-array migration, kept for safety)
    cdef public dict mem
    cdef public object _arch  # set by _load for CBRANCH PC lookup

    def __init__(self):
        cdef int i
        for i in range(REGS_ARR_SIZE):
            self.regs_set[i] = 0
        for i in range(32):
            self.uniq_set[i] = 0
            self.uniq_hi_set[i] = 0
            self.uniq_protected[i] = 0
        self.dirty_count = 0
        self.regs      = {}
        self.reg_sizes = {}
        self.mem       = {}
        self._arch     = None

    cdef inline void _write_reg(self, long off, int sz, uint64_t val) noexcept:
        cdef uint64_t masked = _mask64(val, sz)
        cdef uint64_t lo_mask
        cdef long invalidate_end, k
        if off >= 0 and off < REGS_ARR_SIZE:
            if not self.regs_set[off]:   # only record first write to each slot
                if self.dirty_count < 48:
                    self.dirty[self.dirty_count] = <int>off
                    self.dirty_count += 1
                self.regs_set[off] = 1
                self.regs_arr[off] = masked
                self.regs_sz [off] = <uint8_t>sz
                # New write — also invalidate sub-writes within range.
                invalidate_end = off + sz
                if invalidate_end > off + 8:
                    invalidate_end = off + 8
                k = off + 1
                while k < invalidate_end and k < REGS_ARR_SIZE:
                    if self.regs_set[k] and <int>self.regs_sz[k] < sz:
                        self.regs_set[k] = 0
                    k += 1
                return
            # Same-offset narrower write (e.g. mov al, bl into a slot that
            # currently holds full RAX): overlay the low `sz` bytes onto
            # the existing wider value rather than clobbering it.  See
            # cell_c.c::frame_write_reg for the full rationale.
            if <int>self.regs_sz[off] > sz:
                if sz >= 8:
                    lo_mask = 0xFFFFFFFFFFFFFFFFULL
                else:
                    lo_mask = ((<uint64_t>1) << (sz * 8)) - 1
                self.regs_arr[off] = (self.regs_arr[off] & ~lo_mask) | (masked & lo_mask)
                # regs_sz stays at the wider size.
            else:
                self.regs_arr[off] = masked
                self.regs_sz [off] = <uint8_t>sz
                # Wider/equal write — invalidate sub-writes within range.
                # See cell_c.c::frame_write_reg for the full rationale: a
                # wider write logically subsumes any per-byte overlays
                # within its range, so we drop them here to prevent a
                # later read from re-merging the stale sub-byte values.
                invalidate_end = off + sz
                if invalidate_end > off + 8:
                    invalidate_end = off + 8
                k = off + 1
                while k < invalidate_end and k < REGS_ARR_SIZE:
                    if self.regs_set[k] and <int>self.regs_sz[k] < sz:
                        self.regs_set[k] = 0
                    k += 1
        else:
            self.regs[off]      = masked
            self.reg_sizes[off] = sz

    cdef inline uint64_t _read_reg(self, long off, int sz) noexcept:
        cdef long     k, byte_off, end_off
        cdef int      k_sz
        cdef uint64_t base, sub_val, sub_mask, lane_mask
        cdef object   kv, v
        cdef uint64_t uv

        if off >= 0 and off < REGS_ARR_SIZE:
            # Step 1 — establish the base value of this register slot.
            # If this exact slot was written, that's the base.  Otherwise look
            # backwards for a parent register that contains this offset (e.g.
            # reading AH after writing only RAX).  Otherwise base is zero.
            base = 0
            if self.regs_set[off]:
                base = self.regs_arr[off]
            else:
                k = off - 1
                while k >= 0 and off - k <= 8:
                    if self.regs_set[k] and k + <long>self.regs_sz[k] > off:
                        byte_off = off - k
                        base = self.regs_arr[k] >> (byte_off * 8)
                        break
                    k -= 1

            # Step 2 — merge in any sub-register writes whose range falls
            # inside our read range.  Critical for x86 partial-register
            # writes like `mov ah, bh`: after the COPY writes byte 1
            # (AH) we read RAX (offset 0, size 8) and must overlay the
            # written AH byte onto the original RAX value.  Without this
            # merge, the read returns the pre-write parent value alone
            # and the partial write is silently lost.
            #
            # We only overlay sub-writes whose start offset is within 8
            # bytes of `off` — beyond that, the overlay would shift past
            # the 64-bit width of `base` (uint64_t) and the SHL by ≥64
            # is undefined / masked-mod-64 on x86-64, which yields the
            # wrong lane_mask and zeroes the low bits.  This guard limits
            # overlay to the representable low-8-byte window, which is
            # all the GP partial-write fix needs.  XMM/YMM-wide reads are
            # handled in separate slots (XMM<n>_LO / XMM<n>_HI) by the
            # engine.
            end_off = off + sz
            k = off + 1
            while k < end_off and k < REGS_ARR_SIZE and (k - off) < 8:
                if self.regs_set[k]:
                    k_sz = <int>self.regs_sz[k]
                    if k_sz <= 0:
                        k += 1
                        continue
                    byte_off = k - off
                    if k_sz >= 8:
                        sub_mask = 0xFFFFFFFFFFFFFFFFULL
                    else:
                        sub_mask = ((<uint64_t>1) << (k_sz * 8)) - 1
                    sub_val = self.regs_arr[k] & sub_mask
                    lane_mask = sub_mask << (byte_off * 8)
                    base = (base & ~lane_mask) | (sub_val << (byte_off * 8))
                    k += k_sz
                else:
                    k += 1

            return _mask64(base, sz)

        # Cold path: dict fallback
        v = self.regs.get(off)
        if v is not None:
            uv = <uint64_t>(v & 0xFFFFFFFFFFFFFFFF)
            return _mask64(uv, sz)
        for k, kv in self.reg_sizes.items():
            k_sz = <int>kv
            if k <= off < k + k_sz:
                v = self.regs.get(k)
                if v is not None:
                    byte_off = off - k
                    uv = <uint64_t>(v & 0xFFFFFFFFFFFFFFFF)
                    return _mask64(uv >> (byte_off * 8), sz)
        return 0

    cdef inline void _write_mem(self, uint64_t addr, uint64_t val, int size) noexcept:
        cdef int i
        val = _mask64(val, size)
        for i in range(size):
            self.mem[addr + i] = (val >> (i * 8)) & 0xFF

    cdef inline uint64_t _read_mem(self, uint64_t addr, int size) noexcept:
        cdef uint64_t result = 0
        cdef int i
        cdef object b
        for i in range(size):
            b = self.mem.get(addr + i)
            if b is not None:
                result |= (<uint64_t><int>b) << (i * 8)
        return _mask64(result, size)

    cdef inline void seed_uniq(self, int slot, uint64_t val) noexcept:
        """Materialize an intermediate: set its value, mark it live and protected
        so a re-executed defining op will not clobber the caller-supplied value."""
        if slot >= 0 and slot < 32:
            self.uniq_arr[slot] = val
            self.uniq_set[slot] = 1
            self.uniq_protected[slot] = 1

    cdef inline void clear(self) noexcept:
        cdef int i
        # Only zero the slots that were actually written (dirty list vs scanning all 1104)
        for i in range(self.dirty_count):
            self.regs_set[self.dirty[i]] = 0
        self.dirty_count = 0
        # Clear compact unique array (both low and high halves) + protection.
        for i in range(32):
            if self.uniq_set[i]:
                self.uniq_set[i] = 0
            if self.uniq_hi_set[i]:
                self.uniq_hi_set[i] = 0
            if self.uniq_protected[i]:
                self.uniq_protected[i] = 0
        if self.regs:
            self.regs.clear()
        if self.reg_sizes:
            self.reg_sizes.clear()
        self.mem.clear()

    # ------------------------------------------------------------------
    # Fast read/write using pre-decoded space IDs (no string comparison)
    # ------------------------------------------------------------------

    cdef inline uint64_t read_d(self, int sp, unsigned long off, int sz) noexcept:
        """Read a varnode given pre-decoded (space_id, offset, size)."""
        if sp == SP_CONST:
            return _mask64(<uint64_t>off, sz)
        if sp == SP_REGISTER:
            return self._read_reg(off, sz)
        if sp == SP_UNIQUE:
            # off is now a compact index (0..31) — direct C array lookup, no dict
            if off < 32 and self.uniq_set[off]:
                return _mask64(self.uniq_arr[off], sz)
            return 0
        if sp == SP_RAM:
            return self._read_mem(off, sz)
        return 0

    cdef inline void write_d(self, int sp, unsigned long off, int sz, uint64_t val) noexcept:
        """Write a varnode given pre-decoded (space_id, offset, size)."""
        val = _mask64(val, sz)
        if sp == SP_REGISTER:
            self._write_reg(off, sz, val)
        elif sp == SP_UNIQUE:
            # off is a compact index (0..31) — direct C array write, no dict.
            # A protected (seeded) slot keeps its materialized value: the
            # defining op that would overwrite it is a no-op in a seeded run.
            if off < 32 and not self.uniq_protected[off]:
                self.uniq_arr[off] = val
                self.uniq_set[off] = 1
        elif sp == SP_RAM:
            self._write_mem(off, val, sz)


# ---------------------------------------------------------------------------
# Core: execute a pre-decoded op list on a frame
# ---------------------------------------------------------------------------

cdef void _execute_decoded(
    _PCodeFrame frame,
    DecodedOps decoded,
    int start_pc = 0,
) except *:
    """
    Execute pre-decoded ops on frame using the C struct buffer, from start_pc
    to the end.  No Python tuple unpacking — all field access is direct C struct
    reads.  Raises PCodeFallbackNeeded if any op requires Unicorn.

    start_pc > 0 runs only a suffix of the op list: used by seeded partial
    re-execution (intermediate-taint materialization).  The caller MUST have
    seeded every unique varnode that ops in [start_pc, n_ops) read but whose
    defining op lies before start_pc; otherwise those reads see zero.
    """
    cdef int          oid, o_sp, o_sz, callother_out, n_ins
    cdef unsigned long o_off
    cdef int          i0_sp, i0_sz
    cdef unsigned long i0_off
    cdef int          i1_sp, i1_sz
    cdef unsigned long i1_off
    cdef int          i2_sp, i2_sz
    cdef unsigned long i2_off
    cdef uint64_t  a, b, c, result, u_result, dest, cond
    cdef int64_t   sa, sb, sresult, rel
    cdef int       sz, bits, i, pc, new_pc
    cdef PCodeOp*  op
    cdef PCodeOp*  ops_base
    cdef int       n_ops
    cdef bint      skip_remaining = 0   # set by an internal CBRANCH (CMOVxx skip)
    cdef uint64_t  next_instr_addr = decoded.next_instr_addr
    cdef uint64_t  imark_addr      = decoded.imark_addr
    cdef dict      imark_to_pc     = decoded.imark_to_pc
    cdef object    _pc_obj
    cdef object    _wide_val   # Python int (arbitrary precision) for 128-bit intermediates
    # Bound back-edges to keep wild differential-evaluator inputs (eg ECX
    # under V|T polarity could be huge) from running forever.  256 covers
    # every realistic ``rep stosb`` / ``rep movsb`` case in the benchmark
    # suite (bytes copied is bounded by stack scratch size, ≤ 64 in
    # practice).  Exceeding the budget triggers a Unicorn fallback —
    # which is also bounded by Unicorn's guest-fuel mechanism.
    cdef int       loop_iters       = 0
    cdef int       MAX_LOOP_ITERS   = 256

    # Hoist out of loop: one Python object access total, then pure C
    ops_base = decoded.buf
    n_ops    = decoded.n_ops
    pc = start_pc

    while pc < n_ops:
        op = ops_base + pc                    # pure C pointer arithmetic
        oid            = op.oid
        # CMOVxx skip: an internal CBRANCH (dest == next x86 instruction)
        # branched over the remaining ops in this cell.  Treat them as no-ops.
        # IMARK is allowed through (it has no semantic effect, and pcode never
        # contains another IMARK after the first).
        if skip_remaining and oid != OP_IMARK:
            pc += 1
            continue
        o_sp           = op.o_sp
        o_off          = op.o_off
        o_sz           = op.o_sz
        callother_out  = op.callother_out
        n_ins          = op.n_ins
        i0_sp          = op.i0_sp;  i0_off = op.i0_off;  i0_sz = op.i0_sz
        i1_sp          = op.i1_sp;  i1_off = op.i1_off;  i1_sz = op.i1_sz
        i2_sp          = op.i2_sp;  i2_off = op.i2_off;  i2_sz = op.i2_sz

        # ── Wide-op path: 16-byte (128-bit) unique intermediates ────────
        # Pcode lifts CQO, widening MUL/IMUL, DIV/IDIV with 16-byte unique
        # varnodes to represent 128-bit values.  The C-typed hot path below
        # only manipulates uint64_t, so we intercept these ops here.
        #
        # _wide_val is a cdef object (arbitrary-precision Python int) that
        # carries the full 128 bits without truncation.  Storage uses the
        # existing uniq_arr[slot] for the low 64 bits and the new
        # uniq_hi[slot] for the high 64 bits.
        #
        # fast-path guard: o_sz <= 8 AND NOT a wide-source SUBPIECE.
        # This is False for >99.9% of ops → zero overhead on the hot path.
        if o_sz > 8 or (oid == OP_SUBPIECE and i0_sz > 8):

            if oid == OP_INT_SEXT and o_sp == SP_UNIQUE and o_sz > 8:
                # Sign-extend source (≤8 bytes) to 16 bytes.
                a = frame.read_d(i0_sp, i0_off, i0_sz)
                if (a >> (i0_sz * 8 - 1)) & 1:
                    # Sign bit set: fill upper bits with 1s
                    _wide_val = a | ((((<object>1) << (o_sz * 8 - i0_sz * 8)) - 1) << (i0_sz * 8))
                else:
                    _wide_val = <object>a
                if o_off < 32:
                    frame.uniq_arr[o_off]    = _wide_val & 0xFFFFFFFFFFFFFFFF
                    frame.uniq_set[o_off]    = 1
                    frame.uniq_hi[o_off]     = (_wide_val >> 64) & 0xFFFFFFFFFFFFFFFF
                    frame.uniq_hi_set[o_off] = 1
                pc += 1
                continue

            if oid == OP_INT_ZEXT and o_sp == SP_UNIQUE and o_sz > 8:
                # Zero-extend: high half is always 0.
                a = frame.read_d(i0_sp, i0_off, i0_sz)
                if o_off < 32:
                    frame.uniq_arr[o_off]    = a
                    frame.uniq_set[o_off]    = 1
                    frame.uniq_hi[o_off]     = 0
                    frame.uniq_hi_set[o_off] = 1
                pc += 1
                continue

            if oid == OP_SUBPIECE and i0_sz > 8:
                # Extract o_sz bytes from a 128-bit unique source.
                b = frame.read_d(i1_sp, i1_off, i1_sz)  # byte offset (0 or 8)
                if b >= 8:
                    # Want high half: read uniq_hi[slot]
                    a = frame.uniq_hi[i0_off] if (i0_off < 32 and frame.uniq_hi_set[i0_off]) else 0
                    a = a >> ((b - 8) * 8)
                else:
                    # Want low half (or straddle): low half is uniq_arr[slot]
                    a = frame.uniq_arr[i0_off] if (i0_off < 32 and frame.uniq_set[i0_off]) else 0
                    a = a >> (b * 8)
                if o_sp != NO_OUT_SPACE:
                    frame.write_d(o_sp, o_off, o_sz, _mask64(a, o_sz))
                pc += 1
                continue

            if oid == OP_INT_LEFT and o_sp == SP_UNIQUE and o_sz > 8:
                # 128-bit left shift.  Build Python int from both halves.
                _wide_val = (<object>(frame.uniq_arr[i0_off] if (i0_off < 32 and frame.uniq_set[i0_off]) else 0))
                _wide_val = _wide_val | ((<object>(frame.uniq_hi[i0_off] if (i0_off < 32 and frame.uniq_hi_set[i0_off]) else 0)) << 64)
                b = frame.read_d(i1_sp, i1_off, i1_sz)
                _wide_val = (_wide_val << b) & (((<object>1) << (o_sz * 8)) - 1)
                if o_off < 32:
                    frame.uniq_arr[o_off]    = _wide_val & 0xFFFFFFFFFFFFFFFF
                    frame.uniq_set[o_off]    = 1
                    frame.uniq_hi[o_off]     = (_wide_val >> 64) & 0xFFFFFFFFFFFFFFFF
                    frame.uniq_hi_set[o_off] = 1
                pc += 1
                continue

            if oid == OP_INT_OR and o_sp == SP_UNIQUE and o_sz > 8:
                # 128-bit OR.
                if o_off < 32:
                    frame.uniq_arr[o_off] = (frame.uniq_arr[i0_off] if (i0_off < 32 and frame.uniq_set[i0_off]) else 0) | \
                                            (frame.uniq_arr[i1_off] if (i1_off < 32 and frame.uniq_set[i1_off]) else 0)
                    frame.uniq_set[o_off] = 1
                    frame.uniq_hi[o_off]  = (frame.uniq_hi[i0_off] if (i0_off < 32 and frame.uniq_hi_set[i0_off]) else 0) | \
                                            (frame.uniq_hi[i1_off] if (i1_off < 32 and frame.uniq_hi_set[i1_off]) else 0)
                    frame.uniq_hi_set[o_off] = 1
                pc += 1
                continue

            if oid == OP_INT_AND and o_sp == SP_UNIQUE and o_sz > 8:
                if o_off < 32:
                    frame.uniq_arr[o_off] = (frame.uniq_arr[i0_off] if (i0_off < 32 and frame.uniq_set[i0_off]) else 0) & \
                                            (frame.uniq_arr[i1_off] if (i1_off < 32 and frame.uniq_set[i1_off]) else 0)
                    frame.uniq_set[o_off] = 1
                    frame.uniq_hi[o_off]  = (frame.uniq_hi[i0_off] if (i0_off < 32 and frame.uniq_hi_set[i0_off]) else 0) & \
                                            (frame.uniq_hi[i1_off] if (i1_off < 32 and frame.uniq_hi_set[i1_off]) else 0)
                    frame.uniq_hi_set[o_off] = 1
                pc += 1
                continue

            if oid == OP_INT_XOR and o_sp == SP_UNIQUE and o_sz > 8:
                if o_off < 32:
                    frame.uniq_arr[o_off] = (frame.uniq_arr[i0_off] if (i0_off < 32 and frame.uniq_set[i0_off]) else 0) ^ \
                                            (frame.uniq_arr[i1_off] if (i1_off < 32 and frame.uniq_set[i1_off]) else 0)
                    frame.uniq_set[o_off] = 1
                    frame.uniq_hi[o_off]  = (frame.uniq_hi[i0_off] if (i0_off < 32 and frame.uniq_hi_set[i0_off]) else 0) ^ \
                                            (frame.uniq_hi[i1_off] if (i1_off < 32 and frame.uniq_hi_set[i1_off]) else 0)
                    frame.uniq_hi_set[o_off] = 1
                pc += 1
                continue

            if oid == OP_COPY and o_sp == SP_UNIQUE and o_sz > 8:
                if o_off < 32:
                    frame.uniq_arr[o_off]    = frame.uniq_arr[i0_off] if (i0_off < 32 and frame.uniq_set[i0_off]) else 0
                    frame.uniq_set[o_off]    = 1
                    frame.uniq_hi[o_off]     = frame.uniq_hi[i0_off] if (i0_off < 32 and frame.uniq_hi_set[i0_off]) else 0
                    frame.uniq_hi_set[o_off] = 1
                pc += 1
                continue

            if oid == OP_INT_MULT and o_sp == SP_UNIQUE and o_sz > 8:
                # 128-bit multiply (widening MUL/IMUL).
                _wide_val = (<object>(frame.uniq_arr[i0_off] if (i0_off < 32 and frame.uniq_set[i0_off]) else 0))
                _wide_val = _wide_val | ((<object>(frame.uniq_hi[i0_off] if (i0_off < 32 and frame.uniq_hi_set[i0_off]) else 0)) << 64)
                _wide_val = _wide_val * (<object>(frame.uniq_arr[i1_off] if (i1_off < 32 and frame.uniq_set[i1_off]) else 0) | \
                                        ((<object>(frame.uniq_hi[i1_off] if (i1_off < 32 and frame.uniq_hi_set[i1_off]) else 0)) << 64))
                _wide_val = _wide_val & (((<object>1) << (o_sz * 8)) - 1)
                if o_off < 32:
                    frame.uniq_arr[o_off]    = _wide_val & 0xFFFFFFFFFFFFFFFF
                    frame.uniq_set[o_off]    = 1
                    frame.uniq_hi[o_off]     = (_wide_val >> 64) & 0xFFFFFFFFFFFFFFFF
                    frame.uniq_hi_set[o_off] = 1
                pc += 1
                continue

            if oid == OP_INT_DIV and o_sz <= 8:
                # 128-bit / 64-bit unsigned divide (DIV quotient).
                _wide_val = (<object>(frame.uniq_arr[i0_off] if (i0_off < 32 and frame.uniq_set[i0_off]) else 0))
                _wide_val = _wide_val | ((<object>(frame.uniq_hi[i0_off] if (i0_off < 32 and frame.uniq_hi_set[i0_off]) else 0)) << 64)
                b = frame.read_d(i1_sp, i1_off, i1_sz)
                if b == 0:
                    frame.write_d(o_sp, o_off, o_sz, 0)
                else:
                    frame.write_d(o_sp, o_off, o_sz, <uint64_t>(_wide_val // <object>b))
                pc += 1
                continue

            if oid == OP_INT_REM and o_sz <= 8:
                # 128-bit % 64-bit unsigned remainder (DIV remainder).
                _wide_val = (<object>(frame.uniq_arr[i0_off] if (i0_off < 32 and frame.uniq_set[i0_off]) else 0))
                _wide_val = _wide_val | ((<object>(frame.uniq_hi[i0_off] if (i0_off < 32 and frame.uniq_hi_set[i0_off]) else 0)) << 64)
                b = frame.read_d(i1_sp, i1_off, i1_sz)
                if b == 0:
                    frame.write_d(o_sp, o_off, o_sz, 0)
                else:
                    frame.write_d(o_sp, o_off, o_sz, <uint64_t>(_wide_val % <object>b))
                pc += 1
                continue

            if oid == OP_INT_SDIV and o_sz <= 8:
                # 128-bit signed / 64-bit signed divide (IDIV quotient).
                _wide_val = (<object>(frame.uniq_arr[i0_off] if (i0_off < 32 and frame.uniq_set[i0_off]) else 0))
                _wide_val = _wide_val | ((<object>(frame.uniq_hi[i0_off] if (i0_off < 32 and frame.uniq_hi_set[i0_off]) else 0)) << 64)
                if _wide_val >> (i0_sz * 8 - 1):  # sign bit set
                    _wide_val = _wide_val - ((<object>1) << (i0_sz * 8))
                _wide_val = _wide_val // <object><int64_t>_signed64(frame.read_d(i1_sp, i1_off, i1_sz), i1_sz)
                frame.write_d(o_sp, o_off, o_sz, <uint64_t>(_wide_val & 0xFFFFFFFFFFFFFFFF))
                pc += 1
                continue

            if oid == OP_INT_SREM and o_sz <= 8:
                # 128-bit signed % 64-bit signed remainder (IDIV remainder).
                _wide_val = (<object>(frame.uniq_arr[i0_off] if (i0_off < 32 and frame.uniq_set[i0_off]) else 0))
                _wide_val = _wide_val | ((<object>(frame.uniq_hi[i0_off] if (i0_off < 32 and frame.uniq_hi_set[i0_off]) else 0)) << 64)
                if _wide_val >> (i0_sz * 8 - 1):
                    _wide_val = _wide_val - ((<object>1) << (i0_sz * 8))
                _wide_val = _wide_val % <object><int64_t>_signed64(frame.read_d(i1_sp, i1_off, i1_sz), i1_sz)
                frame.write_d(o_sp, o_off, o_sz, <uint64_t>(_wide_val & 0xFFFFFFFFFFFFFFFF))
                pc += 1
                continue

            # Unrecognised wide op — fall through to normal (narrow) dispatch.

        # ── Hot path: most frequent pcode ops first ─────────────────
        if oid == OP_IMARK or oid == OP_RETURN or oid == OP_CALL:
            pass
        elif oid == OP_BRANCH:
            # Three accepted BRANCH patterns (others fall back during predecode):
            #   (1) const-space target: pcode-relative signed offset.
            #   (2) ram-space target == any in-sequence IMARK: jump to that op.
            #   (3) ram-space target == next_instr_addr: forward skip to end.
            if i0_sp == SP_CONST:
                # Sign-extend offset based on size (i0_sz bytes).
                rel = _signed64(<uint64_t>i0_off, i0_sz)
                pc = pc + <int>rel
                if pc < 0 or pc > n_ops:
                    raise PCodeFallbackNeeded('BRANCH out of pcode range')
                if rel <= 0:
                    loop_iters += 1
                    if loop_iters > MAX_LOOP_ITERS:
                        raise PCodeFallbackNeeded('iteration budget exceeded (const BRANCH)')
                continue
            if i0_sp == SP_RAM:
                dest = <uint64_t>i0_off
                if dest == next_instr_addr:
                    pc = n_ops  # exit the cell
                    continue
                _pc_obj = imark_to_pc.get(dest)
                if _pc_obj is not None:
                    new_pc = <int>_pc_obj
                    # Backward jump => loop iteration; cap to keep wild
                    # taint-polarity inputs from looping forever.
                    if new_pc <= pc:
                        loop_iters += 1
                        if loop_iters > MAX_LOOP_ITERS:
                            raise PCodeFallbackNeeded('iteration budget exceeded (rep loop)')
                    pc = new_pc
                    continue
                raise PCodeFallbackNeeded('BRANCH to unsupported ram target')
            raise PCodeFallbackNeeded('BRANCH to unsupported space')
        elif oid == OP_INT_XOR:
            if o_sp != NO_OUT_SPACE:
                frame.write_d(o_sp, o_off, o_sz,
                    frame.read_d(i0_sp, i0_off, i0_sz) ^ frame.read_d(i1_sp, i1_off, i1_sz))
        elif oid == OP_INT_AND:
            if o_sp != NO_OUT_SPACE:
                frame.write_d(o_sp, o_off, o_sz,
                    frame.read_d(i0_sp, i0_off, i0_sz) & frame.read_d(i1_sp, i1_off, i1_sz))
        elif oid == OP_INT_OR:
            if o_sp != NO_OUT_SPACE:
                frame.write_d(o_sp, o_off, o_sz,
                    frame.read_d(i0_sp, i0_off, i0_sz) | frame.read_d(i1_sp, i1_off, i1_sz))
        elif oid == OP_INT_ADD:
            if o_sp != NO_OUT_SPACE:
                frame.write_d(o_sp, o_off, o_sz,
                    frame.read_d(i0_sp, i0_off, i0_sz) + frame.read_d(i1_sp, i1_off, i1_sz))
        elif oid == OP_INT_SUB:
            if o_sp != NO_OUT_SPACE:
                frame.write_d(o_sp, o_off, o_sz,
                    frame.read_d(i0_sp, i0_off, i0_sz) - frame.read_d(i1_sp, i1_off, i1_sz))
        elif oid == OP_COPY or oid == OP_INT_ZEXT or oid == OP_INT_TRUNC or oid == OP_CAST:
            if o_sp != NO_OUT_SPACE:
                frame.write_d(o_sp, o_off, o_sz, frame.read_d(i0_sp, i0_off, i0_sz))
        elif oid == OP_INT_EQUAL:
            if o_sp != NO_OUT_SPACE:
                frame.write_d(o_sp, o_off, o_sz,
                    1 if frame.read_d(i0_sp, i0_off, i0_sz) == frame.read_d(i1_sp, i1_off, i1_sz) else 0)
        elif oid == OP_INT_NOTEQUAL:
            if o_sp != NO_OUT_SPACE:
                frame.write_d(o_sp, o_off, o_sz,
                    0 if frame.read_d(i0_sp, i0_off, i0_sz) == frame.read_d(i1_sp, i1_off, i1_sz) else 1)
        elif oid == OP_INT_LESS:
            if o_sp != NO_OUT_SPACE:
                frame.write_d(o_sp, o_off, o_sz,
                    1 if frame.read_d(i0_sp, i0_off, i0_sz) < frame.read_d(i1_sp, i1_off, i1_sz) else 0)
        elif oid == OP_INT_RIGHT:
            # P-code spec (Ghidra pcodedescription.html): "If input1 is larger
            # than the number of bits in output, the result is zero."  The
            # output-width threshold is `o_sz * 8` bits (input0/output have the
            # same size per the spec).  Using `& 0x3F` here was an x86-64-isa
            # mask that wrongly turned shifts ≥ width into 0-shifts: e.g. for
            # SHLD's lifted `RBX >> (0x40 - cl)` with cl=0, mask collapsed 64
            # to 0, so the cell returned RBX instead of 0 and the differential
            # lost the RAX-side taint (id=3252).
            if o_sp != NO_OUT_SPACE:
                b = frame.read_d(i1_sp, i1_off, i1_sz)
                if b >= <uint64_t>(o_sz * 8):
                    frame.write_d(o_sp, o_off, o_sz, 0)
                else:
                    frame.write_d(o_sp, o_off, o_sz, frame.read_d(i0_sp, i0_off, i0_sz) >> b)
        elif oid == OP_INT_LEFT:
            # Same rule as INT_RIGHT: shift ≥ output-width-in-bits → 0.
            # Without this, BEXTR's length-mask `(1 << length) - 1` with
            # length=92 wrapped to `(1 << 28) - 1` and dropped any source bit
            # that landed at result position ≥ 28 (id=5337).
            if o_sp != NO_OUT_SPACE:
                b = frame.read_d(i1_sp, i1_off, i1_sz)
                if b >= <uint64_t>(o_sz * 8):
                    frame.write_d(o_sp, o_off, o_sz, 0)
                else:
                    frame.write_d(o_sp, o_off, o_sz, frame.read_d(i0_sp, i0_off, i0_sz) << b)
        # ── Less frequent ops ────────────────────────────────────────
        elif oid == OP_CBRANCH:
            cond = frame.read_d(i1_sp, i1_off, i1_sz)
            # CBRANCH semantics: if cond is true, go to dest; else fall through.
            # Accepted dest patterns:
            #   (1) const-space: pcode-relative signed offset (BSF/BSR).
            #   (2) ram-space == next_instr_addr: forward skip-to-end (CMOVxx,
            #       rep-loop exit when ECX==0).
            #   (3) ram-space == any in-sequence IMARK address: in-cell jump
            #       (forward or backward).  Backward => loop iteration.
            #   (4) ram-space, other: real x86 conditional jump out of cell
            #       (JL/JE/JNE) — write RIP so the differential XOR sees the
            #       taken/not-taken decision as RIP taint.
            if i0_sp == SP_CONST:
                if cond:
                    rel = _signed64(<uint64_t>i0_off, i0_sz)
                    pc = pc + <int>rel
                    if pc < 0 or pc > n_ops:
                        raise PCodeFallbackNeeded('CBRANCH out of pcode range')
                    if rel <= 0:
                        loop_iters += 1
                        if loop_iters > MAX_LOOP_ITERS:
                            raise PCodeFallbackNeeded('iteration budget exceeded (const CBRANCH)')
                    continue
                # not taken: fall through (pc += 1 at the end)
            elif i0_sp == SP_RAM:
                dest = <uint64_t>i0_off
                if dest == next_instr_addr:
                    if cond:
                        skip_remaining = 1
                else:
                    _pc_obj = imark_to_pc.get(dest)
                    if _pc_obj is not None:
                        if cond:
                            new_pc = <int>_pc_obj
                            if new_pc <= pc:
                                loop_iters += 1
                                if loop_iters > MAX_LOOP_ITERS:
                                    raise PCodeFallbackNeeded('iteration budget exceeded (CBRANCH loop)')
                            pc = new_pc
                            continue
                        # not taken: fall through
                    else:
                        # Pattern (4): real x86 conditional branch.
                        result = dest if cond else dest + 1
                        pc_tup = _ARCH_PC.get(str(frame._arch))
                        if pc_tup is not None:
                            frame.write_d(SP_REGISTER, pc_tup[0], pc_tup[1], result)
            else:
                raise PCodeFallbackNeeded('CBRANCH to unsupported space')
        elif oid == OP_BRANCHIND or oid == OP_CALLIND:
            # Write the jump target into the architectural PC register so that
            # SimCell read-back gives the correct concrete value for the
            # differential: SimH and SimL will differ exactly when the target
            # varnode (i0) depends on a tainted register, which is what
            # map_outputs_to_targets expects to see via the AVALANCHE expression.
            #
            # Without this write PC stays at its pristine value (0), making
            # SimH = SimL = 0 and collapsing the differential to 0.  Then only
            # the AVALANCHE floor term (T_input) would cover taint, which is
            # sound but loses the bit-precise differential for BRANCHIND targets
            # that are simple register reads (e.g. JMP rax, BR x0).
            #
            # Note: all OTHER register writes for the instruction (link-register
            # save, RSP adjustment) are already emitted as pcode ops that precede
            # this terminal op and were executed above.
            pc_tup = _ARCH_PC.get(str(frame._arch))
            if pc_tup is not None:
                frame.write_d(SP_REGISTER, pc_tup[0], pc_tup[1],
                              frame.read_d(i0_sp, i0_off, i0_sz))
        elif oid == OP_CALLOTHER:
            if callother_out:
                raise PCodeFallbackNeeded('CALLOTHER with output')
        elif oid == OP_FLOAT_ANY or oid == OP_TRUNC_FLOAT:
            raise PCodeFallbackNeeded('Float opcode')
        elif oid == OP_UNKNOWN:
            raise PCodeFallbackNeeded('Unknown opcode')
        elif oid == OP_UNIMPLEMENTED or oid == OP_SEGMENT or oid == OP_CPOOLREF or oid == OP_NEW or oid == OP_INSERT or oid == OP_EXTRACT:
            pass


        elif oid == OP_LOAD:
            if o_sp != NO_OUT_SPACE:
                a = frame.read_d(i1_sp, i1_off, i1_sz)
                frame.write_d(o_sp, o_off, o_sz, frame._read_mem(a, o_sz))

        elif oid == OP_STORE:
            a = frame.read_d(i1_sp, i1_off, i1_sz)
            frame._write_mem(a, frame.read_d(i2_sp, i2_off, i2_sz), i2_sz)

        elif oid == OP_MULTIEQUAL or oid == OP_INDIRECT:
            if o_sp != NO_OUT_SPACE and n_ins > 0:
                frame.write_d(o_sp, o_off, o_sz, frame.read_d(i0_sp, i0_off, i0_sz))


        elif oid == OP_INT_MULT:
            if o_sp != NO_OUT_SPACE:
                frame.write_d(o_sp, o_off, o_sz,
                    frame.read_d(i0_sp, i0_off, i0_sz) * frame.read_d(i1_sp, i1_off, i1_sz))

        elif oid == OP_INT_DIV:
            if o_sp != NO_OUT_SPACE:
                b = frame.read_d(i1_sp, i1_off, i1_sz)
                frame.write_d(o_sp, o_off, o_sz,
                    0 if b == 0 else frame.read_d(i0_sp, i0_off, i0_sz) // b)

        elif oid == OP_INT_SDIV:
            if o_sp != NO_OUT_SPACE:
                sz = i0_sz
                sa = _signed64(frame.read_d(i0_sp, i0_off, i0_sz), sz)
                sb = _signed64(frame.read_d(i1_sp, i1_off, i1_sz), sz)
                if sb == 0:
                    frame.write_d(o_sp, o_off, o_sz, 0)
                else:
                    sresult = sa // sb
                    if (sa ^ sb) < 0 and sresult * sb != sa:
                        sresult += 1
                    frame.write_d(o_sp, o_off, o_sz, <uint64_t>sresult)

        elif oid == OP_INT_REM:
            if o_sp != NO_OUT_SPACE:
                b = frame.read_d(i1_sp, i1_off, i1_sz)
                frame.write_d(o_sp, o_off, o_sz,
                    0 if b == 0 else frame.read_d(i0_sp, i0_off, i0_sz) % b)

        elif oid == OP_INT_SREM:
            if o_sp != NO_OUT_SPACE:
                sz = i0_sz
                sa = _signed64(frame.read_d(i0_sp, i0_off, i0_sz), sz)
                sb = _signed64(frame.read_d(i1_sp, i1_off, i1_sz), sz)
                if sb == 0:
                    frame.write_d(o_sp, o_off, o_sz, 0)
                else:
                    sresult = sa - sb * (sa // sb)
                    frame.write_d(o_sp, o_off, o_sz, <uint64_t>sresult)

        elif oid == OP_INT_2COMP:
            if o_sp != NO_OUT_SPACE:
                frame.write_d(o_sp, o_off, o_sz,
                    <uint64_t>(-<int64_t>frame.read_d(i0_sp, i0_off, i0_sz)))

        elif oid == OP_INT_NEGATE:
            if o_sp != NO_OUT_SPACE:
                frame.write_d(o_sp, o_off, o_sz, ~frame.read_d(i0_sp, i0_off, i0_sz))



        elif oid == OP_INT_SRIGHT:
            # P-code spec: "If input1 is larger than the number of bits in
            # output, the result is zero or all 1-bits (-1), depending on the
            # original sign of input0."  Encode the all-1s case as a width-
            # mask so the result fits the output varnode.
            if o_sp != NO_OUT_SPACE:
                sz  = i0_sz
                sa  = _signed64(frame.read_d(i0_sp, i0_off, i0_sz), sz)
                b   = frame.read_d(i1_sp, i1_off, i1_sz)
                if b >= <uint64_t>(o_sz * 8):
                    if sa < 0:
                        # All 1-bits within the output width.
                        if o_sz >= 8:
                            frame.write_d(o_sp, o_off, o_sz, <uint64_t>0xFFFFFFFFFFFFFFFF)
                        else:
                            frame.write_d(o_sp, o_off, o_sz,
                                (<uint64_t>1 << (o_sz * 8)) - 1)
                    else:
                        frame.write_d(o_sp, o_off, o_sz, 0)
                else:
                    frame.write_d(o_sp, o_off, o_sz, <uint64_t>(sa >> b))


        elif oid == OP_INT_LESSEQUAL:
            if o_sp != NO_OUT_SPACE:
                frame.write_d(o_sp, o_off, o_sz,
                    1 if frame.read_d(i0_sp, i0_off, i0_sz) <= frame.read_d(i1_sp, i1_off, i1_sz) else 0)

        elif oid == OP_INT_SLESS:
            if o_sp != NO_OUT_SPACE:
                sz = i0_sz
                sa = _signed64(frame.read_d(i0_sp, i0_off, i0_sz), sz)
                sb = _signed64(frame.read_d(i1_sp, i1_off, i1_sz), sz)
                frame.write_d(o_sp, o_off, o_sz, 1 if sa < sb else 0)

        elif oid == OP_INT_SLESSEQUAL:
            if o_sp != NO_OUT_SPACE:
                sz = i0_sz
                sa = _signed64(frame.read_d(i0_sp, i0_off, i0_sz), sz)
                sb = _signed64(frame.read_d(i1_sp, i1_off, i1_sz), sz)
                frame.write_d(o_sp, o_off, o_sz, 1 if sa <= sb else 0)

        elif oid == OP_INT_CARRY:
            if o_sp != NO_OUT_SPACE:
                bits = i0_sz * 8
                a    = frame.read_d(i0_sp, i0_off, i0_sz)
                b    = frame.read_d(i1_sp, i1_off, i1_sz)
                if bits >= 64:
                    # uint64_t addition wraps mod 2^64; detect overflow by checking
                    # if the sum is smaller than either operand.
                    frame.write_d(o_sp, o_off, o_sz, 1 if (a + b) < a else 0)
                else:
                    frame.write_d(o_sp, o_off, o_sz,
                        1 if (a + b) >= (<uint64_t>1 << bits) else 0)

        elif oid == OP_INT_SCARRY:
            if o_sp != NO_OUT_SPACE:
                sz       = i0_sz
                sa       = _signed64(frame.read_d(i0_sp, i0_off, i0_sz), sz)
                sb       = _signed64(frame.read_d(i1_sp, i1_off, i1_sz), sz)
                # Unsigned add then re-sign: avoids C signed overflow UB.
                u_result = (<uint64_t>sa + <uint64_t>sb)
                sresult  = _signed64(u_result, sz)
                # Overflow: both inputs same sign, result has different sign.
                frame.write_d(o_sp, o_off, o_sz,
                    1 if ((sa < 0) == (sb < 0)) and ((sa < 0) != (sresult < 0)) else 0)

        elif oid == OP_INT_SBORROW:
            if o_sp != NO_OUT_SPACE:
                sz       = i0_sz
                sa       = _signed64(frame.read_d(i0_sp, i0_off, i0_sz), sz)
                sb       = _signed64(frame.read_d(i1_sp, i1_off, i1_sz), sz)
                # Unsigned sub then re-sign: avoids C signed overflow UB.
                u_result = (<uint64_t>sa - <uint64_t>sb)
                sresult  = _signed64(u_result, sz)
                # Overflow: inputs have different signs, result sign differs from dividend.
                frame.write_d(o_sp, o_off, o_sz,
                    1 if ((sa < 0) != (sb < 0)) and ((sa < 0) != (sresult < 0)) else 0)


        elif oid == OP_INT_SEXT:
            if o_sp != NO_OUT_SPACE:
                sz = i0_sz
                frame.write_d(o_sp, o_off, o_sz,
                    <uint64_t>_signed64(frame.read_d(i0_sp, i0_off, i0_sz), sz))


        elif oid == OP_POPCOUNT:
            if o_sp != NO_OUT_SPACE:
                a = frame.read_d(i0_sp, i0_off, i0_sz)
                result = 0
                while a:
                    a &= a - 1
                    result += 1
                frame.write_d(o_sp, o_off, o_sz, result)

        elif oid == OP_LZCOUNT:
            if o_sp != NO_OUT_SPACE:
                bits = i0_sz * 8
                a    = frame.read_d(i0_sp, i0_off, i0_sz)
                if a == 0:
                    frame.write_d(o_sp, o_off, o_sz, bits)
                else:
                    result = 0
                    while not (a & (<uint64_t>1 << (bits - 1))):
                        result += 1
                        a <<= 1
                    frame.write_d(o_sp, o_off, o_sz, result)

        elif oid == OP_PIECE:
            if o_sp != NO_OUT_SPACE:
                a = frame.read_d(i0_sp, i0_off, i0_sz)
                b = frame.read_d(i1_sp, i1_off, i1_sz)
                frame.write_d(o_sp, o_off, o_sz, (a << (i1_sz * 8)) | b)

        elif oid == OP_SUBPIECE:
            if o_sp != NO_OUT_SPACE:
                sz = <int>frame.read_d(i1_sp, i1_off, i1_sz)
                frame.write_d(o_sp, o_off, o_sz,
                    frame.read_d(i0_sp, i0_off, i0_sz) >> (sz * 8))

        elif oid == OP_PTRADD:
            if o_sp != NO_OUT_SPACE:
                a = frame.read_d(i0_sp, i0_off, i0_sz)
                b = frame.read_d(i1_sp, i1_off, i1_sz)
                c = frame.read_d(i2_sp, i2_off, i2_sz)
                frame.write_d(o_sp, o_off, o_sz, a + b * c)

        elif oid == OP_PTRSUB:
            if o_sp != NO_OUT_SPACE:
                frame.write_d(o_sp, o_off, o_sz,
                    frame.read_d(i0_sp, i0_off, i0_sz) - frame.read_d(i1_sp, i1_off, i1_sz))

        elif oid == OP_BOOL_AND:
            if o_sp != NO_OUT_SPACE:
                frame.write_d(o_sp, o_off, o_sz,
                    1 if (frame.read_d(i0_sp, i0_off, i0_sz) and frame.read_d(i1_sp, i1_off, i1_sz)) else 0)

        elif oid == OP_BOOL_OR:
            if o_sp != NO_OUT_SPACE:
                frame.write_d(o_sp, o_off, o_sz,
                    1 if (frame.read_d(i0_sp, i0_off, i0_sz) or frame.read_d(i1_sp, i1_off, i1_sz)) else 0)

        elif oid == OP_BOOL_XOR:
            if o_sp != NO_OUT_SPACE:
                a = frame.read_d(i0_sp, i0_off, i0_sz)
                b = frame.read_d(i1_sp, i1_off, i1_sz)
                frame.write_d(o_sp, o_off, o_sz, 1 if bool(a) ^ bool(b) else 0)

        elif oid == OP_BOOL_NEGATE:
            if o_sp != NO_OUT_SPACE:
                frame.write_d(o_sp, o_off, o_sz,
                    0 if frame.read_d(i0_sp, i0_off, i0_sz) else 1)

        # Default fall-through: advance to next pcode op.
        pc += 1

    # Note on end-of-cell RIP behavior:
    #
    # Unicorn's emu_start always advances RIP past the executed bytes.  For
    # the JL/JE pattern (Pattern (4) above), the CBRANCH handler already
    # writes a dest-dependent RIP so the differential XOR sees taken vs
    # not-taken as RIP taint.  For all other instructions the pcode
    # interpreter does not write RIP — the architectural register stays
    # whatever the caller loaded.  Static-rule generation handles
    # incrementing RIP through normal instruction sequencing.
    #
    # In particular, an in-cell loop (rep stosb/movsb, BSF/BSR) does NOT
    # produce RIP taint here — Unicorn's apparent RIP taint for these
    # instructions comes from a different layer of the rule generator,
    # not from per-cell evaluation, and is over-conservative.


# ---------------------------------------------------------------------------
# Register map cache
# ---------------------------------------------------------------------------

# Architecture-specific aliases: state_format name → Sleigh register name.
# Used when the state_format uses a friendly name (e.g. 'Z') but the Sleigh
# spec uses a different name for the same register (e.g. 'ZR' in ARM64).
# Must match the aliases in engine.py StateMapper.arm_aliases.
_ARCH_REG_ALIASES: dict[str, dict[str, str]] = {
    'ARM64': {'N': 'NG', 'Z': 'ZR', 'C': 'CY', 'V': 'OV'},
}


# PC register offset per arch — populated by _build_reg_maps on first call.
_ARCH_PC: dict = {}  # arch_str -> (reg_offset: long, reg_size: int)


@functools.lru_cache(maxsize=8)
def _build_reg_maps(arch):
    ctx = get_context(arch)
    offsets, sizes = {}, {}
    for name, vn in ctx.registers.items():
        key = name.upper()
        offsets[key] = vn.offset
        sizes[key]   = vn.size
    # Add friendly aliases so _read_output can resolve state_format names
    # that differ from the raw Sleigh register names (e.g. ARM64 'Z' → 'ZR').
    _arch_parts = str(arch).upper().split('.')
    arch_str = _arch_parts[len(_arch_parts) - 1]  # e.g. 'ARM64' from Architecture.ARM64
    for alias_str in ('ARM64', 'AMD64', 'X86'):
        if alias_str in arch_str:
            for friendly, sleigh in _ARCH_REG_ALIASES.get(alias_str, {}).items():
                if sleigh in offsets and friendly not in offsets:
                    offsets[friendly] = offsets[sleigh]
                    sizes[friendly]   = sizes[sleigh]
            break
    # XMM<n>_LO / XMM<n>_HI: split each 128-bit XMM register into two
    # 64-bit halves (LO at base offset, HI at base+8).  Mirrors the
    # wrapper.X64_FORMAT layout so the cell.pyx _load can populate the
    # P-code frame from XMM_LO/_HI keys in InstructionCellExpr inputs.
    if 'AMD64' in arch_str or 'X86' in arch_str:
        for i in range(16):
            full_key = f'XMM{i}'
            if full_key in offsets:
                base_off = offsets[full_key]
                offsets[f'XMM{i}_LO'] = base_off
                sizes[f'XMM{i}_LO']   = 8
                offsets[f'XMM{i}_HI'] = base_off + 8
                sizes[f'XMM{i}_HI']   = 8
    arch_key = str(arch)
    for _pc_name in ('RIP', 'EIP', 'PC'):
        if _pc_name in offsets:
            _ARCH_PC[arch_key] = (offsets[_pc_name], sizes[_pc_name])
            break

    return offsets, sizes


# ---------------------------------------------------------------------------
# PCodeCellEvaluator
# ---------------------------------------------------------------------------

cdef class PCodeCellEvaluator:
    """
    Native P-code differential evaluator (Cython, pre-decoded hot path).
    Identical public interface to cell.py PCodeCellEvaluator.
    """
    cdef object      arch
    cdef _PCodeFrame _frame_a
    cdef _PCodeFrame _frame_b
    cdef public dict _offsets
    cdef public dict _sizes
    cdef public int  native_calls
    cdef public int  fallback_calls

    def __init__(self, arch):
        self.arch           = arch
        self._frame_a       = _PCodeFrame()
        self._frame_b       = _PCodeFrame()
        self._offsets, self._sizes = _build_reg_maps(arch)
        self.native_calls   = 0
        self.fallback_calls = 0

    cdef void _load(self, _PCodeFrame frame, dict inputs):
        frame._arch = str(self.arch)  # for CBRANCH PC lookup
        cdef object   name, val, off_obj, sz_obj
        cdef str      key, body, head
        cdef long     off
        cdef uint64_t addr_u64
        cdef int      sz, size, sep, second_us
        cdef uint64_t v
        cdef int64_t  signed_off
        cdef list     deferred_mem

        frame.clear()

        # Two-pass: register/value writes FIRST so MEM_<reg>_<off>_<size>
        # entries (which need to look up <reg> from the frame) can resolve
        # the address register correctly. Memory writes are deferred and
        # executed after all register writes complete.
        deferred_mem = []

        for name, val in inputs.items():
            # Mask to 64-bit unsigned BEFORE casting: Python AND stays in Python
            # int domain (no overflow), then Cython casts the bounded value.
            v = <uint64_t>(val & 0xFFFFFFFFFFFFFFFF)
            # Fast prefix check — direct slice comparison is faster than
            # str.startswith(); benchmark-verified.
            if (<str>name)[:4] == 'MEM_':
                deferred_mem.append((<str>name, v))
            else:
                key     = (<str>name).upper()
                off_obj = self._offsets.get(key)
                if off_obj is not None:
                    off    = <long>off_obj
                    sz_obj = self._sizes.get(key)
                    sz     = <int>sz_obj if sz_obj is not None else 8
                    frame._write_reg(off, sz, _mask64(v, sz))

        # Pass 2: resolve MEM_<...> entries.
        # Format A:  MEM_<hex>_<size>           (static address)
        # Format B:  MEM_<reg>_<offset>_<size>  (dynamic register-relative)
        for name, val in deferred_mem:
            v = <uint64_t>(val & 0xFFFFFFFFFFFFFFFF)
            body = (<str>name)[4:]
            sep  = body.rfind('_')
            if sep < 0:
                continue

            # Try Format A: MEM_<hex>_<size>
            head = body[:sep]
            if head[:2] == '0x' or head[:3] == '-0x':
                try:
                    addr_u64 = <uint64_t>int(head, 16)
                    size = int(body[sep + 1:])
                    frame._write_mem(addr_u64, v, size)
                    continue
                except (ValueError, OverflowError):
                    pass

            # Format B: MEM_<reg>_<offset>_<size>
            # body looks like 'RBP_-16_8'.  rfind('_') gave us the size split,
            # so head='RBP_-16'. Find the next '_' between the reg name and
            # the signed integer offset.
            second_us = head.rfind('_')
            if second_us < 0:
                continue
            try:
                key = head[:second_us].upper()
                signed_off = <int64_t>int(head[second_us + 1:])
                size = int(body[sep + 1:])
            except (ValueError, OverflowError):
                continue

            off_obj = self._offsets.get(key)
            if off_obj is None:
                continue
            sz_obj = self._sizes.get(key)
            sz     = <int>sz_obj if sz_obj is not None else 8
            addr_u64 = frame._read_reg(<long>off_obj, sz)
            addr_u64 = (addr_u64 + <uint64_t>signed_off) & 0xFFFFFFFFFFFFFFFF
            frame._write_mem(addr_u64, v, size)

    cdef void _load_state(self, _PCodeFrame frame, dict regs, dict mem):
        """
        Load a frame directly from MachineState's `regs` and `mem` dicts.

        Avoids the (regs + MEM_<hex>_<size> string-keyed flat dict) round-trip
        that simulator.py's `evaluate_concrete` would otherwise build per call.
        Saves a dict copy and the MEM_ key parsing for every cell evaluation.
        """
        frame._arch = str(self.arch)  # for CBRANCH PC lookup
        cdef object   name, val, off_obj, sz_obj, addr, mval
        cdef str      key
        cdef long     off
        cdef uint64_t v, mv
        cdef int      sz, size

        frame.clear()
        # --- Registers ---
        for name, val in regs.items():
            v = <uint64_t>(val & 0xFFFFFFFFFFFFFFFF)
            key     = (<str>name).upper()
            off_obj = self._offsets.get(key)
            if off_obj is not None:
                off    = <long>off_obj
                sz_obj = self._sizes.get(key)
                sz     = <int>sz_obj if sz_obj is not None else 8
                frame._write_reg(off, sz, _mask64(v, sz))

        # --- Memory ---
        # mem is keyed by integer address; size derived from bit_length, like
        # simulator.py used to do when building the flat dict.
        for addr, mval in mem.items():
            mv = <uint64_t>(mval & 0xFFFFFFFFFFFFFFFF)
            if mval:
                size = ((<int>mval.bit_length()) + 7) // 8
                if size < 1:
                    size = 1
            else:
                size = 8
            frame._write_mem(<uint64_t>addr, mv, size)

    cdef uint64_t _read_output(self, _PCodeFrame frame, str out_reg,
                               int bit_start, int bit_end):
        cdef object   off_obj, sz_obj
        cdef str      key, body, head
        cdef long     off
        cdef uint64_t addr_u64
        cdef int      sz, size, sep, second_us, width
        cdef int64_t  signed_off
        cdef uint64_t val, mask

        width = bit_end - bit_start + 1

        # Direct slice comparison — faster than str.startswith.
        if out_reg[:4] == 'MEM_':
            body = out_reg[4:]
            sep  = body.rfind('_')
            if sep < 0:
                return 0

            # Format A: MEM_<hex>_<size> — static address
            head = body[:sep]
            if head[:2] == '0x' or head[:3] == '-0x':
                try:
                    addr_u64 = <uint64_t>int(head, 16)
                    size = int(body[sep + 1:])
                    val  = frame._read_mem(addr_u64, size)
                    if width >= 64:
                        return val >> bit_start
                    mask = (<uint64_t>1 << width) - 1
                    return (val >> bit_start) & mask
                except (ValueError, OverflowError):
                    return 0

            # Format B: MEM_<reg>_<offset>_<size> — register-relative.
            second_us = head.rfind('_')
            if second_us < 0:
                return 0
            try:
                key = head[:second_us].upper()
                signed_off = <int64_t>int(head[second_us + 1:])
                size = int(body[sep + 1:])
            except (ValueError, OverflowError):
                return 0

            off_obj = self._offsets.get(key)
            if off_obj is None:
                return 0
            sz_obj = self._sizes.get(key)
            sz     = <int>sz_obj if sz_obj is not None else 8
            addr_u64 = frame._read_reg(<long>off_obj, sz)
            addr_u64 = (addr_u64 + <uint64_t>signed_off) & 0xFFFFFFFFFFFFFFFF
            val = frame._read_mem(addr_u64, size)
            if width >= 64:
                return val >> bit_start
            mask = (<uint64_t>1 << width) - 1
            return (val >> bit_start) & mask

        key     = out_reg.upper()
        off_obj = self._offsets.get(key)
        if off_obj is None:
            return 0
        off    = <long>off_obj
        sz_obj = self._sizes.get(key)
        sz     = <int>sz_obj if sz_obj is not None else 8
        val    = frame._read_reg(off, sz)
        # x86 EFLAGS (offset 640, size 4) is never written directly by pcode —
        # the Sleigh spec writes individual flag registers (CF@512, ZF@518, etc.).
        # Reconstruct EFLAGS from those when the direct read returns 0.
        if val == 0 and off == 640 and sz == 4:
            val = (frame._read_reg(512, 1)       |   # CF  bit 0
                   (frame._read_reg(514, 1) << 2) |   # PF  bit 2
                   (frame._read_reg(518, 1) << 6) |   # ZF  bit 6
                   (frame._read_reg(519, 1) << 7) |   # SF  bit 7
                   (frame._read_reg(522, 1) << 10)|   # DF  bit 10
                   (frame._read_reg(523, 1) << 11))   # OF  bit 11
        if width >= 64:
            return val >> bit_start
        mask   = (<uint64_t>1 << width) - 1
        return (val >> bit_start) & mask

    # --- intermediate (unique) varnode read/seed for materialization ---

    cdef uint64_t _read_uniq(self, _PCodeFrame frame, DecodedOps decoded,
                             unsigned long raw_off, int bit_start, int bit_end):
        """Read a materialized intermediate by its RAW Sleigh unique offset."""
        cdef object   slot_obj = decoded.uniq_map.get(raw_off)
        cdef int      slot, width
        cdef uint64_t val, mask
        if slot_obj is None:
            return 0
        slot = <int>slot_obj
        if slot < 0 or slot >= 32 or not frame.uniq_set[slot]:
            return 0
        val   = frame.uniq_arr[slot]
        width = bit_end - bit_start + 1
        if width >= 64:
            return val >> bit_start
        mask = (<uint64_t>1 << width) - 1
        return (val >> bit_start) & mask

    cdef uint64_t _read_output_any(self, _PCodeFrame frame, DecodedOps decoded,
                                   str out_reg, int bit_start, int bit_end):
        """Read a register / MEM_ / UNIQ_<rawoffset> output."""
        if out_reg[:5] == 'UNIQ_':
            return self._read_uniq(frame, decoded,
                                   <unsigned long>int(out_reg[5:]), bit_start, bit_end)
        return self._read_output(frame, out_reg, bit_start, bit_end)

    cdef void _seed(self, _PCodeFrame frame, DecodedOps decoded,
                    object raw_off_obj, object val_obj):
        """Seed one intermediate (raw unique offset -> value) into a frame."""
        cdef object slot_obj = decoded.uniq_map.get(raw_off_obj)
        if slot_obj is None:
            return
        frame.seed_uniq(<int>slot_obj, <uint64_t>(val_obj & 0xFFFFFFFFFFFFFFFF))

    def evaluate_concrete(self, cell, flat_inputs):
        cdef _PCodeFrame frame = self._frame_a
        decoded = _get_decoded(self.arch, bytes.fromhex(cell.instruction))
        if decoded.has_fallback:
            raise PCodeFallbackNeeded('instruction requires Unicorn')
        self._load(frame, flat_inputs)
        _execute_decoded(frame, decoded)
        self.native_calls += 1
        return self._read_output(frame, cell.out_reg, cell.out_bit_start, cell.out_bit_end)

    def evaluate_concrete_state(self, cell, dict regs, dict mem):
        """
        Same as evaluate_concrete but takes MachineState's `regs` and `mem`
        dicts directly — no flat-dict copy, no MEM_<hex>_<size> key building.
        Hot path used by simulator.py for the use_unicorn=False configuration.
        """
        cdef _PCodeFrame frame = self._frame_a
        decoded = _get_decoded(self.arch, bytes.fromhex(cell.instruction))
        if decoded.has_fallback:
            raise PCodeFallbackNeeded('instruction requires Unicorn')
        self._load_state(frame, regs, mem)
        _execute_decoded(frame, decoded)
        self.native_calls += 1
        return self._read_output(frame, cell.out_reg, cell.out_bit_start, cell.out_bit_end)

    def evaluate_differential(self, cell, or_inputs, and_inputs):
        cdef _PCodeFrame fa = self._frame_a
        cdef _PCodeFrame fb = self._frame_b
        cdef uint64_t out_or, out_and
        decoded = _get_decoded(self.arch, bytes.fromhex(cell.instruction))
        if decoded.has_fallback:
            raise PCodeFallbackNeeded('instruction requires Unicorn')
        self._load(fa, or_inputs)
        _execute_decoded(fa, decoded)
        out_or = self._read_output(fa, cell.out_reg, cell.out_bit_start, cell.out_bit_end)
        self._load(fb, and_inputs)
        _execute_decoded(fb, decoded)
        out_and = self._read_output(fb, cell.out_reg, cell.out_bit_start, cell.out_bit_end)
        self.native_calls += 1
        return out_or ^ out_and

    # ------------------------------------------------------------------
    # Intermediate-taint materialization API
    # (docs/design/intermediate-taint-materialization.md)
    # ------------------------------------------------------------------

    def evaluate_uniq_concrete(self, instruction_hex, flat_inputs,
                               unsigned long raw_off, int bit_start, int bit_end):
        """Run the full instruction and read an intermediate (unique) varnode
        named by its RAW Sleigh offset.  Used to obtain value(t) and, via two
        replicas, the taint T(t) of a cut intermediate."""
        cdef _PCodeFrame frame = self._frame_a
        decoded = _get_decoded(self.arch, bytes.fromhex(instruction_hex))
        if decoded.has_fallback:
            raise PCodeFallbackNeeded('instruction requires Unicorn')
        self._load(frame, flat_inputs)
        _execute_decoded(frame, decoded)
        self.native_calls += 1
        return self._read_uniq(frame, decoded, raw_off, bit_start, bit_end)

    def uniq_start_pc(self, instruction_hex, unsigned long raw_off):
        """pc just past an intermediate's defining op (for a seeded suffix run),
        or 0 if the offset is not a known unique."""
        decoded = _get_decoded(self.arch, bytes.fromhex(instruction_hex))
        cdef object slot_obj = decoded.uniq_map.get(raw_off)
        if slot_obj is None:
            return 0
        cdef object pc_obj = decoded.uniq_def_pc.get(slot_obj)
        if pc_obj is None:
            return 0
        return <int>pc_obj + 1

    def evaluate_differential_seeded(self, instruction_hex, or_inputs, and_inputs,
                                     seeds_or, seeds_and, out_reg,
                                     int bit_start, int bit_end, int start_pc=0):
        """Differential over a downstream segment with materialized intermediates.

        or_inputs / and_inputs give the register+memory V|T and V&~T replicas.
        seeds_or / seeds_and map a raw unique offset -> that intermediate's value
        in each replica (already lifted by the intermediate's own taint mask, i.e.
        t|T(t) and t&~T(t)).  Execution starts at start_pc so the arithmetic core
        that produced the intermediate is not recomputed.  out_reg may be a
        register, MEM_..., or 'UNIQ_<rawoffset>'.  Returns out_or ^ out_and."""
        cdef _PCodeFrame fa = self._frame_a
        cdef _PCodeFrame fb = self._frame_b
        cdef uint64_t out_or, out_and
        cdef object off_obj, val_obj
        decoded = _get_decoded(self.arch, bytes.fromhex(instruction_hex))
        if decoded.has_fallback:
            raise PCodeFallbackNeeded('instruction requires Unicorn')
        if seeds_or is None:
            seeds_or = {}
        if seeds_and is None:
            seeds_and = {}

        self._load(fa, or_inputs)
        for off_obj, val_obj in seeds_or.items():
            self._seed(fa, decoded, off_obj, val_obj)
        _execute_decoded(fa, decoded, start_pc)
        out_or = self._read_output_any(fa, decoded, out_reg, bit_start, bit_end)

        self._load(fb, and_inputs)
        for off_obj, val_obj in seeds_and.items():
            self._seed(fb, decoded, off_obj, val_obj)
        _execute_decoded(fb, decoded, start_pc)
        out_and = self._read_output_any(fb, decoded, out_reg, bit_start, bit_end)

        self.native_calls += 1
        return out_or ^ out_and

    @property
    def fallback_rate(self):
        total = self.native_calls + self.fallback_calls
        return self.fallback_calls / total if total else 0.0

    def stats(self):
        return {
            'native_calls':   self.native_calls,
            'fallback_calls': self.fallback_calls,
            'fallback_rate':  self.fallback_rate,
        }
