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
    cdef public int    n_ops
    cdef public bint   has_fallback
    cdef public object input_reg_offsets  # set of SP_REGISTER input offsets


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
    result.input_reg_offsets = set()
    has_fallback = False
    # Compact unique-space mapping: raw offset → sequential index 0,1,2,...
    cdef dict uniq_map = {}
    cdef int  uniq_next = 0
    cdef PCodeOp* op_ptr
    cdef int _ii, _cb_count, _cb_last, _has_bi

    for op in ops:
        oid    = _opcode_id(op.opcode.name)
        out    = op.output
        ins    = op.inputs

        # Check for fallback conditions
        if oid == OP_BRANCHIND or oid == OP_CALLIND:
            has_fallback = True
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
            result.n_ops += 1

    # Simple conditional branches (JL, JE, …) have exactly ONE CBRANCH
    # as the very last decoded op and no BRANCHIND/CALLIND.  We handle
    # those natively by writing the PC in _execute_decoded, so no Unicorn
    # fallback is needed.  Complex cases (BSF/BSR = multiple CBRANCHes;
    # CMOVNZ = CBRANCH not last) keep has_fallback=True.
    if has_fallback:
        _cb_count = 0; _cb_last = -1; _has_bi = 0
        for _ii in range(result.n_ops):
            if result.buf[_ii].oid == OP_CBRANCH:
                _cb_count += 1; _cb_last = _ii
            elif result.buf[_ii].oid == OP_BRANCHIND or result.buf[_ii].oid == OP_CALLIND:
                _has_bi = 1
        if _cb_count == 1 and _cb_last == result.n_ops - 1 and not _has_bi:
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
    # Compact unique-space array: indices 0..15 replacing the uniq dict
    cdef uint64_t uniq_arr[32]
    cdef uint8_t  uniq_set[32]  # which slots are written
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
        self.dirty_count = 0
        self.regs      = {}
        self.reg_sizes = {}
        self.mem       = {}
        self._arch     = None

    cdef inline void _write_reg(self, long off, int sz, uint64_t val) noexcept:
        cdef uint64_t masked = _mask64(val, sz)
        if off >= 0 and off < REGS_ARR_SIZE:
            if not self.regs_set[off]:   # only record first write to each slot
                self.regs_set[off] = 1
                if self.dirty_count < 48:
                    self.dirty[self.dirty_count] = <int>off
                    self.dirty_count += 1
            self.regs_arr[off] = masked
            self.regs_sz [off] = <uint8_t>sz
        else:
            self.regs[off]      = masked
            self.reg_sizes[off] = sz

    cdef inline uint64_t _read_reg(self, long off, int sz) noexcept:
        cdef long     k, byte_off
        cdef int      k_sz
        cdef object   kv, v
        cdef uint64_t uv

        if off >= 0 and off < REGS_ARR_SIZE:
            if self.regs_set[off]:
                return _mask64(self.regs_arr[off], sz)
            # Sub-register overlap scan for C-array registers.
            # e.g. reading AH (off=1, sz=1) after writing RAX (off=0, sz=8):
            # regs_set[0]=1, regs_sz[0]=8, 0 <= 1 < 0+8 → extract byte 1.
            # Search backwards up to 8 bytes (max register size).
            k = off - 1
            while k >= 0 and off - k <= 8:
                if self.regs_set[k] and k + <long>self.regs_sz[k] > off:
                    byte_off = off - k
                    return _mask64(self.regs_arr[k] >> (byte_off * 8), sz)
                k -= 1
            return 0

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

    cdef inline void clear(self) noexcept:
        cdef int i
        # Only zero the slots that were actually written (dirty list vs scanning all 1104)
        for i in range(self.dirty_count):
            self.regs_set[self.dirty[i]] = 0
        self.dirty_count = 0
        # Clear compact unique array
        for i in range(32):
            if self.uniq_set[i]:
                self.uniq_set[i] = 0
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
            # off is a compact index (0..31) — direct C array write, no dict
            if off < 32:
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
) except *:
    """
    Execute all pre-decoded ops on frame using the C struct buffer.
    No Python tuple unpacking — all field access is direct C struct reads.
    Raises PCodeFallbackNeeded if any op requires Unicorn.
    """
    cdef int          oid, o_sp, o_sz, callother_out, n_ins
    cdef unsigned long o_off
    cdef int          i0_sp, i0_sz
    cdef unsigned long i0_off
    cdef int          i1_sp, i1_sz
    cdef unsigned long i1_off
    cdef int          i2_sp, i2_sz
    cdef unsigned long i2_off
    cdef uint64_t  a, b, c, result, u_result
    cdef int64_t   sa, sb, sresult
    cdef int       sz, bits, i
    cdef PCodeOp*  op
    cdef PCodeOp*  ops_base
    cdef int       n_ops

    # Hoist out of loop: one Python object access total, then pure C
    ops_base = decoded.buf
    n_ops    = decoded.n_ops

    for i in range(n_ops):
        op = ops_base + i                    # pure C pointer arithmetic
        oid            = op.oid
        o_sp           = op.o_sp
        o_off          = op.o_off
        o_sz           = op.o_sz
        callother_out  = op.callother_out
        n_ins          = op.n_ins
        i0_sp          = op.i0_sp;  i0_off = op.i0_off;  i0_sz = op.i0_sz
        i1_sp          = op.i1_sp;  i1_off = op.i1_off;  i1_sz = op.i1_sz
        i2_sp          = op.i2_sp;  i2_off = op.i2_off;  i2_sz = op.i2_sz

        # ── Hot path: most frequent pcode ops first ─────────────────
        if oid == OP_IMARK or oid == OP_BRANCH or oid == OP_RETURN or oid == OP_CALL:
            pass
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
            if o_sp != NO_OUT_SPACE:
                b = frame.read_d(i1_sp, i1_off, i1_sz) & 0x3F
                frame.write_d(o_sp, o_off, o_sz, frame.read_d(i0_sp, i0_off, i0_sz) >> b)
        elif oid == OP_INT_LEFT:
            if o_sp != NO_OUT_SPACE:
                b = frame.read_d(i1_sp, i1_off, i1_sz) & 0x3F
                frame.write_d(o_sp, o_off, o_sz, frame.read_d(i0_sp, i0_off, i0_sz) << b)
        # ── Less frequent ops ────────────────────────────────────────
        elif oid == OP_CBRANCH:
            cond   = frame.read_d(i1_sp, i1_off, i1_sz)
            dest   = <uint64_t>i0_off  # branch target from ram varnode
            result = dest if cond else dest + 1
            pc_tup = _ARCH_PC.get(str(frame._arch))
            if pc_tup is not None:
                frame.write_d(SP_REGISTER, pc_tup[0], pc_tup[1], result)
        elif oid == OP_BRANCHIND or oid == OP_CALLIND:
            raise PCodeFallbackNeeded('Control-flow opcode')
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
            if o_sp != NO_OUT_SPACE:
                sz  = i0_sz
                sa  = _signed64(frame.read_d(i0_sp, i0_off, i0_sz), sz)
                b   = frame.read_d(i1_sp, i1_off, i1_sz) & 0x3F
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
        cdef str      key, body
        cdef long     off
        cdef uint64_t addr_u64
        cdef int      sz, size, sep
        cdef uint64_t v

        frame.clear()
        for name, val in inputs.items():
            # Mask to 64-bit unsigned BEFORE casting: Python AND stays in Python
            # int domain (no overflow), then Cython casts the bounded value.
            v = <uint64_t>(val & 0xFFFFFFFFFFFFFFFF)
            if (<str>name).startswith('MEM_'):
                body = (<str>name)[4:]
                sep  = body.rfind('_')
                if sep >= 0:
                    try:
                        addr_u64 = int(body[:sep], 16)
                        size = int(body[sep + 1:])
                        frame._write_mem(addr_u64, v, size)
                    except (ValueError, OverflowError):
                        pass
            else:
                key     = (<str>name).upper()
                off_obj = self._offsets.get(key)
                if off_obj is not None:
                    off    = <long>off_obj
                    sz_obj = self._sizes.get(key)
                    sz     = <int>sz_obj if sz_obj is not None else 8
                    frame._write_reg(off, sz, _mask64(v, sz))

    cdef uint64_t _read_output(self, _PCodeFrame frame, str out_reg,
                               int bit_start, int bit_end):
        cdef object   off_obj, sz_obj
        cdef str      key, body
        cdef long     off
        cdef uint64_t addr_u64
        cdef int      sz, size, sep, width
        cdef uint64_t val, mask

        width = bit_end - bit_start + 1

        if out_reg.startswith('MEM_'):
            body = out_reg[4:]
            sep  = body.rfind('_')
            if sep >= 0:
                try:
                    addr_u64 = int(body[:sep], 16)
                    size = int(body[sep + 1:])
                    val  = frame._read_mem(addr_u64, size)
                    if width >= 64:
                        return val >> bit_start
                    mask = (<uint64_t>1 << width) - 1
                    return (val >> bit_start) & mask
                except (ValueError, OverflowError):
                    return 0
            return 0

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

    def evaluate_concrete(self, cell, flat_inputs):
        cdef _PCodeFrame frame = self._frame_a
        decoded = _get_decoded(self.arch, bytes.fromhex(cell.instruction))
        if decoded.has_fallback:
            raise PCodeFallbackNeeded('instruction requires Unicorn')
        self._load(frame, flat_inputs)
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
