# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
hook_core — Cython implementation of the per-instruction Unicorn hook.

This eliminates the per-call Python overhead of `_instruction_evaluator_raw`
(attribute lookups, frame setup, dict alloc, etc.) by moving the entire
body into Cython with C-API direct calls.

The hook function is exposed as a callable Python object (`InstructionHook`)
with a `__call__` method. Cython compiles `__call__` into a tight C function
that directly drives the dispatcher logic; only the Python frame entry from
Unicorn's binding is unavoidable.

This module is the V5 hot path. It's drop-in replacement for the body
of `MicrotaintWrapper._instruction_evaluator_raw` and is constructed once
in `_setup_hooks()`. The wrapper class fields it accesses are read once
at construction time and stored as typed cdef fields.

Tier 3 instruction cache is built directly into the hook for maximum
speed: the cache lookup uses PyDict_GetItem (no attribute load).
"""
from cpython.dict cimport (
    PyDict_New, PyDict_GetItem, PyDict_SetItem, PyDict_DelItem,
    PyDict_Size, PyDict_Clear, PyDict_Items, PyDict_Copy, PyDict_Next,
    PyDict_Contains,
)
from cpython.object cimport PyObject_RichCompareBool, Py_EQ
from cpython.long cimport PyLong_AsUnsignedLongLong, PyLong_FromUnsignedLongLong
from cpython.bytes cimport (
    PyBytes_FromStringAndSize, PyBytes_AS_STRING, PyBytes_GET_SIZE,
)
from cpython.set cimport PySet_Add, PySet_Discard, PySet_Contains
from cpython.exc cimport PyErr_Clear, PyErr_Occurred, PyErr_ExceptionMatches
from cpython.ref cimport Py_INCREF, Py_DECREF, Py_XINCREF, Py_XDECREF, PyObject
from cpython.pycapsule cimport PyCapsule_GetPointer

from microtaint.emulator.shadow cimport BitPreciseShadowMemory
from microtaint.instrumentation.ast cimport LogicCircuit
from libc.stdint cimport uint64_t
from libc.stdlib cimport malloc, calloc, free, realloc
from libc.string cimport memcpy, memcmp, memset

# uc_reg_read_batch(uc_engine *uc, int *regs, void **vals, int count) -> uc_err.
# Called through this C function pointer (address extracted from Unicorn's ctypes
# handle) to skip ctypes' per-call ffi_prep / ConvParam(isinstance) overhead on
# the register-read boundary, which runs on every slow-path instruction and every
# value-dependent cache hit.
ctypedef int (*uc_reg_read_batch_ft)(void *uc, void *regs, void *vals, int count) noexcept nogil

# uc_mem_read(uc_engine *uc, uint64_t address, void *bytes, size_t size) -> uc_err.
# Called through this C function pointer for the guest-memory reads that
# OP_PUSH_MEM_VALUE performs.  perf attributes ~14-15% of the taint phase to this
# path when it goes through ctypes: ConvParam / _stginfo_from_type /
# PyCSimpleType_from_param / PyCArgObject_new boxing plus ffi_call_int, and the
# isinstance + PyObject_Malloc/Free traffic they generate.  The fnptr call skips
# all of it (same trick already used for uc_reg_read_batch).
ctypedef int (*uc_mem_read_ft)(void *uc, uint64_t addr, void *buf, size_t size) noexcept nogil

# circuit_c's C API (circuit_c_api.h).  Calling the array evaluators as Python
# methods costs, per instruction, an attribute lookup + tuple build +
# PyArg_ParseTuple + PyLong boxing of both array addresses.  Through the capsule
# they are plain C calls taking uint64_t*.  Return convention is identical to the
# methods (new ref; Py_None = declined; NULL = error), so the fallback logic is
# unchanged.
ctypedef object (*eval_arr_ptr_ft)(object compiled, uint64_t *taint, uint64_t *val,
                                   int n_slots, object pcode, object name_to_slot)
ctypedef object (*eval_mem_ptr_ft)(object compiled, uint64_t *taint, uint64_t *val,
                                   int n_slots, object pcode, object shadow,
                                   object mem_reader, object name_to_slot)
# C guest-memory reader handed to circuit_c so OP_PUSH_MEM_VALUE never enters
# Python: 0 = read ok (*out set), anything else = "use the Python reader".
ctypedef int (*mt_mem_read_ft)(void *ctx, uint64_t addr, int size, uint64_t *out) noexcept nogil
ctypedef object (*eval_mem_ptr_c_ft)(object compiled, uint64_t *taint, uint64_t *val,
                                     int n_slots, object pcode, object shadow,
                                     object mem_reader, object name_to_slot,
                                     mt_mem_read_ft mem_fn, void *mem_ctx)

cdef struct MemReadCtx:
    unsigned long long uc_handle
    unsigned long long uc_mr_addr
    unsigned long long buf_addr

cdef int _mt_mem_read_c(void *ctx, uint64_t addr, int size, uint64_t *out) noexcept nogil:
    """Pure-C guest read: uc_mem_read through the fnptr, little-endian assembly
    out of the buffer.  Identical semantics to LiveMemReader's fast path.
    Declines (non-zero) for widths it does not handle or a failed read, so the
    Python reader keeps its Qiling fallback."""
    cdef MemReadCtx *c = <MemReadCtx*>ctx
    cdef unsigned char *p
    cdef uint64_t v = 0
    cdef int i
    if c == NULL or c.uc_mr_addr == 0 or c.buf_addr == 0:
        return 1
    if size <= 0 or size > 8:
        return 1
    if (<uc_mem_read_ft>(<void*>c.uc_mr_addr))(
            <void*>c.uc_handle, addr, <void*>c.buf_addr, <size_t>size) != 0:
        return 1
    p = <unsigned char*>c.buf_addr
    for i in range(size):
        v |= (<uint64_t>p[i]) << (8 * i)
    out[0] = v
    return 0

# Integer-returning forms of the same two evaluators.  The PyObject forms hand
# back an int count or a list of (addr, size, taint) tuples, so every memory
# instruction allocated a list, a tuple and three PyLongs (all GC-tracked) purely
# to move 17 bytes of result across the boundary.  These write into caller-owned
# C storage and return a plain int, so the steady state allocates nothing.
#
# `except? -2` matches MT_EVAL_ERROR: on that value Cython checks for a pending
# exception and propagates it, so a C-level failure surfaces exactly like the
# PyObject forms' NULL return did.
ctypedef int (*eval_arr_ptr_i_ft)(object compiled, uint64_t *taint, uint64_t *val,
                                  int n_slots, object pcode, object name_to_slot,
                                  int implicit_policy) except? -2
ctypedef int (*eval_mem_ptr_ci_ft)(object compiled, uint64_t *taint, uint64_t *val,
                                   int n_slots, object pcode, object shadow,
                                   object mem_reader, object name_to_slot,
                                   mt_mem_read_ft mem_fn, void *mem_ctx,
                                   MtMemWrite *out, int out_cap,
                                   int implicit_policy) except? -2

ctypedef int (*compiled_flags_ft)(object compiled) noexcept
ctypedef int (*compiled_prefilter_ft)(object compiled, object name_to_slot,
                                      void *pf) noexcept
ctypedef int (*uc_reg_read_batch_c_ft)(void *uc, void *ids, void *ptrs, int count) noexcept nogil

# ---------------------------------------------------------------------------
# The C hot path (fastpath.h).
#
# The per-instruction steady state lives there as plain C so that the absence of
# Python from it is mechanical rather than a claim: see that file's header.  All
# the shared structs are DEFINED there and only declared here, so the two cannot
# drift.  What stays on this side is everything that legitimately needs Python:
# first-time decode, the dict fallback, implicit-taint reporting, SMC.
# ---------------------------------------------------------------------------
cdef extern from "fastpath.h":
    ctypedef struct MtMemWrite:
        uint64_t addr
        uint64_t taint
        int size

    ctypedef struct CircuitCAPI:
        eval_arr_ptr_ft eval_arr_ptr
        eval_mem_ptr_ft eval_mem_ptr
        eval_mem_ptr_c_ft eval_mem_ptr_c
        eval_arr_ptr_i_ft eval_arr_ptr_i
        eval_mem_ptr_ci_ft eval_mem_ptr_ci
        compiled_flags_ft compiled_flags
        compiled_prefilter_ft compiled_prefilter

    ctypedef struct AddrEntry "MtAddrEntry":
        int size
        PyObject *instr_bytes
        PyObject *circuit
        PyObject *uc_arrs
        int *slots
        int n_in
        int have_slots
        unsigned long long ids_addr
        unsigned long long ptrs_addr
        unsigned long long vals_addr
        int n_calls_i
        int need_ef
        int have_regs
        unsigned long long ir_ids_addr
        unsigned long long ir_ptrs_addr
        unsigned long long ir_vals_addr
        int *ir_slots
        int ir_n_in
        int ir_n_calls_i
        int ir_need_ef
        int have_ir_regs
        uint64_t *in_snap
        uint64_t *out_snap
        uint64_t *val_snap
        int snap_n
        int have_snap
        int have_val
        void *ir_fn
        int ir_tried
        int ir_n_acc
        int ir_writes_pc
        signed char ir_acc_kind[8]
        signed char ir_acc_size[8]
        signed char ir_acc_needval[8]
        int express

    ctypedef struct AddrMap "MtAddrMap":
        uint64_t *keys
        AddrEntry **vals
        Py_ssize_t cap
        Py_ssize_t n

    ctypedef struct MtFastCtx:
        uint64_t **g_taint
        uint64_t **g_val
        int *n_slots
        AddrMap *map
        unsigned long long *uc_handle_addr
        uc_reg_read_batch_c_ft uc_reg_read_batch
        const CircuitCAPI *capi
        PyObject *pcode
        PyObject *shadow
        PyObject *mem_reader
        PyObject *slot_map
        mt_mem_read_ft mem_fn
        void *mem_ctx
        uint64_t (*shadow_read_mask)(object shadow, uint64_t addr, int size) noexcept
        void (*shadow_write_mask)(object shadow, uint64_t addr, uint64_t mask, int size) noexcept
        uint64_t *ir_val
        uint64_t *ir_taint
        uint64_t *ir_out
        int *eflags_slot
        int **ef_slots
        int **ef_bits
        int *ef_n
        int *rip_slot
        MtMemWrite *ltw
        int *ltw_n
        int implicit_policy
        int instr_cache_enabled
        int use_cregs
        int use_cmem
        unsigned long *hits
        unsigned long *misses
        unsigned long *fast_done
        unsigned long *prefilter_hits
        unsigned *express_gen
        unsigned long *express_done
        unsigned long *express_miss
        unsigned long *instr_total
        int *arr_loaded
        int *check_aiw
        PyObject **ltw_set
        PyObject *owner
        long long regs_read
        long long regs_circuit

    uint64_t MT_EMPTY_ADDR

    void mt_am_clear(AddrMap *m)
    void mt_am_free(AddrMap *m)
    AddrEntry *mt_am_get(AddrMap *m, uint64_t key) noexcept
    AddrEntry *mt_am_new(AddrMap *m, uint64_t key)
    # Returns MT_FAST_DONE / MT_FAST_SLOW, or MT_FAST_ERROR with an exception set.
    int mt_fast_step(MtFastCtx *c, uint64_t address, AddrEntry *ent,
                     object compiled, int cflags) except? 2
    # Neither of these can raise: the express lane touches no PyObject, and
    # arming it only reads per-address constants the full path already resolved.
    void mt_express_arm(MtFastCtx *c, AddrEntry *ent, object compiled,
                        int cflags) noexcept
    int mt_fast_step_nogil(MtFastCtx *c, uint64_t address,
                           unsigned int size) noexcept nogil

cdef extern from "memhook.h":
    ctypedef struct MtMemHookUD:
        void *shadow
        void (*clear)(void *shadow, uint64_t address, int size) noexcept nogil
        int (*is_poisoned)(void *shadow, uint64_t address, int size) noexcept nogil
        PyObject *owner
        int *check_uaf
        const MtMemWrite *ltw
        const int *ltw_n
        PyObject **lw_set
        const uint64_t *code_lo
        const uint64_t *code_hi

    # Both return 0 when they have handled the access and non-zero when the
    # caller must take the GIL; neither commits anything in the latter case.
    int mt_mem_read_nogil(MtMemHookUD *u, uint64_t address, int size) noexcept nogil
    int mt_mem_write_nogil(MtMemHookUD *u, uint64_t address, int size) noexcept nogil

# Mirrors of the header's constants.  Cython needs compile-time literals to size
# an array and to fold branch tests, so these are duplicated deliberately; the
# header is the source of truth and the values must match it.
DEF MT_FAST_DONE_AIW = 3
DEF MT_EVAL_DECLINED = -1
DEF MT_EVAL_ERROR = -2
# Compiled-program memory state: must match MT_IR_MEM_BASE / MT_IR_MAX_ACC in
# fastpath.h, and MEM_SLOT_BASE / MAX_ACCESSES in taint_ir/engine_glue.py.
DEF MT_IR_MEM_BASE = 512
DEF MT_IR_MAX_ACC = 8
DEF MT_IR_SLOTS = MT_IR_MEM_BASE + 4 * MT_IR_MAX_ACC

DEF MT_MAX_MEM_WRITES = 32
DEF MT_FAST_DONE = 0
DEF MT_FAST_SLOW = 1
DEF MT_FAST_ERROR = 2
# Circuit capabilities the hot path consults per instruction.  Reading them as
# Python attributes cost two PyObject_GenericGetAttr lookups per instruction.
DEF MT_CF_C_EVALUABLE = 0x01
DEF MT_CF_C_MEM_EVALUABLE = 0x02
DEF MT_CF_HAS_MEM_OPS = 0x04
DEF MT_CF_VALUE_INDEP = 0x08
DEF MT_CF_PC_TARGET = 0x10
DEF MT_CF_PY_FALLBACK = 0x20

cdef CircuitCAPI *_circuit_capi = NULL

cdef void _load_circuit_capi():
    global _circuit_capi
    cdef object mod, cap
    try:
        from microtaint.instrumentation.cell_c import circuit_c as _cc
        cap = getattr(_cc, '_circuit_capi', None)
        if cap is not None:
            _circuit_capi = <CircuitCAPI*>PyCapsule_GetPointer(
                cap, b"microtaint.instrumentation.cell_c.circuit_c._circuit_capi")
    except Exception:  # noqa: BLE001 - not fatal, we fall back to the methods
        _circuit_capi = NULL
        PyErr_Clear()

_load_circuit_capi()

# Shadow memory at the C level, for the compiled-program memory path.  Mirrors
# _ShadowCAPI in shadow.pyx; when the capsule is unavailable the pointers stay
# NULL and that path is simply not taken.
cdef struct _MtShadowCAPI:
    uint64_t (*read_mask)(object shadow, uint64_t address, int size) noexcept
    void     (*write_mask)(object shadow, uint64_t address, uint64_t mask, int size) noexcept
    # The memory callbacks call these two with no GIL held; see shadow.pyx.
    void     (*clear)(void *shadow, uint64_t address, int size) noexcept nogil
    int      (*is_poisoned)(void *shadow, uint64_t address, int size) noexcept nogil

cdef _MtShadowCAPI *_shadow_capi = NULL

cdef void _load_shadow_capi():
    global _shadow_capi
    cdef object mod, cap
    try:
        from microtaint.emulator import shadow as _sh
        cap = getattr(_sh, '_shadow_capi', None)
        if cap is not None:
            _shadow_capi = <_MtShadowCAPI*>PyCapsule_GetPointer(
                cap, b"microtaint.emulator.shadow._shadow_capi")
    except Exception:  # noqa: BLE001 - not fatal; the circuit path still works
        _shadow_capi = NULL
        PyErr_Clear()

_load_shadow_capi()

import ctypes
from microtaint.types import ImplicitTaintError as _ImplicitTaintError

cdef object ImplicitTaintError = _ImplicitTaintError
cdef object EMPTY_FROZENSET = frozenset()

# Use the GIL-free C taint evaluator (CompiledCircuit.evaluate_c) for eligible
# circuits (register-only, no PC target -> no implicit-taint check needed).  It is
# bit-identical to circuit.evaluate and skips the PyObject-heavy eval loop.  Set
# MICROTAINT_DISABLE_DO_EVALUATE_C=1 to force the classic path.
import os as _os
cdef bint _USE_DEVAL_C = _os.environ.get('MICROTAINT_DISABLE_DO_EVALUATE_C') != '1'

# Memory circuits (loads/stores/mem-ALU) run the C-array interior via
# CompiledCircuit.evaluate_c_mem (shadow read/write at the C level), skipping the
# PyObject-heavy do_evaluate path.  Bit-identical to circuit.evaluate; returns
# None to fall back for anything it cannot handle.  MICROTAINT_DISABLE_CMEM=1
# forces the classic path.  MICROTAINT_DIFF_CMEM=1 additionally runs the full
# do_evaluate and asserts equality on every memory instruction (soundness gate,
# for the test suite -- slow, never in production).
cdef bint _USE_CMEM  = _os.environ.get('MICROTAINT_DISABLE_CMEM') != '1'
cdef bint _DIFF_CMEM = _os.environ.get('MICROTAINT_DIFF_CMEM') == '1'

# The compiled-taint-program path.  On by default; MICROTAINT_TAINT_IR=0 sends
# every instruction back through the evaluator.  It replaced the evaluator on
# the hot path once the whole suite had been run with it on, both ways.
cdef bint _NULL_HOOK = _os.environ.get('MICROTAINT_NULL_HOOK', '') not in ('', '0')

_taint_ir_on = _os.environ.get('MICROTAINT_TAINT_IR') != '0'
_taint_ir_program_for = None
if _taint_ir_on:
    try:
        from microtaint.taint_ir.engine_glue import program_for as _taint_ir_program_for
    except Exception:
        _taint_ir_on = False


# Read input registers by calling uc_reg_read_batch through a C function pointer
# (and reading the value array via a uint64*), instead of the Python ctypes call
# + per-slot ctypes indexing.  MICROTAINT_DISABLE_CREGS=1 forces the ctypes path.
cdef bint _USE_CREGS = _os.environ.get('MICROTAINT_DISABLE_CREGS') != '1'

# Address-keyed decode cache: on a repeat visit to an address, skip the
# uc_mem_read + ctypes buffer slice + cached_gen_rule tuplehash (~3.0 us of the
# ~4.5 us cache-hit floor) by reusing the (bytes, circuit) decoded last time.
# The Tier-3/Tier-4 output caches already assume address->instruction stability
# (they replay output_state by address without re-checking bytes), so this adds
# no new correctness assumption.  Set MICROTAINT_DISABLE_DECODE_CACHE=1 to force
# a fresh read+decode on every instruction.
cdef bint _USE_DECODE_CACHE = _os.environ.get('MICROTAINT_DISABLE_DECODE_CACHE') != '1'

# ---------------------------------------------------------------------------
# ARRAY-NATIVE TAINT STATE (rework Phase 1.3c).
#
# The dict hot path spends ~23.5% of the taint phase in PyObject churn (measured
# by perf: _PyObject_Malloc/_Free, _Py_dict_lookup/PyDict_SetItem/dict_dealloc,
# attribute lookups, tuple dealloc) building a pre_regs dict, copying pre_taint,
# building an output_state dict, refilling register_taint, and keying the Tier-3/4
# caches on frozenset(dict.items()) + dict-equality.
#
# The array path keeps the taint AND the input values in flat slot-indexed uint64
# C arrays, evaluates through CompiledCircuit.evaluate_c_arr_ptr (registers) and
# evaluate_c_mem_ptr (memory) which write those arrays in place, and keys the
# caches on memcmp of an array snapshot.  No dict is built per instruction.
#
# `register_taint` remains the external contract (wrapper / tests / the Python
# fallback): the array is authoritative only DURING a Cython-hook run, loaded from
# the dict on the first instruction and synced back on demand via
# sync_taint_to_dict().  That is safe because during a hook run nothing else
# touches register_taint (verified: wrapper.py's _instruction_evaluator_raw is the
# Python fallback, active only when this hook is NOT installed).
#
# ON by default: proven bit-equivalent (full suite green, identical pass/skip
# counts to the dict path) and ~17% faster e2e.  MICROTAINT_ARR_HOOK=0 forces the
# classic dict path, which stays in place as the fallback and escape hatch.
cdef bint _USE_ARR_HOOK = _os.environ.get('MICROTAINT_ARR_HOOK') != '0'
# The GIL-free express lane (fastpath.h).  On by default; MICROTAINT_EXPRESS=0
# leaves every instruction on the GIL path, which is what the differential test
# compares against and what to reach for if it is ever suspected.
cdef bint _EXPRESS = _os.environ.get('MICROTAINT_EXPRESS') != '0'
# The GIL-free memory callbacks (memhook.h).  Same contract, same off-switch:
# MICROTAINT_NOGIL_MEM=0 sends every load and store back through the GIL path.
cdef bint _NOGIL_MEM = _os.environ.get('MICROTAINT_NOGIL_MEM') != '0'


# ---------------------------------------------------------------------------
# Per-address C cache.
#
# The table itself (MtAddrMap / MtAddrEntry and the mt_am_* operations) lives in
# fastpath.h so the C hot path and this module share one definition.  It replaced
# three Python dicts keyed by address (decode_cache, slots_cache, arr_cache),
# whose every lookup cost a PyLong key and a hashed probe, and whose every store
# allocated PyBytes snapshots.
#
# The PyObject fields on an entry (instruction bytes, circuit, uc_arrays tuple)
# stay PyObjects because the decoder and evaluator genuinely need them, but they
# are owned pointers now: no dict lookup, tuple indexing or boxing per visit.
# ---------------------------------------------------------------------------


cdef class InstructionHook:
    """
    Compiled hook callable. Holds a typed reference to the wrapper's fields
    and exposes __call__(uc, address, size, user_data) which Unicorn invokes
    on every instruction in the hooked range.

    All hot-path mutations go through PyDict_* directly, with no Python
    attribute lookups in the steady state.
    """
    # Wrapper-state fields, captured once at construction.
    cdef public object wrapper            # MicrotaintWrapper, for slow-path callbacks
    cdef public dict   register_taint     # mutated in-place
    cdef public set    last_tainted_writes
    # C form of the same information, used by the array path.  The set version
    # boxes a PyLong per tainted byte on the way in and another per written byte
    # on the way out (MemWriteHook's membership test), for data that is at most
    # a handful of (addr, size, mask) triples.  Holding the triples instead makes
    # both sides pure integer work and lets the hot path stay callable from C.
    # The set stays authoritative for the dict/fallback paths, which still fill
    # it; MemWriteHook consults both, so the two representations never disagree.
    cdef MtMemWrite ltw[MT_MAX_MEM_WRITES]
    cdef int ltw_n
    cdef public dict   instr_cache        # address -> (frozenset, dict)  legacy entries
    cdef public dict   instr_cache_v      # address -> (taint_version, dict, taint_snapshot)
    cdef public dict   py_decode_cache    # address -> (size, bytes, circuit); dict path only
    # Bounds of every address currently in decode_cache, [code_lo, code_hi).  The
    # mem-write hook uses this to detect self-modifying / JIT'd code: a guest
    # write intersecting this range invalidates the decode + output caches so the
    # rewritten instruction is re-read and re-decoded on its next execution.
    cdef public unsigned long long code_lo
    cdef public unsigned long long code_hi
    cdef public BitPreciseShadowMemory shadow_mem  # cdef class — direct C-level method dispatch
    cdef public object sim                # CellSimulator
    # sim._pcode, captured once.  CellSimulator assigns it in its __init__ and
    # never again, so re-reading it per instruction only bought a
    # PyObject_GenericGetAttr on the hot path.
    cdef object pcode_obj
    cdef public object policy             # ImplicitTaintPolicy
    cdef public object reporter
    cdef public object ql                 # Qiling for emu_stop / mem.read fallback
    cdef public bint   check_bof
    cdef public bint   check_sc
    cdef public bint   check_aiw
    cdef public bint   instr_cache_enabled

    # Versioned-state tracking: incremented every time register_taint mutates.
    # Used as a fast cache key — avoids frozenset(register_taint.items())
    # on every callback (~1 us savings per call × 1.19M = ~1.3 s).
    cdef public unsigned long long taint_version

    # Captured ctypes function pointers and helpers.
    cdef object       uc_handle           # ctypes c_void_p (unicorn engine handle)
    cdef object       uc_mem_read         # ctypes function
    cdef object       uc_reg_read_batch   # ctypes function
    # C-level register-read boundary: uc_reg_read_batch's raw C address + the uc
    # engine handle as an integer (cached lazily; the handle is fixed once
    # emulation starts).  Used by _read_pre_regs when _USE_CREGS is on.
    cdef unsigned long long uc_rrb_addr
    cdef unsigned long long uc_handle_addr
    cdef object       mem_buf             # ctypes buffer
    cdef object       arch                # Architecture enum
    cdef object       cached_gen_rule     # _cached_generate_static_rule
    cdef object       x64_format_key      # tuple
    cdef object       get_decoded         # _get_decoded
    cdef object       build_offsets_arrs  # _RegisterFile.offsets_arrays
    cdef object       eflags_bits         # dict: flag name -> bit in the parent
    cdef object       flag_parent         # str, or None where the ISA has none
    cdef object       pc_reg_name         # str
    cdef object       eval_context_cls    # EvalContext
    cdef dict         taint_ir_reg_arrs   # instr bytes -> the reduced read's arrays
    cdef object       read_live_memory    # bound method
    cdef object       get_live_registers  # bound method
    cdef object       disasm              # bound method

    # Counters
    cdef public unsigned long instr_cache_hits
    cdef public unsigned long instr_cache_misses
    # Instructions the C fast path completed without entering Python at all,
    # and the total it was offered, so coverage is a ratio and not a guess.
    cdef public unsigned long fast_done
    cdef public unsigned long instr_total
    # Instructions dismissed outright because no input was tainted.
    cdef public unsigned long prefilter_hits
    # Instructions the GIL-free express lane finished on its own, and the
    # generation counter that invalidates its cached prefilters.
    cdef public unsigned long express_done
    cdef unsigned long express_miss_c[16]
    cdef unsigned express_gen
    cdef PyObject *ltw_set_ptr
    # Why instructions leave the C path, so the biggest reason is a measurement.
    cdef public unsigned long fb_pc
    cdef public unsigned long fb_pyfall
    cdef public unsigned long fb_mem
    cdef public unsigned long fb_other
    cdef public unsigned long fb_nocircuit

    # --- array-native taint state (Phase 1.3c; see _USE_ARR_HOOK note) ---
    cdef uint64_t *g_taint            # slot-indexed taint, authoritative during a run
    cdef uint64_t *g_val              # slot-indexed live input values
    cdef int n_slots                  # slots in use
    cdef int cap_slots                # slots allocated
    cdef public dict slot_map         # register name -> slot index
    cdef public dict arr_cache        # address -> (in_snap, out_snap, val_snap, slots)
    cdef public dict slots_cache      # instruction bytes -> list of input slots
    cdef public dict taint_ir_progs   # instruction bytes -> compiled taint program (keeps its code alive)
    cdef uint64_t *ir_scratch         # 3 x MT_IR_SLOTS, for the two-pass memory protocol
    cdef AddrMap addr_map             # pure-C per-address cache (array path)
    # Handed to mt_fast_step.  Holds POINTERS to the mutable fields above rather
    # than copies, so slot growth and lazy slot interning are picked up with no
    # refresh step to forget; see the note in fastpath.h.
    cdef MtFastCtx fctx
    cdef bint arr_loaded              # register_taint has been loaded into g_taint
    cdef bint block_mode              # block mode owns g_taint for the whole run
    #: Called by invalidate_smc when block mode is on, to drop its plan cache
    #: too.  The block runtime is a separate module and its cache is keyed by
    #: (address, size), so a rewritten block would otherwise run the plan
    #: compiled for what used to be there.
    cdef public object block_invalidate
    cdef public unsigned long arr_fallbacks   # instructions that had to use the dict path
    cdef MemReadCtx mem_ctx           # C guest-read context handed to circuit_c
    cdef bint mem_ctx_ready
    cdef int rip_slot                 # cached slot of the PC register
    cdef int eflags_slot              # cached slot of EFLAGS
    cdef list eflags_slot_bits        # cached [(flag slot, bit index), ...]
    cdef int *ef_slots                # C form of the above, built once
    cdef int *ef_bits
    cdef int ef_n

    def __init__(self, wrapper, *,
                 uc_handle, uc_mem_read, uc_reg_read_batch, mem_buf,
                 arch, cached_gen_rule, x64_format_key,
                 get_decoded, build_offsets_arrs, eflags_bits,
                 eval_context_cls, uc_reg_read_batch_addr=0,
                 flag_parent='EFLAGS', pc_reg_name='RIP'):
        self.wrapper = wrapper
        self.register_taint = wrapper.register_taint
        self.last_tainted_writes = wrapper._last_tainted_writes
        self.instr_cache = wrapper._instr_cache
        # Version-keyed companion cache.  address -> (taint_version, output_state).
        # On hit, no dict-equality check needed: same version means same state.
        self.instr_cache_v = {}
        # Address-keyed decode cache (see _USE_DECODE_CACHE note at module top).
        self.py_decode_cache = {}
        # Empty range: code_lo > code_hi means "no cached code yet".
        self.code_lo = 0xFFFFFFFFFFFFFFFF
        self.code_hi = 0
        self.shadow_mem = wrapper.shadow_mem
        self.sim = wrapper.sim
        self.pcode_obj = getattr(wrapper.sim, '_pcode', None)
        self.policy = wrapper._policy
        self.reporter = wrapper.reporter
        self.ql = wrapper.ql
        self.check_bof = wrapper.check_bof
        self.check_sc = wrapper.check_sc
        self.check_aiw = wrapper.check_aiw
        self.instr_cache_enabled = wrapper._instr_cache_enabled
        self.taint_version = 0
        # uc_handle is already a ctypes c_void_p (ql.uc._uch), so a c_void_p-typed
        # ctypes call needs no per-call from_param conversion for it.
        self.uc_handle = uc_handle
        self.uc_mem_read = uc_mem_read
        self.uc_reg_read_batch = uc_reg_read_batch
        # C-level reg-read boundary.  The fnptr address is stable; the uc handle
        # integer is resolved lazily on first use (it is fixed once emulation
        # starts, and may not be set at hook-construction time).
        self.uc_rrb_addr = <unsigned long long>(uc_reg_read_batch_addr or 0)
        self.uc_handle_addr = 0
        self.mem_buf = mem_buf
        self.arch = arch
        self.cached_gen_rule = cached_gen_rule
        self.x64_format_key = x64_format_key
        self.get_decoded = get_decoded
        self.build_offsets_arrs = build_offsets_arrs
        self.eflags_bits = eflags_bits
        self.flag_parent = flag_parent
        self.pc_reg_name = pc_reg_name
        self.eval_context_cls = eval_context_cls
        self.read_live_memory = wrapper._live_mem_reader
        self.get_live_registers = wrapper._get_live_registers
        self.disasm = wrapper._disasm
        self.instr_cache_hits = 0
        self.instr_cache_misses = 0

        # Register with the wrapper so its `register_taint` property can sync the
        # array back on read.  Done here (not only in _setup_hooks) because tests
        # and tools build a hook via _make_cython_hook() and drive it directly;
        # without this the property would find no hook and hand back a stale dict.
        # A hook that is built but never runs has arr_loaded False, and
        # sync_taint_to_dict early-returns then, so this cannot clobber the dict.
        try:
            wrapper._instr_hook_obj = self
        except Exception:  # noqa: BLE001 - wrapper may forbid the attribute
            pass

        # --- array-native taint state (Phase 1.3c) ---
        self.g_taint = NULL
        self.g_val = NULL
        self.n_slots = 0
        self.express_done = 0
        cdef int _mi
        for _mi in range(16):
            self.express_miss_c[_mi] = 0
        self.express_gen = 1
        self.ltw_set_ptr = NULL
        self.cap_slots = 0
        self.slot_map = {}
        self.taint_ir_progs = {}
        # The ctypes arrays behind each program's reduced register read, held
        # for as long as the entries cache their addresses.
        self.taint_ir_reg_arrs = {}
        self.arr_cache = {}
        self.slots_cache = {}
        self.addr_map.keys = NULL
        self.addr_map.vals = NULL
        self.addr_map.cap = 0
        self.addr_map.n = 0
        self.arr_loaded = False
        self.block_mode = False
        self.block_invalidate = None
        self.arr_fallbacks = 0
        self.mem_ctx_ready = False
        self.rip_slot = -1
        self.eflags_slot = -1
        self.eflags_slot_bits = None
        self.ef_slots = NULL
        self.ef_bits = NULL
        self.ef_n = 0
        self.ltw_n = 0
        self.fast_done = 0
        self.instr_total = 0
        self.prefilter_hits = 0
        self.fb_pc = 0
        self.fb_pyfall = 0
        self.fb_mem = 0
        self.fb_other = 0
        self.fb_nocircuit = 0
        self._init_fast_ctx()
        # Pre-intern the arch's state-format registers so the slot space is
        # stable for the common case (SIMD VL_ lanes are interned on demand).
        cdef object _nm
        if x64_format_key is not None:
            for _entry in x64_format_key:
                _nm = _entry[0] if isinstance(_entry, tuple) else _entry
                self._slot_for(_nm)

    cdef void _init_fast_ctx(self):
        """Point the C hot-path context at this object's own fields.

        Everything mutable is passed as the ADDRESS of the field, which is fixed
        for the object's lifetime, so the context never needs refreshing and can
        never hold a stale g_taint after slot growth reallocates it.  The
        PyObject members are borrowed: this object owns them and outlives every
        call into the C path."""
        self.fctx.g_taint = &self.g_taint
        self.fctx.g_val = &self.g_val
        self.fctx.n_slots = &self.n_slots
        self.fctx.map = &self.addr_map
        self.fctx.uc_handle_addr = &self.uc_handle_addr
        self.fctx.uc_reg_read_batch = <uc_reg_read_batch_c_ft>(<void*>self.uc_rrb_addr)
        self.fctx.capi = _circuit_capi
        # Py_None rather than NULL for absent objects: the evaluator already
        # treats None as "not provided", and a NULL would be a crash instead.
        self.fctx.pcode = <PyObject*>self.pcode_obj if self.pcode_obj is not None else <PyObject*>None
        self.fctx.shadow = <PyObject*>self.shadow_mem if self.shadow_mem is not None else <PyObject*>None
        self.fctx.mem_reader = <PyObject*>self.read_live_memory \
            if self.read_live_memory is not None else <PyObject*>None
        self.fctx.slot_map = <PyObject*>self.slot_map
        # The compiled-program memory path: shadow access as plain C, and the
        # scratch state its two passes need (nothing may touch the live arrays
        # until both have succeeded).  Allocated once; if either is missing the
        # path is not taken and the circuit evaluator handles memory as before.
        if _shadow_capi != NULL:
            self.fctx.shadow_read_mask = _shadow_capi.read_mask
            self.fctx.shadow_write_mask = _shadow_capi.write_mask
        self.ir_scratch = <uint64_t*>calloc(3 * MT_IR_SLOTS, sizeof(uint64_t))
        if self.ir_scratch != NULL:
            self.fctx.ir_val = self.ir_scratch
            self.fctx.ir_taint = self.ir_scratch + MT_IR_SLOTS
            self.fctx.ir_out = self.ir_scratch + 2 * MT_IR_SLOTS
        self.fctx.mem_fn = _mt_mem_read_c
        self.fctx.mem_ctx = <void*>&self.mem_ctx
        self.fctx.eflags_slot = &self.eflags_slot
        self.fctx.ef_slots = &self.ef_slots
        self.fctx.ef_bits = &self.ef_bits
        self.fctx.ef_n = &self.ef_n
        self.fctx.rip_slot = &self.rip_slot
        self.fctx.ltw = self.ltw
        self.fctx.ltw_n = &self.ltw_n
        # int() of the ImplicitTaintPolicy IntEnum; the C side mirrors the values.
        try:
            self.fctx.implicit_policy = <int>int(self.policy)
        except Exception:  # noqa: BLE001 - unknown policy: take the Python path
            self.fctx.implicit_policy = 1   # MT_POLICY_WARN -> always defers
        self.fctx.instr_cache_enabled = 1 if self.instr_cache_enabled else 0
        self.fctx.use_cregs = 1 if _USE_CREGS else 0
        self.fctx.use_cmem = 1 if _USE_CMEM else 0
        self.fctx.hits = &self.instr_cache_hits
        self.fctx.misses = &self.instr_cache_misses
        self.fctx.fast_done = &self.fast_done
        self.fctx.prefilter_hits = &self.prefilter_hits
        # ---- the GIL-free express lane (fastpath.h) --------------------
        # user_data for the callback is &fctx, so the context carries the owner
        # rather than the other way round: reaching it must not cost a cast of a
        # PyObject, which is the very thing the lane exists to avoid.
        self.fctx.owner = <PyObject*>self
        self.fctx.express_gen = &self.express_gen
        self.fctx.express_done = &self.express_done
        self.fctx.express_miss = self.express_miss_c
        self.fctx.instr_total = &self.instr_total
        self.fctx.arr_loaded = <int*>&self.arr_loaded
        self.fctx.check_aiw = <int*>&self.check_aiw
        # Borrowed, and only when it really is a set: the lane reads its size
        # through PySet_GET_SIZE, which does not check.  The wrapper clears this
        # set in place and never rebinds it, so one pointer stays correct.
        if isinstance(self.last_tainted_writes, set):
            self.ltw_set_ptr = <PyObject*>self.last_tainted_writes
            self.fctx.ltw_set = &self.ltw_set_ptr
        else:
            self.fctx.ltw_set = NULL

    #: Reason names for express_miss, in index order (see fastpath.h).
    EXPRESS_MISS_REASONS = ('cold', 'unarmed', 'dirty_no_program',
                            'claims_standing', 'no_register_read',
                            'unused', 'memory_declined', 'unused',
                            'mem_state_too_big', 'mem_no_shadow',
                            'mem_guest_read_failed', 'mem_pc_policy',
                            'unused', 'unused', 'unused', 'unused')

    @property
    def express_miss(self):
        """Per-reason counts of express-lane declines, in index order.

        Pair with EXPRESS_MISS_REASONS.  Cheap enough to leave on: it counts
        only on a path that is already leaving for the GIL."""
        return tuple(self.express_miss_c[i] for i in range(16))

    def __dealloc__(self):
        mt_am_free(&self.addr_map)
        if self.g_taint != NULL:
            free(self.g_taint)
            self.g_taint = NULL
        if self.g_val != NULL:
            free(self.g_val)
            self.g_val = NULL
        if self.ef_slots != NULL:
            free(self.ef_slots)
            self.ef_slots = NULL
        if self.ef_bits != NULL:
            free(self.ef_bits)
            self.ef_bits = NULL
        if self.ir_scratch != NULL:
            free(self.ir_scratch)
            self.ir_scratch = NULL

    # ------------------------------------------------------------------
    # Array-native taint state helpers (Phase 1.3c)
    # ------------------------------------------------------------------
    def prepare_block_mode(self, names):
        """Everything the per-instruction path would have done lazily.

        SETUP, not the hot path.  Two things are resolved on first use by the
        instruction path, and block mode never runs it:

          * a slot for every register.  Without this the slot map is empty and
            every block program is compiled against nothing.
          * the Unicorn engine handle, which the batch register read needs.
            Measured: without it, `blk_read_regs` failed for all 10,540 blocks
            of bench_untainted and not one was handled.
          * the C guest-memory read context.  Measured the same way: with the
            handle resolved but this still zeroed, 10,240 of those blocks
            declined on the memory read alone.

        Called once, before the run.
        """
        cdef int last = -1
        if self.uc_handle is not None and self.uc_handle.value:
            self.uc_handle_addr = <unsigned long long>self.uc_handle.value
        if not self.mem_ctx_ready:
            self._init_mem_ctx()
        for name in names:
            last = self._slot_for(name)
        # The array is the state for the whole run: the per-instruction hook is
        # never armed alongside block mode, so nothing else loads the seed in or
        # syncs the answer back.  Without this, register taint seeded by the
        # caller was dropped and register taint COMPUTED by the block runtime
        # was invisible through `wrapper.register_taint` -- measured on
        # bench_sparse, RBX and R11 came back clean while every memory word
        # agreed, which reads as a block-mode under-taint.
        self.block_mode = True
        self._load_dict_to_arr()
        return last

    cdef int _slot_for(self, object name) except -1:
        """Intern a register name to a slot, growing the arrays as needed.

        Growth invalidates arr_cache because cached snapshots are length-keyed
        to n_slots.  Names are finite per program, so this converges quickly."""
        cdef object got = self.slot_map.get(name)
        cdef int slot
        cdef int newcap
        cdef uint64_t *nt
        cdef uint64_t *nv
        if got is not None:
            return <int>got
        slot = self.n_slots
        if slot >= self.cap_slots:
            newcap = 64 if self.cap_slots == 0 else self.cap_slots * 2
            nt = <uint64_t*>realloc(self.g_taint, newcap * sizeof(uint64_t))
            if nt == NULL:
                raise MemoryError()
            self.g_taint = nt
            nv = <uint64_t*>realloc(self.g_val, newcap * sizeof(uint64_t))
            if nv == NULL:
                raise MemoryError()
            self.g_val = nv
            memset(<void*>(self.g_taint + self.cap_slots), 0,
                   (newcap - self.cap_slots) * sizeof(uint64_t))
            memset(<void*>(self.g_val + self.cap_slots), 0,
                   (newcap - self.cap_slots) * sizeof(uint64_t))
            self.cap_slots = newcap
        self.g_taint[slot] = 0
        self.g_val[slot] = 0
        self.n_slots = slot + 1
        PyDict_SetItem(self.slot_map, name, slot)
        # Snapshots are sized to the old n_slots; drop them.  Slot INDICES stay
        # valid across growth, so the decode/slots part of each entry is kept.
        self._drop_snapshots()
        if self.arr_cache:
            self.arr_cache.clear()
        return slot

    cdef void _drop_snapshots(self):
        """Invalidate cached taint snapshots (they are sized to n_slots) while
        keeping decode + slot data, which stays valid when the arrays grow."""
        # The express lane's cached prefilters were resolved against the old
        # layout; bumping the generation retires all of them at once, and the
        # full path re-arms each address the next time it runs one.
        self.express_gen += 1
        cdef Py_ssize_t i
        cdef AddrEntry *e
        if self.addr_map.vals == NULL:
            return
        for i in range(self.addr_map.cap):
            if self.addr_map.keys[i] != MT_EMPTY_ADDR:
                e = self.addr_map.vals[i]
                if e != NULL:
                    e.have_snap = 0
                    e.have_val = 0

    cdef void _load_dict_to_arr(self):
        """register_taint (the external contract) -> g_taint.  Called once at the
        start of a run so externally seeded taint is picked up."""
        cdef object key, val
        cdef int slot
        memset(<void*>self.g_taint, 0, self.n_slots * sizeof(uint64_t))
        for key, val in self.register_taint.items():
            slot = self._slot_for(key)
            self.g_taint[slot] = <uint64_t>(<object>(int(val) & 0xFFFFFFFFFFFFFFFF))
        self.arr_loaded = True

    cpdef sync_taint_to_dict(self):
        """g_taint -> register_taint, and hand authority back to the dict.

        The wrapper's `register_taint` property calls this so the external
        contract is unchanged: tests seed the dict before a run, read it during or
        after, then seed it again.  Clearing arr_loaded means the NEXT instruction
        reloads the array from the dict, so an external re-seed is always picked
        up (otherwise a second run would silently ignore it)."""
        if not self.arr_loaded:
            return
        # In block mode the array stays authoritative: there is no next
        # instruction to reload it from the dict, so clearing the flag here
        # would make every read after the first hand back a stale snapshot.
        if not self.block_mode:
            self.arr_loaded = False
        cdef object name
        cdef object slot_obj
        cdef int slot
        cdef uint64_t v
        if PyDict_Size(self.register_taint):
            PyDict_Clear(self.register_taint)
        for name, slot_obj in self.slot_map.items():
            slot = <int>(<object>slot_obj)
            v = self.g_taint[slot]
            if v:
                PyDict_SetItem(self.register_taint, name,
                               PyLong_FromUnsignedLongLong(v))

    def __call__(self, uc, address, size, user_data):
        # Cython compiles this to a typed C function. The Python frame
        # for `__call__` is unavoidable (Unicorn calls it through a Python
        # binding), but everything inside is C.
        self._evaluate(<unsigned long long>address, <int>size)

    cdef inline _evaluate(self, unsigned long long address, int size):
        # Phase 1.3c: array-native state path (opt-in until proven).
        if _USE_ARR_HOOK:
            self._evaluate_arr(address, size)
            return
        cdef dict register_taint = self.register_taint

        # Early exit: no taint, no work. (Rare once any taint exists.)
        # NOTE: _any_taint check is implicit — if register_taint is empty
        # AND no shadow taint exists, the wrapper would not have armed
        # this hook. Once armed, we always run.

        # Read + decode the instruction.  On a repeat visit the address-keyed
        # decode cache returns the bytes and circuit decoded last time, skipping
        # uc_mem_read + the ctypes buffer slice + the cached_gen_rule tuplehash
        # (~3.0 us of the ~4.5 us cache-hit floor).  Keyed by (address, size); a
        # size change forces a fresh read+decode.
        cdef int err
        cdef bytes instruction_bytes
        cdef object circuit
        cdef AddrEntry *aent = NULL
        if _USE_DECODE_CACHE:
            dentry = self.py_decode_cache.get(address)
            if dentry is not None and (<int>(<object>dentry[0])) == size:
                instruction_bytes = <bytes>dentry[1]
                circuit = <object>dentry[2]
            else:
                err = self.uc_mem_read(self.uc_handle, address, self.mem_buf, size)
                if err == 0:
                    instruction_bytes = bytes(self.mem_buf[:size])
                else:
                    instruction_bytes = bytes(self.ql.mem.read(address, size))
                circuit = self.cached_gen_rule(self.arch, instruction_bytes, self.x64_format_key)
                self.py_decode_cache[address] = (size, instruction_bytes, circuit)
                # Extend the cached-code bounds so the mem-write hook can spot a
                # write that lands on any cached instruction (self-modifying code).
                if address < self.code_lo:
                    self.code_lo = address
                if address + <unsigned long long>size > self.code_hi:
                    self.code_hi = address + <unsigned long long>size
        else:
            err = self.uc_mem_read(self.uc_handle, address, self.mem_buf, size)
            if err == 0:
                instruction_bytes = bytes(self.mem_buf[:size])
            else:
                instruction_bytes = bytes(self.ql.mem.read(address, size))
            circuit = self.cached_gen_rule(self.arch, instruction_bytes, self.x64_format_key)

        # circuit._compiled is read fresh each call: it is lazily populated on the
        # first circuit.evaluate and can be invalidated inside evaluate, so the
        # decode cache stores the circuit object, not its compiled snapshot.
        cdef object compiled_circuit = circuit._compiled

        # Tier 3 fast path: per-address memoization.
        cdef object cache_key = None
        cdef object cached
        cdef object output_state
        cdef object key, val
        cdef long long val_int
        cdef object v_entry
        cdef unsigned long long live_version
        cdef unsigned long long out_version
        cdef dict input_snapshot
        cdef bint can_cache = (
            self.instr_cache_enabled
            and compiled_circuit is not None
            and compiled_circuit is not False
            and not compiled_circuit.has_mem_ops
        )

        # Value-independence fast path.  When the circuit's taint output is a
        # pure function of the input taints (mov/xor/lea/... -- it reads no
        # operand value), the cache can key on the taint signature alone and we
        # skip reading operand values entirely on a hit.  Only VALUE-dependent
        # circuits (and/or/add/... whose taint depends on operand values) need
        # the values read up-front for the value-aware key.  Non-cacheable
        # circuits fall through to the slow path, which reads pre_regs there.
        cdef bint value_indep = can_cache and (<object>compiled_circuit).value_independent
        cdef object pre_regs = None
        if can_cache and not value_indep:
            pre_regs = self._read_pre_regs(instruction_bytes, address)

        if can_cache:
            # Tier 4: version-cache fast path.
            #
            # Each entry stores (in_version, out_version, output_state,
            # input_snapshot).  in_version == taint_version is a fast first
            # check; input_snapshot equality is the actual correctness
            # guard.  Versions can collide because taint_version is a
            # mix of content-hashes (cacheable slow path, Tier-3 hit) and
            # monotonic increments (has-mem-ops slow path).  A collision
            # would otherwise replay output computed for a different
            # register_taint state and corrupt taint propagation; see
            # test_step3_tier4_cache.py for the concrete failure case.
            v_entry = self.instr_cache_v.get(address)
            live_version = self.taint_version
            if v_entry is not None and (<unsigned long long>(<object>v_entry[0])) == live_version:
                # Adopt the cached output only if BOTH the input taint
                # (v_entry[3]) AND the input register values (v_entry[4]) match.
                # The taint transfer is value-dependent, so a value change must
                # miss and recompute (soundness); two dict-equality calls.
                if (PyObject_RichCompareBool(<object>v_entry[3], register_taint, Py_EQ) == 1
                        and (value_indep
                             or PyObject_RichCompareBool(<object>v_entry[4], pre_regs, Py_EQ) == 1)):
                    output_state = <object>v_entry[2]
                    self.instr_cache_hits += 1
                    if self.last_tainted_writes:
                        self.last_tainted_writes.clear()
                    if PyDict_Size(register_taint):
                        PyDict_Clear(register_taint)
                    for key, val in output_state.items():
                        if val:
                            PyDict_SetItem(register_taint, key, val)
                    # Adopt the cached output_version: register_taint now has
                    # exactly the content that was assigned this version.
                    self.taint_version = <unsigned long long>(<object>v_entry[1])
                    return

            # Legacy frozenset-keyed cache (still useful for first-visit
            # cold paths and for cross-version equivalence).
            if PyDict_Size(register_taint) == 0:
                cache_key = EMPTY_FROZENSET
            else:
                cache_key = frozenset(register_taint.items())
            cached = self.instr_cache.get(address)
            if (cached is not None and cached[0] == cache_key
                    and (value_indep
                         or PyObject_RichCompareBool(<object>cached[2], pre_regs, Py_EQ) == 1)):
                # Cache hit: taint signature matches (and, for value-dependent
                # circuits, the operand values too), so replay.
                output_state = cached[1]
                self.instr_cache_hits += 1
                if self.last_tainted_writes:
                    self.last_tainted_writes.clear()
                # Snapshot register_taint BEFORE clearing, so the Tier-4
                # entry's input snapshot reflects the actual inputs this
                # output was computed for.
                input_snapshot = dict(register_taint)
                if PyDict_Size(register_taint):
                    PyDict_Clear(register_taint)
                # Populate the version cache too, so the next visit
                # at this address skips the frozenset construction.
                # We need a deterministic output_version derived from
                # output_state's content.  Use frozenset hash (one-shot
                # cost; only happens on cold version-cache misses).
                out_version = (<Py_ssize_t>hash(frozenset(output_state.items())))
                self.instr_cache_v[address] = (live_version, out_version, output_state, input_snapshot,
                                               None if value_indep else dict(pre_regs))
                # No-mem-ops circuits never produce MEM_ keys; just refill.
                for key, val in output_state.items():
                    if val:
                        PyDict_SetItem(register_taint, key, val)
                self.taint_version = out_version
                return
            self.instr_cache_misses += 1

        # Slow path.  For value-dependent (and non-cacheable) circuits pre_regs
        # was read above; for a value-INDEPENDENT circuit we skipped it on the
        # (fast) hit path, so read it now that we must actually evaluate.
        if pre_regs is None:
            pre_regs = self._read_pre_regs(instruction_bytes, address)
        # Snapshot register_taint for the EvalContext.
        cdef dict pre_taint = PyDict_Copy(register_taint)
        # Tell wrapper for AIW check + slow path consistency.
        self.wrapper._pre_regs = pre_regs
        self.wrapper._pre_taint = pre_taint

        # Evaluate the circuit. Catch ImplicitTaintError for SC/BOF reporting.
        try:
            output_state = None
            if _USE_DEVAL_C and compiled_circuit is not None and compiled_circuit is not False:
                # GIL-free C path for register-only, non-PC circuits (bit-identical
                # to evaluate; returns None to fall back for mem / PC / non-eligible).
                output_state = compiled_circuit.evaluate_c(pre_taint, pre_regs, self.sim._pcode)
                if output_state is None and _USE_CMEM:
                    # Memory circuits (loads/stores/mem-ALU): C-array interior with
                    # shadow read/write at the C level.  Bit-identical to
                    # circuit.evaluate; returns None to fall back (PC, wide SIMD,
                    # python fallback, missing shadow capsule).
                    output_state = compiled_circuit.evaluate_c_mem(
                        pre_taint, pre_regs, self.sim._pcode,
                        self.shadow_mem, self.read_live_memory)
                    if _DIFF_CMEM and output_state is not None:
                        # Soundness gate: prove evaluate_c_mem == do_evaluate for
                        # every memory instruction the suite exercises.
                        ref_ectx = self.eval_context_cls(
                            input_taint=pre_taint, input_values=pre_regs,
                            simulator=self.sim, implicit_policy=self.policy,
                            shadow_memory=self.shadow_mem, mem_reader=self.read_live_memory,
                        )
                        ref = circuit.evaluate(ref_ectx)
                        if ref != output_state:
                            raise AssertionError(
                                f'evaluate_c_mem mismatch @ {address:#x} '
                                f'({instruction_bytes.hex()}):\n  cmem={output_state}\n  ref ={ref}')
            if output_state is None:
                # Only the fallback path needs an EvalContext; evaluate_c reads
                # pre_taint / pre_regs directly, so for the common register-only
                # case we skip building (and discarding) a per-instruction object.
                ectx = self.eval_context_cls(
                    input_taint=pre_taint,
                    input_values=pre_regs,
                    simulator=self.sim,
                    implicit_policy=self.policy,
                    shadow_memory=self.shadow_mem,
                    mem_reader=self.read_live_memory,
                )
                output_state = circuit.evaluate(ectx)
        except BaseException as e:
            if isinstance(e, ImplicitTaintError):
                self._handle_implicit_taint(instruction_bytes, address, e)
                return
            raise

        # Cache the output for next time at this address with this taint sig.
        # We need a stable output_version derived from output_state content,
        # so that future cache hits can adopt this version atomically.
        cdef unsigned long long out_version_slow = 0
        cdef object value_snap
        if cache_key is not None:
            # Value-dependent circuits carry an input-VALUES snapshot so future
            # hits verify operand values match, not just the taint signature.
            # Value-INDEPENDENT circuits store None (their taint cannot change
            # with values, so the hit path skips the value check).
            value_snap = None if value_indep else dict(pre_regs)
            self.instr_cache[address] = (cache_key, dict(output_state), value_snap)
            # Compute output_version once (one frozenset hash; happens
            # ~167k times across the bench, not 1.19M).
            out_version_slow = (<Py_ssize_t>hash(frozenset(output_state.items())))
            # Tier-4 entry snapshots register_taint (pre_taint) AND (for
            # value-dependent circuits) the input values at instruction entry.
            # On future visits the hit path verifies both match the live state
            # before adopting the cached output_state, guarding against
            # taint_version collisions AND value-dependent taint changes.
            self.instr_cache_v[address] = (
                self.taint_version, out_version_slow,
                dict(output_state), dict(pre_taint), value_snap,
            )

        # Post-processing: clear writes set, clear register_taint, walk output.
        # This mutates register_taint to a new state — adopt the deterministic
        # output_version derived above (or fall back to a fresh increment if
        # the version cache wasn't populated, e.g. has_mem_ops circuit).
        if cache_key is not None:
            self.taint_version = out_version_slow
        else:
            self.taint_version += 1
        if self.last_tainted_writes:
            self.last_tainted_writes.clear()
        if PyDict_Size(register_taint):
            PyDict_Clear(register_taint)

        cdef list mem_writes = []
        cdef str skey, sbody
        cdef long mem_addr_l
        cdef int mem_size_i, ii
        cdef unsigned long long val_ll
        cdef BitPreciseShadowMemory shadow_mem = self.shadow_mem
        cdef set last_writes = self.last_tainted_writes
        cdef bint check_aiw = self.check_aiw

        for key, val in output_state.items():
            skey = <str>key
            if len(skey) >= 4 and skey[0] == 'M' and skey[1] == 'E' and skey[2] == 'M' and skey[3] == '_':
                # MEM_<hex>_<size>
                sbody = skey[4:]
                last = sbody.rfind('_')
                if last < 0:
                    continue
                try:
                    mem_addr_l = int(sbody[:last], 16)
                    mem_size_i = int(sbody[last + 1:])
                except (ValueError, OverflowError):
                    continue
                shadow_mem.write_mask(mem_addr_l, val, mem_size_i)
                if val:
                    # val may be a Python int up to 64 bits (0..0xFFFF_FFFF_FFFF_FFFF).
                    # Cast through unsigned long long, not signed long, to avoid
                    # OverflowError on values >= 2^63.
                    val_ll = <unsigned long long>(int(val) & 0xFFFFFFFFFFFFFFFFULL)
                    for ii in range(mem_size_i):
                        if (val_ll >> (ii * 8)) & 0xFF:
                            last_writes.add(mem_addr_l + ii)
                    if check_aiw:
                        mem_writes.append((mem_addr_l, mem_size_i, int(val)))
            elif val:
                PyDict_SetItem(register_taint, key, val)

        # AIW check (rare on most paths).
        if check_aiw and PyDict_Size(register_taint) and len(mem_writes) > 0:
            self._aiw_check(mem_writes, pre_regs, pre_taint, instruction_bytes, address)

    @property
    def regs_read(self):
        """Register values actually fetched from the CPU."""
        return self.fctx.regs_read

    @property
    def regs_circuit(self):
        """What the circuit's input set would have fetched for the same
        instructions.  `regs_circuit - regs_read` is what the compiled
        program's live-input set saved, at about 11 ns each."""
        return self.fctx.regs_circuit

    cdef void _attach_taint_ir(self, AddrEntry *ent, object instruction_bytes) noexcept:
        """Compile this instruction's taint program against the live slot map.

        Returns quietly on any refusal -- an unsupported p-code shape, a
        register with no slot yet, an emitter that declined.  The circuit path
        stays in place for all of them, so this can only ever be an
        acceleration, never a change of answer.
        """
        cdef unsigned long long addr_val
        try:
            got = _taint_ir_program_for(self.arch, instruction_bytes, self.slot_map)
        except BaseException:
            return
        if got is None:
            return
        cap, addr_val, accesses, writes_pc, live_regs = got
        if len(accesses) > 8:
            return
        # The capsule owns the emitted code; hold it for the hook's lifetime.
        self.taint_ir_progs[instruction_bytes] = cap
        ent.ir_n_acc = len(accesses)
        ent.ir_writes_pc = 1 if writes_pc else 0
        for i in range(len(accesses)):
            ent.ir_acc_kind[i] = <signed char>accesses[i][0]
            ent.ir_acc_size[i] = <signed char>accesses[i][1]
            ent.ir_acc_needval[i] = <signed char>accesses[i][2]
        ent.ir_fn = <void*><size_t>addr_val
        self._attach_ir_reg_read(ent, instruction_bytes, live_regs)

    cdef void _attach_ir_reg_read(self, AddrEntry *ent, object instruction_bytes,
                                  object live_regs) noexcept:
        """Build the reduced register read for a program that has been attached.

        The circuit reads every register its p-code mentions; the program reads
        only the ones whose VALUE survived its dead-code pass, which over the
        bank is 1.48 registers of 3.69.  At about 11 ns a register that is
        worth having, and it is the whole difference on an architecture whose
        instructions name three operands rather than two.

        A refusal here is not a refusal of the program: `have_ir_regs` stays 0
        and the express lane falls back to reading the full set."""
        cdef object arrs, ids, ptrs, vals, names, n_calls_int, ids_addr
        cdef object ptrs_addr, vals_addr
        cdef Py_ssize_t n, i
        cdef bint need_ef
        ent.have_ir_regs = 0
        if ent.ir_slots != NULL:
            free(ent.ir_slots); ent.ir_slots = NULL
        ent.ir_n_in = 0
        try:
            arrs = self.build_offsets_arrs(frozenset(live_regs))
            (ids, vals, ptrs, n, names, need_ef, n_calls,
             ids_addr, ptrs_addr, vals_addr, n_calls_int) = arrs
        except BaseException:
            return
        # Hold the ctypes buffers for as long as their addresses are cached.
        self.taint_ir_reg_arrs[instruction_bytes] = arrs
        if ids is None or n == 0:
            # Nothing to read at all -- a rule that only routes taint.  That is
            # a valid descriptor, and the best one: no batch call is made.
            ent.ir_ids_addr = 0
            ent.ir_ptrs_addr = 0
            ent.ir_vals_addr = 0
            ent.ir_n_calls_i = 0
            ent.ir_need_ef = 1 if need_ef else 0
            ent.have_ir_regs = 1
            return
        ent.ir_slots = <int*>malloc(<size_t>n * sizeof(int))
        if ent.ir_slots == NULL:
            return
        for i in range(n):
            ent.ir_slots[i] = self._slot_for(names[i])
        ent.ir_n_in = <int>n
        ent.ir_ids_addr = <unsigned long long>ids_addr
        ent.ir_ptrs_addr = <unsigned long long>ptrs_addr
        ent.ir_vals_addr = <unsigned long long>vals_addr
        ent.ir_n_calls_i = <int>n_calls_int
        ent.ir_need_ef = 1 if need_ef else 0
        ent.have_ir_regs = 1

    cdef int _fill_vals_arr(self, bytes instruction_bytes,
                            unsigned long long address) except -1:
        """Read this instruction's live input-register values straight into the
        slot-indexed g_val array.  Same C reg-read boundary as _read_pre_regs, but
        it writes uint64 slots instead of building a Python dict (the ~938 ns
        pre_regs dict build disappears)."""
        cdef object decoded, uc_arrs, ids, vals, ptrs, names, n_calls
        cdef object ids_addr, ptrs_addr, vals_addr, n_calls_int, slots
        cdef object fname, fbit, k, v, pre
        cdef AddrEntry *ent
        cdef Py_ssize_t n, i
        cdef bint need_ef
        cdef uint64_t *vptr
        cdef uint64_t ef
        cdef list slist
        cdef object entry
        try:
            # Per-address cache of (slots, uc_arrays).  Both are derived purely
            # from the instruction bytes, and invalidate_smc drops this along with
            # the other address-keyed caches when those bytes change.  Caching
            # uc_arrays as well avoids calling get_decoded() -- an lru_cache keyed
            # on (arch, bytes), so a bytes hash plus wrapper -- on EVERY
            # instruction just to re-read the same _uc_arrays attribute.
            # (The slot list has to live here rather than on `decoded` because
            # that is a cdef class and will not take a new attribute.)
            ent = mt_am_get(&self.addr_map, address)
            if ent != NULL and ent.have_regs:
                # Steady state: everything the read needs is already in the C
                # entry, so nothing here touches a Python object.
                if _USE_CREGS and self.uc_rrb_addr != 0 and \
                        (ent.n_in == 0 or ent.slots != NULL):
                    if ent.n_in > 0:
                        if self.uc_handle_addr == 0:
                            self.uc_handle_addr = <unsigned long long>self.uc_handle.value
                        (<uc_reg_read_batch_ft>(<void*>self.uc_rrb_addr))(
                            <void*>self.uc_handle_addr,
                            <void*>ent.ids_addr,
                            <void*>ent.ptrs_addr,
                            ent.n_calls_i)
                        vptr = <uint64_t*>(<void*>ent.vals_addr)
                        for i in range(ent.n_in):
                            self.g_val[ent.slots[i]] = vptr[i]
                    if ent.need_ef:
                        self._explode_eflags()
                    if self.rip_slot < 0:
                        self.rip_slot = self._slot_for(self._pc_name())
                    self.g_val[self.rip_slot] = <uint64_t>address
                    return 0
                # No registers to read, or the C batch read is unavailable: fall
                # through to the general path below, which re-derives from
                # uc_arrs.  Rare, so it is not worth a second specialisation.
            if ent != NULL and ent.have_slots and ent.uc_arrs != NULL:
                uc_arrs = <object>ent.uc_arrs
                (ids, vals, ptrs, n, names, need_ef, n_calls,
                 ids_addr, ptrs_addr, vals_addr, n_calls_int) = uc_arrs
            else:
                decoded = self.get_decoded(self.arch, instruction_bytes)
                uc_arrs = decoded._uc_arrays
                if uc_arrs is None:
                    uc_arrs = self.build_offsets_arrs(decoded.input_reg_offsets)
                    decoded._uc_arrays = uc_arrs
                (ids, vals, ptrs, n, names, need_ef, n_calls,
                 ids_addr, ptrs_addr, vals_addr, n_calls_int) = uc_arrs
                if ent == NULL:
                    ent = mt_am_new(&self.addr_map, address)
                if ent != NULL:
                    Py_XINCREF(<PyObject*>uc_arrs)
                    Py_XDECREF(ent.uc_arrs)
                    ent.uc_arrs = <PyObject*>uc_arrs
                    if ent.slots != NULL:
                        free(ent.slots); ent.slots = NULL
                    ent.n_in = 0
                    if ids is not None and n > 0:
                        ent.slots = <int*>malloc(<size_t>n * sizeof(int))
                        if ent.slots != NULL:
                            for i in range(n):
                                ent.slots[i] = self._slot_for(names[i])
                            ent.n_in = <int>n
                    ent.have_slots = 1
                    # Unpack the parts the steady-state read needs into the
                    # entry, so later visits never touch the tuple again.
                    if (ids is not None and ent.slots != NULL and ent.n_in == <int>n
                            and ids_addr is not None and ptrs_addr is not None
                            and vals_addr is not None):
                        ent.ids_addr = <unsigned long long>ids_addr
                        ent.ptrs_addr = <unsigned long long>ptrs_addr
                        ent.vals_addr = <unsigned long long>vals_addr
                        ent.n_calls_i = <int>n_calls_int
                        ent.need_ef = 1 if need_ef else 0
                        ent.have_regs = 1
            if ids is not None:
                if _USE_CREGS and self.uc_rrb_addr != 0:
                    if self.uc_handle_addr == 0:
                        self.uc_handle_addr = <unsigned long long>self.uc_handle.value
                    (<uc_reg_read_batch_ft>(<void*>self.uc_rrb_addr))(
                        <void*>self.uc_handle_addr,
                        <void*>(<unsigned long long>ids_addr),
                        <void*>(<unsigned long long>ptrs_addr),
                        <int>n_calls_int)
                    vptr = <uint64_t*>(<void*>(<unsigned long long>vals_addr))
                    if ent != NULL and ent.slots != NULL:
                        for i in range(n):
                            self.g_val[ent.slots[i]] = vptr[i]
                else:
                    self.uc_reg_read_batch(self.uc_handle, ids, ptrs, n_calls)
                    if ent != NULL and ent.slots != NULL:
                        for i in range(n):
                            self.g_val[ent.slots[i]] = \
                                <uint64_t>(int(vals[i]) & 0xFFFFFFFFFFFFFFFF)
                if need_ef:
                    self._explode_eflags()
        except Exception:  # noqa: BLE001 - mirror _read_pre_regs' fallback
            pre = self.get_live_registers(self.uc_handle)
            for k, v in pre.items():
                self.g_val[self._slot_for(k)] = \
                    <uint64_t>(int(v) & 0xFFFFFFFFFFFFFFFF)
        if self.rip_slot < 0:
            self.rip_slot = self._slot_for(self._pc_name())
        self.g_val[self.rip_slot] = <uint64_t>address
        return 0

    cdef void _explode_eflags(self):
        """Scatter the flag bits of the packed flags register into their own
        value slots.  The (slot, bit) pairs are derived once from the arch's flag
        map and then live in C arrays, so the per-instruction work is a shift and
        a store per flag with no tuple indexing."""
        cdef int i
        cdef uint64_t ef
        if self.eflags_slot < 0:
            if self.flag_parent is None:
                # MIPS and RISC-V have no packed flags register: compares write
                # a general register instead, so there is nothing to scatter.
                self.eflags_slot = 0
                self.eflags_slot_bits = []
            else:
                self.eflags_slot = self._slot_for(self.flag_parent)
                self.eflags_slot_bits = [
                    (self._slot_for(_f), int(_b)) for _f, _b in self.eflags_bits.items()]
            self.ef_n = 0
            if self.ef_slots != NULL:
                free(self.ef_slots); self.ef_slots = NULL
            if self.ef_bits != NULL:
                free(self.ef_bits); self.ef_bits = NULL
            i = len(self.eflags_slot_bits)
            if i > 0:
                self.ef_slots = <int*>malloc(<size_t>i * sizeof(int))
                self.ef_bits = <int*>malloc(<size_t>i * sizeof(int))
                if self.ef_slots != NULL and self.ef_bits != NULL:
                    for i, fpair in enumerate(self.eflags_slot_bits):
                        self.ef_slots[i] = <int>(<object>fpair[0])
                        self.ef_bits[i] = <int>(<object>fpair[1])
                    self.ef_n = len(self.eflags_slot_bits)
        ef = self.g_val[self.eflags_slot]
        if self.ef_n > 0:
            for i in range(self.ef_n):
                self.g_val[self.ef_slots[i]] = (ef >> self.ef_bits[i]) & 1
        else:
            for fpair in self.eflags_slot_bits:
                self.g_val[<int>(<object>fpair[0])] = \
                    (ef >> <int>(<object>fpair[1])) & 1

    cdef void _init_mem_ctx(self):
        """Populate the C read context from the LiveMemReader that already
        resolved the uc_mem_read fnptr and buffer address.  If anything is
        missing the context stays zeroed and _mt_mem_read_c declines, so
        circuit_c falls back to the Python reader."""
        cdef LiveMemReader r
        self.mem_ctx.uc_handle = 0
        self.mem_ctx.uc_mr_addr = 0
        self.mem_ctx.buf_addr = 0
        try:
            if isinstance(self.read_live_memory, LiveMemReader):
                r = <LiveMemReader>self.read_live_memory
                self.mem_ctx.uc_handle = r.uc_handle
                self.mem_ctx.uc_mr_addr = r.uc_mr_addr
                self.mem_ctx.buf_addr = r.mem_buf_addr
        except Exception:  # noqa: BLE001
            pass
        self.mem_ctx_ready = True

    cdef object _pc_name(self):
        """PC register name for this arch (the wrapper owns the mapping)."""
        return self.pc_reg_name

    cdef inline _evaluate_arr(self, unsigned long long address, int size):
        """Array-native hot path (Phase 1.3c).  Taint and input values live in
        slot-indexed uint64 C arrays; eval writes them in place via
        evaluate_c_arr_ptr / evaluate_c_mem_ptr; the output cache is keyed by
        memcmp of an array snapshot.  No per-instruction dict is built."""
        if not self.arr_loaded:
            self._load_dict_to_arr()
        self.instr_total += 1

        cdef int err
        cdef bytes instruction_bytes
        # Typed so `circuit._compiled` below is a C struct read (see ast.pxd),
        # not an attribute lookup.  The assignment is type-checked by Cython, so
        # a rule generator returning anything else raises rather than misreads.
        cdef LogicCircuit circuit
        cdef AddrEntry *aent = NULL
        if _USE_DECODE_CACHE:
            aent = mt_am_get(&self.addr_map, address)
            if aent != NULL and aent.size == size and aent.circuit != NULL:
                instruction_bytes = <bytes>(<object>aent.instr_bytes)
                circuit = <object>aent.circuit
            else:
                err = self.uc_mem_read(self.uc_handle, address, self.mem_buf, size)
                if err == 0:
                    instruction_bytes = bytes(self.mem_buf[:size])
                else:
                    instruction_bytes = bytes(self.ql.mem.read(address, size))
                circuit = self.cached_gen_rule(self.arch, instruction_bytes, self.x64_format_key)
                aent = mt_am_new(&self.addr_map, address)
                if aent != NULL:
                    Py_XINCREF(<PyObject*>instruction_bytes)
                    Py_XDECREF(aent.instr_bytes)
                    aent.instr_bytes = <PyObject*>instruction_bytes
                    Py_XINCREF(<PyObject*>circuit)
                    Py_XDECREF(aent.circuit)
                    aent.circuit = <PyObject*>circuit
                    aent.size = size
                    # different bytes at this address: slots + snapshots are stale
                    aent.have_slots = 0
                    aent.have_regs = 0
                    aent.have_snap = 0
                    aent.have_val = 0
                    aent.ir_fn = NULL
                    aent.ir_tried = 0
                    aent.ir_n_acc = 0
                    aent.ir_writes_pc = 0
                    # The express lane's cache describes the PREVIOUS bytes at
                    # this address; different bytes make every part of it wrong.
                    aent.express = 0
                if address < self.code_lo:
                    self.code_lo = address
                if address + <unsigned long long>size > self.code_hi:
                    self.code_hi = address + <unsigned long long>size
        else:
            err = self.uc_mem_read(self.uc_handle, address, self.mem_buf, size)
            if err == 0:
                instruction_bytes = bytes(self.mem_buf[:size])
            else:
                instruction_bytes = bytes(self.ql.mem.read(address, size))
            circuit = self.cached_gen_rule(self.arch, instruction_bytes, self.x64_format_key)

        # A taint program compiled from this instruction's p-code replaces the
        # whole per-output circuit evaluation with one call.  Resolved here
        # rather than in C because building it is Python; retried while it is
        # only waiting on a register slot the engine has not interned yet.
        if _taint_ir_on and aent != NULL and aent.ir_fn == NULL and aent.ir_tried < 4:
            aent.ir_tried += 1
            self._attach_taint_ir(aent, instruction_bytes)

        # circuit._compiled is read fresh each call: it is lazily populated on the
        # first evaluate and can be invalidated inside it, so the decode cache
        # stores the circuit, not its compiled snapshot.  Through ast.pxd this is
        # a C struct read.
        cdef object compiled_circuit = circuit._compiled
        cdef bint compiled_ok = (compiled_circuit is not None and compiled_circuit is not False)
        cdef int cflags = 0
        cdef bint has_mem
        if compiled_ok and _circuit_capi != NULL:
            cflags = _circuit_capi.compiled_flags(compiled_circuit)
            has_mem = (cflags & MT_CF_HAS_MEM_OPS) != 0
        else:
            has_mem = compiled_ok and compiled_circuit.has_mem_ops
        cdef bint can_cache = self.instr_cache_enabled and compiled_ok and not has_mem
        cdef bint value_indep
        if can_cache and _circuit_capi != NULL:
            value_indep = (cflags & MT_CF_VALUE_INDEP) != 0
        else:
            value_indep = can_cache and (<object>compiled_circuit).value_independent

        # ---- C hot path -----------------------------------------------
        # Everything from here to the evaluator result is plain C (fastpath.h).
        # It returns MT_FAST_SLOW for anything needing Python -- a first visit
        # whose entry is not prepared, a lazily-interned slot that does not
        # exist yet, or an evaluator that declined -- and in that case it has
        # committed nothing to the taint array, so the fallback below starts
        # from an unmodified state.
        cdef int frc = MT_FAST_SLOW
        if compiled_ok and aent != NULL and _circuit_capi != NULL:
            if not self.mem_ctx_ready:
                self._init_mem_ctx()
            try:
                frc = mt_fast_step(&self.fctx, address, aent, compiled_circuit, cflags)
            except BaseException as e:
                # Same handling the Python eval path gets: an implicit-taint
                # report is a policy decision, not a failure.
                if isinstance(e, ImplicitTaintError):
                    self._handle_implicit_taint(instruction_bytes, address, e)
                    return
                raise
            if frc == MT_FAST_DONE:
                if self.last_tainted_writes:
                    self.last_tainted_writes.clear()
                if self.check_aiw and self.ltw_n > 0:
                    self._aiw_check_from_ltw(instruction_bytes, address)
                # This address just completed entirely in C, so everything the
                # express lane would need is resolved.  Cache it, and the next
                # visit skips the GIL and this whole prologue.
                if _EXPRESS:
                    mt_express_arm(&self.fctx, aent, compiled_circuit, cflags)
                return

        cdef Py_ssize_t nbytes = self.n_slots * sizeof(uint64_t)
        cdef Py_ssize_t i
        cdef bint hit = False
        cdef bint vals_filled = False

        # Values are needed for evaluation and for the value-aware cache key of
        # value-dependent circuits.  Value-INDEPENDENT circuits skip the read on a
        # cache hit (same optimisation the dict path makes).
        if not value_indep:
            self._fill_vals_arr(instruction_bytes, address)
            vals_filled = True

        # ---- output cache probe (C snapshots on the per-address entry) ----
        if can_cache and aent != NULL and aent.have_snap and aent.snap_n == self.n_slots:
            if memcmp(<void*>self.g_taint, <void*>aent.in_snap, nbytes) == 0:
                hit = True
                if aent.have_val:
                    # A value-dependent circuit must re-check the operand values;
                    # n_in == 0 would make that check vacuous, so such entries are
                    # never stored with have_val set.
                    for i in range(aent.n_in):
                        if self.g_val[aent.slots[i]] != aent.val_snap[i]:
                            hit = False
                            break
                if hit:
                    memcpy(<void*>self.g_taint, <void*>aent.out_snap, nbytes)
                    self.instr_cache_hits += 1
                    self.ltw_n = 0
                    if self.last_tainted_writes:
                        self.last_tainted_writes.clear()
                    return
        if can_cache:
            self.instr_cache_misses += 1

        if not vals_filled:
            self._fill_vals_arr(instruction_bytes, address)
        # _fill_vals_arr may have created the entry (or grown the slot arrays)
        if aent == NULL:
            aent = mt_am_get(&self.addr_map, address)

        # Snapshot the pre-state taint (and operand values) BEFORE eval, while
        # g_taint still holds the inputs; eval rewrites it in place.
        cdef bint want_store = False
        if can_cache and aent != NULL:
            if not value_indep and (not aent.have_slots or aent.n_in <= 0):
                # no operand-value snapshot possible -> do not cache, else a later
                # hit would ignore changed values
                want_store = False
            else:
                if aent.snap_n != self.n_slots:
                    if aent.in_snap != NULL: free(aent.in_snap)
                    if aent.out_snap != NULL: free(aent.out_snap)
                    aent.in_snap = <uint64_t*>malloc(<size_t>nbytes)
                    aent.out_snap = <uint64_t*>malloc(<size_t>nbytes)
                    aent.snap_n = self.n_slots if (aent.in_snap != NULL and aent.out_snap != NULL) else 0
                if aent.snap_n == self.n_slots:
                    # Same rule as fastpath.h: in_snap is written before the
                    # evaluation and out_snap only after one that commits, so
                    # the entry is invalid in between.  Leaving have_snap set
                    # pairs a new input with a stale output, and the next probe
                    # replays it -- dropping the implicit-taint check for a
                    # branch whose evaluation deliberately did not commit.
                    aent.have_snap = 0
                    memcpy(<void*>aent.in_snap, <void*>self.g_taint, nbytes)
                    if value_indep:
                        aent.have_val = 0
                    else:
                        if aent.val_snap == NULL:
                            aent.val_snap = <uint64_t*>malloc(<size_t>aent.n_in * sizeof(uint64_t))
                        if aent.val_snap != NULL:
                            for i in range(aent.n_in):
                                aent.val_snap[i] = self.g_val[aent.slots[i]]
                            aent.have_val = 1
                        else:
                            aent.have_val = 0
                    want_store = (value_indep or aent.have_val)

        # Memory writes come back in C storage on the stack; nothing is boxed.
        cdef MtMemWrite mw[MT_MAX_MEM_WRITES]
        cdef int n_mw = 0
        cdef int rc = MT_EVAL_DECLINED
        cdef object result = None
        cdef object output_state = None
        cdef bint used_arr = False
        cdef bint used_capi = False
        try:
            if compiled_ok:
                if has_mem:
                    if _USE_CMEM:
                        if _circuit_capi != NULL:
                            if not self.mem_ctx_ready:
                                self._init_mem_ctx()
                            rc = _circuit_capi.eval_mem_ptr_ci(
                                compiled_circuit, self.g_taint, self.g_val,
                                self.n_slots, self.pcode_obj, self.shadow_mem,
                                self.read_live_memory, self.slot_map,
                                _mt_mem_read_c, <void*>&self.mem_ctx,
                                mw, MT_MAX_MEM_WRITES, self.fctx.implicit_policy)
                            # rc >= 0 is a real write count.  Every negative code
                            # (DECLINED, ERROR, PC_REPORT) means the C path did
                            # NOT commit and Python must run the instruction --
                            # testing only for DECLINED let MT_EVAL_PC_REPORT
                            # read as success and silently dropped the
                            # implicit-taint report for tainted `ret`.
                            used_arr = rc >= 0
                            used_capi = used_arr
                            if used_arr:
                                n_mw = rc
                        else:
                            result = compiled_circuit.evaluate_c_mem_ptr(
                                <unsigned long long>(<size_t>self.g_taint),
                                <unsigned long long>(<size_t>self.g_val),
                                self.n_slots, self.pcode_obj, self.shadow_mem,
                                self.read_live_memory, self.slot_map)
                            used_arr = result is not None
                else:
                    if _circuit_capi != NULL:
                        rc = _circuit_capi.eval_arr_ptr_i(
                            compiled_circuit, self.g_taint, self.g_val,
                            self.n_slots, self.pcode_obj, self.slot_map,
                            self.fctx.implicit_policy)
                        used_arr = rc >= 0
                    else:
                        result = compiled_circuit.evaluate_c_arr_ptr(
                            <unsigned long long>(<size_t>self.g_taint),
                            <unsigned long long>(<size_t>self.g_val),
                            self.n_slots, self.pcode_obj, self.slot_map)
                        used_arr = result is not None
            if not used_arr:
                # `fb_other` used to hold two different things: an instruction
                # with no compiled circuit at all, and one that had a circuit
                # and still declined for a reason nothing here names.  They ask
                # for different fixes, so they are counted apart.
                if not compiled_ok:
                    self.fb_nocircuit += 1
                elif cflags & MT_CF_PC_TARGET:
                    self.fb_pc += 1
                elif cflags & MT_CF_PY_FALLBACK:
                    self.fb_pyfall += 1
                elif has_mem:
                    self.fb_mem += 1
                else:
                    self.fb_other += 1
                output_state = self._eval_fallback_dict(
                    circuit, compiled_circuit, compiled_ok, instruction_bytes, address)
        except BaseException as e:
            if isinstance(e, ImplicitTaintError):
                self._handle_implicit_taint(instruction_bytes, address, e)
                return
            raise

        # ---- apply ----
        self.ltw_n = 0
        if self.last_tainted_writes:
            self.last_tainted_writes.clear()
        # Only the AIW check reads this, so build the list only when it is on;
        # otherwise every instruction allocated (and GC-tracked) an empty list
        # that nothing ever looked at.  All three appenders are check_aiw-guarded.
        cdef list mem_writes = [] if self.check_aiw else None
        if used_arr:
            if has_mem:
                if used_capi:
                    self._apply_mem_writes_c(mw, n_mw, mem_writes)
                else:
                    self._apply_mem_writes(result, mem_writes)
        else:
            self._apply_output_state_to_arr(output_state, mem_writes)

        if want_store and used_arr and aent != NULL and aent.snap_n == self.n_slots:
            memcpy(<void*>aent.out_snap, <void*>self.g_taint, nbytes)
            aent.have_snap = 1

        if self.check_aiw and len(mem_writes) > 0:
            self._aiw_check_arr(mem_writes, instruction_bytes, address)

    cdef object _eval_fallback_dict(self, object circuit, object compiled_circuit,
                                    bint compiled_ok, bytes instruction_bytes,
                                    unsigned long long address):
        """Rare path: the array evals declined (SIMD VL_ lanes not interned, PC
        write, python fallback...).  Run the existing dict evaluation and let the
        caller fold the result back into the arrays.  Correct, just not fast."""
        self.arr_fallbacks += 1
        cdef dict pre_taint = {}
        cdef object name, slot_obj
        cdef int slot
        cdef uint64_t v
        for name, slot_obj in self.slot_map.items():
            v = self.g_taint[<int>(<object>slot_obj)]
            if v:
                pre_taint[name] = PyLong_FromUnsignedLongLong(v)
        cdef object pre_regs = self._read_pre_regs(instruction_bytes, address)
        self.wrapper._pre_regs = pre_regs
        self.wrapper._pre_taint = pre_taint
        cdef object output_state = None
        if _USE_DEVAL_C and compiled_ok:
            output_state = compiled_circuit.evaluate_c(pre_taint, pre_regs, self.sim._pcode)
            if output_state is None and _USE_CMEM:
                output_state = compiled_circuit.evaluate_c_mem(
                    pre_taint, pre_regs, self.sim._pcode,
                    self.shadow_mem, self.read_live_memory)
        if output_state is None:
            ectx = self.eval_context_cls(
                input_taint=pre_taint,
                input_values=pre_regs,
                simulator=self.sim,
                implicit_policy=self.policy,
                shadow_memory=self.shadow_mem,
                mem_reader=self.read_live_memory,
            )
            output_state = circuit.evaluate(ectx)
        return output_state

    cdef void _apply_output_state_to_arr(self, object output_state, list mem_writes):
        """Fold a dict output_state (full post-state + MEM_ keys) back into
        g_taint / the shadow.  Mirrors the dict path's clear-then-refill."""
        cdef object key, val
        cdef str skey, sbody
        cdef long mem_addr_l
        cdef int mem_size_i, ii, last
        cdef unsigned long long val_ll
        cdef BitPreciseShadowMemory shadow_mem = self.shadow_mem
        cdef set last_writes = self.last_tainted_writes
        if output_state is None:
            return
        memset(<void*>self.g_taint, 0, self.n_slots * sizeof(uint64_t))
        for key, val in output_state.items():
            skey = <str>key
            if len(skey) >= 4 and skey[0] == 'M' and skey[1] == 'E' and skey[2] == 'M' and skey[3] == '_':
                sbody = skey[4:]
                last = sbody.rfind('_')
                if last < 0:
                    continue
                try:
                    mem_addr_l = int(sbody[:last], 16)
                    mem_size_i = int(sbody[last + 1:])
                except (ValueError, OverflowError):
                    continue
                shadow_mem.write_mask(mem_addr_l, val, mem_size_i)
                if val:
                    val_ll = <unsigned long long>(int(val) & 0xFFFFFFFFFFFFFFFFULL)
                    for ii in range(mem_size_i):
                        if (val_ll >> (ii * 8)) & 0xFF:
                            last_writes.add(mem_addr_l + ii)
                    if self.check_aiw:
                        mem_writes.append((mem_addr_l, mem_size_i, int(val)))
            elif val:
                self.g_taint[self._slot_for(key)] = \
                    <uint64_t>(int(val) & 0xFFFFFFFFFFFFFFFFULL)

    cdef void _aiw_after_express(self, unsigned long long address):
        """The AIW check for an instruction the GIL-free lane already committed.

        The lane cannot run this itself (it builds dicts and reports), so it
        commits the taint, records the tainted stores in ltw, and asks for the
        GIL only when there are some.  The instruction bytes come back off the
        address entry rather than being passed down, so the lane's signature
        stays free of Python."""
        cdef AddrEntry *e = mt_am_get(&self.addr_map, address)
        if e == NULL or e.instr_bytes == NULL or self.ltw_n <= 0:
            return
        self._aiw_check_from_ltw(<bytes>(<object>e.instr_bytes), address)

    cdef _aiw_check_from_ltw(self, bytes instruction_bytes,
                             unsigned long long address):
        """AIW check for an instruction the C path handled.

        The C path deliberately builds no Python list; the writes it recorded
        carry the same (addr, size, mask) triples the list used to, so the list
        is materialised here, only when the check is actually enabled."""
        cdef list mem_writes = []
        cdef int k
        for k in range(self.ltw_n):
            mem_writes.append((self.ltw[k].addr, self.ltw[k].size, self.ltw[k].taint))
        if mem_writes:
            self._aiw_check_arr(mem_writes, instruction_bytes, address)

    cdef _aiw_check_arr(self, list mem_writes, bytes instruction_bytes,
                        unsigned long long address):
        """AIW needs dicts; build them lazily here (rare) rather than per
        instruction."""
        cdef dict pre_taint = {}
        cdef object name, slot_obj
        cdef uint64_t v
        for name, slot_obj in self.slot_map.items():
            v = self.g_taint[<int>(<object>slot_obj)]
            if v:
                pre_taint[name] = PyLong_FromUnsignedLongLong(v)
        if not pre_taint:
            return
        cdef object pre_regs = self._read_pre_regs(instruction_bytes, address)
        self._aiw_check(mem_writes, pre_regs, pre_taint, instruction_bytes, address)

    cdef void _apply_mem_writes_c(self, MtMemWrite *writes, int n, list mem_writes):
        """C-struct form of _apply_mem_writes: same effect, nothing boxed.

        The shadow was already written inside the evaluator; what remains is to
        record which bytes this instruction legitimately tainted, so the
        UC_HOOK_MEM_WRITE callback does not immediately clear them again.  The
        (addr, size, mask) triples are kept as-is rather than expanded to one
        entry per byte: MemWriteHook can answer the same membership question
        from them directly."""
        cdef int k, ii
        cdef uint64_t t
        for k in range(n):
            t = writes[k].taint
            if t == 0:
                continue
            if self.ltw_n < MT_MAX_MEM_WRITES:
                self.ltw[self.ltw_n] = writes[k]
                self.ltw_n += 1
            else:
                # Cannot happen today (the evaluator declines above
                # MT_MAX_MEM_WRITES), but dropping an entry would let the write
                # hook clear taint this instruction legitimately produced, i.e.
                # a silent under-taint.  Spill to the unbounded set instead.
                for ii in range(writes[k].size):
                    if (t >> (ii * 8)) & 0xFF:
                        self.last_tainted_writes.add(writes[k].addr + <uint64_t>ii)
            if self.check_aiw:
                mem_writes.append((writes[k].addr, writes[k].size, t))

    cdef void _apply_mem_writes(self, object writes, list mem_writes):
        """Feed last_tainted_writes (+ AIW list) from evaluate_c_mem_ptr's
        (addr, size, taint) results.  The shadow itself was already written at the
        C level, so there is no MEM_<hex>_<size> string round-trip."""
        cdef object w
        cdef long mem_addr
        cdef int msize, ii
        cdef unsigned long long val_ll
        cdef set last_writes = self.last_tainted_writes
        if writes is None:
            return
        for w in writes:
            mem_addr = <long>(<object>w[0])
            msize = <int>(<object>w[1])
            val_ll = <unsigned long long>(int(<object>w[2]) & 0xFFFFFFFFFFFFFFFFULL)
            if val_ll:
                for ii in range(msize):
                    if (val_ll >> (ii * 8)) & 0xFF:
                        last_writes.add(mem_addr + ii)
                if self.check_aiw:
                    mem_writes.append((mem_addr, msize, int(<object>w[2])))

    @property
    def decode_cache(self):
        """Introspection view of the address-keyed decode cache.

        The array path keeps decode entries in a C table so the hot path holds no
        Python dict; this rebuilds a {address: size} mapping on demand so tests
        and tools can still verify the cache is populated and invalidated.  Built
        only when read, never on the hot path.  With MICROTAINT_ARR_HOOK=0 the
        dict path's own cache is returned instead.
        """
        cdef Py_ssize_t i
        cdef dict out
        if not _USE_ARR_HOOK:
            return self.py_decode_cache
        out = {}
        if self.addr_map.vals != NULL:
            for i in range(self.addr_map.cap):
                if self.addr_map.keys[i] != MT_EMPTY_ADDR and self.addr_map.vals[i] != NULL:
                    out[self.addr_map.keys[i]] = self.addr_map.vals[i].size
        return out

    def code_range_addrs(self) -> tuple:
        """Addresses of `code_lo` and `code_hi`, for the block runtime.

        Block mode plans blocks the instruction hook never decodes, so it has
        to widen the same range: it is what the mem-write hook's
        self-modifying-code guard tests, on both the nogil and the GIL path.
        """
        return (<unsigned long long>&self.code_lo,
                <unsigned long long>&self.code_hi)

    cpdef void invalidate_smc(self):
        """Drop every address-keyed cache after a write hit cached code.

        Called by the mem-write hook when a guest write intersects
        [code_lo, code_hi).  The decode cache would otherwise replay the
        pre-write bytes/circuit, and the Tier-3/Tier-4 output caches would
        replay taint computed for the old instruction, so all three must go;
        the rewritten instruction is re-read and re-decoded on its next
        execution.  Rare (only self-modifying / JIT'd code writes into the
        code range), so a full clear + lazy rebuild is fine.
        """
        if self.py_decode_cache:
            self.py_decode_cache.clear()
        if self.instr_cache_v:
            self.instr_cache_v.clear()
        if self.instr_cache:
            self.instr_cache.clear()
        # The array path's output cache is address-keyed too, so it replays taint
        # computed for the pre-write instruction unless dropped here.
        if self.arr_cache:
            self.arr_cache.clear()
        if self.slots_cache:
            self.slots_cache.clear()
        mt_am_clear(&self.addr_map)
        self.code_lo = 0xFFFFFFFFFFFFFFFF
        self.code_hi = 0
        # Block mode's plans are keyed by (address, size) in a separate module,
        # and it is what widened the range that brought us here.
        if self.block_invalidate is not None:
            self.block_invalidate()

    cdef object _read_pre_regs(self, bytes instruction_bytes, unsigned long long address):
        """Read this instruction's live input-register values into a dict (with
        RIP set to the runtime PC).  Used for the value-aware cache key of
        value-dependent circuits and by the slow-path evaluation.  Skipped for
        value-independent circuits on a cache hit."""
        cdef object pre_regs, decoded, uc_arrs, ids, vals, ptrs, names, ef, n_calls, fname, fbit
        cdef object ids_addr, ptrs_addr, vals_addr, n_calls_int
        cdef Py_ssize_t n_slots, i_slot
        cdef bint need_ef
        cdef uint64_t* vptr
        try:
            decoded = self.get_decoded(self.arch, instruction_bytes)
            uc_arrs = decoded._uc_arrays
            if uc_arrs is None:
                uc_arrs = self.build_offsets_arrs(decoded.input_reg_offsets)
                decoded._uc_arrays = uc_arrs
            (ids, vals, ptrs, n_slots, names, need_ef, n_calls,
             ids_addr, ptrs_addr, vals_addr, n_calls_int) = uc_arrs
            if ids is None:
                pre_regs = {}
            elif _USE_CREGS and self.uc_rrb_addr != 0:
                # C-level boundary: call uc_reg_read_batch through the fnptr and
                # read the value array via a uint64* -- no ctypes ffi/ConvParam,
                # no per-slot ctypes indexing.  Identical to the ctypes path: same
                # C function, same arrays, same little-endian uint64 slots.
                if self.uc_handle_addr == 0:
                    self.uc_handle_addr = <unsigned long long>self.uc_handle.value
                (<uc_reg_read_batch_ft>(<void*>self.uc_rrb_addr))(
                    <void*>self.uc_handle_addr,
                    <void*>(<unsigned long long>ids_addr),
                    <void*>(<unsigned long long>ptrs_addr),
                    <int>n_calls_int)
                vptr = <uint64_t*>(<void*>(<unsigned long long>vals_addr))
                pre_regs = {}
                for i_slot in range(n_slots):
                    pre_regs[names[i_slot]] = vptr[i_slot]
                if need_ef:
                    ef = pre_regs.get(self.flag_parent, 0)
                    for fname, fbit in self.eflags_bits.items():
                        pre_regs[fname] = (ef >> fbit) & 1
            else:
                self.uc_reg_read_batch(self.uc_handle, ids, ptrs, n_calls)
                pre_regs = {names[i_slot]: int(vals[i_slot]) for i_slot in range(n_slots)}
                if need_ef:
                    ef = pre_regs.get(self.flag_parent, 0)
                    for fname, fbit in self.eflags_bits.items():
                        pre_regs[fname] = (ef >> fbit) & 1
        except Exception:
            pre_regs = self.get_live_registers(self.uc_handle)
        pre_regs[self.pc_reg_name] = address
        return pre_regs

    cdef _aiw_check(self, list mem_writes, dict pre_regs, dict pre_taint,
                    bytes instruction_bytes, unsigned long long address):
        # Pure Python fallback for the rare AIW path.
        cdef long mem_addr
        for entry in mem_writes:
            mem_addr = entry[0]
            for reg_name, reg_taint in pre_taint.items():
                if reg_taint == 0:
                    continue
                reg_val = pre_regs.get(reg_name, 0)
                if reg_val == 0:
                    continue
                if abs(int(mem_addr) - int(reg_val)) <= 4096:
                    mnemonic, asm_str = self.disasm(instruction_bytes, address)
                    self.reporter.aiw(
                        address,
                        pointer_taint=reg_taint,
                        instruction=asm_str,
                    )
                    self.ql.emu_stop()
                    return

    cdef _handle_implicit_taint(self, bytes instruction_bytes,
                                 unsigned long long address, exc):
        mnemonic, asm_str = self.disasm(instruction_bytes, address)
        is_hijack = mnemonic.startswith('ret') or mnemonic in ('jmp', 'call')
        if is_hijack and self.check_bof:
            self.reporter.bof(address, instruction=asm_str)
            self.ql.emu_stop()
        elif not is_hijack and self.check_sc:
            taint_mask = 0
            try:
                for part in str(exc).split():
                    if part.startswith('0x'):
                        taint_mask = int(part, 16)
                        break
            except Exception:
                pass
            self.reporter.side_channel(address, instruction=asm_str, taint_mask=taint_mask)
            self.ql.emu_stop()
        else:
            self.ql.emu_stop()


# ---------------------------------------------------------------------------
# Pure-C UC_HOOK_CODE trampoline.  Registered with uc_hook_add as a raw C
# function pointer so Unicorn calls it with NO per-instruction Python frame:
# measured per-instruction callback cost was ~523 ns for a ctypes
# CFUNCTYPE(python) against ~34 ns for a raw pointer.
#
# It holds no GIL.  Unicorn releases the GIL for the duration of emulation, so
# a `with gil` declaration here re-acquires it once per guest instruction --
# which, measured against a pure-C Unicorn harness (benchmark/taint_density/
# hookcost.c), was about 183 ns of the ~200 ns this callback cost when it did
# nothing at all.  The express lane in fastpath.h answers what it can without
# Python and _hook_slow takes the GIL for the rest, which is where all the
# existing logic still lives, unchanged.
#
# user_data is the hook's MtFastCtx, not the hook: a nogil callback cannot cast
# a PyObject, and the context carries a borrowed pointer back.  The wrapper
# keeps the hook alive.  noexcept prints-and-clears any unhandled exception,
# matching the ctypes-callback behaviour.
# ---------------------------------------------------------------------------
cdef void _hook_aiw(void *owner, unsigned long long address):
    """The one piece of an express-lane instruction that needs the GIL."""
    (<InstructionHook>owner)._aiw_after_express(address)


cdef void _hook_slow(void *owner, unsigned long long address, int size):
    """Everything the express lane declined, with the GIL held."""
    cdef InstructionHook hook = <InstructionHook>owner
    # Straight to the array path when it is in use.  `_evaluate` only
    # dispatches between the two, and a whole Cython call to decide something
    # fixed at import time is about 3.5% of the run.
    if _USE_ARR_HOOK:
        hook._evaluate_arr(address, size)
    else:
        hook._evaluate(address, size)


cdef void _c_instruction_hook(void *uc, unsigned long long address,
                              unsigned int size, void *user_data) noexcept nogil:
    # MICROTAINT_NULL_HOOK: return before doing any taint work, so the cost of
    # BEING hooked (Unicorn's dispatch and the trampoline) can be measured apart
    # from the cost of the work inside.  Purely diagnostic -- the taint state is
    # not updated, so a run with it set produces no findings.
    if _NULL_HOOK:
        return
    # Unicorn releases the GIL for the duration of emulation, so `with gil` here
    # would re-acquire it once per guest instruction.  Measured against a pure-C
    # Unicorn harness on the same workload, Unicorn's own per-instruction
    # dispatch is about 18 ns while this callback returning immediately cost
    # about 200 ns -- nearly all of the difference being that acquire.  The
    # express lane answers what it can without it and hands over the rest.
    cdef MtFastCtx *c = <MtFastCtx*>user_data
    cdef int rc = mt_fast_step_nogil(c, address, size)
    if rc == MT_FAST_DONE:
        return
    if rc == MT_FAST_DONE_AIW:
        # Handled, but this instruction made tainted stores and the
        # attacker-influenced-write check has to see them.
        with gil:
            _hook_aiw(c.owner, address)
        return
    with gil:
        _hook_slow(c.owner, address, size)


def c_instruction_hook_ptr():
    """Address (int) of the pure-C UC_HOOK_CODE trampoline, for uc_hook_add.

    Register it with c_instruction_hook_ud(hook) as user_data; the caller MUST
    keep that InstructionHook alive for the hook's lifetime.
    """
    return <unsigned long long>&_c_instruction_hook


def c_instruction_hook_ud(InstructionHook hook):
    """user_data for the trampoline: the hook's C context, not the hook.

    The callback runs without the GIL, so it cannot cast a PyObject to reach
    the context; the context carries a borrowed pointer back to the hook for
    the path that does need Python.  Valid for as long as `hook` is alive.
    """
    return <unsigned long long>&hook.fctx


# ---------------------------------------------------------------------------
# Memory hooks — Cython port of _mem_write_clear_hook, _mem_access_hook,
# and _uaf_unmapped_write_hook.  Same design as InstructionHook: capture
# typed references to wrapper state once, dispatch into shadow_mem via
# C-level cpdef calls, no Python attribute lookups in the hot path.
# ---------------------------------------------------------------------------

cdef class MemWriteClearHook:
    """
    Memory-write callback (UC_HOOK_MEM_WRITE).

    Two responsibilities:
      1. UAF detection: if the target address is poisoned (was munmap'd),
         report and stop.
      2. Taint clearing: any guest write to an address NOT in
         `last_tainted_writes` clears the shadow-memory taint at that
         address.  This is how the engine "forgets" stale taint when
         the program writes a fresh, untainted value.
    """
    cdef public object wrapper
    cdef public BitPreciseShadowMemory shadow_mem
    cdef public set last_tainted_writes
    cdef public object reporter
    cdef public object ql
    cdef public bint check_uaf
    # What the GIL-free trampoline needs, as plain C (memhook.h).  It is the
    # callback's user_data, so reaching it costs no PyObject cast.
    cdef MtMemHookUD ud
    cdef PyObject *lw_set_ptr
    cdef InstructionHook _instr_hook

    # The instruction hook whose decode/output caches this write hook must
    # invalidate on self-modifying code.  None when the Cython hook is not in
    # use (the Python fallback re-reads bytes every instruction, so it needs no
    # invalidation).  Wired by the wrapper once both hooks exist -- which is
    # after registration, so assigning it re-resolves the C context.
    property instr_hook:
        def __get__(self):
            return self._instr_hook
        def __set__(self, value):
            self._instr_hook = value
            self._wire_ud()

    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.shadow_mem = wrapper.shadow_mem
        self.last_tainted_writes = wrapper._last_tainted_writes
        self.reporter = wrapper.reporter
        self.ql = wrapper.ql
        self.check_uaf = wrapper.check_uaf
        self._instr_hook = None
        self.lw_set_ptr = NULL
        self._wire_ud()

    cdef void _wire_ud(self):
        """Resolve the C context the GIL-free path reads.

        Everything mutable is taken as the ADDRESS of a field, so this survives
        the instruction hook filling ltw during a later instruction.  Called
        again whenever instr_hook is assigned, because registration happens
        before that wiring exists."""
        self.ud.owner = <PyObject*>self
        self.ud.check_uaf = <int*>&self.check_uaf
        self.ud.shadow = <void*>self.shadow_mem if self.shadow_mem is not None else NULL
        if _shadow_capi != NULL:
            self.ud.clear = _shadow_capi.clear
            self.ud.is_poisoned = _shadow_capi.is_poisoned
        else:
            self.ud.clear = NULL
            self.ud.is_poisoned = NULL
        if isinstance(self.last_tainted_writes, set):
            self.lw_set_ptr = <PyObject*>self.last_tainted_writes
            self.ud.lw_set = &self.lw_set_ptr
        else:
            self.ud.lw_set = NULL
        cdef InstructionHook ih = self._instr_hook
        if ih is not None:
            self.ud.ltw = ih.ltw
            self.ud.ltw_n = &ih.ltw_n
            # `unsigned long long` on the hook, uint64_t in the struct: the
            # same 64 bits, but distinct types to the C compiler.
            self.ud.code_lo = <const uint64_t*>&ih.code_lo
            self.ud.code_hi = <const uint64_t*>&ih.code_hi
        else:
            self.ud.ltw = NULL
            self.ud.ltw_n = NULL
            self.ud.code_lo = NULL
            self.ud.code_hi = NULL

    def __call__(self, object _uc, int _access, unsigned long long address,
                 int size, long long _value, object _user_data=None):
        # ctypes-callback entry point, kept so the non-C registration path (and
        # any direct caller) still works; the C trampoline calls _dispatch.
        self._dispatch(address, size)

    cdef void _dispatch(self, unsigned long long address, int size):
        # Hot path. Keep everything cdef.
        cdef BitPreciseShadowMemory sm = self.shadow_mem
        cdef set lw = self.last_tainted_writes
        cdef int i
        cdef unsigned long long a

        # Self-modifying / JIT'd code: if this write lands on any cached
        # instruction, drop the address-keyed caches so the rewritten bytes are
        # re-decoded on next execution.  Two C-level integer compares reject the
        # overwhelmingly common data write (stack/heap, outside the code range).
        cdef InstructionHook ih = self._instr_hook
        if (ih is not None and ih.code_hi > ih.code_lo
                and address < ih.code_hi
                and address + <unsigned long long>size > ih.code_lo):
            ih.invalidate_smc()

        if self.check_uaf and sm.is_poisoned(address, size):
            self.reporter.uaf(address, size)
            self.ql.emu_stop()
            return

        # Bytes the current instruction legitimately tainted are exempt from the
        # clear.  The array path records them as (addr, size, mask) triples on the
        # instruction hook; the dict and Python fallback paths still use the set.
        # Both are consulted, so whichever produced this instruction's writes is
        # honoured and a byte is cleared only if NEITHER claims it.
        cdef int n_ltw = ih.ltw_n if ih is not None else 0
        if n_ltw == 0 and not lw:
            sm.clear(address, size)
            return
        cdef int k
        cdef unsigned long long off
        cdef bint keep
        for i in range(size):
            a = address + <unsigned long long>i
            keep = False
            for k in range(n_ltw):
                off = a - ih.ltw[k].addr   # unsigned: a below the range wraps large
                if off < <unsigned long long>ih.ltw[k].size:
                    # off >= 8 means the write is wider than its 64-bit mask can
                    # describe (not reachable today: targets are <= 8 bytes, wide
                    # vectors arrive pre-split).  Keeping the byte over-taints;
                    # shifting past the mask width would drop taint.
                    if off >= 8 or ((ih.ltw[k].taint >> (off * 8)) & 0xFF):
                        keep = True
                        break
            if not keep and lw and PySet_Contains(lw, a):
                keep = True
            if not keep:
                sm.clear(a, 1)


cdef class MemAccessHook:
    """
    Memory-read callback (UC_HOOK_MEM_READ).

    Reports a UAF if the read targets a poisoned region and stops the
    emulator.  Trivial enough to inline but lives here for symmetry
    with the write hook and to share the same Cython compilation unit.
    """
    cdef public object wrapper
    cdef public BitPreciseShadowMemory shadow_mem
    cdef public object reporter
    cdef public object ql
    cdef public bint check_uaf
    cdef MtMemHookUD ud

    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.shadow_mem = wrapper.shadow_mem
        self.reporter = wrapper.reporter
        self.ql = wrapper.ql
        self.check_uaf = wrapper.check_uaf
        self.ud.owner = <PyObject*>self
        self.ud.check_uaf = <int*>&self.check_uaf
        self.ud.shadow = <void*>self.shadow_mem if self.shadow_mem is not None else NULL
        self.ud.is_poisoned = _shadow_capi.is_poisoned if _shadow_capi != NULL else NULL
        self.ud.clear = NULL
        self.ud.lw_set = NULL
        self.ud.ltw = NULL
        self.ud.ltw_n = NULL
        self.ud.code_lo = NULL
        self.ud.code_hi = NULL

    def __call__(self, object _uc, int _access, unsigned long long address,
                 int size, long long _value, object _user_data=None):
        self._dispatch(address, size)

    cdef void _dispatch(self, unsigned long long address, int size):
        if self.check_uaf and self.shadow_mem.is_poisoned(address, size):
            self.reporter.uaf(address, size)
            self.ql.emu_stop()


cdef class UafUnmappedWriteHook:
    """
    Invalid-memory-write callback (UC_HOOK_MEM_WRITE_UNMAPPED).

    Catches the mmap → munmap → write UAF pattern where the target
    page is fully unmapped (Unicorn would otherwise call its own crash
    handler).  Returns False to terminate emulation.
    """
    cdef public object wrapper
    cdef public BitPreciseShadowMemory shadow_mem
    cdef public object reporter
    cdef public object ql

    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.shadow_mem = wrapper.shadow_mem
        self.reporter = wrapper.reporter
        self.ql = wrapper.ql

    def __call__(self, object _uc, int _access, unsigned long long address,
                 int size, long long _value, object _user_data=None):
        if self.shadow_mem.is_poisoned(address, size):
            self.reporter.uaf(address, size)
        self.ql.emu_stop()
        return False


# ---------------------------------------------------------------------------
# LiveMemReader — Cython port of MicrotaintWrapper._read_live_memory.
#
# Called ~256k times per bench run from circuit_c's OP_PUSH_MEM_VALUE
# bytecode op when a circuit needs to read guest memory.  Original was a
# Python method on the wrapper (~1.48 us/call frame overhead).  This
# version exposes a callable cdef class that circuit_c invokes directly.
# Signature is identical (address, size) -> int so it's a drop-in
# replacement for the bound method passed via mem_reader=.
# ---------------------------------------------------------------------------

cdef class LiveMemReader:
    """Callable that reads `size` bytes from Unicorn's address space.

    The wrapper used to do this through a Python method that called
    ctypes-wrapped uc_mem_read into a pre-allocated buffer.  Moving the
    body to a cdef class drops the Python frame setup per call (~0.5 us
    out of 1.48 us total).  The remaining ~1 us is the actual ctypes
    call into libunicorn's uc_mem_read — which is C, just C through a
    Python ABI.
    """
    cdef public object wrapper        # the MicrotaintWrapper that owns us
    cdef public object uc_mem_read    # ctypes function for uc_mem_read
    cdef public object mem_buf        # pre-allocated ctypes buffer
    cdef public object mem_ptrs       # dict[int, ctypes.POINTER(uintN_t)]
    cdef public object ql             # Qiling, for the slow-path fallback
    cdef unsigned long long uc_handle # Unicorn engine pointer
    # C-level read boundary: address of uc_mem_read and of the ctypes buffer, so
    # the read needs no ctypes marshalling at all.  0 => use the ctypes path.
    cdef unsigned long long uc_mr_addr
    cdef unsigned long long mem_buf_addr

    def __init__(self, wrapper, *, uc_mem_read, mem_buf, mem_ptrs,
                 uc_mem_read_addr=0, mem_buf_addr=0):
        self.wrapper = wrapper
        self.uc_mem_read = uc_mem_read
        self.mem_buf = mem_buf
        self.mem_ptrs = mem_ptrs
        self.ql = wrapper.ql
        self.uc_mr_addr = <unsigned long long>(uc_mem_read_addr or 0)
        self.mem_buf_addr = <unsigned long long>(mem_buf_addr or 0)
        # uc_handle is a ctypes c_void_p — extract the integer once for
        # the hot path.  ctypes will accept the int argument directly.
        cdef object handle = wrapper._uc_handle
        if hasattr(handle, 'value'):
            self.uc_handle = <unsigned long long>(handle.value or 0)
        else:
            self.uc_handle = <unsigned long long>(handle or 0)

    def __call__(self, unsigned long long address, int size):
        """Read `size` bytes at `address` from the live Unicorn engine."""
        cdef int err
        cdef object ptr
        cdef unsigned char *p
        cdef uint64_t v
        cdef int i
        # C-level fast path: call uc_mem_read through the fnptr and assemble the
        # little-endian value straight out of the buffer.  Bit-identical to the
        # ctypes path (same C function, same buffer, same LE semantics) but with
        # no ConvParam / PyCArgObject boxing and no ctypes pointer deref.  Widths
        # above 8 bytes keep the bytes path.
        if self.uc_mr_addr != 0 and self.mem_buf_addr != 0 and 0 < size <= 8:
            err = (<uc_mem_read_ft>(<void*>self.uc_mr_addr))(
                <void*>self.uc_handle, <uint64_t>address,
                <void*>self.mem_buf_addr, <size_t>size)
            if err == 0:
                p = <unsigned char*>self.mem_buf_addr
                v = 0
                for i in range(size):
                    v |= (<uint64_t>p[i]) << (8 * i)
                return v
            try:
                return int.from_bytes(self.ql.mem.read(address, size), 'little')
            except Exception:  # noqa: BLE001
                return 0
        try:
            err = self.uc_mem_read(self.uc_handle, address, self.mem_buf, size)
            if err == 0:
                ptr = self.mem_ptrs.get(size)
                if ptr is not None:
                    return ptr[0]
                return int.from_bytes(self.mem_buf[:size], 'little')
            # Unicorn returned an error; fall back to Qiling's mem.read
            # (slower but knows about Qiling-mapped regions Unicorn
            # doesn't expose directly).
            return int.from_bytes(self.ql.mem.read(address, size), 'little')
        except Exception:
            return 0

# The memory hooks got the same treatment as the instruction hook above, and for
# the same reason: registered as a ctypes CFUNCTYPE they pay libffi's closure
# machinery (_CallPythonObject -> classify_argument -> per-argument isinstance
# and boxing) plus a GIL acquire on EVERY guest load and store, which perf
# attributes to _CallPythonObject/classify_argument/object_recursive_isinstance.
# Through a raw C function pointer the callback is a direct call into the Cython
# _dispatch, so only the GIL acquire remains.  Same bodies, same results.
# Unicorn calls these once per guest load and store, so a `with gil`
# declaration here means the program acquires the GIL on every memory access it
# makes.  Their usual answers -- "not poisoned", "clear the shadow bytes this
# write covered" -- are arithmetic over the shadow's page table, which has
# `nogil` accessors, so memhook.h answers them directly and only the rare cases
# (a report, a write onto decoded code, the dict path's Python claim set) come
# back here.
cdef void _c_mem_write_hook(void *uc, int access, unsigned long long address,
                            int size, long long value, void *user_data) noexcept nogil:
    cdef MtMemHookUD *u = <MtMemHookUD*>user_data
    if _NOGIL_MEM and mt_mem_write_nogil(u, address, size) == 0:
        return
    with gil:
        (<MemWriteClearHook>u.owner)._dispatch(address, size)


cdef void _c_mem_access_hook(void *uc, int access, unsigned long long address,
                             int size, long long value, void *user_data) noexcept nogil:
    cdef MtMemHookUD *u = <MtMemHookUD*>user_data
    if _NOGIL_MEM and mt_mem_read_nogil(u, address, size) == 0:
        return
    with gil:
        (<MemAccessHook>u.owner)._dispatch(address, size)


def c_mem_write_hook_ptr():
    """Address (int) of the pure-C UC_HOOK_MEM_WRITE trampoline.

    Register with c_mem_write_hook_ud(hook) as user_data; the caller MUST keep
    that MemWriteClearHook alive for the hook's lifetime.
    """
    return <unsigned long long>&_c_mem_write_hook


def c_mem_access_hook_ptr():
    """Address (int) of the pure-C UC_HOOK_MEM_READ trampoline.  Same contract
    as c_mem_write_hook_ptr, with c_mem_access_hook_ud(hook) as user_data."""
    return <unsigned long long>&_c_mem_access_hook


def c_mem_write_hook_ud(MemWriteClearHook hook):
    """user_data for the write trampoline: its C context, not the hook."""
    return <unsigned long long>&hook.ud


def c_mem_access_hook_ud(MemAccessHook hook):
    """user_data for the read trampoline: its C context, not the hook."""
    return <unsigned long long>&hook.ud
