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
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.set cimport PySet_Add
from cpython.exc cimport PyErr_Clear, PyErr_Occurred, PyErr_ExceptionMatches
from cpython.ref cimport Py_INCREF, Py_DECREF, PyObject

import ctypes
from microtaint.types import ImplicitTaintError as _ImplicitTaintError

cdef object ImplicitTaintError = _ImplicitTaintError
cdef object EMPTY_FROZENSET = frozenset()


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
    cdef public dict   instr_cache        # address -> (frozenset, dict)  legacy entries
    cdef public dict   instr_cache_v      # address -> (taint_version, dict, taint_snapshot)
    cdef public object shadow_mem         # BitPreciseShadowMemory (cdef class)
    cdef public object sim                # CellSimulator
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
    cdef object       uc_handle           # int (cached uc handle)
    cdef object       uc_mem_read         # ctypes function
    cdef object       uc_reg_read_batch   # ctypes function
    cdef object       mem_buf             # ctypes buffer
    cdef object       arch                # Architecture enum
    cdef object       cached_gen_rule     # _cached_generate_static_rule
    cdef object       x64_format_key      # tuple
    cdef object       get_decoded         # _get_decoded
    cdef object       build_offsets_arrs  # _build_offsets_arrays
    cdef object       eflags_bits         # dict
    cdef object       eval_context_cls    # EvalContext
    cdef object       read_live_memory    # bound method
    cdef object       get_live_registers  # bound method
    cdef object       disasm              # bound method

    # Counters
    cdef public unsigned long instr_cache_hits
    cdef public unsigned long instr_cache_misses

    def __init__(self, wrapper, *,
                 uc_handle, uc_mem_read, uc_reg_read_batch, mem_buf,
                 arch, cached_gen_rule, x64_format_key,
                 get_decoded, build_offsets_arrs, eflags_bits,
                 eval_context_cls):
        self.wrapper = wrapper
        self.register_taint = wrapper.register_taint
        self.last_tainted_writes = wrapper._last_tainted_writes
        self.instr_cache = wrapper._instr_cache
        # Version-keyed companion cache.  address -> (taint_version, output_state).
        # On hit, no dict-equality check needed: same version means same state.
        self.instr_cache_v = {}
        self.shadow_mem = wrapper.shadow_mem
        self.sim = wrapper.sim
        self.policy = wrapper._policy
        self.reporter = wrapper.reporter
        self.ql = wrapper.ql
        self.check_bof = wrapper.check_bof
        self.check_sc = wrapper.check_sc
        self.check_aiw = wrapper.check_aiw
        self.instr_cache_enabled = wrapper._instr_cache_enabled
        self.taint_version = 0
        self.uc_handle = uc_handle
        self.uc_mem_read = uc_mem_read
        self.uc_reg_read_batch = uc_reg_read_batch
        self.mem_buf = mem_buf
        self.arch = arch
        self.cached_gen_rule = cached_gen_rule
        self.x64_format_key = x64_format_key
        self.get_decoded = get_decoded
        self.build_offsets_arrs = build_offsets_arrs
        self.eflags_bits = eflags_bits
        self.eval_context_cls = eval_context_cls
        self.read_live_memory = wrapper._read_live_memory
        self.get_live_registers = wrapper._get_live_registers
        self.disasm = wrapper._disasm
        self.instr_cache_hits = 0
        self.instr_cache_misses = 0

    def __call__(self, uc, address, size, user_data):
        # Cython compiles this to a typed C function. The Python frame
        # for `__call__` is unavoidable (Unicorn calls it through a Python
        # binding), but everything inside is C.
        self._evaluate(<unsigned long long>address, <int>size)

    cdef inline _evaluate(self, unsigned long long address, int size):
        cdef dict register_taint = self.register_taint

        # Early exit: no taint, no work. (Rare once any taint exists.)
        # NOTE: _any_taint check is implicit — if register_taint is empty
        # AND no shadow taint exists, the wrapper would not have armed
        # this hook. Once armed, we always run.

        # Read instruction bytes.
        cdef int err = self.uc_mem_read(self.uc_handle, address, self.mem_buf, size)
        cdef bytes instruction_bytes
        if err == 0:
            instruction_bytes = bytes(self.mem_buf[:size])
        else:
            instruction_bytes = bytes(self.ql.mem.read(address, size))

        cdef object circuit = self.cached_gen_rule(self.arch, instruction_bytes, self.x64_format_key)
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
        cdef bint can_cache = (
            self.instr_cache_enabled
            and compiled_circuit is not None
            and compiled_circuit is not False
            and not compiled_circuit.has_mem_ops
        )
        if can_cache:
            # Tier 4: version-cache fast path.
            v_entry = self.instr_cache_v.get(address)
            live_version = self.taint_version
            if v_entry is not None and (<unsigned long long>(<object>v_entry[0])) == live_version:
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
            if cached is not None and cached[0] == cache_key:
                # Cache hit: replay output_state directly.
                output_state = cached[1]
                self.instr_cache_hits += 1
                if self.last_tainted_writes:
                    self.last_tainted_writes.clear()
                if PyDict_Size(register_taint):
                    PyDict_Clear(register_taint)
                # Populate the version cache too, so the next visit
                # at this address skips the frozenset construction.
                # We need a deterministic output_version derived from
                # output_state's content.  Use frozenset hash (one-shot
                # cost; only happens on cold version-cache misses).
                out_version = (<Py_ssize_t>hash(frozenset(output_state.items())))
                self.instr_cache_v[address] = (live_version, out_version, output_state)
                # No-mem-ops circuits never produce MEM_ keys; just refill.
                for key, val in output_state.items():
                    if val:
                        PyDict_SetItem(register_taint, key, val)
                self.taint_version = out_version
                return
            self.instr_cache_misses += 1

        # Slow path: do everything.
        # Read live register values.
        cdef object pre_regs
        try:
            decoded = self.get_decoded(self.arch, instruction_bytes)
            uc_arrs = decoded._uc_arrays
            if uc_arrs is None:
                uc_arrs = self.build_offsets_arrs(decoded.input_reg_offsets)
                decoded._uc_arrays = uc_arrs
            ids, vals, ptrs, n, names, need_ef = uc_arrs
            if ids is None:
                pre_regs = {}
            else:
                self.uc_reg_read_batch(self.uc_handle, ids, ptrs, n)
                pre_regs = {names[i]: int(vals[i]) for i in range(n)}
                if need_ef:
                    ef = pre_regs.get('EFLAGS', 0)
                    for f, b in self.eflags_bits.items():
                        pre_regs[f] = (ef >> b) & 1
        except Exception:
            pre_regs = self.get_live_registers(self.uc_handle)

        # Snapshot register_taint for EvalContext.
        cdef dict pre_taint = PyDict_Copy(register_taint)
        # Tell wrapper for AIW check + slow path consistency.
        self.wrapper._pre_regs = pre_regs
        self.wrapper._pre_taint = pre_taint

        ctx = self.eval_context_cls(
            input_taint=pre_taint,
            input_values=pre_regs,
            simulator=self.sim,
            implicit_policy=self.policy,
            shadow_memory=self.shadow_mem,
            mem_reader=self.read_live_memory,
        )

        # Evaluate the circuit. Catch ImplicitTaintError for SC/BOF reporting.
        try:
            output_state = circuit.evaluate(ctx)
        except BaseException as e:
            if isinstance(e, ImplicitTaintError):
                self._handle_implicit_taint(instruction_bytes, address, e)
                return
            raise

        # Cache the output for next time at this address with this taint sig.
        # We need a stable output_version derived from output_state content,
        # so that future cache hits can adopt this version atomically.
        cdef unsigned long long out_version_slow = 0
        if cache_key is not None:
            self.instr_cache[address] = (cache_key, dict(output_state))
            # Compute output_version once (one frozenset hash; happens
            # ~167k times across the bench, not 1.19M).
            out_version_slow = (<Py_ssize_t>hash(frozenset(output_state.items())))
            self.instr_cache_v[address] = (self.taint_version, out_version_slow, dict(output_state))

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
        cdef object shadow_mem = self.shadow_mem
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
