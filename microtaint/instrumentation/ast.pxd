# cython: language_level=3
"""
Cython declarations for LogicCircuit.

Exists so other Cython modules can `cimport LogicCircuit` and read its fields as
C struct members.  Without this .pxd, `circuit._compiled` from another module is
a PyObject_GenericGetAttr on every instruction, which the taint hot path pays
once per executed instruction purely to reach a pointer it already owns.

The attribute list must stay in sync with the class body in ast.pyx: Cython
requires the declarations to live here and not be repeated there.
"""

cdef class EvalContext:
    cdef public dict input_taint
    cdef public dict input_values
    cdef public object simulator
    cdef public object implicit_policy
    cdef public object shadow_memory
    cdef public object mem_reader
    cdef public str arch_str  # cached once, avoids str(simulator.arch) per TaintOperand miss
    cdef public bint share_frames  # frame-recycle: route cell exec through the per-evaluate frame cache


cdef class LogicCircuit:
    cdef public list assignments
    cdef public object architecture
    cdef public str instruction
    cdef public list state_format
    cdef public str _pc_target      # pre-computed: 'RIP'/'EIP'/'PC' or None
    cdef public bint has_unicorn_cells  # True if any assignment uses InstructionCellExpr
    cdef public object input_reg_names  # set of register names needed as value inputs
    cdef public object _compiled       # cached CompiledCircuit (or None if compile failed/disabled)
    # True iff the taint output is a pure function of the input taints (all p-code
    # ops are value-independent: COPY/XOR/NOT/ZEXT/SEXT/const-shift).  Set by the
    # circuit builder; the compiled circuit copies it and the instruction cache
    # then keys on the taint signature alone.  Default False (value-dependent).
    cdef public bint value_independent

    cpdef precompile(self, object simulator)
    cdef _compile_now(self, object simulator)
    cdef _warm_decode(self)
    cpdef dict evaluate(self, EvalContext context)
