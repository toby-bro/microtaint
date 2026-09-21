"""Type stubs for the compiled taint-IR runtime.

`taint_ir_c` is a plain CPython C extension: it takes a serialized IR program,
compiles it, optionally emits native code for it, and runs it over flat slot
arrays.  Four modules import it -- the engine glue, the block compiler, the
public taint API and the tests -- and none of them had a declaration to check
against.
"""
from microtaint.taint_ir.exec import SerializedForC

class _Capsule: ...

def compile(program: SerializedForC) -> _Capsule:
    """Compile a serialized IR program.  The capsule owns the emitted code."""

def jit(program: _Capsule) -> bool:
    """Emit native code for the program.  True if the host emitter took it."""

def fn_addr(program: _Capsule) -> int:
    """Address of the emitted function, or 0 if it was not emitted."""

#: The interpreter, for another native extension to call.  Imported by the
#: block runtime at module init via PyCapsule_Import("taint_ir_c._taint_ir_capi")
#: so a region whose program the emitter declined can still be RUN rather than
#: making the whole block unhandleable.  Not intended for Python use.
_taint_ir_capi: object

def prog_addr(program: _Capsule) -> int:
    """Address of the program itself, for the interpreter capsule.

    The block runtime runs a region this way when the host emitter declined
    the program (it declines division and count-leading-zeros deliberately).
    The capsule owns the program, so a caller holding this address has to keep
    the capsule alive.
    """

def has_backend() -> bool:
    """True if this build has a native emitter for the host.

    Without this, a False from `jit` is ambiguous, and dangerously so: it is
    the right answer on a host with no backend, and it is equally what a
    FAILED page allocation looks like, because both leave the caller using the
    interpreter and therefore still correct.  No correctness test can tell the
    two apart, so a platform where the emitter silently never engages would
    look exactly like a healthy one.  Where this is True, a trivial program
    must be taken.
    """

def jit_size(program: _Capsule) -> int:
    """Bytes of native code, or 0."""

def n_nodes(program: _Capsule) -> int:
    """Node count of a compiled program."""

def run(program: _Capsule, values: list[int], taint: list[int]) -> list[int]:
    """Run once.

    Returns the updated taint slots.  The output aliases the taint input, which
    is sound because a compiled program reads every input before it stores any
    output -- the guarantee the block runtime's in-place pass relies on.
    """

def bench(program: _Capsule, values: list[int], taint: list[int],
          iters: int = ...) -> tuple[float, int]:
    """Time the program itself: (nanoseconds per iteration, a sink value).

    The state arrays are prepared once, so what is measured is the propagation
    and nothing around it.
    """
