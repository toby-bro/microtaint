"""Type stubs for the compiled taint-IR runtime.

`taint_ir_c` is a plain CPython C extension: it takes a serialized IR program,
compiles it, optionally emits native code for it, and runs it over flat slot
arrays.  Four modules import it -- the engine glue, the block compiler, the
public taint API and the tests -- and none of them had a declaration to check
against.
"""
from typing import Any

class _Capsule: ...

def compile(program: dict[str, Any]) -> _Capsule:
    """Compile a serialized IR program.  The capsule owns the emitted code."""

def jit(program: _Capsule) -> bool:
    """Emit native code for the program.  True if the host emitter took it."""

def fn_addr(program: _Capsule) -> int:
    """Address of the emitted function, or 0 if it was not emitted."""

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
