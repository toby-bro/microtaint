"""Emit C for a lowered taint program, and compile a batch of them.

Two uses, and the second is the point.

As a BACKEND: a taint program is straight-line 64-bit integer code with no
memory traffic except its inputs and outputs, which is exactly the shape a C
compiler turns into good machine code -- so handing the program to clang and
dlopening the result is a legitimate way to run taint at native speed for a
working set of instructions known ahead of time.

As a MEASUREMENT: it establishes the ceiling.  An optimising compiler with a
real register allocator and instruction selector is the best any hand-written
emitter is going to do, so the gap between this and a direct JIT says whether
the JIT is worth its complexity, and the gap between this and the interpreter
says what dispatch is costing.

The generated function takes the register state as two flat arrays and writes
taint into a third, matching the C evaluator's convention exactly, so the two
are interchangeable and can be diff-tested against each other.
"""
# ruff: noqa: PLC0415, S603, S607
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from microtaint.taint_ir.exec import SlotOf
    from microtaint.taint_ir.ir import IRProg

import ctypes
import os
import subprocess
import tempfile

from microtaint.taint_ir import ir as _ir

_BIN = {
    _ir.AND: '&', _ir.OR: '|', _ir.XOR: '^', _ir.ADD: '+', _ir.SUB: '-',
    _ir.MUL: '*',
}


def _slot(slot_of: SlotOf, key: Any) -> int:
    s = slot_of(key)
    if s is None:
        raise KeyError(f'no state slot for register {key!r}')
    return s


def emit_c(prog: IRProg, slot_of: SlotOf, name: str) -> str:
    """C source for one finalized program."""
    p = prog.finalize() if any(op == _ir.BOOLSYM for op, *_r in prog.nodes) else prog
    inv = {n: k for (kind, k), n in p.inputs.items() if kind == 'v'}
    intn = {n: k for (kind, k), n in p.inputs.items() if kind == 't'}
    lines = [f'void {name}(const uint64_t *v, const uint64_t *t, uint64_t *o) {{']
    for n, (op, a, b, c, imm) in enumerate(p.nodes):
        if not p.live[n]:
            continue
        d = f'    uint64_t x{n} = '
        if op == _ir.CONST:
            lines.append(f'{d}UINT64_C({imm});')
        elif op == _ir.INV:
            lines.append(f'{d}v[{_slot(slot_of, inv[n])}];')
        elif op == _ir.INT:
            lines.append(f'{d}t[{_slot(slot_of, intn[n])}];')
        elif op in _BIN:
            lines.append(f'{d}x{a} {_BIN[op]} x{b};')
        elif op == _ir.SHL:
            lines.append(f'{d}(x{b} < 64) ? (x{a} << x{b}) : 0;')
        elif op == _ir.SHR:
            lines.append(f'{d}(x{b} < 64) ? (x{a} >> x{b}) : 0;')
        elif op == _ir.SAR:
            lines.append(f'{d}(uint64_t)(((int64_t)x{a}) >> '
                         f'((x{b} < 64) ? x{b} : 63));')
        elif op == _ir.NOT:
            lines.append(f'{d}~x{a};')
        elif op == _ir.NEG:
            lines.append(f'{d}(uint64_t)(-(int64_t)x{a});')
        elif op == _ir.ULT:
            lines.append(f'{d}(x{a} < x{b}) ? 1 : 0;')
        elif op == _ir.SLT:
            lines.append(f'{d}((int64_t)x{a} < (int64_t)x{b}) ? 1 : 0;')
        elif op == _ir.EQ:
            lines.append(f'{d}(x{a} == x{b}) ? 1 : 0;')
        elif op == _ir.NEZ:
            lines.append(f'{d}x{a} ? 1 : 0;')
        elif op == _ir.SEL:
            lines.append(f'{d}x{c} ? x{a} : x{b};')
        elif op == _ir.POPCNT:
            lines.append(f'{d}(uint64_t)__builtin_popcountll(x{a});')
        elif op == _ir.CLZ:
            lines.append(f'{d}x{a} ? (uint64_t)__builtin_clzll(x{a}) : 64;')
        elif op == _ir.MULHI:
            lines.append(f'{d}(uint64_t)(((unsigned __int128)x{a} * '
                         f'(unsigned __int128)x{b}) >> 64);')
        elif op == _ir.UDIV:
            lines.append(f'{d}x{b} ? x{a} / x{b} : 0;')
        elif op == _ir.UREM:
            lines.append(f'{d}x{b} ? x{a} % x{b} : 0;')
        elif op == _ir.SDIV:
            lines.append(f'{d}x{b} ? (uint64_t)((int64_t)x{a} / (int64_t)x{b}) : 0;')
        elif op == _ir.SREM:
            lines.append(f'{d}x{b} ? (uint64_t)((int64_t)x{a} %% (int64_t)x{b}) : 0;'
                         .replace('%%', '%'))
        else:
            raise ValueError(f'no C lowering for {op}')
    for key, n in p.outputs:
        s = slot_of(key)
        if s is not None:
            lines.append(f'    o[{s}] = x{n};')
    lines.append('}')
    return '\n'.join(lines)


PROLOGUE = '#include <stdint.h>\n#include <time.h>\n'


def emit_driver(names: list[str]) -> str:
    """A C-side timing loop over the generated functions.

    Timing them from Python would measure the FFI, which costs several times
    what the taint program does.  The call through the table is deliberate: it
    keeps the compiler from inlining the body into the loop and optimising the
    work away, and a real integration pays for a call too.
    """
    table = ',\n    '.join(names)
    return f"""
typedef void (*mt_fn)(const uint64_t *, const uint64_t *, uint64_t *);
static mt_fn MT_FNS[] = {{
    {table}
}};
double mt_bench(int idx, const uint64_t *v, const uint64_t *t, uint64_t *o,
                long iters) {{
    mt_fn f = MT_FNS[idx];
    struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    for (long k = 0; k < iters; k++) f(v, t, o);
    clock_gettime(CLOCK_MONOTONIC, &b);
    return ((double)(b.tv_sec - a.tv_sec) * 1e9
            + (double)(b.tv_nsec - a.tv_nsec)) / (double)iters;
}}
void mt_call(int idx, const uint64_t *v, const uint64_t *t, uint64_t *o) {{
    MT_FNS[idx](v, t, o);
}}
"""


def compile_batch(sources: list[str], *, opt: str = '-O3', cc: str | None = None,
                  workdir: str | None = None) -> tuple[Any, float]:
    """Compile many generated functions into one shared object and dlopen it.

    Returns (ctypes.CDLL, compile_seconds).  Batching matters: the compiler's
    own start-up dominates a single 60-operation function, so per-function
    compile time is only meaningful measured over a realistic set.
    """
    import time
    cc = cc or os.environ.get('CC', 'clang')
    src = PROLOGUE + '\n'.join(sources) + '\n'
    d = workdir or tempfile.mkdtemp(prefix='mt_taint_ir_')
    cpath = os.path.join(d, 'taint_progs.c')
    sopath = os.path.join(d, 'taint_progs.so')
    with open(cpath, 'w') as f:
        f.write(src)
    t0 = time.perf_counter()
    subprocess.run([cc, opt, '-fPIC', '-shared', '-march=native',
                    '-fno-plt', cpath, '-o', sopath], check=True,
                   capture_output=True)
    dt = time.perf_counter() - t0
    return ctypes.CDLL(sopath), dt


_FN = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_uint64),
                       ctypes.POINTER(ctypes.c_uint64),
                       ctypes.POINTER(ctypes.c_uint64))


def bind_driver(lib: Any) -> tuple[Any, Any]:
    """(bench, call) bound to the compiled batch."""
    lib.mt_bench.restype = ctypes.c_double
    lib.mt_bench.argtypes = [ctypes.c_int] + [ctypes.POINTER(ctypes.c_uint64)] * 3 \
        + [ctypes.c_long]
    lib.mt_call.restype = None
    lib.mt_call.argtypes = [ctypes.c_int] + [ctypes.POINTER(ctypes.c_uint64)] * 3
    return lib.mt_bench, lib.mt_call


def bind(lib: Any, name: str) -> Any:
    fn = getattr(lib, name)
    fn.restype = None
    fn.argtypes = [ctypes.POINTER(ctypes.c_uint64)] * 3
    return fn
