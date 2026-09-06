#!/usr/bin/env python3
"""bench_c_hook.py — per-instruction cost of a UC_HOOK_CODE by registration style.

Isolates the cost Unicorn pays to enter a code hook, over a tight `inc rax`
loop, for four registration styles:

  (0) no hook                              -> Unicorn native TCG execution
  (1) ctypes CFUNCTYPE(python) callback    -> the classic Python-callback path
  (2) pure-C function pointer              -> no per-instruction Python frame
  (3) pure-C + PyGILState_Ensure/Release   -> a C trampoline that then re-enters
                                              Python/Cython (what microtaint's
                                              MICROTAINT_C_HOOK path costs)

The (1)-(2) delta is the Python-frame cost a pure-C hook removes; (3) is the
floor for any hook that still runs the (Cython) taint logic under the GIL.

Self-contained: it compiles a tiny C probe to a temp .so with the system cc.
Run:  python benchmark/overhead/bench_c_hook.py [--n 400000]
"""
# ruff: noqa: PLC0415, ARG005, S603
from __future__ import annotations

import argparse
import ctypes
import subprocess
import sys
import sysconfig
import tempfile
import time
from pathlib import Path

_PROBE_C = r"""
#include <Python.h>
#include <stdint.h>
static volatile uint64_t g_count = 0;
void mt_empty_code_hook(void *uc, uint64_t a, uint32_t s, void *ud) {
    (void)uc;(void)a;(void)s;(void)ud; g_count++;
}
void mt_gil_code_hook(void *uc, uint64_t a, uint32_t s, void *ud) {
    (void)uc;(void)a;(void)s;(void)ud;
    PyGILState_STATE g = PyGILState_Ensure(); g_count++; PyGILState_Release(g);
}
uint64_t mt_hook_count(void){ return g_count; }
"""


def _build_probe() -> ctypes.CDLL:
    tmp = Path(tempfile.mkdtemp(prefix='mt_hook_probe_'))
    src = tmp / 'probe.c'
    src.write_text(_PROBE_C)
    so = tmp / 'probe.so'
    inc = sysconfig.get_path('include')
    libdir = sysconfig.get_config_var('LIBDIR') or ''
    ldver = sysconfig.get_config_var('LDVERSION') or f'{sys.version_info.major}.{sys.version_info.minor}'
    cmd = ['cc', '-O2', '-shared', '-fPIC', f'-I{inc}', '-o', str(so), str(src)]
    if libdir:
        cmd += [f'-L{libdir}', f'-lpython{ldver}']
    subprocess.run(cmd, check=True)
    lib = ctypes.CDLL(str(so))
    lib.mt_hook_count.restype = ctypes.c_uint64
    return lib


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=400_000)
    args = ap.parse_args()
    n = args.n

    try:
        import unicorn.unicorn_py3.unicorn as _uu
        from unicorn import UC_ARCH_X86, UC_HOOK_CODE, UC_MODE_64, Uc
        from unicorn.x86_const import UC_X86_REG_RAX
    except Exception as e:
        print('unicorn not available:', e)
        return 1

    lib = _build_probe()
    code_addr = 0x1000000
    body = b'\x48\xff\xc0'  # inc rax
    code = body * 64
    code += b'\xe9' + ((-(len(body) * 64 + 5)) & 0xFFFFFFFF).to_bytes(4, 'little')

    _uclib = _uu.uclib
    add = _uclib.uc_hook_add
    add.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]
    add.restype = ctypes.c_int
    cfunc_t = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint64,
                               ctypes.c_uint32, ctypes.c_void_p)

    def mk():
        uc = Uc(UC_ARCH_X86, UC_MODE_64)
        uc.mem_map(code_addr, 0x10000)
        uc.mem_write(code_addr, code)
        uc.reg_write(UC_X86_REG_RAX, 0)
        return uc

    def handle(uc):
        for attr in ('_uch', 'uch', '_uc'):
            h = getattr(uc, attr, None)
            if h is not None:
                return ctypes.c_void_p(int(h) if not isinstance(h, ctypes.c_void_p) else h.value)
        raise RuntimeError('no uc handle')

    def run_binding(label, hook):
        uc = mk()
        hid = uc.hook_add(UC_HOOK_CODE, hook) if hook is not None else None
        t0 = time.perf_counter_ns()
        uc.emu_start(code_addr, code_addr + len(code), 0, n)
        dt = time.perf_counter_ns() - t0
        if hid is not None:
            uc.hook_del(hid)
        print(f'  {label:44s} {dt / n:8.1f} ns/instr')
        return dt / n

    def run_ptr(label, ptr):
        uc = mk()
        hh = ctypes.c_size_t()
        rc = add(handle(uc), ctypes.byref(hh), UC_HOOK_CODE, ctypes.cast(ptr, ctypes.c_void_p),
                 None, ctypes.c_uint64(code_addr), ctypes.c_uint64(code_addr + len(code)))
        if rc != 0:
            print(f'  {label:44s} uc_hook_add FAILED rc={rc}')
            return -1.0
        t0 = time.perf_counter_ns()
        uc.emu_start(code_addr, code_addr + len(code), 0, n)
        dt = time.perf_counter_ns() - t0
        print(f'  {label:44s} {dt / n:8.1f} ns/instr')
        return dt / n

    print(f'UC_HOOK_CODE registration-style cost (N={n} inc-rax):')
    t0 = run_binding('(0) no hook (native TCG)', None)
    py_cb = cfunc_t(lambda uc, a, s, u: None)  # keep alive
    t1 = run_ptr('(1) ctypes CFUNCTYPE(python)', ctypes.cast(py_cb, ctypes.c_void_p))
    t2 = run_ptr('(2) pure-C fn ptr', lib.mt_empty_code_hook)
    t3 = run_ptr('(3) pure-C + PyGILState (Cython floor)', lib.mt_gil_code_hook)
    print()
    print(f'  Python-frame cost removed (1)-(2): {t1 - t2:.1f} ns/instr')
    print(f'  GIL floor for C+Cython   (3)-(0): {t3 - t0:.1f} ns/instr')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
