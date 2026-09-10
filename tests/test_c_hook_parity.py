# ruff: noqa: S603, S607, S110, PLC0415
"""The pure-C UC_HOOK_CODE trampoline (MICROTAINT_C_HOOK) must produce IDENTICAL
detection results to the ctypes CFUNCTYPE(python) path.

Both register the SAME InstructionHook._evaluate; the only difference is how
Unicorn enters it (a raw C function pointer with a `with gil` trampoline vs a
ctypes Python callback).  So a taint-carrying run must report the same findings.
This is the correctness gate for the C-hook optimisation; the speed win itself is
measured by benchmark/overhead/bench_c_hook.py.
"""
from __future__ import annotations

import os
import platform
import subprocess
import tempfile
from io import StringIO

import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != 'Linux', reason='emulator compilation tests require Linux',
)

_SRC = r"""
long sys_read(int fd, void *buf, unsigned long count){long r;
  __asm__ volatile("syscall":"=a"(r):"0"(0),"D"(fd),"S"(buf),"d"(count):"rcx","r11","memory");return r;}
long sys_exit(int status){long r;
  __asm__ volatile("syscall":"=a"(r):"0"(60),"D"(status):"rcx","r11","memory");return r;}
void vulnerable(){ char buf[16]; sys_read(0, buf, 32); }
void _start(){ vulnerable(); sys_exit(0); }
"""


def _compile(src: str) -> str:
    fd, path = tempfile.mkstemp(suffix='.elf')
    os.close(fd)
    subprocess.run(['gcc', '-nostdlib', '-O0', '-fno-stack-protector', '-o', path, '-x', 'c', '-'],
                   input=src.encode(), check=True)
    return path


def _run(binary: str, stdin_data: bytes, use_c_hook: bool) -> tuple[list[str], bool]:
    from qiling import Qiling
    from qiling.const import QL_VERBOSE
    from qiling.extensions import pipe

    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper
    from microtaint.sleigh.engine import _cached_generate_static_rule

    _cached_generate_static_rule.cache_clear()
    prev = os.environ.get('MICROTAINT_C_HOOK')
    os.environ['MICROTAINT_C_HOOK'] = '1' if use_c_hook else '0'
    try:
        ql = Qiling([binary], '/', verbose=QL_VERBOSE.OFF)
        ql.os.stdin = pipe.SimpleInStream(0)
        ql.os.stdin.write(stdin_data)
        stream = StringIO()
        reporter = Reporter(json_mode=False, stream=stream)
        wrapper = MicrotaintWrapper(ql, check_bof=True, check_uaf=False, check_sc=False,
                                    reporter=reporter)
        used_c = wrapper._use_c_hook
        try:
            ql.run()
        except Exception:
            pass
        reporter.finalize()
        return stream.getvalue().splitlines(), used_c
    finally:
        if prev is None:
            os.environ.pop('MICROTAINT_C_HOOK', None)
        else:
            os.environ['MICROTAINT_C_HOOK'] = prev


def _norm(lines: list[str]) -> list[str]:
    # Drop volatile fields (addresses/timings can differ run to run); keep the
    # finding LABELS which are what a detection asserts on.
    out = []
    for ln in lines:
        for tag in ('[BOF]', 'buffer_overflow', '[UAF]', 'use_after_free', '[SC]', 'side_channel'):
            if tag in ln:
                out.append(tag)
    return sorted(set(out))


def test_c_hook_matches_ctypes_path_on_bof() -> None:
    binary = _compile(_SRC)
    payload = b'A' * 32
    logs_c, used_c = _run(binary, payload, use_c_hook=True)
    logs_py, used_py = _run(binary, payload, use_c_hook=False)

    assert used_c is True, 'C-hook path should be selected when MICROTAINT_C_HOOK=1'
    assert used_py is False, 'ctypes path should be selected when MICROTAINT_C_HOOK=0'
    # Both must detect the buffer overflow, with the SAME finding labels.
    assert _norm(logs_c), f'C-hook path reported no finding: {logs_c}'
    assert _norm(logs_c) == _norm(logs_py), (
        f'C-hook vs ctypes finding mismatch:\n  C : {_norm(logs_c)}\n  py: {_norm(logs_py)}'
    )
