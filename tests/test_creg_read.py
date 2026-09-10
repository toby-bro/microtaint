# ruff: noqa: S603, S607, PLC0415
"""The C-level register-read path must be bit-identical to the ctypes path.

_read_pre_regs can read the instruction's live input registers either through
Python ctypes (uc_reg_read_batch + per-slot ctypes indexing) or, by default,
through a C function pointer + a uint64* over the value array (MICROTAINT_DISABLE_CREGS
toggles it).  Both call the same Unicorn C function on the same arrays, so they
must agree exactly; a wrong address or cast in the C path would corrupt the
register VALUES that drive memory-address computation and value-dependent taint.

This runs a register-value-dependent addressing sequence (store through a pointer
register, then load it back, plus a value-dependent AND) through the real Cython
hook with the C path ON and OFF, and asserts the final register + shadow taint
match.  Emulation runs in a subprocess (multiple Qiling instances in one pytest
process can segfault via a Unicorn/gevent interaction).
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from typing import TypedDict

import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != 'Linux', reason='emulator hook tests require Linux + gcc',
)

FULL = 0xFFFFFFFFFFFFFFFF


class CregRun(TypedDict):
    """What the child below prints: the final register taint by name, and the
    shadow masks at the two addresses it wrote.  Every value is a hex string,
    so the two runs compare literally."""

    final_taint: dict[str, str]
    shadow_D: str
    shadow_D8: str


def _run(disable_cregs: bool) -> CregRun:
    import os
    env = dict(os.environ)
    env['MICROTAINT_DISABLE_CREGS'] = '1' if disable_cregs else '0'
    out = subprocess.run(
        [sys.executable, __file__], capture_output=True, text=True, check=True, env=env,
    ).stdout
    run: CregRun = json.loads(out.strip().splitlines()[-1])
    return run


def test_creg_path_matches_ctypes_path() -> None:
    on = _run(disable_cregs=False)   # C-level register read (default)
    off = _run(disable_cregs=True)   # ctypes register read
    assert on == off, (
        'C-level register read diverged from the ctypes path:\n'
        f'  C-path ={on}\n  ctypes ={off}\n'
        '(a wrong address/cast in the fnptr register read would corrupt the '
        'register values feeding address computation / value-dependent taint)'
    )
    # And the sequence must actually propagate taint (guard against a trivial
    # both-empty match).
    assert on['final_taint'], 'expected non-empty taint after the sequence'


def _main() -> None:
    import os
    import tempfile

    from qiling import Qiling
    from qiling.const import QL_VERBOSE
    from unicorn.x86_const import UC_X86_REG_RAX, UC_X86_REG_RBX

    from microtaint.emulator.hook_core import InstructionHook
    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper

    # A register-value-dependent addressing sequence.  RBX holds the pointer
    # VALUE (read via pre_regs), so a wrong C-path register read would mis-address
    # the store/load and diverge the shadow taint.
    prog = [
        ('488903',   3),   # mov [rbx], rax        store tainted RAX at [RBX]
        ('488b0b',   3),   # mov rcx, [rbx]        load it back into RCX
        ('4821c1',   3),   # and rcx, rax          value-dependent taint
        ('48894b08', 4),   # mov [rbx+8], rcx      store at [RBX+8]
    ]
    code = b''.join(bytes.fromhex(hx) for hx, _ in prog)

    stub = r'void _start(){ __asm__ volatile("syscall"::"a"(60),"D"(0):"rcx","r11"); }'
    fd, binary = tempfile.mkstemp(suffix='.elf')
    os.close(fd)
    subprocess.run(['gcc', '-nostdlib', '-static', '-no-pie', '-fno-pie', '-O0',
                    '-fno-stack-protector', '-o', binary, '-x', 'c', '-'],
                   input=stub.encode(), check=True)
    try:
        ql = Qiling([binary], '/', verbose=QL_VERBOSE.OFF)
        wrapper = MicrotaintWrapper(ql, check_bof=False, check_uaf=False, check_sc=False,
                                    reporter=Reporter(json_mode=False))
        hook = wrapper._make_cython_hook()
        assert isinstance(hook, InstructionHook)
        X = 0x4446000       # code page
        D = 0x4447000       # data page (pointer target)
        for base in (X, D):
            try:
                ql.uc.mem_map(base, 0x1000)
            except Exception:
                pass
        ql.uc.mem_write(X, code)

        wrapper.register_taint.clear()
        wrapper.register_taint['RAX'] = FULL
        wrapper._any_taint = True
        ql.uc.reg_write(UC_X86_REG_RAX, 0xCAFEF00DDEADBEEF)
        ql.uc.reg_write(UC_X86_REG_RBX, D)  # pointer value -> address computation

        offset = 0
        for _, size in prog:
            hook(None, X + offset, size, None)
            offset += size

        final_taint = {k: v for k, v in wrapper.register_taint.items() if v}
        # shadow taint at the two written addresses
        shadow_D = wrapper.shadow_mem.read_mask(D, 8)
        shadow_D8 = wrapper.shadow_mem.read_mask(D + 8, 8)
        print(json.dumps({
            'final_taint': {k: hex(v) for k, v in sorted(final_taint.items())},
            'shadow_D': hex(shadow_D), 'shadow_D8': hex(shadow_D8),
        }))
    finally:
        os.unlink(binary)


if __name__ == '__main__':
    _main()
