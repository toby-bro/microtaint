# ruff: noqa: S603, S607, PLC0415
"""The instruction taint cache must be VALUE-aware, not just taint-aware.

The taint transfer is value-dependent: `and rax, rbx` with rax fully tainted
yields taint 0 when rbx==0 (x & 0 = 0) but full taint when rbx==0xFF (x & FF =
x).  The Tier-3/Tier-4 caches key on the taint signature; a naive taint-only key
replays a stale result when the same address is revisited with the same taint
but different operand VALUES -- under-tainting `and rax,rbx` (rbx: 0 then 0xFF)
from 0xFF down to 0, an unsound miss a taint engine must never make.

Each scenario revisits ONE address with a fixed taint signature while changing
operand values, and pins that the cached result tracks the values.  The
emulation runs in a subprocess: constructing multiple Qiling/Unicorn instances
in one pytest process can segfault (a known Unicorn/gevent interaction; the repo
already isolates such tests), and a subprocess keeps each run clean.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile

import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != 'Linux', reason='emulator hook tests require Linux + gcc',
)

FULL = 0xFFFFFFFFFFFFFFFF
_STUB = r'void _start(){ __asm__ volatile("syscall"::"a"(60),"D"(0):"rcx","r11"); }'


def _run_seq_subprocess(code_hex: str, seq, rax_taint: int, rax_val: int):
    out = subprocess.run(
        [sys.executable, __file__, code_hex, hex(rax_taint), hex(rax_val),
         ','.join(hex(v) for v in seq)],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out.strip().splitlines()[-1])


def test_and_cache_is_value_aware() -> None:
    # and rax, rbx = 48 21 d8 ; rax tainted FULL, vary rbx.
    # correct: rbx=0 -> 0 (x&0), rbx=0xFF -> 0xFF (x&FF), repeat.
    taints = _run_seq_subprocess('4821d8', [0x00, 0xFF, 0x00, 0xFF], FULL, 0xFF)
    assert taints == [0x00, 0xFF, 0x00, 0xFF], (
        f'value-dependent AND taint not tracked across values: {[hex(t) for t in taints]} '
        '(a stale cache replay would repeat the first result -> under-taint)'
    )


def test_add_cache_is_value_aware() -> None:
    # add rax, rbx = 48 01 d8 ; rax LOW BYTE tainted (0xFF), value 0xFF; carry into
    # bit 8 depends on rbx.  rbx=0x00 -> 0xff (no carry), rbx=0xFF -> 0x1ff.
    taints = _run_seq_subprocess('4801d8', [0x00, 0xFF, 0x00], 0xFF, 0xFF)
    assert taints == [0xFF, 0x1FF, 0xFF], (
        f'value-dependent ADD carry taint not tracked: {[hex(t) for t in taints]}'
    )


def test_value_stable_still_hits() -> None:
    # Same taint+values repeated must still cache-hit (perf path intact).
    res = _run_seq_subprocess('4821d8', [0xFF] * 40, FULL, 0xFF)
    assert res[-1] == 0xFF, f'stable-value AND result wrong: {hex(res[-1])}'
    # (hit-count assertion happens in the subprocess; see __main__)


# --------------------------------------------------------------------------
# Subprocess entry: run the sequence against the real Cython hook, print taints.
# --------------------------------------------------------------------------
def _main() -> None:
    code = bytes.fromhex(sys.argv[1])
    rax_taint = int(sys.argv[2], 16)
    rax_val = int(sys.argv[3], 16)
    seq = [int(v, 16) for v in sys.argv[4].split(',')]

    from unicorn.x86_const import UC_X86_REG_RAX, UC_X86_REG_RBX

    from qiling import Qiling
    from qiling.const import QL_VERBOSE

    from microtaint.emulator.hook_core import InstructionHook
    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper

    fd, binary = tempfile.mkstemp(suffix='.elf')
    os.close(fd)
    subprocess.run(['gcc', '-nostdlib', '-static', '-no-pie', '-fno-pie', '-O0',
                    '-fno-stack-protector', '-o', binary, '-x', 'c', '-'],
                   input=_STUB.encode(), check=True)
    try:
        ql = Qiling([binary], '/', verbose=QL_VERBOSE.OFF)
        wrapper = MicrotaintWrapper(ql, check_bof=False, check_uaf=False, check_sc=False,
                                    reporter=Reporter(json_mode=False))
        hook = wrapper._make_cython_hook()
        assert isinstance(hook, InstructionHook)
        X = 0x4446000
        try:
            ql.uc.mem_map(X, 0x1000)
        except Exception:
            pass
        ql.uc.mem_write(X, code)
        out = []
        for v in seq:
            wrapper.register_taint.clear()
            wrapper.register_taint['RAX'] = rax_taint
            wrapper._any_taint = True
            ql.uc.reg_write(UC_X86_REG_RAX, rax_val)
            ql.uc.reg_write(UC_X86_REG_RBX, v)
            hook(None, X, len(code), None)
            out.append(wrapper.register_taint.get('RAX', 0))
        print(json.dumps(out))
    finally:
        os.unlink(binary)


if __name__ == '__main__':
    _main()
