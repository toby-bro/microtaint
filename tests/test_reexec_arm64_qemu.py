# ruff: noqa: S603
"""The AArch64 native re-execution trampoline works (cross-compiled + qemu).

Proves ISA-generality of microtaint.reexec: the same C harness (reexec.c) drives
the AArch64 trampoline (reexec_arm64.S). We cross-compile the built-in self-test
for aarch64 and run it under qemu-aarch64, asserting each register-only
instruction executes on the (emulated) host CPU with the right result + NZCV.

Skipped when the aarch64 cross-compiler or qemu-aarch64 is unavailable.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

_CC = shutil.which('aarch64-linux-gnu-gcc')
_QEMU = shutil.which('qemu-aarch64')

pytestmark = pytest.mark.skipif(
    _CC is None or _QEMU is None,
    reason='needs aarch64-linux-gnu-gcc and qemu-aarch64',
)

_REEXEC = Path(__file__).resolve().parent.parent / 'microtaint' / 'reexec'


def test_arm64_reexec_selftest_under_qemu() -> None:
    tmp = Path(tempfile.mkdtemp(prefix='reexec_arm64_'))
    exe = tmp / 'rx_arm64'
    subprocess.run(
        [_CC, '-O2', '-static', '-DREEXEC_SELFTEST', '-o', str(exe),
         str(_REEXEC / 'reexec.c'), str(_REEXEC / 'reexec_arm64.S')],
        check=True,
    )
    out = subprocess.run([_QEMU, str(exe)], capture_output=True, text=True, timeout=60, check=False)
    combined = out.stdout + out.stderr
    assert 'ALL OK' in combined, f'arm64 reexec self-test failed:\n{combined}'
    # Each curated instruction must report OK (no FAIL lines).
    assert 'FAIL' not in combined, f'arm64 reexec self-test had failures:\n{combined}'
    assert out.returncode == 0, f'arm64 self-test exit {out.returncode}:\n{combined}'
