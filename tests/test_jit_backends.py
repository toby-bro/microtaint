"""Every host backend against the interpreter, on the host and cross-compiled.

The emitter is the only part of the engine whose correctness cannot be checked
by importing it: it produces machine code for whatever host it was built for,
and the host this suite runs on is one of two.  So the differential harness is
a standalone C program -- interpreter and emitter side by side, no Python -- and
the AArch64 backend is checked by cross-compiling it and running it under
qemu-user.  A disagreement is then attributable to the backend rather than to
the harness, because the same harness passes on the host backend.

Both backends are exercised the same two ways:

  * DIRECTED, one opcode at a time over the values its rule turns on (0, 1, 63,
    64, the sign bit), which says WHICH encoding is wrong when one is.
  * RANDOM programs in the shape the lowering produces, which is what finds the
    register-allocator faults an opcode-at-a-time sweep cannot: both of the
    ones this harness found on first run were allocator faults, not encodings.

Skipped, not failed, where the cross toolchain or qemu is missing: those are
the developer's tools, and a machine without them can still run everything else.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sysconfig

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_HERE, 'jit_a64', 'jit_selftest.c')
_INC = os.path.join(_ROOT, 'microtaint', 'instrumentation', 'cell_c')

#: Enough programs to reach the spill paths on both hosts without making the
#: suite wait: the x86-64 pool is ten registers and the AArch64 one is
#: twenty-five, so a program has to be a few dozen nodes deep before either
#: spills, and the generator sizes them for that.
_ITERS = '3000'


def _build(cc: str, extra: list[str], out: str) -> str:
    subprocess.run([cc, '-O2', *extra, f'-I{_INC}', '-o', out, _SRC],  # noqa: S603
                   check=True, capture_output=True)
    return out


def _run(argv: list[str]) -> str:
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)  # noqa: S603
    assert proc.returncode == 0, (
        f'{argv[0]} reported a disagreement between the emitter and the '
        f'interpreter:\n{proc.stdout}\n{proc.stderr}')
    return proc.stdout


@pytest.fixture(scope='module')
def host_harness(tmp_path_factory) -> str:
    cc = sysconfig.get_config_var('CC') or 'cc'
    cc = cc.split()[0]
    if not shutil.which(cc):
        pytest.skip(f'no host compiler ({cc})')
    out = str(tmp_path_factory.mktemp('jit') / 'selftest_host')
    try:
        return _build(cc, [], out)
    except subprocess.CalledProcessError as exc:
        if b'no backend for this host' in (exc.stderr or b''):
            pytest.skip('no emitter for this host; the interpreter is the path here')
        raise


@pytest.fixture(scope='module')
def a64_harness(tmp_path_factory) -> str:
    cc = 'aarch64-linux-gnu-gcc'
    if not shutil.which(cc):
        pytest.skip('no aarch64 cross compiler')
    if not shutil.which('qemu-aarch64'):
        pytest.skip('no qemu-aarch64 to run it under')
    out = str(tmp_path_factory.mktemp('jit') / 'selftest_a64')
    return _build(cc, ['-static'], out)


def test_host_backend_matches_the_interpreter_per_opcode(host_harness: str) -> None:
    out = _run([host_harness, 'directed'])
    assert '0 opcodes wrong' in out, out


def test_host_backend_matches_the_interpreter_on_random_programs(host_harness: str) -> None:
    out = _run([host_harness, _ITERS])
    assert '0 mismatches' in out, out
    # A backend that declines everything would pass the comparison and mean
    # nothing, so the count that matters is what it accepted.
    compiled = int(out.split('backend:')[1].split()[0])
    assert compiled > int(_ITERS) // 2, f'the emitter took almost nothing: {out}'


def test_aarch64_backend_matches_the_interpreter_per_opcode(a64_harness: str) -> None:
    out = _run(['qemu-aarch64', a64_harness, 'directed'])
    assert '0 opcodes wrong' in out, out


def test_aarch64_backend_matches_the_interpreter_on_random_programs(a64_harness: str) -> None:
    out = _run(['qemu-aarch64', a64_harness, _ITERS])
    assert '0 mismatches' in out, out
    compiled = int(out.split('backend:')[1].split()[0])
    assert compiled > int(_ITERS) // 2, f'the emitter took almost nothing: {out}'
