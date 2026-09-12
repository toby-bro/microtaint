"""The dynamic loader reading libc.so is not the program reading its input.

Every `read` is a taint source, which is right for a program's input and wrong
for ld.so pulling shared libraries off disk before the program has started.
The consequence is not a little extra taint: the loader then branches on data
the engine considers tainted, the engine reports a side channel in ld.so and
stops, and the program under analysis never runs at all.

Measured on a dynamically linked BOF guest: 29 instructions analysed, one
`taint_source` of 1024 bytes from fd=3, one `side_channel` inside
ld-linux-x86-64.so.2, and the buffer overflow in `main` never reached.

This hid behind the per-instruction hook's main-image address filter, which
happened to exclude the loader as well as libc.  Removing that filter -- it was
dropping taint through JIT'd code, see test_taint_outside_the_main_image.py --
exposed it.

So: a read that happens before the guest reaches its OWN entry point is the
loader loading the program.  A static binary has no such reads and is
unaffected.
"""
from __future__ import annotations

import io
import os
import platform
import subprocess
import tempfile
from collections.abc import Iterator

import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != 'Linux', reason='emulator tests require Linux',
)

_SRC = r"""
#include <unistd.h>
void vulnerable(void) { char buf[16]; read(0, buf, 64); }
int main(void) { vulnerable(); return 0; }
"""


@pytest.fixture(scope='module')
def dynamic_guest() -> Iterator[str]:
    fd, path = tempfile.mkstemp(suffix='.elf')
    os.close(fd)
    built = subprocess.run(['gcc', '-O0', '-o', path, '-x', 'c', '-'],
                           input=_SRC.encode(), capture_output=True, check=False)
    if built.returncode != 0:
        os.unlink(path)
        pytest.skip(f'cannot build the guest: {built.stderr.decode()[:300]}')
    yield path
    os.unlink(path)


def _run(guest: str, block: bool,
         ) -> tuple[list[dict[str, object]], tuple[int, int]]:
    """-> (findings, the guest's own image range)."""
    from qiling import Qiling
    from qiling.const import QL_VERBOSE

    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper

    prev = os.environ.get('MICROTAINT_BLOCK')
    os.environ['MICROTAINT_BLOCK'] = '1' if block else '0'
    saved, dn = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    try:
        rep = Reporter(json_mode=True, stream=io.StringIO())
        ql = Qiling([guest], '/', verbose=QL_VERBOSE.OFF)
        ql.os.stdin = io.BytesIO(b'A' * 64)
        w = MicrotaintWrapper(ql, reporter=rep)
        os.dup2(dn, 1)
        ok = True
        try:
            ql.run()
        except Exception:
            ok = False
        os.dup2(saved, 1)
        if block:
            w.block_mode_finish(ok)
        image = ql.loader.images[0]
        return ([f.to_dict() for f in rep.findings], (image.base, image.end))
    finally:
        os.dup2(saved, 1)
        os.close(dn)
        os.close(saved)
        if prev is None:
            os.environ.pop('MICROTAINT_BLOCK', None)
        else:
            os.environ['MICROTAINT_BLOCK'] = prev


@pytest.mark.parametrize('block', [True, False], ids=['block', 'instruction'])
def test_the_loaders_own_reads_are_not_taint_sources(
        dynamic_guest: str, block: bool) -> None:
    """Nothing the loader reads off disk counts as the program's input."""
    findings, _image = _run(dynamic_guest, block)
    sources = [f for f in findings if f.get('kind') == 'taint_source']
    assert sources, 'no taint was introduced at all, so this proves nothing'
    stray = [f for f in sources if f.get('source') not in (None, 'stdin')]
    assert not stray, (
        f"the loader's own reads were taken as program input: {stray[:3]}")


@pytest.mark.parametrize('block', [True, False], ids=['block', 'instruction'])
def test_the_program_actually_runs(dynamic_guest: str, block: bool) -> None:
    """And the program under analysis gets to run.

    Reported before the fix: 29 instructions, and a side channel inside
    ld-linux rather than anything belonging to the guest.
    """
    findings, (base, end) = _run(dynamic_guest, block)
    leaks = [f for f in findings if f.get('kind') != 'taint_source']
    assert leaks, 'nothing was reported at all, so this proves nothing'
    where = [int(str(f.get('address')), 16) for f in leaks]
    assert any(base <= a < end for a in where), (
        f"every finding is outside the guest's own image "
        f'[0x{base:x}, 0x{end:x}): {[hex(a) for a in where]}.  The engine '
        f'never got past the dynamic loader, so the program under analysis '
        f'did not run.')
