"""`qiling_stats=False` turns off Qiling's own statistics, and nothing else.

Qiling records every syscall's name and arguments, and every string it sees,
for a summary that is logged at debug level and that microtaint never reads.
Measured on a checkpoint loop that is 1.16 M instructions an iteration of 8.97
-- 13% -- so a harness running one guest thousands of times wants it off.

Qiling ships `QlOsNullStats` for exactly this, which is why the option is one
line and not a patch over somebody else's internals.

The point of the test is that turning it off changes NOTHING about the answer:
the same findings, the same taint.  An optimisation that quietly changed what
the engine reports would be worth nothing.
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

_GUEST = r"""
#include <stdio.h>
#include <string.h>
int main(void) {
    char buf[128];
    if (!fgets(buf, sizeof buf, stdin)) return 1;
    size_t n = strlen(buf);
    unsigned long h = 5381;
    for (size_t i = 0; i < n; i++) h = h * 33 + (unsigned char)buf[i];
    if (h & 1) puts("odd"); else puts("even");
    return 0;
}
"""


@pytest.fixture(scope='module')
def guest() -> Iterator[str]:
    fd, path = tempfile.mkstemp(suffix='.elf')
    os.close(fd)
    built = subprocess.run(['gcc', '-static', '-O1', '-o', path, '-x', 'c', '-'],
                           input=_GUEST.encode(), capture_output=True, check=False)
    if built.returncode != 0:
        os.unlink(path)
        pytest.skip(f'cannot build the guest: {built.stderr.decode()[:300]}')
    yield path
    os.unlink(path)


def _run(guest: str, qiling_stats: bool) -> tuple[list[dict[str, object]], str]:
    from qiling import Qiling
    from qiling.const import QL_VERBOSE

    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper

    prev = os.environ.get('MICROTAINT_BLOCK')
    os.environ['MICROTAINT_BLOCK'] = '1'
    saved, dn = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    try:
        rep = Reporter(json_mode=True, stream=io.StringIO())
        ql = Qiling([guest], '/', verbose=QL_VERBOSE.OFF)
        ql.os.stdin = io.BytesIO(b'hello tainted world\n')
        w = MicrotaintWrapper(ql, reporter=rep, qiling_stats=qiling_stats)
        kind = type(ql.os.stats).__name__
        os.dup2(dn, 1)
        ok = True
        try:
            ql.run()
        except Exception:
            ok = False
        os.dup2(saved, 1)
        w.block_mode_finish(ok)
        return [f.to_dict() for f in rep.findings], kind
    finally:
        os.dup2(saved, 1)
        os.close(dn)
        os.close(saved)
        if prev is None:
            os.environ.pop('MICROTAINT_BLOCK', None)
        else:
            os.environ['MICROTAINT_BLOCK'] = prev


def test_the_option_selects_qilings_own_null_implementation(
        guest: str) -> None:
    """The premise: it is Qiling's switch, not ours."""
    _on, kind_on = _run(guest, True)
    _off, kind_off = _run(guest, False)
    assert kind_on == 'QlOsStats', kind_on
    assert kind_off == 'QlOsNullStats', kind_off


def test_turning_the_statistics_off_changes_no_finding(guest: str) -> None:
    """The property.  Speed that changes the answer is not speed."""
    with_stats, _a = _run(guest, True)
    without, _b = _run(guest, False)
    assert with_stats, (
        'the guest branches on a hash of its input and nothing was reported, '
        'so this compares two empty lists')
    assert without == with_stats, (
        f'{len(without)} findings without Qiling statistics against '
        f'{len(with_stats)} with them')
