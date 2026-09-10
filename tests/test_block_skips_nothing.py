"""Block mode must not skip a block, on a binary made of real library code.

A block the lowering refuses is SKIPPED by the C hook: nothing computes its
taint, and a tainted value that moves through one comes out clean.  That is an
under-taint, the one direction that is never acceptable, and it is invisible
without asking -- the run completes, the answer looks like an answer, and the
only trace is a counter nobody reads.

The hand-written `-nostdlib` guests in benchmark/taint_density refuse nothing at
all, so none of this shows there.  A static-glibc binary does: the trace covers
the vectorised string and memory routines, thread-local storage, and the bit
scans, which is where every refusal has been found so far.

Marked `slow` because it compiles a real binary and emulates it; it is the kind
of check that belongs on a release rather than on every edit.
"""
from __future__ import annotations

import io
import os
import platform
import subprocess
import tempfile
from collections.abc import Iterator

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(platform.system() != 'Linux',
                       reason='emulator tests require Linux'),
]

#: Deliberately ordinary: fgets, strlen, malloc, memcpy, a byte loop and
#: snprintf.  What matters is that glibc's own implementations get executed,
#: not that the program is interesting.
_GUEST = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    char buf[128];
    if (!fgets(buf, sizeof buf, stdin)) return 1;
    size_t n = strlen(buf);
    char *copy = malloc(n + 1);
    memcpy(copy, buf, n + 1);
    for (size_t i = 0; i < n; i++) copy[i] = (char)(copy[i] ^ 0x20);
    unsigned long h = 5381;
    for (size_t i = 0; i < n; i++) h = h * 33 + (unsigned char)copy[i];
    char out[64];
    snprintf(out, sizeof out, "%lu %zu", h, n);
    fputs(out, stdout);
    free(copy);
    return (int)(h & 1);
}
"""


@pytest.fixture(scope='module')
def guest() -> Iterator[str]:
    fd, path = tempfile.mkstemp(suffix='.elf')
    os.close(fd)
    built = subprocess.run(['gcc', '-static', '-O1', '-o', path, '-x', 'c', '-'],
                           input=_GUEST.encode(), capture_output=True,
                           check=False)
    if built.returncode != 0:
        os.unlink(path)
        pytest.skip(f'cannot build a static guest: {built.stderr.decode()[:300]}')
    yield path
    os.unlink(path)


@pytest.fixture(scope='module')
def stats(guest: str) -> dict[str, int]:
    from qiling import Qiling
    from qiling.const import QL_VERBOSE

    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper

    prev = os.environ.get('MICROTAINT_BLOCK')
    os.environ['MICROTAINT_BLOCK'] = '1'
    saved, dn = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    try:
        ql = Qiling([guest], '/', verbose=QL_VERBOSE.OFF)
        ql.os.stdin = io.BytesIO(b'hello world, this is a tainted line\n')
        w = MicrotaintWrapper(ql, reporter=Reporter(json_mode=False,
                                                    stream=io.StringIO()))
        os.dup2(dn, 1)
        ok = True
        try:
            ql.run()
        except Exception:
            ok = False
        os.dup2(saved, 1)
        w.block_mode_finish(ok)
        got = w.block_mode_stats()
        assert got is not None, 'block mode did not install'
        return got
    finally:
        os.dup2(saved, 1)
        os.close(dn)
        os.close(saved)
        if prev is None:
            os.environ.pop('MICROTAINT_BLOCK', None)
        else:
            os.environ['MICROTAINT_BLOCK'] = prev


def test_the_guest_exercises_real_library_code(stats: dict[str, int]) -> None:
    """The premise.  A trace of a dozen blocks says nothing about coverage, and
    a run where block mode never installed says nothing at all."""
    assert stats['blocks'] > 400, (
        f'only {stats["blocks"]} block executions; this is meant to be a real '
        f'binary, and a short trace cannot show what it is here to show')
    assert stats['planned'] > 100, (
        f'only {stats["planned"]} distinct blocks planned')


@pytest.mark.xfail(strict=True, reason=(
    'A p-code LOOP is the last shape the block lowering refuses, and on this '
    'guest every instance is `bsf` inside glibc"s memchr and strlen: SLEIGH '
    'models a bit scan as a loop over bit positions, not as an opcode.  The '
    'taint IR is straight-line by construction, so a backward BRANCH cannot '
    'be expressed as one program.  The fix is to UNROLL such a loop to the '
    'operand width -- the predication machinery already turns each exit test '
    'into a select, so the unrolled form is exact -- with a runtime-guarded '
    'floor for the case the unrolling does not provably finish.  Three '
    'distinct blocks; every other cause has been removed.  Strict, so that '
    'finishing it fails here and says to delete this marker.'))
def test_block_mode_skips_nothing(stats: dict[str, int]) -> None:
    """The property.  Not "few blocks are skipped": none are.

    A skipped block is unanalysed code, and taint that flows through one is
    simply lost, so any number above zero is an under-taint waiting for the
    right input.
    """
    assert stats['unhandled'] == 0, (
        f'{stats["unhandled"]} of {stats["blocks"]} block executions were '
        f'skipped, so their taint was never computed: {stats}')
