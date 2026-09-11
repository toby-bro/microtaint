"""The second run of a binary must compile nothing, and find the same leaks.

This is the property the whole compiled-block cache exists for, and it is the
one that cannot be checked without a real guest: it depends on two separately
constructed wrappers agreeing, to the slot, about what the register file looks
like.  If they ever stopped agreeing -- a register interned in a different
order, an architecture detected differently -- every cache key would differ,
every block would be compiled again, and nothing anywhere would say so.  Block
mode would simply go back to being slower than the path it replaces.

Measured on bench_dense before the cache existed, three runs in one process:

    block mode        451.8 ms   428.6 ms   426.8 ms
    per instruction   659.5 ms   170.7 ms   170.1 ms

The per-instruction path has a process-wide rule cache and drops 3.9x on its
second run.  Block mode had none and lost to it from run 2 onward, which is the
only regime a fuzzer is ever in.  Afterwards: 771 ms, then 82 ms, then 82 ms.

Marked `slow` because it compiles a real binary and emulates it twice.
"""
from __future__ import annotations

import io
import os
import platform
import subprocess
import tempfile
from collections.abc import Iterator

import pytest

from microtaint.taint_ir import blockcompile as bc

#: (what the compiler did, what the block hook saw, the findings).
_Run = tuple[dict[str, int], dict[str, int], list[dict[str, object]]]

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(platform.system() != 'Linux',
                       reason='emulator tests require Linux'),
]

#: Reads a line, hashes it, and branches on the hash.  Ordinary on purpose: the
#: point is that glibc's own vectorised string and memory routines run, because
#: those are the blocks a real binary is actually made of.
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
    unsigned long h = 5381;
    for (size_t i = 0; i < n; i++) h = h * 33 + (unsigned char)copy[i];
    /* A branch whose direction depends on the input, which is the finding. */
    if (h & 1) puts("odd"); else puts("even");
    char out[64];
    snprintf(out, sizeof out, "%lu %zu", h, n);
    fputs(out, stdout);
    free(copy);
    return (int)(h & 3);
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


def _run(guest: str) -> tuple[dict[str, int], dict[str, int], list[dict[str, object]]]:
    """One whole run under block mode.

    -> (what the compiler did, what the block hook saw, the findings).

    The findings are read off the reporter as structured records rather than as
    rendered text: what is being compared is what was FOUND, not how it printed.
    """
    from qiling import Qiling
    from qiling.const import QL_VERBOSE

    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper

    before = bc.cache_stats()
    reporter = Reporter(json_mode=True, stream=io.StringIO())
    prev = os.environ.get('MICROTAINT_BLOCK')
    os.environ['MICROTAINT_BLOCK'] = '1'
    saved, dn = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    try:
        ql = Qiling([guest], '/', verbose=QL_VERBOSE.OFF)
        ql.os.stdin = io.BytesIO(b'hello world, this is a tainted line\n')
        w = MicrotaintWrapper(ql, reporter=reporter)
        os.dup2(dn, 1)
        ok = True
        try:
            ql.run()
        except Exception:
            ok = False
        os.dup2(saved, 1)
        w.block_mode_finish(ok)
        stats = w.block_mode_stats()
        assert stats is not None, 'block mode did not install'
    finally:
        os.dup2(saved, 1)
        os.close(dn)
        os.close(saved)
        if prev is None:
            os.environ.pop('MICROTAINT_BLOCK', None)
        else:
            os.environ['MICROTAINT_BLOCK'] = prev
    after = bc.cache_stats()
    did = {k: after[k] - before[k] for k in ('hits', 'misses', 'refusals')}
    return did, dict(stats), [f.to_dict() for f in reporter.findings]


@pytest.fixture(scope='module')
def two_runs(guest: str) -> list[_Run]:
    bc.cache_clear()
    return [_run(guest), _run(guest)]


def test_the_first_run_actually_compiled_something(
        two_runs: list[_Run]) -> None:
    """The premise.  A run that planned nothing would make every assertion
    below true for the wrong reason."""
    did, stats, _ = two_runs[0]
    assert stats['planned'] > 100, (
        f'only {stats["planned"]} distinct blocks planned; this is meant to be '
        f'a real binary')
    assert did['misses'] > 50, (
        f'the first run compiled only {did["misses"]} blocks, so there is '
        f'almost nothing for the second run to avoid recompiling')


def test_the_second_run_compiles_nothing(
        two_runs: list[_Run]) -> None:
    """The property.  Not "fewer": none.

    A single miss on the second run means two wrappers disagreed about
    something the key carries, and one disagreement is every block.
    """
    first, second = two_runs[0][0], two_runs[1][0]
    assert second['misses'] == 0, (
        f'the second run compiled {second["misses"]} blocks again, of the '
        f'{first["misses"]} the first run compiled: two wrappers disagree '
        f'about the cache key, so block mode pays the compiler on every run')
    assert second['hits'] > 0, 'the second run did not consult the cache at all'


def test_the_second_run_plans_the_same_blocks(
        two_runs: list[_Run]) -> None:
    """And it is the same work, not a shorter run that simply met less code."""
    a, b = two_runs[0][1], two_runs[1][1]
    assert b['planned'] == a['planned'], (a, b)
    assert b['blocks'] == a['blocks'], (a, b)
    assert b['unhandled'] == 0, (
        f'{b["unhandled"]} block executions were skipped on the second run, '
        f'so their taint was never computed: {b}')


def test_a_run_off_the_cache_finds_exactly_what_a_compiled_one_found(
        two_runs: list[_Run]) -> None:
    """The soundness half.  Speed that changes the answer is not speed.

    The guest branches on a hash of its input, so there is something to find;
    a run that reported nothing would compare two empty strings and pass.
    """
    first, second = two_runs[0][2], two_runs[1][2]
    assert first, (
        'the first run reported no leak at all, so this compares nothing; the '
        'guest branches on a hash of its input and that branch is the finding')
    assert len(second) == len(first), (
        f'the second run, off cached blocks, reported {len(second)} findings '
        f'against {len(first)}')
    assert second == first, (
        'the second run, which compiled nothing and ran entirely off cached '
        'blocks, reported different leaks from the first')
