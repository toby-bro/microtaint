"""Every counter the block hook claims to expose is actually in the dict.

`py_hook_stats` builds its dict with one `s:` pair per key in a hand-written
`Py_BuildValue` format string, and Py_BuildValue does NOT check that the format
and the argument list agree: a format one pair short silently drops the LAST
key and returns a perfectly valid dict.  That has now happened twice.  Adding
`abandoned` once made `reports_pending` disappear, and adding
`reports_duplicate`/`reports_sites` made `committed_writes` read as absent --
which looked exactly like "the guest committed no taint writes", a plausible
number that no assertion anywhere contradicted.

So this test names the keys.  It is deliberately a literal list rather than a
loop over something derived from the C, because the whole failure mode is the C
and the declaration drifting apart.
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

#: Every key `hook_stats` is documented to return, in blockpath_c.pyi.
_KEYS = frozenset({
    'blocks', 'handled', 'unhandled', 'planned', 'no_plan', 'no_regs',
    'cache_full', 'regs_clean', 'miss', 'invalidations', 'abandoned',
    'last_bad_addr', 'last_bad_size', 'reports', 'reports_pending',
    'reports_duplicate', 'reports_sites', 'sites_lost', 'sinks',
    'sinks_pending', 'sinks_distinct', 'reused', 'no_code',
    'committed_writes',
})

_GUEST = r"""
long sys_read(int fd, void *buf, unsigned long n){long r;
  __asm__ volatile("syscall":"=a"(r):"0"(0),"D"(fd),"S"(buf),"d"(n):"rcx","r11","memory");return r;}
void sys_exit(int c){__asm__ volatile("syscall"::"a"(60),"D"((long)c):"rcx","r11");__builtin_unreachable();}
volatile unsigned long g_out;
void _start(void){
  unsigned char in[32];
  long n = sys_read(0, in, 32);
  if (n <= 0) sys_exit(1);
  unsigned long acc = 0;
  for (unsigned k = 0; k < (unsigned)(in[0] & 31); k++) acc += k;
  g_out = acc;
  sys_exit(0);
}
"""


@pytest.fixture(scope='module')
def stats() -> Iterator[dict[str, int]]:
    fd, path = tempfile.mkstemp(suffix='.elf')
    os.close(fd)
    built = subprocess.run(
        ['gcc', '-nostdlib', '-static', '-no-pie', '-O1', '-fno-stack-protector',
         '-o', path, '-x', 'c', '-'],
        input=_GUEST.encode(), capture_output=True, check=False)
    if built.returncode != 0:
        os.unlink(path)
        pytest.skip(f'cannot build the guest: {built.stderr.decode()[:300]}')
    from qiling import Qiling
    from qiling.const import QL_VERBOSE

    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper
    prev = os.environ.get('MICROTAINT_BLOCK')
    os.environ['MICROTAINT_BLOCK'] = '1'
    saved, dn = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    try:
        rep = Reporter(json_mode=False, stream=io.StringIO())
        ql = Qiling([path], '/', verbose=QL_VERBOSE.OFF)
        ql.os.stdin = io.BytesIO(bytes((i * 7 + 13) & 0xFF for i in range(32)))
        w = MicrotaintWrapper(ql, reporter=rep)
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
        yield got
    finally:
        os.dup2(saved, 1)
        os.close(dn)
        os.close(saved)
        if prev is None:
            os.environ.pop('MICROTAINT_BLOCK', None)
        else:
            os.environ['MICROTAINT_BLOCK'] = prev
        os.unlink(path)


def test_no_documented_counter_is_missing(stats: dict[str, int]) -> None:
    missing = sorted(_KEYS - set(stats))
    assert not missing, (
        f'hook_stats dropped {missing}; the Py_BuildValue format string is '
        f'almost certainly one `s:` pair short of its argument list')


def test_no_undocumented_counter_appeared(stats: dict[str, int]) -> None:
    """The other direction: a key here that this test does not know about is a
    counter nobody declared, which is how the format string drifts."""
    extra = sorted(set(stats) - _KEYS)
    assert not extra, f'hook_stats returned undeclared keys {extra}'


def test_the_counters_that_must_move_did(stats: dict[str, int]) -> None:
    """The keys whose value this guest is guaranteed to move are non-zero.

    This is a premise check, not the drift gate: Py_BuildValue drops a
    displaced key ENTIRELY rather than zeroing it, so the missing-key test
    above is what catches the format-string bug.  This one catches the other
    way a stats assertion goes quiet, which is a guest that stopped exercising
    the thing being counted.  `committed_writes` is deliberately NOT here: this
    guest legitimately commits none, and asserting on it made the test fail for
    a reason that had nothing to do with the dict.
    """
    assert stats['blocks'] > 0, 'the block hook never fired'
    assert stats['handled'] > 0, 'no block was analysed'
    assert stats['reports'] > 0, (
        'the guest branches on a byte it read from stdin, so the runtime must '
        'have filed at least one finding')
