"""Which leak sites did THIS input reach?

Deduplication describes a site once for the life of the hook, which answers
"where does this binary leak" and deliberately does not answer "what did input
437 touch": input 437 emits no finding at all once the site is known.  A fuzzer
and a concolic loop need the second answer, and paying a Python call per site
per input to get it gives back half of what deduplication saved.

So the runtime records it instead.  Each distinct finding is assigned a dense
site id when it is first seen, and each iteration sets a bit per site it hit.
`RunOutcome.sites` is that bitmap resolved back to (address, mask), computed
without entering Python once per finding.

The guest below takes one of two secret-dependent paths, so the answer is
genuinely per input rather than the same set every time -- which is the only
way this test can tell a real bitmap from one that always returns everything.
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

#: One always-reached secret branch, then one of two secret-dependent loops.
#: An input with bit 0 of the first byte set runs the first loop and never the
#: second, so the two inputs must produce DIFFERENT site sets with a common
#: element.  A trip count cannot be turned into a conditional move, which is
#: what keeps the loops branches at -O1.
_GUEST = r"""
long sys_read(int fd, void *buf, unsigned long n){long r;
  __asm__ volatile("syscall":"=a"(r):"0"(0),"D"(fd),"S"(buf),"d"(n):"rcx","r11","memory");return r;}
void sys_exit(int c){__asm__ volatile("syscall"::"a"(60),"D"((long)c):"rcx","r11");__builtin_unreachable();}
volatile unsigned long g_out;
__attribute__((noinline)) void body(void) {
  unsigned char in[32];
  long n = sys_read(0, in, 32);
  if (n <= 0) return;
  unsigned long acc = 0;
  if (in[0] & 1) {
    for (unsigned k = 0; k < (unsigned)(in[1] & 15); k++) acc += k;
  } else {
    for (unsigned k = 0; k < (unsigned)(in[1] & 15); k++) acc += k * 7;
  }
  g_out = acc;
}
void _start(void) { body(); sys_exit(0); }
"""


@pytest.fixture(scope='module')
def guest() -> Iterator[tuple[str, int]]:
    fd, path = tempfile.mkstemp(suffix='.elf')
    os.close(fd)
    built = subprocess.run(
        ['gcc', '-nostdlib', '-static', '-no-pie', '-O1', '-fno-stack-protector',
         '-o', path, '-x', 'c', '-'],
        input=_GUEST.encode(), capture_output=True, check=False)
    if built.returncode != 0:
        os.unlink(path)
        pytest.skip(f'cannot build the guest: {built.stderr.decode()[:300]}')
    nm = subprocess.run(['nm', path], capture_output=True, text=True,
                        check=False).stdout
    syms = {line.split()[-1]: int(line.split()[0], 16)
            for line in nm.splitlines() if len(line.split()) >= 3}
    yield path, syms['body']
    os.unlink(path)


#: (per-iteration site sets, the hook's counters at the end).
type Runs = tuple[list[frozenset[int]], dict[str, int]]


@pytest.fixture(scope='module')
def runs(guest: tuple[str, int]) -> Runs:
    """Four inputs through one checkpoint loop: odd, even, odd, even."""
    from qiling import Qiling
    from qiling.const import QL_VERBOSE

    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper

    binary, body = guest
    prev = os.environ.get('MICROTAINT_BLOCK')
    os.environ['MICROTAINT_BLOCK'] = '1'
    saved, dn = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    out: list[frozenset[int]] = []
    try:
        rep = Reporter(json_mode=True, stream=io.StringIO())
        ql = Qiling([binary], '/', verbose=QL_VERBOSE.OFF)
        ql.os.stdin = io.BytesIO(b'')
        w = MicrotaintWrapper(ql, reporter=rep)
        os.dup2(dn, 1)
        w.run_to(body)
        cp = w.checkpoint()
        for i in range(4):
            w.restore(cp)
            # byte 0 selects the path; byte 1 gives both loops a trip count
            ql.os.stdin = io.BytesIO(bytes([i & 1, 9] + [0] * 30))
            rep.findings.clear()
            outcome = w.resume(cp)
            out.append(frozenset(a for a, _m in outcome.sites))
        stats = w.block_mode_stats() or {}
        os.dup2(saved, 1)
    finally:
        os.dup2(saved, 1)
        os.close(dn)
        os.close(saved)
        if prev is None:
            os.environ.pop('MICROTAINT_BLOCK', None)
        else:
            os.environ['MICROTAINT_BLOCK'] = prev
    return out, stats


def test_every_iteration_reports_sites(runs: Runs) -> None:
    """The premise.  An empty bitmap everywhere would satisfy "the sets differ
    per input" only vacuously, and would satisfy nothing else here."""
    sites, _ = runs
    assert len(sites) == 4, f'expected 4 iterations, got {len(sites)}'
    empty = [i for i, s in enumerate(sites) if not s]
    assert not empty, (
        f'iterations {empty} reported no leak site at all, so this guest '
        f'cannot show per-input attribution')


def test_the_sets_are_per_input_not_per_binary(runs: Runs) -> None:
    """The property.  A bitmap that always returned every known site would
    pass every other test in this file."""
    sites, _ = runs
    odd, even = sites[1], sites[0]      # input i=1 has bit 0 set, i=0 does not
    assert odd != even, (
        f'both paths reported the same sites ({sorted(hex(a) for a in odd)}), '
        f'so the answer is not per input; the compiler most likely made the '
        f'selecting branch a conditional move')


def test_the_same_input_gives_the_same_answer(runs: Runs) -> None:
    """Iterations 0 and 2 run identical input, as do 1 and 3.  A bitmap that
    is not cleared between iterations would grow instead of repeating."""
    sites, _ = runs
    assert sites[0] == sites[2], (
        f'the same input gave different answers on iterations 0 and 2: '
        f'{sorted(hex(a) for a in sites[0] ^ sites[2])}')
    assert sites[1] == sites[3], (
        f'the same input gave different answers on iterations 1 and 3: '
        f'{sorted(hex(a) for a in sites[1] ^ sites[3])}')


def test_the_paths_share_the_branch_that_selects_them(runs: Runs) -> None:
    """The `if` is reached by every input, so the two sets must overlap.  Two
    disjoint sets would mean the bitmap is keyed on the input rather than on
    what the input reached."""
    sites, _ = runs
    assert sites[0] & sites[1], (
        f'the two paths share no leak site, but both run the same '
        f'secret-dependent `if`: {sorted(hex(a) for a in sites[0])} vs '
        f'{sorted(hex(a) for a in sites[1])}')


def test_every_site_is_one_the_runtime_knows(runs: Runs) -> None:
    """A site id that resolved to a bogus address would be an out-of-range
    read dressed up as an answer."""
    sites, stats = runs
    union = set().union(*sites)
    assert len(union) <= stats['reports_sites'], (
        f'{len(union)} distinct addresses reported across the iterations but '
        f'the runtime only ever recorded {stats["reports_sites"]} sites')
    assert all(a > 0 for a in union), f'a site resolved to address 0: {union}'


def test_deduplication_still_holds(runs: Runs) -> None:
    """Per-input attribution must not have been bought by re-describing
    everything: the occurrences still collapse."""
    _, stats = runs
    assert stats['reports_duplicate'] > 0, (
        'nothing was suppressed, so the bitmap was paid for by giving up '
        'deduplication')
    assert stats['reports'] > stats['reports_sites'], (
        f"{stats['reports']} occurrences at {stats['reports_sites']} sites: "
        f"no site repeated, so this guest cannot show the two coexisting")
