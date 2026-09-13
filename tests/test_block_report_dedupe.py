"""A leak site is described once, not once per time the branch executes.

The runtime files a report every time it finds the program counter tainted, so
a secret-dependent branch inside a loop is filed once per iteration.  Each one
then crossed into Python to be disassembled and formatted.  On a static-glibc
guest that was 109 reports per input at 51 distinct addresses, the SAME 51
every input, and the whole reporting path cost 0.89 M of a 7.65 M iteration.

Deduping is only safe if it loses nothing, so that is what these tests pin:

* every address reported without deduping is still reported with it,
* the total occurrence count is unchanged, because suppressing a duplicate
  must not make the engine claim the branch fired fewer times,
* and the suppressed count accounts for exactly the difference.

The last one matters most.  A dedupe that silently dropped findings and a
dedupe that correctly collapsed them look identical from the outside unless
the arithmetic is checked, which is why `reports` keeps counting every
occurrence and `reports_duplicate` is exposed next to it.
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

#: (addresses reported, block counters).
type Leaks = tuple[list[int], dict[str, int]]

#: Two secret-dependent loop trip counts.  A LOOP is the point: the branch is
#: taken many times per run, so the same address is reported many times, which
#: is the redundancy being removed.  A straight-line `if` would be reported
#: once and could not show the difference.
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
  for (unsigned k = 0; k < (unsigned)(in[1] & 31); k++) acc += k * 3;
  g_out = acc;
  sys_exit(0);
}
"""


def _build(src: str) -> str:
    fd, path = tempfile.mkstemp(suffix='.elf')
    os.close(fd)
    built = subprocess.run(
        ['gcc', '-nostdlib', '-static', '-no-pie', '-O1', '-fno-stack-protector',
         '-o', path, '-x', 'c', '-'],
        input=src.encode(), capture_output=True, check=False)
    if built.returncode != 0:
        os.unlink(path)
        pytest.skip(f'cannot build the guest: {built.stderr.decode()[:300]}')
    return path


def _run(guest: str, *, dedupe: bool) -> Leaks:
    from qiling import Qiling
    from qiling.const import QL_VERBOSE

    from microtaint.emulator.reporter import FindingKind, Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper

    prev = os.environ.get('MICROTAINT_BLOCK')
    os.environ['MICROTAINT_BLOCK'] = '1'
    saved, dn = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    try:
        rep = Reporter(json_mode=False, stream=io.StringIO())
        ql = Qiling([guest], '/', verbose=QL_VERBOSE.OFF)
        ql.os.stdin = io.BytesIO(bytes((i * 7 + 13) & 0xFF for i in range(32)))
        w = MicrotaintWrapper(ql, reporter=rep, dedupe_reports=dedupe)
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
        leaks = [f.address for f in rep.findings
                 if f.kind in (FindingKind.SIDE_CHANNEL, FindingKind.BOF)]
        return leaks, stats
    finally:
        os.dup2(saved, 1)
        os.close(dn)
        os.close(saved)
        if prev is None:
            os.environ.pop('MICROTAINT_BLOCK', None)
        else:
            os.environ['MICROTAINT_BLOCK'] = prev


@pytest.fixture(scope='module')
def guest() -> Iterator[str]:
    path = _build(_GUEST)
    yield path
    os.unlink(path)


@pytest.fixture(scope='module')
def both(guest: str) -> tuple[Leaks, Leaks]:
    return _run(guest, dedupe=False), _run(guest, dedupe=True)


def test_there_was_redundancy_to_remove(both: tuple[Leaks, Leaks]) -> None:
    """The premise.  Without repeated reports at one address every assertion
    below would hold trivially against a dedupe that does nothing at all."""
    (plain, plain_stats), _ = both
    assert plain, 'the guest reported no control-flow leak at all'
    assert len(plain) > len(set(plain)), (
        f'no address was reported twice ({len(plain)} reports at '
        f'{len(set(plain))} addresses), so this guest cannot show that '
        f'deduplication collapses anything; the loop was probably made '
        f'branchless')
    assert plain_stats['reports_duplicate'] == 0, (
        'reports_duplicate must be 0 when deduping is off')


def test_no_leak_site_is_lost(both: tuple[Leaks, Leaks]) -> None:
    """The property that makes it safe.  Fewer findings is only acceptable if
    the set of places the engine points at is identical."""
    (plain, _), (deduped, _) = both
    missed = sorted(set(plain) - set(deduped))
    assert not missed, (
        f'deduping lost {len(missed)} leak site(s) entirely: '
        f'{[hex(a) for a in missed]}')
    assert set(deduped) == set(plain), (
        f'deduping changed the set of sites: '
        f'{sorted(hex(a) for a in set(deduped) ^ set(plain))}')


def test_each_site_is_described_once(both: tuple[Leaks, Leaks]) -> None:
    _, (deduped, _) = both
    dupes = [a for a in set(deduped) if deduped.count(a) > 1]
    assert not dupes, (
        f'these addresses were still described more than once: '
        f'{[hex(a) for a in dupes]}')


def test_the_occurrence_count_is_not_hidden(both: tuple[Leaks, Leaks]) -> None:
    """Suppressing a duplicate must not make the engine claim the branch fired
    fewer times.  `reports` counts occurrences either way.

    Anchored to the findings the un-deduped run actually produced, not merely
    to the other arm's counter.  Comparing the two counters ALONE passes when
    both are broken together: a mutation that removed the increment entirely
    left both at zero and every assertion here green.
    """
    (plain, plain_stats), (_, dedup_stats) = both
    assert plain_stats['reports'] == len(plain), (
        f"the occurrence counter disagrees with the findings the un-deduped "
        f"run produced: {plain_stats['reports']} counted, {len(plain)} "
        f"reported")
    assert dedup_stats['reports'] == len(plain), (
        f"deduping changed the occurrence count: {len(plain)} occurrences "
        f"without it, {dedup_stats['reports']} counted with it")


def test_the_suppressed_count_accounts_for_the_difference(
        both: tuple[Leaks, Leaks]) -> None:
    """The arithmetic that tells a correct collapse from a silent drop."""
    (plain, _), (deduped, dedup_stats) = both
    assert dedup_stats['reports_duplicate'] > 0, (
        'nothing was suppressed, so deduping did not run')
    assert dedup_stats['reports_sites'] == len(set(deduped)), (
        f"the runtime remembers {dedup_stats['reports_sites']} distinct "
        f"findings but described {len(set(deduped))}")
    assert (len(deduped) + dedup_stats['reports_duplicate']
            == len(plain)), (
        f'{len(deduped)} described + {dedup_stats["reports_duplicate"]} '
        f'suppressed != {len(plain)} reported without deduping')


def test_deduping_is_the_default(guest: str) -> None:
    """The option exists to turn it OFF; a caller who passes nothing gets the
    collapsed form, because describing a loop's branch once per iteration is
    noise rather than information."""
    from qiling import Qiling
    from qiling.const import QL_VERBOSE

    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper
    rep = Reporter(json_mode=False, stream=io.StringIO())
    ql = Qiling([guest], '/', verbose=QL_VERBOSE.OFF)
    w = MicrotaintWrapper(ql, reporter=rep)
    assert w.dedupe_reports is True
