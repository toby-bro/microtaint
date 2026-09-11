"""Block mode must REPORT the leaks it finds, not just compute them.

The C runtime detects a secret-dependent program counter -- it checks the PC
slot's taint at the end of every region and files a report -- and the test
runner has drained those since the day it was written.  The live UC_HOOK_BLOCK
path never did.  So on a real binary block mode computed that a branch in
glibc's `_IO_getline_info` depended on stdin, held the finding with the block's
taint, and then dropped it on the next commit: the run finished, the answer
looked like an answer, and the leak was simply not in it.

That is the worst shape of under-report there is, because the whole point of
the engine is the finding.  It hid because the tests drove the RUNNER and the
binaries drove the HOOK, and only one of the two had a reader.

The gate is a comparison against the per-instruction path on the same guest,
which is the path that was already right.  Block mode is allowed to report MORE
(it does not stop the emulator, so it keeps finding leaks after the first),
but every address the per-instruction path reports has to be in its answer.
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

#: One run's answer: the addresses it reported as control-flow leaks, and the
#: block counters (None when block mode did not install, which would mean the
#: comparison had run the same path twice).
type Leaks = tuple[list[int], dict[str, int] | None]

#: A branch whose DIRECTION depends on a byte that came from stdin, plus a
#: second one later so "found the first and stopped looking" is visible.
#:
#: Written as loops rather than as `if (x) a; else b;` on purpose: at -O1 a
#: compiler folds the latter into a conditional move, which is branchless and
#: therefore not a leak at all.  A trip count cannot be cmov'd.  The premise
#: test below fails loudly if that ever stops being true.
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
  for (unsigned k = 0; k < (unsigned)(in[0] & 7); k++) acc += k;   /* secret */
  for (unsigned k = 0; k < (unsigned)(in[1] & 7); k++) acc += k * 3; /* again */
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


def _run(guest: str, *, block: bool) -> Leaks:
    """(addresses reported as control-flow leaks, block counters)."""
    from qiling import Qiling
    from qiling.const import QL_VERBOSE

    from microtaint.emulator.reporter import FindingKind, Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper

    prev = os.environ.get('MICROTAINT_BLOCK')
    os.environ['MICROTAINT_BLOCK'] = '1' if block else '0'
    saved, dn = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    try:
        rep = Reporter(json_mode=False, stream=io.StringIO())
        ql = Qiling([guest], '/', verbose=QL_VERBOSE.OFF)
        ql.os.stdin = io.BytesIO(bytes((i * 7 + 13) & 0xFF for i in range(32)))
        w = MicrotaintWrapper(ql, reporter=rep)
        os.dup2(dn, 1)
        ok = True
        try:
            ql.run()
        except Exception:            # a guest that faults is still a comparison
            ok = False
        os.dup2(saved, 1)
        w.block_mode_finish(ok)
        leaks = [f.address for f in rep.findings
                 if f.kind in (FindingKind.SIDE_CHANNEL, FindingKind.BOF)]
        return leaks, w.block_mode_stats()
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
    return _run(guest, block=False), _run(guest, block=True)


def test_the_guest_actually_leaks(both: tuple[Leaks, Leaks]) -> None:
    """The premise, and it is not a formality.

    If the compiler turned the secret-dependent loop into branchless code there
    would be no leak to find, and every assertion below would pass against a
    block mode that reports nothing whatsoever -- which is precisely the defect
    they exist to catch.  So the path known to be right has to find one first.
    """
    (instr_leaks, _), _ = both
    assert instr_leaks, (
        'the per-instruction path found no control-flow leak in a guest '
        'written to have two, so this comparison cannot show anything; the '
        'likely cause is the compiler making the branch branchless')


def test_block_mode_installed(both: tuple[Leaks, Leaks]) -> None:
    """A run where block mode never installed compares the same path twice."""
    _, (_, stats) = both
    assert stats is not None, 'block mode did not install'
    assert stats['blocks'] > 0, 'the block hook never fired'
    assert stats['unhandled'] == 0, (
        f'{stats["unhandled"]} blocks were skipped, so this guest cannot show '
        f'whether a HANDLED block reports: {stats}')


def test_block_mode_reports_every_leak_the_instruction_path_does(both: tuple[Leaks, Leaks]) -> None:
    """The property.  Not "block mode reports something": it reports these.

    Block mode may report MORE -- it does not stop the emulator, so it goes on
    finding leaks where the per-instruction path stops at the first -- but a
    finding the other path makes and this one does not is a leak that a user
    running with `MICROTAINT_BLOCK=1` would never hear about.
    """
    (instr_leaks, _), (block_leaks, _) = both
    missed = sorted(set(instr_leaks) - set(block_leaks))
    assert not missed, (
        f'block mode missed {len(missed)} of {len(set(instr_leaks))} leaks: '
        f'{[hex(a) for a in missed]}; it reported {[hex(a) for a in block_leaks]}')


def test_the_runtime_counts_every_finding_it_makes(both: tuple[Leaks, Leaks]) -> None:
    """`reports` counts findings even when the ring overflows.

    The ring is fixed size and the drain empties it, so a run that found more
    than it could hold must still say how many there were -- otherwise "fewer
    findings" and "fewer leaks" are the same number and neither is trustworthy.
    """
    _, (block_leaks, stats) = both
    assert stats is not None
    assert stats['reports'] >= len(block_leaks), (
        f'{stats["reports"]} reports counted but {len(block_leaks)} emitted')
    assert stats['reports'] > 0, 'the runtime counted no findings at all'
    assert stats['reports_pending'] == 0, (
        f'{stats["reports_pending"]} findings still undrained after finish')
