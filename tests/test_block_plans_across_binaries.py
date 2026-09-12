"""One process, two binaries, the same addresses: the plan table's identity.

A compiled block is kept for the life of the PROCESS now, and a fresh emulator
finds it in C without asking Python.  That is the whole point -- a fuzzer makes
a new emulator per input, and a campaign runs a hundred binaries in one
process -- and it is safe only because a kept plan is keyed on the block's
BYTES rather than on where it sits.

An address is not an identity.  Two static binaries built the same way load at
the same addresses, so a table keyed on (address, length) would hand the second
binary the plans compiled for the first, silently: taint computed for
instructions that are not there.

The two guests are therefore the same image with four bytes changed, at a
four-byte function written as raw bytes so that nothing else moves:

    48 89 f8 c3   mov rax, rdi ; ret      -> returns its argument,  TAINTED
    48 31 c0 c3   xor rax, rax ; ret      -> returns 0,             clean

Every block sits at the address and the length it did in the other image, and
exactly one differs.  Both orderings are run because they fail in opposite
directions:

    tainted after clean   a stale plan calls the result clean    -> UNDER-taint
    clean after tainted   a stale plan calls the result tainted  -> over-taint

An immediate is NOT enough to catch this and an earlier version of this file
patched one: changing a constant changes the value a block computes but not how
taint moves through it, so both plans agree and the test passed with the byte
check removed.  The rewrite has to change the DATAFLOW.

Marked `slow`: it builds a static guest and emulates it four times.
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

#: (what the block hook saw, the findings).
_Run = tuple[dict[str, int], list[dict[str, object]]]

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(platform.system() != 'Linux',
                       reason='emulator tests require Linux'),
]

#: The stub, and eight bytes after its `ret` that nothing ever executes.  They
#: are there to make the stub findable in the image exactly once: the four
#: instruction bytes alone occur all over a linked binary.
_TAINTED = bytes((0x48, 0x89, 0xf8, 0xc3))
_CLEAN = bytes((0x48, 0x31, 0xc0, 0xc3))
_ANCHOR = bytes((0xde, 0xad, 0xbe, 0xef, 0xca, 0xfe, 0xba, 0xbe))

#: No libc and raw syscalls, so the only tainted value in the program is what
#: the stub returns and a finding can be counted rather than filtered.
_GUEST = r"""
static long sys_read(int fd, void *buf, unsigned long n) {
  long r; __asm__ volatile("syscall" : "=a"(r)
      : "0"(0), "D"((long)fd), "S"(buf), "d"(n) : "rcx","r11","memory"); return r;
}
static void sys_write(int fd, const void *buf, unsigned long n) {
  long r; __asm__ volatile("syscall" : "=a"(r)
      : "0"(1), "D"((long)fd), "S"(buf), "d"(n) : "rcx","r11","memory"); (void)r;
}
static void sys_exit(int c) {
  __asm__ volatile("syscall" :: "a"(60), "D"((long)c) : "rcx","r11");
  __builtin_unreachable();
}

/* volatile, or gcc deletes the store as dead in a _start that never returns. */
volatile unsigned long slot;

__asm__(".globl mystub\n.align 16\nmystub:\n"
        " .byte 0x48,0x89,0xf8,0xc3\n"
        " .byte 0xde,0xad,0xbe,0xef,0xca,0xfe,0xba,0xbe\n");
extern unsigned long mystub(unsigned long);

void _start(void) {
  unsigned long v = 0;
  sys_read(0, &v, 8);                       /* v is the only tainted value */
  slot = mystub(v);
  if (slot & 1) sys_write(1, "T", 1); else sys_write(1, "t", 1);
  sys_exit(0);
}
"""


@pytest.fixture(scope='module')
def guests() -> Iterator[tuple[str, str]]:
    """(the image whose stub is tainted, the same image with a clean stub)."""
    fd, tainted = tempfile.mkstemp(suffix='.elf')
    os.close(fd)
    built = subprocess.run(
        ['gcc', '-static', '-nostdlib', '-O1', '-fno-stack-protector',
         '-o', tainted, '-x', 'c', '-'],
        input=_GUEST.encode(), capture_output=True, check=False)
    if built.returncode != 0:
        os.unlink(tainted)
        pytest.skip(f'cannot build the guest: {built.stderr.decode()[:300]}')
    image = open(tainted, 'rb').read()
    needle = _TAINTED + _ANCHOR
    if image.count(needle) != 1:
        os.unlink(tainted)
        pytest.skip(f'the stub appears {image.count(needle)} times, so patching '
                    f'it would not be the controlled change this needs')
    fd, clean = tempfile.mkstemp(suffix='.elf')
    os.close(fd)
    with open(clean, 'wb') as fh:
        fh.write(image.replace(needle, _CLEAN + _ANCHOR))
    os.chmod(clean, 0o700)
    yield tainted, clean
    os.unlink(tainted)
    os.unlink(clean)


def _leaks(findings: list[dict[str, object]]) -> list[dict[str, object]]:
    """The findings that are a leak.

    A run also reports where taint was INTRODUCED, which both images do
    identically and which says nothing about how taint moved.  Comparing the
    whole list would make the clean image look like it found something.
    """
    return [f for f in findings if f.get('kind') != 'taint_source']


def _run(guest: str) -> _Run:
    """One whole run under block mode.  -> (what the hook saw, the findings)."""
    from qiling import Qiling
    from qiling.const import QL_VERBOSE

    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper

    reporter = Reporter(json_mode=True, stream=io.StringIO())
    prev = os.environ.get('MICROTAINT_BLOCK')
    os.environ['MICROTAINT_BLOCK'] = '1'
    saved, dn = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    try:
        ql = Qiling([guest], '/', verbose=QL_VERBOSE.OFF)
        ql.os.stdin = io.BytesIO(b'\x01\x02\x03\x04\x05\x06\x07\x08')
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
    return dict(stats), [f.to_dict() for f in reporter.findings]


@pytest.fixture(scope='module')
def runs(guests: tuple[str, str]) -> dict[str, _Run]:
    """Each image alone, and each after the other one ran in this process."""
    tainted, clean = guests
    out: dict[str, _Run] = {}
    bc.cache_clear()
    out['tainted_alone'] = _run(tainted)
    bc.cache_clear()
    out['clean_alone'] = _run(clean)
    bc.cache_clear()
    _run(clean)
    out['tainted_after'] = _run(tainted)
    bc.cache_clear()
    _run(tainted)
    out['clean_after'] = _run(clean)
    return out


def test_the_two_images_are_the_same_program_at_the_same_addresses(
        runs: dict[str, _Run]) -> None:
    """The premise.  If the second image did not meet the first one's blocks,
    nothing below is about sharing at all."""
    for name in ('tainted_after', 'clean_after'):
        stats = runs[name][0]
        assert stats['reused'] > 0, (
            f'{name}: no plan at all was reused from the other image, so this '
            f'says nothing about what happens when one is')
        assert stats['no_code'] == 0, (
            f"{name}: {stats['no_code']} blocks' bytes could not be read, and a "
            f'block that cannot be read is never shared')
    assert runs['tainted_alone'][0]['blocks'] == runs['tainted_after'][0]['blocks']
    assert runs['clean_alone'][0]['blocks'] == runs['clean_after'][0]['blocks']


def test_the_two_images_disagree_when_each_runs_alone(
        runs: dict[str, _Run]) -> None:
    """And the oracle.  The tainted stub returns its argument and the branch
    below it is secret-dependent; the clean one returns zero and there is
    nothing to find.  Two images that agreed would make every comparison below
    true for the wrong reason."""
    assert _leaks(runs['tainted_alone'][1]), (
        'the tainted image reported no leak, so there is nothing for a stale '
        'plan to lose')
    assert _leaks(runs['clean_alone'][1]) == [], (
        f'the clean image reported {_leaks(runs["clean_alone"][1])}, so there '
        f'is nothing for a stale plan to invent')


def test_a_stale_plan_does_not_lose_the_leak(runs: dict[str, _Run]) -> None:
    """The under-taint direction, and the one that matters.

    The tainted image runs second, so its stub sits where the clean one's did.
    A plan kept by address alone would compute the clean stub's taint for it and
    the leak would simply be gone.
    """
    assert _leaks(runs['tainted_after'][1]) == _leaks(runs['tainted_alone'][1]), (
        f'after the other image ran, the tainted image reported '
        f'{_leaks(runs["tainted_after"][1])} against '
        f'{_leaks(runs["tainted_alone"][1])} alone')


def test_a_stale_plan_does_not_invent_a_leak(runs: dict[str, _Run]) -> None:
    """And the over-taint direction, for the same reason in reverse."""
    assert _leaks(runs['clean_after'][1]) == _leaks(runs['clean_alone'][1]), (
        f'after the other image ran, the clean image reported '
        f'{_leaks(runs["clean_after"][1])} against '
        f'{_leaks(runs["clean_alone"][1])} alone')


def test_the_unhandled_count_does_not_move(runs: dict[str, _Run]) -> None:
    """A block served the wrong plan need not report differently: it can simply
    decline, and a declined block is skipped rather than reported."""
    for alone, after in (('tainted_alone', 'tainted_after'),
                         ('clean_alone', 'clean_after')):
        assert runs[after][0]['unhandled'] == runs[alone][0]['unhandled'], (
            f'{runs[after][0]["unhandled"]} block executions were skipped in '
            f'{after}, against {runs[alone][0]["unhandled"]} in {alone}')
