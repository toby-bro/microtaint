"""Block mode must drop a cached plan when the guest rewrites the code under it.

A block plan is keyed by (address, size) alone, so a guest that patches an
instruction in place -- same length, same block -- keeps the key it had.  Until
this was wired, block mode kept running the plan compiled for the bytes that
used to be there: measured on the guest below, the taint the patched
instruction propagates was lost entirely, which is an under-taint.

The guard already existed for the instruction path.  It is armed by
`[code_lo, code_hi)`, the range some cache holds a decode of, and the
mem-write hook tests it on both the nogil and the GIL path.  Block mode plans
blocks the instruction hook never decodes, so two things had to be added: the
block hook widens that same range as it plans, and `invalidate_smc` drops the
block plan cache alongside the instruction caches.

The guest needs a writable code segment, which `-Wl,--omagic` gives by putting
everything in one RWX LOAD segment.  That is not a normal layout, so nothing
here compares the two engines' whole answers on it -- what is asserted is the
one byte the patched instruction is responsible for.
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

#: `patchme` starts as `xor %r12,%r12`, which CLEARS, and is rewritten in place
#: to `mov %rbx,%r12`, which COPIES the tainted value.  Both are three bytes, so
#: the block keeps its address and its size and therefore its cache key -- which
#: is the whole point: a plan cache that keyed on the bytes would not need this.
_GUEST = r"""
long sys_read(int fd, void *buf, unsigned long n){long r;
  __asm__ volatile("syscall":"=a"(r):"0"(0),"D"(fd),"S"(buf),"d"(n):"rcx","r11","memory");return r;}
void sys_exit(int c){__asm__ volatile("syscall"::"a"(60),"D"((long)c):"rcx","r11");__builtin_unreachable();}

__attribute__((noinline)) unsigned long step(unsigned long x){
  register unsigned long rbx __asm__("rbx") = x;
  register unsigned long r12 __asm__("r12");
  __asm__ volatile("patchme: .byte 0x4d,0x31,0xe4" : "=r"(r12) : "r"(rbx));
  return r12;
}
extern unsigned char patchme[];
volatile unsigned long out1, out2;
void _start(void){
  unsigned char in[8];
  if (sys_read(0, in, 8) <= 0) sys_exit(1);
  out1 = step(in[0]);                                        /* xor: clean */
  patchme[0] = 0x49; patchme[1] = 0x89; patchme[2] = 0xdc;   /* mov %rbx,%r12 */
  out2 = step(in[0]);                                        /* mov: TAINTED */
  sys_exit(0);
}
"""


def _build() -> str:
    fd, path = tempfile.mkstemp(suffix='.elf')
    os.close(fd)
    built = subprocess.run(
        ['gcc', '-nostdlib', '-static', '-no-pie', '-O1', '-fno-stack-protector',
         '-Wl,--omagic', '-o', path, '-x', 'c', '-'],
        input=_GUEST.encode(), capture_output=True, check=False)
    if built.returncode != 0:
        os.unlink(path)
        pytest.skip(f'cannot build the guest: {built.stderr.decode()[:300]}')
    return path


def _symbol(elf: str, name: str) -> int:
    """A symbol's address, from nm.  The globals are what is asserted on, so
    the test has to name them rather than guess where the linker put them."""
    out = subprocess.run(['nm', elf], capture_output=True, text=True, check=False)
    if out.returncode != 0:
        pytest.skip('nm is unavailable, so the globals cannot be located')
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == name:
            return int(parts[0], 16)
    pytest.fail(f'{name} is not in the guest: the C above no longer defines it')


class Result:
    """One run: the shadow mask at each global, and the block counters."""

    def __init__(self, out1: int, out2: int, stats: dict[str, int] | None) -> None:
        self.out1 = out1
        self.out2 = out2
        self.stats = stats


def _run(guest: str) -> Result:
    from qiling import Qiling
    from qiling.const import QL_VERBOSE

    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper

    a1, a2 = _symbol(guest, 'out1'), _symbol(guest, 'out2')
    prev = os.environ.get('MICROTAINT_BLOCK')
    os.environ['MICROTAINT_BLOCK'] = '1'
    saved, dn = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    try:
        ql = Qiling([guest], '/', verbose=QL_VERBOSE.OFF)
        ql.os.stdin = io.BytesIO(bytes(range(1, 9)))
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
        return Result(w.shadow_mem.read_mask(a1, 8),
                      w.shadow_mem.read_mask(a2, 8),
                      w.block_mode_stats())
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
    path = _build()
    yield path
    os.unlink(path)


@pytest.fixture(scope='module')
def result(guest: str) -> Result:
    return _run(guest)


def test_block_mode_installed_and_handled_the_blocks(result: Result) -> None:
    """The premise.  A run where block mode never installed says nothing about
    a block plan cache, and an unhandled block is skipped rather than replanned,
    which would hide the same defect for a different reason."""
    stats = result.stats
    assert stats is not None, 'block mode did not install'
    assert stats['blocks'] > 0, 'the block hook never fired'
    assert stats['unhandled'] == 0, f'blocks were skipped: {stats}'


def test_the_write_into_the_code_invalidated_the_plan_cache(
        result: Result) -> None:
    """The mechanism: the guard fired and the cache was dropped.

    It is armed by the block hook widening `[code_lo, code_hi)` as it plans, so
    a zero here means either that the range was never widened or that
    `invalidate_smc` no longer reaches the block cache.
    """
    stats = result.stats
    assert stats is not None
    assert stats['invalidations'] >= 1, (
        f'the guest rewrote three bytes of its own code and no plan cache was '
        f'dropped: {stats}')


def test_the_patched_instruction_propagates_its_taint(result: Result) -> None:
    """The answer.  `out2` is written from the register the PATCHED instruction
    copies the tainted value into, so a stale plan -- which still holds the
    `xor` that clears it -- leaves it clean."""
    assert result.out2, (
        'out2 is clean: the block ran the plan compiled for the instruction '
        'that used to be there, which cleared the taint instead of copying it')


def test_the_unpatched_instruction_did_not_propagate_it(result: Result) -> None:
    """The other half, which is what makes the test above mean something.

    Before the patch the same block runs `xor %r12,%r12`, so `out1` must be
    CLEAN.  If it were tainted, `out2` being tainted would prove nothing about
    the cache: both would just be tainted all along.
    """
    assert not result.out1, (
        f'out1 is tainted ({result.out1:#x}), but the instruction that '
        f'produced it clears its output, so the comparison above is vacuous')
