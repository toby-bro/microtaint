"""Taint must reach code the guest ran BEFORE its input arrived.

A Unicorn hook is compiled into a translated block, and microtaint arms its
hooks lazily, on the first taint injection -- which for a real program is the
`read` that brings the input in.  Everything the program did before that is
already translated, and those blocks keep running without the hook.  So any
function a program warms up before reading its input is never analysed again,
however tainted the values flowing through it become.

This is not a corner case.  Every program initialises, then reads, then works;
a static-glibc guest re-enters `strlen`, `memcpy` and `snprintf` after the read
having already run them during startup.  Measured there before the fix: of the
785 block executions after the taint arrived, 150 were never seen, and 17 of 54
secret-dependent branches were never reported.  Nothing raised.  The run
completed.  It simply found less.

The guest below is the smallest thing that shows it: call a function, read the
input, call THE SAME function on it.  With the translations from before arming
still in place, the result comes back completely clean.
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

#: Raw syscalls and no libc, so the only tainted value in the program is what
#: `read` brings in and the only interesting code is `mix`.
_GUEST = r"""
static long sys_read(int fd, void *buf, unsigned long n) {
  long r; __asm__ volatile("syscall" : "=a"(r)
      : "0"(0), "D"((long)fd), "S"(buf), "d"(n) : "rcx","r11","memory"); return r;
}
static void sys_exit(int c) {
  __asm__ volatile("syscall" :: "a"(60), "D"((long)c) : "rcx","r11");
  __builtin_unreachable();
}
/* volatile, or gcc drops the stores as dead in a _start that never returns. */
volatile unsigned long out;

__attribute__((noinline)) unsigned long mix(unsigned long x) {
  unsigned long a = x;
  for (int i = 0; i < 8; i++) a = a * 31u + (a >> 3);
  return a;
}

void _start(void) {
  unsigned long v = 0;
  unsigned long warm = 0;
  /* Runs `mix` while nothing is tainted: this is what gets translated before
     the hooks exist, and it is the same code the tainted call below uses. */
  for (int i = 0; i < 4; i++) warm += mix((unsigned long)i);
  out = warm;
  sys_read(0, &v, 8);              /* taint arrives HERE, and arms the hooks */
  out = mix(v);                    /* the SAME blocks, now carrying taint */
  sys_exit(0);
}
"""

_INPUT = bytes(range(8))


def _mix(x: int) -> int:
    a = x & 0xFFFFFFFFFFFFFFFF
    for _ in range(8):
        a = (a * 31 + (a >> 3)) & 0xFFFFFFFFFFFFFFFF
    return a


@pytest.fixture(scope='module')
def guest() -> Iterator[tuple[str, int]]:
    """(the binary, the address of its `out` global)."""
    fd, path = tempfile.mkstemp(suffix='.elf')
    os.close(fd)
    built = subprocess.run(
        ['gcc', '-static', '-nostdlib', '-O1', '-fno-stack-protector',
         '-o', path, '-x', 'c', '-'],
        input=_GUEST.encode(), capture_output=True, check=False)
    if built.returncode != 0:
        os.unlink(path)
        pytest.skip(f'cannot build the guest: {built.stderr.decode()[:300]}')
    nm = subprocess.run(['nm', path], capture_output=True, text=True, check=False)
    addr = next((int(line.split()[0], 16) for line in nm.stdout.splitlines()
                 if line.split()[-1:] == ['out']), None)
    if addr is None:
        os.unlink(path)
        pytest.skip("cannot locate the guest's `out` symbol")
    yield path, addr
    os.unlink(path)


def _run(guest: str, addr: int, block: bool) -> tuple[int, int]:
    """-> (the value the guest computed, the taint mask of that value)."""
    from qiling import Qiling
    from qiling.const import QL_VERBOSE

    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper

    prev = os.environ.get('MICROTAINT_BLOCK')
    os.environ['MICROTAINT_BLOCK'] = '1' if block else '0'
    saved, dn = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    try:
        ql = Qiling([guest], '/', verbose=QL_VERBOSE.OFF)
        ql.os.stdin = io.BytesIO(_INPUT)
        w = MicrotaintWrapper(
            ql, reporter=Reporter(json_mode=True, stream=io.StringIO()))
        os.dup2(dn, 1)
        ok = True
        try:
            ql.run()
        except Exception:
            ok = False
        os.dup2(saved, 1)
        if block:
            w.block_mode_finish(ok)
        value = int.from_bytes(bytes(ql.mem.read(addr, 8)), 'little')
        return value, w.shadow_mem.read_mask(addr, 8)
    finally:
        os.dup2(saved, 1)
        os.close(dn)
        os.close(saved)
        if prev is None:
            os.environ.pop('MICROTAINT_BLOCK', None)
        else:
            os.environ['MICROTAINT_BLOCK'] = prev


@pytest.mark.parametrize('block', [False, True], ids=['instruction', 'block'])
def test_the_guest_ran_both_calls(guest: tuple[str, int], block: bool) -> None:
    """The premise.  If the second call never happened, or the first one never
    warmed anything, the taint assertion below would be about nothing.

    `out` holds the warm-up sum until the tainted call overwrites it, so a
    result equal to `mix(input)` proves both calls ran, in that order.
    """
    path, addr = guest
    value, _mask = _run(path, addr, block)
    warm = sum(_mix(i) for i in range(4)) & 0xFFFFFFFFFFFFFFFF
    assert value == _mix(int.from_bytes(_INPUT, 'little')), (
        f'the guest computed 0x{value:016x}, not mix(input); it did not run '
        f'the tainted call')
    assert value != warm, (
        'the guest still holds the warm-up sum, so the tainted call never '
        'overwrote it')


@pytest.mark.parametrize('block', [False, True], ids=['instruction', 'block'])
def test_taint_reaches_code_translated_before_the_hooks_were_armed(
        guest: tuple[str, int], block: bool) -> None:
    """The property.  Every bit of the result is tainted, because every bit of
    it came from the input.

    Before `_flush_translations`, this came back 0x0 on BOTH paths: the blocks
    of `mix` were translated during the warm-up, the hooks were armed
    afterwards by the `read`, and Unicorn kept running the translations that
    had no hook in them.
    """
    path, addr = guest
    _value, mask = _run(path, addr, block)
    assert mask == 0xFFFFFFFFFFFFFFFF, (
        f'the guest hashed its whole 8-byte input into `out` and the result is '
        f'masked 0x{mask:016x}: the taint of code that was already translated '
        f'when the hooks were armed was lost')
