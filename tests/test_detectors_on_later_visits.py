# ruff: noqa: S603, S607, PLC0415, S110
"""Every detector must still fire on a site it has already seen behave.

The engine caches per address, and the implicit-taint detector was found
reporting a key-dependent branch only on the visits that missed that cache (see
tests/test_branch_report_after_cache.py).  The failure was not specific to
branches: what the cache replays is the whole taint state, so any check that
reads taint AFTER an instruction can be starved the same way.

So each detector gets the same shape: a single site executed several times,
harmless at first and a real finding later.  A detector that only ever answers
for a cold address passes the ordinary tests, where the very first execution is
the bad one, and fails these.
"""
from __future__ import annotations

import platform
import subprocess
import tempfile
from io import StringIO

import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != 'Linux', reason='emulator tests require Linux',
)


def _compile_freestanding(src: str) -> str:
    fd, path = tempfile.mkstemp(suffix='.elf')
    import os
    os.close(fd)
    subprocess.run(['gcc', '-nostdlib', '-O0', '-fno-stack-protector',
                    '-o', path, '-x', 'c', '-'],
                   input=src.encode(), check=True, capture_output=True)
    return path


_SYSCALLS = r"""
long sys_read(int fd, void *buf, unsigned long count){long r;
  __asm__ volatile("syscall":"=a"(r):"0"(0),"D"(fd),"S"(buf),"d"(count):"rcx","r11","memory");return r;}
long sys_exit(int status){long r;
  __asm__ volatile("syscall":"=a"(r):"0"(60),"D"(status):"rcx","r11","memory");return r;}
"""

#: One read site, reached three times.  The first two calls fit the buffer; the
#: third overruns it and lands on the saved return address.  A loop would have
#: been the obvious way to repeat the site, but the overflow clobbers the loop
#: counter on the stack, and the tainted comparison that follows ends the run
#: before the return -- so the repetition is unrolled into three calls instead.
_BOF_SRC = _SYSCALLS + r"""
static void fill(unsigned long n){ char buf[16]; sys_read(0, buf, n); }
void _start(void){ fill(4); fill(4); fill(64); sys_exit(0); }
"""


def _run_freestanding(binary: str, stdin_data: bytes, **checks: bool) -> list[str]:
    from qiling import Qiling
    from qiling.const import QL_VERBOSE
    from qiling.extensions import pipe

    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper
    from microtaint.sleigh.engine import _cached_generate_static_rule

    _cached_generate_static_rule.cache_clear()
    ql = Qiling([binary], '/', verbose=QL_VERBOSE.OFF)
    ql.os.stdin = pipe.SimpleInStream(0)
    ql.os.stdin.write(stdin_data)
    stream = StringIO()
    reporter = Reporter(json_mode=False, stream=stream)
    MicrotaintWrapper(ql, check_bof=checks.get('bof', False),
                      check_uaf=checks.get('uaf', False),
                      check_sc=checks.get('sc', False),
                      check_aiw=checks.get('aiw', False), reporter=reporter)
    try:
        ql.run()
    except Exception:
        pass
    reporter.finalize()
    return stream.getvalue().splitlines()


def test_overflow_on_a_read_site_already_seen_behaving() -> None:
    """The overflowing read is the THIRD execution of that call site."""
    binary = _compile_freestanding(_BOF_SRC)
    logs = _run_freestanding(binary, b'A' * 64, bof=True)
    assert any('[BOF]' in line or 'buffer_overflow' in line for line in logs), (
        f'the third pass writes 64 bytes into a 16-byte buffer and was not '
        f'reported; the first two passes were benign: {logs}')


#: One read site, reached four times against the same buffer.  The first two
#: reads happen while it is live; the harness poisons it (what HeapTracker does
#: on free) and the last two are then use-after-free.  `touch` is not inlined at
#: -O0, so all four reads are the same instruction at the same address.
#:
#: The buffer is a static, and the poisoning is driven from the harness rather
#: than by malloc/free, because HeapTracker resolves libc symbols and does not
#: get them everywhere (tests/test_heap.py skips on the same condition).  What
#: is under test here is the read side -- whether a read site that has already
#: been seen reading LIVE memory is still checked once the memory is dead --
#: and that does not depend on where the poison came from.
_UAF_SRC = _SYSCALLS + r"""
long sys_write(int fd, const void *buf, unsigned long count){long r;
  __asm__ volatile("syscall":"=a"(r):"0"(1),"D"(fd),"S"(buf),"d"(count):"rcx","r11","memory");return r;}
char blob[64];
static char touch(void){ return blob[0]; }
void _start(void){
  volatile char a = touch();
  volatile char b = touch();       /* the site is warm from here on */
  sys_write(1, "x", 1);            /* marker: the harness poisons blob here */
  volatile char c = touch();       /* use after free */
  volatile char d = touch();
  (void)a; (void)b; (void)c; (void)d;
  sys_exit(0);
}
"""


def _symbol_address(binary: str, name: str) -> int | None:
    """Absolute address of a symbol in a static, non-PIE binary."""
    try:
        out = subprocess.run(['nm', binary], capture_output=True, check=True).stdout.decode()
    except (OSError, subprocess.CalledProcessError):
        return None
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == name:
            return int(parts[0], 16)
    return None


def test_use_after_free_read_at_a_site_already_seen_live() -> None:
    """The offending reads have already run twice against LIVE memory."""
    import os

    from qiling import Qiling
    from qiling.const import QL_VERBOSE
    from qiling.extensions import pipe

    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper

    fd, binary = tempfile.mkstemp(suffix='.elf')
    os.close(fd)
    try:
        subprocess.run(['gcc', '-nostdlib', '-static', '-no-pie', '-O0',
                        '-fno-stack-protector', '-o', binary, '-x', 'c', '-'],
                       input=_UAF_SRC.encode(), check=True, capture_output=True)
        blob = _symbol_address(binary, 'blob')
        if blob is None:
            pytest.skip('cannot locate the guest buffer (nm unavailable?)')

        ql = Qiling([binary], '/', verbose=QL_VERBOSE.OFF)
        ql.os.stdin = pipe.SimpleInStream(0)
        stream = StringIO()
        reporter = Reporter(json_mode=True, stream=stream)
        wrapper = MicrotaintWrapper(ql, check_bof=False, check_uaf=True, check_sc=False,
                                    check_aiw=False, reporter=reporter)

        # The marker write is the "free": poison the buffer exactly once, after
        # the site has already been visited twice against live memory.
        poisoned: list[int] = []

        def on_write(ql_, fd_, buf, count, *rest):
            if not poisoned:
                wrapper.shadow_mem.poison(blob, 64)
                poisoned.append(1)

        ql.os.set_syscall('write', on_write)
        try:
            ql.run()
        except Exception:
            pass

        assert poisoned, 'the marker syscall never ran; the guest did not reach the free point'
        uaf = [f for f in getattr(reporter, 'findings', [])
               if str(f.kind).endswith('use_after_free')]
        assert uaf, (
            'the last two reads come from poisoned memory at a site that had '
            'already read it while live, and nothing was reported')
    finally:
        os.unlink(binary)
