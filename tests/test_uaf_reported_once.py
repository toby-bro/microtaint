"""One access to freed memory is one finding.

In UAF mode three memory hooks are registered, and for a write to a page that
`munmap` has removed Unicorn fires BOTH the write hook and the unmapped-write
hook.  Each checks the shadow, each finds the address poisoned, and each
reports: the artifact's own guest, whose whole body is

    p = mmap(...); munmap(p, 4096); p[0] = 'A';

printed two identical `[UAF]` lines for that single store, against a README that
says every guest prints exactly one finding of its kind.

Reporting the same freed address twice also says nothing new when the access sits
in a loop, so the address is the identity: `Reporter.uaf` reports each freed
address once, whichever hook reaches it first.  The existing detector tests
assert only that SOMETHING was reported, which is why two went unnoticed.
"""
from __future__ import annotations

import os
import platform
import subprocess
import tempfile
from io import StringIO

import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != 'Linux', reason='emulator tests require Linux',
)

#: mmap a page, munmap it, then write through the stale pointer: one access.
_UAF_SRC = r"""
static long sys_mmap(void *addr, unsigned long len, int prot, int flags, int fd, long off) {
    long ret;
    register long r10 __asm__("r10") = flags;
    register long r8  __asm__("r8")  = fd;
    register long r9  __asm__("r9")  = off;
    __asm__ volatile("syscall" : "=a"(ret)
                     : "0"(9), "D"(addr), "S"(len), "d"(prot), "r"(r10), "r"(r8), "r"(r9)
                     : "rcx", "r11", "memory");
    return ret;
}
static long sys_munmap(void *addr, unsigned long len) {
    long ret;
    __asm__ volatile("syscall" : "=a"(ret) : "0"(11), "D"(addr), "S"(len)
                     : "rcx", "r11", "memory");
    return ret;
}
static long sys_exit(int c) {
    long ret;
    __asm__ volatile("syscall" : "=a"(ret) : "0"(60), "D"(c) : "rcx", "r11", "memory");
    return ret;
}
void _start(void) {
    char *p = (char *)sys_mmap(0, 4096, 3, 34, -1, 0);
    sys_munmap(p, 4096);
    p[0] = 'A';
    sys_exit(0);
}
"""


def test_one_access_to_freed_memory_is_one_finding() -> None:
    """The guest frees once and writes once, so the report has one finding."""
    from qiling import Qiling
    from qiling.const import QL_VERBOSE

    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper

    fd, binary = tempfile.mkstemp(suffix='.elf')
    os.close(fd)
    try:
        subprocess.run(['gcc', '-nostdlib', '-static', '-no-pie', '-O0',
                        '-fno-stack-protector', '-o', binary, '-x', 'c', '-'],
                       input=_UAF_SRC.encode(), check=True, capture_output=True)
        ql = Qiling([binary], '/', verbose=QL_VERBOSE.OFF)
        reporter = Reporter(json_mode=True, stream=StringIO())
        MicrotaintWrapper(ql, check_bof=False, check_uaf=True, check_sc=False,
                          check_aiw=False, reporter=reporter)
        try:
            ql.run()
        except Exception:
            pass

        uaf = [f for f in reporter.findings if str(f.kind).endswith('use_after_free')]
        assert uaf, 'the write through the stale pointer was not reported at all'
        assert len(uaf) == 1, (
            f'one store to freed memory produced {len(uaf)} findings at '
            f'{[hex(f.address) for f in uaf]}: a write to an unmapped page makes '
            f'Unicorn fire both the write hook and the unmapped-write hook, and '
            f'both report'
        )
    finally:
        os.unlink(binary)


# ---------------------------------------------------------------------------
# The identity, without an emulator.
# ---------------------------------------------------------------------------

def test_the_same_freed_address_reports_once() -> None:
    from microtaint.emulator.reporter import Reporter

    r = Reporter(json_mode=True, stream=StringIO())
    r.uaf(0x4000, 1)
    r.uaf(0x4000, 1)
    r.uaf(0x4000, 8)
    assert len(r.findings) == 1, 'the same freed address reported more than once'


def test_distinct_freed_addresses_each_report() -> None:
    """Dedup must not collapse two different bugs into one."""
    from microtaint.emulator.reporter import Reporter

    r = Reporter(json_mode=True, stream=StringIO())
    r.uaf(0x4000, 1)
    r.uaf(0x8000, 1)
    assert [f.address for f in r.findings] == [0x4000, 0x8000]
