"""Taint stored into an mmap'd page must survive being read back.

A guest maps a page, copies its tainted input into it, hashes it back out.  The
taint-IR path tracks that.  The LogicCircuit path (MICROTAINT_TAINT_IR=0) does
not: the store into the mapped page never reaches the shadow, and the result
comes back completely clean.

What that has been narrowed to, by elimination:

  * not the ADDRESS being high -- the same store into a stack buffer, also
    above 0x7fff00000000, keeps its taint on both paths;
  * not the POINTER -- the same store through a register pointer into a static
    buffer keeps its taint on both paths;
  * not WHEN the region appears -- mapping the page before the input arrives
    fails identically;
  * not address truncation -- nothing is tainted at the 32- or 48-bit
    truncations of the address either;
  * not the mem-write clear hook -- the taint is absent with that hook
    disabled, so it is never written rather than written and wiped.

Every one of those was correct, and together they pointed away from the real
variable, which is not the memory at all: it is the ADDRESSING MODE.  gcc
compiled this loop with a biased pointer --

    sub   rdi, rax                     rdi = buf - base
    movzx ecx, BYTE PTR [rdi+rax*1]    rdi + rax == buf + i

-- and `resolve_ptr_with_offset` had nowhere to put a second register, so an
INT_ADD of two registers returned the left one and dropped the right.  The load
read RDI alone, which is buf-base, an address holding nothing.  A stack buffer
uses `[rsp+disp]` and a static buffer a constant address; both are single-register
forms, which is why they kept their taint.  Fixed by giving MemMapping an index
term; see tests/test_base_index_addressing.py for the minimal case.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Iterator

import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != 'Linux', reason='emulator tests require Linux',
)

_GUEST = r"""
static long sys6(long n, long a1, long a2, long a3, long a4, long a5, long a6) {
  long r;
  register long r10 __asm__("r10") = a4;
  register long r8  __asm__("r8")  = a5;
  register long r9  __asm__("r9")  = a6;
  __asm__ volatile("syscall" : "=a"(r)
      : "0"(n), "D"(a1), "S"(a2), "d"(a3), "r"(r10), "r"(r8), "r"(r9)
      : "rcx","r11","memory");
  return r;
}
static void sys_exit(int c) {
  __asm__ volatile("syscall" :: "a"(60), "D"((long)c) : "rcx","r11");
  __builtin_unreachable();
}
volatile unsigned long out;
void _start(void) {
  unsigned char buf[16];
  if (sys6(0, 0, (long)buf, 16, 0, 0, 0) <= 0) sys_exit(1);
  long p = sys6(9, 0, 65536, 3, 0x22, -1, 0);   /* mmap RW anonymous */
  if (p < 0) sys_exit(2);
  volatile unsigned char *q = (volatile unsigned char *)p;
  for (int i = 0; i < 16; i++) q[i] = buf[i];   /* tainted store */
  unsigned long a = 0;
  for (int i = 0; i < 16; i++) a = a * 31u + q[i];
  out = a;
  sys_exit(0);
}
"""


@pytest.fixture(scope='module')
def guest() -> Iterator[tuple[str, int]]:
    fd, path = tempfile.mkstemp(suffix='.elf')
    os.close(fd)
    built = subprocess.run(
        ['gcc', '-static', '-nostdlib', '-O1', '-fno-stack-protector',
         '-o', path, '-x', 'c', '-'],
        input=_GUEST.encode(), capture_output=True, check=False)
    if built.returncode != 0:
        os.unlink(path)
        pytest.skip(f'cannot build the guest: {built.stderr.decode()[:300]}')
    nm = subprocess.run(['nm', path], capture_output=True, text=True,
                        check=False).stdout
    addr = next((int(line.split()[0], 16) for line in nm.splitlines()
                 if line.split()[-1:] == ['out']), None)
    if addr is None:
        os.unlink(path)
        pytest.skip("cannot locate the guest's `out` symbol")
    yield path, addr
    os.unlink(path)


#: Run each measurement in its OWN process.  `MICROTAINT_TAINT_IR` is latched
#: the first time the engine consults it, so setting it between two calls in
#: one process changes nothing: an earlier version of this file did exactly
#: that, measured the taint-IR path twice, and reported the LogicCircuit path
#: as working.  Same family as every other "an engine default is not what you
#: asked for" trap in this repo.
_PROBE = r"""
import io, os, subprocess, sys
guest, addr = sys.argv[1], int(sys.argv[2])
from qiling import Qiling
from qiling.const import QL_VERBOSE
from microtaint.emulator.reporter import Reporter
from microtaint.emulator.wrapper import MicrotaintWrapper
ql = Qiling([guest], '/', verbose=QL_VERBOSE.OFF)
ql.os.stdin = io.BytesIO(bytes(range(1, 17)))
w = MicrotaintWrapper(ql, reporter=Reporter(json_mode=True, stream=io.StringIO()))
saved, dn = os.dup(1), os.open(os.devnull, os.O_WRONLY)
os.dup2(dn, 1)
ok = True
try:
    ql.run()
except Exception:
    ok = False
os.dup2(saved, 1)
if os.environ.get('MICROTAINT_BLOCK') == '1':
    w.block_mode_finish(ok)
sys.stderr.write('MASK=%d\n' % w.shadow_mem.read_mask(addr, 8))
"""


def _taint_of_out(guest: str, addr: int, taint_ir: bool, block: bool) -> int:
    env = dict(os.environ)
    env['MICROTAINT_BLOCK'] = '1' if block else '0'
    env['MICROTAINT_TAINT_IR'] = '1' if taint_ir else '0'
    done = subprocess.run([sys.executable, '-c', _PROBE, guest, str(addr)],
                          env=env, capture_output=True, check=False)
    for line in done.stderr.decode().splitlines():
        if line.startswith('MASK='):
            return int(line[5:])
    # `pytest.skip` does not return, so nothing follows it.
    pytest.skip(f'the probe did not report: {done.stderr.decode()[-300:]}')


def test_the_taint_ir_path_tracks_it(guest: tuple[str, int]) -> None:
    """The premise, and the oracle: the default path gets this right, so the
    guest really does carry taint through the mapped page."""
    path, addr = guest
    assert _taint_of_out(path, addr, taint_ir=True, block=False) != 0
    assert _taint_of_out(path, addr, taint_ir=True, block=True) != 0


def test_the_logic_circuit_path_tracks_it(guest: tuple[str, int]) -> None:
    """The property, on the other path."""
    path, addr = guest
    mask = _taint_of_out(path, addr, taint_ir=False, block=False)
    assert mask != 0, (
        f'the guest hashed its tainted input through a page it mapped itself '
        f'and the result is 0x{mask:016x}')
