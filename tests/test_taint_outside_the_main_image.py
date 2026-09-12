"""Taint must survive code that is not in the main binary.

The per-instruction hook is registered with a C-level address filter set to the
main image, so Unicorn never calls it for anything outside.  The comment where
that is set says it "eliminates the entire Qiling hook dispatch overhead for
libc/loader instructions".  What it actually eliminates is the TAINT: every
instruction outside the image runs unanalysed, so a value that flows through
one comes out clean.

Two ordinary programs hit it:

  * a dynamically linked binary, where `strlen` and `memcpy` live in libc.so.
    Measured before the fix: the hash of a tainted line came back 0x0 on the
    per-instruction path and fully tainted in block mode, same binary, same
    address.
  * anything that JITs, because an mmap'd code page is nowhere near the image.

It stayed invisible because every test, benchmark and campaign guest in this
repo is `-static`, and then libc IS the image.

Block mode has no such filter, so it is the oracle here as well as the second
subject: the two paths must agree.
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

_DYN = r"""
#include <unistd.h>
#include <string.h>
/* Dynamically linked, so `memcpy` is in libc.so -- outside the main image.
   Nothing here branches on the tainted bytes before `out` is stored: the
   per-instruction path halts at the first branch on tainted data BY DESIGN,
   and a guest that hits one never reaches the assertion.  An earlier version
   of this used fgets+strlen and the path stopped after 29 instructions, which
   looked exactly like the bug and was not.  The length is fixed, so memcpy
   branches only on alignment, which is untainted. */
volatile unsigned long out;
int main(void) {
    unsigned char buf[32];
    unsigned char copy[32];
    if (read(0, buf, 32) != 32) return 1;
    memcpy(copy, buf, 32);           /* libc.so, outside the image */
    unsigned long h = 5381;
    for (int i = 0; i < 32; i++) h = h * 33 + copy[i];
    out = h;
    return 0;
}
"""

#: mmap's a page, writes `mov rax,rdi ; add rax,rdi ; ret` into it and calls it
#: with the tainted value.  x86-64 only, which is why it is skipped elsewhere.
_JIT = r"""
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
  unsigned char buf[8];
  if (sys6(0, 0, (long)buf, 8, 0, 0, 0) <= 0) sys_exit(1);
  unsigned long v = 0;
  for (int i = 0; i < 8; i++) v |= ((unsigned long)buf[i]) << (8 * i);
  long p = sys6(9, 0, 4096, 7, 0x22, -1, 0);        /* mmap RWX */
  if (p < 0) sys_exit(2);
  unsigned char *c = (unsigned char *)p;
  c[0] = 0x48; c[1] = 0x89; c[2] = 0xf8;            /* mov rax, rdi */
  c[3] = 0x48; c[4] = 0x01; c[5] = 0xf8;            /* add rax, rdi */
  c[6] = 0xc3;                                      /* ret          */
  out = ((unsigned long (*)(unsigned long))(void *)c)(v);
  sys_exit(0);
}
"""


def _build(src: str, *flags: str) -> str:
    fd, path = tempfile.mkstemp(suffix='.elf')
    os.close(fd)
    built = subprocess.run(['gcc', *flags, '-O1', '-o', path, '-x', 'c', '-'],
                           input=src.encode(), capture_output=True, check=False)
    if built.returncode != 0:
        os.unlink(path)
        pytest.skip(f'cannot build the guest: {built.stderr.decode()[:300]}')
    return path


@pytest.fixture(scope='module')
def dynamic_guest() -> Iterator[str]:
    path = _build(_DYN)
    yield path
    os.unlink(path)


@pytest.fixture(scope='module')
def jit_guest() -> Iterator[str]:
    path = _build(_JIT, '-static', '-nostdlib', '-fno-stack-protector')
    yield path
    os.unlink(path)


def _taint_of_out(guest: str, block: bool, stdin: bytes) -> int:
    """The taint mask of the guest's `out` global after a run."""
    from qiling import Qiling
    from qiling.const import QL_VERBOSE

    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper

    prev = os.environ.get('MICROTAINT_BLOCK')
    os.environ['MICROTAINT_BLOCK'] = '1' if block else '0'
    saved, dn = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    try:
        ql = Qiling([guest], '/', verbose=QL_VERBOSE.OFF)
        ql.os.stdin = io.BytesIO(stdin)
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
        nm = subprocess.run(['nm', guest], capture_output=True, text=True,
                            check=False).stdout
        sym = next((int(line.split()[0], 16) for line in nm.splitlines()
                    if line.split()[-1:] == ['out']), None)
        if sym is None:
            pytest.skip("cannot locate the guest's `out` symbol")
        base = getattr(ql.loader, 'load_address', 0) or 0
        addr = base + sym if sym < 0x400000 else sym
        return w.shadow_mem.read_mask(addr, 8)
    finally:
        os.dup2(saved, 1)
        os.close(dn)
        os.close(saved)
        if prev is None:
            os.environ.pop('MICROTAINT_BLOCK', None)
        else:
            os.environ['MICROTAINT_BLOCK'] = prev


def test_taint_survives_libc_in_a_dynamically_linked_guest(
        dynamic_guest: str) -> None:
    """An ordinary `gcc` binary: the hash of its tainted input is tainted.

    `memcpy` is in libc.so, which is not the main image.

    BLOCK MODE ONLY, and the reason is worth writing down.  Every `read` is a
    taint source, including the ones the DYNAMIC LOADER makes to pull libc.so
    off disk -- so ld.so branches on data the engine considers tainted, long
    before `main`, and the per-instruction path stops there because stopping at
    the first implicit-taint site is what it does.  Measured: 29 instructions
    analysed, the finding inside ld-linux-x86-64.so.2, `out` never written.
    That is not this bug, and asserting it here would be asserting the wrong
    thing.  The JIT case below covers both paths.
    """
    mask = _taint_of_out(dynamic_guest, True, bytes(range(32)))
    assert mask == 0xFFFFFFFFFFFFFFFF, (
        f'the guest hashed a tainted line into `out` and its taint is '
        f'0x{mask:016x}: taint that flowed through libc.so was dropped '
        f'because libc is not inside the main image')


@pytest.mark.parametrize('block', [True, False], ids=['block', 'instruction'])
def test_taint_survives_code_the_guest_wrote_itself(
        jit_guest: str, block: bool) -> None:
    """A JIT'd page is nowhere near the image, and the value it returns is
    computed entirely from the input."""
    mask = _taint_of_out(jit_guest, block, bytes(range(1, 9)))
    assert mask != 0, (
        'the guest JITted `rax = rdi + rdi`, called it with its tainted input '
        'and stored the result: the result came back clean, so the code it '
        "wrote into an mmap'd page was never analysed")
