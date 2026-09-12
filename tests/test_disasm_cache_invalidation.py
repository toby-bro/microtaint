"""The remembered disassembly must not survive the guest rewriting the code.

Reporting a secret-dependent branch reads 16 bytes of guest memory and runs
capstone, and a fuzz iteration reports the same branches every time: 110
findings at about 54 sites on a static-glibc guest, and the same sites for the
next input.  Measured, that was 6.06 M instructions of an 18.08 M restored
iteration -- a third of it -- spent re-answering a question whose answer had
not changed.

So the answer is remembered, keyed by address, which is only safe while the
bytes at that address are the same.  The one thing that changes them is the
guest writing onto its own code, and `InstructionHook.smc_invalidations`
counts exactly that; a restore puts back the bytes the checkpoint had, and if
the guest rewrote code in between, the counter says so.

The guest here rewrites a four-byte function from `mov rax, rdi ; ret` to
`xor rax, rax ; ret`, which is the same shape the block runtime's own
self-modifying-code tests use.
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
__asm__(".globl mystub\n.align 16\nmystub:\n .byte 0x48,0x89,0xf8,0xc3\n");
extern unsigned long mystub(unsigned long);

void _start(void) {
  unsigned char buf[8];
  if (sys6(0, 0, (long)buf, 8, 0, 0, 0) <= 0) sys_exit(1);
  unsigned long v = 0;
  for (int i = 0; i < 8; i++) v |= ((unsigned long)buf[i]) << (8 * i);
  out = mystub(v);
  unsigned char *p = (unsigned char *)(void *)mystub;
  p[0] = 0x48; p[1] = 0x31; p[2] = 0xc0; p[3] = 0xc3;   /* xor rax,rax ; ret */
  out = mystub(v);
  sys_exit(0);
}
"""


@pytest.fixture(scope='module')
def guest() -> Iterator[tuple[str, int]]:
    fd, path = tempfile.mkstemp(suffix='.elf')
    os.close(fd)
    built = subprocess.run(
        ['gcc', '-static', '-nostdlib', '-O1', '-fno-stack-protector',
         '-Wl,-N', '-o', path, '-x', 'c', '-'],
        input=_GUEST.encode(), capture_output=True, check=False)
    if built.returncode != 0:
        os.unlink(path)
        pytest.skip(f'cannot build the guest: {built.stderr.decode()[:300]}')
    nm = subprocess.run(['nm', path], capture_output=True, text=True,
                        check=False).stdout
    addr = next((int(line.split()[0], 16) for line in nm.splitlines()
                 if line.split()[-1:] == ['mystub']), None)
    if addr is None:
        os.unlink(path)
        pytest.skip('cannot locate `mystub`')
    yield path, addr
    os.unlink(path)


def test_a_rewritten_instruction_is_disassembled_again(
        guest: tuple[str, int]) -> None:
    """Ask before the run, let the guest rewrite it, ask again."""
    from qiling import Qiling
    from qiling.const import QL_VERBOSE

    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper

    path, stub = guest
    prev = os.environ.get('MICROTAINT_BLOCK')
    os.environ['MICROTAINT_BLOCK'] = '1'
    saved, dn = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    try:
        ql = Qiling([path], '/', verbose=QL_VERBOSE.OFF)
        ql.os.stdin = io.BytesIO(bytes(range(1, 9)))
        w = MicrotaintWrapper(
            ql, reporter=Reporter(json_mode=True, stream=io.StringIO()))
        os.dup2(dn, 1)
        before = w._disasm_at(stub)
        ok = True
        try:
            ql.run()
        except Exception:
            ok = False
        os.dup2(saved, 1)
        w.block_mode_finish(ok)
        after = w._disasm_at(stub)
        final = bytes(ql.mem.read(stub, 4))
    finally:
        os.dup2(saved, 1)
        os.close(dn)
        os.close(saved)
        if prev is None:
            os.environ.pop('MICROTAINT_BLOCK', None)
        else:
            os.environ['MICROTAINT_BLOCK'] = prev

    assert final == bytes((0x48, 0x31, 0xC0, 0xC3)), (
        f'the guest did not rewrite its stub ({final.hex()}), so this test '
        f'compares the same instruction with itself')
    assert before[0] == 'mov', (
        f'the stub did not start as `mov rax, rdi`: {before}')
    assert after[0] == 'xor', (
        f'the stub was rewritten to `xor rax, rax` and is still reported as '
        f'{after[1]!r}: the remembered disassembly outlived the code it '
        f'described')
