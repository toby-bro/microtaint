# ruff: noqa: S603, S607
"""A branch that becomes tainted only on a LATER visit must still be reported.

The hot path caches an instruction's taint answer per address: the whole input
taint array is snapshotted, the instruction is evaluated, and the resulting
array is stored as the output.  The input snapshot has to be taken BEFORE the
evaluation (it rewrites the array in place) and the output one after, so between
the two the entry holds a new input beside the previous output.  `have_snap` said
the pair was usable throughout that window.

That window is not hypothetical: an evaluation that commits nothing ends inside
it.  A conditional jump whose target depends on tainted data is exactly that
case -- the evaluator refuses to commit precisely so the implicit-taint policy
can report the leak from Python first -- and it returns without ever reaching
the store.  The entry is then left claiming that the tainted input produces the
output computed for the CLEAN one.  The caller re-probes the same entry on its
way to the reporting path, matches, replays the stale output, and returns: the
taint state is silently rolled back and the leak is never reported.

It takes two visits to arm and one to lose: a branch tainted on its first or
second visit was still reported, which is why this only ever showed up as
"the third key bit onwards is missing" rather than as a dead detector.

The guest is one bit-serial loop, the textbook square-and-multiply shape from
`crypto/test_constant_time.c`: iteration i branches on bit i of a secret byte.
Tainting exactly one bit makes the branch clean on every visit but one, so the
visit that must fire is chosen by the test.
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

#: Iteration i branches on bit i of the secret; 8 iterations, no libc.
_SRC = r"""
long sys_read(int fd, void *buf, unsigned long count){long r;
  __asm__ volatile("syscall":"=a"(r):"0"(0),"D"(fd),"S"(buf),"d"(count):"rcx","r11","memory");return r;}
long sys_exit(int status){long r;
  __asm__ volatile("syscall":"=a"(r):"0"(60),"D"(status):"rcx","r11","memory");return r;}
void _start(){
  unsigned char buf[1];
  sys_read(0, buf, 1);
  unsigned int e = buf[0];
  unsigned int r = 1;
  for (int i = 0; i < 8; i++) {
      if (e & 1) { r += 3; }      /* the key-dependent branch */
      e >>= 1;
  }
  sys_exit((int)(r & 1));
}
"""


@pytest.fixture(scope='module')
def binary() -> str:
    fd, path = tempfile.mkstemp(suffix='.elf')
    os.close(fd)
    try:
        subprocess.run(
            ['gcc', '-nostdlib', '-O0', '-fno-stack-protector', '-o', path, '-x', 'c', '-'],
            input=_SRC.encode(), check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f'cannot build the guest: {exc}')
    yield path
    os.unlink(path)


def _side_channel_reports(binary: str, secret_bit: int) -> list[int]:
    """Run with ONLY `secret_bit` of the input byte tainted -> reported addresses."""
    from qiling import Qiling
    from qiling.const import QL_VERBOSE
    from qiling.extensions import pipe

    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper

    ql = Qiling([binary], '/', verbose=QL_VERBOSE.OFF)
    ql.os.stdin = pipe.SimpleInStream(0)
    ql.os.stdin.write(b'\xff')
    wrapper = MicrotaintWrapper(
        ql, check_bof=False, check_uaf=False, check_sc=True, check_aiw=False,
        reporter=Reporter(json_mode=False, stream=StringIO()))

    # Taint one bit of the byte the guest reads, not the whole byte: the branch
    # is then tainted on exactly one visit, and which one is `secret_bit`.
    def one_bit(address: int, size: int) -> None:
        for i in range(size):
            wrapper.taint_region(address + i, bytes([1 << secret_bit]) if i == 0 else b'\x00')

    wrapper._taint_bytes = one_bit

    seen: list[int] = []
    report = wrapper.reporter.side_channel

    def record(address: int, instruction: str = '', taint_mask: int = 0) -> None:
        seen.append(address)
        report(address, instruction=instruction, taint_mask=taint_mask)

    wrapper.reporter.side_channel = record  # type: ignore[method-assign]
    try:
        ql.run()
    except Exception:
        pass
    return seen


# 0 and 1 are the visits that were reported even with the bug; 2 upwards are the
# ones it swallowed.  Both are here so a fix that trades one for the other fails.
@pytest.mark.parametrize('secret_bit', [0, 1, 2, 3, 5, 7])
def test_tainted_branch_is_reported_on_any_visit(binary: str, secret_bit: int) -> None:
    reports = _side_channel_reports(binary, secret_bit)
    assert reports, (
        f'bit {secret_bit} of the secret decides the branch on visit {secret_bit}, '
        f'and nothing was reported; the earlier visits are clean, so this is the '
        f'cached answer for a CLEAN branch being replayed for a tainted one')
    assert len(set(reports)) == 1, (
        f'expected the one key-dependent branch, got {[hex(a) for a in set(reports)]}')
