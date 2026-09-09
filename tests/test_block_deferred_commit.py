# ruff: noqa: PLC0415, S110, S603, S607, ARG005
"""A block's taint must not take effect until the block has completed.

A block hook fires BEFORE the block executes, so its taint is speculative.  That
is not a theoretical worry: measured on a guest whose block loads through an
unmapped pointer, the hook announces a 65-byte block and six of its instructions
run before the fault.  Committing the whole block would have written taint for
instructions that never executed.

And that is an UNDER-taint, not a safe over-approximation.  `mov rax, rbx` with a
clean rbx clears RAX's taint, while RAX still holds its old tainted value because
the instruction never ran.  A detector reading RAX afterwards sees clean.

`BlockRunner` holds a block's taint pending and commits it when the NEXT block
arrives, because reaching a new block is the proof that the previous one
finished.  Abandonment then costs nothing: the pending state is simply dropped,
and there is nothing to undo.
"""
from __future__ import annotations

import os
import platform
import subprocess
import tempfile

import pytest

from microtaint.taint_ir.blockrun import BlockRunner, Pending

pytestmark = pytest.mark.skipif(
    platform.system() != 'Linux', reason='emulator tests require Linux',
)


def _runner(applied: list):
    return BlockRunner(
        compute=lambda a, s: Pending(a, {'RAX': a}),
        apply=lambda p: applied.append(p.address),
    )


def test_a_block_is_not_committed_until_the_next_one_arrives() -> None:
    applied: list[int] = []
    runner = _runner(applied)
    runner.on_block(0x100, 4)
    assert applied == [], 'the first block was committed before anything proved it ran'
    runner.on_block(0x200, 4)
    assert applied == [0x100], 'reaching a second block should have committed the first'
    runner.on_block(0x300, 4)
    assert applied == [0x100, 0x200]


def test_an_abandoned_block_leaves_nothing_behind() -> None:
    """The point of holding it: a block that does not finish applies nothing,
    so there is no state to roll back and no fault hook to get right."""
    applied: list[int] = []
    runner = _runner(applied)
    runner.on_block(0x100, 4)
    runner.on_block(0x200, 4)          # commits 0x100
    runner.abandon()                   # 0x200 faulted
    assert applied == [0x100], 'the abandoned block was applied anyway'
    assert runner.dropped == 1
    assert runner.pending_address is None


def test_the_last_block_is_committed_only_if_the_run_finished() -> None:
    for completed, expect in ((True, [0x100]), (False, [])):
        applied: list[int] = []
        runner = _runner(applied)
        runner.on_block(0x100, 4)
        runner.finish(completed=completed)
        assert applied == expect, (
            f'finish(completed={completed}) should have applied {expect}')


#: A block whose load faults part way, so several instructions after it never
#: run.  This is what makes the rule necessary rather than tidy.
_FAULTING = r"""
long sys_read(int fd, void *buf, unsigned long n){long r;
  __asm__ volatile("syscall":"=a"(r):"0"(0),"D"(fd),"S"(buf),"d"(n):"rcx","r11","memory");return r;}
void sys_exit(int c){__asm__ volatile("syscall"::"a"(60),"D"((long)c):"rcx","r11");__builtin_unreachable();}
unsigned long out;
void _start(void){
  unsigned char in[8]; sys_read(0, in, 8);
  unsigned long a = in[0], b = in[1];
  unsigned long *bad = (unsigned long *)0xdead0000UL;   /* unmapped */
  a += b;            /* runs   */
  a ^= 0x5555;       /* runs   */
  a += *bad;         /* FAULTS */
  a *= 3;            /* never  */
  a -= b;            /* never  */
  out = a;           /* never  */
  sys_exit(0);
}
"""


def test_a_real_block_really_is_abandoned_part_way() -> None:
    """The premise, measured rather than assumed: the hook is told about a block
    of which only some instructions run.

    If this ever stops being true -- if Unicorn started announcing only the part
    it will execute -- the deferred commit would be unnecessary, and it is worth
    finding that out from a failing test rather than by reasoning.
    """
    import io

    from qiling import Qiling
    from qiling.const import QL_VERBOSE
    from unicorn import UC_HOOK_BLOCK, UC_HOOK_CODE

    fd, binary = tempfile.mkstemp(suffix='.elf')
    os.close(fd)
    try:
        built = subprocess.run(
            ['gcc', '-nostdlib', '-static', '-no-pie', '-O1',
             '-fno-stack-protector', '-o', binary, '-x', 'c', '-'],
            input=_FAULTING.encode(), capture_output=True)
        if built.returncode != 0:
            pytest.skip(f'cannot build the guest: {built.stderr.decode()[:300]}')

        ql = Qiling([binary], '/', verbose=QL_VERBOSE.OFF)
        ql.os.stdin = io.BytesIO(b'\x11\x22\x33\x44\x55\x66\x77\x88')
        blocks: list[tuple[int, int]] = []
        instrs: list[int] = []
        ql.uc.hook_add(UC_HOOK_BLOCK, lambda uc, a, s, u: blocks.append((a, s)))
        ql.uc.hook_add(UC_HOOK_CODE, lambda uc, a, s, u: instrs.append(a))
        saved, devnull = os.dup(1), os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        try:
            ql.run()
        except Exception:      # the fault ends the run, which is the point
            pass
        os.dup2(saved, 1)

        assert blocks, 'no block hook fired at all'
        addr, size = blocks[-1]
        ran = [x for x in instrs if addr <= x < addr + size]
        assert len(ran) >= 1
        # The block hook was told about more code than actually executed.
        end = addr + size
        assert max(ran) + 1 < end, (
            f'the last block spans {addr:#x}..{end:#x} and execution reached '
            f'{max(ran):#x}; if the whole block really ran, the premise behind '
            f'deferring the commit no longer holds and it should be revisited')
    finally:
        os.unlink(binary)


def test_the_deferred_commit_would_have_prevented_the_under_taint() -> None:
    """The failure the rule exists to stop, in miniature.

    A block that would clear RAX's taint is computed, then abandoned.  Committing
    eagerly loses the taint RAX legitimately still has; holding it does not.
    """
    state = {'RAX': 0xFF}          # RAX is tainted before the block

    def apply(p: Pending) -> None:
        state.update(p.taint)

    runner = BlockRunner(compute=lambda a, s: Pending(a, {'RAX': 0}), apply=apply)
    runner.on_block(0x100, 16)     # the block would clear RAX
    runner.abandon()               # ... but it faults before that instruction
    assert state['RAX'] == 0xFF, (
        'RAX lost its taint to a block that never ran the instruction which '
        'would have cleared it: exactly the under-taint the deferral prevents')
