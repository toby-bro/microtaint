"""A block's taint must not take effect until the block has completed.

A block hook fires BEFORE the block executes, so its taint is speculative.  That
is not a theoretical worry: measured on a guest whose block loads through an
unmapped pointer, the hook announces a 65-byte block and six of its instructions
run before the fault.  Committing the whole block would have written taint for
instructions that never executed.

And that is an UNDER-taint, not a safe over-approximation.  `mov rax, rbx` with a
clean rbx clears RAX's taint, while RAX still holds its old tainted value because
the instruction never ran.  A detector reading RAX afterwards sees clean.

The rule lives in `blockpath.h`: a block's taint is held pending and committed
when the NEXT block arrives, because reaching a new block is the proof that the
previous one finished.  Abandonment then costs nothing -- the pending state is
dropped and there is nothing to undo -- and it needs no fault hook.

These drive the C runtime through `blockpath_c`, because that is where the rule
is.  A Python re-statement of it would be a second implementation to keep in
step with the one that actually runs.
"""
from __future__ import annotations

import os
import platform
import subprocess
import tempfile
from typing import TYPE_CHECKING

import pytest

from microtaint.emulator import blockpath_c as B
from microtaint.taint_ir import frompcode
from microtaint.taint_ir.blockcompile import compile_block
from microtaint.types import Architecture

if TYPE_CHECKING:    # stub-only: opaque PyCapsule handles into blockpath.h
    from microtaint.emulator.blockpath_c import _Capsule as Handle

pytestmark = pytest.mark.skipif(
    platform.system() != 'Linux', reason='emulator tests require Linux',
)

_ARCH = Architecture.AMD64
_ARENA, _ARENA_LEN = 0x7000, 0x1000
#: `mov rax, rbx` -- the instruction from the docstring above.  With a clean
#: RBX it CLEARS RAX's taint, so a block committed early under-taints.
_MOV_RAX_RBX = bytes.fromhex('4889d8')
#: A later block that touches neither RAX nor memory.
_INC_RCX = bytes.fromhex('48ffc1')


@pytest.fixture(scope='module')
def kit() -> dict[str, int]:
    builder = frompcode.builder_for(_ARCH)
    names = sorted(set(builder.name_by_off.values()))
    return {n: i for i, n in enumerate(names)}


def _fresh(layout: dict[str, int], *, rax_tainted: bool = True,
           ) -> tuple[Handle, Handle, list[int]]:
    """A runner whose RAX is tainted, and a plan for `mov rax, rbx`."""
    mem = B.mem_new(_ARENA, _ARENA_LEN)
    runner = B.runner_new(len(layout), -1, mem)
    seed = [0] * len(layout)
    if rax_tainted:
        seed[layout['RAX']] = (1 << 64) - 1
    B.runner_seed(runner, seed)
    got = compile_block(_ARCH, _MOV_RAX_RBX, 0x401000, layout,
                        publish_all_values=True)
    assert got is not None, 'mov rax, rbx did not compile'
    return runner, got[0], [0] * len(layout)


def _rax(runner: Handle, layout: dict[str, int]) -> int:
    return B.runner_taint(runner)[layout['RAX']]


def test_a_block_is_not_committed_until_the_next_one_arrives(kit: dict[str, int]) -> None:
    layout = kit
    runner, plan, regs = _fresh(layout)
    B.runner_on_block(runner, plan, 0x401000, regs)
    assert _rax(runner, layout) == (1 << 64) - 1, (
        'the block was committed before anything proved it ran')
    assert B.runner_stats(runner)['committed'] == 0

    B.runner_on_block(runner, plan, 0x401010, regs)
    assert _rax(runner, layout) == 0, (
        'reaching a second block should have committed the first, which clears '
        'RAX by moving a clean RBX into it')
    assert B.runner_stats(runner)['committed'] == 1


def test_an_abandoned_block_leaves_nothing_behind(kit: dict[str, int]) -> None:
    """The point of holding it: a block that does not finish applies nothing,
    so there is no state to roll back and no fault hook to get right.

    Checking the state straight after abandoning is not enough -- nothing has
    committed yet either way.  What has to hold is that the abandoned block is
    GONE: a later block must not carry it into effect.  (Measured: without that
    second half, an `abandon` that merely forgot to clear the pending flag
    passed this test.)
    """
    layout = kit
    runner, plan, regs = _fresh(layout)
    B.runner_on_block(runner, plan, 0x401000, regs)
    B.runner_abandon(runner)
    assert _rax(runner, layout) == (1 << 64) - 1, (
        'the abandoned block was applied anyway, clearing taint for an '
        'instruction that never ran')

    # A later block, which must commit ITSELF and nothing else.
    got = compile_block(_ARCH, _INC_RCX, 0x402000, layout,
                        publish_all_values=True)
    assert got is not None, 'inc rcx did not compile'
    B.runner_on_block(runner, got[0], 0x402000, regs)
    B.runner_finish(runner, True)
    assert _rax(runner, layout) == (1 << 64) - 1, (
        'a later block dragged the abandoned one into effect: RAX lost taint '
        'to `mov rax, rbx` from a block that never ran')
    stats = B.runner_stats(runner)
    assert stats['dropped'] == 1
    assert stats['committed'] == 1, (
        f'expected only the later block to commit, got {stats}')


def test_the_last_block_is_committed_only_if_the_run_finished(kit: dict[str, int]) -> None:
    layout = kit
    for completed, expect in ((True, 0), (False, (1 << 64) - 1)):
        runner, plan, regs = _fresh(layout)
        B.runner_on_block(runner, plan, 0x401000, regs)
        B.runner_finish(runner, completed)
        assert _rax(runner, layout) == expect, (
            f'finish(completed={completed}) left RAX {_rax(runner, layout):#x}')


def test_the_runtime_carries_no_python() -> None:
    """`blockpath.h` must compile with no Python headers reachable at all.

    Stated as a compile rather than as a comment, because a comment does not
    fail when someone reaches for a PyObject on the hot path.  Everything in
    that header runs per BLOCK EXECUTION, where touching CPython costs more
    than the work being wrapped.
    """
    import shutil
    cc = shutil.which('cc') or shutil.which('gcc')
    if cc is None:
        pytest.skip('no C compiler')
    header = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'microtaint', 'emulator', 'blockpath.h')
    proc = subprocess.run([cc, '-fsyntax-only', header],
                          capture_output=True, text=True, check=False)
    assert proc.returncode == 0, (
        'blockpath.h no longer compiles without Python.  Something on the '
        f'block hot path reached for CPython:\n{proc.stderr[:2000]}')


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
            input=_FAULTING.encode(), capture_output=True, check=False)
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
