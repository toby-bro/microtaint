"""The C block hook, against the per-instruction path, on a real guest.

`blockpath.h` is gated in isolation by tests/test_block_runtime_c.py, which is
where the two-pass protocol, the overlays and the deferred commit are attacked.
This is the other half: that the hook is WIRED correctly -- that it fires, that
its plans compile against the engine's own slot map, that it reads the register
file and the shadow the engine uses, and that the taint it leaves equals what
the instruction path leaves.

Wiring is where this kind of thing goes wrong quietly.  Two bugs found here were
both of one shape: the per-instruction path resolves something lazily on its
first use, and block mode never runs it.  The engine handle was one (all 10,540
blocks declined on the register read) and the C memory-read context was the
other (10,240 declined on the memory read).  Neither changed a single answer --
they just silently handled nothing -- which is why `unhandled` is asserted here
and not merely the taint.
"""
from __future__ import annotations

import io
import os
import platform
import subprocess
import tempfile
from collections.abc import Callable, Iterator
from typing import TypedDict

import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != 'Linux', reason='emulator tests require Linux',
)

#: Stores tainted bytes through a computed address, reloads them, indexes a
#: public table with a tainted value and branches on one: the shapes a block has
#: to get right.  The loop bounds come from the READ's return value, so the
#: compiler cannot unroll them into one straight-line block -- at a constant 16
#: it did exactly that, and the guest produced a single block, which is no
#: evidence about block handling at all.
_GUEST = r"""
long sys_read(int fd, void *buf, unsigned long n){long r;
  __asm__ volatile("syscall":"=a"(r):"0"(0),"D"(fd),"S"(buf),"d"(n):"rcx","r11","memory");return r;}
void sys_exit(int c){__asm__ volatile("syscall"::"a"(60),"D"((long)c):"rcx","r11");__builtin_unreachable();}
static const unsigned char TBL[16] = {3,1,4,1,5,9,2,6,5,3,5,8,9,7,9,3};
unsigned long out;
void _start(void){
  unsigned char in[32], work[32];
  long n = sys_read(0, in, 32);
  if (n <= 0) sys_exit(1);
  unsigned long acc = 0;
  for (int round = 0; round < 8; round++) {
    for (long i = 0; i < n; i++) work[i] = in[i] ^ (unsigned char)(i + round);
    for (long i = 0; i < n; i++) work[i] = TBL[work[i] & 15];
    for (long i = 0; i < n; i++) { acc = acc * 31 + work[i]; if (work[i] & 1) acc ^= 0xa5; }
    for (long i = 0; i < n; i++) in[i] = work[i];
  }
  out = acc;
  sys_exit(0);
}
"""


def _build() -> str:
    fd, path = tempfile.mkstemp(suffix='.elf')
    os.close(fd)
    built = subprocess.run(
        ['gcc', '-nostdlib', '-static', '-no-pie', '-O1', '-fno-stack-protector',
         '-o', path, '-x', 'c', '-'],
        input=_GUEST.encode(), capture_output=True, check=False)
    if built.returncode != 0:
        os.unlink(path)
        pytest.skip(f'cannot build the guest: {built.stderr.decode()[:300]}')
    return path


class RunTaint(TypedDict):
    """One run's whole answer: register taint by name, memory taint by
    address, and the block-mode counters (None when block mode never
    installed, which is what tells the two runs apart)."""

    regs: dict[str, int]
    mem: dict[int, int]
    stats: dict[str, int] | None


#: The two runs under comparison: the instruction path, then block mode.
BothRuns = tuple[RunTaint, RunTaint]


def _run(guest: str, block: bool) -> RunTaint:
    from qiling import Qiling
    from qiling.const import QL_VERBOSE

    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper

    prev = os.environ.get('MICROTAINT_BLOCK')
    os.environ['MICROTAINT_BLOCK'] = '1' if block else '0'
    saved, dn = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    try:
        ql = Qiling([guest], '/', verbose=QL_VERBOSE.OFF)
        ql.os.stdin = io.BytesIO(bytes((i * 7 + 13) & 0xFF for i in range(32)))
        w = MicrotaintWrapper(ql, reporter=Reporter(json_mode=False, stream=io.StringIO()))
        os.dup2(dn, 1)
        ok = True
        try:
            ql.run()
        except Exception:  # a guest that faults is still a comparison
            ok = False
        os.dup2(saved, 1)
        w.block_mode_finish(ok)
        mem: dict[int, int] = {}
        for lo, hi, *_ in ql.mem.map_info:
            if hi - lo > (1 << 20):
                continue
            for a in range(lo, hi, 8):
                m = w.shadow_mem.read_mask(a, 8)
                if m:
                    mem[a] = m
        return RunTaint(regs={k: v for k, v in w.register_taint.items() if v},
                        mem=mem, stats=w.block_mode_stats())
    finally:
        os.dup2(saved, 1)
        os.close(dn)
        if prev is None:
            os.environ.pop('MICROTAINT_BLOCK', None)
        else:
            os.environ['MICROTAINT_BLOCK'] = prev


@pytest.fixture(scope='module')
def both() -> Iterator[BothRuns]:
    guest = _build()
    try:
        yield _run(guest, block=False), _run(guest, block=True)
    finally:
        os.unlink(guest)


def test_the_block_hook_actually_handles_the_blocks(
        both: BothRuns) -> None:
    """Every block, or the comparison below proves nothing.

    A hook that declined everything would leave the taint state untouched, and
    on a guest whose taint all comes from one `read` that can look identical to
    a correct run.
    """
    _instr, block = both
    stats = block['stats']
    assert stats is not None, 'block mode did not install'
    assert stats['blocks'] > 100, f'the hook barely fired: {stats}'
    assert stats['handled'] == stats['blocks'], (
        f'the block hook declined {stats["unhandled"]} of {stats["blocks"]} '
        f'blocks, which is unanalysed code: {stats}')


def test_the_instruction_path_did_not_also_run(
        both: BothRuns) -> None:
    """Block mode OWNS the taint.  Both hooks armed would write the same state
    and clobber each other, and the comparison would measure whichever ran
    last -- which is exactly how an earlier version of this reported success."""
    instr, _block = both
    assert instr['stats'] is None, (
        'the instruction run installed block mode too, so the two runs are not '
        'the two implementations')


def test_block_and_instruction_paths_agree(
        both: BothRuns) -> None:
    instr, block = both
    assert instr['regs'] or instr['mem'], (
        'the instruction path found no taint at all, so there is nothing to '
        'compare and this test would pass against anything')

    lost = {k: (v, block['regs'].get(k, 0)) for k, v in instr['regs'].items()
            if v & ~block['regs'].get(k, 0)}
    assert not lost, ('block mode LOST register taint: '
                      + ', '.join(f'{k} {a:#x} -> {b:#x}' for k, (a, b) in sorted(lost.items())))
    lost_mem = {k: (v, block['mem'].get(k, 0)) for k, v in instr['mem'].items()
                if v & ~block['mem'].get(k, 0)}
    assert not lost_mem, (
        f'block mode LOST taint in {len(lost_mem)} memory words, first few: '
        + ', '.join(f'{k:#x} {a:#018x} -> {b:#018x}'
                    for k, (a, b) in sorted(lost_mem.items())[:4]))
    assert instr['regs'] == block['regs'], 'register taint differs'
    assert instr['mem'] == block['mem'], 'memory taint differs'


def test_a_plan_that_reads_more_registers_than_the_scratch_holds_declines(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The plan says how many registers to read; the buffer they land in is the
    hook's, sized to the whole register file.

    Those two facts used to come from the same place.  They no longer do: a
    plan is kept for the life of the process and run by whichever emulator
    meets the block next, so "this plan's read fits my buffer" is an assumption
    about somebody else rather than a local fact.  If it were ever wrong,
    `uc_reg_read_batch` would write past the end of that buffer -- a heap
    overflow, not a wrong answer, and the kind that surfaces somewhere else
    entirely.

    Declining is the sound response: the block is counted as unhandled, which
    is unanalysed code and visible in the stats, rather than read into memory
    the hook does not own.  Forced here by handing the hook a scratch sized for
    a single read call while its blocks ask for more; nothing else changes.

    The taint of a declined block is simply not computed, which is why this
    asserts the COUNTERS and not the answer.  An engine that declined silently
    would look exactly like one that worked.
    """
    from microtaint.emulator import blockpath_c

    real_hook_new = blockpath_c.hook_new

    def one_call_scratch(fastctx: int, compiler: Callable[[int, int], object | None],
                         ids: int, ptrs: int, vals: int, n_calls: int,
                         slots: list[int], *rest: int) -> object:
        """`hook_new` with the scratch forced to a single read call.

        The parameter types mirror `blockpath_c.hook_new` exactly rather than
        being `object`: this forwards them straight through, so a stand-in that
        does not match the real signature is the one thing that would make the
        substitution meaningless.
        """
        return real_hook_new(fastctx, compiler, ids, ptrs, vals, 1, slots, *rest)

    monkeypatch.setattr(blockpath_c, 'hook_new', one_call_scratch)
    guest = _build()
    try:
        got = _run(guest, block=True)
    finally:
        os.unlink(guest)

    stats = got['stats']
    assert stats is not None, 'block mode did not install'
    assert stats['blocks'] > 100, (
        f'the hook barely fired ({stats["blocks"]} blocks), so the plans that '
        f'overrun the scratch may never have been reached: {stats}')
    assert stats['no_regs'] > 0, (
        f'no block declined on its register read, so a plan asking for more '
        f'words than the scratch holds was read into it anyway: {stats}')
    assert stats['unhandled'] >= stats['no_regs'], (
        f'{stats["no_regs"]} blocks declined on their register read but only '
        f'{stats["unhandled"]} were counted unhandled: a skipped block that is '
        f'not counted is unanalysed code nobody knows about')
    assert stats['handled'] + stats['unhandled'] <= stats['blocks'], stats
