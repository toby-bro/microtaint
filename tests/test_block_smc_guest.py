"""A guest that rewrites its own code, which is what keys the cache on bytes.

A compiled block is kept for the whole process now, and the C runtime holds its
plans by (address, size).  Self-modifying and JIT'd code is the one thing that
makes those two statements dangerous: the same address, the same length,
different instructions.  A cache that keyed a compiled block on where it sits
rather than on what it says would run the plan compiled for the code that used
to be there, and would do it silently.

The guest rewrites a four-byte function in its own text segment:

    48 31 c0 c3   xor rax, rax ; ret      -> returns 0,        clean
    48 89 f8 c3   mov rax, rdi ; ret      -> returns its arg,  TAINTED

Four bytes both times at the same address, so the runtime's (address, size) key
is identical and only noticing the rewrite can save it.  The two orderings fail
in opposite directions, which is why both are run:

    tainted second   a stale plan calls the result clean    -> UNDER-taint
    tainted first    a stale plan calls the result tainted  -> over-taint

Measured: with the cache keyed on (address, length) rather than on the bytes,
the first ordering leaves the result clean and reports nothing at all.

The per-instruction path is the oracle rather than a hand-written expectation.
It has no plan cache to go stale, it computes the same guest, and it agrees on
both orderings -- so a disagreement is block mode's, and there is no argument
about what the answer should have been.
"""
from __future__ import annotations

import io
import os
import platform
import subprocess
import tempfile
from collections.abc import Iterator
from typing import NamedTuple

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(platform.system() != 'Linux',
                       reason='emulator tests require Linux'),
]

#: Raw syscalls and no libc, so the only tainted value in the whole program is
#: what one of the two stubs returns and a finding can be counted rather than
#: filtered.  `mystub` is written as raw bytes at top level so it is EXACTLY the
#: four bytes that get rewritten, with no prologue in front of them.
_GUEST = r"""
static long sys_read(int fd, void *buf, unsigned long n) {
  long r; __asm__ volatile("syscall" : "=a"(r)
      : "0"(0), "D"((long)fd), "S"(buf), "d"(n) : "rcx","r11","memory"); return r;
}
static void sys_write(int fd, const void *buf, unsigned long n) {
  long r; __asm__ volatile("syscall" : "=a"(r)
      : "0"(1), "D"((long)fd), "S"(buf), "d"(n) : "rcx","r11","memory"); (void)r;
}
static void sys_exit(int c) {
  __asm__ volatile("syscall" :: "a"(60), "D"((long)c) : "rcx","r11");
  __builtin_unreachable();
}

/* volatile, or gcc deletes every store below as dead in a _start that never
   returns, and the whole test would compare zero against zero. */
volatile unsigned long slot_first, slot_second;

__asm__(".globl mystub\n.align 16\nmystub:\n .byte 0x48,0x31,0xc0,0xc3\n");
extern unsigned long mystub(unsigned long);

#define CLEAN(p)   do { (p)[0]=0x48; (p)[1]=0x31; (p)[2]=0xc0; (p)[3]=0xc3; } while (0)
#define TAINTED(p) do { (p)[0]=0x48; (p)[1]=0x89; (p)[2]=0xf8; (p)[3]=0xc3; } while (0)

void _start(void) {
  unsigned long v = 0;
  sys_read(0, &v, 8);                       /* v is the only tainted value */
  unsigned char *p = (unsigned char *)(void *)mystub;

#ifdef TAINTED_FIRST
  TAINTED(p);
#else
  CLEAN(p);
#endif
  slot_first = mystub(v);

#ifdef TAINTED_FIRST
  CLEAN(p);
#else
  TAINTED(p);
#endif
  slot_second = mystub(v);                  /* same address, same four bytes */

  /* Branch on BOTH, so whichever ordering is built there is exactly one
     secret-dependent branch for the per-instruction path to find and hold
     block mode to.  Branching only on the second leaves the other ordering
     with no leak at all, and a test whose oracle found nothing proves
     nothing. */
  if (slot_first  & 1) sys_write(1, "F", 1); else sys_write(1, "f", 1);
  if (slot_second & 1) sys_write(1, "S", 1); else sys_write(1, "s", 1);
  sys_exit(0);
}
"""


class _Answer(NamedTuple):
    """What one run concluded: the taint of each global, and the findings."""

    taint: dict[str, int]
    findings: list[tuple[str, int]]
    stats: dict[str, int] | None


def _symbols(path: str) -> dict[str, int]:
    out = subprocess.run(['nm', path], capture_output=True, text=True,
                         check=False).stdout
    got = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2].startswith('slot_'):
            got[parts[2]] = int(parts[0], 16)
    return got


@pytest.fixture(scope='module', params=[False, True],
                ids=['tainted stub second', 'tainted stub first'])
def guest(request: pytest.FixtureRequest) -> Iterator[tuple[str, dict[str, int], bool]]:
    fd, path = tempfile.mkstemp(suffix='.elf')
    os.close(fd)
    argv = ['gcc', '-static', '-nostdlib', '-O1', '-fno-stack-protector',
            # OMAGIC: a writable text segment, so the guest can rewrite itself
            # IN ITS OWN IMAGE.  Rewriting a freshly mmap'd page instead would
            # drag in a separate, unrelated defect: the hook tracks the code it
            # has planned as ONE [lo, hi) interval, so a far-away JIT page
            # stretches that interval over the whole image and every ordinary
            # data write in between then looks like self-modifying code.
            '-Wl,-N', '-o', path, '-x', 'c', '-']
    if request.param:
        argv.insert(1, '-DTAINTED_FIRST')
    built = subprocess.run(argv, input=_GUEST.encode(), capture_output=True,
                           check=False)
    if built.returncode != 0:
        os.unlink(path)
        pytest.skip(f'cannot build the guest: {built.stderr.decode()[:300]}')
    slots = _symbols(path)
    if len(slots) != 2:
        os.unlink(path)
        pytest.skip(f'cannot locate the guest globals: {slots}')
    yield path, slots, request.param
    os.unlink(path)


def _run(path: str, slots: dict[str, int], *, block: bool) -> _Answer:
    from qiling import Qiling
    from qiling.const import QL_VERBOSE

    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper

    reporter = Reporter(json_mode=True, stream=io.StringIO())
    prev = os.environ.get('MICROTAINT_BLOCK')
    os.environ['MICROTAINT_BLOCK'] = '1' if block else '0'
    saved, dn = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    try:
        ql = Qiling([path], '/', verbose=QL_VERBOSE.OFF)
        ql.os.stdin = io.BytesIO(b'12345678')
        w = MicrotaintWrapper(ql, reporter=reporter)
        os.dup2(dn, 1)
        ok = True
        try:
            ql.run()
        except Exception:
            ok = False
        os.dup2(saved, 1)
        stats = None
        if block:
            w.block_mode_finish(ok)
            got = w.block_mode_stats()
            assert got is not None, 'block mode did not install'
            stats = dict(got)
        taint = {n: w.shadow_mem.read_mask(a, 8) for n, a in slots.items()}
    finally:
        os.dup2(saved, 1)
        os.close(dn)
        os.close(saved)
        if prev is None:
            os.environ.pop('MICROTAINT_BLOCK', None)
        else:
            os.environ['MICROTAINT_BLOCK'] = prev
    findings = [(str(getattr(f.kind, 'value', f.kind)), f.address)
                for f in reporter.findings]
    return _Answer(taint, findings, stats)


@pytest.fixture(scope='module')
def answers(guest: tuple[str, dict[str, int], bool]) -> tuple[_Answer, _Answer, bool]:
    path, slots, tainted_first = guest
    return (_run(path, slots, block=True),
            _run(path, slots, block=False),
            tainted_first)


def test_the_guest_really_rewrote_its_own_code(
        answers: tuple[_Answer, _Answer, bool]) -> None:
    """The premise.  If the rewrite were never noticed, everything below would
    be true about a program that did nothing interesting."""
    block, _instr, _first = answers
    assert block.stats is not None
    assert block.stats['invalidations'] >= 1, (
        f'the self-modifying-code guard never fired, so the second stub was '
        f'never even a question: {block.stats}')
    assert block.stats['unhandled'] == 0, (
        f'{block.stats["unhandled"]} block executions were skipped, so a '
        f'missing taint below could just be unanalysed code: {block.stats}')


def test_the_tainted_stub_taints_its_result(
        answers: tuple[_Answer, _Answer, bool]) -> None:
    """The under-taint direction, in both orderings.

    One stub returns the tainted input; whichever order the two are written in,
    that result must come back tainted.  Running the stale plan for it gives a
    clean answer, and a clean answer to a tainted question is the failure that
    never announces itself.

    Measured: with the compiled-block cache keyed on (address, length) instead
    of on the bytes, this is exactly what happens -- the global comes back
    clean and the secret-dependent branch it feeds is never reported.
    """
    block, _instr, tainted_first = answers
    want = 'slot_first' if tainted_first else 'slot_second'
    assert block.taint[want] != 0, (
        f'{want} came back clean: block mode ran the plan compiled for the '
        f'code that used to be at that address, and the taint is simply gone '
        f'({block.taint})')


def test_block_mode_never_finds_less_than_the_per_instruction_path(
        answers: tuple[_Answer, _Answer, bool]) -> None:
    """The soundness differential, with the per-instruction path as the oracle.

    It has no plan cache to go stale and computes the same guest, so it settles
    what the answer should be without a hand-written expectation.  The
    assertion is containment rather than equality, and deliberately so: block
    mode is allowed to be less precise, never less complete.

    Where the two differ today is one known over-taint, which has nothing to do
    with self-modifying code: block mode does not clear a register zeroed by
    `xor reg,reg`.  The IR folds the VALUE of `XOR(x, x)` to a constant, but
    the taint rule for a binary op is `t_a | t_b` and does not notice that a
    provably constant result cannot depend on anything.  Reproduced in three
    lines against the C runtime, with no guest at all:

        compile `48 31 c0` with RAX tainted -> RAX comes back fully tainted

    The per-instruction path gets it right, which is why this test compares in
    one direction only.
    """
    block, instr, _first = answers
    assert any(instr.taint.values()), (
        f'the per-instruction path found no taint either, so it cannot serve '
        f'as an oracle here: {instr.taint}')
    for name, got in instr.taint.items():
        assert block.taint[name] | got == block.taint[name], (
            f'{name}: the per-instruction path found taint {got:#x} that block '
            f'mode missed ({block.taint[name]:#x}) -- block mode may be less '
            f'precise than the instruction path, never less complete')


def test_the_findings_block_mode_makes_are_the_ones_that_are_there(
        answers: tuple[_Answer, _Answer, bool]) -> None:
    """Every branch the per-instruction path calls secret-dependent must also
    be one block mode calls secret-dependent."""
    block, instr, _first = answers
    want = {f for f in instr.findings if f[0] == 'side_channel'}
    got = {f for f in block.findings if f[0] == 'side_channel'}
    assert want, (
        'the per-instruction path reported no leak, so there is nothing here '
        'to hold block mode to')
    assert want <= got, (
        f'block mode missed {sorted(want - got)}; it reported {sorted(got)}')
