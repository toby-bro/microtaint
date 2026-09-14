"""A restored emulator must answer exactly what a fresh one answers.

Returning to a checkpoint instead of building a new emulator per input is worth
about 5x on a fuzzer-shaped run, and it is worth nothing at all if the answer
changes.  So every guest here is run twice: once in a brand-new emulator, once
from a checkpoint, and the two are compared on what they FOUND, on the value
and taint of the guest's own result, and on whether the memory map came back.

The guests are not arbitrary.  Each one is a way a restore has actually gone
wrong, and every one of them failed here before the corresponding line of
`MicrotaintWrapper.restore` existed:

    leak      maps a region and never unmaps it.  Qiling's restore re-maps
              what is MISSING and never unmaps what the guest added, so the
              map grew from 8 regions to 16 over 8 iterations.
    mprotect  drops its own write permission.  A region still mapped with
              different permissions keeps them, so the next iteration faulted
              on a store it had made happily the first time.
    smc       rewrites a four-byte function.  Unicorn kept running the
              translation of the rewritten form after the bytes were put back
              (wrong VALUE), and the engine's address-keyed plan cache kept the
              plan compiled for it (right value, lost TAINT).
    jit       writes code into a fresh page and calls it.
    crash     faults on half its inputs, because a fuzzer's inputs do.
    warm      calls a function, reads its input, calls the same function on it.

Both engine paths are driven, because two of those bugs were fixed on one path
and still live on the other: the signal for "the guest rewrote code" came from
block mode's stats, which do not exist per-instruction, so it read zero there
forever.
"""
from __future__ import annotations

import io
import os
import platform
import subprocess
import tempfile
from collections.abc import Iterator
from typing import TYPE_CHECKING, NamedTuple

import pytest

if TYPE_CHECKING:
    from qiling import Qiling

    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper

pytestmark = pytest.mark.skipif(
    platform.system() != 'Linux', reason='emulator tests require Linux',
)

#: The x86-64 syscall ABI is rdi, rsi, rdx, r10, r8, r9.  Leaving RDX out is
#: quiet and ruinous: a read's count lands in r10 and the call reads nothing,
#: which made an early version of the `leak` guest map nothing at all and pass.
_SYS = r"""
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
"""

_LEAK = _SYS + r"""
__attribute__((noinline)) void body(void) {
  unsigned char buf[16];
  if (sys6(0, 0, (long)buf, 16, 0, 0, 0) <= 0) sys_exit(1);
  long p = sys6(9, 0, 65536, 3, 0x22, -1, 0);      /* mmap, never unmapped */
  if (p < 0) sys_exit(2);
  volatile unsigned char *q = (volatile unsigned char *)p;
  for (int i = 0; i < 16; i++) q[i] = buf[i];
  unsigned long a = 0;
  for (int i = 0; i < 16; i++) a = a * 31u + q[i];
  out = a;
}
void _start(void) { body(); sys_exit(0); }
"""

_MPROTECT = _SYS + r"""
__attribute__((noinline)) void body(void) {
  unsigned char buf[16];
  if (sys6(0, 0, (long)buf, 16, 0, 0, 0) <= 0) sys_exit(1);
  unsigned long a = 0;
  for (int i = 0; i < 16; i++) a = a * 31u + buf[i];
  out = a;                                          /* needs write permission */
  unsigned long page = ((unsigned long)&out) & ~0xFFFUL;
  sys6(10, (long)page, 4096, 1, 0, 0, 0);           /* mprotect PROT_READ */
}
void _start(void) { body(); sys_exit(0); }
"""

_SMC = _SYS + r"""
/* mov rax, rdi ; ret -- returns its argument, so the result is TAINTED */
__asm__(".globl mystub\n.align 16\nmystub:\n .byte 0x48,0x89,0xf8,0xc3\n");
extern unsigned long mystub(unsigned long);
__attribute__((noinline)) void body(void) {
  unsigned char buf[8];
  if (sys6(0, 0, (long)buf, 8, 0, 0, 0) <= 0) sys_exit(1);
  unsigned long v = 0;
  for (int i = 0; i < 8; i++) v |= ((unsigned long)buf[i]) << (8 * i);
  out = mystub(v);
  unsigned char *p = (unsigned char *)(void *)mystub;
  p[0] = 0x48; p[1] = 0x31; p[2] = 0xc0; p[3] = 0xc3;  /* xor rax,rax ; ret */
  (void)mystub(v);
}
void _start(void) { body(); sys_exit(0); }
"""

_JIT = _SYS + r"""
__attribute__((noinline)) void body(void) {
  unsigned char buf[8];
  if (sys6(0, 0, (long)buf, 8, 0, 0, 0) <= 0) sys_exit(1);
  unsigned long v = 0;
  for (int i = 0; i < 8; i++) v |= ((unsigned long)buf[i]) << (8 * i);
  long p = sys6(9, 0, 4096, 7, 0x22, -1, 0);        /* mmap RWX */
  if (p < 0) sys_exit(2);
  unsigned char *c = (unsigned char *)p;
  c[0] = 0x48; c[1] = 0x89; c[2] = 0xf8;            /* mov rax, rdi */
  c[3] = 0x48; c[4] = 0x01; c[5] = 0xf8;            /* add rax, rdi */
  c[6] = 0xc3;
  out = ((unsigned long (*)(unsigned long))(void *)c)(v);
  sys6(11, p, 4096, 0, 0, 0, 0);
}
void _start(void) { body(); sys_exit(0); }
"""

#: Faults on the LENGTH, not the content.  Branching on tainted data halts the
#: per-instruction path at that branch by design, so a content-dependent fault
#: was never reached there and the faulting half was not exercised at all.
_CRASH = _SYS + r"""
__attribute__((noinline)) void body(void) {
  unsigned char buf[16];
  long n = sys6(0, 0, (long)buf, 16, 0, 0, 0);
  if (n <= 0) sys_exit(1);
  unsigned long a = 0;
  for (int i = 0; i < 16; i++) a = a * 31u + buf[i];
  out = a;
  if (n & 1) { ((void (*)(void))0x10)(); }          /* jump to unmapped */
}
void _start(void) { body(); sys_exit(0); }
"""

_WARM = _SYS + r"""
__attribute__((noinline)) unsigned long mix(unsigned long x) {
  unsigned long a = x;
  for (int i = 0; i < 8; i++) a = a * 31u + (a >> 3);
  return a;
}
__attribute__((noinline)) void body(void) {
  unsigned char buf[8];
  unsigned long warm = 0;
  for (int i = 0; i < 4; i++) warm += mix((unsigned long)i);
  out = warm;
  if (sys6(0, 0, (long)buf, 8, 0, 0, 0) <= 0) sys_exit(1);
  unsigned long v = 0;
  for (int i = 0; i < 8; i++) v |= ((unsigned long)buf[i]) << (8 * i);
  out = mix(v);
}
void _start(void) { body(); sys_exit(0); }
"""


_REGS = _SYS + r"""
volatile unsigned long sink;      /* the tainted hash */
__attribute__((noinline)) void body(void) {
  unsigned long snap;
  __asm__ volatile("mov %%rbx, %0" : "=r"(snap) :: );
  out = snap;                     /* rbx as body found it: must stay CLEAN */
  unsigned char buf[16];
  if (sys6(0, 0, (long)buf, 16, 0, 0, 0) <= 0) sys_exit(1);
  unsigned long a = 0;
  for (int i = 0; i < 16; i++) a = a * 31u + buf[i];
  sink = a;
}
void _start(void) {
  body();
  unsigned long t = sink;         /* tainted */
  /* Past body's epilogue, so this SURVIVES the iteration.  Leaving it inside
     body does not: rbx is callee-saved and the epilogue puts it back, taint
     and all, which is why an earlier version of this guest caught nothing. */
  __asm__ volatile("mov %0, %%rbx" :: "r"(t) : "rbx");
  sys_exit(0);
}
"""


#: Makes its own READ-ONLY data writable and scribbles on it.  The restore
#: skips read-only regions -- two thirds of a checkpoint's memory is the
#: target's text and rodata, which the guest cannot write -- and this is the
#: case that makes "cannot" false.  `seen` publishes the byte BEFORE the
#: scribble, so it must read the original on every iteration.
_RODATA = _SYS + r"""
__attribute__((section(".rodata"))) const unsigned char marker[16] = {
  0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
  0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00
};
volatile unsigned long seen;
__attribute__((noinline)) void body(void) {
  seen = marker[0];                       /* 0x11 unless a scribble survived */
  unsigned char buf[16];
  if (sys6(0, 0, (long)buf, 16, 0, 0, 0) <= 0) sys_exit(1);
  unsigned long a = 0;
  for (int i = 0; i < 16; i++) a = a * 31u + buf[i];
  out = a;
  unsigned long page = ((unsigned long)marker) & ~0xFFFUL;
  if (sys6(10, (long)page, 4096, 3, 0, 0, 0) != 0) sys_exit(3);  /* mprotect RW */
  *(volatile unsigned char *)marker = 0xAA;                      /* scribble */
}
void _start(void) { body(); sys_exit(0); }
"""


class Guest(NamedTuple):
    name: str
    source: str
    inputs: list[bytes]
    #: The guest maps something the checkpoint did not have.
    maps: bool = False
    #: Some inputs fault and some do not; both outcomes must occur.
    faults: bool = False
    #: Globals compared by value AND taint.  `out` is always one of them.
    probes: tuple[str, ...] = ('out',)
    #: `out` must come back CLEAN here: it publishes a register the guest had
    #: not tainted yet, and the premise is carried by another global instead.
    out_is_clean: bool = False
    #: Link with a writable text segment.  The default, because the `smc`
    #: guest rewrites its own code -- but it also makes every segment
    #: writable, so a guest that is about READ-ONLY memory must not have it.
    omagic: bool = True


_GUESTS = [
    Guest('leak', _LEAK, [bytes([i] * 16) for i in range(1, 7)], maps=True),
    Guest('mprotect', _MPROTECT, [bytes([i] * 16) for i in range(1, 7)]),
    Guest('smc', _SMC, [bytes([i] * 8) for i in range(1, 7)]),
    Guest('jit', _JIT, [bytes([i] * 8) for i in range(1, 7)], maps=True),
    Guest('crash', _CRASH, [bytes([i] * (15 + i % 2)) for i in range(1, 9)],
          faults=True),
    Guest('warm', _WARM, [bytes([i] * 8) for i in range(1, 7)]),
    Guest('regs', _REGS, [bytes([i] * 16) for i in range(1, 7)],
          probes=('out', 'sink'), out_is_clean=True),
    Guest('rodata', _RODATA, [bytes([i] * 16) for i in range(1, 7)],
          probes=('out', 'seen'), omagic=False),
]


#: The first byte of the `rodata` guest's marker array.
_ORIGINAL_MARKER = 0x11


def _marker_addr(binary: str) -> int:
    nm = subprocess.run(['nm', binary], capture_output=True, text=True,
                        check=False).stdout
    addr = next((int(line.split()[0], 16) for line in nm.splitlines()
                 if line.split()[-1:] == ['marker']), None)
    if addr is None:
        pytest.skip('cannot locate the rodata marker')
    return addr


def _build(source: str, omagic: bool = True) -> str:
    fd, path = tempfile.mkstemp(suffix='.elf')
    os.close(fd)
    flags = ['-static', '-nostdlib', '-O1', '-fno-stack-protector']
    if omagic:
        flags.append('-Wl,-N')
    built = subprocess.run(
        ['gcc', *flags, '-o', path, '-x', 'c', '-'],
        input=source.encode(), capture_output=True, check=False)
    if built.returncode != 0:
        os.unlink(path)
        pytest.skip(f'cannot build the guest: {built.stderr.decode()[:300]}')
    return path


@pytest.fixture(scope='module')
def built() -> Iterator[dict[str, tuple[str, int, tuple[int, ...]]]]:
    """name -> (binary, address of `body`, addresses of the probed globals)."""
    made: dict[str, tuple[str, int, tuple[int, ...]]] = {}
    for guest in _GUESTS:
        path = _build(guest.source, guest.omagic)
        nm = subprocess.run(['nm', path], capture_output=True, text=True,
                            check=False).stdout
        syms = {line.split()[-1]: int(line.split()[0], 16)
                for line in nm.splitlines() if len(line.split()) >= 3}
        made[guest.name] = (path, syms['body'],
                            tuple(syms[name] for name in guest.probes))
    yield made
    for path, _b, _probes in made.values():
        os.unlink(path)


class Answer(NamedTuple):
    """Everything a run is compared on."""

    completed: bool
    findings: tuple[str, ...]
    #: (value, taint mask) per probed global
    probes: tuple[tuple[int, int], ...]
    regions: int
    mapped: int


def _fresh(binary: str, probes: tuple[int, ...], inp: bytes,
           block: bool) -> Answer:
    from qiling import Qiling
    from qiling.const import QL_VERBOSE

    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper

    saved, dn = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    try:
        rep = Reporter(json_mode=True, stream=io.StringIO())
        ql = Qiling([binary], '/', verbose=QL_VERBOSE.OFF)
        ql.os.stdin = io.BytesIO(inp)
        w = MicrotaintWrapper(ql, reporter=rep)
        os.dup2(dn, 1)
        ok = True
        try:
            ql.run()
        except Exception:
            ok = False
        os.dup2(saved, 1)
        w.block_mode_finish(ok)
        # The SAME definition of "finished" the restored path uses: the guest
        # reached its own exit.  Inferring it from "nothing was raised" on one
        # side and from the exit syscall on the other compares two different
        # questions, which is exactly what it did when first written.
        return _answer(ql, w, rep, ok and w.guest_exited, probes)
    finally:
        os.dup2(saved, 1)
        os.close(dn)
        os.close(saved)


def _restored(binary: str, entry: int, probes: tuple[int, ...],
              inputs: list[bytes], block: bool) -> list[Answer]:
    from qiling import Qiling
    from qiling.const import QL_VERBOSE

    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper

    saved, dn = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    try:
        rep = Reporter(json_mode=True, stream=io.StringIO())
        ql = Qiling([binary], '/', verbose=QL_VERBOSE.OFF)
        ql.os.stdin = io.BytesIO(b'')
        w = MicrotaintWrapper(ql, reporter=rep)
        os.dup2(dn, 1)
        w.run_to(entry)
        cp = w.checkpoint()
        out = []
        for inp in inputs:
            w.restore(cp)
            ql.os.stdin = io.BytesIO(inp)
            rep.findings.clear()
            ok = bool(w.resume(cp))
            out.append(_answer(ql, w, rep, ok, probes))
        os.dup2(saved, 1)
        return out
    finally:
        os.dup2(saved, 1)
        os.close(dn)
        os.close(saved)


def _answer(ql: Qiling, w: MicrotaintWrapper, rep: Reporter, ok: bool,
            probes: tuple[int, ...]) -> Answer:
    """Everything a run is compared on.

    Typed rather than `object`: this reads five attributes off three of them,
    and `object` meant none of those reads were checked -- a renamed field
    would have surfaced as a run-time AttributeError inside a helper that only
    the comparison calls.
    """
    return Answer(
        completed=ok,
        # `hex(f.address)` rather than `f.to_dict()['address']`: the same
        # string by construction, but the dict erases its type.
        findings=tuple(sorted(hex(f.address) for f in rep.findings)),
        probes=tuple((int.from_bytes(bytes(ql.mem.read(a, 8)), 'little'),
                      w.shadow_mem.read_mask(a, 8)) for a in probes),
        regions=len(ql.mem.map_info),
        mapped=sum(hi - lo for lo, hi, *_ in ql.mem.map_info),
    )


@pytest.fixture(scope='module', params=[True, False], ids=['block', 'instr'])
def block_mode(request: pytest.FixtureRequest) -> Iterator[bool]:
    prev = os.environ.get('MICROTAINT_BLOCK')
    os.environ['MICROTAINT_BLOCK'] = '1' if request.param else '0'
    yield bool(request.param)
    if prev is None:
        os.environ.pop('MICROTAINT_BLOCK', None)
    else:
        os.environ['MICROTAINT_BLOCK'] = prev


@pytest.fixture(scope='module')
def runs(built: dict[str, tuple[str, int, tuple[int, ...]]], block_mode: bool,
         ) -> dict[str, tuple[list[Answer], list[Answer]]]:
    """name -> (a fresh emulator per input, one restored emulator)."""
    out = {}
    for guest in _GUESTS:
        binary, body, probes = built[guest.name]
        fresh = [_fresh(binary, probes, i, block_mode) for i in guest.inputs]
        rest = _restored(binary, body, probes, guest.inputs, block_mode)
        out[guest.name] = (fresh, rest)
    return out


@pytest.mark.parametrize('guest', _GUESTS, ids=lambda g: g.name)
def test_the_guests_do_what_they_are_here_to_do(
        runs: dict[str, tuple[list[Answer], list[Answer]]], guest: Guest,
        ) -> None:
    """The premise.  A guest that mapped nothing, faulted never, or carried no
    taint would make the comparison below true for the wrong reason -- and
    three of these guests have silently done exactly that while being written.
    """
    fresh, _restored_runs = runs[guest.name]
    # `leak` and `jit` are about the memory MAP, and on the LogicCircuit path
    # their result is not tainted at all -- taint through an mmap'd page is
    # dropped there, which is a separate engine bug with its own test
    # (test_taint_through_mmapped_memory.py).  Requiring taint here would make
    # this file fail for something it is not about.
    if not guest.maps:
        assert any(t for a in fresh for _v, t in a.probes), (
            f'{guest.name}: no run produced a tainted probe, so comparing '
            f'taint proves nothing')
    if guest.out_is_clean:
        assert all(a.probes[0][1] == 0 for a in fresh), (
            f'{guest.name}: `out` publishes a register from before any input '
            f'existed and a FRESH run already finds it tainted, so a restored '
            f'run finding it tainted would prove nothing')
    if guest.maps:
        assert len({a.regions for a in fresh}) == 1
        assert fresh[0].regions > 4, (
            f'{guest.name}: is meant to map something and did not')
    if guest.faults:
        n = sum(1 for a in fresh if not a.completed)
        assert 0 < n < len(fresh), (
            f'{guest.name}: {n} of {len(fresh)} fresh runs faulted; both '
            f'outcomes have to occur for this to be about faults at all')


@pytest.mark.parametrize('guest', _GUESTS, ids=lambda g: g.name)
def test_a_restored_run_answers_what_a_fresh_one_answers(
        runs: dict[str, tuple[list[Answer], list[Answer]]], guest: Guest,
        ) -> None:
    """The property, for every input: same findings, same result, same taint of
    that result, same completion."""
    fresh, rest = runs[guest.name]
    for i, (a, b) in enumerate(zip(fresh, rest, strict=True)):
        assert b == a, (
            f'{guest.name} input {i}: a restored emulator answered\n'
            f'  restored={b}\n  fresh   ={a}')


@pytest.mark.parametrize('guest', _GUESTS, ids=lambda g: g.name)
def test_a_restored_emulator_does_not_drift(
        runs: dict[str, tuple[list[Answer], list[Answer]]], guest: Guest,
        ) -> None:
    """And it is still the same emulator at the end as at the start.

    A map that grows by a region an iteration changes no answer for a long
    while and then changes all of them.
    """
    _fresh_runs, rest = runs[guest.name]
    assert rest[0].regions == rest[-1].regions, (
        f'{guest.name}: {rest[0].regions} regions after the first iteration, '
        f'{rest[-1].regions} after the last')
    assert rest[0].mapped == rest[-1].mapped, (
        f'{guest.name}: {rest[0].mapped} bytes mapped after the first '
        f'iteration, {rest[-1].mapped} after the last')


def test_restoring_abandons_a_block_the_previous_run_still_held(
        built: dict[str, tuple[str, int, tuple[int, ...]]],
        block_mode: bool) -> None:
    """A caller that drives the emulator itself leaves a block held.

    Block mode holds a block's taint until the NEXT block proves it completed,
    so a run that ends mid-block leaves one pending.  `resume` finishes it; a
    caller using `ql.run` directly does not, and then the held block commits
    into the FOLLOWING iteration -- the previous input's taint landing on this
    input's result.  Measured with that line removed: `out`, which publishes a
    register from before any input existed, came back fully tainted from the
    second iteration on.
    """
    if not block_mode:
        pytest.skip("the held block is block mode's deferred commit")
    from qiling import Qiling
    from qiling.const import QL_VERBOSE

    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper

    binary, body, probes = built['regs']
    out_addr, sink_addr = probes
    saved, dn = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    seen = []
    try:
        rep = Reporter(json_mode=True, stream=io.StringIO())
        ql = Qiling([binary], '/', verbose=QL_VERBOSE.OFF)
        ql.os.stdin = io.BytesIO(b'')
        w = MicrotaintWrapper(ql, reporter=rep)
        os.dup2(dn, 1)
        w.run_to(body)
        cp = w.checkpoint()
        for i in range(4):
            w.restore(cp)
            ql.os.stdin = io.BytesIO(bytes([i + 1] * 16))
            rep.findings.clear()
            ql.exit_point = None
            try:
                ql.run(begin=cp.address)   # deliberately NOT resume()
            except Exception:
                pass
            seen.append((w.shadow_mem.read_mask(out_addr, 8),
                         w.shadow_mem.read_mask(sink_addr, 8)))
        os.dup2(saved, 1)
    finally:
        os.dup2(saved, 1)
        os.close(dn)
        os.close(saved)
    assert all(sink for _out, sink in seen), (
        f'the guest never tainted anything, so this proves nothing: {seen}')
    assert all(out == 0 for out, _sink in seen), (
        f'`out` publishes a register from before the input was read and came '
        f'back tainted: the block held by the previous iteration committed '
        f'into this one. masks={[hex(o) for o, _s in seen]}')


def test_each_iteration_reports_its_own_counts(
        built: dict[str, tuple[str, int, tuple[int, ...]]],
        block_mode: bool) -> None:
    """`resume` says what THIS input did, not what every input has done.

    The hook's counters run for the life of the emulator, so in a checkpoint
    loop they answer a question nobody asked: a fuzzer wants to know whether
    the input it just ran left blocks unanalysed, and a number that only ever
    grows cannot say.
    """
    if not block_mode:
        pytest.skip('block counts come from block mode')
    from qiling import Qiling
    from qiling.const import QL_VERBOSE

    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper

    binary, body, _probes = built['warm']
    saved, dn = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    seen = []
    try:
        rep = Reporter(json_mode=True, stream=io.StringIO())
        ql = Qiling([binary], '/', verbose=QL_VERBOSE.OFF)
        ql.os.stdin = io.BytesIO(b'')
        w = MicrotaintWrapper(ql, reporter=rep)
        os.dup2(dn, 1)
        w.run_to(body)
        cp = w.checkpoint()
        for i in range(4):
            w.restore(cp)
            ql.os.stdin = io.BytesIO(bytes([i + 1] * 8))
            rep.findings.clear()
            seen.append(w.resume(cp))
        os.dup2(saved, 1)
    finally:
        os.dup2(saved, 1)
        os.close(dn)
        os.close(saved)
    assert all(r.blocks > 0 for r in seen), (
        f'an iteration reported no blocks at all: {seen}')
    assert len({r.blocks for r in seen}) == 1, (
        f'the counts differ between iterations of the same guest, so they are '
        f'accumulating rather than being per-iteration: '
        f'{[r.blocks for r in seen]}')
    assert all(r.handled == r.blocks and r.unhandled == 0 for r in seen), (
        f'blocks went unanalysed in a restored run: {seen}')


def test_read_only_memory_the_guest_made_writable_is_put_back(
        built: dict[str, tuple[str, int, tuple[int, ...]]],
        block_mode: bool) -> None:
    """Restoring skips regions the guest cannot have written.

    Two thirds of a checkpoint's memory is the target's own text and rodata --
    740 KB of 1,116 on a static-glibc guest -- and rewriting it every iteration
    is work to put back bytes that are already there.  A read-only region
    cannot be written by the guest, so it is skipped.

    This is the case that makes "cannot" false: the guest mprotects its own
    rodata writable and scribbles on it.  Its PERMISSIONS then differ from the
    checkpoint's, so the region is unmapped and written in full.

    Asserted by reading the guest's memory directly rather than by comparing
    two runs.  The comparison does not catch it -- measured, with the restore
    deliberately broken the page came back ZEROED and every fresh-vs-restored
    assertion still passed -- so the thing being claimed is checked head on.
    """
    from qiling import Qiling
    from qiling.const import QL_VERBOSE

    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper

    binary, body, _probes = built['rodata']
    marker = _marker_addr(binary)
    original = _ORIGINAL_MARKER
    saved, dn = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    seen = []
    try:
        rep = Reporter(json_mode=True, stream=io.StringIO())
        ql = Qiling([binary], '/', verbose=QL_VERBOSE.OFF)
        ql.os.stdin = io.BytesIO(b'')
        w = MicrotaintWrapper(ql, reporter=rep)
        os.dup2(dn, 1)
        w.run_to(body)
        cp = w.checkpoint()
        for i in range(3):
            ql.os.stdin = io.BytesIO(bytes([i + 1] * 16))
            rep.findings.clear()
            w.resume(cp)
            scribbled = bytes(ql.mem.read(marker, 1))[0]
            w.restore(cp)
            seen.append((scribbled, bytes(ql.mem.read(marker, 1))[0]))
        os.dup2(saved, 1)
    finally:
        os.dup2(saved, 1)
        os.close(dn)
        os.close(saved)
    assert all(after_run == 0xAA for after_run, _after_restore in seen), (
        f'the guest never scribbled on its rodata, so this proves nothing: '
        f'{seen}')
    assert all(after_restore == original
               for _after_run, after_restore in seen), (
        f'rodata the guest made writable and scribbled on did not come back: '
        f'{[(hex(a), hex(b)) for a, b in seen]}, expected 0x{original:02x}')


def test_a_checkpoint_after_taint_exists_is_refused(
        built: dict[str, tuple[str, int, tuple[int, ...]]],
        block_mode: bool) -> None:
    """Restoring CLEARS the taint rather than putting it back.

    That is right for what this is for -- checkpoint before the input, drive
    inputs from there -- and it would silently throw away taint a caller had
    seeded before the checkpoint.  Refusing beats losing it quietly.  Putting
    it back instead needs the shadow to be snapshottable, which it is not.
    """
    from qiling import Qiling
    from qiling.const import QL_VERBOSE

    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.snapshot import CheckpointError
    from microtaint.emulator.wrapper import MicrotaintWrapper

    binary, body, _probes = built['warm']
    saved, dn = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    try:
        rep = Reporter(json_mode=True, stream=io.StringIO())
        ql = Qiling([binary], '/', verbose=QL_VERBOSE.OFF)
        ql.os.stdin = io.BytesIO(b'')
        w = MicrotaintWrapper(ql, reporter=rep, qiling_stats=False)
        os.dup2(dn, 1)
        w.run_to(body)
        # A checkpoint here is fine: nothing is tainted yet.
        w.checkpoint()
        # Seed taint the way a caller would, then ask again.
        w.taint_region(ql.arch.regs.arch_sp - 64, b'\xff' * 8)
        os.dup2(saved, 1)
    finally:
        os.dup2(saved, 1)
        os.close(dn)
        os.close(saved)
    with pytest.raises(CheckpointError, match='taint already exists'):
        w.checkpoint()
