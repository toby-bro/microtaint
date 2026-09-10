# ruff: noqa: PLW1510
"""The array-native taint path must not under-taint against the dict path.

Register taint lives in slot-indexed C arrays (MICROTAINT_ARR_HOOK=1, the
default) or in a Python dict (=0).  They are two implementations of one
specification, and the dict path is the older of the two: it is what v0.6.10
ran, before the hot path moved into C.  So the dict path is the reference and
the array path is the thing under test.

Nothing compared them.  The release gate varies MICROTAINT_TAINT_IR, which
selects between the compiled taint program and the differential, and leaves this
axis alone -- so the DEFAULT path had no parity gate against the reference at
all, and a divergence in it could only be found by noticing a wrong answer
downstream.  One was: on a store through a tainted index the array path ends a
run holding LESS taint than the dict path.

What is asserted is the soundness direction, not equality.  Over-taint is a
precision cost the engine is allowed to pay and the two paths do reach it
differently; taint that the reference holds and the array path does not is an
under-taint, and there is no acceptable amount of that.  Precision differences
are reported in the failure message when a test fails for the real reason, so a
run that trades soundness for precision is still legible.

The guests are small and end in a syscall.  Each is run twice, in two
subprocesses, because MICROTAINT_ARR_HOOK is read once at import.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TypedDict

import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != 'Linux', reason='emulator tests require Linux',
)

_ROOT = Path(__file__).resolve().parent.parent

_SYSCALLS = r"""
long sys_read(int fd, void *buf, unsigned long count){long r;
  __asm__ volatile("syscall":"=a"(r):"0"(0),"D"(fd),"S"(buf),"d"(count):"rcx","r11","memory");return r;}
long sys_write(int fd, const void *buf, unsigned long count){long r;
  __asm__ volatile("syscall":"=a"(r):"0"(1),"D"(fd),"S"(buf),"d"(count):"rcx","r11","memory");return r;}
long sys_exit(int status){long r;
  __asm__ volatile("syscall":"=a"(r):"0"(60),"D"(status):"rcx","r11","memory");return r;}
"""

#: Each guest is one taint shape.  `scatter` is the one that found the
#: divergence: it both loads and stores through an index derived from the
#: secret, which is where the two paths resolve an address differently.
_GUESTS = {
    'arith': _SYSCALLS + r"""
void _start(void){
  unsigned char b[8]; sys_read(0, b, 8);
  unsigned long a = 0;
  for (int i = 0; i < 8; i++) { a = (a << 7) ^ (a + b[i]); a ^= a >> 11; }
  sys_write(1, &a, 8);
  sys_exit((int)(a & 1));
}
""",
    'load_store': _SYSCALLS + r"""
unsigned char table[64];
void _start(void){
  unsigned char b[8]; sys_read(0, b, 8);
  for (int i = 0; i < 64; i++) { table[i] = b[i & 7] ^ (unsigned char)i; }
  unsigned long a = 0;
  for (int i = 0; i < 64; i++) { a += table[i]; }
  sys_write(1, &a, 8);
  sys_exit((int)(a & 1));
}
""",
    'scatter': _SYSCALLS + r"""
unsigned char state[64];
void _start(void){
  unsigned char b[8]; sys_read(0, b, 8);
  for (int i = 0; i < 64; i++) { state[i] = b[i & 7] + (unsigned char)i; }
  /* index derived from the secret: tainted LOAD and tainted STORE addresses */
  for (int i = 0; i < 64; i++) {
      unsigned char idx = (unsigned char)(state[i] & 63);
      unsigned char tmp = state[i];
      state[i]   = state[idx];
      state[idx] = tmp;
  }
  unsigned long a = 0;
  for (int i = 0; i < 64; i++) { a = a * 31 + state[i]; }
  sys_write(1, &a, 8);
  sys_exit((int)(a & 1));
}
""",
    'flags_and_branch': _SYSCALLS + r"""
void _start(void){
  unsigned char b[4]; sys_read(0, b, 4);
  unsigned int e = b[0] | (b[1] << 8);
  int acc = 0;
  for (int i = 0; i < 12; i++) { acc += (e & 1) ? 3 : 5; e >>= 1; }
  sys_write(1, &acc, 4);
  sys_exit(acc & 1);
}
""",
}

_STDIN = bytes(range(1, 17))


def _build(source: str) -> str:
    fd, path = tempfile.mkstemp(suffix='.elf')
    os.close(fd)
    built = subprocess.run(
        ['gcc', '-nostdlib', '-static', '-no-pie', '-O1', '-fno-stack-protector',
         '-o', path, '-x', 'c', '-'],
        input=source.encode(), capture_output=True)
    if built.returncode != 0:
        os.unlink(path)
        pytest.skip(f'cannot build the guest: {built.stderr.decode()[:400]}')
    return path


class Probe(TypedDict):
    """One run of `tests.arr_hook_probe`.

    `findings` arrives as JSON lists, not the (kind, address) tuples the probe
    built, so it is compared and printed rather than indexed.  `mem` is keyed
    by the decimal address as a string, for the same reason: JSON object keys
    are strings.
    """

    engine: str
    arr_hook: str
    findings: list[list[object]]
    regs: dict[str, int]
    mem: dict[str, int]


def _probe(guest: str, arr_hook: str) -> Probe:
    env = {**os.environ, 'MICROTAINT_ARR_HOOK': arr_hook}
    run = subprocess.run(
        [sys.executable, '-m', 'tests.arr_hook_probe', guest, _STDIN.hex()],
        capture_output=True, cwd=str(_ROOT), env=env, timeout=900)
    if run.returncode != 0:
        pytest.fail(f'the probe failed with ARR_HOOK={arr_hook}: '
                    f'{run.stderr.decode()[-2000:]}')
    probe: Probe = json.loads(run.stdout.decode())
    return probe


def _under_taint(reference: dict[str, int],
                 candidate: dict[str, int]) -> dict[str, tuple[int, int]]:
    """Bits the reference holds that the candidate does not, per key."""
    lost: dict[str, tuple[int, int]] = {}
    for key, ref in reference.items():
        got = candidate.get(key, 0)
        if ref & ~got:
            lost[key] = (ref, got)
    return lost


#: The taint-density benchmarks, prebuilt and checked in.  They are the reason
#: this file exists: the divergence was found on bench_dense, whose scatter loop
#: stores through an index derived from a 64-bit tainted accumulator, and none of
#: the small guests above reproduce it.  A gate that only ran shapes it already
#: passes would prove nothing.
_BENCH_DIR = _ROOT / 'benchmark' / 'taint_density'
_BENCHES = ('bench_untainted', 'bench_sparse', 'bench_dense')


def _compare(guest: str, label: str) -> None:
    dict_path = _probe(guest, '0')
    array_path = _probe(guest, '1')

    assert dict_path['findings'] == array_path['findings'], (
        f'{label}: the two paths reported different findings\n'
        f'  dict  : {dict_path["findings"]}\n'
        f'  array : {array_path["findings"]}')

    lost_regs = _under_taint(dict_path['regs'], array_path['regs'])
    lost_mem = _under_taint(dict_path['mem'], array_path['mem'])
    if lost_regs or lost_mem:
        gained_mem = _under_taint(array_path['mem'], dict_path['mem'])
        pytest.fail(
            f'{label}: the array path lost taint the dict path holds.\n'
            f'  registers: '
            + ', '.join(f'{k} {ref:#x} -> {got:#x}' for k, (ref, got) in
                        sorted(lost_regs.items()))
            + f'\n  memory ({len(lost_mem)} words): '
            + ', '.join(f'{int(k):#x} {ref:#018x} -> {got:#018x}'
                        for k, (ref, got) in sorted(lost_mem.items())[:6])
            + f'\n  (for context, {len(gained_mem)} memory words are tainted only '
              f'on the array path, which is the harmless direction)')


#: bench_dense used to fail here under a strict xfail, and that marker is what
#: reported the fix: it XPASSed the moment the cause was removed.
#:
#: The cause was NOT the array path.  It was the compiled taint path lowering
#: every load under the 'concrete' pointer policy, so the S-box load at 0x401133
#: (`movzbl (%rax,%rcx,1),%eax`, a CLEAN public table at a TAINTED index) came
#: back clean and the store at 0x401137 wrote a clean byte.  The array path was
#: only where it happened to show, because the dict path reached the differential
#: for those instructions more often.  Fixed by making 'avalanche' the default
#: policy; see tests/test_tainted_pointer_load.py.
#:
#: The note that stood here said "it is not the tainted-address policy on its
#: own -- that same load in isolation is tainted on both paths".  That was
#: wrong: in isolation the compiled path answered 0x0 for it.


@pytest.mark.parametrize('name', _BENCHES)
def test_benchmarks_do_not_under_taint(name: str) -> None:
    guest = _BENCH_DIR / f'{name}.elf'
    if not guest.exists():
        pytest.skip(f'{guest} is not built')
    _compare(str(guest), name)


@pytest.mark.parametrize('name', sorted(_GUESTS))
def test_array_path_does_not_under_taint(name: str) -> None:
    guest = _build(_GUESTS[name])
    try:
        dict_path = _probe(guest, '0')
        array_path = _probe(guest, '1')
    finally:
        os.unlink(guest)

    assert dict_path['findings'] == array_path['findings'], (
        f'{name}: the two paths reported different findings\n'
        f'  dict  : {dict_path["findings"]}\n'
        f'  array : {array_path["findings"]}')

    lost_regs = _under_taint(dict_path['regs'], array_path['regs'])
    lost_mem = _under_taint(dict_path['mem'], array_path['mem'])
    if lost_regs or lost_mem:
        gained_mem = _under_taint(array_path['mem'], dict_path['mem'])
        pytest.fail(
            f'{name}: the array path lost taint the dict path holds.\n'
            f'  registers: '
            + ', '.join(f'{k} {ref:#x} -> {got:#x}' for k, (ref, got) in
                        sorted(lost_regs.items()))
            + f'\n  memory ({len(lost_mem)} words): '
            + ', '.join(f'{int(k):#x} {ref:#018x} -> {got:#018x}'
                        for k, (ref, got) in sorted(lost_mem.items())[:6])
            + f'\n  (for context, {len(gained_mem)} memory words are tainted only '
              f'on the array path, which is the harmless direction)')
