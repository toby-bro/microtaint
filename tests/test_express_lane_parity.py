# ruff: noqa: S603, S607
"""The GIL-free paths must produce EXACTLY the state the GIL paths do.

`mt_fast_step_nogil` answers an instruction without acquiring the GIL, from
constants cached on the address entry: the resolved untainted-input prefilter
and the compiled taint program.  That makes it a second implementation of
decisions the full path also makes, so the thing to check is not that it finds
bugs but that the two agree completely -- every register's taint, every shadow
byte, and every finding.

The memory callbacks got the same treatment (memhook.h): Unicorn calls them
once per guest load and store, and their usual answers are arithmetic over the
shadow's page table.  MICROTAINT_NOGIL_MEM=0 puts them back on the GIL path, so
the same comparison covers them.

The comparison is run in subprocesses because both flags are read when the
extension is imported, and it is repeated with the compiled-program path off and
on, because three of the four express shapes only exist when it is on.

Both directions of the count assertion matter.  Without them a change that made
the lane decline everything would leave this test passing while measuring
nothing, which is the failure mode a parity test is most prone to.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile

import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != 'Linux', reason='emulator tests require Linux',
)

# Reads attacker input, copies it around, does arithmetic on it and stores it
# back: register taint, memory taint, branches and a stack overflow, so all four
# express shapes (prefilter, register program, PC program, memory program) are
# exercised in one run.
_SRC = r"""
long sys_read(int fd, void *buf, unsigned long count){long r;
  __asm__ volatile("syscall":"=a"(r):"0"(0),"D"(fd),"S"(buf),"d"(count):"rcx","r11","memory");return r;}
long sys_exit(int status){long r;
  __asm__ volatile("syscall":"=a"(r):"0"(60),"D"(status):"rcx","r11","memory");return r;}
static char out[64];
void work(void){
  char buf[16];
  sys_read(0, buf, 48);
  unsigned long acc = 0;
  for (int i = 0; i < 32; i++) {
    unsigned char c = (unsigned char)buf[i & 15];
    acc = (acc << 3) ^ (acc >> 5) ^ (c + i);
    if (c & 1) acc += c * 7u; else acc -= c ^ 0x5au;
    out[i & 63] = (char)(acc & 0xff);
  }
  out[63] = (char)acc;
}
void _start(void){ work(); sys_exit(0); }
"""

_CHILD = r"""
import hashlib, io, json, os, sys
from qiling import Qiling
from qiling.const import QL_VERBOSE
from microtaint.emulator.reporter import Reporter
from microtaint.emulator.wrapper import MicrotaintWrapper

binary, payload_hex = sys.argv[1], sys.argv[2]
stream = io.StringIO()
ql = Qiling([binary], "/", verbose=QL_VERBOSE.OFF)
ql.os.stdin = io.BytesIO(bytes.fromhex(payload_hex))
reporter = Reporter(json_mode=False, stream=stream)
w = MicrotaintWrapper(ql, check_bof=True, check_uaf=False, check_sc=False,
                      reporter=reporter)
saved, devnull = os.dup(1), os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull, 1)
try:
    ql.run()
except Exception:
    pass
os.dup2(saved, 1)
reporter.finalize()

hook = w._instr_hook_obj
regs = {}
if hook is not None:
    hook.sync_taint_to_dict()
    regs = {k: int(v) for k, v in hook.register_taint.items() if int(v)}

# Every shadow byte in every mapped region, hashed.  This is the whole memory
# taint state, not a sample of it.
h = hashlib.sha256()
sm = w.shadow_mem
for entry in sorted(ql.mem.map_info, key=lambda e: e[0]):
    lo, hi = entry[0], entry[1]
    if hi - lo > (8 << 20):
        continue
    h.update(b"%016x" % lo)
    h.update(bytes(sm.read_bytes(lo, hi - lo)))

print("@@RESULT@@" + json.dumps({
    "findings": sorted(set(stream.getvalue().splitlines())),
    "regs": {k: v for k, v in sorted(regs.items())},
    "shadow": h.hexdigest(),
    "instr_total": int(getattr(hook, "instr_total", 0) or 0),
    "express_done": int(getattr(hook, "express_done", 0) or 0),
    "fast_done": int(getattr(hook, "fast_done", 0) or 0),
}))
"""


def _compile(src: str) -> str:
    fd, path = tempfile.mkstemp(suffix='.elf')
    os.close(fd)
    subprocess.run(
        ['gcc', '-nostdlib', '-O1', '-fno-stack-protector', '-static',
         '-o', path, '-x', 'c', '-'],
        input=src.encode(), check=True)
    return path


def _run(binary: str, payload: bytes, *, express: bool, taint_ir: bool,
         nogil_mem: bool = True) -> dict[str, int]:
    env = dict(os.environ)
    env['MICROTAINT_EXPRESS'] = '1' if express else '0'
    env['MICROTAINT_NOGIL_MEM'] = '1' if nogil_mem else '0'
    env['MICROTAINT_TAINT_IR'] = '1' if taint_ir else '0'
    out = subprocess.run([sys.executable, '-c', _CHILD, binary, payload.hex()],
                         capture_output=True, text=True, env=env, check=True,
                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for line in out.stdout.splitlines():
        if line.startswith('@@RESULT@@'):
            return json.loads(line[len('@@RESULT@@'):])
    raise AssertionError(f'child produced no result:\n{out.stdout}\n{out.stderr}')


@pytest.fixture(scope='module')
def binary():
    return _compile(_SRC)


@pytest.mark.parametrize('taint_ir', [False, True], ids=['circuit', 'compiled'])
def test_express_lane_matches_the_gil_path(binary, taint_ir):
    payload = bytes((i * 37 + 11) & 0xFF for i in range(48))
    on = _run(binary, payload, express=True, taint_ir=taint_ir)
    off = _run(binary, payload, express=False, taint_ir=taint_ir)

    assert off['express_done'] == 0, \
        'MICROTAINT_EXPRESS=0 must leave every instruction on the GIL path'
    assert on['express_done'] > 0, \
        'the express lane never engaged, so this run compared nothing'

    assert on['instr_total'] == off['instr_total'], 'the runs diverged'
    assert on['regs'] == off['regs'], 'register taint differs'
    assert on['shadow'] == off['shadow'], 'memory taint differs'
    assert on['findings'] == off['findings'], 'findings differ'


@pytest.mark.parametrize('taint_ir', [False, True], ids=['circuit', 'compiled'])
def test_nogil_memory_callbacks_match_the_gil_path(binary, taint_ir):
    """Same comparison for the load/store callbacks, which own the shadow.

    Every byte of memory taint is in the digest, so a callback that cleared a
    byte the GIL path kept -- an under-taint, the one failure that is never
    acceptable -- cannot pass this.
    """
    payload = bytes((i * 37 + 11) & 0xFF for i in range(48))
    on = _run(binary, payload, express=True, taint_ir=taint_ir, nogil_mem=True)
    off = _run(binary, payload, express=True, taint_ir=taint_ir, nogil_mem=False)

    assert on['instr_total'] == off['instr_total'], 'the runs diverged'
    assert on['shadow'] == off['shadow'], 'memory taint differs'
    assert on['regs'] == off['regs'], 'register taint differs'
    assert on['findings'] == off['findings'], 'findings differ'


@pytest.mark.parametrize('taint_ir', [False, True], ids=['circuit', 'compiled'])
def test_express_lane_carries_most_of_the_run(binary, taint_ir):
    """A coverage floor, so a change that quietly disabled the lane is visible.

    Deliberately loose: what it must catch is the lane falling to nothing, not
    a few points of movement as the shapes it accepts change.
    """
    payload = bytes((i * 37 + 11) & 0xFF for i in range(48))
    on = _run(binary, payload, express=True, taint_ir=taint_ir)
    share = on['express_done'] / max(1, on['instr_total'])
    assert share > 0.20, f'express lane covered only {share:.1%} of the run'
