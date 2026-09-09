#!/usr/bin/env python3
"""
diagnose_worker.py
==================
Identify which worker_microtaint.py the benchmark is actually using.

The benchmark spawns a subprocess via:
    .venv_microtaint/bin/python worker_microtaint.py
which is NOT necessarily the worker_microtaint.py in your repo root.
The interpreter is `.venv_microtaint/bin/python` — that interpreter's
sys.path determines which `microtaint` package gets imported, and the
script `worker_microtaint.py` is resolved relative to whatever
directory the benchmark was launched from.

This script:
  1. Locates and prints the worker file the benchmark would actually run
  2. Prints the size and last-modified timestamp of that file
  3. Imports it as a module and inspects its `_REGS` to verify whether
     the soundness fix (16 GP regs) is in place
  4. Imports the engine.py the worker would actually load and verifies
     the `limit = i` temporal-ordering fix is present
  5. Sends the canonical rep-stosb test case (id 8009) through the worker
     and reports whether the output is sound or matches the
     un-fixed `0x7706aa7ab587952e` value

Run from your benchmark directory:
    .venv_microtaint/bin/python diagnose_worker.py
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def banner(s: str) -> None:
    print()
    print('=' * 70)
    print('  ' + s)
    print('=' * 70)


# ----------------------------------------------------------------------------
# 1. Locate the actual worker_microtaint.py
# ----------------------------------------------------------------------------

banner('1. Worker file location')

cwd = Path.cwd()
candidates = [
    cwd / 'worker_microtaint.py',
    cwd.parent / 'worker_microtaint.py',
]
worker_path: Path | None = None
for p in candidates:
    if p.is_file():
        worker_path = p
        break

if worker_path is None:
    print(f'ERROR: no worker_microtaint.py found in {cwd} or parent.')
    print('       This script must be run from the benchmark directory.')
    sys.exit(1)

stat = worker_path.stat()
print(f'  Path:          {worker_path}')
print(f'  Size:          {stat.st_size} bytes')
print(f'  Last modified: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))}')


# ----------------------------------------------------------------------------
# 2. Import the worker as a module and inspect _REGS
# ----------------------------------------------------------------------------

banner('2. Worker _REGS state_format')

spec = importlib.util.spec_from_file_location('worker_under_test', worker_path)
worker_mod = importlib.util.module_from_spec(spec)
sys.modules['worker_under_test'] = worker_mod
try:
    spec.loader.exec_module(worker_mod)
except Exception as exc:
    print(f'ERROR: failed to load worker: {exc}')
    sys.exit(2)

regs = getattr(worker_mod, '_REGS', None)
if regs is None:
    print('ERROR: worker has no module-level _REGS attribute.')
    sys.exit(2)

reg_names = [r.name for r in regs]
print(f'  _REGS contains {len(reg_names)} registers')
gp_regs = [n for n in reg_names if not n.startswith('XMM')]
xmm_regs = [n for n in reg_names if n.startswith('XMM')]
print(f'  GP regs ({len(gp_regs)}):  {gp_regs}')
print(f'  XMM regs ({len(xmm_regs)}): present' if xmm_regs else '  XMM regs: MISSING')

required = {'RAX', 'RBX', 'RCX', 'RDX', 'RSI', 'RDI', 'RSP', 'RBP',
            'R8', 'R9', 'R10', 'R11', 'R12', 'R13', 'R14', 'R15'}
missing = required - set(gp_regs)
if missing:
    print()
    print(f'  ** SOUNDNESS FIX MISSING **')
    print(f'  Missing GP regs: {sorted(missing)}')
    print(f'  Without these, the engine cannot resolve stack-relative LOAD/STORE')
    print(f'  pointers, and falls back to garbage Unicorn memory on rep stosb.')
    soundness_fix_present_in_worker = False
else:
    print(f'  OK: all 16 GP registers present in _REGS.')
    soundness_fix_present_in_worker = True


# ----------------------------------------------------------------------------
# 3. Locate and check engine.py the worker imports
# ----------------------------------------------------------------------------

banner('3. engine.py temporal-ordering fix')

import microtaint.sleigh.engine as engine_mod  # noqa: E402

print(f'  Path: {engine_mod.__file__}')
src = inspect.getsource(engine_mod)
has_temporal = 'limit = i' in src
print(f'  Has "limit = i" marker: {has_temporal}')
if not has_temporal:
    print('  ** ENGINE FIX MISSING **')
    print('  resolve_ptr_with_offset is the pre-fix version — STORE addresses')
    print('  for rep stosb resolve to V_RDI+1 instead of V_RDI.')

engine_fix_present = has_temporal


# ----------------------------------------------------------------------------
# 4. Send canonical test case through the actual worker process
# ----------------------------------------------------------------------------

banner('4. End-to-end test through worker subprocess')

# Find the same Python interpreter the benchmark would use.  The
# benchmark.py spec is:
#   "microtaint": ".venv_microtaint/bin/python worker_microtaint.py"
# so we invoke that interpreter explicitly.
benchmark_path = cwd / 'benchmark.py'
worker_cmd = None
if benchmark_path.is_file():
    try:
        bm_src = benchmark_path.read_text()
        for line in bm_src.splitlines():
            if 'microtaint' in line and 'worker_microtaint.py' in line and ':' in line:
                # e.g. "    \"microtaint\": \".venv_microtaint/bin/python worker_microtaint.py\","
                rhs = line.split(':', 1)[1].strip().strip(',').strip('"').strip("'")
                worker_cmd = rhs.split()
                break
    except Exception:
        pass

if worker_cmd is None:
    print('  Could not parse benchmark.py for worker command; using sys.executable.')
    worker_cmd = [sys.executable, str(worker_path)]
else:
    # Fix relative path — interpret it from cwd.
    worker_cmd = [str(cwd / worker_cmd[0])] + worker_cmd[1:]
    if worker_cmd[1] == 'worker_microtaint.py':
        worker_cmd[1] = str(worker_path)

print(f'  Spawning: {" ".join(worker_cmd)}')
if not Path(worker_cmd[0]).is_file():
    print(f'  ERROR: interpreter {worker_cmd[0]} not found.')
    sys.exit(2)

# Test case: rep stosb id 8009 from report
test_case = {
    'arch': 'x86_64',
    'bytes': 'fc48894424c0488d7c24c04889d8b908000000f3aa488b4424c0',
    'state': {'RAX': 10940498380929573403, 'RBX': 1830928842394036844,
              'RCX': 19, 'RDX': 5767559093351484470},
    'taint': {'RAX': 4352, 'RBX': 2, 'RCX': 0, 'RDX': 8796093022208},
}

proc = subprocess.Popen(
    worker_cmd,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd=str(cwd),
)
ready_line = proc.stdout.readline().decode().strip()
if ready_line != 'READY':
    print(f'  ERROR: worker did not say READY (got: {ready_line!r})')
    err = proc.stderr.read().decode()
    if err:
        print(f'  stderr: {err[:500]}')
    proc.kill()
    sys.exit(2)

proc.stdin.write((json.dumps(test_case) + '\n').encode())
proc.stdin.flush()
result_line = proc.stdout.readline().decode().strip()
proc.stdin.write(b'QUIT\n')
proc.stdin.flush()
try:
    proc.wait(timeout=5)
except subprocess.TimeoutExpired:
    proc.kill()

result = json.loads(result_line)
mt_rax = result.get('output_taint', {}).get('RAX', 0)
print(f'  Worker output for id 8009: RAX = 0x{mt_rax:016x}')
print(f'  Expected (sound):           RAX = 0x0202020202021302')
print(f'  Un-fixed (broken):          RAX = 0x7706aa7ab587952e')

is_sound = (mt_rax == 0x0202020202021302)
is_broken = (mt_rax == 0x7706aa7ab587952e)


# ----------------------------------------------------------------------------
# 5. Verdict
# ----------------------------------------------------------------------------

banner('VERDICT')

if is_sound and soundness_fix_present_in_worker and engine_fix_present:
    print('  PASS: Both fixes are in place and the worker produces sound output.')
    print('        Your benchmark run should now show 0 unsound on rep stosb.')
    print()
    print('  If your most recent benchmark report still shows the broken output,')
    print('  the most likely causes are:')
    print('    (a) The report was generated BEFORE you applied the fix.')
    print('    (b) A persistent worker process kept running with the old code')
    print('        in memory.  Kill any running benchmark.py and worker')
    print('        processes, then re-run.')
    print('    (c) Stale __pycache__ entries somewhere.  Find and remove:')
    print('          find . -name __pycache__ -type d -exec rm -rf {} +')
    sys.exit(0)
elif is_broken:
    print('  FAIL: Worker still produces the broken un-fixed output')
    print('        RAX = 0x7706aa7ab587952e.')
    print()
    print('  Diagnostic flags:')
    print(f'    Worker has all 16 GP regs in _REGS:    {soundness_fix_present_in_worker}')
    print(f'    engine.py has temporal-ordering fix:    {engine_fix_present}')
    print()
    if not soundness_fix_present_in_worker:
        print('  ** The worker file is missing the _REGS soundness fix. **')
        print(f'  Edit {worker_path} and ensure _REGS includes:')
        print('    RSI, RDI, RSP, RBP, R8, R9, R10, R11, R12, R13, R14, R15')
        print('  in addition to RAX, RBX, RCX, RDX and the XMM lanes.')
    if not engine_fix_present:
        print('  ** The engine.py at the path printed above is missing the fix. **')
        print(f'  The worker imported engine.py from: {engine_mod.__file__}')
        print('  Replace that file with the fixed version.  Note that this is')
        print('  the path inside the WORKER\'s venv site-packages — different')
        print('  from your source-tree microtaint/sleigh/engine.py.')
    sys.exit(1)
else:
    print(f'  UNKNOWN: Worker produced unexpected output 0x{mt_rax:016x}.')
    print('  This is neither the known-sound nor the known-broken value.')
    print(f'  Worker has all 16 GP regs in _REGS: {soundness_fix_present_in_worker}')
    print(f'  engine.py has temporal-ordering fix: {engine_fix_present}')
    sys.exit(2)
