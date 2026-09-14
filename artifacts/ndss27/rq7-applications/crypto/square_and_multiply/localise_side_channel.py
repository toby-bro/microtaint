#!/usr/bin/env python3
"""
localise_side_channel.py -- demonstrates (not asserts) the constant-time
LOCALISATION claim of the paper's Table "Constant-time leak attribution".

check_side_channel.py already shows the binary verdict (microtaint fires a
side-channel on pow_branch, none on pow_ct). This harness shows the finer,
bit-precise claim: if a SINGLE exponent bit k is the secret, microtaint's
--check-sc fires at EXACTLY step k+1 of the square-and-multiply ladder, naming
the leaking key bit (bit 0 -> step 1, ..., bit 31 -> step 32).

A byte- or register-granular engine cannot taint a single exponent bit; tainting
"the secret" taints the whole exponent, so every one of the 32 loop iterations
reads a tainted bit and its branch is flagged (all 32 steps, no localisation).
That whole-exponent behaviour is measured for the baselines in
../../applications_multitool (libdft64/TaintGrind/Triton: 32 tainted branches).

Here we drive the real pow_branch binary under microtaint + Qiling, taint one
input bit at a time (which byte-granular sources cannot express), and confirm
the fired step. Run:
    env -u VIRTUAL_ENV uv run --project <engine> python localise_side_channel.py
"""

# Experiment script, not library code: see artifacts/ndss27/README.md,
# "Lint and type checking", for why annotations are not required here.
# mypy: disable-error-code="no-untyped-def, no-untyped-call, type-arg"

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from qiling import Qiling
from qiling.const import QL_VERBOSE

from microtaint.emulator.reporter import Reporter
from microtaint.emulator.wrapper import MicrotaintWrapper

SCRIPT_DIR = Path(__file__).resolve().parent
GCC = ['gcc', '-O0', '-g', '-static', '-no-pie', '-fno-stack-protector']


class _Stdin:
    def __init__(self, data: bytes):
        self._buf = bytearray(data)

    def read(self, count: int) -> bytes:
        chunk = bytes(self._buf[:count])
        del self._buf[:count]
        return chunk


def _find_key_branch(binary: Path) -> int:
    """Return the PC of pow_branch's key-dependent branch: the `je` right after
    `and $0x1,%eax ; test %eax,%eax` (the `if (e & 1)`)."""
    out = subprocess.check_output(['objdump', '-d', '--no-show-raw-insn', str(binary)], text=True)
    lines, in_fn = [], False
    for ln in out.splitlines():
        if '<pow_branch>:' in ln:
            in_fn = True
        elif in_fn and re.match(r'^[0-9a-f]+ <\w+>:', ln.strip()):
            break
        if in_fn:
            lines.append(ln)
    for i, ln in enumerate(lines):
        if 'and    $0x1,%eax' in ln and i + 2 < len(lines) and 'test' in lines[i + 1]:
            m = re.match(r'\s*([0-9a-f]+):\s*je', lines[i + 2])
            if m:
                return int(m.group(1), 16)
    raise RuntimeError('could not locate pow_branch key branch')


def fired_step(binary: Path, branch_pc: int, k: int) -> int | None:
    """Taint ONLY exponent bit k; return the loop step (1-indexed) at which
    --check-sc first fires, or None."""
    ql = Qiling([str(binary), 'vuln'], '/', verbose=QL_VERBOSE.OFF)
    ql.os.stdin = _Stdin(b'\xff\xff\xff\xff')  # value irrelevant; the taint mask decides
    w = MicrotaintWrapper(ql, check_bof=False, check_uaf=False, check_sc=True, check_aiw=False, reporter=Reporter())
    state = {'off': 0}

    def selective(address: int, n: int) -> None:  # taint exactly bit k of the 4-byte exponent
        for i in range(n):
            g = state['off'] + i
            mask = bytes([1 << (k % 8)]) if g == (k // 8) else b'\x00'
            w.taint_region(address + i, mask)
        state['off'] += n

    w._taint_bytes = selective  # type: ignore[method-assign]

    cnt = {'n': 0}
    fired: dict[str, int | None] = {'at': None}
    ql.hook_address(lambda ql: cnt.__setitem__('n', cnt['n'] + 1), branch_pc)
    orig = w.reporter.side_channel

    def cap(address, instruction='', taint_mask=0):
        if fired['at'] is None:
            fired['at'] = cnt['n']
        return orig(address, instruction=instruction, taint_mask=taint_mask)

    w.reporter.side_channel = cap  # type: ignore[method-assign]
    try:
        ql.run()
    except Exception:
        pass
    return fired['at']


def _engine_provenance() -> dict:
    """Which engine produced this result: commit, version, dirty flag.

    Never raises: an installed wheel has no git repository, and that is not a
    reason for the experiment to stop.
    """
    try:
        from microtaint.provenance import engine_provenance
        return engine_provenance()
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--json', metavar='PATH',
                    help='also write the per-bit localisation result as JSON, so '
                         'the 32/32 claim can be processed rather than only read')
    args = ap.parse_args()

    src = SCRIPT_DIR / 'test_constant_time.c'
    build = Path(tempfile.mkdtemp())
    binary = build / 'ct'
    try:
        r = subprocess.run([*GCC, '-o', str(binary), str(src)], capture_output=True)
        if r.returncode != 0:
            print('FAIL build:\n' + r.stderr.decode(), file=sys.stderr)
            return 1
        branch_pc = _find_key_branch(binary)
        print(f'pow_branch key branch at {branch_pc:#x}; tainting one exponent bit at a time')
        ok = True
        bits = []
        for k in range(32):
            step = fired_step(binary, branch_pc, k)
            good = step == k + 1
            ok = ok and good
            bits.append({
                'secret_bit': k,
                'fired_step': step,
                'expected_step': k + 1,
                'ok': bool(good),
            })
            print(
                f'  secret bit {k:2d} -> side-channel at step {step}  (expect {k + 1})  {"OK" if good else "MISMATCH"}',
            )
        if args.json:
            n_exact = sum(1 for b in bits if b['ok'])
            with open(args.json, 'w') as fh:
                json.dump({
                    'experiment': 'side-channel-localisation',
                    'engine': _engine_provenance(),
                    'binary': 'pow_branch',
                    'branch_pc': branch_pc,
                    'branch_pc_hex': f'{branch_pc:#x}',
                    'n_bits': len(bits),
                    'n_exact': n_exact,
                    'localisation': f'{n_exact}/{len(bits)}',
                    'passed': bool(ok),
                    'bits': bits,
                }, fh, indent=2)
                fh.write('\n')
            print(f'[json] {args.json}')
        print(
            'PASS: microtaint localises every single-bit secret to its exact step'
            if ok
            else 'FAIL: localisation mismatch',
        )
        return 0 if ok else 1
    finally:
        import shutil

        shutil.rmtree(build, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
