#!/usr/bin/env python3
"""
Eval: square-and-multiply SC detection.

Compiles test_constant_time.c and checks:
  - pow_branch (naive)   -> microtaint fires >= 1 SC finding
  - pow_ct (mask-select) -> microtaint fires 0 SC findings
"""

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
GCC = ['gcc', '-O0', '-g', '-static', '-no-pie', '-fno-stack-protector']


def die(msg: str) -> None:
    print(f'FAIL: {msg}', file=sys.stderr)
    sys.exit(1)


def run_mt(binary: Path, variant: str, input_bytes: bytes) -> int:
    fd, inp = tempfile.mkstemp()
    try:
        os.write(fd, input_bytes)
        os.close(fd)
        cmd = ['microtaint', '--check-sc', '--json', '--quiet', '--input', inp, '--', str(binary), variant]
        r = subprocess.run(cmd, capture_output=True, timeout=60)  # noqa: PLW1510, S603
        text = r.stdout.decode('ascii', errors='replace')
        # The target binary may write to stdout before microtaint appends JSON;
        # find the JSON blob by locating the first '{'.
        idx = text.find('{')
        if idx == -1:
            return 0
        try:
            data = json.loads(text[idx:])
        except json.JSONDecodeError:
            return 0
        return data.get('summary', {}).get('side_channel', 0)  # type: ignore[no-any-return]
    finally:
        os.unlink(inp)


def main() -> None:
    src = SCRIPT_DIR / 'test_constant_time.c'
    build = Path(tempfile.mkdtemp())
    binary = build / 'test_constant_time'

    try:
        r = subprocess.run([*GCC, '-o', str(binary), str(src)], capture_output=True)  # noqa: PLW1510, S603
        if r.returncode != 0:
            die('build failed:\n' + r.stderr.decode())

        exponent = struct.pack('<I', 5)  # secret; base/modulus are compile-time constants

        n_vuln = run_mt(binary, 'vuln', exponent)
        n_ct = run_mt(binary, 'ct', exponent)

        ok = True
        if n_vuln >= 1:
            print(f'PASS  pow_branch: {n_vuln} SC finding(s)')
        else:
            print('FAIL  pow_branch: expected >= 1 SC finding, got 0')
            ok = False

        if n_ct == 0:
            print('PASS  pow_ct: 0 SC findings')
        else:
            print(f'FAIL  pow_ct: expected 0 SC findings, got {n_ct}')
            ok = False

        sys.exit(0 if ok else 1)
    finally:
        shutil.rmtree(build, ignore_errors=True)


if __name__ == '__main__':
    main()
