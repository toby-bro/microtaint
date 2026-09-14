#!/usr/bin/env python3
"""
Eval: square-and-multiply SC detection.

Compiles test_constant_time.c and checks:
  - pow_branch (naive)   -> microtaint fires >= 1 SC finding
  - pow_ct (mask-select) -> microtaint fires 0 SC findings
"""

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
GCC = ['gcc', '-O0', '-g', '-static', '-no-pie', '-fno-stack-protector']


def _microtaint_cmd() -> list[str]:
    """Locate the microtaint CLI.

    The bare name `microtaint` only resolves when the venv's bin directory is on
    PATH.  That is true under `uv run`, which is how the artifact drivers invoke
    this, but not when a reviewer runs the script with an explicit interpreter,
    and the failure was a bare FileNotFoundError naming only 'microtaint'.
    Prefer an explicit override, then the console script installed next to the
    running interpreter, then PATH, and fall back to `-m` (cli.py has a
    __main__ guard) so there is always a way through.
    """
    override = os.environ.get('MICROTAINT_CLI')
    if override:
        return [override]
    sibling = Path(sys.executable).parent / 'microtaint'
    if sibling.exists():
        return [str(sibling)]
    found = shutil.which('microtaint')
    if found:
        return [found]
    return [sys.executable, '-m', 'microtaint.emulator.cli']


def die(msg: str) -> None:
    print(f'FAIL: {msg}', file=sys.stderr)
    sys.exit(1)


def run_mt(binary: Path, variant: str, input_bytes: bytes) -> int:
    fd, inp = tempfile.mkstemp()
    try:
        os.write(fd, input_bytes)
        os.close(fd)
        cmd = [*_microtaint_cmd(), '--check-sc', '--json', '--quiet', '--input', inp, '--', str(binary), variant]
        r = subprocess.run(cmd, capture_output=True, timeout=60)
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


def _engine_provenance() -> Mapping[str, object]:
    """Which engine produced this result: commit, version, dirty flag.

    Never raises: an installed wheel has no git repository, and that is not a
    reason for the experiment to stop.
    """
    try:
        from microtaint.provenance import engine_provenance
        return engine_provenance()
    except Exception:
        return {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--json', metavar='PATH',
                    help='also write the two-sided verdict as JSON, so the result '
                         'can be processed rather than only read')
    args = ap.parse_args()

    src = SCRIPT_DIR / 'test_constant_time.c'
    build = Path(tempfile.mkdtemp())
    binary = build / 'test_constant_time'

    try:
        r = subprocess.run([*GCC, '-o', str(binary), str(src)], capture_output=True)
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

        if args.json:
            with open(args.json, 'w') as fh:
                json.dump({
                    'experiment': 'constant-time-side-channel',
                    'engine': _engine_provenance(),
                    'passed': bool(ok),
                    'variants': {
                        # Two-sided on purpose: a detector that fires on
                        # everything would pass the first check and fail this one.
                        'pow_branch': {'sc_findings': n_vuln,
                                       'expected': '>=1', 'ok': n_vuln >= 1},
                        'pow_ct': {'sc_findings': n_ct,
                                   'expected': '0', 'ok': n_ct == 0},
                    },
                }, fh, indent=2)
                fh.write('\n')
            print(f'[json] {args.json}')

        sys.exit(0 if ok else 1)
    finally:
        shutil.rmtree(build, ignore_errors=True)


if __name__ == '__main__':
    main()
