#!/usr/bin/env python3
"""Re-check every unsound case a run reports, with a fresh oracle.

    verify_unsound.py REPORT.json [--tool microtaint] [--json OUT.json]

Why this exists: the ground truth is 2^k Unicorn executions, and on a long batch
Unicorn occasionally returns a register that no assignment could have produced.
Measured on a 9,858-case run at seed 1: 22 cases came back unsound, and every one
of them was the oracle.  One of the 22 was `and rax, 0` -- an instruction whose
output cannot depend on any input -- with the oracle claiming 32 tainted bits in
RAX.  Re-running that case on its own gave the correct empty mask five times out
of five, so the fault does not reproduce outside the batch.

That makes a single run's unsound count an upper bound, not a result.  This is
the same two-pass discipline rq6-generalisation already uses: pass 1 is a wide
net that can only over-report, pass 2 re-checks each report in isolation, and a
finding is only a finding if it survives pass 2.

Re-checking with the SAME oracle is not enough, because some of its errors are
deterministic.  On one case -- `imul rax,rcx; add rbx,rax; shr rbx,1; xor
rcx,rbx; add rdx,rcx` -- the batch oracle claimed RBX bit 63 moves, and so did a
fresh run of it.  It cannot: RBX is last written by a logical shift right, so its
top bit is 0 whatever the inputs.  The difference is that the fast oracle drives
2^k emulations through ONE reused Unicorn instance (carefully reset each time)
while this one builds a new instance per emulation, and on that case the reused
instance reported bits across four registers that no assignment produces.

So each candidate is re-scored against an ISOLATED oracle (a fresh Unicorn per
emulation) and a freshly started engine worker.  Verdicts:

  CONFIRMED       the isolated oracle still demands bits the fresh engine misses
  ORACLE ARTIFACT the isolated oracle does not demand them
  ENGINE FLAKE    the oracle agrees with the batch, the fresh engine now covers it

Isolation costs about 3 ms per emulation, so a k=15 candidate takes ~100 s.  That
is affordable because under-taints are rare; it would not be as the main scoring
path, which is why the fast oracle stays where it is.
"""

# Experiment script, not library code: see artifacts/ndss27/README.md,
# "Lint and type checking", for why annotations are not required here.
# mypy: disable-error-code="no-untyped-def, no-untyped-call, type-arg"

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

REGS = ('RAX', 'RBX', 'RCX', 'RDX')

# Runs one case in a clean interpreter, one Unicorn instance per emulation.
_CHILD = r"""
import json, sys
import unicorn
import unicorn.x86_const as ux

REGS = ('RAX', 'RBX', 'RCX', 'RDX')
IDS = {r: getattr(ux, 'UC_X86_REG_' + r) for r in REGS}
MASK64 = (1 << 64) - 1
CODE_BASE, STACK_BASE = 0x1000, 0x100000

def run(code, vals):
    uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
    uc.mem_map(CODE_BASE, 0x10000)
    uc.mem_map(STACK_BASE, 0x10000)
    uc.mem_write(CODE_BASE, code)
    for r, i in IDS.items():
        uc.reg_write(i, vals[r])
    uc.reg_write(ux.UC_X86_REG_RSP, STACK_BASE + 0x8000)
    uc.reg_write(ux.UC_X86_REG_EFLAGS, 0x2)
    uc.emu_start(CODE_BASE, CODE_BASE + len(code))
    return {r: uc.reg_read(i) & MASK64 for r, i in IDS.items()}

tc = json.loads(sys.argv[1])
code = bytes.fromhex(tc['bytes'])
taint = {r: int(tc['taint'].get(r, 0)) for r in REGS}
state = {r: int(tc['state'].get(r, 0)) for r in REGS}
base = {r: state[r] & ~taint[r] & MASK64 for r in REGS}
pos = [(r, b) for r in REGS for b in range(64) if (taint[r] >> b) & 1]

outs = []
for assign in range(1 << len(pos)):
    vals = dict(base)
    for idx, (r, b) in enumerate(pos):
        if (assign >> idx) & 1:
            vals[r] = (vals[r] | (1 << b)) & MASK64
    try:
        outs.append(run(code, vals))
    except Exception:
        pass          # a trap is not a defined output; drop it, as the harness does
mask = {}
for r in REGS:
    if not outs:
        mask[r] = 0
        continue
    o = a = outs[0][r]
    for x in outs[1:]:
        o |= x[r]
        a &= x[r]
    mask[r] = (o ^ a) & MASK64
print(json.dumps(mask))
"""


def isolated_ground_truth(tc: dict, timeout: float = 1800.0) -> dict[str, int] | None:
    """The oracle's answer for one case, one fresh Unicorn per emulation."""
    try:
        proc = subprocess.run(
            [sys.executable, '-c', _CHILD, json.dumps(tc)],
            capture_output=True, text=True, check=False, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    parsed: dict[str, int] = json.loads(proc.stdout.strip().splitlines()[-1])
    return parsed


def fresh_engine(cases: list[dict], worker: str) -> list[dict[str, int]]:
    """The engine's answer for each case, from one freshly started worker."""
    proc = subprocess.Popen(
        [worker, str(HERE / 'worker_microtaint.py')],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    if proc.stdout.readline().strip() != 'READY':
        proc.kill()
        raise RuntimeError('worker did not start')
    out = []
    for tc in cases:
        proc.stdin.write(json.dumps(tc) + '\n')
        proc.stdin.flush()
        out.append(json.loads(proc.stdout.readline()).get('output_taint') or {})
    proc.stdin.write('QUIT\n')
    proc.stdin.flush()
    proc.wait(timeout=10)
    return out


def under(truth: dict[str, int], got: dict[str, int]) -> dict[str, int]:
    return {r: t & ~got.get(r, 0) for r, t in truth.items() if t & ~got.get(r, 0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('report')
    ap.add_argument('--tool', default='microtaint')
    ap.add_argument('--worker', default=str(HERE / '.venv_microtaint' / 'bin' / 'python'))
    ap.add_argument('--json', dest='out_json', default=None)
    args = ap.parse_args()

    report = json.loads(Path(args.report).read_text())
    candidates = []
    for r in report['results']:
        truth = (r['tool_results'].get('ground_truth') or {}).get('output_taint') or {}
        got = (r['tool_results'].get(args.tool) or {}).get('output_taint') or {}
        miss = under(truth, got)
        if miss:
            candidates.append((r, miss))

    print(f'{len(candidates)} case(s) reported unsound for {args.tool}')
    if not candidates:
        return 0

    cases = [{'arch': 'x86_64', 'bytes': r['instruction']['bytes'],
              'state': r['instruction']['state'], 'taint': r['instruction']['taint']}
             for r, _ in candidates]
    engine_now = fresh_engine(cases, args.worker)

    rows: list[dict[str, Any]] = []
    confirmed = 0
    for (r, batch_miss), tc, got_now in zip(candidates, cases, engine_now, strict=True):
        truth_now = isolated_ground_truth(tc)
        if truth_now is None:
            verdict, miss_now = 'ORACLE FAILED', {}
        else:
            miss_now = under(truth_now, got_now)
            if miss_now:
                verdict = 'CONFIRMED'
                confirmed += 1
            elif under(truth_now, (r['tool_results'].get(args.tool) or {}).get('output_taint') or {}):
                verdict = 'ENGINE FLAKE'
            else:
                verdict = 'ORACLE ARTIFACT'
        rows.append({'id': r['id'], 'category': r['instruction'].get('category'),
                     'asm': r['instruction'].get('assembly'), 'bytes': r['instruction']['bytes'],
                     'verdict': verdict,
                     'batch_under': {k: hex(v) for k, v in batch_miss.items()},
                     'fresh_under': {k: hex(v) for k, v in miss_now.items()}})
        print(f"  {verdict:16s} id={r['id']:<6d} {str(r['instruction'].get('assembly'))[:52]:52s} "
              f"batch={ {k: hex(v) for k, v in batch_miss.items()} }")

    print(f'\nconfirmed under-taints: {confirmed} of {len(candidates)}')
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(rows, indent=1))
        print(f'wrote {args.out_json}')
    return 1 if confirmed else 0


if __name__ == '__main__':
    sys.exit(main())
