#!/usr/bin/env python3
"""Soundness fuzzer for MicroTaint.

Runs MicroTaint against a linear single-bit-flip noninterference oracle on a
large stream of RANDOMLY generated instruction test cases, to strengthen the
empirical soundness guarantee well beyond the fixed 9858-case benchmark corpus.

For each random case the MicroTaint output taint must CONTAIN the bit-flip lower
bound (a necessary soundness condition: every bit reachable by flipping a single
tainted input bit must be tainted). Any under-taint is logged in full
(assembly, bytes, state, taint, output, lower bound, missed mask) so it is
immediately reproducible.

Run (tee both streams to a file):
    python soundness_fuzz.py --hours 4 --seed 1 2>&1 | tee fuzz_$(date +%Y%m%d_%H%M%S).log
Smoke test:
    python soundness_fuzz.py --max-cases 200
"""
from __future__ import annotations

import argparse
import random
import sys
import time

import benchmark
import bitflip_oracle_scratch as bf
import worker_microtaint as wm

REGISTERS = ['RAX', 'RBX', 'RCX', 'RDX']
MASK64 = (1 << 64) - 1


def _mask(v: object) -> int:
    return 0 if not isinstance(v, int) else v & MASK64


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--hours', type=float, default=4.0, help='wall-clock budget')
    ap.add_argument('--max-cases', type=int, default=0, help='stop after N cases (0 = time only)')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--arch', default='x86_64')
    ap.add_argument('--beat', type=float, default=30.0, help='heartbeat interval (s)')
    args = ap.parse_args()

    random.seed(args.seed)
    sim = benchmark.GroundTruthSimulator()
    t0 = time.time()
    deadline = t0 + args.hours * 3600
    last_beat = t0
    n = sound = uncovered = errors = 0
    unsound: list[dict] = []

    print(
        f'[fuzz] start budget={args.hours}h seed={args.seed} arch={args.arch} '
        f'max_cases={args.max_cases or "inf"}',
        flush=True,
    )
    while time.time() < deadline and (args.max_cases == 0 or n < args.max_cases):
        n += 1
        try:
            tc = (
                benchmark.generate_single_test(args.arch)
                if random.random() < 0.5
                else benchmark.generate_sequence_test(args.arch)
            )
            state = {r: int(tc['state'].get(r, 0)) for r in REGISTERS}
            taint = {r: int(tc['taint'].get(r, 0)) for r in REGISTERS}
            bs = bytes.fromhex(tc['bytes'])
            mt = wm.run_one(tc).get('output_taint', {})
            lb, cov = bf.bitflip_lower_bound(sim, bs, state, bf.tainted_positions(taint))
            if not cov:
                uncovered += 1
                continue
            missed = 0
            for r in REGISTERS:
                missed |= _mask(lb.get(r, 0)) & ~_mask(mt.get(r, 0))
            if missed:
                rec = {
                    'assembly': tc.get('assembly', ''),
                    'bytes': tc['bytes'],
                    'category': tc.get('category', ''),
                    'state': state,
                    'taint': taint,
                    'mt': {r: _mask(mt.get(r, 0)) for r in REGISTERS},
                    'lb': {r: _mask(lb.get(r, 0)) for r in REGISTERS},
                    'missed': missed,
                }
                unsound.append(rec)
                print(
                    f'[UNSOUND] n={n} cat={rec["category"]} asm={rec["assembly"]!r} '
                    f'bytes={rec["bytes"]} state={state} taint={taint} '
                    f'mt={{{", ".join(f"{r}:{v:#x}" for r, v in rec["mt"].items())}}} '
                    f'lb={{{", ".join(f"{r}:{v:#x}" for r, v in rec["lb"].items())}}} '
                    f'missed={missed:#x}',
                    flush=True,
                )
            else:
                sound += 1
        except Exception as e:  # noqa: BLE001 -- a bad case must not kill the fuzzer
            errors += 1
            if errors <= 25:
                print(f'[error] n={n}: {type(e).__name__}: {e}', flush=True)
        now = time.time()
        if now - last_beat >= args.beat:
            last_beat = now
            rate = n / max(now - t0, 1e-9)
            print(
                f'[beat] {int(now - t0)}s n={n} sound={sound} UNSOUND={len(unsound)} '
                f'uncovered={uncovered} errors={errors} {rate:.0f} cases/s '
                f'{int(max(deadline - now, 0))}s left',
                flush=True,
            )

    print('=' * 72, flush=True)
    print(
        f'[fuzz] DONE total={n} sound={sound} UNSOUND={len(unsound)} '
        f'uncovered={uncovered} errors={errors} elapsed={int(time.time() - t0)}s',
        flush=True,
    )
    if unsound:
        print(f'[fuzz] {len(unsound)} UNDER-TAINT case(s) found -- see [UNSOUND] lines above.', flush=True)
    else:
        print('[fuzz] NO under-taints: MicroTaint sound on every covered case.', flush=True)
    return 1 if unsound else 0


if __name__ == '__main__':
    sys.exit(main())
