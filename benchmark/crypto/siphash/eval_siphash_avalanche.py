#!/usr/bin/env python3
"""
eval_siphash_avalanche.py
=========================
Bit-precise avalanche evaluation of SipHash-2-4 under microtaint.

Compiles the C sources from the same directory, then for each of the 128
input bit positions (16 bytes x 8 bits) runs the engine with exactly ONE
bit tainted via MicrotaintWrapper.taint_bit() and reads the predicted
taint mask from shadow memory after the hash executes.

Three quantities are reported per input bit across N random messages:

  engine_bits    -- bits the engine predicts as tainted on the 8-byte output
  empirical_bits -- bits that actually flip when the input bit is toggled
                    (ground truth: hash(msg) XOR hash(msg with bit flipped))
  precision_bits -- bits in both masks (engine AND empirical)

Expected results for a correct, bit-precise engine on SipHash-2-4:
  engine_bits    ~ 32/64   (engine marks ~half the output bits)
  empirical_bits ~ 32/64   (SipHash PRF/avalanche guarantee)
  soundness      :  empirical bits are always a subset of engine bits
                    (no output bit actually flips without engine prediction)

Usage
-----
    # Quick smoke-test (1 message, 8 bits, 4 workers):
    uv run python eval_siphash_avalanche.py --messages 1 --bits 8

    # Full evaluation (1000 messages, all 128 bits, 8 parallel workers):
    uv run python eval_siphash_avalanche.py --messages 1000 --workers 8

    # Save results for the paper:
    uv run python eval_siphash_avalanche.py --messages 1000 --workers 8 \\
        --csv results_avalanche.csv --matrix
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Locate sources relative to this script
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
SIPHASH_SOURCES = [
    SCRIPT_DIR / 'test_avalanche_siphash.c',
    SCRIPT_DIR / 'siphash_ref.c',
]
GCC_FLAGS = ['-O0', '-g', '-static', '-no-pie', '-fno-stack-protector']


def build_binary(output: Path) -> None:
    """Compile the SipHash harness. Raises RuntimeError on failure."""
    for src in SIPHASH_SOURCES:
        if not src.exists():
            raise FileNotFoundError(f'Source file not found: {src}')
    cmd = ['gcc', *GCC_FLAGS, '-o', str(output), *[str(s) for s in SIPHASH_SOURCES]]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f'Compilation failed:\n{r.stderr.decode("utf-8", errors="replace")}')


# ---------------------------------------------------------------------------
# Single-run worker (top-level for multiprocessing pickling)
# ---------------------------------------------------------------------------


def _run_one(task: tuple) -> dict[str, Any]:
    """
    Run the SipHash binary once under microtaint with exactly one tainted bit.

    task = (binary_path_str, message_bytes, taint_byte_idx, taint_bit_idx)

    Returns:
        taint_byte, taint_bit  -- which input bit was tainted
        engine_mask            -- 64-bit mask of predicted tainted output bits
        empirical_mask         -- 64-bit mask of bits that actually flipped
    """
    binary_str, msg, taint_byte, taint_bit = task

    # ---- Silence all output in this worker process ----
    # Qiling emits [!] rseq warnings via Python logging and writes the
    # emulated binary's stdout to the real process stdout, both polluting
    # the parent's progress bar.  Redirect everything to /dev/null.
    import logging as _log
    import os as _os
    import sys as _sys

    _log.disable(_log.CRITICAL)
    _nul = open(_os.devnull, 'w')
    _sys.stdout = _nul
    _sys.stderr = _nul

    # ---- microtaint imports (inside worker subprocess) ----
    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper
    from qiling import Qiling
    from qiling.const import QL_INTERCEPT, QL_VERBOSE

    # We capture write(fd=1, buf, 8) and read shadow at THAT moment.
    # Reading after ql.run() returns is too late: libc's exit cleanup
    # runs many instructions that can clear the shadow at the buffer's
    # address (stack writes whose ranges overlap the output buffer).
    output_buf: list[int | None] = [None]
    engine_mask_holder: list[int] = [0]

    # The write hook needs access to wrapper.shadow_mem, but wrapper is
    # created below.  We use a holder list to break the chicken-and-egg.
    wrapper_holder: list[Any] = [None]

    def _write_hook(ql: Qiling, fd: int, buf: int, count: int, *_: Any) -> int:
        """Capture output buffer + shadow mask at write() time, then suppress."""
        if fd == 1 and count == 8 and output_buf[0] is None:
            output_buf[0] = buf
            w = wrapper_holder[0]
            if w is not None:
                engine_mask_holder[0] = w.shadow_mem.read_mask(buf, 8)
        return count

    # ---- Qiling + wrapper setup ----
    ql = Qiling([binary_str], '/', verbose=QL_VERBOSE.OFF)

    class _FixedStdin:
        def read(self, n: int) -> bytes:
            return msg[:n]

    ql.os.stdin = _FixedStdin()
    ql.os.set_syscall(1, _write_hook, QL_INTERCEPT.CALL)

    reporter = Reporter(json_mode=True, stream=io.StringIO())
    wrapper = MicrotaintWrapper(
        ql,
        check_sc=False,
        check_bof=False,
        check_uaf=False,
        check_aiw=False,
        reporter=reporter,
    )
    wrapper_holder[0] = wrapper

    # Override the sys_read hook: write data into the buffer, then use
    # taint_bit() to mark exactly one bit rather than the whole buffer.
    def _read_hook(ql: Qiling, fd: int, buf: int, count: int) -> int:
        if fd != 0:
            return 0
        data = msg[:count]
        if not data:
            return 0
        ql.mem.write(buf, data)
        wrapper.taint_bit(buf + taint_byte, taint_bit)
        return len(data)

    ql.os.set_syscall(0, _read_hook, QL_INTERCEPT.CALL)

    try:
        ql.run()
    except Exception:
        pass

    engine_mask = engine_mask_holder[0]

    # ---- Empirical XOR (binary run, independent of engine) ----
    def _hash(m: bytes) -> int:
        r = subprocess.run([binary_str], input=m, capture_output=True, timeout=5)
        return int.from_bytes(r.stdout, 'little') if len(r.stdout) == 8 else 0

    msg_flipped = bytearray(msg)
    msg_flipped[taint_byte] ^= 1 << taint_bit
    empirical_mask = _hash(msg) ^ _hash(bytes(msg_flipped))

    return {
        'taint_byte': taint_byte,
        'taint_bit': taint_bit,
        'engine_mask': engine_mask,
        'empirical_mask': empirical_mask,
    }


# ---------------------------------------------------------------------------
# Statistics aggregation
# ---------------------------------------------------------------------------


def _popcount(x: int) -> int:
    return bin(x).count('1')


def _aggregate(results: list[dict]) -> dict[tuple[int, int], dict]:
    """Group results by (taint_byte, taint_bit) and compute per-bit stats."""
    from collections import defaultdict

    groups: dict[tuple, list] = defaultdict(list)
    for r in results:
        groups[(r['taint_byte'], r['taint_bit'])].append(r)

    stats = {}
    for key, runs in groups.items():
        engine_c = [_popcount(r['engine_mask']) for r in runs]
        empirical_c = [_popcount(r['empirical_mask']) for r in runs]
        precision_c = [_popcount(r['engine_mask'] & r['empirical_mask']) for r in runs]
        # soundness violation: empirical bit set but engine didn't predict it
        unsound = sum(1 for r in runs if r['empirical_mask'] & ~r['engine_mask'])
        n = len(runs)
        stats[key] = {
            'n': n,
            'engine_mean': sum(engine_c) / n,
            'empirical_mean': sum(empirical_c) / n,
            'precision_mean': sum(precision_c) / n,
            'soundness_failures': unsound,
        }
    return stats


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _ansi(code: str, t: str) -> str:
    return f'\033[{code}m{t}\033[0m' if sys.stdout.isatty() else t


def bold(t: str) -> str:
    return _ansi('1', t)


def dim(t: str) -> str:
    return _ansi('2', t)


def green(t: str) -> str:
    return _ansi('1;32', t)


def red(t: str) -> str:
    return _ansi('1;31', t)


def yellow(t: str) -> str:
    return _ansi('1;33', t)


def _fmt(v: float, lo: float = 24.0, hi: float = 40.0) -> str:
    s = f'{v:5.1f}'
    return (green if lo <= v <= hi else yellow)(s)


def print_summary(stats: dict, n_messages: int, elapsed: float) -> None:
    all_bits = list(stats.values())
    total_runs = sum(s['n'] for s in all_bits)
    unsound_runs = sum(s['soundness_failures'] for s in all_bits)

    e_vals = [s['engine_mean'] for s in all_bits]
    em_vals = [s['empirical_mean'] for s in all_bits]
    pr_vals = [s['precision_mean'] for s in all_bits]

    e_avg = sum(e_vals) / len(e_vals)
    em_avg = sum(em_vals) / len(em_vals)
    pr_avg = sum(pr_vals) / len(pr_vals)

    print()
    print(bold('=== SipHash-2-4 bit-precise avalanche results ==='))
    print(f'  Input bits tested  : {len(stats)} / 128')
    print(f'  Messages per bit   : {n_messages}')
    print(f'  Total engine runs  : {total_runs}  ' f'({elapsed:.1f}s total, {elapsed/total_runs*1000:.0f}ms/run)')
    print()
    print(bold('Averages across all (bit, message) pairs:'))
    print(f'  Engine predicted bits   : {_fmt(e_avg)} / 64  {dim("(expected ~32)")}')
    print(f'  Empirical flipped bits  : {_fmt(em_avg)} / 64  {dim("(expected ~32)")}')
    print(f'  Precision (engine∩empi) : {_fmt(pr_avg)} / 64  ' f'{dim("(should approach empirical)")}')
    print()

    if unsound_runs == 0:
        print(green('  SOUNDNESS: PASS') + dim(f' — no under-tainting across {total_runs} runs'))
    else:
        print(red('  SOUNDNESS: FAIL') + f' — {unsound_runs} violations out of {total_runs} runs')
        print(dim('  A violation: an output bit flipped empirically but the engine'))
        print(dim('  did not predict it (under-tainting).'))

    # Over-tainting ratio
    if em_avg > 0:
        ot_ratio = (e_avg - pr_avg) / em_avg * 100
        if ot_ratio < 10:
            ot_str = green(f'{ot_ratio:.1f}%')
        elif ot_ratio < 30:
            ot_str = yellow(f'{ot_ratio:.1f}%')
        else:
            ot_str = red(f'{ot_ratio:.1f}%')
        print(f'  Over-tainting ratio     : {ot_str}  ' f'{dim("(engine bits not in empirical / empirical bits)")}')

    print()
    print(bold('Per-input-byte averages:'))
    print(f'  {"Byte":>5}  {"Engine":>8}  {"Empirical":>10}  {"Precision":>10}  ' f'{"Unsound":>8}')
    print(f'  {"-----":>5}  {"--------":>8}  {"----------":>10}  {"----------":>10}  ' f'{"--------":>8}')
    for byte_i in range(16):
        bstats = [v for (b, _), v in stats.items() if b == byte_i]
        if not bstats:
            continue
        be = sum(s['engine_mean'] for s in bstats) / len(bstats)
        bem = sum(s['empirical_mean'] for s in bstats) / len(bstats)
        bpr = sum(s['precision_mean'] for s in bstats) / len(bstats)
        buns = sum(s['soundness_failures'] for s in bstats)
        uns_str = red(str(buns)) if buns else dim('0')
        print(f'  {byte_i:>5}  {_fmt(be):>8}  {_fmt(bem):>10}  {_fmt(bpr):>10}  ' f'{uns_str:>8}')


def print_matrix(stats: dict) -> None:
    """Print a 16x8 grid of engine_mean values."""
    print()
    print(bold('Engine-predicted tainted output bits — input [byte x bit] matrix:'))
    header = '  ' + '      '.join(f'bit{j}' for j in range(8))
    print(f'  {"":>6}  {header}')
    for byte_i in range(16):
        row = []
        for bit_j in range(8):
            s = stats.get((byte_i, bit_j))
            row.append(_fmt(s['engine_mean']) if s else '  n/a')
        print(f'  B{byte_i:<5}  ' + '  '.join(row))


def save_csv(stats: dict, path: Path) -> None:
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                'input_byte',
                'input_bit',
                'n_messages',
                'engine_mean',
                'empirical_mean',
                'precision_mean',
                'soundness_failures',
            ],
        )
        w.writeheader()
        for (byte_i, bit_j), s in sorted(stats.items()):
            w.writerow(
                {
                    'input_byte': byte_i,
                    'input_bit': bit_j,
                    'n_messages': s['n'],
                    'engine_mean': round(s['engine_mean'], 3),
                    'empirical_mean': round(s['empirical_mean'], 3),
                    'precision_mean': round(s['precision_mean'], 3),
                    'soundness_failures': s['soundness_failures'],
                }
            )
    print(f'\n  Results saved to: {path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Bit-precise SipHash-2-4 avalanche evaluation under microtaint',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--messages', type=int, default=10, help='Random messages to test per input bit (default: 10)')
    ap.add_argument('--bits', default='all', help='"all" for all 128 bits, or an integer (default: all)')
    ap.add_argument('--workers', type=int, default=4, help='Parallel worker processes (default: 4)')
    ap.add_argument('--seed', type=int, default=42, help='RNG seed for reproducibility (default: 42)')
    ap.add_argument('--csv', default=None, metavar='FILE', help='Save per-bit statistics to a CSV file')
    ap.add_argument('--matrix', action='store_true', help='Print the 16x8 taint matrix')
    args = ap.parse_args()

    # ---- Build binary ----
    import atexit
    import shutil
    import tempfile

    build_dir = Path(tempfile.mkdtemp(prefix='microtaint_siphash_'))
    atexit.register(shutil.rmtree, build_dir, True)

    binary = build_dir / 'test_avalanche_siphash'
    print(bold('\n=== SipHash-2-4 bit-precise avalanche evaluation ==='))
    print(f'  Compiling: {" + ".join(s.name for s in SIPHASH_SOURCES)} ...', end=' ')
    sys.stdout.flush()
    try:
        build_binary(binary)
        print(green('OK'))
    except (FileNotFoundError, RuntimeError) as e:
        print(red('FAIL'))
        print(f'  {e}', file=sys.stderr)
        sys.exit(1)

    # ---- Build task list ----
    rng = random.Random(args.seed)
    all_bits = [(b, j) for b in range(16) for j in range(8)]
    if args.bits == 'all':
        bits_to_test = all_bits
    else:
        bits_to_test = all_bits[: int(args.bits)]

    tasks = [
        (str(binary), bytes(rng.randint(0, 255) for _ in range(16)), byte_i, bit_j)
        for byte_i, bit_j in bits_to_test
        for _ in range(args.messages)
    ]
    total = len(tasks)

    print(f'  Input bits : {len(bits_to_test)} / 128')
    print(f'  Messages   : {args.messages} per bit')
    print(f'  Total runs : {total}')
    print(f'  Workers    : {args.workers}')
    print()

    # ---- Run with progress bar ----
    results: list[dict] = []
    done = 0
    t0 = time.perf_counter()
    bar_width = 32

    def _tick(r: dict) -> None:
        nonlocal done
        results.append(r)
        done += 1
        pct = done / total
        filled = int(bar_width * pct)
        bar = '█' * filled + '░' * (bar_width - filled)
        elapsed = time.perf_counter() - t0
        eta = (elapsed / done) * (total - done) if done else 0
        print(
            f'\r  [{bar}] {100*pct:5.1f}%  {done}/{total}  ' f'{elapsed:.0f}s elapsed  ETA {eta:.0f}s  ',
            end='',
            flush=True,
        )

    with Pool(processes=args.workers) as pool:
        for r in pool.imap_unordered(_run_one, tasks, chunksize=1):
            _tick(r)

    elapsed = time.perf_counter() - t0
    print()  # end progress bar line

    # ---- Aggregate and report ----
    stats = _aggregate(results)
    print_summary(stats, args.messages, elapsed)

    if args.matrix:
        print_matrix(stats)

    if args.csv:
        save_csv(stats, Path(args.csv))


if __name__ == '__main__':
    main()
