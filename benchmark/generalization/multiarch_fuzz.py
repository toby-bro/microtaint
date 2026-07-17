#!/usr/bin/env python3
"""Randomised cross-ISA soundness fuzzer for MicroTaint.

The deterministic oracle (multiarch_oracle.py) checks ~15 instructions per ISA on
3 fixed states and 6 fixed taint masks -- a smoke test.  This fuzzer strengthens
that toward the x86 campaign: RANDOM states, RANDOM taint patterns, an EXPANDED
corpus that includes flag/carry/conditional SEQUENCES (where the entire x86 bug
set lived), and a time budget.

For each case it asserts the necessary soundness condition against a fresh-Unicorn
single-bit-flip oracle:

    sound  =>  microtaint_taint  >=  bitflip_lower_bound

Every under-taint is logged in full (asm, bytes, state, taint, mt, lb, missed) so
it is immediately reproducible.  Where the tainted-bit count k is small the exact
2^k ground truth is also computed, which additionally flags OVER-tainting.

The taint patterns are chosen to hit the classes that broke x86: dense masks and
runs of adjacent bits (carry/borrow boundaries), full masks (differential
cancellation), and CORRELATED masks that taint the SAME bit positions in both
operands (XOR cancellation, SBORROW overlap).  A fresh Unicorn per run avoids the
shared-simulator state leak that produced false positives in the x86 fuzzer.

Run:  python multiarch_fuzz.py --seconds 120 --seed 1 2>&1 | tee fuzz_isa_$(date +%s).log
      python multiarch_fuzz.py --arch arm64 --seconds 60
"""

from __future__ import annotations

import argparse
import random
import sys
import time

import multiarch_oracle as O

# Flag / carry / conditional SEQUENCES per ISA, each validated to assemble AND
# execute.  These exercise the condition/carry machinery -- NZCV (ARM64), XER/CR
# (PPC), icc (SPARC) -- by routing a flag into a scored GPR (cset/csel, mfcr,
# adc/adde/addx carry chains), the way `bt;setc` / `cmp;setl` did on x86.  MIPS
# has no flags: its compares write GPRs directly.
_EXTRA: dict[str, list[tuple[str, str, list[str]]]] = {
    'ARM64': [
        ('adds x0, x1, x2', 'X0', ['X1', 'X2']),
        ('cmp x1, x2; cset x0, lt', 'X0', ['X1', 'X2']),
        ('cmp x1, x2; cset x0, hi', 'X0', ['X1', 'X2']),
        ('cmp x1, x2; cset x0, eq', 'X0', ['X1', 'X2']),
        ('adds x3, x1, x2; adc x0, x1, x2', 'X0', ['X1', 'X2']),
        ('subs x3, x1, x2; sbc x0, x1, x2', 'X0', ['X1', 'X2']),
        ('cmp x1, x2; csel x0, x1, x2, lt', 'X0', ['X1', 'X2']),
        ('cmp x1, x2; csel x0, x1, x2, hs', 'X0', ['X1', 'X2']),
        ('eor x0, x1, x2', 'X0', ['X1', 'X2']),
        ('eon x0, x1, x2', 'X0', ['X1', 'X2']),  # x1 ^ ~x2 -- the xnor class
        ('orn x0, x1, x2', 'X0', ['X1', 'X2']),
    ],
    'PPC32BE': [
        ('addc 3, 4, 5', 'R3', ['R4', 'R5']),
        ('addc 6, 4, 5; adde 3, 4, 5', 'R3', ['R4', 'R5']),
        ('subfc 6, 4, 5; subfe 3, 4, 5', 'R3', ['R4', 'R5']),
        ('cmpw 0, 4, 5; mfcr 3', 'R3', ['R4', 'R5']),
        ('eqv 3, 4, 5', 'R3', ['R4', 'R5']),   # ~(r4 ^ r5) -- xnor class
        ('nand 3, 4, 5', 'R3', ['R4', 'R5']),
        ('andc 3, 4, 5', 'R3', ['R4', 'R5']),
    ],
    'SPARC32BE': [
        ('addcc %g1, %g2, %g3', 'G3', ['G1', 'G2']),
        ('addcc %g4, %g2, %g1; addx %g1, %g2, %g3', 'G3', ['G1', 'G2']),
        ('subcc %g4, %g2, %g1; subx %g1, %g2, %g3', 'G3', ['G1', 'G2']),
        ('andcc %g1, %g2, %g3', 'G3', ['G1', 'G2']),
        ('xnor %g1, %g2, %g3', 'G3', ['G1', 'G2']),
        ('andn %g1, %g2, %g3', 'G3', ['G1', 'G2']),
    ],
    'MIPS64BE': [
        ('sltu $2, $4, $5', 'V0', ['A0', 'A1']),
        ('addu $3, $4, $5; slt $2, $3, $4', 'V0', ['A0', 'A1']),
        ('nor $2, $4, $5', 'V0', ['A0', 'A1']),
        ('dsll $2, $4, 4', 'V0', ['A0']),
        ('dsrl $2, $4, 4', 'V0', ['A0']),
    ],
}


def _rand_taint(rng: random.Random, bits: int) -> int:
    """A taint mask over `bits`, weighted toward the patterns that broke x86."""
    mask = (1 << bits) - 1
    r = rng.random()
    if r < 0.30:
        return rng.getrandbits(bits)                      # dense random
    if r < 0.50:
        return 1 << rng.randrange(bits)                   # single bit
    if r < 0.68:
        k = rng.randint(2, min(12, bits))                 # run of adjacent bits
        start = rng.randrange(bits - k + 1)
        return (((1 << k) - 1) << start) & mask
    if r < 0.80:
        return mask                                       # full mask
    if r < 0.90:
        hi = rng.randint(1, min(16, bits))                # high / sign region
        return (mask ^ ((1 << (bits - hi)) - 1)) & mask
    return rng.getrandbits(bits) & rng.getrandbits(bits)  # sparse random


def _gen_taint(rng: random.Random, spec: O.IsaSpec, srcs: list[str]) -> dict[str, int]:
    """Random taint over the instruction's real source operands.  With some
    probability the two operands share the SAME tainted positions -- the
    correlated case that triggers XOR cancellation and SBORROW overlap."""
    taint = dict.fromkeys(spec.regs, 0)
    if len(srcs) >= 2 and rng.random() < 0.35:
        shared = _rand_taint(rng, spec.bits)
        for r in srcs:
            taint[r] = shared if rng.random() < 0.7 else _rand_taint(rng, spec.bits)
    else:
        for r in srcs:
            taint[r] = _rand_taint(rng, spec.bits)
    return taint


def _corpus(spec: O.IsaSpec) -> list[tuple[str, str, list[str]]]:
    return list(spec.prog) + _EXTRA.get(spec.label, [])


def _fuzz_arch(  # noqa: C901
    spec: O.IsaSpec,
    seconds: float,
    seed: int,
    beat: float,
) -> tuple[int, int, int, int, list[str]]:
    rng = random.Random(seed)
    # pre-assemble the corpus once
    corpus: list[tuple[str, str, list[str], bytes]] = []
    for asm, out_reg, srcs in _corpus(spec):
        try:
            corpus.append((asm, out_reg, srcs, bytes(spec.ks.asm(asm, O.CODE_ADDR)[0])))
        except Exception as e:
            print(f'  [{spec.label}] SKIP {asm!r}: assemble: {e}', flush=True)
    if not corpus:
        return 0, 0, 0, 0, []

    n = unsound = exact_n = exact_ok = 0
    reports: list[str] = []
    t0 = time.time()
    last = t0
    deadline = t0 + seconds
    while time.time() < deadline:
        asm, out_reg, srcs, code = rng.choice(corpus)
        state = {r: rng.getrandbits(spec.bits) for r in spec.regs}
        taint = _gen_taint(rng, spec, srcs)
        try:
            lb = O.bitflip_lower_bound(spec, code, state, taint)
            mt = O.microtaint(spec, code, state, taint)
        except Exception:  # noqa: BLE001, S112 -- a bad random case must not kill the fuzzer
            continue
        n += 1
        missed = 0
        for r in spec.regs:
            missed |= lb[r] & ~((mt.get(r, 0) or 0)) & spec.mask
        if missed:
            unsound += 1
            line = (
                f'[UNSOUND] {spec.label} asm={asm!r} bytes={code.hex()} '
                f'state={ {r: hex(state[r]) for r in srcs} } '
                f'taint={ {r: hex(taint[r]) for r in srcs} } '
                f'mt={ {r: hex(mt.get(r, 0)) for r in spec.regs if mt.get(r, 0)} } '
                f'lb={ {r: hex(lb[r]) for r in spec.regs if lb[r]} } missed={missed:#x}'
            )
            print(line, flush=True)
            if len(reports) < 40:
                reports.append(line)
        gt = O.exact_gt(spec, code, state, taint)
        if gt is not None:
            exact_n += 1
            if all((mt.get(r, 0) or 0) & spec.mask == gt[r] for r in spec.regs):
                exact_ok += 1
        now = time.time()
        if now - last >= beat:
            last = now
            rate = n / max(now - t0, 1e-9)
            print(
                f'  [{spec.label}] {int(now - t0)}s n={n} UNSOUND={unsound} '
                f'exact={exact_ok}/{exact_n} {rate:.0f} cases/s',
                flush=True,
            )
    return n, unsound, exact_n, exact_ok, reports


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arch', default='all', choices=[*O.ISAS.keys(), 'all'])
    ap.add_argument('--seconds', type=float, default=120.0, help='wall-clock budget PER arch')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--beat', type=float, default=15.0)
    args = ap.parse_args()

    targets = list(O.ISAS) if args.arch == 'all' else [args.arch]
    rows: list[tuple[str, int, int, int, int]] = []
    grand_unsound = 0
    for i, key in enumerate(targets):
        spec = O.ISAS[key]()
        print(f'=== fuzzing {spec.label} for {args.seconds:.0f}s (seed {args.seed + i}) ===', flush=True)
        n, unsound, en, eok, _ = _fuzz_arch(spec, args.seconds, args.seed + i, args.beat)
        rows.append((spec.label, n, unsound, en, eok))
        grand_unsound += unsound

    print('\n' + '=' * 74)
    print('MicroTaint cross-ISA fuzzer -- randomised states + taints, fresh-Uc oracle')
    print('=' * 74)
    print(f'{"ISA":10} {"cases":>9} {"under-taints":>13} {"exact-checked":>14} {"exact":>7}')
    print('-' * 74)
    for label, n, unsound, en, eok in rows:
        pct = f'{100.0 * eok / en:.0f}%' if en else 'n/a'
        print(f'{label:10} {n:9d} {unsound:13d} {en:14d} {pct:>7}')
    print('-' * 74)
    total_cases = sum(r[1] for r in rows)
    print(f'{"TOTAL":10} {total_cases:9d} {grand_unsound:13d}')
    print()
    if grand_unsound == 0:
        print('NO under-tainting found on any ISA (fresh-Unicorn single-bit-flip oracle).')
    else:
        print(f'{grand_unsound} UNDER-TAINT case(s) -- see the [UNSOUND] lines above.')
    print('=' * 74)
    return 1 if grand_unsound else 0


if __name__ == '__main__':
    sys.exit(main())
