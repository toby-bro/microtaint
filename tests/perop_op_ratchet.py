"""Per-instruction taint-propagation OP COUNT, ratcheted, over every ISA.

The end goal for the taint engine is to compile each instruction's taint
propagation to native code.  What that costs is set by how many primitive
operations the propagation needs -- for the WHOLE instruction, flags included --
so that number is measured here per instruction per ISA, recorded in a baseline,
and allowed to move in one direction only.

Two things are ratcheted:

  * per-instruction op count -- an instruction may never need MORE operations
    than the baseline records.  This is the optimisation target.
  * coverage -- an instruction the pass answers today may not start declining.
    Without this, "fewer ops" could be bought by answering fewer instructions.

A decline costs nothing and is recorded as such; making a declining instruction
answerable is an improvement and is accepted (it lowers `declined`).

    .venv/bin/python -m tests.perop_op_ratchet            # report + check
    .venv/bin/python -m tests.perop_op_ratchet --update   # rewrite baseline

The vectors are fixed by a per-label seed, so the count is deterministic: it
depends on the p-code program and the taint rules, not on which random values a
run happened to draw.  Values matter (the rules are value-aware), so they are
drawn once per label and pinned.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal, TypedDict

from tests.perop_c_bank import Declined, perop_c_step


class IsaOps(TypedDict):
    """One measured ISA: the op count per instruction it answered, and the
    labels it declined."""

    ops: dict[str, int]
    declined: list[str]


Measured = dict[str, IsaOps]
#: (isa, label, baseline, now) -- baseline is 'declined' for a new answer.
Change = tuple[str, str, int | Literal['declined'], int]
#: (isa, label, baseline op count).  A decline has no 'now' to report.
NewDecline = tuple[str, str, int]

BASELINE = Path(__file__).parent / 'perop_op_baseline.json'
MASK64 = 0xFFFFFFFFFFFFFFFF


def _vector(reg_names: list[str], label: str,
            ) -> tuple[dict[str, int], dict[str, int]]:
    """The pinned (taint, values) pair for one instruction.

    Dense taint on the first few registers: the op count must reflect the work
    of a real propagation, and several rules take a cheaper branch when an input
    is provably clean.  Measuring the clean case would flatter every number.
    """
    rng = random.Random(f'opcount:{label}')
    taint: dict[str, int] = {}
    for i, r in enumerate(reg_names):
        taint[r] = MASK64 if i < 4 else 0
    values = {r: rng.randint(1, MASK64) for r in reg_names}
    return taint, values


def measure(isas: list[str] | None = None) -> Measured:
    """-> {isa: {'ops': {label: n}, 'declined': [label, ...]}}"""
    from benchmark.instruction_bank import load_bank

    out: Measured = {}
    specs = load_bank(isas=set(isas) if isas else None)
    for isa, spec in sorted(specs.items()):
        ops: dict[str, int] = {}
        declined: list[str] = []
        reg_names = [r.name for r in spec.regs]
        for ins in spec.instructions:
            taint, values = _vector(reg_names, ins.label)
            try:
                _t, _v, cost = perop_c_step(spec.arch, ins.bytes, spec.regs,
                                            taint, values)
            except Declined:
                declined.append(ins.label)
                continue
            except Exception:
                declined.append(ins.label)
                continue
            ops[ins.label] = cost['ops']
        out[isa] = {'ops': ops, 'declined': sorted(set(declined))}
    return out


def _quantile(xs: list[int], q: float) -> int:
    if not xs:
        return 0
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def summarize(measured: Measured) -> str:
    lines: list[str] = []
    grand_ops = grand_n = grand_dec = 0
    for isa, d in sorted(measured.items()):
        vals = list(d['ops'].values())
        n = len(vals)
        dec = len(d['declined'])
        grand_ops += sum(vals)
        grand_n += n
        grand_dec += dec
        cov = 100 * n / (n + dec) if (n + dec) else 0
        lines.append(
            f'  {isa:12s} n={n:4d} declined={dec:3d} ({cov:5.1f}% answered)  '
            f'ops mean={sum(vals)/n if n else 0:6.1f} p50={_quantile(vals,0.5):4d} '
            f'p95={_quantile(vals,0.95):4d} p100={max(vals) if vals else 0:4d}')
    total = grand_n + grand_dec
    lines.append(f'  {"TOTAL":12s} n={grand_n:4d} declined={grand_dec:3d} '
                 f'({100*grand_n/total if total else 0:5.1f}% answered)  '
                 f'ops mean={grand_ops/grand_n if grand_n else 0:6.1f}')
    return '\n'.join(lines)


def compare(measured: Measured, baseline: Measured,
            ) -> tuple[list[Change], list[Change], list[NewDecline]]:
    """-> (regressions, improvements, new_declines) with per-instruction detail."""
    regressions: list[Change] = []
    improvements: list[Change] = []
    new_declines: list[NewDecline] = []
    for isa, d in measured.items():
        base = baseline.get(isa)
        if base is None:
            continue
        base_ops = base['ops']
        base_dec = set(base['declined'])
        for label, n in d['ops'].items():
            b = base_ops.get(label)
            if b is None:
                if label in base_dec:
                    improvements.append((isa, label, 'declined', n))
                continue
            if n > b:
                regressions.append((isa, label, b, n))
            elif n < b:
                improvements.append((isa, label, b, n))
        for label in d['declined']:
            if label in base_ops:
                new_declines.append((isa, label, base_ops[label]))
    return regressions, improvements, new_declines


def load_baseline() -> Measured | None:
    if not BASELINE.exists():
        return None
    loaded: Measured = json.loads(BASELINE.read_text())
    return loaded


def save_baseline(measured: Measured) -> None:
    BASELINE.write_text(json.dumps(measured, indent=1, sort_keys=True) + '\n')


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--update', action='store_true',
                    help='rewrite the baseline from this run')
    ap.add_argument('--isas', nargs='*', default=None)
    ap.add_argument('--top', type=int, default=12,
                    help='show the N most expensive instructions per ISA')
    args = ap.parse_args(argv)

    measured = measure(args.isas)
    print('taint-propagation op count per instruction (whole instruction, flags included)')
    print(summarize(measured))

    if args.top:
        print('\nmost expensive:')
        for isa, d in sorted(measured.items()):
            worst = sorted(d['ops'].items(), key=lambda kv: -kv[1])[:args.top]
            if worst:
                print(f'  {isa}: ' + ', '.join(f'{lbl}={n}' for lbl, n in worst))

    if args.update:
        save_baseline(measured)
        print(f'\nbaseline written: {BASELINE}')
        return 0

    baseline = load_baseline()
    if baseline is None:
        save_baseline(measured)
        print(f'\nno baseline; wrote {BASELINE}')
        return 0

    regressions, improvements, new_declines = compare(measured, baseline)
    print(f'\nvs baseline: {len(improvements)} cheaper, {len(regressions)} more '
          f'expensive, {len(new_declines)} newly declined')
    for isa, label, b, n in regressions[:20]:
        print(f'  REGRESSION {isa} {label}: {b} -> {n}')
    for isa, label, b in new_declines[:20]:
        print(f'  NEW DECLINE {isa} {label}: was {b} ops')
    return 1 if (regressions or new_declines) else 0


if __name__ == '__main__':
    raise SystemExit(main())
