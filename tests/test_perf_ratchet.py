"""tests/test_perf_ratchet.py
=================================
Per-instruction / per-ISA taint-propagation PERFORMANCE RATCHET.

Goal: with every future change we ONLY EVER get faster (or equal) on every ISA
and every instruction; a regression on any single instruction fails CI.

Everything measured here is the taint-propagation HOT PATH -- ``circuit.evaluate``
alone. The circuits are built once, up front, in ``_cases`` (rule generation,
which is LRU-cached in production and runs once per instruction), so
``generate_static_rule`` is never inside the timed / counted region.

The corpus is the unified instruction bank (``benchmark/instruction_bank/``),
~1,500 forms across AMD64 / ARM64 / MIPS64BE / PPC32BE / RISCV64, shared with
the benchmark scripts.

Two artifacts, deliberately separate:

  * ``tests/perf_baseline.json`` (COMMITTED, machine-independent) -- the ratchet
    reference. It stores DETERMINISTIC work metrics per instruction:
      - ``cells``   : native cell re-executions per ``circuit.evaluate`` (the
                      real hot-path cost -- one flag-setting instruction re-runs
                      the whole instruction in the cell once per output). Counting,
                      not timing, so it is identical on every machine, never flaky.
      - ``assigns`` : number of top-level taint assignments the static rule emits
                      (structural size of the propagation the hot path executes).
      - ``nodes``   : total AST nodes across all of an instruction's taint
                      expressions. An exact prop should be a handful of ops
                      (polarised masks + a XOR + a sext/zext), so a 900-1500-node
                      flag tree is a red flag; this metric makes that visible and
                      only-ever-decreasing.
    The gate asserts current <= baseline for every instruction. Wall-clock ns is
    machine-dependent, so it is measured, printed and logged but NEVER asserted.

  * ``tests/perf.log.d/`` (GITIGNORED, per-machine) -- one JSON file per run with
    the raw perf of that run: deterministic metrics plus, when timing is enabled,
    wall-clock ns and the full stats block (tp/s, tp/s/i, mean, median, p50, p90,
    p99, p100, min, max, stdev, geomean). Lets you track progression over time.

Usage
-----
Fast deterministic gate (runs in the normal suite; no "bench" in the name)::

    pytest tests/test_perf_ratchet.py::test_instruction_cost_ratchet

Full timing benchmark (excluded from the fast gate by the "not bench" filter;
note the filename contains "ratchet", so select the bench by its node id)::

    pytest tests/test_perf_ratchet.py::test_perf_timing_bench -s

Regenerate the baseline after a genuine optimization (ratchets DOWN only)::

    UPDATE_PERF_BASELINE=1 pytest tests/test_perf_ratchet.py::test_instruction_cost_ratchet
    # refuses to RAISE any baseline value unless PERF_ALLOW_REGRESSION=1 too.
"""


from __future__ import annotations

import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pytest

from microtaint.instrumentation.ast import EvalContext, LogicCircuit
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import ImplicitTaintPolicy

_HERE = Path(__file__).parent
BASELINE_PATH = _HERE / 'perf_baseline.json'
LOG_DIR = _HERE / 'perf.log.d'

# The unified instruction bank lives in the (durable) benchmark tree and is
# shared with the benchmark scripts; it stores pre-assembled bytes, so the gate
# needs no keystone. See benchmark/instruction_bank/README.md.
sys.path.insert(0, str(_HERE.parent / 'benchmark'))
from instruction_bank import load_bank  # type: ignore[import-not-found]  # noqa: E402

FULL = 0xFFFFFFFFFFFFFFFF

# Deterministic per-instruction metrics the ratchet gates (only ever decrease).
_METRICS = ('cells', 'assigns', 'nodes')


# ===========================================================================
# Corpus: the unified instruction bank (~1,500 forms across AMD64 / ARM64 /
# MIPS64BE / PPC32BE / RISCV64), covering move / arith / logic / shift (imm +
# variable) / compare / mul / div / flag / bit-scan / bit-field / rotate /
# memory, so the flag-setting and variable-shift tail (the p99/p100) is
# represented on every ISA. Bytes are pre-assembled -> no keystone here.
# ===========================================================================


class _Case:
    __slots__ = ('asm', 'circ', 'ectx', 'isa', 'sim')

    def __init__(self, isa: str, asm: str, circ: LogicCircuit, ectx: EvalContext,
                 sim: CellSimulator) -> None:
        self.isa, self.asm, self.circ, self.ectx, self.sim = isa, asm, circ, ectx, sim


@lru_cache(maxsize=1)
def _cases() -> tuple[_Case, ...]:
    """Build every circuit once (rule-gen is the expensive part; cached)."""
    cases: list[_Case] = []
    for spec in load_bank().values():
        sim = CellSimulator(spec.arch)  # production default: C kernel unless disabled
        vals = {r.name: (0x1234567 + 7 * i) for i, r in enumerate(spec.regs)}
        # Which inputs are tainted does not change the deterministic work metric
        # (cells = number of output InstructionCellExprs, structural), so a fixed
        # subset keeps the gate reproducible.
        taint = {r.name: FULL for r in spec.regs[1:4]}
        for ins in spec.instructions:
            circ = generate_static_rule(spec.arch, ins.bytes, spec.regs)
            ectx = EvalContext(
                input_values=vals,
                input_taint=taint,
                simulator=sim,
                implicit_policy=ImplicitTaintPolicy.KEEP,
            )
            cases.append(_Case(spec.name, ins.label, circ, ectx, sim))
    return tuple(cases)


# ===========================================================================
# Measurement
# ===========================================================================


def _count_nodes(expr: object, depth: int = 0) -> int:
    """Total AST nodes in a taint expression (structural size of the rule).

    A high count is the symptom the user flagged: an exact taint prop should be a
    handful of ops (polarised masks + a XOR + a sext/zext), so 900-1500-node flag
    trees signal wasteful rule generation. Tracked so the ratchet drives it down
    and catches regressions."""
    if expr is None or depth > 200:
        return 0
    total = 1
    for attr in ('lhs', 'rhs', 'expr', 'operand', 'address_expr'):
        child = getattr(expr, attr, None)
        if child is not None:
            total += _count_nodes(child, depth + 1)
    inputs = getattr(expr, 'inputs', None)
    if isinstance(inputs, dict):
        for child in inputs.values():
            total += _count_nodes(child, depth + 1)
    return total


def _work(case: _Case) -> tuple[int, int, int]:
    """Deterministic work per evaluate: (cells, assigns, nodes). Machine-independent."""
    for _ in range(3):  # warm any per-call memoization
        case.circ.evaluate(case.ectx)
    pcode = case.sim._pcode
    assert pcode is not None, 'a CellSimulator always has an evaluator'
    n0 = pcode.native_calls
    case.circ.evaluate(case.ectx)
    cells = pcode.native_calls - n0
    nodes = sum(_count_nodes(a.expression) for a in case.circ.assignments)
    return cells, len(case.circ.assignments), nodes


def _time_ns(case: _Case, reps: int = 600, batches: int = 3) -> float:
    """Steady-state ns/evaluate: min over batches of reps-call means (GC-robust).
    Modest reps because the bank is large and ns is informational, not gated."""
    for _ in range(100):
        case.circ.evaluate(case.ectx)
    best = float('inf')
    for _ in range(batches):
        t0 = time.perf_counter_ns()
        for _ in range(reps):
            case.circ.evaluate(case.ectx)
        best = min(best, (time.perf_counter_ns() - t0) / reps)
    return best


# ===========================================================================
# Stats
# ===========================================================================


def _pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = min(len(s) - 1, round(p / 100 * (len(s) - 1)))
    return s[k]


#: One measured instruction.  The metric fields are ints and the two labels
#: are strings; the metrics are read with a NAME from _METRICS, which is why
#: this is a plain dict and not a TypedDict -- a TypedDict cannot be indexed
#: by a variable.  The timing rows carry `ns` and `tp_s` as floats.
Row = dict[str, str | int | float]
#: A whole run as written to the perf log: rows plus derived summaries.
Record = dict[str, object]


def _number(row: Row, name: str) -> float:
    """A measured quantity off a row, as a float (the timings are not ints)."""
    v = row[name]
    assert isinstance(v, (int, float)), f'{name} is {type(v).__name__}'
    return float(v)


def _label(row: Row, name: str) -> str:
    """The isa or asm off a row, as a string."""
    v = row[name]
    assert isinstance(v, str), f'{name} is {type(v).__name__}, not a label'
    return v


def _recorded(baseline: Record, isa: str, asm: str) -> dict[str, int] | None:
    """The recorded floor for one instruction, or None if it is new.

    The baseline is isa -> asm -> {metric: count} with a `_meta` block sitting
    beside the ISAs, so both lookups are checked here rather than at each of
    the four places that walk it.
    """
    per_isa = baseline.get(isa)
    if not isinstance(per_isa, dict):
        return None
    entry = per_isa.get(asm)
    return entry if isinstance(entry, dict) else None


def _metric(row: Row, name: str) -> int:
    """One metric off a row.  The names in _METRICS all address ints; this is
    where that is checked rather than assumed."""
    v = row[name]
    assert isinstance(v, int), f'{name} is {type(v).__name__}, not a count'
    return v


def _stats(vals: list[float]) -> dict[str, float]:
    """Every distribution stat we can think of, over a list of per-instr values."""
    if not vals:
        return {}
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    pos = [v for v in vals if v > 0]
    geomean = math.exp(sum(math.log(v) for v in pos) / len(pos)) if pos else 0.0
    return {
        'n': n,
        'min': min(vals),
        'mean': mean,
        'median': _pct(vals, 50),
        'p50': _pct(vals, 50),
        'p90': _pct(vals, 90),
        'p99': _pct(vals, 99),
        'p100': max(vals),
        'max': max(vals),
        'stdev': math.sqrt(var),
        'geomean': geomean,
    }


def _throughput(ns_vals: list[float]) -> dict[str, float]:
    """tp/s (taint-propagations per second) figures.

    tp_s_aggregate : replay one of each instruction back to back -> props/sec
                     = n / sum(seconds).
    tp_s_per_instr : mean single-instruction throughput = mean_i(1e9 / ns_i)
                     ("tp/s/i" -- per-instruction props/sec, averaged).
    tp_s_geomean   : geometric-mean single-instruction throughput (robust).
    """
    pos = [v for v in ns_vals if v > 0]
    if not pos:
        return {}
    total_s = sum(pos) / 1e9
    per = [1e9 / v for v in pos]
    geo = math.exp(sum(math.log(v) for v in per) / len(per))
    return {
        'tp_s_aggregate': len(pos) / total_s,
        'tp_s_per_instr': sum(per) / len(per),
        'tp_s_geomean': geo,
    }


# ===========================================================================
# Baseline + per-run log persistence
# ===========================================================================


def _git(git: str, *args: str) -> str:
    # args are all controlled literals (git subcommands); git is a resolved abs path.
    out = subprocess.check_output([git, *args], cwd=_HERE, stderr=subprocess.DEVNULL)  # noqa: S603
    return out.decode().strip()


def _git_sha() -> tuple[str, bool]:
    git = shutil.which('git')  # full path -> no S607 (partial-path) risk
    if git is None:
        return 'unknown', False
    try:
        return _git(git, 'rev-parse', '--short', 'HEAD'), bool(_git(git, 'status', '--porcelain'))
    except Exception:
        return 'unknown', False


def _load_baseline() -> Record:
    if not BASELINE_PATH.exists():
        return {}
    data: Record = json.loads(BASELINE_PATH.read_text())
    return data


def _write_log(record: Record) -> Path | None:
    """One JSON file per run in perf.log.d/, timestamp-sortable name. Best effort."""
    try:
        LOG_DIR.mkdir(exist_ok=True)
        gi = LOG_DIR / '.gitignore'
        if not gi.exists():  # keep the dir self-ignoring even on a fresh clone
            gi.write_text('*\n!.gitignore\n')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        path = LOG_DIR / f'{stamp}-{record["git"]}-{record["kind"]}.json'
        path.write_text(json.dumps(record, indent=2, sort_keys=True))
        return path
    except OSError:
        return None


def _base_record(kind: str, rows: list[Row]) -> Record:
    sha, dirty = _git_sha()
    return {
        'ts': datetime.now(timezone.utc).isoformat(),
        'kind': kind,
        'git': sha,
        'dirty': dirty,
        'host': socket.gethostname(),
        'python': platform.python_version(),
        'use_c': os.environ.get('MICROTAINT_DISABLE_C_KERNEL') != '1',
        'n_instr': len(rows),
        'instr': rows,
    }


# ===========================================================================
# THE GATE: deterministic per-instruction work never regresses (non-flaky).
# ===========================================================================


def _deterministic_rows() -> tuple[list[Row], Record]:
    """Measure (cells, assigns) per instruction; return rows + a log record."""
    cases = _cases()
    assert cases, 'no instructions assembled -- corpus/keystone broken'
    rows: list[Row] = []
    by_isa: dict[str, list[Row]] = {}
    for c in cases:
        cells, assigns, nodes = _work(c)
        row: Row = {'isa': c.isa, 'asm': c.asm, 'cells': cells,
                   'assigns': assigns, 'nodes': nodes}
        rows.append(row)
        by_isa.setdefault(c.isa, []).append(row)
    rec = _base_record('ratchet', rows)
    rec['summary'] = {m: _stats([_metric(r, m) for r in rows]) for m in _METRICS}
    rec['per_isa'] = {
        isa: {m: _stats([_metric(r, m) for r in rs]) for m in _METRICS}
        for isa, rs in by_isa.items()
    }
    return rows, rec


def _regenerate_baseline(rows: list[Row], baseline: Record,
                         allow_reg: bool) -> None:
    new: dict[str, object] = {
        '_meta': {
            'description': 'Deterministic per-instruction taint-prop work. RATCHET: '
            'values only ever decrease. Regenerate with '
            'UPDATE_PERF_BASELINE=1 (add PERF_ALLOW_REGRESSION=1 to '
            'permit an intentional increase).',
            'metrics': {
                'cells': 'native cell re-executions per evaluate',
                'assigns': 'top-level taint assignments emitted by the rule',
                'nodes': 'total AST nodes across the rule expressions (structural size)',
            },
            'generated': datetime.now(timezone.utc).isoformat(),
        },
    }
    raised = []
    for r in rows:
        old = _recorded(baseline, _label(r, 'isa'), _label(r, 'asm'))
        raised += [
            f'{_label(r, "isa")} {_label(r, "asm")} {m}: {old[m]} -> {_metric(r, m)}'
            for m in _METRICS
            if old is not None and m in old and _metric(r, m) > old[m]
        ]
        per_isa = new.setdefault(_label(r, 'isa'), {})
        assert isinstance(per_isa, dict)
        per_isa[_label(r, 'asm')] = {m: _metric(r, m) for m in _METRICS}
    if raised and not allow_reg:
        pytest.fail(
            'UPDATE_PERF_BASELINE refused: these would RAISE the baseline '
            '(a regression). Set PERF_ALLOW_REGRESSION=1 to force.\n  ' + '\n  '.join(raised),
        )
    BASELINE_PATH.write_text(json.dumps(new, indent=2, sort_keys=True))
    print(f'\nwrote baseline: {BASELINE_PATH} ({len(rows)} instructions)')


def _diff_baseline(rows: list[Row],
                   baseline: Record) -> tuple[list[str], list[str], list[str]]:
    regressions, improvements, missing = [], [], []
    for r in rows:
        old = _recorded(baseline, _label(r, 'isa'), _label(r, 'asm'))
        if old is None:
            missing.append(f'{r["isa"]} {r["asm"]}')
            continue
        for m in _METRICS:
            if m not in old:
                continue
            got = _metric(r, m)
            where = f'{_label(r, "isa")} {_label(r, "asm")} {m}'
            if got > old[m]:
                regressions.append(f'{where}: {old[m]} -> {got} (+{got - old[m]})')
            elif got < old[m]:
                improvements.append(f'{where}: {old[m]} -> {got} ({got - old[m]})')
    return regressions, improvements, missing


def test_instruction_cost_ratchet() -> None:
    rows, rec = _deterministic_rows()
    baseline = _load_baseline()

    if os.environ.get('UPDATE_PERF_BASELINE') == '1':
        _regenerate_baseline(rows, baseline, os.environ.get('PERF_ALLOW_REGRESSION') == '1')
        _write_log(rec)
        return

    _write_log(rec)
    if not baseline:
        pytest.skip(
            'no perf_baseline.json yet -- create it with '
            'UPDATE_PERF_BASELINE=1 pytest '
            'tests/test_perf_ratchet.py::test_instruction_cost_ratchet',
        )

    regressions, improvements, missing = _diff_baseline(rows, baseline)
    if improvements:
        print('\nperf IMPROVED (re-baseline with UPDATE_PERF_BASELINE=1 to lock in):')
        print('\n'.join('  ' + line for line in improvements))
    if missing:
        print(
            f'\n{len(missing)} instruction(s) not in baseline (add via '
            f'UPDATE_PERF_BASELINE=1): ' + ', '.join(missing[:8]) + (' ...' if len(missing) > 8 else ''),
        )

    assert not regressions, (
        f'taint-propagation work REGRESSED on {len(regressions)} instruction(s) '
        '-- microtaint must only ever get faster. If this is an intentional, '
        'justified cost increase, re-baseline with UPDATE_PERF_BASELINE=1 '
        'PERF_ALLOW_REGRESSION=1.\n  ' + '\n  '.join(regressions)
    )


# ===========================================================================
# THE BENCHMARK: wall-clock + full stats, logged for progression tracking.
# "bench" in the name -> excluded from the fast gate by `-k "not bench"`.
# ===========================================================================


def test_perf_timing_bench() -> None:
    cases = _cases()
    assert cases
    rows: list[Row] = []
    by_isa: dict[str, list[Row]] = {}
    for c in cases:
        cells, assigns, nodes = _work(c)
        ns = _time_ns(c)
        row: Row = {
            'isa': c.isa,
            'asm': c.asm,
            'cells': cells,
            'assigns': assigns,
            'nodes': nodes,
            'ns': ns,
            'tp_s': (1e9 / ns if ns else 0.0),
        }
        rows.append(row)
        by_isa.setdefault(c.isa, []).append(row)

    ns_all = [_number(r, 'ns') for r in rows]
    summary = {
        'ns': _stats(ns_all),
        'cells': _stats([_number(r, 'cells') for r in rows]),
        'assigns': _stats([_number(r, 'assigns') for r in rows]),
        'throughput': _throughput(ns_all),
    }
    per_isa = {}
    for isa, rs in by_isa.items():
        ns_i = [_number(r, 'ns') for r in rs]
        per_isa[isa] = {'ns': _stats(ns_i), 'throughput': _throughput(ns_i), 'cells': _stats([_number(r, 'cells') for r in rs])}

    rec = _base_record('timing', rows)
    rec['summary'] = summary
    rec['per_isa'] = per_isa
    path = _write_log(rec)

    # ---- human-readable report ----
    s = summary['ns']
    tp = summary['throughput']
    print(f'\n=== taint-propagation timing ({len(rows)} instructions, {len(by_isa)} ISAs) ===')
    print(
        '  ns/evaluate:  '
        f'min={s["min"]:.0f}  p50={s["p50"]:.0f}  mean={s["mean"]:.0f}  '
        f'median={s["median"]:.0f}  p90={s["p90"]:.0f}  p99={s["p99"]:.0f}  '
        f'p100={s["p100"]:.0f}  stdev={s["stdev"]:.0f}  geomean={s["geomean"]:.0f}',
    )
    print(
        '  throughput:   '
        f'tp/s(aggregate)={tp["tp_s_aggregate"]:,.0f}  '
        f'tp/s/i(mean)={tp["tp_s_per_instr"]:,.0f}  '
        f'tp/s(geomean)={tp["tp_s_geomean"]:,.0f}',
    )
    cs = summary['cells']
    print(f'  cells/eval:   p50={cs["p50"]:.0f}  mean={cs["mean"]:.1f}  p99={cs["p99"]:.0f}  p100={cs["p100"]:.0f}')
    print('  per-ISA (ns p50 / p100, tp/s aggregate):')
    for isa, d in per_isa.items():
        print(
            f'    {isa:9s} p50={d["ns"]["p50"]:8.0f}  p100={d["ns"]["p100"]:8.0f}  '
            f'tp/s={d["throughput"]["tp_s_aggregate"]:,.0f}',
        )
    print('  slow tail (top 8 by ns):')
    for r in sorted(rows, key=lambda r: -_number(r, 'ns'))[:8]:
        print(f'    {_number(r, "ns"):9.0f} ns  {_number(r, "cells"):5.1f} cells  '
              f'{_label(r, "isa"):9s} {_label(r, "asm")}')
    if path:
        print(f'  logged: {path.relative_to(_HERE.parent)}')

    # ns is machine-dependent: never a hard assertion here. The deterministic
    # ratchet (test_instruction_cost_ratchet) is the gate; this run's numbers
    # are recorded in perf.log.d/ for tracking progression over time.
    assert summary['ns']['min'] > 0


if __name__ == '__main__':  # allow running as a plain script for the ns report
    sys.exit(pytest.main([__file__, '-k', 'bench', '-s', '-q']))
