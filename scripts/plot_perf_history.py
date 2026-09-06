#!/usr/bin/env python3
"""scripts/plot_perf_history.py
=================================
Plot the perf progression recorded by ``tests/test_perf_ratchet.py``.

Every run of the ratchet gate (and of the timing bench) drops one JSON file in
``tests/perf.log.d/``. This script draws **one subplot per ISA**, each showing
the wall-clock cost of a taint propagation (ns per ``circuit.evaluate``) at
several percentiles, and marks every run at which the *deterministic* work of
that ISA moved with a **vertical line**.

That separation is the point of the figure:

  * a step in the ns curves **with** a vertical line  -> the taint formulas got
    cheaper (fewer cell re-executions / assignments / AST nodes);
  * a step **without** a vertical line                -> the engine (hot path,
    C kernel, frame recycling, ...) got faster on the same formulas.

Deterministic metrics used as change detectors (summed over the ISA's
instructions, so any per-instruction improvement registers even when the
percentiles do not move):

  cells    native cell re-executions per evaluate      (every run)
  assigns  top-level taint assignments the rule emits  (every run)
  nodes    total AST nodes across the rule expressions (runs since 140bdb1)

Change detection always runs on the per-ISA *common cohort* (the instructions
present in every plotted run of that ISA), so growing the instruction bank
never registers as a work change. ``--cohort`` additionally restricts the
plotted ns curves to that cohort.

Schema drift, and how it is handled
-----------------------------------
* ``ns`` exists on ``timing`` runs only (~1 in 5), so the ns curves are sparse;
  their markers are the real measurements. ``ratchet`` runs still contribute
  their deterministic work, which is what gives the vertical lines their
  precise attribution to a commit.
* ``nodes`` only exists from ``20260905T154726Z-140bdb1`` onward, so it cannot
  produce a vertical line before that run. Nothing is faked.
* ``timing`` runs record ``nodes`` per instruction but omit it from their
  ``summary``; everything here is recomputed from the ``instr[]`` rows, which
  recovers it (and reproduces the logged ``summary`` numbers exactly).
* The instruction bank grew 87 -> 1535 -> 1559 -> 1569 forms. The 87-form era
  used different asm spellings and is dropped by default (``--include-legacy``).
  The two SIMD ISAs appear later, which is why their curves start later.
* **Every run is one x slot** -- nothing is averaged or collapsed per commit,
  because a commit was often measured many times against a dirty tree and the
  runs disagree. Two runs of 39c1998, for instance, report AMD64 cells 1284 and
  3560; only by seeing both can you tell one of them was taken mid-edit. Runs
  made with the C kernel disabled (``use_c: false``) are plotted too, shaded in
  red, but never generate a change rule (``--exclude-no-c`` drops them
  entirely).

Usage
-----
    .venv/bin/python scripts/plot_perf_history.py
    .venv/bin/python scripts/plot_perf_history.py --ncols 1 --show-increases
    .venv/bin/python scripts/plot_perf_history.py --cohort --csv perf.csv
    .venv/bin/python scripts/plot_perf_history.py --y tp_s --pcts 50,90,99,100
    .venv/bin/python scripts/plot_perf_history.py --show-increases   # see run-to-run wobble
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use('Agg')  # headless: this script only ever writes files
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

_HERE = Path(__file__).resolve().parent
DEFAULT_LOG_DIR = _HERE.parent / 'tests' / 'perf.log.d'

# What can go on the y axis: metric -> (axis label, log scale?).
Y_METRICS: dict[str, tuple[str, bool]] = {
    'ns': ('ns / evaluate', True),
    'tp_s': ('taint-props / s', True),
}
# Deterministic work metrics -> the colour of their vertical change lines.
WORK_METRICS: dict[str, str] = {'cells': '#c00000', 'assigns': '#006d2c', 'nodes': '#4a1486'}

# Stable colours so a given ISA keeps its colour across every figure.
ISA_COLORS: dict[str, str] = {
    'AMD64': '#1f77b4',
    'ARM64': '#2ca02c',
    'MIPS64BE': '#d62728',
    'PPC32BE': '#9467bd',
    'RISCV64': '#8c564b',
    'SPARC32BE': '#7f7f7f',
    'AMD64_SIMD': '#ff7f0e',
    'ARM64_SIMD': '#17becf',
    'ALL': '#000000',
}
# p50 solid, p90 dash-dot, p99 dashed, p100 dotted; anything else falls back.
PCT_STYLES: dict[int, tuple[str, float]] = {50: ('-', 1.8), 90: ('-.', 1.2), 99: ('--', 1.3), 100: (':', 1.3)}

# Runs below this instruction count are the pre-bank (87-form) era whose asm
# spellings do not match anything later; dropped unless --include-legacy.
LEGACY_MAX_N = 1000

XLABEL = 'run (linear in run index; labels are the commit of that run, * = dirty tree)'


def cpu_label() -> str:
    """CPU model + clock, e.g. 'AMD Ryzen 7 5700U @ 1.78 GHz (boost off)'.

    Better provenance than a hostname: it states the conditions the numbers were
    produced under, and leaks nothing about the machine's owner.
    """
    model = ''
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                model = line.split(':', 1)[1].strip()
                break
    except OSError:
        pass
    model = re.sub(r'\s+with\s+.*$', '', model)          # drop "with Radeon Graphics"
    model = re.sub(r'\((?:R|TM)\)|\bCPU\b', '', model).strip()
    parts = [model or 'unknown CPU']
    try:
        khz = int(Path('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq').read_text())
        parts.append(f'@ {khz / 1e6:.2f} GHz')
    except (OSError, ValueError):
        pass
    try:
        if Path('/sys/devices/system/cpu/cpufreq/boost').read_text().strip() == '0':
            parts.append('(boost off)')
    except OSError:
        pass
    return ' '.join(parts)


def pct(vals: list[float], p: float) -> float:
    """Percentile, byte-identical to ``_pct`` in tests/test_perf_ratchet.py."""
    if not vals:
        return 0.0
    s = sorted(vals)
    k = min(len(s) - 1, round(p / 100 * (len(s) - 1)))
    return s[k]


@dataclass
class Run:
    """One perf.log.d file: its provenance plus the per-instruction rows by ISA."""

    path: Path
    ts: datetime
    kind: str
    git: str
    dirty: bool
    host: str
    python: str
    use_c: bool
    n_instr: int
    by_isa: dict[str, list[dict[str, Any]]]
    # Instructions of the per-ISA common cohort only; the change detector reads
    # these so that growing the bank never looks like a work change.
    cohort_by_isa: dict[str, list[dict[str, Any]]]

    @property
    def label(self) -> str:
        return f'{self.git}{"*" if self.dirty else ""}'

    def values(self, isa: str, metric: str, *, cohort: bool = False) -> list[float]:
        """All per-instruction values of ``metric`` for ``isa`` ('ALL' = pooled)."""
        src = self.cohort_by_isa if cohort else self.by_isa
        rows = [r for rs in src.values() for r in rs] if isa == 'ALL' else src.get(isa, [])
        if metric == 'tp_s':  # derived; 1e9/ns, same as the logged tp_s
            return [1e9 / r['ns'] for r in rows if r.get('ns')]
        return [float(r[metric]) for r in rows if metric in r]

    def work(self, isa: str, metric: str) -> float | None:
        """Total deterministic work of the ISA's cohort, or None if unmeasured."""
        vals = self.values(isa, metric, cohort=True)
        return sum(vals) if vals else None


def load_runs(
    log_dir: Path, *, include_legacy: bool, include_no_c: bool, host: str | None,
) -> tuple[list[Run], list[str]]:
    """Read every *.json in ``log_dir`` into chronologically sorted Runs.

    Also returns one note per exclusion, so a filtered run is never silent.
    """
    runs: list[Run] = []
    dropped: dict[str, int] = {}
    for path in sorted(log_dir.glob('*.json')):
        try:
            d = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f'skip {path.name}: {exc}', file=sys.stderr)
            continue
        rows = d.get('instr') or []
        if not rows or 'ts' not in d:
            continue
        if not include_legacy and d.get('n_instr', len(rows)) <= LEGACY_MAX_N:
            dropped['pre-bank corpus (--include-legacy to keep)'] = (
                dropped.get('pre-bank corpus (--include-legacy to keep)', 0) + 1
            )
            continue
        if not include_no_c and d.get('use_c') is False:
            dropped['C kernel disabled (--exclude-no-c)'] = dropped.get('C kernel disabled (--exclude-no-c)', 0) + 1
            continue
        if host and d.get('host') != host:
            dropped[f'other host (--host {host})'] = dropped.get(f'other host (--host {host})', 0) + 1
            continue
        by_isa: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            by_isa.setdefault(r['isa'], []).append(r)
        runs.append(
            Run(
                path=path,
                ts=datetime.fromisoformat(d['ts']),
                kind=d.get('kind', '?'),
                git=d.get('git', '?'),
                dirty=bool(d.get('dirty')),
                host=d.get('host', '?'),
                python=d.get('python', '?'),
                use_c=bool(d.get('use_c', True)),
                n_instr=d.get('n_instr', len(rows)),
                by_isa=by_isa,
                cohort_by_isa={},
            ),
        )
    runs.sort(key=lambda r: r.ts)
    return runs, [f'{n} run(s) excluded: {why}' for why, n in sorted(dropped.items())]


def common_cohort(runs: list[Run]) -> dict[str, set[str]]:
    """Per ISA, the instructions present in *every* run that has that ISA."""
    common: dict[str, set[str]] = {}
    for run in runs:
        for isa, rows in run.by_isa.items():
            asms = {r['asm'] for r in rows}
            common[isa] = asms if isa not in common else (common[isa] & asms)
    return common


def attach_cohorts(runs: list[Run], *, plot_cohort: bool) -> dict[str, int]:
    """Fill ``cohort_by_isa``; with ``plot_cohort`` also restrict the plotted rows."""
    common = common_cohort(runs)
    for run in runs:
        run.cohort_by_isa = {
            isa: [r for r in rows if r['asm'] in common[isa]] for isa, rows in run.by_isa.items()
        }
        if plot_cohort:
            run.by_isa = {isa: list(rows) for isa, rows in run.cohort_by_isa.items()}
    return {isa: len(s) for isa, s in common.items()}


@dataclass
class Change:
    """A run at which an ISA's total deterministic work moved."""

    index: int
    run: Run
    isa: str
    metric: str
    before: float
    after: float

    @property
    def is_decrease(self) -> bool:
        return self.after < self.before

    @property
    def rel(self) -> float:
        return (self.after - self.before) / self.before * 100 if self.before else 0.0

    def annotation(self) -> str:
        return f'{self.metric} {self.rel:+.1f}%'


def work_changes(runs: list[Run], isa: str, metrics: list[str], min_change: float = 0.0) -> list[Change]:
    """Every run where the ISA's summed cohort work changed, per metric.

    ``min_change`` (percent) drops sub-threshold wobble; the running baseline
    still follows every value, so small steps accumulate into a later crossing
    instead of being lost.
    """
    out: list[Change] = []
    for metric in metrics:
        prev: float | None = None
        for i, run in enumerate(runs):
            if not run.use_c:  # different kernel: plotted, but not a formula change
                continue
            total = run.work(isa, metric)
            if total is None:  # metric did not exist yet in this run
                continue
            if prev is not None and total != prev:
                ch = Change(index=i, run=run, isa=isa, metric=metric, before=prev, after=total)
                if abs(ch.rel) >= min_change:
                    out.append(ch)
            prev = total
    return sorted(out, key=lambda c: (c.index, c.metric))


def series(runs: list[Run], isa: str, metric: str, p: float) -> tuple[list[int], list[float]]:
    """(x indices, y values) for one ISA/metric/percentile, skipping runs lacking it."""
    xs: list[int] = []
    ys: list[float] = []
    for i, run in enumerate(runs):
        vals = run.values(isa, metric)
        if vals:
            xs.append(i)
            ys.append(pct(vals, p))
    return xs, ys


def isas_of(runs: list[Run]) -> list[str]:
    """Every ISA seen, ordered: known ones first (scalar then SIMD), then the rest."""
    seen = {isa for run in runs for isa in run.by_isa}
    known = [i for i in ISA_COLORS if i in seen and i != 'ALL']
    return known + sorted(seen - set(known))


def color_of(isa: str, fallback: dict[str, str]) -> str:
    if isa not in ISA_COLORS and isa not in fallback:
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        fallback[isa] = cycle[len(fallback) % len(cycle)]
    return ISA_COLORS.get(isa, fallback.get(isa, '#333333'))


def draw_changes(ax: Any, changes: list[Change], metrics: list[str], *, show_increases: bool) -> None:
    """Vertical rule + rotated label per work change; labels staggered by metric."""
    for ch in changes:
        if not ch.is_decrease and not show_increases:
            continue
        c = WORK_METRICS[ch.metric]
        ax.axvline(
            ch.index,
            color=c,
            ls='-' if ch.is_decrease else (0, (1, 2)),
            lw=1.3 if ch.is_decrease else 0.9,
            alpha=0.65 if ch.is_decrease else 0.4,
            zorder=0,
        )
        # Stagger the metrics vertically so simultaneous changes stay legible.
        y = 0.975 - 0.30 * metrics.index(ch.metric)
        ax.annotate(
            ch.annotation(),
            xy=(ch.index, y),
            xycoords=('data', 'axes fraction'),
            rotation=90,
            fontsize=5.8,
            color=c,
            alpha=0.95 if ch.is_decrease else 0.55,
            ha='right',
            va='top',
            annotation_clip=False,
        )


def set_xaxis(ax: Any, runs: list[Run], *, max_ticks: int, bottom: bool, mode: str) -> None:
    """Label the (linear) run-index axis with short shas.

    ``even``   ticks at a fixed run stride, so spacing on screen is uniform;
               a commit measured many times simply spans several ticks.
    ``commit`` one tick at the first run of each commit: exact commit
               boundaries, but visually bunched, since commits have between 1
               and a dozen runs each.
    """
    if mode == 'commit':
        firsts: list[tuple[int, str]] = []
        prev = None
        for i, run in enumerate(runs):
            if run.git != prev:
                firsts.append((i, run.label))
                prev = run.git
        step = max(1, math.ceil(len(firsts) / max_ticks))
        ticks = firsts[::step]
    else:
        step = max(1, math.ceil(len(runs) / max_ticks))
        ticks = [(i, runs[i].label) for i in range(0, len(runs), step)]
    ax.set_xlim(-0.5, len(runs) - 0.5)
    ax.set_xticks([i for i, _ in ticks])
    ax.set_xticklabels([lbl for _, lbl in ticks] if bottom else [], rotation=90, fontsize=6.5)
    # sharex hides labels on every non-bottom-row axes; a short column's last
    # panel is not on the bottom row, so re-enable them explicitly.
    ax.tick_params(labelbottom=bottom)


def mark_no_c(ax: Any, runs: list[Run]) -> None:
    """Shade runs made with the C kernel disabled (only present with --include-no-c)."""
    for i, run in enumerate(runs):
        if not run.use_c:
            ax.axvspan(i - 0.4, i + 0.4, color='#d62728', alpha=0.10, lw=0)


def draw_isa_panel(
    ax: Any,
    runs: list[Run],
    isa: str,
    y_metric: str,
    pcts: list[float],
    work: list[str],
    *,
    show_increases: bool,
    min_change: float,
    fallback: dict[str, str],
) -> None:
    """One ISA: its percentile curves plus the vertical work-change rules."""
    color = color_of(isa, fallback)
    label, want_log = Y_METRICS[y_metric]
    drawn: list[float] = []
    for p in pcts:
        xs, ys = series(runs, isa, y_metric, p)
        if not xs:
            continue
        ls, lw = PCT_STYLES.get(int(p), ('-', 1.2))
        # The y metric lives on timing runs only, so mark the real measurements.
        ms = 3.0 if len(xs) > 0.4 * len(runs) else 4.5
        ax.plot(xs, ys, ls, color=color, lw=lw, marker='.', ms=ms, alpha=0.95, zorder=3)
        drawn += ys

    if work:
        draw_changes(ax, work_changes(runs, isa, work, min_change), work, show_increases=show_increases)
    mark_no_c(ax, runs)

    if not drawn:
        ax.text(0.5, 0.5, f'no {y_metric} data', ha='center', va='center', transform=ax.transAxes, color='#888888')
    elif want_log and any(v > 0 for v in drawn):
        ax.set_yscale('log')
        ax.yaxis.set_minor_formatter(NullFormatter())  # decades only; panels are narrow
        lo, hi = min(v for v in drawn if v > 0), max(drawn)
        ax.set_ylim(lo / 1.9, hi * 2.6)  # headroom for the rotated change labels
    ax.set_ylabel(label, fontsize=8.5)
    ax.grid(True, which='major', axis='y', alpha=0.18, lw=0.5)
    ax.grid(True, axis='x', alpha=0.08, lw=0.5)
    ax.tick_params(labelsize=8)


def isa_title(runs: list[Run], isa: str, sizes: dict[str, int]) -> str:
    n_now = max((len(r.by_isa.get(isa, [])) for r in runs), default=0)
    cohort_n = sizes.get(isa, 0)
    return f'{isa}  ({n_now} instr' + (f', cohort {cohort_n}' if cohort_n != n_now else '') + ')'


def build_figure(
    runs: list[Run],
    isas: list[str],
    y_metric: str,
    pcts: list[float],
    work: list[str],
    sizes: dict[str, int],
    *,
    ncols: int,
    show_increases: bool,
    min_change: float,
    x_ticks: str,
    title: str,
) -> Any:
    """One subplot per ISA, laid out on an ``ncols``-wide grid with a shared x axis."""
    fallback: dict[str, str] = {}
    nrows = math.ceil(len(isas) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.6 * ncols, 2.75 * nrows + 1.8),
        squeeze=False,
        sharex=True,
        layout='constrained',
    )
    flat = [ax for row in axes for ax in row]
    for ax, isa in zip(flat, isas, strict=False):
        draw_isa_panel(
            ax, runs, isa, y_metric, pcts, work,
            show_increases=show_increases, min_change=min_change, fallback=fallback,
        )
        ax.set_title(isa_title(runs, isa, sizes), fontsize=9.5, color=color_of(isa, fallback), fontweight='bold')
    for ax in flat[len(isas):]:
        ax.set_visible(False)
    # Bottom-most *visible* axes of each column carries the sha tick labels.
    for ci in range(ncols):
        col = [(ri, axes[ri][ci]) for ri in range(nrows) if axes[ri][ci].get_visible()]
        for ri, ax in col:
            set_xaxis(ax, runs, max_ticks=26, bottom=(ri, ax) == col[-1], mode=x_ticks)

    handles = [
        Line2D([], [], color='#444444', lw=1.5, ls=PCT_STYLES.get(int(p), ('-', 1.2))[0], label=f'p{int(p)}')
        for p in pcts
    ]
    handles += [Line2D([], [], color=WORK_METRICS[m], lw=1.6, label=f'{m} decreased') for m in work]
    if show_increases:
        handles += [Line2D([], [], color='#444444', lw=1.0, ls=(0, (1, 2)), label='(dotted) work increased')]
    fig.suptitle(title, fontsize=11)
    fig.legend(
        handles=handles,
        loc='outside lower center',
        ncol=min(7, len(handles)),
        fontsize=8,
        frameon=False,
        title=f'{XLABEL}\nvertical rule = taint-formula change; a step without one = engine/hot-path change',
        title_fontsize=8.5,
    )
    return fig


def write_csv(
    path: Path, runs: list[Run], isas: list[str], y_metric: str, pcts: list[float], work: list[str],
    *, cpu: str = '',
) -> None:
    """Tidy dump: the plotted percentiles plus the summed work behind the rules."""
    head = ['ts', 'git', 'dirty', 'kind', 'cpu', 'python', 'use_c', 'isa', 'n', 'cohort_n', 'metric', 'stat', 'value']
    with path.open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(head)
        for run in runs:
            for isa in [*isas, 'ALL']:
                prov = [run.ts.isoformat(), run.git, int(run.dirty), run.kind, cpu, run.python, int(run.use_c)]
                counts = [isa, len(run.values(isa, 'cells')), len(run.values(isa, 'cells', cohort=True))]
                vals = run.values(isa, y_metric)
                for p in pcts:
                    if vals:
                        w.writerow([*prov, *counts, y_metric, f'p{int(p)}', pct(vals, p)])
                for m in work:
                    total = run.work(isa, m)
                    if total is not None:
                        w.writerow([*prov, *counts, m, 'sum', total])


def report(
    runs: list[Run], isas: list[str], y_metric: str, pcts: list[float], work: list[str], min_change: float = 0.0,
) -> None:
    """First -> last percentile per ISA, then the work changes behind the rules."""
    print(f'\n{"isa":11s} {"stat":>5s} {"first":>13s} {"last":>13s} {"change":>9s}   points')
    for isa in isas:
        for p in pcts:
            ys = series(runs, isa, y_metric, p)[1]
            if not ys:
                continue
            first, last = ys[0], ys[-1]
            rel = f'{(last - first) / first * 100:+.1f}%' if first else 'n/a'
            print(f'{isa:11s} {"p" + str(int(p)):>5s} {first:13.2f} {last:13.2f} {rel:>9s}   {len(ys)}')

    if not work:
        return
    print(
        f"\ndeterministic work changes (summed over each ISA's common cohort)\n"
        f'{"isa":11s} {"commit":10s} {"metric":8s} {"before":>10s} {"after":>10s} {"change":>9s}',
    )
    for isa in isas:
        for ch in work_changes(runs, isa, work, min_change):
            tag = '' if ch.is_decrease else '  (INCREASE)'
            print(
                f'{isa:11s} {ch.run.label:10s} {ch.metric:8s} {ch.before:10.0f} {ch.after:10.0f} '
                f'{ch.rel:+8.1f}%{tag}',
            )


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--log-dir', type=Path, default=DEFAULT_LOG_DIR, help='perf.log.d directory')
    ap.add_argument('--out', type=Path, default=None, help='output PNG (default <log-dir>/perf-history.png)')
    ap.add_argument('--y', default='ns', choices=tuple(Y_METRICS), help='what the curves show (default ns)')
    ap.add_argument('--pcts', default='50,99,100', help='comma list of percentiles (default 50,99,100)')
    ap.add_argument(
        '--work', default=','.join(WORK_METRICS),
        help=f'work metrics drawn as vertical change rules, from {",".join(WORK_METRICS)} ("" = none)',
    )
    ap.add_argument('--show-increases', action='store_true', help='also rule work increases (dotted)')
    ap.add_argument(
        '--min-change', type=float, default=0.0,
        help='hide work changes smaller than this percent (default 0 = show every change)',
    )
    ap.add_argument('--ncols', type=int, default=2, help='ISA subplots per row (default 2)')
    ap.add_argument(
        '--x-ticks', choices=('even', 'commit'), default='even',
        help='x ticks at a uniform run stride (default) or at each commit boundary',
    )
    ap.add_argument('--isa', default=None, help='comma list of ISAs to keep (default: all)')
    ap.add_argument('--cohort', action='store_true', help='also restrict the plotted curves to the common cohort')
    ap.add_argument('--include-legacy', action='store_true', help='keep the pre-bank 87-instruction runs')
    ap.add_argument(
        '--exclude-no-c', action='store_true',
        help='drop runs made with the C kernel disabled (kept and shaded red by default)',
    )
    ap.add_argument('--host', default=None, help='only runs from this host (default: all)')
    ap.add_argument('--since', default=None, help='only runs whose file name sorts >= this (e.g. 20260905)')
    ap.add_argument('--until', default=None, help='only runs whose file name sorts <= this')
    ap.add_argument('--kind', choices=('all', 'ratchet', 'timing'), default='all', help='restrict to one log kind')
    ap.add_argument(
        '--order-file', type=Path, default=None,
        help='file of commit shas (oldest first): order the x axis by history rather than by timestamp',
    )
    ap.add_argument('--csv', type=Path, default=None, help='also dump the plotted values as tidy CSV')
    ap.add_argument(
        '--cpu', default=None,
        help='CPU label for the title and CSV (default: auto-detected model + clock). '
             'The machine name is never shown; the raw logs still record it.',
    )
    ap.add_argument('--dpi', type=int, default=140)
    ap.add_argument('--quiet', action='store_true', help='no stdout summary table')
    return ap.parse_args(argv)


def order_by_commits(runs: list[Run], order_file: Path) -> list[Run]:
    """Sort runs by position in a commit list rather than by wall-clock time.

    A parallel sweep hands commits to workers round-robin, so a later commit can
    finish before an earlier one. The x axis must follow history, not the clock.
    """
    shas = [ln.strip() for ln in order_file.read_text().splitlines() if ln.strip()]
    pos = {sha: i for i, sha in enumerate(shas)}

    def key(run: Run) -> tuple[int, datetime]:
        for sha, i in pos.items():
            if sha.startswith(run.git):
                return (i, run.ts)
        return (len(shas), run.ts)  # unknown commits sort to the end

    return sorted(runs, key=key)


def select_isas(runs: list[Run], wanted: str | None) -> list[str]:
    """ISAs to plot; empty (falsy) if an explicit --isa filter matched nothing."""
    isas = isas_of(runs)
    if not wanted:
        return isas
    want = {w.strip() for w in wanted.split(',') if w.strip()}
    kept = [i for i in isas if i in want]
    if not kept:
        print(f'no ISA matched {sorted(want)}; available: {isas}', file=sys.stderr)
    return kept


def select_runs(args: argparse.Namespace) -> tuple[list[Run], list[str]]:
    """Load the log dir and apply every run-level filter from the CLI."""
    runs, notes = load_runs(
        args.log_dir, include_legacy=args.include_legacy, include_no_c=not args.exclude_no_c, host=args.host,
    )
    if args.since:
        runs = [r for r in runs if r.path.name >= args.since]
    if args.until:
        runs = [r for r in runs if r.path.name <= args.until]
    if args.kind != 'all':
        runs = [r for r in runs if r.kind == args.kind]
    return runs, notes


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.log_dir.is_dir():
        print(f'no such log dir: {args.log_dir}', file=sys.stderr)
        return 2

    runs, notes = select_runs(args)
    for note in notes:  # never filter a run silently
        print(note, file=sys.stderr)
    if not runs:
        print(f'no runs matched in {args.log_dir}', file=sys.stderr)
        return 1

    if args.order_file:
        runs = order_by_commits(runs, args.order_file)
    sizes = attach_cohorts(runs, plot_cohort=args.cohort)
    isas = select_isas(runs, args.isa)
    if not isas:
        return 1

    pcts = [float(p) for p in args.pcts.split(',') if p.strip()]
    work = [m.strip() for m in args.work.split(',') if m.strip()]
    unknown = [m for m in work if m not in WORK_METRICS]
    if unknown:
        print(f'unknown work metric(s) {unknown}; known: {list(WORK_METRICS)}', file=sys.stderr)
        return 2
    if args.ncols < 1:
        print('--ncols must be >= 1', file=sys.stderr)
        return 2

    # min/max, not first/last: --order-file sorts by history, not by clock
    span = f'{min(r.ts for r in runs):%Y-%m-%d %H:%M} .. {max(r.ts for r in runs):%Y-%m-%d %H:%M} UTC'
    cpu = args.cpu or cpu_label()
    curves = 'cohort-only curves' if args.cohort else 'curves as recorded'
    title = f'microtaint taint-propagation perf history  |  {len(runs)} runs  |  {span}  |  {cpu}  |  {curves}'

    out = args.out or args.log_dir / 'perf-history.png'
    fig = build_figure(
        runs, isas, args.y, pcts, work, sizes,
        ncols=args.ncols, show_increases=args.show_increases, min_change=args.min_change,
        x_ticks=args.x_ticks, title=title,
    )
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    print(f'wrote {out}')

    if args.csv:
        write_csv(args.csv, runs, isas, args.y, pcts, work, cpu=cpu)
        print(f'wrote {args.csv}')

    if not args.quiet:
        report(runs, isas, args.y, pcts, work, args.min_change)
    return 0


if __name__ == '__main__':
    sys.exit(main())
