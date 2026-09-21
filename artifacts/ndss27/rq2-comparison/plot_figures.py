"""
plot_figures.py  --  the paper's five evaluation figures, from a benchmark run.

Usage:
    python plot_figures.py [REPORT.json] [overhead_results.json]
    python plot_figures.py --report REPORT.json --overhead overhead_results.json
    python plot_figures.py --overhead overhead_results.json   # RQ5 figure only

Output files:
    fig_unsoundness.pdf      Figure 4, RQ2: unsound cases per engine
    fig_precision.pdf        Figure 5, RQ3: sound%, exact%, mean Jaccard
    fig_perf_latency.pdf     Figure 6, RQ4: p50 and p99 per-step latency
    fig_perf_throughput.pdf  Figure 7, RQ4: throughput per engine
    fig_overhead.pdf         Figure 8, RQ5: wall/CPU time and peak RSS

The overhead figure reads the ladder JSON next to the overhead results; the
other four read the benchmark report.  The two inputs are independent, so a
run with no engine comparison (--no-baselines skips it) still gets its RQ5
figure instead of nothing at all.
"""

# Experiment script, not library code: see artifacts/ndss27/README.md,
# "Lint and type checking", for why annotations are not required here.
# mypy: disable-error-code="no-untyped-def, no-untyped-call, type-arg"

import json
import os
import sys

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------------- #
# Load metrics from the merged JSON report                                      #
# --------------------------------------------------------------------------- #
def _parse_argv(argv):
    """(report, overhead) from the named flags, the positionals, or both.

    The positional form is what `run-all.sh` and the README have always used.
    The named form exists so the report can be left out: a --no-baselines run
    has no engine comparison but does have an overhead ladder.
    """
    report = overhead = None
    rest = []
    it = iter(argv)
    for a in it:
        if a in ('--report', '--overhead'):
            try:
                v = next(it)
            except StopIteration:
                raise SystemExit(f'{a} needs a path') from None
            if a == '--report':
                report = v
            else:
                overhead = v
        elif a in ('-h', '--help'):
            raise SystemExit(__doc__)
        elif a.startswith('-'):
            raise SystemExit(f'unknown option: {a}')
        else:
            rest.append(a)
    if rest and report is None:
        report = rest.pop(0)
    if rest and overhead is None:
        overhead = rest.pop(0)
    if rest:
        raise SystemExit(f'unexpected arguments: {" ".join(rest)}')
    return report, overhead


REPORT_PATH, OVERHEAD_PATH = _parse_argv(sys.argv[1:])
# No default for the overhead path.  It used to fall back to
# ../rq5-overhead/overhead_results.json, so plotting one run's report picked up
# whatever ladder happened to be in the tree and wrote a figure 8 that belonged
# to a different run.  A figure comes from a named input or not at all.
# REPORT stays None when there is no engine comparison to plot.  The four
# figures that need it are skipped by name rather than crashing on load, so a
# reduced run still produces the figure it did measure.
REPORT = None
if REPORT_PATH is not None:
    with open(REPORT_PATH) as _f:
        REPORT = json.load(_f)

# Map JSON tool keys → display names used in the figures
DISPLAY_NAME = {
    'microtaint': 'MicroTaint',
    'angr':       'angr',
    'maat':       'Maat',
    'triton':     'Triton',
    'panda':      'PANDA',
    'taintgrind': 'TaintGrind',
    'libdft64':   'libdft64',
}

def _report():
    """The loaded report, or a clear refusal if none was given."""
    if REPORT is None:
        raise SystemExit('this figure needs an engine-comparison report')
    return REPORT

def _gt(tool):
    """Return metrics.ground_truth.per_tool[tool]."""
    return _report()['metrics']['ground_truth']['per_tool'][tool]

def _pt(tool):
    """Return metrics.per_tool[tool]."""
    return _report()['metrics']['per_tool'][tool]


# --------------------------------------------------------------------------- #
# Shared style                                                                  #
# --------------------------------------------------------------------------- #
plt.rcParams.update(
    {
        'font.family': 'serif',
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.dpi': 300,
        'pdf.fonttype': 42,  # embed fonts for camera-ready
        'ps.fonttype': 42,
    },
)

MICROTAINT_COLOR = 'orange'
OTHER_COLOR = 'lightgray'
HATCHES = ['', '//', 'xx', '..', '\\\\', 'oo', '++']  # one per engine

# Global size knob: scales every figure's HEIGHT. Lower = shorter plots (the
# paper includes them at width=\\linewidth, so only height sets vertical
# footprint). Retune all plots from this one line.
PLOT_SCALE = 0.7  # cumulative height scale (was 0.8, then 0.8 again)


def bar_colors(engines, highlight='MicroTaint'):
    """Return a list of colors, highlighting MicroTaint."""
    return [MICROTAINT_COLOR if highlight.lower() in e.lower() else OTHER_COLOR for e in engines]


def save(fig, name):
    fig.savefig(name, bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)
    print(f'  saved {name}')


# --------------------------------------------------------------------------- #
# Figure 4 - RQ2, unsound cases per engine                                     #
# --------------------------------------------------------------------------- #
def plot_unsoundness():
    tool_keys = ['microtaint', 'angr', 'taintgrind', 'libdft64', 'triton', 'maat', 'panda']
    engines = [DISPLAY_NAME[t] for t in tool_keys]
    unsound = [_gt(t)['unsound_cases'] for t in tool_keys]

    fig, ax = plt.subplots(figsize=(4, 2.3 * PLOT_SCALE))
    colors = bar_colors(engines)
    bars = ax.bar(engines, unsound, color=colors, edgecolor='black', linewidth=0.6)

    ax.set_ylabel('Unsound cases')
    ax.set_xticks(range(len(engines)))
    ax.set_xticklabels(engines, rotation=15, ha='right')

    # label every bar (including 0)
    for bar, val in zip(bars, unsound, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            str(val),
            ha='center',
            va='bottom',
            fontsize=7,
        )

    ax.set_ylim(0, max(unsound) * 1.30)
    fig.tight_layout()
    save(fig, 'fig_unsoundness.pdf')


# --------------------------------------------------------------------------- #
# Figure 5 - RQ3, soundness and precision per engine                           #
# --------------------------------------------------------------------------- #
def plot_precision():
    # bit-precise engines first, then register-level
    tool_keys = ['microtaint', 'angr', 'maat', 'taintgrind', 'libdft64', 'triton', 'panda']
    engines = [DISPLAY_NAME[t] for t in tool_keys]
    sound_pct = [round(_gt(t)['soundness_rate'] * 100, 1) for t in tool_keys]
    exact_pct = [round(_gt(t)['exact_case_rate'] * 100, 1) for t in tool_keys]

    x = np.arange(len(engines))
    width = 0.35

    fig, ax = plt.subplots(figsize=(4, 2.2 * PLOT_SCALE))
    colors = bar_colors(engines)

    ax.bar(x - width / 2, sound_pct, width, label='Sound %', color=colors, edgecolor='black', linewidth=0.6)
    ax.bar(
        x + width / 2, exact_pct, width, label='Exact %', color=colors, edgecolor='black', linewidth=0.6, hatch='//',
    )

    ax.set_ylabel('Soundness/precision (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(engines, rotation=15, ha='right', fontsize=8)
    ax.set_ylim(0, 125)
    ax.axhline(100, color='gray', linewidth=0.5, linestyle='--')
    ax.legend(loc='lower left', fontsize=7)

    # divider between bit-precise and register-level groups
    ax.axvline(2.5, color='black', linewidth=0.7, linestyle=':')
    ax.text(0.9, 107, 'bit-precise', fontsize=7, color='dimgray')
    ax.text(3.3, 107, 'register-level', fontsize=7, color='dimgray')

    fig.tight_layout()
    save(fig, 'fig_precision.pdf')


# --------------------------------------------------------------------------- #
# Figure 6 - RQ4, per-step latency                                             #
# --------------------------------------------------------------------------- #
def plot_perf_latency():
    tool_keys = ['microtaint', 'panda', 'triton', 'maat', 'libdft64', 'taintgrind', 'angr']
    engines = [DISPLAY_NAME[t] for t in tool_keys]
    # PER-STEP, not per test: a sequence test is up to 32 instructions, so the
    # per-test tail measures sequence length rather than propagation cost.
    p50_us = [_pt(t)['latency_p50_per_instr_ms'] * 1000.0 for t in tool_keys]
    p99_us = [_pt(t)['latency_p99_per_instr_ms'] * 1000.0 for t in tool_keys]
    p100_us = [_pt(t)['latency_p100_per_instr_ms'] * 1000.0 for t in tool_keys]

    x = np.arange(len(engines))
    # Three bars where there were two, each two-thirds as wide, so the group
    # occupies the same space.
    width = 0.35 * 2 / 3

    fig, ax = plt.subplots(figsize=(5.5, 3.0 * PLOT_SCALE))
    colors = bar_colors(engines)

    ax.bar(x - width, p50_us, width, label='p50 (per step)', color=colors, edgecolor='black', linewidth=0.6)
    ax.bar(
        x, p99_us, width, label='p99 (per step)',
        color=colors, edgecolor='black', linewidth=0.6, hatch='//',
    )
    ax.bar(
        x + width, p100_us, width, label='p100 (per step)',
        color=colors, edgecolor='black', linewidth=0.6, hatch='xx',
    )

    ax.set_ylabel('Latency (µs, log scale)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(engines, rotation=15, ha='right', fontsize=10)
    ax.set_yscale('log')
    ax.legend(fontsize=10)

    fig.tight_layout()
    save(fig, 'fig_perf_latency.pdf')


# --------------------------------------------------------------------------- #
# Figure 7 - RQ4, throughput                                                   #
# --------------------------------------------------------------------------- #
def plot_perf_throughput():
    tool_keys = ['microtaint', 'panda', 'triton', 'maat', 'libdft64', 'taintgrind', 'angr']
    engines = [DISPLAY_NAME[t] for t in tool_keys]
    tps = [round(_pt(t)['throughput_per_s']) for t in tool_keys]

    fig, ax = plt.subplots(figsize=(4.5, 2.8 * PLOT_SCALE))
    colors = bar_colors(engines)
    bars = ax.bar(engines, tps, color=colors, edgecolor='black', linewidth=0.6)

    ax.set_ylabel('Throughput (tests/s, log scale)')
    ax.set_xticks(range(len(engines)))
    ax.set_xticklabels(engines, rotation=15, ha='right')
    ax.set_yscale('log')
    # extra headroom on log scale so the topmost label doesn't clip
    ax.set_ylim(top=max(tps) * 3.0)

    for bar, val in zip(bars, tps, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.08,
            f'{val:,}',
            ha='center',
            va='bottom',
            fontsize=8,
        )

    fig.tight_layout()
    save(fig, 'fig_perf_throughput.pdf')


# --------------------------------------------------------------------------- #
# Figure 8 - RQ5, end-to-end overhead                                          #
# --------------------------------------------------------------------------- #
def plot_overhead():
    """The overhead ladder: four rungs, wall time and peak memory.

    Reads rq5-overhead/overhead_ladder.json rather than overhead_results.json.
    The ladder is what makes the number attributable: `native` and
    `microtaint-all` alone say how slow, not whose cost it is, and the two rungs
    between them carry the answer.  `Qiling + hooks` is a pure-C per-instruction
    hook that reads four guest registers -- the plumbing any per-instruction
    dynamic analysis owes -- and `MicroTaint plumbing` is the real engine armed
    with nothing tainted, so no propagation runs.

    Time is LOG scale, deliberately: the range is 0.002 s to 6.968 s, a factor of
    4,115, and on a linear axis the native bar is invisible and Qiling+hooks
    nearly so, which erases exactly the two rungs the figure exists to show.

    `microtaint-none` is deliberately NOT a bar.  The detectors cost nothing
    measurable now, so a fifth bar the same height as the fourth would tell the
    reader that nothing was measured; the rung stays in the appendix table,
    where the exact numbers belong.
    Memory is linear, where the range is only about 6x and the growth is the
    point.  Exact values are in the appendix table rather than on the bars, which
    keeps this the same size as the two-panel figure it replaces.
    """
    ladder_path = os.path.join(os.path.dirname(OVERHEAD_PATH), 'overhead_ladder.json')
    with open(ladder_path) as f:
        lad = json.load(f)

    keys = ['native', 'c-codehook-regs', 'microtaint-plumbing', 'microtaint-all']
    labels = ['native', 'Qiling + hooks', 'MicroTaint plumbing', 'MicroTaint all']
    # Wall seconds of the measured phase: ql.run() for the emulated rungs, the
    # whole subprocess for `native`, which has no ql.run to isolate.
    wall_s = [lad['layers'][k]['run_s'] for k in keys]
    rss_mib = [lad['layers'][k].get('peak_rss_mib') or 0.0 for k in keys]
    colors = [OTHER_COLOR, OTHER_COLOR, MICROTAINT_COLOR, MICROTAINT_COLOR]
    x = np.arange(len(keys))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.2, 2.5 * PLOT_SCALE))

    ax1.bar(x, wall_s, 0.6, color=colors, edgecolor='black', linewidth=0.6)
    ax1.set_yscale('log')
    ax1.set_ylabel('Wall time (s)', fontsize=10)
    ax1.set_title('Time', fontsize=10)

    ax2.bar(x, rss_mib, 0.6, color=colors, edgecolor='black', linewidth=0.6)
    ax2.set_ylabel('Peak RSS (MiB)', fontsize=10)
    ax2.set_title('Peak memory', fontsize=10)
    ax2.set_ylim(0)

    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)

    fig.tight_layout()
    save(fig, 'fig_overhead.pdf')


# --------------------------------------------------------------------------- #
# Run all plots                                                                 #
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    made = []
    if REPORT is not None:
        print(f'Generating plots from {REPORT_PATH} ...')
        plot_unsoundness()
        plot_precision()
        plot_perf_latency()
        plot_perf_throughput()
        made += ['fig_unsoundness.pdf', 'fig_precision.pdf',
                 'fig_perf_latency.pdf', 'fig_perf_throughput.pdf']
    else:
        print('No engine-comparison report given: skipping figures 4 to 7.')

    if OVERHEAD_PATH is None:
        print('No --overhead given: skipping figure 8.')
    else:
        ladder = os.path.join(os.path.dirname(OVERHEAD_PATH), 'overhead_ladder.json')
        if os.path.exists(ladder):
            plot_overhead()
            made.append('fig_overhead.pdf')
        else:
            print(f'No overhead ladder at {ladder}: skipping figure 8.')

    # Producing nothing and exiting zero is how a plotting step disappears from
    # a run without anybody noticing.
    if not made:
        raise SystemExit('no figure could be generated: neither input was given')
    print('Done: ' + ', '.join(made))
