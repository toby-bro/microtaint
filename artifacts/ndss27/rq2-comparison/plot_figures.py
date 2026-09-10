"""
plot_figures.py  --  the paper's five evaluation figures, from a benchmark run.

Usage:
    python plot_figures.py REPORT.json [overhead_results.json]

Output files:
    fig_unsoundness.pdf      Figure 4, RQ2: unsound cases per engine
    fig_precision.pdf        Figure 5, RQ3: sound%, exact%, mean Jaccard
    fig_perf_latency.pdf     Figure 6, RQ4: p50 and p99 per-step latency
    fig_perf_throughput.pdf  Figure 7, RQ4: throughput per engine
    fig_overhead.pdf         Figure 8, RQ5: wall/CPU time and peak RSS

The overhead figure reads rq5-overhead/overhead_results.json; the other four
read the benchmark report.
"""

# Experiment script, not library code: see artifacts/ndss27/README.md,
# "Lint and type checking", for why annotations are not required here.
# mypy: disable-error-code="no-untyped-def, no-untyped-call, type-arg"

import json
import sys

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------- #
# Load metrics from the merged JSON report                                      #
# --------------------------------------------------------------------------- #
REPORT_PATH = sys.argv[1] if len(sys.argv) > 1 else 'report_merged_fixed_microtaint.json'
OVERHEAD_PATH = sys.argv[2] if len(sys.argv) > 2 else '../rq5-overhead/overhead_results.json'
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

def _gt(tool):
    """Return metrics.ground_truth.per_tool[tool]."""
    return REPORT['metrics']['ground_truth']['per_tool'][tool]

def _pt(tool):
    """Return metrics.per_tool[tool]."""
    return REPORT['metrics']['per_tool'][tool]


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
    with open(OVERHEAD_PATH) as f:
        ovh = json.load(f)
    keys = ['native', 'qiling-only', 'microtaint-all']
    configs = ['native', 'qiling-only', 'MicroTaint-all']
    wall_s = [ovh[k]['wall_s'] for k in keys]
    cpu_s = [ovh[k]['user_cpu_s'] + ovh[k]['sys_cpu_s'] for k in keys]
    rss_mib = [ovh[k]['peak_rss_mib'] for k in keys]

    x = np.arange(len(configs))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.2, 2.5 * PLOT_SCALE))

    # ---- left: wall and CPU time ----
    colors = [OTHER_COLOR, OTHER_COLOR, MICROTAINT_COLOR]
    ax1.bar(x - width / 2, wall_s, width, label='Wall (s)', color=colors, edgecolor='black', linewidth=0.6)
    ax1.bar(x + width / 2, cpu_s, width, label='CPU (s)', color=colors, edgecolor='black', linewidth=0.6, hatch='//')
    ax1.set_ylabel('Time (s)', fontsize=10)
    ax1.set_title('Wall and CPU time')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=20, ha='right', fontsize=10)
    ax1.legend(fontsize=10)

    # ---- right: peak RSS ----
    ax2.bar(configs, rss_mib, color=colors, edgecolor='black', linewidth=0.6)
    ax2.set_ylabel('Peak RSS (MiB)', fontsize=10)
    ax2.set_title('Peak memory usage')
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels(configs, rotation=20, ha='right', fontsize=10)

    for ax in (ax1, ax2):
        ax.set_ylim(0)

    fig.tight_layout()
    save(fig, 'fig_overhead.pdf')


# --------------------------------------------------------------------------- #
# Run all plots                                                                 #
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    print(f'Generating plots from {REPORT_PATH} ...')
    plot_unsoundness()
    plot_precision()
    plot_perf_latency()
    plot_perf_throughput()
    plot_overhead()
    print('Done.')
