"""
plot_figures.py  --  the paper's five evaluation figures, from a benchmark run.

Usage:
    python plot_figures.py [REPORT.json] [overhead_results.json]
    python plot_figures.py --report REPORT.json --overhead overhead_results.json
    python plot_figures.py --overhead overhead_results.json   # RQ5 figure only

Output files:
    fig_unsoundness.pdf      Figure 2, RQ2: unsound cases per engine
    fig_precision.pdf        Figure 3, RQ3: sound%, exact%, mean Jaccard
    fig_perf_latency.pdf     Figure 4, RQ4: p50 and p99 per-step latency
    fig_perf_throughput.pdf  Figure 5, RQ4: throughput per engine
    fig_overhead.pdf         Figure 6, RQ5: wall/CPU time and peak RSS

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


def slot_x(n, slot):
    """Bar centres spaced `slot` apart instead of 1.0.

    Widening bars to close a gap changes how heavy the figure reads.  Moving the
    centres closer closes the same gap at the bar's original width, so `slot`
    minus the group's width IS the visible gap.
    """
    return np.arange(n) * slot


#: Axis units of clearance left either side of the outermost bar.
EDGE_MARGIN = 0.1

#: A legend that sits ABOVE the axes reads as a title unless it is visibly a
#: box.  A faint grey RULE around it says "key"; the inside stays white, so the
#: swatches keep their own ground and nothing competes with the bars.
LEGEND_BOX = {
    'frameon': True,
    'facecolor': 'white',
    'edgecolor': '0.75',
    'framealpha': 1.0,
    'borderpad': 0.35,
}

#: One scale for every figure, so no panel reads heavier than its neighbours.
#: The latency figure used to set its own 11pt y-label against everyone else's
#: 9pt, and the two legends disagreed at 7 and 9.
AXIS_LABEL_SIZE = 10
LEGEND_SIZE = 9

#: Point size of the value printed on top of a bar AND of the engine names
#: under the axis, so the two read at the same weight.  The throughput headroom
#: reads it too: a rotated label is as tall as this size makes it.
VALUE_LABEL_SIZE = 10


def slot_geometry(n, slot, group_w, margin=EDGE_MARGIN):
    """Axis limits that hug the bars, and the factor the FIGURE shrinks by.

    Closing the gaps shortens the axis.  Two things then have to move together
    or the bars change size:

      * the x-limits hug the content, so no dead margin appears inside the axes
      * the figure's WIDTH scales by the same factor, which holds
        pixels-per-axis-unit constant, so a bar keeps its width on the page

    The figure is therefore narrower than the column, and the paper includes it
    at `width = <factor> \\linewidth` rather than at full width.  Scaling the
    x-limits without the figure inflates the bars; scaling neither leaves the
    gap space as dead margin inside the axes.
    """
    span_old = (n - 1) + group_w + 2 * margin
    span_new = (n - 1) * slot + group_w + 2 * margin
    lo = -group_w / 2 - margin
    hi = (n - 1) * slot + group_w / 2 + margin
    return (lo, hi), span_new / span_old


def legend_handles(labels, hatches):
    """Legend swatches on a WHITE ground, differing only by hatch.

    The bars are coloured to highlight MicroTaint, so reusing the bar colours in
    the legend paints the p50 swatch orange and implies the series belongs to
    one engine.  The series is distinguished by HATCH, so the swatch carries the
    hatch on white and the text stays black.
    """
    return [
        plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black',
                      linewidth=0.6, hatch=h, label=lab)
        for lab, h in zip(labels, hatches, strict=True)
    ]


def bar_colors(engines, highlight='MicroTaint'):
    """Return a list of colors, highlighting MicroTaint."""
    return [MICROTAINT_COLOR if highlight.lower() in e.lower() else OTHER_COLOR for e in engines]


def save(fig, name):
    # metadata CreationDate=None drops the timestamp the PDF backend would
    # otherwise stamp in, which is the only part of matplotlib's output that
    # changes between two runs of the same script on the same data.  Without
    # it, regenerating a figure produces a different FILE for an identical
    # PICTURE, so every re-run shows up as a diff and the artifact cannot claim
    # its figures are reproducible.
    fig.savefig(name, bbox_inches='tight', pad_inches=0.01,
                metadata={'CreationDate': None})
    plt.close(fig)
    print(f'  saved {name}')


# --------------------------------------------------------------------------- #
# Figure 2 - RQ2, unsound cases per engine                                     #
# --------------------------------------------------------------------------- #
def plot_unsoundness():
    # Ordered by the quantity plotted, so the bars stay ranked when an engine's
    # count moves.  A hardcoded order silently mis-ranks the figure instead.
    tool_keys = ['microtaint', 'angr', 'taintgrind', 'libdft64', 'triton', 'maat', 'panda']
    tool_keys.sort(key=lambda t: _gt(t)['unsound_cases'])
    engines = [DISPLAY_NAME[t] for t in tool_keys]
    unsound = [_gt(t)['unsound_cases'] for t in tool_keys]

    fig, ax = plt.subplots(figsize=(4, 2.3 * PLOT_SCALE))
    colors = bar_colors(engines)
    slot, bw = 0.88, 0.8
    x = slot_x(len(engines), slot)
    xlim, kw = slot_geometry(len(engines), slot, bw)
    fig.set_size_inches(4 * kw, 2.3 * PLOT_SCALE)
    bars = ax.bar(x, unsound, width=bw, color=colors, edgecolor='black', linewidth=0.6)

    ax.set_ylabel('Unsound cases', fontsize=AXIS_LABEL_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(engines, rotation=30, ha='right', fontsize=VALUE_LABEL_SIZE)
    ax.set_xlim(*xlim)

    # label every bar (including 0)
    for bar, val in zip(bars, unsound, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            str(val),
            ha='center',
            va='bottom',
            fontsize=VALUE_LABEL_SIZE,
        )

    ax.set_ylim(0, max(unsound) * 1.30)
    fig.tight_layout()
    save(fig, 'fig_unsoundness.pdf')


# --------------------------------------------------------------------------- #
# Figure 3 - RQ3, soundness and precision per engine                           #
# --------------------------------------------------------------------------- #
def plot_precision():
    # bit-precise engines first, then register-level
    tool_keys = ['microtaint', 'angr', 'maat', 'taintgrind', 'libdft64', 'triton', 'panda']
    engines = [DISPLAY_NAME[t] for t in tool_keys]
    sound_pct = [round(_gt(t)['soundness_rate'] * 100, 1) for t in tool_keys]
    exact_pct = [round(_gt(t)['exact_case_rate'] * 100, 1) for t in tool_keys]

    # Bars at their original width; the slot closes the gap instead.
    slot, width = 0.80, 0.35
    x = slot_x(len(engines), slot)
    xlim, kw = slot_geometry(len(engines), slot, 2 * width)

    fig, ax = plt.subplots(figsize=(4 * kw, 2.2 * PLOT_SCALE))
    colors = bar_colors(engines)

    ax.bar(x - width / 2, sound_pct, width, label='Sound %', color=colors, edgecolor='black', linewidth=0.6)
    ax.bar(
        x + width / 2, exact_pct, width, label='Exact %', color=colors, edgecolor='black', linewidth=0.6, hatch='//',
    )

    ax.set_ylabel('Cases (%)', fontsize=AXIS_LABEL_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(engines, rotation=30, ha='right', fontsize=VALUE_LABEL_SIZE)
    ax.set_xlim(*xlim)
    # Headroom for the two group labels at y=107: at 125 the axis frame sat on
    # top of them.
    ax.set_ylim(0, 130)
    ax.axhline(100, color='gray', linewidth=0.5, linestyle='--')
    # Above the axes on a white ground: inside the axes the box sat over the
    # bars, and the series differ by hatch, not by colour, so the swatches are
    # neutral and the labels black.
    ax.legend(handles=legend_handles(['Sound %', 'Exact %'], ['', '//']),
              fontsize=LEGEND_SIZE, ncol=2, loc='lower left', bbox_to_anchor=(0.0, 1.01),
              columnspacing=1.2, handletextpad=0.5, **LEGEND_BOX)

    # divider between bit-precise and register-level groups
    ax.axvline(2.5 * slot, color='black', linewidth=0.7, linestyle=':')
    ax.text(0.9 * slot, 107, 'bit-precise', fontsize=7, color='black')
    ax.text(3.3 * slot, 107, 'register-level', fontsize=7, color='black')

    fig.tight_layout()
    save(fig, 'fig_precision.pdf')


# --------------------------------------------------------------------------- #
# Figure 4 - RQ4, per-step latency                                             #
# --------------------------------------------------------------------------- #
def plot_perf_latency():
    # Ordered by the p50 the figure ranks on.  TaintGrind and libdft64 changed
    # places when TaintGrind's harness changed, which a hardcoded order missed.
    tool_keys = ['microtaint', 'panda', 'triton', 'maat', 'libdft64', 'taintgrind', 'angr']
    tool_keys.sort(key=lambda t: _pt(t)['latency_p50_per_instr_ms'])
    engines = [DISPLAY_NAME[t] for t in tool_keys]
    # PER-STEP, not per test: a sequence test is up to 32 instructions, so the
    # per-test tail measures sequence length rather than propagation cost.
    p50_us = [_pt(t)['latency_p50_per_instr_ms'] * 1000.0 for t in tool_keys]
    p99_us = [_pt(t)['latency_p99_per_instr_ms'] * 1000.0 for t in tool_keys]
    p100_us = [_pt(t)['latency_p100_per_instr_ms'] * 1000.0 for t in tool_keys]

    # Three bars at their original width; the slot closes the gap.
    slot = 0.80
    x = slot_x(len(engines), slot)
    width = 0.35 * 2 / 3
    xlim, kw = slot_geometry(len(engines), slot, 3 * width)

    fig, ax = plt.subplots(figsize=(5.5 * kw, 3.0 * PLOT_SCALE))
    colors = bar_colors(engines)

    ax.bar(x - width, p50_us, width, color=colors, edgecolor='black', linewidth=0.6)
    ax.bar(x, p99_us, width, color=colors, edgecolor='black', linewidth=0.6, hatch='//')
    ax.bar(x + width, p100_us, width, color=colors, edgecolor='black', linewidth=0.6, hatch='xx')

    ax.set_ylabel('Latency (µs)', fontsize=AXIS_LABEL_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(engines, rotation=30, ha='right', fontsize=VALUE_LABEL_SIZE)
    ax.set_xlim(*xlim)
    ax.set_yscale('log')
    # Above the axes, not inside them.  matplotlib's default loc='best' put the
    # box in the upper left, which is where PANDA's p100 bar reaches: the tallest
    # bar of the left half was drawn behind the legend and read as missing.
    # A horizontal strip above the plot cannot collide with any bar.
    ax.legend(handles=legend_handles(['p50', 'p99', 'p100'], ['', '//', 'xx']),
              fontsize=LEGEND_SIZE, ncol=3, loc='lower left', bbox_to_anchor=(0.0, 1.01),
              columnspacing=1.2, handletextpad=0.5, **LEGEND_BOX)

    fig.tight_layout()
    save(fig, 'fig_perf_latency.pdf')


# --------------------------------------------------------------------------- #
# Figure 5 - RQ4, throughput                                                   #
# --------------------------------------------------------------------------- #
def plot_perf_throughput():
    tool_keys = ['microtaint', 'panda', 'triton', 'maat', 'taintgrind', 'libdft64', 'angr']
    tool_keys.sort(key=lambda t: -_pt(t)['throughput_per_s'])
    engines = [DISPLAY_NAME[t] for t in tool_keys]
    tps = [round(_pt(t)['throughput_per_s']) for t in tool_keys]

    slot, bw = 0.80, 0.70
    xlim, kw = slot_geometry(len(engines), slot, bw)
    fig, ax = plt.subplots(figsize=(4.5 * kw, 2.8 * PLOT_SCALE))
    colors = bar_colors(engines)
    # Narrower bars, because the value sits VERTICALLY above each one and a wide
    # bar under a tall label reads as a column of text rather than a bar.  The
    # slot closes the gap they would otherwise leave.
    x = slot_x(len(engines), slot)
    bars = ax.bar(x, tps, width=bw, color=colors, edgecolor='black', linewidth=0.6)

    ax.set_ylabel('Tests/s', fontsize=AXIS_LABEL_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(engines, rotation=45, ha='right', fontsize=VALUE_LABEL_SIZE)
    ax.set_xlim(*xlim)
    ax.set_yscale('log')
    # Headroom for the rotated labels: a rotated number is as TALL as it is
    # long, and the tallest bar carries the longest number, so the clearance is
    # set by that label's length rather than by a fixed number of decades.
    ax.set_ylim(top=max(tps) * 10 ** (0.050 * VALUE_LABEL_SIZE * len(f'{max(tps):,}')))

    for bar, val in zip(bars, tps, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.08,
            f'{val:,}',
            ha='center',
            va='bottom',
            fontsize=VALUE_LABEL_SIZE,
            rotation=90,
        )

    fig.tight_layout()
    save(fig, 'fig_perf_throughput.pdf')


# --------------------------------------------------------------------------- #
# Figure 6 - RQ5, end-to-end overhead                                          #
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
    # Short forms: the rung names ran into each other under the axis and cost
    # more room than the bars.  \S\ref{sec:eval:overhead} spells each one out.
    labels = ['native', 'ql + hooks', 'MT plumb.', 'MT all']
    # Wall seconds of the measured phase: ql.run() for the emulated rungs, the
    # whole subprocess for `native`, which has no ql.run to isolate.
    wall_s = [lad['layers'][k]['run_s'] for k in keys]
    rss_mib = [lad['layers'][k].get('peak_rss_mib') or 0.0 for k in keys]
    colors = [OTHER_COLOR, OTHER_COLOR, MICROTAINT_COLOR, MICROTAINT_COLOR]
    # Bars keep their width; the slot closes the gap, on BOTH panels.
    slot = 0.72
    x = slot_x(len(keys), slot)
    xlim, kw = slot_geometry(len(keys), slot, 0.6)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.2 * kw, 2.9 * PLOT_SCALE))

    ax1.bar(x, wall_s, 0.6, color=colors, edgecolor='black', linewidth=0.6)
    ax1.set_yscale('log')
    ax1.set_ylabel('Wall time (s)', fontsize=AXIS_LABEL_SIZE)
    ax1.set_title('Time', fontsize=AXIS_LABEL_SIZE)

    ax2.bar(x, rss_mib, 0.6, color=colors, edgecolor='black', linewidth=0.6)
    ax2.set_ylabel('Peak RSS (MiB)', fontsize=AXIS_LABEL_SIZE)
    ax2.set_title('Peak memory', fontsize=AXIS_LABEL_SIZE)
    ax2.set_ylim(0)

    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=VALUE_LABEL_SIZE)
        ax.set_xlim(*xlim)
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
