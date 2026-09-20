"""The RQ2/RQ3/RQ4 evaluation tables, from a benchmark.py report.

These seven tables were transcribed into LaTeX by hand, so a re-run could move
a number without the table following.  Every value in them is already in the
report, which means they can be generated instead:

    python gen_eval_tables.py REPORT.json --tex eval_tables.tex
    python gen_eval_tables.py REPORT.json --md eval_tables.md

Tables emitted, and what each one needs from the report:

  tab:soundness        metrics.ground_truth.per_tool
  tab:gt-coverage      metrics.ground_truth
  tab:over-taint       metrics.ground_truth.per_tool, bit-precise engines only
  tab:unsound-summary  metrics.ground_truth.per_tool unsound_modes/categories
  tab:perf             metrics.per_tool latency and throughput
  tab:f1-vs-ref        metrics.per_tool precision/recall/f1
  tab:pe               metrics.path_explosion_scaling_median

A table whose inputs the run did not produce is skipped with a note on stderr,
rather than emitted with zeros.  A run with --no-baselines has one engine, and
a one-row comparison table is not a comparison.
"""
# Experiment script, not library code: see artifacts/ndss27/README.md,
# "Lint and type checking", for why annotations are not required here.
# mypy: disable-error-code="no-untyped-def, no-untyped-call, type-arg"
from __future__ import annotations

import argparse
import json
import sys

#: Display name per report key.  Anything the report names but this does not is
#: still emitted, under its own key, so a new engine is not silently dropped.
NAME = {'microtaint': 'microtaint', 'angr': 'angr', 'maat': 'Maat',
        'triton': 'Triton', 'panda': 'PANDA', 'taintgrind': 'TaintGrind',
        'libdft64': 'libdft64'}

#: Test-mode pillar to the one-letter code the unsoundness table uses.
MODE_CODE = {'single': 'S', 'sequence': 'Q', 'sweep': 'W'}

#: How many categories the unsoundness table names before it stops.
TOP_CATEGORIES = 6

#: The report carries the oracle under this key alongside the engines.  It is
#: what the others are scored against, so it never gets a row of its own.
ORACLE = 'ground_truth'

DASH = '---'


def name(key):
    return NAME.get(key, key)


def tex_name(key):
    """The paper writes engine names with \\textsc and marks its own."""
    return r'\textsc{%s}\xspace' % name(key).lower()


def texnum(x, decimals=0):
    """A number with the paper's thin-space thousands separator."""
    s = f'{x:,.{decimals}f}'
    whole, _, frac = s.partition('.')
    whole = whole.replace(',', r'\,')
    return whole + ('.' + frac if frac else '')


# --------------------------------------------------------------- table bodies
# Each builder returns (label, caption, column spec, header cells, rows, rules)
# where `rules` holds row indices after which a horizontal rule is drawn.  Both
# emitters then render the same structure, so the LaTeX and the Markdown cannot
# drift apart.


def t_soundness(rep):
    gt = rep['metrics']['ground_truth']['per_tool']
    gran = rep['metadata'].get('granularity', {})
    if len(gt) < 2:
        return None
    groups = [('bit', 'bit-precise (compared bit-by-bit)'),
              ('reg', 'register-level (compared at register granularity)')]
    rows: list[tuple] = []
    rules, seen = [], 0
    for g, title in groups:
        keys = [k for k in gt if k != ORACLE and gran.get(k, 'reg') == g]
        if not keys:
            continue
        # Most sound first, then most exact: the reading order the table's
        # argument follows.
        keys.sort(key=lambda k: (-gt[k]['soundness_rate'], -gt[k]['exact_case_rate']))
        rows.append(('GROUP', title))
        seen += 1
        for k in keys:
            d = gt[k]
            jac = d.get('mean_jaccard_bit')
            # A register-level engine is scored per register, so its bit totals
            # are structurally zero.  Reporting those zeros would read as
            # "never wrong" for the engines that are wrong most often.
            over, under = ((d['over_bits_total'], d['under_bits_total'])
                           if g == 'bit'
                           else (d['regs_over_only'], d['regs_under_only']))
            rows.append((k, f"{100 * d['soundness_rate']:.1f}",
                         f"{100 * d['exact_case_rate']:.1f}",
                         f'{jac:.4f}' if jac is not None else DASH,
                         over, under, d['unsound_cases']))
            seen += 1
        rules.append(seen)
    rules.pop()  # the last group is closed by the table's own bottom rule
    return ('tab:soundness',
            'Soundness and precision against exhaustive ground truth',
            'lrrrrrr',
            ['Engine', r'Sound\,\%', r'Exact\,\%', 'Mean Jaccard',
             'Over-bits', 'Under-bits', 'Unsound cases'],
            rows, rules)


def t_gt_coverage(rep):
    gt = rep['metrics']['ground_truth']
    if not gt.get('cases_total'):
        return None
    budget = gt['budget_bits']
    mean_k, max_k = gt.get('skipped_k_mean'), gt.get('skipped_k_max')
    if mean_k is None:
        skipped = f'Skipped ($k > {budget}$)'
    else:
        skipped = (f'Skipped ($k > {budget}$, mean $k = {mean_k:g}$, '
                   f'max $k = {max_k}$)')
    rows = [('TEXT', f'Within budget ($k \\le {budget}$)', gt['cases_within_budget']),
            ('TEXT', skipped, gt['cases_skipped']),
            ('RULE',),
            ('TEXT', 'Total', gt['cases_total'])]
    return ('tab:gt-coverage', 'Ground-truth coverage of the corpus', 'lr',
            ['Status', 'Cases'], rows, [])


def t_over_taint(rep):
    gt = rep['metrics']['ground_truth']['per_tool']
    gran = rep['metadata'].get('granularity', {})
    keys = [k for k in gt
            if k != ORACLE and gran.get(k, 'reg') == 'bit'
            and gt[k]['cases_compared']]
    if len(keys) < 2:
        return None
    # Spurious bits per case is the over-taint the table is about, so the least
    # over-tainting engine leads.
    keys.sort(key=lambda k: gt[k]['over_bits_total'] / gt[k]['cases_compared'])
    rows = []
    for k in keys:
        d = gt[k]
        jac = d.get('mean_jaccard_bit')
        rows.append((k, f"{d['over_bits_total'] / d['cases_compared']:.2f}",
                     f'{jac:.4f}' if jac is not None else DASH))
    return ('tab:over-taint', 'Over-taint among the bit-precise engines', 'lrr',
            ['Engine', 'Spurious bits / case', 'Mean Jaccard'], rows, [])


def t_unsound_summary(rep):
    gt = rep['metrics']['ground_truth']['per_tool']
    if len(gt) < 3 or not any('unsound_modes' in d for d in gt.values()):
        return None
    keys = sorted((k for k in gt if k != ORACLE),
                  key=lambda k: (gt[k]['unsound_cases'], k))
    rows = []
    for k in keys:
        d = gt[k]
        modes = d.get('unsound_modes') or {}
        cats = d.get('unsound_categories') or {}
        # S, Q, W in the pillar order the caption gives, not in whichever order
        # the run happened to hit them.
        code = ', '.join(c for m, c in MODE_CODE.items() if modes.get(m)) or DASH
        top = sorted(cats.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_CATEGORIES]
        cat_s = ', '.join(f'{c}({n})' for c, n in top) or DASH
        rows.append((k, d['unsound_cases'], code, cat_s))
    caption = ('Per-engine unsoundness summary on the ground-truth-evaluable '
               'cases.  Modes: S=single-instruction, Q=sequence, W=sweep.')
    return ('tab:unsound-summary', caption, 'lrll', ['Engine', 'Unsound', 'Modes', 'Top affected categories (count)'],
            rows, [])


def t_perf(rep):
    pt = rep['metrics']['per_tool']
    keys = [k for k in pt if k != ORACLE and pt[k].get('latency_p50_ms')]
    if len(keys) < 2:
        return None
    keys.sort(key=lambda k: pt[k]['latency_p50_ms'])
    rows = []
    for k in keys:
        d = pt[k]
        rows.append((k, texnum(d['latency_p50_ms'] * 1000, 1),
                     texnum(d['latency_p99_ms'] * 1000),
                     texnum(d.get('throughput_per_s') or 0),
                     texnum(d.get('throughput_per_s_per_instr') or 0)))
    return ('tab:perf', 'Per-propagation latency and throughput', 'lrrrr',
            ['Engine', r'p50 ($\mu$s)', r'p99 ($\mu$s)', 'tp/s', 'tp/s/i'],
            rows, [])


def t_f1(rep, reference):
    pt = rep['metrics']['per_tool']
    keys = [k for k in pt if k not in (reference, ORACLE) and pt[k].get('f1')]
    if len(keys) < 2:
        return None
    keys.sort(key=lambda k: -pt[k]['f1'])
    # The paper bolds the engines that agree with the reference on better than
    # 99% of the harmonic mean.  That is a threshold, not a podium, so it is
    # written as one.
    rows = [(k, f"{pt[k]['precision']:.4f}", f"{pt[k]['recall']:.4f}",
             (f"{pt[k]['f1']:.4f}", pt[k]['f1'] >= 0.99)) for k in keys]
    return ('tab:f1-vs-ref',
            f'Agreement with {name(reference)} as the reference labelling',
            'lrrr', ['Engine', 'Precision', 'Recall', 'F1'], rows, [])


def t_pe(rep):
    pe = (rep['metrics'].get('path_explosion_scaling_median')
          or {}).get('path_explosion_branching')
    if not pe:
        return None
    ns = sorted(pe, key=int)
    tools = sorted({t for n in ns for t in pe[n]},
                   key=lambda t: min(pe[n].get(t, float('inf')) for n in ns))
    # Engines that abandoned every case have no row of numbers, but leaving them
    # out would read as if they had not been tested.
    for t in rep['metrics']['per_tool']:
        if t != ORACLE and t not in tools:
            tools.append(t)
    rows = [(t, *[texnum(pe[n][t] * 1000) if t in pe[n] else DASH for n in ns])
            for t in tools]
    return ('tab:pe', r'Path-explosion latency scaling (median $\mu$s per test)',
            'l' + 'r' * len(ns), ['Engine'] + [f'$N{{=}}{n}$' for n in ns],
            rows, [])


BUILDERS = [t_soundness, t_gt_coverage, t_over_taint, t_unsound_summary,
            t_perf, t_pe]


# ------------------------------------------------------------------- emitters

def _cell(v, tex):
    """A row cell.  A (text, bold) pair is emphasised, everything else is not."""
    if isinstance(v, tuple):
        text, bold = v
        return (r'\textbf{%s}' % text) if (tex and bold) else text
    return str(v)


def as_tex(tables):
    out = ['% Generated by gen_eval_tables.py.  Do not edit by hand.']
    for label, caption, spec, header, rows, rules in tables:
        out += ['', r'\begin{table}[t]', '    ' + r'\centering',
                f'    \\caption{{{caption}}}', f'    \\label{{{label}}}',
                f'    \\begin{{tabular}}{{{spec}}}', '        ' + r'\toprule',
                '        ' + ' & '.join(header) + r' \\', '        ' + r'\midrule']
        for i, row in enumerate(rows):
            if row[0] == 'RULE':
                out.append('        ' + r'\midrule')
                continue
            if row[0] == 'GROUP':
                out.append('        ' + r'\multicolumn{%d}{l}{\emph{%s}} \\'
                           % (len(header), row[1]))
            elif row[0] == 'TEXT':
                out.append('        ' + ' & '.join(_cell(v, True) for v in row[1:])
                           + r' \\')
            else:
                cells = [tex_name(row[0])] + [_cell(v, True) for v in row[1:]]
                out.append('        ' + ' & '.join(cells) + r' \\')
            if i + 1 in rules:
                out.append('        ' + r'\midrule')
        out += ['        ' + r'\bottomrule', '    ' + r'\end{tabular}',
                r'\end{table}']
    return '\n'.join(out) + '\n'


def _plain(s):
    """The LaTeX our own captions and headers use, as readable text."""
    for a, b in ((r'$\mu$', 'u'), (r'\le', '<='), (r'\,', ''), (r'\%', '%'),
                 ('{=}', '='), ('$', '')):
        s = s.replace(a, b)
    return s


def as_md(tables):
    out = ['<!-- Generated by gen_eval_tables.py. Do not edit by hand. -->']
    for label, caption, _spec, header, rows, _rules in tables:
        # The caption is LaTeX, and a Markdown reader should not have to parse
        # it.  Only the markup the captions actually use is stripped.
        cap = _plain(caption).replace(r'\,', ' ')
        out += ['', f'### {cap} ({label})', '',
                '| ' + ' | '.join(_plain(h) for h in header) + ' |',
                '| ' + ' | '.join('---' for _ in header) + ' |']
        for row in rows:
            if row[0] == 'RULE':
                continue
            if row[0] == 'GROUP':
                out.append(f'| **{row[1]}** |' + ' |' * (len(header) - 1))
                continue
            cells = ([] if row[0] == 'TEXT' else [name(row[0])])
            cells += [_cell(v, False) for v in row[1:]]
            cells += [''] * (len(header) - len(cells))
            out.append('| ' + ' | '.join(_plain(c) for c in cells) + ' |')
    return '\n'.join(out) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('report', help='a benchmark.py report JSON')
    ap.add_argument('--tex', metavar='PATH')
    ap.add_argument('--md', metavar='PATH')
    ap.add_argument('--reference', default='microtaint',
                    help='the engine the F1 table scores the others against')
    args = ap.parse_args()

    with open(args.report) as fh:
        rep = json.load(fh)

    tables = []
    for build in BUILDERS:
        t = build(rep)
        if t is None:
            print(f'[-] {build.__name__}: this run has no inputs for it, skipped',
                  file=sys.stderr)
            continue
        tables.append(t)
    f1 = t_f1(rep, args.reference)
    if f1 is not None:
        tables.insert(1, f1)
    else:
        print('[-] t_f1: this run has no inputs for it, skipped', file=sys.stderr)

    if not tables:
        raise SystemExit(
            f'{args.report} supports none of the evaluation tables.  A report '
            f'from a --no-baselines run cannot produce a comparison.',
        )
    if args.tex:
        with open(args.tex, 'w') as fh:
            fh.write(as_tex(tables))
        print(f'[+] wrote {args.tex} ({len(tables)} tables)')
    if args.md:
        with open(args.md, 'w') as fh:
            fh.write(as_md(tables))
        print(f'[+] wrote {args.md} ({len(tables)} tables)')
    if not (args.tex or args.md):
        print(as_md(tables))
    return 0


if __name__ == '__main__':
    sys.exit(main())
