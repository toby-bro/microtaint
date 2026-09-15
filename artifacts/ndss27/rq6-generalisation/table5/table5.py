#!/usr/bin/env python3
# ruff: noqa: W505, E501
#   Style only, and suppressed rather than rewritten: this harness is
#   vendored from the campaign that produced the published numbers, and
#   e.g. adding zip(strict=True) would change behaviour where the
#   original silently truncated.  Correctness is gated by
#   validate_oracle.py, not by restyling proven code.
# Vendored verbatim from the soundness-campaign harness; see README.md.
# Experiment script, not library code: see artifacts/ndss27/README.md,
# "Lint and type checking", for why annotations are not required here.
# mypy: disable-error-code="no-untyped-def, no-untyped-call, type-arg, no-any-return, var-annotated, assignment, index, arg-type, union-attr, operator, attr-defined, misc, call-overload, return-value, unreachable"
"""Generate Table 5 (cross-ISA soundness and precision) from the campaign shards.

Columns, and where each comes from:

  Instrs      instruction forms with at least one CHECKED case.  Forms that were
              only ever skipped are not claimed as covered.
  Cases       sum of `checked`, i.e. cases where the ground truth was EXACT.
              Cases that fell back to the lower bound, or whose runs did not
              complete, are counted as `skipped` and excluded here.
  Under       cases where `gt & ~mt` was non-empty: a real under-taint.
  Bit-exact   exact / checked, where exact means the engine's mask EQUALS the
              ground truth on every scored register.
  Over-taint  the MEAN of the per-case tainted-to-minimum bit ratio, which is
              what the paper's caption describes.  It is NOT sum(mt)/sum(gt):
              that weights large-taint cases more heavily and gives a materially
              different number (2.80x vs 1.68x on AMD64).  The mean is
              accumulated during the run because it cannot be recovered from
              summed bits afterwards.

The script REFUSES to emit a table if any ISA looks vacuous, i.e. it has checked
cases but zero ground-truth bits.  A run like that reports a perfect zero while
measuring nothing, and a table generated from it would be worse than no table:
19,085,696 MIPS cases were scored that way on 2026-09-12.

Run:  python table5.py [--dir DIR] [--json OUT] [--tex OUT]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

# Paper label and column order, so the emitted table is directly comparable.
LABEL = {'AMD64': 'x86-64', 'ARM64': 'ARM64', 'MIPS64BE': 'MIPS64',
         'PPC32BE': 'PPC32', 'RISCV64': 'RV64GC'}
ORDER = ['AMD64', 'ARM64', 'MIPS64BE', 'PPC32BE', 'RISCV64']


def load(d):
    rows = {}
    for f in sorted(glob.glob(os.path.join(d, 'campaign_*.json'))):
        try:
            doc = json.load(open(f))
        except Exception as exc:
            print(f'  skipping {os.path.basename(f)}: {exc}', file=sys.stderr)
            continue
        m = doc.get('metrics') or {}
        agg = {
            'instrs': sum(1 for x in m.values() if x.get('checked')),
            'checked': sum(x.get('checked', 0) for x in m.values()),
            'skipped': sum(x.get('skipped', 0) for x in m.values()),
            'under': sum(x.get('under', 0) for x in m.values()),
            'exact': sum(x.get('exact', 0) for x in m.values()),
            'over': sum(x.get('over', 0) for x in m.values()),
            'gt_bits': sum(x.get('gt_bits', 0) for x in m.values()),
            'mt_bits': sum(x.get('mt_bits', 0) for x in m.values()),
            'ratio_sum': sum(x.get('ratio_sum', 0.0) for x in m.values()),
            'ratio_n': sum(x.get('ratio_n', 0) for x in m.values()),
            'elapsed_h': doc.get('elapsed_h'),
            'rounds': doc.get('rounds'),
            'provenance': doc.get('provenance', {}),
            'unsound_instrs': doc.get('unsound_instrs'),
        }
        rows[doc.get('isa', os.path.basename(f))] = agg
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument('--json', metavar='PATH')
    ap.add_argument('--tex', metavar='PATH')
    ap.add_argument('--allow-partial', action='store_true',
                    help='emit even if an ISA is missing (still refuses vacuous)')
    args = ap.parse_args()

    rows = load(args.dir)
    if not rows:
        print('no campaign_*.json found', file=sys.stderr)
        return 2

    problems = []
    for isa, r in rows.items():
        if r['checked'] == 0:
            # An empty run must never certify.  Guarding only the
            # checked-but-no-taint case leaves the emptier failure wide open,
            # which is the same blind spot in a smaller disguise.
            problems.append(f'{isa}: ZERO cases were checked, so this row is an '
                            f'absence of measurement, not a result')
            continue
        if r['gt_bits'] == 0:
            problems.append(f'{isa}: {r["checked"]} checked cases but ZERO '
                            f'ground-truth bits, so no under-taint could ever '
                            f'have been reported')
        if r['checked'] and r['ratio_n'] == 0:
            problems.append(f'{isa}: no per-case ratio recorded; re-run with a '
                            f'harness that accumulates ratio_sum/ratio_n')
    missing = [k for k in ORDER if k not in rows]
    if missing and not args.allow_partial:
        problems.append(f'missing ISAs: {", ".join(missing)}')

    hdr = ('%-9s %7s %10s %7s %10s %11s %9s' %
           ('ISA', 'Instrs', 'Cases', 'Under', 'Bit-exact', 'Over-taint', 'skipped'))
    print(hdr)
    print('-' * len(hdr))
    tot = dict.fromkeys(('instrs', 'checked', 'under', 'exact', 'skipped',
                         'ratio_n'), 0)
    tot['ratio_sum'] = 0.0
    out_rows = []
    for isa in [k for k in ORDER if k in rows] + [k for k in rows if k not in ORDER]:
        r = rows[isa]
        ex = r['exact'] / r['checked'] if r['checked'] else 0.0
        ratio = r['ratio_sum'] / r['ratio_n'] if r['ratio_n'] else float('nan')
        print('%-9s %7d %10s %7d %9.1f%% %10.2fx %9d' % (
            LABEL.get(isa, isa), r['instrs'], f'{r["checked"]/1e6:.1f}M',
            r['under'], 100 * ex, ratio, r['skipped']))
        for k in tot:
            if k in r:
                tot[k] += r[k]
        # Both aggregations are recorded because they differ materially and
        # the paper's caption must say which one it means: the mean of per-case
        # ratios (what the caption describes) and the ratio of summed bits.
        sum_ratio = r['mt_bits'] / r['gt_bits'] if r['gt_bits'] else float('nan')
        out_rows.append({'isa': isa, 'label': LABEL.get(isa, isa), **r,
                         'bit_exact': ex, 'over_taint_mean': ratio,
                         'over_taint_sum_ratio': sum_ratio})
    ex_t = tot['exact'] / tot['checked'] if tot['checked'] else 0.0
    ratio_t = tot['ratio_sum'] / tot['ratio_n'] if tot['ratio_n'] else float('nan')
    print('-' * len(hdr))
    print('%-9s %7d %10s %7d %9.1f%% %10.2fx %9d' % (
        'Overall', tot['instrs'], f'{tot["checked"]/1e6:.1f}M', tot['under'],
        100 * ex_t, ratio_t, tot['skipped']))

    if args.tex:
        with open(args.tex, 'w') as fh:
            fh.write('% Generated by campaign_ae/table5.py.  Do not edit by hand.\n')
            for r in out_rows:
                fh.write('        %-8s & %-7d & %6.1f\\,M & %d     & %.1f\\%%    '
                         '& $%.2f\\times$ \\\\\n' % (
                             r['label'], r['instrs'], r['checked'] / 1e6,
                             r['under'], 100 * r['bit_exact'], r['over_taint_mean']))
            fh.write('        \\hline\n')
            fh.write('        Overall & %d & %.1f\\,M & %d     & %.1f\\%%    '
                     '& ---          \\\\\n' % (
                         tot['instrs'], tot['checked'] / 1e6, tot['under'],
                         100 * ex_t))
        print(f'[tex] {args.tex}')

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'experiment': 'rq6-table5', 'rows': out_rows,
                       'overall': {'instrs': tot['instrs'],
                                   'cases': tot['checked'],
                                   'under': tot['under'],
                                   'bit_exact': ex_t,
                                   'over_taint_mean': ratio_t,
                                   'skipped': tot['skipped']},
                       'problems': problems}, fh, indent=2)
            fh.write('\n')
        print(f'[json] {args.json}')

    if problems:
        print('\nREFUSING to certify this table:')
        for p in problems:
            print(f'  * {p}')
        return 1
    print('\nEvery ISA produced ground-truth taint, so the zeros above are '
          'measurements rather than absences.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
