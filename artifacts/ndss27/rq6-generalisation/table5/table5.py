#!/usr/bin/env python3
# ruff: noqa: W505, E501, RUF100, I001
#   RUF100 and I001 are config differences, not defects: the source repo
#   selects BLE001/E402/C901 (so those noqa ARE used there) and sorts
#   `microtaint` as third-party, while it is first-party here.  No single
#   spelling satisfies both repos, and the body must stay byte-identical
#   to the harness that produced the published numbers.
#   Style only, suppressed rather than rewritten: vendored from the campaign
#   that produced the published numbers.  Correctness is gated by
#   validate_oracle.py, not by restyling proven code.
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


def _merge_named(metrics, key):
    out = {}
    for x in metrics.values():
        for k, v in (x.get(key) or {}).items():
            out[k] = out.get(k, 0) + v
    return out


def _merge_per_flag(metrics):
    out = {}
    for x in metrics.values():
        for f, c in (x.get('per_flag') or {}).items():
            a = out.setdefault(f, {})
            for k, v in c.items():
                a[k] = a.get(k, 0) + v
    return out


def _flag_exact_excluding(hist, order, excluded):
    """Case-level flag exactness ignoring `excluded` flags.

    Recomputed from the per-case failure mask, so it is the true conjunction
    over the remaining flags.  Deriving it from per-flag totals would be wrong:
    a case can fail on PF alone or on PF and CF together, and totals cannot tell
    those apart.
    """
    if not hist:
        return None
    keep = 0
    for name, bit in (order or {}).items():
        if name not in excluded:
            keep |= 1 << bit
    tot = ok = 0
    for mask_s, n in hist.items():
        tot += n
        if (int(mask_s) & keep) == 0:
            ok += n
    return (ok / tot) if tot else None


def _merge_hist(metrics):
    """Sum the per-form log2 ratio histograms into one."""
    out = {}
    for x in metrics.values():
        for k, v in (x.get('ratio_hist') or {}).items():
            out[k] = out.get(k, 0) + v
    return out


def _median_bucket(hist):
    """Median over-taint ratio, as the log2 bucket containing the median case.

    The MEAN is dominated by avalanche cases, where one tainted bit legitimately
    taints all 64, so it measures how many multiplies are in the corpus more
    than it measures the engine.  The median says what a typical case looks
    like.  Bucket k holds ratios in [2^(k-1), 2^k); bucket 0 holds ratio <= 1.
    """
    total = sum(hist.values())
    if not total:
        return None, None
    seen = 0
    for k in sorted(hist, key=int):
        seen += hist[k]
        if seen * 2 >= total:
            ki = int(k)
            if ki == 0:
                return 1.0, '<=1'
            return float(1 << (ki - 1)), '[%d,%d)' % (1 << (ki - 1), 1 << ki)
    return None, None


def load(d):
    rows = {}
    for f in sorted(glob.glob(os.path.join(d, 'campaign_*.json'))):
        try:
            doc = json.load(open(f))
        except Exception as exc:  # noqa: BLE001
            print(f'  skipping {os.path.basename(f)}: {exc}', file=sys.stderr)
            continue
        m = doc.get('metrics') or {}
        # A form whose ground truth is EMPTY over its whole run was not measured.
        # `mul r8` multiplies by a register the oracle pins to zero, so no tainted
        # input bit can move an output bit and the ground truth is empty.
        #
        # The engine does NOT answer nothing on it, and that is the point.  The
        # state handed to the engine is built from isa.gprs, so r8 is not in it:
        # the oracle knows the multiplier is zero and the engine does not.  The
        # engine assumes an unknown multiplier, avalanches, and is charged 4,192
        # over-taint cases and 540,768 over-taint bits for a fact the harness hid
        # from it.  The two sides are being asked about DIFFERENT machine states.
        #
        # So these forms are not merely unmeasured, they are actively misscored,
        # in both directions: the ones the engine stays silent on score as free
        # bit-exact passes (on AMD64, 14.1% of cases and 7.2 points of exactness),
        # and the ones it does not score as over-taint it does not owe.  The real
        # repair is to model every GPR so the states agree; until then they are
        # reported separately and folded into nothing.
        #
        # The split is driven by each form's own measured gt_bits over thousands
        # of cases, not by a list of forms or a sampled probe.
        seen = {a: x for a, x in m.items() if x.get('checked')}
        obs = {a: x for a, x in seen.items() if x.get('gt_bits', 0) > 0}
        blind = {a: x for a, x in seen.items() if x.get('gt_bits', 0) <= 0}
        agg = {
            # Instrs and Cases describe the OBSERVABLE population, the same one
            # every percentage is over, so a reader can multiply a column by the
            # case count and get the right answer.  The attempted totals are kept
            # alongside rather than dropped.
            'instrs': len(obs),
            'instrs_attempted': len(seen),
            'checked': sum(x.get('checked', 0) for x in obs.values()),
            'checked_attempted': sum(x.get('checked', 0) for x in seen.values()),
            'skipped': sum(x.get('skipped', 0) for x in m.values()),
            'under': sum(x.get('under', 0) for x in seen.values()),
            # Precision is reported over the OBSERVABLE population only.
            'observable_checked': sum(x.get('checked', 0) for x in obs.values()),
            'exact': sum(x.get('exact', 0) for x in obs.values()),
            'over': sum(x.get('over', 0) for x in obs.values()),
            'gt_bits': sum(x.get('gt_bits', 0) for x in obs.values()),
            'mt_bits': sum(x.get('mt_bits', 0) for x in obs.values()),
            'ratio_sum': sum(x.get('ratio_sum', 0.0) for x in obs.values()),
            'ratio_n': sum(x.get('ratio_n', 0) for x in obs.values()),
            'ratio_hist': _merge_hist(obs),
            'per_flag': _merge_per_flag(obs),
            'flag_fail_hist': _merge_named(obs, 'flag_fail_hist'),
            'flag_order': doc.get('flag_order', {}),
            # Unobservable population, reported separately and never blended in.
            'blind_instrs': len(blind),
            'blind_checked': sum(x.get('checked', 0) for x in blind.values()),
            'blind_engine_silent': sum(x.get('exact', 0) for x in blind.values()),
            'blind_over': sum(x.get('over', 0) for x in blind.values()),
            'blind_under': sum(x.get('under', 0) for x in blind.values()),
            # Exactness over cases that witnessed SOMETHING.  `chk` scores every
            # modelled GPR including ones the instruction only reads, and for
            # those the ground truth is tautologically the input mask: a no-op
            # would produce it identically.  Reporting exactness over all cases
            # therefore partly reports that a copy survives.
            'informative_checked': sum(x.get('informative_checked', 0)
                                       for x in obs.values()),
            'informative_exact': sum(x.get('informative_exact', 0)
                                     for x in obs.values()),
            'info_bits': sum(x.get('info_bits', 0) for x in obs.values()),
            'blind_over_bits': sum(x.get('over_bits', 0) for x in blind.values()),
            'blind_forms': sorted(blind),
            **{k: sum(x.get(k, 0) for x in obs.values()) for k in (
                'val_checked', 'val_exact', 'val_over', 'val_under',
                'flag_checked', 'flag_exact', 'flag_over', 'flag_under',
                'val_gt_bits', 'val_mt_bits', 'val_over_bits',
                'flag_gt_bits', 'flag_mt_bits', 'flag_over_bits')},
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
    ap.add_argument('--exclude-flags', default='',
                    help='comma-separated flags to EXCLUDE from the flag-exact '
                         'column, e.g. PF.  Recomputed exactly from the per-case '
                         'failure-mask histogram, not approximated.')
    ap.add_argument('--per-flag', action='store_true',
                    help='also print the per-flag exact/over breakdown')
    ap.add_argument('--allow-partial', action='store_true',
                    help='emit even if an ISA is missing (still refuses vacuous)')
    args = ap.parse_args()

    excluded = {f.strip() for f in args.exclude_flags.split(',') if f.strip()}
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
    # Shards must agree about WHAT PRODUCED THEM.  A 0.089h smoke shard from a
    # DIRTY engine, carrying the superseded "PC completion check" oracle string,
    # was sitting in this directory and certified happily with --allow-partial,
    # NaN in both exactness columns and all.  Nothing compared a shard's oracle,
    # engine commit or dirty flag against the run being certified.
    provs = {isa: (r.get('provenance') or {}) for isa, r in rows.items()}
    oracles = {isa: p.get('oracle') for isa, p in provs.items() if p.get('oracle')}
    if len(set(oracles.values())) > 1:
        problems.append(
            'shards disagree about which ORACLE produced them, so they are not '
            'one campaign: ' + '; '.join(f'{k}={v!r}' for k, v in sorted(oracles.items())),
        )
    commits = {isa: p.get('engine_commit') for isa, p in provs.items()
               if p.get('engine_commit')}
    if len(set(commits.values())) > 1:
        problems.append(
            'shards measured DIFFERENT engine commits, so the rows cannot be '
            'compared: ' + '; '.join(f'{k}={(v or "")[:12]}' for k, v in sorted(commits.items())),
        )
    dirty = sorted(isa for isa, p in provs.items() if p.get('engine_dirty'))
    if dirty:
        problems.append(
            f'{", ".join(dirty)}: measured a DIRTY engine checkout, so the '
            f'commit recorded does not identify the code that ran',
        )
    for isa, r in rows.items():
        if r['checked'] and not r.get('val_checked'):
            problems.append(
                f'{isa}: shard predates the register/flag exactness split, so '
                f'its precision columns are NaN rather than measurements',
            )
    missing = [k for k in ORDER if k not in rows]
    if missing and not args.allow_partial:
        problems.append(f'missing ISAs: {", ".join(missing)}')

    hdr = ('%-9s %7s %11s %8s %11s %12s %10s %11s' %
           ('ISA', 'Instrs', 'Cases', 'Under', 'exact regs%', 'exact flags%',
            'Over-taint', 'skipped'))
    print(hdr)
    print('-' * len(hdr))
    tot = dict.fromkeys(('instrs', 'checked', 'checked_attempted', 'under',
                         'exact', 'skipped', 'ratio_n', 'blind_instrs',
                         'blind_checked', 'blind_engine_silent',
                         'blind_over_bits'), 0)
    tot['ratio_sum'] = 0.0
    out_rows = []
    for isa in [k for k in ORDER if k in rows] + [k for k in rows if k not in ORDER]:
        r = rows[isa]
        obs_n = r['checked']
        ex = r['exact'] / obs_n if obs_n else 0.0
        ratio = r['ratio_sum'] / r['ratio_n'] if r['ratio_n'] else float('nan')
        vex = r['val_exact'] / r['val_checked'] if r.get('val_checked') else float('nan')
        fex = r['flag_exact'] / r['flag_checked'] if r.get('flag_checked') else float('nan')
        if excluded:
            alt = _flag_exact_excluding(r.get('flag_fail_hist'),
                                        r.get('flag_order'), excluded)
            if alt is not None:
                fex = alt
        print('%-9s %7d %11s %8d %10.1f%% %11.1f%% %11.2fx %10d' % (
            LABEL.get(isa, isa), r['instrs'], format(r['checked'], ','),
            r['under'], 100 * vex, 100 * fex, ratio, r['skipped']))
        for k in tot:
            if k in r:
                tot[k] += r[k]
        # Both aggregations are recorded because they differ materially and
        # the paper's caption must say which one it means: the mean of per-case
        # ratios (what the caption describes) and the ratio of summed bits.
        sum_ratio = r['mt_bits'] / r['gt_bits'] if r['gt_bits'] else float('nan')
        med, med_label = _median_bucket(r.get('ratio_hist') or {})
        out_rows.append({'isa': isa, 'label': LABEL.get(isa, isa), **r,
                         'bit_exact': ex, 'over_taint_mean': ratio,
                         'over_taint_sum_ratio': sum_ratio,
                         'over_taint_median_lower': med,
                         'over_taint_median_bucket': med_label})
    ex_t = tot['exact'] / tot['checked'] if tot['checked'] else 0.0
    ratio_t = tot['ratio_sum'] / tot['ratio_n'] if tot['ratio_n'] else float('nan')
    print('-' * len(hdr))
    vt = sum(x.get('val_exact', 0) for x in rows.values())
    vc = sum(x.get('val_checked', 0) for x in rows.values())
    ft = sum(x.get('flag_exact', 0) for x in rows.values())
    fc = sum(x.get('flag_checked', 0) for x in rows.values())
    fex_t = (ft / fc) if fc else float('nan')
    if excluded:
        # Merge every ISA's mask histogram under its OWN bit assignment: the
        # same flag has a different bit index on different ISAs, so the masks
        # are not comparable and must be reduced per ISA before summing.
        ok = tt = 0
        for rr in rows.values():
            a = _flag_exact_excluding(rr.get('flag_fail_hist'),
                                      rr.get('flag_order'), excluded)
            n = sum((rr.get('flag_fail_hist') or {}).values())
            if a is not None:
                ok += a * n
                tt += n
        if tt:
            fex_t = ok / tt
    print('%-9s %7d %11s %8d %10.1f%% %11.1f%% %11.2fx %10d' % (
        'Overall', tot['instrs'], format(tot['checked'], ','), tot['under'],
        100 * vt / vc if vc else float('nan'),
        100 * fex_t, ratio_t, tot['skipped']))
    print()
    if excluded:
        print()
        print('exact flags%% EXCLUDES %s (recomputed from the per-case failure mask)'
              % ', '.join(sorted(excluded)))
    if args.per_flag:
        print()
        print('%-9s %-6s %10s %9s %9s %7s' % ('ISA', 'flag', 'checked', 'exact%', 'over%', 'under'))
        for isa in [k for k in ORDER if k in rows]:
            pf = rows[isa].get('per_flag') or {}
            for f, a in sorted(pf.items(), key=lambda z: -z[1].get('checked', 0)):
                c = a.get('checked', 0) or 1
                print('%-9s %-6s %10s %8.1f%% %8.1f%% %7d' % (
                    LABEL.get(isa, isa), f, format(a.get('checked', 0), ','),
                    100 * a.get('exact', 0) / c, 100 * a.get('over', 0) / c,
                    a.get('under', 0)))
    print()
    blind_n = tot['blind_checked']
    if blind_n:
        print()
        print('NOT MEASURED, and excluded from the table above (all columns):')
        print('%-9s %7s %11s %13s %13s %7s' % (
            'ISA', 'forms', 'cases', 'engine silent', 'over-taint bits', 'under'))
        for isa in [k for k in ORDER if k in rows] + [k for k in rows if k not in ORDER]:
            r = rows[isa]
            if not r.get('blind_checked'):
                continue
            print('%-9s %7d %11s %13s %13s %7d' % (
                LABEL.get(isa, isa), r['blind_instrs'],
                format(r['blind_checked'], ','),
                format(r['blind_engine_silent'], ','),
                format(r.get('blind_over_bits', 0), ','),
                r.get('blind_under', 0)))
        print('These forms name a register the oracle pins to zero, so no tainted')
        print('bit can move an output and the ground truth is empty.  The engine is')
        print('not told the register exists (state_format comes from isa.gprs), so')
        print('the two sides are asked about DIFFERENT machine states.  Cases the')
        print('engine stays silent on would score as free bit-exact passes; cases it')
        print('does not are charged as over-taint it does not owe.  Neither is a')
        print('measurement, so neither is folded in.  Model every GPR to fix this.')
        print()

    ic = sum(r.get('informative_checked', 0) for r in rows.values())
    ie = sum(r.get('informative_exact', 0) for r in rows.values())
    ib = sum(r.get('info_bits', 0) for r in rows.values())
    gb = sum(r.get('gt_bits', 0) for r in rows.values())
    if ic:
        print()
        print('%-9s %11s %11s %9s %14s' % (
            'ISA', 'cases', 'informative', 'exact%', 'signal/gt bits'))
        for isa in [k for k in ORDER if k in rows] + [k for k in rows if k not in ORDER]:
            r = rows[isa]
            n = r.get('informative_checked', 0)
            if not n:
                continue
            print('%-9s %11s %11s %8.1f%% %13.1f%%' % (
                LABEL.get(isa, isa), format(r['checked'], ','), format(n, ','),
                100 * r.get('informative_exact', 0) / n,
                100 * r.get('info_bits', 0) / r['gt_bits'] if r['gt_bits'] else 0))
        print('%-9s %11s %11s %8.1f%% %13.1f%%' % (
            'Overall', format(tot['checked'], ','), format(ic, ','),
            100 * ie / ic, 100 * ib / gb if gb else 0))
        print('"informative" excludes cases whose ground truth is exactly the input')
        print('mask, which a no-op of the same length would reproduce; the last')
        print('column is how much of the ground truth is signal rather than')
        print('registers passing through untouched.')
        print()
    print('every column is over the Cases shown, which already excludes the')
    print('not-measured population listed above (its under-taints are shown there).')
    print('exact regs%  = cases where the engine matched the ground truth on every')
    print('               scored GENERAL-PURPOSE register')
    print('exact flags% = same, on every scored FLAG.  A case counts as bit-exact')
    print('               overall only if BOTH hold, which is why one coarse flag')
    print('               sinks an otherwise perfect result.')

    # The refusal must come BEFORE the files are written.  Emitting first meant
    # "REFUSING to certify this table" still left a paste-ready table5.tex on
    # disk -- in the reproduced case, a row reading `MIPS64 & 0 & 0.0\,M & 0 &
    # 0.0\%% & $nan\times$`.  A refusal that still hands over the artefact is
    # not a refusal, and run_table5.sh tells the reviewer to run this exact
    # command without checking its exit code.
    if problems:
        print('\nREFUSING to certify this table:')
        for p in problems:
            print(f'  * {p}')
        for path in (args.tex, args.json):
            if path and os.path.exists(path):
                os.unlink(path)
                print(f'  (removed stale {path})')
        return 1

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

    print('\nEvery ISA produced ground-truth taint, so the zeros above are '
          'measurements rather than absences.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
