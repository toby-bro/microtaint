#!/usr/bin/env python3
"""Splice a microtaint-only run into the seven-engine reference report.

The six baselines are expensive to run (containers, Pin, Valgrind: about three
hours) and have not changed. microtaint has, so re-measuring it alone against the
same corpus and splicing the result gives every engine's most recent number on
identical inputs, in ten minutes instead of three hours.

The six baseline engines have not changed since the reference run, and the
corpus is fixed by the seed, so their numbers carry over unchanged; microtaint's
do not, because the engine has moved.  This produces the report that has each
engine's most recent measurement on the SAME corpus.

    merge_reports.py REFERENCE.json NEW_MICROTAINT.json OUT.json [--timing TIMED.json]

Soundness and precision need the ground truth and do not care about machine load
(a stopped emulation is dropped, not misread).  Per-step latency is the opposite:
it needs no ground truth at all -- `--no-ground-truth` finishes in seconds -- and
it does need a quiet machine.  So they are naturally two runs, and `--timing`
takes the per-step block from the second one.

Refuses to merge if the two runs did not score the same cases, which is the only
thing that makes the splice meaningful.
"""
# Experiment script, not library code: see artifacts/ndss27/README.md,
# "Lint and type checking", for why annotations are not required here.
# mypy: disable-error-code="no-untyped-def, no-untyped-call, type-arg, explicit-any"
import json
import math
import sys


def case_key(r):
    i = r['instruction']
    return (i.get('assembly'), i.get('bytes'),
            json.dumps(i.get('state'), sort_keys=True),
            json.dumps(i.get('taint'), sort_keys=True))


def _pct(values, p):
    v = sorted(values)
    if not v:
        return 0.0
    k = (len(v) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] if lo == hi else v[lo] + (v[hi] - v[lo]) * (k - lo)


def add_per_step_metrics(report):
    """Percentiles of ONE PROPAGATION STEP, for every engine, from the raw cases.

    The stored p50/p95/p99 are per TEST, and a test in the sequence pillars is up
    to 32 instructions, so their tail says how long a sequence takes rather than
    how long a step takes.  That made microtaint look worse at p99 than Triton
    while being 5x better at p50 -- an artifact of the corpus mix.  Recomputed
    here rather than read from the reports, because the reference run predates the
    fields and the two must be comparable.  Path-explosion is excluded, as it is
    from throughput.
    """
    for tool, m in report['metrics']['per_tool'].items():
        if tool == 'ground_truth':
            continue
        per_step, single = [], []
        for r in report['results']:
            ns = (r['tool_results'].get(tool) or {}).get('time_ns')
            if not ns:
                continue
            i = r['instruction']
            if 'path_explosion' in str(i.get('category')):
                continue
            n = max(1, (i.get('assembly') or '').count(';') + 1)
            per_step.append(ns / 1e6 / n)
            if n == 1:
                single.append(ns / 1e6)
        m['latency_p50_per_instr_ms'] = round(_pct(per_step, 50), 4)
        m['latency_p95_per_instr_ms'] = round(_pct(per_step, 95), 4)
        m['latency_p99_per_instr_ms'] = round(_pct(per_step, 99), 4)
        m['latency_p99_single_instr_ms'] = round(_pct(single, 99), 4)
        m['latency_p100_per_instr_ms'] = round(max(per_step) if per_step else 0.0, 4)


def main() -> int:
    argv = sys.argv[1:]
    timing_path = None
    if '--timing' in argv:
        i = argv.index('--timing')
        timing_path = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    ref_path, new_path, out_path = argv[0], argv[1], argv[2]
    ref = json.load(open(ref_path))
    new = json.load(open(new_path))

    rk = [case_key(r) for r in ref['results']]
    nk = [case_key(r) for r in new['results']]
    if rk != nk:
        print(f'REFUSED: corpora differ ({len(rk)} vs {len(nk)} cases, '
              f'{sum(a != b for a, b in zip(rk, nk, strict=False))} mismatched)')
        return 1
    print(f'corpus identical: {len(rk)} cases, same order')

    # The ground truth is the same oracle over the same inputs; check it agrees
    # before trusting either run's scoring of it.
    gt_diff = 0
    for a, b in zip(ref['results'], new['results'], strict=True):
        ga = (a['tool_results'].get('ground_truth') or {}).get('output_taint')
        gb = (b['tool_results'].get('ground_truth') or {}).get('output_taint')
        if ga != gb:
            gt_diff += 1
    print(f'ground-truth disagreements between the two runs: {gt_diff}')

    for a, b in zip(ref['results'], new['results'], strict=True):
        if 'microtaint' in b['tool_results']:
            a['tool_results']['microtaint'] = b['tool_results']['microtaint']

    ref['metrics']['per_tool']['microtaint'] = new['metrics']['per_tool']['microtaint']
    ref['metrics']['ground_truth']['per_tool']['microtaint'] = \
        new['metrics']['ground_truth']['per_tool']['microtaint']

    if timing_path is not None:
        timed = json.load(open(timing_path))
        tk = [case_key(r) for r in timed['results']]
        if tk != rk:
            print(f'REFUSED: the timing run scored a different corpus ({len(tk)} cases)')
            return 1
        # An ERRORED case carries `time_ns: 0`.  Splicing that in drops the error
        # field but keeps the zero, so the case reads as a completed run that took
        # no time -- it pulls the median down and removes the case from the tail,
        # invisibly.  This is not hypothetical: `precompile` raised AttributeError
        # on every ChainedCircuit, 753 of 943 sequence cases errored, and all 753
        # zeros were merged into the report the paper's latency figures came from.
        n_err = sum(1 for r in timed['results']
                    if (r['tool_results'].get('microtaint') or {}).get('error'))
        n_zero = sum(1 for r in timed['results']
                     if (r['tool_results'].get('microtaint') or {}).get('time_ns') == 0
                     and not (r['tool_results'].get('microtaint') or {}).get('error'))
        if n_err or n_zero:
            print(f'REFUSED: the timing run has {n_err} errored and {n_zero} zero-latency '
                  f'microtaint cases out of {len(tk)}. Splicing it would publish those as '
                  f'completed runs of zero duration. Fix the worker and re-run.')
            return 1

        ref['metrics']['per_tool']['microtaint'] = timed['metrics']['per_tool']['microtaint']
        # Per-CASE timings too, not just the summary: otherwise anything that
        # recomputes a percentile from `results` silently mixes the timing run's
        # summary with the soundness run's samples.
        for a, b in zip(ref['results'], timed['results'], strict=True):
            tb = b['tool_results'].get('microtaint')
            ta = a['tool_results'].get('microtaint')
            if tb is not None and ta is not None and 'time_ns' in tb:
                ta['time_ns'] = tb['time_ns']
        pt = ref['metrics']['per_tool']['microtaint']
        print(f"per-step block taken from {timing_path}: "
              f"p50={pt['latency_p50_ms'] * 1000:.0f}us tps={pt['throughput_per_s']:,.0f}")

    add_per_step_metrics(ref)

    ref['metadata']['merged_from'] = {
        'baselines': ref_path, 'microtaint': new_path,
        'timing': timing_path,
        'note': 'six baselines from the reference run; microtaint re-measured on the same corpus',
    }
    json.dump(ref, open(out_path, 'w'))
    g = ref['metrics']['ground_truth']['per_tool']
    print(f'\nwrote {out_path}')
    print(f"{'engine':12s} {'unsound':>8s} {'exact%':>8s}")
    for t, v in g.items():
        print(f"{t:12s} {v['unsound_cases']:8d} {v['exact_case_rate'] * 100:8.1f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
