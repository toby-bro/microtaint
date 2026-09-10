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

    merge_reports.py REFERENCE.json NEW_MICROTAINT.json OUT.json

Refuses to merge if the two runs did not score the same cases, which is the only
thing that makes the splice meaningful.
"""
# Experiment script, not library code: see artifacts/ndss27/README.md,
# "Lint and type checking", for why annotations are not required here.
# mypy: disable-error-code="no-untyped-def, no-untyped-call, type-arg, explicit-any"
import json
import sys


def case_key(r):
    i = r['instruction']
    return (i.get('assembly'), i.get('bytes'),
            json.dumps(i.get('state'), sort_keys=True),
            json.dumps(i.get('taint'), sort_keys=True))


def main() -> int:
    ref_path, new_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
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

    for block in ('per_tool',):
        if 'microtaint' in new['metrics'][block]:
            ref['metrics'][block]['microtaint'] = new['metrics'][block]['microtaint']
    ref['metrics']['ground_truth']['per_tool']['microtaint'] = \
        new['metrics']['ground_truth']['per_tool']['microtaint']

    ref['metadata']['merged_from'] = {
        'baselines': ref_path, 'microtaint': new_path,
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
