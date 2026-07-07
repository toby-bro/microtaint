#!/usr/bin/env python3
"""Generate the paper's benchmark \\newcommand block from a benchmark report JSON.

This is the single command that turns experiment output into the LaTeX numbers,
so after a re-run there is no manual transcription: the values in main.tex's
"Benchmark result numbers" block are produced verbatim by this script.

Usage:
    python gen_paper_macros.py REPORT.json [--overhead overhead_results.json]
                                           [--benchmark benchmark.py]
                                           [--out benchmark_numbers.tex]
                                           [--no-performance] [--no-overhead]

    - REPORT.json          : a benchmark.py report (report_*.json).
    - --overhead           : overhead/overhead_results.json (for the 3 overhead macros).
    - --benchmark          : benchmark.py, imported for the structural template/class
                             counts (defaults to ./benchmark.py next to this script).
    - --out                : also write the block to a file (e.g. paper/benchmark_numbers.tex,
                             which main.tex can \\input); otherwise prints to stdout.
    - --no-performance     : skip the latency/throughput/speedup macros (keep the paper's
                             existing perf numbers -- useful when the run was not a clean,
                             isolated latency measurement).
    - --no-overhead        : skip the overhead macros.

Only benchmark-derived numbers are emitted. Copy the block into main.tex, or point
main.tex at the --out file with \\input.
"""

import argparse
import importlib.util
import json
import os
import sys

# macro-prefix -> report tool key
ENGINES = {
    'mt': 'microtaint',
    'an': 'angr',
    'ma': 'maat',
    'tr': 'triton',
    'ld': 'libdft64',
    'tg': 'taintgrind',
    'pa': 'panda',
}


def texint(n):
    """Integer with LaTeX thousands separators: 9858 -> '9{,}858'."""
    return f'{int(round(n)):,}'.replace(',', '{,}')


def pct(rate):
    """Rate in [0,1] -> one-decimal percentage string: 0.8244 -> '82.4'."""
    return f'{rate * 100:.1f}'


def jac(x):
    return f'{x:.3f}'


def speedup(mt_tps, other_tps):
    """MicroTaint-relative speedup; integer when >= 10, else one decimal (4.3, 34, 6.3)."""
    r = mt_tps / other_tps
    return str(int(round(r))) if r >= 10 else f'{r:.1f}'


def load_structural(benchmark_py):
    """Import benchmark.py and read the generator's template/class counts."""
    spec = importlib.util.spec_from_file_location('bench_gen', benchmark_py)
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except SystemExit:
        pass
    pool = m.INSTRUCTION_POOL
    seq = m.INSTRUCTION_SEQUENCES
    pool_cats = {c for _, c in pool}
    seq_cats = {c for *_, c in seq}
    return {
        'numPoolTmpl': len(pool),
        'numPoolClasses': len(pool_cats),
        'numSeqTmpl': len(seq),
        'numSeqClasses': len(seq_cats),
    }


def corpus_derived(report):
    """Mnemonic and behaviour-class counts derived from the generated corpus."""
    mnem, classes = set(), set()
    for r in report['results']:
        inst = r['instruction']
        if isinstance(inst, str):
            import ast
            inst = ast.literal_eval(inst)
        mode = inst.get('mode')
        for part in str(inst.get('assembly', '')).split(';'):
            tok = part.strip().split()
            if tok:
                mnem.add(tok[0].lower())
        if mode in ('single', 'sequence'):
            classes.add(inst.get('category'))
    return {'numMnem': len(mnem), 'numClasses': len(classes)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('report', help='benchmark report_*.json')
    ap.add_argument('--overhead', default=None, help='overhead_results.json')
    ap.add_argument('--benchmark', default=None, help='benchmark.py (defaults to sibling file)')
    ap.add_argument('--out', default=None, help='write the block to this file as well as stdout')
    ap.add_argument('--no-performance', action='store_true')
    ap.add_argument('--no-overhead', action='store_true')
    args = ap.parse_args()

    with open(args.report) as f:
        rep = json.load(f)
    gt = rep['metrics']['ground_truth']
    gtt = gt['per_tool']
    pt = rep['metrics']['per_tool']
    md = rep['metadata']

    v = {}  # macro name -> already-formatted string

    # ---- Corpus / workload ----
    n_tests = rep['metrics']['total_cases']
    n_gt = gt['cases_within_budget']
    n_random = md['n_single']
    n_seq = md['n_sequence']
    n_sweep = md['n_sweep']
    v['numTests'] = texint(n_tests)
    v['numGT'] = texint(n_gt)
    v['pctGT'] = str(int(round(100 * n_gt / n_tests)))
    v['numRandom'] = texint(n_random)
    v['numSeq'] = texint(n_seq)
    v['numSweep'] = texint(n_sweep)
    v['numCurated'] = texint(n_tests - n_random - n_seq - n_sweep)
    v.update({k: texint(x) for k, x in corpus_derived(rep).items()})

    bench_py = args.benchmark or os.path.join(os.path.dirname(os.path.abspath(args.report)), 'benchmark.py')
    if not os.path.exists(bench_py):
        bench_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark.py')
    try:
        v.update({k: texint(x) for k, x in load_structural(bench_py).items()})
    except Exception as e:  # noqa: BLE001
        print(f'% WARNING: could not import {bench_py} for template counts ({e});', file=sys.stderr)
        print('%          numPoolTmpl/numPoolClasses/numSeqTmpl/numSeqClasses left for manual update.', file=sys.stderr)

    # ---- Ground-truth soundness / precision (per engine) ----
    for pre, key in ENGINES.items():
        g = gtt.get(key)
        if g is None:
            print(f'% WARNING: engine {key} absent from report; its macros are skipped.', file=sys.stderr)
            continue
        v[pre + 'Sound'] = pct(g['soundness_rate'])
        v[pre + 'Exact'] = pct(g['exact_case_rate'])
        v[pre + 'Uns'] = str(g['unsound_cases'])
        if g.get('mean_jaccard_bit') is not None:
            v[pre + 'Jac'] = jac(g['mean_jaccard_bit'])
        v[pre + 'Over'] = texint(g['over_bits_total'])
        v[pre + 'Under'] = texint(g['under_bits_total'])

    # ---- Per-step performance ----
    if not args.no_performance:
        mt_tps = pt['microtaint']['throughput_per_s']
        for pre, key in ENGINES.items():
            p = pt.get(key)
            if p is None:
                continue
            v[pre + 'Lat'] = texint(p['latency_p50_ms'] * 1000.0)
            v[pre + 'Tps'] = texint(p['throughput_per_s'])
            if key != 'microtaint':
                v[pre + 'Speedup'] = speedup(mt_tps, p['throughput_per_s'])

    # ---- End-to-end overhead ----
    if args.overhead and not args.no_overhead:
        with open(args.overhead) as f:
            ov = json.load(f)
        nat = ov['native']['wall_s']
        qil = ov['qiling-only']['wall_s']
        mta = ov['microtaint-all']['wall_s']
        run = ov['microtaint-all']['extra'].get('run_s') or mta
        v['ovhSlow'] = f'{mta / qil:.2f}'
        v['ovhNativeX'] = str(int(round(mta / nat)))
        v['ovhAmort'] = f'{run / nat:.1f}'

    # ---- Emit, in the same order/grouping as main.tex ----
    order = [
        ('Corpus / workload', ['numTests', 'numGT', 'pctGT', 'numMnem', 'numClasses', 'numRandom',
                               'numSeq', 'numSweep', 'numCurated', 'numPoolTmpl', 'numPoolClasses',
                               'numSeqTmpl', 'numSeqClasses']),
        ('Ground-truth soundness / precision (per engine)',
         [pre + suf for pre in ENGINES for suf in ('Sound', 'Exact', 'Jac', 'Over', 'Under', 'Uns')]),
        ('Per-step performance', [pre + suf for pre in ENGINES for suf in ('Lat', 'Tps', 'Speedup')]),
        ('End-to-end overhead', ['ovhSlow', 'ovhAmort', 'ovhNativeX']),
    ]
    lines = ['% Auto-generated by gen_paper_macros.py from ' + os.path.basename(args.report)
             + f" (seed {md.get('seed')}). Do not edit by hand; re-run the script."]
    for title, names in order:
        emitted = [n for n in names if n in v]
        if not emitted:
            continue
        lines.append(f'% ------- {title} -------')
        for n in emitted:
            lines.append(f'\\newcommand{{\\{n}}}{{{v[n]}}}')
    block = '\n'.join(lines) + '\n'

    sys.stdout.write(block)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(block)
        print(f'% wrote {args.out}', file=sys.stderr)


if __name__ == '__main__':
    main()
