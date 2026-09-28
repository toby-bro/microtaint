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

# Experiment script, not library code: see artifacts/ndss27/README.md,
# "Lint and type checking", for why annotations are not required here.
# mypy: disable-error-code="no-untyped-def, no-untyped-call, type-arg"

import argparse
import importlib.util
import json
import os
import sys

# macro-prefix -> report tool key
#: Mirrors benchmark.GT_MIN_ANSWER_RATE.  Kept as a literal so this script
#: refuses an under-answered report even when run against an old benchmark.py.
MIN_ANSWER_RATE = 0.5

#: Instruction families the evaluation prose names when it breaks an engine's
#: unsoundness down.  The paper used to carry these counts as literals, and they
#: went stale: it said TaintGrind missed "29x cmov" with a three-case tail on
#: variable shifts, which was 32 cases, while the run it shipped beside reported
#: 7.  Each family emits <prefix>Uns<Family> when the engine has any, so the
#: sentence and the run cannot disagree.
#:
#: Matched on the MNEMONIC, longest prefix first, so `cmovnp` is a cmov and not a
#: `cmp`.  A mnemonic in no family lands in `Other`, which is emitted too: a
#: breakdown that silently omits cases is a breakdown that adds up to less than
#: the total.
UNSOUND_FAMILIES = (
    ('Cmov', ('cmov',)),
    # No other x86 mnemonic begins with `set`, so this is the setcc family.
    ('Setcc', ('set',)),
    ('Shift', ('shrd', 'shld', 'shlx', 'shrx', 'sarx', 'shl', 'shr', 'sar',
               'rol', 'ror', 'rcl', 'rcr')),
    # pdep and pext are named individually because the prose names them
    # individually; `BitPerm` keeps the rest of the family.
    ('Pdep', ('pdep',)),
    ('Pext', ('pext',)),
    ('BitPerm', ('bzhi', 'bextr')),
    ('Mul', ('imul', 'mul')),
    ('Lea', ('lea',)),
)


def unsound_families(rep, tool):
    """Unsound cases per instruction family, for one tool.

    Unsound means the ground truth tainted something the tool left clean.  What
    "something" is depends on the tool's granularity, and it must be decided the
    same way benchmark.py decides it, or the breakdown describes a different
    population from the total it is broken down from:

      bit-granular tools  a MISSED BIT (`gt & ~got`), so a tool that taints the
                          register but not every dependent bit is unsound
      reg-granular tools  binarised, a register the tool left entirely clean

    Scoring maat (bit) with the register test found 93 of its 300 unsound cases,
    because most of its misses are individual bits inside a register it did
    taint.  The caller's sum check is what caught that.
    """
    regs = ('RAX', 'RBX', 'RCX', 'RDX')
    mask64 = (1 << 64) - 1
    gran = (rep['metadata'].get('granularity') or {}).get(tool, 'bit')
    counts = {}
    for case in rep['results']:
        tr = case['tool_results']
        gt_m = (tr.get('ground_truth') or {}).get('output_taint')
        t_m = (tr.get(tool) or {}).get('output_taint')
        if gt_m is None or t_m is None:
            continue
        if gran == 'reg':
            unsound = any(gt_m.get(r, 0) and not t_m.get(r, 0) for r in regs)
        else:
            unsound = any((gt_m.get(r, 0) & ~t_m.get(r, 0)) & mask64
                          for r in regs)
        if not unsound:
            continue
        mnem = (case['instruction'].get('assembly') or '').split()
        mnem = mnem[0].lower() if mnem else ''
        fam = next((name for name, pres in UNSOUND_FAMILIES
                    if any(mnem.startswith(x) for x in pres)), 'Other')
        counts[fam] = counts.get(fam, 0) + 1
    return counts

#: Below this, the detector overhead is indistinguishable from noise and is not
#: emitted as a percentage.  The six detector configurations in the shipped run
#: all fall within 0.5% of one another.
OVH_DETECT_MIN_PCT = 0.5

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
    return f'{round(n):,}'.replace(',', '{,}')


def pct(rate):
    """Rate in [0,1] -> one-decimal percentage string: 0.8244 -> '82.4'."""
    return f'{rate * 100:.1f}'


def jac(x):
    return f'{x:.3f}'


def speedup(mt_tps, other_tps):
    """MicroTaint-relative speedup; integer when >= 10, else one decimal (4.3, 34, 6.3)."""
    r = mt_tps / other_tps
    return str(round(r)) if r >= 10 else f'{r:.1f}'


def load_structural(benchmark_py):
    """Import benchmark.py and read the generator's template/class counts."""
    spec = importlib.util.spec_from_file_location('bench_gen', benchmark_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import {benchmark_py} as a module')
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
    ap.add_argument(
        '--allow-unresolved',
        action='store_true',
        help=(
            'Emit the macros that ARE resolvable and omit the ones this run '
            'cannot measure, instead of refusing outright. Intended for the '
            'reduced corpus, whose overhead ladder has too few repetitions to '
            'separate the detectors from the noise. The omitted names are '
            'listed in the output, so a paper that uses one fails to build '
            'rather than quoting a number nobody measured.'
        ),
    )
    args = ap.parse_args()

    with open(args.report) as f:
        rep = json.load(f)
    gt = rep['metrics']['ground_truth']
    gtt = gt['per_tool']
    pt = rep['metrics']['per_tool']
    md = rep['metadata']

    v = {}  # macro name -> already-formatted string
    unresolved = []  # macros this run could not measure (see --allow-unresolved)

    # ---- Corpus / workload ----
    n_tests = rep['metrics']['total_cases']
    n_gt = gt['cases_within_budget']
    n_random = md['n_single']
    n_seq = md['n_sequence']
    n_sweep = md['n_sweep']
    v['numTests'] = texint(n_tests)
    v['numGT'] = texint(n_gt)
    v['pctGT'] = str(round(100 * n_gt / n_tests))
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
    except Exception as e:
        print(f'% WARNING: could not import {bench_py} for template counts ({e});', file=sys.stderr)
        print('%          numPoolTmpl/numPoolClasses/numSeqTmpl/numSeqClasses left for manual update.', file=sys.stderr)

    # ---- Ground-truth soundness / precision (per engine) ----
    for pre, key in ENGINES.items():
        g = gtt.get(key)
        if g is None:
            print(f'% WARNING: engine {key} absent from report; its macros are skipped.', file=sys.stderr)
            continue
        # A rate computed over a self-selected subset must not reach the paper.
        # A tool that ERRORS on the cases it cannot handle and answers the rest
        # scores 100% sound on what is left: the reviewer's saboteur answered 27
        # of 157 cases and this emitted \mtSound{100.0} from it.  `answer_rate`
        # is absent from reports written before that was counted, so an old
        # report is refused rather than trusted.
        rate = g.get('answer_rate')
        if rate is None:
            raise SystemExit(
                f'{key}: this report predates answer-rate accounting, so the '
                f'fraction of cases it actually answered is unknown and its '
                f'soundness macro cannot be certified.  Re-run benchmark.py.',
            )
        if rate < MIN_ANSWER_RATE:
            raise SystemExit(
                f'{key}: answered only {g["cases_compared"]} of '
                f'{g["cases_attempted"]} cases ({100 * rate:.1f}%), erroring on '
                f'{g["cases_errored"]}.  Its rates describe a subset it selected '
                f'for itself, so they are not emitted as macros.',
            )
        v[pre + 'Sound'] = pct(g['soundness_rate'])
        v[pre + 'Exact'] = pct(g['exact_case_rate'])
        v[pre + 'Uns'] = str(g['unsound_cases'])
        if g.get('mean_jaccard_bit') is not None:
            v[pre + 'Jac'] = jac(g['mean_jaccard_bit'])
        v[pre + 'Over'] = texint(g['over_bits_total'])
        v[pre + 'Under'] = texint(g['under_bits_total'])
        # The breakdown the prose quotes, from the same run as the total.
        _fams = unsound_families(rep, key)
        for _fam, _n in sorted(_fams.items()):
            v[f'{pre}Uns{_fam}'] = str(_n)
        if _fams and sum(_fams.values()) != g['unsound_cases']:
            raise SystemExit(
                f'{key}: the unsound-family breakdown sums to '
                f'{sum(_fams.values())} but the report says '
                f"{g['unsound_cases']} unsound cases.  One of the two is "
                f'measuring a different population, so neither is emitted.',
            )

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
        # overhead_results.json is still read, but ONLY for the vacuity check
        # below: every published overhead macro now comes from the ladder, so a
        # ratio in one sentence cannot describe a different run from the next.
        # A run in which the guest never executed the workload is FAST and clean,
        # so no threshold on the timings can catch it -- only the guest's own
        # output can.  The shipped 2026-05 json had `guest_bytes` absent entirely
        # and 18 ns/instr of "taint", and its three macros went into the paper.
        for _label in ('native', 'qiling-only', 'microtaint-all'):
            _gb = ov[_label].get('extra', {}).get('guest_bytes')
            if _gb is None:
                raise SystemExit(
                    f'{args.overhead}: {_label} has no guest_bytes, so it predates the '
                    f'workload-ran check and cannot be trusted. Re-run overhead_bench.py '
                    f'(it now records it) rather than publishing these numbers.')
            if _gb <= 0:
                raise SystemExit(
                    f'{args.overhead}: {_label} guest wrote 0 bytes -- the workload never '
                    f'ran, so this file measures process startup, not taint.')

        # ovhVsQilingHooks is microtaint-all against `c-codehook-regs`: a pure-C
        # per-instruction hook that reads four guest registers, i.e. the work any
        # per-instruction dynamic analysis owes before it analyses anything.
        # That rung is the paper's `Qiling+hooks` configuration, and it is the
        # ONLY baseline the main text quotes -- bare Qiling appears nowhere
        # outside the appendix, so no macro should divide by it.
        #
        # It comes from the LADDER, not overhead_results.json, because the floor
        # rung only exists there.
        lad_path = os.path.join(os.path.dirname(args.overhead), 'overhead_ladder.json')
        try:
            with open(lad_path) as f:
                lad = json.load(f)
        except OSError as exc:
            raise SystemExit(
                f'{lad_path}: needed for ovhSlow (the plumbing floor rung lives only in '
                f'the ladder). Run overhead_ladder.py.') from exc
        floor = lad['layers']['c-codehook-regs']['run_s']
        mt_all = lad['layers']['microtaint-all']['run_s']
        v['ovhVsQilingHooks'] = f'{mt_all / floor:.1f}'
        v['ovhFloorNs'] = f"{floor * 1e9 / lad['guest_instructions']:,.1f}".replace(',', '{,}')
        # The TAINT PHASE (`ql.run`) of each, for the parenthetical in the
        # prose.  These were called ovhAllWall/ovhFloorWall and documented as
        # "wall-clock of the whole process" while assigning run_s, so the paper
        # said "wall" about a number that excluded interpreter start-up and
        # emulator construction.  Named for what they hold.
        v['ovhAllRun'] = f'{mt_all:.3f}'
        v['ovhFloorRun'] = f'{floor:.3f}'
        _rss = lad['layers']['microtaint-all'].get('peak_rss_mib') or 0
        _rss0 = lad['layers']['c-codehook-regs'].get('peak_rss_mib') or 0
        v['ovhRssFloor'] = f'{round(_rss0)}'
        v['ovhRssAll'] = f'{round(_rss)}'
        v['ovhRssDelta'] = f'{round(_rss) - round(_rss0)}'
        # One-shot setup = wall minus the taint phase.  The old prose said setup
        # cost twice the propagation phase; on a 3.8M-instruction workload the
        # relation is the other way round, so the sentence has to state which.
        _wall = lad['layers']['microtaint-all'].get('wall_s') or 0
        v['ovhSetupS'] = f'{max(_wall - mt_all, 0):.2f}'
        # Detection's share of the engine, for the RQ5 summary sentence.
        #
        # This can come out NEGATIVE, and on the shipped ladder it does: the
        # detectors measure 0.24% FASTER than running without them, so the macro
        # evaluated to `-0` and the sentence "the detectors add \ovhDetectPct\%"
        # rendered "the detectors add -0%".  A difference that small is noise
        # between two configurations whose six variants all sit within 0.5% of
        # each other, not a measured cost, and emitting it as a percentage
        # claims a precision the data does not have.  Refuse instead: a
        # measurement that cannot distinguish the two must not become a sentence
        # asserting one is dearer.
        _none = lad['layers']['microtaint-none']['run_s']
        _detect_pct = 100 * (mt_all - _none) / mt_all
        if _detect_pct < OVH_DETECT_MIN_PCT:
            _msg = (
                f'ovhDetectPct is {_detect_pct:.2f}%: with the detectors on, the '
                f'run measured {_none - mt_all:.4f}s FASTER than with them off, '
                f"so their cost is below this experiment's noise floor and "
                f'cannot be stated as a percentage.  Re-run the ladder with more '
                f'repetitions, or drop the claim.'
            )
            if not args.allow_unresolved:
                raise SystemExit(_msg)
            print(f'[!] omitting ovhDetectPct: {_msg}', file=sys.stderr)
            unresolved.append('ovhDetectPct')
        else:
            v['ovhDetectPct'] = f'{_detect_pct:.0f}'
        # Both of these come from the LADDER too, so every overhead macro is from
        # one measurement.  Mixing a ratio from overhead_results.json with one
        # from the ladder is how two numbers in the same paragraph end up
        # describing two different runs.
        v['ovhNativeX'] = f'{mt_all / lad["layers"]["native"]["run_s"]:,.0f}'.replace(',', '{,}')

        # The workload's size, counted by a code hook on this same ladder, so
        # that no caption or sentence has to carry it as a literal.
        _instr = lad['guest_instructions']
        v['ovhInstrExact'] = f'{_instr:,}'.replace(',', '{,}')
        v['ovhInstrM'] = f'{_instr / 1e6:.1f}'

        # The emulator's OWN share of the engine's run.  This is the number that
        # decides the paragraph's conclusion: if it is small, the dominant cost
        # is what we hang off the emulator, not the emulator.
        _qil = lad['layers']['qiling-only']['run_s']
        v['ovhQilRun'] = f'{_qil:.3f}'
        v['ovhQilPct'] = f'{100 * _qil / mt_all:.1f}'

        # The hosting-language penalty, from the two rungs that differ ONLY in
        # whether the identical empty hook body is C or Python.
        _c_hook = lad['layers']['c-codehook']['run_s']
        _py_hook = lad['layers']['codehook']['run_s']
        v['ovhCHookRun'] = f'{_c_hook:.3f}'
        v['ovhPyHookRun'] = f'{_py_hook:.3f}'
        v['ovhPyPenalty'] = f'{_py_hook / _c_hook:.0f}'
        v['ovhPyHookRegsRun'] = f'{lad["layers"]["codehook-regs"]["run_s"]:.1f}'
        v['ovhPlumbingRun'] = f'{lad["layers"]["microtaint-plumbing"]["run_s"]:.3f}'
        v['ovhNoneRun'] = f'{_none:.3f}'

        # Repetitions, so the appendix's method sentence is generated too.  The
        # two Python-hosted rungs get fewer because at seconds per run they
        # would otherwise dominate the ladder's wall-clock.
        v['ovhRuns'] = str(lad['layers']['microtaint-all'].get('n_runs') or 0)
        v['ovhRunsPyHook'] = str(lad['layers']['codehook'].get('n_runs') or 0)
        v['ovhRunsPyHookRegs'] = str(lad['layers']['codehook-regs'].get('n_runs') or 0)

    # ---- Emit, in the same order/grouping as main.tex ----
    order = [
        ('Corpus / workload', ['numTests', 'numGT', 'pctGT', 'numMnem', 'numClasses', 'numRandom',
                               'numSeq', 'numSweep', 'numCurated', 'numPoolTmpl', 'numPoolClasses',
                               'numSeqTmpl', 'numSeqClasses']),
        ('Ground-truth soundness / precision (per engine)',
         [pre + suf for pre in ENGINES for suf in ('Sound', 'Exact', 'Jac', 'Over', 'Under', 'Uns')]),
        ('Unsoundness by instruction family (per engine)',
         [f'{pre}Uns{fam}' for pre in ENGINES
          for fam in (*(n for n, _ in UNSOUND_FAMILIES), 'Other')]),
        ('Per-step performance', [pre + suf for pre in ENGINES for suf in ('Lat', 'Tps', 'Speedup')]),
        ('End-to-end overhead', ['ovhVsQilingHooks', 'ovhFloorNs', 'ovhAllRun', 'ovhFloorRun',
                                 'ovhRssFloor', 'ovhRssAll', 'ovhRssDelta', 'ovhSetupS', 'ovhDetectPct',
                                 'ovhNativeX', 'ovhInstrExact', 'ovhInstrM', 'ovhQilRun', 'ovhQilPct',
                                 'ovhCHookRun', 'ovhPyHookRun', 'ovhPyPenalty', 'ovhPyHookRegsRun',
                                 'ovhPlumbingRun', 'ovhNoneRun', 'ovhRuns',
                                 'ovhRunsPyHook', 'ovhRunsPyHookRegs']),
    ]
    lines = ['% Auto-generated by gen_paper_macros.py from ' + os.path.basename(args.report)
             + f" (seed {md.get('seed')}). Do not edit by hand; re-run the script."]
    if unresolved:
        # Name them here rather than leaving a silent hole.  A paper that uses
        # one of these will fail to build, which is the point: the run could not
        # measure it, so no value should be quoted for it.
        lines.append('% NOT MEASURED by this run, and therefore not defined below: '
                     + ', '.join(unresolved) + '.')
        lines.append('% The reduced corpus cannot resolve these. Run the full corpus for them.')
    for title, names in order:
        emitted = [n for n in names if n in v]
        if not emitted:
            continue
        lines.append(f'% ------- {title} -------')
        for n in emitted:
            lines.append(f'\\newcommand{{\\{n}}}{{{v[n]}}}')
    # `order` is a whitelist, so a macro computed above but not named in it is
    # dropped WITHOUT a word.  That is how the unsound-family macros first came
    # out of this script as an empty section: they were measured, put in `v`, and
    # silently discarded.  Refuse instead.
    _listed = {n for _, names in order for n in names}
    _dropped = sorted(set(v) - _listed)
    if _dropped:
        raise SystemExit(
            'these macros were computed but are not listed in `order`, so they '
            'would be silently dropped: ' + ', '.join(_dropped) +
            '.  Add them to a group in `order`.',
        )
    block = '\n'.join(lines) + '\n'

    sys.stdout.write(block)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(block)
        print(f'% wrote {args.out}', file=sys.stderr)


if __name__ == '__main__':
    main()
