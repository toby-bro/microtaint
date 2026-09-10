# Engine comparison: soundness, precision, per-step cost (RQ2, RQ3, RQ4)

One run of `benchmark.py` answers all three questions, because all three score
the same corpus against the same ground truth. It is the paper's central
experiment and the most expensive to reproduce, since it needs all six
baselines.

```sh
./setup_envs.sh                 # once, installs the six baselines
uv run python benchmark.py      # ~3 h, dominated by container life-cycle
```

`benchmark.py` drives one worker per engine (`worker_angr.py`, `worker_maat.py`,
`worker_triton.py`, `worker_panda.py`, `worker_microtaint.py`) and compiles the
C harnesses for libdft64 and TaintGrind itself. An engine whose environment is
missing is recorded as skipped rather than crashing the run, so check the engine
list in the summary before concluding anything about a comparison.

The corpus is 9,858 tests over 127 mnemonics: random single instructions, short
sequences, systematic sweeps with five canonical taint masks, and the curated
cases (`imul` semantics, real-world idioms, bug seeds, path-explosion stress).
For the 3,263 tests with at most 15 tainted bits, every input combination is
enumerated on Unicorn, and that subset is where soundness and precision are
scored, because only there is the truth known rather than approximated.

What to read off the report:

- **RQ2** (Figure 4): unsound cases per engine. The claim is zero for
  microtaint and non-zero for every other engine.
- **RQ3** (Figure 5): exact%, mean Jaccard, over- and under-bits. Precision only
  means something conditional on soundness, since an engine that taints nothing
  is maximally precise and useless. Byte-granular engines promote a tainted bit
  to its whole byte; scoring accounts for that, which is why they come out
  sound-but-imprecise rather than unsound.
- **RQ4** (Figures 6 and 7): p50/p99 per-step latency and throughput. The
  path-explosion tests and tool startup are excluded; Appendix G gives each
  tool's timed region, which differ in scope.

Then regenerate the paper's numbers and figures from the report:

```sh
uv run python gen_paper_macros.py REPORT.json --overhead ../rq5-overhead/overhead_results.json
uv run --with matplotlib --with numpy python plot_figures.py REPORT.json ../rq5-overhead/overhead_results.json
```

The first writes the LaTeX macros the paper's text reads, so the prose and the
run cannot drift apart silently. The second writes the five figures. Both accept
the reference run shipped with the engine,
`benchmark/precision_soundess/report_1783378403.json` (seed 12), which is the
report the paper's numbers were generated from, so a reviewer can check the
tables and figures without waiting three hours first.

Exact and deterministic: fixed corpus, fixed masks, enumerated ground truth. A
rerun that finds an unsound microtaint case is a real disagreement with the
paper and is worth reporting.

`proofs/` holds the Z3 proofs of the category rules (Appendix A).
