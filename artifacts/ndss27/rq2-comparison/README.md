# Engine comparison: soundness, precision, per-step cost (RQ2, RQ3, RQ4)

One run of `benchmark.py` answers all three questions, because all three score
the same corpus against the same ground truth. It is the paper's central
experiment and the most expensive to reproduce, since it needs all six
baselines.

```sh
./setup_envs.sh                 # once, installs the six baselines
uv run python benchmark.py --seed 12    # ~3 h, dominated by container life-cycle
```

Seed 12 is the one the paper's corpus was drawn with. `benchmark.py` defaults
to no seed, which draws a fresh corpus on every run, so pass it explicitly
whenever two runs need to be comparable. `run-all.sh` passes it for you, and
`RQ2_SEED` overrides it.

This is the only experiment whose corpus is drawn at run time and whose seed
therefore has to be given. The cross-ISA campaign uses seed 1, TaintInduce uses
1 for inference and a separate seed for its held-out scoring set, the overhead
benchmark generates its input from a fixed seed, and the avalanche workloads
take fixed input and draw nothing.

`setup_envs.sh` needs, beyond the artifact's own requirements:

* **docker**, for three images: libdft64 and TaintGrind are built from
  commit-pinned upstream repositories, and PANDA is pulled by digest. Your user
  must be able to run `docker` without sudo, which after
  `sudo usermod -aG docker $USER` needs a NEW login session to take effect.
* **valgrind**, on the host. `benchmark.py` compiles TaintGrind's C harness
  itself with `-I/usr/include/valgrind`, so without the headers that engine
  reports `compile failed: valgrind.h: No such file or directory` and drops out
  of the comparison. This is separate from the TaintGrind container.
* **network**, for roughly 5GB: a 35MB Pin tarball, a ~1GB PANDA image, its
  **3GB** guest qcow, and the packages each container build installs. The qcow
  is fetched during setup on purpose: `benchmark.py` allows a worker 600s to
  boot, which a 3GB download cannot meet, so leaving it to the first benchmark
  run drops PANDA from the comparison.

The script is idempotent. Re-running it after a failure picks up where it
stopped rather than redoing the downloads, which matters because most of its
runtime is network and a partial run is the normal failure.

An engine whose environment is missing is recorded as errored and the run still
exits 0, so **read the per-engine table before quoting anything**: a comparison
missing a baseline looks exactly like a successful run in the exit code.

**TaintGrind's harness is built without libc**, unlike every other engine's.
TaintGrind's `expr2vbits_Unop` has no case for `Iop_Ctz64`/`Iop_Clz64`, which
glibc's string and startup routines emit, so a glibc-linked harness makes it
panic (`the 'impossible' happened`) before reaching the instruction under test.
The harness therefore defines its own `_start` and calls `write(2)` and
`clock_gettime(2)` directly. Its taint semantics are unaffected, since
`TNT_TAINT`/`TNT_IS_TAINTED` are Valgrind client requests rather than library
calls, but the measured region contains less surrounding code than the other
engines', which is worth knowing when reading its per-step cost.

`benchmark.py` drives one worker per engine (`worker_angr.py`, `worker_maat.py`,
`worker_triton.py`, `worker_panda.py`, `worker_microtaint.py`) and compiles the
C harnesses for libdft64 and TaintGrind itself. An engine whose environment is
missing is recorded as skipped rather than crashing the run, so check the engine
list in the summary before concluding anything about a comparison.

The corpus is several thousand tests over more than a hundred mnemonics: random
single instructions, short
sequences, systematic sweeps with five canonical taint masks, and the curated
cases (`imul` semantics, real-world idioms, bug seeds, path-explosion stress).
For the tests within the enumeration budget, every input combination is
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

The corpus and the masks are fixed by the seed, and the ground truth is an
enumeration, so the *inputs* are deterministic. The oracle's answers are not,
quite: it drives 2^k Unicorn emulations through one reused instance, and on a
long batch a few of them come back with register values no assignment produces.
That can add apparent under-taints, so a run's unsound count is an upper bound.

So verify before believing:

```sh
uv run python verify_unsound.py REPORT.json
```

It re-scores every reported unsound case against an isolated oracle (a fresh
Unicorn per emulation) and a freshly started worker, and prints CONFIRMED or
ORACLE ARTIFACT for each. Measured over five seeds of the full corpus: seeds 7
and 99 reported none, seed 42 reported one and seed 1 reported twenty-two, and
**every one of the twenty-three was the oracle**. One of them was `and rax, 0`,
an instruction whose output cannot depend on any input, with the oracle claiming
32 tainted bits in RAX.

A CONFIRMED case is a real disagreement with the paper and worth reporting.

## Cost

Measured on sixteen cores with CPU boost disabled.

| corpus | cases | time |
| --- | --- | --- |
| `--quick` (500 single + 100 sequence) | 600 | 18 min |
| the paper's corpus (7500 + 1000 + sweep) | 9858 | about 3 h |

This is the most demanding experiment for resources as well as time. It runs
seven engines at once, and peak memory across a whole `--quick` run was
1.65 GB, reached here rather than anywhere else, with angr the largest single
contributor. Installing the six engines needs about 5 GB of downloads and 13 GB
of disk, dominated by the container images.

## Regenerating the paper's tables

`gen_eval_tables.py` builds seven of the paper's tables from a report, in both
LaTeX and Markdown:

```sh
uv run python gen_eval_tables.py report_<stamp>.json --tex eval.tex --md eval.md
```

It emits the soundness and precision table, the ground-truth coverage, the
over-taint comparison, the per-engine unsoundness summary, the latency and
throughput table, the agreement against the reference labelling, and the
path-explosion scaling. A table whose inputs a run did not produce is named on
stderr and left out: a `--no-baselines` report has one engine, and a one-row
comparison is not a comparison. The path-explosion table needs the extra
suites, which the paper's corpus runs and a `--quick` corpus does not.

It refuses to state a soundness figure from a report that cannot say how many
cases each engine declined to answer, because an engine that errors on the hard
ones scores perfectly on a corpus it selected for itself. `--allow-uncertified`
emits the tables anyway and writes the caveat into them.

`run-all.sh` calls this itself and writes the result to the run directory.
