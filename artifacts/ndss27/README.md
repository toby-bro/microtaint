# microtaint — NDSS 2027 artifact

## Presentation of the artifacts

All of the experiments mentionned in our submission are detailed and reproducible here.

For each we provide the exact command to reproduce the tables, and graphs in our paper.
Nevertheless as some of them are many hours long, we also provide the commands to run faster versions of each of these.

### Reproductibility

In order to facilitate reproductibility all the commands we ran in the paper are reported in this directory, this includes the randomness seeds.
Naturally the timings measured are subject to variations, for comparison the results we obtained were on a `AMD Ryzen 7 5700U (16) @ 1.80 GHz` on which boost was deactivated, and `cpupower` set to performance to stabilise the CPU's frequency and obtain meaningful results.

### Requirements

No dedicated hardware is needed, a commodity computer is sufficient.

The requirements below were established by installing the artifact from scratch
in a minimal Debian 12 virtual machine, a desktop install should provides all of the requirements.

On a Debian or Ubuntu system, the core experiments need exactly:

```sh
sudo apt install git build-essential curl
```

`git` because the artifact is a repository and the evaluated version is a tag,
`build-essential` because microtaint compiles Cython and C extensions and
because several experiments build their guest binaries from source, and `curl`
only to fetch `uv` (skip it if `uv` is already installed).

All the provided scripts use [uv](https://docs.astral.sh/uv) to manage the dependencies and the Python version, and we highly recommend installing it to prevent any grueling work.
`uv` downloads its own interpreter, so the system Python does not matter.
Python 3.12 through 3.14 are supported and `uv` selects one automatically.

Three experiments need more than that, and each is skipped cleanly if the extra
software is absent:

* The engine comparison needs the six baselines, and
  `rq2-comparison/setup_envs.sh` installs them all: for the python libraries
  (triton, maat, angr, microtaint) `uv` is the only thing needed, but panda and
  taintgrind need docker. Lastly for libdft64 we made a script which patches
  the source code to enable it to be compiled in 2026, and compiles it. Two
  host packages beyond the core three are required for it:

  ```sh
  sudo apt install docker.io valgrind
  sudo usermod -aG docker "$USER"   # then start a NEW login session
  ```

  `valgrind` because `benchmark.py` compiles TaintGrind's C harness on the host
  against `/usr/include/valgrind`; without it that engine silently drops out of
  the comparison. The script downloads roughly 5GB in total (mostly PANDA's 3GB guest image) and is idempotent,
  so a run interrupted by a network failure can simply be run again. See
  [`rq2-comparison/README.md`](./rq2-comparison/README.md).
* The taintinduce comparison needs our fork of taintinduce, which
  `rq1-synthesis_vs_inference/setup_taintinduce.sh` clones, pins and installs.
* Two of the three `base64` avalanche workloads are fetched on first run (a
  pinned Debian package and a pinned upstream release), so that build step
  needs network access once. The third measures the machine's own
  `/usr/bin/base64` and needs nothing.

#### How long it takes, and how much it costs

Measured on the minimal Debian 12 virtual machine described above, from a
cold start, with four virtual cores and 4GB of memory.

| step | time | disk after |
| --- | --- | --- |
| `apt install git build-essential` | 1 min 31 s | |
| install `uv` | 3 s | |
| clone and check out the tag | 1 s | |
| `uv sync --locked --all-extras` | 1 min 58 s | 2.1 GB |
| **microtaint, total** | **3 min 33 s** | **2.1 GB** |
| the six engines and TaintInduce | about half an hour of work, plus 5 GB of downloads | 15 GB |

The last line is mostly network rather than work, so its wall-clock depends on
the connection: 5 GB of downloads, dominated by PANDA's 3 GB guest image.

Memory never exceeded 500 MB while installing microtaint, and peaked at
1.65 GB during an engine comparison with all seven engines running at once.
Neither ever touched swap on a machine that had none, so 4 GB is enough for
every experiment. We would still suggest 8 GB for the full corpus, since angr
is the memory-hungry engine and its footprint grows with the number of
symbolic states a case produces.

For the experiments themselves, `./run-all.sh --quick --no-baselines` took
1 h 45 min on eight virtual cores. The full `./run-all.sh` takes about two
days, dominated by the cross-ISA campaign. As a smaller calibration point,
forty single-instruction cases scored against all seven engines take one
hundred seconds.

## Installing microtaint

The version evaluated in the paper is `v0.7.2` of microtaint, it can be installed locally, (the whole compilation process takes a few minutes at most).

```sh
git checkout v0.7.2                   # the engine the paper evaluates
uv sync --locked --all-extras
```

Or using a pre-compiled version from PyPI.

```sh
uv init --bare --no-workspace --python 3.13 && uv add 'microtaint==0.7.2'
```

Every number in the paper comes from that one tag, measured in a single
re-run, so a figure and the prose around it cannot disagree about which engine
produced them. Experiments that take an engine path explicitly (the Table 5
campaign) accept `$MT_ENGINE_ROOT` so a frozen checkout can be measured while
the harness stays current.

## Running everything

`run-all.sh` runs every experiment in order, cheapest first, so a broken
environment surfaces in minutes rather than hours. Each one's raw output and a
verdict land under `results/<timestamp>/`.

```sh
./run-all.sh                 # the paper's corpus; about two days, dominated by RQ6
./run-all.sh --quick         # reduced corpus, about two hours; NOT the paper's numbers
./run-all.sh --no-baselines  # skip RQ1 and RQ2, the only steps needing the other engines
```

Every experiment appends a `PASS`, `FAIL` or `SKIP` line to
`results/<timestamp>/SUMMARY.md`, and the script exits non-zero if anything
failed. A `SKIP` names the command that builds the missing guest binary. Nothing
is overwritten: runs are timestamped so that two of them can be compared.

The four memory-safety detectors are checked against their *expected* exit code
rather than against zero, because for those a finding is the result; see
[`rq7-applications/README.md`](./rq7-applications/README.md).

The script pins `MICROTAINT_TAINT_IR=0` for every child process, and finishes by
regenerating the paper's macro files into the results directory.

The afore mentionned environment variable ensures that the microtaint engine evaluated by these scripts is the one described in the paper (as work on the engine has progressed whilst microtaint was under review).

## What each experiment shows, and how to tell it worked

Each claim in the artifact abstract is supported by one experiment. `run-all.sh`
runs them all in the order below, cheapest first, and writes a verdict line per
experiment into `results/<timestamp>/SUMMARY.md`.

**The first success criterion is mechanical**: every line in `SUMMARY.md` reads
`PASS`, and the script exits zero. A `FAIL` is a broken run. A `SKIP` means the
experiment did not run at all, usually because software it needs is absent, and
is never counted as a success. The second criterion is that the numbers below
land near the paper's. They will not match exactly, and the third column says
how far off is reasonable.

| claim | experiment | what the paper reports | what a re-run should show |
| --- | --- | --- | --- |
| Taint rules can be generated for any instruction, unlike TaintInduce | `rq1-synthesis_vs_inference/run_rq1.sh` | TaintInduce converges on the narrow families only, and not at 64 bits | microtaint synthesises a rule for every instruction attempted. TaintInduce's failures are timeouts and non-convergence, so their exact count moves with the time budget |
| microtaint is sound on x86-64, and the other engines are not | RQ2, `rq2-comparison/benchmark.py` | 0 unsound cases for microtaint, over 9858 cases. angr 43, TaintGrind 32, Triton 206, libdft64 210, Maat 300, PANDA 464 | 0 for microtaint, exactly, whatever the corpus. The others are counts, not rates, so they fall with the corpus: `--quick` scores 600 cases, a sixteenth as many, and we measured 1, 0, 16, 7, 21 and 62 for the same six. Compare the shape, not the numbers, and expect engines whose paper counts are close (Triton and libdft64 differ by 2%) to swap places |
| microtaint is precise | RQ3, same run | 100.0% sound, 83.8% bit-exact, mean Jaccard 0.962 (the accepted version reports 82.4% and 0.953, from an earlier run of the same experiment) | within a point or so of either: a `--quick` run of our own gave 100.0%, 84.9% and 0.949. Sound% must stay at 100 |
| microtaint is faster per propagation than every other engine | RQ4, same run | see Figures 6 and 7 | absolute times are hardware-dependent and will differ. What must hold is that microtaint has the lowest p50 per-step latency of the seven |
| the overhead is plumbing, not taint propagation | RQ5, `rq5-overhead/` | the taint work is a small part of the total, 18.1x the emulation floor | the ratio is hardware-dependent. Disable CPU boost, or the rungs are not comparable with each other |
| microtaint is sound across ISAs, with little porting effort | RQ6, `rq6-generalisation/` | no under-taint on any of the five ISAs | zero under-taints. This is a soundness claim, so any non-zero result is a real finding and worth reporting to us |
| bit precision finds what byte-granular engines cannot | RQ7, `rq7-applications/` | the DNS case is a false positive for every byte-granular engine | microtaint separates the two fields, the byte-granular engines do not |
| how much the avalanche fallback costs | `avalanche/` | base64 16.1% of data bits, nftables and siphash 0.0% | identical, if the same binaries are measured. See the note on base64 below |

Two of these deserve a warning.

**The avalanche base64 row depends on the binary.** Three builds of GNU base64
gave 18.74%, 18.74% and 15.97% avalanche on one machine, so the number is a
property of the binary as much as of the engine. The artifact therefore
measures three: a hash-pinned Debian package, a build from pinned upstream
source, and the machine's own `/usr/bin/base64`. Read the `debian12` column,
which is the pinned one and the reference.

The accepted version of the paper measured the development machine's own
coreutils 9.11 rather than a pinned binary. The pinned Debian package is
coreutils 9.1, and it gives the same 370 instructions, 16.1% of data bits and
53.3% of flag bits, so pinning changed which binary is named but not the
values. The other two columns are there to show how far the number moves when
the binary changes, and are not comparable with the paper.

**If one of the six other engines is not working, the comparison fails rather
than leaving it out.** A failure here points at that engine's installation,
not at microtaint, and the message names the engines that produced nothing.
Re-running `rq2-comparison/setup_envs.sh` is the usual fix, and
`--allow-missing-engines` accepts a partial comparison deliberately.

### What a run regenerates

Everything lands in `results/<timestamp>/`, alongside the raw output:

* `SUMMARY.md`, the verdict per experiment, which is the first thing to read
* `benchmark_numbers.tex` and `avalanche_numbers.tex`, every number the paper
  quotes, regenerated from this run. Each line is one value, named as it
  appears in the text, so they can be read against the paper directly
* `fig_unsoundness.pdf`, `fig_precision.pdf`, `fig_perf_latency.pdf`,
  `fig_perf_throughput.pdf` and `fig_overhead.pdf`, which are Figures 4 to 8
* the JSON report behind each experiment, so any number can be recomputed

For comparison, `reference-runs/` holds the full output of runs of our own,
including a complete cross-ISA campaign, so a new run can be read against ours
without waiting for the long experiments to finish.

## Structure of the repository

### Taintinduce - rule synthesis comparison (`RQ1`)

While there is no functionnal implementation of taintinduce left on internet (the original repository has dead dependencies).
We will use a version we fixed.

The experiments are in [`rq1-synthesis_vs_inference`](./rq1-synthesis_vs_inference/).
A dedicated [README](./rq1-synthesis_vs_inference/README.md) is present in this directory to explain how to run the experiment.

### Engine comparison (`RQ2-4`)

The results obtained in the paper were through a 5h long campaign (because some tools are much slower than microtaint).
The exact command we used in [`rq2-comparison`](./rq2-comparison/) is detailed in the dedicated [README](./rq2-comparison/README.md) along with indications on how to run a shorter test.

The results that this dir produce answer the soundness (`RQ2`), as well as the precision (`RQ3`) and speed (`RQ4`) questions with a detailed comparison of all the different engines at our disposal.
One run of `benchmark.py` scores all three, and the same directory regenerates the paper's figures and its LaTeX macros from that run.

It also holds [`proofs`](./rq2-comparison/proofs/), the Z3 proofs that each category's rule can never under-taint. These are not part of the paper's artifacts officially, they are "bonus".

### Overhead evaluation (`RQ5`)

The overhead of using microtaint to instrument a binary is evaluated in [rq5-overhead](./rq5-overhead/), and a dedicated [README](./rq5-overhead/README.md) explains how to run the experiment.

This experiment instruments a binary and compares execution time when it is executed natively, through qiling, and with microtaint.

### ISA generalisation (`RQ6`)

To evaluate the ISA generalisation, we started off the `RQ2` harness, but only evaluate microtaint on `x86-64`, `ARM64`, `MIPS64`, `PowerPC32`, `RISCV64GC`.
Some of these ISAs are little endian, others big endian, some are 32bits, others 64bits.

The detailed directory of this experiment are unsurprisignly in [rq6-generalisation](./rq6-generalisation/), and you will find a dedicated [README](./rq6-generalisation/README.md).

### Security analyses (`RQ7`)

The last part of our evaluation shows two programs in which bit-level granularity enables security analyses that were not achievable before.
The two examples are a DNS header parser, and a square and multiply implementation.
We also include four vulnerable binaries in [memory-safety](./rq7-applications/memory-safety/) to show that microtaint finds buffer overflows, use-after-free, side channels and writes through attacker-controlled pointers in classical binaries, which is the battery every engine is expected to have.
In [rq7-applications](./rq7-applications/) is the dedicated [README](./rq7-applications/README.md), and in [other-engines](./rq7-applications/other-engines/) the same two analyses run against the six baselines previously used in `RQ2-4`

### Avalanche's cost

At the behest of our gracious reviewers we also added an experiment to caracterise where and when an avalanche was triggered, and the amount of bits which are consequence of avalanche.
It runs on three real workloads (coreutils `base64`, the `nft_byteorder` routine of the Linux nftables module, and SipHash-2-4) and produces the two halves of the category table.

All related experiments and [README](./avalanche/README.md) can be found in the [avalanche](./avalanche) directory.

## Code quality

Whilst all of the code presented in this directory has been AI-generated,
all the scripts were re-read by the authors, and to the best of our knowledge,
they seem to do what we want them to.

We applied the same quality-requirements as the main code to try and prevent as many bugs as possible, and facilitate comprehension.
This includes strict type checking (with `mypy`) and linting (with `ruff`).
