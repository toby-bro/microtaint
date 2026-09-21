# microtaint — NDSS 2027 artifact

## Presentation of the artifacts

All of the experiments mentionned in our submission are detailed and reproducible here.

For each we provide the exact command to reproduce the tables, and graphs in our paper.
Nevertheless as some of them are many hours long, we also provide the commands to run faster versions of each of these.

### Reproductibility

In order to facilitate reproductibility all the commands we ran in the paper are reported in this directory, this includes the randomness seeds.
Naturally the timings measured are subject to variations, for comparison the results we obtained were on a `AMD Ryzen 7 5700U (16) @ 1.80 GHz` on which boost was deactivated, and `cpupower` set to performance to stabilise the CPU's frequency and obtain meaningful results.

## Structure of the repository

### Taintinduce - rule synthesis comparison (`RQ1`)

While there is no functionnal implementation of taintinduce left on internet (the original repository has dead dependencies).
We will use a version we fixed.

The experiments are in [`rq1-synthesis_vs_inference`](./rq1-synthesis_vs_inference/).
A dedicated [README](./rq1-synthesis_vs_inference/README.md) is present in this directory to explain how to run the experiment.

### Engine comparison (`RQ2-4`)

The results obtained in the paper were through a campaign of about three hours (because some tools are much slower than microtaint).
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
In [rq7-applications](./rq7-applications/) is the dedicated [README](./rq7-applications/README.md), and in [other-engines](./rq7-applications/other-engines/) the same two analyses run against the six baselines used in `RQ2-4`

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

## tl;dr

If you have a standard linux machine running an x86 CPU this should work fine.

```sh
./setup-all.sh
./run-all.sh --quick
```

## Requirements

No dedicated hardware is needed, a commodity computer is sufficient.

The requirements below were established by installing the artifact from scratch
in a minimal Debian 12 virtual machine, a desktop install should provides all of the requirements.

### Short recap

For building microtaint the requirements are

- git (we version using vcs)
- compilers

For running the scripts

- a real shell (powershell does not count)
- uv

For compiling and running all the other tools

- docker, accessible from usermode
- valgrind (for taintgrind)
- openssl (to verify the intel pin signature for libdft)

Building all the depenencies takes less than half an hour of compute, the exact duration will largely depend on the network speed, to download the other tools.

Running all the experiments consumes less than 15GB of disk space, and can run on 4GB of ram, we recommend more.

### Long version (AI written)

On a Debian (and derivatives) system, the core experiments need exactly:

```sh
sudo apt install git build-essential curl
```

On Arch

```sh
sudo pacman -S --needed git base-devel curl python
```

<details>
<summary>Why ?</summary>

*Debian*: `git` because the artifact is a repository and the evaluated version is a tag, `build-essential` because microtaint compiles Cython and C extensions and because several experiments build their guest binaries from source, and `curl` only to fetch `uv` (skip it if `uv` is already installed).

*Arch*: `base-devel` is what provides `gcc`, `make` and `binutils`, and `binutils` is
where `ar` comes from. `python` only runs the harness scripts: `uv` brings its
own interpreter for everything the experiments import.

</details>

#### uv

All the provided scripts use [uv](https://docs.astral.sh/uv) to manage the dependencies and the Python version, and we highly recommend installing it to prevent any grueling work.

`uv` downloads its own interpreter, so the system Python does not matter.
Python 3.12 through 3.14 are supported and `uv` selects one automatically.

<details>
<summary> Why `>=3.12` ?</summary>

pypcode requires a python version later than 3.12, and when developping MicroTaint, we resorted to using `TypeAlias`es which only appear in python versions later than 3.12.

Support for the later versions (>=3.15) will come as soon as our dependencies support them (especially pypcode and qiling (and it's dependencies)).

</details>

#### Other engines

Three experiments need more than that, and each is skipped cleanly if the extra software is absent:

- The engine comparison needs the six baselines, and
  `rq2-comparison/setup_envs.sh` installs them all: for the python libraries
  (triton, maat, angr, microtaint) `uv` is the only thing needed, but panda and
  taintgrind need docker. Lastly for libdft64 we made a script which patches
  the source code to enable it to be compiled in 2026, and compiles it. Two
  host packages beyond the core three are required for it:

  ```sh
  sudo apt install docker.io valgrind # install docker how you want, but this works
  sudo usermod -aG docker "$USER"     # then start a NEW login session
  ```

  On Arch, where the `docker` package does not start the daemon for you:

  ```sh
  sudo pacman -S --needed docker valgrind wget openssl
  sudo systemctl enable --now docker
  sudo usermod -aG docker "$USER"   # then start a NEW login session
  ```

  `valgrind` because `benchmark.py` compiles TaintGrind's C harness on the host
  against `/usr/include/valgrind`; without it that engine silently drops out of
  the comparison. The script downloads roughly 5GB in total (mostly PANDA's 3GB guest image) and is quite idempotent,
  so a run interrupted by a network failure can simply be run again. See
  [`rq2-comparison/README.md`](./rq2-comparison/README.md).
- The taintinduce comparison needs our fork of taintinduce, which
  `rq1-synthesis_vs_inference/setup_taintinduce.sh` clones, pins and installs.
- Two of the three `base64` avalanche workloads are fetched on first run (a
  pinned Debian package and a pinned upstream release), so that build step
  needs network access once. The third measures the machine's own
  `/usr/bin/base64` and needs nothing.

#### How long it takes, and how much it costs

Measured on the minimal Debian 12 virtual machine described above, from a
cold start, with eight virtual cores and 4GB of memory.

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

For the experiments themselves, here is where a `--quick` run spends its time,
step by step, measured on sixteen cores with CPU boost disabled. The step names
are the ones in `SUMMARY.md`, and the order is the order they run in.

| step | time | note |
| --- | --- | --- |
| `avalanche-*` (six steps) | 28 s | calibration, three `base64` builds, nftables, siphash |
| `rq7-bof` `-uaf` `-sc` `-aiw` | 3 s | about a second each |
| `rq7-crypto-check` `-localise` | 8 s | |
| `rq7-dns` | 1 s | |
| `rq1-synthesis` | 10 min | needs TaintInduce; `--no-baselines` skips it |
| `rq7-other-*` (eight steps) | **43 min** | the six compared engines on the same two workloads; `--no-baselines` skips it. PANDA is 40 of those minutes, everything else is under two |
| `rq5-ladder` | **58 min** | the longest step by far, and it is one Python process |
| `rq5-bench` | 4 min | |
| `rq2-soundness` | 18 min | all seven engines; `--no-baselines` skips it |
| `rq6-pass1` | 8 min | five ISAs, 20000 cases each |
| `rq6-pass2` | 5 s | re-checks only what pass 1 flagged |
| **total** | **2 h 22 min** | 71 min with `--no-baselines` |

Two steps are most of it. The RQ5 ladder is 58 minutes because it measures each
rung repeatedly to separate configurations whose costs are close together, and
it is single-threaded, so more cores do not help it. The RQ7 baselines are 43,
of which PANDA is 40: it is the only full-system engine of the seven, so each
of its three workloads boots an entire Ubuntu guest under QEMU before any taint
runs. That is a fixed cost per workload and says nothing about its propagation
speed, which is the second fastest of the seven on the RQ2 corpus. Everything
else put together finishes inside forty minutes.

`--no-baselines` drops RQ1, RQ2 and the RQ7 baselines, which is why it is
71 minutes rather than 142. The same `./run-all.sh --quick --no-baselines` took
1 h 45 min on eight virtual cores. The full `./run-all.sh` takes about two
days, dominated by the cross-ISA campaign. As a smaller calibration point,
forty single-instruction cases scored against all seven engines take one
hundred seconds.

## Installing microtaint and the other engines

### Short version

If you want to install microtaint without using the `./setup-all.sh` script, then it can be done automatically with `uv` by compiling the code provided in this repository or downloading a pre-compiled version from the PyPi.

If all the dependencies are installed, then the following script downloads, compiles, builds... all, for linux machines.

```sh
./setup-all.sh
```

### Long version

The version evaluated in the paper is `v0.7.2` of microtaint, it can be installed locally (the whole compilation process takes a few minutes at most).

```sh
git checkout v0.7.2                   # the engine the paper evaluates
uv sync --locked --all-extras
```

Or using a pre-compiled version from PyPI.

```sh
uv init --bare --no-workspace --python 3.13 && uv add 'microtaint==0.7.2'
```

`./setup-all.sh` does the above and the six compared engines and TaintInduce in
one command, checking the dependencies first so a missing one is reported
before the downloads rather than after them. `./setup-all.sh --no-baselines`
installs microtaint alone. It is written for Debian (and spin-offs) package names
and only tested on debian 12, and arch linux (august 2026).

## Running everything

### Short version

```sh
./run-all.sh                # for a full 2 days run that reproduces the results in the paper
./run-all.sh --quick        # for a 2h version showing the claims of the paper
./run-all.sh --no-baselines # to remove the comparisons to the other engines (can be combined with --quick)
```

Each experiment writes its results to the `./results/` subdir.
The script regenerates the macros used in the paper, the tables and plots of the paper.

### Long version

`run-all.sh` runs every experiment in order, cheapest first, so a broken
environment surfaces in minutes rather than hours. Each one's raw output and a
verdict land under `results/<timestamp>/`.

```sh
./run-all.sh                    # the paper's corpus; about two days, dominated by RQ6
./run-all.sh --quick            # reduced corpus, about two hours; NOT the paper's numbers
./run-all.sh --no-baselines     # skip RQ1 and RQ2, the only steps needing the other engines
./run-all.sh --microtaint-only  # score RQ2 for this engine alone, with no oracle
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

#### What each experiment shows, and how to tell it worked

Each claim in the artifact abstract is supported by one experiment. `run-all.sh`
runs them all in the order below, cheapest first, and writes a verdict line per
experiment into `results/<timestamp>/SUMMARY.md`.

**The first success criterion is mechanical**: every line in `SUMMARY.md` reads
`PASS`, and the script exits zero. A `FAIL` is a broken run. A `SKIP` means the
experiment did not run at all, usually because software it needs is absent, and
is never counted as a success. One line is expected to skip: `tables-multiisa`
comes from the open-ended campaign in `rq6-generalisation/table5/`, which
`run-all.sh` does not start because it runs to a deadline the caller chooses.
On `--quick`, `tables-eval` also leaves the path-explosion table out, for the
same reason the reduced corpus leaves out the suites that feed it. The second criterion is that the numbers below
land near the paper's. They will not match exactly, and the third column says
how far off is reasonable.

| claim | experiment | what the paper reports | what a re-run should show |
| --- | --- | --- | --- |
| Taint rules can be generated for any instruction, unlike TaintInduce | `rq1-synthesis_vs_inference/run_rq1.sh` | TaintInduce converges on the narrow families only, and not at 64 bits | microtaint synthesises a rule for every instruction attempted. TaintInduce's failures are timeouts and non-convergence, so their exact count moves with the time budget |
| microtaint is sound on x86-64, and the other engines are not | RQ2, `rq2-comparison/benchmark.py` | 0 unsound cases for microtaint, over 9858 cases. angr 43, TaintGrind 32, Triton 206, libdft64 210, Maat 300, PANDA 464 | 0 for microtaint, exactly, whatever the corpus. The others are counts, not rates, so they fall with the corpus: `--quick` scores 600 cases, a sixteenth as many, and we measured 1, 0, 16, 7, 21 and 62 for the same six. Compare the shape, not the numbers, and expect engines whose paper counts are close (Triton and libdft64 differ by 2%) to swap places |
| microtaint is precise | RQ3, same run | 100.0% sound, 83.8% bit-exact, mean Jaccard 0.962 (the accepted version reports 82.4% and 0.953, from an earlier run of the same experiment) | within a point or so of either: a `--quick` run of our own gave 100.0%, 84.9% and 0.949. Sound% must stay at 100 |
| microtaint is faster per propagation than every other engine | RQ4, same run | see the two RQ4 figures, Figures 4 and 5 | absolute times are hardware-dependent and will differ. What must hold is that microtaint has the lowest p50 per-step latency of the seven |
| the overhead is plumbing, not taint propagation | RQ5, `rq5-overhead/` | the taint work is a small part of the total, 18.1x the emulation floor | the ratio is hardware-dependent. Disable CPU boost, or the rungs are not comparable with each other |
| microtaint is sound across ISAs, with little porting effort | RQ6, `rq6-generalisation/` | no under-taint on any of the five ISAs | zero under-taints. |
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

- `SUMMARY.md`, the verdict per experiment, which is the first thing to read
- `benchmark_numbers.tex` and `avalanche_numbers.tex`, every number the paper
  quotes, regenerated from this run. Each line is one value, named as it
  appears in the text, so they can be read against the paper directly
- `fig_unsoundness.pdf`, `fig_precision.pdf`, `fig_perf_latency.pdf`,
  `fig_perf_throughput.pdf` and `fig_overhead.pdf`, which are Figures 2 to 6,
  from `rq2-comparison/plot_figures.py`. The first four read the engine
  comparison and the fifth reads the overhead ladder, so a run that measured
  only one of the two gets only its figures
- `eval_tables.tex` and `eval_tables.md`: the soundness, ground-truth coverage,
  over-taint, unsoundness-summary, latency, F1 and path-explosion tables. These
  are the long-form counterparts of Figures 2 to 5 rather than tables in the
  paper, which reports those results as figures and as macros in the prose. A
  table whose inputs a run did not produce is skipped by name, so a `--quick`
  run emits the subset it measured instead of a table of zeroes
- `ladder_table.tex` and `ladder_table.md`: the appendix's overhead ladder,
  every rung with its wall, CPU, memory and per-instruction cost and its
  ratio against native and against bare Qiling
- `apps_tables.tex` and `apps_tables.md`: the two RQ7 application tables,
  from this run's constant-time and DNS results plus the checked-in verdicts
  of the other engines
- `table5.tex` and `table5.json`, the cross-ISA table, but only when the
  separate campaign in `rq6-generalisation/table5/` has been run. The RQ6 pass
  in `run-all.sh` uses a different harness and cannot produce it
- the JSON report behind each experiment, so any number can be recomputed

For comparison, `reference-runs/` holds the full output of runs of our own, so
a new run can be read against ours without waiting for the long experiments to
finish. It has four: a whole `--quick --no-baselines` run, the paper's own
9858-case engine comparison with its four figures regenerated from it, a
full-tier TaintInduce comparison, and a 24 hour cross-ISA campaign. Its
[README](./reference-runs/README.md) gives the command that regenerates each
table and figure from the JSON beside it.
