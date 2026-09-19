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
in a minimal Debian 12 virtual machine (the `genericcloud` image, 324 packages,
no compiler and no git), rather than by listing what the authors' machines
happened to have. A desktop install already provides all of it.

On a Debian or Ubuntu system, the core experiments need exactly:

```sh
sudo apt install git build-essential curl
```

`git` because the artifact is a repository and the evaluated version is a tag,
`build-essential` because microtaint compiles Cython and C extensions and
because several experiments build their guest binaries from source, and `curl`
only to fetch `uv` (skip it if `uv` is already installed).

All the provided scripts use [uv](https://docs.astral.sh/uv) to manage the
dependencies and the Python version, and we highly recommend installing it to
prevent any grueling work. `uv` downloads its own interpreter, so the system
Python does not matter: Debian 12 ships 3.11, which is below what microtaint
requires, and everything still works. Python 3.12 through 3.14 are supported
and `uv` selects one automatically.

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
