# microtaint — NDSS 2027 artifact

In this directory you will find all the experiments that were described in our paper.
The dir is split by research question.

## Presentation of the artifacts

All of the experiments mentionned in our submission are detailed and reproducible here.

For each we provide the exact command (seeds included) to be able to reproduce the tables, and graphs in our paper.
Nevertheless as some of them are many hours long, we also provide the commands to run faster versions of each of these.

### Reproductibility

In order to facilitate reproductibility all the commands we ran in the paper are reported in this directory, this includes the randomness seeds.
Naturally the timings measured are subject to variations, for comparison the results we obtained were on a `AMD Ryzen 7 5700U (16) @ 1.80 GHz` on which boost was deactivated, and `cpupower` set to performance, this stabilizes the CPU's frequency so as to have meaningful results.

### Requirements

No dedicated hardware is needed, a commodity computer is sufficient.

All the microtaints experiments can be run with a standard python installation running python 3.12 to 3.14.

All the provided scripts use [uv](https://docs.astral.sh/uv) to manage the dependencies, python versions... we highly recommend installing it, to prevent any grueling work.

The only experiment needing more software is the engine comparison one.
To that effect we include a script which installs them all.
For all the python libraries (triton, maat, angr, microtaint) `uv` is the only thing needed. But for panda and taintgrind docker is needed.

Lastly for libdft64 we made a script which patches the source code to enable it to be compiled in 2026, and compiles it.
Classical dev dependencies such as `make` and a C compiler are naturally required.

## Installing microtaint

The version evaluated in the paper is `v0.6.15` of microtaint, it can be installed locally, (the whole compilation process takes a few minutes at most).

```sh
git checkout v0.6.15                  # the engine the paper evaluates
uv sync --locked --all-extras
```

Or using pre-compiled version on the PyPi.

```sh
uv init --bare --no-workspace --python 3.13 && uv add 'microtaint==v0.6.15'
```

## Structure of the repository

### Taintinduce - rule synthesis comparison (`RQ1`)

While there is no functionnal implementation of taintinduce left on internet (the original repository has dead dependencies).
We will use a version we fixed.

The experiments are in [`rq1-synthesis_vs_inference`](./rq1-synthesis_vs_inference/).
A dedicated [README](./rq1-synthesis_vs_inference/README.md) is present in this directory to explain how to run the experiment.

### Engine comparison (`RQ2-4`)

The results obtained in the paper were through a 5h long campaign (because some tools are much slower than microtaint).
The exact command we used in [`rq2-comparison](./rq2-comparison/) is detailed in the dedicated [README](./rq2-comparison/README.md) along with indications on how to run a shorter test.

The results that this dir produce answer the soundness (`RQ2`), as well as the precision (`RQ3`) and speed (`RQ4`) questions with a detailed comparison of all the different engines at our disposal.

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
in the [rq7-applications](./rq7-applications/) you will find the dedicated [README](./rq7-applications/README.md)
