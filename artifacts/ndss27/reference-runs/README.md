# Reference run

One run of `./run-all.sh --quick --no-baselines` on a minimal Debian 12 machine.
It is in `run-all-debian12/20260918-121340`, with the summary, the build log, and the stdout, the stderr and the JSON output of every experiment.

The results agree with the paper.
The nftables and siphash avalanche figures are identical (nftables 35 instructions and 60.0% of flag bits, siphash 325 and 63.6%), and the five architectures each compared 20000 cases with no under-taint.

The base64 column needs a word, because there are three base64 binaries and the paper's number is one of them.
The paper's 370 instructions, 16.1% of data bits and 53.3% of flag bits are the machine's own `/usr/bin/base64`, which this run reproduces to the digit under `\avlBsfSystem*`.
`run-all.sh` now puts the pinned Debian 12 coreutils in the headline column instead, and that binary gives 990 instructions, 3.5% of data bits and 53.6% of flag bits.
The flag share barely moves, the data share does not: how much of a program avalanches depends on which build of it you measure, which is the point of measuring three.

The overhead figures are the exception.
The machine is virtualised, so everything is slower and the timings are noisier than on real hardware: the emulation floor is 0.588 s here against 0.385 s in the paper, and the rest of the ladder follows.
The ordering of the rungs is what to check, not the exact magnitudes.

The engine comparison is absent because `--no-baselines` skips it, and with it the benchmark macros, the four engine figures and the evaluation tables that all read its report.

## What was generated from these runs afterwards

Both directories were run before the table and figure generators existed, so their outputs were produced afterwards from the JSON each run had already written.
Every command below reads only files that are in this repository.

From `run-all-debian12/20260918-121340`:

```sh
cd ../rq7-applications
uv run python gen_apps_tables.py \
    --results-dir ../reference-runs/run-all-debian12/20260918-121340 \
    --tex ../reference-runs/run-all-debian12/20260918-121340/apps_tables.tex \
    --md  ../reference-runs/run-all-debian12/20260918-121340/apps_tables.md

cd ../avalanche
uv run python gen_avalanche_macros.py \
    ../reference-runs/run-all-debian12/20260918-121340/avalanche_base64_pinned.json \
    ../reference-runs/run-all-debian12/20260918-121340/avalanche_nftables.json \
    ../reference-runs/run-all-debian12/20260918-121340/avalanche_siphash.json \
    --out ../reference-runs/run-all-debian12/20260918-121340/avalanche_numbers.tex

cd ../rq2-comparison
uv run --with matplotlib --with numpy python plot_figures.py \
    --overhead ../reference-runs/run-all-debian12/20260918-121340/overhead_results.json
mv fig_overhead.pdf ../reference-runs/run-all-debian12/20260918-121340/
```

That gives `apps_tables.tex` and `apps_tables.md` (the two RQ7 application tables, both matching the paper), `avalanche_numbers.tex` (the category table's macros) and `fig_overhead.pdf` (Figure 8).
The other four figures and the seven evaluation tables need the engine comparison, which this run skipped, so they cannot be produced from it.

From `rq6-campaign-ryzen5-3600-24h`:

```sh
cd ../rq6-generalisation/table5
uv run python table5.py --dir ../../reference-runs/rq6-campaign-ryzen5-3600-24h \
    --tex ../../reference-runs/rq6-campaign-ryzen5-3600-24h/table5.tex \
    --json ../../reference-runs/rq6-campaign-ryzen5-3600-24h/table5.json
```

That gives the cross-ISA table, with the full run printed to `table5.txt`.
It is a larger campaign than the paper's, 89.8 million cases against 55.2 million, so the percentages are not the paper's and are not meant to be.
What carries over is the column that matters: zero under-taints on all five architectures.

## Running it

The machine was a Debian 12 `genericcloud` image with eight cores and 4 GB of memory, holding 324 packages, no compiler and no git.

```sh
sudo apt-get update
sudo apt-get install -y git build-essential      # curl is already in the image

curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

git clone https://github.com/toby-bro/microtaint microtaint
cd microtaint
git checkout v0.7.2
uv sync --locked --all-extras

cd artifacts/ndss27
./run-all.sh --quick --no-baselines
```

It took one hour and forty five minutes.

## Running everything

The six compared engines and TaintInduce need more.
Docker must work without `sudo`, and group membership only takes effect in a new login session, so log out and back in after the `usermod`.

```sh
sudo apt-get install -y docker.io valgrind wget openssl
sudo usermod -aG docker "$USER"
# log out and back in here

cd microtaint/artifacts/ndss27
./rq2-comparison/setup_envs.sh                   # the six engines, about 5 GB
./rq1-synthesis_vs_inference/setup_taintinduce.sh

./run-all.sh
```

There is also a `./setup-all.sh` that runs both setup scripts and checks the dependencies first.
It is what we used on a fresh Debian 12 and it is only tested there, so on any other distribution the commands above are the reference and the package names are yours to translate.

`valgrind` is needed on the host and not only in the container, because the TaintGrind harness is compiled here against `/usr/include/valgrind`.
Both setup scripts can be run again if the network drops part way through; they pick up where they stopped rather than starting over.
`setup_envs.sh` downloads about 5 GB, most of it PANDA's guest image, and the whole tree then occupies 15 GB.
The full `./run-all.sh` takes about two days, dominated by the cross-ISA campaign.
