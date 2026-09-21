# Reference runs

Four directories, one per experiment that has a long run behind it.

| directory | experiment | what it holds |
| --- | --- | --- |
| `run-all-debian12/20260918-121340` | RQ5, RQ6 quick, RQ7, avalanche | a whole `--quick --no-baselines` run |
| `rq2-comparison-paper-corpus` | RQ2, RQ3, RQ4 | the paper's own 9858-case engine comparison |
| `rq1-taintinduce` | RQ1 | the full-tier TaintInduce comparison |
| `rq6-campaign-ryzen5-3600-24h` | RQ6 | a 24 hour, 89.8 million case cross-ISA campaign |

## Unpack first

The bulky evidence is stored compressed, one `.tar.xz` per directory named
after it. The engine comparison report is 12.7 MB of JSON and the rules
TaintInduce synthesised are another 6.2 MB, which xz takes to 1.2 MB and
0.35 MB together. Summaries, tables and figures are left as they are, so what
a reader wants to look at is readable without unpacking anything.

```sh
./extract.sh
```

It unpacks every archive in place, skips one whose contents are already there,
and leaves the archives alone, so it can be run again safely. Every command
below assumes it has been run.

## The `--quick --no-baselines` run

One run of `./run-all.sh --quick --no-baselines` on a minimal Debian 12 machine.
It is in `run-all-debian12/20260918-121340`, with the summary, the build log, and the stdout, the stderr and the JSON output of every experiment.

The results agree with the paper.
The nftables and siphash avalanche figures are identical (nftables 35 instructions and 60.0% of flag bits, siphash 325 and 63.6%), and the five architectures each compared 20000 cases with no under-taint.

The base64 column needs a word, because this run predates the three-column split.
What it calls `base64-pinned` is a from-source coreutils 9.11 built `-O0 -static`, which is what is now called `base64-static`, and it gives 990 instructions and 3.5% of data bits.
Its `base64-system` is the machine's own binary and reproduces the paper to the digit: 370 instructions, 16.1% of data bits and 53.3% of flag bits.
The column the artifact now leads with, `base64-debian12`, is the hash-pinned Debian package (coreutils 9.1), and it gives the paper's 370 and 16.14% as well.
So the two packaged builds agree and the `-O0 -static` one does not: what moves the number is the build, not the version.

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

```sh
cd ../rq5-overhead
uv run python gen_ladder_table.py \
    ../reference-runs/run-all-debian12/20260918-121340/overhead_ladder.json \
    --tex ../reference-runs/run-all-debian12/20260918-121340/ladder_table.tex \
    --md  ../reference-runs/run-all-debian12/20260918-121340/ladder_table.md
```

That gives `apps_tables.tex` and `apps_tables.md` (the two RQ7 application tables, both matching the paper), `avalanche_numbers.tex` (the category table's macros), `fig_overhead.pdf` (Figure 6) and `ladder_table.tex` and `ladder_table.md` (the appendix's full overhead ladder).
The ladder is where the note about virtualisation above becomes concrete: every rung is slower than the paper's, `qiling-only` is 48 times native here against 52, and `microtaint-all` is 3674 against 4115.
The chain is in the same order and the marginal costs sit in the same places, which is what the rungs are for.
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

## The engine comparison, on the paper's corpus

`rq2-comparison-paper-corpus.tar.xz` holds `report_merged_v3.json`, the report the paper's Figures 2 to 5 and its benchmark macros were generated from.
It is 9858 cases at seed 12 (7500 single, 1000 sequence, 1310 sweep, plus the extra suites), scored against the exhaustive oracle, with all seven engines.
Its unsound counts are the paper's: 0 for microtaint, 32 TaintGrind, 43 angr, 206 Triton, 210 libdft64, 300 Maat, 464 PANDA.

The four figures beside it were regenerated from it here and are **pixel for pixel identical** to the ones in the paper.

```sh
cd ../rq2-comparison
uv run --with matplotlib --with numpy python plot_figures.py \
    --report ../reference-runs/rq2-comparison-paper-corpus/report_merged_v3.json
mv fig_*.pdf ../reference-runs/rq2-comparison-paper-corpus/

uv run python gen_eval_tables.py \
    ../reference-runs/rq2-comparison-paper-corpus/report_merged_v3.json \
    --allow-uncertified \
    --tex ../reference-runs/rq2-comparison-paper-corpus/eval_tables.tex \
    --md  ../reference-runs/rq2-comparison-paper-corpus/eval_tables.md
```

Figure 6, the overhead one, is not here: it reads the overhead ladder, which has nothing to do with this report, and its reference is in the `--quick` run above.

`--allow-uncertified` is needed, and it is worth saying why.
A case an engine ERRORED on is not evidence that it is sound on that case, so both generators check what fraction of the corpus each engine actually answered before they will state a soundness figure.
This report predates that accounting: it records how many cases each engine was compared on and not how many it refused.
The numbers are almost certainly right, and a re-run on the same corpus gives the same ones with a full answer rate, but this file cannot establish that on its own, so the caveat is written into the bottom of the generated tables rather than dropped.
`gen_paper_macros.py` applies the same rule and refuses outright, which is why there is no `benchmark_numbers.tex` here.

The seven tables generated from it are not the ones in the accepted paper.
The paper's `eval_tables.tex` comes from an older and much smaller run: 2003 cases against 9858, a 12 bit oracle budget against 15, and 694 ground-truth-evaluable cases against 3263.
That is why its microtaint row reads 84.9% bit-exact and 0.9602 Jaccard while this corpus gives 83.8% and 0.9617, and why its unsoundness summary names four `cmov` cases where this corpus finds 29.
The paper's macros and the paper's tables were generated from different runs.

## The TaintInduce comparison

`rq1-taintinduce` is one full-tier `run_rq1.sh` (no `--quick`), with the per-instruction JSON loose and the logs and the synthesised rules in its archive.
`SUMMARY.md` is the table: correct on bit-moving, logic and control flow, correct on 4 bit arithmetic, and unsound from 8 bits upward, which is the carry row of the paper's table.
The held-out under-taint column is the evidence: 74 cases on `add al,bl`, 220 on `add ax,bx`, 562 on `add eax,ebx`.
It took about an hour.

## The cross-ISA campaign

See the commands above.
