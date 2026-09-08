# microtaint — NDSS 2027 artifact

Everything the paper measures lives here, one directory per research question.
The experiments *are* these files — they were moved in, not wrapped — so what
you run is what produced the numbers.

---

## Quick start

```sh
git checkout v0.6.15                  # the engine the paper evaluates
uv sync --locked --all-extras
cd artifacts/ndss27
./run-all.sh --no-baselines           # ~100 min, needs nothing but microtaint
```

That covers RQ4–RQ7 and is enough to see the system work end to end. Results
land in `results/<timestamp>/`, with a `SUMMARY.md`.

For the full evaluation, including the six baseline engines it is compared
against, see `INSTALL.md` step 4 and then:

```sh
./run-all.sh                          # ~5 h
```

**`v0.6.15` matters.** It is the last version with a single taint
implementation, which is what the paper describes: taint is answered by
generating a `LogicCircuit` and evaluating it. Later versions add a compiled
taint program that is faster and has *different precision* (215 cases tighter,
28 looser of 2,072 on AMD64), so running from `main` reproduces a different
system rather than a faster one. This check should FAIL:

```sh
uv run python -c "import microtaint.taint_ir"    # ModuleNotFoundError = correct
```

---

## Claims, and where each is reproduced

| # | Claim (paper §) | Directory | Needs | Time |
|---|---|---|---|---|
| RQ1 | Synthesis is sound where observation-based inference does not converge (§6.1) | `rq1-synthesis-vs-inference/` | TaintInduce | — |
| RQ2 | Sound on every x86-64 test case, where the six baselines are not (§6.2) | `rq2-soundness/` | all baselines | ~3 h |
| RQ3 | Highest bit-level precision among sound engines (§6.3) | `rq3-precision/` | all baselines | with RQ2 |
| RQ4 | Per-step cost is competitive and independent of operand width (§6.4) | `rq4-per-step-cost/` | — | ~20 min |
| RQ5 | End-to-end overhead under full-system emulation (§6.5) | `rq5-overhead/` | — | ~15 min |
| RQ6 | Generalises unchanged to ARM64, MIPS64, PPC32, RV64GC (§6.6) | `rq6-generalisation/` | — | ~32 min |
| RQ7 | Enables analyses coarser engines cannot do (§6.7) | `rq7-applications/` | — | ~20 min |

Each directory's `README.md` gives the exact command, what it writes, which
number in the paper it corresponds to, and the tolerance a rerun should meet.

**RQ1 is not yet wired up.** TaintInduce needs its own environment and a
comparison harness that is not in this repository. It is listed rather than
dropped so the claim-to-experiment mapping stays complete.

## Running one experiment at a time

Every directory is self-contained. From `artifacts/ndss27`:

```sh
cd rq5-overhead    && uv run python overhead_bench.py --build-bench bench.c ...
cd rq6-generalisation && uv run python campaign.py pass1 --n 20000 --arch all --seed 1 --out camp
cd rq4-per-step-cost  && uv run python bench_width_scaling.py && uv run python plot_width_scaling.py
cd rq7-applications/crypto/square_and_multiply && uv run python check_side_channel.py
cd rq7-applications/dns && uv run python dns_experiment.py
```

The per-directory README has the full invocation with its flags. Two of them
have flags that matter and are easy to get wrong — `overhead_bench.py` in
particular silently reports a figure about eighty times too fast if you pass its
arguments in the obvious order — so copy the command from the README rather than
improvising it.

## What is exact and what is not

**Exact.** Soundness and precision. The corpus is fixed, the taint masks are
fixed, and ground truth is an exhaustive enumeration over every input
combination for the 3,263 cases with at most 15 tainted bits. A rerun that
disagrees is a real disagreement, and worth telling us about.

**Not exact.** Timings. The paper's are from an AMD Ryzen 7 5700U at ~3 GHz;
yours will differ. Every timing claim in the per-directory READMEs is stated as
a *ratio* between engines or configurations, because those hold across machines
where absolute numbers do not.

## Layout

    README.md INSTALL.md REQUIREMENTS.md LICENSE   this, and how to get set up
    run-all.sh                                     every experiment, in order
    rq2-soundness/      benchmark.py, one worker per engine, setup_envs.sh,
                        proofs/ (the Z3 category-soundness proofs)
    rq3-precision/      scored in RQ2's pass; avalanche/ for the cost of the
                        avalanche category (§6.8)
    rq4-per-step-cost/  the width-scaling experiment and its figure
    rq5-overhead/       the full-system overhead harness and its guest
    rq6-generalisation/ the multi-ISA campaign and its oracle
    rq7-applications/   crypto/ and dns/ for microtaint; other-engines/ for the
                        same two analyses written against angr, Maat, Triton
                        and PANDA
    results/            created on first run; nothing here is overwritten

`benchmark/` at the repository root is a scratchpad and is **not** part of the
artifact. It keeps only the shared instruction corpus that the test suite reads,
and internal performance tooling.
