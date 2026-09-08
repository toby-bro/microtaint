# Installing

## 1. The engine, at the version the paper evaluates

```sh
git clone <repository> microtaint && cd microtaint
git checkout v0.6.15
uv sync --locked --all-extras
```

`v0.6.15` matters. It is the last version with a single taint implementation:
taint is answered by generating a `LogicCircuit` and evaluating it, which is
what §4 and §5 of the paper describe. Later versions add a compiled taint
program that is faster and has *different precision* (215 cases tighter and 28
looser of 2,072 on AMD64), so running the evaluation from `main` reproduces a
different system, not a faster one.

Check you have the right thing:

```sh
uv run python -c "import microtaint, microtaint.taint_ir" 2>&1 | tail -1
# ModuleNotFoundError: No module named 'microtaint.taint_ir'   <- correct
```

If that import *succeeds*, you are on a later version and the numbers will not
match the paper.

## 2. Confirm the engine works before running anything long

```sh
uv run pytest -o addopts="" -q -x tests/test_oracle_harness.py
```

About a minute. It checks the engine against per-bit ground truth from Unicorn
on a slice of the instruction bank — the same oracle the evaluation uses — so if
this passes, the machinery the experiments rest on is sound on your machine.

## 3. Run the experiments that need nothing else

```sh
./run-all.sh --no-baselines        # RQ4-RQ7, about 100 minutes
```

Results land in `results/`, one subdirectory per RQ, each with the raw JSON and
a short `SUMMARY.md` comparing what you got against what the paper reports.

## 4. The baselines, if you are reproducing RQ1-RQ3

Six engines with mutually incompatible dependencies, so each gets its own
virtualenv. The repository already carries the script that builds them:

```sh
cd artifacts/ndss27/rq2-soundness && ./setup_envs.sh
```

It creates `.venv_master`, `.venv_triton`, `.venv_angr`, `.venv_maat` and the
rest beside itself. Two of them cannot be fully automated and the script says so
when it reaches them:

- **Pin 3.20** (needed by Triton and libdft64) requires accepting Intel's
  licence, so it has to be downloaded by hand.
- **PANDA** runs from the `panda-re/panda` Docker image, so Docker must be
  running.

Every experiment **skips a baseline it cannot find and records that it skipped
it**, so a partial install gives partial results rather than a failure, and the
summary says which comparisons are missing.

Then:

```sh
./run-all.sh                          # everything, about five hours
```
