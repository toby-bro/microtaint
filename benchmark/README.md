# Performance harness

What the CI performance job runs, and the instruction corpus it draws from.
This is not part of the NDSS artifact: that lives in
[`artifacts/ndss27/`](../artifacts/ndss27/) and shares nothing with this
directory.

| path | what it is |
| --- | --- |
| `bench_suite.py` | the suite the `benchmark` CI job runs on every pull request. Prints deterministic *gates* (counters, coverage, operation counts), which fail the job on a regression, and *timings*, which are ratios against a baseline measured in the same run and are advisory |
| `instruction_bank/` | the live corpus: 1569 instructions across seven ISAs, plus the loader and `build.py`, which regenerates `instructions.jsonl` |
| `taint_density/` | freestanding `-nostdlib` guests the suite measures, and `hookcost.c`. The optimisation level of each is part of its definition, not a detail: changing it changes the instruction mix and silently breaks comparability with older numbers |

## Two copies of the bank, on purpose

`scripts/sweep/pinned/instruction_bank/` is a frozen snapshot of this one, held
still by `MANIFEST.sha256`. The performance sweep lays it over every historical
commit it measures, so that every revision is scored against the same corpus.
This directory is the live one, and the only place `build.py` can sit:
regenerating inside the pinned copy would overwrite the very bytes the manifest
exists to protect.

The tests and `bench_suite.py` read the pinned copy, because what they want is
the corpus that does not move. `build.py` writes this one.

## Run it

```sh
uv run python bench_suite.py --json out.json    # about 45 s
```
