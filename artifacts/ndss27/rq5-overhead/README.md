# RQ5 — End-to-end overhead

**Claim (§6.5).** Running a real program under microtaint in a full-system
emulation harness (Qiling + Unicorn) costs a stated factor over native
execution, and a stated factor over the same harness without taint.

## Run

```sh
cd ../../../benchmark/overhead
uv run python overhead_bench.py --build-bench bench.c --gen-input 256 --runs 100 \
    --only native --only qiling-only --only microtaint-all \
    --native-timeout 5 --qiling-timeout 120 --microtaint-timeout 1800 \
    --json overhead_results.json
```

About 15 minutes.

## Read the invocation carefully

The positional `binary` argument takes `argparse.REMAINDER`, so
**`overhead_bench.py bench.elf --gen-input 256` passes those flags to the guest
program, not to the benchmark.** It then runs one repetition with no tainted
input and reports a time roughly eighty times too low. Use the form above, with
`--build-bench`, exactly as written.

## What to look at

`overhead_results.json` has three entries. The claim rests on two ratios:

- `microtaint-all.wall_s / native.wall_s` — the overhead over native
- `microtaint-all.extra.run_s / qiling-only.extra.run_s` — the cost of taint
  itself, with the emulator's own cost divided out

`run_s` is the `ql.run()` phase alone and is the more meaningful of the two:
`wall_s` on this benchmark is dominated by Python startup and module import,
which has nothing to do with taint.

## Tolerance

Machine-dependent in absolute terms. The ratio over `qiling-only` is the stable
figure and is what the paper's claim is stated in.
