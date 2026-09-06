# scripts/sweep — normalized per-commit perf sweep

Measures every commit in a range with **one pinned harness against one pinned
instruction bank**, so the only thing differing between runs is the engine.
Produces `tests/perf.log.norm.d/` (see its README for method and results).

## Update after adding commits

    ./scripts/sweep/sweep.sh

That is the whole workflow. It is **idempotent**: a commit that already has a
snapshot is not rebuilt, one that already has a log is not re-measured. So after
new commits land, this builds and measures only those, then replots. With
nothing new it is a ~7s replot.

    RANGE=v0.6.10..HEAD ./scripts/sweep/sweep.sh   # default range
    PLOT_ONLY=1 ./scripts/sweep/sweep.sh           # just redraw
    WORK=/scratch/sweep ./scripts/sweep/sweep.sh   # where clone+snapshots live

| env | default | meaning |
|---|---|---|
| `RANGE` | `v0.6.10..HEAD` | commits to measure |
| `WORK` | `$TMPDIR/microtaint-perf-sweep` | clone, worktrees, snapshots (never inside the repo) |
| `OUT` | `tests/perf.log.norm.d` | where logs and the plot land |
| `BUILD_WORKERS` | 4 | parallel builds (measurement is never parallel with a build) |
| `BENCH_CPUS` | `2 10` | cpus for measurement workers, one per L3 domain |

## Files

    sweep.sh       the entrypoint: setup -> build missing -> measure missing -> plot
    find_solo.py   flags runs not measured under steady parallel load
    pinned/        THE CONTRACT: corpus + harness shared by every run
      instruction_bank/  1569 instructions (data + loader)
      test_perf_ratchet.py  the harness, incl. its reps=600 x 3 timing loop
      drive.py       runs the bench against the checked-out engine
      MANIFEST.sha256

## `pinned/` must not drift

`sweep.sh` verifies `pinned/` against its manifest and refuses to run if it
changed: existing logs were measured against the old corpus, so mixing them
would be comparing different benchmarks. To adopt a newer bank, update
`pinned/`, refresh the manifest, delete `tests/perf.log.norm.d/*.json`, and
re-measure the whole range.

`pinned/` carries two deliberate deviations from the repo copies, applied once
and used by every run:

1. the harness skips an instruction whose rule an engine revision cannot build,
   recording it in the log's `skipped` list, rather than aborting the run;
2. the bank loader drops an ISA group whose register geometry the engine cannot
   construct (the SIMD profiles need `microtaint.debug.reg_aliases`, which only
   exists from commit 18/82 onward).

`pinned/instruction_bank/build.py` is intentionally absent: it is the bank
*generator* and is not needed to load the bank, so pinning it would only add a
second copy to keep in sync.
