#!/usr/bin/env bash
# Every experiment, in order, into results/.
#
#   ./run-all.sh                 everything, about five hours
#   ./run-all.sh --no-baselines  RQ4-RQ7 only, about 100 minutes, needs only
#                                microtaint (RQ1 and RQ2 need the other engines)
#
# Each experiment writes its raw output under results/<rq>/ and appends a line
# to results/SUMMARY.md.  Nothing here deletes a previous run: results are
# timestamped, because a reviewer comparing two runs is a thing that happens and
# an artifact that overwrites its own evidence is unhelpful.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$HERE/results/$STAMP"
mkdir -p "$OUT"
BASELINES=1
[ "${1:-}" = "--no-baselines" ] && BASELINES=0

log() { printf '\n=== %s ===\n' "$1" | tee -a "$OUT/log.txt"; }
run() {  # run <rq> <dir-relative-to-this-file> <command...>
  local rq=$1 dir=$2; shift 2
  log "$rq"
  mkdir -p "$OUT/$rq"
  ( cd "$HERE/$dir" && "$@" ) >"$OUT/$rq/stdout.txt" 2>"$OUT/$rq/stderr.txt"
  local rc=$?
  echo "$rq: exit $rc" | tee -a "$OUT/SUMMARY.md"
  return 0            # a failing experiment must not stop the rest
}

echo "# microtaint NDSS 2027 artifact — run $STAMP" > "$OUT/SUMMARY.md"
echo "engine version: $(git -C "$REPO" describe --tags 2>/dev/null)" >> "$OUT/SUMMARY.md"

if [ "$BASELINES" = 1 ]; then
  # RQ1 needs the TaintInduce checkout that
  # rq1-synthesis_vs_inference/setup_taintinduce.sh creates.  If it is not there,
  # run_rq1.sh says so and exits; the rest of this script carries on.
  run rq1-synthesis  rq1-synthesis_vs_inference ./run_rq1.sh --quick
  run rq2-soundness rq2-soundness uv run python benchmark.py
  echo 'rq3-precision: scored in the same pass as rq2' >> "$OUT/SUMMARY.md"
fi

run rq4-per-step-cost rq4-per-step-cost uv run python bench_width_scaling.py
run rq5-overhead      rq5-overhead      uv run python overhead_bench.py \
    --build-bench bench.c --gen-input 256 --runs 100 \
    --only native --only qiling-only --only microtaint-all \
    --native-timeout 5 --qiling-timeout 120 --microtaint-timeout 1800 \
    --json overhead_results.json
run rq6-generalisation rq6-generalisation uv run python campaign.py \
    pass1 --n 20000 --arch all --seed 1 --out camp
run rq7-ct   rq7-applications/crypto/square_and_multiply uv run python check_side_channel.py
run rq7-dns  rq7-applications/dns                       uv run python dns_experiment.py

printf '\nResults in %s\n' "$OUT"
