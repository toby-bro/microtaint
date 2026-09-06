#!/usr/bin/env bash
# =============================================================================
# Normalized per-commit perf sweep.
#
# Measures every commit in a range with ONE pinned harness against ONE pinned
# instruction bank, so the only thing that differs between runs is the engine.
# Output: one timing JSON per commit in tests/perf.log.norm.d/, plus the plot.
#
#   ./scripts/sweep/sweep.sh                 # update: build+measure what's missing, replot
#   RANGE=v0.6.10..HEAD ./scripts/sweep/sweep.sh
#   PLOT_ONLY=1 ./scripts/sweep/sweep.sh     # just redraw from existing logs
#
# It is idempotent: a commit that already has a snapshot is not rebuilt, and one
# that already has a log is not re-measured. So adding commits and re-running
# only does the new work.
#
# Env knobs (all optional):
#   RANGE         commit range                  (default v0.6.10..HEAD)
#   WORK          scratch dir for clone/worktrees/snapshots
#                                               (default $TMPDIR/microtaint-perf-sweep)
#   OUT           where the logs land           (default <repo>/tests/perf.log.norm.d)
#   BUILD_WORKERS parallel builds               (default 4)
#   BENCH_CPUS    cpus for measurement workers  (default "2 10": one per L3 domain)
#   PLOT_ONLY=1   skip build+measure
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
PINNED="$HERE/pinned"
RANGE=${RANGE:-v0.6.10..HEAD}
WORK=${WORK:-${TMPDIR:-/tmp}/microtaint-perf-sweep}
OUT=${OUT:-$REPO/tests/perf.log.norm.d}
BUILD_WORKERS=${BUILD_WORKERS:-4}
read -r -a BENCH_CPUS <<< "${BENCH_CPUS:-2 10}"
PY=${PY:-$REPO/.venv/bin/python}

mkdir -p "$WORK" "$OUT"
CLONE=$WORK/repo
LIST=$WORK/commits.order

# --- the pinned corpus must never drift, or the series stops being comparable --
verify_pinned() {
  (cd "$PINNED" && sha256sum --quiet -c MANIFEST.sha256) || {
    echo "FATAL: scripts/sweep/pinned/ has changed since the series was recorded." >&2
    echo "Existing logs were measured against the old corpus and are no longer" >&2
    echo "comparable. Either restore it, or delete $OUT and re-measure everything." >&2
    exit 2
  }
}

setup() {
  [ -d "$CLONE/.git" ] || git clone --quiet --local "$REPO" "$CLONE"
  git -C "$CLONE" fetch --quiet origin 2>/dev/null || git -C "$CLONE" fetch --quiet "$REPO" 2>/dev/null || true
  git -C "$REPO" rev-list --reverse "$RANGE" > "$LIST"
  echo "range $RANGE -> $(wc -l < "$LIST") commits"
  for i in $(seq 0 $((BUILD_WORKERS - 1))); do
    [ -d "$WORK/wt$i" ] || git -C "$CLONE" worktree add --quiet --detach "$WORK/wt$i" HEAD
  done
}

# Lay the pinned corpus + harness over a worktree. This is what makes runs
# comparable: same instructions, same measurement loop, every commit.
apply_pinned() {
  local wt=$1
  mkdir -p "$wt/benchmark" "$wt/tests"
  rm -rf "$wt/benchmark/instruction_bank"
  cp -a "$PINNED/instruction_bank" "$wt/benchmark/instruction_bank"
  cp "$PINNED/test_perf_ratchet.py" "$wt/tests/test_perf_ratchet.py"
}

build_missing() {
  mkdir -p "$WORK/snap"
  rm -f "$WORK"/blist.*
  awk -v w="$WORK" -v n="$BUILD_WORKERS" '{print > (w "/blist." ((NR-1)%n))}' "$LIST"
  for i in $(seq 0 $((BUILD_WORKERS - 1))); do
    (
      wt=$WORK/wt$i
      while read -r sha; do
        short=$(git -C "$wt" rev-parse --short "$sha")
        [ -f "$WORK/snap/$sha.tgz" ] && continue
        git -C "$wt" checkout --quiet --force "$sha" || { echo "[b$i] $short CHECKOUT-FAIL"; continue; }
        git -C "$wt" clean -xfdq microtaint
        apply_pinned "$wt"
        if ! (cd "$wt" && uv sync --quiet --reinstall-package microtaint) >> "$WORK/build.log" 2>&1; then
          echo "[b$i] $short BUILD-FAIL"; continue
        fi
        tar czf "$WORK/snap/$sha.tgz.tmp" -C "$wt" microtaint && mv "$WORK/snap/$sha.tgz.tmp" "$WORK/snap/$sha.tgz"
        echo "[b$i] $short built"
      done < "$WORK/blist.$i"
    ) &
  done
  wait
  echo "snapshots: $(ls "$WORK/snap" 2>/dev/null | grep -c '\.tgz$') / $(wc -l < "$LIST")"
}

# Measurement: no builds run here, and each worker owns a physical core in its
# own L3 domain, so every commit is measured under the same conditions.
bench_missing() {
  local n=${#BENCH_CPUS[@]}
  rm -f "$WORK"/rlist.*
  awk -v w="$WORK" -v n="$n" '{print > (w "/rlist." ((NR-1)%n))}' "$LIST"
  for i in $(seq 0 $((n - 1))); do
    (
      wt=$WORK/wt$i cpu=${BENCH_CPUS[$i]}
      while read -r sha; do
        short=$(git -C "$wt" rev-parse --short "$sha")
        compgen -G "$OUT/*-$short-timing.json" > /dev/null && continue
        [ -f "$WORK/snap/$sha.tgz" ] || { echo "[r$i] $short NO-SNAPSHOT"; continue; }
        git -C "$wt" checkout --quiet --force "$sha" || { echo "[r$i] $short CHECKOUT-FAIL"; continue; }
        git -C "$wt" clean -xfdq microtaint
        tar xzf "$WORK/snap/$sha.tgz" -C "$wt"
        apply_pinned "$wt"
        rm -rf "$wt/tests/perf.log.d"
        if ! (cd "$wt" && taskset -c "$cpu" .venv/bin/python "$PINNED/drive.py" "$wt/tests") >> "$WORK/bench.log" 2>&1; then
          echo "[r$i] $short BENCH-FAIL"; continue
        fi
        cp "$wt"/tests/perf.log.d/*-timing.json "$OUT/" 2>/dev/null
        echo "[r$i] $short measured"
      done < "$WORK/rlist.$i"
    ) &
  done
  wait
  echo "logs: $(ls "$OUT"/*-timing.json 2>/dev/null | wc -l) / $(wc -l < "$LIST")"
}

plot() {
  cp "$LIST" "$OUT/commits.order"
  "$PY" "$HERE/../plot_perf_history.py" --log-dir "$OUT" --order-file "$OUT/commits.order" \
      --out "$OUT/perf-history-norm.png" --csv "$OUT/perf-history-norm.csv" --quiet "$@"
}

verify_pinned
if [ "${PLOT_ONLY:-0}" != 1 ]; then
  setup
  build_missing
  bench_missing
  "$PY" "$HERE/find_solo.py" "$OUT" "$WORK" || true
else
  git -C "$REPO" rev-list --reverse "$RANGE" > "$LIST"
fi
plot "$@"
