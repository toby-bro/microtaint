#!/usr/bin/env bash
# Every experiment, in order, into results/.
#
#   ./run-all.sh                 the paper's corpus.  About two days, dominated
#                                by RQ6 (5 ISAs x 1M cases).
#   ./run-all.sh --quick         reduced corpus, about two hours.  Checks that
#                                every experiment RUNS.  It does NOT reproduce
#                                the paper's numbers and must not be quoted.
#   ./run-all.sh --no-baselines  skip RQ1 and RQ2, the only steps that need the
#                                other engines installed.
#   ./run-all.sh --microtaint-only
#                                run RQ2/3/4 for this engine alone, without the
#                                other engines and without the ground-truth
#                                oracle.  Re-checks THIS engine cheaply; the
#                                comparison columns are absent by construction
#                                and the run is marked as not quotable.
#
# Each experiment writes its raw output under results/<rq>/ and appends a verdict
# to results/SUMMARY.md.  Nothing here deletes a previous run: results are
# timestamped, because a reviewer comparing two runs is a thing that happens and
# an artifact that overwrites its own evidence is unhelpful.
#
# Order is cheapest-first so a broken environment surfaces in minutes rather than
# hours, with the timing measurement (RQ5) before the two long CPU-saturating
# runs and RQ6 last.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$HERE/results/$STAMP"
mkdir -p "$OUT"

# The artifact measures the LogicCircuit engine.  taint_ir is a separate
# in-progress path and no paper number comes from it, so pin it off once here
# for every child process instead of trusting each script to pin it itself.
export MICROTAINT_TAINT_IR=0

QUICK=0
BASELINES=1
for a in "$@"; do
  case "$a" in
    --quick)        QUICK=1 ;;
    --no-baselines) BASELINES=0 ;;
    --microtaint-only) MTONLY=1 ;;
    -h|--help)      sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

if [ "$QUICK" = 1 ]; then
  RQ2_ARGS=(--number 500 --sequences 100)
  RQ6_N=20000
  MODE='quick (reduced corpus, NOT the paper numbers)'
else
  # benchmark.py defaults to 500 single + 100 sequence + no sweep = 600 cases.
  # The paper's corpus is 9,858.  Passing nothing here is what silently produced
  # a 600-case "full" run once, so the flags are explicit and not defaulted.
  RQ2_ARGS=(--number 7500 --sequences 1000 --sweep)
  RQ6_N=1000000
  MODE='full (the paper corpus)'
fi

# --microtaint-only: run RQ2/3/4 for THIS engine alone, with no ground-truth
# oracle.  RQ2 otherwise needs angr, Maat, Triton, PANDA, taintgrind and
# libdft64 all installed, and the 2^k oracle is the slowest part of the pass.
# This keeps the experiment runnable for a reviewer who has none of the other
# engines, and makes re-checking THIS engine after a change cheap.  The
# comparison columns are absent by construction, so the run is explicitly
# marked: it cannot be quoted as the cross-engine result.
if [ "${MTONLY:-0}" = 1 ]; then
  RQ2_ARGS+=(--workers microtaint --no-ground-truth)
  MODE="$MODE, microtaint only (no baselines, no oracle: NOT the comparison)"
fi

{
  echo "# microtaint NDSS 2027 artifact, run $STAMP"
  echo
  echo "mode: $MODE"
  echo "engine version: $(git -C "$REPO" describe --tags --dirty 2>/dev/null || echo unknown)"
  echo "MICROTAINT_TAINT_IR=$MICROTAINT_TAINT_IR"
  echo
} > "$OUT/SUMMARY.md"

log() { printf '\n=== %s ===\n' "$1" | tee -a "$OUT/log.txt"; }

record() {  # record <rq> <verdict> <detail>
  printf '%-24s %-4s %s\n' "$1" "$2" "$3" >> "$OUT/SUMMARY.md"
  printf '    %-4s %s %s\n' "$2" "$1" "$3" | tee -a "$OUT/log.txt"
}

_prepare() {  # _prepare <rq> <dir> -> 0 if runnable
  mkdir -p "$OUT/$1"
  if [ ! -d "$HERE/$2" ]; then
    record "$1" FAIL "missing directory $2"
    return 1
  fi
  return 0
}

run() {  # run <rq> <dir> <cmd...>   a normal experiment: 0 is the pass
  local rq=$1 dir=$2; shift 2
  log "$rq"
  _prepare "$rq" "$dir" || return 0
  ( cd "$HERE/$dir" && "$@" ) >"$OUT/$rq/stdout.txt" 2>"$OUT/$rq/stderr.txt"
  local rc=$?
  if [ $rc -eq 0 ]; then record "$rq" PASS "exit 0"
  else                   record "$rq" FAIL "exit $rc, see results/$STAMP/$rq/stderr.txt"; fi
  return 0
}

run_detector() {  # run_detector <rq> <dir> <kind> <cmd...>
  # microtaint's CLI exit code is 0 = no security findings, 1 = at least one,
  # 2 = CLI error.  Each of these four targets contains exactly one planted bug,
  # so 1 is the expected result and 0 means the detector MISSED it.
  #
  # But the exit code ALONE cannot be trusted: a Python traceback also exits 1.
  # Reproduced -- pointing the CLI at a non-existent rootfs made it raise before
  # any analysis, exit 1, print nothing, and file as "PASS finding detected".
  # Any import error, any uv resolution failure, and _resolve_rootfs on a
  # non-Linux host do the same, so a macOS reviewer could collect four PASSes
  # from a detector that never ran once.
  #
  # The harness already passes --json and threw the document away.  Read it:
  # PASS requires a finding OF THE RIGHT KIND to actually be in the summary.
  local rq=$1 dir=$2 kind=$3; shift 3
  log "$rq"
  _prepare "$rq" "$dir" || return 0
  # stdin from /dev/null.  The guests read stdin, and rq7-uaf is invoked without
  # --input, so with an inherited stdin it BLOCKS FOREVER waiting for a read
  # that never completes: measured on a compute node, 0 bytes of output and 0%
  # CPU after 21 minutes, while the same command with </dev/null finishes in a
  # second and reports its finding.  An artifact that hangs at step four of RQ7
  # is worse than one that fails there.
  ( cd "$HERE/$dir" && "$@" ) >"$OUT/$rq/stdout.txt" 2>"$OUT/$rq/stderr.txt" </dev/null
  local rc=$?
  local n
  n=$(python3 - "$OUT/$rq/stdout.txt" "$kind" <<'PYEOF'
import json, sys
# The CLI prints a status line before the JSON unless --quiet, so the document
# does not start at byte 0 and json.load() on the whole file fails.  Scan to the
# first '{', which is what check_side_channel.py already does for the same
# reason.  Getting this wrong reported all four detectors as FAIL on a run where
# every one of them correctly found its planted bug.
try:
    text = open(sys.argv[1]).read()
    i = text.find('{')
    doc = json.loads(text[i:]) if i >= 0 else None
    print(int(doc['summary'][sys.argv[2]]))
except Exception:
    print(-1)
PYEOF
)
  if [ "$n" -lt 0 ]; then
    record "$rq" FAIL "no parseable --json output (exit $rc): the detector did not run, see results/$STAMP/$rq/stderr.txt"
  elif [ "$n" -eq 0 ]; then
    record "$rq" FAIL "no $kind finding: the planted bug was MISSED"
  elif [ "$rc" -ne 1 ]; then
    record "$rq" FAIL "$n $kind finding(s) but exit $rc, expected 1"
  else
    record "$rq" PASS "$n $kind finding(s) detected"
  fi
  return 0
}

need() {  # need <rq> <file> <how-to-build> -> 0 if present
  if [ ! -e "$HERE/$1" ]; then
    record "$2" SKIP "missing $1, build it with: $3"
    return 1
  fi
  return 0
}

# ------------------------------------------------------------------ build
# The guest binaries are gitignored build products, so a fresh clone has none of
# them and every experiment that needs one would fail one by one.  Build what we
# can up front and report the rest as SKIP with the command to run.
log 'build guests'
( cd "$HERE/rq7-applications/memory-safety" && make ) \
    >"$OUT/build.log" 2>&1 && echo 'memory-safety guests: built' >>"$OUT/build.log"

# ------------------------------------------------------------------ avalanche
# Table 6.  Cheap, and it exercises the whole lifting path, so it fails fast.
run avalanche-calibrate avalanche uv run python calibrate.py \
    --json-out "$OUT/avalanche_calibrate.json"
run avalanche-base64 avalanche uv run python avalanche_freq.py --title base64 \
    --stdin 'The quick brown fox jumps over the lazy dog' \
    --json-out "$OUT/avalanche_base64.json" /usr/bin/base64
if need avalanche/nftables/nftables_harness avalanche-nftables 'make -C avalanche/nftables'; then
  run avalanche-nftables avalanche uv run python avalanche_freq.py --title nftables \
      --stdin-bytes "$(python3 -c 'print((bytes([80,2,0])+bytes(range(80))).hex())')" \
      --json-out "$OUT/avalanche_nftables.json" ./nftables/nftables_harness
fi
if need avalanche/siphash/siphash_bin avalanche-siphash 'make -C avalanche/siphash'; then
  run avalanche-siphash avalanche uv run python avalanche_freq.py --title siphash \
      --stdin 'sixteen byte msg' \
      --json-out "$OUT/avalanche_siphash.json" ./siphash/siphash_bin
fi

# ------------------------------------------------------------------ RQ7
D=rq7-applications/memory-safety
run_detector rq7-bof "$D" bof uv run microtaint --json --check-bof --input input.bin -- ./bof.elf
run_detector rq7-uaf "$D" uaf uv run microtaint --json --check-uaf                   -- ./uaf.elf
run_detector rq7-sc  "$D" side_channel uv run microtaint --json --check-sc  --input input.bin -- ./sc.elf
run_detector rq7-aiw "$D" aiw uv run microtaint --json --check-aiw --input input.bin -- ./aiw.elf
run rq7-crypto-check    rq7-applications/crypto/square_and_multiply uv run python check_side_channel.py \
    --json "$OUT/rq7_ct_check.json"
run rq7-crypto-localise rq7-applications/crypto/square_and_multiply uv run python localise_side_channel.py \
    --json "$OUT/rq7_ct_localise.json"
run rq7-dns             rq7-applications/dns                        uv run python dns_experiment.py \
    --json "$OUT/rq7_dns.json"

# ------------------------------------------------------------------ RQ1
if [ "$BASELINES" = 1 ]; then
  # Needs the TaintInduce checkout that setup_taintinduce.sh creates.  If it is
  # absent run_rq1.sh says so and exits; the rest of this script carries on.
  # espresso's DNF minimisation dominates the runtime and a TIMEOUT is one of
  # the results rather than a failure.
  if [ "$QUICK" = 1 ]; then run rq1-synthesis rq1-synthesis_vs_inference ./run_rq1.sh --quick
  else                      run rq1-synthesis rq1-synthesis_vs_inference ./run_rq1.sh; fi
fi

# ------------------------------------------------------------------ RQ5
# A timing measurement, so it runs before the two CPU-saturating steps below.
# Both scripts refuse to run with CPU boost enabled (ALLOW_CPU_BOOST=1 overrides
# and the boost state is recorded in the JSON either way).
# BUILD the two artefacts rather than skipping.  Neither bench.elf nor
# ladder_hooks.so is tracked (git ls-files confirms), so a fresh clone has
# neither, and the old SKIP remedy named `make -C rq5-overhead bench.elf` when
# there is no Makefile in that directory at all.  Worse, the single `need`
# wrapped BOTH rq5 steps while naming only rq5-ladder, so on a fresh clone
# SUMMARY.md contained no line whatsoever for rq5-bench: the headline overhead
# experiment silently did not exist in the verdict list.
if [ ! -e "$HERE/rq5-overhead/bench.elf" ]; then
  ( cd "$HERE/rq5-overhead" && uv run python overhead_bench.py --build-bench bench.c --runs 0 ) \
      >>"$OUT/build.log" 2>&1 && echo 'rq5 bench.elf: built' >>"$OUT/build.log"
fi
if [ ! -e "$HERE/rq5-overhead/ladder_hooks.so" ]; then
  ( cd "$HERE/rq5-overhead" && gcc -O2 -fPIC -shared -o ladder_hooks.so ladder_hooks.c ) \
      >>"$OUT/build.log" 2>&1 && echo 'rq5 ladder_hooks.so: built' >>"$OUT/build.log"
fi
if need rq5-overhead/bench.elf rq5-ladder 'cd rq5-overhead && uv run python overhead_bench.py --build-bench bench.c --runs 0'; then
  if need rq5-overhead/ladder_hooks.so rq5-ladder 'cd rq5-overhead && gcc -O2 -fPIC -shared -o ladder_hooks.so ladder_hooks.c'; then
    run rq5-ladder rq5-overhead uv run python overhead_ladder.py bench.elf --gen-input 64 \
        --runs 100 --runs-for codehook=30 --runs-for codehook-regs=15 \
        --json "$OUT/overhead_ladder.json"
  fi
fi
# rq5-bench gets its OWN need, so it always produces a verdict line.
if need rq5-overhead/bench.elf rq5-bench 'cd rq5-overhead && uv run python overhead_bench.py --build-bench bench.c --runs 0'; then
  run rq5-bench rq5-overhead uv run python overhead_bench.py --gen-input 64 --runs 5 \
      --instr-count --native-timeout 10 --qiling-timeout 300 --microtaint-timeout 900 \
      --json "$OUT/overhead_results.json" bench.elf
fi

# ------------------------------------------------------------------ RQ2/3/4
if [ "$BASELINES" = 1 ]; then
  # RQ3 (precision) and RQ4 (per-step cost) are scored in this same pass; there
  # is no separate directory for either.  BATCH_TIMEOUT defaults to 0, meaning
  # no ceiling: a non-zero value silently truncates slow engines mid-corpus and
  # still produces a well-formed report.  Set it only as a deliberate guard.
  run rq2-soundness rq2-comparison uv run python benchmark.py "${RQ2_ARGS[@]}"
  cp "$HERE"/rq2-comparison/report_*.json "$OUT/" 2>/dev/null
  echo 'rq3-precision            NOTE scored in the rq2 pass' >> "$OUT/SUMMARY.md"
  echo 'rq4-per-step-cost        NOTE scored in the rq2 pass' >> "$OUT/SUMMARY.md"
fi

# ------------------------------------------------------------------ RQ6
# Last: the longest step by a wide margin.  pass1 fuzzes, pass2 re-checks every
# candidate under-taint that pass1 flagged.
run rq6-pass1 rq6-generalisation uv run python campaign.py pass1 \
    --n "$RQ6_N" --arch all --seed 1 --out "$OUT/camp"
run rq6-pass2 rq6-generalisation uv run python campaign.py pass2 --in "$OUT/camp"

# ------------------------------------------------------------------ macros
# Regenerate the paper's number macros from the run that just happened, into the
# results directory.  Copy them over the paper's copies yourself once you have
# read the diff; this script does not write into the paper.
log 'paper macros'
if [ -f "$OUT/avalanche_base64.json" ] && [ -f "$OUT/avalanche_nftables.json" ] \
   && [ -f "$OUT/avalanche_siphash.json" ]; then
  ( cd "$HERE/avalanche" && uv run python gen_avalanche_macros.py \
      "$OUT/avalanche_base64.json" "$OUT/avalanche_nftables.json" \
      "$OUT/avalanche_siphash.json" --out "$OUT/avalanche_numbers.tex" ) \
    >>"$OUT/log.txt" 2>&1 \
    && record macros-avalanche PASS "$OUT/avalanche_numbers.tex" \
    || record macros-avalanche FAIL 'gen_avalanche_macros.py failed'
else
  record macros-avalanche SKIP 'needs all three avalanche reports'
fi

RQ2_REPORT="$(ls -t "$OUT"/report_*.json 2>/dev/null | head -1)"
if [ -n "$RQ2_REPORT" ]; then
  OV=()
  [ -f "$OUT/overhead_results.json" ] && OV=(--overhead "$OUT/overhead_results.json")
  ( cd "$HERE/rq2-comparison" && uv run python gen_paper_macros.py "$RQ2_REPORT" \
      "${OV[@]}" --out "$OUT/benchmark_numbers.tex" ) \
    >>"$OUT/log.txt" 2>&1 \
    && record macros-benchmark PASS "$OUT/benchmark_numbers.tex" \
    || record macros-benchmark FAIL 'gen_paper_macros.py failed'
else
  record macros-benchmark SKIP 'no rq2 report to generate from'
fi

printf '\n%s\n' "Results in $OUT" | tee -a "$OUT/log.txt"
printf 'Verdicts:\n' ; grep -E '^\S+\s+(PASS|FAIL|SKIP|NOTE)' "$OUT/SUMMARY.md" || true
if grep -qE '^\S+\s+FAIL' "$OUT/SUMMARY.md"; then
  printf '\nAt least one experiment FAILED; see results/%s/SUMMARY.md\n' "$STAMP"
  exit 1
fi
# A run in which everything SKIPped used to exit 0 and print "Results in ...",
# which is indistinguishable from a run in which everything passed.  SKIP is an
# absence of evidence, so it must not read as success.
n_pass=$(grep -cE '^\S+\s+PASS' "$OUT/SUMMARY.md" || true)
n_skip=$(grep -cE '^\S+\s+SKIP' "$OUT/SUMMARY.md" || true)
if [ "${n_pass:-0}" -eq 0 ]; then
  printf '\nNOTHING PASSED (%s skipped); see results/%s/SUMMARY.md\n' "${n_skip:-0}" "$STAMP"
  exit 1
fi
if [ "${n_skip:-0}" -gt 0 ]; then
  printf '\n%s experiment(s) SKIPPED and did not run; see results/%s/SUMMARY.md\n' \
      "${n_skip:-0}" "$STAMP"
fi
exit 0
