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
# Every experiment below runs through `uv`, so find it before doing anything
# else: installed-but-not-on-PATH is the normal state right after uv's own
# installer, and failing here with one message beats failing in each experiment.
. "$HERE/find_uv.sh"
find_uv || exit 1
. "$HERE/check_deps.sh"

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$HERE/results/$STAMP"
mkdir -p "$OUT"

# The artifact measures the LogicCircuit engine.  taint_ir is a separate
# in-progress path and no paper number comes from it, so pin it off once here
# for every child process instead of trusting each script to pin it itself.
export MICROTAINT_TAINT_IR=0


# The seed the paper's engine comparison was drawn with.  benchmark.py defaults
# to none, which draws a fresh corpus every time: the run is then not
# reproducible even against itself, and two runs cannot be compared case by
# case.  Every other experiment already pins its randomness (RQ1 infer_seed=1,
# RQ5 gen_input 0xC0FFEE, RQ6 --seed 1, the avalanche workloads take fixed
# input), so this was the only one left.
RQ2_SEED="${RQ2_SEED:-12}"

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

check_deps "$BASELINES" || exit 1


if [ "$QUICK" = 1 ]; then
  RQ2_ARGS=(--number 500 --sequences 100 --seed "$RQ2_SEED")
  RQ6_N=20000
  MODE='quick (reduced corpus, NOT the paper numbers)'
else
  # benchmark.py defaults to 500 single + 100 sequence + no sweep = 600 cases.
  # The paper's corpus is 9,858.  Passing nothing here is what silently produced
  # a 600-case "full" run once, so the flags are explicit and not defaulted.
  RQ2_ARGS=(--number 7500 --sequences 1000 --sweep --seed "$RQ2_SEED")
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
: >"$OUT/build.log"

# Build EVERY guest binary the default (non-baseline) path needs, not just the
# memory-safety ones.  The avalanche harnesses used to be left to the `need`
# guards below, so a fresh clone reported two SKIPs and told the reader to go
# run `make` by hand -- for binaries this script is perfectly able to build.
# A SKIP should mean "this needs software we cannot install for you", not "we
# did not bother to run make".
build_guest() {  # build_guest <label> <dir> <cmd...>
  local label=$1 dir=$2; shift 2
  if ( cd "$HERE/$dir" && "$@" ) >>"$OUT/build.log" 2>&1; then
    echo "$label: built" >>"$OUT/build.log"
  else
    echo "$label: BUILD FAILED, see the output above" >>"$OUT/build.log"
  fi
}

build_guest 'memory-safety guests'   rq7-applications/memory-safety make
build_guest 'avalanche nftables'     avalanche/nftables            make
build_guest 'avalanche siphash'      avalanche/siphash             make
# Pinned upstream coreutils, so the base64 row measures the same program
# everywhere.  Downloads a tarball on first run; if there is no network the
# step below reports SKIP with the command, and base64-system still runs.
build_guest 'avalanche base64'       avalanche/base64              make

# Neither rq5 artefact is tracked (git ls-files confirms), so a fresh clone has
# neither.  Built here with the others rather than half-way down the script.
[ -e "$HERE/rq5-overhead/bench.elf" ] || \
  build_guest 'rq5 bench.elf' rq5-overhead uv run python overhead_bench.py --build-bench bench.c --runs 0
[ -e "$HERE/rq5-overhead/ladder_hooks.so" ] || \
  build_guest 'rq5 ladder_hooks.so' rq5-overhead gcc -O2 -fPIC -shared -o ladder_hooks.so ladder_hooks.c

# ------------------------------------------------------------------ avalanche
# Table 6.  Cheap, and it exercises the whole lifting path, so it fails fast.
run avalanche-calibrate avalanche uv run python calibrate.py \
    --json-out "$OUT/avalanche_calibrate.json"
# THREE base64 columns, because the binary decides the answer and the spread
# should be visible rather than accidental.  See avalanche/base64/Makefile for
# the measured spread and why each one is here.
#
#   debian12  Debian 12's own coreutils, extracted from a hash-pinned .deb.  A
#             prebuilt binary is bit-identical everywhere (no compiler in the
#             loop), so this is the REFERENCE column.
#   static    pinned upstream release built here, -O0 -static: self-contained,
#             but reproducible only given the same compiler.
#   system    whatever /usr/bin/base64 this machine has, deliberately unpinned,
#             to show what this machine reports.
if need avalanche/base64/base64_debian12 avalanche-base64-debian12 'make -C avalanche/base64'; then
  run avalanche-base64-debian12 avalanche uv run python avalanche_freq.py --title base64-debian12 \
      --stdin 'The quick brown fox jumps over the lazy dog' \
      --json-out "$OUT/avalanche_base64_debian12.json" ./base64/base64_debian12
fi
if need avalanche/base64/base64_static avalanche-base64-static 'make -C avalanche/base64'; then
  run avalanche-base64-static avalanche uv run python avalanche_freq.py --title base64-static \
      --stdin 'The quick brown fox jumps over the lazy dog' \
      --json-out "$OUT/avalanche_base64_static.json" ./base64/base64_static
fi
run avalanche-base64-system avalanche uv run python avalanche_freq.py --title base64-system \
    --stdin 'The quick brown fox jumps over the lazy dog' \
    --json-out "$OUT/avalanche_base64_system.json" /usr/bin/base64
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
else
  # An experiment that leaves no line at all is worse than one that says SKIP:
  # the summary then reads all-PASS for a run that never attempted it.
  record rq1-synthesis SKIP '--no-baselines'
fi

# ---------------------------------------------------- RQ7, the baselines
# The same two analyses on the six compared engines.  Each writes
# other-engines/results/<tool>.json, which is what the application tables
# read.  They need the environments setup_envs.sh builds, so they only run
# with the baselines, and PANDA boots three full-system guests, which is most
# of the time this block takes.
if [ "$BASELINES" = 1 ]; then
  V="$HERE/rq2-comparison"
  OE=rq7-applications/other-engines
  run rq7-other-angr   "$OE" "$V/.venv_angr/bin/python"   detect_angr_apps.py
  run rq7-other-angr-loc "$OE" "$V/.venv_angr/bin/python" localise_angr_ct.py
  run rq7-other-maat   "$OE" "$V/.venv_maat/bin/python"   detect_maat.py
  run rq7-other-maat-loc "$OE" "$V/.venv_maat/bin/python" localise_maat_ct.py
  run rq7-other-triton "$OE" "$V/.venv_triton/bin/python" detect_triton.py
  run rq7-other-libdft "$OE" python3 detect_libdft64.py
  run rq7-other-tgrind "$OE" python3 detect_taintgrind.py
  # PANDA is 40 of this block's 43 minutes, because it is the only full-system
  # engine and boots a guest per workload.  The five above exercise every other
  # environment in about three minutes, so a broken one is reported before that
  # cost is paid rather than after it.
  run rq7-other-panda  "$OE" python3 detect_panda.py
  cp "$HERE/$OE"/results/*.json "$OUT/" 2>/dev/null
else
  record rq7-other-engines SKIP '--no-baselines'
fi

# ------------------------------------------------------------------ RQ5
# A timing measurement, so it runs before the two CPU-saturating steps below.
# Both scripts refuse to run with CPU boost enabled (ALLOW_CPU_BOOST=1 overrides
# and the boost state is recorded in the JSON either way).
# Both rq5 artefacts are built up front with the other guests.  Each rq5 step
# still gets its OWN `need` below: the single `need` used to wrap BOTH steps
# while naming only rq5-ladder, so on a fresh clone SUMMARY.md contained no line
# whatsoever for rq5-bench and the headline overhead experiment silently did not
# exist in the verdict list.
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
  # benchmark.py writes report_<unixtime>.json into its own directory, which
  # accumulates one per run ever made there.  Copying them all put 28 reports
  # and 140 MB of somebody else's runs into this one, and left RQ2_REPORT below
  # choosing between them on an `ls -t` tie, since cp gives them all the same
  # mtime.  Take the note of what was there first and copy only what appeared.
  _before=$(ls "$HERE"/rq2-comparison/report_*.json 2>/dev/null | sort)
  run rq2-soundness rq2-comparison uv run python benchmark.py "${RQ2_ARGS[@]}"
  comm -13 <(printf '%s\n' "$_before") \
           <(ls "$HERE"/rq2-comparison/report_*.json 2>/dev/null | sort) \
    | while read -r _r; do [ -n "$_r" ] && cp "$_r" "$OUT/"; done
  echo 'rq3-precision            NOTE scored in the rq2 pass' >> "$OUT/SUMMARY.md"
  echo 'rq4-per-step-cost        NOTE scored in the rq2 pass' >> "$OUT/SUMMARY.md"
else
  record rq2-soundness     SKIP '--no-baselines'
  record rq3-precision     SKIP '--no-baselines (scored in the rq2 pass)'
  record rq4-per-step-cost SKIP '--no-baselines (scored in the rq2 pass)'
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
# The paper's base64 macros come from the REFERENCE column (Debian 12's own
# coreutils, hash-pinned, identical on every machine).  The static and system
# builds are measured alongside it to show the spread, but feeding all three
# here would also skew the cross-workload aggregates, which average over
# workloads and would then count base64 three times.
if [ -f "$OUT/avalanche_base64_debian12.json" ] && [ -f "$OUT/avalanche_nftables.json" ] \
   && [ -f "$OUT/avalanche_siphash.json" ]; then
  ( cd "$HERE/avalanche" && uv run python gen_avalanche_macros.py \
      "$OUT/avalanche_base64_debian12.json" "$OUT/avalanche_nftables.json" \
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
  # The reduced corpus runs the overhead ladder with too few repetitions to
  # separate the detectors from the noise, and gen_paper_macros.py refuses to
  # state a percentage from noise.  On --quick, emit every macro that IS
  # resolvable and name the one that is not, rather than producing nothing.
  UNRES=()
  [ "$QUICK" = 1 ] && UNRES=(--allow-unresolved)
  [ -f "$OUT/overhead_results.json" ] && OV=(--overhead "$OUT/overhead_results.json")
  ( cd "$HERE/rq2-comparison" && uv run python gen_paper_macros.py "$RQ2_REPORT" \
      "${OV[@]}" "${UNRES[@]}" --out "$OUT/benchmark_numbers.tex" ) \
    >>"$OUT/log.txt" 2>&1 \
    && record macros-benchmark PASS "$OUT/benchmark_numbers.tex" \
    || record macros-benchmark FAIL 'gen_paper_macros.py failed'
else
  record macros-benchmark SKIP 'no rq2 report to generate from'
fi

# ------------------------------------------------------------------ figures
# The paper's five evaluation figures, from the run that just happened.  These
# need matplotlib and numpy, which the artifact does not otherwise depend on,
# so they come from `uv run --with` rather than from the project environment.
# The two inputs are independent: figures 4 to 7 need the engine comparison,
# figure 8 needs the overhead ladder.  A --no-baselines run has only the second,
# and used to get no figure at all rather than the one it had measured.
FIGARGS=()
[ -n "$RQ2_REPORT" ] && FIGARGS+=(--report "$RQ2_REPORT")
[ -f "$OUT/overhead_results.json" ] && FIGARGS+=(--overhead "$OUT/overhead_results.json")
if [ ${#FIGARGS[@]} -gt 0 ]; then
  ( cd "$HERE/rq2-comparison" && uv run --with matplotlib --with numpy \
      python plot_figures.py "${FIGARGS[@]}" \
      && mv -f fig_*.pdf "$OUT/" ) \
    >>"$OUT/log.txt" 2>&1 \
    && record figures PASS "$OUT/fig_*.pdf" \
    || record figures FAIL 'plot_figures.py failed'
else
  record figures SKIP 'neither an rq2 report nor an overhead ladder to plot from'
fi

# ------------------------------------------------------------------- tables
# The paper's tables, from this run rather than transcribed by hand.  Each
# generator refuses to emit a table whose inputs the run did not produce, so a
# --quick or --no-baselines run gets the subset it actually measured.
if [ -n "$RQ2_REPORT" ]; then
  ( cd "$HERE/rq2-comparison" && uv run python gen_eval_tables.py "$RQ2_REPORT" \
      --tex "$OUT/eval_tables.tex" --md "$OUT/eval_tables.md" ) \
    >>"$OUT/log.txt" 2>&1 \
    && record tables-eval PASS "$OUT/eval_tables.md" \
    || record tables-eval FAIL 'gen_eval_tables.py failed'
else
  record tables-eval SKIP 'no rq2 report to generate from'
fi

# The appendix's overhead ladder, from the rungs RQ5 measured.
if [ -f "$OUT/overhead_ladder.json" ]; then
  ( cd "$HERE/rq5-overhead" && uv run python gen_ladder_table.py \
      "$OUT/overhead_ladder.json" \
      --tex "$OUT/ladder_table.tex" --md "$OUT/ladder_table.md" ) \
    >>"$OUT/log.txt" 2>&1 \
    && record tables-ladder PASS "$OUT/ladder_table.md" \
    || record tables-ladder FAIL 'gen_ladder_table.py failed'
else
  record tables-ladder SKIP 'rq5 produced no overhead ladder'
fi

# The two application tables need this run's RQ7 output and the checked-in
# verdicts of the other engines, which the detect_* scripts produce.
if [ -f "$OUT/rq7_ct_localise.json" ] && [ -f "$OUT/rq7_dns.json" ]; then
  ( cd "$HERE/rq7-applications" && uv run python gen_apps_tables.py \
      --results-dir "$OUT" --tex "$OUT/apps_tables.tex" --md "$OUT/apps_tables.md" ) \
    >>"$OUT/log.txt" 2>&1 \
    && record tables-apps PASS "$OUT/apps_tables.md" \
    || record tables-apps FAIL 'gen_apps_tables.py failed'
else
  record tables-apps SKIP 'rq7 produced no localisation or dns result'
fi

# The cross-ISA table comes from the separate, open-ended table5 campaign
# (rq6-generalisation/table5/run_table5.sh), not from the RQ6 pass above: the
# two use different harnesses and different file formats.  Generate it only if
# that campaign has been run, and never pretend the RQ6 pass produced it.
if ls "$HERE"/rq6-generalisation/table5/campaign_*.json >/dev/null 2>&1; then
  ( cd "$HERE/rq6-generalisation/table5" && uv run python table5.py \
      --tex "$OUT/table5.tex" --json "$OUT/table5.json" ) \
    >>"$OUT/log.txt" 2>&1 \
    && record tables-multiisa PASS "$OUT/table5.tex" \
    || record tables-multiisa FAIL 'table5.py refused to certify the campaign'
else
  record tables-multiisa SKIP 'table5 campaign not run (see rq6-generalisation/table5/)'
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
