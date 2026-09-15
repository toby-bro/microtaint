#!/usr/bin/env bash
# Table 5: the cross-ISA soundness and precision campaign.
#
#   ./run_table5.sh <hours>        all five ISAs in parallel, until the deadline
#   ./run_table5.sh 12             the paper's scale (about 55M cases)
#   ./run_table5.sh 0.2            a smoke run: proves it works, proves nothing else
#
# Pin the engine under test with $MT_ENGINE_ROOT; without it the campaign
# measures this repository.  The paper's numbers come from a frozen tag:
#
#   MT_ENGINE_ROOT=/path/to/pcode-taint-engine-at-v0.7.2 ./run_table5.sh 12
#
# Each ISA runs under a supervisor loop because the harness has historically
# segfaulted roughly every 10k cases; it resumes from its own checkpoint and
# appends witnesses, so a restart costs at most the chunk in flight.
#
# Results: campaign_<isa>.json per ISA, under_<isa>.jsonl for witnesses.  Turn
# them into the table with `python table5.py --tex ... --json ...`, which
# REFUSES to certify a run in which any ISA measured nothing.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
HOURS="${1:-12}"

# The artifact measures the LogicCircuit engine; taint_ir and block mode are
# separate paths and no paper number comes from either.
export MICROTAINT_TAINT_IR=0
export MICROTAINT_BLOCK=0

PY="${PY:-uv run --project "$(cd "$HERE/../../../.." && pwd)" python}"
DEADLINE=$(python3 -c "import time;print(time.time()+$HOURS*3600)")
mkdir -p "$HERE/logs"
echo "deadline $(date -d "@${DEADLINE%%.*}")  (${HOURS}h)"
echo "engine   ${MT_ENGINE_ROOT:-$(cd "$HERE/../../../.." && pwd)}"

for isa in x86_64 arm64 mips ppc riscv; do
    rm -f "$HERE/under_${isa}.jsonl" "$HERE/campaign_${isa}.json"
    (
        n=0
        while [ "$(python3 -c 'import time;print(int(time.time()))')" -lt "${DEADLINE%%.*}" ]; do
            n=$((n+1))
            echo "=== attempt $n at $(date) ===" >> "$HERE/logs/${isa}.log"
            ( cd "$HERE" && $PY run_campaign.py "$isa" "$DEADLINE" ) \
                >> "$HERE/logs/${isa}.log" 2>&1
            echo "=== exited rc=$? at $(date) ===" >> "$HERE/logs/${isa}.log"
        done
    ) &
    echo "  $isa supervisor pid $!"
done

echo "all launched; when it finishes:  python table5.py --tex table5.tex --json table5.json"
wait
