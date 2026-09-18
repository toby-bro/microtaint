#!/bin/bash
# RQ1: run TaintInduce over the behaviour families of the paper's Table 3 and
# score every rule it produces against a held-out ground truth.
#
#   ./run_rq1.sh --quick   the families that finish fast, about 13 minutes
#   ./run_rq1.sh           everything except the 64-bit non-convergence probe,
#                          about an hour (measured: 59 min)
#   ./run_rq1.sh --full    the same plus the 64-bit probe, up to an hour longer
#
# Almost all of that is espresso: DNF minimisation on a 96-bit x86 state costs
# minutes per instruction, while the observation phase costs seconds.
#
# Each case writes results/<stamp>/<case>.{log,json} and one row of
# results/<stamp>/SUMMARY.md.  A case that fails or times out does not stop the
# others: a timeout *is* one of the results this experiment is looking for.
set -u

here=$(cd "$(dirname "$0")" && pwd)
. "$here/../find_uv.sh"
find_uv || exit 1
ti="$here/external/taintinduce"

if [ ! -d "$ti/.venv" ]; then
    echo '[!] external/taintinduce is not set up. Run ./setup_taintinduce.sh first.'
    exit 1
fi

tier='default'
case "${1:-}" in
    --quick) tier='quick' ;;
    --full)  tier='full' ;;
    '')      ;;
    *) echo "usage: $0 [--quick|--full]"; exit 2 ;;
esac

stamp=$(date +%Y%m%d-%H%M%S)
out="$here/results/$stamp"
mkdir -p "$out"

# How long a single instruction may take before we call it non-convergence.  The
# paper's x86-64 two-register row is a one-hour-per-encoding timeout, so that is
# what --full uses; the shorter tiers cap earlier because a reviewer should not
# have to wait an hour to learn that something did not finish.
case "$tier" in
    quick) budget=900 ;;
    full)  budget=3600 ;;
    *)     budget=1800 ;;
esac

# Held-out scoring: how many fresh seed states the rule is tested on.  Each state
# costs one bit-flip execution per input bit, so 32 states of a 96-bit x86 state
# is about 3k executions: negligible next to the inference itself.
holdout_states=32
# The held-out seed fixes the scoring set exactly: those states are drawn in the
# scorer's own process.  The inference seed is best-effort only -- observation
# generation runs across a process pool and each worker draws from an RNG state
# it inherited at fork, so the training set moves a little between runs and
# between machines.  Verdicts do not move; flow counts do.  See README, Traps.
infer_seed=1
holdout_seed=1000

summary="$out/SUMMARY.md"
{
    echo "# RQ1 - TaintInduce vs held-out ground truth ($stamp, tier=$tier)"
    echo
    echo "| Family | Instruction | W | Dataflows | Silent on forced chains | Random: always/partial/never | Never exercised | Held-out under-taint | Verdict |"
    echo "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
} > "$summary"

# run_case <tier-needed> <family> <width> <arch> <bytes> <label>
#
# tier-needed is the cheapest tier that includes this case: a case marked "full"
# runs only under --full, one marked "quick" runs under every tier.
run_case() {
    local need=$1 family=$2 width=$3 arch=$4 bytes=$5 label=$6
    case "$need:$tier" in
        full:quick|full:default) return ;;
        default:quick) return ;;
    esac

    local case_id="${family}_${arch}_${bytes}"
    echo
    echo "[*] $family / $label  (${arch} ${bytes}, budget ${budget}s)"

    # The family fixes the dataflow shape the rule is checked against:
    #   arithmetic -> triangle, because a carry out of bit i reaches every j >= i
    #   logic      -> diagonal, because bit i reaches bit i and nothing else
    # Everything else (bit-moving, control flow) has no single algebraic shape,
    # so it gets the held-out scoring only.
    local carry_args=()
    case "$family" in
        arithmetic) carry_args=(--carry-width "$width" --carry-shape triangle) ;;
        logic)      carry_args=(--carry-width "$width" --carry-shape diagonal) ;;
    esac

    local t0=$(date +%s)
    timeout "$budget" uv run --project "$ti" python "$here/score_rule.py" \
        --arch "$arch" --bytes "$bytes" --name "$label" \
        --infer-seed "$infer_seed" --holdout-seed "$holdout_seed" \
        --holdout-states "$holdout_states" ${carry_args[@]+"${carry_args[@]}"} \
        --obs-dir "$out/rules" --json "$out/$case_id.json" \
        > "$out/$case_id.log" 2>&1
    local rc=$?
    local wall=$(( $(date +%s) - t0 ))

    if [ "$rc" = 124 ]; then
        echo "    did not converge within ${budget}s"
        echo "| $family | \`$label\` | $width | - | - | - | - | - | did not converge (>${budget}s) |" >> "$summary"
        return
    fi
    if [ "$rc" != 0 ]; then
        echo "    FAILED (exit $rc), see $out/$case_id.log"
        echo "| $family | \`$label\` | $width | - | - | - | - | - | error (exit $rc) |" >> "$summary"
        return
    fi

    # Pull the numbers out of the JSON the scorer wrote rather than re-parsing
    # its prose, so the table and the raw data cannot drift apart.
    uv run --project "$ti" python - "$out/$case_id.json" "$family" "$label" "$width" "$wall" >> "$summary" <<'PYSUM'
import json, sys
path, family, label, width, wall = sys.argv[1:6]
with open(path) as fh:
    r = json.load(fh)
c = r['counts']
w = (r.get('forced_carry') or {}).get('totals')
if w:
    required = str(w['required'])
    silent = str(w['forced_silent'])
    mix = f"{w['rand_cells_always']}/{w['rand_cells_partial']}/{w['rand_cells_never']}"
    unseen = str(w['rand_cells_unseen'])
else:
    required = silent = mix = unseen = '-'
print(f"| {family} | `{label}` | {width} | {required} | {silent} | {mix} | {unseen} "
      f"| {c['under']} ({c['under_bits']} bits) | {r['verdict']} ({wall}s) |")
PYSUM
    grep -E '^(VERDICT|FORCED-CARRY|STRUCTURE|  exact|  data-register|  flag flows|  over-taint lands)' \
        "$out/$case_id.log" | sed 's/^/    /'
}

echo "[*] Results will land in $out"

# ---------------------------------------------------------------------------
# The behaviour families of Table 3.  Any ISA-agnostic rule inducer has to
# handle all of them; they are ordered here from the ones TaintInduce recovers
# to the ones it does not.
# ---------------------------------------------------------------------------

# Bit-moving.  A byte swap is pure routing: every output bit depends on exactly
# one input bit, unconditionally.  This is the easiest possible shape for a
# per-bit DNF and the inference gets it, in seconds.
run_case quick bit-moving 32 X86 0fce 'bswap esi'

# Control flow.  The displacement has to be non-zero.  TaintInduce decides
# whether an instruction touches the program counter by executing it a hundred
# times and checking whether the PC ever ends up anywhere other than
# start + len(insn); `jz .+2` branches to its own fall-through, so the PC never
# enters the state format and the case scores flag-to-flag identity flows and
# nothing else.  With `.+16` the state gains EIP and the ZF-reaches-the-PC
# dependency is actually compared.
run_case quick control-flow 32 X86 740e 'jz .+16'

# Logic.  and/or/xor are bit-local: output bit i depends on input bits i of the
# two operands and nothing else.  No carry, so nothing to miss.
run_case quick logic 32 X86 31d8 'xor eax, ebx'
run_case default logic 32 X86 21d8 'and eax, ebx'
run_case default logic 32 X86 09d8 'or eax, ebx'

# Arithmetic, by operand width.  This is the row the experiment exists for.
#
# At 4 bits the whole machine state is 12 bits, which is under the observation
# engine's 2^14 threshold, so it enumerates the state space instead of sampling
# it.  The inference has seen every input, and it is right.
run_case quick arithmetic 4 JN 0 'ADD R1, R2 (JN)'
#
# From 8 bits up, the state no longer fits and the seeds are samples.  Carry
# propagation from bit i to bit j needs the intermediate bits of *both* operands
# to be configured so a carry passes through, and each (i,j) pair is a separate
# dependency that only a seed realising that exact configuration can reveal.  The
# number of such pairs grows with the square of the width while the seed budget
# does not, so this is where the rules start missing flows.
run_case quick   arithmetic 8  X86   00d8   'add al, bl'
run_case default arithmetic 16 X86   6601d8 'add ax, bx'
run_case default arithmetic 32 X86   01d8   'add eax, ebx'
run_case default arithmetic 32 X86   29d8   'sub eax, ebx'
#
# x86-64, two register operands.  The paper reports this one as not converging
# within an hour per encoding; --full is the tier that actually waits that long.
run_case full arithmetic 64 AMD64 4801d8 'add rax, rbx'

echo
echo "[+] Done. Summary:"
echo
cat "$summary"
