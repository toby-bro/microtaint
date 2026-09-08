#!/usr/bin/env python3
"""
diagnose_poison_search.py
=========================
Three-stage diagnostic that builds on diagnose_bench_replay.py:

STAGE 1 — Backend reproduction matrix
    Replay all preceding tests through the worker pipeline with three
    different CellSimulator configurations:
      a) (use_unicorn=True,  use_c=False) — pure Unicorn
      b) (use_unicorn=False, use_c=False) — Cython P-code evaluator
      c) (use_unicorn=False, use_c=True ) — C P-code evaluator
    For each, report whether id=8009 produces the broken value.

STAGE 2 — Lower-bound bisection
    Binary-search the smallest START index N such that running tests
    [N..target_idx] STILL reproduces the broken value.  This identifies
    the EARLIEST test index that contributes to the poisoning.  Use the
    backend that reproduced in STAGE 1.

STAGE 3 — Necessary-set extraction
    Starting from the lower bound found in STAGE 2, walk forward index
    by index.  For each index i in [lower_bound, target_idx), check
    whether running [{lower_bound, i, target_idx}] (and progressively
    accumulating mandatory predecessors) reproduces the broken value.
    The result: a minimal set of test indices that, played through ONE
    shared simulator in order, reproduce the bug.

Usage:
    .venv_microtaint/bin/python diagnose_poison_search.py REPORT.json [target_id]

target_id defaults to 8009.

Output is also saved to ``poison_search.log`` for sharing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

cwd = Path.cwd()
worker_path = next(
    (p for p in [cwd / 'worker_microtaint.py', cwd.parent / 'worker_microtaint.py']
     if p.is_file()),
    None,
)
if worker_path is None:
    print(f'ERROR: worker_microtaint.py not found in {cwd} or its parent.')
    sys.exit(1)
sys.path.insert(0, str(worker_path.parent))

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule, _cached_generate_static_rule
from microtaint.types import Architecture, Register

# Silence the simulator's per-failure error logger — bisection runs many
# tests, including some that legitimately fail (e.g. instructions Unicorn
# doesn't support); the resulting log spam buries the diagnostic output.
import logging
logging.getLogger('microtaint').setLevel(logging.CRITICAL)
logging.getLogger('microtaint.simulator').setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)

# --- Worker mirror constants -----------------------------------------------

_REGS = (
    [Register('RAX', 64), Register('RBX', 64), Register('RCX', 64), Register('RDX', 64)]
    + [Register('RSI', 64), Register('RDI', 64), Register('RSP', 64), Register('RBP', 64)]
    + [Register(f'R{n}', 64) for n in range(8, 16)]
    + [Register(f'XMM{n}_LO', 64) for n in range(8)]
    + [Register(f'XMM{n}_HI', 64) for n in range(8)]
)
_DEFAULT_RSP = 0x80000000

EXPECTED_SOUND  = 0x0202020202021302
EXPECTED_BROKEN = 0x7706aa7ab587952e


# --- Logging ----------------------------------------------------------------

LOG_PATH = Path('poison_search.log')
_log_fh = open(LOG_PATH, 'w')


def log(msg: str = '') -> None:
    print(msg)
    _log_fh.write(msg + '\n')
    _log_fh.flush()


# --- Args -------------------------------------------------------------------

if len(sys.argv) < 2:
    log('Usage: diagnose_poison_search.py REPORT.json [target_id]')
    sys.exit(1)
report_path = sys.argv[1]
target_id = int(sys.argv[2]) if len(sys.argv) > 2 else 8009

with open(report_path) as f:
    report = json.load(f)
results = report['results']
target_idx = next((i for i, e in enumerate(results) if e['id'] == target_id), None)
if target_idx is None:
    log(f'id={target_id} not found in report.')
    sys.exit(1)

target_inst = results[target_idx]['instruction']
target_state = target_inst['state']
target_taint = target_inst['taint']
target_bytes_hex = target_inst['bytes']

log(f'Report:    {report_path}')
log(f'Target:    id={target_id} at index {target_idx}')
log(f'Target bs: {target_bytes_hex}')
log(f'Expected sound:  0x{EXPECTED_SOUND:016x}')
log(f'Expected broken: 0x{EXPECTED_BROKEN:016x}')


# --- Core replay primitive --------------------------------------------------

def replay(indices: list[int], sim_kwargs: dict) -> int:
    """Run the given test indices in order through ONE shared
    CellSimulator built with the given kwargs.  Returns the RAX taint
    output produced when the LAST index in the list is evaluated.

    The last index MUST be target_idx — we always end on the target.
    """
    if not indices or indices[-1] != target_idx:
        raise ValueError('indices must end with target_idx')

    # Each replay needs a fresh simulator AND fresh circuit cache, so
    # only the chosen `indices` are responsible for any pollution.
    _cached_generate_static_rule.cache_clear()
    sim = CellSimulator(Architecture.AMD64, **sim_kwargs)

    last_rax = 0
    for i in indices:
        e = results[i]
        inst = e['instruction']
        bs = bytes.fromhex(inst['bytes'])
        state = {r.name: inst['state'].get(r.name, 0) for r in _REGS}
        if state.get('RSP', 0) == 0:
            state['RSP'] = _DEFAULT_RSP
        taint = {r.name: inst['taint'].get(r.name, 0) for r in _REGS}
        try:
            circuit = generate_static_rule(Architecture.AMD64, bs, _REGS)
            ctx = EvalContext(input_values=state, input_taint=taint, simulator=sim)
            raw = circuit.evaluate(ctx)
        except Exception:
            continue
        last_rax = raw.get('RAX', 0) & ((1 << 64) - 1)
    return last_rax


def verdict(rax: int) -> str:
    if rax == EXPECTED_SOUND:  return 'SOUND'
    if rax == EXPECTED_BROKEN: return 'BROKEN'
    return f'OTHER(0x{rax:016x})'


# ============================================================================
# STAGE 1 — Backend reproduction matrix
# ============================================================================

log('\n' + '=' * 72)
log('STAGE 1 — Backend reproduction')
log('=' * 72)

backends = [
    ('unicorn', dict(use_unicorn=True,  use_c=False)),
    ('cython',  dict(use_unicorn=False, use_c=False)),
    ('c',       dict(use_unicorn=False, use_c=True)),
]

# Run all 8009 prior tests + target through each backend.
all_indices = list(range(target_idx + 1))

stage1_results: dict[str, int] = {}
for name, kw in backends:
    log(f'\n  Backend {name} (use_unicorn={kw["use_unicorn"]}, use_c={kw["use_c"]}):')
    log(f'    Replaying {len(all_indices)} test cases ...')
    rax = replay(all_indices, kw)
    stage1_results[name] = rax
    log(f'    RESULT id={target_id}: RAX = 0x{rax:016x}  [{verdict(rax)}]')

reproducing_backends = [name for name, rax in stage1_results.items() if rax == EXPECTED_BROKEN]
log(f'\n  Backends reproducing the broken value: {reproducing_backends}')

if not reproducing_backends:
    log('\n  No backend reproduced the broken value — cannot proceed with bisection.')
    log('  Either the bug requires a different report, or the environment')
    log('  on this machine is not the one where the bug appears.')
    _log_fh.close()
    sys.exit(1)

# Use the first reproducing backend for bisection.  In practice all three
# should reproduce per the user's claim; we pick the cheapest one.
bisect_backend_name = reproducing_backends[0]
bisect_kw = next(kw for n, kw in backends if n == bisect_backend_name)
log(f'\n  Using backend "{bisect_backend_name}" for bisection.')


# ============================================================================
# STAGE 2 — Lower-bound bisection
# ============================================================================
#
# Find smallest N such that replay([N..target_idx]) reproduces BROKEN.
# Sanity bounds:
#   N=0           -> known BROKEN (all prior cases)
#   N=target_idx  -> known SOUND  (target alone)
#
# Binary-search: invariant is `lo` reproduces BROKEN, `hi` does not.
# We search for the largest N that still produces BROKEN.

log('\n' + '=' * 72)
log('STAGE 2 — Lower-bound bisection')
log('=' * 72)

# Sanity check
log('\n  Sanity: replay just target (no prior) ...')
rax_alone = replay([target_idx], bisect_kw)
log(f'    target alone: 0x{rax_alone:016x}  [{verdict(rax_alone)}]')
if rax_alone == EXPECTED_BROKEN:
    log('  Target alone reproduces broken — bug is not prior-test-dependent.')
    log('  Bisection cannot narrow further.')
    _log_fh.close()
    sys.exit(0)

lo = 0              # known BROKEN
hi = target_idx     # known SOUND
# We search for the largest N (call it threshold) such that
# replay([N..target_idx]) is BROKEN.  threshold ∈ [lo, hi-1].

log(f'\n  Bisecting on the start index N (range [{lo}, {hi}]):')
log(f'    Goal: largest N such that replay([N..{target_idx}]) is BROKEN.')

# Binary-search — at each step ~13 iterations for 0..8009.
iterations = 0
while hi - lo > 1:
    mid = (lo + hi) // 2
    indices = list(range(mid, target_idx + 1))
    rax = replay(indices, bisect_kw)
    v = verdict(rax)
    log(f'    [{iterations:2d}] N={mid:5d}  ({len(indices):5d} tests)  -> {v}')
    if rax == EXPECTED_BROKEN:
        lo = mid     # still broken — try later start
    else:
        hi = mid     # sound — start must be earlier
    iterations += 1

threshold = lo
log(f'\n  Bisection result: largest N producing BROKEN is N = {threshold}')
log(f'    replay([{threshold}..{target_idx}]) -> BROKEN')
log(f'    replay([{threshold + 1}..{target_idx}]) -> not BROKEN')
log(f'\n  Test that FIRST contributes to the poisoning:')
log(f'    index = {threshold}  id = {results[threshold]["id"]}')
log(f'    asm   = {results[threshold]["instruction"]["assembly"][:120]}')
log(f'    bytes = {results[threshold]["instruction"]["bytes"]}')
log(f'    cat   = {results[threshold]["instruction"].get("category")}')


# ============================================================================
# STAGE 3 — Necessary-set extraction
# ============================================================================
#
# Walk forward from threshold+1 to target_idx-1.  For each index i, decide
# whether it is needed:
#   - Run replay(necessary + [i, target_idx])
#   - If BROKEN: i is NOT needed (we got broken without including it as
#     mandatory-already)
#       Actually wait — that logic is wrong.  Let me redo it.
#
# The right algorithm: greedy elimination.  Start with `keep = [threshold,
# target_idx]`.  For each index i in (threshold, target_idx):
#   - Try replay(sorted(keep + [i]))
#   - If BROKEN: don't add i to keep — wasn't needed
#   - If SOUND: add i to keep — it's necessary to reach BROKEN
# At the end, `keep` is a minimal set whose replay produces BROKEN.
#
# Wait that's also wrong.  We want: sequence of preceding tests + target ->
# BROKEN.  We start with [threshold, target] which we KNOW is BROKEN.  We
# want to confirm: which OTHER indices in (threshold, target) are
# necessary to ALSO include for the bug to manifest, vs. which are
# incidental?
#
# Actually since [threshold, target_idx] alone is BROKEN, the answer is:
# nothing else in (threshold, target_idx) is necessary.  threshold alone
# is sufficient.
#
# So the right next question: is just {threshold, target_idx} BROKEN?
# If yes: minimal set is size 2.  Otherwise, the bisection invariant
# violated something — investigate.
# ============================================================================

log('\n' + '=' * 72)
log('STAGE 3 — Minimal-set verification')
log('=' * 72)

log(f'\n  Verifying: replay([{threshold}, {target_idx}]) ...')
rax_pair = replay([threshold, target_idx], bisect_kw)
log(f'    -> 0x{rax_pair:016x}  [{verdict(rax_pair)}]')

if rax_pair == EXPECTED_BROKEN:
    log(f'\n  >>> Minimal poisoning set found: {{{threshold}, {target_idx}}}')
    log(f'      i.e. running test idx={threshold} (id={results[threshold]["id"]})')
    log(f'      followed by test idx={target_idx} (id={target_id})')
    log(f'      reproduces the broken value through ONE shared simulator.')
    log(f'\n  Test idx={threshold} details:')
    inst_p = results[threshold]['instruction']
    log(f'    bytes:    {inst_p["bytes"]}')
    log(f'    asm:      {inst_p.get("assembly", "")[:200]}')
    log(f'    category: {inst_p.get("category")}')
    log(f'    state:    {inst_p["state"]}')
    log(f'    taint:    {inst_p["taint"]}')
else:
    log(f'\n  Pair alone does NOT reproduce — but bisection said')
    log(f'  [{threshold}..{target_idx}] does.  This means MULTIPLE prior')
    log(f'  tests are needed to set up the polluted state.')
    log(f'\n  Iterative expansion: keep adding indices from ({threshold}, {target_idx})')
    log(f'  to the necessary set until replay(necessary + [target]) becomes BROKEN.')

    # Greedy walk-forward: necessary = [threshold].  At each step, scan
    # forward through (last(necessary), target_idx) to find the first index
    # i such that replay(necessary + [i, target_idx]) is BROKEN.  Add it
    # to necessary, then repeat.  Terminates either when a single new
    # index flips us to BROKEN, or when no single addition is sufficient.
    necessary: list[int] = [threshold]
    iteration = 0
    while True:
        iteration += 1
        scan_start = necessary[-1] + 1
        log(f'\n  Iteration {iteration}: scanning ({scan_start}, {target_idx}) ...')
        log(f'    current necessary set: {necessary}')

        found_index = None
        last_progress_log = scan_start
        for i in range(scan_start, target_idx):
            trial = sorted(set(necessary + [i, target_idx]))
            rax_i = replay(trial, bisect_kw)
            if rax_i == EXPECTED_BROKEN:
                inst_i = results[i]['instruction']
                log(f'    [add idx={i:5d} id={results[i]["id"]:5d} '
                    f'cat={inst_i.get("category", "?"):12s}] -> BROKEN')
                necessary.append(i)
                found_index = i
                break
            if i - last_progress_log >= 500:
                log(f'    [progress idx={i:5d}] still SOUND')
                last_progress_log = i

        if found_index is None:
            log(f'\n  No single additional index in ({scan_start}, {target_idx})')
            log(f'  flips replay to BROKEN.  The poisoning is distributed over')
            log(f'  more than `len(necessary)` tests.  Trying a wider expansion:')
            log(f'  test "necessary + ALL of (scan_start, target_idx) + target"')
            indices_full = sorted(set(necessary + list(range(scan_start, target_idx)) + [target_idx]))
            rax_full = replay(indices_full, bisect_kw)
            log(f'    full set replay: {verdict(rax_full)}')
            if rax_full == EXPECTED_BROKEN:
                log(f'  Full set is BROKEN — every index in (scan_start, target_idx)')
                log(f'  collectively contributes; no single one alone is enough.')
                log(f'  Listing the gap as a block (scan_start..target_idx-1):')
                log(f'    block: indices {scan_start} to {target_idx - 1} '
                    f'({target_idx - scan_start} tests)')
            else:
                log(f'  Full set is also SOUND — necessary set without the gap')
                log(f'  works only when the OUTER bisection range is intact.')
                log(f'  This indicates a non-monotonic poisoning behaviour.')
            break

        # We added one index; check if necessary alone now reproduces
        rax_check = replay(necessary + [target_idx], bisect_kw)
        log(f'    replay(necessary + [target]) -> {verdict(rax_check)}')
        if rax_check == EXPECTED_BROKEN:
            log(f'\n  >>> Necessary-set search converged.')
            log(f'      Minimal poisoning set: {necessary} + [{target_idx}]')
            log(f'      Total: {len(necessary) + 1} test cases.\n')
            for idx in necessary + [target_idx]:
                inst_x = results[idx]['instruction']
                log(f'    idx={idx:5d}  id={results[idx]["id"]:5d}  '
                    f'bytes={inst_x["bytes"][:32]:32s}  '
                    f'cat={inst_x.get("category", "?")}')
            break
        # Else: continue expanding
        if iteration >= 50:
            log(f'\n  Iteration limit reached without convergence.')
            break

log(f'\nLog saved to: {LOG_PATH.resolve()}')
_log_fh.close()
