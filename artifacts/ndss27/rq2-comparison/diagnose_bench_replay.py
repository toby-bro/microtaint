#!/usr/bin/env python3
"""
diagnose_bench_replay.py
========================
Replay the benchmark test cases through a worker-like loop, IN-PROCESS
(no subprocess), so we can introspect the state at the moment id=8009
diverges.

This is the KEY diagnostic: previous diagnostics ran rep-stosb in
isolation and got SOUND output, but the full benchmark gives BROKEN.
The polluter is a specific sequence of preceding test cases.  This
script runs that exact sequence and watches what happens.

Usage:
    .venv_microtaint/bin/python diagnose_bench_replay.py PATH/TO/report.json

Pass the path to a benchmark report JSON.  The script:
  1. Reads test_cases up to and including id=8009
  2. Runs them through ONE in-process worker loop (single _SIM, single
     LogicCircuit cache — i.e., what the worker process sees)
  3. Captures id=8009's RAX taint output and prints internal state at
     that point (compiled circuit identity, _pcode identity,
     CellHandle status, etc.)

Critically, this runs in-process so we can inspect EVERY internal
attribute when the divergence happens.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

cwd = Path.cwd()
worker_path = next((p for p in [cwd / 'worker_microtaint.py', cwd.parent / 'worker_microtaint.py'] if p.is_file()), None)
if worker_path is None:
    print(f'ERROR: worker_microtaint.py not found in {cwd} or its parent.')
    sys.exit(1)
sys.path.insert(0, str(worker_path.parent))

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule, _cached_generate_static_rule
from microtaint.types import Architecture, Register

_REGS = (
    [Register('RAX', 64), Register('RBX', 64), Register('RCX', 64), Register('RDX', 64)]
    + [Register('RSI', 64), Register('RDI', 64), Register('RSP', 64), Register('RBP', 64)]
    + [Register(f'R{n}', 64) for n in range(8, 16)]
    + [Register(f'XMM{n}_LO', 64) for n in range(8)]
    + [Register(f'XMM{n}_HI', 64) for n in range(8)]
)
_DEFAULT_RSP = 0x80000000

REPORT_PATH = sys.argv[1] if len(sys.argv) > 1 else None
if REPORT_PATH is None:
    print('Usage: diagnose_bench_replay.py PATH/TO/report.json')
    sys.exit(1)

with open(REPORT_PATH) as f:
    report = json.load(f)

results = report['results']
target_idx = next((i for i, e in enumerate(results) if e['id'] == 8009), None)
if target_idx is None:
    print('id=8009 not found in report.')
    sys.exit(1)

print(f'Found id=8009 at index {target_idx}')
print(f'Will replay {target_idx + 1} test cases through one in-process worker loop.')
print()

# Single shared simulator (just like the worker daemon)
_SIM = CellSimulator(Architecture.AMD64)
print(f'Built one shared CellSimulator: id=0x{id(_SIM):x}, pcode id=0x{id(_SIM._pcode):x}')
print()

# Track state of LogicCircuit objects — when does the rep-stosb circuit
# first appear?
target_bytestring = bytes.fromhex(results[target_idx]['instruction']['bytes'])
target_circuit_first_seen_at = None
target_circuit_id = None

REPSTOSB_BYTES = target_bytestring

# Replay loop
divergence_seen = False
for i in range(target_idx + 1):
    e = results[i]
    inst = e['instruction']
    bs = bytes.fromhex(inst['bytes'])

    # Worker-like setup
    state = {r.name: inst['state'].get(r.name, 0) for r in _REGS}
    if state.get('RSP', 0) == 0:
        state['RSP'] = _DEFAULT_RSP
    taint = {r.name: inst['taint'].get(r.name, 0) for r in _REGS}

    try:
        circuit = generate_static_rule(Architecture.AMD64, bs, _REGS)
    except Exception as exc:
        print(f'  [{i:5d}] id={e["id"]} generate_static_rule failed: {exc}')
        continue

    # On first sight of the target bytestring, record everything
    if bs == REPSTOSB_BYTES and target_circuit_first_seen_at is None:
        target_circuit_first_seen_at = i
        target_circuit_id = id(circuit)
        print(f'  >> First sight of rep-stosb bytestring at index {i} (id={e["id"]})')
        print(f'     LogicCircuit id: 0x{id(circuit):x}')
        print(f'     _compiled before evaluate: {circuit._compiled is not None and circuit._compiled is not False}')

    ctx = EvalContext(input_values=state, input_taint=taint, simulator=_SIM)
    try:
        raw = circuit.evaluate(ctx)
    except Exception as exc:
        print(f'  [{i:5d}] id={e["id"]} evaluate failed: {exc}')
        continue
    rax_out = raw.get('RAX', 0) & ((1 << 64) - 1)

    # Compare with what the report shows for this case
    expected = (e.get('tool_results', {}).get('microtaint', {})
                  .get('output_taint', {}).get('RAX', 0)) & ((1 << 64) - 1)

    is_target = (e['id'] == 8009)
    if is_target:
        print()
        print('=' * 72)
        print(f'  TARGET: id=8009 reached')
        print('=' * 72)
        print(f'    LogicCircuit id:    0x{id(circuit):x}')
        print(f'    Same as first?      {id(circuit) == target_circuit_id}')
        print(f'    _compiled is set:   {circuit._compiled is not None and circuit._compiled is not False}')
        if circuit._compiled is not None and circuit._compiled is not False:
            print(f'    _compiled type:     {type(circuit._compiled).__name__}')
            try:
                stats = circuit._compiled.stats()
                print(f'    _compiled stats:    {stats}')
            except Exception as exc:
                print(f'    _compiled stats:    error ({exc})')
        print(f'    _SIM id:            0x{id(_SIM):x}')
        print(f'    _SIM._pcode id:     0x{id(_SIM._pcode):x}')
        print(f'    _SIM._pcode native_calls:    {_SIM._pcode.native_calls}')
        print(f'    _SIM._pcode fallback_calls:  {_SIM._pcode.fallback_calls}')
        print()
        print(f'    REPLAY produced RAX = 0x{rax_out:016x}')
        print(f'    REPORT recorded RAX = 0x{expected:016x}')
        print(f'    Match: {rax_out == expected}')
        print()
        if rax_out == 0x0202020202021302:
            print('    >>> Replay produces SOUND output')
        elif rax_out == 0x7706aa7ab587952e:
            print('    >>> Replay reproduces the BROKEN report value')
        else:
            print(f'    >>> Replay produces UNEXPECTED value')

        if rax_out != expected:
            print()
            print('    !!! REPLAY DOES NOT MATCH REPORT !!!')
            print('    The in-process replay diverges from what the benchmark recorded.')
            print('    This means the benchmark is using a different code path or')
            print('    state than this script — possibly subprocess isolation, env')
            print('    vars, or something we have not modeled.')
        divergence_seen = (rax_out != 0x0202020202021302)
        break

    # Intermediate progress
    if i % 1000 == 0 and i > 0:
        print(f'  [{i:5d}] id={e["id"]} cat={inst.get("category", "?")[:15]} ok')
    elif bs == REPSTOSB_BYTES:
        # Other rep-stosb cases — interesting datapoints
        sound = (rax_out & 0xFF) * 0x0101010101010101 == rax_out or rax_out == 0
        flag = '' if sound else '   <-- non-byte-repeated!'
        print(f'  [{i:5d}] id={e["id"]} REP STOSB: RAX_out=0x{rax_out:016x}{flag}')

print()
print('=' * 72)
print('SUMMARY')
print('=' * 72)
if divergence_seen:
    print('  IN-PROCESS REPLAY REPRODUCED THE BROKEN VALUE')
    print('  This proves the bug is in the LogicCircuit/_compiled/_pcode layer,')
    print('  not in subprocess isolation, environment vars, or the Unicorn GT.')
    print('  Next step: diff _compiled.stats() between a fresh-circuit run and')
    print('  this replay run to find what differs.')
else:
    print('  IN-PROCESS REPLAY GAVE SOUND OUTPUT')
    print('  This is significant: the benchmark report shows the BROKEN value')
    print('  for this same case, but running through one shared simulator and')
    print('  one shared circuit cache in-process gives the SOUND value.')
    print()
    print('  Implications:')
    print('    - Bug is NOT just stale cache state in LogicCircuit/_compiled')
    print('    - Bug requires something the in-process replay does not model')
    print('    - Most likely candidate: subprocess interaction (stdin/stdout')
    print('      buffering causes the worker to see test cases in a different')
    print('      order, OR subprocess child-process initial memory layout')
    print('      differs from the parent in a way that affects Unicorn).')
    print()
    print('  Verify by running the benchmark with a SINGLE worker process')
    print('  and instrumenting what the worker actually receives via stdin.')
