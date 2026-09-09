#!/usr/bin/env python3
"""
diagnose_internal_state.py
==========================
Trace every internal step of the rep-stosb id=8009 evaluation, dumping
the intermediate values that should be identical between the working
"with-patch" run and the broken "without-patch" run.

Goal: identify which intermediate value diverges between the two runs.

Run with the user's broken behavior:
    .venv_microtaint/bin/python diagnose_internal_state.py

Run from the benchmark directory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

cwd = Path.cwd()
worker_path = next((p for p in [cwd / 'worker_microtaint.py', cwd.parent / 'worker_microtaint.py'] if p.is_file()), None)
if worker_path is None:
    print(f'ERROR: worker_microtaint.py not found in {cwd} or its parent.')
    sys.exit(1)
sys.path.insert(0, str(worker_path.parent))

from microtaint.instrumentation.ast import (
    EvalContext,
    MemoryDifferentialExpr,
    LogicCircuit,
)
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule, _cached_generate_static_rule
from microtaint.types import Architecture, Register


# ---- Mirror the worker's _REGS ---------------------------------------------

_REGS = (
    [Register('RAX', 64), Register('RBX', 64), Register('RCX', 64), Register('RDX', 64)]
    + [Register('RSI', 64), Register('RDI', 64), Register('RSP', 64), Register('RBP', 64)]
    + [Register(f'R{n}', 64) for n in range(8, 16)]
    + [Register(f'XMM{n}_LO', 64) for n in range(8)]
    + [Register(f'XMM{n}_HI', 64) for n in range(8)]
)

TEST_8009 = {
    'state': {'RAX': 10940498380929573403, 'RBX': 1830928842394036844,
              'RCX': 19, 'RDX': 5767559093351484470},
    'taint': {'RAX': 4352, 'RBX': 2, 'RCX': 0, 'RDX': 8796093022208},
    'bytes': 'fc48894424c0488d7c24c04889d8b908000000f3aa488b4424c0',
}

EXPECTED_SOUND = 0x0202020202021302
EXPECTED_BROKEN = 0x7706aa7ab587952e


def banner(s: str) -> None:
    print(); print('=' * 72); print('  ' + s); print('=' * 72)


def make_context(sim):
    state = {r.name: TEST_8009['state'].get(r.name, 0) for r in _REGS}
    if state.get('RSP', 0) == 0:
        state['RSP'] = 0x80000000
    taint = {r.name: TEST_8009['taint'].get(r.name, 0) for r in _REGS}
    return EvalContext(input_values=state, input_taint=taint, simulator=sim)


def evaluate_and_describe(circuit, ctx, label):
    """Evaluate the circuit and dump each step."""
    print(f'  --- {label} ---')
    print(f'  circuit._compiled is None: {circuit._compiled is None}')
    print(f'  circuit._compiled is False: {circuit._compiled is False}')
    if circuit._compiled is not None and circuit._compiled is not False:
        print(f'  circuit._compiled type: {type(circuit._compiled).__name__}')
    raw = circuit.evaluate(ctx)
    rax = raw.get('RAX', 0) & ((1 << 64) - 1)
    print(f'  RESULT RAX = 0x{rax:016x}')
    if rax == EXPECTED_SOUND:
        verdict = 'SOUND'
    elif rax == EXPECTED_BROKEN:
        verdict = 'BROKEN'
    else:
        verdict = 'OTHER'
    print(f'  verdict: {verdict}')
    print(f'  pcode.fallback_calls: {ctx.simulator._pcode.fallback_calls}')
    print(f'  pcode.native_calls:   {ctx.simulator._pcode.native_calls}')
    return raw, verdict


# ---------------------------------------------------------------------------

banner('SETUP — clear all caches, build a fresh circuit')
_cached_generate_static_rule.cache_clear()
bs = bytes.fromhex(TEST_8009['bytes'])

# Fresh sim
sim_a = CellSimulator(Architecture.AMD64)
print(f'  sim_a id: 0x{id(sim_a):016x}')
print(f'  sim_a._pcode id: 0x{id(sim_a._pcode):016x}')
print(f'  sim_a.use_unicorn: {sim_a.use_unicorn}')
print(f'  sim_a.use_c: {sim_a.use_c}')

# Generate the circuit
circuit = generate_static_rule(Architecture.AMD64, bs, _REGS)
print(f'  circuit id: 0x{id(circuit):016x}')
print(f'  n_assignments: {len(circuit.assignments)}')
for i, a in enumerate(circuit.assignments):
    if hasattr(a.target, 'address_expr'):
        kind = 'MEM'
        try:
            target_str = str(a.target)
        except Exception:
            target_str = f'<MemoryTarget at 0x{id(a.target):x}>'
    else:
        kind = a.target.name
        target_str = kind
    expr_kind = type(a.expression).__name__
    print(f'    [{i}] target={target_str:50s}  expr={expr_kind}')


banner('EXPERIMENT A — Single evaluate on fresh sim (baseline)')
ctx_a = make_context(sim_a)
print(f'  ctx_a.input_values is the worker dict: keys={len(ctx_a.input_values)}')
print(f'  ctx_a.input_taint  is the worker dict: keys={len(ctx_a.input_taint)}')
print(f'  RAX value:  0x{ctx_a.input_values["RAX"]:016x}')
print(f'  RAX taint:  0x{ctx_a.input_taint["RAX"]:016x}')
print(f'  RBX taint:  0x{ctx_a.input_taint["RBX"]:016x}')
raw_a, verdict_a = evaluate_and_describe(circuit, ctx_a, 'first eval')


banner('EXPERIMENT B — Same circuit, same sim, second evaluate')
ctx_b = make_context(sim_a)
raw_b, verdict_b = evaluate_and_describe(circuit, ctx_b, 'second eval')

if raw_a == raw_b:
    print(f'  Match: yes — second eval gives identical output to first')
else:
    print(f'  *** MISMATCH between first and second eval ***')
    for k in set(raw_a) | set(raw_b):
        a_v = int(raw_a.get(k, 0)) & ((1 << 64) - 1)
        b_v = int(raw_b.get(k, 0)) & ((1 << 64) - 1)
        if a_v != b_v:
            print(f'    {k}: first=0x{a_v:016x}  second=0x{b_v:016x}')


banner('EXPERIMENT C — Fresh sim, fresh circuit, single evaluate')
_cached_generate_static_rule.cache_clear()
sim_c = CellSimulator(Architecture.AMD64)
circuit_c = generate_static_rule(Architecture.AMD64, bs, _REGS)
print(f'  is the same circuit object? {circuit is circuit_c}  (should be False after cache_clear)')
ctx_c = make_context(sim_c)
raw_c, verdict_c = evaluate_and_describe(circuit_c, ctx_c, 'fresh-cache fresh-sim eval')


banner('EXPERIMENT D — Cached circuit (already _compiled), fresh sim')
sim_d = CellSimulator(Architecture.AMD64)
print(f'  circuit._compiled is set: {circuit._compiled is not None and circuit._compiled is not False}')
print(f'  Reusing the SAME circuit (with possibly stale _compiled) on a NEW sim.')
ctx_d = make_context(sim_d)
raw_d, verdict_d = evaluate_and_describe(circuit, ctx_d, 'cached circuit, new sim')


banner('EXPERIMENT E — Force MICROTAINT_DISABLE_COMPILED_CIRCUIT=1')
print('  Disable the compiled-bytecode fast path; use the Cython AST walker.')
os.environ['MICROTAINT_DISABLE_COMPILED_CIRCUIT'] = '1'
_cached_generate_static_rule.cache_clear()
sim_e = CellSimulator(Architecture.AMD64)
circuit_e = generate_static_rule(Architecture.AMD64, bs, _REGS)
ctx_e = make_context(sim_e)
raw_e, verdict_e = evaluate_and_describe(circuit_e, ctx_e, 'AST walker only (no _compiled)')
del os.environ['MICROTAINT_DISABLE_COMPILED_CIRCUIT']


banner('VERDICT MATRIX')
results = [
    ('A: fresh sim, first eval', verdict_a, raw_a.get('RAX', 0) & ((1 << 64) - 1)),
    ('B: fresh sim, second eval', verdict_b, raw_b.get('RAX', 0) & ((1 << 64) - 1)),
    ('C: cache_clear + fresh sim', verdict_c, raw_c.get('RAX', 0) & ((1 << 64) - 1)),
    ('D: cached _compiled, new sim', verdict_d, raw_d.get('RAX', 0) & ((1 << 64) - 1)),
    ('E: no _compiled (AST walker)', verdict_e, raw_e.get('RAX', 0) & ((1 << 64) - 1)),
]
for label, verdict, val in results:
    print(f'  {label:35s} = 0x{val:016x}  [{verdict}]')

print()
if all(v == 'SOUND' for _, v, _ in results):
    print('  All experiments produced SOUND output.  The bug does NOT reproduce')
    print('  in this isolated probe.  Please run this same script INSIDE the')
    print('  full benchmark process flow (e.g. invoke after running a chunk of')
    print('  preceding test cases through the same simulator).')
elif all(v == 'BROKEN' for _, v, _ in results):
    print('  All experiments produced BROKEN output.  The bug is environmental')
    print('  and reproduces from a clean simulator on the very first call.')
    print('  Most likely: a build-tagged ABI mismatch or a Python version issue.')
else:
    print('  Mixed results — the bug is partial.  The DIVERGING experiments')
    print('  pinpoint the layer at fault:')
    print()
    print('    A vs B differ          -> _compiled gets corrupted on first use')
    print('    A vs C differ          -> per-circuit state pollution (LogicCircuit)')
    print('    A vs D differ          -> _compiled captures simulator-specific state')
    print('    A vs E differ          -> compiled fast path is the broken layer;')
    print('                              the AST walker fallback is correct')
    print('    A=B=C=D fast all wrong -> issue is in MemoryDifferentialExpr or')
    print('                              the cell.pyx differential, not _compiled')
