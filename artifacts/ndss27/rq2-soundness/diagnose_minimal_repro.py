#!/usr/bin/env python3
"""
diagnose_minimal_repro.py
=========================
Maximally-instrumented reproduction of the cross-test leakage bug.

Plays exactly three test cases through one shared CellSimulator:
  idx=7989  rep movsb sequence with fully-tainted RAX
  idx=7990  mov al,bl ; mov ah,cl
  idx=8009  rep stosb sequence (the target)

This minimal sequence reproduces the broken value 0x7706aa7ab587952e
on Python 3.13 across all three CellSimulator backends.  The script
dumps every observable piece of state at every step, with the goal of
identifying WHERE in the pipeline the divergence happens between
working (3.12) and broken (3.13) behavior.

Run on your Python 3.13 environment:
    .venv_microtaint/bin/python diagnose_minimal_repro.py

The output is also saved to ``minimal_repro.log``.
"""

from __future__ import annotations

import json
import logging
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

logging.getLogger().setLevel(logging.CRITICAL)

_REGS = (
    [Register('RAX', 64), Register('RBX', 64), Register('RCX', 64), Register('RDX', 64)]
    + [Register('RSI', 64), Register('RDI', 64), Register('RSP', 64), Register('RBP', 64)]
    + [Register(f'R{n}', 64) for n in range(8, 16)]
    + [Register(f'XMM{n}_LO', 64) for n in range(8)]
    + [Register(f'XMM{n}_HI', 64) for n in range(8)]
)

# Hard-coded inputs from report_1778423780.json (values copied so this script
# is self-contained — no need to ship the report).
TEST_7989 = {
    'bytes': 'fc48894424e0488d7424e0488d7c24c0b908000000f3a4488b4424c0',
    'state': {'RAX': 15470179137417423390, 'RBX': 5284306197202325960,
              'RCX': 91, 'RDX': 13740211632721712556},
    'taint': {'RAX': 18446744073709551615, 'RBX': 35184372088832,
              'RCX': 12207444339575697730, 'RDX': 1828153273910131988},
    'asm': 'cld; mov [rsp-32], rax; lea rsi, [rsp-32]; lea rdi, [rsp-64]; '
           'mov ecx, 8; rep movsb; mov rax, [rsp-64]',
}
TEST_7990 = {
    'bytes': '88d888cc',
    'state': {'RAX': 7314663058932324252, 'RBX': 14501752426502692987,
              'RCX': 2069826156496372160, 'RDX': 8251849794471225177},
    'taint': {'RAX': 2097152, 'RBX': 1073741824, 'RCX': 0, 'RDX': 0},
    'asm': 'mov al, bl; mov ah, cl',
}
TEST_8009 = {
    'bytes': 'fc48894424c0488d7c24c04889d8b908000000f3aa488b4424c0',
    'state': {'RAX': 10940498380929573403, 'RBX': 1830928842394036844,
              'RCX': 19, 'RDX': 5767559093351484470},
    'taint': {'RAX': 4352, 'RBX': 2, 'RCX': 0, 'RDX': 8796093022208},
    'asm': 'cld; mov [rsp-64], rax; lea rdi, [rsp-64]; mov rax, rbx; '
           'mov ecx, 8; rep stosb; mov rax, [rsp-64]',
}

EXPECTED_SOUND  = 0x0202020202021302
EXPECTED_BROKEN = 0x7706aa7ab587952e

# Watch addresses — the spill locations both rep sequences touch
WATCHED = [0x7FFFFFC0, 0x7FFFFFC8, 0x7FFFFFD0, 0x7FFFFFE0, 0x7FFFFFE8, 0x7FFFFFF0, 0x7FFFFFF8]

# --- Logging helper ---------------------------------------------------------
LOG_PATH = Path('minimal_repro.log')
_log_fh = open(LOG_PATH, 'w')


def log(msg: str = '') -> None:
    print(msg)
    _log_fh.write(msg + '\n')
    _log_fh.flush()


# --- Helpers ----------------------------------------------------------------

def make_state(tc):
    state = {r.name: tc['state'].get(r.name, 0) for r in _REGS}
    if state.get('RSP', 0) == 0:
        state['RSP'] = 0x80000000
    return state


def make_taint(tc):
    return {r.name: tc['taint'].get(r.name, 0) for r in _REGS}


def dump_sim(sim, tag):
    log(f'  [{tag}]')
    log(f'    _dirtied_memory ({len(sim._dirtied_memory)} addrs): {sorted(sim._dirtied_memory)}')
    log(f'    _mapped_pages: {sorted(sim._mapped_pages)}')
    if hasattr(sim, '_pcode') and sim._pcode is not None:
        nc = getattr(sim._pcode, 'native_calls', '?')
        fc = getattr(sim._pcode, 'fallback_calls', '?')
        log(f'    pcode native_calls={nc}  fallback_calls={fc}')
    for addr in WATCHED:
        try:
            data = bytes(sim.uc.mem_read(addr, 8))
            in_dirty = addr in sim._dirtied_memory
            log(f'    mem[0x{addr:08x}] = {data.hex()}  in_dirty={in_dirty}')
        except Exception as e:
            log(f'    mem[0x{addr:08x}] = unreadable ({e})')


def run_test(sim, tc, label):
    log(f'\n  >> Running {label}: bytes={tc["bytes"][:32]}...')
    log(f'     asm: {tc["asm"][:120]}')
    log(f'     state: RAX=0x{tc["state"]["RAX"]:016x} RBX=0x{tc["state"]["RBX"]:016x} '
        f'RCX=0x{tc["state"]["RCX"]:016x} RDX=0x{tc["state"]["RDX"]:016x}')
    log(f'     taint: RAX=0x{tc["taint"]["RAX"]:016x} RBX=0x{tc["taint"]["RBX"]:016x} '
        f'RCX=0x{tc["taint"]["RCX"]:016x} RDX=0x{tc["taint"]["RDX"]:016x}')
    state = make_state(tc)
    taint = make_taint(tc)
    bs = bytes.fromhex(tc['bytes'])
    circuit = generate_static_rule(Architecture.AMD64, bs, _REGS)
    log(f'     circuit type={type(circuit).__name__}  id=0x{id(circuit):x}')
    compiled = getattr(circuit, '_compiled', None)
    if compiled is not None:
        log(f'     _compiled={compiled is not None and compiled is not False}')
    ctx = EvalContext(input_values=state, input_taint=taint, simulator=sim)
    raw = circuit.evaluate(ctx)
    rax = raw.get('RAX', 0) & ((1 << 64) - 1)
    log(f'     RESULT RAX = 0x{rax:016x}')
    return rax


# --- Main scenarios ---------------------------------------------------------

def scenario(name, sim_kwargs):
    log('\n' + '=' * 72)
    log(f'SCENARIO: {name} ({sim_kwargs})')
    log('=' * 72)

    _cached_generate_static_rule.cache_clear()
    sim = CellSimulator(Architecture.AMD64, **sim_kwargs)
    log(f'  CellSimulator id=0x{id(sim):x}, pcode id=0x{id(sim._pcode):x}')

    dump_sim(sim, 'initial state')

    # Run 7989
    rax_a = run_test(sim, TEST_7989, '7989 (rep movsb)')
    dump_sim(sim, 'after 7989')

    # Run 7990
    rax_b = run_test(sim, TEST_7990, '7990 (partial_write)')
    dump_sim(sim, 'after 7990')

    # Run 8009 — the target
    rax_c = run_test(sim, TEST_8009, '8009 (rep stosb) [TARGET]')
    dump_sim(sim, 'after 8009')

    # Verdict
    log(f'\n  Final 8009 RAX taint: 0x{rax_c:016x}')
    if rax_c == EXPECTED_SOUND:
        log(f'  SOUND  (expected on a healthy run)')
    elif rax_c == EXPECTED_BROKEN:
        log(f'  BROKEN  (reproduces the report bug)')
    else:
        log(f'  OTHER  (different from both sound and broken)')


# Run all three backends
scenario('unicorn', dict(use_unicorn=True, use_c=False))
scenario('cython',  dict(use_unicorn=False, use_c=False))
scenario('c',       dict(use_unicorn=False, use_c=True))


# --- Variations to localize the cause ---------------------------------------

log('\n' + '=' * 72)
log('VARIATIONS — localize what triggers the bug')
log('=' * 72)


def replay(tcs, sim_kwargs, label):
    log(f'\n  Variation: {label}  ({sim_kwargs})')
    _cached_generate_static_rule.cache_clear()
    sim = CellSimulator(Architecture.AMD64, **sim_kwargs)
    last_rax = 0
    for tc, name in tcs:
        state = make_state(tc); taint = make_taint(tc)
        bs = bytes.fromhex(tc['bytes'])
        circuit = generate_static_rule(Architecture.AMD64, bs, _REGS)
        ctx = EvalContext(input_values=state, input_taint=taint, simulator=sim)
        raw = circuit.evaluate(ctx)
        last_rax = raw.get('RAX', 0) & ((1 << 64) - 1)
        log(f'    after {name}: RAX = 0x{last_rax:016x}')
    return last_rax


# Backend that previously showed the bug — pick c-kernel for these variations
KW = dict(use_unicorn=False, use_c=True)

replay([(TEST_8009, '8009')], KW, 'just 8009')
replay([(TEST_7989, '7989'), (TEST_8009, '8009')], KW, '7989 -> 8009')
replay([(TEST_7990, '7990'), (TEST_8009, '8009')], KW, '7990 -> 8009')
replay([(TEST_7989, '7989'), (TEST_7990, '7990'), (TEST_8009, '8009')], KW, '7989 -> 7990 -> 8009')
replay([(TEST_7990, '7990'), (TEST_7989, '7989'), (TEST_8009, '8009')], KW, '7990 -> 7989 -> 8009 (swapped order)')

# Repeat 8009 multiple times after poisoning
log(f'\n  Stability check: poison then run 8009 many times')
_cached_generate_static_rule.cache_clear()
sim = CellSimulator(Architecture.AMD64, **KW)
run_test(sim, TEST_7989, '7989')
run_test(sim, TEST_7990, '7990')
for n in range(5):
    rax = run_test(sim, TEST_8009, f'8009 #{n+1}')
    log(f'    iteration {n+1}: 0x{rax:016x}')


log(f'\n  Log saved to: {LOG_PATH.resolve()}')
_log_fh.close()
