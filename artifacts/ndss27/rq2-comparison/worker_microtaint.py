#!/usr/bin/env python3
"""
worker_microtaint.py — persistent daemon mode.

Protocol (stdin/stdout, line-delimited JSON):
  ← {"arch": ..., "bytes": ..., "state": {...}, "taint": {...}}
  → {"output_taint": {...}, "time_ns": N}
  ← "QUIT"  → exits cleanly
"""

import json
import sys
import time
from typing import Any

from microtaint.debug.reg_aliases import RegisterAliases
from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, Register

# State format: every GP register plus XMM<n>_LO/_HI lanes for n=0..7.
#
# Why we need all 16 GP regs (not just RAX-RDX):
# ---------------------------------------------
# The benchmark only sets values/taint for RAX-RDX, but the test
# bytestrings routinely USE other GP registers as pointers (RSP for
# stack, RDI/RSI for string ops, RBP for frames).  When the engine's
# dep extractor encounters a LOAD or STORE whose address is computed
# from one of those registers (e.g. `mov rax, [rsp-64]`), it calls
# resolve_ptr_with_offset to figure out the base register and constant
# offset.  That resolver maps the P-code register varnode back to a
# state_format entry via mapper.map_to_state.  If the register isn't
# in state_format, the resolver returns (None, 0) and the engine
# cannot tag the LOAD as a memory-input dependency — so the engine
# falls back to a pure-register differential that runs Unicorn with
# RSP=0, hits the safe-stack-fallback heuristic at 0x80000000, reads
# uninitialised memory there, and produces a non-deterministic
# differential.  See engine.py and report 1778266748 for the full
# trace of this on rep stosb id 8009/8337.
#
# Including RSI/RDI/RSP/RBP/R8-R15 in state_format means the dep
# extractor can resolve LOAD/STORE pointers correctly, the engine
# routes through the memory-aware MemoryDifferentialExpr path, and
# the differential preserves the values stored to and loaded from
# the stack.
#
# The XMM lanes are required for soundness on instructions that
# transit taint through the SIMD register file (e.g.
# movq xmm0, rax; paddq; movq rax, xmm0).
#
# They must be named the way the compute path names them, which is NOT
# 'XMM0_LO': the engine tracks vector registers as geometry-derived 8-byte
# lanes called VL_<sleigh-byte-offset>.  A state_format entry the mapper
# cannot resolve is silently inert -- no error, just an instruction whose
# SIMD taint has nowhere to live -- so `movq xmm0,rax; paddq xmm0,xmm1;
# movq rax,xmm0` comes back reporting less taint than moved.  RegisterAliases
# derives the lane names from pypcode's own register geometry, so this stays
# correct if the offsets move.
_ALIASES = RegisterAliases(Architecture.AMD64)
_XMM_LANES = [name for n in range(8) for name in _ALIASES.to_engine_names(f'XMM{n}')]
_REGS = (
    [Register('RAX', 64), Register('RBX', 64), Register('RCX', 64), Register('RDX', 64)]
    + [Register('RSI', 64), Register('RDI', 64), Register('RSP', 64), Register('RBP', 64)]
    + [Register(f'R{n}', 64) for n in range(8, 16)]
    + [Register(name, 64) for name in _XMM_LANES]
)

# Module-level singleton.  Building a CellSimulator allocates a Unicorn
# instance and an internal P-code evaluator: roughly 2 ms of startup
# cost per construction.  Allocating a fresh one per test case takes
# the per-case latency from ~17 us to ~2.3 ms — a 130x slowdown that
# turned the daemon's median per-test latency from 8.6 us into 41 us
# and dropped throughput from 116 k/s to 24 k/s on the 9858-case suite.
#
# The simulator's internal Unicorn state is safe to share across test
# cases: simulator.py's clear_memory_and_registers() is called at the
# start of every emulation that takes the Unicorn fallback path, which
# resets all GP/XMM/segment registers and clears the dirty-memory set.
# The native P-code evaluator path (the common case) doesn't touch
# Unicorn memory at all — it operates on a local _PCodeFrame.
_SIM = CellSimulator(Architecture.AMD64, use_unicorn=False, use_c=True)

# A safe default RSP for tests that don't carry one.  The simulator's
# Unicorn fallback uses this same value when state.RSP == 0; mirroring
# it here ensures the dep extractor and the Unicorn run agree on where
# the stack lives.
_DEFAULT_RSP = 0x80000000


def run_one(tc: dict[str, Any]) -> dict[str, Any]:
    bytestring = bytes.fromhex(tc['bytes'])
    circuit = generate_static_rule(Architecture.AMD64, bytestring, _REGS)
    # Fill in zeros for any register the test case doesn't mention.
    # The benchmark only sets values/taint for the four GP regs we
    # report on; other GPs and XMMs default to (state=0, taint=0).
    state = {r.name: tc['state'].get(r.name, 0) for r in _REGS}
    if state.get('RSP', 0) == 0:
        state['RSP'] = _DEFAULT_RSP
    taint = {r.name: tc['taint'].get(r.name, 0) for r in _REGS}
    ctx = EvalContext(
        input_values=state,
        input_taint=taint,
        simulator=_SIM,
    )
    t0 = time.process_time_ns()
    raw = circuit.evaluate(ctx)
    t1 = time.process_time_ns()
    # Report only the GP regs back to the benchmark — that's what
    # the comparison framework expects.
    output_taint = {
        reg: (val & 0xFFFFFFFFFFFFFFFF if isinstance(val, int) else 0)
        for reg, val in raw.items()
        if reg in ('RAX', 'RBX', 'RCX', 'RDX')
    }
    return {'output_taint': output_taint, 'time_ns': t1 - t0}


def main() -> None:
    # Signal ready
    sys.stdout.write('READY\n')
    sys.stdout.flush()

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        if line == 'QUIT':
            break
        try:
            tc = json.loads(line)
            result = run_one(tc)
        except Exception:
            import traceback

            result = {'error': traceback.format_exc(), 'time_ns': 0}
        sys.stdout.write(json.dumps(result) + '\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
