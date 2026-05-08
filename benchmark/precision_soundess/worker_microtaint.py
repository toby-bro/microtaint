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

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, Register

# State format: every GP register plus XMM<n>_LO/_HI lanes for n=0..7.
#
# Why we need all 16 GP regs (not just RAX-RDX):
# ---------------------------------------------
# The benchmark only sets values/taint for RAX-RDX, but the test bytestrings
# routinely USE other GP registers as pointers (RSP for stack, RDI/RSI for
# string ops, RBP for frames).  When the engine's dep extractor encounters
# a LOAD or STORE whose address is computed from one of those registers
# (e.g. `mov rax, [rsp-64]`), it calls `resolve_ptr_with_offset` to figure
# out the base register and constant offset.  That resolver maps the P-code
# register varnode back to a state_format entry via mapper.map_to_state.
# If the register isn't in state_format, the resolver returns (None, 0)
# and the engine cannot tag the LOAD as a memory-input dependency.
#
# Concrete consequence: for `rep stosb` sequences, the differential
# evaluator falls back to Unicorn for the concrete polarity (because the
# native P-code evaluator can't handle the rep BRANCH).  When Unicorn
# runs with RSP=0 → the simulator's RSP-fallback heuristic kicks in
# (RSP=0x80000000) → the load reads uninitialised memory → the polarity
# C1 returns garbage → the differential C1^C2 produces an unsound taint
# mask that is missing taint bits the GT proves are dependent.
#
# Including RSI/RDI/RSP/RBP/R8-R15 in state_format means the dep extractor
# can resolve LOAD/STORE pointers correctly, the engine routes through the
# memory-aware MemoryDifferentialExpr path, and the differential preserves
# the values stored to and loaded from the stack.
#
# The XMM lanes are required for soundness on instructions that transit
# taint through the SIMD register file (e.g. movq xmm0, rax; paddq;
# movq rax, xmm0).
#
# Cost: ~28 extra slots in the state vector — engine handles this in O(1)
# per slot, no measurable slowdown on the per-test workload.
_REGS = (
    [Register('RAX', 64), Register('RBX', 64), Register('RCX', 64), Register('RDX', 64)]
    + [Register('RSI', 64), Register('RDI', 64), Register('RSP', 64), Register('RBP', 64)]
    + [Register(f'R{n}', 64) for n in range(8, 16)]
    + [Register(f'XMM{n}_LO', 64) for n in range(8)]
    + [Register(f'XMM{n}_HI', 64) for n in range(8)]
)
_SIM = CellSimulator(Architecture.AMD64)

# A safe default RSP for tests that don't carry one.  The simulator's
# Unicorn fallback uses this same value when state.RSP == 0; mirroring
# it here ensures the dep extractor and the Unicorn run agree on where
# the stack lives.
_DEFAULT_RSP = 0x80000000


def run_one(tc: dict) -> dict:
    bytestring = bytes.fromhex(tc['bytes'])
    circuit = generate_static_rule(Architecture.AMD64, bytestring, _REGS)
    # Fill in zeros for any register the test case doesn't mention.  The
    # benchmark only sets values/taint for the four GP regs we report on;
    # other GPs and XMMs default to (state=0, taint=0).
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
    # Report only the GP regs back to the benchmark — that's what the
    # comparison framework expects.
    output_taint = {
        reg: (val & 0xFFFFFFFFFFFFFFFF if isinstance(val, int) else 0)
        for reg, val in raw.items()
        if reg in ('RAX', 'RBX', 'RCX', 'RDX')
    }
    return {"output_taint": output_taint, "time_ns": t1 - t0}


def main():
    # Signal ready
    sys.stdout.write("READY\n")
    sys.stdout.flush()

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        if line == "QUIT":
            break
        try:
            tc = json.loads(line)
            result = run_one(tc)
        except Exception:
            import traceback

            result = {"error": traceback.format_exc(), "time_ns": 0}
        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
