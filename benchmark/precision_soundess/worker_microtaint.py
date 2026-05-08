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

# State format: GP regs RAX-RDX plus XMM<n>_LO/_HI lanes for n=0..7.
# The XMM lanes are required for soundness on instructions that transit
# taint through the SIMD register file (e.g. movq xmm0, rax; paddq;
# movq rax, xmm0).  Without these slots, the engine's StateMapper has
# nowhere to write the intermediate XMM taint, and any taint flowing
# GP → XMM → GP is silently dropped.
# Cost: ~16 extra slots in the state vector — engine handles this in O(1)
# per slot, no measurable slowdown on the per-test workload.
_REGS = (
    [Register('RAX', 64), Register('RBX', 64), Register('RCX', 64), Register('RDX', 64)]
    + [Register(f'XMM{n}_LO', 64) for n in range(8)]
    + [Register(f'XMM{n}_HI', 64) for n in range(8)]
)
_SIM = CellSimulator(Architecture.AMD64)


def run_one(tc: dict) -> dict:
    bytestring = bytes.fromhex(tc['bytes'])
    circuit = generate_static_rule(Architecture.AMD64, bytestring, _REGS)
    # Fill in zeros for any register the test case doesn't mention.  The
    # benchmark only sets values/taint for the four GP regs we report on;
    # XMM regs default to (state=0, taint=0).
    state = {r.name: tc['state'].get(r.name, 0) for r in _REGS}
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
