#!/usr/bin/env python3
"""
worker_angr.py — persistent daemon mode.

Protocol (stdin/stdout, line-delimited JSON):
  ← {"arch": ..., "bytes": ..., "state": {...}, "taint": {...}}
  → {"output_taint": {...}, "time_ns": N, "n_states": N}
  ← "QUIT"  → exits cleanly

Execution model
---------------
The whole bytestring is executed end-to-end via ``simgr.explore(find=end)``.
This causes angr to fork at every conditional branch and follow every
feasible path until each one reaches the end of the code blob.  Output
taint is the UNION of every found state's per-register taint mask: a bit
of register R is reported tainted if the bit is symbolic in ANY of the
final states.

This is the correct semantics for path-explosion benchmarks: a sequence
like ``test rbx,1; jz +3; add rax,rbx; shr rbx,1`` (×N) forks 2^N ways
on a tainted RBX, and the union over all 2^N final states is exactly
"is RAX tainted in any reachable execution".

A SIGALRM-based wall-clock timeout protects against pathological inputs
that would otherwise spin in the SMT solver indefinitely.  The previous
implementation of this worker used a custom explore loop with state-cap
and time-budget guards; it accessed ``simgr.found`` defensively before
the stash existed (angr only creates the ``found`` stash on the first
``explore()`` call) which raised ``AttributeError`` on every test case
and left every result errored.  The simple pattern below avoids that.
"""
import io
import json
import logging
import signal
import sys
import time

import angr
import claripy

logging.getLogger('angr').setLevel(logging.ERROR)
logging.getLogger('cle').setLevel(logging.ERROR)
logging.getLogger('claripy').setLevel(logging.ERROR)
logging.getLogger('pyvex').setLevel(logging.ERROR)

# Wall-clock ceiling for a single test case.  Most tests finish in
# milliseconds; the explicit-branch path-explosion at N=12 with 2^12 = 4096
# paths takes ~30 s.  We allow 60 s before raising TimeoutError.
ANGR_PER_CASE_TIMEOUT_S = 60


class _AngrTimeout(Exception):
    """Raised by SIGALRM handler when a single test exceeds the budget."""


def _sigalrm_handler(signum, frame):
    raise _AngrTimeout()


def make_tainted_bv(reg_name: str, val: int, taint_mask: int, reg_size: int):
    """Build a bitvector with concrete bits where the taint mask is 0 and
    fresh symbolic bits where the mask is 1.

    Equivalent to: for each bit, pick from a symbolic BVS if tainted, else
    from a constant BVV with the concrete value.  We concatenate per-bit
    extractions because that's the cleanest way to get bit-precise mixed
    concrete/symbolic registers in claripy.
    """
    if taint_mask == 0:
        return claripy.BVV(val, reg_size)
    if taint_mask == (1 << reg_size) - 1:
        return claripy.BVS(f"taint_{reg_name}", reg_size)
    sym = claripy.BVS(f"taint_{reg_name}", reg_size)
    con = claripy.BVV(val, reg_size)
    bits = []
    for bit in range(reg_size - 1, -1, -1):
        if (taint_mask >> bit) & 1:
            bits.append(claripy.Extract(bit, bit, sym))
        else:
            bits.append(claripy.Extract(bit, bit, con))
    return claripy.Concat(*bits)


def _bit_mask_of_symbolic(val, reg_size: int) -> int:
    """Return mask of which bits of `val` are symbolic.

    A non-symbolic bitvector contributes 0.  A fully-symbolic single BVS
    contributes ``(1<<reg_size) - 1``.  Otherwise we walk every bit and
    test each one — this is what claripy supports.  The bit-walk is
    O(reg_size) per register so cheap even for many states.
    """
    if not val.symbolic:
        return 0
    if val.depth == 1 and val.op == "BVS":
        return (1 << reg_size) - 1
    mask = 0
    for bit in range(reg_size):
        if claripy.Extract(bit, bit, val).symbolic:
            mask |= 1 << bit
    return mask


def run_one(tc: dict) -> dict:
    arch_map = {"x86": "x86", "x86_64": "amd64"}
    angr_arch = arch_map[tc['arch']]
    reg_size = 32 if tc['arch'] == "x86" else 64

    code = bytes.fromhex(tc['bytes'])
    base_addr = 0x400000
    end_addr = base_addr + len(code)

    # Build a fresh project per call.  This is what the standalone
    # path_explosion_explicit.py does and it keeps the run reproducible —
    # angr caches some lifter state on Project that's hard to reset.
    proj = angr.Project(
        io.BytesIO(code),
        main_opts={'backend': 'blob', 'arch': angr_arch, 'base_addr': base_addr},
    )
    state = proj.factory.blank_state(addr=base_addr)

    # Install the symbolic+concrete register mix.  Registers not in tc['state']
    # remain at angr's default (typically zero or symbolic depending on
    # blank_state options); we don't override them.
    for reg_name, val in tc['state'].items():
        taint = tc['taint'].get(reg_name, 0)
        bv = make_tainted_bv(reg_name, val, taint, reg_size)
        state.registers.store(reg_name.lower(), bv)

    t0 = time.process_time_ns()
    simgr = proj.factory.simgr(state)

    aborted = False
    abort_reason = None

    # Wall-clock guard via SIGALRM.  We can't use angr's internal timeouts
    # because they apply per-step and don't cap the total exploration time.
    # SIGALRM is process-wide but we only set it during the explore call so
    # the protocol loop in main() is unaffected.
    prev_handler = signal.signal(signal.SIGALRM, _sigalrm_handler)
    signal.alarm(ANGR_PER_CASE_TIMEOUT_S)
    try:
        simgr.explore(find=end_addr)
    except _AngrTimeout:
        aborted = True
        abort_reason = f"timeout after {ANGR_PER_CASE_TIMEOUT_S}s"
    except Exception as exc:
        aborted = True
        abort_reason = f"{type(exc).__name__}: {str(exc)[:200]}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_handler)

    t1 = time.process_time_ns()

    # Pull the states we'll union over.  After explore() the simgr always
    # has a 'found' stash if any state reached end_addr.  Use stashes.get()
    # so we don't blow up on the empty case (which can happen if explore
    # was aborted or every path errored before reaching end_addr).
    found = simgr.stashes.get('found', [])
    deadended = simgr.stashes.get('deadended', [])
    active = simgr.stashes.get('active', [])

    # Found is canonical; fall back to deadended (paths that finished but
    # didn't hit end_addr exactly — e.g. lifted past the end into invalid
    # bytes), then to whatever remains active when we time out.
    final_states = list(found) or list(deadended) or list(active)

    output_taint = {reg: 0 for reg in tc['state']}
    if final_states:
        for st in final_states:
            for reg_name in tc['state']:
                val = st.registers.load(reg_name.lower())
                output_taint[reg_name] |= _bit_mask_of_symbolic(val, reg_size)

    result = {
        "output_taint": output_taint,
        "time_ns": t1 - t0,
        "n_states": len(final_states),
    }
    if aborted:
        result["aborted"] = True
        result["abort_reason"] = abort_reason
    return result


def main():
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
