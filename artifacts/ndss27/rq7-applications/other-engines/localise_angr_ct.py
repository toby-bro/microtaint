#!/usr/bin/env python3
"""angr single-bit CT localisation (backs the "angr localises" cell of the
constant-time attribution table).

detect_angr_apps.py shows the binary verdict with the whole exponent symbolic.
Here we make ONLY exponent bit k symbolic and, at every execution of
pow_branch's key branch, ask whether the secret bit can flip the branch outcome
(the exit guard is satisfiable both taken and not-taken). It is flippable at
exactly one step -- step k+1 -- for every k, i.e. angr localises the leaking key
bit, matching microtaint. (Iterations are counted at the per-iteration
`and $0x1,%eax`; angr's `exit` breakpoint fires once per successor, so the key
branch itself would double-count.)

Run: /home/jns/Documents/Telecom/PRIM/benchmark/.venv_angr/bin/python localise_angr_ct.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import angr
import claripy

logging.getLogger("angr").setLevel(logging.CRITICAL)

ELF = str((Path(__file__).resolve().parent / "bin" / "test_constant_time"))
POW_BRANCH = 0x402f75
AND = 0x402fa3           # `and $0x1,%eax` -- once per loop iteration
KEY = 0x402fa8           # the key-dependent `je`
RET = 0x13370000


def _exponent_with_symbolic_bit(k: int):
    bk = claripy.BVS("bk", 1, explicit_name=True)
    parts = [p for p in (claripy.BVV(0, 31 - k) if 31 - k > 0 else None, bk,
                         claripy.BVV(0, k) if k > 0 else None) if p is not None]
    e32 = parts[0]
    for p in parts[1:]:
        e32 = claripy.Concat(e32, p)
    return claripy.Concat(claripy.BVV(0, 32), e32)


def localise(proj, k: int) -> list[int]:
    st = proj.factory.call_state(POW_BRANCH, 7, _exponent_with_symbolic_bit(k),
                                 101, ret_addr=RET)
    it = [0]
    steps = set()

    def on_and(s):  # noqa: ANN001
        it[0] += 1

    def on_exit(s):  # noqa: ANN001
        if s.addr != KEY:
            return
        g = s.inspect.exit_guard
        if g is None or not g.symbolic:
            return
        # genuine secret-dependence: guard satisfiable both taken and not-taken
        if (s.solver.satisfiable(extra_constraints=[g])
                and s.solver.satisfiable(extra_constraints=[claripy.Not(g)])):
            steps.add(it[0])

    st.inspect.b("instruction", when=angr.BP_BEFORE, instruction=AND, action=on_and)
    st.inspect.b("exit", when=angr.BP_BEFORE, action=on_exit)
    sm = proj.factory.simgr(st)
    n = 0
    while sm.active and n < 3000:
        sm.step(); n += 1
    return sorted(steps)


def main() -> int:
    proj = angr.Project(ELF, auto_load_libs=False)
    ok = True
    for k in range(32):
        steps = localise(proj, k)
        good = steps == [k + 1]
        ok = ok and good
        print(f"secret bit {k:2d} -> flippable branch at step {steps}  "
              f"(expect [{k + 1}])  {'OK' if good else 'MISMATCH'}")
    print("PASS: angr localises every single-bit secret to its exact step"
          if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
