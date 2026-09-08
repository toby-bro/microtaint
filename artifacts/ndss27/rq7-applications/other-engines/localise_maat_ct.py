#!/usr/bin/env python3
"""Maat single-bit CT localisation (backs the "Maat localises" cell of the
constant-time attribution table).

detect_maat_ct.py shows the binary verdict with the WHOLE exponent concolic.
Here we make ONLY exponent bit k symbolic (the other 31 bits concrete 0) and, at
every execution of pow_branch's key branch (the `je` after `and $1; test`), ask
whether the secret bit can flip the branch outcome (the branch condition is
satisfiable both true and false). It is flippable at exactly one step -- step
k+1 -- for every k, i.e. Maat localises the leaking key bit, matching microtaint
(crypto/square_and_multiply/localise_side_channel.py).

Run: /home/jns/Documents/Telecom/PRIM/benchmark/.venv_maat/bin/python localise_maat_ct.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.dup2(2, 1)  # keep Maat's C++ ANSI logger off stdout

from maat import (ACTION, ARCH, BIN, Concat, Cst, EVENT, MaatEngine, OS, PERM,
                  Solver, Var, WHEN)

ELF = str((Path(__file__).resolve().parent / "bin" / "test_constant_time"))
POW_BRANCH = 0x402f75
KEY = 0x402fa8            # `je` after `and $0x1,%eax ; test %eax,%eax`
RET = 0x13370000
STACK_TOP = 0x7ffffff00000


def _exponent_with_symbolic_bit(k: int):
    """32-bit exponent: bit k is a fresh 1-bit symbol `bk`, all other bits 0."""
    parts = [p for p in (Cst(31 - k, 0) if 31 - k > 0 else None,
                         Var(1, "bk"),
                         Cst(k, 0) if k > 0 else None) if p is not None]
    low = parts[0]
    for p in parts[1:]:
        low = Concat(low, p)
    return Concat(Cst(32, 0), low)


def _both_feasible(cond) -> bool:
    """True iff the branch condition is satisfiable both ways over the secret."""
    s_t = Solver(); s_t.add(cond)
    s_f = Solver(); s_f.add(cond.invert())
    return s_t.check() and s_f.check()


def localise(k: int) -> list[int]:
    eng = MaatEngine(ARCH.X64, OS.LINUX)
    eng.load(ELF, BIN.ELF64)
    eng.mem.map(STACK_TOP - 0x10000, STACK_TOP + 0x1000, PERM.RW)
    sp = STACK_TOP - 0x800
    eng.mem.write(sp, RET, 8)
    eng.cpu.rsp = sp; eng.cpu.rbp = sp
    eng.cpu.rdi = 7; eng.cpu.rdx = 101                 # base, mod (public)
    eng.cpu.rsi = _exponent_with_symbolic_bit(k)
    eng.vars.set("bk", 1)
    state = {"it": 0, "steps": []}

    def on_branch(e):  # noqa: ANN001
        try:
            cond = e.info.branch.cond
        except Exception:  # noqa: BLE001
            return ACTION.CONTINUE
        if cond is None or e.cpu.rip.as_uint(e.vars) != KEY:
            return ACTION.CONTINUE
        state["it"] += 1
        if _both_feasible(cond):
            state["steps"].append(state["it"])
        return ACTION.CONTINUE

    def on_exec(e):  # noqa: ANN001
        return ACTION.HALT if e.cpu.rip.as_uint(e.vars) == RET else ACTION.CONTINUE

    eng.hooks.add(EVENT.BRANCH, WHEN.BEFORE, callbacks=[on_branch])
    eng.hooks.add(EVENT.EXEC, WHEN.BEFORE, callbacks=[on_exec])
    eng.run_from(POW_BRANCH)
    return state["steps"]


def main() -> int:
    ok = True
    for k in range(32):
        steps = localise(k)
        good = steps == [k + 1]
        ok = ok and good
        print(f"secret bit {k:2d} -> flippable branch at step {steps}  "
              f"(expect [{k + 1}])  {'OK' if good else 'MISMATCH'}")
    print("PASS: Maat localises every single-bit secret to its exact step"
          if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
