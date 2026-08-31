#!/usr/bin/env python3
"""Maat baseline for the constant-time workload (test_constant_time.c).

Maat is symbolic: the exponent is a concolic 32-bit var "exp" (concrete value 5),
base/mod public. Drive pow_branch / pow_ct directly (glibc start-up can't be
emulated). At each conditional BRANCH a leak = "exp" appears in the branch
condition's contained_vars(); a variant leaks if any branch is secret-dependent.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

# Maat's C++ logger writes ANSI to stdout; keep it off our data path.
os.dup2(2, 1)

from maat import (ACTION, ARCH, BIN, Concat, Cst, EVENT, MaatEngine, OS, PERM,
                  Var, WHEN)

ELF = Path(__file__).resolve().parent / "bin" / "test_constant_time"
POW_BRANCH = 0x402f75
POW_CT = 0x402ff8
RET_SENTINEL = 0x13370000
STACK_TOP = 0x7ffffff00000

BASE = 7        # public
EXPONENT = 5    # secret (stdin), == 0b101
MOD = 101       # public
SECRET_VAR = "exp"


def run_variant(entry: int, name: str) -> dict:
    engine = MaatEngine(ARCH.X64, OS.LINUX)
    engine.load(str(ELF), BIN.ELF64)

    engine.mem.map(STACK_TOP - 0x10000, STACK_TOP + 0x1000, PERM.RW)
    sp = STACK_TOP - 0x800
    engine.mem.write(sp, RET_SENTINEL, 8)
    engine.cpu.rsp = sp
    engine.cpu.rbp = sp
    engine.cpu.rdi = BASE
    # Only the exponent is tainted: low 32 bits are concolic var "exp",
    # upper 32 bits concrete 0.  Concrete valuation EXPONENT lets Maat resolve
    # every branch and execute all 32 iterations.
    engine.cpu.rsi = Concat(Cst(32, 0), Var(32, SECRET_VAR))
    engine.vars.set(SECRET_VAR, EXPONENT)
    engine.cpu.rdx = MOD

    leak_execs: list[str] = []       # dynamic tainted-branch executions
    leak_sites: set[int] = set()     # unique branch PCs with tainted cond
    cond_branches = [0]
    error = [""]

    def on_branch(eng):  # noqa: ANN001
        try:
            cond = eng.info.branch.cond
        except Exception:  # unconditional branch/call: no condition set
            return ACTION.CONTINUE
        if cond is None:
            return ACTION.CONTINUE
        cond_branches[0] += 1
        if SECRET_VAR in set(cond.contained_vars()):
            pc = eng.cpu.rip.as_uint(eng.vars)
            leak_sites.add(pc)
            leak_execs.append(hex(pc))
        return ACTION.CONTINUE

    def on_exec(eng):  # noqa: ANN001
        if eng.cpu.rip.as_uint(eng.vars) == RET_SENTINEL:
            return ACTION.HALT
        return ACTION.CONTINUE

    engine.hooks.add(EVENT.BRANCH, WHEN.BEFORE, callbacks=[on_branch])
    engine.hooks.add(EVENT.EXEC, WHEN.BEFORE, callbacks=[on_exec])

    try:
        engine.run_from(entry)
    except Exception as exc:  # noqa: BLE001
        error[0] = f"{type(exc).__name__}: {exc}"

    return {
        "name": name,
        "entry": hex(entry),
        "cond_branches_total": cond_branches[0],
        "leak_count_static": len(leak_sites),
        "leak_branch_execs": len(leak_execs),
        "leak_sites": sorted(hex(p) for p in leak_sites),
        "leak_exec_site_counts": dict(Counter(leak_execs)),
        "stop": int(engine.info.stop),
        "error": error[0],
    }


def main() -> int:
    vuln = run_variant(POW_BRANCH, "vuln")
    ct = run_variant(POW_CT, "ct")

    print(f"[maat CT] saw: pow_branch {vuln['leak_branch_execs']} secret-dependent "
          f"branches ({vuln['leak_count_static']} sites) | not: pow_ct "
          f"{ct['leak_branch_execs']}")

    out = {"vuln": vuln, "ct": ct}
    Path(__file__).resolve().parent.joinpath(
        "results", "_maat_ct_raw.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
