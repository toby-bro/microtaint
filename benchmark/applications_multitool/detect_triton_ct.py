#!/usr/bin/env python3
"""Triton baseline for the constant-time workload (test_constant_time.c).

Emulate pow_branch / pow_ct directly, taint only the exponent (ESI, the secret),
and count conditional branches that read a tainted flag -- i.e. branch on the
secret. A leak = at least one such branch. base/mod are public (untainted).
"""
from __future__ import annotations

import sys
from pathlib import Path

from triton import ARCH, EXCEPTION, Instruction, MemoryAccess, TritonContext

sys.path.insert(0, str(Path(__file__).resolve().parent / "triton_tool"))
from elfload import load_segments  # noqa: E402

ELF = Path(__file__).resolve().parent / "bin" / "test_constant_time"
POW_BRANCH = 0x402f75
POW_CT = 0x402ff8
RET_SENTINEL = 0x13370000
STACK_TOP = 0x7FFFFFFFF000
INSTR_BUDGET = 100_000

BASE = 7        # public
EXPONENT = 5    # secret (stdin), == 0b101
MOD = 101       # public

# x86 conditional-jump mnemonics (everything that starts with 'j' except jmp).
_UNCOND = {"jmp"}


def _mem_bytes(ctx, addr, size):
    return bytes(ctx.getConcreteMemoryAreaValue(addr, size))


def _is_cond_branch(inst) -> bool:
    m = inst.getDisassembly().split()[0].lower()
    return m.startswith("j") and m not in _UNCOND


def run_variant(entry: int, name: str) -> dict:
    ctx = TritonContext(ARCH.X86_64)
    _entry, segs = load_segments(ELF)
    for seg in segs:
        if seg.data:
            ctx.setConcreteMemoryAreaValue(seg.vaddr, seg.data)

    rsp = STACK_TOP - 0x800
    ctx.setConcreteMemoryValue(MemoryAccess(rsp, 8), RET_SENTINEL)
    ctx.setConcreteRegisterValue(ctx.registers.rsp, rsp)
    ctx.setConcreteRegisterValue(ctx.registers.rbp, rsp)
    ctx.setConcreteRegisterValue(ctx.registers.rdi, BASE)
    ctx.setConcreteRegisterValue(ctx.registers.rsi, EXPONENT)
    ctx.setConcreteRegisterValue(ctx.registers.rdx, MOD)
    ctx.setConcreteRegisterValue(ctx.registers.rip, entry)

    # Taint ONLY the exponent (the secret).  We taint the 32-bit ESI that
    # carries `e`; base (EDI) and mod (EDX) stay untainted (public).
    ctx.taintRegister(ctx.registers.esi)

    leaks = []                 # dynamic tainted-branch executions
    static_leak_pcs = set()    # unique branch PCs that were tainted
    n_tainted_execs = 0        # dynamic count of tainted-branch executions
    cond_branches = 0
    steps = 0
    reached_ret = False
    error = ""
    pc = entry

    while steps < INSTR_BUDGET:
        if pc == RET_SENTINEL:
            reached_ret = True
            break
        inst = Instruction(pc, _mem_bytes(ctx, pc, 16))
        fault = ctx.processing(inst)
        if fault != EXCEPTION.NO_FAULT or inst.getSize() == 0:
            error = f"fault {fault} at {pc:#x} ({inst.getDisassembly()})"
            break
        steps += 1

        if _is_cond_branch(inst):
            cond_branches += 1
            # Registers this Jcc reads == the EFLAGS bits that decide it.
            read_regs = [r for r, _ast in inst.getReadRegisters()]
            tainted_flags = [r.getName() for r in read_regs
                             if ctx.isRegisterTainted(r)]
            if tainted_flags:
                static_leak_pcs.add(inst.getAddress())
                n_tainted_execs += 1
                if len(leaks) < 40:
                    leaks.append({
                        "pc": hex(inst.getAddress()),
                        "insn": inst.getDisassembly(),
                        "tainted_flags": tainted_flags,
                    })

        pc = ctx.getConcreteRegisterValue(ctx.registers.rip)

    if steps >= INSTR_BUDGET and not reached_ret:
        error = error or "instruction budget exhausted"

    return {
        "name": name,
        "entry": hex(entry),
        "cond_branches_executed": cond_branches,
        "leak_count_static": len(static_leak_pcs),
        "n_tainted_branch_execs": n_tainted_execs,
        "leak_pcs": sorted(hex(p) for p in static_leak_pcs),
        "leaks_sample": leaks[:5],
        "steps": steps,
        "reached_ret": reached_ret,
        "error": error,
    }


def main() -> int:
    vuln = run_variant(POW_BRANCH, "vuln")
    ct = run_variant(POW_CT, "ct")

    import json
    Path(__file__).resolve().parent.joinpath("results", "_ct_raw.json").write_text(
        json.dumps({"vuln": vuln, "ct": ct}, indent=2))
    print(f"[triton CT] saw: pow_branch {vuln['n_tainted_branch_execs']} secret-dependent "
          f"branches ({vuln['leak_count_static']} sites) | not: pow_ct "
          f"{ct['n_tainted_branch_execs']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
