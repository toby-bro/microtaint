#!/usr/bin/env python3
"""angr baseline for the two application workloads (CT and DNS).

angr models the secret as symbolic bitvectors ("tainted" = the expression
depends on the input symbol), rather than shadow-tainting bytes.

CT (test_constant_time.c): symbolic exponent, count conditional branches whose
guard depends on it. pow_branch leaks, pow_ct does not. The exponent is pinned to
its concrete value (5) to follow one path while guards stay symbolic.

DNS (AND al,0x78 ; SHR al,3): make the flag byte symbolic and test per-bit
dependence of the output -- bit 7 (QR, masked away) vs bits 6..3 (OPCODE).
"""
from __future__ import annotations

import json
import logging
import signal
import subprocess
import sys
from pathlib import Path

import angr
import claripy

logging.getLogger("angr").setLevel(logging.CRITICAL)
logging.getLogger("cle").setLevel(logging.CRITICAL)
logging.getLogger("pyvex").setLevel(logging.CRITICAL)
logging.getLogger("claripy").setLevel(logging.CRITICAL)

HERE = Path(__file__).resolve().parent
ELF = HERE / "bin" / "test_constant_time_angr"
RESULTS = HERE / "results"

BASE = 7        # public constant
EXPONENT = 5    # secret (stdin, 4-byte LE) == 0b101
MOD = 101       # public constant

STEP_BUDGET = 4000
WALLCLOCK_S = 240


# --------------------------------------------------------------------------- #
# WORKLOAD 1 : constant-time control-flow side channel
# --------------------------------------------------------------------------- #
def _sym_addr(elf: Path, name: str) -> int:
    out = subprocess.check_output(["nm", str(elf)], text=True)
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == name:
            return int(parts[0], 16)
    raise RuntimeError(f"symbol {name!r} not found in {elf}")


class _Timeout(Exception):
    pass


def _alarm(signum, frame):  # noqa: ANN001
    raise _Timeout()


def run_ct_variant(entry: int, name: str) -> dict:
    """Emulate pow_branch/pow_ct directly (like the Triton baseline) with a
    SYMBOLIC exponent argument, and record every conditional-branch guard that
    depends on the exponent symbol."""
    proj = angr.Project(str(ELF), auto_load_libs=False)

    exponent = claripy.BVS("exponent", 32, explicit_name=True)
    exp_name = exponent.args[0]

    # call_state lays down a clean frame with a sentinel return addr so the
    # function dead-ends on `ret`. Drop LAZY_SOLVES so unsatisfiable successors
    # (the not-taken side of a branch, given exponent==5) are pruned eagerly and
    # we follow the single concrete path -- no 2^32 loop fork.
    state = proj.factory.call_state(
        entry,
        remove_options={angr.options.LAZY_SOLVES},
    )
    # System V AMD64: base=RDI, e=RSI, mod=RDX. Only the exponent is symbolic.
    state.regs.rdi = claripy.BVV(BASE, 64)
    state.regs.rsi = exponent.zero_extend(32)
    state.regs.rdx = claripy.BVV(MOD, 64)
    # Fair concretization: pin the symbolic exponent to the tested value so the
    # loop bound / shift chain is deterministic (single path) while the guard
    # ASTs stay symbolic functions of `exponent`.
    state.add_constraints(exponent == EXPONENT)

    # Distinct static branch SITES whose guard depends on the exponent, and the
    # dynamic count of such branch executions.
    leak_sites: dict[int, dict] = {}
    n_tainted_execs = {"n": 0}
    all_cond_sites: set[int] = set()

    def on_exit(st):  # noqa: ANN001
        guard = st.inspect.exit_guard
        jk = st.inspect.exit_jumpkind
        if guard is None or jk != "Ijk_Boring":
            return
        # Unconditional/default exits have a concrete guard (True); skip them.
        if not guard.symbolic:
            return
        ins = st.scratch.ins_addr  # address of the branch instruction
        all_cond_sites.add(ins)
        gvars = set(st.solver.variables(guard))
        if exp_name in gvars:
            n_tainted_execs["n"] += 1
            if ins not in leak_sites:
                leak_sites[ins] = {
                    "pc": hex(ins),
                    "guard": str(guard)[:160],
                    "guard_vars": sorted(gvars),
                }

    state.inspect.b("exit", when=angr.BP_BEFORE, action=on_exit)

    simgr = proj.factory.simgr(state)
    err = ""
    prev = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(WALLCLOCK_S)
    steps = 0
    try:
        while simgr.active and steps < STEP_BUDGET:
            simgr.step()
            steps += 1
    except _Timeout:
        err = f"wallclock timeout after {WALLCLOCK_S}s"
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)

    return {
        "name": name,
        "entry": hex(entry),
        "steps": steps,
        "n_deadended": len(simgr.deadended),
        "cond_branch_sites_total": len(all_cond_sites),
        "secret_dep_branch_sites": sorted(hex(p) for p in leak_sites),
        "secret_dep_branch_count": len(leak_sites),
        "secret_dep_branch_execs": n_tainted_execs["n"],
        "leaks": list(leak_sites.values()),
        "error": err,
    }


# --------------------------------------------------------------------------- #
# WORKLOAD 2 : DNS bit-field (bit-precise QR vs OPCODE discrimination)
# --------------------------------------------------------------------------- #
DNS_CODE = bytes.fromhex("2478c0e803")  # AND AL,0x78 ; SHR AL,3


def _run_dns_bytes(flag_ast):
    """Run the two LDNS_OPCODE_WIRE instructions in angr on a symbolic (or
    partly-symbolic) AL and return the resulting 8-bit AL AST."""
    proj = angr.load_shellcode(DNS_CODE, arch="amd64")
    st = proj.factory.blank_state(
        addr=proj.entry,
        remove_options={angr.options.LAZY_SOLVES},
    )
    st.regs.rax = claripy.BVV(0, 64)
    st.regs.al = flag_ast  # inject the (partly) symbolic flag byte into AL
    succ = st.step(num_inst=2)
    out_state = succ.successors[0]
    return out_state.regs.al  # 8-bit result


def _depends_on_bit(bit: int) -> bool:
    """Does the output depend on input bit `bit`?  Run the two instructions on
    two symbolic inputs that agree on every OTHER bit and ask whether the
    outputs can differ (SAT) -- a sound per-bit dependence test, fully via
    angr's instruction semantics."""
    a = claripy.BVS("a", 8)
    b = claripy.BVS("b", 8)
    out_a = _run_dns_bytes(a)
    out_b = _run_dns_bytes(b)
    s = claripy.Solver()
    for j in range(8):
        if j != bit:
            s.add(claripy.Extract(j, j, a) == claripy.Extract(j, j, b))
    s.add(out_a != out_b)
    return s.satisfiable()


def run_dns() -> dict:
    # (1) fully symbolic flag byte -> record the output AST, test per-bit deps.
    flag = claripy.BVS("flag", 8, explicit_name=True)
    out = _run_dns_bytes(flag)

    qr_dep = _depends_on_bit(7)                               # bit7 (QR)
    opcode_bits = {i: _depends_on_bit(i) for i in (6, 5, 4, 3)}
    low_bits = {i: _depends_on_bit(i) for i in (2, 1, 0)}
    opcode_dep = any(opcode_bits.values())

    # (2) LITERAL single-bit taint: make ONLY QR (bit7) symbolic, rest concrete;
    #     then ONLY OPCODE (bits6..3) symbolic. angr can do this because claripy
    #     bitvectors are bit-addressable (Concat/Extract) -- a byte engine cannot.
    qr_only = claripy.Concat(claripy.BVS("qr", 1, explicit_name=True),
                             claripy.BVV(0, 7))          # only bit7 symbolic
    out_qr_only = _run_dns_bytes(qr_only)
    op_only = claripy.Concat(claripy.BVV(0, 1),
                             claripy.BVS("opc", 4, explicit_name=True),
                             claripy.BVV(0, 3))          # only bits6..3 symbolic
    out_op_only = _run_dns_bytes(op_only)

    # With only QR symbolic the output must be CONCRETE (QR masked away);
    # with only OPCODE symbolic the output must be SYMBOLIC (OPCODE survives).
    qr_only_out_symbolic = out_qr_only.symbolic
    op_only_out_symbolic = out_op_only.symbolic

    qr_reaches = bool(qr_dep) or bool(qr_only_out_symbolic)
    opcode_reaches = bool(opcode_dep) or bool(op_only_out_symbolic)
    can_discriminate = (qr_reaches != opcode_reaches)

    return {
        "granularity": "bit",
        "can_taint_single_bit": True,
        "output_ast": str(out)[:200],
        "per_bit_dependence": {
            "bit7_QR": bool(qr_dep),
            "bit6": bool(opcode_bits[6]),
            "bit5": bool(opcode_bits[5]),
            "bit4": bool(opcode_bits[4]),
            "bit3": bool(opcode_bits[3]),
            "bit2": bool(low_bits[2]),
            "bit1": bool(low_bits[1]),
            "bit0": bool(low_bits[0]),
        },
        "single_bit_symbolic_check": {
            "qr_only_symbolic_input": True,
            "qr_only_output_symbolic": bool(qr_only_out_symbolic),
            "opcode_only_symbolic_input": True,
            "opcode_only_output_symbolic": bool(op_only_out_symbolic),
        },
        "qr_reaches_output": qr_reaches,
        "opcode_reaches_output": opcode_reaches,
        "can_discriminate_qr_vs_opcode": bool(can_discriminate),
    }


# --------------------------------------------------------------------------- #
def main() -> int:
    pow_branch = _sym_addr(ELF, "pow_branch")
    pow_ct = _sym_addr(ELF, "pow_ct")

    vuln = run_ct_variant(pow_branch, "vuln")
    ct = run_ct_variant(pow_ct, "ct")

    dns = run_dns()
    vuln_leaks = vuln["secret_dep_branch_count"]
    ct_leaks = ct["secret_dep_branch_count"]
    matches_microtaint = (vuln_leaks >= 1 and ct_leaks == 0)

    out = {
        "tool": "angr",
        "family": "symbolic",
        "ct": {
            "vuln_branch_leaks": vuln_leaks,
            "ct_branch_leaks": ct_leaks,
            "matches_microtaint": bool(matches_microtaint),
        },
        "dns": {
            "granularity": dns["granularity"],
            "can_taint_single_bit": dns["can_taint_single_bit"],
            "qr_reaches_output": dns["qr_reaches_output"],
            "opcode_reaches_output": dns["opcode_reaches_output"],
            "can_discriminate_qr_vs_opcode": dns["can_discriminate_qr_vs_opcode"],
        },
        "raw": {"vuln": vuln, "ct": ct, "dns_full": dns},
    }

    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / "angr.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"[angr] CT vuln={vuln_leaks} ct={ct_leaks} (match={matches_microtaint}) | "
          f"DNS discriminate={dns['can_discriminate_qr_vs_opcode']} -> {p}")

    ok = (not vuln["error"]) and (not ct["error"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
