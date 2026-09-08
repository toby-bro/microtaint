#!/usr/bin/env python3
"""
worker_maat.py — persistent daemon mode.

Protocol (stdin/stdout, line-delimited JSON):
  ← {"arch": ..., "bytes": ..., "state": {...}, "taint": {...}}
  → {"output_taint": {...}, "time_ns": N}
  ← "QUIT"  → exits cleanly

Taint extraction method: dual concretization.
See original worker_maat.py for full explanation.

IMPORTANT: maat's C++ logger writes coloured ANSI output to stdout by
default, which corrupts the line-delimited JSON protocol.  We redirect
stdout to stderr at the OS level before importing maat so the logger's
output never reaches the orchestrator.
"""
import json
import os
import sys
import time
import traceback as _tb

# Redirect maat's stdout-going logger to stderr BEFORE importing maat.
# We dup the real stdout fd, redirect fd 1 → fd 2, import, then restore.
_real_stdout_fd = os.dup(1)
os.dup2(2, 1)  # fd 1 now points at stderr (maat logger goes here)

from maat import ARCH, OS, PERM, Concat, Cst, MaatEngine, Var

_out = os.fdopen(_real_stdout_fd, "w", 1)  # use the saved fd directly


def _emit(obj: dict) -> None:
    _out.write(json.dumps(obj) + "\n")
    _out.flush()


_ARCH_MAP = {"x86": ARCH.X86, "x86_64": ARCH.X64}


def _reg_bits(arch: str) -> int:
    return 32 if arch == "x86" else 64


def _ud_suffix(arch: str) -> bytes:
    return b"\x0f\x0b" if "x86" in arch else b""


def _build_symbolic_value(val_int: int, taint_mask: int, name: str, reg_bits: int):
    if taint_mask == 0:
        return val_int, {}
    if taint_mask == (1 << reg_bits) - 1:
        return Var(reg_bits, name), {name: reg_bits}
    result = None
    var_names = {}
    for bit in range(reg_bits - 1, -1, -1):
        if (taint_mask >> bit) & 1:
            vname = f"{name}_b{bit}"
            piece = Var(1, vname)
            var_names[vname] = 1
        else:
            piece = Cst(1, (val_int >> bit) & 1)
        result = piece if result is None else Concat(result, piece)
    return result, var_names


def _setup_registers(engine, tc: dict, reg_bits: int, call_id: int) -> dict[str, dict[str, int]]:
    # Return per-register var maps, not a flat merged dict
    per_reg_vars: dict[str, dict[str, int]] = {}
    for reg_name, val in tc["state"].items():
        val_int = int(val, 16) if isinstance(val, str) else int(val)
        raw_mask = tc["taint"].get(reg_name, 0)
        mask = int(raw_mask, 16) if isinstance(raw_mask, str) else int(raw_mask)
        sym_val, var_names = _build_symbolic_value(val_int, mask, f"taint_{reg_name}_{call_id}", reg_bits)
        per_reg_vars[reg_name] = var_names  # isolated per register
        setattr(engine.cpu, reg_name.lower(), sym_val)
    return per_reg_vars


def _extract_taint_dual(engine, reg_name: str, reg_bits: int, per_reg_vars: dict) -> int:
    # Only the vars that were introduced for THIS query register
    # — but we need to flip ALL input vars to detect propagation correctly.
    # Use all_vars = union of all per_reg_vars values.
    all_vars = {k: v for d in per_reg_vars.values() for k, v in d.items()}
    try:
        val = getattr(engine.cpu, reg_name.lower())
        if val.is_concrete(engine.vars):
            return 0
        if not all_vars:
            return (1 << reg_bits) - 1
        ctx = engine.vars

        # Concretize all inputs to 0
        for vname, vbits in all_vars.items():
            ctx.set(vname, 0)
        try:
            concrete_0 = val.as_uint(ctx)
        except Exception:
            _cleanup_vars(ctx, all_vars)
            return (1 << reg_bits) - 1

        # Concretize all inputs to all-ones
        for vname, vbits in all_vars.items():
            ctx.set(vname, (1 << vbits) - 1)
        try:
            concrete_1 = val.as_uint(ctx)
        except Exception:
            _cleanup_vars(ctx, all_vars)
            return (1 << reg_bits) - 1

        _cleanup_vars(ctx, all_vars)
        return (concrete_0 ^ concrete_1) & ((1 << reg_bits) - 1)
    except Exception:
        return 0


def _cleanup_vars(ctx, all_vars):
    for vname in all_vars:
        try:
            ctx.remove(vname)
        except Exception:
            pass


_call_id = 0


def run_one(tc: dict) -> dict:
    global _call_id
    _call_id += 1
    arch = tc["arch"]
    reg_bits = _reg_bits(arch)
    engine = MaatEngine(_ARCH_MAP[arch], OS.LINUX)
    BASE = 0x400000
    engine.mem.map(BASE, BASE + 0x1000, PERM.RWX)
    raw = bytes.fromhex(tc["bytes"]) + _ud_suffix(arch)
    engine.mem.write(BASE, raw, len(raw))
    pc = "eip" if arch == "x86" else "rip"
    setattr(engine.cpu, pc, BASE)
    per_reg_vars = _setup_registers(engine, tc, reg_bits, _call_id)
    t0 = time.process_time_ns()
    try:
        engine.run(1)
    except Exception:
        pass
    t1 = time.process_time_ns()

    output_taint = {reg: _extract_taint_dual(engine, reg, reg_bits, per_reg_vars) for reg in tc["state"]}
    return {"output_taint": output_taint, "time_ns": t1 - t0}


def main():
    _out.write("READY\n")
    _out.flush()

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        if line == "QUIT":
            break
        try:
            tc = json.loads(line)
            result = run_one(tc)
        except Exception:
            result = {"error": _tb.format_exc()[:400], "time_ns": 0}
        # Always emit exactly one JSON line — never let an exception skip this
        try:
            _emit(result)
        except Exception:
            pass  # if stdout is broken, nothing we can do


if __name__ == "__main__":
    main()
