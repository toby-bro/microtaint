#!/usr/bin/env python3
"""Maat baseline for the DNS bit-field workload.

OPCODE is extracted by `AND al,0x78 ; SHR al,3`; QR is bit 7, masked away. Maat
is bit-granular: build AL as a Concat of eight 1-bit terms, making only the
tested bits symbolic. Taint QR (bit7) alone, then OPCODE (bits6..3) alone, and
test by dual concretization (tainted bits set to 0 vs 1) whether the output byte
changes. Ground truth: QR -> output clean, OPCODE -> output tainted; Maat
separates the two fields, which byte-granular engines cannot.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.dup2(2, 1)  # keep Maat's ANSI logger off stdout

from maat import (ARCH, Concat, Cst, Extract, MaatEngine, OS, PERM, Var)

AND_AL_78 = bytes.fromhex("2478")     # and al, 0x78
SHR_AL_3 = bytes.fromhex("c0e803")    # shr al, 3
CODE = AND_AL_78 + SHR_AL_3
BASE = 0x400000
# Flag byte: QR(bit7)=1, OPCODE(bits6..3)=0b0101=5, low bits 0 -> 0xA8.
FLAG = 0xA8


def _build_byte(tainted_bits: set[int], name: str):
    """Concat (MSB first) of eight 1-bit terms; tainted bits are Var(1)."""
    pieces = []
    var_names = []
    for bit in range(7, -1, -1):
        if bit in tainted_bits:
            vn = f"{name}_b{bit}"
            pieces.append(Var(1, vn))
            var_names.append(vn)
        else:
            pieces.append(Cst(1, (FLAG >> bit) & 1))
    expr = pieces[0]
    for p in pieces[1:]:
        expr = Concat(expr, p)
    return expr, var_names


def run(tainted_bits: set[int], name: str) -> dict:
    engine = MaatEngine(ARCH.X64, OS.LINUX)
    engine.mem.map(BASE, BASE + 0x1000, PERM.RWX)
    engine.mem.write(BASE, CODE, len(CODE))
    engine.cpu.rip = BASE

    byte, var_names = _build_byte(tainted_bits, name)
    # AL is the low byte of RAX; keep the upper 56 bits concrete 0.
    engine.cpu.rax = Concat(Cst(56, 0), byte)
    for vn in var_names:
        engine.vars.set(vn, (FLAG >> int(vn.split("_b")[1])) & 1)

    engine.run(2)  # AND then SHR

    al = Extract(engine.cpu.rax, 7, 0)

    # Dual concretization: does the output byte depend on the tainted bit(s)?
    for vn in var_names:
        engine.vars.set(vn, 0)
    out_zero = al.as_uint(engine.vars)
    for vn in var_names:
        engine.vars.set(vn, 1)
    out_one = al.as_uint(engine.vars)
    depends = out_zero != out_one

    return {
        "config": name,
        "tainted_bits": sorted(tainted_bits),
        "n_tainted_bits": len(var_names),
        "flag_in": hex(FLAG),
        "out_al_inputs0": hex(out_zero),
        "out_al_inputs1": hex(out_one),
        "output_depends_on_tainted_bits": depends,
    }


def main() -> int:
    qr = run({7}, "QR")          # taint only QR bit7 (benign)
    op = run({6, 5, 4, 3}, "OPCODE")  # taint only OPCODE bits6..3

    can_taint_single_bit = qr["n_tainted_bits"] == 1
    qr_reaches = qr["output_depends_on_tainted_bits"]
    op_reaches = op["output_depends_on_tainted_bits"]
    can_discriminate = qr_reaches != op_reaches

    print(f"[maat DNS] saw: OPCODE taints output={op_reaches}, single-bit taint "
          f"(discriminates QR/OPCODE={can_discriminate}) | not: QR taints "
          f"output={qr_reaches} (masked by AND 0x78)")

    out = {
        "qr": qr,
        "opcode": op,
        "can_taint_single_bit": can_taint_single_bit,
        "qr_reaches_output": qr_reaches,
        "opcode_reaches_output": op_reaches,
        "can_discriminate_qr_vs_opcode": can_discriminate,
    }
    Path(__file__).resolve().parent.joinpath(
        "results", "_maat_dns_raw.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
