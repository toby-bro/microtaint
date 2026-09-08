#!/usr/bin/env python3
"""Triton baseline for the DNS bit-field workload.

OPCODE is extracted from the flag byte by `AND al,0x78 ; SHR al,3`. QR is bit 7,
masked away by the AND. Ground truth: taint QR only -> output clean; taint OPCODE
-> output tainted. Triton taints a whole byte (no bit API), so "QR only" and
"OPCODE only" are the same input and give the same output: it cannot separate the
two fields (a false positive on QR).
"""
from __future__ import annotations

import json
from pathlib import Path

from triton import ARCH, Instruction, TritonContext

BASE = 0x400000
AND_AL_78 = bytes.fromhex("2478")      # and al, 0x78
SHR_AL_3 = bytes.fromhex("c0e803")     # shr al, 3
FLAGS = 0xA8  # 1010_1000 : QR(bit7)=1, OPCODE(bits6..3)=0b0101=5, rest 0


def run(al_value: int, taint_al: bool, label: str) -> dict:
    ctx = TritonContext(ARCH.X86_64)
    ctx.setConcreteRegisterValue(ctx.registers.rax, 0)
    ctx.setConcreteRegisterValue(ctx.registers.al, al_value)
    if taint_al:
        ctx.taintRegister(ctx.registers.al)

    in_tainted = ctx.isRegisterTainted(ctx.registers.al)

    pc = BASE
    trace = []
    for raw in (AND_AL_78, SHR_AL_3):
        inst = Instruction(pc, raw)
        ctx.processing(inst)
        trace.append({
            "pc": hex(pc),
            "insn": inst.getDisassembly(),
            "al_after": hex(ctx.getConcreteRegisterValue(ctx.registers.al)),
            "al_tainted_after": bool(ctx.isRegisterTainted(ctx.registers.al)),
        })
        pc += inst.getSize()

    out_val = ctx.getConcreteRegisterValue(ctx.registers.al)
    out_tainted = ctx.isRegisterTainted(ctx.registers.al)
    return {
        "label": label,
        "al_in": hex(al_value),
        "input_al_tainted": bool(in_tainted),
        "trace": trace,
        "output_al": hex(out_val),
        "output_al_tainted": bool(out_tainted),
    }


def probe_subbit_api() -> dict:
    """Confirm Triton's finest taint unit is a byte (no bit-level API)."""
    ctx = TritonContext(ARCH.X86_64)
    facts = {}
    # Smallest register you can name is an 8-bit one (al/ah); no bit selector.
    ctx.taintRegister(ctx.registers.al)
    facts["taint_al_taints_whole_al"] = bool(ctx.isRegisterTainted(ctx.registers.al))
    # Tainting AL taints the byte; there is no ctx.taintRegisterBit / bit index.
    facts["has_bit_level_taint_api"] = any(
        hasattr(ctx, n) for n in
        ("taintRegisterBit", "taintBit", "taintRegisterBits", "setTaintBit")
    )
    facts["taint_api_methods"] = sorted(
        n for n in dir(ctx) if "taint" in n.lower()
    )
    return facts


def main() -> int:
    # Config 1: intent "taint QR (bit7) only" -- Triton can only taint whole AL.
    qr = run(FLAGS, taint_al=True, label="intent=QR-bit7-only (Triton taints whole AL)")
    # Config 2: intent "taint OPCODE (bits6..3) only" -- again whole AL.
    op = run(FLAGS, taint_al=True, label="intent=OPCODE-bits6..3-only (Triton taints whole AL)")
    # Control: input untainted -> output must be clean (pipeline sanity).
    clean = run(FLAGS, taint_al=False, label="control: AL untainted")

    api = probe_subbit_api()
    can_taint_single_bit = api["has_bit_level_taint_api"]
    # Discriminate QR vs OPCODE iff the two attribution policies can yield
    # different output-taint verdicts.  They cannot: identical whole-byte input.
    can_discriminate = (qr["output_al_tainted"] != op["output_al_tainted"])

    print(f"[triton] DNS QR_out_tainted={qr['output_al_tainted']} OPCODE_out_tainted={op['output_al_tainted']} "
          f"single_bit={can_taint_single_bit} discriminate={can_discriminate}")

    out = {
        "qr_intent": qr, "opcode_intent": op, "control_untainted": clean,
        "api": api,
        "can_taint_single_bit": can_taint_single_bit,
        "can_discriminate_qr_vs_opcode": can_discriminate,
        "byte_taint_output_al": qr["output_al"],
    }
    Path(__file__).resolve().parent.joinpath("results", "_dns_raw.json").write_text(
        json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    main()
