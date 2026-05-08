#!/usr/bin/env python3
"""
worker_triton.py — persistent daemon mode.

Protocol (stdin/stdout, line-delimited JSON):
  ← {"arch": ..., "bytes": ..., "state": {...}, "taint": {...}}
  → {"output_taint": {...}, "time_ns": N}
  ← "QUIT"  → exits cleanly

Multi-instruction sequences
---------------------------
Triton's ctx.processing(inst) decodes and executes exactly ONE instruction
per call, stopping at the first opcode boundary.  For multi-instruction
sequences (mode == "sequence", "realworld", "bugdetect", "path_explosion",
"oracle") we must call it once per instruction, advancing the address each
time.  We detect this by checking whether the test case carries an
"asm_lines" list (more than one entry) or by noting that the concatenated
bytes are longer than a single decoded instruction.

The approach: set PC to BASE, then call ctx.processing() in a loop,
incrementing the address by each instruction's size (from getSize()) until
we've consumed all the bytes.
"""
import json
import sys
import time

from triton import ARCH, Instruction, TritonContext

_ARCH_MAP = {"x86": ARCH.X86, "x86_64": ARCH.X86_64}
BASE_ADDR = 0x400000


def run_one(tc: dict) -> dict:
    ctx = TritonContext(_ARCH_MAP[tc['arch']])
    regs = list(tc['state'].keys())

    # Set concrete register values and taint
    for reg_name, val in tc['state'].items():
        reg = getattr(ctx.registers, reg_name.lower())
        ctx.setConcreteRegisterValue(reg, val)
        if tc['taint'].get(reg_name, 0):
            ctx.taintRegister(reg)

    raw_bytes = bytes.fromhex(tc['bytes'])
    pc = BASE_ADDR

    t0 = time.process_time_ns()

    # x86-64 has an architectural limit of 15 bytes per instruction.
    # libtriton's Instruction(pc, buf) constructor enforces a 16-byte
    # ceiling on `buf` and raises TypeError("Invalid size (too big)")
    # for anything larger.  Bytestrings in the benchmark routinely
    # exceed 16 bytes (sequence-mode tests average 17, path_explosion
    # reaches 183), so we MUST slice the input to a single-instruction
    # window before passing it.  16 is the safe upper bound.
    MAX_INSTR_BYTES = 16

    offset = 0
    while offset < len(raw_bytes):
        # Build an Instruction from at most 16 bytes at current offset.
        # Triton decodes only the first instruction in the buffer; the
        # ceiling protects libtriton's internal opcode buffer.
        chunk = raw_bytes[offset:offset + MAX_INSTR_BYTES]
        inst = Instruction(pc, chunk)
        ok = ctx.processing(inst)
        if not ok or inst.getSize() == 0:
            # Cannot decode — stop here (e.g. UD2 terminator or unsupported opcode)
            break
        offset += inst.getSize()
        pc += inst.getSize()

    t1 = time.process_time_ns()

    output_taint = {
        reg_name: (1 if ctx.isRegisterTainted(getattr(ctx.registers, reg_name.lower())) else 0) for reg_name in regs
    }
    return {"output_taint": output_taint, "time_ns": t1 - t0}


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
