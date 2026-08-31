#!/usr/bin/env python3
"""
panda_ct_worker.py -- runs INSIDE the pandare/panda:latest container.

Full-system x86_64 PANDA + taint2 driver for the constant-time / control-flow
side-channel workload (crypto/square_and_multiply/test_constant_time.c).

Detection question (identical to every other tool in the comparison):
does attacker/secret taint reach the FLAGS that drive a conditional branch?

Method (mirrors libdft cf_leak.cpp: "latch source-operand taint at every
RFLAGS-writing instruction, check it at each conditional branch"):
  1. revert the bionic guest to its 'root' snapshot, copy the static harness
     in, run it with the 4-byte secret exponent on stdin;
  2. taint2 is byte-precise. At the entry of the target routine (after the
     prologue has spilled the 32-bit exponent argument ESI to its stack slot)
     we apply taint2 labels to those 4 RAM bytes -- the secret exponent (the
     stdin value 5). BASE and MOD stay untainted (RSA/DH threat model);
  3. instrument every instruction of the target routine (insn_translate gates
     the per-insn callback to the function PC range, so the rest of the guest
     runs uninstrumented). Using capstone we classify each instruction:
       * a FLAGS writer  -> latch = OR of taint over its source operands
                            (registers via taint2_query_reg, memory via
                            taint2 RAM query on the computed effective addr);
       * a conditional branch (Jcc) -> if the latch is set, the branch outcome
                            depends on the secret: record a control-flow leak.

Output: JSON blob on the last stdout line prefixed with 'RESULT_JSON: '.
argv: 1=harness path in /work, 2=argv1 ('vuln'|'ct'), 3=payload_b64 (stdin),
      4=arm_pc hex, 5=e_slot_disp (signed int, off rbp), 6=func_lo hex,
      7=func_hi hex.
"""
import base64
import json
import sys
import traceback

import capstone
from pandare import Panda

HARNESS = sys.argv[1]
ARGV1 = sys.argv[2]
PAYLOAD_B64 = sys.argv[3]
PAYLOAD = base64.b64decode(PAYLOAD_B64)
ARM_PC = int(sys.argv[4], 16)
E_DISP = int(sys.argv[5])          # signed displacement off RBP of the exponent
FUNC_LO = int(sys.argv[6], 16)
FUNC_HI = int(sys.argv[7], 16)

INVALID_PA = 0xFFFFFFFFFFFFFFFF

# capstone reg-name -> qemu/taint2 register index (env.regs order: A C D B ...)
_REG2IDX = {}
for idx, names in enumerate([
    ("RAX", "EAX", "AX", "AL", "AH"),
    ("RCX", "ECX", "CX", "CL", "CH"),
    ("RDX", "EDX", "DX", "DL", "DH"),
    ("RBX", "EBX", "BX", "BL", "BH"),
    ("RSP", "ESP", "SP", "SPL"),
    ("RBP", "EBP", "BP", "BPL"),
    ("RSI", "ESI", "SI", "SIL"),
    ("RDI", "EDI", "DI", "DIL"),
    ("R8", "R8D", "R8W", "R8B"),
    ("R9", "R9D", "R9W", "R9B"),
    ("R10", "R10D", "R10W", "R10B"),
    ("R11", "R11D", "R11W", "R11B"),
    ("R12", "R12D", "R12W", "R12B"),
    ("R13", "R13D", "R13W", "R13B"),
    ("R14", "R14D", "R14W", "R14B"),
    ("R15", "R15D", "R15W", "R15B"),
]):
    for n in names:
        _REG2IDX[n] = idx

state = {
    "armed": False,
    "labeled_bytes": 0,
    "decoded": 0,
    "cond_branches_total": 0,          # dynamic executions of any Jcc in range
    "leak_execs": 0,                   # Jcc executions with tainted latch
    "leak_sites": {},                  # hex(pc) -> count of tainted executions
    "site_exec": {},                   # hex(pc) -> total executions
    "error": "",
    "serial": "",
    "serial_timeout": False,
}

# Per-PC decoded metadata, filled lazily on first arm.
meta = {"cond": {}, "flag": {}}   # cond: pc->True ; flag: pc->(reg_idxs, mems)

md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
md.detail = True

panda = Panda(generic="x86_64")


def _decode_function(cpu):
    """Read the target routine from guest RAM and classify each instruction."""
    try:
        code = panda.virtual_memory_read(cpu, FUNC_LO, FUNC_HI - FUNC_LO)
    except Exception:
        state["error"] = "decode read: " + traceback.format_exc()
        return
    n = 0
    for insn in md.disasm(bytes(code), FUNC_LO):
        n += 1
        mnem = insn.mnemonic
        regs_read, regs_written = insn.regs_access()
        _fl = ("rflags", "eflags", "flags")
        writes_flags = any(md.reg_name(r) in _fl for r in regs_written)
        reads_flags = any(md.reg_name(r) in _fl for r in regs_read)
        is_jcc = mnem.startswith("j") and mnem != "jmp" and reads_flags
        if is_jcc:
            meta["cond"][insn.address] = True
            continue
        if writes_flags:
            reg_idxs = []
            for r in regs_read:
                nm = md.reg_name(r)
                if nm is None:
                    continue
                idx = _REG2IDX.get(nm.upper())
                if idx is not None:
                    reg_idxs.append(idx)
            mems = []
            for op in insn.operands:
                if op.type == capstone.x86.X86_OP_MEM:
                    base = md.reg_name(op.mem.base) if op.mem.base else None
                    index = md.reg_name(op.mem.index) if op.mem.index else None
                    mems.append((base.upper() if base else None,
                                 index.upper() if index else None,
                                 op.mem.scale, op.mem.disp, op.size))
            meta["flag"][insn.address] = (reg_idxs, mems)
    state["decoded"] = n


def _reg_tainted(idx):
    try:
        for off in range(8):
            if panda.plugins["taint2"].taint2_query_reg(idx, off) > 0:
                return True
    except Exception:
        pass
    return False


def _mem_tainted(cpu, base, index, scale, disp, size):
    try:
        ea = disp & 0xFFFFFFFFFFFFFFFF
        if base:
            ea += panda.arch.get_reg(cpu, base)
        if index and scale:
            ea += panda.arch.get_reg(cpu, index) * scale
        ea &= 0xFFFFFFFFFFFFFFFF
        for j in range(max(1, size)):
            pa = panda.virt_to_phys(cpu, ea + j)
            if pa in (INVALID_PA, 0xFFFFFFFF, 0):
                continue
            if panda.taint_check_ram(pa):
                return True
    except Exception:
        pass
    return False


@panda.cb_after_machine_init
def machine_init(cpu):
    panda.load_plugin("taint2")
    panda.enable_precise_pc()


@panda.cb_insn_translate
def should_instrument(cpu, pc):
    return FUNC_LO <= pc < FUNC_HI


latch = {"tainted": False}


@panda.cb_insn_exec
def on_insn(cpu, pc):
    try:
        if not state["armed"]:
            if pc != ARM_PC:
                return 0
            if not panda.taint_enabled():
                panda.taint_enable()
            rbp = panda.arch.get_reg(cpu, "RBP")
            base = (rbp + E_DISP) & 0xFFFFFFFFFFFFFFFF
            n = 0
            for off in range(4):
                pa = panda.virt_to_phys(cpu, base + off)
                if pa in (INVALID_PA, 0xFFFFFFFF, 0):
                    continue
                panda.taint_label_ram(pa, off)
                n += 1
            state["labeled_bytes"] = n
            state["armed"] = True
            _decode_function(cpu)
            return 0

        if pc in meta["cond"]:
            state["cond_branches_total"] += 1
            k = hex(pc)
            state["site_exec"][k] = state["site_exec"].get(k, 0) + 1
            if latch["tainted"]:
                state["leak_execs"] += 1
                state["leak_sites"][k] = state["leak_sites"].get(k, 0) + 1
            return 0

        fm = meta["flag"].get(pc)
        if fm is not None:
            reg_idxs, mems = fm
            t = any(_reg_tainted(i) for i in reg_idxs)
            if not t:
                for (b, ix, sc, ds, sz) in mems:
                    if _mem_tainted(cpu, b, ix, sc, ds, sz):
                        t = True
                        break
            latch["tainted"] = t
    except Exception:
        if not state["error"]:
            state["error"] = "insn: " + traceback.format_exc()
    return 0


@panda.queue_blocking
def driver():
    try:
        import os
        import shutil
        panda.revert_sync("root")
        share = "/tmp/hshare"
        if os.path.isdir(share):
            shutil.rmtree(share)
        os.makedirs(share)
        shutil.copy("/work/" + HARNESS, share + "/h")
        panda.copy_to_guest(share)
        panda.run_serial_cmd("chmod +x /root/hshare/h", timeout=60)
        panda.run_serial_cmd(
            "echo " + PAYLOAD_B64 + " | base64 -d > /root/payload", timeout=60)
        try:
            out = panda.run_serial_cmd(
                "cd /root && ./hshare/h " + ARGV1 + " < payload | xxd",
                timeout=600)
            state["serial"] = out
        except Exception as exc:
            state["serial"] = "timeout: " + str(exc)
            state["serial_timeout"] = True
    except Exception:
        state["error"] = "driver: " + traceback.format_exc()
    finally:
        panda.end_analysis()


panda.run()
print("RESULT_JSON: " + json.dumps(state))
