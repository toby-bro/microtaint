#!/usr/bin/env python3
"""
panda_dns_worker.py -- runs INSIDE the pandare/panda:latest container.

Full-system x86_64 PANDA + taint2 driver for the DNS OPCODE bit-field probe
(dns_bitfield.c). Reproduces the LDNS_OPCODE_WIRE extract on the RFC1035
flag byte:
    AND AL, 0x78   (24 78)     keep bits 6..3 (OPCODE), drop QR (bit7)
    SHR AL, 3      (C0 E8 03)  right-align OPCODE into bits 3..0

The flag byte packs QR at bit7 (benign, masked away by AND 0x78) and OPCODE
at bits6..3 (security-relevant). taint2 is BYTE-precise: the finest source it
can label is the whole flag byte. We taint that one byte, execute the two
instructions, and observe:
  * whether AL stays tainted after AND and after SHR,
  * whether the stored output byte `out` is tainted.
Because taint2 cannot label a single bit, it cannot express "QR only" vs
"OPCODE only": both collapse to the whole-byte label and the output is
reported tainted regardless -- the same byte-granularity false positive as
libdft/taintgrind/Triton under a QR-attribution policy.

argv: 1=harness path in /work, 2=payload_b64 (1 stdin flag byte),
      3=arm_pc hex (after read()), 4=flag_slot_disp (signed, off rbp),
      5=and_pc hex, 6=shr_pc hex, 7=store_pc hex, 8=out_addr hex,
      9=func_lo hex, 10=func_hi hex.
"""
import base64
import json
import sys
import traceback

from pandare import Panda

HARNESS = sys.argv[1]
PAYLOAD_B64 = sys.argv[2]
PAYLOAD = base64.b64decode(PAYLOAD_B64)
ARM_PC = int(sys.argv[3], 16)
FLAG_DISP = int(sys.argv[4])
AND_PC = int(sys.argv[5], 16)
SHR_PC = int(sys.argv[6], 16)
STORE_PC = int(sys.argv[7], 16)
OUT_ADDR = int(sys.argv[8], 16)
FUNC_LO = int(sys.argv[9], 16)
FUNC_HI = int(sys.argv[10], 16)

INVALID_PA = 0xFFFFFFFFFFFFFFFF

state = {
    "armed": False,
    "labeled_bytes": 0,
    "input_flag": PAYLOAD[0] if PAYLOAD else None,
    "al_tainted_before_and": None,
    "al_tainted_after_and": None,
    "al_tainted_after_shr": None,
    "out_tainted": None,
    "out_value": None,
    "error": "",
    "serial": "",
    "serial_timeout": False,
}

panda = Panda(generic="x86_64")


def _al_tainted():
    # AL is byte 0 of RAX (reg index 0).
    try:
        return panda.plugins["taint2"].taint2_query_reg(0, 0) > 0
    except Exception:
        return None


def _ram_tainted(cpu, va, size):
    t = False
    for j in range(size):
        try:
            pa = panda.virt_to_phys(cpu, va + j)
            if pa in (INVALID_PA, 0xFFFFFFFF, 0):
                continue
            if panda.taint_check_ram(pa):
                t = True
        except Exception:
            pass
    return t


@panda.cb_after_machine_init
def machine_init(cpu):
    panda.load_plugin("taint2")
    panda.enable_precise_pc()


@panda.cb_insn_translate
def should_instrument(cpu, pc):
    return FUNC_LO <= pc < FUNC_HI


@panda.cb_insn_exec
def on_insn(cpu, pc):
    try:
        if not state["armed"]:
            if pc != ARM_PC:
                return 0
            if not panda.taint_enabled():
                panda.taint_enable()
            rbp = panda.arch.get_reg(cpu, "RBP")
            va = (rbp + FLAG_DISP) & 0xFFFFFFFFFFFFFFFF
            pa = panda.virt_to_phys(cpu, va)
            if pa not in (INVALID_PA, 0xFFFFFFFF, 0):
                panda.taint_label_ram(pa, 0)
                state["labeled_bytes"] = 1
            state["armed"] = True
            return 0

        if pc == AND_PC:
            state["al_tainted_before_and"] = _al_tainted()
        elif pc == SHR_PC:
            state["al_tainted_after_and"] = _al_tainted()
        elif pc == STORE_PC:
            state["al_tainted_after_shr"] = _al_tainted()
    except Exception:
        if not state["error"]:
            state["error"] = "insn: " + traceback.format_exc()
    return 0


@panda.cb_after_block_exec
def after_block(cpu, tb, exit_code):
    # Once the store to `out` has executed, sample the output byte's taint.
    try:
        if state["armed"] and state["out_tainted"] is None \
                and state["al_tainted_after_shr"] is not None:
            state["out_tainted"] = _ram_tainted(cpu, OUT_ADDR, 1)
            try:
                state["out_value"] = panda.virtual_memory_read(
                    cpu, OUT_ADDR, 1, fmt="int")
            except Exception:
                pass
    except Exception:
        if not state["error"]:
            state["error"] = "afterblk: " + traceback.format_exc()


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
                "cd /root && ./hshare/h < payload | xxd", timeout=600)
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
