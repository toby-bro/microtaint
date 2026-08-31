#!/usr/bin/env python3
"""
panda_worker.py -- runs INSIDE the pandare/panda:latest container.

Full-system x86_64 PANDA + taint2 driver for the nft_byteorder OOB write.

Method
------
The static -no-pie harness lives at fixed virtual addresses (guest is
no-ASLR/no-KASLR). We:
  1. revert the bionic guest to its 'root' snapshot,
  2. copy the harness in and run it with the trigger payload on stdin,
  3. hook the entry of nft_byteorder_eval (fixed PC); at that point the 80
     attacker-controlled register bytes have already been read from stdin
     into the register file, so we apply byte-positional taint2 labels to
     that RAM (this is the attacker source, "color A"),
  4. on every guest virtual memory WRITE, if the destination lands in the
     canary_after sentinel region [regs+80, regs+80+256) we query taint2 on
     the freshly-written RAM bytes and record whether the stored value is
     tainted (byte-precise forward data-flow).

We also record in-bounds byteswap writes into the register file as a taint
sanity check (those MUST be tainted, proving taint2 is propagating through
the ntohs/htons byteswap at all).

Output: a JSON blob on the last stdout line prefixed with 'RESULT_JSON: '.
Args: argv[1] = path to harness inside /work, argv[2] = base64 payload,
      argv[3] = eval_pc hex, argv[4] = regs_lo hex, argv[5] = canary_lo hex.
"""
import base64
import json
import sys
import traceback

from pandare import Panda

HARNESS = sys.argv[1]
PAYLOAD = base64.b64decode(sys.argv[2])
EVAL_PC = int(sys.argv[3], 16)
REGS_LO = int(sys.argv[4], 16)
REGS_HI = REGS_LO + 80
CANARY_LO = int(sys.argv[5], 16)
CANARY_HI = CANARY_LO + 256
PAYLOAD_B64 = sys.argv[2]

INVALID_PA = 0xFFFFFFFFFFFFFFFF

state = {
    "armed": False,
    "labeled_bytes": 0,
    "oob": [],                 # writes into canary_after
    "inbounds_total": 0,       # byteswap writes into the register file
    "inbounds_tainted": 0,
    "serial": "",
    "serial_timeout": False,
    "error": "",
}

panda = Panda(generic="x86_64")


@panda.cb_after_machine_init
def machine_init(cpu):
    panda.load_plugin("taint2")
    panda.enable_precise_pc()
    panda.enable_memcb()


@panda.hook(EVAL_PC, kernel=False)
def on_eval_entry(cpu, tb, h):
    # Fires when a user-mode block starting at nft_byteorder_eval executes.
    if state["armed"]:
        return
    try:
        if not panda.taint_enabled():
            panda.taint_enable()
        n = 0
        for off in range(REGS_HI - REGS_LO):
            va = REGS_LO + off
            pa = panda.virt_to_phys(cpu, va)
            if pa == INVALID_PA or pa == 0xFFFFFFFF or pa == 0:
                continue
            # positional label: byte `off` of the register file
            panda.taint_label_ram(pa, off)
            n += 1
        state["labeled_bytes"] = n
        state["armed"] = True
    except Exception:
        state["error"] = "hook: " + traceback.format_exc()


@panda.cb_virt_mem_after_write
def on_write(cpu, pc, addr, size, buf):
    if not state["armed"]:
        return
    try:
        # in-bounds byteswap writes -> taint-propagation sanity check
        if REGS_LO <= addr < REGS_HI:
            state["inbounds_total"] += 1
            pa = panda.virt_to_phys(cpu, addr)
            if pa not in (INVALID_PA, 0xFFFFFFFF, 0) and panda.taint_check_ram(pa):
                state["inbounds_tainted"] += 1
            return

        # does this write overlap the canary_after sentinel region?
        if addr >= CANARY_HI or addr + size <= CANARY_LO:
            return

        tainted_bytes = 0
        labels = set()
        for j in range(size):
            va = addr + j
            if not (CANARY_LO <= va < CANARY_HI):
                continue
            pa = panda.virt_to_phys(cpu, va)
            if pa in (INVALID_PA, 0xFFFFFFFF, 0):
                continue
            if panda.taint_check_ram(pa):
                tainted_bytes += 1
                tq = panda.taint_get_ram(pa)
                if tq is not None:
                    try:
                        labels.update(tq.get_labels())
                    except Exception:
                        pass
        try:
            val = panda.virtual_memory_read(cpu, addr, size, fmt="int")
        except Exception:
            val = None
        state["oob"].append({
            "pc": pc,
            "addr": addr,
            "size": size,
            "value": val,
            "tainted_bytes": tainted_bytes,
            "labels": sorted(labels),
        })
    except Exception:
        if not state["error"]:
            state["error"] = "write: " + traceback.format_exc()


@panda.queue_blocking
def driver():
    try:
        import os
        import shutil
        panda.revert_sync("root")
        # The guest VM cannot see the container's filesystem; copy_to_guest
        # builds an ISO of a directory and mounts+copies it inside the guest.
        share = "/tmp/hshare"
        if os.path.isdir(share):
            shutil.rmtree(share)
        os.makedirs(share)
        shutil.copy("/work/" + HARNESS, share + "/h")
        panda.copy_to_guest(share)  # -> guest ~/hshare/h
        panda.run_serial_cmd("chmod +x /root/hshare/h", timeout=60)
        panda.run_serial_cmd(
            "echo " + PAYLOAD_B64 + " | base64 -d > /root/payload", timeout=60)
        # taint2 slows the guest ~100x; the harness completes (all writes are
        # captured by the callbacks) but the post-command shell prompt can be
        # slow enough to trip run_serial_cmd's expect() timeout. That is a
        # cosmetic sync artifact -- salvage the serial buffer either way.
        try:
            out = panda.run_serial_cmd(
                "cd /root && ./hshare/h < payload", timeout=600)
            state["serial"] = out
        except Exception as exc:
            sc = getattr(panda, "serial_console", None)
            try:
                lines = list(getattr(sc, "prior_lines", []))
                cur = getattr(sc, "current_line", b"")
                cur = bytes(cur).decode("latin-1") if isinstance(cur, (bytes, bytearray)) else str(cur)
                state["serial"] = "\n".join([str(x) for x in lines] + [cur])
            except Exception:
                state["serial"] = str(exc)
            state["serial_timeout"] = True
    except Exception:
        state["error"] = "driver: " + traceback.format_exc()
    finally:
        panda.end_analysis()


panda.run()

print("RESULT_JSON: " + json.dumps(state))
