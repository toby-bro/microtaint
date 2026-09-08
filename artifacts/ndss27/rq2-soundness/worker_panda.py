#!/usr/bin/env python3
"""
worker_panda.py  –  Persistent PANDA/taint2 worker, register-level output.
Runs INSIDE pandare/panda container. No arguments.

Protocol
--------
  stdout → "READY\n"
  stdin  ← JSON test-case line | "QUIT\n"
  stdout → JSON result line per case
            {"output_taint": {"RAX": 0|1, ...}, "time_ns": N}

Granularity note
----------------
taint_label_reg(reg, label) applies one label to all bytes of a register.
With a single label, cb_mask always returns all-ones for tainted registers —
it cannot distinguish which bits are affected. True bit-level output would
require per-byte distinct labels via the lower-level C API not exposed here.
We therefore report 0 (clean) or 1 (tainted) per register, which is correct.

TCG cache fix
-------------
QEMU's TCG (Tiny Code Generator) JIT-compiles guest basic blocks and caches
the translations keyed by guest physical address. If two successive tests
write different byte sequences to the same physical address, the second test
will execute the FIRST test's stale cached translation — producing wrong
results (typically a 0ms "instant" result that matches the previous output).

Fix: each test executes at a DIFFERENT physical address within a pre-mapped
1 MB region. QEMU has never translated the new address, so it must JIT-compile
the new bytes fresh. The address advances by one page (0x1000 = 4096 bytes)
per test, which is always larger than any sequence we emit (max ~500 bytes
for N=32). After 256 tests the counter wraps — the oldest cached translation
is almost certainly evicted from QEMU's finite-size TB cache by then.

Execution region
----------------
panda.map_memory() creates a QEMU-level RWX region at a physical address
that TCG can always execute from, bypassing guest page tables. We map 1 MB
at EXEC_PA_BASE to accommodate the rotating addresses.
"""

import os
import sys

_real_stdout_fd = os.dup(1)
os.dup2(2, 1)
_json_out = os.fdopen(_real_stdout_fd, "w", 1)

import json
import threading
import time

from pandare import Panda

R_EAX, R_ECX, R_EDX, R_EBX = 0, 1, 2, 3
REG_MAP = {"RAX": R_EAX, "RBX": R_EBX, "RCX": R_ECX, "RDX": R_EDX}

RET = b"\xc3"
EXEC_PA_BASE = 0x4000000  # base physical address of the execution region
EXEC_REGION = 0x100000  # 1 MB — enough for 256 page-strided tests
EXEC_STRIDE = 0x1000  # one page per test — always larger than any sequence
EXEC_NAME = "taint_exec"

# How many distinct addresses before we wrap around.
# QEMU's TB cache typically holds ~65536 entries; after WRAP_AFTER tests the
# oldest translations are very likely evicted, making wrap-around safe.
WRAP_AFTER = EXEC_REGION // EXEC_STRIDE  # = 256


def emit(obj):
    _json_out.write(json.dumps(obj) + "\n")
    _json_out.flush()


def main():
    panda = Panda(generic="x86_64")
    ev_run = threading.Event()
    ev_done = threading.Event()
    shared = {
        "test_case": None,
        "result": None,
        "quit": False,
        "sc_end": None,
        "exec_pa": EXEC_PA_BASE,  # current execution address (rotates per test)
        "t_start": 0,
    }

    @panda.cb_after_machine_init
    def on_init(cpu):
        # Map the full 1 MB region once — all per-test offsets live inside it.
        panda.map_memory(EXEC_NAME, EXEC_REGION, EXEC_PA_BASE)
        panda.taint_enable()

    @panda.cb_after_block_exec
    def after_block(cpu, tb, exit_code):
        sc_end = shared["sc_end"]
        if sc_end is None or shared["result"] is not None:
            return
        # Collect results when the PC has advanced past the sequence bytes
        # (i.e. it has reached or passed the RET instruction at sc_end).
        if panda.arch.get_pc(cpu) < sc_end:
            return
        elapsed = time.perf_counter_ns() - shared["t_start"]
        output = {n: (1 if panda.taint_check_reg(i) else 0) for n, i in REG_MAP.items()}
        shared["result"] = {"output_taint": output, "time_ns": elapsed}
        shared["sc_end"] = None
        ev_done.set()

    # Counter lives in a mutable container so the closure can update it.
    _test_counter = [0]

    @panda.queue_blocking
    def panda_loop():
        panda.revert_sync("root")

        _json_out.write("READY\n")
        _json_out.flush()

        while True:
            ev_run.wait()
            ev_run.clear()
            if shared["quit"]:
                panda.end_analysis()
                return

            tc = shared["test_case"]
            try:
                instr = bytes.fromhex(tc["bytes"])
                sc = instr + RET

                # ── Rotate execution address ──────────────────────────────
                # Each test gets a fresh physical address that QEMU's TCG
                # has never translated, eliminating stale-cache results.
                slot = _test_counter[0] % WRAP_AFTER
                exec_pa = EXEC_PA_BASE + slot * EXEC_STRIDE
                sc_end = exec_pa + len(instr)  # RET is at this address
                _test_counter[0] += 1

                # Validate the sequence fits in one stride.
                assert len(sc) <= EXEC_STRIDE, f"Sequence too long ({len(sc)} bytes) for stride {EXEC_STRIDE}"

                panda.revert_sync("root")
                cpu = panda.get_cpu()

                # Write the new bytes to the fresh address.
                panda.physical_memory_write(exec_pa, sc)

                # Set register state.
                for name in REG_MAP:
                    panda.arch.set_reg(cpu, name, tc["state"].get(name, 0))
                panda.arch.set_pc(cpu, exec_pa)

                # Apply taint AFTER revert (revert wipes all taint state).
                for name, idx in REG_MAP.items():
                    if tc["taint"].get(name, 0) != 0:
                        panda.taint_label_reg(idx, idx + 1)

                shared["sc_end"] = sc_end
                shared["t_start"] = time.perf_counter_ns()

            except Exception:
                import traceback

                shared["result"] = {"error": traceback.format_exc(), "time_ns": 0}
                shared["sc_end"] = None
                ev_done.set()

    def stdin_reader():
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            if line == "QUIT":
                shared["quit"] = True
                ev_run.set()
                return
            try:
                tc = json.loads(line)
            except json.JSONDecodeError as e:
                emit({"error": f"JSON parse: {e}", "time_ns": 0})
                continue
            shared.update({"test_case": tc, "result": None})
            ev_done.clear()
            ev_run.set()
            ev_done.wait()
            emit(shared["result"] or {"error": "no result", "time_ns": 0})
        shared["quit"] = True
        ev_run.set()

    threading.Thread(target=stdin_reader, daemon=True).start()
    panda.run()


if __name__ == "__main__":
    main()
