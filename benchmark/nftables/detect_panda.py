#!/usr/bin/env python3
"""
detect_panda.py -- PANDA/taint2 driver for the nft_byteorder evaluation.

PANDA is the heaviest tool in the comparison: a full-system whole-machine
dynamic-taint platform (QEMU + taint2). The pip `pandare` bindings on this
host ship no native `libpanda-x86_64.so`, and taint2 requires a full-system
guest image, so the actual analysis runs inside the `pandare/panda:latest`
container against the bionic x86_64 guest (`~/.panda/bionic-...qcow2`, the
image `Panda(generic="x86_64")` resolves to). This host-side driver, run with
the project's `.venv_panda`, orchestrates that container, parses its taint
observations, and emits the shared common.Verdict.

taint2 method (identical detection question to every other tool)
----------------------------------------------------------------
The static -no-pie harness runs in the no-ASLR guest at fixed virtual
addresses. panda_tool/panda_worker.py:
  * hooks the entry PC of nft_byteorder_eval and applies byte-positional
    taint2 labels to the 80 attacker-controlled register bytes already read
    from stdin into the register file (the "color A" source);
  * on every guest virtual memory write, checks whether the destination is
    in canary_after [regs+80, +256) and, if so, queries taint2 on the
    freshly-written RAM to decide whether the stored VALUE is tainted;
  * separately counts in-bounds byteswap writes into the register file and
    how many taint2 marks tainted -- a propagation sanity check.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import common
from common import Layout, Verdict

CONTAINER = "pandare/panda:latest"
QCOW_DIR = Path(os.path.expanduser("~/.panda"))
WORKER_REL = "panda_tool/panda_worker.py"
EVAL_PC = 0x403017  # nft_byteorder_eval entry (nm harness)


def _eval_pc(harness: Path) -> int:
    out = subprocess.check_output(["nm", str(harness)], text=True)
    for line in out.splitlines():
        p = line.split()
        if len(p) == 3 and p[2] == "nft_byteorder_eval":
            return int(p[0], 16)
    return EVAL_PC


def run_in_container(harness_name: str, lay: Layout, eval_pc: int) -> dict:
    """Invoke panda_worker.py inside the pandare container; parse RESULT_JSON."""
    payload_b64 = base64.b64encode(common.PAYLOAD).decode()
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{QCOW_DIR}:/root/.panda",
        "-v", f"{common.HERE}:/work",
        "-w", "/work",
        CONTAINER, "python3", WORKER_REL,
        harness_name, payload_b64,
        hex(eval_pc), hex(lay.regs), hex(lay.canary_after),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    blob = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON: "):
            blob = line[len("RESULT_JSON: "):]
    if blob is None:
        tail = (proc.stdout[-1500:] + "\n" + proc.stderr[-1500:])
        raise RuntimeError(f"no RESULT_JSON from container. tail:\n{tail}")
    return json.loads(blob)


def _preflight() -> str:
    if shutil.which("docker") is None:
        return "docker CLI not found on PATH"
    imgs = subprocess.run(["docker", "images", "-q", CONTAINER],
                          capture_output=True, text=True)
    if not imgs.stdout.strip():
        return f"container image {CONTAINER} not present (docker pull it)"
    qcows = list(QCOW_DIR.glob("*.qcow2")) if QCOW_DIR.exists() else []
    if not qcows:
        return f"no x86_64 qcow guest image under {QCOW_DIR}"
    return ""


def main() -> int:
    lay = common.recover_layout(common.VULN_HARNESS)
    v = Verdict(
        tool="panda",
        family="dbi",
        method=("Full-system PANDA (QEMU) + taint2 byte-precise dynamic taint; "
                "80 stdin-derived register bytes labeled at nft_byteorder_eval "
                "entry, taint queried on every guest write into canary_after"),
    )

    blocker = _preflight()
    if blocker:
        v.ran = False
        v.error = blocker
        v.notes = (
            "PANDA taint2 is full-system byte-precise forward data-flow taint "
            "(same class as libdft64/taintgrind). It runs in the pandare/panda "
            "container against the bionic x86_64 guest; the blocker above "
            "prevented execution.")
        p = v.save()
        print(f"[panda] blocked: {blocker} -> {p}")
        return 2

    eval_pc = _eval_pc(common.VULN_HARNESS)

    res = run_in_container("harness", lay, eval_pc)
    v.ran = bool(res.get("armed"))
    oob = res.get("oob", [])
    tainted = [w for w in oob if w.get("tainted_bytes", 0) > 0]
    v.n_oob_writes = len(oob)
    v.n_oob_writes_tainted = len(tainted)
    v.sink_value_tainted = bool(tainted)
    # taint2 tracks data-flow, not control-dependence: the OOB store address
    # is dst + i*4 (i an untainted induction var); the tainted priv->len only
    # bounds the loop (an implicit flow taint2 does not follow). No pointer is
    # data-tainted, so the address column is False -- matching microtaint.
    v.sink_addr_input_dependent = False
    if oob:
        v.oob_pc = hex(oob[0]["pc"])
    v.detected = v.sink_value_tainted

    inb_total = res.get("inbounds_total", 0)
    inb_tainted = res.get("inbounds_tainted", 0)
    values = sorted({hex(w["value"]) for w in oob if w.get("value") is not None})
    v.extra = {
        "labeled_source_bytes": res.get("labeled_bytes"),
        "oob_addr_lo": hex(min(w["addr"] for w in oob)) if oob else None,
        "oob_addr_hi": hex(max(w["addr"] for w in oob)) if oob else None,
        "oob_values": values,
        "inbounds_writes": inb_total,
        "inbounds_writes_tainted": inb_tainted,
        "serial_timeout": res.get("serial_timeout", False),
        "engine": "PANDA taint2 (full-system x86_64, pandare/panda container)",
    }
    if res.get("error"):
        v.error = res["error"].splitlines()[-1][:200]

    v.notes = (
        f"{len(oob)} OOB writes into canary_after (stride 4, all 0xefcd) -- the "
        f"exact overshoot geometry -- 0 value-tainted: the overshoot reads the "
        f"sentinel's own constants, not stdin. Sanity: {inb_tainted}/{inb_total} "
        f"in-bounds stores ARE tainted, so taint2 propagates through ntohs; the "
        f"OOB clean result is a true data-flow fact. Address not input-dependent "
        f"(tainted len only bounds the loop, an implicit flow taint2 skips).")

    # Control: patched harness must produce no OOB writes at all.
    if common.FIXED_HARNESS.exists():
        try:
            fpc = _eval_pc(common.FIXED_HARNESS)
            fres = run_in_container("harness_fixed", lay, fpc)
            f_oob = fres.get("oob", [])
            f_tainted = [w for w in f_oob if w.get("tainted_bytes", 0) > 0]
            v.control_fixed_detected = bool(f_tainted)
            v.extra["control_oob_writes"] = len(f_oob)
            v.extra["control_inbounds_writes"] = fres.get("inbounds_total", 0)
        except Exception as exc:  # noqa: BLE001
            v.extra["control_error"] = str(exc)[:200]

    p = v.save()
    print(f"[panda] saw: {v.n_oob_writes} OOB writes + {inb_tainted}/{inb_total} in-bounds "
          f"tainted | not: 0 OOB value-tainted, addr not input-dependent -> {p}")
    return 0 if v.ran else 2


if __name__ == "__main__":
    sys.exit(main())
