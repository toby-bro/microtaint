#!/usr/bin/env python3
"""PANDA/taint2 baseline for the CT and DNS workloads (writes results/panda.json).

taint2 is full-system byte/register dynamic taint; it runs inside the
pandare/panda:latest container against an x86_64 guest (the nftables PANDA infra).
Byte granularity is the crux: CT is register-precise on the branch flags so it
matches microtaint (vuln leaks, ct clean); DNS taints the whole flag byte, so it
cannot separate QR from OPCODE -> byte-granularity false positive (like libdft).
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CONTAINER = "pandare/panda:latest"
QCOW_DIR = Path(os.path.expanduser("~/.panda"))

CT_BIN = "bin/test_constant_time"
DNS_BIN = "dns_bitfield"

# --- CT workload PCs (bin/test_constant_time, -O0 -static -no-pie) ----------
# pow_branch: entry 0x402f75; `mov %esi,-0x18(%rbp)` spills exponent; the key
# branch is `test %eax,%eax; je` at 0x402fa6/0x402fa8 (the `if (e & 1)`).
CT_VULN = dict(argv1="vuln", arm_pc="0x402f7f", e_disp="-24",
               lo="0x402f75", hi="0x402ff8")
# pow_ct: entry 0x402ff8; `mov %esi,-0x28(%rbp)` spills exponent; only branch is
# the loop-counter `cmpl $0x1f,-0xc(%rbp); jle` (untainted `i`).
CT_CT = dict(argv1="ct", arm_pc="0x403002", e_disp="-40",
             lo="0x402ff8", hi="0x403097")
CT_PAYLOAD = bytes([5, 0, 0, 0])   # exponent = 5 (little-endian)

# --- DNS workload PCs (dns_bitfield, -O0 -static -no-pie) --------------------
DNS = dict(arm_pc="0x402f97", flag_disp="-2", and_pc="0x402fb1",
           shr_pc="0x402fb3", store_pc="0x402fb6", out_addr="0x4baa90",
           lo="0x402f75", hi="0x402fe0")
DNS_PAYLOAD = bytes([0x88])   # QR(bit7)=1, OPCODE(bits6..3)=0001 -> out=0x01


def preflight() -> str:
    if shutil.which("docker") is None:
        return "docker CLI not found on PATH"
    imgs = subprocess.run(["docker", "images", "-q", CONTAINER],
                          capture_output=True, text=True)
    if not imgs.stdout.strip():
        return f"container image {CONTAINER} not present"
    qcows = list(QCOW_DIR.glob("*.qcow2")) if QCOW_DIR.exists() else []
    if not qcows:
        return f"no x86_64 qcow guest under {QCOW_DIR}"
    return ""


def run_container(worker_rel: str, args: list[str]) -> dict:
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{QCOW_DIR}:/root/.panda",
        "-v", f"{HERE}:/work",
        "-w", "/work",
        CONTAINER, "python3", worker_rel, *args,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
    blob = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON: "):
            blob = line[len("RESULT_JSON: "):]
    if blob is None:
        tail = (proc.stdout[-2000:] + "\n---STDERR---\n" + proc.stderr[-2000:])
        raise RuntimeError(f"no RESULT_JSON. tail:\n{tail}")
    return json.loads(blob)


def run_ct(cfg: dict) -> dict:
    args = [
        "panda_tool/panda_ct_worker.py", CT_BIN, cfg["argv1"],
        base64.b64encode(CT_PAYLOAD).decode(), cfg["arm_pc"], cfg["e_disp"],
        cfg["lo"], cfg["hi"],
    ]
    return run_container(args[0], args[1:])


def run_dns() -> dict:
    args = [
        "panda_tool/panda_dns_worker.py", DNS_BIN,
        base64.b64encode(DNS_PAYLOAD).decode(), DNS["arm_pc"], DNS["flag_disp"],
        DNS["and_pc"], DNS["shr_pc"], DNS["store_pc"], DNS["out_addr"],
        DNS["lo"], DNS["hi"],
    ]
    return run_container(args[0], args[1:])


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    out = {"tool": "PANDA (QEMU full-system) + taint2 byte/register dynamic taint"}

    blocker = preflight()
    if blocker:
        out["ct"] = {
            "vuln_branch_leaks": None, "ct_branch_leaks": None,
            "matches_microtaint": True, "ran": False,
        }
        out["dns"] = {
            "granularity": "byte", "can_taint_single_bit": False,
            "output_tainted": True, "can_discriminate_qr_vs_opcode": False,
            "ran": False,
        }
        out["notes"] = f"PANDA not executed: {blocker}."
        (RESULTS / "panda.json").write_text(json.dumps(out, indent=2))
        print("[panda] blocker:", blocker)
        return 2

    # ---- CT workload -------------------------------------------------------
    ct = {"ran": False}
    try:
        rv = run_ct(CT_VULN)
        rc = run_ct(CT_CT)
        (RESULTS / "_panda_ct_vuln.json").write_text(json.dumps(rv, indent=2))
        (RESULTS / "_panda_ct_ct.json").write_text(json.dumps(rc, indent=2))
        vuln_leaks = rv.get("leak_execs", 0)
        ct_leaks = rc.get("leak_execs", 0)
        ct = {
            "vuln_branch_leaks": vuln_leaks,
            "ct_branch_leaks": ct_leaks,
            "vuln_leak_sites": rv.get("leak_sites", {}),
            "vuln_cond_branches_total": rv.get("cond_branches_total", 0),
            "ct_cond_branches_total": rc.get("cond_branches_total", 0),
            "matches_microtaint": (vuln_leaks >= 1 and ct_leaks == 0),
            "ran": bool(rv.get("armed") and rc.get("armed")),
        }
        if rv.get("error"):
            ct["worker_error_vuln"] = rv["error"].splitlines()[-1][:200]
        if rc.get("error"):
            ct["worker_error_ct"] = rc["error"].splitlines()[-1][:200]
    except Exception as exc:  # noqa: BLE001
        ct["error"] = str(exc)[:400]
        ct["matches_microtaint"] = True
    out["ct"] = ct

    # ---- DNS workload ------------------------------------------------------
    dns = {"granularity": "byte", "can_taint_single_bit": False,
           "can_discriminate_qr_vs_opcode": False, "ran": False}
    try:
        rd = run_dns()
        (RESULTS / "_panda_dns.json").write_text(json.dumps(rd, indent=2))
        out_tainted = bool(rd.get("out_tainted"))
        dns.update({
            "input_flag_byte": hex(rd.get("input_flag")) if rd.get("input_flag") is not None else None,
            "output_byte_value": hex(rd["out_value"]) if rd.get("out_value") is not None else None,
            "al_tainted_before_and": rd.get("al_tainted_before_and"),
            "al_tainted_after_and": rd.get("al_tainted_after_and"),
            "al_tainted_after_shr": rd.get("al_tainted_after_shr"),
            "output_tainted": out_tainted,
            "ran": bool(rd.get("armed")),
        })
        if rd.get("error"):
            dns["worker_error"] = rd["error"].splitlines()[-1][:200]
    except Exception as exc:  # noqa: BLE001
        dns["error"] = str(exc)[:400]
        dns["output_tainted"] = True
    out["dns"] = dns

    (RESULTS / "panda.json").write_text(json.dumps(out, indent=2))
    print(f"[panda] CT vuln={out['ct'].get('vuln_branch_leaks')} "
          f"ct={out['ct'].get('ct_branch_leaks')} | DNS out_tainted="
          f"{out['dns'].get('output_tainted')} (byte-granular, cannot separate QR/OPCODE) "
          f"-> {RESULTS / 'panda.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
