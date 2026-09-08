#!/usr/bin/env python3
"""detect_maat.py -- assemble the Maat per-tool baseline (results/maat.json)
for the two RQ7 applications by running the CT and DNS drivers and folding
their raw traces into the shared schema.

Run:
  /home/jns/Documents/Telecom/PRIM/benchmark/.venv_maat/bin/python detect_maat.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

import detect_maat_ct
import detect_maat_dns

# Reference (microtaint) numbers for the CT workload.
MT_VULN_MIN = 1
MT_CT = 0


def main() -> int:
    detect_maat_ct.main()
    detect_maat_dns.main()

    ct_raw = json.loads((RESULTS / "_maat_ct_raw.json").read_text())
    dns_raw = json.loads((RESULTS / "_maat_dns_raw.json").read_text())

    vuln = ct_raw["vuln"]
    ct = ct_raw["ct"]
    vuln_leaks = vuln["leak_count_static"]
    ct_leaks = ct["leak_count_static"]

    matches_microtaint = (vuln_leaks >= MT_VULN_MIN) and (ct_leaks == MT_CT)
    qr_reaches = dns_raw["qr_reaches_output"]
    op_reaches = dns_raw["opcode_reaches_output"]
    can_bit = dns_raw["can_taint_single_bit"]
    can_disc = dns_raw["can_discriminate_qr_vs_opcode"]

    result = {
        "tool": "maat",
        "family": "symbolic",
        "ct": {
            "workload": "test_constant_time.c (square-and-multiply modexp)",
            "vuln_branch_leaks": vuln_leaks,
            "vuln_branch_leak_sites": vuln["leak_sites"],
            "vuln_branch_leak_execs": vuln["leak_branch_execs"],
            "ct_branch_leaks": ct_leaks,
            "matches_microtaint": matches_microtaint,
        },
        "dns": {
            "workload": "test_bitpacked_dns.c: AND al,0x78 ; SHR al,3",
            "granularity": "bit",
            "can_taint_single_bit": can_bit,
            "qr_reaches_output": qr_reaches,
            "opcode_reaches_output": op_reaches,
            "can_discriminate_qr_vs_opcode": can_disc,
        },
    }

    out_path = RESULTS / "maat.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[maat] CT vuln={vuln_leaks} ct={ct_leaks} (match={matches_microtaint}) | "
          f"DNS QR_out={qr_reaches} OPCODE_out={op_reaches} discriminate={can_disc} -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
