#!/usr/bin/env python3
"""
report.py -- aggregate every tool's results/<tool>.json into one comparison
table for the nft_byteorder (CVE-2023-35001) taint evaluation.

Run each detect_<tool>.py first (each writes results/<tool>.json), then:
    python report.py            # pretty table to stdout
    python report.py --md       # Markdown table (for the README)
"""
from __future__ import annotations

import sys

import common
import ground_truth

# Presentation order: DBI whole-program taint first, then symbolic/emulation.
ORDER = ["microtaint", "libdft64", "taintgrind", "panda",
         "angr", "triton", "maat"]


def _score(v: dict) -> tuple[int, str]:
    """(over_taint, under_taint-or-'?') for a verdict vs ground truth."""
    s = ground_truth.score(
        v.get("n_oob_writes_tainted", 0),
        v["n_inbounds_tainted"] if v.get("n_inbounds_tainted", -1) >= 0 else None,
    )
    under = "?" if s["under_taint"] is None else str(s["under_taint"])
    return s["over_taint"], under


def _yn(v) -> str:
    if v is None:
        return "-"
    return "yes" if v else "no"


def rows() -> list[dict]:
    verdicts = {v["tool"]: v for v in common.load_all_verdicts()}
    out = []
    for t in ORDER:
        if t in verdicts:
            out.append(verdicts[t])
    # any tool not in ORDER, appended at the end
    for t, v in verdicts.items():
        if t not in ORDER:
            out.append(v)
    return out


def pretty() -> str:
    rs = rows()
    if not rs:
        return "(no results yet -- run the detect_<tool>.py drivers first)"
    hdr = ["tool", "family", "ran", "#oob", "#oob-tainted",
           "over-taint", "under-taint", "control(fixed)"]
    lines = [hdr]
    for v in rs:
        over, under = _score(v)
        lines.append([
            v["tool"], v["family"], _yn(v["ran"]),
            str(v["n_oob_writes"]), str(v["n_oob_writes_tainted"]),
            str(over), under,
            _yn(v["control_fixed_detected"]),
        ])
    w = [max(len(r[i]) for r in lines) for i in range(len(hdr))]
    buf = []
    for j, r in enumerate(lines):
        buf.append("  ".join(c.ljust(w[i]) for i, c in enumerate(r)))
        if j == 0:
            buf.append("  ".join("-" * w[i] for i in range(len(hdr))))
    # error/blocker notes
    notes = [f"  {v['tool']}: {v['error'] or v['notes']}"
             for v in rs if v.get("error") or not v["ran"]]
    if notes:
        buf.append("\nblockers / notes:")
        buf.extend(notes)
    return "\n".join(buf)


def markdown() -> str:
    rs = rows()
    if not rs:
        return "_(no results yet)_"
    hdr = ["Tool", "Family", "OOB writes seen", "OOB value-tainted",
           "Over-taint (FP)", "Under-taint (FN)", "Control (patched) clean"]
    buf = ["| " + " | ".join(hdr) + " |",
           "|" + "|".join(["---"] * len(hdr)) + "|"]
    for v in rs:
        ctrl = v["control_fixed_detected"]
        over, under = _score(v)
        buf.append("| " + " | ".join([
            f"`{v['tool']}`", v["family"],
            str(v["n_oob_writes"]), str(v["n_oob_writes_tainted"]),
            f"**{over}**" if over else "0",
            under,
            _yn(None if ctrl is None else (not ctrl)),
        ]) + " |")
    return "\n".join(buf)


if __name__ == "__main__":
    print(markdown() if "--md" in sys.argv else pretty())
