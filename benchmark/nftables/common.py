#!/usr/bin/env python3
"""
common.py -- shared contract for the multi-tool nft_byteorder taint evaluation.

Every per-tool driver (detect_<tool>.py) imports this module so the *source*,
the *sink*, and the *verdict schema* are identical across all seven engines.
That uniformity is what makes the comparison fair: the only thing that varies
between drivers is the taint engine, never the question we ask it.

The bug (CVE-2023-35001, nft_byteorder alignment/stride mismatch): with
priv->size == 2 the evaluator iterates priv->len/2 times at stride 4, writing
2 bytes each iteration, so it overshoots the validated register file by ~len/2
bytes into the memory that follows it.

Source  (color A): the 83 bytes the harness reads from stdin
                   (3 control bytes + 80 register bytes). All attacker-derived.
Sink    (color B): the 256-byte `canary_after` region placed immediately after
                   the 80-byte register file. Any write here is out of bounds.

Detection question, asked identically of every tool:
  Does attacker-tainted data (A) reach a write into sentinel territory (B)?
We record two independent columns at the sink:
  * sink_value_tainted        -- the VALUE stored is attacker-tainted
  * sink_addr_input_dependent -- the ADDRESS written is attacker/input-dependent
For this bug the overshoot geometry is constant given len=80, so the value
column is the primary A-cap-B signal; the address column separates tools that
reason about pointer provenance from those that only track data.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
VULN_HARNESS = HERE / "harness"
FIXED_HARNESS = HERE / "harness_fixed"
RESULTS_DIR = HERE / "results"

# ---------------------------------------------------------------------------
# Layout (must match harness.c `struct layout`)
#   [ canary_before : 256 ][ regs : 80 ][ canary_after : 256 ]
# ---------------------------------------------------------------------------
CANARY_BEFORE_SIZE = 256
REGS_SIZE = 80
CANARY_AFTER_SIZE = 256

# ---------------------------------------------------------------------------
# Trigger payload -- the 83 attacker-controlled stdin bytes (the taint source).
#   ctl[0] = priv->len  = 80  (max the validator accepts for dreg=0)
#   ctl[1] = priv->size = 2   (the branch with the stride/element mismatch)
#   ctl[2] = priv->op   = 0   (NFT_BYTEORDER_NTOH)
#   then 80 register bytes (source of the htons/ntohs), all tainted.
# ---------------------------------------------------------------------------
CTL_BYTES = bytes([80, 2, 0])
REGS_PAYLOAD = bytes([0x42] * REGS_SIZE)
PAYLOAD = CTL_BYTES + REGS_PAYLOAD  # 83 bytes; every byte is color A
PAYLOAD_LEN = len(PAYLOAD)


@dataclass
class Layout:
    base: int            # address of global struct `L` (== canary_before)
    canary_before: int
    regs: int
    canary_after: int
    canary_before_size: int = CANARY_BEFORE_SIZE
    regs_size: int = REGS_SIZE
    canary_after_size: int = CANARY_AFTER_SIZE

    def in_canary_after(self, addr: int, size: int = 1) -> bool:
        return not (addr + size <= self.canary_after
                    or addr >= self.canary_after + self.canary_after_size)

    def in_canary_before(self, addr: int, size: int = 1) -> bool:
        return not (addr + size <= self.canary_before
                    or addr >= self.canary_before + self.canary_before_size)

    def canary_after_offset(self, addr: int) -> int:
        return addr - self.canary_after


def recover_layout(harness: Path = VULN_HARNESS) -> Layout:
    """Deterministic addresses from the ELF symbol table.

    The harness is built `-static -no-pie`, so the link-time address of the
    global `L` struct is also its runtime address under every tool (native
    ptrace/Pin/Valgrind, Qiling, angr, Triton, Maat). No run required.
    """
    out = subprocess.check_output(["nm", str(harness)], text=True)
    base = None
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == "L":
            base = int(parts[0], 16)
            break
    if base is None:
        raise RuntimeError(f"symbol `L` not found in {harness}")
    return Layout(
        base=base,
        canary_before=base,
        regs=base + CANARY_BEFORE_SIZE,
        canary_after=base + CANARY_BEFORE_SIZE + REGS_SIZE,
    )


# ---------------------------------------------------------------------------
# Verdict schema -- every driver writes one of these to results/<tool>.json
# ---------------------------------------------------------------------------
@dataclass
class Verdict:
    tool: str
    # family: "dbi" (runs the real binary) or "symbolic" (models input)
    family: str
    method: str                       # one line: how this tool tracks taint
    ran: bool = False                 # did the engine execute/analyze the harness
    # --- sink observations on the VULNERABLE harness ---
    n_oob_writes: int = 0             # writes landing in canary_after
    n_oob_writes_tainted: int = 0     # of those, value carried attacker taint
    sink_value_tainted: bool = False  # >=1 tainted value reached the sentinel
    sink_addr_input_dependent: bool = False  # write ADDRESS was input-dependent
    oob_pc: str = ""                  # representative PC of the OOB store
    # --- in-bounds observations (for exact under-taint scoring) ---
    # the 20 in-bounds u16 stores (regs offsets 0,4,..,76) carry attacker data;
    # a tool that fails to taint them is under-tainting.
    n_inbounds_writes: int = -1       # in-bounds byteswap stores seen (-1 = not measured)
    n_inbounds_tainted: int = -1      # of those, value tainted (-1 = not measured)
    # --- final verdict + control ---
    detected: bool = False            # attacker taint reached sentinel (A cap B)
    control_fixed_detected: bool | None = None  # patched harness: must be False
    error: str = ""
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def save(self) -> Path:
        RESULTS_DIR.mkdir(exist_ok=True)
        p = RESULTS_DIR / f"{self.tool}.json"
        p.write_text(json.dumps(asdict(self), indent=2))
        return p


def load_all_verdicts() -> list[dict]:
    if not RESULTS_DIR.exists():
        return []
    out = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:  # noqa: BLE001
            pass
    return out


if __name__ == "__main__":
    lay = recover_layout()
    print(f"L={lay.base:#x} regs={lay.regs:#x}+{lay.regs_size} victim={lay.canary_after:#x}+{lay.canary_after_size} "
          f"payload={PAYLOAD_LEN}B (len={CTL_BYTES[0]} size={CTL_BYTES[1]} op={CTL_BYTES[2]})")
