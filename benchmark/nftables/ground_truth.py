#!/usr/bin/env python3
"""
ground_truth.py -- the exact taint ground truth for the nft_byteorder
(CVE-2023-35001) reproduction, worked out by hand from the loop rather than by
running any tool, so every tool's over-taint and under-taint can be scored.

The realistic bug, faithfully: with priv->sreg == priv->dreg == 0, size == 2,
op == NTOH, len == 80, nft_byteorder_eval runs an IN-PLACE byteswap:

    for i in 0 .. len/2 - 1:          # 40 iterations
        d[i].u16 = ntohs(s[i].u16)    # s == d == regs base, STRIDE 4 (union sizeof)

Each iteration writes 2 bytes (the low u16 of a 4-byte element) and SKIPS the
next 2. So per element i the byte map is:

    regs_base + 4*i      : WRITTEN   (low byte of the u16)
    regs_base + 4*i + 1  : WRITTEN   (high byte of the u16)
    regs_base + 4*i + 2  : untouched
    regs_base + 4*i + 3  : untouched

The 80-byte register file (bytes 0..79) is attacker-tainted (stdin). The victim
region begins at byte 80 and is initialised to a constant sentinel pattern (not
attacker data). Because the operation is in place, iteration i reads its source
from the SAME offset it writes, so:

    i = 0 .. 19  (4*i = 0..76  < 80):   source in regs  -> value is ATTACKER-TAINTED, dest in-bounds
    i = 20 .. 39 (4*i = 80..156 >= 80): source in victim -> value is a byteswap of an
                                        untainted CONSTANT, dest OUT OF BOUNDS

Hence the exact ground truth for value-flow taint at the out-of-bounds sink is
ZERO tainted bytes: the overshoot corrupts the victim, but the *values* written
there are not attacker-derived. The attacker's control over the corruption is
through the tainted loop bound `len` (a control dependence), not the data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import common

LEN = common.CTL_BYTES[0]      # 80
SIZE = common.CTL_BYTES[1]     # 2
STRIDE = 4                     # sizeof(union {u32; u16;})
REGS_BYTES = common.REGS_SIZE  # 80
N_ITERS = LEN // SIZE          # 40


@dataclass
class ByteFact:
    victim_off: int            # offset into the victim region (canary_after)
    written: bool              # did the buggy loop write this byte?
    value_attacker_tainted: bool  # is the written value attacker-derived (data flow)?


@dataclass
class GroundTruth:
    # write-level (u16 stores)
    n_writes_total: int            # loop iterations that store (40)
    n_writes_inbounds: int         # dest within regs (attacker value)  -> 20
    n_writes_oob: int              # dest in victim (constant value)     -> 20
    # value-flow taint at the OOB sink
    oob_value_tainted_bytes: int   # GROUND TRUTH: 0
    inbounds_value_tainted_bytes: int  # 40 (20 u16 * 2 bytes)
    # per victim-byte map for the corrupted span
    victim_written_offsets: list[int] = field(default_factory=list)   # bytes actually stomped
    victim_unwritten_offsets: list[int] = field(default_factory=list)  # skipped by the stride
    # control dependence
    overshoot_len_dependent: bool = True   # tainted `len` drives the trip count


def compute() -> GroundTruth:
    written = []
    unwritten = []
    n_oob = 0
    n_inb = 0
    for i in range(N_ITERS):
        base = STRIDE * i            # dest byte offset from regs base
        lo, hi = base, base + 1      # the two written bytes
        skip = (base + 2, base + 3)  # the two skipped bytes
        dest_oob = base >= REGS_BYTES
        if dest_oob:
            n_oob += 1
            for b in (lo, hi):
                written.append(b - REGS_BYTES)          # offset into victim
            for b in skip:
                if b >= REGS_BYTES:
                    unwritten.append(b - REGS_BYTES)
        else:
            n_inb += 1
    return GroundTruth(
        n_writes_total=N_ITERS,
        n_writes_inbounds=n_inb,
        n_writes_oob=n_oob,
        oob_value_tainted_bytes=0,          # <-- the exact truth
        inbounds_value_tainted_bytes=n_inb * SIZE,
        victim_written_offsets=sorted(written),
        victim_unwritten_offsets=sorted(unwritten),
    )


def score(n_oob_writes_tainted: int, n_inbounds_tainted: int | None = None) -> dict:
    """Score one tool against the ground truth.

    over_taint  = OOB writes a tool calls value-tainted that GT says are not (FP).
    under_taint = in-bounds attacker-tainted writes a tool misses (FN); only
                  scored when the tool reported an in-bounds tainted count.
    """
    gt = compute()
    over = max(0, n_oob_writes_tainted - gt.oob_value_tainted_bytes // SIZE)
    under = None
    if n_inbounds_tainted is not None:
        under = max(0, gt.n_writes_inbounds - n_inbounds_tainted)
    return {"over_taint": over, "under_taint": under}


if __name__ == "__main__":
    gt = compute()
    print(f"stores={gt.n_writes_total} (in={gt.n_writes_inbounds} oob={gt.n_writes_oob}) "
          f"gt_oob_value_tainted={gt.oob_value_tainted_bytes} len_driven={gt.overshoot_len_dependent} | "
          f"written[{len(gt.victim_written_offsets)}]={gt.victim_written_offsets} "
          f"unwritten[{len(gt.victim_unwritten_offsets)}]={gt.victim_unwritten_offsets}")
