#!/usr/bin/env python3
"""
detect_microtaint.py -- microtaint driver for the nft_byteorder evaluation.

Reference implementation of the shared contract (common.py). This is the same
two-color method as the original detect.py, refactored to emit a Verdict so it
lines up column-for-column with the other six engines.

microtaint runs the real harness under Qiling and propagates bit-precise taint.
The stdin bytes are the source (color A, tainted at read()); a second static
shadow marks canary_after (color B). A write into color-B territory whose value
also carries color-A taint is the A-cap-B memory-corruption signal.
"""
from __future__ import annotations

import sys
from pathlib import Path

from qiling import Qiling
from qiling.const import QL_VERBOSE

from microtaint.emulator.shadow import BitPreciseShadowMemory
from microtaint.emulator.reporter import Reporter
from microtaint.emulator.wrapper import MicrotaintWrapper

import common
from common import Layout, Verdict


class _Stdin:
    def __init__(self, data: bytes):
        self._buf = bytearray(data)

    def read(self, count: int) -> bytes:
        chunk = bytes(self._buf[:count])
        del self._buf[:count]
        return chunk


def run_case(harness: Path, lay: Layout) -> dict:
    """Run one harness; return sink observations."""
    ql = Qiling([str(harness)], "/", verbose=QL_VERBOSE.OFF)
    ql.os.stdin = _Stdin(common.PAYLOAD)

    reporter = Reporter()
    wrapper = MicrotaintWrapper(
        ql, check_bof=False, check_uaf=False, check_sc=False,
        check_aiw=False, reporter=reporter,
    )
    color_a: BitPreciseShadowMemory = wrapper.shadow_mem

    color_b = BitPreciseShadowMemory()
    for off in range(lay.canary_after_size):
        color_b.write_mask(lay.canary_after + off, 0xFF, 1)
    for off in range(lay.canary_before_size):
        color_b.write_mask(lay.canary_before + off, 0xFF, 1)

    oob: list[dict] = []
    inbounds: list[dict] = []

    def on_write(ql, access, address, size, value):  # noqa: ANN001
        # count only the buggy stride stores (size-2 u16 byteswaps), so the
        # number is comparable to the other tools; the harness's own byte-wise
        # canary-init loop (size-1 constant writes) is not the overshoot.
        if size == 2 and color_b.is_tainted(address, size):
            ca = color_a.read_mask(address, min(size, 8))
            oob.append({"pc": ql.arch.regs.read("rip"), "addr": address,
                        "size": size, "value_taint": ca})
        elif size == 2 and lay.regs <= address < lay.regs + lay.regs_size:
            # the 20 in-bounds byteswap stores (regs offsets 0,4,..,76)
            ca = color_a.read_mask(address, min(size, 8))
            inbounds.append({"addr": address, "value_taint": ca})

    ql.hook_mem_write(on_write)
    try:
        ql.run()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}", "oob": oob, "inbounds": inbounds}
    return {"error": "", "oob": oob, "inbounds": inbounds}


def main() -> int:
    lay = common.recover_layout(common.VULN_HARNESS)
    v = Verdict(
        tool="microtaint",
        family="dbi",
        method=("Qiling emulation + bit-precise shadow memory; stdin tainted "
                "at read(), taint checked at every guest write into canary_after"),
    )

    res = run_case(common.VULN_HARNESS, lay)
    v.ran = True
    if res["error"]:
        v.error = res["error"]
    oob = res["oob"]
    tainted = [w for w in oob if w["value_taint"]]
    v.n_oob_writes = len(oob)
    v.n_oob_writes_tainted = len(tainted)
    v.sink_value_tainted = bool(tainted)
    inb = res.get("inbounds", [])
    v.n_inbounds_writes = len(inb)
    v.n_inbounds_tainted = sum(1 for w in inb if w["value_taint"])
    # microtaint tracks data-flow taint on memory, not pointer provenance; the
    # OOB address here is computed from an untainted loop index.
    v.sink_addr_input_dependent = False
    if tainted:
        v.oob_pc = hex(tainted[0]["pc"])
    v.detected = v.sink_value_tainted
    v.notes = ("flags all 20 OOB stride-4 stores value-tainted -- an OVER-TAINT. "
               "The taint is the len control byte, NOT attacker register data "
               "(clearing len -> 0 tainted; clearing all 80 register bytes -> still "
               "20). It is laundered by two imprecisions: mov reg<-mem does not "
               "clear stale register taint, so len's taint from the loop-bound "
               "computation survives into the untainted loop index; then loading "
               "through that spuriously-tainted address taints the value "
               "(pointer-avalanche). Sound (never misses the write) but imprecise.")

    # Control: the patched harness must NOT reach the sentinel.
    if common.FIXED_HARNESS.exists():
        fres = run_case(common.FIXED_HARNESS, lay)
        f_tainted = [w for w in fres["oob"] if w["value_taint"]]
        v.control_fixed_detected = bool(f_tainted)

    p = v.save()
    print(f"[microtaint] saw: {v.n_oob_writes_tainted}/{v.n_oob_writes} OOB stores flagged "
          f"value-tainted (over-taint) | not: address is a clean loop index -> {p}")
    return 0 if v.detected else 1


if __name__ == "__main__":
    sys.exit(main())
