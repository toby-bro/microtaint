"""Lean microtaint-only timer for A/B env-flag deltas (no phase profiler)."""
from __future__ import annotations
import io, os, sys, time

BENCH = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 64
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
STDIN = bytes((i * 7 + 13) & 0xFF for i in range(N))

# Silence the guest's own stdout (it writes binary garbage) by pointing fd 1 at
# /dev/null during emulation; restore it for the result line.
_saved_fd1 = os.dup(1)
_devnull = os.open(os.devnull, os.O_WRONLY)

from qiling import Qiling
from qiling.const import QL_VERBOSE
from microtaint.emulator.reporter import Reporter
from microtaint.emulator.wrapper import MicrotaintWrapper

best = float('inf'); instrs = 0; cells = 0
for _ in range(ITERS):
    ql = Qiling([BENCH], '/', verbose=QL_VERBOSE.OFF)
    ql.os.stdin = io.BytesIO(STDIN)
    w = MicrotaintWrapper(ql, reporter=Reporter(json_mode=False, stream=sys.stderr))
    os.dup2(_devnull, 1)
    t0 = time.perf_counter()
    try:
        ql.run()
    except Exception:
        pass
    best = min(best, time.perf_counter() - t0)
    os.dup2(_saved_fd1, 1)
    try:
        cells = w.sim._pcode.native_calls
        h = w._instr_hook_obj
        instrs = h.instr_cache_hits + h.instr_cache_misses
    except Exception:
        pass

# instruction count: use a fixed reference (from the counting run) if available
print(f"{best*1e3:9.2f} ms   cells={cells}")
