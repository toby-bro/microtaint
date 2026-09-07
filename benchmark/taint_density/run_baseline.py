"""Bare Qiling with no taint engine attached: the denominator for overhead.

The mission's target is stated as a multiple of Qiling's own speed, so that
number has to be measured on the same binary, the same input and the same
machine as the taint runs -- not quoted from a different setup.
"""
from __future__ import annotations

import io
import os
import sys
import time

BENCH = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 64
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
STDIN = bytes((i * 7 + 13) & 0xFF for i in range(N))

_saved_fd1 = os.dup(1)
_devnull = os.open(os.devnull, os.O_WRONLY)

from qiling import Qiling            # noqa: E402
from qiling.const import QL_VERBOSE  # noqa: E402

best = float('inf')
for _ in range(ITERS):
    ql = Qiling([BENCH], '/', verbose=QL_VERBOSE.OFF)
    ql.os.stdin = io.BytesIO(STDIN)
    os.dup2(_devnull, 1)
    t0 = time.perf_counter()
    try:
        ql.run()
    except Exception:  # noqa: BLE001
        pass
    dt = time.perf_counter() - t0
    os.dup2(_saved_fd1, 1)
    best = min(best, dt)

print(f'{best * 1e3:9.2f} ms   (no taint)')
