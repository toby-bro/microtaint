"""Where the per-instruction overhead goes, in nanoseconds.

Four runs of the same binary with the same input, so the differences mean
something:

  bare          Qiling with nothing attached -- the denominator.
  python hook   Qiling with an empty Python `hook_code` callback.  This is what
                a per-instruction hook costs before doing any work at all, and
                it is why microtaint installs a C trampoline instead.
  microtaint    the taint engine, however the environment has it configured.

Run it when deciding what to optimise next.  Once the taint computation itself
is a few nanoseconds, the answer stops being "make the taint cheaper".

    .venv/bin/python run_overhead.py bench_untainted.elf
"""
from __future__ import annotations

import io
import os
import sys
import time

BENCH = sys.argv[1] if len(sys.argv) > 1 else 'bench_untainted.elf'
N_IN = int(sys.argv[2]) if len(sys.argv) > 2 else 64
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
STDIN = bytes((i * 7 + 13) & 0xFF for i in range(N_IN))

_saved = os.dup(1)
_devnull = os.open(os.devnull, os.O_WRONLY)

from qiling import Qiling            # noqa: E402
from qiling.const import QL_VERBOSE  # noqa: E402


def _timed(build):
    """Best of ITERS, with the guest's own output silenced."""
    best = float('inf')
    extra = None
    for _ in range(ITERS):
        ql = Qiling([BENCH], '/', verbose=QL_VERBOSE.OFF)
        ql.os.stdin = io.BytesIO(STDIN)
        state = build(ql)
        os.dup2(_devnull, 1)
        t0 = time.perf_counter()
        try:
            ql.run()
        except Exception:  # noqa: BLE001
            pass
        dt = time.perf_counter() - t0
        os.dup2(_saved, 1)
        if dt < best:
            best, extra = dt, state
    return best, extra


def _bare(_ql):
    return None


def _python_hook(ql):
    box = [0]

    def cb(_ql, _addr, _size):
        box[0] += 1

    ql.hook_code(cb)
    return box


def _microtaint(ql):
    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper
    w = MicrotaintWrapper(ql, reporter=Reporter(json_mode=False, stream=sys.stderr))
    return w


def main() -> int:
    bare, _ = _timed(_bare)
    pyhook, box = _timed(_python_hook)
    taint, wrapper = _timed(_microtaint)

    n = box[0] if box else 0
    hook = getattr(wrapper, '_instr_hook_obj', None)
    if hook is not None and getattr(hook, 'instr_total', 0):
        n = hook.instr_total

    print(f'{BENCH}  {n} instructions, best of {ITERS}')
    print(f'  {"bare qiling":18s} {bare*1e3:9.2f} ms  {bare*1e9/max(1,n):8.1f} ns/instruction')
    print(f'  {"empty python hook":18s} {pyhook*1e3:9.2f} ms  {pyhook*1e9/max(1,n):8.1f} ns/instruction'
          f'   (+{(pyhook-bare)*1e9/max(1,n):.0f} for the callback alone)')
    print(f'  {"microtaint":18s} {taint*1e3:9.2f} ms  {taint*1e9/max(1,n):8.1f} ns/instruction'
          f'   (+{(taint-bare)*1e9/max(1,n):.0f} over bare)')
    if hook is not None:
        tot = max(1, getattr(hook, 'instr_total', 1))
        print(f'  hook coverage: {100*hook.prefilter_hits/tot:.0f}% dismissed with no tainted '
              f'input, {100*hook.fast_done/tot:.0f}% finished in C')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
