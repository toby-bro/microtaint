"""Where the taint engine's per-instruction work actually goes.

Wall-clock alone cannot tell you WHY a change helped, and the counters that can
were being re-derived by hand every time. This prints them: how many
instructions the C fast path finished on its own, how many were dismissed
outright as untainted, how well the output cache is doing, and -- when
instructions do leave the C path -- the reason they left.

    python taint_stats.py bench_dense.elf [input_bytes]

Read it alongside run_bench.py: that one says whether a change was faster, this
one says whether it was faster for the reason you thought.

The two ratios worth watching:

  fast path   -- instructions completed with no Python at all. A change that
                 barely moves the clock but drops this has quietly pushed work
                 back onto the slow path; a change that raises it without
                 moving the clock was not on the critical path.
  fallbacks   -- broken down by cause, so the biggest one is a measurement
                 rather than a guess. That breakdown is what showed 99% of
                 fallbacks were PC-writing instructions (branches) taking the
                 Python path purely for an implicit-taint check that had
                 nothing to decide.
"""

from __future__ import annotations

import io
import os
import sys

from qiling import Qiling
from qiling.const import QL_VERBOSE

from microtaint.emulator.reporter import Reporter
from microtaint.emulator.wrapper import MicrotaintWrapper


def _pct(n: int, d: int) -> str:
    return f'{100.0 * n / d:5.1f}%' if d else '    -'


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    binary = sys.argv[1]
    n_in = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    stdin = bytes((i * 7 + 13) & 0xFF for i in range(n_in))

    ql = Qiling([binary], '/', verbose=QL_VERBOSE.OFF)
    ql.os.stdin = io.BytesIO(stdin)
    w = MicrotaintWrapper(ql, reporter=Reporter(json_mode=False, stream=io.StringIO()))

    # The guest writes binary noise to stdout; silence it during emulation only.
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(1)
    os.dup2(devnull, 1)
    try:
        ql.run()
    except Exception:  # noqa: BLE001 - a guest crash still leaves valid counters
        pass
    finally:
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)

    h = w._instr_hook_obj  # noqa: SLF001 - the counters live on the hook
    if h is None:
        print('no Cython instruction hook was installed (nothing to report)')
        return 1

    total = h.instr_total
    probes = h.instr_cache_hits + h.instr_cache_misses
    fb = h.fb_pc + h.fb_pyfall + h.fb_mem + h.fb_other

    print(f'binary                : {binary}')
    print(f'instructions hooked   : {total}')
    print(f'  C fast path         : {h.fast_done:>9}  {_pct(h.fast_done, total)}')
    print(f'  of which prefiltered: {h.prefilter_hits:>9}  {_pct(h.prefilter_hits, total)}'
          '   (no input tainted)')
    print(f'output cache          : {h.instr_cache_hits} hit / {probes} probed'
          f'  {_pct(h.instr_cache_hits, probes)}')
    print(f'cell re-executions    : {w.sim._pcode.native_calls}')  # noqa: SLF001
    print(f'Python fallbacks      : {fb:>9}  {_pct(fb, total)}')
    if fb:
        print(f'    PC write (branch) : {h.fb_pc:>9}  {_pct(h.fb_pc, fb)} of fallbacks')
        print(f'    python assignment : {h.fb_pyfall:>9}  {_pct(h.fb_pyfall, fb)}')
        print(f'    memory circuit    : {h.fb_mem:>9}  {_pct(h.fb_mem, fb)}')
        print(f'    other             : {h.fb_other:>9}  {_pct(h.fb_other, fb)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
