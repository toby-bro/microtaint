"""Where the per-instruction overhead goes, in nanoseconds.

Four configurations of the same binary with the same input, so the differences
mean something:

  bare          Qiling with nothing attached -- the denominator.
  python hook   Qiling with an empty Python `hook_code` callback.  This is what
                a per-instruction hook costs before doing any work at all, and
                it is why the engine installs a C trampoline instead.
  null hook     the engine's own C trampoline, returning immediately
                (MICROTAINT_NULL_HOOK=1).  The gap from `bare` is the cost of
                BEING hooked -- Unicorn's dispatch, and the GIL acquire the
                callback declaration implies.
  microtaint    the engine, however the environment has it configured.  The gap
                from `null hook` is the taint work itself.

Each configuration runs in its own process, for two reasons: the null-hook flag
is read when the extension is imported, so it cannot be toggled in-flight, and
running them together lets one warm the caches for the next and quietly flatters
whichever goes last.

Run it when deciding what to optimise.  Once the taint computation is a few
nanoseconds, the answer stops being "make the taint cheaper".

    .venv/bin/python run_overhead.py bench_untainted.elf
"""
# ruff: noqa: S603
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time

MODES = ('bare', 'pyhook', 'null', 'taint')


def _one(mode: str, bench: str, n_in: int, iters: int) -> dict:
    """Time one configuration in this process and report it as JSON."""
    stdin_bytes = bytes((i * 7 + 13) & 0xFF for i in range(n_in))
    saved, devnull = os.dup(1), os.open(os.devnull, os.O_WRONLY)

    from qiling import Qiling
    from qiling.const import QL_VERBOSE

    best = float('inf')
    count = 0
    coverage = None
    for _ in range(iters):
        ql = Qiling([bench], '/', verbose=QL_VERBOSE.OFF)
        ql.os.stdin = io.BytesIO(stdin_bytes)
        box = [0]
        wrapper = None
        if mode == 'pyhook':
            def cb(_ql, _addr, _size, box=box):
                box[0] += 1
            ql.hook_code(cb)
        elif mode in ('null', 'taint'):
            from microtaint.emulator.reporter import Reporter
            from microtaint.emulator.wrapper import MicrotaintWrapper
            wrapper = MicrotaintWrapper(ql, reporter=Reporter(json_mode=False,
                                                              stream=sys.stderr))
        os.dup2(devnull, 1)
        t0 = time.perf_counter()
        try:
            ql.run()
        except Exception:  # noqa: BLE001
            pass
        dt = time.perf_counter() - t0
        os.dup2(saved, 1)
        # The hook object is installed during the run, so it is read afterwards.
        hook_obj = getattr(wrapper, '_instr_hook_obj', None) if wrapper else None
        if dt < best:
            best = dt
            count = box[0] or (getattr(hook_obj, 'instr_total', 0) or 0)
            if mode == 'taint' and hook_obj is not None:
                tot = max(1, hook_obj.instr_total)
                coverage = (100 * hook_obj.prefilter_hits / tot,
                            100 * hook_obj.fast_done / tot)
    return {'mode': mode, 'seconds': best, 'instructions': count,
            'coverage': coverage}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == '--one':
        print(json.dumps(_one(argv[1], argv[2], int(argv[3]), int(argv[4]))))
        return 0

    bench = argv[0] if argv else 'bench_untainted.elf'
    n_in = int(argv[1]) if len(argv) > 1 else 64
    iters = int(argv[2]) if len(argv) > 2 else 3

    results = {}
    for mode in MODES:
        env = dict(os.environ)
        if mode == 'null':
            env['MICROTAINT_NULL_HOOK'] = '1'
        else:
            env.pop('MICROTAINT_NULL_HOOK', None)
        out = subprocess.run(
            [sys.executable, os.path.abspath(__file__), '--one', mode, bench,
             str(n_in), str(iters)],
            capture_output=True, text=True, env=env, check=True,
            cwd=os.path.dirname(os.path.abspath(__file__)))
        results[mode] = json.loads(out.stdout.strip().splitlines()[-1])

    n = max((r['instructions'] for r in results.values()), default=0) or 1

    def ns(mode):
        return results[mode]['seconds'] * 1e9 / n

    print(f'{bench}  {n} instructions, best of {iters}, each in its own process')
    print(f'  {"bare qiling":18s} {results["bare"]["seconds"]*1e3:9.2f} ms  '
          f'{ns("bare"):8.1f} ns/instruction')
    print(f'  {"empty python hook":18s} {results["pyhook"]["seconds"]*1e3:9.2f} ms  '
          f'{ns("pyhook"):8.1f} ns/instruction   '
          f'(+{ns("pyhook")-ns("bare"):.0f} for the callback alone)')
    print(f'  {"null hook":18s} {results["null"]["seconds"]*1e3:9.2f} ms  '
          f'{ns("null"):8.1f} ns/instruction   '
          f'(+{ns("null")-ns("bare"):.0f} just to be hooked)')
    print(f'  {"microtaint":18s} {results["taint"]["seconds"]*1e3:9.2f} ms  '
          f'{ns("taint"):8.1f} ns/instruction   '
          f'(+{ns("taint")-ns("null"):.0f} of taint work)')
    cov = results['taint']['coverage']
    if cov:
        print(f'  hook coverage: {cov[0]:.0f}% dismissed with no tainted input, '
              f'{cov[1]:.0f}% finished in C')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
