"""tests/test_fastpath_coverage_ratchet.py
==========================================
Ratchet on WHERE the per-instruction work happens, not how long it takes.

The hot-path optimisations are all of the same shape: keep an instruction inside
the C fast path, or dismiss it entirely, instead of falling back to Python or
re-executing a SLEIGH cell. The failure mode when one of them silently stops
applying is nasty, because wall clock barely moves -- the work shifts back to a
slower path that is still only a few percent of a benchmark dominated by other
things -- while the mechanism the code claims to have is simply gone.

So this asserts the deterministic counters instead:

  cells      SLEIGH cell re-executions. Counting, not timing, so it is identical
             on every machine and never flaky. This is the one that collapsed
             from ~38,000 to ~50 when loads became dismissible; if a change puts
             it back, that is a regression however fast the suite feels.
  fallbacks  instructions that left C for the Python evaluator.
  fast path  instructions the C path finished on its own.

Ceilings carry headroom so ordinary churn does not trip them; they exist to
catch a mechanism disappearing, not to police single-digit drift. Raising one
is a deliberate act that belongs in a commit message with a reason.

The binaries come from benchmark/taint_density (see its README); the test skips
if they have not been built, so a checkout without a compiler still runs green.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pytest

from qiling import Qiling
from qiling.const import QL_VERBOSE

from microtaint.emulator.reporter import Reporter
from microtaint.emulator.wrapper import MicrotaintWrapper

_BENCH_DIR = Path(__file__).resolve().parent.parent / 'benchmark' / 'taint_density'

# binary -> (max cells, max Python fallbacks, min fast-path fraction,
#            min fraction dismissed as untainted)
# Measured values at the time of writing are roughly half of each ceiling for
# cells and fallbacks, so these trip on a mechanism breaking, not on noise.
# Baselined AFTER the PC-relative memory-taint fix. Before it, taint entering
# through a RIP-relative load was silently dropped, so the untainted-input exit
# dismissed instructions it had no right to dismiss and these counters looked far
# better than the engine deserved (bench_untainted read 43 cells; the honest
# figure is 36 only because the guest genuinely never reads its tainted bytes).
# A ratchet baselined on buggy behaviour would have locked the bug in, so these
# numbers are only meaningful together with tests/test_pc_relative_mem_taint.py.
LIMITS = {
    'bench_untainted.elf': (500, 500, 0.99, 0.85),
    'bench_sparse.elf':    (5_000, 500, 0.99, 0.85),
    'bench_dense.elf':     (200_000, 1_000, 0.99, 0.60),
}


def _counters(elf: Path):
    stdin = bytes((i * 7 + 13) & 0xFF for i in range(64))
    ql = Qiling([str(elf)], '/', verbose=QL_VERBOSE.OFF)
    ql.os.stdin = io.BytesIO(stdin)
    w = MicrotaintWrapper(ql, reporter=Reporter(json_mode=False, stream=io.StringIO()))
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
    h = w._instr_hook_obj  # noqa: SLF001
    if h is None:
        pytest.skip('Cython instruction hook not installed')
    return h, w.sim._pcode.native_calls  # noqa: SLF001


@pytest.mark.parametrize('name', sorted(LIMITS))
def test_hot_path_coverage_ratchet(name: str) -> None:
    elf = _BENCH_DIR / name
    if not elf.exists():
        pytest.skip(f'{name} not built (run `make` in {_BENCH_DIR})')
    max_cells, max_fb, min_fast, min_pre = LIMITS[name]
    h, cells = _counters(elf)

    total = h.instr_total
    assert total > 1000, f'{name}: only {total} instructions hooked; the workload did not run'

    fb = h.fb_pc + h.fb_pyfall + h.fb_mem + h.fb_other
    fast_frac = h.fast_done / total
    pre_frac = h.prefilter_hits / total

    assert cells <= max_cells, (
        f'{name}: {cells} SLEIGH cell re-executions, ceiling {max_cells}. '
        f'Instructions that used to be dismissed or handled in C are being '
        f're-executed again -- a hot-path mechanism stopped applying.'
    )
    assert fb <= max_fb, (
        f'{name}: {fb} Python fallbacks, ceiling {max_fb} '
        f'(pc={h.fb_pc} pyfall={h.fb_pyfall} mem={h.fb_mem} other={h.fb_other}). '
        f'Instructions are leaving the C path again.'
    )
    assert fast_frac >= min_fast, (
        f'{name}: C fast path finished only {fast_frac:.1%} of instructions, '
        f'floor {min_fast:.0%}.'
    )
    assert pre_frac >= min_pre, (
        f'{name}: only {pre_frac:.1%} of instructions were dismissed as having '
        f'no tainted input, floor {min_pre:.0%}. The untainted-input exit is '
        f'covering less than it should.'
    )
