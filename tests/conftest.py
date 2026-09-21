"""tests/conftest.py — when MICROTAINT_USE_C=1, force all CellSimulator
instances to use the pure-C evaluator. This validates the C module as a
drop-in replacement for the Cython evaluator.
"""

import atexit
import os
import sys

import pytest

from microtaint.instrumentation.cell import PCodeCellEvaluator
from microtaint.instrumentation.cell_c.cell_c import PCodeCellEvaluatorC
from microtaint.simulator import CellSimulator
from microtaint.sleigh.lifter import clear_contexts

# Make cell_c importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'microtaint', 'instrumentation', 'cell_c'))

# The instruction bank the perf and coverage tests draw their cases from.  It
# used to be imported as `benchmark.instruction_bank`, from a `benchmark/`
# directory that was never in the repository, so twenty-two test modules failed
# at collection in a clean clone and `uv run pytest` did not start.  The bank
# itself is here, under scripts/sweep/pinned/, pinned by MANIFEST.sha256
# because the perf ratchet compares against it.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts', 'sweep', 'pinned'))

if os.environ.get('MICROTAINT_USE_C') == '1':
    import microtaint.simulator as sim_mod

    _original_init = sim_mod.CellSimulator.__init__

    from microtaint.types import Architecture

    def _patched_init(self: CellSimulator, arch: Architecture,
                      use_unicorn: bool = False,
                      use_c: bool | None = False) -> None:
        _original_init(self, arch, use_unicorn=use_unicorn, use_c=True)

    # Deliberate: this conftest exists to force the C evaluator for a run.
    sim_mod.CellSimulator.__init__ = _patched_init  # type: ignore[method-assign]


# A SLEIGH context holds a lot of nanobind-managed state -- 1441 instances for
# x86-64 alone -- and nanobind reports it at shutdown as leaked.  It is a cache
# that was never emptied rather than a leak, but the message is
# indistinguishable from a real refcount bug and buries one, so the test run
# empties it.  Three block-mode end-to-end files are what trip it today.
#
# Registered HERE and not in the library: the destructors cost ~16 ms for one
# architecture and ~57 ms for all seven, which a process about to exit would
# otherwise get for free from the kernel.  Paying that once per test session is
# nothing; paying it once per execution of a fuzzing target is not.
atexit.register(clear_contexts)


# ---------------------------------------------------------------------------
# Two tiers.
#
# The suite's wall time under xdist is set by its LONGEST test, not by its
# total: four full-bank sweeps in test_oracle_harness take 204, 150, 87 and 61
# seconds, so nothing else matters until they move.  They are release-grade
# checks -- "is the engine sound across 1500 instructions and many vectors" --
# and the answer does not change between two adjacent commits touching an
# emitter.  Running them on every edit, twice over (once per implementation),
# spends most of the day re-answering a settled question.
#
# So they are marked `slow` and deselected unless asked for.  The randomised
# fuzzers stay in the fast tier at a REDUCED, FIXED-SEED budget rather than
# being removed: a fuzzer that never runs finds nothing, and one that runs a few
# deterministic vectors still catches the gross breakage that matters on an
# edit, while the full budget runs at release.
#
# The rule that keeps this honest: the slow tier is MANDATORY on a release, not
# merely available.  A marker nobody is obliged to run is a marker that quietly
# deletes its tests.

_SLOW_ENV = 'MICROTAINT_SLOW_TESTS'


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        '--slow', action='store_true', default=False,
        help='also run the release-grade tests: the full-bank soundness sweeps '
             'and the fuzzers at full budget.')


def slow_tier_enabled(config: pytest.Config | None = None) -> bool:
    """Is this a release-grade run?

    Honours an environment variable as well as the flag, because the gate and
    CI drive pytest through wrappers that already pass `-o addopts=...`, and one
    more positional flag to thread through each of them is one more place to
    forget it.
    """
    if os.environ.get(_SLOW_ENV, '') not in ('', '0'):
        return True
    return bool(config is not None and config.getoption('--slow', default=False))


def fuzz_budget(full: int, config: pytest.Config | None = None) -> int:
    """How many random vectors this run should draw.

    Full budget at release; a small deterministic slice otherwise.  Never zero:
    a fuzzer reduced to nothing is a test that passes because it did nothing.
    """
    return full if slow_tier_enabled(config) else max(1, full // 4)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        'markers',
        'slow: a release-grade check (full-bank sweep or full-budget fuzz); '
        'deselected unless --slow or MICROTAINT_SLOW_TESTS=1')
    config.addinivalue_line(
        'markers',
        'smoke: runs against an INSTALLED wheel on a bare runner, so it may '
        'use no fixture and no repository file but the guest beside it')


def pytest_collection_modifyitems(config: pytest.Config,
                                  items: list[pytest.Item]) -> None:
    if slow_tier_enabled(config):
        return
    skip = pytest.mark.skip(
        reason='release-grade; run with --slow or MICROTAINT_SLOW_TESTS=1')
    for item in items:
        if 'slow' in item.keywords:
            item.add_marker(skip)


# ---------------------------------------------------------------------------
# The cell evaluator behind a simulator.
#
# `CellSimulator._pcode` is None when the native evaluator could not be built.
# Reading a counter or calling an entry point through that None is not a
# failure a test should absorb: it means the kernel under test never ran, and
# the assertion that follows would be about nothing.


def cell_kernel(sim: CellSimulator) -> PCodeCellEvaluator | PCodeCellEvaluatorC:
    """Either cell evaluator, whichever this simulator built."""
    kernel = sim._pcode
    assert kernel is not None, (
        'this simulator has no native cell evaluator, so there is nothing to '
        'measure or compare')
    return kernel


def c_cell_kernel(sim: CellSimulator) -> PCodeCellEvaluatorC:
    """The pure-C evaluator specifically.

    Some entry points exist only there: the C-level shadow access goes through
    a C-API capsule the Cython kernel does not publish.
    """
    kernel = cell_kernel(sim)
    assert isinstance(kernel, PCodeCellEvaluatorC), (
        f'this entry point needs the C kernel; the simulator has '
        f'{type(kernel).__name__}')
    return kernel
