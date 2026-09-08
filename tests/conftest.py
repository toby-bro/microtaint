"""tests/conftest.py — when MICROTAINT_USE_C=1, force all CellSimulator
instances to use the pure-C evaluator. This validates the C module as a
drop-in replacement for the Cython evaluator.
"""

# mypy: disable-error-code="method-assign,no-untyped-def"
import os
import sys

# Make cell_c importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'microtaint', 'instrumentation', 'cell_c'))

if os.environ.get('MICROTAINT_USE_C') == '1':
    import microtaint.simulator as sim_mod

    _original_init = sim_mod.CellSimulator.__init__

    def _patched_init(self, arch, use_unicorn=False, use_c=False):  # noqa: ARG001
        return _original_init(self, arch, use_unicorn=use_unicorn, use_c=True)

    sim_mod.CellSimulator.__init__ = _patched_init


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
import pytest

_SLOW_ENV = 'MICROTAINT_SLOW_TESTS'


def pytest_addoption(parser):
    parser.addoption(
        '--slow', action='store_true', default=False,
        help='also run the release-grade tests: the full-bank soundness sweeps '
             'and the fuzzers at full budget.')


def slow_tier_enabled(config=None) -> bool:
    """Is this a release-grade run?

    Honours an environment variable as well as the flag, because the gate and
    CI drive pytest through wrappers that already pass `-o addopts=...`, and one
    more positional flag to thread through each of them is one more place to
    forget it.
    """
    if os.environ.get(_SLOW_ENV, '') not in ('', '0'):
        return True
    return bool(config is not None and config.getoption('--slow', default=False))


def fuzz_budget(full: int, config=None) -> int:
    """How many random vectors this run should draw.

    Full budget at release; a small deterministic slice otherwise.  Never zero:
    a fuzzer reduced to nothing is a test that passes because it did nothing.
    """
    return full if slow_tier_enabled(config) else max(1, full // 4)


def pytest_configure(config):
    config.addinivalue_line(
        'markers',
        'slow: a release-grade check (full-bank sweep or full-budget fuzz); '
        'deselected unless --slow or MICROTAINT_SLOW_TESTS=1')


def pytest_collection_modifyitems(config, items):
    if slow_tier_enabled(config):
        return
    skip = pytest.mark.skip(
        reason='release-grade; run with --slow or MICROTAINT_SLOW_TESTS=1')
    for item in items:
        if 'slow' in item.keywords:
            item.add_marker(skip)
