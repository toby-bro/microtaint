"""
conftest_use_c.py — pytest plugin that forces all CellSimulator instances
to use the pure-C evaluator. Used to verify drop-in correctness.

Activate with:
    PYTHONPATH=... pytest -p tests.conftest_use_c tests/

Or copy contents into tests/conftest.py for a one-shot run.
"""

from __future__ import annotations

import microtaint.simulator as sim_mod

_original_init = sim_mod.CellSimulator.__init__


def _patched_init(self, arch, use_unicorn=False, use_c=False):  # type: ignore[no-untyped-def]
    return _original_init(self, arch, use_unicorn=use_unicorn, use_c=True)


# Deliberate: this conftest exists to force the C evaluator for a whole run.
sim_mod.CellSimulator.__init__ = _patched_init  # type: ignore[method-assign]
