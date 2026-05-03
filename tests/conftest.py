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
