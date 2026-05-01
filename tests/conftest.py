"""tests/conftest.py — make the cell_c hand-written extension importable
during test runs (it lives in a non-package directory).

The runtime default is now to use the C kernel.  To force the Cython
kernel for differential testing, set MICROTAINT_DISABLE_C_KERNEL=1.

Legacy MICROTAINT_USE_C=1 (which used to FORCE the C kernel) is now a
no-op since the C kernel is already the default.
"""
import os
import sys

# Make cell_c importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'microtaint',
                                 'instrumentation', 'cell_c'))
