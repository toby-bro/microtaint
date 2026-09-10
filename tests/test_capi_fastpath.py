"""The compiled-circuit fast cell path must be active on a NORMAL import.

circuit_c calls cell_eval_fast (the fast C cell path) via a CAPI capsule it
imports at module init.  Historically it only imported when `cell_c` was
importable as a TOP-LEVEL module (tests/conftest.py puts its directory on
sys.path); a normal import (the emulator) left the CAPI unloaded and every
OP_CALL_CELL fell back to the ~3x-slower Python evaluate_concrete path.

circuit_c now also imports cell_c by its full dotted name, so the fast path
works with no sys.path dependency.  This test runs a FRESH interpreter (which
does NOT inherit conftest's sys.path insert) to prove it.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap


def test_cell_capi_loaded_without_cell_c_on_syspath() -> None:
    code = textwrap.dedent(
        """
        import sys
        import microtaint.instrumentation.cell_c.circuit_c as c
        assert 'cell_c' not in sys.modules, 'cell_c unexpectedly top-level'
        print('CAPI_LOADED' if c.cell_capi_loaded() else 'CAPI_SLOW')
        """,
    )
    out = subprocess.run(
        [sys.executable, '-c', code], capture_output=True, text=True, timeout=120, check=False,
    )
    assert 'CAPI_LOADED' in out.stdout, (
        f'fast cell path not active on a normal import:\n{out.stdout}\n{out.stderr}'
    )
