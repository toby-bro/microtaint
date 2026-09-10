"""Regression: frame-recycle must stay bit-exact for control-flow circuits.

Bug (fixed)
-----------
The frame-recycle pool frames (``EvalC.frame_pool``) were created without the
``arch_pc_off`` / ``arch_pc_sz`` PC seed that ``frame_a`` / ``frame_b`` receive.
A relative branch executed in a pool frame therefore computed its target from
PC=0 and the recycled frame reported a wrong (zero) PC.  For a tainted
conditional branch (``test rdi,1 ; jz``) every polarity corner of the PC
differential then read the same zero PC, so the control-flow taint collapsed to
0.  With recycle ON (the default) the implicit-taint interceptor saw an
untainted PC and did NOT strip / warn / stop, silently losing implicit taint.

The bank compiled-vs-walker validation never caught it because the instruction
bank has no branches; this test pins the invariant directly.

The fix seeds ``arch_pc_off`` / ``arch_pc_sz`` on every pool frame, so a
recycled branch computes the same PC as the unshared path.
"""
# ruff: noqa: S603

from __future__ import annotations

import subprocess
import sys

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, ImplicitTaintPolicy, Register

# test rdi, 1 ; jz +3  -- ZF (hence the branch target / PC) depends on RDI.
_BRANCH_BYTES_HEX = '48f7c7010000007403'
_REGS = ('RDI', 'RIP')


def _eval_pc_taint() -> int:
    """PC taint of the tainted conditional branch, KEEP policy (no stripping),
    using the engine exactly as configured in this process."""
    regs = [Register(name=r, bits=64) for r in _REGS]
    circ = generate_static_rule(Architecture.AMD64, bytes.fromhex(_BRANCH_BYTES_HEX), regs)
    sim = CellSimulator(Architecture.AMD64)
    ectx = EvalContext(
        input_values={'RDI': 1, 'RIP': 0x400000},
        input_taint={'RDI': 0xFFFFFFFFFFFFFFFF, 'RIP': 0},
        simulator=sim,
        implicit_policy=ImplicitTaintPolicy.KEEP,
    )
    return circ.evaluate(ectx).get('RIP', 0)


def _eval_pc_taint_recycle_off() -> int:
    """Same, in a fresh process with frame-recycle disabled (the reference)."""
    code = (
        'from tests.test_frame_recycle_control_flow import _eval_pc_taint;'
        'print(_eval_pc_taint())'
    )
    out = subprocess.check_output(
        [sys.executable, '-c', code],
        env={'MICROTAINT_RECYCLE_FRAMES': '0', 'PATH': '/usr/bin:/bin'},
        cwd='.',
        text=True,
    )
    return int(out.strip())


def test_control_flow_taint_survives_frame_recycle() -> None:
    """With recycle ON (default) the branch's PC must be tainted (non-zero) and
    bit-identical to the recycle-OFF reference."""
    on = _eval_pc_taint()
    off = _eval_pc_taint_recycle_off()
    assert off != 0, 'reference (recycle off) should taint the PC of a tainted branch'
    assert on == off, (
        f'frame-recycle changed the control-flow PC taint: recycle-on={on:#x} '
        f'recycle-off={off:#x}. The pool frames are not bit-exact for branches.'
    )


def test_implicit_taint_stripped_under_recycle() -> None:
    """The IGNORE policy must still strip a tainted PC when recycle is on."""
    regs = [Register(name=r, bits=64) for r in _REGS]
    circ = generate_static_rule(Architecture.AMD64, bytes.fromhex(_BRANCH_BYTES_HEX), regs)
    sim = CellSimulator(Architecture.AMD64)
    ectx = EvalContext(
        input_values={'RDI': 1, 'RIP': 0x400000},
        input_taint={'RDI': 0xFFFFFFFFFFFFFFFF, 'RIP': 0},
        simulator=sim,
        implicit_policy=ImplicitTaintPolicy.IGNORE,
    )
    assert 'RIP' not in circ.evaluate(ectx)
