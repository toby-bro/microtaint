"""Regression for the RMW transport-OR floor removal (commit 94c91ca).

A relocating read-modify-write on memory (e.g. `shr dword [rbp-4], 1`) must
taint the EXACT relocated bits, not OR the original taint back at its old
positions.  The deleted floor OR'd each value-dependency back into the output at
its ORIGINAL bit positions, so a right shift by 1 kept bit 31 tainted
(0xffffffff) instead of clearing it (the shifted-in constant 0).  The exact
taint of `>>1` on 32 fully-tainted bits is 0x7fffffff.

Runs entirely in-process (generate_static_rule + EvalContext), like
test_store_clearing.py.
"""

from __future__ import annotations

import pytest

from microtaint.emulator.shadow import BitPreciseShadowMemory
from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, ImplicitTaintPolicy, Register

X64 = [Register(n, 64) for n in ('RAX', 'RBX', 'RCX', 'RDX', 'RBP', 'RSP', 'RIP')] + [
    Register('EFLAGS', 32), Register('ZF', 1), Register('CF', 1),
    Register('SF', 1), Register('OF', 1), Register('PF', 1),
]
RBP_VAL = 0x80000200


@pytest.fixture(scope='module')
def sim() -> CellSimulator:
    return CellSimulator(Architecture.AMD64)


def _mem_outputs(hex_bytes: str, shadow: BitPreciseShadowMemory, sim: CellSimulator) -> dict[str, int]:
    circuit = generate_static_rule(Architecture.AMD64, bytes.fromhex(hex_bytes), X64)
    ctx = EvalContext(
        input_taint={},
        input_values={'RBP': RBP_VAL},
        simulator=sim,
        implicit_policy=ImplicitTaintPolicy.KEEP,
        shadow_memory=shadow,
        mem_reader=lambda addr, sz: 0,  # noqa: ARG005
    )
    out = circuit.evaluate(ctx)
    return {k: v for k, v in out.items() if k.startswith('MEM_')}


def test_shr_mem_relocates_taint_exactly(sim: CellSimulator) -> None:
    """`shr dword [rbp-4], 1` (D1 6D FC) on fully-tainted memory taints T>>1, not T."""
    addr = RBP_VAL - 4
    shadow = BitPreciseShadowMemory()
    shadow.write_mask(addr, 0xFFFFFFFF, 4)  # 32-bit memory fully tainted

    mem = _mem_outputs('D16DFC', shadow, sim)

    assert len(mem) == 1, f'shr [mem],1 must produce exactly one MEM_ output; got {mem}'
    taint = next(iter(mem.values()))
    # Exact relocated taint of a right shift by 1: bit 31 becomes the shifted-in
    # constant 0, every other bit moves down one.  The old floor would OR the
    # original mask back -> 0xffffffff (bit 31 wrongly tainted).
    assert taint == 0x7FFFFFFF, (
        f'relocating RMW must taint exactly T>>1 = 0x7fffffff; the transport-OR '
        f'floor would over-taint to 0xffffffff. got {hex(taint)}'
    )
