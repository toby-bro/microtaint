"""tests/test_pc_relative_mem_taint.py
=====================================
PC-relative memory operands must carry taint at the program's REAL load address.

Position-independent x86-64 code reaches globals through RIP-relative operands
(`movzbl 0x2004(%rip),%eax`), so this is not a corner case: it is how essentially
every modern binary touches static data. ARM64 and PPC literal pools are the same
shape.

The rule for such an instruction is built once at a fixed lift base
(`_TRANSLATE_BASE`, 0x1000) and the engine rebases the operand onto the runtime PC
register, which is correct in principle. What this pins down is that taint only
actually survives when the runtime PC EQUALS that lift base. At any other PC --
i.e. every real program, which loads at 0x400000 and up -- the taint is lost.

Consequence: a value loaded from tainted memory through a RIP-relative operand
arrives untainted, and every dependency downstream of it is silently dropped.

Fixed by resolving PC-relative MEM_ inputs against the cell's lift base rather
than the runtime PC, in both the flat and fast paths of the cell kernel. The
p-code is lifted at a fixed base, so a PC-relative operand is BAKED into it at
that base; seeding the operand at the runtime PC left the load reading untouched
memory, both differential corners saw identical bytes, and the taint vanished.
It only ever worked when the runtime PC happened to equal the lift base.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'benchmark'))
from instruction_bank import isa_registers  # type: ignore[import-not-found]

from microtaint.emulator.shadow import BitPreciseShadowMemory
from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, ImplicitTaintPolicy

# movzbl 0x2004(%rip),%eax -- 7 bytes, so the operand resolves to PC + 7 + 0x2004.
MOVZBL_RIPREL = bytes([0x0F, 0xB6, 0x05, 0x04, 0x20, 0x00, 0x00])
DISP_FROM_PC = 0x200B  # 0x2004 + instruction length
LIFT_BASE = 0x1000     # engine's _TRANSLATE_BASE


def _propagates(pc: int) -> bool:
    """True if a tainted byte at the operand's effective address taints RAX."""
    arch = Architecture.AMD64
    regs = list(isa_registers(arch))
    sim = CellSimulator(arch)
    circ = generate_static_rule(arch, MOVZBL_RIPREL, regs)

    values = {r.name: 0 for r in regs}
    values['RIP'] = pc
    shadow = BitPreciseShadowMemory()
    shadow.write_mask(pc + DISP_FROM_PC, 0xFF, 1)   # the byte the load reads

    ectx = EvalContext(
        input_values=values,
        input_taint={r.name: 0 for r in regs},
        simulator=sim,
        implicit_policy=ImplicitTaintPolicy.KEEP,
        shadow_memory=shadow,
        # A real reader: returning a constant 0 for everything is enough to make
        # the cell path look broken when it is not, which cost real debugging
        # time once already.
        mem_reader=lambda a, s: sum(((0x11 * (i + 1)) & 0xFF) << (8 * i) for i in range(s)),
    )
    return any(v for v in circ.evaluate(ectx).values())


def test_pc_relative_load_at_lift_base_propagates() -> None:
    """Control: at the lift base it works, so the corpus and rule are sound."""
    assert _propagates(LIFT_BASE), (
        'even at the lift base a RIP-relative load did not carry taint; the '
        'failure is broader than the PC rebasing and these tests are mis-aimed'
    )


@pytest.mark.parametrize('pc', [0x401018, 0x400000, 0x1001, 0x8000, 0x555555555000])
def test_pc_relative_load_propagates_at_real_addresses(pc: int) -> None:
    assert _propagates(pc), (
        f"PC={pc:#x}: a byte tainted at the operand's effective address "
        f'({pc + DISP_FROM_PC:#x}) did not taint the destination. Every value '
        f'a real program loads from a global through a RIP-relative operand '
        f'arrives untainted.'
    )
