"""RIP-relative (PC-relative / absolute-address) memory taint soundness.

x86-64 `mov eax,[rip+disp]` / `mov [rip+disp],al` lift to a LOAD/STORE whose
pointer is a CONSTANT absolute address (the assembler folds rip+disp at
translate time), not a register expression. The address resolver only produced
a MemMapping for register-based pointers, so RIP-relative accesses created no
memory dependency at all:

  * a RIP-relative LOAD from tainted memory returned an UNTAINTED register
    (missed taint -> under-taint / unsound), and
  * a RIP-relative STORE of a tainted register wrote NO tainted shadow byte
    (dropped taint -> under-taint / unsound).

This is the mechanism behind the DNS end-to-end sink dropping taint (the `out`
global is written RIP-relative). Reg-based loads/stores are unaffected. These
tests fail until the resolver handles constant absolute addresses.
"""

import pytest

from microtaint.emulator.shadow import BitPreciseShadowMemory
from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, Register

# generate_static_rule translates the bytes at this base; RIP-relative
# displacements resolve relative to the address after the instruction.
_TRANSLATE_BASE = 0x1000


@pytest.fixture(scope='module')
def simulator() -> CellSimulator:
    return CellSimulator(Architecture.AMD64)


@pytest.fixture(scope='module')
def regs() -> list[Register]:
    return [Register(name=n, bits=64) for n in ('RAX', 'RBX', 'RCX', 'RDX', 'RBP', 'RSP', 'RIP')]


def test_riprel_load_reads_tainted_global(simulator: CellSimulator, regs: list[Register]) -> None:
    """MOV EAX, [RIP+0] (8B 05 00000000) from a tainted global must taint EAX."""
    circuit = generate_static_rule(Architecture.AMD64, bytes.fromhex('8b0500000000'), regs)
    shadow = BitPreciseShadowMemory()
    # Taint a wide window around every plausible RIP-relative target so the
    # assertion cannot pass or fail on an address off-by-one.
    for addr in range(_TRANSLATE_BASE - 0x100, _TRANSLATE_BASE + 0x100):
        shadow.write_mask(addr, 0xFF, 1)
    ctx = EvalContext(
        input_values={'RIP': _TRANSLATE_BASE, 'RAX': 0},
        input_taint={},
        simulator=simulator,
        shadow_memory=shadow,
    )
    out = circuit.evaluate(ctx)
    assert out.get('RAX', 0) != 0, (
        'RIP-relative load from tainted memory must taint the destination '
        '(under-taint / unsound if 0)'
    )


def test_riprel_store_taints_global(simulator: CellSimulator, regs: list[Register]) -> None:
    """MOV [RIP+0], AL (88 05 00000000) of a tainted AL must taint the target byte."""
    circuit = generate_static_rule(Architecture.AMD64, bytes.fromhex('880500000000'), regs)
    shadow = BitPreciseShadowMemory()
    ctx = EvalContext(
        input_values={'RIP': _TRANSLATE_BASE, 'RAX': 0},
        input_taint={'RAX': 0xFF},  # AL tainted
        simulator=simulator,
        shadow_memory=shadow,
    )
    out = circuit.evaluate(ctx)
    tainted_mem = {k: v for k, v in out.items() if k.startswith('MEM') and v}
    assert tainted_mem, (
        'RIP-relative store of a tainted register must produce a tainted memory '
        'output (dropped taint / unsound if none)'
    )
