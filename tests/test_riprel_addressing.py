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
    return [Register(name=n, bits=64) for n in ('RAX', 'RBX', 'RCX', 'RDX', 'RBP', 'RSP', 'RIP')] + [
        Register('EFLAGS', 32), Register('CF', 1), Register('ZF', 1),
        Register('SF', 1), Register('OF', 1), Register('PF', 1),
    ]


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


def test_riprel_arith_flows_memory_operand(simulator: CellSimulator, regs: list[Register]) -> None:
    """ADD EAX, [RIP+0] (03 05 00000000) must flow the ABSOLUTE memory operand's
    taint through the arithmetic (carry-aware differential), not merely transport
    it.  This is the case a transport-only shortcut gets wrong: the memory operand
    participates in the INT_ADD carry chain with EAX, so it MUST go through the
    same differential machinery as a register / register-relative operand.
    """
    circuit = generate_static_rule(Architecture.AMD64, bytes.fromhex('030500000000'), regs)
    shadow = BitPreciseShadowMemory()
    for addr in range(_TRANSLATE_BASE - 0x100, _TRANSLATE_BASE + 0x100):
        shadow.write_mask(addr, 0xFF, 1)  # the global is tainted, EAX is not
    ctx = EvalContext(
        input_values={'RIP': _TRANSLATE_BASE, 'RAX': 0},
        input_taint={},
        simulator=simulator,
        shadow_memory=shadow,
        mem_reader=lambda addr, sz: 0,  # noqa: ARG005
    )
    out = circuit.evaluate(ctx)
    assert out.get('RAX', 0) != 0, (
        'absolute memory operand must flow through arithmetic; under-taint if 0'
    )


def test_arm64_pcrel_literal_load_reads_tainted() -> None:
    """Cross-ISA: ARM64 `ldr w0, #8` (40 00 00 18) loads a PC-relative literal
    whose address folds to a compile-time constant (a LOAD with a constant
    pointer, resolve_ptr_with_offset -> (None, addr)).  A tainted literal must
    taint w0.  Same constant-address handling as x86 RIP-relative -- one fix
    covers every ISA that lifts absolute / PC-relative addressing to a folded
    constant address.
    """
    arm_sim = CellSimulator(Architecture.ARM64)
    arm_regs = [Register(f'x{i}', 64) for i in range(31)] + [Register('sp', 64), Register('pc', 64)]
    circuit = generate_static_rule(Architecture.ARM64, bytes.fromhex('40000018'), arm_regs)
    shadow = BitPreciseShadowMemory()
    for addr in range(0x1000, 0x1020):
        shadow.write_mask(addr, 0xFF, 1)  # the literal (at PC+8) is tainted
    ctx = EvalContext(
        input_values={'pc': 0x1000},
        input_taint={},
        simulator=arm_sim,
        shadow_memory=shadow,
        mem_reader=lambda addr, sz: 0,  # noqa: ARG005
    )
    out = circuit.evaluate(ctx)
    assert out.get('x0', 0) != 0, (
        'ARM64 PC-relative literal load from tainted memory must taint the '
        'destination (under-taint / unsound if 0)'
    )
