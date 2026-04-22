from __future__ import annotations

from microtaint.instrumentation.ast import InstructionCellExpr
from microtaint.simulator import CellSimulator
from microtaint.types import Architecture


def test_cell_simulator_add() -> None:
    sim = CellSimulator(Architecture.AMD64)
    # ADD EAX, EBX -> \x01\xd8
    bytestring = b'\x01\xd8'

    # Let's say v_state: RAX=10, RBX=5
    # t_state: RAX=0, RBX=0xffffffff
    # Expected output taint for RAX: it should see that bits of RBX affect RAX.

    # We just want to make sure it runs without crashing for now and returns an int.
    v_state = {'RAX': 10, 'RBX': 5}
    t_state = {'RAX': 0, 'RBX': 0xFFFFFFFF}

    out_taint = sim.evaluate_cell_differential(bytestring, 'RAX', v_state, t_state)
    assert isinstance(out_taint, int)

    # It should be 0xFFFFFFFF (or 0xFFFFFFFFFFFFFFFF depending on how ADD EAX, EBX propagates over RAX)
    # Actually ADD EAX, EBX clears upper 32-bits of RAX. So it should propagate at least the lower 32-bits.
    assert out_taint != 0


def test_unicorn_simulator_x86_add() -> None:
    # ADD EAX, EBX -> 01 d8
    cell = InstructionCellExpr(Architecture.X86, '01d8', 'EAX', 0, 31, {})
    sim = CellSimulator(Architecture.X86)

    # 0x10 + 0x20 = 0x30
    res = sim.evaluate_concrete(cell, {'EAX': 0x10, 'EBX': 0x20})
    assert res == 0x30


def test_unicorn_simulator_x86_flags() -> None:
    # ADD EAX, EBX -> 01 d8
    # 0xffffffff + 1 -> ZF = 1, CF = 1
    cell_zf = InstructionCellExpr(Architecture.X86, '01d8', 'ZF', 0, 7, {})
    cell_cf = InstructionCellExpr(Architecture.X86, '01d8', 'CF', 0, 7, {})

    sim = CellSimulator(Architecture.X86)
    res_zf = sim.evaluate_concrete(cell_zf, {'EAX': 0xFFFFFFFF, 'EBX': 1})
    res_cf = sim.evaluate_concrete(cell_cf, {'EAX': 0xFFFFFFFF, 'EBX': 1})

    assert res_zf == 1
    assert res_cf == 1


def test_unicorn_simulator_amd64_mov() -> None:
    # MOV RAX, RBX -> 48 89 d8
    cell = InstructionCellExpr(Architecture.AMD64, '4889d8', 'RAX', 0, 63, {})
    sim = CellSimulator(Architecture.AMD64)
    res = sim.evaluate_concrete(cell, {'RAX': 0, 'RBX': 0xDEADBEEF})
    assert res == 0xDEADBEEF


def test_unicorn_simulator_arm64_add() -> None:
    # ADD X0, X1, X2 -> 20 00 02 8b
    cell = InstructionCellExpr(Architecture.ARM64, '2000028b', 'X0', 0, 63, {})
    sim = CellSimulator(Architecture.ARM64)
    res = sim.evaluate_concrete(cell, {'X0': 0, 'X1': 50, 'X2': 100})
    assert res == 150
