"""Comprehensive tests for x86 flag taint propagation."""

from __future__ import annotations

from microtaint.simulator import CellSimulator, MachineState
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, Register


def test_add_taints_all_flags() -> None:
    arch = Architecture.X86
    code = bytes.fromhex('01d8')
    regs = [
        Register(name='EAX', bits=32),
        Register(name='EBX', bits=32),
        Register(name='CF', bits=1),
        Register(name='PF', bits=1),
        Register(name='ZF', bits=1),
        Register(name='SF', bits=1),
        Register(name='OF', bits=1),
    ]
    rule = generate_static_rule(arch, code, regs)
    targets = {a.target.name for a in rule.assignments}
    assert 'CF' in targets
    assert 'PF' in targets
    assert 'ZF' in targets
    assert 'SF' in targets
    assert 'OF' in targets


def test_test_taints_flags() -> None:
    arch = Architecture.X86
    code = bytes.fromhex('85d8')
    regs = [
        Register(name='EAX', bits=32),
        Register(name='EBX', bits=32),
        Register(name='CF', bits=1),
        Register(name='PF', bits=1),
        Register(name='ZF', bits=1),
        Register(name='SF', bits=1),
        Register(name='OF', bits=1),
    ]
    rule = generate_static_rule(arch, code, regs)
    targets = {a.target.name for a in rule.assignments}
    assert 'PF' in targets
    assert 'ZF' in targets
    assert 'SF' in targets


def test_simulator_flag_taint_add() -> None:
    arch = Architecture.X86
    code = bytes.fromhex('01d8')
    sim = CellSimulator(arch)
    v_state = MachineState({'EAX': 0xFF, 'EBX': 0x01})
    t_state = MachineState({'EAX': 0x00, 'EBX': 0x01})

    assert sim.evaluate_cell_differential(code, 'EAX', v_state, t_state) != 0
    assert sim.evaluate_cell_differential(code, 'CF', v_state, t_state) != 0
    assert sim.evaluate_cell_differential(code, 'ZF', v_state, t_state) != 0
    assert sim.evaluate_cell_differential(code, 'SF', v_state, t_state) != 0
    assert sim.evaluate_cell_differential(code, 'OF', v_state, t_state) != 0


def test_simulator_flag_taint_cmp() -> None:
    arch = Architecture.X86
    code = bytes.fromhex('39d8')
    sim = CellSimulator(arch)
    v_state = MachineState({'EAX': 0x10, 'EBX': 0x10})
    t_state = MachineState({'EAX': 0xEF, 'EBX': 0x10})

    assert sim.evaluate_cell_differential(code, 'ZF', v_state, t_state) != 0
    assert sim.evaluate_cell_differential(code, 'CF', v_state, t_state) != 0
    assert sim.evaluate_cell_differential(code, 'SF', v_state, t_state) != 0


def test_simulator_flag_taint_test() -> None:
    arch = Architecture.X86
    code = bytes.fromhex('85d8')
    sim = CellSimulator(arch)
    v_state = MachineState({'EAX': 0xFF, 'EBX': 0xFF})
    t_state = MachineState({'EAX': 0x00, 'EBX': 0xFF})

    assert sim.evaluate_cell_differential(code, 'ZF', v_state, t_state) != 0
    assert sim.evaluate_cell_differential(code, 'PF', v_state, t_state) != 0
    assert sim.evaluate_cell_differential(code, 'SF', v_state, t_state) != 0
