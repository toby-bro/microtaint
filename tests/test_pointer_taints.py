from __future__ import annotations

import pytest

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, Register


@pytest.fixture
def amd64_registers() -> list[Register]:
    return [
        Register(name='RAX', bits=64),
        Register(name='RBX', bits=64),
        Register(name='RCX', bits=64),
        Register(name='RDX', bits=64),
        Register(name='RSP', bits=64),
        Register(name='RBP', bits=64),
        Register(name='RIP', bits=64),
        Register(name='EFLAGS', bits=32),
    ]


def test_mov_clean_pointer_propagates_exact_memory_taint(amd64_registers: list[Register]) -> None:
    """
    MOV RAX, [RBX].
    Pointer is clean. Memory has partial taint.
    RAX should inherit exact partial taint. Fast-path used.
    """
    sim = CellSimulator(Architecture.AMD64)
    bytestring = bytes.fromhex('488b03')  # MOV RAX, QWORD PTR [RBX]
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)

    ctx = EvalContext(
        input_values={'RBX': 0x1000},
        input_taint={'RBX': 0, 'MEM_0x1000_8': 0x00000000000000FF},  # Low 8 bits tainted
        simulator=sim,
    )

    out = circuit.evaluate(ctx)
    assert out.get('RAX', 0) == 0xFF


def test_mov_tainted_pointer_avalanches_output(amd64_registers: list[Register]) -> None:
    """
    MOV RAX, [RBX].
    Memory is clean. Pointer is tainted by just 1 bit.
    RAX should completely avalanche to 0xFFFFFFFFFFFFFFFF.
    """
    sim = CellSimulator(Architecture.AMD64)
    bytestring = bytes.fromhex('488b03')  # MOV RAX, QWORD PTR [RBX]
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)

    ctx = EvalContext(
        input_values={'RBX': 0x1000},
        input_taint={'RBX': 0x01, 'MEM_0x1000_8': 0},  # Memory is clean, pointer is compromised
        simulator=sim,
    )

    out = circuit.evaluate(ctx)
    assert out.get('RAX', 0) == 0xFFFFFFFFFFFFFFFF


def test_add_tainted_pointer_avalanches_output(amd64_registers: list[Register]) -> None:
    """
    ADD RAX, [RBX].
    RAX is clean. Memory is clean. Pointer is tainted.
    Since it pulled an unknown value from an arbitrary address, the result
    must be fully avalanched.
    """
    sim = CellSimulator(Architecture.AMD64)
    bytestring = bytes.fromhex('480303')  # ADD RAX, QWORD PTR [RBX]
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)

    ctx = EvalContext(
        input_values={'RAX': 0, 'RBX': 0x1000},
        input_taint={'RAX': 0, 'RBX': 0x01, 'MEM_0x1000_8': 0},
        simulator=sim,
    )

    out = circuit.evaluate(ctx)
    assert out.get('RAX', 0) == 0xFFFFFFFFFFFFFFFF
