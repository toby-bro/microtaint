import pytest

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, Register


@pytest.fixture(scope='module')
def arm64_simulator() -> CellSimulator:
    return CellSimulator(Architecture.ARM64)


@pytest.fixture(scope='module')
def arm64_registers() -> list[Register]:
    return [
        Register(name='X0', bits=64),
        Register(name='X1', bits=64),
        Register(name='NZCV', bits=4),
        Register(name='N', bits=1),
        Register(name='Z', bits=1),
        Register(name='C', bits=1),
        Register(name='V', bits=1),
    ]


def extract_flag(ast_output: dict[str, int], flag_name: str) -> int:
    if flag_name in ast_output:
        return ast_output[flag_name]
    nzcv = ast_output.get('NZCV', 0)
    if flag_name == 'N':
        return (nzcv >> 31) & 1
    if flag_name == 'Z':
        return (nzcv >> 30) & 1
    if flag_name == 'C':
        return (nzcv >> 29) & 1
    if flag_name == 'V':
        return (nzcv >> 28) & 1
    return 0


# ==========================================
# ARM64: Dedicated Flag Taint Tests
# ==========================================


def test_arm64_cmp_z_flag_taint(arm64_simulator: CellSimulator, arm64_registers: list[Register]) -> None:
    """CMP X0, X1 -> 1F0001EB. Tests Zero flag taint propagation."""
    bytestring = bytes.fromhex('1F0001EB')
    circuit = generate_static_rule(Architecture.ARM64, bytestring, arm64_registers)
    ctx = EvalContext(
        input_values={'X0': 5, 'X1': 5},
        input_taint={'X0': 0x10, 'X1': 0},
        simulator=arm64_simulator,
    )
    assert extract_flag(circuit.evaluate(ctx), 'Z') == 1


def test_arm64_cmp_n_flag_taint(arm64_simulator: CellSimulator, arm64_registers: list[Register]) -> None:
    """CMP X0, X1 -> 1F0001EB. Tests Negative flag taint propagation."""
    bytestring = bytes.fromhex('1F0001EB')
    circuit = generate_static_rule(Architecture.ARM64, bytestring, arm64_registers)
    ctx = EvalContext(
        input_values={'X0': 0, 'X1': 1},
        input_taint={'X0': 0x10, 'X1': 0},
        simulator=arm64_simulator,
    )
    assert extract_flag(circuit.evaluate(ctx), 'N') == 1


def test_arm64_adds_c_flag_taint(arm64_simulator: CellSimulator, arm64_registers: list[Register]) -> None:
    """ADDS X0, X0, X1 -> 000001AB. Tests Carry flag taint propagation."""
    bytestring = bytes.fromhex('000001AB')
    circuit = generate_static_rule(Architecture.ARM64, bytestring, arm64_registers)
    ctx = EvalContext(
        input_values={'X0': 0xFFFFFFFFFFFFFFFF, 'X1': 1},
        input_taint={'X0': 0x10, 'X1': 0},
        simulator=arm64_simulator,
    )
    assert extract_flag(circuit.evaluate(ctx), 'C') == 1


def test_arm64_subs_v_flag_taint(arm64_simulator: CellSimulator, arm64_registers: list[Register]) -> None:
    """SUBS X0, X0, X1 -> 000001EB. Tests Overflow flag taint propagation."""
    bytestring = bytes.fromhex('000001EB')
    circuit = generate_static_rule(Architecture.ARM64, bytestring, arm64_registers)
    ctx = EvalContext(
        input_values={'X0': 0x8000000000000000, 'X1': 1},
        input_taint={'X0': 0, 'X1': 0x1},  # Taint the bit crossing the boundary
        simulator=arm64_simulator,
    )
    assert extract_flag(circuit.evaluate(ctx), 'V') == 1
