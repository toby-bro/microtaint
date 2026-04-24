import pytest

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, Register


@pytest.fixture(scope='module')
def simulator() -> CellSimulator:
    return CellSimulator(Architecture.AMD64)


@pytest.fixture(scope='module')
def amd64_registers() -> list[Register]:
    return [
        Register(name='RAX', bits=64),
        Register(name='RBX', bits=64),
        Register(name='RCX', bits=64),
        Register(name='RDX', bits=64),
        Register(name='EFLAGS', bits=32),
        # FIX: The AST engine needs these defined to track them!
        Register(name='ZF', bits=1),
        Register(name='CF', bits=1),
        Register(name='SF', bits=1),
        Register(name='OF', bits=1),
        Register(name='PF', bits=1),
    ]


def extract_flag(ast_output: dict[str, int], flag_name: str) -> int:
    """Helper to extract boolean flag taint."""
    # Sleigh often maps individual flags directly if they are in the state format
    if flag_name in ast_output:
        return ast_output[flag_name]

    # Fallback to EFLAGS bit shifts just in case
    eflags = ast_output.get('EFLAGS', 0)
    if flag_name == 'CF':
        return (eflags >> 0) & 1
    if flag_name == 'PF':
        return (eflags >> 2) & 1
    if flag_name == 'ZF':
        return (eflags >> 6) & 1
    if flag_name == 'SF':
        return (eflags >> 7) & 1
    if flag_name == 'OF':
        return (eflags >> 11) & 1
    return 0


# ==========================================
# 1. XOR Tests (Clearing vs Propagating)
# ==========================================


def test_xor_clears_taint(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('4831c0')  # XOR RAX, RAX
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)

    ctx = EvalContext(
        input_values={'RAX': 0xDEADBEEF},
        input_taint={'RAX': 0xFFFFFFFFFFFFFFFF},
        simulator=simulator,
    )
    assert circuit.evaluate(ctx).get('RAX', 0) == 0x0


def test_xor_propagates_taint(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('4831d8')  # XOR RAX, RBX
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)

    ctx = EvalContext(
        input_values={'RAX': 0x0, 'RBX': 0x0},
        input_taint={'RAX': 0xAAAAAAAAAAAAAAAA, 'RBX': 0x5555555555555555},
        simulator=simulator,
    )
    assert circuit.evaluate(ctx).get('RAX', 0) == 0xFFFFFFFFFFFFFFFF


# ==========================================
# 2. AND / OR Tests (Constant Masking)
# ==========================================


def test_and_constant_masks_taint(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('250f0f0f0f')  # AND EAX, 0x0F0F0F0F
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)

    ctx = EvalContext(
        input_values={'RAX': 0xFFFFFFFF},
        input_taint={'RAX': 0xFFFFFFFF},
        simulator=simulator,
    )
    assert circuit.evaluate(ctx).get('RAX', 0) == 0x0F0F0F0F


def test_or_constant_masks_taint(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('0d0f0f0f0f')  # OR EAX, 0x0F0F0F0F
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)

    ctx = EvalContext(
        input_values={'RAX': 0x00000000},
        input_taint={'RAX': 0xFFFFFFFF},
        simulator=simulator,
    )
    assert circuit.evaluate(ctx).get('RAX', 0) == 0xF0F0F0F0


# ==========================================
# 3. Shift Tests (Translatable & Avalanche)
# ==========================================


def test_shl_concrete_propagates_taint(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('c1e004')  # SHL EAX, 4
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)

    ctx = EvalContext(
        input_values={'RAX': 0x0000000F},
        input_taint={'RAX': 0x0000000F},
        simulator=simulator,
    )
    assert circuit.evaluate(ctx).get('RAX', 0) == 0x000000F0


def test_shr_concrete_propagates_taint(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('c1e804')  # SHR EAX, 4
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)

    ctx = EvalContext(
        input_values={'RAX': 0x000000F0},
        input_taint={'RAX': 0x000000F0},
        simulator=simulator,
    )
    assert circuit.evaluate(ctx).get('RAX', 0) == 0x0000000F


def test_shl_symbolic_avalanches_taint(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('d3e0')  # SHL EAX, CL
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)

    ctx = EvalContext(
        input_values={'RAX': 0x1, 'RCX': 0x1},
        input_taint={'RAX': 0x0, 'RCX': 0x1},
        simulator=simulator,
    )
    output = circuit.evaluate(ctx)

    # FIX: AMD64 zero-extends 32-bit outputs to the full 64-bit register.
    # Therefore, an Avalanche on EAX correctly cascades through the entire RAX!
    assert output.get('RAX', 0) == 0xFFFFFFFFFFFFFFFF


# ==========================================
# 4. Flag Precision Tests
# ==========================================


def test_cmp_toggles_zf(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('39d8')  # CMP EAX, EBX
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)

    ctx = EvalContext(
        input_values={'RAX': 5, 'RBX': 4},
        input_taint={'RAX': 1, 'RBX': 0},
        simulator=simulator,
    )
    assert extract_flag(circuit.evaluate(ctx), 'ZF') == 1


def test_cmp_toggles_cf(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('39d8')  # CMP EAX, EBX
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)

    ctx = EvalContext(
        input_values={'RAX': 0, 'RBX': 1},
        input_taint={'RAX': 1, 'RBX': 0},
        simulator=simulator,
    )
    assert extract_flag(circuit.evaluate(ctx), 'CF') == 1
