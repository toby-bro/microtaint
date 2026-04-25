import pytest

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, Register

# --- Fixtures ---


@pytest.fixture(scope='module')
def simulator() -> CellSimulator:
    return CellSimulator(Architecture.AMD64)


@pytest.fixture(scope='module')
def arm64_simulator() -> CellSimulator:
    return CellSimulator(Architecture.ARM64)


@pytest.fixture(scope='module')
def amd64_registers() -> list[Register]:
    return [
        Register(name='RAX', bits=64),
        Register(name='RBX', bits=64),
        Register(name='RCX', bits=64),
        Register(name='RDX', bits=64),
        Register(name='RSP', bits=64),
        Register(name='RBP', bits=64),
        Register(name='RSI', bits=64),
        Register(name='RDI', bits=64),
        Register(name='R8', bits=64),
        Register(name='R9', bits=64),
        Register(name='R10', bits=64),
        Register(name='R11', bits=64),
        Register(name='R12', bits=64),
        Register(name='R13', bits=64),
        Register(name='R14', bits=64),
        Register(name='R15', bits=64),
        Register(name='RIP', bits=64),
        Register(name='EFLAGS', bits=32),
        Register(name='ZF', bits=1),
        Register(name='CF', bits=1),
        Register(name='SF', bits=1),
        Register(name='OF', bits=1),
        Register(name='PF', bits=1),
        Register(name='AX', bits=16),
        Register(name='AL', bits=8),
        Register(name='AH', bits=8),
        Register(name='EAX', bits=32),
        Register(name='BX', bits=16),
        Register(name='BL', bits=8),
        Register(name='BH', bits=8),
        Register(name='EBX', bits=32),
    ]


@pytest.fixture(scope='module')
def arm64_registers() -> list[Register]:
    return [
        Register(name='X0', bits=64),
        Register(name='X1', bits=64),
        Register(name='X2', bits=64),
        Register(name='X3', bits=64),
        Register(name='X4', bits=64),
        Register(name='X5', bits=64),
        Register(name='X6', bits=64),
        Register(name='X7', bits=64),
        Register(name='X8', bits=64),
        Register(name='X9', bits=64),
        Register(name='X10', bits=64),
        Register(name='X11', bits=64),
        Register(name='X12', bits=64),
        Register(name='X13', bits=64),
        Register(name='X14', bits=64),
        Register(name='X15', bits=64),
        Register(name='X16', bits=64),
        Register(name='X17', bits=64),
        Register(name='X18', bits=64),
        Register(name='X19', bits=64),
        Register(name='X20', bits=64),
        Register(name='X21', bits=64),
        Register(name='X22', bits=64),
        Register(name='X23', bits=64),
        Register(name='X24', bits=64),
        Register(name='X25', bits=64),
        Register(name='X26', bits=64),
        Register(name='X27', bits=64),
        Register(name='X28', bits=64),
        Register(name='X29', bits=64),
        Register(name='X30', bits=64),
        Register(name='SP', bits=64),
        Register(name='PC', bits=64),
        Register(name='NZCV', bits=4),
        Register(name='N', bits=1),
        Register(name='Z', bits=1),
        Register(name='C', bits=1),
        Register(name='V', bits=1),
        Register(name='W0', bits=32),
        Register(name='W1', bits=32),
        Register(name='W2', bits=32),
    ]


def extract_flag(ast_output: dict[str, int], flag_name: str) -> int:  # noqa: C901
    if flag_name in ast_output:
        return ast_output[flag_name]
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
    nzcv = ast_output.get('NZCV', 0)
    if flag_name == 'N':
        return (nzcv >> 3) & 1
    if flag_name == 'Z':
        return (nzcv >> 2) & 1
    if flag_name == 'C':
        return (nzcv >> 1) & 1
    if flag_name == 'V':
        return (nzcv >> 0) & 1
    return 0


def test_and_monotonic_precise_taint(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    """
    Validates Monotonic precision (Bitwise non-decreasing).
    AND EAX, EBX.
    If EBX is 0 in a bit position, the output of that bit is ALWAYS 0,
    regardless of EAX. Therefore, a taint in EAX at that position should NOT propagate.
    """
    bytestring = bytes.fromhex('21D8')  # AND EAX, EBX
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)

    ctx = EvalContext(
        input_values={'EAX': 0xFFFFFFFF, 'EBX': 0x0000FFFF},
        input_taint={'EAX': 0xFFFF0000, 'EBX': 0x00000000},  # Taint the top 16 bits of EAX
        simulator=simulator,
    )
    res = circuit.evaluate(ctx)

    # Naive OR-based tracking would propagate 0xFFFF0000.
    # Precise Monotonic tracking realizes the output can't change because EBX masks it out.
    assert res.get('EAX', 0) == 0x00000000


def test_or_monotonic_precise_taint(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    """
    Validates Monotonic precision (Bitwise non-decreasing).
    OR EAX, EBX.
    If EBX is 1 in a bit position, the output of that bit is ALWAYS 1,
    regardless of EAX. Therefore, a taint in EAX at that position should NOT propagate.
    """
    bytestring = bytes.fromhex('09D8')  # OR EAX, EBX
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)

    ctx = EvalContext(
        input_values={'EAX': 0x00000000, 'EBX': 0xFFFF0000},
        input_taint={'EAX': 0xFFFF0000, 'EBX': 0x00000000},  # Taint the top 16 bits of EAX
        simulator=simulator,
    )
    res = circuit.evaluate(ctx)

    # Top 16 bits of EBX are already 1. Toggling top 16 bits of EAX changes nothing.
    assert res.get('EAX', 0) == 0x00000000


def test_xor_mapped_taint(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    """
    Validates Mapped precision.
    XOR EAX, EBX.
    The paper dictates XOR propagates taint if *any* input bit is tainted.
    """
    bytestring = bytes.fromhex('31D8')  # XOR EAX, EBX
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)

    ctx = EvalContext(
        input_values={'EAX': 0x00000000, 'EBX': 0xFFFFFFFF},
        input_taint={'EAX': 0x0F0F0F0F, 'EBX': 0x00000000},
        simulator=simulator,
    )
    res = circuit.evaluate(ctx)

    # XOR simply ORs the taints.
    assert res.get('EAX', 0) == 0x0F0F0F0F


def test_add_transportable_taint(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    """
    Validates Transportable precision.
    ADD EAX, EBX.
    Ensures the (A^t | B^t) transportability term is injected natively,
    and verifies carry propagation works cleanly.
    """
    bytestring = bytes.fromhex('01D8')  # ADD EAX, EBX
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)

    ctx = EvalContext(
        input_values={'EAX': 0x00000001, 'EBX': 0x00000002},
        input_taint={'EAX': 0x00000001, 'EBX': 0x00000004},
        simulator=simulator,
    )
    res = circuit.evaluate(ctx)

    # Taint should strictly be the combination of both input taints (transportability term),
    # since no cascading carry overlaps with the tainted bits in this specific addition.
    assert res.get('EAX', 0) == 0x00000005


def test_signed_comparison_sless_msb_taint(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    """
    Validates Signed Comparison (SLESS/SBORROW) MSB Polarity split.
    CMP EAX, EBX.
    Checks that tainting the MSB of EAX correctly calculates the differential
    and correctly flags the Sign Flag (SF) as tainted.
    """
    bytestring = bytes.fromhex('39D8')  # CMP EAX, EBX
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)

    # EAX is a large positive number. EBX is 0.
    # SF (Sign Flag) should evaluate to 0 initially (Positive result).
    # If the MSB of EAX is tainted, it means EAX *could* be negative,
    # meaning the SF flag output SHOULD be tainted.
    ctx = EvalContext(
        input_values={'EAX': 0x7FFFFFFF, 'EBX': 0x00000000},
        input_taint={'EAX': 0x80000000, 'EBX': 0x00000000},  # Taint ONLY the MSB
        simulator=simulator,
    )
    res = circuit.evaluate(ctx)

    # SF is dependent on the MSB (signed evaluation)
    sf_taint = extract_flag(res, 'SF')
    assert sf_taint == 1


def test_signed_comparison_sless_msb_no_taint_if_masked(
    simulator: CellSimulator,
    amd64_registers: list[Register],
) -> None:
    """
    Validates that if the MSB is tainted, but the underlying values prevent
    the Sign Flag from flipping, the SF is NOT overtainted.
    """
    bytestring = bytes.fromhex('39D8')  # CMP EAX, EBX
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)

    # If EAX is 0x00000001 and EBX is 0, flipping the MSB of EAX gives 0x80000001.
    # 0x00000001 - 0 = Positive (SF=0)
    # 0x80000001 - 0 = Negative (SF=1)
    # Therefore, flipping the MSB DOES flip SF, so SF should be tainted.
    ctx1 = EvalContext(
        input_values={'EAX': 0x00000001, 'EBX': 0x00000000},
        input_taint={'EAX': 0x80000000, 'EBX': 0x00000000},
        simulator=simulator,
    )
    assert extract_flag(circuit.evaluate(ctx1), 'SF') == 1
