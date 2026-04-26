import pytest

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, Register


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


def test_push_pop_no_crash(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('505b')
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)
    ctx = EvalContext(
        # Use 64-bit RAX/RBX since PUSH/POP in 64-bit mode are 64-bit operations
        input_values={'RAX': 0x12345678, 'RBX': 0},
        input_taint={'RAX': 0xFFFFFFFF, 'RBX': 0},
        simulator=simulator,
    )

    # We expect the 0xFFFFFFFF taint to completely transfer to RBX
    assert circuit.evaluate(ctx).get('RBX', 0) == 0xFFFFFFFF


def test_implicit_stack_memory_operations_do_not_crash() -> None:
    """
    Tests that instructions implicitly using the stack (like PUSH and POP)
    do not underflow the stack pointer and crash Unicorn with UC_ERR_MAP.
    """
    amd64_registers = [
        Register(name='RAX', bits=64),
        Register(name='RBX', bits=64),
        Register(name='RSP', bits=64),
    ]

    sim = CellSimulator(Architecture.AMD64)

    # 50 -> PUSH RAX
    # 5b -> POP RBX
    bytestring = bytes.fromhex('505b')

    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)

    ctx = EvalContext(
        input_values={'RAX': 0xDEADBEEF, 'RBX': 0},
        input_taint={'RAX': 0xFFFFFFFFFFFFFFFF, 'RBX': 0},
        simulator=sim,
    )

    out = circuit.evaluate(ctx)

    # The instruction should execute cleanly without throwing UC_ERR_MAP.
    # Furthermore, because RAX was pushed and then popped into RBX,
    # the taint must have successfully transferred between the registers.
    assert out.get('RBX', 0) == 0xFFFFFFFFFFFFFFFF, 'Taint failed to propagate through the stack'
