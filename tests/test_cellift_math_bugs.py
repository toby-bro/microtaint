import pytest

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator, MachineState
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, Register


@pytest.fixture(scope='module')
def simulator() -> CellSimulator:
    return CellSimulator(Architecture.X86)


@pytest.fixture(scope='module')
def state_format() -> list[Register]:
    return [
        Register('EAX', 32),
        Register('EBX', 32),
        Register('ECX', 32),
        Register('EFLAGS', 32),
        Register('ZF', 1),
        Register('SF', 1),
    ]


def check_ast_vs_golden_taint(
    simulator: CellSimulator,
    bytestring: bytes,
    out_reg: str,
    V_dict: dict[str, int],
    T_dict: dict[str, int],
    state_format: list[Register],
    expected_override: int | None = None,
) -> None:
    """
    Exhaustively proves mathematically whether the AST taint logic is sound by
    running the raw CPU state through all 2^k tainted input permutations.
    """
    # 1. Evaluate AST Logic
    rule = generate_static_rule(Architecture.X86, bytestring, state_format)
    ctx = EvalContext(input_taint=T_dict.copy(), input_values=V_dict.copy(), simulator=simulator)
    ast_taint = rule.evaluate(ctx).get(out_reg, 0)

    # 2. Compute Golden Taint via 2^k state exhaustive permutations
    t_vars: list[tuple[str, int]] = []
    for reg, t_val in T_dict.items():
        for b in range(32):
            if (t_val >> b) & 1:
                t_vars.append((reg, b))

    assert len(t_vars) <= 8, 'Too many tainted bits for exhaustive mathematical test.'

    base_V = {reg: V_dict.get(reg, 0) & ~T_dict.get(reg, 0) for reg in V_dict}
    outputs: set[int] = set()

    for i in range(1 << len(t_vars)):
        current_V = base_V.copy()
        for bit_idx, (reg, b) in enumerate(t_vars):
            if (i >> bit_idx) & 1:
                current_V[reg] |= 1 << b

        m_state = MachineState(regs=current_V)
        simulator.clear_memory_and_registers()
        simulator.setup_registers_and_memory(m_state, None)
        simulator._execute(bytestring, m_state)  # pyright: ignore[reportPrivateUsage]
        outputs.add(simulator._read_reg(out_reg))  # pyright: ignore[reportPrivateUsage]

    golden_taint = 0
    for b in range(32):
        # If the bit toggled in ANY of the states, the math says it must be tainted
        bit_vals = {(out >> b) & 1 for out in outputs}
        if len(bit_vals) > 1:
            golden_taint |= 1 << b

    # Allow conservative approximations (Avalanching) for un-trackable Sleigh ops
    if expected_override is not None:
        assert ast_taint == expected_override, f'Expected AST {expected_override:#x}, got {ast_taint:#x}'
    else:
        assert ast_taint == golden_taint, f'Golden math says {golden_taint:#x}, but AST computed {ast_taint:#x}'


def test_bug_1_xor_is_not_monotonic(simulator: CellSimulator, state_format: list[Register]) -> None:
    """
    XOR: 31 d8 (xor eax, ebx)
    Math Bug: C(V|T) ^ C(V & ~T) on an XOR where both inputs are tainted gives 0 ^ 0 = 0.
    """
    # V_EAX=0, V_EBX=0. T_EAX=1, T_EBX=1
    check_ast_vs_golden_taint(
        simulator,
        b'\x31\xd8',
        'EAX',
        V_dict={'EAX': 0, 'EBX': 0},
        T_dict={'EAX': 1, 'EBX': 1},
        state_format=state_format,
    )


def test_bug_2_translatable_shifts_ignored(simulator: CellSimulator, state_format: list[Register]) -> None:
    """
    Shift: d3 e0 (shl eax, cl)
    Math Bug: Shifts are NOT monotonic. If offset is tainted, AST must avalanche.
    """
    # V_EAX=1, V_ECX=1. T_EAX=0, T_ECX=1
    check_ast_vs_golden_taint(
        simulator,
        b'\xd3\xe0',
        'EAX',
        V_dict={'EAX': 1, 'ECX': 1},
        T_dict={'EAX': 0, 'ECX': 1},
        state_format=state_format,
        expected_override=0xFFFFFFFF,  # CellIFT approximation policy requirement
    )


def test_bug_3_conditionally_transportable_equalities(simulator: CellSimulator, state_format: list[Register]) -> None:
    """
    Equality: 39 d8 (cmp eax, ebx) -> sets ZF
    Math Bug: Conditional equality requires explicit combination rules.
    """
    # V_EAX=5, V_EBX=4. T_EAX=1, T_EBX=0
    check_ast_vs_golden_taint(
        simulator,
        b'\x39\xd8',
        'ZF',
        V_dict={'EAX': 5, 'EBX': 4},
        T_dict={'EAX': 1, 'EBX': 0},
        state_format=state_format,
        expected_override=1,  # Boolean avalanche approximation
    )


def test_bug_4_missing_negation_transport_term(simulator: CellSimulator, state_format: list[Register]) -> None:
    """
    Negation: f7 d8 (neg eax)
    Math Bug: Negation is transportable. Without the transport term ( | T ),
    intermediate bits fail to toggle correctly in the differential extreme.
    """
    # V_EAX=0, T_EAX=3 (Bits 0, 1)
    # The pure differential gives 0xFFFFFFFD ^ 0 = 0xFFFFFFFD (Bit 1 is ZERO - MISSED TAINT!)
    # The transport logic (| 3) adds bit 1 back in.
    check_ast_vs_golden_taint(
        simulator,
        b'\xf7\xd8',
        'EAX',
        V_dict={'EAX': 0},
        T_dict={'EAX': 3},
        state_format=state_format,
    )
