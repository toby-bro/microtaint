import pytest

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator, MachineState
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, Register


@pytest.fixture(scope='module')
def simulator() -> CellSimulator:
    # Make sure simulator.py has 'EFLAGS': uc_x86_const.UC_X86_REG_EFLAGS mapped for X86!
    return CellSimulator(Architecture.X86)


@pytest.fixture(scope='module')
def state_format() -> list[Register]:
    return [
        Register('EAX', 32),
        Register('EBX', 32),
        Register('ECX', 32),
        Register('EDX', 32),
        Register('EFLAGS', 32),
        Register('ZF', 1),
        Register('SF', 1),
        Register('CF', 1),
    ]


def check_property_vs_golden_model(  # noqa: C901
    simulator: CellSimulator,
    bytestring: bytes,
    out_reg: str,
    v_dict: dict[str, int],
    t_dict: dict[str, int],
    state_format: list[Register],
    expected_override: int | None = None,
) -> None:
    """
    Exhaustively proves mathematically whether the AST taint logic is sound by
    running the raw CPU state through all 2^k tainted input permutations.
    """
    # 1. Evaluate AST Logic
    rule = generate_static_rule(Architecture.X86, bytestring, state_format)
    ctx = EvalContext(input_taint=t_dict.copy(), input_values=v_dict.copy(), simulator=simulator)

    # Read the raw dictionary output
    ast_output = rule.evaluate(ctx)

    # Extract the AST taint using the proper EFLAGS bit shifts for boolean flags
    if out_reg == 'CF':
        ast_taint = (ast_output.get('EFLAGS', 0) >> 0) & 1
    elif out_reg == 'PF':
        ast_taint = (ast_output.get('EFLAGS', 0) >> 2) & 1
    elif out_reg == 'ZF':
        ast_taint = (ast_output.get('EFLAGS', 0) >> 6) & 1
    elif out_reg == 'SF':
        ast_taint = (ast_output.get('EFLAGS', 0) >> 7) & 1
    elif out_reg == 'OF':
        ast_taint = (ast_output.get('EFLAGS', 0) >> 11) & 1
    else:
        ast_taint = ast_output.get(out_reg, 0)

    # 2. Extract exactly which bits are tainted
    t_vars: list[tuple[str, int]] = []
    for reg, t_val in t_dict.items():
        for b in range(32):
            if (t_val >> b) & 1:
                t_vars.append((reg, b))

    assert len(t_vars) <= 10, 'Too many tainted bits for fast exhaustive mathematical test.'

    # 3. Base values with all tainted bits set to 0
    base_v: dict[str, int] = {reg: v_dict.get(reg, 0) & ~t_dict.get(reg, 0) for reg in v_dict}
    outputs: set[int] = set()

    # 4. Brute force all 2^k permutations of the tainted bits natively
    for i in range(1 << len(t_vars)):
        current_v = base_v.copy()
        for bit_idx, (reg, b) in enumerate(t_vars):
            if (i >> bit_idx) & 1:
                current_v[reg] |= 1 << b

        m_state = MachineState(regs=current_v)
        simulator._execute(bytestring, m_state)
        outputs.add(simulator._read_reg(out_reg))

    # 5. Compute Golden Taint: A bit is tainted if it toggled in ANY of the states
    golden_taint = 0
    reg_size = next((r.bits for r in state_format if r.name == out_reg), 32)

    for b in range(reg_size):
        bit_vals = {(out >> b) & 1 for out in outputs}
        if len(bit_vals) > 1:
            golden_taint |= 1 << b

    # Check assertions
    if expected_override is not None:
        assert ast_taint == expected_override, f'Expected AST {expected_override:#x}, got {ast_taint:#x}'
    else:
        assert ast_taint == golden_taint, f'Golden math says {golden_taint:#x}, but AST computed {ast_taint:#x}'


def test_property_mapped_xor(simulator: CellSimulator, state_format: list[Register]) -> None:
    """
    PROPERTY: MAPPED / SIMPLE
    Instruction: xor eax, ebx (31 d8)
    Why: A pure differential C(V|T) ^ C(V&~T) drops taint when both inputs are 1.
    XOR must simply OR the input taints together directly.
    """
    check_property_vs_golden_model(
        simulator,
        b'\x31\xd8',
        'EAX',
        v_dict={'EAX': 0, 'EBX': 0},
        t_dict={'EAX': 1, 'EBX': 1},
        state_format=state_format,
    )


def test_property_transportable_addition(simulator: CellSimulator, state_format: list[Register]) -> None:
    """
    PROPERTY: TRANSPORTABLE
    Instruction: add eax, ebx (01 d8)
    Why: Addition uses the differential PLUS a transport term (T_a | T_b)
    to ensure intermediate carries don't drop bit taints.
    """
    check_property_vs_golden_model(
        simulator,
        b'\x01\xd8',
        'EAX',
        v_dict={'EAX': 0x0F, 'EBX': 0x01},
        t_dict={'EAX': 0x02, 'EBX': 0x00},
        state_format=state_format,
    )


def test_property_transportable_negation(simulator: CellSimulator, state_format: list[Register]) -> None:
    """
    PROPERTY: TRANSPORTABLE
    Instruction: neg eax (f7 d8)
    Why: Arithmetic negation is transportable. Without the transport term,
    the differential drops intermediate bit taints.
    """
    check_property_vs_golden_model(
        simulator,
        b'\xf7\xd8',
        'EAX',
        v_dict={'EAX': 0},
        t_dict={'EAX': 3},  # Bits 0 and 1
        state_format=state_format,
    )


def test_property_translatable_shift(simulator: CellSimulator, state_format: list[Register]) -> None:
    """
    PROPERTY: TRANSLATABLE
    Instruction: shl eax, cl (d3 e0)
    Why: Shifts are not monotonic. If the shift offset is tainted, we must
    Avalanche (taint the entire register) per the CELLIFT approximation policy.
    """
    check_property_vs_golden_model(
        simulator,
        b'\xd3\xe0',
        'EAX',
        v_dict={'EAX': 1, 'ECX': 1},
        t_dict={'EAX': 0, 'ECX': 1},
        state_format=state_format,
        expected_override=0xFFFFFFFF,
    )


def test_property_cond_transportable_equality(simulator: CellSimulator, state_format: list[Register]) -> None:
    """
    PROPERTY: CONDITIONALLY TRANSPORTABLE
    Instruction: cmp eax, ebx (39 d8) setting ZF
    Why: Equalities avalanche the specific boolean output flag if ANY
    input bit is tainted.
    """
    check_property_vs_golden_model(
        simulator,
        b'\x39\xd8',
        'ZF',
        v_dict={'EAX': 5, 'EBX': 4},
        t_dict={'EAX': 1, 'EBX': 0},
        state_format=state_format,
        expected_override=1,
    )


def test_property_avalanche_signed_comparison(simulator: CellSimulator, state_format: list[Register]) -> None:
    """
    PROPERTY: AVALANCHE
    Instruction: cmp eax, ebx (39 d8) setting SF
    Why: Signed comparisons have split polarities (MSB is inverted).
    Treating the 1-bit output as an Avalanche mathematically perfectly preserves
    the boolean taint without requiring static polarity tracking.
    """
    check_property_vs_golden_model(
        simulator,
        b'\x39\xd8',
        'SF',
        v_dict={'EAX': 0, 'EBX': 1},
        t_dict={'EAX': 0x80000000, 'EBX': 0},  # Tainting the MSB sign bit
        state_format=state_format,
        expected_override=1,
    )
