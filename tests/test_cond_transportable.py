import pytest

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, Register


@pytest.fixture(scope='module')
def simulator() -> CellSimulator:
    return CellSimulator(Architecture.AMD64)


@pytest.fixture
def amd64_registers() -> list[Register]:
    return [
        Register(name='RAX', bits=64),
        Register(name='RBX', bits=64),
        Register(name='RCX', bits=64),
        Register(name='RDX', bits=64),
        Register(name='ZF', bits=1),
        Register(name='CF', bits=1),
        Register(name='SF', bits=1),
        Register(name='OF', bits=1),
    ]


@pytest.mark.parametrize(
    ('v_eax', 'v_ebx', 't_eax', 't_ebx', 'expected_zf_taint', 'description'),
    [
        # --- SCENARIO 1: Untainted bits match ---
        # The untainted bits are perfectly identical. The tainted bit(s) could flip
        # the equality state, so the Zero Flag (ZF) MUST be tainted.
        (0x10, 0x10, 0x01, 0x00, 1, 'Untainted match, single bit taint on A'),
        (0x10, 0x10, 0x00, 0x01, 1, 'Untainted match, single bit taint on B'),
        (0xF0, 0xF0, 0x0F, 0x0F, 1, 'Untainted match, overlapping multi-bit taint'),
        # --- SCENARIO 2: Untainted bits mismatch ---
        # The untainted bits are different. No matter what the tainted bits do,
        # the operands can NEVER be equal. ZF can never flip to 1, so ZF MUST NOT be tainted.
        (0x10, 0x20, 0x01, 0x00, 0, 'Untainted mismatch, single bit taint'),
        (0xF0, 0xE0, 0x0F, 0x0F, 0, 'Untainted mismatch, overlapping multi-bit taint'),
        # --- SCENARIO 3: Disjoint Taints ---
        # EAX is tainted at bit 0. EBX is tainted at bit 1.
        # The mask of untainted bits excludes BOTH bit 0 and bit 1.
        (0x100, 0x100, 0x01, 0x02, 1, 'Untainted match, disjoint taints'),
        (0x100, 0x200, 0x01, 0x02, 0, 'Untainted mismatch, disjoint taints'),
        # --- SCENARIO 4: Extreme / Edge Cases ---
        # If one input is fully tainted, there are technically no untainted bits to compare (Mask = 0).
        # 0 == 0 evaluates to True, meaning equality is possible. ZF MUST be tainted.
        (0x00000000, 0x12345678, 0xFFFFFFFF, 0x00000000, 1, 'Full taint on EAX vs arbitrary EBX'),
        # If there is absolutely no taint in the system, the output cannot be tainted.
        (0x10, 0x10, 0x00, 0x00, 0, 'No taint present at all'),
    ],
)
def test_extensive_conditional_transportability_equality(
    simulator: CellSimulator,
    amd64_registers: list[Register],
    v_eax: int,
    v_ebx: int,
    t_eax: int,
    t_ebx: int,
    expected_zf_taint: int,
    description: str,
) -> None:
    """
    Extensively tests COND_TRANSPORTABLE precision for equality cells.
    Instruction: CMP EAX, EBX -> 39 d8 (Checks equality via ZF)
    """
    arch = Architecture.AMD64
    bytestring = bytes.fromhex('39d8')  # CMP EAX, EBX

    # 1. Generate the AST rule statically
    circuit = generate_static_rule(arch, bytestring, amd64_registers)

    # 2. Evaluate the AST against the specific parameterized state
    ctx = EvalContext(
        input_values={'RAX': v_eax, 'RBX': v_ebx},
        input_taint={'RAX': t_eax, 'RBX': t_ebx},
        simulator=simulator,
    )
    result = circuit.evaluate(ctx)

    # 3. Extract the Zero Flag (ZF) taint
    zf_taint = result.get('ZF')

    # Fallback: Just in case the engine mapped it into the parent EFLAGS register
    if zf_taint is None:
        eflags = result.get('EFLAGS', 0)
        zf_taint = (eflags >> 6) & 1  # ZF is bit 6 in x86 EFLAGS

    assert zf_taint == expected_zf_taint, f'Failed on case: {description}'
