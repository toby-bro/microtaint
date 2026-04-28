# ruff: noqa: PLC0415
import pytest

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, ImplicitTaintPolicy, Register

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
        return (nzcv >> 31) & 1
    if flag_name == 'Z':
        return (nzcv >> 30) & 1
    if flag_name == 'C':
        return (nzcv >> 29) & 1
    if flag_name == 'V':
        return (nzcv >> 28) & 1
    return 0


# ==========================================
# AMD64: MOV Tests
# ==========================================


def test_mov_reg_reg_propagates_taint(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('4889C3')  # MOV RBX, RAX
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)
    ctx = EvalContext(
        input_values={'RAX': 0x1234, 'RBX': 0},
        input_taint={'RAX': 0xFFFFFFFFFFFFFFFF, 'RBX': 0},
        simulator=simulator,
    )
    assert circuit.evaluate(ctx).get('RBX', 0) == 0xFFFFFFFFFFFFFFFF


def test_mov_imm_reg_clears_taint(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('48B80100000000000000')  # MOV RAX, 1
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)
    ctx = EvalContext(input_values={'RAX': 0}, input_taint={'RAX': 0xFFFFFFFFFFFFFFFF}, simulator=simulator)
    assert circuit.evaluate(ctx).get('RAX', 0) == 0


def test_mov_partial_reg_propagates_low_32(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('89C3')  # MOV EBX, EAX
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)
    ctx = EvalContext(
        input_values={'EAX': 0x12345678, 'EBX': 0},
        input_taint={'EAX': 0xFFFFFFFF, 'EBX': 0},
        simulator=simulator,
    )
    assert circuit.evaluate(ctx).get('EBX', 0) == 0xFFFFFFFF


# ==========================================
# AMD64: ADD/SUB Tests
# ==========================================


def test_add_propagates_taint(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('4801D8')  # ADD RAX, RBX
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)
    ctx = EvalContext(
        input_values={'RAX': 0, 'RBX': 0},
        input_taint={'RAX': 0xAAAAAAAAAAAAAAAA, 'RBX': 0x5555555555555555},
        simulator=simulator,
    )
    assert circuit.evaluate(ctx).get('RAX', 0) == 0xFFFFFFFFFFFFFFFF


def test_add_partial_reg_propagates_low_16(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('6601C3')  # ADD BX, AX
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)
    ctx = EvalContext(
        input_values={'AX': 0x1234, 'BX': 0x5678},
        input_taint={'AX': 0xFFFF, 'BX': 0xFFFF},
        simulator=simulator,
    )
    assert circuit.evaluate(ctx).get('BX', 0) == 0xFFFF


def test_add_overflow_sets_of(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('01D8')  # ADD EAX, EBX
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)
    ctx = EvalContext(
        input_values={'EAX': 0x7FFFFFFF, 'EBX': 1},
        input_taint={'EAX': 0x10, 'EBX': 0},
        simulator=simulator,
    )
    # The output flips between an overflow state and a non-overflow state, tainting OF
    assert extract_flag(circuit.evaluate(ctx), 'OF') == 1


def test_sub_propagates_taint(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('4829D8')  # SUB RAX, RBX
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)
    ctx = EvalContext(
        input_values={'RAX': 0, 'RBX': 0},
        input_taint={'RAX': 0xAAAAAAAAAAAAAAAA, 'RBX': 0x5555555555555555},
        simulator=simulator,
    )
    assert circuit.evaluate(ctx).get('RAX', 0) == 0xFFFFFFFFFFFFFFFF


def test_sub_borrow_sets_cf(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('29D8')  # SUB EAX, EBX
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)
    ctx = EvalContext(
        input_values={'EAX': 0, 'EBX': 1},
        input_taint={'EAX': 0x10, 'EBX': 0},
        simulator=simulator,
    )
    assert extract_flag(circuit.evaluate(ctx), 'CF') == 1


# ==========================================
# AMD64: AND/OR/XOR Tests (Partial Registers)
# ==========================================


def test_and_partial_mask_propagates_low_16(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('6621C3')  # AND BX, AX
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)
    ctx = EvalContext(
        input_values={'AX': 0xFFFF, 'BX': 0xFFFF},
        input_taint={'AX': 0x0000FFFF, 'BX': 0x0000FFFF},  # Use strict 16-bit taints
        simulator=simulator,
    )
    assert circuit.evaluate(ctx).get('BX', 0) == 0x0000FFFF


def test_and_imm_partial_mask(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('25FF000000')  # AND EAX, 0x000000FF
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)
    ctx = EvalContext(input_values={'EAX': 0xFFFFFFFF}, input_taint={'EAX': 0xFFFFFFFF}, simulator=simulator)
    assert circuit.evaluate(ctx).get('EAX', 0) == 0x000000FF


def test_or_partial_propagates_high_8(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('08D8')  # OR AL, BL
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)
    ctx = EvalContext(input_values={'AL': 0x00, 'BL': 0xFF}, input_taint={'AL': 0x00, 'BL': 0xFF}, simulator=simulator)
    assert circuit.evaluate(ctx).get('AL', 0) == 0xFF


# ==========================================
# AMD64: BSWAP, ROL, ROR Tests
# ==========================================


def test_bswap_propagates_full_taint(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('480FC8')  # BSWAP RAX
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)
    ctx = EvalContext(
        input_values={'RAX': 0x0123456789ABCDEF},
        input_taint={'RAX': 0xFFFFFFFFFFFFFFFF},
        simulator=simulator,
    )
    assert circuit.evaluate(ctx).get('RAX', 0) == 0xFFFFFFFFFFFFFFFF


def test_rol_propagates_taint(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('48C1C008')  # ROL RAX, 8
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)
    ctx = EvalContext(
        input_values={'RAX': 0x00FF00FF00FF00FF},
        input_taint={'RAX': 0x00FF00FF00FF00FF},
        simulator=simulator,
    )
    # The taint mask itself gets rotated
    assert circuit.evaluate(ctx).get('RAX', 0) == 0xFF00FF00FF00FF00


def test_ror_preserves_taint_if_zero_shift(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('48C1C800')  # ROR RAX, 0
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)
    ctx = EvalContext(input_values={'RAX': 0x1234}, input_taint={'RAX': 0xFFFFFFFFFFFFFFFF}, simulator=simulator)
    # Taint does not clear on a shift by zero, it preserves perfectly.
    assert circuit.evaluate(ctx).get('RAX', 0) == 0xFFFFFFFFFFFFFFFF


# ==========================================
# AMD64: MUL/IMUL/DIV/IDIV Tests
# ==========================================


def test_mul_propagates_taint_to_rdx_rax(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('48F7E2')  # MUL RDX
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)
    ctx = EvalContext(
        input_values={'RAX': 2, 'RDX': 3},
        input_taint={'RAX': 0xFFFFFFFFFFFFFFFF, 'RDX': 0},
        simulator=simulator,
    )
    output = circuit.evaluate(ctx)
    assert output.get('RAX', 0) == 0xFFFFFFFFFFFFFFFF
    assert output.get('RDX', 0) == 0xFFFFFFFFFFFFFFFF


def test_imul_propagates_taint(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('480FAFD8')  # IMUL RBX, RAX
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)
    ctx = EvalContext(
        input_values={'RAX': 2, 'RBX': 3},
        input_taint={'RAX': 0xFFFFFFFFFFFFFFFF, 'RBX': 0},
        simulator=simulator,
    )
    # 2-operand IMUL outputs strictly to the destination register (RBX)
    assert circuit.evaluate(ctx).get('RBX', 0) == 0xFFFFFFFFFFFFFFFF


# ==========================================
# AMD64: CMP/TEST/SETcc Tests
# ==========================================


def test_cmp_equal_sets_zf(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('39D8')  # CMP EAX, EBX
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)
    ctx = EvalContext(
        input_values={'EAX': 5, 'EBX': 5},
        input_taint={'EAX': 0xFFFFFFFF, 'EBX': 0},
        simulator=simulator,
    )
    assert extract_flag(circuit.evaluate(ctx), 'ZF') == 1


def test_test_propagates_taint_to_flags(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('85D2')  # TEST EDX, EDX
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)
    ctx = EvalContext(input_values={'RDX': 0}, input_taint={'RDX': 0x10}, simulator=simulator)
    output = circuit.evaluate(ctx)
    assert extract_flag(output, 'ZF') == 1


def test_setz_propagates_zf_taint(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('0F94C0')  # SETZ AL
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)
    ctx = EvalContext(input_values={'EFLAGS': 0x40}, input_taint={'ZF': 1}, simulator=simulator)
    assert circuit.evaluate(ctx).get('AL', 0) == 1


# ==========================================
# AMD64: LEA, POP, PUSH, XCHG Tests
# ==========================================


def test_lea_propagates_taint(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('488D0418')  # LEA RAX, [RAX + RBX]
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)
    ctx = EvalContext(
        input_values={'RAX': 0x1000, 'RBX': 0x20},
        input_taint={'RAX': 0xFFFFFFFFFFFFFFFF, 'RBX': 0},
        simulator=simulator,
    )
    assert circuit.evaluate(ctx).get('RAX', 0) == 0xFFFFFFFFFFFFFFFF


def test_xchg_propagates_taint_bidirectional(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('4891')  # XCHG RAX, RCX (64-bit explicit)
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)
    ctx = EvalContext(
        input_values={'RAX': 0, 'RCX': 0},
        input_taint={'RAX': 0x10, 'RCX': 0x20},
        simulator=simulator,
    )
    output = circuit.evaluate(ctx)
    assert output.get('RAX', 0) == 0x20
    assert output.get('RCX', 0) == 0x10


# ==========================================
# AMD64: INC/DEC/NEG/NOT Tests
# ==========================================


def test_inc_propagates_taint(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('FFC0')  # INC EAX (32-bit zero extends)
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)
    ctx = EvalContext(input_values={'EAX': 0}, input_taint={'EAX': 0xFFFFFFFF}, simulator=simulator)
    assert circuit.evaluate(ctx).get('EAX', 0) == 0xFFFFFFFF


def test_neg_propagates_taint(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('48F7D8')  # NEG RAX
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)
    ctx = EvalContext(input_values={'RAX': 5}, input_taint={'RAX': 0xFFFFFFFFFFFFFFFF}, simulator=simulator)
    assert circuit.evaluate(ctx).get('RAX', 0) == 0xFFFFFFFFFFFFFFFF


def test_not_propagates_taint(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('48F7D0')  # NOT RAX
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)
    ctx = EvalContext(
        input_values={'RAX': 0xFFFFFFFFFFFFFFFF},
        input_taint={'RAX': 0xFFFFFFFFFFFFFFFF},
        simulator=simulator,
    )
    assert circuit.evaluate(ctx).get('RAX', 0) == 0xFFFFFFFFFFFFFFFF


# ==========================================
# ARM64: MOV/MOVK Tests
# ==========================================


def test_arm64_mov_propagates_taint(arm64_simulator: CellSimulator, arm64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('E00300AA')  # MOV X0, X0
    circuit = generate_static_rule(Architecture.ARM64, bytestring, arm64_registers)
    ctx = EvalContext(input_values={'X0': 0}, input_taint={'X0': 0xFFFFFFFFFFFFFFFF}, simulator=arm64_simulator)
    assert circuit.evaluate(ctx).get('X0', 0) == 0xFFFFFFFFFFFFFFFF


def test_arm64_mov_imm_clears_taint(arm64_simulator: CellSimulator, arm64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('00008052')  # MOV W0, #0
    circuit = generate_static_rule(Architecture.ARM64, bytestring, arm64_registers)
    ctx = EvalContext(input_values={'W0': 0}, input_taint={'W0': 0xFFFFFFFF}, simulator=arm64_simulator)
    assert circuit.evaluate(ctx).get('W0', 0) == 0


def test_arm64_movk_propagates_taint(arm64_simulator: CellSimulator, arm64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('0000A072')  # MOVK W0, #0, LSL #0
    circuit = generate_static_rule(Architecture.ARM64, bytestring, arm64_registers)
    ctx = EvalContext(input_values={'W0': 0}, input_taint={'W0': 0xFFFFFFFF}, simulator=arm64_simulator)
    # MOVK replaces lower 16 bits with 0, leaving top 16 bits intact
    assert circuit.evaluate(ctx).get('W0', 0) == 0x0000FFFF


# ==========================================
# ARM64: ADD/SUB Tests
# ==========================================


def test_arm64_add_propagates_taint(arm64_simulator: CellSimulator, arm64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('0000008B')  # ADD X0, X0, X0
    circuit = generate_static_rule(Architecture.ARM64, bytestring, arm64_registers)
    ctx = EvalContext(input_values={'X0': 0}, input_taint={'X0': 0xFFFFFFFFFFFFFFFF}, simulator=arm64_simulator)
    assert circuit.evaluate(ctx).get('X0', 0) == 0xFFFFFFFFFFFFFFFF


def test_arm64_add_imm_propagates_taint(arm64_simulator: CellSimulator, arm64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('00000011')  # ADD W0, W0, #0
    circuit = generate_static_rule(Architecture.ARM64, bytestring, arm64_registers)
    ctx = EvalContext(input_values={'W0': 0}, input_taint={'W0': 0xFFFFFFFF}, simulator=arm64_simulator)
    assert circuit.evaluate(ctx).get('W0', 0) == 0xFFFFFFFF


def test_arm64_sub_propagates_taint(arm64_simulator: CellSimulator, arm64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('000001CB')  # SUB X0, X0, X1
    circuit = generate_static_rule(Architecture.ARM64, bytestring, arm64_registers)
    ctx = EvalContext(
        input_values={'X0': 5, 'X1': 2},
        input_taint={'X0': 0xFFFFFFFFFFFFFFFF, 'X1': 0},
        simulator=arm64_simulator,
    )
    assert circuit.evaluate(ctx).get('X0', 0) == 0xFFFFFFFFFFFFFFFF


# ==========================================
# ARM64: AND/ORR/EOR Tests
# ==========================================


def test_arm64_and_propagates_taint(arm64_simulator: CellSimulator, arm64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('0000008A')  # AND X0, X0, X0
    circuit = generate_static_rule(Architecture.ARM64, bytestring, arm64_registers)
    ctx = EvalContext(
        input_values={'X0': 0xFFFFFFFFFFFFFFFF},
        input_taint={'X0': 0xFFFFFFFFFFFFFFFF},
        simulator=arm64_simulator,
    )
    assert circuit.evaluate(ctx).get('X0', 0) == 0xFFFFFFFFFFFFFFFF


def test_arm64_orr_propagates_taint(arm64_simulator: CellSimulator, arm64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('000000AA')  # ORR X0, X0, X0
    circuit = generate_static_rule(Architecture.ARM64, bytestring, arm64_registers)
    ctx = EvalContext(input_values={'X0': 0}, input_taint={'X0': 0xFFFFFFFFFFFFFFFF}, simulator=arm64_simulator)
    assert circuit.evaluate(ctx).get('X0', 0) == 0xFFFFFFFFFFFFFFFF


def test_arm64_eor_propagates_taint(arm64_simulator: CellSimulator, arm64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('000001CA')  # EOR X0, X0, X1
    circuit = generate_static_rule(Architecture.ARM64, bytestring, arm64_registers)
    ctx = EvalContext(
        input_values={'X0': 0, 'X1': 0},
        input_taint={'X0': 0xFFFFFFFFFFFFFFFF, 'X1': 0},
        simulator=arm64_simulator,
    )
    assert circuit.evaluate(ctx).get('X0', 0) == 0xFFFFFFFFFFFFFFFF


# ==========================================
# ARM64: LSL/LSR/ASR/ROR Tests
# ==========================================


def test_arm64_ubfx_right_shift_propagates_taint(
    arm64_simulator: CellSimulator,
    arm64_registers: list[Register],
) -> None:
    bytestring = bytes.fromhex('007C1053')  # Right shifts W0 by 16 (>> 0x10), zero extends to X0
    circuit = generate_static_rule(Architecture.ARM64, bytestring, arm64_registers)
    ctx = EvalContext(input_values={'W0': 0x10000}, input_taint={'W0': 0x10000}, simulator=arm64_simulator)
    # Taint bit 16 shifts right by 16 positions (0x10000 >> 16 -> 0x1).
    assert circuit.evaluate(ctx).get('X0', 0) == 0x1


def test_arm64_lsr_propagates_taint(arm64_simulator: CellSimulator, arm64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('007C0053')  # LSR W0, W0, #1
    circuit = generate_static_rule(Architecture.ARM64, bytestring, arm64_registers)
    ctx = EvalContext(input_values={'W0': 0xFFFFFFFF}, input_taint={'W0': 0xFFFFFFFF}, simulator=arm64_simulator)
    assert circuit.evaluate(ctx).get('W0', 0) == 0xFFFFFFFF


# ==========================================
# ARM64: CMP/CMN/TST/TEQ Tests
# ==========================================


def test_arm64_cmp_sets_nzcv(arm64_simulator: CellSimulator, arm64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('1F0001EB')  # CMP X0, X1
    circuit = generate_static_rule(Architecture.ARM64, bytestring, arm64_registers)
    ctx = EvalContext(
        input_values={'X0': 5, 'X1': 5},
        input_taint={'X0': 0x10, 'X1': 0},
        simulator=arm64_simulator,
    )
    output = circuit.evaluate(ctx)
    assert extract_flag(output, 'Z') == 1


def test_arm64_tst_propagates_taint_to_nzcv(arm64_simulator: CellSimulator, arm64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('1F0001EA')  # TST X0, X1
    circuit = generate_static_rule(Architecture.ARM64, bytestring, arm64_registers)
    ctx = EvalContext(
        input_values={'X0': 0, 'X1': 0xFF},
        input_taint={'X0': 0xFF, 'X1': 0x0},
        simulator=arm64_simulator,
    )
    output = circuit.evaluate(ctx)
    assert extract_flag(output, 'Z') == 1


# ==========================================
# ARM64: MUL/DIV Tests
# ==========================================


def test_arm64_mul_propagates_taint(arm64_simulator: CellSimulator, arm64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('007C009B')  # MUL X0, X0, X0
    circuit = generate_static_rule(Architecture.ARM64, bytestring, arm64_registers)
    ctx = EvalContext(input_values={'X0': 2}, input_taint={'X0': 0xFFFFFFFFFFFFFFFF}, simulator=arm64_simulator)
    assert circuit.evaluate(ctx).get('X0', 0) == 0xFFFFFFFFFFFFFFFF


# ==========================================
# ARM64: LDR/STR Tests
# ==========================================


def test_arm64_ldr_propagates_taint(arm64_simulator: CellSimulator, arm64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('000040F9')  # LDR X0, [X0]
    circuit = generate_static_rule(Architecture.ARM64, bytestring, arm64_registers)
    ctx = EvalContext(
        input_values={'X0': 0x1000},
        input_taint={'MEM_0x1000_8': 0xFFFFFFFFFFFFFFFF},  # Explicit size suffix _8 required
        simulator=arm64_simulator,
    )
    assert circuit.evaluate(ctx).get('X0', 0) == 0xFFFFFFFFFFFFFFFF


# ==========================================
# ARM64: Flag Tests (NZCV)
# ==========================================


def test_arm64_add_sets_carry(arm64_simulator: CellSimulator, arm64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('000001AB')  # ADDS X0, X0, X1 (Flag-setting variant)
    circuit = generate_static_rule(Architecture.ARM64, bytestring, arm64_registers)
    ctx = EvalContext(
        input_values={'X0': 0xFFFFFFFFFFFFFFFF, 'X1': 1},
        input_taint={'X0': 0x10, 'X1': 0},
        simulator=arm64_simulator,
    )
    assert extract_flag(circuit.evaluate(ctx), 'C') == 1


def test_arm64_sub_sets_negative(arm64_simulator: CellSimulator, arm64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('000001EB')  # SUBS X0, X0, X1 (Flag-setting variant)
    circuit = generate_static_rule(Architecture.ARM64, bytestring, arm64_registers)
    ctx = EvalContext(
        input_values={'X0': 0, 'X1': 1},
        input_taint={'X0': 0x10, 'X1': 0},
        simulator=arm64_simulator,
    )
    assert extract_flag(circuit.evaluate(ctx), 'N') == 1


def test_ret_propagates_taint_to_rip(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    bytestring = bytes.fromhex('C3')  # RET
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)

    # We need a concrete stack pointer to resolve the memory read
    stack_ptr = 0x80000000

    ctx = EvalContext(
        input_values={
            'RSP': stack_ptr,
        },
        input_taint={
            'RSP': 0,
            f'MEM_{hex(stack_ptr)}_8': 0xFFFFFFFFFFFFFFFF,  # Taint the memory where the return address lives
        },
        simulator=simulator,
        implicit_policy=ImplicitTaintPolicy.KEEP,
    )

    output = circuit.evaluate(ctx)

    assert (
        output.get('RIP', 0) == 0xFFFFFFFFFFFFFFFF
    ), f"Expected RIP to be fully tainted, got {hex(output.get('RIP', 0))}"


def test_cond_transportable_cmp_with_immediate(amd64_registers: list[Register]) -> None:
    """
    cmp al, 0x58  (3c 58)
    Tests equality comparison against a constant immediate.

    Key insight: when a register is compared against a constant,
    the tainted bits of the register could take the value of the constant,
    so if the constant falls within the tainted range, the flag is uncertain.
    """
    from microtaint.instrumentation.ast import EvalContext
    from microtaint.simulator import CellSimulator
    from microtaint.sleigh.engine import generate_static_rule
    from microtaint.types import Architecture

    sim = CellSimulator(Architecture.AMD64)
    bytestring = bytes.fromhex('3c58')  # cmp al, 0x58
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)

    # Case 1: AL is fully tainted, value happens to equal the constant.
    # The constant 0x58 is reachable → ZF must be tainted.
    ctx = EvalContext(
        input_values={'RAX': 0x58},
        input_taint={'RAX': 0xFF},
        simulator=sim,
    )
    out = circuit.evaluate(ctx)
    assert out.get('ZF', 0) == 1, 'ZF should be tainted: fully tainted AL could equal 0x58'

    # Case 2: AL is fully tainted, value does NOT equal constant.
    # But since AL is fully tainted, 0x58 is still reachable → ZF must be tainted.
    ctx2 = EvalContext(
        input_values={'RAX': 0x00},
        input_taint={'RAX': 0xFF},
        simulator=sim,
    )
    out2 = circuit.evaluate(ctx2)
    assert out2.get('ZF', 0) == 1, 'ZF should be tainted: fully tainted AL can reach 0x58'

    # Case 3: Only bit 0 of AL is tainted, AL=0x10.
    # Tainted AL can be 0x10 or 0x11 — neither equals 0x58 → ZF NOT tainted.
    ctx3 = EvalContext(
        input_values={'RAX': 0x10},
        input_taint={'RAX': 0x01},
        simulator=sim,
    )
    out3 = circuit.evaluate(ctx3)
    assert out3.get('ZF', 0) == 0, 'ZF should NOT be tainted: tainted bit cannot make AL reach 0x58'

    # Case 4: Bits 0-5 tainted, AL=0x58.
    # AL can range 0x58..0x5F and 0x40..0x7F etc — 0x58 is reachable → ZF tainted.
    ctx4 = EvalContext(
        input_values={'RAX': 0x58},
        input_taint={'RAX': 0x3F},
        simulator=sim,
    )
    out4 = circuit.evaluate(ctx4)
    assert out4.get('ZF', 0) == 1, 'ZF should be tainted: 0x58 is reachable with tainted lower 6 bits'

    # Case 5: No taint → no taint on output.
    ctx5 = EvalContext(
        input_values={'RAX': 0x58},
        input_taint={},
        simulator=sim,
    )
    out5 = circuit.evaluate(ctx5)
    assert out5.get('ZF', 0) == 0, 'ZF should NOT be tainted: no input taint'
