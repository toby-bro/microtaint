import pytest

from microtaint.emulator.shadow import BitPreciseShadowMemory
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
    ]


# ==========================================
# AMD64: Partial Register Zero-Extension vs Preservation
# ==========================================


def test_amd64_32bit_mov_zero_extends_clearing_taint(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    """
    MOV EAX, EBX (89 D8)
    In 64-bit mode, writing to a 32-bit register MUST zero-extend to 64-bits.
    Therefore, the upper 32 bits of RAX lose their original taint and become untainted (0).
    """
    bytestring = bytes.fromhex('89D8')
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)

    ctx = EvalContext(
        input_values={'RAX': 0xFFFFFFFFFFFFFFFF, 'RBX': 0x0000000011112222},
        # Top 32-bits of RAX are highly tainted. Lower 32-bits of EBX are tainted.
        input_taint={'RAX': 0xFFFFFFFF00000000, 'RBX': 0x00000000AABBCCDD},
        simulator=simulator,
    )
    output = circuit.evaluate(ctx)

    # The upper 32 bits of RAX must be cleared of taint (0x00000000).
    # The lower 32 bits of RAX perfectly inherit EBX's taint.
    assert output.get('RAX', 0) == 0x00000000AABBCCDD


def test_amd64_16bit_mov_preserves_upper_taint(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    """
    MOV AX, BX (66 89 D8)
    Writing to a 16-bit register DOES NOT zero extend. The upper 48 bits are preserved.
    Therefore, the upper 48 bits of RAX must retain their exact previous taint.
    """
    bytestring = bytes.fromhex('6689D8')
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)

    ctx = EvalContext(
        input_values={'RAX': 0, 'RBX': 0},
        input_taint={'RAX': 0xFFFFFFFFFFFF0000, 'RBX': 0x000000000000BBBB},
        simulator=simulator,
    )
    output = circuit.evaluate(ctx)

    # Upper 48 bits of RAX are preserved (0xFFFFFFFFFFFF).
    # Lower 16 bits are overwritten with BX taint (0xBBBB).
    assert output.get('RAX', 0) == 0xFFFFFFFFFFFFBBBB


def test_amd64_8bit_mov_preserves_upper_taint(simulator: CellSimulator, amd64_registers: list[Register]) -> None:
    """
    MOV AL, BL (88 D8)
    Writing to an 8-bit register preserves the upper 56 bits.
    """
    bytestring = bytes.fromhex('88D8')
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)

    ctx = EvalContext(
        input_values={'RAX': 0, 'RBX': 0},
        input_taint={'RAX': 0xFFFFFFFFFFFFFF00, 'RBX': 0x00000000000000CC},
        simulator=simulator,
    )
    output = circuit.evaluate(ctx)

    # Only the lowest byte gets overwritten by BL's taint.
    assert output.get('RAX', 0) == 0xFFFFFFFFFFFFFFCC


def test_amd64_32bit_load_from_untainted_mem_clears_taint(
    simulator: CellSimulator, amd64_registers: list[Register],
) -> None:
    """
    MOV EAX, [RBP-8] (8B 45 F8) loading from UNTAINTED memory MUST strong-update:
    a previously-tainted EAX becomes untainted (and the upper 32 bits of RAX clear
    via zero-extension).

    Regression: reg<-reg writes strong-updated correctly, but reg<-MEMORY loads did
    NOT clear the destination's stale taint when the loaded memory was untainted --
    they carried the register's old taint forward. This weak-update-on-zero was the
    root cause of the nft (CVE-2023-35001) and constant-time over-taint: a stale
    len/exponent taint survived a clean load, then seeded a pointer-avalanche / div
    over-taint. See the nft and CT application experiments.
    """
    regs = [*amd64_registers, Register(name='RBP', bits=64), Register(name='RSP', bits=64)]
    shadow = BitPreciseShadowMemory()  # entirely untainted
    circuit = generate_static_rule(Architecture.AMD64, bytes.fromhex('8B45F8'), regs)

    ctx = EvalContext(
        input_values={'RAX': 0, 'RBP': 0x7FFFF000},
        input_taint={'RAX': 0x00000000FFFFFFFF},  # stale low-32 taint on the destination
        simulator=simulator,
        shadow_memory=shadow,
    )
    output = circuit.evaluate(ctx)

    assert output.get('RAX', 0) == 0x0, (
        'load from untainted memory must clear the destination register (strong '
        f'update); got {output.get("RAX", 0):#x}'
    )


def test_amd64_32bit_xor_zeroing_idiom_clears_64bit_taint(
    simulator: CellSimulator, amd64_registers: list[Register],
) -> None:
    """
    XOR EAX, EAX (31 C0)
    This is the standard x86 zeroing idiom. Because it targets a 32-bit register,
    it zero-extends to 64-bits. The ENTIRE 64-bit RAX register must be cleared of taint.
    """
    bytestring = bytes.fromhex('31C0')
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)

    ctx = EvalContext(
        input_values={'RAX': 0},
        input_taint={'RAX': 0xFFFFFFFFFFFFFFFF},  # 100% tainted
        simulator=simulator,
    )
    output = circuit.evaluate(ctx)

    # Taint completely neutralized by identical-register XOR + zero extension
    assert output.get('RAX', 0) == 0x0
