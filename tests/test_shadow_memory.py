import pytest

from microtaint.instrumentation.ast import Expr
from microtaint.simulator import CellSimulator
from microtaint.types import Register


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


def test_shadow_memory_base_plus_offset_read() -> None:
    """
    Simulates: movzbl -0x8(%rbp), %eax
    RBP = 0x80000000de38
    Taint written at 0x80000000de30 (= RBP - 8), size 8
    Reading 1 byte at RBP - 8 must return tainted byte.
    """
    from microtaint.emulator.shadow import BitPreciseShadowMemory

    shadow = BitPreciseShadowMemory()

    rbp = 0x80000000DE38
    taint_addr = rbp - 8  # 0x80000000de30

    # sys_read taints 8 bytes at buf_addr
    shadow.write_mask(taint_addr, 0xFFFFFFFFFFFFFFFF, 8)

    # movzbl reads 1 byte at rbp - 8
    result = shadow.read_mask(taint_addr, 1)
    assert result == 0xFF, f'Expected 0xFF got {hex(result)}'

    # Reading byte 0 through 7 should all be tainted
    for i in range(8):
        result = shadow.read_mask(taint_addr + i, 1)
        assert result == 0xFF, f'Byte {i} at {hex(taint_addr + i)} expected 0xFF got {hex(result)}'

    # Reading beyond the tainted region should return 0
    result = shadow.read_mask(taint_addr + 8, 1)
    assert result == 0, f'Expected 0 beyond taint region, got {hex(result)}'


def test_shadow_memory_partial_write_read() -> None:
    """
    Write taint to bytes 0-7, read back at various sub-ranges.
    """
    from microtaint.emulator.shadow import BitPreciseShadowMemory

    shadow = BitPreciseShadowMemory()
    base = 0x1000

    # Only taint the first 4 bytes
    shadow.write_mask(base, 0xFFFFFFFF, 4)

    assert shadow.read_mask(base, 1) == 0xFF  # byte 0: tainted
    assert shadow.read_mask(base + 1, 1) == 0xFF  # byte 1: tainted
    assert shadow.read_mask(base + 3, 1) == 0xFF  # byte 3: tainted
    assert shadow.read_mask(base + 4, 1) == 0x00  # byte 4: clean
    assert shadow.read_mask(base, 4) == 0xFFFFFFFF  # full 4 bytes: tainted
    assert shadow.read_mask(base, 8) == 0xFFFFFFFF  # 8 bytes: only first 4 tainted


def test_shadow_memory_cross_page_write_read() -> None:
    """
    Write taint straddling a page boundary (page size = 4096).
    """
    from microtaint.emulator.shadow import BitPreciseShadowMemory

    shadow = BitPreciseShadowMemory()

    # Write starting 4 bytes before a page boundary
    page_boundary = 0x80001000
    addr = page_boundary - 4

    shadow.write_mask(addr, 0xFFFFFFFFFFFFFFFF, 8)

    # All 8 bytes should be tainted, crossing the page
    for i in range(8):
        result = shadow.read_mask(addr + i, 1)
        assert result == 0xFF, f'Cross-page byte {i} expected 0xFF got {hex(result)}'


def test_movzbl_rbp_offset_taint_propagation(amd64_registers: list[Register]) -> None:
    from microtaint.emulator.shadow import BitPreciseShadowMemory
    from microtaint.instrumentation.ast import EvalContext
    from microtaint.sleigh.engine import generate_static_rule
    from microtaint.types import Architecture

    shadow = BitPreciseShadowMemory()
    rbp = 0x80000000DE38
    taint_addr = rbp - 8  # 0x80000000de30

    shadow.write_mask(taint_addr, 0xFF, 1)

    # Verify shadow memory is correct before anything else
    assert shadow.read_mask(taint_addr, 1) == 0xFF, 'Shadow memory write/read broken'
    assert shadow.read_mask(rbp, 1) == 0x00, 'Shadow memory should be clean at RBP itself'

    bytestring = bytes.fromhex('0fb645f8')  # movzbl -0x8(%rbp), %eax
    circuit = generate_static_rule(Architecture.AMD64, bytestring, amd64_registers)

    # Print the circuit assignments
    print('\nCircuit for movzbl -0x8(%rbp), %eax:')
    for a in circuit.assignments:
        print(f'  {a}')

    ctx = EvalContext(
        input_taint={},
        input_values={'RBP': rbp},
        shadow_memory=shadow,
        simulator=CellSimulator(Architecture.AMD64),  # needed otherwise ast.pyx exits
    )

    # Manually evaluate each assignment and trace
    print('\nManual expression evaluation:')
    for a in circuit.assignments:
        if a.expression is not None:
            val = a.expression.evaluate(ctx)
            print(f'  {a.target} => {hex(val)}')

        # If it's a memory operand in the expression tree, trace it
        from microtaint.instrumentation.ast import BinaryExpr, MemoryOperand, TaintOperand

        def trace_expr(expr: Expr, depth: int = 0) -> None:
            indent = '    ' * depth
            if isinstance(expr, MemoryOperand):
                addr = expr.address_expr.evaluate(ctx)
                taint_val = shadow.read_mask(addr, expr.size) if expr.is_taint else None
                print(
                    f"{indent}MemoryOperand(addr={hex(addr)}, size={expr.size}, is_taint={expr.is_taint}) => taint={hex(taint_val) if taint_val is not None else 'N/A'}",
                )
            elif isinstance(expr, TaintOperand):
                val = expr.evaluate(ctx)
                print(f'{indent}TaintOperand({expr.name}, is_taint={expr.is_taint}) => {hex(val)}')
            elif isinstance(expr, BinaryExpr):
                print(f'{indent}BinaryExpr({expr.op})')
                trace_expr(expr.lhs, depth + 1)
                trace_expr(expr.rhs, depth + 1)
            else:
                val = expr.evaluate(ctx)
                print(f'{indent}{type(expr).__name__} => {hex(val)}')

        if a.expression is not None:
            trace_expr(a.expression)

    out = circuit.evaluate(ctx)
    print(f'\nFull output: {out}')

    assert (
        out.get('RAX', 0) != 0
    ), f"EAX should be tainted after loading from tainted memory at RBP-8, got {hex(out.get('RAX', 0))}"
