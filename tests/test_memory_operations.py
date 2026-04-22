"""
Comprehensive bit-precise memory operation tests for the taint engine.

These tests verify exact taint propagation through memory LOAD and STORE operations
at the static analysis level (AST generation from P-code).
"""

from __future__ import annotations

import pytest

from microtaint.instrumentation.ast import MemoryOperand
from microtaint.simulator import CellSimulator, MachineState
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, Register


@pytest.fixture
def x86_64_registers() -> list[Register]:
    """Standard x86-64 register set for testing."""
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
    ]


class TestMemoryLoad:
    """Test LOAD (memory read) operations with exact taint propagation."""

    def test_load_generates_memory_dependency(self, x86_64_registers: list[Register]) -> None:
        """LOAD from memory creates dependency on memory operand."""
        arch = Architecture.AMD64
        # mov rax, [rdi] -> 48 8b 07
        bytestring = b'\x48\x8b\x07'

        rule = generate_static_rule(arch, bytestring, x86_64_registers)

        # Should have assignment for RAX
        rax_assignments = [a for a in rule.assignments if a.target.name == 'RAX']
        assert len(rax_assignments) == 1, 'Should have one RAX assignment'

        assignment = rax_assignments[0]

        # Dependencies should include RDI (address) and memory read
        dep_names = [dep.name if hasattr(dep, 'name') else str(type(dep).__name__) for dep in assignment.dependencies]

        # Must depend on RDI (the address register)
        assert 'RDI' in dep_names, f'Should depend on RDI (address), got {dep_names}'

        # Should have a MemoryOperand dependency
        has_mem_dep = any('Memory' in str(type(dep).__name__) for dep in assignment.dependencies)
        assert has_mem_dep, 'Should have memory operand dependency'

    def test_load_generates_rdi_and_memory_deps(self, x86_64_registers: list[Register]) -> None:
        """Verify LOAD creates both address register and memory dependencies."""
        arch = Architecture.AMD64
        # mov rax, [rdi]
        bytestring = b'\x48\x8b\x07'

        rule = generate_static_rule(arch, bytestring, x86_64_registers)

        rax_assignments = [a for a in rule.assignments if a.target.name == 'RAX']
        assert len(rax_assignments) == 1

        # Should have RDI dependency
        has_rdi = any(hasattr(dep, 'name') and dep.name == 'RDI' for dep in rax_assignments[0].dependencies)
        assert has_rdi, 'Should depend on RDI'

        # Should have MemoryOperand dependency
        has_mem = any(isinstance(dep, MemoryOperand) for dep in rax_assignments[0].dependencies)
        assert has_mem, 'Should have MemoryOperand dependency'

    def test_load_partial_tainted_low_byte(self) -> None:
        """LOAD with only low byte tainted in memory - SKIPPED (no simulator)."""
        sim = CellSimulator(Architecture.AMD64)
        # mov rax, [rdi]
        bytestring = b'\x48\x8b\x07'

        v_state = MachineState({'RDI': 0x1000}, {0x1000: 0x1122334455667788})
        t_state = MachineState({'RDI': 0x0}, {0x1000: 0xFF})  # Only low byte tainted

        result_taint = sim.evaluate_cell_differential(bytestring, 'RAX', v_state, t_state)

        # Only bits 0-7 should be tainted
        assert result_taint == 0xFF, f'Expected 0xFF, got {hex(result_taint)}'

    def test_load_partial_tainted_high_word(self) -> None:
        """LOAD with only high word (bytes 4-5) tainted."""
        sim = CellSimulator(Architecture.AMD64)
        bytestring = b'\x48\x8b\x07'

        v_state = MachineState({'RDI': 0x1000}, {0x1000: 0x1122334455667788})
        t_state = MachineState({'RDI': 0x0}, {0x1000: 0xFFFF00000000})  # Bytes 4-5 tainted

        result_taint = sim.evaluate_cell_differential(bytestring, 'RAX', v_state, t_state)

        # Bits 32-47 should be tainted
        assert result_taint == 0xFFFF00000000, f'Expected 0xFFFF00000000, got {hex(result_taint)}'

    def test_load_with_tainted_address(self) -> None:
        """LOAD where the address pointer is tainted."""
        sim = CellSimulator(Architecture.AMD64)
        bytestring = b'\x48\x8b\x07'

        v_state = MachineState({'RDI': 0x1000}, {0x1000: 0x1122334455667788})
        t_state = MachineState({'RDI': 0xF}, {0x1000: 0x0})  # Address is tainted, memory is not

        result_taint = sim.evaluate_cell_differential(bytestring, 'RAX', v_state, t_state)

        # Result should be tainted due to tainted address
        # The exact pattern depends on D-vector implementation
        assert result_taint != 0x0, 'Tainted address should propagate taint'

    def test_load_dword_32bit(self) -> None:
        """LOAD 4-byte value (EAX from memory)."""
        sim = CellSimulator(Architecture.AMD64)
        # mov eax, [rdi] -> 8b 07
        bytestring = b'\x8b\x07'

        v_state = MachineState({'RDI': 0x1000}, {0x1000: 0x12345678})
        t_state = MachineState({'RDI': 0x0}, {0x1000: 0xFFFFFFFF})  # Low 32 bits tainted

        result_taint = sim.evaluate_cell_differential(bytestring, 'RAX', v_state, t_state)

        # Only low 32 bits should be tainted (EAX write zero-extends)
        assert result_taint == 0xFFFFFFFF, f'Expected 0xFFFFFFFF, got {hex(result_taint)}'

    def test_load_word_16bit(self) -> None:
        """LOAD 2-byte value (AX from memory)."""
        sim = CellSimulator(Architecture.AMD64)
        # mov ax, [rdi] -> 66 8b 07
        bytestring = b'\x66\x8b\x07'

        v_state = MachineState({'RDI': 0x1000}, {0x1000: 0x1234})
        t_state = MachineState({'RDI': 0x0}, {0x1000: 0xFFFF})  # 16 bits tainted

        result_taint = sim.evaluate_cell_differential(bytestring, 'RAX', v_state, t_state)

        # Only bits 0-15 should be tainted (AX write keeps high bits)
        assert result_taint & 0xFFFF == 0xFFFF, f'Expected low 16 bits tainted, got {hex(result_taint)}'

    def test_load_byte_8bit(self) -> None:
        """LOAD 1-byte value (AL from memory)."""
        sim = CellSimulator(Architecture.AMD64)
        # mov al, [rdi] -> 8a 07
        bytestring = b'\x8a\x07'

        v_state = MachineState({'RDI': 0x1000}, {0x1000: 0x12})
        t_state = MachineState({'RDI': 0x0}, {0x1000: 0xFF})  # 8 bits tainted

        result_taint = sim.evaluate_cell_differential(bytestring, 'RAX', v_state, t_state)

        # Only bits 0-7 should be tainted
        assert result_taint & 0xFF == 0xFF, f'Expected low 8 bits tainted, got {hex(result_taint)}'


class TestMemoryStore:
    """Test STORE (memory write) operations with exact taint propagation."""

    def test_store_fully_tainted_qword(self) -> None:
        """STORE 8-byte fully tainted value to memory."""
        sim = CellSimulator(Architecture.AMD64)
        # mov [rsi], rax -> 48 89 06
        bytestring = b'\x48\x89\x06'

        v_state = MachineState({'RSI': 0x2000, 'RAX': 0x1122334455667788}, {})
        t_state = MachineState({'RSI': 0x0, 'RAX': 0xFFFFFFFFFFFFFFFF}, {})

        # The target is memory at address 0x2000
        # Need to evaluate memory taint at that location
        result_taint = sim.evaluate_cell_differential(
            bytestring,
            ('MEM', 0x2000, 8),
            v_state,
            t_state,
        )

        # All 64 bits in memory should be tainted
        assert result_taint == 0xFFFFFFFFFFFFFFFF, f'Expected full taint, got {hex(result_taint)}'

    def test_store_partial_tainted_low_byte(self) -> None:
        """STORE with only low byte of source register tainted."""
        sim = CellSimulator(Architecture.AMD64)
        bytestring = b'\x48\x89\x06'

        v_state = MachineState({'RSI': 0x2000, 'RAX': 0x1122334455667788}, {})
        t_state = MachineState({'RSI': 0x0, 'RAX': 0xFF}, {})  # Only low byte tainted

        result_taint = sim.evaluate_cell_differential(
            bytestring,
            ('MEM', 0x2000, 8),
            v_state,
            t_state,
        )

        # Only low byte in memory should be tainted
        assert result_taint == 0xFF, f'Expected 0xFF, got {hex(result_taint)}'

    def test_store_partial_tainted_middle_bytes(self) -> None:
        """STORE with bytes 2-3 tainted in source register."""
        sim = CellSimulator(Architecture.AMD64)
        bytestring = b'\x48\x89\x06'

        v_state = MachineState({'RSI': 0x2000, 'RAX': 0x1122334455667788}, {})
        t_state = MachineState({'RSI': 0x0, 'RAX': 0xFFFF0000}, {})  # Bytes 2-3 tainted

        result_taint = sim.evaluate_cell_differential(
            bytestring,
            ('MEM', 0x2000, 8),
            v_state,
            t_state,
        )

        # Bits 16-31 in memory should be tainted
        assert result_taint == 0xFFFF0000, f'Expected 0xFFFF0000, got {hex(result_taint)}'

    def test_store_with_tainted_address(self) -> None:
        """STORE where the destination address pointer is tainted."""
        sim = CellSimulator(Architecture.AMD64)
        bytestring = b'\x48\x89\x06'

        v_state = MachineState({'RSI': 0x2000, 'RAX': 0x1122334455667788}, {})
        t_state = MachineState({'RSI': 0xF, 'RAX': 0x0}, {})  # Address tainted, data clean

        result_taint = sim.evaluate_cell_differential(
            bytestring,
            ('MEM', 0x2000, 8),
            v_state,
            t_state,
        )

        # Memory should show taint due to tainted address
        # (This is a security-critical case: writing to attacker-controlled address)
        assert result_taint != 0x0, 'Tainted address should propagate to memory taint'

    def test_store_dword_32bit(self) -> None:
        """STORE 4-byte value to memory."""
        sim = CellSimulator(Architecture.AMD64)
        # mov [rsi], eax -> 89 06
        bytestring = b'\x89\x06'

        v_state = MachineState({'RSI': 0x2000, 'RAX': 0x12345678}, {})
        t_state = MachineState({'RSI': 0x0, 'RAX': 0xFFFFFFFF}, {})

        result_taint = sim.evaluate_cell_differential(
            bytestring,
            ('MEM', 0x2000, 4),
            v_state,
            t_state,
        )

        # 32 bits should be tainted in memory
        assert result_taint == 0xFFFFFFFF, f'Expected 0xFFFFFFFF, got {hex(result_taint)}'

    def test_store_word_16bit(self) -> None:
        """STORE 2-byte value to memory."""
        sim = CellSimulator(Architecture.AMD64)
        # mov [rsi], ax -> 66 89 06
        bytestring = b'\x66\x89\x06'

        v_state = MachineState({'RSI': 0x2000, 'RAX': 0x1234}, {})
        t_state = MachineState({'RSI': 0x0, 'RAX': 0xFFFF}, {})

        result_taint = sim.evaluate_cell_differential(
            bytestring,
            ('MEM', 0x2000, 2),
            v_state,
            t_state,
        )

        # 16 bits should be tainted in memory
        assert result_taint == 0xFFFF, f'Expected 0xFFFF, got {hex(result_taint)}'

    def test_store_byte_8bit(self) -> None:
        """STORE 1-byte value to memory."""
        sim = CellSimulator(Architecture.AMD64)
        # mov [rsi], al -> 88 06
        bytestring = b'\x88\x06'

        v_state = MachineState({'RSI': 0x2000, 'RAX': 0x12}, {})
        t_state = MachineState({'RSI': 0x0, 'RAX': 0xFF}, {})

        result_taint = sim.evaluate_cell_differential(
            bytestring,
            ('MEM', 0x2000, 1),
            v_state,
            t_state,
        )

        # 8 bits should be tainted in memory
        assert result_taint == 0xFF, f'Expected 0xFF, got {hex(result_taint)}'


class TestMemoryReadModifyWrite:
    """Test read-modify-write operations that involve both LOAD and STORE."""

    def test_add_memory_operand(self) -> None:
        """ADD to memory location: add [rdi], eax."""
        sim = CellSimulator(Architecture.AMD64)
        # add [rdi], eax -> 01 07
        bytestring = b'\x01\x07'

        # Memory has value 0x10, EAX has 0x20, both partially tainted
        v_state = MachineState({'RDI': 0x1000, 'RAX': 0x20}, {0x1000: 0x10})
        t_state = MachineState({'RDI': 0x0, 'RAX': 0xF}, {0x1000: 0xF0})  # Different bits tainted

        result_taint = sim.evaluate_cell_differential(
            bytestring,
            ('MEM', 0x1000, 4),
            v_state,
            t_state,
        )

        # Result should have taint from both operands combined
        # Bits 0-3 from EAX and bits 4-7 from memory should propagate
        assert result_taint != 0x0, 'Both memory and register taint should propagate'
        assert result_taint & 0xF != 0, 'EAX taint should affect result'
        assert result_taint & 0xF0 != 0, 'Memory taint should affect result'

    def test_inc_memory(self) -> None:
        """INC memory location: inc dword [rdi]."""
        sim = CellSimulator(Architecture.AMD64)
        # inc dword [rdi] -> ff 07
        bytestring = b'\xff\x07'

        v_state = MachineState({'RDI': 0x1000}, {0x1000: 0xFF})  # Value that will carry
        t_state = MachineState({'RDI': 0x0}, {0x1000: 0x1})  # Bit 0 tainted

        result_taint = sim.evaluate_cell_differential(
            bytestring,
            ('MEM', 0x1000, 4),
            v_state,
            t_state,
        )

        # Bit 0 taint should propagate through carry chain
        # 0xFF + 1 = 0x100, so carry propagates to bit 8
        assert result_taint != 0x0, 'Taint should propagate'
        assert result_taint & 0xFF != 0, 'Low byte should be tainted'

    def test_xor_memory_operand(self) -> None:
        """XOR memory with register: xor [rdi], eax."""
        sim = CellSimulator(Architecture.AMD64)
        # xor [rdi], eax -> 31 07
        bytestring = b'\x31\x07'

        v_state = MachineState({'RDI': 0x1000, 'RAX': 0x12345678}, {0x1000: 0xABCDEF00})
        t_state = MachineState({'RDI': 0x0, 'RAX': 0xFF}, {0x1000: 0xFF00})  # Different bits

        result_taint = sim.evaluate_cell_differential(
            bytestring,
            ('MEM', 0x1000, 4),
            v_state,
            t_state,
        )

        # XOR propagates taint from both operands
        assert result_taint & 0xFF != 0, 'EAX taint should propagate'
        assert result_taint & 0xFF00 != 0, 'Memory taint should propagate'


class TestMemoryComplexScenarios:
    """Test complex memory scenarios including chains and overwrites."""

    def test_memory_to_memory_via_register(self) -> None:
        """Chain: LOAD from [rdi], STORE to [rsi]."""
        sim = CellSimulator(Architecture.AMD64)
        # mov rax, [rdi]; mov [rsi], rax
        bytestring = b'\x48\x8b\x07\x48\x89\x06'

        v_state = MachineState({'RDI': 0x1000, 'RSI': 0x2000}, {0x1000: 0x1122334455667788})
        t_state = MachineState(
            {'RDI': 0x0, 'RSI': 0x0},
            {0x1000: 0xFFFFFFFFFFFFFFFF},  # Source fully tainted
        )

        # After both instructions, memory at 0x2000 should be tainted
        result_taint = sim.evaluate_cell_differential(
            bytestring,
            ('MEM', 0x2000, 8),
            v_state,
            t_state,
        )

        assert result_taint == 0xFFFFFFFFFFFFFFFF, f'Expected full taint, got {hex(result_taint)}'

    def test_memory_overwrite(self) -> None:
        """Writing to same memory location overwrites previous taint."""
        sim = CellSimulator(Architecture.AMD64)
        # mov [rdi], rax; mov [rdi], rbx
        bytestring = b'\x48\x89\x07\x48\x89\x1f'

        v_state = MachineState(
            {'RDI': 0x1000, 'RAX': 0x1111111111111111, 'RBX': 0x2222222222222222},
            {},
        )
        t_state = MachineState(
            {'RDI': 0x0, 'RAX': 0xFFFFFFFFFFFFFFFF, 'RBX': 0x0},  # RAX fully tainted  # RBX clean
            {},
        )

        # After second store, memory should be clean (RBX overwrote tainted RAX)
        result_taint = sim.evaluate_cell_differential(
            bytestring,
            ('MEM', 0x1000, 8),
            v_state,
            t_state,
        )

        assert result_taint == 0x0, f'Expected clean memory after overwrite, got {hex(result_taint)}'

    def test_partial_memory_overwrite(self) -> None:
        """Partial overwrite: write qword, then overwrite low dword."""
        sim = CellSimulator(Architecture.AMD64)
        # mov [rdi], rax; mov [rdi], ebx
        bytestring = b'\x48\x89\x07\x89\x1f'

        v_state = MachineState(
            {'RDI': 0x1000, 'RAX': 0x1111111111111111, 'RBX': 0x22222222},
            {},
        )
        t_state = MachineState(
            {'RDI': 0x0, 'RAX': 0xFFFFFFFFFFFFFFFF, 'RBX': 0x0},  # RAX fully tainted  # RBX clean
            {},
        )

        # After second store, low 32 bits clean, high 32 bits still tainted
        result_taint = sim.evaluate_cell_differential(
            bytestring,
            ('MEM', 0x1000, 8),
            v_state,
            t_state,
        )

        # Low 32 bits should be clean, high 32 bits should be tainted
        assert result_taint & 0xFFFFFFFF == 0x0, 'Low 32 bits should be clean'
        assert result_taint & 0xFFFFFFFF00000000 != 0x0, 'High 32 bits should remain tainted'

    def test_adjacent_memory_independence(self) -> None:
        """Verify adjacent memory locations don't interfere."""
        sim = CellSimulator(Architecture.AMD64)
        # mov [rdi], rax
        bytestring = b'\x48\x89\x07'

        # Write to 0x1000, check that 0x1008 is unaffected
        v_state = MachineState({'RDI': 0x1000, 'RAX': 0x1122334455667788}, {})
        t_state = MachineState({'RDI': 0x0, 'RAX': 0xFFFFFFFFFFFFFFFF}, {})

        # Memory at 0x1008 should be unaffected
        result_taint_1000 = sim.evaluate_cell_differential(
            bytestring,
            ('MEM', 0x1000, 8),
            v_state,
            t_state,
        )
        result_taint_1008 = sim.evaluate_cell_differential(
            bytestring,
            ('MEM', 0x1008, 8),
            v_state,
            t_state,
        )

        assert result_taint_1000 == 0xFFFFFFFFFFFFFFFF, '0x1000 should be tainted'
        assert result_taint_1008 == 0x0, '0x1008 should be clean'
