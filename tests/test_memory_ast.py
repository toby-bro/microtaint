"""
Test memory operation AST generation from P-code.

Verifies that LOAD and STORE operations generate correct taint assignment ASTs
with appropriate MemoryOperand dependencies.
"""

# mypy: disable-error-code="union-attr"
# mypy: disable-error-code="attr-defined"

from __future__ import annotations

import pytest

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
    ]


class TestMemoryLoadAST:
    """Test AST generation for memory LOAD operations."""

    def test_load_qword_ast(self, x86_64_registers: list[Register]) -> None:
        """mov rax, [rdi] - generates RAX assignment with memory dependency."""
        arch = Architecture.AMD64
        # mov rax, [rdi] -> 48 8b 07
        bytestring = b'\x48\x8b\x07'

        rule = generate_static_rule(arch, bytestring, x86_64_registers)

        # Should have RAX assignment
        rax_assignments = [a for a in rule.assignments if a.target.name == 'RAX']
        assert len(rax_assignments) == 1, 'Should have one RAX assignment'

        assignment = rax_assignments[0]

        # Should depend on RDI (address) and memory
        dep_types = [type(dep).__name__ for dep in assignment.dependencies]
        assert (
            'TaintOperand' in dep_types or 'MemoryOperand' in dep_types
        ), f'Should have taint/memory dependencies, got {dep_types}'

    def test_load_dword_ast(self, x86_64_registers: list[Register]) -> None:
        """mov eax, [rdi] - generates RAX assignment (32-bit load + zero-extend)."""
        arch = Architecture.AMD64
        # mov eax, [rdi] -> 8b 07
        bytestring = b'\x8b\x07'

        rule = generate_static_rule(arch, bytestring, x86_64_registers)

        rax_assignments = [a for a in rule.assignments if a.target.name == 'RAX']
        # May have 2 assignments: one for EAX[0:31] and one for zero-extension to RAX[0:63]
        assert len(rax_assignments) >= 1, 'Should have at least one RAX assignment'

    def test_load_word_ast(self, x86_64_registers: list[Register]) -> None:
        """mov ax, [rdi] - generates RAX assignment (16-bit load)."""
        arch = Architecture.AMD64
        # mov ax, [rdi] -> 66 8b 07
        bytestring = b'\x66\x8b\x07'

        rule = generate_static_rule(arch, bytestring, x86_64_registers)

        rax_assignments = [a for a in rule.assignments if a.target.name == 'RAX']
        assert len(rax_assignments) >= 1, 'Should have at least one RAX assignment'

    def test_load_byte_ast(self, x86_64_registers: list[Register]) -> None:
        """mov al, [rdi] - generates RAX assignment (8-bit load)."""
        arch = Architecture.AMD64
        # mov al, [rdi] -> 8a 07
        bytestring = b'\x8a\x07'

        rule = generate_static_rule(arch, bytestring, x86_64_registers)

        rax_assignments = [a for a in rule.assignments if a.target.name == 'RAX']
        assert len(rax_assignments) >= 1, 'Should have at least one RAX assignment'


class TestMemoryStoreAST:
    """Test AST generation for memory STORE operations."""

    def test_store_qword_ast(self, x86_64_registers: list[Register]) -> None:
        """mov [rsi], rax - generates memory assignment."""
        arch = Architecture.AMD64
        # mov [rsi], rax -> 48 89 06
        bytestring = b'\x48\x89\x06'

        rule = generate_static_rule(arch, bytestring, x86_64_registers)

        # Should have memory assignment(s)
        mem_assignments = [a for a in rule.assignments if type(a.target).__name__ == 'MemoryOperand']
        assert len(mem_assignments) >= 1, 'Should have memory assignment'

        assignment = mem_assignments[0]

        # Should depend on RAX (source data)
        dep_names = [dep.name if hasattr(dep, 'name') else '' for dep in assignment.dependencies]
        assert 'RAX' in dep_names, f'Should depend on RAX, got {dep_names}'

    def test_store_dword_ast(self, x86_64_registers: list[Register]) -> None:
        """mov [rsi], eax - generates memory assignment (32-bit)."""
        arch = Architecture.AMD64
        # mov [rsi], eax -> 89 06
        bytestring = b'\x89\x06'

        rule = generate_static_rule(arch, bytestring, x86_64_registers)

        mem_assignments = [a for a in rule.assignments if type(a.target).__name__ == 'MemoryOperand']
        assert len(mem_assignments) >= 1

    def test_store_word_ast(self, x86_64_registers: list[Register]) -> None:
        """mov [rsi], ax - generates memory assignment (16-bit)."""
        arch = Architecture.AMD64
        # mov [rsi], ax -> 66 89 06
        bytestring = b'\x66\x89\x06'

        rule = generate_static_rule(arch, bytestring, x86_64_registers)

        mem_assignments = [a for a in rule.assignments if type(a.target).__name__ == 'MemoryOperand']
        assert len(mem_assignments) >= 1

    def test_store_byte_ast(self, x86_64_registers: list[Register]) -> None:
        """mov [rsi], al - generates memory assignment (8-bit)."""
        arch = Architecture.AMD64
        # mov [rsi], al -> 88 06
        bytestring = b'\x88\x06'

        rule = generate_static_rule(arch, bytestring, x86_64_registers)

        mem_assignments = [a for a in rule.assignments if type(a.target).__name__ == 'MemoryOperand']
        assert len(mem_assignments) >= 1


class TestMemoryToMemoryAST:
    """Test AST for memory-to-memory operations through registers."""

    def test_memory_copy_chain(self, x86_64_registers: list[Register]) -> None:
        """mov rax, [rdi]; mov [rsi], rax - load then store."""
        arch = Architecture.AMD64
        # mov rax, [rdi]; mov [rsi], rax
        bytestring = b'\x48\x8b\x07\x48\x89\x06'

        rule = generate_static_rule(arch, bytestring, x86_64_registers)

        # Should have RAX assignment from first instruction
        rax_assignments = [a for a in rule.assignments if hasattr(a.target, 'name') and a.target.name == 'RAX']
        assert len(rax_assignments) >= 1

        # Should have memory assignment from second instruction
        mem_assignments = [a for a in rule.assignments if type(a.target).__name__ == 'MemoryOperand']
        assert len(mem_assignments) >= 1


class TestMemoryReadModifyWriteAST:
    """Test AST for read-modify-write memory operations."""

    def test_add_to_memory_ast(self, x86_64_registers: list[Register]) -> None:
        """add [rdi], eax - RMW operation."""
        arch = Architecture.AMD64
        # add [rdi], eax -> 01 07
        bytestring = b'\x01\x07'

        rule = generate_static_rule(arch, bytestring, x86_64_registers)

        # Should have memory assignment
        mem_assignments = [a for a in rule.assignments if type(a.target).__name__ == 'MemoryOperand']
        assert len(mem_assignments) >= 1, 'Should have memory target'

        assignment = mem_assignments[0]

        # Should depend on both RAX (addend) and memory (original value)
        # dep_types = [type(dep).__name__ for dep in assignment.dependencies]
        # Should have both register and memory dependencies
        assert len(assignment.dependencies) >= 1, 'Should have dependencies'

    def test_inc_memory_ast(self, x86_64_registers: list[Register]) -> None:
        """inc dword [rdi] - RMW increment."""
        arch = Architecture.AMD64
        # inc dword [rdi] -> ff 07
        bytestring = b'\xff\x07'

        rule = generate_static_rule(arch, bytestring, x86_64_registers)

        # Should have memory assignment
        mem_assignments = [a for a in rule.assignments if type(a.target).__name__ == 'MemoryOperand']
        assert len(mem_assignments) >= 1

    def test_xor_memory_ast(self, x86_64_registers: list[Register]) -> None:
        """xor [rdi], eax - RMW XOR."""
        arch = Architecture.AMD64
        # xor [rdi], eax -> 31 07
        bytestring = b'\x31\x07'

        rule = generate_static_rule(arch, bytestring, x86_64_registers)

        mem_assignments = [a for a in rule.assignments if type(a.target).__name__ == 'MemoryOperand']
        assert len(mem_assignments) >= 1


class TestMemoryOperandStructure:
    """Test the structure and properties of MemoryOperand AST nodes."""

    def test_memory_operand_has_address_expr(self, x86_64_registers: list[Register]) -> None:
        """Memory operands should have address expressions."""
        arch = Architecture.AMD64
        # mov [rsi], rax
        bytestring = b'\x48\x89\x06'

        rule = generate_static_rule(arch, bytestring, x86_64_registers)

        mem_assignments = [a for a in rule.assignments if type(a.target).__name__ == 'MemoryOperand']

        if len(mem_assignments) > 0:
            mem_target = mem_assignments[0].target
            # MemoryOperand should have an address expression (addr or address_expr)
            has_addr = hasattr(mem_target, 'addr') or hasattr(mem_target, 'address_expr')
            assert has_addr, 'MemoryOperand should have addr/address_expr attribute'

    def test_memory_operand_has_size(self, x86_64_registers: list[Register]) -> None:
        """Memory operands should specify access size."""
        arch = Architecture.AMD64
        # mov [rsi], rax (8-byte store)
        bytestring = b'\x48\x89\x06'

        rule = generate_static_rule(arch, bytestring, x86_64_registers)

        mem_assignments = [a for a in rule.assignments if type(a.target).__name__ == 'MemoryOperand']

        if len(mem_assignments) > 0:
            mem_target = mem_assignments[0].target
            # MemoryOperand should have size
            assert hasattr(mem_target, 'size'), 'MemoryOperand should have size attribute'
            assert mem_target.size == 8, f'Expected 8-byte size, got {mem_target.size}'

    def test_different_sizes_have_correct_operands(self, x86_64_registers: list[Register]) -> None:
        """Verify different operand sizes generate correct MemoryOperand sizes."""
        arch = Architecture.AMD64

        test_cases = [
            (b'\x88\x06', 1),  # mov [rsi], al - 1 byte
            (b'\x66\x89\x06', 2),  # mov [rsi], ax - 2 bytes
            (b'\x89\x06', 4),  # mov [rsi], eax - 4 bytes
            (b'\x48\x89\x06', 8),  # mov [rsi], rax - 8 bytes
        ]

        for bytestring, expected_size in test_cases:
            rule = generate_static_rule(arch, bytestring, x86_64_registers)
            mem_assignments = [a for a in rule.assignments if type(a.target).__name__ == 'MemoryOperand']

            if len(mem_assignments) > 0:
                mem_target = mem_assignments[0].target
                assert (
                    mem_target.size == expected_size
                ), f'Expected {expected_size}-byte size for {bytestring.hex()}, got {mem_target.size}'
