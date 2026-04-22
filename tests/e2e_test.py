from __future__ import annotations

import pytest

from microtaint.classifier.categories import InstructionCategory
from microtaint.sleigh.engine import generate_static_rule
from microtaint.sleigh.lifter import get_context
from microtaint.types import Architecture, Register


@pytest.fixture
def x86_registers() -> list[Register]:
    return [
        Register(name='RAX', bits=64),
        Register(name='RBX', bits=64),
        Register(name='RCX', bits=64),
        Register(name='RDX', bits=64),
        Register(name='RSP', bits=64),
        Register(name='RBP', bits=64),
        Register(name='RSI', bits=64),
        Register(name='RDI', bits=64),
        Register(name='ZF', bits=1),
        Register(name='CF', bits=1),
        Register(name='SF', bits=1),
        Register(name='OF', bits=1),
        Register(name='PF', bits=1),
    ]


def test_mov_register(x86_registers: list[Register]) -> None:
    # MOV RAX, RBX -> \x48\x89\xd8
    arch = Architecture.AMD64
    byte_string = b'\x48\x89\xd8'

    rule = generate_static_rule(arch, byte_string, x86_registers)

    # Filter to only RAX
    rax_assignments = [a for a in rule.assignments if a.target.name == 'RAX']
    assert len(rax_assignments) == 1

    assignment = rax_assignments[0]
    deps = [dep.name for dep in assignment.dependencies]

    assert 'RBX' in deps


def test_add_register(x86_registers: list[Register]) -> None:
    # ADD RAX, RBX -> \x48\x01\xd8
    arch = Architecture.AMD64
    byte_string = b'\x48\x01\xd8'

    rule = generate_static_rule(arch, byte_string, x86_registers)

    targets = {a.target.name for a in rule.assignments}
    assert 'RAX' in targets
    assert 'ZF' in targets
    assert 'CF' in targets
    assert 'SF' in targets
    assert 'OF' in targets

    rax_assignments = [a for a in rule.assignments if a.target.name == 'RAX']
    assignment = rax_assignments[0]
    deps = [dep.name for dep in assignment.dependencies]

    assert 'RBX' in deps
    assert 'RAX' in deps


def test_xor_self(x86_registers: list[Register]) -> None:
    # XOR RAX, RAX -> \x48\x31\xc0
    arch = Architecture.AMD64
    byte_string = b'\x48\x31\xc0'

    rule = generate_static_rule(arch, byte_string, x86_registers)
    rax_assignments = [a for a in rule.assignments if a.target.name == 'RAX']
    assert len(rax_assignments) == 1


def test_load_memory(x86_registers: list[Register]) -> None:
    # MOV RAX, [RBX] -> \x48\x8b\x03
    arch = Architecture.AMD64
    byte_string = b'\x48\x8b\x03'

    rule = generate_static_rule(arch, byte_string, x86_registers)

    rax_assignments = [a for a in rule.assignments if a.target.name == 'RAX']
    assert len(rax_assignments) == 1
    assignment = rax_assignments[0]

    # Get names from dependencies that have a .name attribute (TaintOperand)
    deps = [dep.name for dep in assignment.dependencies if hasattr(dep, 'name')]
    assert 'RBX' in deps  # It needs the address
    # Should also have a MemoryOperand dependency
    assert any(type(dep).__name__ == 'MemoryOperand' for dep in assignment.dependencies)


def test_push_register(x86_registers: list[Register]) -> None:
    # PUSH RAX -> \x50
    arch = Architecture.AMD64
    byte_string = b'\x50'

    rule = generate_static_rule(arch, byte_string, x86_registers)

    # PUSH writes to memory, so we look for memory targets instead of RSP register targets
    rsp_assignments = [a for a in rule.assignments if hasattr(a.target, 'name') and a.target.name == 'RSP']
    if rsp_assignments:
        assignment = rsp_assignments[0]
        deps = [dep.name for dep in assignment.dependencies if hasattr(dep, 'name')]
        assert 'RSP' in deps
    # PUSH also creates a memory store assignment
    mem_assignments = [a for a in rule.assignments if type(a.target).__name__ == 'MemoryOperand']
    assert len(mem_assignments) >= 1, 'PUSH should create memory store'


def test_mul_register(x86_registers: list[Register]) -> None:
    # MUL RBX -> \x48\xf7\xe3
    arch = Architecture.AMD64
    byte_string = b'\x48\xf7\xe3'

    rule = generate_static_rule(arch, byte_string, x86_registers)
    targets = {a.target.name for a in rule.assignments}
    assert 'RAX' in targets
    assert 'RDX' in targets


def test_branch_instruction(x86_registers: list[Register]) -> None:
    # JMP RAX -> \xff\xe0
    # Includes RIP as well
    regs = [*x86_registers, Register(name='RIP', bits=64)]
    arch = Architecture.AMD64
    byte_string = b'\xff\xe0'

    rule = generate_static_rule(arch, byte_string, regs)
    rip_assignments = [a for a in rule.assignments if a.target.name == 'RIP']
    assert len(rip_assignments) > 0
    assert 'RAX' in [d.name for d in rip_assignments[0].dependencies]

    # Conditional branch: JZ +10 (74 0a)
    byte_string_jz = b'\x74\x0a'
    rule_jz = generate_static_rule(arch, byte_string_jz, regs)
    jz_assignments = [a for a in rule_jz.assignments if a.target.name == 'RIP']
    assert len(jz_assignments) > 0
    deps = [d.name for d in jz_assignments[0].dependencies]
    assert 'ZF' in deps


def test_x86_flags_mapping() -> None:
    # X86 flags mapping test (when FLAGS register is provided)
    arch = Architecture.X86
    regs = [Register(name='EAX', bits=32), Register(name='EBX', bits=32), Register(name='EFLAGS', bits=32)]
    # ADD EAX, EBX -> \x01\xd8
    byte_string = b'\x01\xd8'
    rule = generate_static_rule(arch, byte_string, regs)
    targets = {a.target.name for a in rule.assignments}
    assert 'EFLAGS' in targets


def test_jump_const(x86_registers: list[Register]) -> None:
    # JMP +10 -> \xeb\x0a
    arch = Architecture.AMD64
    byte_string = b'\xeb\x0a'
    regs = [*x86_registers, Register(name='RIP', bits=64)]
    rule = generate_static_rule(arch, byte_string, regs)
    assert len(rule.assignments) == 0


def test_missing_pc_reg(x86_registers: list[Register]) -> None:
    # JMP RAX -> \xff\xe0
    # Omitting RIP from regs should cause the branch PC block to skip
    arch = Architecture.AMD64
    byte_string = b'\xff\xe0'
    rule = generate_static_rule(arch, byte_string, x86_registers)  # x86_registers has no RIP
    # It might still generate assignments for RAX if there were any, but mostly harmless
    # We just want to make sure it doesn't crash
    assert rule.architecture == arch


def test_sub_polarity(x86_registers: list[Register]) -> None:
    # SUB RAX, RBX -> \x48\x29\xd8
    arch = Architecture.AMD64
    byte_string = b'\x48\x29\xd8'
    rule = generate_static_rule(arch, byte_string, x86_registers)
    rax_assignments = [a for a in rule.assignments if a.target.name == 'RAX']
    # This hits lines 142/143 where polarity p==0
    assert len(rax_assignments) > 0


def test_invalid_register() -> None:
    # Test register falling back to None context mapping
    arch = Architecture.AMD64
    byte_string = b'\x48\x89\xd8'
    regs = [Register(name='INVALID_REG', bits=64)]
    rule = generate_static_rule(arch, byte_string, regs)
    assert len(rule.assignments) == 0


def test_unknown_category() -> None:
    # CPUID -> \x0f\xa2
    arch = Architecture.AMD64
    regs = [Register('RAX', 64), Register('RBX', 64), Register('RCX', 64), Register('RDX', 64)]
    byte_string = b'\x0f\xa2'
    rule = generate_static_rule(arch, byte_string, regs)
    # CPUID is usually completely unmapped or heavily complex
    t = {a.target.name for a in rule.assignments}
    assert 'RAX' in t


def test_avalanche_pc(x86_registers: list[Register]) -> None:
    # Need to hit lines 196 (Avalanche on Mapped PCs) and 202 (Avalanche on Unknown PCs)
    # CBRANCH is mapped to EIP normally
    arch = Architecture.AMD64
    # RET -> \xc3
    byte_string = b'\xc3'
    regs = [*x86_registers, Register(name='RIP', bits=64)]
    rule = generate_static_rule(arch, byte_string, regs)
    assigned = [a.target.name for a in rule.assignments]
    assert 'RIP' in assigned


def test_imul_avalanche(x86_registers: list[Register]) -> None:
    # IMUL RBX -> \x48\xf7\xe3
    arch = Architecture.AMD64
    byte_string = b'\x48\xf7\xe3'
    rule = generate_static_rule(arch, byte_string, x86_registers)
    t = {a.target.name for a in rule.assignments}
    assert 'RAX' in t


def test_not_polarity(x86_registers: list[Register]) -> None:
    # NOT RAX -> \x48\xf7\xd0
    arch = Architecture.AMD64
    byte_string = b'\x48\xf7\xd0'
    rule = generate_static_rule(arch, byte_string, x86_registers)
    rax_assignments = [a for a in rule.assignments if a.target.name == 'RAX']
    assert len(rax_assignments) > 0


def test_unsupported_arch_error() -> None:
    with pytest.raises(ValueError, match='Unsupported architecture'):
        get_context('XYZ')


def test_unknown_category_str() -> None:
    assert str(InstructionCategory.UNKNOWN) == 'Unknown'
