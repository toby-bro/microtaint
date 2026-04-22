"""
Pre-Release Smoke Test for Microtaint Pipeline.
Ensures that lifting, AST generation, and Unicorn differential simulation
are fully operational before packaging.
"""

from __future__ import annotations

import sys

from microtaint.instrumentation.ast import EvalContext, MemoryOperand
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, Register


def main() -> int:
    print('[*] Initializing Microtaint Smoke Test...')
    arch = Architecture.AMD64
    registers = [
        Register(name='RAX', bits=64),
        Register(name='RBX', bits=64),
        Register(name='CF', bits=1),
    ]

    try:
        simulator = CellSimulator(arch)
        print('[+] Simulator initialized successfully.')
    except Exception as e:
        print(f'[-] FATAL: Failed to initialize Unicorn simulator: {e}')
        return 1

    try:
        # 1. Test MAPPED & Constant Path: MOV RAX, 0x42 (b8 42 00 00 00)
        rule_mov = generate_static_rule(arch, bytes.fromhex('b842000000'), registers)
        assert len(rule_mov.assignments) > 0, 'No assignments for MOV'
        ctx = EvalContext(input_values={'RAX': 0xFF}, input_taint={'RAX': 0xFF}, simulator=simulator)
        assert rule_mov.evaluate(ctx).get('RAX', -1) == 0, 'MOV constant failed to clear taint'
        print('[+] MAPPED/Constant path operational.')

        # 2. Test TRANSPORTABLE Path: ADD RAX, RBX (48 01 d8)
        rule_add = generate_static_rule(arch, bytes.fromhex('4801d8'), registers)

        # 2a. Test Internal Ripple (0xFF + 1 = 0x100 rippling up to bit 8)
        ctx_ripple = EvalContext(
            input_values={'RAX': 0xFF, 'RBX': 0x01},
            input_taint={'RAX': 0x01, 'RBX': 0x00},
            simulator=simulator,
        )
        out_ripple = rule_add.evaluate(ctx_ripple)
        assert out_ripple.get('RAX', 0) == 0x1FF, 'ADD failed to ripple carry taint inside RAX'
        assert out_ripple.get('CF', -1) == 0, "CF improperly tainted when 64-bit boundary wasn't crossed"

        # 2b. Test Boundary Carry (0xFFFFFFFFFFFFFFFF + 1 = 0x0 triggering CF)
        ctx_carry = EvalContext(
            input_values={'RAX': 0xFFFFFFFFFFFFFFFF, 'RBX': 0x00},
            input_taint={'RAX': 0x00, 'RBX': 0x01},
            simulator=simulator,
        )
        out_carry = rule_add.evaluate(ctx_carry)
        assert out_carry.get('CF', -1) == 1, 'ADD failed to trigger CF taint on 64-bit boundary'

        print('[+] TRANSPORTABLE (Carry/Differential) path operational.')

        # 3. Test MEMORY Path: MOV RAX, [RBX] (48 8b 03)
        rule_mem = generate_static_rule(arch, bytes.fromhex('488b03'), registers)
        _mem_assignments = [
            a
            for a in rule_mem.assignments
            if isinstance(a.target, MemoryOperand) or 'MemoryOperand' in str(type(a.target))
        ]
        # Note: If it's a LOAD, the target is a register (RAX), but the dependencies contain a MemoryOperand.
        has_mem_dep = any('Memory' in str(type(d).__name__) for a in rule_mem.assignments for d in a.dependencies)
        assert has_mem_dep, 'Memory Operand parsing failed in AST'
        print('[+] Memory Operand parsing operational.')

        # 4. Test AVALANCHE Path: IMUL RAX, RBX (48 0f af c3)
        rule_imul = generate_static_rule(arch, bytes.fromhex('480fafc3'), registers)
        ctx = EvalContext(
            input_values={'RAX': 0x02, 'RBX': 0x03},
            input_taint={'RAX': 0x01, 'RBX': 0x00},
            simulator=simulator,
        )
        out_imul = rule_imul.evaluate(ctx)
        assert out_imul.get('RAX', 0) == 0xFFFFFFFFFFFFFFFF, 'Avalanche failed to fully taint register'
        print('[+] AVALANCHE path operational.')

    except AssertionError as e:
        print(f'[-] SMOKE TEST FAILED: {e}')
        return 1
    except Exception as e:
        print(f'[-] UNEXPECTED EXCEPTION: {e}')
        return 1

    print('[*] All systems go. Microtaint is ready for PyPI release!')
    return 0


if __name__ == '__main__':
    sys.exit(main())
