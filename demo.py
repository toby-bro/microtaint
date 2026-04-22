"""
Microtaint Demonstration Script
Showcases the power of bit-precise CellIFT AST generation and execution.
"""

from __future__ import annotations

import pypcode

from microtaint.instrumentation.ast import EvalContext, MemoryOperand
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import _map_sleigh_to_state, generate_static_rule
from microtaint.sleigh.lifter import get_context
from microtaint.sleigh.slicer import get_varnode_id, slice_backward
from microtaint.types import Architecture, Register


def print_header(title: str) -> None:
    print(f"\n{'-'*60}")
    print(f' {title.upper()}')
    print(f"{'-'*60}")


def extract_assignment_slices(
    arch: Architecture,
    state_format: list[Register],
    ctx: pypcode.Context,
    translation: pypcode.Translation,
    out_name: str,
    matched_ops: list[str],
    op: pypcode.PcodeOp,
) -> None:
    """Helper function to print the backward P-Code slice for a given target."""
    if op.output and op.output.space.name == 'register':
        mapped = _map_sleigh_to_state(ctx, arch.name, state_format, op.output.offset, op.output.size)

        if mapped and mapped.name == out_name:
            s_ops = slice_backward(translation.ops, op.output)
            for s_op in reversed(s_ops):
                inps: list[str] = []
                for inp in s_op.inputs:
                    if inp.space.name == 'register':
                        r_map = _map_sleigh_to_state(ctx, arch.name, state_format, inp.offset, inp.size)
                        inps.append(r_map.name if r_map else get_varnode_id(inp))
                    elif inp.space.name == 'const':
                        inps.append(hex(inp.offset))
                    else:
                        inps.append(get_varnode_id(inp))

                out = ''
                if s_op.output:
                    if s_op.output.space.name == 'register':
                        r_map = _map_sleigh_to_state(
                            ctx,
                            arch.name,
                            state_format,
                            s_op.output.offset,
                            s_op.output.size,
                        )
                        out = r_map.name if r_map else get_varnode_id(s_op.output)
                    else:
                        out = get_varnode_id(s_op.output)
                    out += ' = '

                op_str = f'   > {out}{s_op.opcode.name}({", ".join(inps)})'
                if op_str not in matched_ops:
                    matched_ops.append(op_str)


def main() -> None:
    print('\nWelcome to Microtaint: Bit-Precise Taint Propagation')
    arch = Architecture.AMD64
    simulator = CellSimulator(arch)
    ctx = get_context(arch)

    # Standard 64-bit registers + 1-bit flags
    state_format = [
        Register(name='RAX', bits=64),
        Register(name='RBX', bits=64),
        Register(name='ZF', bits=1),
        Register(name='CF', bits=1),
    ]

    # =========================================================================
    # SCENARIO 1: Exact Bit Masking via Unicorn Differentials
    # =========================================================================
    print_header('Scenario 1: Exact Bit Masking (AND EAX, 0x0F0F0F0F)')
    print("A naive engine would say: 'EAX is tainted, we ANDed it, it's still tainted.'")
    print('Microtaint uses CellIFT differentials to prove exactly which bits survive.')

    # AND EAX, 0x0F0F0F0F -> 25 0f 0f 0f 0f
    byte_str_and = bytes.fromhex('250f0f0f0f')
    rule_and = generate_static_rule(arch, byte_str_and, state_format)
    translation_and = ctx.translate(byte_str_and, 0x1000)

    print('\n[AST Logic Circuit Generated]')
    for assign in rule_and.assignments:
        tgt = assign.target
        if isinstance(tgt, MemoryOperand):
            print(f'  [+] Target: MEM [{tgt.address_expr}] ({tgt.size * 8} bits)')
            out_name = f'MEM_{tgt.size}'
        else:
            bits = tgt.bit_end - tgt.bit_start + 1
            print(f'  [+] Target: {tgt.name} [{tgt.bit_start}:{tgt.bit_end}] ({bits} bits)')
            out_name = tgt.name

        print(f'  [-] Formula: {assign.expression}')
        print('  [-] P-Code Slice:')
        matched_ops: list[str] = []
        for op in translation_and.ops:
            extract_assignment_slices(arch, state_format, ctx, translation_and, out_name, matched_ops, op)
        for p in matched_ops:
            print(p)

    # Evaluate the circuit dynamically
    ctx_and = EvalContext(
        input_values={'RAX': 0xFFFFFFFF},
        input_taint={'RAX': 0xFFFFFFFF},  # Fully tainted
        simulator=simulator,
    )

    out_and = rule_and.evaluate(ctx_and)
    print('\n[Execution Results]')
    print(f'  Input Taint:  {hex(0xFFFFFFFF)}')
    print(f'  Applied Mask: {hex(0x0F0F0F0F)}')
    print(f"  Output Taint: {hex(out_and.get('RAX', 0))} (Bits forced to 0 safely lost their taint!)")

    # =========================================================================
    # SCENARIO 2: Ripple Carry and Flag Boundaries
    # =========================================================================
    print_header('Scenario 2: Ripple Carry & Flags (ADD RAX, RBX)')
    print('Microtaint detects transportable arithmetic and calculates exact carry chains.')

    # ADD RAX, RBX -> 48 01 d8
    byte_str_add = bytes.fromhex('4801d8')
    rule_add = generate_static_rule(arch, byte_str_add, state_format)
    translation_add = ctx.translate(byte_str_add, 0x1000)

    print('\n[AST Logic Circuit Generated for RAX & CF]')
    for assign in rule_add.assignments:
        target_name = getattr(assign.target, 'name', '')
        if target_name in ('RAX', 'CF'):
            print(f'  [+] Target: {target_name}')
            print(f'  [-] Formula: {assign.expression}')
            print('  [-] P-Code Slice:')
            matched_ops = []
            for op in translation_add.ops:
                extract_assignment_slices(arch, state_format, ctx, translation_add, target_name, matched_ops, op)
            for p in matched_ops:
                print(p)
            print('')

    # We add 0xFF + 0x01. Bit 0 is tainted. The carry ripples all the way to bit 8!
    ctx_add = EvalContext(
        input_values={'RAX': 0xFF, 'RBX': 0x01},
        input_taint={'RAX': 0x01, 'RBX': 0x00},
        simulator=simulator,
    )

    out_add = rule_add.evaluate(ctx_add)
    print('[Execution Results]')
    print('  Operation: 0xFF + 0x01')
    print('  Input Taint: Bit 0 only (0x1)')
    print(f"  Output RAX Taint: {hex(out_add.get('RAX', 0))} (The taint successfully rippled up 8 bits!)")
    print(f"  Output CF Taint:  {out_add.get('CF', 0)} (No 64-bit boundary overflow occurred, so CF is untainted)")


if __name__ == '__main__':
    main()
