from __future__ import annotations

import pypcode

from microtaint.sleigh.engine import _map_sleigh_to_state, generate_static_rule
from microtaint.sleigh.lifter import get_context
from microtaint.sleigh.slicer import _get_varnode_id, slice_backward
from microtaint.types import Architecture, Register


def main() -> None:
    arch = Architecture.AMD64

    # We define the 64-bit registers and the 8-bit flags used by SLEIGH
    state_format = [
        Register(name='RAX', bits=64),
        Register(name='RBX', bits=64),
        Register(name='ZF', bits=8),
        Register(name='CF', bits=8),
        Register(name='SF', bits=8),
        Register(name='OF', bits=8),
    ]

    # ADD RAX, RBX
    byte_string = b'\x48\x01\xd8'

    print(f'Analyzing Instruction: ADD RAX, RBX (Bytes: {byte_string.hex()})')
    print('=' * 60)

    ctx = get_context(arch)
    translation = ctx.translate(byte_string, 0x1000)

    rule = generate_static_rule(arch, byte_string, state_format)

    for assignment in rule.assignments:
        tgt = assignment.target
        bits = tgt.bit_end - tgt.bit_start + 1

        print(f'[+] Target: {tgt.name} [{tgt.bit_start}:{tgt.bit_end}] ({bits} bits)')
        print(f'[-] Dependencies: {[dep.name for dep in assignment.dependencies]}')
        print('[-] Sleigh Semantics Slice:')

        out_name = tgt.name
        matched_ops: list[str] = []

        # Output all operations that construct this assignment target
        for op in translation.ops:
            extract_assignment_slices(arch, state_format, ctx, translation, out_name, matched_ops, op)

        for p in matched_ops:
            print(p)
        print(f'[-] Formula:\n   {assignment.expression}')
        print('-' * 60)


def extract_assignment_slices(
    arch: Architecture,
    state_format: list[Register],
    ctx: pypcode.Context,
    translation: pypcode.Translation,
    out_name: str,
    matched_ops: list[str],
    op: pypcode.PcodeOp,
) -> None:
    if op.output and op.output.space.name == 'register':
        mapped = _map_sleigh_to_state(ctx, arch, state_format, op.output.offset, op.output.size)
        if mapped and mapped[0] == out_name:
            # we found the root sink Op for this register !
            # slice backward
            s_ops = slice_backward(translation.ops, op.output)
            for s_op in reversed(s_ops):
                inps = []
                for inp in s_op.inputs:
                    if inp.space.name == 'register':
                        r_map = _map_sleigh_to_state(ctx, arch, state_format, inp.offset, inp.size)
                        inps.append(r_map[0] if r_map else _get_varnode_id(inp))
                    elif inp.space.name == 'const':
                        inps.append(hex(inp.offset))
                    else:
                        inps.append(_get_varnode_id(inp))

                out = ''
                if s_op.output:
                    if s_op.output.space.name == 'register':
                        r_map = _map_sleigh_to_state(
                            ctx,
                            arch,
                            state_format,
                            s_op.output.offset,
                            s_op.output.size,
                        )
                        out = r_map[0] if r_map else _get_varnode_id(s_op.output)
                    else:
                        out = _get_varnode_id(s_op.output)
                    out += ' = '

                op_str = f'   > {out}{s_op.opcode.name}({", ".join(inps)})'  # type: ignore[attr-defined]
                if op_str not in matched_ops:
                    matched_ops.append(op_str)


if __name__ == '__main__':
    main()
