"""A compiled taint program belongs to the slot layout it was compiled for.

The program's slot numbers are baked into its machine code.  The cache was
keyed on the instruction alone, so two callers with different layouts shared
one program: it then reads the wrong words and reports the answer against the
wrong register.  It never showed in production because the emulator hook was
the sole caller and every hook interns its registers in the same order -- which
is exactly the kind of accident that stops being true the moment a second
caller exists.
"""
from __future__ import annotations

import pytest

from microtaint.instrumentation.cell_c import taint_ir_c
from microtaint.taint_ir.engine_glue import Compiled, enabled, program_for
from microtaint.types import Architecture

#: `add rax, rbx` -- two register inputs, a result and five flags.
ADD_RAX_RBX = bytes.fromhex('4801d8')
_FLAGS = ('CF', 'OF', 'SF', 'ZF', 'PF', 'AF')


def _layout(first: str, second: str) -> dict[str, int]:
    names = [first, second, 'RIP', *_FLAGS]
    return {n: i for i, n in enumerate(names)}


def _taint_after(prog: Compiled, layout: dict[str, int],
                 tainted: str) -> dict[str, int]:
    """Run the program with exactly one register tainted -> {name: mask}."""
    n = max(layout.values()) + 1
    values = [0] * n
    taints = [0] * n
    taints[layout[tainted]] = 0xFF
    out = taint_ir_c.run(prog[0], values, taints)
    return {name: out[slot] for name, slot in layout.items() if out[slot]}


@pytest.mark.skipif(not enabled(), reason='the compiled taint path is turned off')
def test_two_layouts_do_not_share_one_program() -> None:
    a = _layout('RAX', 'RBX')
    b = _layout('RBX', 'RAX')          # the same registers, opposite slots
    prog_a = program_for(Architecture.AMD64, ADD_RAX_RBX, a)
    prog_b = program_for(Architecture.AMD64, ADD_RAX_RBX, b)
    if prog_a is None or prog_b is None:
        pytest.skip('the emitter declined this instruction on this host')

    assert prog_a[0] is not prog_b[0], (
        'both layouts were handed the SAME compiled program; its slot numbers '
        'are compiled in, so one of them is reading the wrong words')

    # Taint RAX under each layout.  RAX is slot 0 in one and slot 1 in the
    # other, so a program shared between them attributes the result to RBX.
    got_a = _taint_after(prog_a, a, 'RAX')
    got_b = _taint_after(prog_b, b, 'RAX')
    assert got_a == got_b, (
        f'the same question answered differently under two layouts: '
        f'{got_a} vs {got_b}')
    assert got_a.get('RAX') == 0xFF, f'RAX should carry the taint, got {got_a}'
    assert 'RBX' not in got_a, (
        f'RBX was never tainted; the answer landed on the wrong register: {got_a}')
