# ruff: noqa: PLC0415
"""A basic block can be lowered as ONE program, and the branch rule that allows it.

Lowering a whole block beats chaining per-instruction programs, because
dead-code elimination then runs across the block: an `add` computes six flags
and the next instruction overwrites most of them, and only a block-wide pass can
see that.  Measured over the taint-density workloads, greedy regions of ~7
instructions cut the executed taint ops per instruction roughly in half and cut
the number of compiled calls by 7.5x.

What stood in the way was a branch rule that is right for one instruction and
wrong for a block.  SLEIGH names an intra-instruction branch target by ADDRESS,
and `_branch_target` resolves those through the IMARK table.  For a single
instruction that table holds one entry, so a jump anywhere else resolves to None
and is treated as what it is: a branch out, which writes the program counter.
Concatenate a block's p-code and the table holds every instruction, so the loop
edge of a `for` resolves to an index inside the region and is rejected as a
p-code loop.  That was 95% of the block executions of bench_untainted and
bench_dense.

Unicorn ends a basic block at any branch, so a branch that leaves its own
instruction can only be the block's last instruction, and its target is the next
block whatever the address arithmetic says about direction.  `block=True` says
so.  A branch back into the SAME instruction is still a p-code loop, which is
how `rep` lifts, and still declines.

Not yet covered here: that a block program computes the same taint as the
per-instruction programs run in sequence.  That is the real correctness
question, and it belongs with the runtime wiring that will have something to
compare against; nothing calls `block=True` yet.
"""
from __future__ import annotations

from typing import Any

import pytest
from pypcode import PcodeOp

from microtaint.taint_ir import frompcode
from microtaint.taint_ir.frompcode import Unsupported
from microtaint.taint_ir.ir import IRProg
from microtaint.types import Architecture

_ARCH, _KEY = Architecture.AMD64, 'AMD64'

#: A straight-line run ending in a BACKWARD conditional branch: the shape of
#: every loop body Unicorn hands to a block hook.
_LOOP_BODY = [
    bytes.fromhex('4883c001'),      # add rax, 1
    bytes.fromhex('4839d8'),        # cmp rax, rbx
    bytes.fromhex('7cf7'),          # jl  -9   (back to the top, exactly)
]


def _ops(seq: list[bytes]) -> tuple[list[Any], int]:
    """Concatenated p-code for `seq`, each instruction at its own lift base."""
    from microtaint.sleigh.lifter import get_context
    ops: list[PcodeOp] = []
    base = frompcode.LIFT_BASE
    for code in seq:
        ops.extend(get_context(_KEY).translate(code, base).ops)
        base += len(code)
    return ops, base


@pytest.fixture(scope='module')
def builder() -> frompcode.Builder:
    return frompcode.Builder(_ARCH, False, 'concrete')


def test_a_blocks_loop_edge_is_an_exit_not_a_loop(builder: frompcode.Builder) -> None:
    ops, end = _ops(_LOOP_BODY)
    with pytest.raises(Unsupported, match='backward CBRANCH'):
        builder.build(ops, end, emit='both')
    prog = builder.build(ops, end, emit='both', block=True)
    assert prog.outputs, 'the block lowered to a program with no outputs'


def test_the_exit_makes_the_counter_depend_on_the_condition(builder: frompcode.Builder) -> None:
    """The block's branch must still be treated as a conditional jump: the
    program counter becomes secret-dependent exactly when its condition is.
    Losing that would silently drop every implicit-flow report in a block."""
    ops, end = _ops(_LOOP_BODY)
    prog = builder.build(ops, end, emit='both', block=True)
    pc_off = builder.pc_off
    assert pc_off is not None
    written = [k for k, _ in prog.outputs if k[0] == 'reg' and k[1] == pc_off]
    assert written, (
        'the block program does not write the program counter; its exit branch '
        'was lowered as something other than a conditional jump')


def test_a_single_instruction_is_unaffected(builder: frompcode.Builder) -> None:
    """block=True must change nothing for one instruction: same outputs, same
    program.  The per-instruction path is what ships."""
    for code in (*_LOOP_BODY, bytes.fromhex('4801d8'), bytes.fromhex('488b4508')):
        ops, end = _ops([code])
        try:
            plain = builder.build(ops, end, emit='both')
        except Unsupported:
            with pytest.raises(Unsupported):
                builder.build(*_ops([code]), emit='both', block=True)
            continue
        as_block = builder.build(*_ops([code]), emit='both', block=True)
        assert [k for k, _ in plain.outputs] == [k for k, _ in as_block.outputs], code.hex()
        assert len(plain.nodes) == len(as_block.nodes), code.hex()


def test_a_self_loop_still_declines_in_block_mode(builder: frompcode.Builder) -> None:
    """A `rep` prefix lifts to a branch back into the SAME instruction.  That is
    a real p-code loop with a runtime trip count, and block mode must not
    smuggle it through as an exit.

    `repne scasb` rather than `rep stosb`: the latter declines earlier, on its
    predicated store, so it would pass this test whatever the branch rule did.
    """
    repne_scasb = bytes.fromhex('f2ae')
    ops, end = _ops([repne_scasb])
    with pytest.raises(Unsupported, match='backward CBRANCH'):
        builder.build(ops, end, emit='both')
    with pytest.raises(Unsupported, match='backward CBRANCH'):
        builder.build(*_ops([repne_scasb]), emit='both', block=True)


def test_the_block_program_is_cheaper_than_the_sum_of_its_parts(builder: frompcode.Builder) -> None:
    """The reason to do this at all.  Cross-instruction dead-code elimination
    has to actually happen, so a block must cost less than its instructions
    lowered separately."""
    from microtaint.taint_ir.exec import serialize_for_c

    names = sorted(set(builder.name_by_off.values()))
    layout = {n: i for i, n in enumerate(names)}
    kinds = ('addr', 'addrt', 'sttaint', 'mem')

    def slot_of(key: tuple[str, int]) -> int | None:
        if key[0] in ('reg', 'regv'):
            name = builder.name_by_off.get(key[1])
            if name is None or name not in layout:
                raise KeyError(key)
            return int(layout[name]) + (len(layout) if key[0] == 'regv' else 0)
        if key[0] in kinds:
            return 2 * len(layout) + 4 * key[1] + kinds.index(key[0])
        raise KeyError(key)

    def ops_of(prog: IRProg) -> int:
        return len(serialize_for_c(prog, slot_of)['op_ids'])

    separate = 0
    for code in _LOOP_BODY:
        try:
            separate += ops_of(builder.build(*_ops([code]), emit='both'))
        except Unsupported:
            separate += 0          # the branch alone does not lower; the block does
    together = ops_of(builder.build(*_ops(_LOOP_BODY), emit='both', block=True))
    assert together < separate, (
        f'the block program costs {together} ops against {separate} for its '
        f'instructions lowered separately; cross-instruction dead-code '
        f'elimination is not happening')
