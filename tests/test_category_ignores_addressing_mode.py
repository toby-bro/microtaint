"""A memory operand's ADDRESSING MODE must not change the data classification.

`is_mapped_permutation` counted a slice's dynamic sources by looking at each
op's inputs, skipping `const` and `unique` varnodes.  A LOAD's pointer is a bare
register only when the address needs no arithmetic:

    and ecx,[rdi]          LOAD reads ('register', RDI)  -> RDI counted
    and ecx,[rdi+rax*4]    LOAD reads ('unique', ...)    -> nothing counted

so the first saw two dynamic sources (RDI and RCX) and correctly refused to call
itself a permutation, while the second saw one (RCX) and reported a permutation
OF RCX.  Same instruction, same operand, opposite classification, and MAPPED is a
claim that the output is a fixed bit-permutation of a single source.

The fix is to count what a LOAD actually contributes: its inputs are an ADDRESS,
not data, and the VALUE it brings in is one dynamic source however the address is
written.  These tests pin the invariant rather than the mechanism, so any future
rewrite of the recogniser is still held to it.
"""
from __future__ import annotations

import pytest
from pypcode import PcodeOp

from microtaint.sleigh.lifter import get_context
from microtaint.sleigh.mapper import determine_category, is_mapped_permutation

# From the module that DEFINES it: engine.py merely imports it, and a
# re-export is not part of that module's interface.
from microtaint.sleigh.slicer import slice_backward

keystone = pytest.importorskip('keystone')

#: The same computation, reached through four addressing modes.
_MODES = ['[rdi]', '[rdi+0x10]', '[rdi+rax*1]', '[rdi+rax*4]', '[rdi+rax*4+0x8]']


def _subslice(asm: str) -> list[PcodeOp]:
    """The DESTINATION register's data slice, with the load pointer not followed.

    engine.py classifies the slice of the target it is computing taint for, so
    this picks the widest register write (the architectural result) rather than
    the last op in the list, which is a flag and would pin a different slice than
    the one the engine acts on.
    """
    ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
    sctx = get_context('AMD64')
    ops = list(sctx.translate(bytes(ks.asm(asm)[0]), 0x1000).ops)
    result = [o for o in ops
              if o.output is not None
              and o.output.space.name == 'register'
              and o.output.size >= 4]
    assert result, f'{asm!r} writes no register result to classify'
    out = result[-1].output
    assert out is not None, 'the comprehension above filtered for a non-None output'
    return slice_backward(ops, out, follow_load_ptr=False)


@pytest.mark.parametrize('op', ['and', 'or', 'xor', 'add', 'sub'])
def test_two_source_op_is_never_a_permutation(op: str) -> None:
    """`<op> ecx,[X]` reads TWO sources (RCX and memory) in every mode."""
    verdicts = {m: is_mapped_permutation(_subslice(f'{op} ecx, {m}'), frozenset())
                for m in _MODES}
    assert not any(verdicts.values()), (
        f'{op} ecx,[X] was called a bit-permutation of one source in some '
        f'addressing mode but not others: {verdicts}'
    )


@pytest.mark.parametrize('op', ['and', 'or', 'xor', 'add', 'sub', 'mov'])
def test_category_is_the_same_in_every_addressing_mode(op: str) -> None:
    """The data category may not depend on how the address is written."""
    cats = {m: determine_category(_subslice(f'{op} ecx, {m}')) for m in _MODES}
    assert len(set(cats.values())) == 1, (
        f'{op} ecx,[X] classified differently per addressing mode: '
        f'{ {m: c.name for m, c in cats.items()} }'
    )


def test_a_plain_load_is_still_mapped() -> None:
    """The guard: a LOAD whose value IS the output stays a single-source mapping.

    Counting the loaded value must not cost `mov ecx,[X]` its MAPPED
    classification -- that is the exact, cheap path for every plain load.
    """
    for mode in _MODES:
        sub = _subslice(f'mov ecx, {mode}')
        assert is_mapped_permutation(sub, frozenset()), (
            f'mov ecx,{mode} stopped being a mapped permutation; a plain load '
            f'has exactly one dynamic source, the value it loads'
        )
