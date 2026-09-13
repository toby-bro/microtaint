"""Everything downstream of the resolver must carry the index too.

Recovering the index register in `resolve_ptr_with_offset` is useless if the
things that consume the resolved address drop it again, and three of them did.
Each failure is silent, and none of them changes the shape of the answer, only
which bytes it was computed from:

* the cell inputs carried the address BASE but not the INDEX, so the replica's
  index register sat at zero and the instruction read an address the masked
  value never reached, making the operand image as "does not matter";
* the polarity key was `(base, displacement)`, so `[rdi]` and `[rdi+rax*4]`
  shared an identity and one location's polarity crossed to another's.

These pin each consumer directly, because an end-to-end taint value can be right
for the wrong reason: a floor elsewhere covers a dropped term often enough that
the whole-answer tests stayed green while all three were broken.
"""
from __future__ import annotations

import pytest

from microtaint.instrumentation.ast import InstructionCellExpr
from microtaint.sleigh.engine import MemMapping, RegMapping, generate_static_rule
from microtaint.types import Architecture, Register

keystone = pytest.importorskip('keystone')

NAMES = ['RAX', 'RBX', 'RCX', 'RDX', 'RDI', 'CF', 'PF', 'ZF', 'SF', 'OF']
FMT = ([Register(n, 64) for n in NAMES[:5]]
       + [Register(f, 1) for f in NAMES[5:]] + [Register('RSP', 64)])

_RDI = RegMapping('RDI', 0, 63)
_RAX = RegMapping('RAX', 0, 63)
_RBX = RegMapping('RBX', 0, 63)


# ---------------------------------------------------------------------------
# The cell inputs.
# ---------------------------------------------------------------------------

def _cell_input_names(asm: str) -> set[str]:
    """Every register name handed to any InstructionCellExpr in the rule."""
    ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
    rule = generate_static_rule(Architecture.AMD64, bytes(ks.asm(asm)[0]), FMT)
    names: set[str] = set()
    seen: set[int] = set()

    def walk(node: object) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        if isinstance(node, InstructionCellExpr):
            names.update(node.inputs)
            for sub in node.inputs.values():
                walk(sub)
            return
        for attr in ('lhs', 'rhs', 'expr', 'address_expr'):
            child = getattr(node, attr, None)
            if child is not None:
                walk(child)

    for a in rule.assignments:
        walk(a.expression)
        walk(a.target)
    return names


def test_the_index_register_reaches_the_cell_inputs() -> None:
    """An indexed operand must supply BOTH address registers to the replica.

    With RDI alone, the replica computes [RDI + 0*scale] and reads bytes the
    masked value never touched.
    """
    names = _cell_input_names('and ecx, dword ptr [rdi + rax*4]')
    assert 'RDI' in names, f'the address base never reached the cell: {sorted(names)}'
    assert 'RAX' in names, (
        f'the address INDEX never reached the cell, so the replica resolves the '
        f'operand to [RDI+0] and reads bytes the mask never reached: '
        f'{sorted(names)}'
    )


def test_a_plain_operand_still_supplies_its_base() -> None:
    """The control: nothing about the single-register case changed."""
    assert 'RDI' in _cell_input_names('and ecx, dword ptr [rdi]')


# ---------------------------------------------------------------------------
# The address identity used to match a LOAD with the STORE at the same place.
# ---------------------------------------------------------------------------

def test_addresses_differing_only_by_index_are_different_identities() -> None:
    """`[rdi]`, `[rdi+rax*4]` and `[rdi+rbx*4]` are three different places."""
    from microtaint.sleigh.engine import _mem_addr_identity

    plain = _mem_addr_identity(MemMapping(0, 4, _RDI, 0))
    scaled = _mem_addr_identity(MemMapping(0, 4, _RDI, 0, _RAX, 4))
    other_reg = _mem_addr_identity(MemMapping(0, 4, _RDI, 0, _RBX, 4))
    other_scale = _mem_addr_identity(MemMapping(0, 4, _RDI, 0, _RAX, 8))
    assert len({plain, scaled, other_reg, other_scale}) == 4, (
        f'two different memory locations share a polarity identity, so one '
        f"location's polarity can be applied to another: "
        f'{plain}, {scaled}, {other_reg}, {other_scale}'
    )


def test_the_two_identity_constructors_agree() -> None:
    """Built from a resolver triple or from a MemMapping: the same identity.

    The producer keys on the triple and the consumer on the mapping; if they
    disagree the lookup silently misses and the polarity is never applied.
    """
    from microtaint.sleigh.engine import _addr_identity, _mem_addr_identity

    assert (_addr_identity(_RDI, -16, (_RAX, 4))
            == _mem_addr_identity(MemMapping(0, 4, _RDI, -16, _RAX, 4)))
    assert (_addr_identity(_RDI, -16, None)
            == _mem_addr_identity(MemMapping(0, 4, _RDI, -16)))


def test_the_displacement_still_separates_identities() -> None:
    """The control: the property the identity already had."""
    from microtaint.sleigh.engine import _mem_addr_identity

    assert (_mem_addr_identity(MemMapping(0, 4, _RDI, 0, _RAX, 4))
            != _mem_addr_identity(MemMapping(0, 4, _RDI, 8, _RAX, 4)))
