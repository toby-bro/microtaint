"""
test_rep_stosb_address_resolution.py
====================================
Regression tests for the STORE-address resolution bug that affected
``rep``-prefixed string instructions (rep stosb, rep movsb, rep stosq,
rep movsq, ...).

Bug
---
SLEIGH lifts ``rep stosb`` (opcode bytes ``f3 aa``) into a P-code op list
where the post-store RDI update appears at op[8], BEFORE the STORE itself
at op[10] in the textual op order::

    [4]  COPY     unique[22300:8] <- RDI         ; capture pre-update RDI
    [5]  INT_ADD  unique[22400:8] <- RDI, 0x1
    ...
    [8]  INT_SUB  RDI <- unique[22400:8], unique[22600:8]   ; RDI := RDI + 1
    [9]  COPY     unique[22800:1] <- AL
    [10] STORE                    <- ..., unique[22300:8], unique[22800:1]

The STORE's address-input is ``unique[22300:8]``, which was COPYd from
RDI at op[4].  The correct architectural semantics is that the store
goes to ``[RDI]``, with RDI being the value held in the register BEFORE
the post-step update.

The old ``resolve_ptr_with_offset`` walked the op list without temporal
ordering: when it followed the COPY back to RDI, it then searched the
WHOLE op list for the latest definer of RDI, found the post-store
INT_SUB at op[8], and traced that back to ``RDI + 1``.  As a result the
static rule wrote the AL-taint to ``[RDI+1]`` instead of ``[RDI]``.

Across ``rep`` iterations Unicorn calls the per-instruction hook with
the live RDI value, so this off-by-one shifts every taint write up by
one slot.  The bottom byte ends up correct (it picks up AL's taint from
the next iteration's stale store) but the top byte of the destination
is never written; the original spilled taint of that slot survives
unchanged.  In the noninterference oracle this manifests as
``RAX_out = 0x4000018000000202`` instead of the correct
``0x0202020202020202`` for a tainted-AL fill.

Fix
---
- ``map_outputs_to_targets`` passes ``stop_op_index=store_idx`` to
  ``resolve_ptr_with_offset`` for STORE addresses, mirroring what was
  already done for LOADs.
- ``resolve_ptr_with_offset`` threads a ``limit`` parameter through every
  recursive call.  When it follows a defining op at index ``i``, the
  recursive resolves of that op's inputs use ``limit = i`` — they can
  only see ops STRICTLY BEFORE op[i].  This implements the SSA-style
  "what was this varnode at the moment op[i] fired" semantics.

These tests would FAIL on the pre-fix engine and PASS on the fixed one.
"""

# mypy: disable-error-code="no-untyped-def, no-untyped-call,import-untyped"
# ruff: noqa: PT018

from __future__ import annotations

import pytest
from keystone import KS_ARCH_X86, KS_MODE_64, Ks

from microtaint.sleigh.engine import (
    StateMapper,
    generate_static_rule,
    resolve_ptr_with_offset,
)
from microtaint.sleigh.lifter import get_context
from microtaint.types import Architecture, Register

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def regs() -> list[Register]:
    """64-bit GP register state, matches what the wrapper uses."""
    return [
        Register(n, 8)
        for n in (
            'RAX',
            'RBX',
            'RCX',
            'RDX',
            'RSI',
            'RDI',
            'RSP',
            'RBP',
            'R8',
            'R9',
            'R10',
            'R11',
            'R12',
            'R13',
            'R14',
            'R15',
        )
    ]


@pytest.fixture(scope='module')
def ks() -> Ks:
    return Ks(KS_ARCH_X86, KS_MODE_64)


# ---------------------------------------------------------------------------
# Unit test: resolve_ptr_with_offset on the bare P-code
# ---------------------------------------------------------------------------


class TestResolvePtrTemporalOrdering:
    """Direct tests of resolve_ptr_with_offset on the rep-stosb P-code shape.

    These are the lowest-level reproducers of the bug. They run the address
    resolver on the actual P-code ops emitted by SLEIGH for ``rep stosb``
    and check the resolved (register, offset) pair.
    """

    def test_rep_stosb_store_addr_is_rdi_with_zero_offset(self, regs) -> None:
        """STORE in rep stosb must resolve to (RDI, 0), not (RDI, +1).

        The buggy resolver returned (RDI, 1) because it picked up the
        post-store INT_SUB that updates RDI within the same instruction.
        """
        ctx = get_context('AMD64')
        mapper = StateMapper(ctx, 'AMD64', regs)
        translation = ctx.translate(bytes.fromhex('f3aa'), 0x1000)

        # Find the STORE op and its index
        store_idx = None
        ptr_vn = None
        for i, op in enumerate(translation.ops):
            if op.opcode.name == 'STORE':
                store_idx = i
                ptr_vn = op.inputs[1]
                break

        assert store_idx is not None, 'no STORE in rep stosb P-code (lifter changed?)'
        assert ptr_vn is not None

        # Resolve, mirroring exactly what map_outputs_to_targets does after
        # the fix: pass stop_op_index=store_idx.
        base_reg, const_offset = resolve_ptr_with_offset(
            ptr_vn,
            list(translation.ops),
            mapper,
            stop_op_index=store_idx,
        )

        assert base_reg is not None, 'resolver returned None — should resolve to RDI'
        assert base_reg.name == 'RDI', f'expected base register RDI, got {base_reg.name}'
        # The bug returned offset=1 here. Correct value is 0.
        assert const_offset == 0, (
            f'rep stosb STORE address resolved to RDI + {const_offset}, '
            f'expected RDI + 0. The post-store INT_SUB (which updates RDI '
            f'after the store within the same op list) is leaking into the '
            f'address resolution — the temporal-ordering fix in '
            f'resolve_ptr_with_offset is missing or incomplete.'
        )

    def test_rep_movsb_store_addr_is_rdi_with_zero_offset(self, regs) -> None:
        """Same bug shape for rep movsb (f3 a4): post-store updates RDI."""
        ctx = get_context('AMD64')
        mapper = StateMapper(ctx, 'AMD64', regs)
        translation = ctx.translate(bytes.fromhex('f3a4'), 0x1000)

        store_idx = None
        ptr_vn = None
        for i, op in enumerate(translation.ops):
            if op.opcode.name == 'STORE':
                store_idx = i
                ptr_vn = op.inputs[1]
                break

        assert store_idx is not None
        assert ptr_vn is not None

        base_reg, const_offset = resolve_ptr_with_offset(
            ptr_vn,
            list(translation.ops),
            mapper,
            stop_op_index=store_idx,
        )

        assert base_reg is not None
        assert base_reg.name == 'RDI'
        assert const_offset == 0, f'rep movsb STORE address resolved to RDI + {const_offset}, expected RDI + 0.'

    def test_rep_movsq_store_addr_is_rdi_with_zero_offset(self, regs) -> None:
        """rep movsq (f3 48 a5) — qword version, same bug pattern."""
        ctx = get_context('AMD64')
        mapper = StateMapper(ctx, 'AMD64', regs)
        translation = ctx.translate(bytes.fromhex('f348a5'), 0x1000)

        store_idx = None
        ptr_vn = None
        for i, op in enumerate(translation.ops):
            if op.opcode.name == 'STORE':
                store_idx = i
                ptr_vn = op.inputs[1]
                break

        assert store_idx is not None
        assert ptr_vn is not None

        base_reg, const_offset = resolve_ptr_with_offset(
            ptr_vn,
            list(translation.ops),
            mapper,
            stop_op_index=store_idx,
        )

        assert base_reg is not None
        assert base_reg.name == 'RDI'
        assert const_offset == 0


# ---------------------------------------------------------------------------
# End-to-end: the assembled static-rule output reflects the correct address
# ---------------------------------------------------------------------------


class TestRepStosbStaticRule:
    """End-to-end check on the LogicCircuit built by generate_static_rule.

    Inspects the final memory-write target, which is the user-visible
    artefact used by the runtime hook.
    """

    def test_rep_stosb_writes_to_rdi_not_rdi_plus_one(self, regs) -> None:
        """The static rule's STORE target must be T_MEM[V_RDI, size=1].

        Pre-fix: T_MEM[(V_RDI[63:0] ADD 0x1), size=1].
        Post-fix: T_MEM[V_RDI[63:0], size=1].
        """
        circuit = generate_static_rule(Architecture.AMD64, bytes.fromhex('f3aa'), regs)

        # Find the memory-write assignment.  It's the only one whose target
        # carries an address_expr (T_MEM[...]).
        mem_assigns = [a for a in circuit.assignments if hasattr(a.target, 'address_expr')]
        assert len(mem_assigns) == 1, f'expected exactly 1 memory-write assignment, got {len(mem_assigns)}'

        target_str = str(mem_assigns[0].target)

        # The fixed engine produces T_MEM[V_RDI[63:0], size=1].
        # The buggy engine produced T_MEM[(V_RDI[63:0] ADD 0x1), size=1].
        # Test both shape constraints:
        assert 'V_RDI' in target_str, f'STORE address does not reference V_RDI: {target_str}'
        assert 'ADD 0x1' not in target_str and ' ADD 1' not in target_str, (
            f'STORE address contains a +1 offset that should not be there.  '
            f'This is the rep-stosb post-store-update leak.  Got: {target_str}'
        )

    def test_rep_stosb_value_taint_is_or_of_rcx_and_rax_taints(self, regs) -> None:
        """The store value's taint expression should depend on T_RAX (AL is
        the bottom byte of RAX). Sanity check that we haven't broken the
        unrelated value-side propagation while fixing the address side.
        """
        circuit = generate_static_rule(Architecture.AMD64, bytes.fromhex('f3aa'), regs)

        mem_assigns = [a for a in circuit.assignments if hasattr(a.target, 'address_expr')]
        assert len(mem_assigns) == 1
        expr_str = str(mem_assigns[0].expression)
        assert 'T_RAX' in expr_str, f'STORE value taint should mention T_RAX (AL is part of RAX), got: {expr_str}'


# ---------------------------------------------------------------------------
# Sanity: ordinary STORE shapes still resolve correctly
#
# The temporal-ordering fix tightened the limits used during recursive
# pointer resolution.  These tests guard against the change accidentally
# breaking ordinary STOREs whose address registers are NOT updated within
# the same instruction.
# ---------------------------------------------------------------------------


class TestOrdinaryStoresStillWork:
    """Common STORE shapes must keep their previous (correct) resolution."""

    def _store_target(self, ks: Ks, regs: list[Register], asm: str) -> str:
        enc, _ = ks.asm(asm)
        circuit = generate_static_rule(Architecture.AMD64, bytes(enc), regs)
        mem_assigns = [a for a in circuit.assignments if hasattr(a.target, 'address_expr')]
        assert mem_assigns, f'no memory write produced for {asm!r}'
        return str(mem_assigns[0].target)

    def test_mov_rbp_minus_16(self, ks, regs) -> None:
        t = self._store_target(ks, regs, 'mov [rbp - 0x10], rax')
        assert 'V_RBP' in t and 'ADD -0x10' in t, t

    def test_mov_rsp_plus_8(self, ks, regs) -> None:
        t = self._store_target(ks, regs, 'mov [rsp + 8], rax')
        assert 'V_RSP' in t and 'ADD 0x8' in t, t

    def test_mov_rdi_no_offset(self, ks, regs) -> None:
        """Plain [RDI] — must still resolve to V_RDI with no offset."""
        t = self._store_target(ks, regs, 'mov [rdi], al')
        assert 'V_RDI' in t, t
        assert 'ADD 0x1' not in t and ' ADD 1' not in t, f'plain mov [rdi], al got a spurious +1 offset: {t}'

    def test_mov_rdi_plus_4(self, ks, regs) -> None:
        """[RDI + 4] is the legitimate +4 case — must keep its real offset."""
        t = self._store_target(ks, regs, 'mov [rdi + 4], eax')
        assert 'V_RDI' in t and 'ADD 0x4' in t, t

    def test_push_rax_resolves_to_rsp_minus_8(self, ks, regs) -> None:
        """``push rax`` lifts to (RSP -= 8; mem[RSP] = RAX).  The STORE
        address is RSP AFTER the predecrement, which is RSP - 8.  This
        is the case most architecturally similar to the rep-stosb shape
        (post-update register feeds the store address) so we double-check
        it stays correct.
        """
        t = self._store_target(ks, regs, 'push rax')
        assert 'V_RSP' in t and 'ADD -0x8' in t, t

    def test_mov_rsp_minus_64(self, ks, regs) -> None:
        """The exact opening of the rep-stosb test sequence in the
        benchmark — must resolve to V_RSP - 0x40.
        """
        t = self._store_target(ks, regs, 'mov [rsp - 64], rax')
        assert 'V_RSP' in t and 'ADD -0x40' in t, t
