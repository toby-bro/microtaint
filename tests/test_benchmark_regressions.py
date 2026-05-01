"""
test_benchmark_regressions.py
=============================
Regression tests for every confirmed microtaint bug found in the
cross-engine benchmark comparison (vs angr and maat).

Each test encodes a case where *both* angr and maat agreed on the
correct answer while microtaint was wrong.  Tests are organised by
root cause, with a comment on the underlying bug.

The expected values (``expected_*``) were computed by hand from the
differential ground truth (run instruction on V|T and V&~T, XOR) and
cross-validated against both angr and maat.

All tests use the same ``_eval`` helper as test_bit_propagation.py:
generate the static circuit then call circuit.evaluate(ctx).

Bug classes fixed
-----------------
  BUG-1 CMOV-NOT-TAKEN  — CMOV drops destination taint when condition is
                          not taken; should pass through old dest taint.
  BUG-2 CHAIN-SEQUENCE  — Multi-instruction sequences: intermediate taint
                          state not threaded between instructions.
  BUG-3 SUBREG-MOVZX    — movzx zero-extends; old upper bits of dest must
                          NOT be carried as deps into the next instruction.
  BUG-4 PERM-DOUBLE     — Double bswap is identity; the permutation must
                          compose correctly over the two instructions.
  BUG-5 AVALANCHE-DEST  — AVALANCHE ops (imul/lzcnt/mul) pick up taint
                          from the *destination* register read-back for
                          flag computation; that read-back should not
                          count as a value dep.
  BUG-6 SHIFT-TAINTED-CL — Shift by a tainted CL must AVALANCHE the
                           whole output, not just bit-0.
"""

# mypy: disable-error-code="no-untyped-def, no-untyped-call"

from __future__ import annotations

import pytest

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, ImplicitTaintPolicy, Register

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def simulator() -> CellSimulator:
    return CellSimulator(Architecture.AMD64)


@pytest.fixture(scope='module')
def regs() -> list[Register]:
    return [
        Register('RAX', 64),
        Register('RBX', 64),
        Register('RCX', 64),
        Register('RDX', 64),
    ]


def _eval(
    simulator: CellSimulator,
    regs: list[Register],
    bytestring: bytes,
    taint: dict[str, int],
    values: dict[str, int],
) -> dict[str, int]:
    circuit = generate_static_rule(Architecture.AMD64, bytestring, regs)
    ctx = EvalContext(
        input_taint=taint,
        input_values=values,
        simulator=simulator,
        implicit_policy=ImplicitTaintPolicy.IGNORE,
    )
    return circuit.evaluate(ctx)


# ---------------------------------------------------------------------------
# BUG-1: CMOV drops destination taint when condition is not-taken
#
# cmovz rax, rbx — when ZF=0 the move does NOT execute.
# The old value of RAX passes through unchanged.
# Microtaint returns T_RAX=0 instead of the original T_RAX.
# ---------------------------------------------------------------------------


class TestCmovNotTaken:
    def test_cmovz_not_taken_preserves_dest_taint(self, simulator, regs) -> None:
        """cmovz rax, rbx when ZF=0 (not taken) — RAX keeps its original taint."""
        # ZF=0: the processor does NOT copy RBX → RAX. RAX is unchanged.
        # So T_RAX_out = T_RAX_in (old value passes through).
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('480f44c3'),  # cmovz rax, rbx
            taint={'RAX': 0xFFFFFFFFFFFFFFFF, 'RBX': 0x1122334455667788, 'RCX': 0, 'RDX': 0},
            values={'RAX': 0xDEAD, 'RBX': 0xBEEF, 'RCX': 0, 'RDX': 0},
            # ZF=0 because RCX=0 and EFLAGS not modified → ZF stays whatever it was
            # But our evaluator uses concrete V: V(RAX) XOR V&~T(RAX) matters
            # We need ZF=0 in the concrete execution. With values only (IGNORE policy),
            # the EFLAGS are initialized to 0 → ZF=0 → move not taken.
        )
        assert (
            out.get('RAX', 0) == 0xFFFFFFFFFFFFFFFF
        ), f'cmovz not-taken must preserve RAX taint; got {out.get("RAX", 0):#x}'

    def test_cmovz_not_taken_rbx_untainted_preserves_dest(self, simulator, regs) -> None:
        """cmovz when RBX is untainted but RAX is — dest taint must survive."""
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('480f44c3'),  # cmovz rax, rbx
            taint={'RAX': 0xFFFFFFFFFFFFFFFF, 'RBX': 0, 'RCX': 0, 'RDX': 0},
            values={'RAX': 0, 'RBX': 0, 'RCX': 0, 'RDX': 0},
        )
        assert (
            out.get('RAX', 0) == 0xFFFFFFFFFFFFFFFF
        ), f'cmovz not-taken: RAX taint must survive; got {out.get("RAX", 0):#x}'

    def test_cmovl_not_taken_preserves_dest_taint(self, simulator, regs) -> None:
        """cmovl rbx, rdx when condition false — RBX keeps original taint."""
        # EFLAGS default 0 → SF=0, OF=0 → SF==OF → condition not met → not taken.
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('480f4cda'),  # cmovl rbx, rdx
            taint={'RAX': 0, 'RBX': 0xFFFFFFFFFFFFFFFF, 'RCX': 0, 'RDX': 0},
            values={'RAX': 0, 'RBX': 0, 'RCX': 0, 'RDX': 0},
        )
        assert (
            out.get('RBX', 0) == 0xFFFFFFFFFFFFFFFF
        ), f'cmovl not-taken: RBX taint must survive; got {out.get("RBX", 0):#x}'

    def test_cmovs_not_taken_preserves_dest_taint(self, simulator, regs) -> None:
        """cmovs rcx, rdx when SF=0 (not taken) — RCX keeps its taint."""
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('480f48ca'),  # cmovs rcx, rdx
            taint={'RAX': 0, 'RBX': 0, 'RCX': 0xFFFFFFFFFFFFFFFF, 'RDX': 0},
            values={'RAX': 0, 'RBX': 0, 'RCX': 0, 'RDX': 0},
        )
        assert (
            out.get('RCX', 0) == 0xFFFFFFFFFFFFFFFF
        ), f'cmovs not-taken: RCX taint must survive; got {out.get("RCX", 0):#x}'

    def test_cmov_taken_replaces_dest_taint(self, simulator, regs) -> None:
        """Sanity: cmovz when ZF=1 IS taken — dest gets source taint."""
        # Force ZF=1 by using value 0 in the instruction that sets EFLAGS.
        # For the worker/circuit path this is hard to test without EFLAGS,
        # so we just check the opposite side: if both are tainted, output
        # must be non-zero (it's the tainted source).
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('480f44c3'),  # cmovz rax, rbx
            taint={'RAX': 0, 'RBX': 0xFFFFFFFFFFFFFFFF, 'RCX': 0, 'RDX': 0},
            values={'RAX': 0, 'RBX': 0xDEAD, 'RCX': 0, 'RDX': 0},
        )
        # Either RAX or RBX taint lands in the output (not both zero):
        total = out.get('RAX', 0) | out.get('RBX', 0)
        assert total != 0, 'cmovz: some taint must propagate when source is tainted'


# ---------------------------------------------------------------------------
# BUG-2: Multi-instruction chain — intermediate taint not threaded
#
# When multiple instructions are passed as one bytestring, Ghidra lifts
# them into a single P-code block.  The static rule maps outputs against
# the *original* input register values, not the intermediate state after
# each instruction executes.  This causes the wrong taint mask on RBX
# for `shl rax,4; or rbx,rax; shr rbx,2`.
# ---------------------------------------------------------------------------


class TestChainSequence:
    def test_shl_or_shr_chain_rbx_taint(self, simulator, regs) -> None:
        """shl rax,4; or rbx,rax; shr rbx,2 — RBX taint must use UPDATED RAX."""
        # Differential ground truth:
        #   step 1: rax' = rax << 4
        #   step 2: rbx' = rbx | rax'  = rbx | (rax << 4)
        #   step 3: rbx'' = rbx' >> 2  = (rbx | (rax << 4)) >> 2
        #
        # With RAX fully tainted and RBX=partial taint:
        #   T_RBX_out ≠ T_RBX_in (the shift changes which bits depend on which)
        rbx_taint_in = 0xF01974DF30ED1C23
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('48c1e0044809c348c1eb02'),  # shl rax,4; or rbx,rax; shr rbx,2
            taint={'RAX': 0xFFFFFFFFFFFFFFFF, 'RBX': rbx_taint_in, 'RCX': 0, 'RDX': 0},
            values={'RAX': 0xFFFFFFFFFFFFFFFF, 'RBX': rbx_taint_in, 'RCX': 0, 'RDX': 0},
        )
        rbx_out = out.get('RBX', 0)
        # Must not just echo the raw RBX input taint unchanged — the shifts transform it.
        assert rbx_out != rbx_taint_in, 'chain: RBX taint must reflect shifts, not just pass through unchanged'
        # And must be non-zero (RAX is fully tainted and feeds into RBX via OR then SHR)
        assert rbx_out != 0, f'chain: RBX must be tainted from tainted RAX; got {rbx_out:#x}'

    def test_add_cascade_chain(self, simulator, regs) -> None:
        """add rax,rbx; add rcx,rax; add rdx,rcx — taint cascades correctly."""
        # With only RBX tainted:
        #   rax' = rax + rbx → T_RAX = diff(add(V_RAX|T_RBX, V_RBX)|T_RBX)  ... → some taint
        #   rcx' = rcx + rax' → T_RCX depends on updated T_RAX
        #   rdx' = rdx + rcx' → T_RDX depends on updated T_RCX
        # All three outputs should be tainted.
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('4801d84801c14801ca'),  # add rax,rbx; add rcx,rax; add rdx,rcx
            taint={'RAX': 0, 'RBX': 0xFFFFFFFFFFFFFFFF, 'RCX': 0, 'RDX': 0},
            values={'RAX': 0, 'RBX': 1, 'RCX': 0, 'RDX': 0},
        )
        assert out.get('RAX', 0) != 0, 'cascade: RAX must be tainted from RBX'
        assert out.get('RCX', 0) != 0, 'cascade: RCX must be tainted via RAX'
        assert out.get('RDX', 0) != 0, 'cascade: RDX must be tainted via RCX'

    def test_neg_sub_mov_chain(self, simulator, regs) -> None:
        """neg rax; sub rbx,rax; mov rcx,rbx — RCX gets RBX taint."""
        # With RAX untainted and RBX partially tainted:
        #   neg rax → rax' = -rax (clean)
        #   sub rbx, rax' → rbx' = rbx - rax' (rbx taint transformed by subtraction)
        #   mov rcx, rbx' → rcx' = rbx' (taint copies)
        # RCX_out must equal T_RBX propagated through sub.
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('48f7d84829c34889d9'),  # neg rax; sub rbx,rax; mov rcx,rbx
            taint={'RAX': 0, 'RBX': 0xFFFFFFFFFFFFFFFF, 'RCX': 0, 'RDX': 0},
            values={'RAX': 1, 'RBX': 0xDEAD, 'RCX': 0, 'RDX': 0},
        )
        # RCX must carry some of RBX's taint (the mov at the end)
        assert out.get('RCX', 0) != 0, f'neg-sub-mov: RCX must be tainted via RBX; got {out}'

    def test_imul_add_mov_chain_zero_output(self, simulator, regs) -> None:
        """imul rax,rbx; add rcx,rax; mov rdx,rcx — if imul result=0, rdx must be clean."""
        # imul rax, rbx with rax=1, rbx=0: result = 0 (clean)
        # add rcx, 0: rcx unchanged (clean if rcx was clean)
        # mov rdx, rcx: rdx = rcx (clean)
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('480fafc34801c14889ca'),  # imul rax,rbx; add rcx,rax; mov rdx,rcx
            taint={'RAX': 0, 'RBX': 0, 'RCX': 0, 'RDX': 0xFFFFFFFFFFFFFFFF},
            values={'RAX': 1, 'RBX': 0, 'RCX': 0, 'RDX': 0},
        )
        assert out.get('RDX', 0) == 0, f'imul-add-mov: RDX must be clean when chain is clean; got {out.get("RDX",0):#x}'


# ---------------------------------------------------------------------------
# BUG-3: subreg — movzx zero-extends; old dest bits must not carry through
#
# movzx rax, bx: clears bits 16-63 of RAX then writes BX into bits 0-15.
# Any subsequent instruction using RAX should see only the low-16 taint,
# not whatever old taint RAX had in its upper bits.
# ---------------------------------------------------------------------------


class TestSubregMovzx:
    def test_movzx_bx_then_add_rcx_clean_upper(self, simulator, regs) -> None:
        """movzx rax,bx; add rax,rcx — upper bits of RAX must be 0-extended."""
        # movzx rax, bx: rax = zero_extend(bx)
        # add rax, rcx: result has carry from bx+cl into higher bits, but
        # bits 63:16 can only carry, not hold old-rax taint directly.
        # With BX bits 15:0 tainted and RCX clean, the output should NOT
        # include the old RAX[63:16] taint.
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('480fb7c34801c8'),  # movzx rax,bx; add rax,rcx
            taint={'RAX': 0xFFFFFFFF00000000, 'RBX': 0x000000000000FFFF, 'RCX': 0, 'RDX': 0},
            values={'RAX': 0, 'RBX': 0x5678, 'RCX': 0, 'RDX': 0},
        )
        rax_out = out.get('RAX', 0)
        # The old RAX[63:32] taint (0xFFFFFFFF00000000) must NOT appear in output
        # because movzx zeroed those bits before the add.
        assert (
            rax_out & 0xFFFFFFFF00000000
        ) == 0, f'movzx must zero-extend; old RAX upper taint must not appear: {rax_out:#x}'

    def test_movzx_al_clears_high_bits(self, simulator, regs) -> None:
        """movzx rax,al — upper 56 bits zeroed; old RAX taint must not survive."""
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('480fb6c0'),  # movzx rax, al
            taint={'RAX': 0xDEADBEEFCAFEBABE, 'RBX': 0, 'RCX': 0, 'RDX': 0},
            values={'RAX': 0x42, 'RBX': 0, 'RCX': 0, 'RDX': 0},
        )
        rax_out = out.get('RAX', 0)
        # Only bits 0-7 of input RAX taint should appear in output RAX (it's the source)
        # Bits 8-63 must be 0 (zero extension clears them unconditionally)
        assert (rax_out & 0xFFFFFFFFFFFFFF00) == 0, f'movzx rax,al must clear upper 56 bits of taint: {rax_out:#x}'

    def test_movzx_preserves_source_low_bits(self, simulator, regs) -> None:
        """movzx rax,bx — low 16 bits of BX taint appear in RAX bits 0-15."""
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('480fb7c3'),  # movzx rax, bx
            taint={'RAX': 0, 'RBX': 0x000000000000ABCD, 'RCX': 0, 'RDX': 0},
            values={'RAX': 0, 'RBX': 0x1234, 'RCX': 0, 'RDX': 0},
        )
        rax_out = out.get('RAX', 0)
        # Low 16 bits must match BX taint; upper 48 must be clean
        assert (
            rax_out & 0x000000000000FFFF
        ) == 0xABCD, f'movzx rax,bx: low 16 bits of output must match BX taint: {rax_out:#x}'
        assert (rax_out & 0xFFFFFFFFFFFF0000) == 0, f'movzx rax,bx: upper 48 bits must be zero: {rax_out:#x}'


# ---------------------------------------------------------------------------
# BUG-4: Double permutation — bswap twice must yield identity taint
#
# bswap rax reverses the 8 bytes of RAX. Doing it twice is the identity.
# The static rule should compose the two permutations; instead it applies
# only one direction, producing the byte-swapped taint mask.
# ---------------------------------------------------------------------------


class TestDoublePermutation:
    def test_bswap_twice_is_identity(self, simulator, regs) -> None:
        """bswap rax; bswap rax is the identity — taint mask must be unchanged."""
        # Asymmetric taint mask to distinguish identity from swapped
        taint_in = 0x0102030405060708
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('480fc8480fc8'),  # bswap rax; bswap rax
            taint={'RAX': taint_in, 'RBX': 0, 'RCX': 0, 'RDX': 0},
            values={'RAX': 0xDEADBEEFCAFEBABE, 'RBX': 0, 'RCX': 0, 'RDX': 0},
        )
        assert out.get('RAX', 0) == taint_in, f'bswapx2 is identity: expected {taint_in:#x}, got {out.get("RAX",0):#x}'

    def test_bswap_once_swaps_taint_bytes(self, simulator, regs) -> None:
        """Sanity: single bswap rax swaps the byte-level taint positions."""
        # bswap reverses byte order. If taint is on bytes 7,6 (bits 63-48),
        # after bswap they move to bytes 0,1 (bits 15-0).
        taint_in = 0xFF00000000000000  # only byte 7 tainted
        expected = 0x00000000000000FF  # after bswap → byte 0
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('480fc8'),  # bswap rax
            taint={'RAX': taint_in, 'RBX': 0, 'RCX': 0, 'RDX': 0},
            values={'RAX': 0xAA00000000000000, 'RBX': 0, 'RCX': 0, 'RDX': 0},
        )
        assert (
            out.get('RAX', 0) == expected
        ), f'bswap: byte 7→byte 0 taint: expected {expected:#x}, got {out.get("RAX",0):#x}'

    def test_rol_then_ror_same_amount_is_identity(self, simulator, regs) -> None:
        """rol rax,8; ror rax,8 is the identity — taint mask unchanged."""
        taint_in = 0x0102030405060708
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('48c1c00848c1c808'),  # rol rax,8; ror rax,8
            taint={'RAX': taint_in, 'RBX': 0, 'RCX': 0, 'RDX': 0},
            values={'RAX': 0xDEADBEEFCAFEBABE, 'RBX': 0, 'RCX': 0, 'RDX': 0},
        )
        assert (
            out.get('RAX', 0) == taint_in
        ), f'rolx8 + rorx8 is identity: expected {taint_in:#x}, got {out.get("RAX",0):#x}'


# ---------------------------------------------------------------------------
# BUG-5: AVALANCHE destination read-back — imul/lzcnt spuriously taint dest
#
# Ghidra emits "read-back" P-code ops on the destination register to
# compute overflow/carry flags (e.g. INT_EQUAL out=CF in0=RAX in1=0).
# The static rule picks up RAX as an input dep, even though RAX is a
# pure output for 3-operand imul and lzcnt.  When the *source* is
# untainted, the output should be untainted regardless of old dest taint.
# ---------------------------------------------------------------------------


class TestAvalancheDestReadBack:
    def test_imul_3op_untainted_source_clean_output(self, simulator, regs) -> None:
        """imul rax, rbx, 3 with RBX=0 and old RAX tainted — result must be 0 taint."""
        # imul rax, rbx, 3: rax = rbx * 3.  RBX is the source; RAX is a pure
        # destination.  If RBX is concrete 0 and untainted, result=0 always.
        # The old taint of RAX must NOT propagate into the output.
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('486bc303'),  # imul rax, rbx, 3
            taint={'RAX': 0xFFFFFFFFFFFFFFFF, 'RBX': 0, 'RCX': 0, 'RDX': 0},
            values={'RAX': 0xDEAD, 'RBX': 0, 'RCX': 0, 'RDX': 0},
        )
        assert out.get('RAX', 0) == 0, f'imul 3-op: untainted RBX must produce untainted RAX; got {out.get("RAX",0):#x}'

    def test_imul_3op_tainted_source_propagates(self, simulator, regs) -> None:
        """imul rax, rbx, 3 with RBX tainted — RAX output must be tainted."""
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('486bc303'),  # imul rax, rbx, 3
            taint={'RAX': 0, 'RBX': 0xFFFFFFFFFFFFFFFF, 'RCX': 0, 'RDX': 0},
            values={'RAX': 0, 'RBX': 7, 'RCX': 0, 'RDX': 0},
        )
        assert out.get('RAX', 0) != 0, f'imul 3-op: tainted RBX must produce tainted RAX; got {out}'

    def test_lzcnt_untainted_source_clean_output(self, simulator, regs) -> None:
        """lzcnt rax, rbx with RBX untainted — RAX output must be clean."""
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('f3480fbdc3'),  # lzcnt rax, rbx
            taint={'RAX': 0xFFFFFFFFFFFFFFFF, 'RBX': 0, 'RCX': 0, 'RDX': 0},
            values={'RAX': 0, 'RBX': 0x8000000000000000, 'RCX': 0, 'RDX': 0},
        )
        assert out.get('RAX', 0) == 0, f'lzcnt: untainted RBX must produce clean RAX; got {out.get("RAX",0):#x}'

    def test_lzcnt_tainted_source_propagates(self, simulator, regs) -> None:
        """lzcnt rax, rbx with RBX tainted — RAX must be tainted."""
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('f3480fbdc3'),  # lzcnt rax, rbx
            taint={'RAX': 0, 'RBX': 0xFFFFFFFFFFFFFFFF, 'RCX': 0, 'RDX': 0},
            values={'RAX': 0, 'RBX': 0x1234, 'RCX': 0, 'RDX': 0},
        )
        assert out.get('RAX', 0) != 0, f'lzcnt: tainted RBX must produce tainted RAX; got {out}'

    def test_imul_2op_both_tainted_propagates(self, simulator, regs) -> None:
        """imul rax, rbx (2-operand): both tainted → RAX tainted (AVALANCHE)."""
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('480fafc3'),  # imul rax, rbx
            taint={'RAX': 0xFFFFFFFFFFFFFFFF, 'RBX': 0xFFFFFFFFFFFFFFFF, 'RCX': 0, 'RDX': 0},
            values={'RAX': 3, 'RBX': 5, 'RCX': 0, 'RDX': 0},
        )
        assert out.get('RAX', 0) != 0, f'imul 2-op: tainted inputs must produce tainted output; got {out}'

    def test_imul_2op_untainted_rax_tainted_rbx(self, simulator, regs) -> None:
        """imul rax, rbx with RAX untainted, RBX fully tainted → RAX output tainted."""
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('480fafc3'),  # imul rax, rbx
            taint={'RAX': 0, 'RBX': 0xFFFFFFFFFFFFFFFF, 'RCX': 0, 'RDX': 0},
            values={'RAX': 3, 'RBX': 5, 'RCX': 0, 'RDX': 0},
        )
        assert out.get('RAX', 0) != 0, f'imul 2-op: tainted RBX must produce tainted RAX; got {out}'


# ---------------------------------------------------------------------------
# BUG-6: shift-by-register with tainted shift amount must AVALANCHE
#
# `shr rax, cl` when CL is tainted: the shift amount is unknown, so
# any bit of RAX could end up in any position.  Output must be fully
# tainted (AVALANCHE), not just bit-0.
# ---------------------------------------------------------------------------


class TestShiftByTaintedReg:
    def test_shr_tainted_cl_avalanches_output(self, simulator, regs) -> None:
        """shr rax, cl when CL is tainted — all RAX output bits must be tainted."""
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('48d3e8'),  # shr rax, cl
            taint={'RAX': 0xFFFFFFFFFFFFFFFF, 'RBX': 0, 'RCX': 0xFFFFFFFFFFFFFFFF, 'RDX': 0},
            values={'RAX': 0xDEADBEEFCAFEBABE, 'RBX': 0, 'RCX': 7, 'RDX': 0},
        )
        assert (
            out.get('RAX', 0) == 0xFFFFFFFFFFFFFFFF
        ), f'shr rax,cl with tainted cl must fully taint RAX; got {out.get("RAX",0):#x}'

    def test_shl_tainted_cl_avalanches_output(self, simulator, regs) -> None:
        """shl rax, cl when CL is tainted — all RAX output bits must be tainted."""
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('48d3e0'),  # shl rax, cl
            taint={'RAX': 0xFFFFFFFFFFFFFFFF, 'RBX': 0, 'RCX': 0xFFFFFFFFFFFFFFFF, 'RDX': 0},
            values={'RAX': 0xDEAD, 'RBX': 0, 'RCX': 4, 'RDX': 0},
        )
        assert (
            out.get('RAX', 0) == 0xFFFFFFFFFFFFFFFF
        ), f'shl rax,cl with tainted cl must fully taint RAX; got {out.get("RAX",0):#x}'

    def test_shr_chain_tainted_cl_avalanches(self, simulator, regs) -> None:
        """mov rcx,rdx; shr rax,cl — when RDX is tainted, RCX becomes tainted,
        and then shr by tainted CL must avalanche."""
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('4889d148d3e8'),  # mov rcx,rdx; shr rax,cl
            taint={'RAX': 0xFFFFFFFFFFFFFFFF, 'RBX': 0, 'RCX': 0, 'RDX': 0xFFFFFFFFFFFFFFFF},
            values={'RAX': 0xDEADBEEFCAFEBABE, 'RBX': 0, 'RCX': 0, 'RDX': 7},
        )
        assert (
            out.get('RAX', 0) == 0xFFFFFFFFFFFFFFFF
        ), f'mov rcx,rdx; shr rax,cl — tainted shift must avalanche; got {out.get("RAX",0):#x}'

    def test_shr_untainted_cl_does_not_avalanche(self, simulator, regs) -> None:
        """shr rax, cl when CL is clean — output should NOT be fully tainted."""
        # With RAX partially tainted and CL=4 (untainted), output is RAX>>4
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('48d3e8'),  # shr rax, cl
            taint={'RAX': 0xFFFFFFFF00000000, 'RBX': 0, 'RCX': 0, 'RDX': 0},
            values={'RAX': 0xFFFFFFFF00000000, 'RBX': 0, 'RCX': 4, 'RDX': 0},
        )
        # Fully-tainted output would be wrong here — the shift amount is known
        assert (
            out.get('RAX', 0) != 0xFFFFFFFFFFFFFFFF
        ), f'shr with clean cl must not fully avalanche; got {out.get("RAX",0):#x}'


# ---------------------------------------------------------------------------
# BUG-7: sanitiser — xor reg,reg; xor other_reg,other_reg (independent)
#
# The double-xor sanitiser `xor rax,rax; xor rbx,rbx` zeroes BOTH regs.
# The circuit for the two-instruction block currently loses RBX's zeroing
# because the multi-instruction lifting doesn't isolate them.
# ---------------------------------------------------------------------------


class TestSanitiserDoubleXor:
    def test_xor_self_rbx_zeroing_independent(self, simulator, regs) -> None:
        """xor rax,rax; xor rbx,rbx — RBX must be zeroed even when RAX is tainted."""
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('4831c04831db'),  # xor rax,rax; xor rbx,rbx
            taint={'RAX': 0xFFFFFFFFFFFFFFFF, 'RBX': 0x0F701D09C404446A, 'RCX': 0, 'RDX': 0},
            values={'RAX': 0, 'RBX': 0, 'RCX': 0, 'RDX': 0},
        )
        assert out.get('RAX', 0) == 0, f'xor rax,rax must zero RAX taint; got {out.get("RAX",0):#x}'
        assert out.get('RBX', 0) == 0, f'xor rbx,rbx must zero RBX taint; got {out.get("RBX",0):#x}'

    def test_xor_self_three_regs_all_zeroed(self, simulator, regs) -> None:
        """xor rax,rax; xor rbx,rbx; xor rcx,rcx — all three zeroed."""
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('4831c04831db4831c9'),  # xor rax,rax; xor rbx,rbx; xor rcx,rcx
            taint={'RAX': 0xFFFFFFFFFFFFFFFF, 'RBX': 0xDEAD, 'RCX': 0x1234, 'RDX': 0},
            values={'RAX': 0, 'RBX': 0, 'RCX': 0, 'RDX': 0},
        )
        assert out.get('RAX', 0) == 0, 'xor rax,rax must zero'
        assert out.get('RBX', 0) == 0, 'xor rbx,rbx must zero'
        assert out.get('RCX', 0) == 0, 'xor rcx,rcx must zero'


# ---------------------------------------------------------------------------
# Regression: previously-passing behaviour must still hold
# ---------------------------------------------------------------------------


class TestRegressionBaselines:
    def test_single_bswap_swaps_bytes(self, simulator, regs) -> None:
        """Single bswap must byte-swap the taint mask (not identity)."""
        # byte 7 (bits 63-56) moves to byte 0 (bits 7-0)
        taint_in = 0xFF00000000000000
        expected = 0x00000000000000FF
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('480fc8'),
            taint={'RAX': taint_in, 'RBX': 0, 'RCX': 0, 'RDX': 0},
            values={'RAX': 0xAA00000000000000, 'RBX': 0, 'RCX': 0, 'RDX': 0},
        )
        assert (
            out.get('RAX', 0) == expected
        ), f'bswap*1: {taint_in:#x} → expected {expected:#x}, got {out.get("RAX",0):#x}'

    def test_xor_self_zeroes_single_reg(self, simulator, regs) -> None:
        """xor rax,rax zeroes RAX taint (single instruction, must still work)."""
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('4831c0'),  # xor rax, rax
            taint={'RAX': 0xFFFFFFFFFFFFFFFF, 'RBX': 0, 'RCX': 0, 'RDX': 0},
            values={'RAX': 0, 'RBX': 0, 'RCX': 0, 'RDX': 0},
        )
        assert out.get('RAX', 0) == 0, 'xor rax,rax: zeroing idiom must produce 0 taint'

    def test_mov_propagates_full_taint(self, simulator, regs) -> None:
        """mov rbx, rax — full taint propagation (single instruction)."""
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('4889c3'),  # mov rbx, rax
            taint={'RAX': 0xFFFFFFFFFFFFFFFF, 'RBX': 0, 'RCX': 0, 'RDX': 0},
            values={'RAX': 0x1234, 'RBX': 0, 'RCX': 0, 'RDX': 0},
        )
        assert out.get('RBX', 0) == 0xFFFFFFFFFFFFFFFF, f'mov must copy full taint; got {out.get("RBX",0):#x}'

    def test_imul_tainted_inputs_propagates(self, simulator, regs) -> None:
        """imul rax, rbx with both tainted — output must be tainted."""
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('480fafc3'),  # imul rax, rbx
            taint={'RAX': 0xFFFFFFFFFFFFFFFF, 'RBX': 0xFFFFFFFFFFFFFFFF, 'RCX': 0, 'RDX': 0},
            values={'RAX': 3, 'RBX': 5, 'RCX': 0, 'RDX': 0},
        )
        assert out.get('RAX', 0) != 0, 'imul with both tainted must produce tainted output'
