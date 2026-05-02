"""
test_flag_and_loop_bugs.py
==========================

Tests for two distinct bug classes found in the benchmark:

BUG-A: CBRANCH passthrough applied to loop instructions
  tzcnt/bsf/bsr lift as software loops in Ghidra P-code (backward CBRANCH).
  Our generic CBRANCH passthrough — which ORs in old destination taint to handle
  the not-taken path of CMOV — wrongly applies to these instructions too.
  Result: old T_RAX leaks into tzcnt/bsf/bsr output even when source is untainted.
  Fix: only apply the passthrough when the CBRANCH is *forward* (skips a write),
  not backward (loops back over a write that is guaranteed to execute on exit).

BUG-B: Flag taint not threaded through ChainedCircuit
  Multi-instruction sequences like `add rax,rbx; adc rcx,rdx` are split into
  per-instruction sub-circuits by ChainedCircuit. The sub-circuit for `add`
  correctly produces T_CF (carry flag taint), and ChainedCircuit threads the
  output taint dict forward — but only if CF is in the state_format. When the
  caller passes a state_format without flag registers, CF never appears in the
  assignments, and `adc` sees T_CF=0 in step 2.
  Fix: ChainedCircuit must evaluate each sub-circuit with a state_format that
  includes the x86 flag registers (CF, OF, ZF, SF, PF), even when the caller's
  outer state_format omits them.
"""

from __future__ import annotations

import pytest

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, ImplicitTaintPolicy, Register

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

AMD64 = Architecture.AMD64
MASK64 = 0xFFFFFFFFFFFFFFFF


@pytest.fixture(scope='module')
def sim() -> CellSimulator:
    return CellSimulator(AMD64)


@pytest.fixture(scope='module')
def gp_regs() -> list[Register]:
    """Minimal GP-only state format — no flag registers."""
    return [
        Register('RAX', 64),
        Register('RBX', 64),
        Register('RCX', 64),
        Register('RDX', 64),
    ]


@pytest.fixture(scope='module')
def gp_and_flag_regs() -> list[Register]:
    """GP registers plus x86 status flags."""
    return [
        Register('RAX', 64),
        Register('RBX', 64),
        Register('RCX', 64),
        Register('RDX', 64),
        Register('CF', 1),
        Register('OF', 1),
        Register('ZF', 1),
        Register('SF', 1),
        Register('PF', 1),
    ]


def _ev(
    sim: CellSimulator,
    regs: list[Register],
    code: bytes,
    taint: dict[str, int],
    values: dict[str, int],
) -> dict[str, int]:
    circuit = generate_static_rule(AMD64, code, regs)
    ctx = EvalContext(
        input_taint=taint,
        input_values=values,
        simulator=sim,
        implicit_policy=ImplicitTaintPolicy.IGNORE,
    )
    return circuit.evaluate(ctx)


# ===========================================================================
# BUG-A: CBRANCH passthrough on loop instructions
#
# tzcnt, bsf, and bsr all lift as software loops in Ghidra P-code using a
# *backward* CBRANCH (loop body jumps back to start). The output register is
# written UNCONDITIONALLY on loop exit. The CMOV passthrough must NOT apply.
# ===========================================================================


class TestCbranchPassthroughLoopInstructions:

    def test_tzcnt_untainted_source_no_dest_passthrough(self, sim: CellSimulator, gp_regs: list[Register]) -> None:
        """tzcnt rax, rbx — RBX untainted, old RAX tainted → output must be 0."""
        out = _ev(
            sim, gp_regs, bytes.fromhex('f3480fbcc3'),
            taint={'RAX': MASK64, 'RBX': 0, 'RCX': 0, 'RDX': 0},
            values={'RAX': 0, 'RBX': 15016014666445342834, 'RCX': 0, 'RDX': 0},
        )
        assert out.get('RAX', 0) == 0, (
            f'tzcnt: untainted RBX must produce untainted RAX, '
            f'got {out.get("RAX", 0):#x} (old T_RAX must NOT leak through)'
        )

    def test_tzcnt_tainted_source_propagates(self, sim: CellSimulator, gp_regs: list[Register]) -> None:
        """tzcnt rax, rbx — RBX tainted → RAX must be tainted (AVALANCHE)."""
        out = _ev(
            sim, gp_regs, bytes.fromhex('f3480fbcc3'),
            taint={'RAX': 0, 'RBX': MASK64, 'RCX': 0, 'RDX': 0},
            values={'RAX': 0, 'RBX': 12, 'RCX': 0, 'RDX': 0},
        )
        assert out.get('RAX', 0) != 0, (
            f'tzcnt: tainted RBX must produce tainted RAX; got {out}'
        )

    def test_bsf_untainted_source_no_dest_passthrough(self, sim: CellSimulator, gp_regs: list[Register]) -> None:
        """bsf rax, rbx — RBX untainted, old RAX tainted → output must be 0."""
        out = _ev(
            sim, gp_regs, bytes.fromhex('480fbcc3'),
            taint={'RAX': MASK64, 'RBX': 0, 'RCX': 0, 'RDX': 0},
            values={'RAX': 0, 'RBX': 1, 'RCX': 0, 'RDX': 0},
        )
        assert out.get('RAX', 0) == 0, (
            f'bsf: untainted RBX must produce untainted RAX, '
            f'got {out.get("RAX", 0):#x}'
        )

    def test_bsf_tainted_source_propagates(self, sim: CellSimulator, gp_regs: list[Register]) -> None:
        """bsf rax, rbx — RBX tainted → RAX must be tainted."""
        out = _ev(
            sim, gp_regs, bytes.fromhex('480fbcc3'),
            taint={'RAX': 0, 'RBX': MASK64, 'RCX': 0, 'RDX': 0},
            values={'RAX': 0, 'RBX': 0x100, 'RCX': 0, 'RDX': 0},
        )
        assert out.get('RAX', 0) != 0, (
            f'bsf: tainted RBX must produce tainted RAX; got {out}'
        )

    def test_bsr_untainted_source_no_dest_passthrough(self, sim: CellSimulator, gp_regs: list[Register]) -> None:
        """bsr rax, rbx — RBX untainted, old RAX tainted → output must be 0."""
        out = _ev(
            sim, gp_regs, bytes.fromhex('480fbdc3'),
            taint={'RAX': MASK64, 'RBX': 0, 'RCX': 0, 'RDX': 0},
            values={'RAX': 0, 'RBX': 0x80, 'RCX': 0, 'RDX': 0},
        )
        assert out.get('RAX', 0) == 0, (
            f'bsr: untainted RBX must produce untainted RAX, '
            f'got {out.get("RAX", 0):#x}'
        )

    def test_bsr_tainted_source_propagates(self, sim: CellSimulator, gp_regs: list[Register]) -> None:
        """bsr rax, rbx — RBX tainted → RAX must be tainted."""
        out = _ev(
            sim, gp_regs, bytes.fromhex('480fbdc3'),
            taint={'RAX': 0, 'RBX': MASK64, 'RCX': 0, 'RDX': 0},
            values={'RAX': 0, 'RBX': 0x1000, 'RCX': 0, 'RDX': 0},
        )
        assert out.get('RAX', 0) != 0, (
            f'bsr: tainted RBX must produce tainted RAX; got {out}'
        )

    def test_cmovz_passthrough_still_works(self, sim: CellSimulator, gp_regs: list[Register]) -> None:
        """Regression: cmovz passthrough must still apply (forward CBRANCH = skip)."""
        out = _ev(
            sim, gp_regs, bytes.fromhex('480f44c3'),   # cmovz rax, rbx
            taint={'RAX': MASK64, 'RBX': 0, 'RCX': 0, 'RDX': 0},
            values={'RAX': 0, 'RBX': 0, 'RCX': 0, 'RDX': 0},
        )
        assert out.get('RAX', 0) == MASK64, (
            f'cmovz not-taken: old T_RAX must survive; got {out.get("RAX", 0):#x}'
        )

    def test_cmovl_passthrough_still_works(self, sim: CellSimulator, gp_regs: list[Register]) -> None:
        """Regression: cmovl passthrough must still apply."""
        out = _ev(
            sim, gp_regs, bytes.fromhex('480f4cda'),   # cmovl rbx, rdx
            taint={'RAX': 0, 'RBX': MASK64, 'RCX': 0, 'RDX': 0},
            values={'RAX': 0, 'RBX': 0, 'RCX': 0, 'RDX': 0},
        )
        assert out.get('RBX', 0) == MASK64, (
            f'cmovl not-taken: old T_RBX must survive; got {out.get("RBX", 0):#x}'
        )


# ===========================================================================
# BUG-B: Flag taint threading through ChainedCircuit
#
# `add rax, rbx` produces T_CF. `adc rcx, rdx` consumes T_CF.
# When evaluated as a ChainedCircuit, the intermediate T_CF must flow from
# step 1 to step 2. This requires that the sub-circuits for intermediate steps
# include CF (and other flags) in their state_format, even when the caller's
# outer state_format only has GP registers.
# ===========================================================================


class TestFlagTaintThreading:

    def test_add_produces_cf_taint(self, sim: CellSimulator, gp_and_flag_regs: list[Register]) -> None:
        """add rax, rbx — both tainted → CF must be tainted in output."""
        out = _ev(
            sim, gp_and_flag_regs, bytes.fromhex('4801d8'),
            taint={**{r.name: 0 for r in gp_and_flag_regs},
                   'RAX': MASK64, 'RBX': MASK64},
            values={**{r.name: 0 for r in gp_and_flag_regs},
                    'RAX': MASK64, 'RBX': 1},
        )
        assert out.get('CF', 0) != 0, (
            f'add with tainted operands must produce tainted CF; got CF={out.get("CF",0):#x}'
        )

    def test_add_produces_of_taint(self, sim: CellSimulator, gp_and_flag_regs: list[Register]) -> None:
        """add rax, rbx — RBX tainted with overflow-causing value → OF/SF must be tainted.

        rep1(RBX = 0x4000... | 0x4000... = 0x4000...) + RAX(0x4000...) = 0x8000... → OF=1, SF=1
        rep2(RBX = 0x4000... & ~0x4000... = 0)         + RAX(0x4000...) = 0x4000... → OF=0, SF=0
        """
        _HALF = 0x4000000000000000
        z = {r.name: 0 for r in gp_and_flag_regs}
        out = _ev(
            sim, gp_and_flag_regs, bytes.fromhex('4801d8'),
            taint={**z, 'RBX': _HALF},
            values={**z, 'RAX': _HALF, 'RBX': _HALF},
        )
        assert out.get('OF', 0) != 0 or out.get('SF', 0) != 0, (
            f'add with RBX tainted (overflow scenario) must taint OF or SF; '
            f'got OF={out.get("OF",0):#x} SF={out.get("SF",0):#x}'
        )

    def test_sub_produces_cf_taint(self, sim: CellSimulator, gp_and_flag_regs: list[Register]) -> None:
        """sub rax, rbx — RBX tainted → CF must be tainted in output.

        rep1(RBX = MASK) produces borrow (0 - MASK = borrow → CF=1),
        rep2(RBX = 0) produces no borrow (0 - 0 = 0 → CF=0).
        Both inputs fully tainted gives equal rep1/rep2 for INT_LESS so we taint only RBX.
        """
        out = _ev(
            sim, gp_and_flag_regs, bytes.fromhex('4829d8'),
            taint={**{r.name: 0 for r in gp_and_flag_regs}, 'RBX': MASK64},
            values={**{r.name: 0 for r in gp_and_flag_regs}, 'RAX': 5, 'RBX': 0},
        )
        assert out.get('CF', 0) != 0, (
            f'sub with tainted RBX must produce tainted CF; got CF={out.get("CF",0):#x}'
        )

    def test_adc_with_tainted_cf_taints_output(self, sim: CellSimulator, gp_and_flag_regs: list[Register]) -> None:
        """adc rcx, rdx — tainted CF alone → RCX must be tainted."""
        out = _ev(
            sim, gp_and_flag_regs, bytes.fromhex('4811d1'),
            taint={**{r.name: 0 for r in gp_and_flag_regs}, 'CF': 1},
            values={**{r.name: 0 for r in gp_and_flag_regs},
                    'RCX': 100, 'RDX': 200, 'CF': 1},
        )
        assert out.get('RCX', 0) != 0, (
            f'adc with tainted CF must produce tainted RCX; got {out}'
        )

    def test_sbb_with_tainted_cf_taints_output(self, sim: CellSimulator, gp_and_flag_regs: list[Register]) -> None:
        """sbb rcx, rdx — tainted CF alone → RCX must be tainted."""
        out = _ev(
            sim, gp_and_flag_regs, bytes.fromhex('4819d1'),
            taint={**{r.name: 0 for r in gp_and_flag_regs}, 'CF': 1},
            values={**{r.name: 0 for r in gp_and_flag_regs},
                    'RCX': 500, 'RDX': 100, 'CF': 1},
        )
        assert out.get('RCX', 0) != 0, (
            f'sbb with tainted CF must produce tainted RCX; got {out}'
        )

    def test_add_adc_chain_flags_threaded(self, sim: CellSimulator, gp_and_flag_regs: list[Register]) -> None:
        """add rax,rbx; adc rcx,rdx — CF from step 1 must reach step 2."""
        z = {r.name: 0 for r in gp_and_flag_regs}
        out = _ev(
            sim, gp_and_flag_regs, bytes.fromhex('4801d84811d1'),
            taint={**z, 'RAX': MASK64, 'RBX': MASK64, 'RCX': 0, 'RDX': 0},
            values={**z, 'RAX': MASK64, 'RBX': 1, 'RCX': 0, 'RDX': 0},
        )
        # The carry from add propagates into adc → RCX must be tainted
        assert out.get('RCX', 0) != 0, (
            f'add;adc chain: carry taint must propagate to RCX; got {out}'
        )

    def test_sub_sbb_chain_flags_threaded(self, sim: CellSimulator, gp_and_flag_regs: list[Register]) -> None:
        """sub rax,rbx; sbb rcx,rdx — borrow CF from step 1 must reach step 2.

        With T_RBX = MASK64 only:
          sub step: rep1(RBX=MASK) gives borrow, rep2(RBX=0) does not → T_CF=1.
          sbb step: sees T_CF=1, computes RCX-RDX-CF → T_RCX = MASK64.
        """
        z = {r.name: 0 for r in gp_and_flag_regs}
        out = _ev(
            sim, gp_and_flag_regs, bytes.fromhex('4829d84819d1'),
            taint={**z, 'RBX': MASK64},
            values={**z, 'RAX': 0, 'RBX': 1},
        )
        assert out.get('RCX', 0) != 0, (
            f'sub;sbb chain: borrow taint must propagate to RCX; got {out}'
        )

    def test_add_adc_chain_gp_only_format_flags_threaded(
        self, sim: CellSimulator, gp_regs: list[Register]
    ) -> None:
        """add rax,rbx; adc rcx,rdx with GP-only state format.

        Even when the caller only provides GP registers in state_format,
        ChainedCircuit must thread flag taint internally so that RCX
        receives carry taint from the add step.
        """
        out = _ev(
            sim, gp_regs, bytes.fromhex('4801d84811d1'),
            taint={'RAX': MASK64, 'RBX': MASK64, 'RCX': 0, 'RDX': 0},
            values={'RAX': MASK64, 'RBX': 1, 'RCX': 0, 'RDX': 0},
        )
        # Even without CF in state_format, the chain must propagate carry taint
        assert out.get('RCX', 0) != 0, (
            f'add;adc chain (GP-only format): carry taint must propagate to RCX; '
            f'got RCX={out.get("RCX", 0):#x}'
        )

    def test_add_zf_sf_pf_tainted(self, sim: CellSimulator, gp_and_flag_regs: list[Register]) -> None:
        """add rax, rbx with tainted inputs → ZF, SF, PF should all be tainted."""
        z = {r.name: 0 for r in gp_and_flag_regs}
        out = _ev(
            sim, gp_and_flag_regs, bytes.fromhex('4801d8'),
            taint={**z, 'RAX': MASK64, 'RBX': MASK64},
            values={**z, 'RAX': 5, 'RBX': 7},
        )
        assert out.get('ZF', 0) != 0, 'add with tainted inputs must taint ZF'
        assert out.get('SF', 0) != 0, 'add with tainted inputs must taint SF'
        assert out.get('PF', 0) != 0, 'add with tainted inputs must taint PF'

    def test_cmp_taints_flags(self, sim: CellSimulator, gp_and_flag_regs: list[Register]) -> None:
        """cmp rax, rbx — tainted RAX → ZF and SF must be tainted.

        Both operands fully tainted gives equal differential results for CF/OF
        (the comparison result is always equal in both reps), so we taint RAX only.
        ZF fires when rep1(RAX=MASK) != RBX but rep2(RAX=0) == RBX.
        SF fires similarly since the sign of the difference changes.
        """
        z = {r.name: 0 for r in gp_and_flag_regs}
        # V_RBX = MASK64 (concrete), T_RAX = MASK64.
        # rep1(RAX = MASK64) cmp MASK64 → result 0 → ZF=1, CF=0
        # rep2(RAX = 0)      cmp MASK64 → 0-MASK borrow → ZF=0, CF=1
        # ZF diff=1, CF diff=1.
        out = _ev(
            sim, gp_and_flag_regs, bytes.fromhex('4839d8'),   # cmp rax, rbx
            taint={**z, 'RAX': MASK64},
            values={**z, 'RAX': 5, 'RBX': MASK64},
        )
        assert out.get('ZF', 0) != 0, (
            f'cmp with tainted RAX must taint ZF; got {out}'
        )
        assert out.get('CF', 0) != 0, (
            f'cmp with tainted RAX must taint CF; got {out}'
        )

    def test_cmp_jnz_chain_flags_thread(self, sim: CellSimulator, gp_and_flag_regs: list[Register]) -> None:
        """cmp produces tainted ZF; setne reads it — ZF taint must not be lost."""
        # setne al = set AL to 1 if ZF=0. With tainted ZF, AL must be tainted.
        z = {r.name: 0 for r in gp_and_flag_regs}
        # cmp rax,rbx; sete al
        code = bytes.fromhex('48 39 d8 0f 94 c0'.replace(' ', ''))
        out = _ev(
            sim, gp_and_flag_regs, code,
            taint={**z, 'RAX': MASK64, 'RBX': MASK64},
            values={**z, 'RAX': 5, 'RBX': 3},
        )
        # sete al writes al (low byte of RAX) — it should be tainted from ZF
        rax_out = out.get('RAX', 0)
        assert (rax_out & 0xFF) != 0, (
            f'sete after cmp with tainted operands: AL must be tainted; '
            f'got RAX={rax_out:#x}'
        )
