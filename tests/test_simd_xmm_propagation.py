"""
test_simd_xmm_propagation.py
============================
End-to-end SIMD taint propagation tests.

These tests generate the LogicCircuit for one SIMD instruction at a time
(via the same generate_static_rule + EvalContext path used by the engine
in production) and verify that taint flows correctly through 128-bit
XMM operations.

Tests are written in architectural terms -- whole XMM registers, 128-bit
values -- via the debug/test-only ``RegisterAliases`` helper, which translates
human names (XMM0, XMM1) to whatever internal geometry the engine uses.  The
tests therefore say nothing about the engine's lane representation and stay
valid across changes to it.

Coverage
--------
- Bit-precise SSE2 logic (PXOR / PAND / POR) on XMM registers (exact taint)
- 128-bit register-to-register data movement (MOVAPS) -- exact copy
- Float arithmetic (ADDSD / MULPS): classified AVALANCHE, any tainted input
  bit taints the affected destination
- CALLOTHER instructions (VPADDD, AESENC, SHA256RNDS2): rule generation and
  avalanche propagation
- Register zeroing idiom (PXOR xmm0, xmm0): zero taint even if XMM0 was tainted
"""

# ruff: noqa: ARG002

from __future__ import annotations

import pytest

from microtaint.debug.reg_aliases import RegisterAliases
from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, ImplicitTaintPolicy, Register

_A = RegisterAliases(Architecture.AMD64)
_FULL64 = 0xFFFFFFFFFFFFFFFF
_FULL128 = (1 << 128) - 1

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def simulator() -> CellSimulator:
    return CellSimulator(Architecture.AMD64)


@pytest.fixture(scope='module')
def regs() -> list[Register]:
    """State format with the GP set + all 16 XMM registers, matching the
    production wrapper.X64_FORMAT layout.  The helper expands each XMM to the
    64-bit geometry lanes the engine actually tracks."""
    gp = ['RAX', 'RBX', 'RCX', 'RDX', 'RSI', 'RDI', 'RBP', 'RSP', 'RIP', 'EFLAGS']
    return _A.state_format([*gp, *(f'XMM{i}' for i in range(16))])


def _eval(
    simulator: CellSimulator,
    regs: list[Register],
    bytestring: bytes,
    taint: dict[str, int],
    values: dict[str, int],
) -> dict[str, int]:
    """Evaluate a rule with human-keyed taint/values and return a human-keyed
    output (whole XMM registers recombined from their lanes)."""
    circuit = generate_static_rule(Architecture.AMD64, bytestring, regs)
    ectx = EvalContext(
        input_taint=_A.to_engine(taint),
        input_values=_A.to_engine(values),
        simulator=simulator,
        implicit_policy=ImplicitTaintPolicy.IGNORE,
        shadow_memory=None,
    )
    return _A.from_engine(circuit.evaluate(ectx))


# ---------------------------------------------------------------------------
# Tier 1: bit-precise SSE2 logic (PXOR / PAND / POR) -- exact taint
# ---------------------------------------------------------------------------


class TestBitwiseSSE:
    def test_pxor_xmm_xmm_propagates_taint(self, simulator: CellSimulator, regs: list[Register]) -> None:
        """pxor xmm0, xmm1 -- XOR relocates no bits, so XMM0's taint is exactly
        XMM1's taint (XMM0 was clean)."""
        # 66 0f ef c1
        out = _eval(
            simulator, regs, bytes([0x66, 0x0F, 0xEF, 0xC1]),
            taint={'XMM1': 0xDEADBEEF},
            values={'XMM1': 0xDEADBEEF},
        )
        assert out.get('XMM0', 0) == 0xDEADBEEF, f'pxor: XMM0 taint {out.get("XMM0", 0):#x} != 0xdeadbeef'

    def test_pxor_zeroing_idiom_emits_no_taint(self, simulator: CellSimulator, regs: list[Register]) -> None:
        """pxor xmm0, xmm0 zeros the register.  The rule generator recognises
        this idiom and emits Constant(0): XMM0 ends fully untainted even though
        it was fully tainted before."""
        # 66 0f ef c0
        out = _eval(
            simulator, regs, bytes([0x66, 0x0F, 0xEF, 0xC0]),
            taint={'XMM0': _FULL128},
            values={'XMM0': (0xBABE << 64) | 0xCAFE},
        )
        assert out.get('XMM0', 0) == 0, f'zeroing pxor leaked taint: {out.get("XMM0", 0):#x}'

    def test_pand_xmm_xmm_propagates_taint(self, simulator: CellSimulator, regs: list[Register]) -> None:
        """pand xmm0, xmm1 -- value-aware AND.  XMM0 bits 0-7 tainted, XMM1 = 0xFF
        (those bits are 1), so each tainted bit passes through: XMM0 taint = 0xFF."""
        # 66 0f db c1
        out = _eval(
            simulator, regs, bytes([0x66, 0x0F, 0xDB, 0xC1]),
            taint={'XMM0': 0xFF},
            values={'XMM1': 0xFF},
        )
        assert out.get('XMM0', 0) == 0xFF, f'pand: XMM0 taint {out.get("XMM0", 0):#x} != 0xff'

    def test_por_xmm_xmm_propagates_taint(self, simulator: CellSimulator, regs: list[Register]) -> None:
        """por xmm0, xmm1 -- OR with a clean-zero XMM0 passes XMM1's taint through
        exactly."""
        # 66 0f eb c1
        out = _eval(
            simulator, regs, bytes([0x66, 0x0F, 0xEB, 0xC1]),
            taint={'XMM1': 0xCAFEBABE},
            values={'XMM1': 0xCAFEBABE},
        )
        assert out.get('XMM0', 0) == 0xCAFEBABE, f'por: XMM0 taint {out.get("XMM0", 0):#x} != 0xcafebabe'


# ---------------------------------------------------------------------------
# Tier 2: 128-bit register moves (MOVAPS) -- exact copy
# ---------------------------------------------------------------------------


class TestSimdMoves:
    def test_movaps_xmm_xmm_propagates_full_taint(self, simulator: CellSimulator, regs: list[Register]) -> None:
        """movaps xmm0, xmm1 -- full 128-bit copy: XMM1 fully tainted -> XMM0
        fully tainted, exactly."""
        # 0f 28 c1
        out = _eval(
            simulator, regs, bytes([0x0F, 0x28, 0xC1]),
            taint={'XMM1': _FULL128},
            values={'XMM1': (0x1122334455667788 << 64) | 0xDEADBEEFCAFEBABE},
        )
        assert out.get('XMM0', 0) == _FULL128, f'movaps: XMM0 taint {out.get("XMM0", 0):#x} != full 128'

    def test_movaps_low_only_taint(self, simulator: CellSimulator, regs: list[Register]) -> None:
        """movaps is an exact copy: tainting only XMM1's low 64 bits leaves XMM0's
        high 64 bits clean (no cross-lane bleed)."""
        # 0f 28 c1
        out = _eval(
            simulator, regs, bytes([0x0F, 0x28, 0xC1]),
            taint={'XMM1': _FULL64},  # low 64 bits only
            values={'XMM1': (0xBABE << 64) | 0xCAFE},
        )
        assert out.get('XMM0', 0) == _FULL64, (
            f'movaps low-only: XMM0 taint {out.get("XMM0", 0):#x} != low-64 (high half must stay clean)'
        )

    def test_movdqu_xmm_mem_generates_rule(self, simulator: CellSimulator, regs: list[Register]) -> None:
        """movdqu xmm0, [rax] -- 128-bit load.  Smoke-test: rule generation must
        not crash and must produce assignments (memory taint needs a shadow, not
        exercised here)."""
        circuit = generate_static_rule(Architecture.AMD64, bytes([0xF3, 0x0F, 0x6F, 0x00]), regs)
        assert circuit.assignments, 'movdqu produced no assignments'


# ---------------------------------------------------------------------------
# Tier 3: float arithmetic (avalanche category)
# ---------------------------------------------------------------------------


class TestFloatAvalanche:
    def test_addsd_taints_low_half_when_input_tainted(self, simulator: CellSimulator, regs: list[Register]) -> None:
        """addsd xmm0, xmm1 -- scalar double add (FLOAT_ADD, AVALANCHE): any
        tainted input bit taints the whole affected destination half (low 64
        bits); the untouched high half stays clean."""
        # f2 0f 58 c1
        out = _eval(
            simulator, regs, bytes([0xF2, 0x0F, 0x58, 0xC1]),
            taint={'XMM1': 0x01},
            values={'XMM1': 0x3FF0000000000001},
        )
        assert out.get('XMM0', 0) == _FULL64, (
            f'addsd avalanche: XMM0 taint {out.get("XMM0", 0):#x} != low-64 avalanche'
        )

    def test_mulps_taints_low_half(self, simulator: CellSimulator, regs: list[Register]) -> None:
        """mulps xmm0, xmm1 -- packed single-precision multiply (AVALANCHE).
        Tainting one input bit must avalanche to non-zero output taint."""
        # 0f 59 c1
        out = _eval(
            simulator, regs, bytes([0x0F, 0x59, 0xC1]),
            taint={'XMM0': 0x01},
            values={'XMM0': 0x40000000_3F800000, 'XMM1': 0x40000000_40000000},
        )
        assert out.get('XMM0', 0) != 0, f'mulps: float avalanche failed; got {out.get("XMM0", 0):#x}'


# ---------------------------------------------------------------------------
# Tier 4: CALLOTHER (VPADDD, AESENC, SHA256RNDS2)
# ---------------------------------------------------------------------------


class TestCallOtherFallback:
    def test_vpaddd_ymm_rule_generation_does_not_crash(self, simulator: CellSimulator, regs: list[Register]) -> None:
        """vpaddd ymm0, ymm1, ymm2 -- uses CALLOTHER in p-code.  Rule generation
        must produce a valid LogicCircuit without raising."""
        # c5 f5 fe c2
        circuit = generate_static_rule(Architecture.AMD64, bytes([0xC5, 0xF5, 0xFE, 0xC2]), regs)
        assert circuit.assignments, 'vpaddd produced no assignments'

    def test_aesenc_avalanches_taint(self, simulator: CellSimulator, regs: list[Register]) -> None:
        """aesenc xmm0, xmm1 -- AES-NI (CALLOTHER, AVALANCHE): a tainted input
        must taint the destination."""
        # 66 0f 38 dc c1
        out = _eval(
            simulator, regs, bytes([0x66, 0x0F, 0x38, 0xDC, 0xC1]),
            taint={'XMM1': _FULL128}, values={'XMM1': _FULL128},
        )
        assert out.get('XMM0', 0) != 0, f'aesenc did not avalanche taint into XMM0; got {out.get("XMM0", 0):#x}'

    def test_sha256rnds2_rule_generation_does_not_crash(self, simulator: CellSimulator, regs: list[Register]) -> None:
        """sha256rnds2 xmm0, xmm1 -- SHA extension (CALLOTHER)."""
        # 0f 38 cb c1
        circuit = generate_static_rule(Architecture.AMD64, bytes([0x0F, 0x38, 0xCB, 0xC1]), regs)
        assert circuit.assignments, 'sha256rnds2 produced no assignments'


# ---------------------------------------------------------------------------
# Regression: GPR baseline must still work alongside the XMM registers
# ---------------------------------------------------------------------------


class TestGprBaselineUnchanged:
    """Sanity: the XMM registers in the state_format must not affect GPR
    propagation."""

    def test_mov_rbx_rax_full_taint_unchanged(self, simulator: CellSimulator, regs: list[Register]) -> None:
        out = _eval(
            simulator, regs, bytes.fromhex('4889c3'),  # mov rbx, rax
            taint={'RAX': _FULL64}, values={'RAX': 0x1234},
        )
        assert out.get('RBX', 0) == _FULL64, f'GPR mov regression: got {out.get("RBX", 0):#x}'

    def test_xor_rax_rbx_orable_unchanged(self, simulator: CellSimulator, regs: list[Register]) -> None:
        out = _eval(
            simulator, regs, bytes.fromhex('4831d8'),  # xor rax, rbx
            taint={'RAX': 0xFF, 'RBX': 0xF0}, values={'RAX': 0, 'RBX': 0},
        )
        assert out.get('RAX', 0) == (0xFF | 0xF0), f'GPR xor regression: got {out.get("RAX", 0):#x}'
