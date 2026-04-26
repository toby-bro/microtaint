"""
Performance benchmarks for microtaint.

Tracks the cost of generate_static_rule (static analysis) and
LogicCircuit.evaluate (runtime taint propagation) across all
InstructionCategory families, for both AMD64 and ARM64.

Run with:
    pytest bench_microtaint.py --benchmark-only
    pytest bench_microtaint.py --benchmark-only --benchmark-sort=mean
    pytest bench_microtaint.py --benchmark-only --benchmark-compare   # compare across runs
    pytest bench_microtaint.py --benchmark-only --benchmark-json=results.json

Drop this file next to your tests/ directory (or anywhere on the pytest path).
It has zero external deps beyond what the project already uses + pytest-benchmark.

Install pytest-benchmark if not present:
    uv add --dev pytest-benchmark
"""

# mypy: disable-error-code="no-untyped-def"


from __future__ import annotations

import pytest

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, Register

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def amd64_sim() -> CellSimulator:
    return CellSimulator(Architecture.AMD64)


@pytest.fixture(scope='module')
def arm64_sim() -> CellSimulator:
    return CellSimulator(Architecture.ARM64)


@pytest.fixture(scope='module')
def amd64_regs() -> list[Register]:
    return [
        Register('RAX', 64),
        Register('RBX', 64),
        Register('RCX', 64),
        Register('RDX', 64),
        Register('RSI', 64),
        Register('RDI', 64),
        Register('RSP', 64),
        Register('RBP', 64),
        Register('R8', 64),
        Register('R9', 64),
        Register('R10', 64),
        Register('R11', 64),
        Register('R12', 64),
        Register('R13', 64),
        Register('R14', 64),
        Register('R15', 64),
        Register('RIP', 64),
        Register('EFLAGS', 32),
        Register('ZF', 1),
        Register('CF', 1),
        Register('SF', 1),
        Register('OF', 1),
        Register('PF', 1),
        Register('EAX', 32),
        Register('EBX', 32),
        Register('ECX', 32),
        Register('EDX', 32),
        Register('AX', 16),
        Register('BX', 16),
        Register('AL', 8),
        Register('AH', 8),
        Register('BL', 8),
        Register('BH', 8),
    ]


@pytest.fixture(scope='module')
def arm64_regs() -> list[Register]:
    regs = [Register(f'X{i}', 64) for i in range(31)]
    regs += [Register(f'W{i}', 32) for i in range(8)]
    regs += [
        Register('SP', 64),
        Register('PC', 64),
        Register('NZCV', 4),
        Register('N', 1),
        Register('Z', 1),
        Register('C', 1),
        Register('V', 1),
    ]
    return regs


# ---------------------------------------------------------------------------
# Pre-built circuits (module-scoped so we pay lifting cost only once per session)
# ---------------------------------------------------------------------------

# Each entry: (fixture_name, hex_bytes, arch, description)
# We build them lazily inside the benchmark functions via a helper.


def _circuit_amd64(hexbytes: str, regs: list[Register]):
    return generate_static_rule(Architecture.AMD64, bytes.fromhex(hexbytes), regs)


def _circuit_arm64(hexbytes: str, regs: list[Register]):
    return generate_static_rule(Architecture.ARM64, bytes.fromhex(hexbytes), regs)


# ---------------------------------------------------------------------------
# ── AMD64: generate_static_rule benchmarks (lifting cost)
# ---------------------------------------------------------------------------


class TestAMD64Lifting:
    """Cost of static analysis (generate_static_rule) — run once per bytecode."""

    # MAPPED / COPY -----------------------------------------------------------

    def test_lift_mov_reg_reg(self, benchmark, amd64_regs):
        """MOV RBX, RAX — pure register copy (MAPPED)."""
        benchmark(_circuit_amd64, '4889C3', amd64_regs)

    def test_lift_mov_imm(self, benchmark, amd64_regs):
        """MOV RAX, 1 — immediate load, clears taint (MAPPED/const)."""
        benchmark(_circuit_amd64, '48B80100000000000000', amd64_regs)

    # TRANSPORTABLE -----------------------------------------------------------

    def test_lift_add_reg_reg(self, benchmark, amd64_regs):
        """ADD RAX, RBX — carry-propagating add (TRANSPORTABLE)."""
        benchmark(_circuit_amd64, '4801D8', amd64_regs)

    def test_lift_sub_reg_reg(self, benchmark, amd64_regs):
        """SUB RAX, RBX — borrow-propagating subtract (TRANSPORTABLE)."""
        benchmark(_circuit_amd64, '4829D8', amd64_regs)

    def test_lift_neg(self, benchmark, amd64_regs):
        """NEG RAX — two's complement negate (TRANSPORTABLE)."""
        benchmark(_circuit_amd64, '48F7D8', amd64_regs)

    def test_lift_lea(self, benchmark, amd64_regs):
        """LEA RAX, [RAX+RBX] — address arithmetic (TRANSPORTABLE)."""
        benchmark(_circuit_amd64, '488D0418', amd64_regs)

    # MONOTONIC ---------------------------------------------------------------

    def test_lift_and_reg_reg(self, benchmark, amd64_regs):
        """AND EAX, EBX — bitwise AND (MONOTONIC)."""
        benchmark(_circuit_amd64, '21D8', amd64_regs)

    def test_lift_or_reg_reg(self, benchmark, amd64_regs):
        """OR EAX, EBX — bitwise OR (MONOTONIC)."""
        benchmark(_circuit_amd64, '09D8', amd64_regs)

    def test_lift_not(self, benchmark, amd64_regs):
        """NOT RAX — bitwise NOT (MONOTONIC)."""
        benchmark(_circuit_amd64, '48F7D0', amd64_regs)

    def test_lift_and_imm_mask(self, benchmark, amd64_regs):
        """AND EAX, 0xFF — constant mask (MONOTONIC, 1 dep)."""
        benchmark(_circuit_amd64, '25FF000000', amd64_regs)

    # ORABLE (XOR) ------------------------------------------------------------

    def test_lift_xor_reg_reg(self, benchmark, amd64_regs):
        """XOR RAX, RBX — general XOR (ORABLE)."""
        benchmark(_circuit_amd64, '4831D8', amd64_regs)

    def test_lift_xor_zeroing(self, benchmark, amd64_regs):
        """XOR EAX, EAX — idiom: zero register, clears taint (ORABLE/zeroing)."""
        benchmark(_circuit_amd64, '31C0', amd64_regs)

    # TRANSLATABLE (shifts) ---------------------------------------------------

    def test_lift_shl_imm(self, benchmark, amd64_regs):
        """SHL RAX, 8 — left shift by immediate (TRANSLATABLE)."""
        benchmark(_circuit_amd64, '48C1E008', amd64_regs)

    def test_lift_shr_imm(self, benchmark, amd64_regs):
        """SHR RAX, 8 — logical right shift (TRANSLATABLE)."""
        benchmark(_circuit_amd64, '48C1E808', amd64_regs)

    def test_lift_sar_imm(self, benchmark, amd64_regs):
        """SAR RAX, 1 — arithmetic right shift (TRANSLATABLE)."""
        benchmark(_circuit_amd64, '48D1F8', amd64_regs)

    def test_lift_rol(self, benchmark, amd64_regs):
        """ROL RAX, 8 — rotate left (MAPPED permutation)."""
        benchmark(_circuit_amd64, '48C1C008', amd64_regs)

    def test_lift_bswap(self, benchmark, amd64_regs):
        """BSWAP RAX — byte-swap (MAPPED permutation)."""
        benchmark(_circuit_amd64, '480FC8', amd64_regs)

    # AVALANCHE (multiply/divide) ---------------------------------------------

    def test_lift_imul_2op(self, benchmark, amd64_regs):
        """IMUL RBX, RAX — 2-op signed multiply (AVALANCHE)."""
        benchmark(_circuit_amd64, '480FAFD8', amd64_regs)

    def test_lift_mul_rdx(self, benchmark, amd64_regs):
        """MUL RDX — widening unsigned multiply → RAX:RDX (AVALANCHE)."""
        benchmark(_circuit_amd64, '48F7E2', amd64_regs)

    # COND_TRANSPORTABLE / flags ----------------------------------------------

    def test_lift_cmp(self, benchmark, amd64_regs):
        """CMP EAX, EBX — sets all integer flags (MONOTONIC + flag outputs)."""
        benchmark(_circuit_amd64, '39D8', amd64_regs)

    def test_lift_test(self, benchmark, amd64_regs):
        """TEST EDX, EDX — AND for flags only (MONOTONIC)."""
        benchmark(_circuit_amd64, '85D2', amd64_regs)

    def test_lift_setz(self, benchmark, amd64_regs):
        """SETZ AL — conditional set from ZF (COND_TRANSPORTABLE)."""
        benchmark(_circuit_amd64, '0F94C0', amd64_regs)

    # XCHG (two mapped outputs) -----------------------------------------------

    def test_lift_xchg(self, benchmark, amd64_regs):
        """XCHG RAX, RCX — swaps two 64-bit registers (two MAPPED outputs)."""
        benchmark(_circuit_amd64, '4891', amd64_regs)

    # INC/DEC -----------------------------------------------------------------

    def test_lift_inc(self, benchmark, amd64_regs):
        """INC EAX (TRANSPORTABLE)."""
        benchmark(_circuit_amd64, 'FFC0', amd64_regs)


# ---------------------------------------------------------------------------
# ── AMD64: LogicCircuit.evaluate benchmarks (runtime taint propagation)
# ---------------------------------------------------------------------------


class TestAMD64Evaluate:
    """Cost of evaluating the pre-built circuit for a concrete taint context."""

    def test_eval_mov_reg_reg(self, benchmark, amd64_sim, amd64_regs):
        circuit = _circuit_amd64('4889C3', amd64_regs)
        ctx = EvalContext(
            input_values={'RAX': 0xDEADBEEF, 'RBX': 0},
            input_taint={'RAX': 0xFFFFFFFFFFFFFFFF, 'RBX': 0},
            simulator=amd64_sim,
        )
        benchmark(circuit.evaluate, ctx)

    def test_eval_add_reg_reg(self, benchmark, amd64_sim, amd64_regs):
        circuit = _circuit_amd64('4801D8', amd64_regs)
        ctx = EvalContext(
            input_values={'RAX': 0xAAAA, 'RBX': 0x5555},
            input_taint={'RAX': 0xAAAAAAAAAAAAAAAA, 'RBX': 0x5555555555555555},
            simulator=amd64_sim,
        )
        benchmark(circuit.evaluate, ctx)

    def test_eval_sub_reg_reg(self, benchmark, amd64_sim, amd64_regs):
        circuit = _circuit_amd64('4829D8', amd64_regs)
        ctx = EvalContext(
            input_values={'RAX': 0x100, 'RBX': 0x1},
            input_taint={'RAX': 0xFFFFFFFFFFFFFFFF, 'RBX': 0xFFFFFFFFFFFFFFFF},
            simulator=amd64_sim,
        )
        benchmark(circuit.evaluate, ctx)

    def test_eval_and_imm_mask(self, benchmark, amd64_sim, amd64_regs):
        circuit = _circuit_amd64('25FF000000', amd64_regs)
        ctx = EvalContext(
            input_values={'EAX': 0xFFFFFFFF},
            input_taint={'EAX': 0xFFFFFFFF},
            simulator=amd64_sim,
        )
        benchmark(circuit.evaluate, ctx)

    def test_eval_xor_zeroing(self, benchmark, amd64_sim, amd64_regs):
        circuit = _circuit_amd64('31C0', amd64_regs)
        ctx = EvalContext(
            input_values={'EAX': 0x1234},
            input_taint={'EAX': 0xFFFFFFFF},
            simulator=amd64_sim,
        )
        benchmark(circuit.evaluate, ctx)

    def test_eval_shl_imm(self, benchmark, amd64_sim, amd64_regs):
        circuit = _circuit_amd64('48C1E008', amd64_regs)
        ctx = EvalContext(
            input_values={'RAX': 0x00FF00FF00FF00FF},
            input_taint={'RAX': 0x00FF00FF00FF00FF},
            simulator=amd64_sim,
        )
        benchmark(circuit.evaluate, ctx)

    def test_eval_rol(self, benchmark, amd64_sim, amd64_regs):
        circuit = _circuit_amd64('48C1C008', amd64_regs)
        ctx = EvalContext(
            input_values={'RAX': 0x00FF00FF00FF00FF},
            input_taint={'RAX': 0x00FF00FF00FF00FF},
            simulator=amd64_sim,
        )
        benchmark(circuit.evaluate, ctx)

    def test_eval_bswap(self, benchmark, amd64_sim, amd64_regs):
        circuit = _circuit_amd64('480FC8', amd64_regs)
        ctx = EvalContext(
            input_values={'RAX': 0x0123456789ABCDEF},
            input_taint={'RAX': 0xFFFFFFFFFFFFFFFF},
            simulator=amd64_sim,
        )
        benchmark(circuit.evaluate, ctx)

    def test_eval_imul_2op(self, benchmark, amd64_sim, amd64_regs):
        circuit = _circuit_amd64('480FAFD8', amd64_regs)
        ctx = EvalContext(
            input_values={'RAX': 2, 'RBX': 3},
            input_taint={'RAX': 0xFFFFFFFFFFFFFFFF, 'RBX': 0},
            simulator=amd64_sim,
        )
        benchmark(circuit.evaluate, ctx)

    def test_eval_cmp(self, benchmark, amd64_sim, amd64_regs):
        circuit = _circuit_amd64('39D8', amd64_regs)
        ctx = EvalContext(
            input_values={'EAX': 5, 'EBX': 5},
            input_taint={'EAX': 0xFFFFFFFF, 'EBX': 0},
            simulator=amd64_sim,
        )
        benchmark(circuit.evaluate, ctx)

    def test_eval_setz(self, benchmark, amd64_sim, amd64_regs):
        circuit = _circuit_amd64('0F94C0', amd64_regs)
        ctx = EvalContext(
            input_values={'EFLAGS': 0x40},
            input_taint={'ZF': 1},
            simulator=amd64_sim,
        )
        benchmark(circuit.evaluate, ctx)

    def test_eval_xchg(self, benchmark, amd64_sim, amd64_regs):
        circuit = _circuit_amd64('4891', amd64_regs)
        ctx = EvalContext(
            input_values={'RAX': 0, 'RCX': 0},
            input_taint={'RAX': 0x10, 'RCX': 0x20},
            simulator=amd64_sim,
        )
        benchmark(circuit.evaluate, ctx)


# ---------------------------------------------------------------------------
# ── ARM64: generate_static_rule benchmarks
# ---------------------------------------------------------------------------


class TestARM64Lifting:
    """Cost of static analysis for ARM64 instructions."""

    def test_lift_mov_reg(self, benchmark, arm64_regs):
        """MOV X0, X0 (ORR alias — MAPPED)."""
        benchmark(_circuit_arm64, 'E00300AA', arm64_regs)

    def test_lift_mov_imm_zero(self, benchmark, arm64_regs):
        """MOV W0, #0 — clears taint (MAPPED/const)."""
        benchmark(_circuit_arm64, '00008052', arm64_regs)

    def test_lift_movk(self, benchmark, arm64_regs):
        """MOVK W0, #0 — partial 16-bit insert (MAPPED permutation)."""
        benchmark(_circuit_arm64, '0000A072', arm64_regs)

    def test_lift_add_reg(self, benchmark, arm64_regs):
        """ADD X0, X0, X0 (TRANSPORTABLE)."""
        benchmark(_circuit_arm64, '0000008B', arm64_regs)

    def test_lift_add_imm(self, benchmark, arm64_regs):
        """ADD W0, W0, #0 (TRANSPORTABLE)."""
        benchmark(_circuit_arm64, '00000011', arm64_regs)

    def test_lift_sub_reg(self, benchmark, arm64_regs):
        """SUB X0, X0, X1 (TRANSPORTABLE)."""
        benchmark(_circuit_arm64, '000001CB', arm64_regs)

    def test_lift_and_reg(self, benchmark, arm64_regs):
        """AND X0, X0, X0 (MONOTONIC)."""
        benchmark(_circuit_arm64, '0000008A', arm64_regs)

    def test_lift_orr_reg(self, benchmark, arm64_regs):
        """ORR X0, X0, X0 (MONOTONIC)."""
        benchmark(_circuit_arm64, '000000AA', arm64_regs)

    def test_lift_eor_reg(self, benchmark, arm64_regs):
        """EOR X0, X0, X1 (ORABLE)."""
        benchmark(_circuit_arm64, '000001CA', arm64_regs)

    def test_lift_lsr_imm(self, benchmark, arm64_regs):
        """LSR W0, W0, #1 (TRANSLATABLE)."""
        benchmark(_circuit_arm64, '007C0053', arm64_regs)

    def test_lift_ubfx(self, benchmark, arm64_regs):
        """UBFX / logical right shift by 16 (TRANSLATABLE)."""
        benchmark(_circuit_arm64, '007C1053', arm64_regs)

    def test_lift_mul(self, benchmark, arm64_regs):
        """MUL X0, X0, X0 (AVALANCHE)."""
        benchmark(_circuit_arm64, '007C009B', arm64_regs)

    def test_lift_cmp(self, benchmark, arm64_regs):
        """CMP X0, X1 / SUBS XZR — sets NZCV (TRANSPORTABLE + flags)."""
        benchmark(_circuit_arm64, '1F0001EB', arm64_regs)

    def test_lift_adds(self, benchmark, arm64_regs):
        """ADDS X0, X0, X1 — flag-setting add (TRANSPORTABLE)."""
        benchmark(_circuit_arm64, '000001AB', arm64_regs)

    def test_lift_subs(self, benchmark, arm64_regs):
        """SUBS X0, X0, X1 — flag-setting subtract (TRANSPORTABLE)."""
        benchmark(_circuit_arm64, '000001EB', arm64_regs)

    def test_lift_tst(self, benchmark, arm64_regs):
        """TST X0, X1 / ANDS XZR — AND for flags (MONOTONIC)."""
        benchmark(_circuit_arm64, '1F0001EA', arm64_regs)

    def test_lift_ldr(self, benchmark, arm64_regs):
        """LDR X0, [X0] — load from memory (MAPPED + memory dep)."""
        benchmark(_circuit_arm64, '000040F9', arm64_regs)


# ---------------------------------------------------------------------------
# ── ARM64: LogicCircuit.evaluate benchmarks
# ---------------------------------------------------------------------------


class TestARM64Evaluate:
    """Runtime taint propagation cost for ARM64."""

    def test_eval_mov_reg(self, benchmark, arm64_sim, arm64_regs):
        circuit = _circuit_arm64('E00300AA', arm64_regs)
        ctx = EvalContext(
            input_values={'X0': 0},
            input_taint={'X0': 0xFFFFFFFFFFFFFFFF},
            simulator=arm64_sim,
        )
        benchmark(circuit.evaluate, ctx)

    def test_eval_add_reg(self, benchmark, arm64_sim, arm64_regs):
        circuit = _circuit_arm64('0000008B', arm64_regs)
        ctx = EvalContext(
            input_values={'X0': 1},
            input_taint={'X0': 0xFFFFFFFFFFFFFFFF},
            simulator=arm64_sim,
        )
        benchmark(circuit.evaluate, ctx)

    def test_eval_sub_reg(self, benchmark, arm64_sim, arm64_regs):
        circuit = _circuit_arm64('000001CB', arm64_regs)
        ctx = EvalContext(
            input_values={'X0': 5, 'X1': 2},
            input_taint={'X0': 0xFFFFFFFFFFFFFFFF, 'X1': 0},
            simulator=arm64_sim,
        )
        benchmark(circuit.evaluate, ctx)

    def test_eval_eor_reg(self, benchmark, arm64_sim, arm64_regs):
        circuit = _circuit_arm64('000001CA', arm64_regs)
        ctx = EvalContext(
            input_values={'X0': 0, 'X1': 0},
            input_taint={'X0': 0xFFFFFFFFFFFFFFFF, 'X1': 0},
            simulator=arm64_sim,
        )
        benchmark(circuit.evaluate, ctx)

    def test_eval_lsr(self, benchmark, arm64_sim, arm64_regs):
        circuit = _circuit_arm64('007C0053', arm64_regs)
        ctx = EvalContext(
            input_values={'W0': 0xFFFFFFFF},
            input_taint={'W0': 0xFFFFFFFF},
            simulator=arm64_sim,
        )
        benchmark(circuit.evaluate, ctx)

    def test_eval_mul(self, benchmark, arm64_sim, arm64_regs):
        circuit = _circuit_arm64('007C009B', arm64_regs)
        ctx = EvalContext(
            input_values={'X0': 2},
            input_taint={'X0': 0xFFFFFFFFFFFFFFFF},
            simulator=arm64_sim,
        )
        benchmark(circuit.evaluate, ctx)

    def test_eval_cmp(self, benchmark, arm64_sim, arm64_regs):
        circuit = _circuit_arm64('1F0001EB', arm64_regs)
        ctx = EvalContext(
            input_values={'X0': 5, 'X1': 5},
            input_taint={'X0': 0x10, 'X1': 0},
            simulator=arm64_sim,
        )
        benchmark(circuit.evaluate, ctx)

    def test_eval_ldr(self, benchmark, arm64_sim, arm64_regs):
        circuit = _circuit_arm64('000040F9', arm64_regs)
        ctx = EvalContext(
            input_values={'X0': 0x1000},
            input_taint={'MEM_0x1000_8': 0xFFFFFFFFFFFFFFFF},
            simulator=arm64_sim,
        )
        benchmark(circuit.evaluate, ctx)
