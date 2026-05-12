"""
Soundness regression tests for the 128-bit pcode-frame truncation bug
identified in report_1778576468.json (May 12 2026 benchmark run, id=7675).

  Pattern — CQO sign-extension output under-taints RDX.

  ``cqo; xor rdx, rcx``

  When T_RAX[63]=1 (sign bit tainted), CQO's output RDX should be fully
  tainted: all 64 bits of RDX are copies of RAX's sign bit via sign-extension.
  Microtaint instead reported T_RDX = T_RAX (only the single differing bit),
  propagating the wrong taint into the subsequent XOR.

  Root cause: the pcode-native evaluators (_PCodeFrame in cell.pyx and Frame
  in cell_core.h) stored all unique-space values as uint64_t.  CQO lifts as:

    u0 (16 bytes) = INT_SEXT(RAX)    ← 128-bit sign extension
    RDX           = SUBPIECE(u0, 8)  ← high 64 bits of u0

  INT_SEXT wrote u0 truncated to 64 bits (losing the all-ones high half).
  SUBPIECE then shifted the 64-bit value right by 64 bits, which is undefined
  behaviour in C and returned the original unshifted value — giving RDX = RAX
  instead of 0xFFFF...FF.

  Fix (cell.pyx): added a uniq_hi[32] / uniq_hi_set[32] companion array.
  The wide-op block at the top of _execute_decoded intercepts any op with
  o_sz > 8 or a SUBPIECE reading a > 8-byte source.  INT_SEXT stores the
  sign-replicated high 64 bits into uniq_hi; SUBPIECE with byte-offset >= 8
  reads from uniq_hi instead of shifting a truncated value.  All other 128-bit
  ops (INT_ZEXT, INT_OR, INT_AND, INT_XOR, INT_COPY, INT_LEFT, INT_MULT for
  widening MUL/IMUL, and the 128÷64 INT_DIV / INT_REM / INT_SDIV / INT_SREM
  for DIV/IDIV) are also handled correctly.

  Fix (cell_core.h): added OP_INT_SEXT to the existing splittable block that
  decomposes 16-byte ops into two 8-byte sub-ops.  The high-half write uses
  the sign bit to fill uniq_arr[i0_off+8] with 0 or UINT64_MAX.  SUBPIECE with
  i0_sz > 8 and byte-offset >= 8 reads from uniq_arr[i0_off+8] (the adjacent
  compact slot written by the splittable block).

Each test computes the brute-force ground truth by 2^k Unicorn enumeration
and asserts ``microtaint_output ⊇ ground_truth`` (no under-tainted bits).
Evaluator-level tests verify both the Cython (PCodeCellEvaluator) and C
(PCodeCellEvaluatorC) pcode engines directly, independent of the taint circuit.
"""

# mypy: disable-error-code="no-untyped-def,no-untyped-call,attr-defined"

from __future__ import annotations

import itertools

import pytest
from unicorn import UC_ARCH_X86, UC_MODE_64
from unicorn.unicorn_py3 import Uc
from unicorn.x86_const import (
    UC_CPU_X86_BROADWELL,
    UC_X86_REG_RAX,
    UC_X86_REG_RBX,
    UC_X86_REG_RCX,
    UC_X86_REG_RDX,
    UC_X86_REG_RSP,
)

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator, MachineState
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, Register

MASK64 = 0xFFFFFFFFFFFFFFFF

_REGS_GP = [
    Register('RAX', 64),
    Register('RBX', 64),
    Register('RCX', 64),
    Register('RDX', 64),
]
_SIM = CellSimulator(Architecture.AMD64)
_SIM_PY = CellSimulator(Architecture.AMD64, use_unicorn=False, use_c=False)
_SIM_C = CellSimulator(Architecture.AMD64, use_unicorn=False, use_c=True)

_UC_REGS = {
    'RAX': UC_X86_REG_RAX,
    'RBX': UC_X86_REG_RBX,
    'RCX': UC_X86_REG_RCX,
    'RDX': UC_X86_REG_RDX,
}
_BASE_ADDR = 0x400000
_PAGE_SIZE = 0x1000
_STACK_BASE = 0x500000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _brute_force_gt(  # noqa: C901
    bytestring: bytes,
    state: dict[str, int],
    taint: dict[str, int],
) -> dict[str, int]:
    """2^k Unicorn enumeration — exact bit-level ground truth."""
    positions: list[tuple[str, int]] = []
    for reg, mask in taint.items():
        for bit in range(64):
            if (mask >> bit) & 1:
                positions.append((reg, bit))
    if len(positions) > 14:
        pytest.skip(f'k={len(positions)} exceeds enumeration budget')

    results: list[dict[str, int]] = []
    for assignment in itertools.product([0, 1], repeat=len(positions)):
        s = dict(state)
        for (reg, bit), val in zip(positions, assignment, strict=False):
            if val:
                s[reg] = (s[reg] ^ (1 << bit)) & MASK64
        uc = Uc(UC_ARCH_X86, UC_MODE_64)
        uc.ctl_set_cpu_model(UC_CPU_X86_BROADWELL)
        uc.mem_map(_BASE_ADDR, _PAGE_SIZE)
        uc.mem_map(_STACK_BASE, _PAGE_SIZE)
        uc.mem_write(_BASE_ADDR, bytestring + b'\x90' * 16)
        for reg in ('RAX', 'RBX', 'RCX', 'RDX'):
            uc.reg_write(_UC_REGS[reg], s.get(reg, 0))
        uc.reg_write(UC_X86_REG_RSP, _STACK_BASE + _PAGE_SIZE - 0x100)
        uc.emu_start(_BASE_ADDR, _BASE_ADDR + len(bytestring))
        results.append({r: uc.reg_read(_UC_REGS[r]) for r in ('RAX', 'RBX', 'RCX', 'RDX')})

    output_taint: dict[str, int] = {}
    for reg in ('RAX', 'RBX', 'RCX', 'RDX'):
        all_or = 0
        all_and = MASK64
        for r in results:
            all_or |= r[reg]
            all_and &= r[reg]
        output_taint[reg] = all_or ^ all_and
    return output_taint


def _eval_microtaint(
    bytestring: bytes,
    state: dict[str, int],
    taint: dict[str, int],
) -> dict[str, int]:
    circuit = generate_static_rule(Architecture.AMD64, bytestring, _REGS_GP)
    full_state = {r.name: state.get(r.name, 0) for r in _REGS_GP}
    full_taint = {r.name: taint.get(r.name, 0) for r in _REGS_GP}
    ctx = EvalContext(input_taint=full_taint, input_values=full_state, simulator=_SIM)
    raw = circuit.evaluate(ctx)
    return {r.name: raw.get(r.name, 0) & MASK64 for r in _REGS_GP}


def _assert_sound(
    label: str,
    bytestring: bytes,
    state: dict[str, int],
    taint: dict[str, int],
    check_regs: tuple[str, ...] = ('RAX', 'RBX', 'RCX', 'RDX'),
) -> None:
    gt = _brute_force_gt(bytestring, state, taint)
    mt = _eval_microtaint(bytestring, state, taint)
    misses = []
    for reg in check_regs:
        gv = gt.get(reg, 0)
        tv = mt.get(reg, 0)
        under = gv & ~tv & MASK64
        if under:
            misses.append(
                f'{reg}: missed=0x{under:016x}  GT=0x{gv:016x}  microtaint=0x{tv:016x}',
            )
    assert not misses, f'[{label}] microtaint under-taints.\n  ' + '\n  '.join(misses)


class _CellProxy:
    """Minimal cell-like object for evaluate_concrete / evaluate_differential."""

    def __init__(self, instr_hex: str, out_reg: str) -> None:
        self.instruction = instr_hex
        self.out_reg = out_reg
        self.out_bit_start = 0
        self.out_bit_end = 63


# ---------------------------------------------------------------------------
# Evaluator-level unit tests: CQO concrete simulation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('sim_label', 'sim'),
    [
        ('cython', _SIM_PY),
        ('c', _SIM_C),
    ],
)
def test_cqo_concrete_sign_positive(sim_label: str, sim: CellSimulator) -> None:
    """CQO with MSB=0: RDX must be 0x0000...0000."""
    cell = _CellProxy('4899', 'RDX')
    result = sim.evaluate_concrete(cell, MachineState(regs={'RAX': 0x4E4C2BA4E3ABA41E}, mem={}))
    assert result == 0, (
        f'[{sim_label}] CQO(MSB=0): expected RDX=0x0, got {result:#018x}.  '
        'The pcode frame must store the low 64 bits of INT_SEXT correctly.'
    )


@pytest.mark.parametrize(
    ('sim_label', 'sim'),
    [
        ('cython', _SIM_PY),
        ('c', _SIM_C),
    ],
)
def test_cqo_concrete_sign_negative(sim_label: str, sim: CellSimulator) -> None:
    """CQO with MSB=1: RDX must be 0xFFFF...FFFF."""
    cell = _CellProxy('4899', 'RDX')
    result = sim.evaluate_concrete(cell, MachineState(regs={'RAX': 0xCE4C2BA4E3ABA41E}, mem={}))
    assert result == MASK64, (
        f'[{sim_label}] CQO(MSB=1): expected RDX=0xffffffffffffffff, got {result:#018x}.  '
        'SUBPIECE must read the sign-replicated high 64 bits, not the truncated low half.'
    )


@pytest.mark.parametrize(
    ('sim_label', 'sim'),
    [
        ('cython', _SIM_PY),
        ('c', _SIM_C),
    ],
)
def test_cqo_differential(sim_label: str, sim: CellSimulator) -> None:
    """CQO differential (SimH XOR SimL) must be 0xFFFF...FFFF when T_RAX[63]=1."""
    cell = _CellProxy('4899', 'RDX')
    # High polarity: RAX = V | T = 0x4E... | 0x8000... = 0xCE...  (MSB=1 → RDX=0xFFFF)
    # Low  polarity: RAX = V & ~T = 0x4E... & ~0x8000... = 0x4E... (MSB=0 → RDX=0x0)
    h = sim.evaluate_concrete(cell, MachineState(regs={'RAX': 0xCE4C2BA4E3ABA41E}, mem={}))
    l = sim.evaluate_concrete(cell, MachineState(regs={'RAX': 0x4E4C2BA4E3ABA41E}, mem={}))  # noqa: E741
    assert h ^ l == MASK64, (
        f'[{sim_label}] CQO differential: expected 0xffffffffffffffff, got {h^l:#018x}. '
        'All 64 bits of RDX depend on RAX bit 63 via sign-extension.'
    )


# ---------------------------------------------------------------------------
# Circuit-level soundness: report id=7675 exact case
# ---------------------------------------------------------------------------


def test_cqo_xor_report_exact() -> None:
    """Report id=7675: cqo; xor rdx, rcx.

    T_RAX[63]=1 (sign bit tainted).  CQO produces all-ones or all-zeros
    in RDX depending on the sign bit, so all 64 bits of RDX are uncertain.
    The subsequent XOR with clean RCX preserves that full uncertainty.
    microtaint must report T_RDX = 0xFFFF...FFFF.
    """
    _assert_sound(
        'cqo_xor_report_exact',
        bytes.fromhex('48994831ca'),  # cqo ; xor rdx, rcx
        state={
            'RAX': 5641932420382696478,
            'RBX': 4898647517017922326,
            'RCX': 3239610637363566272,
            'RDX': 1148298835101224732,
        },
        taint={
            'RAX': 9223372036854775808,  # 0x8000000000000000 — bit 63
            'RBX': 0,
            'RCX': 0,
            'RDX': 72057594037927936,  # 0x100000000000000 — bit 56
        },
        check_regs=('RDX',),
    )


def test_cqo_sign_bit_propagates_to_all_rdx_bits() -> None:
    """CQO alone: T_RAX[63]=1 must produce T_RDX=0xFFFF...FFFF."""
    _assert_sound(
        'cqo_sign_bit_alone',
        bytes.fromhex('4899'),  # cqo
        state={'RAX': 0x8000000000000000, 'RBX': 0, 'RCX': 0, 'RDX': 0},
        taint={'RAX': 1 << 63, 'RBX': 0, 'RCX': 0, 'RDX': 0},
        check_regs=('RDX',),
    )


def test_cqo_low_bits_do_not_taint_rdx() -> None:
    """CQO: non-sign bits of RAX must not propagate to RDX.

    This is a precision test: when only a low bit of RAX is tainted,
    RDX depends only on the sign bit of RAX, which is concrete here.
    microtaint may over-taint (sound but imprecise), but must not under-taint.
    The GT says T_RDX=0; we only assert soundness, not precision.
    """
    # k=1 so GT is computable; RAX bit 0 tainted, sign bit (63) is concrete 0.
    gt = _brute_force_gt(
        bytes.fromhex('4899'),
        state={'RAX': 0x1234, 'RBX': 0, 'RCX': 0, 'RDX': 0},
        taint={'RAX': 1, 'RBX': 0, 'RCX': 0, 'RDX': 0},
    )
    mt = _eval_microtaint(
        bytes.fromhex('4899'),
        state={'RAX': 0x1234, 'RBX': 0, 'RCX': 0, 'RDX': 0},
        taint={'RAX': 1, 'RBX': 0, 'RCX': 0, 'RDX': 0},
    )
    under = gt.get('RDX', 0) & ~mt.get('RDX', 0) & MASK64
    assert under == 0, (
        f'T_RDX under-tainted: missed=0x{under:016x} GT=0x{gt.get("RDX",0):016x} MT=0x{mt.get("RDX",0):016x}'
    )


def test_cqo_followed_by_and() -> None:
    """cqo; and rdx, rax — T_RAX[63] taints RDX, then AND propagates it."""
    _assert_sound(
        'cqo_and_rdx_rax',
        bytes.fromhex('489921c2'),  # cqo ; and rdx, rax
        state={'RAX': 0xFF00FF00FF00FF00, 'RBX': 0, 'RCX': 0, 'RDX': 0},
        taint={'RAX': 1 << 63, 'RBX': 0, 'RCX': 0, 'RDX': 0},
        check_regs=('RDX',),
    )


def test_cqo_followed_by_neg() -> None:
    """cqo; neg rdx — T_RAX[63] taints RDX, negation propagates it."""
    _assert_sound(
        'cqo_neg_rdx',
        bytes.fromhex('489948f7da'),  # cqo ; neg rdx
        state={'RAX': 0xA000000000000000, 'RBX': 0, 'RCX': 0, 'RDX': 0},
        taint={'RAX': 1 << 63, 'RBX': 0, 'RCX': 0, 'RDX': 0},
        check_regs=('RDX',),
    )
