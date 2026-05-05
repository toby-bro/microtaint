"""
test_riscv_microtaint.py
========================

Comprehensive soundness test for microtaint on RV64GC, built around the
bit-flip oracle pattern from test_bit_precision_edge_cases.py.

Coverage strategy
-----------------
Each "primitive" RISC-V instruction is canonicalised to a single register
form (e.g. ADD is always ``add t0, t1, t2``) — register choice does not
change microtaint's behaviour, so varying it would just inflate the test
matrix without exercising new code paths.

For each canonical instruction we vary the *concrete operand values* across
a battery of edge-case patterns (zero, one, all-ones, sign boundaries,
alternating bits, random) and the *taint mask* across full-width, byte-
slices, sign-bit-only, etc. This dimension is where value-dependent bugs
hide — exactly how the ADDW/SUBW sign-extension bug surfaced.

Soundness check
---------------
The oracle is "true taint by exhaustive bit-flip": for each tainted input
bit, flip it once inside Unicorn, XOR the output against the base run, and
OR all those differences. A bit set in the oracle but missing in microtaint
is an UNSOUND under-taint. Microtaint may legally over-approximate (set
bits the oracle did not see) — that's just imprecision, not a bug.

Sections
--------
1.  ARITHMETIC_OPS — register-arithmetic primitives (RV64I + RV64M).
    Tested with the bit-flip oracle.  This is the soundness gate.
2.  Backend agreement — Unicorn-vs-P-code microtaint backends must match.
3.  Memory ops (loads/stores) — backend agreement only; bit-flip oracle
    isn't meaningful for memory addresses.
4.  Control-flow ops (branches/jumps) — backend agreement only.
5.  System ops (ecall/ebreak/fence/nop) — backend agreement only.
6.  Throughput benchmarks (pytest-benchmark).
7.  Sequential 10-instruction taint chain.
8.  Diagnostic dump and fallback-rate report.

Run
---
    pytest test_riscv_microtaint.py -v
    pytest test_riscv_microtaint.py -v --benchmark-warmup=on --benchmark-min-rounds=20
"""

# mypy: disable-error-code="type-arg,no-any-return,no-untyped-def"
# ruff: noqa: ARG001, S101, PLR2004, PLR0913

from __future__ import annotations

import sys
from typing import Any

import pytest
import unicorn
import unicorn.riscv_const as ur

sys.path.insert(0, '/home/claude')
from riscv_encoder import encode  # noqa: E402

from microtaint.instrumentation.ast import EvalContext  # noqa: E402
from microtaint.simulator import CellSimulator  # noqa: E402
from microtaint.sleigh.engine import generate_static_rule  # noqa: E402
from microtaint.types import Architecture, ImplicitTaintPolicy, Register  # noqa: E402

# ===========================================================================
# State format — RISC-V uses ABI-name registers (matches Ghidra SLEIGH spec)
# ===========================================================================

ABI_NAMES: list[str] = [
    'zero', 'ra', 'sp', 'gp', 'tp',
    't0', 't1', 't2',
    's0', 's1',
    'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7',
    's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11',
    't3', 't4', 't5', 't6',
]

RISCV_REGS: list[Register] = [
    Register(name=n.upper(), bits=64) for n in ABI_NAMES
] + [Register(name='PC', bits=64)]

STATE_NAMES: list[str] = [r.name for r in RISCV_REGS]

# Map state-format names → Unicorn register IDs (mirrors simulator._UC_REGS)
_UC_GP: dict[str, int] = {
    'ZERO': ur.UC_RISCV_REG_X0, 'RA': ur.UC_RISCV_REG_X1, 'SP': ur.UC_RISCV_REG_X2,
    'GP':   ur.UC_RISCV_REG_X3, 'TP': ur.UC_RISCV_REG_X4,
    'T0': ur.UC_RISCV_REG_X5, 'T1': ur.UC_RISCV_REG_X6, 'T2': ur.UC_RISCV_REG_X7,
    'S0': ur.UC_RISCV_REG_X8, 'S1': ur.UC_RISCV_REG_X9,
    'A0': ur.UC_RISCV_REG_X10, 'A1': ur.UC_RISCV_REG_X11, 'A2': ur.UC_RISCV_REG_X12,
    'A3': ur.UC_RISCV_REG_X13, 'A4': ur.UC_RISCV_REG_X14, 'A5': ur.UC_RISCV_REG_X15,
    'A6': ur.UC_RISCV_REG_X16, 'A7': ur.UC_RISCV_REG_X17,
    'S2': ur.UC_RISCV_REG_X18, 'S3': ur.UC_RISCV_REG_X19, 'S4': ur.UC_RISCV_REG_X20,
    'S5': ur.UC_RISCV_REG_X21, 'S6': ur.UC_RISCV_REG_X22, 'S7': ur.UC_RISCV_REG_X23,
    'S8': ur.UC_RISCV_REG_X24, 'S9': ur.UC_RISCV_REG_X25, 'S10': ur.UC_RISCV_REG_X26,
    'S11': ur.UC_RISCV_REG_X27,
    'T3': ur.UC_RISCV_REG_X28, 'T4': ur.UC_RISCV_REG_X29, 'T5': ur.UC_RISCV_REG_X30,
    'T6': ur.UC_RISCV_REG_X31,
    'PC': ur.UC_RISCV_REG_PC,
}

MASK64 = 0xFFFFFFFFFFFFFFFF
MASK32 = 0xFFFFFFFF
FULL_TAINT_64 = MASK64
SIGN64 = 1 << 63
SIGN32 = 1 << 31

# ===========================================================================
# Concrete-value patterns — one battery applied to *every* arithmetic op
# ===========================================================================

VALUE_PATTERNS: list[tuple[str, int, int]] = [
    ('zero',         0,                       0),
    ('ones',         MASK64,                  MASK64),
    ('one_one',      1,                       1),
    ('sign_min',     SIGN64,                  SIGN64),
    ('alt_a',        0xAAAAAAAAAAAAAAAA,      0x5555555555555555),
    ('alt_b',        0x5555555555555555,      0xAAAAAAAAAAAAAAAA),
    ('low_byte',     0x00000000000000FF,      0x00000000000000FF),
    ('high_byte',    0xFF00000000000000,      0xFF00000000000000),
    ('cross_32',     0x000000007FFFFFFF,      1),
    ('neg_one',      MASK64,                  1),
    ('big_small',    0xCAFEBABE12345678,      0x10),
    ('shift_amt',    0x123456789ABCDEF0,      4),
    ('shift_high',   1 << 63,                 63),
    ('shift_zero',   0xDEADBEEF,              0),
    ('shift_full',   0xCAFEBABE,              63),
    ('div_by_zero',  100,                     0),
    ('div_min',      SIGN64,                  MASK64),
]
_VALUE_T1 = {lbl: t1 for lbl, t1, _ in VALUE_PATTERNS}
_VALUE_T2 = {lbl: t2 for lbl, _, t2 in VALUE_PATTERNS}

TAINT_PATTERNS: list[tuple[str, int]] = [
    ('full',         FULL_TAINT_64),
    ('sign_only',    SIGN64),
    ('low_byte',     0xFF),
    ('high_byte',    0xFF00000000000000),
    ('alt',          0xAAAAAAAAAAAAAAAA),
    ('cross_32',     0x000000007FFFFFFF),
    ('low_word',     0x00000000FFFFFFFF),
    ('high_word',    0xFFFFFFFF00000000),
    ('one_bit',      1),
]
_TAINT_MASK = dict(TAINT_PATTERNS)


# ===========================================================================
# Instruction primitives
# ===========================================================================

RV64I_R: list[str] = [
    'add t0, t1, t2', 'sub t0, t1, t2',
    'sll t0, t1, t2', 'srl t0, t1, t2', 'sra t0, t1, t2',
    'and t0, t1, t2', 'or  t0, t1, t2', 'xor t0, t1, t2',
    'slt t0, t1, t2', 'sltu t0, t1, t2',
]

RV64I_W: list[str] = [
    'addw t0, t1, t2', 'subw t0, t1, t2',
    'sllw t0, t1, t2', 'srlw t0, t1, t2', 'sraw t0, t1, t2',
]

RV64M_R: list[str] = [
    'mul t0, t1, t2', 'mulh t0, t1, t2',
    'mulhsu t0, t1, t2', 'mulhu t0, t1, t2',
    'div t0, t1, t2', 'divu t0, t1, t2',
    'rem t0, t1, t2', 'remu t0, t1, t2',
]

RV64M_W: list[str] = [
    'mulw t0, t1, t2',
    'divw t0, t1, t2', 'divuw t0, t1, t2',
    'remw t0, t1, t2', 'remuw t0, t1, t2',
]

RV64I_IMM_TEMPLATES: list[tuple[str, list[int]]] = [
    ('addi t0, t1, {imm}',  [0, 1, -1, 100, -100, 2047, -2048]),
    ('xori t0, t1, {imm}',  [0, -1, 0xff, -2048]),
    ('ori  t0, t1, {imm}',  [0, -1, 0xff, 0x555]),
    ('andi t0, t1, {imm}',  [0, -1, 0xff, 0x555, 0x800]),
    ('slti t0, t1, {imm}',  [0, 1, -1, 100, -100]),
    ('sltiu t0, t1, {imm}', [0, 1, -1, 100, 2047]),
    ('addiw t0, t1, {imm}', [0, 1, -1, 100, 2047, -2048]),
]

RV64I_SHIFTI_TEMPLATES: list[tuple[str, list[int]]] = [
    ('slli t0, t1, {imm}',  [0, 1, 4, 31, 32, 63]),
    ('srli t0, t1, {imm}',  [0, 1, 4, 31, 32, 63]),
    ('srai t0, t1, {imm}',  [0, 1, 4, 31, 32, 63]),
    ('slliw t0, t1, {imm}', [0, 1, 4, 15, 31]),
    ('srliw t0, t1, {imm}', [0, 1, 4, 15, 31]),
    ('sraiw t0, t1, {imm}', [0, 1, 4, 15, 31]),
]

U_TYPE: list[str] = ['lui t0, 0x12345', 'auipc t0, 0x12345']


def _expand_imm_templates(templates: list[tuple[str, list[int]]]) -> list[str]:
    out = []
    for tmpl, imms in templates:
        for imm in imms:
            out.append(tmpl.format(imm=imm))
    return out


ARITHMETIC_PRIMITIVES: list[str] = (
    RV64I_R + RV64I_W + RV64M_R + RV64M_W
    + _expand_imm_templates(RV64I_IMM_TEMPLATES)
    + _expand_imm_templates(RV64I_SHIFTI_TEMPLATES)
    + U_TYPE
)


# ===========================================================================
# Memory / control-flow / system — backend agreement only
# ===========================================================================

LOADS_STORES: list[tuple[str, dict, dict]] = [
    ('lb t0, 8(t1)',     {'T1': 0x80000100},      {}),
    ('lh t0, 8(t1)',     {'T1': 0x80000100},      {}),
    ('lw t0, 8(t1)',     {'T1': 0x80000100},      {}),
    ('ld t0, 8(t1)',     {'T1': 0x80000100},      {}),
    ('lbu t0, 8(t1)',    {'T1': 0x80000100},      {}),
    ('lhu t0, 8(t1)',    {'T1': 0x80000100},      {}),
    ('lwu t0, 8(t1)',    {'T1': 0x80000100},      {}),
    ('lw t0, -8(t1)',    {'T1': 0x80000100},      {}),
    ('sb t2, 8(t1)',     {'T1': 0x80000100, 'T2': 0xCAFEBABE}, {'T2': FULL_TAINT_64}),
    ('sh t2, 8(t1)',     {'T1': 0x80000100, 'T2': 0xCAFEBABE}, {'T2': FULL_TAINT_64}),
    ('sw t2, 8(t1)',     {'T1': 0x80000100, 'T2': 0xCAFEBABE}, {'T2': FULL_TAINT_64}),
    ('sd t2, 8(t1)',     {'T1': 0x80000100, 'T2': 0xCAFEBABE}, {'T2': FULL_TAINT_64}),
]

CONTROL_FLOW: list[tuple[str, dict, dict]] = [
    # Use small forward displacement (4 = next instruction); avoids
    # Unicorn UC_ERR_EXCEPTION when the branch is taken to unmapped code
    # in microtaint's bounded CODE region.
    ('beq t1, t2, 4',  {'T1': 0, 'T2': 0},       {'T1': FULL_TAINT_64}),
    ('bne t1, t2, 4',  {'T1': 0, 'T2': 1},       {'T1': FULL_TAINT_64}),
    ('blt t1, t2, 4',  {'T1': 0, 'T2': 1},       {'T1': FULL_TAINT_64}),
    ('bge t1, t2, 4',  {'T1': 1, 'T2': 0},       {'T1': FULL_TAINT_64}),
    ('bltu t1, t2, 4', {'T1': 0, 'T2': 1},       {'T1': FULL_TAINT_64}),
    ('bgeu t1, t2, 4', {'T1': 1, 'T2': 0},       {'T1': FULL_TAINT_64}),
    ('jal t0, 4',      {},                       {}),
    ('jalr t0, t1, 0', {'T1': 0x1004},           {'T1': FULL_TAINT_64}),
]

SYSTEM_OPS: list[tuple[str, dict, dict]] = [
    ('nop',    {}, {}),
    ('fence',  {}, {}),
]


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope='session')
def sim_unicorn() -> CellSimulator:
    return CellSimulator(Architecture.RISCV64, use_unicorn=True)


@pytest.fixture(scope='session')
def sim_pcode() -> CellSimulator:
    return CellSimulator(Architecture.RISCV64, use_unicorn=False)


@pytest.fixture(scope='session')
def circuit_cache() -> dict[str, Any]:
    """One generate_static_rule call per unique asm string, reused everywhere."""
    return {}


def _circuit(asm: str, cache: dict) -> Any:
    if asm not in cache:
        cache[asm] = generate_static_rule(
            Architecture.RISCV64, encode(asm), RISCV_REGS,
        )
    return cache[asm]


# ===========================================================================
# Helpers
# ===========================================================================

def _run_microtaint(sim: CellSimulator, asm: str, values: dict, taint: dict,
                    cache: dict) -> dict:
    circuit = _circuit(asm, cache)
    ctx = EvalContext(
        input_values=values, input_taint=taint, simulator=sim,
        implicit_policy=ImplicitTaintPolicy.KEEP,
    )
    return circuit.evaluate(ctx)


_BASE_CODE = 0x1000
_BASE_DATA = 0x80000000
_DATA_SIZE = 0x10000


def _new_uc() -> unicorn.Uc:
    uc = unicorn.Uc(unicorn.UC_ARCH_RISCV, unicorn.UC_MODE_RISCV64)
    uc.mem_map(_BASE_CODE, 0x4000)
    uc.mem_map(_BASE_DATA, _DATA_SIZE)
    return uc


def _run_unicorn_oracle(code: bytes, regs: dict[str, int]) -> dict[str, int]:
    """Execute one instruction in pristine Unicorn; return all GPRs + PC."""
    uc = _new_uc()
    uc.mem_write(_BASE_CODE, code)
    if 'SP' not in regs:
        uc.reg_write(_UC_GP['SP'], _BASE_DATA + _DATA_SIZE // 2)
    for name, val in regs.items():
        rid = _UC_GP.get(name.upper())
        if rid is not None:
            uc.reg_write(rid, val & MASK64)
    try:
        uc.emu_start(_BASE_CODE, _BASE_CODE + len(code))
    except unicorn.UcError:
        pass
    out: dict[str, int] = {}
    for name, rid in _UC_GP.items():
        out[name] = uc.reg_read(rid) & MASK64
    return out


def _true_taint_riscv(
    code: bytes,
    taint: dict[str, int],
    values: dict[str, int],
) -> dict[str, int]:
    """
    Per-bit sensitivity oracle.  For each tainted input bit, flip it once
    in Unicorn and OR the XOR-difference into the running result.  This is
    the SOUND minimum taint a correct engine must produce.
    """
    base_vals = {n: values.get(n, 0) & ~taint.get(n, 0) & MASK64 for n in STATE_NAMES}
    base_out = _run_unicorn_oracle(code, base_vals)
    result: dict[str, int] = dict.fromkeys(STATE_NAMES, 0)
    for reg, t in taint.items():
        if t == 0:
            continue
        for bit in range(64):
            if not (t >> bit) & 1:
                continue
            mut = dict(base_vals)
            mut[reg] = base_vals[reg] ^ (1 << bit)
            mut_out = _run_unicorn_oracle(code, mut)
            for out_reg in STATE_NAMES:
                diff = (base_out.get(out_reg, 0) ^ mut_out.get(out_reg, 0)) & MASK64
                result[out_reg] |= diff
    return result


def _diff_dicts(a: dict, b: dict) -> dict:
    return {
        k: (a.get(k, 0), b.get(k, 0))
        for k in set(a) | set(b)
        if a.get(k, 0) != b.get(k, 0)
    }


def _short_id(asm: str, vlabel: str, tlabel: str) -> str:
    return asm.replace(' ', '').replace(',', '_') + f'|v={vlabel}|t={tlabel}'


# ===========================================================================
# Test matrix builder — one parametrise row per (asm, value, taint)
# ===========================================================================

# Per-mnemonic value-pattern subsets.  Trims patterns that are pointless for
# specific opcodes (e.g. shift_high values for ADD), keeping the matrix
# tractable while still exercising each instruction's edge regions.
_PATTERN_SUBSETS: dict[str, list[str]] = {
    # Shifts: focus on shift-amount edge cases
    'sll':   ['zero', 'ones', 'sign_min', 'shift_amt', 'shift_high', 'shift_zero', 'shift_full', 'low_byte', 'high_byte', 'cross_32'],
    'srl':   ['zero', 'ones', 'sign_min', 'shift_amt', 'shift_high', 'shift_zero', 'shift_full', 'low_byte', 'high_byte', 'cross_32'],
    'sra':   ['zero', 'ones', 'sign_min', 'shift_amt', 'shift_high', 'shift_zero', 'shift_full', 'low_byte', 'high_byte', 'cross_32'],
    'sllw':  ['zero', 'ones', 'sign_min', 'shift_amt', 'shift_full', 'low_byte', 'high_byte', 'cross_32'],
    'srlw':  ['zero', 'ones', 'sign_min', 'shift_amt', 'shift_full', 'low_byte', 'high_byte', 'cross_32'],
    'sraw':  ['zero', 'ones', 'sign_min', 'shift_amt', 'shift_full', 'low_byte', 'high_byte', 'cross_32'],
    # Division: include divide-by-zero and signed-min/-1 overflow case
    'div':   ['zero', 'ones', 'one_one', 'sign_min', 'big_small', 'div_by_zero', 'div_min', 'cross_32'],
    'divu':  ['zero', 'ones', 'one_one', 'big_small', 'div_by_zero', 'cross_32'],
    'rem':   ['zero', 'ones', 'one_one', 'sign_min', 'big_small', 'div_by_zero', 'div_min', 'cross_32'],
    'remu':  ['zero', 'ones', 'one_one', 'big_small', 'div_by_zero', 'cross_32'],
    'divw':  ['zero', 'ones', 'one_one', 'sign_min', 'big_small', 'div_by_zero', 'cross_32'],
    'divuw': ['zero', 'ones', 'one_one', 'big_small', 'div_by_zero', 'cross_32'],
    'remw':  ['zero', 'ones', 'one_one', 'sign_min', 'big_small', 'div_by_zero', 'cross_32'],
    'remuw': ['zero', 'ones', 'one_one', 'big_small', 'div_by_zero', 'cross_32'],
    # Multiply: avalanche; exercise sign and high-byte
    'mul':    ['zero', 'ones', 'one_one', 'sign_min', 'low_byte', 'high_byte', 'big_small', 'cross_32'],
    'mulh':   ['zero', 'ones', 'one_one', 'sign_min', 'low_byte', 'high_byte', 'big_small', 'cross_32'],
    'mulhsu': ['zero', 'ones', 'one_one', 'sign_min', 'low_byte', 'high_byte', 'big_small', 'cross_32'],
    'mulhu':  ['zero', 'ones', 'one_one', 'sign_min', 'low_byte', 'high_byte', 'big_small', 'cross_32'],
    'mulw':   ['zero', 'ones', 'one_one', 'sign_min', 'cross_32', 'high_byte', 'low_byte', 'big_small'],
    # W-suffix: cross_32 is THE pattern that finds sign-extension bugs
    'addw':  ['zero', 'ones', 'sign_min', 'alt_a', 'cross_32', 'high_byte', 'low_byte', 'big_small'],
    'subw':  ['zero', 'ones', 'sign_min', 'alt_a', 'cross_32', 'high_byte', 'low_byte', 'big_small'],
    # U-type: no register inputs to vary
    'lui':   ['zero'],
    'auipc': ['zero'],
}

_DEFAULT_PATTERNS = ['zero', 'ones', 'one_one', 'sign_min', 'alt_a',
                     'low_byte', 'high_byte', 'cross_32', 'big_small', 'neg_one']

_DEFAULT_TAINT_SUBSET = ['full', 'sign_only', 'low_byte', 'high_byte',
                         'cross_32', 'one_bit']


def _patterns_for(mnem: str) -> list[str]:
    return _PATTERN_SUBSETS.get(mnem, _DEFAULT_PATTERNS)


def _build_oracle_matrix() -> list[tuple[str, str, str, dict, dict]]:
    """Returns list of (asm, value_label, taint_label, values, taint_t1)."""
    matrix: list[tuple[str, str, str, dict, dict]] = []
    for asm in ARITHMETIC_PRIMITIVES:
        mnem = asm.split()[0].lower()
        patterns = _patterns_for(mnem)
        for vlabel in patterns:
            v_t1 = _VALUE_T1[vlabel]
            v_t2 = _VALUE_T2[vlabel]
            values = {'T1': v_t1, 'T2': v_t2}
            for tlabel in _DEFAULT_TAINT_SUBSET:
                taint = {'T1': _TAINT_MASK[tlabel]}
                matrix.append((asm, vlabel, tlabel, values, taint))
    return matrix


ORACLE_MATRIX = _build_oracle_matrix()

# Instructions that don't read T2 — for the T2-source variant tests
_NO_T2_MNEMS: set[str] = {
    'addi', 'xori', 'ori', 'andi', 'slti', 'sltiu', 'addiw',
    'slli', 'srli', 'srai', 'slliw', 'srliw', 'sraiw',
    'lui', 'auipc',
}


# ===========================================================================
# PART 1 — Bit-flip oracle on every arithmetic primitive (T1 source)
# ===========================================================================

@pytest.mark.parametrize(
    ('asm', 'vlabel', 'tlabel', 'values', 'taint'),
    ORACLE_MATRIX,
    ids=[_short_id(a, v, t) for a, v, t, _, _ in ORACLE_MATRIX],
)
def test_oracle_soundness(
    sim_unicorn: CellSimulator, circuit_cache: dict,
    asm: str, vlabel: str, tlabel: str,
    values: dict, taint: dict,
) -> None:
    """
    For every (instruction, value_pattern, taint_pattern) cell, microtaint
    must produce taint that contains every bit the bit-flip oracle saw.
    Missing bits = under-tainting = unsoundness.
    """
    code = encode(asm)
    oracle = _true_taint_riscv(code, taint, values)
    micro = _run_microtaint(sim_unicorn, asm, values, taint, circuit_cache)
    unsound = {}
    for reg in STATE_NAMES:
        o = oracle.get(reg, 0) & MASK64
        m = micro.get(reg, 0) & MASK64
        missing = o & ~m
        if missing:
            unsound[reg] = (o, m, missing)
    assert not unsound, (
        f'\nUNSOUND: {asm!r}  values=({vlabel})  taint=({tlabel})\n'
        f'  values={values}\n  taint ={taint}\n'
        + '\n'.join(
            f'    {r}: oracle={hex(o)}  micro={hex(m)}  MISSING={hex(miss)}'
            for r, (o, m, miss) in unsound.items()
        )
    )


# ===========================================================================
# PART 2 — Same matrix, taint applied to T2 (catches asymmetric handling)
# ===========================================================================

@pytest.mark.parametrize(
    ('asm', 'vlabel', 'tlabel', 'values', 'taint_t1'),
    ORACLE_MATRIX,
    ids=[_short_id(a, v, t) + '|src=T2' for a, v, t, _, _ in ORACLE_MATRIX],
)
def test_oracle_soundness_t2(
    sim_unicorn: CellSimulator, circuit_cache: dict,
    asm: str, vlabel: str, tlabel: str,
    values: dict, taint_t1: dict,
) -> None:
    """As test_oracle_soundness, but taint applied to T2."""
    mnem = asm.split()[0].lower()
    if mnem in _NO_T2_MNEMS:
        pytest.skip(f'{mnem} does not read T2')

    code = encode(asm)
    tmask = next(iter(taint_t1.values()))
    taint = {'T2': tmask}
    oracle = _true_taint_riscv(code, taint, values)
    micro = _run_microtaint(sim_unicorn, asm, values, taint, circuit_cache)
    unsound = {}
    for reg in STATE_NAMES:
        o = oracle.get(reg, 0) & MASK64
        m = micro.get(reg, 0) & MASK64
        missing = o & ~m
        if missing:
            unsound[reg] = (o, m, missing)
    assert not unsound, (
        f'\nUNSOUND (T2 source): {asm!r}  values=({vlabel})  taint=({tlabel})\n'
        f'  values={values}\n  taint ={taint}\n'
        + '\n'.join(
            f'    {r}: oracle={hex(o)}  micro={hex(m)}  MISSING={hex(miss)}'
            for r, (o, m, miss) in unsound.items()
        )
    )


# ===========================================================================
# PART 3 — Backend agreement on every arithmetic primitive
# ===========================================================================

@pytest.mark.parametrize(
    ('asm', 'vlabel', 'tlabel', 'values', 'taint'),
    ORACLE_MATRIX,
    ids=[_short_id(a, v, t) + '|backends' for a, v, t, _, _ in ORACLE_MATRIX],
)
def test_backends_agree_arithmetic(
    sim_unicorn: CellSimulator, sim_pcode: CellSimulator, circuit_cache: dict,
    asm: str, vlabel: str, tlabel: str,
    values: dict, taint: dict,
) -> None:
    """Unicorn-backed and P-code-backed CellSimulator must produce identical taint."""
    out_u = _run_microtaint(sim_unicorn, asm, values, taint, circuit_cache)
    out_p = _run_microtaint(sim_pcode,   asm, values, taint, circuit_cache)
    diffs = _diff_dicts(out_u, out_p)
    assert not diffs, (
        f'\nBACKEND DISAGREE: {asm!r}  values=({vlabel})  taint=({tlabel})\n'
        + '\n'.join(f'    {k}: u={hex(u)} p={hex(p)}' for k, (u, p) in diffs.items())
    )


# ===========================================================================
# PART 4 — Memory ops: backend agreement
# ===========================================================================

@pytest.mark.parametrize(
    ('asm', 'values', 'taint'), LOADS_STORES,
    ids=[a.replace(' ', '_').replace(',', '').replace('(', '_').replace(')', '')
         for a, _, _ in LOADS_STORES],
)
def test_memory_backends_agree(
    sim_unicorn: CellSimulator, sim_pcode: CellSimulator, circuit_cache: dict,
    asm: str, values: dict, taint: dict,
) -> None:
    out_u = _run_microtaint(sim_unicorn, asm, values, taint, circuit_cache)
    out_p = _run_microtaint(sim_pcode,   asm, values, taint, circuit_cache)
    diffs = _diff_dicts(out_u, out_p)
    assert not diffs, f'asm={asm!r}\n' + '\n'.join(
        f'  {k}: u={hex(u)} p={hex(p)}' for k, (u, p) in diffs.items()
    )


# ===========================================================================
# PART 5 — Control-flow / system: backend agreement
# ===========================================================================

@pytest.mark.parametrize(
    ('asm', 'values', 'taint'), CONTROL_FLOW,
    ids=[a.replace(' ', '_').replace(',', '').replace('-', 'm') for a, _, _ in CONTROL_FLOW],
)
def test_controlflow_backends_agree(
    sim_unicorn: CellSimulator, sim_pcode: CellSimulator, circuit_cache: dict,
    asm: str, values: dict, taint: dict,
) -> None:
    out_u = _run_microtaint(sim_unicorn, asm, values, taint, circuit_cache)
    out_p = _run_microtaint(sim_pcode,   asm, values, taint, circuit_cache)
    diffs = _diff_dicts(out_u, out_p)
    assert not diffs, f'asm={asm!r}\n' + '\n'.join(
        f'  {k}: u={hex(u)} p={hex(p)}' for k, (u, p) in diffs.items()
    )


@pytest.mark.parametrize(
    ('asm', 'values', 'taint'), SYSTEM_OPS,
    ids=[a for a, _, _ in SYSTEM_OPS],
)
def test_system_backends_agree(
    sim_unicorn: CellSimulator, sim_pcode: CellSimulator, circuit_cache: dict,
    asm: str, values: dict, taint: dict,
) -> None:
    out_u = _run_microtaint(sim_unicorn, asm, values, taint, circuit_cache)
    out_p = _run_microtaint(sim_pcode,   asm, values, taint, circuit_cache)
    diffs = _diff_dicts(out_u, out_p)
    assert not diffs, f'asm={asm!r}\n' + '\n'.join(
        f'  {k}: u={hex(u)} p={hex(p)}' for k, (u, p) in diffs.items()
    )


# ===========================================================================
# PART 6 — Throughput benchmarks (one per representative opcode family)
# ===========================================================================

BENCH_SET: list[tuple[str, str, dict, dict]] = [
    ('ADD',    'add t0, t1, t2',  {'T1': 1, 'T2': 1}, {'T1': FULL_TAINT_64}),
    ('SUB',    'sub t0, t1, t2',  {'T1': 5, 'T2': 2}, {'T1': FULL_TAINT_64}),
    ('XOR',    'xor t0, t1, t2',  {'T1': 1, 'T2': 2}, {'T1': FULL_TAINT_64}),
    ('AND',    'and t0, t1, t2',  {'T1': 0xFF, 'T2': 0xFF}, {'T1': FULL_TAINT_64}),
    ('SLL',    'sll t0, t1, t2',  {'T1': 0xFF, 'T2': 4}, {'T1': 0xFF}),
    ('SRA',    'sra t0, t1, t2',  {'T1': SIGN64, 'T2': 4}, {'T1': SIGN64}),
    ('SLT',    'slt t0, t1, t2',  {'T1': 1, 'T2': 5}, {'T1': FULL_TAINT_64}),
    ('MUL',    'mul t0, t1, t2',  {'T1': 2, 'T2': 3}, {'T1': FULL_TAINT_64}),
    ('DIV',    'div t0, t1, t2',  {'T1': 100, 'T2': 7}, {'T1': FULL_TAINT_64}),
    ('ADDIW',  'addiw t0, t1, 1', {'T1': 0xFFFFFFFE}, {'T1': FULL_TAINT_64}),
    ('ADDW',   'addw t0, t1, t2', {'T1': 0x7FFFFFFF, 'T2': 1}, {'T1': FULL_TAINT_64}),
    ('SLLI',   'slli t0, t1, 4',  {'T1': 0xFF}, {'T1': 0xFF}),
    ('SRAI',   'srai t0, t1, 60', {'T1': SIGN64}, {'T1': SIGN64}),
    ('LUI',    'lui t0, 0x12345', {}, {}),
]


@pytest.mark.parametrize(
    ('label', 'asm', 'values', 'taint'),
    BENCH_SET, ids=[lab for lab, _, _, _ in BENCH_SET],
)
def test_bench_unicorn(
    benchmark: Any, sim_unicorn: CellSimulator, circuit_cache: dict,
    label: str, asm: str, values: dict, taint: dict,
) -> None:
    circuit = _circuit(asm, circuit_cache)

    def _go() -> dict:
        ctx = EvalContext(input_values=values, input_taint=taint,
                          simulator=sim_unicorn,
                          implicit_policy=ImplicitTaintPolicy.KEEP)
        return circuit.evaluate(ctx)
    benchmark.pedantic(_go, rounds=50, warmup_rounds=3)


@pytest.mark.parametrize(
    ('label', 'asm', 'values', 'taint'),
    BENCH_SET, ids=[lab for lab, _, _, _ in BENCH_SET],
)
def test_bench_pcode(
    benchmark: Any, sim_pcode: CellSimulator, circuit_cache: dict,
    label: str, asm: str, values: dict, taint: dict,
) -> None:
    circuit = _circuit(asm, circuit_cache)

    def _go() -> dict:
        ctx = EvalContext(input_values=values, input_taint=taint,
                          simulator=sim_pcode,
                          implicit_policy=ImplicitTaintPolicy.KEEP)
        return circuit.evaluate(ctx)
    benchmark.pedantic(_go, rounds=50, warmup_rounds=3)


# ===========================================================================
# PART 7 — Sequential 10-instruction taint chain
# ===========================================================================

TAINT_CHAIN: list[str] = [
    'add t1, t0, t0',
    'xor t2, t1, t0',
    'and t3, t2, t1',
    'sll t4, t3, t0',
    'or  t5, t4, t0',
    'mul t6, t5, t1',
    'sub a0, t6, t1',
    'slt a1, t6, t1',
    'addi a2, a0, 1',
    'srli a3, a2, 4',
]


def _run_chain(sim: CellSimulator, cache: dict) -> dict:
    values: dict = {'T0': 8, 'T1': 0x100, 'T2': 0x200, 'T3': 0x300,
                    'SP': _BASE_DATA + _DATA_SIZE // 2}
    taint: dict = {'T0': FULL_TAINT_64}
    for asm in TAINT_CHAIN:
        circuit = _circuit(asm, cache)
        ctx = EvalContext(input_values=values, input_taint=taint,
                          simulator=sim, implicit_policy=ImplicitTaintPolicy.KEEP)
        taint = circuit.evaluate(ctx)
    return taint


def test_chain_backends_agree(
    sim_unicorn: CellSimulator, sim_pcode: CellSimulator, circuit_cache: dict,
) -> None:
    final_u = _run_chain(sim_unicorn, circuit_cache)
    final_p = _run_chain(sim_pcode, circuit_cache)
    diffs = _diff_dicts(final_u, final_p)
    assert not diffs, '10-instr chain final state differs:\n' + '\n'.join(
        f'  {k}: u={hex(u)} p={hex(p)}' for k, (u, p) in diffs.items()
    )


# ===========================================================================
# PART 8 — Diagnostic dump and fallback rate (informational)
# ===========================================================================

def test_diagnostic_summary(
    sim_unicorn: CellSimulator, sim_pcode: CellSimulator, circuit_cache: dict,
    capsys: Any,
) -> None:
    """
    Report MATCH/UNSOUND/BACKEND_DIFF for each primitive on a generic taint.
    Always passes — informational only.
    """
    rows: list[tuple[str, str]] = []
    for asm in ARITHMETIC_PRIMITIVES:
        values = {'T1': 0x12345678ABCDEF01, 'T2': 0x10}
        taint = {'T1': FULL_TAINT_64}
        try:
            out_u = _run_microtaint(sim_unicorn, asm, values, taint, circuit_cache)
            out_p = _run_microtaint(sim_pcode,   asm, values, taint, circuit_cache)
            code = encode(asm)
            oracle = _true_taint_riscv(code, taint, values)
            unsound_bits = sum(
                bin((oracle.get(r, 0) & MASK64) & ~(out_u.get(r, 0) & MASK64)).count('1')
                for r in STATE_NAMES
            )
            backend_match = not _diff_dicts(out_u, out_p)
            if unsound_bits and backend_match:
                status = f'UNSOUND ({unsound_bits} bits)'
            elif unsound_bits and not backend_match:
                status = f'UNSOUND+DIFF ({unsound_bits} bits)'
            elif not backend_match:
                status = 'BACKEND_DIFF'
            else:
                status = 'OK'
            rows.append((asm, status))
        except Exception as exc:
            rows.append((asm, f'ERROR {type(exc).__name__}: {exc}'))

    with capsys.disabled():
        print('\n=== RV64 PRIMITIVE INSTRUCTION DIAGNOSTIC ===')
        for asm, status in rows:
            mark = '   ' if status == 'OK' else ' ! '
            print(f'  {mark}{asm:35s} {status}')
        n_ok = sum(1 for _, s in rows if s == 'OK')
        print(f'\n  {n_ok}/{len(rows)} primitives OK on this scenario')


def test_pcode_fallback_rate(
    sim_pcode: CellSimulator, circuit_cache: dict, capsys: Any,
) -> None:
    """Report the pcode→Unicorn fallback rate after running the primitive matrix."""
    for asm in ARITHMETIC_PRIMITIVES:
        try:
            _run_microtaint(sim_pcode, asm,
                            {'T1': 1, 'T2': 1}, {'T1': FULL_TAINT_64},
                            circuit_cache)
        except Exception:
            pass
    pcode = sim_pcode._pcode
    if pcode is None:
        pytest.skip('pcode evaluator inactive')
    rate = getattr(pcode, 'fallback_rate', None)
    if rate is None:
        pytest.skip('fallback_rate metric not exposed')
    with capsys.disabled():
        print(f'\n[INFO] RV64 P-code fallback rate over primitives: {rate:.1%}')
    assert 0.0 <= rate <= 1.0
