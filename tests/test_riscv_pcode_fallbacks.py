"""
test_riscv_pcode_fallbacks.py
=============================

Identifies which RISC-V instructions the pcode-native evaluator cannot handle
and must hand off to Unicorn.

The predecode step in ``_predecode_ops`` sets ``has_fallback = True`` when it
encounters an op that cannot be evaluated in the pcode C/Cython engine:
  - BRANCHIND / CALLIND  (indirect branches — target is data-dependent)
  - CALLOTHER with output
  - FLOAT_ANY / TRUNC_FLOAT
  - UNKNOWN opcodes

This test probes every instruction in the test suite's canonical instruction
set via ``_get_decoded(...).has_fallback`` — a pure predecode check that costs
nothing at runtime and is deterministic regardless of register values.

Run:
    pytest tests/test_riscv_pcode_fallbacks.py -v
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, 'tests')   # riscv_encoder lives in tests/
from riscv_encoder import encode  # noqa: E402

from microtaint.instrumentation.cell import _get_decoded
from microtaint.types import Architecture

# ---------------------------------------------------------------------------
# Canonical instruction set (mirrors test_riscv_microtaint.py exactly)
# ---------------------------------------------------------------------------

_RV64I_R = [
    'add t0, t1, t2', 'sub t0, t1, t2',
    'sll t0, t1, t2', 'srl t0, t1, t2', 'sra t0, t1, t2',
    'and t0, t1, t2', 'or  t0, t1, t2', 'xor t0, t1, t2',
    'slt t0, t1, t2', 'sltu t0, t1, t2',
]

_RV64I_W = [
    'addw t0, t1, t2', 'subw t0, t1, t2',
    'sllw t0, t1, t2', 'srlw t0, t1, t2', 'sraw t0, t1, t2',
]

_RV64M_R = [
    'mul t0, t1, t2', 'mulh t0, t1, t2', 'mulhsu t0, t1, t2', 'mulhu t0, t1, t2',
    'div t0, t1, t2', 'divu t0, t1, t2', 'rem t0, t1, t2', 'remu t0, t1, t2',
]

_RV64M_W = [
    'mulw t0, t1, t2', 'divw t0, t1, t2', 'divuw t0, t1, t2',
    'remw t0, t1, t2', 'remuw t0, t1, t2',
]

_IMM_TEMPLATES: list[tuple[str, list[int]]] = [
    ('addi t0, t1, {imm}',  [0, 1, -1, 100, -100, 2047, -2048]),
    ('xori t0, t1, {imm}',  [0, -1, 0xFF, -2048]),
    ('ori  t0, t1, {imm}',  [0, -1, 0xFF, 0x555]),
    ('andi t0, t1, {imm}',  [0, -1, 0xFF, 0x555, 0x800]),
    ('slti t0, t1, {imm}',  [0, 1, -1, 100, -100]),
    ('sltiu t0, t1, {imm}', [0, 1, -1, 100, 2047]),
    ('addiw t0, t1, {imm}', [0, 1, -1, 100, 2047, -2048]),
]

_SHIFTI_TEMPLATES: list[tuple[str, list[int]]] = [
    ('slli t0, t1, {imm}',  [0, 1, 4, 31, 32, 63]),
    ('srli t0, t1, {imm}',  [0, 1, 4, 31, 32, 63]),
    ('srai t0, t1, {imm}',  [0, 1, 4, 31, 32, 63]),
    ('slliw t0, t1, {imm}', [0, 1, 4, 15, 31]),
    ('srliw t0, t1, {imm}', [0, 1, 4, 15, 31]),
    ('sraiw t0, t1, {imm}', [0, 1, 4, 15, 31]),
]

_U_TYPE = ['lui t0, 0x12345', 'auipc t0, 0x12345']

_LOADS = [
    'lb t0, 8(t1)', 'lh t0, 8(t1)', 'lw t0, 8(t1)', 'ld t0, 8(t1)',
    'lbu t0, 8(t1)', 'lhu t0, 8(t1)', 'lwu t0, 8(t1)',
]

_STORES = ['sb t2, 8(t1)', 'sh t2, 8(t1)', 'sw t2, 8(t1)', 'sd t2, 8(t1)']

_BRANCHES = [
    'beq t1, t2, 4', 'bne t1, t2, 4',
    'blt t1, t2, 4', 'bge t1, t2, 4',
    'bltu t1, t2, 4', 'bgeu t1, t2, 4',
    'jal t0, 4',
    'jalr t0, t1, 0',
]

_SYSTEM = ['nop', 'fence']


def _expand(templates: list[tuple[str, list[int]]]) -> list[str]:
    return [tmpl.format(imm=imm) for tmpl, imms in templates for imm in imms]


ALL_INSTRUCTIONS: list[str] = (
    _RV64I_R + _RV64I_W
    + _RV64M_R + _RV64M_W
    + _expand(_IMM_TEMPLATES)
    + _expand(_SHIFTI_TEMPLATES)
    + _U_TYPE
    + _LOADS + _STORES
    + _BRANCHES + _SYSTEM
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _has_fallback(asm: str) -> bool:
    """Return True if the pcode predecoder marks this instruction as requiring Unicorn."""
    return bool(_get_decoded(Architecture.RISCV64, encode(asm)).has_fallback)


# ---------------------------------------------------------------------------
# Parametrised test: every instruction that should be native must be native
# ---------------------------------------------------------------------------

# Instructions known to require Unicorn due to indirect control flow.
# jalr lifts to CALLIND (indirect call to a data-dependent target), which
# the pcode engine cannot resolve without executing concretely in Unicorn.
_KNOWN_UNICORN_FALLBACKS: frozenset[str] = frozenset({
    'jalr t0, t1, 0',
})

_KNOWN_NATIVE: list[str] = [i for i in ALL_INSTRUCTIONS if i not in _KNOWN_UNICORN_FALLBACKS]


@pytest.mark.parametrize('asm', _KNOWN_NATIVE, ids=lambda s: s.replace(' ', '_'))
def test_instruction_is_native(asm: str) -> None:
    """Instruction must be handled by the pcode engine without Unicorn fallback."""
    assert not _has_fallback(asm), (
        f'"{asm}" unexpectedly requires Unicorn fallback.  '
        'If this is a new regression, add it to _KNOWN_UNICORN_FALLBACKS '
        'and open a bug.'
    )


@pytest.mark.parametrize('asm', sorted(_KNOWN_UNICORN_FALLBACKS), ids=lambda s: s.replace(' ', '_'))
def test_instruction_requires_unicorn(asm: str) -> None:
    """Instructions in the known-fallback set must still require Unicorn.

    If this test starts failing it means the pcode engine gained native
    support for this instruction — remove it from _KNOWN_UNICORN_FALLBACKS.
    """
    assert _has_fallback(asm), (
        f'"{asm}" no longer requires Unicorn fallback — pcode engine now handles it.  '
        'Remove it from _KNOWN_UNICORN_FALLBACKS.'
    )


# ---------------------------------------------------------------------------
# Summary test: enumerate all fallbacks and print a human-readable report
# ---------------------------------------------------------------------------

def test_fallback_summary(capsys: pytest.CaptureFixture) -> None:  # type: ignore[type-arg]
    """Print a complete fallback/native breakdown for all canonical instructions.

    Always passes — informational only.  Run with -s to see the output.
    """
    fallback_instrs = [i for i in ALL_INSTRUCTIONS if _has_fallback(i)]
    native_instrs   = [i for i in ALL_INSTRUCTIONS if not _has_fallback(i)]

    with capsys.disabled():
        print(f'\n=== RV64GC pcode fallback report ({len(ALL_INSTRUCTIONS)} instructions) ===')
        print()
        if fallback_instrs:
            print(f'  Unicorn fallback ({len(fallback_instrs)}):')
            for asm in fallback_instrs:
                print(f'    {asm}')
        else:
            print('  Unicorn fallback: none')
        print()
        print(f'  Pcode-native ({len(native_instrs)}): {len(native_instrs)}/{len(ALL_INSTRUCTIONS)}')
        print()
        # Group native instructions by mnemonic family for readability
        for group, instrs in [
            ('RV64I-R', _RV64I_R), ('RV64I-W', _RV64I_W),
            ('RV64M-R', _RV64M_R), ('RV64M-W', _RV64M_W),
            ('IMM',     _expand(_IMM_TEMPLATES)),
            ('SHIFTI',  _expand(_SHIFTI_TEMPLATES)),
            ('U-type',  _U_TYPE),
            ('Loads',   _LOADS), ('Stores',  _STORES),
            ('Branches',_BRANCHES), ('System', _SYSTEM),
        ]:
            n_native = sum(1 for i in instrs if not _has_fallback(i))
            marker = '' if n_native == len(instrs) else f'  ← {len(instrs)-n_native} fallback(s)'
            print(f'    {group:12s}: {n_native:3d}/{len(instrs):3d} native{marker}')
