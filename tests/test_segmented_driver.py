"""M4 (core-materialization half): the MICROTAINT_SEGMENTED driver path emits a
cut result's taint as an ``is_intermediate`` assignment, while leaving the visible
OUTPUT (non-intermediate) assignments byte-identical to the default path.

This pins the partition -> UNIQ-target -> two-phase-assembly plumbing before the
downstream rewrite consumes the intermediates.  Runtime safety and output
equivalence under evaluation are covered by running the circuit-level suites with
MICROTAINT_SEGMENTED=1 (they pass unchanged).
"""

# mypy: disable-error-code="no-untyped-def,no-untyped-call,attr-defined,union-attr"

from __future__ import annotations

import pytest

import microtaint.sleigh.engine as engine
from microtaint.emulator.wrapper import X64_FORMAT
from microtaint.types import Architecture

keystone = pytest.importorskip('keystone')

_KS = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
ARCH = Architecture.AMD64


def _build(asm: str, segmented: bool):
    engine._SEGMENTED = segmented
    engine._cached_generate_static_rule.cache_clear()
    code = bytes(_KS.asm(asm, 0x1000)[0])
    return engine.generate_static_rule(ARCH, code, X64_FORMAT)


def _tname(a):
    return a.target.name if hasattr(a.target, 'name') else str(a.target)


def _sig(assigns):
    return sorted(
        f'{_tname(a)}[{getattr(a.target, "bit_start", "?")}:'
        f'{getattr(a.target, "bit_end", "?")}]={a.expression}'
        for a in assigns
    )


CORPUS = [
    'cmp rax, rbx', 'sub rax, rbx', 'add rax, rbx', 'and rax, rbx', 'xor rax, rbx',
    'or rax, rbx', 'test rax, rbx', 'mov rax, rbx', 'sub rax, 5', 'cmp eax, ebx',
    'inc rax', 'neg rax', 'shl rax, 4', 'imul rax, rbx', 'adc rax, rbx', 'sbb rax, rbx',
]


@pytest.fixture(autouse=True)
def _restore_gate():
    saved = engine._SEGMENTED
    yield
    engine._SEGMENTED = saved
    engine._cached_generate_static_rule.cache_clear()


def test_output_assignments_byte_identical_with_gate():
    """Gate on must not change any non-intermediate (output) assignment."""
    for asm in CORPUS:
        off = _build(asm, False)
        on = _build(asm, True)
        reg_on = [a for a in on.assignments if not getattr(a, 'is_intermediate', False)]
        assert _sig(off.assignments) == _sig(reg_on), f'output diverged for {asm!r}'


def test_intermediates_emitted_for_flag_producers():
    """cmp/sub cut at the subtraction result, so the gate adds an intermediate."""
    for asm in ('cmp rax, rbx', 'sub rax, rbx'):
        on = _build(asm, True)
        inter = [a for a in on.assignments if getattr(a, 'is_intermediate', False)]
        assert inter, f'expected an intermediate for {asm!r}'
        assert all(_tname(a).startswith('UNIQ_') for a in inter)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
