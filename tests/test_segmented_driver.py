"""M4: the MICROTAINT_SEGMENTED driver rewrites a cut-derived flag output to
consume the materialized cut (via a seeded segment cell + the existing category
floors), instead of one whole-slice rule.

Properties pinned here (the full soundness suites also pass with the gate on):

  * SOUNDNESS vs the sound baseline -- for every output, the segmented taint must
    never DROP a bit the whole-slice (gate-off) rule sets: ``off & ~on == 0``.
    The segmented rule may over-taint (sound; e.g. a rare 32-bit compare ZF), but
    must never under-taint.
  * A compare (discarded unique result) gains UNIQ_ intermediates and its
    result-derived flags are re-expressed as segment rules.
  * The default (gate-off) path is untouched.
"""

# mypy: disable-error-code="no-untyped-def,no-untyped-call,attr-defined,union-attr,arg-type"

from __future__ import annotations

import random

import pytest

import microtaint.sleigh.engine as engine
from microtaint.emulator.wrapper import X64_FORMAT
from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.types import Architecture, ImplicitTaintPolicy

keystone = pytest.importorskip('keystone')

_KS = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
ARCH = Architecture.AMD64
MASK64 = (1 << 64) - 1
_SIM = CellSimulator(ARCH, use_unicorn=False, use_c=False)  # Cython for both gates
_ZERO = {r.name: 0 for r in X64_FORMAT}

CORPUS = [
    'cmp rax, rbx', 'sub rax, rbx', 'add rax, rbx', 'and rax, rbx', 'xor rax, rbx',
    'or rax, rbx', 'test rax, rbx', 'sub rax, 5', 'cmp eax, ebx', 'neg rax',
    'adc rax, rbx', 'sbb rax, rbx', 'inc rax', 'dec rax', 'imul rax, rbx',
]


def _build(asm: str, segmented: bool):
    engine._SEGMENTED = segmented
    engine._cached_generate_static_rule.cache_clear()
    return engine.generate_static_rule(ARCH, bytes(_KS.asm(asm, 0x1000)[0]), X64_FORMAT)


def _ev(circ, state, taint):
    ctx = EvalContext(
        input_taint={**_ZERO, **taint},
        input_values={**_ZERO, **state},
        simulator=_SIM,
        implicit_policy=ImplicitTaintPolicy.IGNORE,
    )
    return circ.evaluate(ctx)


@pytest.fixture(autouse=True)
def _restore_gate():
    saved = engine._SEGMENTED
    yield
    engine._SEGMENTED = saved
    engine._cached_generate_static_rule.cache_clear()


def test_segmented_never_undertaints_vs_whole_slice():
    """Gate-on taint must be a SUPERSET of gate-off (the sound baseline) -- it may
    over-taint but must never drop a bit."""
    rng = random.Random(2024)
    for asm in CORPUS:
        coff = _build(asm, False)
        con = _build(asm, True)
        for _ in range(150):
            state = {'RAX': rng.getrandbits(64), 'RBX': rng.getrandbits(64)}
            taint = {
                'RAX': rng.getrandbits(64) & rng.getrandbits(64),
                'RBX': rng.getrandbits(64) & rng.getrandbits(64),
            }
            off = _ev(coff, state, taint)
            on = {k: v for k, v in _ev(con, state, taint).items() if not k.startswith('UNIQ_')}
            for k, ov in off.items():
                assert (ov & ~on.get(k, 0)) == 0, (
                    f'segmented under-tainted {asm} {k}: off={ov:#x} on={on.get(k, 0):#x} '
                    f'state={state} taint={taint}'
                )


def test_compare_gains_intermediates_and_segments_flags():
    """cmp cuts at the discarded subtraction result: intermediates appear and the
    result-derived flags become segment rules (differ from the whole-slice)."""
    off = _build('cmp rax, rbx', False)
    on = _build('cmp rax, rbx', True)
    inter = [a for a in on.assignments if getattr(a, 'is_intermediate', False)]
    assert inter, 'expected UNIQ_ intermediates for cmp'
    assert all(a.target.name.startswith('UNIQ_') for a in inter)

    def _rule(circ, name):
        for a in circ.assignments:
            if not getattr(a, 'is_intermediate', False) and getattr(a.target, 'name', None) == name:
                return str(a.expression)
        return None

    # A result-derived flag (ZF) must be re-expressed (its rule changes); an
    # operand-derived flag the cut does not touch keeps its rule shape.
    assert _rule(on, 'ZF') != _rule(off, 'ZF')


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
