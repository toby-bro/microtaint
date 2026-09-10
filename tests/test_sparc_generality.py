"""ISA-generality acceptance test: SPARC32BE with ZERO engine changes.

SPARC32BE is in the Architecture enum and the Unicorn map but has never been in
the instruction bank or any test.  This test proves the taint engine (rule
generation + the production compiled C fast path) handles it correctly with NO
engine modification: the only SPARC-specific code here is a human-facing register
list derived from the SLEIGH spec via _build_reg_maps.

Bit-exactness of the compiled path vs the Cython walker for SPARC was separately
confirmed 0-mismatch over these cases (scratchpad harness, two processes; an
in-process toggle is unreliable because rule objects cache their compiled form).

If a future change breaks ISA-generality, this test fails.

Run: uv run pytest tests/test_sparc_generality.py -v
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from microtaint.instrumentation.ast import EvalContext
from microtaint.instrumentation.cell import _build_reg_maps
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, ImplicitTaintPolicy, Register

ks = pytest.importorskip('keystone')

if TYPE_CHECKING:
    import keystone

FULL32 = 0xFFFFFFFF


def _sparc_regs() -> list[Register]:
    """State format derived from the SLEIGH register map (the only SPARC-specific
    addition; a test helper, not engine code)."""
    _offs, sizes = _build_reg_maps(Architecture.SPARC32BE)
    names = [f'{b}{i}' for b in ('G', 'O', 'L', 'I') for i in range(8)]
    regs = [Register(n, sizes[n] * 8) for n in names if n in sizes]
    for flg in ('CF', 'VF', 'ZF', 'NF', 'Y'):
        if flg in sizes:
            regs.append(Register(flg, sizes[flg] * 8))
    return regs


# asm -> (dest reg, expected exact taint, or None to require nonzero/value-aware)
CASES: list[tuple[str, str, int | None]] = [
    ('add %g1, %g2, %g3', 'G3', FULL32),      # g1 fully tainted -> carry fully taints g3
    ('xor %g1, %g2, %g3', 'G3', FULL32),      # xor is transparent to taint
    ('sub %g1, %g2, %g3', 'G3', FULL32),
    ('or %g1, %g2, %g3', 'G3', None),         # value-aware partial
    ('and %g1, %g2, %g3', 'G3', None),
    ('andn %g1, %g2, %g3', 'G3', None),
    ('sll %g1, %g2, %g3', 'G3', None),
    ('srl %g1, %g2, %g3', 'G3', None),
    ('add %g1, 5, %g3', 'G3', FULL32),        # immediate
    ('xor %g1, -1, %g3', 'G3', FULL32),       # negative immediate
    ('addcc %g1, %g2, %g3', 'G3', FULL32),    # also sets condition codes
    ('umul %g1, %g2, %g3', 'G3', None),
]


@pytest.fixture(scope='module')
def sparc_regs() -> list[Register]:
    return _sparc_regs()


@pytest.fixture(scope='module')
def ks_engine() -> keystone.Ks:
    return ks.Ks(ks.KS_ARCH_SPARC, ks.KS_MODE_SPARC32 | ks.KS_MODE_BIG_ENDIAN)


@pytest.fixture(scope='module')
def sim() -> CellSimulator:
    return CellSimulator(Architecture.SPARC32BE)


@pytest.mark.parametrize(('asm', 'dest', 'expected'), CASES)
def test_sparc_taint_and_generality(asm: str,
                                    dest: str,
                                    expected: int | None,
                                    sparc_regs: list[Register],
                                    ks_engine: keystone.Ks,
                                    sim: CellSimulator) -> None:
    bs = bytes(ks_engine.asm(asm, 0)[0])
    # Rule generation must succeed with zero engine changes.
    circ = generate_static_rule(Architecture.SPARC32BE, bs, sparc_regs)
    assert circ.assignments, f'no rule generated for {asm}'

    vals = {r.name: 0x12345670 + 7 * i for i, r in enumerate(sparc_regs)}
    taint = {'G1': 0xFFFFFFFFFFFFFFFF, 'G2': 0x0000FFFF}
    ectx = EvalContext(input_values=vals, input_taint=taint, simulator=sim,
                      implicit_policy=ImplicitTaintPolicy.KEEP)
    res = {k: v for k, v in circ.evaluate(ectx).items() if v}

    # Destination must carry taint (no silent under-taint of a tainted input).
    assert dest in res, f'{asm}: {dest} untainted, got {res}'
    assert res[dest] != 0, f'{asm}: {dest} untainted, got {res}'
    if expected is not None:
        assert res[dest] == expected, f'{asm}: {dest} taint {res[dest]:#x} != {expected:#x}'
    else:
        # value-aware ops: taint present and within the 32-bit dest width.
        assert 0 < res[dest] <= FULL32, f'{asm}: {dest} taint {res[dest]:#x} out of range'
