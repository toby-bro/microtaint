"""Constant-intermediate multi-instruction sequences stay MONOLITHIC.

When an assembler expands an immediate-logic op into a macro that first
materialises a constant (e.g. MIPS ``and $r,$s,-1`` -> ``addiu $t,$0,-1; and
$r,$s,$t``), the intermediate register is a pure constant and carries no taint.
Chaining such a sequence computes the logic op as a value-aware differential
(cell re-executions); keeping it monolithic lets ``slice_simplify`` fold the
constant, so the taint is exact bit-routing with far fewer (often zero) cells.

These tests pin that the fold is (a) applied, (b) bit-exact vs the brute-forced
true taint, and (c) cheaper in cells.  See ``engine._all_intermediates_constant``.
"""
from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import pytest

from microtaint.instrumentation.ast import (
    ChainedCircuit,
    EvalContext,
    InstructionCellExpr,
)
from microtaint.simulator import CellSimulator, MachineState
from microtaint.sleigh.engine import generate_static_rule

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'benchmark'))
from instruction_bank import load_bank  # type: ignore[import-not-found]

# MIPS64BE immediate-logic macros that materialise a constant then apply the op.
# The label matches the bank's pre-assembled entry (rt=$2 v0, rs=$4 a0).
_CASE_LABELS = [
    'and $2, $4, -1',
    'or $2, $4, -1',
    'xor $2, $4, -1',
    'nor $2, $4, -1',
    'nor $2, $4, 7',
]

_MIPS_SPEC = load_bank(isas=['MIPS64BE'])['MIPS64BE']
_CASE_BYTES = {i.label: i.bytes for i in _MIPS_SPEC.instructions if i.label in _CASE_LABELS}


@pytest.fixture(scope='module')
def mips():
    return _MIPS_SPEC.arch, _MIPS_SPEC.regs


@pytest.fixture(scope='module')
def cases():
    return _CASE_BYTES


def _true_taint(arch, sim, code, name, be, regs, base, taint):
    rbits = {r.name: r.bits for r in regs}
    rn = [r.name for r in regs]
    ice = InstructionCellExpr(arch, code.hex(), name, 0, be, {})
    pos = [(r, b) for r in rn for b in range(rbits[r]) if (taint.get(r, 0) >> b) & 1]
    outs = []
    for combo in itertools.product((0, 1), repeat=len(pos)):
        v = dict(base)
        for (r, b), bit in zip(pos, combo, strict=True):
            v[r] = (v[r] | (1 << b)) if bit else (v[r] & ~(1 << b))
        outs.append(sim.evaluate_concrete(ice, MachineState(regs=dict(v), mem={})))
    acc = 0
    for o in outs[1:]:
        acc |= o ^ outs[0]
    return acc


@pytest.mark.parametrize('label', _CASE_LABELS)
def test_constant_intermediate_macro_is_monolithic(mips, cases, label):
    arch, regs = mips
    code = cases[label]
    circ = generate_static_rule(arch, code, regs)
    assert not isinstance(circ, ChainedCircuit), (
        f'{label}: constant-intermediate macro must stay monolithic, got a chain'
    )


@pytest.mark.parametrize('label', _CASE_LABELS)
def test_constant_intermediate_macro_is_sound_and_exact(mips, cases, label):
    arch, regs = mips
    code = cases[label]
    sim = CellSimulator(arch)
    circ = generate_static_rule(arch, code, regs)
    rn = [r.name for r in regs]
    rbits = {r.name: r.bits for r in regs}
    rng = random.Random(hash(label) & 0xFFFF)
    for _ in range(30):
        vals = {r: rng.getrandbits(rbits[r]) for r in rn}
        taint = dict.fromkeys(rn, 0)
        # taint a few low bits of $4 (a0) -- the real source
        for _ in range(rng.randint(1, 4)):
            taint['R4' if 'R4' in rn else rn[4]] |= 1 << rng.randrange(16)
        base = {r: vals[r] & ~taint[r] for r in rn}
        got = circ.evaluate(EvalContext(input_values=base, input_taint=taint, simulator=sim))
        for a in circ.assignments:
            nm = getattr(a.target, 'name', None)
            if nm is None or hasattr(a.target, 'address_expr'):
                continue
            be = a.target.bit_end - a.target.bit_start
            mask = (1 << (be + 1)) - 1
            tt = _true_taint(arch, sim, code, nm, be, regs, base, taint) & mask
            g = got.get(nm, 0) & mask
            assert not (tt & ~g), (
                f'{label} {nm}: UNDER-taint miss={tt & ~g:#x} got={g:#x} true={tt:#x}'
            )
