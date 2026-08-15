"""The single-call MAPPED rule must equal the 2-replica differential.

``engine.make_mapped_single_call`` replaces the differential ``f(V|T) XOR f(V&~T)``
with the value-independent ``L(T) = f(T) XOR f|_{x->0}`` for pure-routing MAPPED
slices. It emits that form under purely structural preconditions (single dynamic
register input, routing opcodes only, no control flow) -- there is no runtime
self-check in the engine. This test is where that structural guarantee is proved:
for every mapped instruction, the emitted rule's taint must equal the differential
(and therefore the exact bit-flip taint) on every random and corner (V, T).
"""
import random

import pytest

from microtaint.instrumentation.ast import EvalContext, InstructionCellExpr
from microtaint.simulator import CellSimulator, MachineState
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, Register

ARCH = Architecture.AMD64
REG_NAMES = ('RAX', 'RBX', 'RCX', 'RDX')
REGS = [Register(n, 64) for n in REG_NAMES]
M64 = (1 << 64) - 1

# Instructions whose data output is MAPPED (pure routing) -- these must fire the
# single-call form, and it must equal the differential.
MAPPED_CASES = {
    'mov eax,ebx': '89d8',
    'movzx eax,bl': '0fb6c3',
    'movsx eax,bl': '0fbec3',      # sign-extension fill
    'shl eax,4': 'c1e004',
    'shr eax,4': 'c1e804',
    'sar eax,4': 'c1f804',
    'bswap eax': '0fc8',
    'mov rax,rbx': '4889d8',
    'movsxd rax,ebx': '4863c3',
    'and eax,0x0f0f0f0f': '250f0f0f0f',    # affine, f0 == 0
    'or eax,0xf0f0f0f0': '0df0f0f0f0',     # affine with nonzero constant part f0 == c
}

# Non-affine / non-mapped controls -- must NOT fire the single-call form.
DIFFERENTIAL_CASES = {
    'add eax,ebx': '01d8',
    'sub eax,ebx': '29d8',
    'imul eax,ebx': '0fafc3',
    'shl eax,cl': 'd3e0',          # variable shift -> translatable
    'and eax,ebx': '21d8',         # two dynamic inputs -> monotonic
}


@pytest.fixture(scope='module')
def sim():
    return CellSimulator(ARCH, use_unicorn=False, use_c=True)


def _rax_expr(hexs):
    circ = generate_static_rule(ARCH, bytes.fromhex(hexs), REGS)
    asn = next((a for a in circ.assignments if getattr(a.target, 'name', None) == 'RAX'), None)
    assert asn is not None, f'no RAX output for {hexs}'
    return circ, asn.expression


def _cell_count(expr_repr):
    return expr_repr.count('InstructionCellExpr')


def _differential(sim, hexs, values, taint):
    """Reference: f(V|T) XOR f(V&~T) on RAX, computed independently of the rule."""
    ice = InstructionCellExpr(ARCH, hexs, 'RAX', 0, 63, {})
    hi = sim.evaluate_concrete(ice, MachineState(regs={r: (values[r] | taint[r]) & M64 for r in REG_NAMES}, mem={}))
    lo = sim.evaluate_concrete(ice, MachineState(regs={r: (values[r] & ~taint[r]) & M64 for r in REG_NAMES}, mem={}))
    return hi ^ lo


def _corners(rng):
    yield dict.fromkeys(REG_NAMES, M64)
    yield dict.fromkeys(REG_NAMES, 1)
    yield dict.fromkeys(REG_NAMES, 1 << 31)
    yield dict.fromkeys(REG_NAMES, 1 << 63)
    for _ in range(50):
        yield {r: rng.getrandbits(64) for r in REG_NAMES}


@pytest.mark.parametrize('label,hexs', MAPPED_CASES.items(), ids=MAPPED_CASES.keys())
def test_mapped_emits_single_call(label, hexs):
    _circ, expr = _rax_expr(hexs)
    assert _cell_count(repr(expr)) == 1, f'{label}: expected single-call (1 cell), got {repr(expr)[:120]}'


@pytest.mark.parametrize('label,hexs', MAPPED_CASES.items(), ids=MAPPED_CASES.keys())
def test_single_call_equals_differential(label, hexs, sim):
    circ, _ = _rax_expr(hexs)
    rng = random.Random(hash(hexs) & 0xFFFF)
    for taint in _corners(rng):
        values = {r: rng.getrandbits(64) for r in REG_NAMES}
        got = circ.evaluate(EvalContext(input_taint=taint, input_values=values, simulator=sim)).get('RAX', 0) & M64
        ref = _differential(sim, hexs, values, taint)
        assert got == ref, f'{label}: single-call {got:#x} != differential {ref:#x} (taint={taint["RAX"]:#x})'


@pytest.mark.parametrize('label,hexs', DIFFERENTIAL_CASES.items(), ids=DIFFERENTIAL_CASES.keys())
def test_non_mapped_keeps_differential(label, hexs):
    _circ, expr = _rax_expr(hexs)
    assert _cell_count(repr(expr)) != 1, f'{label}: should keep differential, not single-call'
