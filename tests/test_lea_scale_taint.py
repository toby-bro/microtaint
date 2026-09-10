"""`lea r, [b + b*s]` is an addition, and its carries must propagate.

The address form `[base + index*scale]` with base == index computes `b * (1+s)`,
which the adder builds as `b + (b << log2 s)`. Two shifted copies of the same
value are added, so a tainted bit at position i lands at i and at i+log2 s, and
the carry out of either can reach every position above it. A rule that unions the
two shifted positions and stops there misses the carry chain between them.

Found by the RQ2 corpus at seed 2026, `lea rax, [rbx + rbx*4]` with RBX bits 44
and 45 tainted. No oracle is needed to see it -- the arithmetic settles it:

    base = 0xfe31c4128b39effd            (RBX with the tainted bits cleared)
    base*5                    = 0xf6f8d45cb821aff1
    (base | 1<<44)*5  differs in 0x1f00000000000   bits 44,45,46,47,48
    (base | 1<<45)*5  differs in 0x1a00000000000   bits 45,47,48

so the union of what a single tainted bit can move is bits 44..48, and the engine
returned bits 44, 45 and 48 only. Bits 46 and 47 are a silent under-taint.

v0.6.9 answered with the full mask -- over-tainted but sound. The narrowing
landed between v0.6.9 and v0.6.10.
"""

from __future__ import annotations

import pytest

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, Register

_GP = ['RAX', 'RBX', 'RCX', 'RDX', 'RSI', 'RDI', 'RSP', 'RBP'] + [f'R{n}' for n in range(8, 16)]
_REGS = [Register(n, 64) for n in _GP]
_MASK64 = (1 << 64) - 1

# lea rax, [rbx + rbx*4] ; lea rax, [rbx + rbx*8] ; lea rax, [rbx + rbx*2]
_CASES = {
    '488d049b': 5,   # scale 4
    '488d04db': 9,   # scale 8
    '488d045b': 3,   # scale 2
}
_RBX = 0xFE31C4128B39EFFD  # the seed-2026 value, tainted bits already cleared


def _engine(code_hex: str, rbx: int, taint: int) -> int:
    circuit = generate_static_rule(Architecture.AMD64, bytes.fromhex(code_hex), _REGS)
    vals = {r.name: 0 for r in _REGS}
    vals['RBX'] = rbx
    vals['RSP'] = 0x80000000
    taints = {r.name: 0 for r in _REGS}
    taints['RBX'] = taint
    ctx = EvalContext(input_values=vals, input_taint=taints, simulator=CellSimulator(Architecture.AMD64))
    out = circuit.evaluate(ctx)
    return int(out.get('RAX', 0)) & _MASK64


def _truth(multiplier: int, rbx: int, taint: int) -> int:
    """Bits of `rbx * multiplier` that any assignment of the tainted bits moves."""
    base = rbx & ~taint & _MASK64
    bits = [b for b in range(64) if (taint >> b) & 1]
    ref = (base * multiplier) & _MASK64
    moved = 0
    for mask in range(1 << len(bits)):
        v = base
        for idx, b in enumerate(bits):
            if (mask >> idx) & 1:
                v |= 1 << b
        moved |= ((v * multiplier) & _MASK64) ^ ref
    return moved


@pytest.mark.xfail(strict=True, reason='lea [b+b*s] unions the two shifted positions and drops the carry between them')
def test_lea_self_scaled_carries() -> None:
    """Every bit the multiplication actually moves must be tainted."""
    missing = {}
    for code, mult in _CASES.items():
        for taint in (0x300000000000, 1 << 44, 1 << 3, 0x9):
            truth = _truth(mult, _RBX, taint)
            got = _engine(code, _RBX, taint)
            if truth & ~got:
                missing[f'{code} x{mult} taint={taint:#x}'] = hex(truth & ~got)
    assert not missing, f'under-tainted: {missing}'
