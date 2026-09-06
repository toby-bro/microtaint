"""Negative immediates / address offsets must stay on the compiled C fast path.

A negative immediate (``mov rax, -1``) or a negative memory displacement
(``mov rax, [rbp-0x10]``) lowers to a NEGATIVE p-code ``Constant``.  The
compiled circuit (circuit_c) stores constants as uint64, so historically the
compile gate rejected negatives (``PyLong_AsUnsignedLongLong`` fails) and the
whole assignment fell back to the slow Python AST evaluator -- for two of the
most common operand shapes in real code ([rbp-x] locals, -1 masks).

circuit_c now decodes a negative Constant as its two's-complement uint64
(``pylong_to_u64`` falls back to ``PyLong_AsLongLong``; the runtime load uses
``PyLong_AsUnsignedLongLongMask``), so these compile and run in C.  This test
pins BOTH properties:

  * the circuit compiles with ``python_fallback == 0`` (stays on the C fast
    path), and
  * the taint is bit-exact -- proving the negative constant is decoded as
    two's-complement, not garbage.

Bytes are pre-assembled (keystone-free), matching the instruction-bank
philosophy: the test needs no assembler.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'benchmark'))
from instruction_bank import isa_registers  # type: ignore[import-not-found]

from microtaint.instrumentation.ast import ChainedCircuit, EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, ImplicitTaintPolicy

FULL = 0xFFFFFFFFFFFFFFFF
_REGS = list(isa_registers('AMD64'))


class _Shadow:
    """Minimal byte-addressable taint shadow for the memory-operand forms."""

    def __init__(self) -> None:
        self._m: dict[int, int] = {}

    def read_mask(self, addr: int, size: int) -> int:
        return sum((self._m.get(addr + i, 0) & 0xFF) << (8 * i) for i in range(size))

    def write_mask(self, addr: int, size: int, mask: int) -> None:
        for i in range(size):
            self._m[addr + i] = (mask >> (8 * i)) & 0xFF


def _mem_reader(addr: int, size: int) -> int:  # noqa: ARG001 - concrete mem reads all-zero here
    return 0


def _eval(hexbytes: str, *, rax_tainted: bool) -> tuple[int, int, bool]:
    """Return (RAX out-taint, max python_fallback over subcircuits, all-compiled)."""
    sim = CellSimulator(Architecture.AMD64)
    circ = generate_static_rule(Architecture.AMD64, bytes.fromhex(hexbytes), _REGS)
    values = {r.name: 0x2000 for r in _REGS}
    taint = {r.name: 0 for r in _REGS}
    if rax_tainted:
        taint['RAX'] = FULL
    ctx = EvalContext(
        input_values=values, input_taint=taint, simulator=sim,
        implicit_policy=ImplicitTaintPolicy.KEEP,
        shadow_memory=_Shadow(), mem_reader=_mem_reader,
    )
    out = circ.evaluate(ctx)
    subs = list(circ.sub_circuits) if isinstance(circ, ChainedCircuit) else [circ]
    all_compiled = all(getattr(s, '_compiled', None) not in (None, False) for s in subs)
    pyfb = 0
    for s in subs:
        comp = getattr(s, '_compiled', None)
        if comp not in (None, False):
            pyfb = max(pyfb, comp.stats().get('python_fallback', 0))
    return out.get('RAX', 0), pyfb, all_compiled


# (bytes, asm) -- negative immediates + negative-displacement memory operands.
_FASTPATH_FORMS = [
    ('4883e0f0', 'and rax, -16'),
    ('480d00ffffff', 'or rax, -256'),
    ('48c7c0ffffffff', 'mov rax, -1'),
    ('4883c0f8', 'add rax, -8'),
    ('4883f0ff', 'xor rax, -1'),
    ('488b45f0', 'mov rax, [rbp-0x10]'),
    ('48034424f8', 'add rax, [rsp-8]'),
    ('483b45e0', 'cmp rax, [rbp-0x20]'),
    ('48334424e8', 'xor rax, [rsp-0x18]'),
    ('488945f0', 'mov [rbp-0x10], rax'),
]


@pytest.mark.parametrize(('hexbytes', 'asm'), _FASTPATH_FORMS)
def test_negative_constant_stays_on_c_fast_path(hexbytes: str, asm: str) -> None:
    _taint, pyfb, compiled = _eval(hexbytes, rax_tainted=True)
    assert compiled, f'{asm}: circuit did not compile (fell back to the Python walker)'
    assert pyfb == 0, f'{asm}: {pyfb} assignment(s) fell back to Python (negative constant not compiled)'


# (bytes, asm, expected RAX out-taint with RAX fully tainted, rest clean).
# These pin two's-complement decoding: a garbage-decoded constant changes which
# output bits survive.
_TAINT_ORACLES = [
    ('4883e0f0', 'and rax, -16', 0xFFFFFFFFFFFFFFF0),   # low 4 bits ANDed to 0 -> clean
    ('480d00ffffff', 'or rax, -256', 0x00000000000000FF),  # high 56 bits ORed to 1 -> clean
    ('48c7c0ffffffff', 'mov rax, -1', 0x0000000000000000),  # constant load -> no taint
    ('4883c0f8', 'add rax, -8', 0xFFFFFFFFFFFFFFFF),    # add carries taint across the word
    ('4883f0ff', 'xor rax, -1', 0xFFFFFFFFFFFFFFFF),    # NOT preserves taint
]


@pytest.mark.parametrize(('hexbytes', 'asm', 'expected'), _TAINT_ORACLES)
def test_negative_immediate_taint_is_bit_exact(hexbytes: str, asm: str, expected: int) -> None:
    taint, _pyfb, _compiled = _eval(hexbytes, rax_tainted=True)
    assert taint == expected, f'{asm}: RAX taint 0x{taint:016x} != oracle 0x{expected:016x}'
