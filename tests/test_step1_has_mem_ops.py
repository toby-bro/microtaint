"""
Regression test for step 1:
    has_mem_ops contract violation for Python-fallback assignments.

Bug: when `circuit_c.compile_circuit` encounters an unsupported expression
class (e.g. MemoryDifferentialExpr), the whole assignment falls back to
Python and `cc->has_mem_ops` is never set, even if the expression tree
contains memory-reading leaves (`MemoryOperand`, etc).  The wrapper's
per-instruction cache uses `has_mem_ops` to decide whether the result
is safe to memoize.  False-negative `has_mem_ops` → stale taint replay.

We probe the failing instruction directly:
    `mov -0x28(%rbp), %rax`   (encoding 48 8b 45 d8)

This instruction reads memory.  The compiled circuit's `has_mem_ops`
attribute MUST be true.
"""
import pytest

from microtaint.sleigh.engine import _cached_generate_static_rule
from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.types import Architecture, Register


X86_64_STATE = tuple(
    (n, 64)
    for n in (
        'RAX', 'RBX', 'RCX', 'RDX', 'RSI', 'RDI', 'RSP', 'RBP',
        'R8', 'R9', 'R10', 'R11', 'R12', 'R13', 'R14', 'R15',
    )
)


def _compile(bs_hex: str):
    """Build a LogicCircuit, trigger lazy compile-to-C, return the compiled object."""
    _cached_generate_static_rule.cache_clear()
    c = _cached_generate_static_rule(
        Architecture.AMD64, bytes.fromhex(bs_hex), X86_64_STATE,
    )
    # First evaluate() triggers compile_circuit() inside ast.pyx
    sim = CellSimulator(Architecture.AMD64, use_c=True)
    regs = [Register(name=n, bits=b) for n, b in X86_64_STATE]
    iv = {r.name: 0 for r in regs}
    it = {r.name: 0 for r in regs}
    ctx = EvalContext(input_values=iv, input_taint=it, simulator=sim)
    c.evaluate(ctx)
    return c._compiled


# The set below contains AMD64 instructions that should each have
# `has_mem_ops == 1` because they read or write memory AND emit at least
# one taint assignment.  These all triggered the false-negative on the
# unpatched build.
#
# `cmp mem, reg` is excluded: although it reads memory, its rule has zero
# assignments (no taint output) — the cache replays an empty output, which
# is harmless regardless of `has_mem_ops`.
MEMORY_INSTRUCTIONS = [
    # mov mem -> reg  (the canonical failing case in min_loop)
    ('488b45d8', 'mov -0x28(%rbp), %rax'),
    ('488b45f8', 'mov -0x8(%rbp), %rax'),
]


@pytest.mark.parametrize('bs_hex,desc', MEMORY_INSTRUCTIONS)
def test_has_mem_ops_set_for_memory_reading_instruction(bs_hex, desc):
    compiled = _compile(bs_hex)
    assert compiled is not None and compiled is not False, (
        f'instruction {bs_hex} ({desc}) failed to compile to C VM'
    )
    assert compiled.has_mem_ops == 1, (
        f'has_mem_ops must be 1 for {bs_hex} ({desc}); '
        f'this instruction reads memory and its cached taint output '
        f'depends on shadow memory contents, so it must NOT be eligible '
        f'for the wrapper`s per-instruction cache.  has_mem_ops==0 would '
        f'let the cache replay stale taint across iterations.'
    )


def test_has_mem_ops_zero_for_pure_register_instruction():
    """Sanity: pure-register instructions correctly report has_mem_ops=0."""
    # movzbl %al, %edx — no memory access
    compiled = _compile('0fb6d0')
    assert compiled.has_mem_ops == 0, (
        'pure-register movzbl wrongly flagged as having memory ops'
    )
    # xor %rdx, %rax — no memory access
    compiled = _compile('4831d0')
    assert compiled.has_mem_ops == 0
