"""`precompile` must leave nothing for the first `evaluate` to pay for.

Two regressions are guarded here, both found on 2026-09-11 while explaining the
RQ2 p100.

1.  **The decode cache was not warmed.**  `precompile` built the CompiledCircuit
    but the first `evaluate` still populated ``cell._get_decoded`` -- the SLEIGH
    translation plus the DecodedOps struct build.  That is the larger half of the
    first-call cost: `shrd rax,rbx,cl` measured 467 us on its first evaluate
    against 8.4 us steady.  With the RQ2 corpus holding 9,858 cases over only 426
    distinct byte-strings, those 426 cold decodes WERE the reported p99 and p100,
    so the published tail described the corpus rather than the engine.

2.  **ChainedCircuit had no `precompile` at all.**  It is a separate cdef class,
    not a LogicCircuit subclass, so the attribute simply did not exist and every
    multi-instruction sequence raised AttributeError.  That failed silently in a
    way worth remembering: the RQ2 worker calls precompile unconditionally, so
    753 of 943 sequence cases errored, the harness recorded ``time_ns: 0`` for
    each, and the merge then carried those zeros forward as completed runs.

Both are about the same contract: precompile is idempotent, depends only on the
rule and the architecture, and never touches values or taint.
"""
from __future__ import annotations

import pytest

from microtaint.instrumentation.ast import ChainedCircuit, EvalContext, LogicCircuit
from microtaint.instrumentation.cell import _get_decoded
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import _cached_generate_static_rule, generate_static_rule
from microtaint.types import Architecture, Register

ARCH = Architecture.AMD64
RN = ('RAX', 'RBX', 'RCX', 'RDX')
REGS = [Register(n, 64) for n in RN] + [Register(n, 1) for n in ('CF', 'OF', 'SF', 'ZF', 'PF', 'AF')]

# label -> hex.  `add;xor` is two instructions, so it lifts to a ChainedCircuit.
SINGLE = {'add rax,rbx': '4801d8', 'shrd rax,rbx,cl': '480fadd8', 'bsf rax,rbx': '480fbcc3'}
SEQUENCE = {'add;xor': '4801d84831d1', 'mov;and': '4889d84821c8'}


@pytest.fixture(scope='module')
def sim() -> CellSimulator:
    return CellSimulator(ARCH, use_unicorn=False, use_c=True)


def _rule(hexs: str) -> LogicCircuit | ChainedCircuit:
    _cached_generate_static_rule.cache_clear()
    return generate_static_rule(ARCH, bytes.fromhex(hexs), REGS)


@pytest.mark.parametrize('hexs', SINGLE.values(), ids=list(SINGLE))
def test_precompile_warms_the_decode_cache(hexs: str, sim: CellSimulator) -> None:
    """After precompile, the decode for this instruction is already cached.

    Asserted as a cache HIT rather than by timing: a wall-clock threshold would
    be flaky on a shared machine, while the hit counter states the actual claim.
    """
    circuit = _rule(hexs)
    _get_decoded.cache_clear()
    circuit.precompile(sim)
    before = _get_decoded.cache_info()
    assert before.currsize >= 1, 'precompile left the decode cache empty'

    _get_decoded(ARCH, bytes.fromhex(hexs))
    after = _get_decoded.cache_info()
    assert after.hits == before.hits + 1, 'the instruction was not the entry precompile warmed'
    assert after.misses == before.misses, 'decoding after precompile still missed'


@pytest.mark.parametrize('hexs', SEQUENCE.values(), ids=list(SEQUENCE))
def test_chained_circuit_precompiles_every_step(hexs: str, sim: CellSimulator) -> None:
    """A multi-instruction sequence precompiles, and warms EVERY step's decode."""
    circuit = _rule(hexs)
    assert isinstance(circuit, ChainedCircuit), 'expected a sequence to lift to ChainedCircuit'
    _get_decoded.cache_clear()
    circuit.precompile(sim)                      # regression: used to raise AttributeError
    assert _get_decoded.cache_info().currsize == len(circuit.sub_circuits)

    for sub in circuit.sub_circuits:
        before = _get_decoded.cache_info()
        _get_decoded(ARCH, bytes.fromhex(sub.instruction))
        assert _get_decoded.cache_info().hits == before.hits + 1, f'{sub.instruction} not warmed'


@pytest.mark.parametrize('hexs', list(SINGLE.values()) + list(SEQUENCE.values()),
                         ids=list(SINGLE) + list(SEQUENCE))
def test_precompile_is_idempotent_and_does_not_change_taint(hexs: str, sim: CellSimulator) -> None:
    """Precompiling twice is a no-op, and the taint equals the un-precompiled taint.

    The second half is the one that matters: precompile is only legitimate if it
    is invisible in the answer.  It takes the simulator and not an EvalContext
    precisely so it CANNOT specialise to the state, and this pins that.
    """
    values = {'RAX': 0x0123456789ABCDEF, 'RBX': 0xFEDCBA9876543210,
              'RCX': 0x1F, 'RDX': 0xDEADBEEF}
    taint = {'RAX': 0x00FF00FF00FF00FF, 'RBX': 0xF0F0, 'RCX': 0x3, 'RDX': 0}

    def _run(pre: int) -> dict[str, int]:
        circuit = _rule(hexs)
        for _ in range(pre):
            circuit.precompile(sim)
        return circuit.evaluate(EvalContext(input_values=dict(values),
                                            input_taint=dict(taint), simulator=sim))

    cold, once, twice = _run(0), _run(1), _run(2)
    assert once == cold, 'precompile changed the taint'
    assert twice == once, 'precompile is not idempotent'
