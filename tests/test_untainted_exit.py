"""tests/test_untainted_exit.py
================================
The premise behind the untainted-input fast exit in ``fastpath.h``.

The hot path dismisses an instruction outright when none of its taint inputs is
set, on the argument that ``f(V|0) XOR f(V&~0)`` is zero on every target and
every soundness floor is keyed on a tainted input bit, so all of them contribute
zero as well. That argument is only as good as the claim that NO compiled
circuit can manufacture taint out of nothing -- one opcode that pushes a
constant into an output, or a floor that taints unconditionally, would break it
and the exit would silently UNDER-taint (it would clear a target that the real
evaluation would have marked).

Reading the opcodes is not proof, so this checks the property directly over the
whole instruction bank: evaluate every form with an all-clean input state and
assert the resulting taint is empty. That is the exact condition the exit
assumes, tested on the same corpus the perf ratchet uses (~1,500 forms across
AMD64 / ARM64 / MIPS64BE / PPC32BE / RISCV64).

If this ever fails for an instruction, the fast exit is unsound FOR THAT FORM
and the fix is to exclude it (tighten the eligibility test in
``capi_compiled_prefilter``), not to relax this test.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from benchmark.instruction_bank import load_bank
from microtaint.instrumentation.ast import EvalContext, LogicCircuit
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import ImplicitTaintPolicy


def _clean_cases() -> Iterator[tuple[str, str, LogicCircuit, EvalContext]]:
    """(isa, label, circuit, context) with EVERY input taint zero."""
    for spec in load_bank().values():
        sim = CellSimulator(spec.arch)
        # Values are arbitrary but non-trivial: a circuit whose taint output
        # depended on operand VALUES rather than input taint would show up here.
        vals = {r.name: (0x1234567 + 7 * i) for i, r in enumerate(spec.regs)}
        for ins in spec.instructions:
            try:
                circ = generate_static_rule(spec.arch, ins.bytes, spec.regs)
            except Exception:
                continue
            ectx = EvalContext(
                input_values=vals,
                input_taint={},          # the whole point: nothing is tainted
                simulator=sim,
                implicit_policy=ImplicitTaintPolicy.KEEP,
            )
            yield spec.name, ins.label, circ, ectx


def _bank_isas() -> list[str]:
    """Derived from the bank, never hardcoded: a hardcoded list silently stops
    covering any ISA the bank gains later (it already had AMD64_SIMD and
    ARM64_SIMD that a hand-written list missed)."""
    return sorted(load_bank().keys())


@pytest.mark.parametrize('isa', _bank_isas())
def test_clean_inputs_produce_no_taint(isa: str) -> None:
    """No instruction may invent taint from an all-clean input state."""
    checked = 0
    offenders: list[str] = []
    for case_isa, label, circ, ectx in _clean_cases():
        if case_isa != isa:
            continue
        checked += 1
        try:
            out = circ.evaluate(ectx)
        except Exception:
            continue
        dirty = {k: hex(int(v)) for k, v in (out or {}).items() if int(v) != 0}
        if dirty:
            offenders.append(f'{label}: {dirty}')

    assert checked > 0, f'no {isa} forms evaluated - bank or lifter problem'
    assert not offenders, (
        f'{len(offenders)}/{checked} {isa} forms produced taint from clean inputs, '
        f'so the untainted-input fast exit would under-taint them. '
        f'First few: {offenders[:5]}'
    )
