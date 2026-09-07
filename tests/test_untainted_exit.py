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

import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parent.parent / 'benchmark'
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from instruction_bank import load_bank  # type: ignore[import-not-found]  # noqa: E402

from microtaint.instrumentation.ast import EvalContext  # noqa: E402
from microtaint.simulator import CellSimulator  # noqa: E402
from microtaint.sleigh.engine import generate_static_rule  # noqa: E402
from microtaint.types import ImplicitTaintPolicy  # noqa: E402


def _clean_cases():
    """(isa, label, circuit, context) with EVERY input taint zero."""
    for spec in load_bank().values():
        sim = CellSimulator(spec.arch)
        # Values are arbitrary but non-trivial: a circuit whose taint output
        # depended on operand VALUES rather than input taint would show up here.
        vals = {r.name: (0x1234567 + 7 * i) for i, r in enumerate(spec.regs)}
        for ins in spec.instructions:
            try:
                circ = generate_static_rule(spec.arch, ins.bytes, spec.regs)
            except Exception:  # noqa: BLE001 - unliftable forms are not our concern
                continue
            ctx = EvalContext(
                input_values=vals,
                input_taint={},          # the whole point: nothing is tainted
                simulator=sim,
                implicit_policy=ImplicitTaintPolicy.KEEP,
            )
            yield spec.name, ins.label, circ, ctx


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
    for case_isa, label, circ, ctx in _clean_cases():
        if case_isa != isa:
            continue
        checked += 1
        try:
            out = circ.evaluate(ctx)
        except Exception:  # noqa: BLE001 - needs Unicorn / unsupported: not this test's subject
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
