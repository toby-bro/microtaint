"""Fast-path coverage: EVERY bank instruction must stay in the compiled C path.

The taint hot path has two C layers:

  * the compiled circuit (circuit_c ``do_evaluate`` / bytecode interpreter) that
    replaces the Cython AST walker, and
  * the C cell kernel (cell_c ``cell_eval_fast`` / ``execute_decoded``) that
    concretely executes an instruction without Unicorn or a Python cell eval.

An instruction is OFF the fast path if any of these happen:

  * ``walker_taint``   -- a (sub)circuit did not compile (``_compiled`` is False),
    so its taint is computed by the Python/Cython AST walker;
  * ``py_fallback``    -- a compiled circuit still has assignments the bytecode
    emitter could not compile (``python_fallback > 0``);
  * ``runtime_cell_fallback`` -- at run time the C kernel could not execute a
    cell (``fallback_calls`` grew), so it raised PCodeFallbackNeeded / bounced to
    the Python cell path;
  * ``unicorn_concrete`` -- a ChainedCircuit threads concrete state between steps
    through Unicorn instead of the native C kernel (see ast._run_concrete_step).

This test is the guard for "no instruction ever falls back to Python or misses
the fast path". It also caught the class the mission's "0 fallbacks" check
missed: the fallback count was measured only on assignments INSIDE circuits that
compiled, so a ChainedCircuit whose wrapper is not a CompiledCircuit (11 MIPS
imm-logic sequences) was invisible -- this test inspects sub-circuits.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from microtaint.instrumentation.ast import ChainedCircuit, EvalContext
from microtaint.simulator import CellSimulator, _native_be_safe
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import ImplicitTaintPolicy

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / 'benchmark'))
from instruction_bank import load_bank  # type: ignore[import-not-found]  # noqa: E402

_BANK = load_bank()
_ISAS = sorted(_BANK.keys())


def _subcircuits(circ: object) -> list:
    return list(circ.sub_circuits) if isinstance(circ, ChainedCircuit) else [circ]


def _fastpath_offenders(isa: str) -> dict[str, list[str]]:
    """Return {reason: [labels]} for every instruction of `isa` off the fast path."""
    spec = _BANK[isa]
    sim = CellSimulator(spec.arch)
    pc = sim._pcode
    taint = {r.name: 0xFFFFFFFFFFFFFFFF for r in spec.regs}
    values = {r.name: 0x0123456789ABCDEF for r in spec.regs}
    off: dict[str, list[str]] = {
        'walker_taint': [], 'py_fallback': [], 'runtime_cell_fallback': [], 'unicorn_concrete': [],
    }
    for ins in spec.instructions:
        circ = generate_static_rule(spec.arch, ins.bytes, spec.regs)
        nf0 = pc.fallback_calls
        circ.evaluate(EvalContext(
            input_taint=dict(taint), input_values=dict(values),
            simulator=sim, implicit_policy=ImplicitTaintPolicy.IGNORE,
        ))
        if pc.fallback_calls - nf0:
            off['runtime_cell_fallback'].append(f'{ins.label} (+{pc.fallback_calls - nf0})')

        subs = _subcircuits(circ)
        if any(getattr(s, '_compiled', None) in (False, None) for s in subs):
            off['walker_taint'].append(ins.label)
        else:
            for s in subs:
                st = s._compiled.stats()
                if st.get('python_fallback', 0):
                    off['py_fallback'].append(f'{ins.label} ({st["python_fallback"]}/{st["n_assignments"]})')
                    break

        if isinstance(circ, ChainedCircuit):
            for s in circ.sub_circuits:
                if not (sim._is_big_endian and _native_be_safe(sim.arch, s.instruction)):
                    off['unicorn_concrete'].append(f'{ins.label} (step {s.instruction})')
                    break
    return off


@pytest.mark.parametrize('isa', _ISAS)
def test_every_instruction_on_fast_path(isa: str) -> None:
    off = _fastpath_offenders(isa)
    total = sum(len(v) for v in off.values())
    if total:
        lines = [f'{isa}: {total} instruction(s) off the C fast path:']
        for reason, labels in off.items():
            if labels:
                lines.append(f'  {reason} ({len(labels)}):')
                lines += [f'    - {x}' for x in labels[:50]]
                if len(labels) > 50:
                    lines.append(f'    ... +{len(labels) - 50} more')
        pytest.fail('\n'.join(lines))
