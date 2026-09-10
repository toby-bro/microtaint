"""The GIL-free C taint evaluator (CompiledCircuit.evaluate_c) is bit-identical.

evaluate_c runs a compiled circuit's taint entirely on the C path: OP_PUSH_TAINT/
VALUE read uint64 arrays (not the Python input dicts), OP_PUSH_CONST reads a
precomputed uint64 array, OP_CALL_CELL uses the fast cell CAPI, and OP_END writes
a uint64 -- no PyObject in the evaluation loop.  For every register-only
(c_evaluable) circuit it must produce exactly the same output taint as the normal
evaluate().  This is the correctness gate; the speed/GIL-freedom is the point.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'benchmark'))
from instruction_bank import load_bank  # type: ignore[import-not-found]

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule


def _check_isa(isa: str) -> tuple[int, int, list[tuple[str, str, str, str]]]:
    spec = load_bank(isas=[isa])[isa]
    sim = CellSimulator(spec.arch)
    rn = [r.name for r in spec.regs]
    rbits = {r.name: r.bits for r in spec.regs}
    ran = 0
    mism = []
    for ins in spec.instructions:
        try:
            circ = generate_static_rule(spec.arch, ins.bytes, spec.regs)
        except Exception:  # noqa: S112
            continue
        vals = dict.fromkeys(rn, 0)
        taint = dict.fromkeys(rn, 0)
        for r in rn[1:5]:
            taint[r] = (1 << min(16, rbits[r])) - 1
        try:
            normal = circ.evaluate(EvalContext(input_values=vals, input_taint=taint, simulator=sim))
        except Exception:  # noqa: S112
            continue
        comp = getattr(circ, '_compiled', None)
        if comp is None or comp is False:
            continue
        cres = comp.evaluate_c(taint, vals, sim._pcode)
        if cres is None:
            continue  # not c_evaluable (mem / fallback / >64-bit const)
        ran += 1
        for k in set(normal) | set(cres):
            if (normal.get(k, 0) or 0) != (cres.get(k, 0) or 0):
                mism.append((ins.label, k, hex(normal.get(k, 0) or 0), hex(cres.get(k, 0) or 0)))
                break
    return ran, len(mism), mism[:10]


def test_evaluate_c_matches_evaluate_amd64() -> None:
    ran, n_mism, examples = _check_isa('AMD64')
    assert ran > 300, f'too few c_evaluable AMD64 circuits exercised: {ran}'
    assert n_mism == 0, f'evaluate_c != evaluate on {n_mism} circuits, e.g. {examples}'


@pytest.mark.parametrize('isa', ['ARM64', 'MIPS64BE', 'PPC32BE', 'RISCV64'])
def test_evaluate_c_matches_evaluate_other_isas(isa):
    _ran, n_mism, examples = _check_isa(isa)
    assert n_mism == 0, f'{isa}: evaluate_c != evaluate on {n_mism} circuits, e.g. {examples}'
