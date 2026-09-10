"""A circuit's name->slot cache must follow the slot map, which grows under it.

The hook interns register names LAZILY: a name gets a slot the first time it is
actually written, so `slot_map` grows for as long as a run keeps meeting new
registers.  The C evaluators translate their string pool to slots once and cache
the result, and that cache was keyed on the POOL size -- which is fixed for the
life of a circuit, because the pool comes from the instruction.  So the first
evaluation froze the mapping: every name the hook had not yet interned was
pinned at -1 for good.

-1 means "not tracked", and both readers honour it, so the circuit went on
reading 0 for an input whose slot was by then holding real taint, and skipped
committing every output whose name was interned after that first visit.  That is
an UNDER-taint: taint that the same circuit computes correctly on a fresh mapping
silently disappears.  It needs no exotic program, only a register whose first
appearance comes after some instruction has already run.

The tests drive the C evaluators directly with two slot maps, the second one
grown the way `_slot_for` grows it (existing indices kept, the new name
appended), and compare against the same circuit evaluated with that mapping from
the start.
"""
from __future__ import annotations

import ctypes

import pytest

from microtaint.instrumentation.ast import EvalContext
from microtaint.instrumentation.cell_c.cell_c import PCodeCellEvaluatorC
from microtaint.instrumentation.cell_c.circuit_c import CompiledCircuit
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, ImplicitTaintPolicy, Register

#: `add rax, rbx` -- two register inputs, a result and the flags it writes.
ADD_RAX_RBX = bytes.fromhex('4801d8')
_FLAGS = ('CF', 'ZF', 'SF', 'OF', 'PF', 'AF')

_REGS = [Register(name='RAX', bits=64), Register(name='RBX', bits=64),
         *[Register(name=f, bits=1) for f in _FLAGS]]


def _compiled() -> tuple[CompiledCircuit, CellSimulator]:
    """A freshly generated circuit plus its CompiledCircuit (built on first
    evaluate).  Fresh per test: the whole point is per-circuit cache state."""
    circuit = generate_static_rule(Architecture.AMD64, ADD_RAX_RBX, _REGS)
    sim = CellSimulator(Architecture.AMD64)
    circuit.evaluate(EvalContext(input_taint={}, input_values={'RAX': 1, 'RBX': 2},
                                 simulator=sim, implicit_policy=ImplicitTaintPolicy.IGNORE))
    comp = getattr(circuit, '_compiled', None)
    if comp is None or comp is False:
        pytest.skip('this instruction has no compiled circuit on this build')
    return comp, sim


def _run_ptr(comp: CompiledCircuit, sim: CellSimulator,
             slot_map: dict[str, int], taint: dict[str, int],
             values: dict[str, int]) -> dict[str, int]:
    kernel = sim._pcode
    assert isinstance(kernel, PCodeCellEvaluatorC), \
        'the C-array path needs the C evaluator'
    """evaluate_c_arr_ptr against a slot-indexed array -> {name: out taint}."""
    n = max(slot_map.values()) + 1
    arr = ctypes.c_uint64 * n
    t, v = arr(), arr()
    for name, slot in slot_map.items():
        t[slot] = taint.get(name, 0)
        v[slot] = values.get(name, 0)
    rc = comp.evaluate_c_arr_ptr(ctypes.addressof(t), ctypes.addressof(v),
                                 n, kernel, slot_map)
    if rc is None:
        pytest.skip('the C array path declined this instruction')
    return {name: int(t[slot]) for name, slot in slot_map.items()}


def _run_list(comp: CompiledCircuit, sim: CellSimulator,
              slot_map: dict[str, int], taint: dict[str, int],
              values: dict[str, int]) -> dict[str, int]:
    kernel = sim._pcode
    assert isinstance(kernel, PCodeCellEvaluatorC), \
        'the C-array path needs the C evaluator'
    """evaluate_c_arr (the list form) -> {name: out taint} for the targets."""
    n = max(slot_map.values()) + 1
    t = [0] * n
    v = [0] * n
    for name, slot in slot_map.items():
        t[slot] = taint.get(name, 0)
        v[slot] = values.get(name, 0)
    out = comp.evaluate_c_arr(t, v, kernel, slot_map)
    if out is None:
        pytest.skip('the C array path declined this instruction')
    return out


# `_slot_for` appends, so growing the map never moves a name already in it.
_WITHOUT_RBX = {'RAX': 0, **{f: 1 + i for i, f in enumerate(_FLAGS)}}
_WITH_RBX = {**_WITHOUT_RBX, 'RBX': len(_WITHOUT_RBX)}
_TAINT_RBX = {'RBX': 0xFF}
_VALUES = {'RAX': 1, 'RBX': 2}


def test_input_interned_late_is_still_read() -> None:
    """RBX gets its slot after the circuit has already run once.  Its taint must
    reach RAX and the flags exactly as if the mapping had always been there."""
    stale_comp, sim = _compiled()
    _run_ptr(stale_comp, sim, _WITHOUT_RBX, {}, {'RAX': 1})   # RBX not interned
    got = _run_ptr(stale_comp, sim, _WITH_RBX, _TAINT_RBX, _VALUES)

    fresh_comp, sim2 = _compiled()
    want = _run_ptr(fresh_comp, sim2, _WITH_RBX, _TAINT_RBX, _VALUES)

    assert got == want, (
        'the circuit answered differently once its slot mapping had been built '
        f'before RBX was interned: {got} vs {want}')
    assert got['RAX'], 'RBX carried taint into RAX; the result came back clean'


def test_output_interned_late_is_still_written() -> None:
    """Same for a TARGET: a flag interned after the first visit must still have
    its taint committed, not dropped for the life of the circuit."""
    without_zf = {n: s for n, s in _WITH_RBX.items() if n != 'ZF'}
    without_zf = {n: i for i, n in enumerate(without_zf)}
    with_zf = {**without_zf, 'ZF': len(without_zf)}

    stale_comp, sim = _compiled()
    _run_ptr(stale_comp, sim, without_zf, {}, {'RAX': 1})     # ZF not interned
    got = _run_ptr(stale_comp, sim, with_zf, _TAINT_RBX, _VALUES)

    fresh_comp, sim2 = _compiled()
    want = _run_ptr(fresh_comp, sim2, with_zf, _TAINT_RBX, _VALUES)

    assert got == want, f'ZF interned late changed the answer: {got} vs {want}'
    assert got['ZF'], 'ZF depends on a tainted operand; its taint was dropped'


def test_list_form_follows_the_growing_map() -> None:
    """evaluate_c_arr kept its own copy of the same cache; it must agree too."""
    stale_comp, sim = _compiled()
    _run_list(stale_comp, sim, _WITHOUT_RBX, {}, {'RAX': 1})
    got = _run_list(stale_comp, sim, _WITH_RBX, _TAINT_RBX, _VALUES)

    fresh_comp, sim2 = _compiled()
    want = _run_list(fresh_comp, sim2, _WITH_RBX, _TAINT_RBX, _VALUES)

    assert got == want, f'the list form under-tainted after a late intern: {got} vs {want}'
    assert got.get('RAX'), 'RBX carried taint into RAX; the result came back clean'


def test_two_slot_maps_are_not_confused() -> None:
    """Two callers, two mappings of the SAME size: the cache must not serve one
    caller's slots to the other (the failure the size check alone would miss)."""
    comp, sim = _compiled()
    a = _WITH_RBX
    b = {**{n: s for n, s in a.items() if n not in ('RAX', 'RBX')},
         'RAX': a['RBX'], 'RBX': a['RAX']}          # same names, swapped slots

    got_a = _run_ptr(comp, sim, a, _TAINT_RBX, _VALUES)
    got_b = _run_ptr(comp, sim, b, _TAINT_RBX, _VALUES)
    assert got_a == got_b, (
        f'the same question answered differently under two mappings: {got_a} vs {got_b}')
