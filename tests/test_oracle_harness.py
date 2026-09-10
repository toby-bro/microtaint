"""Standing gate for the rework oracle harness (Phase 0 of unify-taint-execution).

Bounded so it runs in CI; the full-bank sweeps run standalone via
`python -m tests.oracle_harness`.  These tests are the safety net every rework
phase is measured against:

  * the harness drives the multi-ISA bank and its comparison is self-consistent;
  * the CURRENT C fast path is bit-exact vs the differential oracle (the invariant
    Phases 1-2 must preserve);
  * the CURRENT engine is SOUND vs Unicorn per-bit ground truth (no under-taint).
"""
# ruff: noqa: PLC0415
# mypy: disable-error-code="no-untyped-def,no-untyped-call,import-untyped"
from __future__ import annotations

import pytest

from tests.oracle_harness import (
    UC_DESCS,
    engine_evaluate_c,
    engine_evaluate_c_arr,
    engine_evaluate_c_arr_ptr,
    reference_taint,
    run_bank,
)


@pytest.mark.slow
def test_harness_self_consistent() -> None:
    """reference vs itself over the register bank: exact everywhere, no under."""
    rep = run_bank(reference_taint, n_dense=2, n_sparse=2, ref='differential')
    assert rep.n_cases > 1000, rep.summary()
    assert rep.n_instrs > 1000, rep.summary()
    assert rep.n_under == 0, rep.summary()
    assert rep.n_exact == rep.n_cases, f'{rep.summary()} :: {rep.mismatches[:5]}'
    assert not rep.errors, rep.errors[:5]
    assert rep.skipped_mem > 0, 'memory forms should be detected + skipped'


@pytest.mark.slow
def test_current_c_path_bit_exact_vs_differential() -> None:
    """The C register fast path must reproduce the differential oracle exactly.
    This is the invariant the whole rework preserves; a non-identity engine
    matching the oracle also proves the harness detects real equivalence."""
    rep = run_bank(engine_evaluate_c, n_dense=3, n_sparse=3, ref='differential')
    assert rep.n_under == 0, f'C path UNDER-taints vs differential: {rep.mismatches[:5]}'
    assert rep.n_over_only == 0, f'C path OVER-taints vs differential: {rep.mismatches[:5]}'
    assert rep.n_exact == rep.n_cases, f'{rep.summary()} :: {rep.mismatches[:5]}'


@pytest.mark.slow
def test_array_gather_bit_exact_vs_differential() -> None:
    """The array-gather register path (evaluate_c_arr) must reproduce the
    differential oracle exactly over the register bank -- the de-risked core of
    the Phase-1 C-native interface, validated before it is wired into the hook."""
    rep = run_bank(engine_evaluate_c_arr, n_dense=4, n_sparse=4, ref='differential')
    assert rep.n_under == 0, f'array-gather UNDER-taints: {rep.mismatches[:5]}'
    assert rep.n_over_only == 0, f'array-gather OVER-taints: {rep.mismatches[:5]}'
    assert rep.n_exact == rep.n_cases, f'{rep.summary()} :: {rep.mismatches[:5]}'


@pytest.mark.slow
def test_array_gather_ptr_bit_exact_vs_differential() -> None:
    """The live-usable pointer form (evaluate_c_arr_ptr) writes target slots of a
    raw uint64 taint array in place; the resulting state must equal the
    differential oracle over the register bank.  This is the eval the hook will
    call directly on its C-array state."""
    rep = run_bank(engine_evaluate_c_arr_ptr, n_dense=4, n_sparse=4, ref='differential')
    assert rep.n_under == 0, f'ptr array-gather UNDER-taints: {rep.mismatches[:5]}'
    assert rep.n_over_only == 0, f'ptr array-gather OVER-taints: {rep.mismatches[:5]}'
    assert rep.n_exact == rep.n_cases, f'{rep.summary()} :: {rep.mismatches[:5]}'


@pytest.mark.slow
def test_current_engine_sound_vs_ground_truth_amd64() -> None:
    """The current differential must not UNDER-taint vs Unicorn per-bit truth
    on AMD64 (over-taint is allowed and expected on undefined flags)."""
    pytest.importorskip('unicorn')
    ud = UC_DESCS['AMD64']()
    rep = run_bank(reference_taint, isas=['AMD64'], n_dense=0, n_sparse=24,
                   ref='ground_truth', uc_desc=ud)
    assert rep.n_cases > 0, 'no ground-truth cases exercised'
    under = [m for m in rep.mismatches if m[1] == 'UNDER']
    assert rep.n_under == 0, f'UNDER-taint vs ground truth: {under[:8]}'


# ---------------------------------------------------------------------------
# `models_disagree` decides which outputs leave the verdict, so an empty result
# from it is indistinguishable from "the two models agree everywhere".  It used
# to reach for `evaluate_concrete_flat` inside `except Exception: continue`,
# and that method exists only on the C evaluator -- so on the Cython one every
# flag was skipped, nothing was compared, and it returned two empty sets.
# ---------------------------------------------------------------------------

#: `add rax, rbx`.  The docstring's own example of the `unmodelled` half:
#: x86 DEFINES AF here, and SLEIGH simply does not compute it.
_ADD_RAX_RBX = bytes.fromhex('4801d8')


def _states() -> list[dict[str, int]]:
    return [{'RAX': v, 'RBX': w, 'RCX': 3, 'RDX': 0, 'RSI': 0, 'RDI': 0}
            for v in (0x1, 0x8000_0000_0000_0001, 0x0F0F_0F0F_0F0F_0F0F)
            for w in (0x1, 0x7FFF_FFFF_FFFF_FFFF)]


def test_models_disagree_returns_two_sets_and_compares_something() -> None:
    """The real path answers, and finds the gap the docstring names."""
    from microtaint.types import Architecture
    from tests.oracle_harness import models_disagree

    got = models_disagree(UC_DESCS['AMD64'](), Architecture.AMD64,
                          _ADD_RAX_RBX, _states())
    assert isinstance(got, tuple), got
    assert len(got) == 2, got
    undefined, unmodelled = got
    assert isinstance(undefined, set), undefined
    assert isinstance(unmodelled, set), unmodelled
    # Reaching here already proves something was compared: the function now
    # raises when it compared nothing.  AF is the specific gap -- the lifter
    # never writes it, so it lands in `unmodelled`, not in `undefined`.
    assert 'AF' in unmodelled, (undefined, unmodelled)
    assert 'AF' not in undefined, undefined


class _NoFlatEvaluator:
    """An evaluator that predates `evaluate_concrete_flat`, like the Cython one."""


def _fake_simulator(pcode: object) -> type:
    class _Sim:
        def __init__(self, arch: object) -> None:
            self.arch = arch
            self._pcode = pcode
    return _Sim


def test_models_disagree_refuses_an_evaluator_that_cannot_compare(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """No `evaluate_concrete_flat` must raise, not return two empty sets."""
    import microtaint.simulator
    from microtaint.types import Architecture
    from tests.oracle_harness import models_disagree

    monkeypatch.setattr(microtaint.simulator, 'CellSimulator',
                        _fake_simulator(_NoFlatEvaluator()))
    with pytest.raises(RuntimeError, match='evaluate_concrete_flat'):
        models_disagree(UC_DESCS['AMD64'](), Architecture.AMD64,
                        _ADD_RAX_RBX, _states())


#: `push rax`.  Its p-code writes no flag, so the cell answers for none of them
#: and the honest result is two empty sets.
_PUSH_RAX = bytes.fromhex('50')


def test_models_disagree_is_empty_when_the_instruction_writes_no_flag() -> None:
    """A per-instruction zero is ordinary, not a broken harness.

    Measured over the AMD64 bank, 43 of the first 400 instructions answer for
    no flag at all: every `push`, and `adcx` / `adox`, whose CF and OF the
    lifter DOES write but the cell declines.  A first version of this fix
    raised on that, which would have failed the sweep on ordinary
    instructions.  The condition worth refusing is the evaluator having no
    `evaluate_concrete_flat` at all, which the test above covers.
    """
    from microtaint.types import Architecture
    from tests.oracle_harness import models_disagree

    undefined, unmodelled = models_disagree(
        UC_DESCS['AMD64'](), Architecture.AMD64, _PUSH_RAX, _states())
    assert undefined == set(), undefined
    assert unmodelled == set(), unmodelled
