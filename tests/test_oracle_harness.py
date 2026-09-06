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
    reference_taint,
    run_bank,
)


def test_harness_self_consistent() -> None:
    """reference vs itself over the register bank: exact everywhere, no under."""
    rep = run_bank(reference_taint, n_dense=2, n_sparse=2, ref='differential')
    assert rep.n_cases > 1000, rep.summary()
    assert rep.n_instrs > 1000, rep.summary()
    assert rep.n_under == 0, rep.summary()
    assert rep.n_exact == rep.n_cases, f'{rep.summary()} :: {rep.mismatches[:5]}'
    assert not rep.errors, rep.errors[:5]
    assert rep.skipped_mem > 0, 'memory forms should be detected + skipped'


def test_current_c_path_bit_exact_vs_differential() -> None:
    """The C register fast path must reproduce the differential oracle exactly.
    This is the invariant the whole rework preserves; a non-identity engine
    matching the oracle also proves the harness detects real equivalence."""
    rep = run_bank(engine_evaluate_c, n_dense=3, n_sparse=3, ref='differential')
    assert rep.n_under == 0, f'C path UNDER-taints vs differential: {rep.mismatches[:5]}'
    assert rep.n_over_only == 0, f'C path OVER-taints vs differential: {rep.mismatches[:5]}'
    assert rep.n_exact == rep.n_cases, f'{rep.summary()} :: {rep.mismatches[:5]}'


def test_array_gather_bit_exact_vs_differential() -> None:
    """The array-gather register path (evaluate_c_arr) must reproduce the
    differential oracle exactly over the register bank -- the de-risked core of
    the Phase-1 C-native interface, validated before it is wired into the hook."""
    rep = run_bank(engine_evaluate_c_arr, n_dense=4, n_sparse=4, ref='differential')
    assert rep.n_under == 0, f'array-gather UNDER-taints: {rep.mismatches[:5]}'
    assert rep.n_over_only == 0, f'array-gather OVER-taints: {rep.mismatches[:5]}'
    assert rep.n_exact == rep.n_cases, f'{rep.summary()} :: {rep.mismatches[:5]}'


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
