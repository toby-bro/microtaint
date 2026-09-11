"""Standing gate for the rework oracle harness (Phase 0 of unify-taint-execution).

Bounded so it runs in CI; the full-bank sweeps run standalone via
`python -m tests.oracle_harness`.  These tests are the safety net every rework
phase is measured against:

  * the harness drives the multi-ISA bank and its comparison is self-consistent;
  * the CURRENT C fast path is bit-exact vs the differential oracle (the invariant
    Phases 1-2 must preserve);
  * the CURRENT engine is SOUND vs Unicorn per-bit ground truth (no under-taint).
"""
from __future__ import annotations

import pytest

from tests.oracle_harness import (
    UC_DESCS,
    Ref,
    engine_evaluate_c,
    engine_evaluate_c_arr,
    engine_evaluate_c_arr_ptr,
    reference_taint,
    run_bank,
)


@pytest.mark.slow
def test_harness_self_consistent() -> None:
    """reference vs itself over the register bank: exact everywhere, no under."""
    rep = run_bank(reference_taint, n_dense=2, n_sparse=2, ref=Ref.DIFFERENTIAL)
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
    rep = run_bank(engine_evaluate_c, n_dense=3, n_sparse=3, ref=Ref.DIFFERENTIAL)
    assert rep.n_under == 0, f'C path UNDER-taints vs differential: {rep.mismatches[:5]}'
    assert rep.n_over_only == 0, f'C path OVER-taints vs differential: {rep.mismatches[:5]}'
    assert rep.n_exact == rep.n_cases, f'{rep.summary()} :: {rep.mismatches[:5]}'


@pytest.mark.slow
def test_array_gather_bit_exact_vs_differential() -> None:
    """The array-gather register path (evaluate_c_arr) must reproduce the
    differential oracle exactly over the register bank -- the de-risked core of
    the Phase-1 C-native interface, validated before it is wired into the hook."""
    rep = run_bank(engine_evaluate_c_arr, n_dense=4, n_sparse=4, ref=Ref.DIFFERENTIAL)
    assert rep.n_under == 0, f'array-gather UNDER-taints: {rep.mismatches[:5]}'
    assert rep.n_over_only == 0, f'array-gather OVER-taints: {rep.mismatches[:5]}'
    assert rep.n_exact == rep.n_cases, f'{rep.summary()} :: {rep.mismatches[:5]}'


@pytest.mark.slow
def test_array_gather_ptr_bit_exact_vs_differential() -> None:
    """The live-usable pointer form (evaluate_c_arr_ptr) writes target slots of a
    raw uint64 taint array in place; the resulting state must equal the
    differential oracle over the register bank.  This is the eval the hook will
    call directly on its C-array state."""
    rep = run_bank(engine_evaluate_c_arr_ptr, n_dense=4, n_sparse=4, ref=Ref.DIFFERENTIAL)
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
                   ref=Ref.GROUND_TRUTH, uc_desc=ud)
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


# ---------------------------------------------------------------------------
# The oracle's own strength.
#
# `ground_truth` is the REFERENCE for `under = truth & ~got`, so truth that is
# too small does not make a test fail -- it makes every no-under-taint gate
# quietly lenient.  That is the one direction nobody notices, so it needs a
# test of its own.
# ---------------------------------------------------------------------------

#: `cmp $0xffffffff,%eax`.  Chosen because it is what glibc's EOF test compiles
#: to, and because its flags depend on the tainted bits only when SEVERAL of
#: them agree: EAX can equal -1 only if every bit is set.
_CMP_MINUS_ONE = bytes.fromhex('83f8ff')


def _zf_truth(rax: int, *, bases: int | None = None) -> int:
    """ZF's truth mask.  `bases=None` uses the harness DEFAULT on purpose: a
    test that pins the strength explicitly would still pass if the default were
    weakened, which is the change that actually matters."""
    from tests.oracle_harness import ground_truth

    desc = UC_DESCS['AMD64']()
    values = dict.fromkeys(desc.gp, 0)
    values['RAX'] = rax
    taint = dict.fromkeys(desc.gp, 0)
    taint['RAX'] = 0xFF
    kw = {} if bases is None else {'bases': bases}
    return ground_truth(desc, _CMP_MINUS_ONE, taint, values, **kw).get('ZF', 0)


def test_the_oracle_sees_a_dependence_that_needs_several_tainted_bits() -> None:
    """With RAX = -1 and its low byte secret, ZF really does move.

    Flipping any one of those eight bits takes EAX away from -1 and clears ZF,
    so ZF depends on them -- but only from a base where the other seven are
    already set.  Probing one bit at a time up from an all-zero base can never
    reach that point, which is how this stayed invisible.
    """
    pytest.importorskip('unicorn')
    assert _zf_truth(0xFFFFFFFF) != 0, (
        'the oracle reports ZF clean for `cmp $-1,%eax` with RAX = -1 and its '
        'low byte tainted, but flipping any of those bits clears ZF; a truth '
        'this small makes every under-taint gate lenient')


def test_the_single_base_oracle_really_did_miss_it() -> None:
    """The counter-example is a counter-example.

    If one base found this too, the extra bases would be cost without benefit
    and this pair of tests would be measuring nothing.
    """
    pytest.importorskip('unicorn')
    assert _zf_truth(0xFFFFFFFF, bases=1) == 0, (
        'one base already finds this, so the multi-base probe is not what is '
        'catching it and these tests no longer pin the reason')


def test_the_oracle_still_calls_a_provably_constant_flag_clean() -> None:
    """Strength must not become noise.

    When EAX holds a byte (which is what `__uflow` returns) the upper bits are
    known zero, so EAX can never equal -1 and ZF is constant however the low
    byte moves.  An oracle that marked it tainted would report the engine's
    correct silence as an under-taint.
    """
    pytest.importorskip('unicorn')
    assert _zf_truth(0x68) == 0, (
        'ZF marked tainted for a comparison it provably cannot change')


# ---------------------------------------------------------------------------
# One Ref, and a comparison that honours it.
# ---------------------------------------------------------------------------

def test_the_two_sweeps_share_one_reference_enum() -> None:
    """`perop_c_bank.Ref` must BE `oracle_harness.Ref`, not a copy of it.

    It was a copy, character for character, and `run_bank_perop_c` selected its
    oracle with `ref is Ref.GROUND_TRUTH`.  A caller importing `Ref` from
    `oracle_harness` -- the obvious place, and where `run_bank` takes it from --
    failed that identity check in silence and was scored against the
    WHOLE-INSTRUCTION DIFFERENTIAL instead of against hardware.

    That is the worst possible way for it to fail.  The taint IR is deliberately
    TIGHTER than the differential in places, so the wrong comparison reports
    those precision gains as under-taints: a sweep asking "is the engine sound"
    answers with a flood of exactly the finding it exists to detect.  Measured,
    1479 of 2645 cases, every one of them spurious.
    """
    from tests import oracle_harness as oh
    from tests import perop_c_bank

    assert perop_c_bank.Ref is oh.Ref, (
        'the two sweeps have separate Ref enums again, so an `is` comparison '
        'in either one silently selects the wrong oracle')


def test_the_reference_may_be_given_as_its_spelling() -> None:
    """The enum's own docstring promises this, and `is` broke the promise.

    `Ref` is a StrEnum so that "a caller may still pass the spelling"; an
    identity comparison rejects the spelling just as silently as it rejected the
    other module's member, and selects the differential.
    """
    from tests import perop_c_bank

    # Typed as `object` so the type checker does not reject the comparison as
    # non-overlapping; a StrEnum member and its spelling really are equal, and
    # that equality is the whole affordance being pinned.
    for spelling, member in (('ground_truth', perop_c_bank.Ref.GROUND_TRUTH),
                             ('differential', perop_c_bank.Ref.DIFFERENTIAL)):
        as_object: object = spelling
        assert as_object == member, (
            f'{spelling!r} no longer equals {member!r}, so a caller passing '
            f'the spelling the docstring invites gets the wrong oracle')
