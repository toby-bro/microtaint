"""Soundness gate for the one-pass per-op taint composer.

The claim under test is narrow and non-negotiable: for every instruction the
pass answers, the taint it reports must cover every bit the real CPU can
actually move.  Precision may improve freely -- and does, the pass is tighter
than the whole-instruction differential on hundreds of cases -- but a bit that
the hardware moves and the pass calls clean is an under-taint, and under-taint
is never acceptable.

Two design points worth stating, because both were mistakes at some stage:

  * The reference is Unicorn per-bit ground truth, NOT the engine's existing
    differential.  Scoring a more precise engine against the old answer reports
    every precision gain as an under-taint.
  * The gate is a SUBSET check -- under(per-op) must be contained in
    under(current engine) -- not equality with the truth.  x86 leaves AF after
    `cmp` and OF after a multi-bit shift architecturally undefined; SLEIGH
    models them as unchanged while Unicorn computes something, so both engines
    "miss" the same bits and neither is at fault.
"""

import pytest

from tests.perop_c_bank import run_bank_perop_c


@pytest.mark.parametrize('isa', ['AMD64', 'ARM64', 'RISCV64'])
def test_no_new_under_taint_vs_ground_truth(isa, request: pytest.FixtureRequest):
    from tests.conftest import fuzz_budget

    rep = run_bank_perop_c(isas=[isa],
                           n_sparse=fuzz_budget(3, request.config),
                           ref='ground_truth')
    assert rep.n_instrs > 0, f'{isa}: no instructions exercised'
    assert rep.n_cases > 0, f'{isa}: no cases evaluated'
    if rep.n_under_new:
        detail = '\n'.join(
            f'  {label}: missing {[(k, hex(b)) for k, b in new.items()]} '
            f'with taint {[(k, hex(x)) for k, x in it.items() if x]}'
            for label, it, _iv, new in rep.new_under_examples[:6])
        pytest.fail(f'{isa}: {rep.n_under_new} case(s) under-taint where the '
                    f'current engine does not:\n{detail}')


@pytest.mark.parametrize('isa,floor', [('AMD64', 0.85), ('ARM64', 0.90),
                                       ('RISCV64', 0.85)])
def test_coverage_does_not_collapse(isa, floor):
    """A decline is safe but not free: it sends the instruction back to the
    slow whole-instruction differential.  Guard the fraction the pass answers so
    a rule change cannot buy correctness by quietly declining more."""
    rep = run_bank_perop_c(isas=[isa], n_sparse=1, ref='ground_truth')
    cov = rep.n_answered / rep.n_instrs
    assert cov >= floor, (f'{isa}: per-op pass answers only {cov:.1%} of the '
                          f'bank (floor {floor:.0%}); '
                          f'{rep.n_declined} declined')
