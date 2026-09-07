"""Ratchet: taint propagation may only ever need FEWER operations.

The op count measured here is the whole cost of propagating taint through one
instruction, flags included.  It is the quantity the compaction work exists to
reduce, and the quantity a future JIT of the taint circuit will be bounded by,
so it is pinned per instruction per ISA and allowed to move one way.

Coverage is ratcheted alongside it: an instruction the pass answers today may
not start declining, or "fewer operations" could be bought by answering fewer
instructions.

Regenerate deliberately, never to make a red test green:
    .venv/bin/python -m tests.perop_op_ratchet --update
"""

import pytest

from tests.perop_op_ratchet import compare, load_baseline, measure


@pytest.fixture(scope='module')
def measured():
    return measure()


def test_baseline_exists():
    assert load_baseline() is not None, (
        'no op-count baseline; run: '
        '.venv/bin/python -m tests.perop_op_ratchet --update')


def test_no_instruction_needs_more_operations(measured):
    baseline = load_baseline()
    regressions, _improvements, _new = compare(measured, baseline)
    if regressions:
        detail = '\n'.join(f'  {isa} {label}: {before} -> {after} ops'
                           for isa, label, before, after in regressions[:20])
        pytest.fail(f'{len(regressions)} instruction(s) got more expensive:\n'
                    f'{detail}')


def test_no_instruction_started_declining(measured):
    baseline = load_baseline()
    _reg, _imp, new_declines = compare(measured, baseline)
    if new_declines:
        detail = '\n'.join(f'  {isa} {label} (was {n} ops)'
                           for isa, label, n in new_declines[:20])
        pytest.fail(f'{len(new_declines)} instruction(s) the per-op pass used '
                    f'to answer now decline:\n{detail}')
