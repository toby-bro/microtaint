"""Taint through memory, on both implementations.

A register-only API cannot answer the question people actually have, which is
whether a value that went into memory comes back out tainted.  These check the
two paths agree about that, because "run the suite against either
implementation" has to mean memory as well as registers.
"""
from __future__ import annotations

import pytest

from microtaint.taint_api import TaintSequence, explain
from microtaint.taint_memory import TaintMemory
from microtaint.types import Architecture

PUSH_RBX = bytes.fromhex('53')
POP_RAX = bytes.fromhex('58')
FULL = (1 << 64) - 1

PATHS = ['compiled', 'differential']


def test_memory_holds_bytes_and_taint_separately() -> None:
    m = TaintMemory()
    m.write(0x7000, 0xDEADBEEF, 8)
    assert m.read(0x7000, 8) == 0xDEADBEEF
    assert m.read_mask(0x7000, 8) == 0, 'writing a value must not invent taint'
    m.taint(0x7000, 4)
    assert m.read_mask(0x7000, 8) == 0xFFFFFFFF
    assert m.read(0x7000, 8) == 0xDEADBEEF, 'tainting must not disturb the value'


def test_never_written_memory_reads_as_zero() -> None:
    m = TaintMemory()
    assert m.read(0x123456, 8) == 0
    assert m.read_mask(0x123456, 8) == 0


@pytest.mark.parametrize('path', PATHS)
def test_a_store_puts_taint_in_memory(path: str) -> None:
    memory = TaintMemory()
    values = {'RBX': 0xCAFEBABE, 'RSP': 0x7000}
    _out, used = explain(Architecture.AMD64, PUSH_RBX, {'RBX': FULL}, values,
                         memory=memory, path=path)
    assert used == path, f'asked for {path}, answered by {used}'
    assert memory.read_mask(0x6FF8, 8) == FULL, (
        'the pushed word should be tainted where the stack pointer put it')


@pytest.mark.parametrize('path', PATHS)
def test_taint_survives_a_round_trip_through_the_stack(path: str) -> None:
    """The question this API exists for: push a tainted register, pop it into
    another, and the taint has to arrive."""
    seq = TaintSequence(Architecture.AMD64, path=path,
                        values={'RBX': 0xCAFEBABE, 'RSP': 0x7000, 'RAX': 0},
                        taint={'RBX': FULL})
    after = seq.run(PUSH_RBX, POP_RAX)
    assert set(seq.paths_used) == {path}, f'fell back: {seq.paths_used}'
    assert after['RAX'] == FULL, 'the taint did not come back out of the stack'
    assert seq.values['RAX'] == 0xCAFEBABE, 'the VALUE did not come back either'
    assert seq.values['RSP'] == 0x7000, 'push then pop must restore the stack pointer'


def test_the_two_paths_agree_about_memory() -> None:
    """Same sequence, both implementations, same answer -- registers AND the
    bytes they went through."""
    results = {}
    for path in PATHS:
        seq = TaintSequence(Architecture.AMD64, path=path,
                            values={'RBX': 0xCAFEBABE, 'RSP': 0x7000, 'RAX': 0},
                            taint={'RBX': FULL})
        after = seq.run(PUSH_RBX, POP_RAX)
        results[path] = (after.get('RAX'), seq.values.get('RAX'),
                         seq.memory.read_mask(0x6FF8, 8))
    assert results['compiled'] == results['differential'], (
        f'the two paths disagree about a stack round trip: {results}')


@pytest.mark.parametrize('path', PATHS)
def test_clean_memory_does_not_invent_taint(path: str) -> None:
    """The direction that matters for false positives: nothing tainted in,
    nothing tainted out."""
    seq = TaintSequence(Architecture.AMD64, path=path,
                        values={'RBX': 0xCAFEBABE, 'RSP': 0x7000, 'RAX': 0},
                        taint={})
    after = seq.run(PUSH_RBX, POP_RAX)
    assert not after.get('RAX'), f'{path} invented taint from a clean push: {after}'


def test_without_a_memory_a_store_falls_back_and_says_so() -> None:
    _out, used = explain(Architecture.AMD64, PUSH_RBX, {'RBX': FULL},
                         {'RBX': 1, 'RSP': 0x7000}, path='compiled')
    assert used == 'differential', (
        'a store was answered by the compiled path with no memory to store into')
