"""M3 regression: partition_slice cut-point analysis.

Structural properties (the byte-identical-circuit property is validated in M4,
once the driver consumes segments):

  * NONE policy returns a single segment equal to slice_backward (parity baseline).
  * CONSERVATIVE cuts x86 cmp at the subtraction result (a discarded unique that
    feeds the flags), and the flag segment reads it as a leaf instead of recomputing.
  * Segments are topologically ordered (a cut precedes its consumers) even when a
    segment reaches a cut through a chain of non-cut ops (PF via POPCOUNT), and the
    union of the segments' ops covers the whole slice.
  * PER_OP cuts at every produced varnode.
"""

# mypy: disable-error-code="no-untyped-def,no-untyped-call,attr-defined,union-attr"

from __future__ import annotations

import pytest

from microtaint.sleigh.lifter import get_context
from microtaint.sleigh.partition import CutPolicy, partition_slice
from microtaint.sleigh.slicer import get_varnode_id, slice_backward

ARCH = 'AMD64'
CMP = '4839d8'  # cmp rax, rbx


def _ops(hex_bytes: str):
    return get_context(ARCH).translate(bytes.fromhex(hex_bytes), 0x1000).ops


def _output_of(ops, opcode):
    """The output varnode of the (first) op with this opcode."""
    for op in ops:
        if op.opcode.name == opcode and op.output is not None:
            return op.output
    return None


def _reg_output_whose_slice_contains(ops, opcode):
    """A register output whose backward slice contains an op of `opcode`
    (e.g. PF is the register output whose slice contains POPCOUNT)."""
    for op in ops:
        o = op.output
        if o is None or o.space.name != 'register':
            continue
        if any(s.opcode.name == opcode for s in slice_backward(ops, o)):
            return o
    return None


def test_none_policy_equals_slice_backward():
    ops = _ops(CMP)
    sf = _output_of(ops, 'INT_SLESS')  # SF = t < 0
    assert sf is not None
    segs = partition_slice(ops, sf, CutPolicy.NONE)
    assert len(segs) == 1
    assert [id(o) for o in segs[0].ops] == [id(o) for o in slice_backward(ops, sf)]
    assert segs[0].cut_inputs == []


def test_conservative_cuts_cmp_at_subtraction_result():
    ops = _ops(CMP)
    sub_out = _output_of(ops, 'INT_SUB')
    sf = _output_of(ops, 'INT_SLESS')  # SF = (t < 0), reads the subtraction result
    assert sub_out is not None
    assert sf is not None
    sub_id = get_varnode_id(sub_out)

    segs = partition_slice(ops, sf, CutPolicy.CONSERVATIVE)
    assert len(segs) >= 2
    assert get_varnode_id(segs[-1].output) == get_varnode_id(sf)

    cut_outputs = {seg.output_id for seg in segs[:-1]}
    assert sub_id in cut_outputs, 'subtraction result must be a materialized cut'

    sf_seg = segs[-1]
    assert sub_id in {get_varnode_id(v) for v in sf_seg.cut_inputs}
    assert not any(o.opcode.name == 'INT_SUB' for o in sf_seg.ops), 'flag segment must not recompute the core'


def test_segments_topologically_ordered_and_cover_slice():
    ops = _ops(CMP)
    pf = _reg_output_whose_slice_contains(ops, 'POPCOUNT')  # PF: multi-cut chain
    assert pf is not None
    segs = partition_slice(ops, pf, CutPolicy.CONSERVATIVE)

    produced: set[str] = set()
    for seg in segs:
        for leaf in seg.cut_inputs:
            assert get_varnode_id(leaf) in produced, 'cut consumed before produced'
        produced.add(seg.output_id)

    whole_ids = {id(o) for o in slice_backward(ops, pf)}
    seg_ids = {id(o) for seg in segs for o in seg.ops}
    assert whole_ids <= seg_ids, 'segments must cover the whole slice'


def test_per_op_policy_cuts_every_produced_varnode():
    ops = _ops(CMP)
    pf = _reg_output_whose_slice_contains(ops, 'POPCOUNT')
    segs = partition_slice(ops, pf, CutPolicy.PER_OP)
    whole = slice_backward(ops, pf)
    produced = {get_varnode_id(o.output) for o in whole if o.output}
    produced.discard(get_varnode_id(pf))
    cut_outputs = {seg.output_id for seg in segs[:-1]}
    assert produced <= cut_outputs


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
