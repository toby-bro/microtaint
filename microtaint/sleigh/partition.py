"""Cut-point analysis for intermediate-taint materialization.

See docs/design/intermediate-taint-materialization.md.

The engine classifies one architectural output's whole backward slice under a
single category (microtaint/sleigh/mapper.py ``determine_category``).  A slice
that mixes an arithmetic *core* with a *flag/condition* extractor -- ``cmp``
computing SF/ZF/PF from a subtraction, ``cmp;cset``, ``cmpw;mfcr`` -- is forced
into one rule, which is right for part of it and fragile for the rest.

``partition_slice`` splits that slice into single-category **segments** joined at
materialized intermediate values.  Each segment is then classified and ruled
independently; the driver (engine.py) materializes each cut varnode's taint once
and threads it downstream.

Cut policy
----------
* ``NONE``          -- no cuts; one segment == the whole slice.  Reproduces the
                       current per-output behaviour exactly (used as the parity
                       baseline and the ``cutset = {}`` case in the design note).
* ``CONSERVATIVE``  -- production policy.  Cut only at genuine category
                       boundaries: a value produced by a non-condition op and
                       consumed by a condition/flag op (C2), or a value whose
                       consumers span more than one category (C1).  Never cuts
                       inside a maximal monotone/arithmetic region, so it can only
                       lose precision by over-tainting (it never does at a true
                       boundary) and never under-taints.
* ``PER_OP``        -- cut at every produced varnode.  This is the maximally
                       aggressive policy = per-op propagation; it over-taints on
                       reconvergent/cancelling dataflow and is NOT a production
                       setting.  Provided as the sound reference oracle / ablation
                       axis (see the design note and M5).

Overlap model
-------------
Producer/consumer matching is by exact varnode identity ``space:offset:size``.
The flag boundaries this targets (a discarded ``unique`` result feeding several
condition ops) are exact matches.  SUBPIECE/partial-overlap cutting is a
deliberate non-goal here -- the whole-slice ``slice_backward`` remains
overlap-aware, and the ``NONE`` policy defers to it verbatim, so nothing that
relies on overlap regresses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from pypcode.pypcode_native import PcodeOp, Varnode

from microtaint.classifier.categories import InstructionCategory
from microtaint.sleigh.mapper import determine_category
from microtaint.sleigh.slicer import get_varnode_id, slice_backward

# Ops that compute a condition / flag from a value.  A value produced by a
# NON-condition op and consumed by one of these sits at the arithmetic->flag
# boundary the cut targets (C2).
CONDITION_OPCODES: frozenset[str] = frozenset({
    'INT_EQUAL',
    'INT_NOTEQUAL',
    'INT_LESS',
    'INT_LESSEQUAL',
    'INT_SLESS',
    'INT_SLESSEQUAL',
    'INT_CARRY',
    'INT_SCARRY',
    'INT_SBORROW',
    'POPCOUNT',
    'LZCOUNT',
    'FLOAT_EQUAL',
    'FLOAT_NOTEQUAL',
    'FLOAT_LESS',
    'FLOAT_LESSEQUAL',
    'FLOAT_NAN',
})


class CutPolicy(str, Enum):
    NONE = 'none'
    CONSERVATIVE = 'conservative'
    PER_OP = 'per_op'


@dataclass
class Segment:
    """A single-category sub-DAG of a slice.

    ``output`` is the varnode this segment computes -- either the slice's
    architectural output, or a cut intermediate that downstream segments consume.
    ``ops`` is the backward slice for ``output`` truncated at cut varnodes.
    ``cut_inputs`` are the cut varnodes this segment reads as materialized leaves.
    """

    output: Varnode
    ops: list[PcodeOp]
    cut_inputs: list[Varnode] = field(default_factory=list)

    @property
    def output_id(self) -> str:
        return get_varnode_id(self.output)


def _op_category(op: PcodeOp) -> InstructionCategory:
    """Single-op category, tolerant of ops no category claims."""
    try:
        return determine_category([op])
    except ValueError:
        return InstructionCategory.MAPPED


def _producers(slice_ops: list[PcodeOp]) -> dict[str, PcodeOp]:
    """First defining op for each produced varnode id in the slice."""
    producer: dict[str, PcodeOp] = {}
    for op in slice_ops:
        if op.output is not None:
            producer.setdefault(get_varnode_id(op.output), op)
    return producer


def _consumers(slice_ops: list[PcodeOp]) -> dict[str, list[PcodeOp]]:
    """Ops that read each varnode id in the slice."""
    consumers: dict[str, list[PcodeOp]] = {}
    for op in slice_ops:
        for inp in op.inputs:
            if inp.space.name == 'const':
                continue
            consumers.setdefault(get_varnode_id(inp), []).append(op)
    return consumers


def _compute_cutset(
    slice_ops: list[PcodeOp],
    out_vn: Varnode,
    policy: CutPolicy,
) -> set[str]:
    """Varnode ids to materialize, per policy.  The architectural output itself is
    never a cut (it is the final segment's own terminal)."""
    if policy is CutPolicy.NONE:
        return set()

    producer = _producers(slice_ops)
    consumers = _consumers(slice_ops)
    out_id = get_varnode_id(out_vn)
    cutset: set[str] = set()

    for vid, prod in producer.items():
        if vid == out_id:
            continue
        cons = consumers.get(vid, [])
        if not cons:
            continue

        if policy is CutPolicy.PER_OP:
            cutset.add(vid)
            continue

        # CONSERVATIVE: cut only at a genuine category boundary.
        prod_is_condition = prod.opcode.name in CONDITION_OPCODES
        cons_cats = {_op_category(c) for c in cons}
        has_condition_consumer = any(c.opcode.name in CONDITION_OPCODES for c in cons)

        # C2: an arithmetic/routing result reused by a condition/flag op.
        c2 = (not prod_is_condition) and has_condition_consumer
        # C1: consumers span more than one category (heterogeneous fan-out).
        c1 = len(cons_cats) > 1
        if c1 or c2:
            cutset.add(vid)

    return cutset


def _slice_until_cuts(  # noqa: C901
    slice_ops: list[PcodeOp],
    boundary: Varnode,
    cutset: set[str],
) -> Segment:
    """Backward slice for ``boundary`` over ``slice_ops``, stopping at any cut
    varnode (recorded as a materialized leaf instead of being recursed into).

    Overlap-aware like ``slice_backward``, but a worklist range is dropped -- and
    the varnode recorded as a cut input -- as soon as it names a cut id.
    """
    def _rng(vn: Varnode) -> tuple[str, int, int]:
        return (vn.space.name, vn.offset, vn.offset + vn.size)

    worklist: list[tuple[str, int, int]] = [_rng(boundary)]
    seg_ops: list[PcodeOp] = []
    cut_inputs: dict[str, Varnode] = {}

    def _overlaps_any(rng: tuple[str, int, int]) -> bool:
        ospace, ostart, oend = rng
        for wspace, wstart, wend in worklist:
            if wspace == ospace and wstart < oend and ostart < wend:
                return True
        return False

    boundary_id = get_varnode_id(boundary)
    for op in reversed(slice_ops):
        if op.output is None:
            continue
        if not _overlaps_any(_rng(op.output)):
            continue
        seg_ops.append(op)
        for inp in op.inputs:
            if inp.space.name == 'const':
                continue
            iid = get_varnode_id(inp)
            # A cut varnode is a materialized leaf: record it, do not recurse.
            # (The boundary itself may equal a cut id when it is an intermediate
            # segment's own output -- that must still be produced, not stopped.)
            if iid in cutset and iid != boundary_id:
                cut_inputs[iid] = inp
                continue
            worklist.append(_rng(inp))

    return Segment(
        output=boundary,
        ops=list(reversed(seg_ops)),
        cut_inputs=list(cut_inputs.values()),
    )


def _topo_sort_segments(segments: list[Segment]) -> list[Segment]:
    """Order segments so each appears after every segment producing a cut it reads.

    Dependencies are the segments' actual ``cut_inputs`` -- a segment may consume a
    cut through a chain of non-cut ops (e.g. PF's segment reaches the AND-result
    cut through POPCOUNT), so ordering must follow the materialized leaves, not the
    producers' direct inputs.
    """
    by_output: dict[str, Segment] = {seg.output_id: seg for seg in segments}
    order: list[Segment] = []
    seen: set[str] = set()

    def visit(seg: Segment) -> None:
        if seg.output_id in seen:
            return
        seen.add(seg.output_id)
        for leaf in seg.cut_inputs:
            dep = by_output.get(get_varnode_id(leaf))
            if dep is not None:
                visit(dep)
        order.append(seg)

    for seg in segments:
        visit(seg)
    return order


def partition_slice(
    ops: list[PcodeOp],
    out_vn: Varnode,
    policy: CutPolicy = CutPolicy.CONSERVATIVE,
) -> list[Segment]:
    """Partition the backward slice of ``out_vn`` into single-category segments.

    Returns segments in topological order: every cut intermediate precedes the
    segments that consume it, and the architectural output's segment is last.
    ``policy is CutPolicy.NONE`` returns a single segment equal to the whole
    ``slice_backward`` result (byte-for-byte parity with today).
    """
    whole = slice_backward(ops, out_vn)
    if policy is CutPolicy.NONE or not whole:
        return [Segment(output=out_vn, ops=whole, cut_inputs=[])]

    cutset = _compute_cutset(whole, out_vn, policy)
    if not cutset:
        return [Segment(output=out_vn, ops=whole, cut_inputs=[])]

    producer = _producers(whole)
    segments: list[Segment] = []
    for vid in cutset:
        prod = producer.get(vid)
        if prod is None or prod.output is None:
            continue
        segments.append(_slice_until_cuts(whole, prod.output, cutset))
    segments.append(_slice_until_cuts(whole, out_vn, cutset))
    return _topo_sort_segments(segments)
