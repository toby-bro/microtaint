"""slice_simplify must be semantics-preserving and must collapse constant selects.

Correctness is checked by evaluating the ORIGINAL slice and the SIMPLIFIED slice
concretely on random inputs (a mini p-code evaluator over constfold._eval) and
requiring the target varnode to match on every trial. Effectiveness is checked
by requiring the op count to drop for constant-count x86 shifts (the dead
count==0 select branch must fold away).
"""

from __future__ import annotations

import random

import pypcode
import pytest

from microtaint.sleigh.constfold import _eval, _key
from microtaint.sleigh.slice_simplify import simplify_slice
from microtaint.sleigh.slicer import slice_backward

_CTX = pypcode.Context('x86:LE:64:default')

# (label, bytes, target register varnode (offset,size)) -- offsets from the
# x86 SLEIGH pcode dumps: EAX=0x0, CF=0x200, OF=0x20b, SF=0x207, ZF=0x206.
_CASES = [
    ('shl eax,7', 'c1e007', 0x0, 4),
    ('shl eax,7 CF', 'c1e007', 0x200, 1),
    ('shl eax,7 OF', 'c1e007', 0x20b, 1),
    ('shl eax,7 SF', 'c1e007', 0x207, 1),
    ('shl eax,7 ZF', 'c1e007', 0x206, 1),
    ('shr eax,3', 'c1e803', 0x0, 4),
    ('sar eax,5', 'c1f805', 0x0, 4),
    ('add eax,ebx', '01d8', 0x0, 4),
    ('shld r10d,r11d,cl', '440fa5da', 0x0, 4),
]


def _eval_slice(ops: list, env: dict[tuple, int]) -> dict[tuple, int]:
    """Concretely evaluate a slice; unknown reads come from `env`. Returns the
    value map (VNKey -> value). Ops constfold._eval cannot model raise."""
    vals: dict[tuple, int] = {}

    def read(vn: object) -> int:
        if vn.space.name == 'const':
            return vn.offset
        k = _key(vn)
        if k in vals:
            return vals[k]
        return env.get(k, 0)

    for op in ops:
        if op.output is None:
            continue
        ins = [read(i) for i in op.inputs]
        bits = [i.size * 8 for i in op.inputs]
        r = _eval(op.opcode.name, ins, bits, op.output.size * 8)
        if r is None:
            raise NotImplementedError(op.opcode.name)
        vals[_key(op.output)] = r
    return vals


def _input_keys(ops: list) -> set[tuple]:
    """Varnodes read but never defined in the slice (the free inputs)."""
    defined = {_key(o.output) for o in ops if o.output is not None}
    reads: set[tuple] = set()
    for o in ops:
        for i in o.inputs:
            if i.space.name != 'const':
                reads.add(_key(i))
    return reads - defined


@pytest.mark.parametrize(('label', 'hexbytes', 'off', 'size'), _CASES)
def test_simplify_preserves_semantics(label: str, hexbytes: str, off: int, size: int) -> None:
    ops = _CTX.translate(bytes.fromhex(hexbytes), 0x1000).ops
    from microtaint.sleigh.slice_simplify import _Vn  # noqa: PLC0415
    target = _Vn('register', off, size)
    sl = slice_backward(ops, target)
    simp = simplify_slice(sl)
    tkey = ('register', off, size)
    rng = random.Random(1234)
    for _ in range(200):
        env = {k: rng.getrandbits(k[2] * 8) for k in (_input_keys(sl) | _input_keys(simp))}
        v0 = _eval_slice(sl, env).get(tkey, 0)
        v1 = _eval_slice(simp, env).get(tkey, 0)
        assert v0 == v1, f'{label}: simplified slice diverged ({v0:#x} != {v1:#x}) env={env}'


def test_simplify_collapses_constant_shift_flag() -> None:
    """A constant-count shift's flag select must shrink (dead branch removed)."""
    ops = _CTX.translate(bytes.fromhex('c1e007'), 0x1000).ops  # shl eax,7
    from microtaint.sleigh.slice_simplify import _Vn  # noqa: PLC0415
    def real(ops: list) -> int:
        # COPY is pure routing (determine_category ignores it); count real work.
        return sum(1 for o in ops if o.opcode.name != 'COPY')
    for off in (0x200, 0x207, 0x206):  # CF, SF, ZF
        sl = slice_backward(ops, _Vn('register', off, 1))
        simp = simplify_slice(sl)
        assert real(simp) < real(sl), (
            f'flag @ {off:#x} did not shrink: real ops {real(sl)} -> {real(simp)}'
        )
        # the count==0 select (INT_AND/INT_OR combine) must be gone as real work
        assert not any(o.opcode.name in ('INT_AND', 'INT_OR') for o in simp), (
            f'flag @ {off:#x} still has a select as real work'
        )
