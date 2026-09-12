"""Regression: memory-RMW CF/OF must not read a slot the circuit has not written.

Found by the 2026-09-12 AE soundness campaign (6.3 M cases).  For a sequence like
``push rax; add qword ptr [rsp], rbx; pop rcx`` the whole thing compiles to ONE
circuit whose assignment order is

    ['RSP', 'CF', 'OF', 'SF', 'ZF', 'PF', 'RCX', 'MEM']

The exact closed-form flag terms resolve the stack LOAD into a ``MemoryOperand``,
but the STORE that fills that slot is assigned LAST, so CF and OF read it empty:
taint 0 AND value 0.  With ``b_val = 0``, ``CF = [~RBX <u 0]`` is false for every
input, which is why tainting RBX lost CF as well as tainting RAX.

The fix declines the exact term when a STORE precedes the LOAD in program order
and floors the declined signed overflow.  A single-instruction RMW loads BEFORE
it stores, so it keeps its exact zero-cell term -- asserted here so the fix is
not silently widened into a precision regression.
"""
from __future__ import annotations

import pytest

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, ImplicitTaintPolicy, Register

keystone = pytest.importorskip('keystone')

NAMES = ['RAX', 'RBX', 'RCX', 'RDX', 'CF', 'PF', 'ZF', 'SF', 'OF']
FMT = ([Register(n, 64) for n in NAMES[:4]]
       + [Register(f, 1) for f in NAMES[4:]] + [Register('RSP', 64)])
SP_VALUE = 0x204000


def _evaluate(asm: str, taint: dict[str, int]) -> dict[str, int]:
    ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
    code = bytes(ks.asm(asm)[0])
    rule = generate_static_rule(Architecture.AMD64, code, FMT)
    vals = dict.fromkeys(NAMES, 0)
    # A carry out of bit 63 that an interior tainted bit can toggle.
    vals['RAX'] = 0xE000_0000_0000_0000
    vals['RBX'] = 0x3000_0000_0000_0000
    vals['RSP'] = SP_VALUE
    tnt = dict.fromkeys(NAMES, 0)
    tnt.update(taint)
    ctx = EvalContext(input_taint=tnt, input_values=vals, simulator=CellSimulator(Architecture.AMD64),
                      implicit_policy=ImplicitTaintPolicy.IGNORE)
    return {n: int(v or 0) for n, v in rule.evaluate(ctx).items()}


@pytest.mark.parametrize('op', ['add', 'adc', 'xadd'])
@pytest.mark.parametrize('src', ['RAX', 'RBX'])
def test_memory_rmw_carry_is_tainted(op: str, src: str) -> None:
    """CF of a store-forwarded RMW must see the tainted operand."""
    res = _evaluate(f'push rax; {op} qword ptr [rsp], rbx; pop rcx', {src: 1 << 61})
    assert res['CF'], f'{op} CF lost the taint on {src} (read the slot before the STORE filled it)'


@pytest.mark.parametrize('op', ['add', 'adc', 'xadd'])
def test_memory_rmw_result_still_propagates(op: str) -> None:
    """The value taint was always right; guard against breaking it."""
    res = _evaluate(f'push rax; {op} qword ptr [rsp], rbx; pop rcx', {'RAX': 1 << 61})
    assert res['RCX'] & (1 << 61), f'{op} lost the result taint through memory'


def test_single_instruction_rmw_keeps_its_exact_term() -> None:
    """A lone RMW loads BEFORE it stores, so the decline must NOT fire for it.

    Guards the discriminator: the fix keys on a STORE PRECEDING the LOAD in
    program order, not on the mere presence of a STORE.  Widening it to any STORE
    would drop this form onto the differential and cost its zero-cell exactness.
    """
    ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
    code = bytes(ks.asm('add qword ptr [rsp], rbx')[0])
    rule = generate_static_rule(Architecture.AMD64, code, FMT)
    cf = next(a for a in rule.assignments if getattr(a.target, 'name', None) == 'CF')
    assert 'ComparisonTaintExpr' in repr(cf.expression), (
        'the lone RMW lost its exact carry term; the store-forwarding guard is too wide'
    )


# A witness the campaign recorded for the OF half of the bug.  The exact signed
# overflow term resolved the same unwritten slot, so tainting RAX's SIGN bit left
# OF clean.  Unlike the carry, signed overflow is NON-monotone, so declining alone
# is not enough -- the declined overflow also has to be floored.
_OF_WITNESS = {
    'asm': 'push rax; adc qword ptr [rsp], rbx; pop rcx',
    'state': {'RAX': 0xC43F97B2D6B954C4, 'RBX': 0x70A25D6F157097EF,
              'RCX': 0x26BCD2D0CFE04A17, 'RDX': 0x23358FE6FB1B9D4F,
              'CF': 1, 'PF': 1, 'ZF': 0, 'SF': 1, 'OF': 1},
    'taint': {'RAX': 1 << 63},
}


def test_memory_rmw_overflow_is_tainted() -> None:
    ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
    code = bytes(ks.asm(_OF_WITNESS['asm'])[0])
    rule = generate_static_rule(Architecture.AMD64, code, FMT)
    vals = dict.fromkeys(NAMES, 0)
    vals.update(_OF_WITNESS['state'])
    vals['RSP'] = SP_VALUE
    tnt = dict.fromkeys(NAMES, 0)
    tnt.update(_OF_WITNESS['taint'])
    ctx = EvalContext(input_taint=tnt, input_values=vals,
                      simulator=CellSimulator(Architecture.AMD64),
                      implicit_policy=ImplicitTaintPolicy.IGNORE)
    res = {n: int(v or 0) for n, v in rule.evaluate(ctx).items()}
    assert res['OF'], 'OF lost the taint on the sign bit of the store-forwarded operand'
