"""`push` and the stack pointer's own taint.

`push %rbp` computes RSP_out = RSP_in - 8 and stores RBP at the new address.
So RSP_out depends on RSP_in and not at all on RBP.  Both halves of that are
checked against Unicorn here rather than argued from the encoding, because the
whole-instruction differential gets both backwards: it drops RSP's own taint
(an under-taint, the one failure mode that is never acceptable) and invents RSP
taint from the pushed register.

The lowered taint program gets both right, so the two evaluators genuinely
disagree on this instruction.  The differential's behaviour is pinned as xfail
rather than deleted: it is a real defect, and a test that quietly accepted it
would let it survive the next rewrite.
"""
# ruff: noqa: PLC0415
from __future__ import annotations

import pytest

CODE_ADDR = 0x1000
STACK = 0x50000
PUSH_RBP = bytes.fromhex('55')


def _uc_push(rsp: int, rbp: int) -> int:
    import unicorn
    import unicorn.x86_const as ux
    uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
    uc.mem_map(CODE_ADDR, 0x1000)
    uc.mem_map(STACK, 0x10000)
    uc.mem_write(CODE_ADDR, PUSH_RBP)
    uc.reg_write(ux.UC_X86_REG_RSP, rsp)
    uc.reg_write(ux.UC_X86_REG_RBP, rbp)
    uc.emu_start(CODE_ADDR, CODE_ADDR + len(PUSH_RBP))
    return uc.reg_read(ux.UC_X86_REG_RSP)


def test_ground_truth_rsp_depends_on_rsp_not_rbp():
    """The premise, measured: flipping RSP moves RSP_out; RBP does not."""
    base = STACK + 0x8000
    moved = 0
    for bit in range(3, 12):
        moved |= _uc_push(base, 0) ^ _uc_push(base | (1 << bit), 0)
    assert moved, 'RSP_out does not depend on RSP_in -- premise is wrong'
    assert _uc_push(base, 0) == _uc_push(base, 0xFF), \
        'RSP_out depends on RBP -- premise is wrong'


def _answers(taint):
    from microtaint.types import Architecture
    from benchmark.instruction_bank import isa_registers
    from tests.oracle_harness import build_circuit, reference_taint
    from tests.perop_c_bank import uc_initial_state
    from tests.oracle_harness import _uc_desc_amd64
    from tests.taint_ir_bank import ir_step

    regs = list(isa_registers('AMD64'))
    names = [r.name for r in regs]
    in_taint = {n: 0 for n in names}
    in_taint.update(taint)
    values = {n: STACK + 0x8000 if n == 'RSP' else 0x2000 + 8 * i
              for i, n in enumerate(names)}
    values.update(uc_initial_state(_uc_desc_amd64()))
    circuit = build_circuit(Architecture.AMD64, PUSH_RBP, regs)
    diff = reference_taint(Architecture.AMD64, PUSH_RBP, regs, in_taint, values,
                           circuit=circuit)
    lowered, _v, _c = ir_step(Architecture.AMD64, PUSH_RBP, regs, in_taint, values)
    return diff, lowered


def test_lowered_program_keeps_rsp_taint():
    _diff, lowered = _answers({'RSP': 0xF0F0F0F0F0F0F0F0})
    assert lowered['RSP'], 'push must not clear the stack pointer\'s own taint'


def test_lowered_program_does_not_invent_rsp_taint_from_the_pushed_register():
    _diff, lowered = _answers({'RBP': 0xFF})
    assert lowered['RSP'] == 0, 'RSP_out does not depend on the pushed register'


@pytest.mark.xfail(reason=(
    'The whole-instruction differential perturbs only RBP for the RSP target, '
    'so RSP\'s own taint never reaches RSP_out. Confirmed against Unicorn as a '
    'real under-taint; fixing it in the rule generator changes taint results '
    'across the suite and needs its own gate run.'), strict=True)
def test_differential_keeps_rsp_taint():
    diff, _lowered = _answers({'RSP': 0xF0F0F0F0F0F0F0F0})
    assert diff['RSP'], 'push must not clear the stack pointer\'s own taint'


@pytest.mark.xfail(reason=(
    'The same slicing mistake in the other direction: the differential taints '
    'RSP from the pushed register, which RSP_out does not depend on.'),
    strict=True)
def test_differential_does_not_invent_rsp_taint():
    diff, _lowered = _answers({'RBP': 0xFF})
    assert diff['RSP'] == 0
