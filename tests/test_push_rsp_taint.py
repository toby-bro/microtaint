"""`push` and the stack pointer's own taint.

`push %rbp` computes RSP_out = RSP_in - 8 and stores RBP at the new address.
So RSP_out depends on RSP_in and not at all on RBP.  Both halves of that are
checked against Unicorn here rather than argued from the encoding, because the
whole-instruction differential gets both backwards: it drops RSP's own taint
(an under-taint, the one failure mode that is never acceptable) and invents RSP
taint from the pushed register.

Both evaluators get it right now.  The differential used to get both backwards
because it extracted dependencies per INSTRUCTION rather than per TARGET, so
`push`'s two outputs -- whose dependencies are disjoint -- shared one set; see
test_rsp_output_depends_on_rsp_not_on_the_pushed_register for the root cause.
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


def test_ground_truth_rsp_depends_on_rsp_not_rbp() -> None:
    """The premise, measured: flipping RSP moves RSP_out; RBP does not."""
    base = STACK + 0x8000
    moved = 0
    for bit in range(3, 12):
        moved |= _uc_push(base, 0) ^ _uc_push(base | (1 << bit), 0)
    assert moved, 'RSP_out does not depend on RSP_in -- premise is wrong'
    assert _uc_push(base, 0) == _uc_push(base, 0xFF), \
        'RSP_out depends on RBP -- premise is wrong'


def _answers(taint):
    from benchmark.instruction_bank import isa_registers
    from microtaint.types import Architecture
    from tests.oracle_harness import _uc_desc_amd64, build_circuit, reference_taint
    from tests.perop_c_bank import uc_initial_state
    from tests.taint_ir_bank import ir_step

    regs = list(isa_registers('AMD64'))
    names = [r.name for r in regs]
    in_taint = dict.fromkeys(names, 0)
    in_taint.update(taint)
    values = {n: STACK + 0x8000 if n == 'RSP' else 0x2000 + 8 * i
              for i, n in enumerate(names)}
    values.update(uc_initial_state(_uc_desc_amd64()))
    circuit = build_circuit(Architecture.AMD64, PUSH_RBP, regs)
    diff = reference_taint(Architecture.AMD64, PUSH_RBP, regs, in_taint, values,
                           circuit=circuit)
    lowered, _v, _c = ir_step(Architecture.AMD64, PUSH_RBP, regs, in_taint, values)
    return diff, lowered


def test_lowered_program_keeps_rsp_taint() -> None:
    _diff, lowered = _answers({'RSP': 0xF0F0F0F0F0F0F0F0})
    assert lowered['RSP'], "push must not clear the stack pointer's own taint"


def test_lowered_program_does_not_invent_rsp_taint_from_the_pushed_register() -> None:
    _diff, lowered = _answers({'RBP': 0xFF})
    assert lowered['RSP'] == 0, 'RSP_out does not depend on the pushed register'


def test_differential_keeps_rsp_taint() -> None:
    diff, _lowered = _answers({'RSP': 0xF0F0F0F0F0F0F0F0})
    assert diff['RSP'], "push must not clear the stack pointer's own taint"


@pytest.mark.xfail(strict=True, reason=(
    "The over-taint half is still open.  The differential extracts VALUE "
    "dependencies per INSTRUCTION, so the pushed register reaches the RSP "
    "target too.  Scoping that collection to the target's backward slice does "
    "fix it -- and MEASURED, it then UNDER-taints bsf/bsr/tzcnt (whose p-code "
    "loops put the dependency outside the slice) and the memory forms of "
    "sub-borrow and signed compare: 9 tests, on both MICROTAINT_TAINT_IR "
    "settings.  Over-tainting is the acceptable direction, so the narrow fix "
    "waits until the flag and memory cases are handled."))
def test_differential_does_not_invent_rsp_taint() -> None:
    diff, _lowered = _answers({'RBP': 0xFF})
    assert diff['RSP'] == 0


def _rsp_dep_set():
    """`extract_dependencies` for the RSP output of `push %rbp`."""
    from microtaint.emulator import archregs
    from microtaint.sleigh import engine as E
    from microtaint.sleigh.lifter import get_context
    from microtaint.types import Architecture

    ctx = get_context('AMD64')
    ops = ctx.translate(PUSH_RBP, CODE_ADDR).ops
    # RSP_out is written by the INT_SUB: `RSP = RSP - 8`.
    rsp_vn = next(o.output for o in ops if o.opcode.name == 'INT_SUB')
    sl = E.slice_backward(ops, rsp_vn)
    mapper = E.StateMapper(ctx, 'AMD64', list(archregs.state_format(Architecture.AMD64)))
    ds = E.extract_dependencies(rsp_vn, sl, E.compute_polarity(sl), ops, mapper)
    return ({getattr(k, 'name', k) for k in ds.value_deps},
            {getattr(k, 'name', k) for k in ds.addr_deps})


def test_the_rsp_slice_contains_only_the_subtraction() -> None:
    """The premise: RSP_out's backward slice is `RSP = RSP - 8` and nothing else.

    If this ever stops holding, the dependency expectations below are about a
    different program and mean nothing.
    """
    from microtaint.sleigh import engine as E
    from microtaint.sleigh.lifter import get_context

    ops = get_context('AMD64').translate(PUSH_RBP, CODE_ADDR).ops
    rsp_vn = next(o.output for o in ops if o.opcode.name == 'INT_SUB')
    names = [o.opcode.name for o in E.slice_backward(ops, rsp_vn)]
    assert names == ['INT_SUB'], f'the RSP slice is no longer just the subtract: {names}'


def test_rsp_output_depends_on_rsp_not_on_the_pushed_register() -> None:
    """The ROOT CAUSE, one level below the two xfails above.

    Dependencies were extracted per INSTRUCTION rather than per TARGET, so every
    output of `push` got the same set: the STORE's address register (RSP) was
    demoted to an address dependency and the STORE's value register (RBP) became
    the only value dependency -- for the RSP output too, whose slice contains
    neither.  That is both halves of the reported defect in one place.
    """
    value_deps, addr_deps = _rsp_dep_set()
    assert 'RSP' in value_deps, (
        f'RSP_out = RSP - 8, so RSP is a VALUE dependency; got value={value_deps} '
        f'addr={addr_deps}.  Classifying it as an address dependency is what '
        f"dropped the stack pointer's own taint: the STORE's pointer register "
        f"was collected for every target of the instruction, not just for the "
        f"store's own.")
