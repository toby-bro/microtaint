"""
Soundness regression tests for the CMOVcc intra-sequence flag taint bug
identified in report_1778576468.json (May 12 2026 benchmark run, id=7665).

  Pattern — CMOVcc after CMP in a multi-instruction sequence loses flag taint.

  ``xor edx, edx; cmp rax, rbx; mov rcx, 1; cmovl rdx, rcx``

  When T_RAX[62]=1 the CMP instruction produces T_SF=1 and T_OF=1 (the sign
  bit of the subtraction depends on RAX bit 62 via carry propagation).  The
  CMOVL condition is L = (SF != OF); when both flags are tainted the taken /
  not-taken decision is uncertain.  CMOVL copies RCX=1 when taken and leaves
  RDX=0 when not taken, so bit 0 of RDX is uncertain: T_RDX[0] = 1.

  Root cause: generate_static_rule checked for *any* CBRANCH in the full
  translated pcode block and fell back to a monolithic (single-circuit)
  representation whenever one was present.  CMOVcc instructions emit a CBRANCH
  internally (a forward skip to next_instr_addr that conditionally skips the
  register write).  This caused the entire sequence to be treated as one flat
  circuit whose assignments all read from the *input* taint snapshot — where
  T_SF = T_OF = 0 — rather than from the CMP-computed intermediate values.

  Fix: _is_architectural_branch() distinguishes real architectural branches
  (ram-space target != next_instr_addr) from CMOVcc skips (ram-space target ==
  next_instr_addr) and pcode-internal loops (const-space target).  Only real
  branches suppress chaining; CMOVcc sequences now use ChainedCircuit, which
  threads the CMP output taint (T_SF=1, T_OF=1) into the CMOVL input.

Each test uses 2^k Unicorn enumeration as ground truth and asserts
``microtaint_output ⊇ ground_truth`` (no under-tainted bits).
"""

# mypy: disable-error-code="no-untyped-def,no-untyped-call,attr-defined"

from __future__ import annotations

import itertools

import pytest
from unicorn import UC_ARCH_X86, UC_MODE_64
from unicorn.unicorn_py3 import Uc
from unicorn.x86_const import (
    UC_CPU_X86_BROADWELL,
    UC_X86_REG_RAX,
    UC_X86_REG_RBX,
    UC_X86_REG_RCX,
    UC_X86_REG_RDX,
    UC_X86_REG_RSP,
)

from microtaint.instrumentation.ast import EvalContext
from microtaint.instrumentation.ast import ChainedCircuit  # noqa: F401 (import to assert type)
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, Register

MASK64 = 0xFFFFFFFFFFFFFFFF

_REGS_GP = [
    Register('RAX', 64),
    Register('RBX', 64),
    Register('RCX', 64),
    Register('RDX', 64),
]
_SIM = CellSimulator(Architecture.AMD64)
_UC_REGS = {
    'RAX': UC_X86_REG_RAX,
    'RBX': UC_X86_REG_RBX,
    'RCX': UC_X86_REG_RCX,
    'RDX': UC_X86_REG_RDX,
}
_BASE_ADDR = 0x400000
_PAGE_SIZE = 0x1000
_STACK_BASE = 0x500000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _brute_force_gt(
    bytestring: bytes,
    state: dict[str, int],
    taint: dict[str, int],
) -> dict[str, int]:
    """Compute exact bit-level ground truth via 2^k Unicorn enumeration."""
    positions: list[tuple[str, int]] = []
    for reg, mask in taint.items():
        for bit in range(64):
            if (mask >> bit) & 1:
                positions.append((reg, bit))
    if len(positions) > 14:
        pytest.skip(f'k={len(positions)} exceeds enumeration budget')

    results: list[dict[str, int]] = []
    for assignment in itertools.product([0, 1], repeat=len(positions)):
        s = dict(state)
        for (reg, bit), val in zip(positions, assignment, strict=False):
            if val:
                s[reg] = (s[reg] ^ (1 << bit)) & MASK64
        uc = Uc(UC_ARCH_X86, UC_MODE_64)
        uc.ctl_set_cpu_model(UC_CPU_X86_BROADWELL)
        uc.mem_map(_BASE_ADDR, _PAGE_SIZE)
        uc.mem_map(_STACK_BASE, _PAGE_SIZE)
        uc.mem_write(_BASE_ADDR, bytestring + b'\x90' * 16)
        for reg in ('RAX', 'RBX', 'RCX', 'RDX'):
            uc.reg_write(_UC_REGS[reg], s.get(reg, 0))
        uc.reg_write(UC_X86_REG_RSP, _STACK_BASE + _PAGE_SIZE - 0x100)
        uc.emu_start(_BASE_ADDR, _BASE_ADDR + len(bytestring))
        results.append({r: uc.reg_read(_UC_REGS[r]) for r in ('RAX', 'RBX', 'RCX', 'RDX')})

    output_taint: dict[str, int] = {}
    for reg in ('RAX', 'RBX', 'RCX', 'RDX'):
        all_or = 0
        all_and = MASK64
        for r in results:
            all_or |= r[reg]
            all_and &= r[reg]
        output_taint[reg] = all_or ^ all_and
    return output_taint


def _eval_microtaint(
    bytestring: bytes,
    state: dict[str, int],
    taint: dict[str, int],
) -> dict[str, int]:
    circuit = generate_static_rule(Architecture.AMD64, bytestring, _REGS_GP)
    full_state = {r.name: state.get(r.name, 0) for r in _REGS_GP}
    full_taint = {r.name: taint.get(r.name, 0) for r in _REGS_GP}
    ctx = EvalContext(input_taint=full_taint, input_values=full_state, simulator=_SIM)
    raw = circuit.evaluate(ctx)
    return {r.name: raw.get(r.name, 0) & MASK64 for r in _REGS_GP}


def _assert_sound(
    label: str,
    bytestring: bytes,
    state: dict[str, int],
    taint: dict[str, int],
    check_regs: tuple[str, ...] = ('RAX', 'RBX', 'RCX', 'RDX'),
) -> None:
    gt = _brute_force_gt(bytestring, state, taint)
    mt = _eval_microtaint(bytestring, state, taint)
    misses = []
    for reg in check_regs:
        gv = gt.get(reg, 0)
        tv = mt.get(reg, 0)
        under = gv & ~tv & MASK64
        if under:
            misses.append(
                f'{reg}: missed=0x{under:016x}  GT=0x{gv:016x}  microtaint=0x{tv:016x}'
            )
    assert not misses, (
        f'[{label}] microtaint under-taints — must cover all GT taint bits.\n  '
        + '\n  '.join(misses)
    )


# ---------------------------------------------------------------------------
# Byte sequences
# ---------------------------------------------------------------------------

# xor edx,edx  (31 d2)
# cmp rax,rbx  (48 39 d8)
# mov rcx,1    (48 c7 c1 01 00 00 00)
# cmovl rdx,rcx (48 0f 4c d1)
_XOR_CMP_MOV_CMOVL = bytes.fromhex('31d24839d848c7c101000000480f4cd1')

# xor edx,edx  (31 d2)
# cmp rax,rbx  (48 39 d8)
# mov rcx,1    (48 c7 c1 01 00 00 00)
# cmovg rdx,rcx (48 0f 4f d1)  — tests the G (greater) condition
_XOR_CMP_MOV_CMOVG = bytes.fromhex('31d24839d848c7c101000000480f4fd1')

# xor edx,edx  (31 d2)
# cmp rax,rbx  (48 39 d8)
# mov rcx,1    (48 c7 c1 01 00 00 00)
# cmovz rdx,rcx (48 0f 44 d1)  — condition depends on ZF only
_XOR_CMP_MOV_CMOVZ = bytes.fromhex('31d24839d848c7c101000000480f44d1')


# ---------------------------------------------------------------------------
# Structural regression: the sequence must use ChainedCircuit, not monolithic
# ---------------------------------------------------------------------------

def test_cmovl_sequence_uses_chained_circuit() -> None:
    """The sequence must be split into a ChainedCircuit, not a monolithic LogicCircuit.

    Before the fix, CMOVcc's internal CBRANCH (a forward skip to next_instr_addr)
    caused generate_static_rule to return a flat LogicCircuit for the whole sequence.
    After the fix, only real architectural branches (JNE, JL …) suppress chaining;
    CMOVcc skips are recognised as intra-instruction and the sequence is correctly
    wrapped in a ChainedCircuit.
    """
    from microtaint.instrumentation.ast import ChainedCircuit as CC  # local import for clarity
    circuit = generate_static_rule(Architecture.AMD64, _XOR_CMP_MOV_CMOVL, _REGS_GP)
    assert isinstance(circuit, CC), (
        f'Expected ChainedCircuit, got {type(circuit).__name__}.  '
        'The CMOVcc internal CBRANCH must no longer suppress sequence chaining.'
    )


# ---------------------------------------------------------------------------
# Soundness: report id=7665 exact case
# ---------------------------------------------------------------------------

def test_cmovl_after_cmp_report_exact() -> None:
    """Report id=7665: xor edx,edx; cmp rax,rbx; mov rcx,1; cmovl rdx,rcx.

    T_RAX[62]=1 taints SF and OF via the CMP subtraction carry chain.
    The CMOVL condition (SF != OF) is therefore uncertain, so T_RDX[0]=1.
    Before the fix microtaint gave T_RDX=0 because the sequence was flattened
    into a monolithic circuit where T_SF=T_OF=0 at evaluation time.
    """
    _assert_sound(
        'cmovl_report_exact',
        _XOR_CMP_MOV_CMOVL,
        state={
            'RAX': 8016027559179816196,   # 0x6f3ea69826a78d04
            'RBX': 7926627367857641452,   # 0x6e010998100b57ec
            'RCX': 13702379345388424796,
            'RDX': 2122900774596652732,
        },
        taint={
            'RAX': 4611686018427387904,   # 0x4000000000000000 — bit 62
            'RBX': 0,
            'RCX': 17179869184,           # 0x400000000 — irrelevant for RDX output
            'RDX': 2199023255616,         # 0x20000000000 — irrelevant for RDX output
        },
        check_regs=('RDX',),
    )


def test_cmovl_only_tainted_bit_is_sign_bit() -> None:
    """Only RAX[63] tainted: affects SF but not OF, L condition still uncertain."""
    _assert_sound(
        'cmovl_sign_bit_only',
        _XOR_CMP_MOV_CMOVL,
        state={
            'RAX': 0x7FFFFFFFFFFFFFFF,  # positive, just below overflow
            'RBX': 0x0000000000000001,
            'RCX': 0,
            'RDX': 0,
        },
        taint={
            'RAX': 1 << 63,  # sign bit only: SF becomes uncertain
            'RBX': 0,
            'RCX': 0,
            'RDX': 0,
        },
        check_regs=('RDX',),
    )


def test_cmovl_taint_on_both_operands() -> None:
    """Both RAX and RBX tainted — flags and CMOVL output fully uncertain."""
    _assert_sound(
        'cmovl_both_operands',
        _XOR_CMP_MOV_CMOVL,
        state={
            'RAX': 0x1000000000000000,
            'RBX': 0x0FFFFFFFFFFFFFFF,
            'RCX': 0,
            'RDX': 0,
        },
        taint={
            'RAX': 1 << 62,
            'RBX': 1 << 62,
            'RCX': 0,
            'RDX': 0,
        },
        check_regs=('RDX',),
    )


def test_cmovl_clean_flags_clean_output() -> None:
    """No taint on RAX or RBX: flags are concrete, CMOVL output must be clean.

    This is a precision check — microtaint must NOT over-taint when the
    condition inputs are fully concrete.
    """
    circuit = generate_static_rule(Architecture.AMD64, _XOR_CMP_MOV_CMOVL, _REGS_GP)
    ctx = EvalContext(
        input_taint={'RCX': 1, 'RDX': 0, 'RAX': 0, 'RBX': 0},
        input_values={'RAX': 5, 'RBX': 3, 'RCX': 1, 'RDX': 0},
        simulator=_SIM,
    )
    out = circuit.evaluate(ctx)
    t_rdx = out.get('RDX', 0)
    # RAX(5) > RBX(3): L is False (concrete), CMOVL not taken, RDX stays 0.
    # T_RCX=1 is the source taint; since branch is not taken it should not flow.
    # The only acceptable T_RDX is 0 (no taint) or at most T_RCX passed through
    # conservatively — but T_RAX=T_RBX=0 so the condition taint is 0.
    # Soundness requires T_RDX >= GT; precision requires T_RDX is not wildly over.
    # GT: concrete condition -> T_RDX = 0. Allow T_RDX = 0.
    assert t_rdx == 0, (
        f'Clean-flag CMOVL should produce T_RDX=0, got {t_rdx:#x}.  '
        'Flag-condition taint must not leak when flags are concrete.'
    )


# ---------------------------------------------------------------------------
# Other CMOVcc conditions: the fix must work for all of them
# ---------------------------------------------------------------------------

def test_cmovg_after_cmp_flag_taint_propagates() -> None:
    """CMOVG: condition is G = (ZF==0 AND SF==OF). T_RAX taints SF -> T_RDX."""
    _assert_sound(
        'cmovg_sign_bit',
        _XOR_CMP_MOV_CMOVG,
        state={
            'RAX': 0x7FFFFFFFFFFFFFFF,
            'RBX': 0x0000000000000001,
            'RCX': 0,
            'RDX': 0,
        },
        taint={
            'RAX': 1 << 63,
            'RBX': 0,
            'RCX': 0,
            'RDX': 0,
        },
        check_regs=('RDX',),
    )


def test_cmovz_after_cmp_flag_taint_propagates() -> None:
    """CMOVZ: condition is ZF. T_RAX can taint ZF -> T_RDX."""
    _assert_sound(
        'cmovz_bit62',
        _XOR_CMP_MOV_CMOVZ,
        state={
            'RAX': 0x4000000000000001,  # differs from RBX only at bit 62 and bit 0
            'RBX': 0x4000000000000000,
            'RCX': 0,
            'RDX': 0,
        },
        taint={
            'RAX': 1 << 0,  # bit 0: RAX and RBX differ here, so ZF depends on it
            'RBX': 0,
            'RCX': 0,
            'RDX': 0,
        },
        check_regs=('RDX',),
    )


# ---------------------------------------------------------------------------
# Regression: real architectural branches must still be monolithic
# ---------------------------------------------------------------------------

def test_real_branch_still_monolithic() -> None:
    """A sequence containing a genuine JNE must remain a monolithic LogicCircuit.

    Real conditional branches have a ram-space CBRANCH target that differs from
    next_instr_addr.  These must NOT be wrapped in ChainedCircuit because splitting
    would lose the cross-instruction ZF -> RIP data path that the slicer traces.

    We use a backward JNE (target == start of the sequence) to ensure the CBRANCH
    destination is well outside [start, next_instr_addr).
    """
    from microtaint.instrumentation.ast import LogicCircuit

    # cmp rax, rbx  (48 39 d8)  — 3 bytes at 0x1000
    # jne -5        (75 fb)     — 2 bytes at 0x1003; target = 0x1003+2-5 = 0x1000
    # Sequence is 5 bytes; next_instr_addr = 0x1005.
    # CBRANCH target 0x1000 != 0x1005 → recognised as a real architectural branch.
    jne_backward = bytes.fromhex('4839d875fb')
    circuit = generate_static_rule(Architecture.AMD64, jne_backward, _REGS_GP)
    assert not hasattr(circuit, 'sub_circuits'), (
        f'Expected monolithic circuit for JNE-backward sequence, got {type(circuit).__name__}.  '
        'Real architectural branches (target != next_instr_addr) must suppress chaining.'
    )
