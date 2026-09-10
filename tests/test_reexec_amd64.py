"""The AMD64 native re-execution path must match SLEIGH concrete execution.

For a set of register-only AMD64 instructions with random inputs, executing the
instruction on the host CPU (microtaint.reexec.NativeReExec) must produce the
SAME result register and flags as microtaint's SLEIGH cell.  Only DEFINED
outputs are compared (the curated instructions have fully-defined OSZAPC), so
this is an exact gate.  Skipped off x86_64 or when a C compiler is unavailable.
"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING

import pytest

from microtaint.reexec import AVAILABLE, FLAG_BITS, NativeReExec

if TYPE_CHECKING:
    from benchmark.instruction_bank import ISASpec
    from microtaint.simulator import CellSimulator

pytestmark = pytest.mark.skipif(not AVAILABLE, reason='native re-exec needs an x86_64 host with cc')

# Curated register-only AMD64 instructions with FULLY-DEFINED OSZAPC flags
# (no count-dependent shifts, so no undefined OF).  (label, bytes, out_reg)
_CASES = [
    ('add rax, rbx', bytes.fromhex('4801d8'), 'RAX'),
    ('sub rax, rbx', bytes.fromhex('4829d8'), 'RAX'),
    ('and rcx, rdx', bytes.fromhex('4821d1'), 'RCX'),
    ('or  rsi, rdi', bytes.fromhex('4809fe'), 'RSI'),
    ('xor r8, r9',   bytes.fromhex('4d31c8'), 'R8'),
    ('inc rax',      bytes.fromhex('48ffc0'), 'RAX'),
    ('dec rbx',      bytes.fromhex('48ffcb'), 'RBX'),
    ('neg rdx',      bytes.fromhex('48f7da'), 'RDX'),
    ('imul rax, rbx', bytes.fromhex('480fafc3'), 'RAX'),
    ('add eax, ebx', bytes.fromhex('01d8'), 'RAX'),
]


@pytest.fixture(scope='module')
def rx() -> NativeReExec:
    return NativeReExec()


@pytest.fixture(scope='module')
def bank_sim() -> tuple[ISASpec, CellSimulator]:
    from benchmark.instruction_bank import load_bank  # noqa: PLC0415
    from microtaint.simulator import CellSimulator  # noqa: PLC0415
    bank = load_bank(isas={'AMD64'})['AMD64']
    return bank, CellSimulator(bank.arch)


@pytest.mark.parametrize(('label', 'code', 'out_reg'), _CASES)
def test_reexec_matches_sleigh(rx: NativeReExec,
                               bank_sim: tuple[ISASpec, CellSimulator],
                               label: str, code: bytes, out_reg: str) -> None:
    from microtaint.instrumentation.ast import InstructionCellExpr  # noqa: PLC0415
    from microtaint.simulator import MachineState  # noqa: PLC0415
    from microtaint.sleigh.engine import generate_static_rule  # noqa: PLC0415

    bank, sim = bank_sim
    reg_names = [r.name for r in bank.regs]
    circ = generate_static_rule(bank.arch, code, bank.regs)
    flag_outs = [nm for a in circ.assignments
                 if (nm := getattr(a.target, 'name', None)) in FLAG_BITS]

    rng = random.Random(hash(label) & 0xFFFF)
    for _ in range(20):
        regs = {n: rng.getrandbits(64) for n in bank_reg_names(bank)}
        out = rx.run(code, regs, 0x202)
        assert out is not None, f'{label}: re-exec faulted'
        out_gpr, out_rflags = out
        ms = MachineState(regs={n: regs.get(n, 0) for n in reg_names}, mem={})

        ice = InstructionCellExpr(bank.arch, code.hex(), out_reg, 0, 63, {})
        sleigh_res = sim.evaluate_concrete(ice, ms) & 0xFFFFFFFFFFFFFFFF
        assert out_gpr[out_reg] == sleigh_res, (
            f'{label}: {out_reg} reexec={out_gpr[out_reg]:#x} sleigh={sleigh_res:#x}'
        )
        for fn in flag_outs:
            ice = InstructionCellExpr(bank.arch, code.hex(), fn, 0, 0, {})
            sleigh_flag = sim.evaluate_concrete(ice, ms) & 1
            assert NativeReExec.flag(out_rflags, fn) == sleigh_flag, (
                f'{label}: {fn} reexec={NativeReExec.flag(out_rflags, fn)} sleigh={sleigh_flag}'
            )


def bank_reg_names(bank: ISASpec) -> list[str]:
    from microtaint.reexec import REG_ORDER  # noqa: PLC0415
    have = {r.name for r in bank.regs}
    return [n for n in REG_ORDER if n in have]
