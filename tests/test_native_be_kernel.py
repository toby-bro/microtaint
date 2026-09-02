"""The native Cython p-code kernel is endianness-aware for sub-register and memory
byte layout.

`_PCodeFrame` stores each register's integer value in a byte-offset-indexed array
with byte `d` at bit `d*8` -- little-endian.  On a big-endian target that is wrong
for a *sub-register* access: SLEIGH emits BE-correct varnode offsets (e.g. PPC
`extsb` reads `register[r4+3:1]` = the low byte under BE; MIPS64 `sll` reads
`register[$4+4:4]` = the low word), but the LE kernel would read the byte/word at
the wrong significance.  `PCodeCellEvaluator` now carries an `_is_big_endian` flag
(from the arch) and inverts the sub-register byte->bit mapping and memory byte
order when set; the little-endian path is byte-identical.

This pins the byte math by asserting the native kernel agrees with Unicorn (the
trusted big-endian reference) on the concrete result of BE sub-register
instructions, over random inputs.  Concrete agreement implies differential
agreement (the differential is `f(V|T) XOR f(V&~T)`, two concrete evaluations).

Caveat encoded by `_canon`: MIPS 32-bit ops assume registers hold sign-extended
32-bit (canonical) values; feeding non-canonical high bits is architecturally
undefined and native/Unicorn may legitimately differ there, so the MIPS inputs are
canonicalised.  PPC GPRs are 32-bit and have no such precondition.
"""

# mypy: disable-error-code="no-untyped-def,no-untyped-call,attr-defined"

from __future__ import annotations

import random

from microtaint.instrumentation.ast import InstructionCellExpr
from microtaint.instrumentation.cell import PCodeCellEvaluator
from microtaint.simulator import CellSimulator, MachineState
from microtaint.types import Architecture


def _agrees(arch: Architecture, instr_hex: str, out_reg: str, out_bits: int,
            in_regs: list[str], canonical: bool, n: int = 150) -> None:
    native = PCodeCellEvaluator(arch)          # BE-aware native kernel
    uc = CellSimulator(arch)                    # BE forces use_unicorn -> reference
    ice = InstructionCellExpr(arch, instr_hex, out_reg, 0, out_bits - 1, {})
    rng = random.Random(0xBE)
    out_mask = (1 << out_bits) - 1

    def _canon(x: int) -> int:
        # MIPS: registers hold sign-extended 32-bit values.
        x &= 0xFFFFFFFF
        return x | 0xFFFFFFFF00000000 if x & 0x80000000 else x

    checked = 0
    for _ in range(n):
        regs = {r: (_canon(rng.getrandbits(32)) if canonical else rng.getrandbits(out_bits)) for r in in_regs}
        nat = native.evaluate_concrete_state(ice, dict(regs), {})
        try:
            state = MachineState(regs=dict(regs), mem={})
            uc._execute(bytes.fromhex(instr_hex), state)
            ref = uc._read_reg(out_reg) & out_mask
        except Exception:  # noqa: S112 -- Unicorn rejected this state, not a kernel concern
            continue
        checked += 1
        assert nat == ref, f'{instr_hex} regs={regs}: native={nat:#x} unicorn={ref:#x}'
    assert checked > 0, 'no comparable states -- Unicorn rejected all inputs'


def test_ppc_subregister_reads_match_unicorn():
    """PPC sign/shift ops read a sub-register byte of a GPR (`register[r4+N:1]`)."""
    ppc = Architecture.PPC32BE
    _agrees(ppc, '7c830774', 'R3', 32, ['R4'], canonical=False)  # extsb 3,4 (low byte)
    _agrees(ppc, '7c830734', 'R3', 32, ['R4'], canonical=False)  # extsh 3,4 (low half)
    _agrees(ppc, '7c832670', 'R3', 32, ['R4'], canonical=False)  # srawi 3,4,4


def test_mips_subregister_ops_match_unicorn():
    """MIPS64 32-bit ops read/write `register[GPR+4:4]` -- the low word under BE."""
    mips = Architecture.MIPS64BE
    _agrees(mips, '00041100', 'V0', 64, ['A0'], canonical=True)        # sll  $2,$4,4
    _agrees(mips, '00041103', 'V0', 64, ['A0'], canonical=True)        # sra  $2,$4,4
    _agrees(mips, '00851021', 'V0', 64, ['A0', 'A1'], canonical=True)  # addu $2,$4,$5


def test_mips_full_width_op_match_unicorn():
    """A 64-bit (maximal-register) op has no sub-register / canonical precondition."""
    _agrees(Architecture.MIPS64BE, '0085102d', 'V0', 64, ['A0', 'A1'], canonical=False)  # daddu


if __name__ == '__main__':
    import pytest

    raise SystemExit(pytest.main([__file__, '-v']))
