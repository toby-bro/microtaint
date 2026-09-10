"""Regression: a standalone load from EXTERNAL tainted memory must taint its
destination, on every ISA.

Guards the bug where MIPS `lw/ld/lhu` dropped external memory taint (dest
untainted) because resolve_ptr_with_offset resolved a multiply-defined address
temp to its FIRST definition (COPY const 0) instead of the LAST-def-before-use
(the real base+offset).  MIPS lifts `lw` as: write addr-temp = COPY(const 0),
then addr-temp = COPY(base+offset), then LOAD addr-temp -- so first-def read the
pointer as 0 and emitted no memory dependency.

The bank's memory instructions are store->load round-trips (internal memory), so
they never exercised a standalone external-memory load; this test does.
"""
from __future__ import annotations

import pytest

from microtaint.emulator.shadow import BitPreciseShadowMemory
from microtaint.instrumentation.ast import EvalContext
from microtaint.instrumentation.cell import _build_reg_maps
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, ImplicitTaintPolicy, Register

ks = pytest.importorskip('keystone')

BASE = 0x4000
# Alternating tainted bytes at [BASE, BASE+8): byte0=FF byte1=00 byte2=FF ...
MEM_TAINT_BYTES = [0xFF if (i % 2 == 0) else 0x00 for i in range(8)]
MEM_VALUE = 0x1122334455667788


def _reader(addr: int, size: int) -> int:
    out = 0
    for i in range(size):
        off = addr + i - BASE
        if 0 <= off < 8:
            out |= ((MEM_VALUE >> (off * 8)) & 0xFF) << (i * 8)
    return out


def _shadow() -> BitPreciseShadowMemory:
    sh = BitPreciseShadowMemory()
    for i, b in enumerate(MEM_TAINT_BYTES):
        if b:
            sh.write_mask(BASE + i, b, 1)
    return sh


def _regs(arch: Architecture, names: list[str]) -> list[Register]:
    _offs, sizes = _build_reg_maps(arch)
    return [Register(n, sizes[n] * 8) for n in names if n in sizes]


# (arch, keystone arch, keystone mode, asm, base-reg name, dest-reg name, min_popcount)
CASES = [
    # MIPS64BE: the regression.  $4=A0 base, $2=V0 dest.
    (Architecture.MIPS64BE, ks.KS_ARCH_MIPS, ks.KS_MODE_MIPS64 | ks.KS_MODE_BIG_ENDIAN,
     'lw $2, 0($4)', 'A0', 'V0', 8),
    (Architecture.MIPS64BE, ks.KS_ARCH_MIPS, ks.KS_MODE_MIPS64 | ks.KS_MODE_BIG_ENDIAN,
     'ld $2, 0($4)', 'A0', 'V0', 16),
    (Architecture.MIPS64BE, ks.KS_ARCH_MIPS, ks.KS_MODE_MIPS64 | ks.KS_MODE_BIG_ENDIAN,
     'lhu $2, 0($4)', 'A0', 'V0', 4),
    # Cross-ISA parity (these already worked; guard they still do).
    (Architecture.PPC32BE, ks.KS_ARCH_PPC, ks.KS_MODE_PPC32 | ks.KS_MODE_BIG_ENDIAN,
     'lwz 3, 0(4)', 'R4', 'R3', 8),
    (Architecture.ARM64, ks.KS_ARCH_ARM64, ks.KS_MODE_LITTLE_ENDIAN,
     'ldr x0, [x1]', 'X1', 'X0', 8),
]

_MIPS_GPRS = (
    ['ZERO', 'AT', 'V0', 'V1', 'A0', 'A1', 'A2', 'A3']
    + [f'T{i}' for i in range(10)]
    + [f'S{i}' for i in range(8)]
)
_PPC_GPRS = [f'R{i}' for i in range(16)]
_ARM_GPRS = [f'X{i}' for i in range(16)]
_GPRS = {Architecture.MIPS64BE: _MIPS_GPRS, Architecture.PPC32BE: _PPC_GPRS, Architecture.ARM64: _ARM_GPRS}


@pytest.mark.parametrize(('arch', 'ksa', 'ksm', 'asm', 'base', 'dest', 'min_pop'), CASES)
def test_external_load_taints_destination(
    arch: Architecture, ksa: int, ksm: int, asm: str, base: str, dest: str, min_pop: int,
) -> None:
    bs = bytes(ks.Ks(ksa, ksm).asm(asm, 0)[0])
    regs = _regs(arch, _GPRS[arch])
    circ = generate_static_rule(arch, bs, regs)
    sim = CellSimulator(arch)
    ectx = EvalContext(input_values={base: BASE}, input_taint={}, simulator=sim,
                      implicit_policy=ImplicitTaintPolicy.KEEP,
                      shadow_memory=_shadow(), mem_reader=_reader)
    res = {k: v for k, v in circ.evaluate(ectx).items() if v}
    assert dest in res, f'{arch.name} {asm!r}: {dest} got no taint (external memory taint dropped): {res}'
    assert bin(res[dest]).count('1') >= min_pop, (
        f'{arch.name} {asm!r}: {dest} taint {res[dest]:#x} popcount '
        f'{bin(res[dest]).count("1")} < {min_pop}')
