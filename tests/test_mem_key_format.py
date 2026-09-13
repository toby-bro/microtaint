"""The memory cell key means the same address to every one of its readers.

A cell key is a STRING protocol between the rule generator, which writes it, and
five readers, which turn it back into an address:

    _load and _read_output            microtaint/instrumentation/cell.pyx
    load_flat and read_output_full    microtaint/instrumentation/cell_c/cell_c.c
    CellSimulator._read_reg           microtaint/simulator.py

Three shapes travel through it:

    Format A   MEM_0x<addr>_<size>                       a compile-time address
    Format B   MEM_<reg>_<off>_<size>                    register-relative
    Format C   MEM_<base>+<idx>*<scale>_<off>_<size>     base+index

Format C exists because Format B had no room for an index, so every reader
rebuilt an indexed address as `reg + off` and dropped the index.  For an INPUT
that puts the masked value at the base rather than at the element, so the
replica reads unmodified bytes and the operand images as "this input does not
matter"; for an OUTPUT it reads the wrong bytes back.  Neither is visible as a
crash, and a reader that disagrees with the others is exactly how an index comes
to be honoured at one site and forgotten at another.

So these tests pin the written form, and drive the key back through the readers.
"""
from __future__ import annotations

import pytest

from microtaint.instrumentation.ast import InstructionCellExpr
from microtaint.instrumentation.cell import PCodeCellEvaluator
from microtaint.instrumentation.cell_c.cell_c import PCodeCellEvaluatorC
from microtaint.sleigh.engine import MemMapping, RegMapping, _mem_cell_key
from microtaint.types import Architecture

keystone = pytest.importorskip('keystone')

_RDI = RegMapping('RDI', 0, 63)
_RAX = RegMapping('RAX', 0, 63)

#: A high, page-aligned data address, so the arithmetic is exercised at 64 bits.
BASE = 0x0000_7FFF_0000_1000
IDX = 6


# ---------------------------------------------------------------------------
# What the generator writes.
# ---------------------------------------------------------------------------

def test_the_written_form_of_each_shape() -> None:
    """A key that does not need an index is byte-identical to Format B.

    The index part is emitted ONLY when there is an index, so adding Format C
    could not disturb any key already in use.
    """
    assert _mem_cell_key(MemMapping(0, 4, _RDI, 0)) == 'MEM_RDI_0_4'
    assert _mem_cell_key(MemMapping(0, 4, _RDI, -16)) == 'MEM_RDI_-16_4'
    assert _mem_cell_key(MemMapping(0, 4, _RDI, 0, _RAX, 1)) == 'MEM_RDI+RAX*1_0_4'
    assert _mem_cell_key(MemMapping(0, 4, _RDI, 0, _RAX, 4)) == 'MEM_RDI+RAX*4_0_4'
    assert _mem_cell_key(MemMapping(0, 1, _RDI, -16, _RAX, 8)) == 'MEM_RDI+RAX*8_-16_1'


# ---------------------------------------------------------------------------
# What the readers make of it.  Both kernels, both directions.
# ---------------------------------------------------------------------------

def _kernels() -> list:
    return [PCodeCellEvaluator(Architecture.AMD64), PCodeCellEvaluatorC(Architecture.AMD64)]


def _asm(text: str) -> str:
    ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
    return bytes(ks.asm(text)[0]).hex()


@pytest.mark.parametrize('scale', [1, 2, 4, 8])
@pytest.mark.parametrize('kernel_idx', [0, 1], ids=['cython', 'c'])
def test_an_input_key_places_the_value_at_base_plus_index(scale: int, kernel_idx: int) -> None:
    """`mov ecx,[rdi+rax*s]` must read the value the INPUT key carries.

    The key names base+index; the frame's RDI and RAX put the element at
    BASE+IDX*scale.  A reader that drops the index writes the value at RDI, the
    instruction reads untouched memory, and RCX comes back 0.
    """
    kernel = _kernels()[kernel_idx]
    code = _asm(f'mov ecx, dword ptr [rdi + rax*{scale}]')
    key = f'MEM_RDI+RAX*{scale}_0_4'
    cell = InstructionCellExpr(Architecture.AMD64, code, 'RCX', 0, 31, {})
    got = kernel.evaluate_concrete(cell, {
        'RDI': BASE, 'RAX': IDX, 'RCX': 0, key: 0xDEADBEEF,
    })
    assert got == 0xDEADBEEF, (
        f'the input key {key!r} did not place its value at RDI+RAX*{scale}; '
        f'the instruction read {got:#x}. A reader that rebuilds the address as '
        f'RDI+0 puts the value a whole {IDX * scale} bytes away from where the '
        f'instruction looks.'
    )


@pytest.mark.parametrize('kernel_idx', [0, 1], ids=['cython', 'c'])
def test_an_input_key_honours_the_displacement_alongside_the_index(kernel_idx: int) -> None:
    """base + index*scale + disp: every term, at once."""
    kernel = _kernels()[kernel_idx]
    code = _asm('mov ecx, dword ptr [rdi + rax*4 + 24]')
    cell = InstructionCellExpr(Architecture.AMD64, code, 'RCX', 0, 31, {})
    got = kernel.evaluate_concrete(cell, {
        'RDI': BASE, 'RAX': IDX, 'RCX': 0, 'MEM_RDI+RAX*4_24_4': 0x11223344,
    })
    assert got == 0x11223344, f'read {got:#x}; the displacement or the index was dropped'


@pytest.mark.parametrize('kernel_idx', [0, 1], ids=['cython', 'c'])
def test_an_output_key_reads_back_from_base_plus_index(kernel_idx: int) -> None:
    """`mov [rdi+rax*4],ecx` must be read back through the OUTPUT key.

    This is the other direction: the store writes at BASE+IDX*4, and the key
    names where to look.  A reader that drops the index looks at RDI and finds
    whatever was there instead.
    """
    kernel = _kernels()[kernel_idx]
    code = _asm('mov dword ptr [rdi + rax*4], ecx')
    cell = InstructionCellExpr(Architecture.AMD64, code, 'MEM_RDI+RAX*4_0_4', 0, 31, {})
    got = kernel.evaluate_concrete(cell, {'RDI': BASE, 'RAX': IDX, 'RCX': 0xCAFEF00D})
    assert got == 0xCAFEF00D, (
        f'read back {got:#x} through the output key; the store landed at '
        f'RDI+RAX*4 and the key must name that same address'
    )


@pytest.mark.parametrize('kernel_idx', [0, 1], ids=['cython', 'c'])
def test_a_key_without_an_index_is_unchanged(kernel_idx: int) -> None:
    """The control: Format B still means exactly what it meant."""
    kernel = _kernels()[kernel_idx]
    code = _asm('mov ecx, dword ptr [rdi + 16]')
    cell = InstructionCellExpr(Architecture.AMD64, code, 'RCX', 0, 31, {})
    got = kernel.evaluate_concrete(cell, {'RDI': BASE, 'RCX': 0, 'MEM_RDI_16_4': 0x5A5A5A5A})
    assert got == 0x5A5A5A5A


def test_the_two_kernels_agree() -> None:
    """The Cython and C readers must not drift apart on the same key."""
    code = _asm('mov ecx, dword ptr [rdi + rax*8]')
    cell = InstructionCellExpr(Architecture.AMD64, code, 'RCX', 0, 31, {})
    flat = {'RDI': BASE, 'RAX': IDX, 'RCX': 0, 'MEM_RDI+RAX*8_0_4': 0x0BADC0DE}
    answers = [k.evaluate_concrete(cell, dict(flat)) for k in _kernels()]
    assert answers[0] == answers[1] == 0x0BADC0DE, (
        f'the two cell kernels disagree on the same key: {answers}'
    )


# ---------------------------------------------------------------------------
# The fifth reader: CellSimulator._read_reg.
# ---------------------------------------------------------------------------

def test_the_unicorn_simulator_resolves_the_same_address() -> None:
    """`CellSimulator._read_reg` must compute base + index*scale + off too.

    It is the Unicorn-backed fallback and parses the key by hand, so it is the
    reader most likely to be forgotten.  The register reads and the memory read
    are stubbed, so this is about the arithmetic and nothing else.
    """
    from microtaint.simulator import CellSimulator

    sim = object.__new__(CellSimulator)
    seen: list[tuple[int, int]] = []
    regs = {'RDI': BASE, 'RAX': IDX}

    def fake_read_mem(addr: int, size: int) -> int:
        seen.append((addr, size))
        return 0

    real_read_reg = CellSimulator._read_reg
    sim._read_mem = fake_read_mem                                   # type: ignore[method-assign]
    sim._read_reg = lambda n: (regs[n] if n in regs                 # type: ignore[method-assign]
                               else real_read_reg(sim, n))

    real_read_reg(sim, 'MEM_RDI+RAX*4_24_4')
    assert seen == [(BASE + IDX * 4 + 24, 4)], (
        f'CellSimulator resolved the key to {seen}, expected '
        f'{[(BASE + IDX * 4 + 24, 4)]}'
    )


# ---------------------------------------------------------------------------
# An address with an index but NO base register.
# ---------------------------------------------------------------------------

def test_the_written_form_without_a_base() -> None:
    """`[rcx*8]` writes its base field as `0`.

    `0+` cannot be confused with the `0x` that marks a Format-A key, and every
    reader adds nothing for it rather than looking up a register named "0".
    """
    from microtaint.sleigh.engine import _NO_BASE_REG

    _rcx = RegMapping('RCX', 0, 63)
    assert _mem_cell_key(MemMapping(0, 4, _NO_BASE_REG, 0, _rcx, 8)) == 'MEM_0+RCX*8_0_4'
    assert _mem_cell_key(MemMapping(0, 1, _NO_BASE_REG, 32, _rcx, 4)) == 'MEM_0+RCX*4_32_1'


@pytest.mark.parametrize('kernel_idx', [0, 1], ids=['cython', 'c'])
def test_a_no_base_input_key_places_the_value_at_index_times_scale(kernel_idx: int) -> None:
    """The readers must resolve `MEM_0+RCX*8_0_4` to RCX*8."""
    kernel = _kernels()[kernel_idx]
    code = _asm('mov edx, dword ptr [rcx*8]')
    cell = InstructionCellExpr(Architecture.AMD64, code, 'RDX', 0, 31, {})
    got = kernel.evaluate_concrete(cell, {
        'RCX': BASE // 8, 'RDX': 0, 'MEM_0+RCX*8_0_4': 0x1234ABCD,
    })
    assert got == 0x1234ABCD, (
        f'read {got:#x}: the reader did not resolve a base-less key to '
        f'index*scale'
    )
