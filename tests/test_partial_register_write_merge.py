"""
Regression test for partial-register write visibility.

Background
----------
Reproducer: ``mov ah, bh`` (id=2199 in report_1778076700.json).

Bug: the C-pcode evaluator (``cell_c.c::frame_read_reg``) and its Cython
counterpart (``cell.pyx::_read_reg``) used to look up only PARENT registers
that contain the requested offset.  After a sub-register write (e.g. ``mov
ah, bh`` writes byte 1 of RAX at offset 0x1, size 1), reading the parent
register (RAX at offset 0x0, size 8) walked backwards from offset 0 — found
the original RAX value still present in the frame from the input setup —
and returned that, **silently dropping the partial write**.

Fix: ``frame_read_reg`` now also scans FORWARD over the read range to
overlay any sub-register writes onto the parent's base value, with the
written byte taking precedence over the original byte.

This test pins the fix at the simulator level — both the native C-pcode
path and the Unicorn reference path must agree on every interesting
sub-register write pattern.
"""
from __future__ import annotations

import pytest

from microtaint.instrumentation.ast import InstructionCellExpr
from microtaint.simulator import CellSimulator, MachineState
from microtaint.types import Architecture

_SIM = CellSimulator(Architecture.AMD64)


def _eval(use_unicorn: bool, bytestring: str, regs_in: dict[str, int],
          out_reg: str, bit_start: int, bit_end: int) -> int:
    """Evaluate a single concrete cell on either the C-pcode or Unicorn
    backend.  Returns the masked output bits."""
    cell = InstructionCellExpr(
        architecture=Architecture.AMD64,
        instruction=bytestring,
        out_reg=out_reg,
        out_bit_start=bit_start,
        out_bit_end=bit_end,
        inputs={},
    )
    state = MachineState(regs=regs_in, mem={})
    saved = _SIM.use_unicorn
    try:
        _SIM.use_unicorn = use_unicorn
        return _SIM.evaluate_concrete(cell, state)
    finally:
        _SIM.use_unicorn = saved


def _assert_c_matches_unicorn(
    label: str,
    bytestring: str,
    regs_in: dict[str, int],
    out_reg: str,
    bit_start: int,
    bit_end: int,
) -> None:
    """Both the C and Unicorn backends must give the same result.  This
    is the strongest possible assertion: Unicorn is the gold-standard
    for x86 semantics, so ``c == unicorn`` means the C path got it
    right."""
    c_out = _eval(False, bytestring, regs_in, out_reg, bit_start, bit_end)
    u_out = _eval(True, bytestring, regs_in, out_reg, bit_start, bit_end)
    assert c_out == u_out, (
        f'{label}: C-pcode evaluator disagrees with Unicorn.\n'
        f'  bytestring={bytestring}  regs_in={regs_in}\n'
        f'  out={out_reg}[{bit_end}:{bit_start}]\n'
        f'  C       = {hex(c_out)}\n'
        f'  Unicorn = {hex(u_out)}'
    )


# =============================================================================
# mov ah, bh — the original failing case
# =============================================================================

_MOV_AH_BH = '88fc'


@pytest.mark.parametrize(
    'rbx_value',
    [
        0x0,
        0xFF,                  # BL all-1, BH all-0 — must produce AH=0
        0x100,                 # BH=0x01, lowest BH bit → AH bit 0
        0x8000,                # BH=0x80, highest BH bit → AH bit 7
        0xFF00,                # BH=0xFF, all BH bits → AH all-1
        0xABCD,                # BH=0xAB
        0x1234567890ABCDEF,    # full register
        0x4004000000008000,    # T_RBX from id=2199 — only bit 15 in BH
    ],
)
def test_mov_ah_bh_c_matches_unicorn(rbx_value: int) -> None:
    """``mov ah, bh`` reads BH (RBX byte 1) and writes AH (RAX byte 1).
    The C evaluator must merge the AH write into the RAX read so that
    reading the full 8-byte RAX after the COPY shows the new AH byte."""
    _assert_c_matches_unicorn(
        f'mov ah, bh with RBX={hex(rbx_value)}',
        bytestring=_MOV_AH_BH,
        regs_in={'RAX': 0, 'RBX': rbx_value},
        out_reg='RAX', bit_start=0, bit_end=63,
    )


def test_mov_ah_bh_preserves_other_rax_bytes() -> None:
    """Writing AH must NOT clobber the other 7 bytes of RAX.  This is
    the second half of the partial-write contract: the merge must be
    selective (overlay only the written bytes, leave the rest)."""
    rax_in = 0xDEADBEEFCAFEBABE
    rbx_in = 0x4242  # BH=0x42
    out = _eval(False, _MOV_AH_BH, {'RAX': rax_in, 'RBX': rbx_in},
                'RAX', 0, 63)
    # Expected: RAX low byte unchanged (0xBE), AH=0x42, top 6 bytes unchanged
    expected = (rax_in & ~0xFF00) | (0x42 << 8)
    assert out == expected, (
        f'AH write clobbered other RAX bytes: '
        f'expected {hex(expected)}, got {hex(out)}'
    )
    # And it must agree with Unicorn (sanity)
    u_out = _eval(True, _MOV_AH_BH, {'RAX': rax_in, 'RBX': rbx_in},
                  'RAX', 0, 63)
    assert out == u_out, f'C={hex(out)} vs Unicorn={hex(u_out)}'


# =============================================================================
# Other partial-write patterns — make sure we didn't break siblings
# =============================================================================

@pytest.mark.parametrize(
    ('label', 'bytestring', 'regs_in'),
    [
        # mov al, bl  — write byte 0 of RAX from byte 0 of RBX
        ('mov al, bl',     '88d8', {'RAX': 0xCAFEBABE_DEADBEEF, 'RBX': 0xAB}),
        # mov ax, bx  — write low 2 bytes of RAX from low 2 of RBX
        ('mov ax, bx',     '6689d8', {'RAX': 0xCAFEBABE_DEADBEEF, 'RBX': 0x1234}),
        # mov eax, ebx — writes low 4 of RAX, ZERO-extends top 4 (x86-64 spec)
        ('mov eax, ebx',   '89d8',   {'RAX': 0xCAFEBABE_DEADBEEF, 'RBX': 0xFFFFFFFF_12345678}),
        # mov bh, al — read AL, write BH
        ('mov bh, al',     '88c7',   {'RAX': 0xAB, 'RBX': 0xDEADBEEF_CAFEBABE}),
        # add ah, bh — RMW of AH using BH as source
        ('add ah, bh',     '00fc',   {'RAX': 0x500, 'RBX': 0x100}),
        # xor al, bl — RMW with XOR
        ('xor al, bl',     '30d8',   {'RAX': 0xFF, 'RBX': 0xAA}),
    ],
)
def test_misc_partial_writes_c_matches_unicorn(
    label: str, bytestring: str, regs_in: dict[str, int],
) -> None:
    """Cover the family of x86 partial-register writes (8/16/32-bit
    moves and RMWs) where the C evaluator must produce results
    bit-identical to Unicorn after the partial-write merge fix."""
    _assert_c_matches_unicorn(
        label, bytestring, regs_in, 'RAX', 0, 63,
    )
    # Also check RBX in case the write was to RBX
    _assert_c_matches_unicorn(
        label + ' (RBX read)', bytestring, regs_in, 'RBX', 0, 63,
    )
