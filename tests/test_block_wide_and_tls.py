"""Block mode must not refuse the shapes real library code is made of.

A block the lowering refuses is SKIPPED by the C hook, so nothing computes its
taint.  That is an under-taint, and on a static-glibc guest it was 71 of 642
block executions before these two fixes, because both causes live in the
vectorised string and memory routines every real binary calls:

  * thread-local storage.  Sleigh names the FS base `FS_OFFSET` and Unicorn
    names it `FS_BASE`, and nothing connected the two, so the engine could not
    read it at all: 25 of 49 refused blocks, and every `%fs:`-relative address
    was computed from zero.
  * a load or store wider than a machine word.  The lowering already split wide
    REGISTER varnodes into 8-byte lanes but refused wide MEMORY, which is what
    every `movdqu` is.

The hand-written `-nostdlib` benchmarks refuse nothing at all, which is why
none of this showed there.
"""
from __future__ import annotations

import pytest

from microtaint.emulator import archregs
from microtaint.taint_ir.blocks import plan_block
from microtaint.taint_ir.ir import Access
from microtaint.types import Architecture

_ARCH = Architecture.AMD64
_BASE = 0x401000

#: `mov rax, qword ptr fs:[0x28]` -- the stack-protector canary read, and the
#: single most common `%fs:` reference in a glibc binary.
_TLS_READ = bytes.fromhex('64488b042528000000')
#: `movdqu (%rsi), %xmm0` and `movdqu %xmm0, (%rdi)`: a 16-byte load and store.
_WIDE_LOAD = bytes.fromhex('f30f6f06')
_WIDE_STORE = bytes.fromhex('f30f7f07')


def _accesses(code: bytes) -> list[Access]:
    """The memory accesses of a one-region lowering of `code`."""
    regions = plan_block(_ARCH, code, _BASE, abs_ram=True)
    assert len(regions) == 1, f'expected one region, got {len(regions)}'
    prog = regions[0].prog
    assert prog is not None, (
        'the lowering refused this, so a block containing it is skipped and '
        'its taint is never computed')
    return list(prog.accesses)


# ---------------------------------------------------------------------------
# Thread-local storage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(('sleigh', 'unicorn_name'),
                         [('FS_OFFSET', 'FS_BASE'), ('GS_OFFSET', 'GS_BASE')])
def test_the_segment_bases_are_readable(sleigh: str, unicorn_name: str) -> None:
    """The two names have to be connected, or the register cannot be read.

    An unreadable register is not merely untracked: its VALUE reads back as
    zero, so an address computed from it points somewhere else entirely, and a
    load resolved at the wrong address reads the shadow at the wrong place and
    comes back clean.
    """
    ux = pytest.importorskip('unicorn.x86_const')
    regs = archregs.for_arch(_ARCH)
    assert sleigh in regs.all_names, (
        f'{sleigh} is not tracked, so block mode has no slot for it')

    from microtaint.instrumentation.cell import _build_reg_maps
    off = _build_reg_maps(_ARCH)[0][sleigh]
    entry = regs.offset_to_uc.get(off)
    assert entry is not None, f'{sleigh} at {off:#x} maps to no Unicorn register'
    name, uc_id, _is_flag = entry
    assert name == sleigh
    assert uc_id == getattr(ux, f'UC_X86_REG_{unicorn_name}'), (
        f"{sleigh} must read through Unicorn's {unicorn_name}")


def test_a_thread_local_read_lowers() -> None:
    """`mov rax, fs:[0x28]`, the canary read.  It was the single biggest cause
    of refused blocks: 25 of 49 on a static-glibc guest."""
    accesses = _accesses(_TLS_READ)
    assert len(accesses) == 1, f'expected one load, got {accesses}'
    assert accesses[0]['kind'] == 'load'
    assert accesses[0]['size'] == 8


# ---------------------------------------------------------------------------
# Wide memory
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(('label', 'code', 'kind'),
                         [('movdqu load', _WIDE_LOAD, 'load'),
                          ('movdqu store', _WIDE_STORE, 'store')])
def test_a_wide_access_becomes_one_access_per_8_byte_lane(
        label: str, code: bytes, kind: str) -> None:
    """A 16-byte access is two 8-byte ones, which is what the runtime can
    resolve and what the shadow is addressed in."""
    accesses = _accesses(code)
    assert len(accesses) == 2, f'{label}: expected two lanes, got {accesses}'
    assert [a['kind'] for a in accesses] == [kind, kind], label
    assert [a['size'] for a in accesses] == [8, 8], (
        f'{label}: a lane wider than 8 bytes is not something the runtime can '
        f'read or the shadow can address')


def test_the_second_lane_is_eight_bytes_further_on() -> None:
    """Lane 1 must address `base + 8`, not `base` again.

    Two lanes at the same address would read the low half twice: the high half
    of the value would never be looked at, and its taint would be lost.  The
    address is a node in the program, so this reads it back through the
    program's own constant folding.
    """
    from microtaint.taint_ir.ir import ADD

    regions = plan_block(_ARCH, _WIDE_LOAD, _BASE, abs_ram=True)
    prog = regions[0].prog
    assert prog is not None
    lo, hi = (a['addr'] for a in prog.accesses)
    assert lo != hi, 'both lanes address the same word'
    op, a, b, _c, _imm = prog.nodes[hi]
    assert op == ADD, f'lane 1 is not an offset from lane 0 ({op})'
    assert a == lo, 'lane 1 is not derived from lane 0'
    off = prog.nodes[b]
    assert off[0] == 'const', f'lane 1 is at a computed offset ({off[0]})'
    assert off[4] == 8, f'lane 1 is at +{off[4]}, expected +8'
