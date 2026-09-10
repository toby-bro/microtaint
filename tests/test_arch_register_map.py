"""The register map has to reach every register a guest actually reads.

This is the test that would have caught the emulator being x86-only.  The map
used to be a hand-written table of 34 x86 names, so `_build_offsets_arrays` on
any other architecture returned empty arrays, every register came out zero, and
the run reported nothing at all -- silently, because nothing here asked.
"""
from __future__ import annotations

import collections

import pytest

from benchmark.instruction_bank import all_instructions
from microtaint.emulator import archregs
from microtaint.emulator.wrapper import X64_FORMAT, register_file
from microtaint.instrumentation.cell import _build_reg_maps, _get_decoded
from microtaint.types import Architecture

# Every architecture the emulator claims to lift for.
ARCHS = (Architecture.AMD64, Architecture.ARM64, Architecture.MIPS64BE,
         Architecture.PPC32BE, Architecture.RISCV64, Architecture.X86,
         Architecture.SPARC32BE)


@pytest.mark.parametrize('arch', ARCHS, ids=lambda a: a.value)
def test_every_architecture_has_a_register_map(arch: Architecture) -> None:
    regs = archregs.for_arch(arch)
    assert regs.offset_to_uc, f'{arch.value}: no register reaches Unicorn'
    assert regs.all_uc_ids, f'{arch.value}: nothing to read for a whole-file snapshot'
    assert len(regs.all_names) == len(regs.all_uc_ids)
    # The program counter is read from the hook's own address rather than the
    # CPU, so it has to be named even where it is not in the map.
    assert regs.pc_name
    assert regs.pc_offset == _build_reg_maps(arch)[0][regs.pc_name]


@pytest.mark.parametrize('arch', ARCHS, ids=lambda a: a.value)
def test_flags_unpack_from_their_parent(arch: Architecture) -> None:
    regs = archregs.for_arch(arch)
    if not regs.flag_bits:
        pytest.skip(f'{arch.value} models no packed flags register')
    values = {regs.flag_parent: (1 << 64) - 1}
    regs.unpack_flags(values)
    for name in regs.flag_bits:
        assert values[name] == 1, f'{arch.value}: {name} did not unpack'
    values = {regs.flag_parent: 0}
    regs.unpack_flags(values)
    assert all(values[n] == 0 for n in regs.flag_bits)


def _bank_offsets() -> dict[str, collections.Counter[int]]:
    """Every register offset the instruction bank's forms actually read."""
    per: dict[str, collections.Counter[int]] = collections.defaultdict(collections.Counter)
    for rec in all_instructions():
        if rec.isa.endswith('_SIMD'):
            continue                      # vector lanes take the vector path
        try:
            arch = Architecture[rec.isa]
        except KeyError:
            continue
        try:
            decoded = _get_decoded(arch, rec.bytes)
        except Exception:  # noqa: S112 - a form that will not lift is not the subject
            continue
        for off in decoded.input_reg_offsets:
            per[rec.isa][off] += 1
    return per


@pytest.mark.parametrize('isa', ['AMD64', 'ARM64', 'MIPS64BE', 'PPC32BE', 'RISCV64'])
def test_map_covers_every_register_the_bank_reads(isa: str) -> None:
    arch = Architecture[isa]
    regs = archregs.for_arch(arch)
    names = {off: name for name, off in _build_reg_maps(arch)[0].items()}
    reads = _bank_offsets()[isa]
    assert reads, f'{isa}: the bank decoded nothing'

    unmapped = {
        off: (names.get(off, '?'), n) for off, n in reads.items()
        # Scratch the lifter allocates for itself has nothing to read from:
        # it is written before it is read, inside the one instruction.
        if off not in regs.offset_to_uc and not regs.is_scratch(off)
    }
    assert not unmapped, (
        f'{isa}: {len(unmapped)} register offsets the bank reads have no way to '
        f'reach Unicorn, so they would be seeded zero: {sorted(unmapped.items())[:8]}')


@pytest.mark.parametrize('isa', ['AMD64', 'ARM64', 'MIPS64BE', 'PPC32BE', 'RISCV64'])
def test_state_format_covers_the_flags_the_isa_models(isa: str) -> None:
    """A register missing from the format gets no assignment, so its taint is
    dropped.  Flags are the ones that go missing quietly."""
    arch = Architecture[isa]
    regs = archregs.for_arch(arch)
    fmt = {r.name: r.bits for r in archregs.state_format(arch)}
    for flag in regs.flag_bits:
        assert flag in fmt, f'{isa}: {flag} taint would be dropped'
        # A chain-threaded one-byte boolean flag has to be declared one bit or
        # a carry threaded through it under-taints.
        assert fmt[flag] == 1, f'{isa}: {flag} is declared {fmt[flag]} bits, not 1'


def test_x64_format_is_still_what_the_baselines_were_taken_against() -> None:
    """x86's format is pinned on purpose (see archregs.state_format); this is
    the assertion that says so out loud, so widening it is a deliberate act."""
    assert [(r.name, r.bits) for r in X64_FORMAT] == [
        ('RAX', 64), ('RBX', 64), ('RCX', 64), ('RDX', 64), ('RSI', 64),
        ('RDI', 64), ('RBP', 64), ('RSP', 64),
        *[(f'R{i}', 64) for i in range(8, 16)],
        ('RIP', 64), ('EFLAGS', 32),
        ('ZF', 1), ('CF', 1), ('SF', 1), ('OF', 1), ('PF', 1), ('AF', 1),
        ('TF', 1), ('IF', 1), ('DF', 1),
    ]


@pytest.mark.parametrize('arch', ARCHS, ids=lambda a: a.value)
def test_register_file_arrays_line_up(arch: Architecture) -> None:
    """One id per call, one name per slot: a mismatch reads the wrong register
    into the wrong slot, which is a wrong VALUE and therefore a wrong taint."""
    f = register_file(arch)
    assert len(f.all_names) == f._n_slots
    assert len(f.all_ids) == f._n_calls
    # Two slots per vector register, one per scalar.
    assert f._n_slots == f._n_calls + len(f.vectors)
