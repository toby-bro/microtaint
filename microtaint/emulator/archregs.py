"""Naming a guest register on both sides of the emulator boundary.

Sleigh identifies a register by its byte offset in the architecture's register
space; Unicorn identifies it by an integer drawn from a per-architecture
constant module.  Every instruction the engine sees has to carry values across
that boundary, so the two namings have to be joined -- and they already agree
on nearly every spelling, because both were read off the same architecture
manual.  The map is therefore DERIVED, by matching the geometry's names against
the constant module's, rather than written out by hand: a hand-written table is
exactly what made this x86-only, silently, for every other guest.

Only what cannot be derived is listed per ISA, and only that:

  * ``_ALIASES`` -- the few spellings the two genuinely disagree on (Sleigh's
    PowerPC ``R7`` is Unicorn's ``7``).
  * ``_FLAGS`` -- condition flags, which Sleigh gives one-byte registers of
    their own and Unicorn only exposes packed inside a parent register.

Everything else falls out of the geometry.  In particular, which name to use
when several share an offset is not a choice: the widest register covering an
offset is the one to read, because one read then fills that whole span and
every narrower view of it.  Registers wider than the eight-byte value slot are
excluded -- reading one would overrun into the next slot -- so a vector file
reaches the state through the lane path, not this one.
"""

# ruff: noqa: PLC0415  (deferred imports: cell and types would cycle)
from __future__ import annotations

import importlib
from dataclasses import dataclass

from microtaint.types import Architecture

# Architecture -> (unicorn constant module, constant-name prefix).
_CONST_MODULE: dict[Architecture, tuple[str, str]] = {
    Architecture.X86: ('unicorn.x86_const', 'UC_X86_REG_'),
    Architecture.AMD64: ('unicorn.x86_const', 'UC_X86_REG_'),
    Architecture.ARM64: ('unicorn.arm64_const', 'UC_ARM64_REG_'),
    Architecture.MIPS64BE: ('unicorn.mips_const', 'UC_MIPS_REG_'),
    Architecture.PPC32BE: ('unicorn.ppc_const', 'UC_PPC_REG_'),
    Architecture.RISCV64: ('unicorn.riscv_const', 'UC_RISCV_REG_'),
    Architecture.SPARC32BE: ('unicorn.sparc_const', 'UC_SPARC_REG_'),
}

# Sleigh spelling -> Unicorn spelling, only where the two disagree.
# PowerPC is the one architecture whose general registers Unicorn numbers
# rather than names; everything else agrees letter for letter.
_ALIASES: dict[Architecture, dict[str, str]] = {
    Architecture.PPC32BE: {f'R{i}': str(i) for i in range(32)},
}

# Condition flags: Sleigh's own one-byte register -> bit position inside the
# parent register Unicorn exposes.  One read of the parent yields them all,
# which is why the parent is named once rather than per flag.
_FLAGS: dict[Architecture, tuple[str, dict[str, int]]] = {
    Architecture.X86: ('EFLAGS', {
        'CF': 0, 'PF': 2, 'AF': 4, 'ZF': 6, 'SF': 7,
        'TF': 8, 'IF': 9, 'DF': 10, 'OF': 11,
    }),
    Architecture.ARM64: ('NZCV', {'NG': 31, 'ZR': 30, 'CY': 29, 'OV': 28}),
    # XER_COUNT is a seven-bit field rather than a flag, so it has no entry
    # here and reaches the state through the whole-register read instead.
    Architecture.PPC32BE: ('XER', {'XER_SO': 31, 'XER_OV': 30, 'XER_CA': 29}),
}
_FLAGS[Architecture.AMD64] = _FLAGS[Architecture.X86]

# Registers Sleigh declares that the architecture does not have: scratch the
# lifter allocates for itself, written before it is read inside the one
# instruction that uses it.  There is nothing in the CPU to read them from, and
# nothing to read: seeding them zero is correct rather than approximate.  Named
# here so a gap in the map can be told apart from a register we simply failed
# to place -- the distinction the completeness test turns on.
_SCRATCH: dict[Architecture, tuple[str, ...]] = {
    Architecture.ARM64: ('TMP', 'SHIFT_CARRY'),
}

#: The value array holds one eight-byte slot per read, so a register wider than
#: that cannot go through this path without overrunning the next slot.
_MAX_WIDTH = 8


@dataclass(frozen=True)
class ArchRegs:
    """Everything the emulator boundary needs to name this guest's registers."""

    arch: Architecture
    #: Sleigh byte offset -> (name to store the value under, Unicorn id,
    #: whether the value has to be unpacked into flag bits afterwards).
    offset_to_uc: dict[int, tuple[str, int, bool]]
    #: Every register worth reading when the whole file is wanted, and the ids
    #: that read them -- same order, one id per name.
    all_names: tuple[str, ...]
    all_uc_ids: tuple[int, ...]
    #: The program counter, which the hook writes from the hook's own address
    #: rather than reading, so that PC-relative operands resolve at run time.
    pc_name: str
    pc_offset: int
    #: Flag unpacking, empty for an architecture whose flags Unicorn names.
    flag_parent: str | None
    flag_bits: dict[str, int]
    #: Offsets of the lifter's own scratch registers: unmapped on purpose.
    scratch_offsets: frozenset[int]

    def is_scratch(self, offset: int) -> bool:
        return offset in self.scratch_offsets

    def unpack_flags(self, values: dict) -> None:
        """Expand the parent's value into the individual flag registers."""
        if not self.flag_bits:
            return
        parent = values.get(self.flag_parent, 0)
        for name, bit in self.flag_bits.items():
            values[name] = (parent >> bit) & 1


_CACHE: dict[Architecture, ArchRegs] = {}


def for_arch(arch: Architecture) -> ArchRegs:
    hit = _CACHE.get(arch)
    if hit is None:
        hit = _CACHE[arch] = _build(arch)
    return hit


def _unicorn_ids(arch: Architecture) -> dict[str, int]:
    """Every register Unicorn names for this architecture, upper-cased."""
    spec = _CONST_MODULE.get(arch)
    if spec is None:
        return {}
    mod_name, prefix = spec
    try:
        mod = importlib.import_module(mod_name)
    except ImportError:
        return {}
    out = {}
    for attr in dir(mod):
        if not attr.startswith(prefix):
            continue
        name = attr[len(prefix):]
        if name in ('INVALID', 'ENDING'):
            continue
        out[name] = getattr(mod, attr)
    return out


def _readable(arch, offsets, sizes) -> list[tuple[int, int, str, int]]:
    """Geometry registers Unicorn also names, narrow enough for one slot."""
    uc_ids = _unicorn_ids(arch)
    alias = _ALIASES.get(arch, {})
    out: list[tuple[int, int, str, int]] = []      # (offset, size, name, uc_id)
    for name, off in offsets.items():
        size = sizes.get(name, 8)
        if size > _MAX_WIDTH:
            continue
        uc_id = uc_ids.get(alias.get(name, name))
        if uc_id is not None:
            out.append((off, size, name, uc_id))
    return out


def _cover(candidates, offsets, sizes) -> dict[int, tuple[str, int, bool]]:
    """Every byte a readable register covers, keyed to the widest read for it.

    Keying only the offsets the geometry names would miss the ones instructions
    actually ask for: a lifted MIPS `A0` field is read at offset 38, six bytes
    into the register that holds it, and no register is declared there.  One
    read supplies that register and every narrower view of it, and the value
    lands in the frame at the right byte with the right width.
    """
    cover: dict[int, tuple[str, int, bool]] = {}
    for c_off, c_size, c_name, c_id in sorted(candidates):
        for off in range(c_off, c_off + c_size):
            prev = cover.get(off)
            if prev is not None:
                # widest wins; then the one that starts earliest; then the
                # name, so the table comes out the same on every run.
                key = (sizes.get(prev[0], 8), -offsets.get(prev[0], off), c_name)
                if key >= (c_size, -c_off, prev[0]):
                    continue
            cover[off] = (c_name, c_id, False)
    return cover


def _build(arch: Architecture) -> ArchRegs:
    from microtaint.instrumentation.cell import _build_reg_maps

    offsets, sizes = _build_reg_maps(arch)
    candidates = _readable(arch, offsets, sizes)

    flag_parent, flag_bits = _FLAGS.get(arch, (None, {}))
    parent_id = _unicorn_ids(arch).get(flag_parent) if flag_parent else None
    if parent_id is None:
        flag_parent, flag_bits = None, {}

    offset_to_uc = _cover(candidates, offsets, sizes)
    # Flags override: their own offsets read the parent and unpack.
    for name in flag_bits:
        off = offsets.get(name)
        if off is not None:
            offset_to_uc[off] = (flag_parent, parent_id, True)

    prefixes = _SCRATCH.get(arch, ())
    scratch_offsets = frozenset(
        off for name, off in offsets.items()
        if off not in offset_to_uc and name.startswith(prefixes)
    ) if prefixes else frozenset()

    pc_name, pc_offset = _pc(offsets)

    # The whole-file read: one entry per distinct register, so a fallback that
    # wants everything gets every offset the map can reach.
    seen: dict[int, str] = {}
    for name, uc_id, is_flag in offset_to_uc.values():
        if is_flag:
            continue
        seen.setdefault(uc_id, name)
    if flag_parent is not None:
        seen.setdefault(parent_id, flag_parent)
    all_names = tuple(seen.values())
    all_uc_ids = tuple(seen.keys())

    return ArchRegs(
        arch=arch,
        offset_to_uc=offset_to_uc,
        all_names=all_names,
        all_uc_ids=all_uc_ids,
        pc_name=pc_name,
        pc_offset=pc_offset,
        flag_parent=flag_parent,
        flag_bits=dict(flag_bits),
        scratch_offsets=scratch_offsets,
    )


def state_format(arch: Architecture) -> list:
    """The register set this architecture's taint state tracks.

    A register missing from the format gets no assignment at all -- its taint
    is silently dropped -- so the list has to cover everything a guest program
    can write.  Deriving it is what makes that true for an architecture nobody
    hand-listed: every register Unicorn can read and the frame can hold, plus
    one entry per condition flag, declared a single bit so a carry threaded
    through a chain of them keeps its width.

    x86 is the exception, and deliberately: its list is pinned to what the
    committed perf baselines and taint expectations were taken against, so
    widening it there is a measured change rather than a side effect of
    teaching the emulator about other architectures.
    """
    from microtaint.types import Register

    pinned = _PINNED_FORMAT.get(arch)
    if pinned is not None:
        return [Register(n, b) for n, b in pinned]

    from microtaint.instrumentation.cell import _build_reg_maps

    regs = for_arch(arch)
    _, sizes = _build_reg_maps(arch)
    names = sorted({n for n, _uc, is_flag in regs.offset_to_uc.values() if not is_flag})
    fmt = [Register(n, min(sizes.get(n, 8), _MAX_WIDTH) * 8) for n in names]
    fmt += [Register(f, 1) for f in sorted(regs.flag_bits)]
    return fmt


# x86's format, pinned: see state_format.  Flag entries are one bit each so a
# carry threaded from one flag to the next keeps its width.
_X86_FORMAT: tuple[tuple[str, int], ...] = (
    tuple((f'R{n}', 64) for n in
          ('AX', 'BX', 'CX', 'DX', 'SI', 'DI', 'BP', 'SP'))
    + tuple((f'R{i}', 64) for i in range(8, 16))
    + (('RIP', 64), ('EFLAGS', 32))
    # Every flag the map unpacks out of EFLAGS is tracked, not just the six
    # arithmetic ones: a flag whose value is read but whose taint is not
    # tracked under-taints the moment a guest writes it from tainted data, and
    # `popf` writes all of them from the stack.  Leaving AF out dropped its
    # taint on the five bank forms that write it (the BCD adjusts).  One bit
    # each, so a carry threaded from one flag to the next keeps its width.
    + tuple((f, 1) for f in ('ZF', 'CF', 'SF', 'OF', 'PF', 'AF', 'TF', 'IF', 'DF'))
)
_PINNED_FORMAT: dict[Architecture, tuple[tuple[str, int], ...]] = {
    Architecture.AMD64: _X86_FORMAT,
    Architecture.X86: _X86_FORMAT,
}


def _pc(offsets: dict[str, int]) -> tuple[str, int]:
    """The architecture's program counter, by the names Sleigh gives it."""
    for name in ('RIP', 'EIP', 'PC'):
        if name in offsets:
            return name, offsets[name]
    raise KeyError('no program counter in this architecture geometry')
