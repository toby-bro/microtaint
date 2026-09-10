"""Human-name <-> engine-name register translation (debug / tests only).

The taint compute path tracks vector registers as geometry-derived 8-byte lanes
named ``VL_<sleigh-byte-offset>`` and scalar registers by their sleigh name.
That is ISA-general but not human friendly: ``VL_0x1240`` means nothing to a
person, and a 128-bit XMM cannot appear in a state_format directly (the mask
path is 64-bit, so a vector is carried as its 64-bit lanes).

``RegisterAliases`` is a bidirectional alias table so a human can write and read
tests, and inspect rules, in architectural terms -- ``XMM1``, ``XMM0[63:0]``,
``RAX`` -- while the engine keeps its geometry names.  The whole table is
derived from pypcode's own register geometry (there is no hand-written per-ISA
table), so the same code names x86 XMM/YMM/ZMM lanes, ARM64 Q/Z lanes, PPC vs
lanes, and so on.

IMPORTANT: nothing in the compute path imports this module.  It is purely a
helper for tests and for humans inspecting a rule, never part of taint
evaluation.  The only thing it borrows from the engine is ``get_context`` (the
canonical Architecture -> pypcode-spec factory), so the geometry it reports is
always the geometry the engine actually uses.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from microtaint.sleigh.lifter import get_context
from microtaint.types import Architecture, Register

LANE_BYTES = 8
LANE_BITS = LANE_BYTES * 8

# A geometry lane can be covered by several overlapping vector registers (x86
# XMM0/YMM0/ZMM0 share a base offset).  When naming a lane we prefer the
# SMALLEST covering register (XMM over YMM over ZMM), and among equal sizes we
# de-prioritise scratch/result/mask/temp registers that some Sleigh specs place
# at the same offset as a real architectural register.
_SCRATCH_MARKERS = ('RESULT', 'MASK', 'TMP', 'BCST', 'DEST', 'DEBUG')

_RANGE_RE = re.compile(r'^(?P<name>[A-Za-z_][A-Za-z0-9]*)\[(?P<hi>\d+):(?P<lo>\d+)\]$')
_HALF_RE = re.compile(r'^(?P<name>[A-Za-z_][A-Za-z0-9]*)_(?P<half>LO|HI)$', re.IGNORECASE)


def _is_scratch(name: str) -> bool:
    up = name.upper()
    return any(m in up for m in _SCRATCH_MARKERS)


class RegisterAliases:
    """Bidirectional human <-> engine register-name map for one architecture."""

    def __init__(self, arch: Architecture | str) -> None:
        self.arch: str = arch.name if isinstance(arch, Architecture) else str(arch)
        self.is_big_endian: bool = self.arch.upper().endswith('BE')
        ctx = get_context(self.arch)

        # name(upper) -> (offset, size); split into scalar (<=8) and vector (>8).
        self._scalars: dict[str, tuple[int, int]] = {}
        self._vectors: dict[str, tuple[int, int]] = {}
        self._canonical: dict[str, str] = {}  # upper -> original-case sleigh name
        for name, vn in ctx.registers.items():
            if vn.space.name != 'register':
                continue
            up = name.upper()
            self._canonical[up] = name
            if vn.size > LANE_BYTES:
                self._vectors[up] = (vn.offset, vn.size)
            else:
                self._scalars[up] = (vn.offset, vn.size)

        # ISA friendly-name aliases: names a human writes that differ from the
        # official Sleigh register name.  This is the correspondence table the
        # compute path no longer carries (ARM64 condition flags are ng/zr/cy/ov
        # in Sleigh, but people write N/Z/C/V).  Arch-gated and geometry-checked
        # (only added when the Sleigh register actually exists).  friendly(upper)
        # -> canonical Sleigh name, plus the reverse for to_human_name.
        self._friendly: dict[str, str] = {}
        self._friendly_rev: dict[str, str] = {}
        if 'ARM' in self.arch.upper():
            for fr, sl in (('N', 'ng'), ('Z', 'zr'), ('C', 'cy'), ('V', 'ov')):
                canon = self._canonical.get(sl.upper())
                if canon is not None:
                    self._friendly[fr] = canon
                    self._friendly_rev[sl.upper()] = fr

        # Geometry lane byte-offset -> canonical human name (e.g. 'XMM0[63:0]').
        self._lane_human: dict[int, str] = {}
        for lane_off in self._all_lane_offsets():
            best = self._smallest_cover(lane_off)
            if best is not None:
                bname, boff, _ = best
                lo = (lane_off - boff) * 8
                self._lane_human[lane_off] = f'{bname}[{lo + LANE_BITS - 1}:{lo}]'

    # -- geometry helpers --------------------------------------------------

    def _all_lane_offsets(self) -> set[int]:
        offs: set[int] = set()
        for off, size in self._vectors.values():
            for b in range(off, off + size, LANE_BYTES):
                offs.add(b)
        return offs

    def _smallest_cover(self, lane_off: int) -> tuple[str, int, int] | None:
        """The (canonical-name, offset, size) of the smallest vector register
        that covers ``lane_off``, de-prioritising scratch registers on ties."""
        best: tuple[str, int, int] | None = None
        best_key: tuple[int, int, str] | None = None
        for up, (off, size) in self._vectors.items():
            if off <= lane_off < off + size:
                cname = self._canonical[up]
                key = (size, 1 if _is_scratch(cname) else 0, cname)
                if best_key is None or key < best_key:
                    best_key = key
                    best = (cname, off, size)
        return best

    def _lane_value_bit(self, vec_off: int, vec_size: int, lane_off: int) -> int:
        """Value-bit index (LSB = 0) at which lane ``lane_off`` starts inside its
        vector.  Endianness-aware: on a big-endian target byte 0 of a register is
        its MOST significant byte, so the lowest-offset lane holds the high bits."""
        rel_byte = lane_off - vec_off
        if self.is_big_endian:
            return (vec_size - rel_byte - LANE_BYTES) * 8
        return rel_byte * 8

    # -- name resolution ---------------------------------------------------

    def _resolve_vector_lanes(self, name_up: str, lo_bit: int, hi_bit: int) -> list[int]:
        """Byte-offsets of the geometry lanes of vector ``name_up`` overlapping
        register value-bit range [lo_bit, hi_bit]."""
        off, size = self._vectors[name_up]
        lanes: list[int] = []
        for b in range(off, off + size, LANE_BYTES):
            v_lo = self._lane_value_bit(off, size, b)
            v_hi = v_lo + LANE_BITS - 1
            if v_lo <= hi_bit and lo_bit <= v_hi:
                lanes.append(b)
        lanes.sort()
        return lanes

    def to_engine_names(self, human: str) -> list[str]:
        """Engine register key(s) for a human name.

        ``RAX`` -> ``['RAX']``; ``XMM0`` -> both lane names; ``XMM0[63:0]`` /
        ``XMM0_LO`` -> the single covering lane.  A name the table does not know
        (already an engine key, e.g. ``VL_0x1240`` or an unlisted register) is
        returned unchanged so callers can pass through raw engine names."""
        s = human.strip()

        m = _RANGE_RE.match(s)
        if m:
            name_up = m.group('name').upper()
            lo, hi = int(m.group('lo')), int(m.group('hi'))
            if name_up in self._vectors:
                return [f'VL_{b:#x}' for b in self._resolve_vector_lanes(name_up, lo, hi)]
            # A bit-slice of a scalar is still the scalar's engine key.
            if name_up in self._scalars:
                return [self._canonical[name_up]]
            return [s]

        m = _HALF_RE.match(s)
        if m and m.group('name').upper() in self._vectors:
            name_up = m.group('name').upper()
            half = m.group('half').upper()
            lo, hi = (0, LANE_BITS - 1) if half == 'LO' else (LANE_BITS, 2 * LANE_BITS - 1)
            return [f'VL_{b:#x}' for b in self._resolve_vector_lanes(name_up, lo, hi)]

        up = s.upper()
        if up in self._vectors:
            off, size = self._vectors[up]
            return [f'VL_{b:#x}' for b in range(off, off + size, LANE_BYTES)]
        if up in self._scalars:
            return [self._canonical[up]]
        if up in self._friendly:
            return [self._friendly[up]]
        return [s]

    def to_human_name(self, engine: str) -> str:
        """Human name for an engine register key (inverse of the common cases of
        :meth:`to_engine_names`).  ``VL_0x1200`` -> ``XMM0[63:0]``; an ISA flag is
        returned in its friendly form (ng -> N); a scalar in its canonical case;
        an unknown key is returned unchanged."""
        friendly = self._friendly_rev.get(engine.upper())
        if friendly is not None:
            return friendly
        if engine.startswith('VL_'):
            try:
                off = int(engine[3:], 16)
            except ValueError:
                return engine
            return self._lane_human.get(off, engine)
        return self._canonical.get(engine.upper(), engine)

    def _entity(self, human: str) -> tuple[list[str], int, tuple[int, int] | None]:
        """Parse a human name into (engine keys, base value-bit, (offset,size) of
        the vector, or None for a scalar).  The base value-bit is the LSB of the
        named entity: 0 for a whole register, `lo` for NAME[hi:lo], 0/64 for
        _LO/_HI.  Shared by :meth:`to_engine` and :meth:`read`."""
        keys = self.to_engine_names(human)
        if not keys or not keys[0].startswith('VL_'):
            return keys, 0, None
        s = human.strip()
        mr = _RANGE_RE.match(s)
        mh = _HALF_RE.match(s)
        if mr:
            vname, base_lo = mr.group('name').upper(), int(mr.group('lo'))
        elif mh:
            vname = mh.group('name').upper()
            base_lo = 0 if mh.group('half').upper() == 'LO' else LANE_BITS
        else:
            vname, base_lo = s.upper(), 0
        return keys, base_lo, self._vectors[vname]

    # -- bulk helpers for tests -------------------------------------------

    def state_format(self, names: Sequence[str | Register]) -> list[Register]:
        """A state_format built from human names, vectors expanded to their
        64-bit geometry lanes (so no >64-bit Register reaches the engine).
        Scalars keep an explicit bit width when given as a ``Register``."""
        out: list[Register] = []
        seen: set[str] = set()
        for item in names:
            human = item.name if isinstance(item, Register) else item
            explicit_bits = item.bits if isinstance(item, Register) else None
            for key in self.to_engine_names(human):
                if key in seen:
                    continue
                seen.add(key)
                if key.startswith('VL_'):
                    out.append(Register(key, LANE_BITS))
                elif explicit_bits is not None:
                    out.append(Register(key, explicit_bits))
                else:
                    up = key.upper()
                    size = self._scalars.get(up, (0, LANE_BYTES))[1]
                    out.append(Register(key, size * 8))
        return out

    def to_engine(self, values: dict[str, int]) -> dict[str, int]:
        """Translate a human-keyed value/taint dict to engine keys, splitting a
        wide vector value across its lanes (endianness-aware)."""
        out: dict[str, int] = {}
        mask = (1 << LANE_BITS) - 1
        for human, val in values.items():
            keys, base_lo, vec = self._entity(human)
            if vec is None:
                out[keys[0]] = out.get(keys[0], 0) | val
                continue
            # Vector: the value is relative to the entity's low bit.  Entity bit v
            # sits at lane-local bit (v - d), d = lane_value_bit - base_lo, so the
            # lane value is `val >> d` (d>=0) or `val << -d` (d<0, when the slice
            # starts above the lane base, e.g. XMM0[95:88] in the byte-8..15 lane).
            off, size = vec
            for key in keys:
                d = self._lane_value_bit(off, size, int(key[3:], 16)) - base_lo
                lane_val = (val >> d) if d >= 0 else (val << -d)
                out[key] = out.get(key, 0) | (lane_val & mask)
        return out

    def read(self, values: dict[str, int], human: str) -> int:
        """Whole value of ONE human-named register or slice, gathered from an
        engine-keyed output dict (the inverse of one :meth:`to_engine` entry).
        ``read(out, 'YMM0')`` combines YMM0's four lanes into one 256-bit value;
        ``read(out, 'XMM0[127:64]')`` returns just that half at bit 0; a scalar
        name returns its value directly.  Unambiguous because the caller names
        the register (unlike :meth:`from_engine`, which must guess per lane)."""
        keys, base_lo, vec = self._entity(human)
        if vec is None:
            return values.get(keys[0], 0)
        off, size = vec
        mask = (1 << LANE_BITS) - 1
        result = 0
        for key in keys:
            # Inverse of to_engine: lane bit b holds entity bit (b + d), d as above.
            d = self._lane_value_bit(off, size, int(key[3:], 16)) - base_lo
            lane_val = values.get(key, 0) & mask
            result |= (lane_val << d) if d >= 0 else (lane_val >> -d)
        return result

    def from_engine(self, values: dict[str, int]) -> dict[str, int]:
        """Recombine an engine-keyed value/taint dict into human names, merging a
        vector's lanes back into one wide value keyed by the register name."""
        out: dict[str, int] = {}
        for key, val in values.items():
            if not key.startswith('VL_'):
                out[self.to_human_name(key)] = val
                continue
            lane_off = int(key[3:], 16)
            best = self._smallest_cover(lane_off)
            if best is None:
                out[key] = val
                continue
            bname, boff, bsize = best
            shift = self._lane_value_bit(boff, bsize, lane_off)
            out[bname] = out.get(bname, 0) | (val << shift)
        return out
