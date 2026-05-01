"""
test_shadow_bit_precision.py
============================
Verifies that BitPreciseShadowMemory tracks taint at bit granularity (not
just byte granularity).

These tests pin down the contract documented in shadow.pyx line 17:
"Taint byte: bit i set means bit i of the corresponding memory byte is
tainted." Each shadow byte carries 8 independent bits of taint, one per
bit of the underlying memory byte. So a 1-byte memory cell can have
3 of its 8 bits tainted (e.g. shadow value 0x07) without polluting the
others; an 8-byte memory cell can have 17 tainted bits in arbitrary
positions; etc.

These properties are the engine's correctness backbone: dropping to
byte-granularity here would cascade into the rule generator and the
runtime, and a tainted nibble write would falsely taint the entire byte.
"""

# mypy: disable-error-code="no-untyped-def, no-untyped-call"

from __future__ import annotations

import pytest

from microtaint.emulator.shadow import BitPreciseShadowMemory

# ---------------------------------------------------------------------------
# Single-byte tests — taint within a byte
# ---------------------------------------------------------------------------


class TestSingleByteBitPrecision:
    def test_single_bit_tainted(self) -> None:
        """Setting bit 3 of one byte must yield mask 0x08 — not 0xFF."""
        shm = BitPreciseShadowMemory()
        shm.write_mask(0x1000, 0x08, 1)  # bit 3 only
        assert shm.read_mask(0x1000, 1) == 0x08

    def test_each_bit_independently(self) -> None:
        """Each of the 8 bits inside a byte is tracked separately."""
        shm = BitPreciseShadowMemory()
        for bit in range(8):
            shm.write_mask(0x2000 + bit, 1 << bit, 1)
            got = shm.read_mask(0x2000 + bit, 1)
            assert got == (1 << bit), f'bit {bit}: expected {1 << bit:#x}, got {got:#x}'

    def test_partial_taint_does_not_leak(self) -> None:
        """Tainting bits 0,2,4,6 leaves bits 1,3,5,7 clean."""
        shm = BitPreciseShadowMemory()
        shm.write_mask(0x3000, 0x55, 1)  # 01010101
        assert shm.read_mask(0x3000, 1) == 0x55
        # The complement bits must read as untainted
        assert (shm.read_mask(0x3000, 1) & 0xAA) == 0

    def test_clear_specific_byte_leaves_neighbours(self) -> None:
        """Clearing one byte's taint must not affect adjacent bytes."""
        shm = BitPreciseShadowMemory()
        shm.write_mask(0x4000, 0xFF, 1)
        shm.write_mask(0x4001, 0xFF, 1)
        shm.write_mask(0x4002, 0xFF, 1)
        shm.clear(0x4001, 1)
        assert shm.read_mask(0x4000, 1) == 0xFF
        assert shm.read_mask(0x4001, 1) == 0x00
        assert shm.read_mask(0x4002, 1) == 0xFF


# ---------------------------------------------------------------------------
# Multi-byte tests — taint mask is little-endian per-byte
# ---------------------------------------------------------------------------


class TestMultiByteLittleEndian:
    def test_write_8byte_mask_each_byte_independent(self) -> None:
        """An 8-byte write packs 8 per-byte taint masks into one uint64.
        Bit group [i*8 .. i*8+7] is the taint mask for memory byte addr+i."""
        shm = BitPreciseShadowMemory()
        # Byte 0: bit 0 only.  Byte 1: bit 1 only. ... Byte 7: bit 7 only.
        # Packed mask bytes (LE): [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80]
        # As uint64: 0x80_40_20_10_08_04_02_01
        packed = 0x8040201008040201
        shm.write_mask(0x5000, packed, 8)
        assert shm.read_mask(0x5000, 8) == packed
        # Each individual byte must read back its own slice
        for i in range(8):
            expected = 1 << i
            got = shm.read_mask(0x5000 + i, 1)
            assert got == expected, f'byte {i}: expected {expected:#x}, got {got:#x}'

    def test_partial_taint_across_bytes(self) -> None:
        """Taint flows in arbitrary bit patterns across byte boundaries."""
        shm = BitPreciseShadowMemory()
        # Byte 0: 0x0F (low nibble), Byte 1: 0xF0 (high nibble),
        # Byte 2: 0xAA (alternating), Byte 3: 0x55 (alternating opposite)
        packed = 0x55AAF00F
        shm.write_mask(0x6000, packed, 4)
        # Read each byte
        assert shm.read_mask(0x6000, 1) == 0x0F
        assert shm.read_mask(0x6001, 1) == 0xF0
        assert shm.read_mask(0x6002, 1) == 0xAA
        assert shm.read_mask(0x6003, 1) == 0x55
        # And the whole thing
        assert shm.read_mask(0x6000, 4) == packed

    def test_partial_read_within_tainted_region(self) -> None:
        """Reading 2 bytes of a 4-byte tainted region returns just those 2."""
        shm = BitPreciseShadowMemory()
        shm.write_mask(0x7000, 0xDEADBEEF, 4)
        # Read middle 2 bytes: 0xADBE  (little-endian: bytes 1,2 = 0xBE, 0xAD)
        assert shm.read_mask(0x7001, 2) == 0xADBE


# ---------------------------------------------------------------------------
# Page-boundary tests — pages are 4 KiB; verify bits flow across pages
# ---------------------------------------------------------------------------


class TestPageBoundary:
    def test_write_spanning_pages_preserves_bits(self) -> None:
        """A multi-byte write that crosses a page boundary must keep
        per-bit precision in both pages."""
        shm = BitPreciseShadowMemory()
        # 4 bytes spanning the 0x1000-page boundary at 0xFFE..0x1001
        addr = 0x0FFE
        # Each byte: a different bit pattern
        packed = 0x55AA_F00F  # bytes [0x0F, 0xF0, 0xAA, 0x55]
        shm.write_mask(addr, packed, 4)
        assert shm.read_mask(addr, 1) == 0x0F  # last byte of page 0
        assert shm.read_mask(addr + 1, 1) == 0xF0  # last byte of page 0 (or first of page 1)
        assert shm.read_mask(addr + 2, 1) == 0xAA  # first or second byte of page 1
        assert shm.read_mask(addr + 3, 1) == 0x55
        # And the round-trip reads back identical
        assert shm.read_mask(addr, 4) == packed

    def test_independent_pages_dont_leak(self) -> None:
        """Tainting one page does not affect a different page."""
        shm = BitPreciseShadowMemory()
        shm.write_mask(0x2000, 0xFFFFFFFF, 4)  # page 0x2000
        # Page 0x3000 must report 0
        assert shm.read_mask(0x3000, 4) == 0
        assert shm.read_mask(0x3FFF, 1) == 0


# ---------------------------------------------------------------------------
# is_tainted query — returns True iff ANY bit is tainted in range
# ---------------------------------------------------------------------------


class TestIsTaintedQuery:
    def test_is_tainted_any_bit(self) -> None:
        """is_tainted returns True if any of the 8 bits of any byte is set."""
        shm = BitPreciseShadowMemory()
        shm.write_mask(0x8000, 0x01, 1)  # only bit 0 of byte 0 tainted
        assert shm.is_tainted(0x8000, 1) is True

    def test_is_tainted_false_when_all_bits_clear(self) -> None:
        shm = BitPreciseShadowMemory()
        shm.write_mask(0x9000, 0x00, 4)  # explicit no-taint write
        assert shm.is_tainted(0x9000, 4) is False

    def test_is_tainted_range_finds_one_tainted_byte(self) -> None:
        """A single tainted byte in the middle of a clean range must
        flip is_tainted to True."""
        shm = BitPreciseShadowMemory()
        shm.write_mask(0xA005, 0x80, 1)  # bit 7 of byte 5 only
        assert shm.is_tainted(0xA000, 16) is True
        # But shifting one byte past it goes back to clean
        assert shm.is_tainted(0xA006, 10) is False


# ---------------------------------------------------------------------------
# Round-trip via write_bytes / read_bytes
# ---------------------------------------------------------------------------


class TestByteArrayRoundTrip:
    def test_write_bytes_read_bytes_identity(self) -> None:
        """write_bytes(addr, taint_array) → read_bytes(addr, n) returns
        the exact same array, byte-for-byte."""
        shm = BitPreciseShadowMemory()
        # 16 bytes of distinct per-bit taint patterns
        taint = bytearray(
            [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0xFF, 0xAA, 0x55, 0x0F, 0xF0, 0x69, 0x96, 0xC3],
        )
        shm.write_bytes(0xB000, taint)
        got = shm.read_bytes(0xB000, len(taint))
        assert got == taint, f'mismatch: wrote {bytes(taint).hex()}, read {bytes(got).hex()}'

    def test_partial_read_does_not_consume_bits(self) -> None:
        """Reading is non-destructive — the same shadow can be read again."""
        shm = BitPreciseShadowMemory()
        shm.write_mask(0xC000, 0xFF00FF00FF00FF00, 8)
        first = shm.read_mask(0xC000, 8)
        second = shm.read_mask(0xC000, 8)
        assert first == second == 0xFF00FF00FF00FF00


# ---------------------------------------------------------------------------
# Clear semantics — clear removes ALL bits in the range
# ---------------------------------------------------------------------------


class TestClearSemantics:
    def test_clear_removes_all_bits_in_range(self) -> None:
        shm = BitPreciseShadowMemory()
        shm.write_mask(0xD000, 0xFFFFFFFF, 4)
        shm.clear(0xD000, 4)
        assert shm.read_mask(0xD000, 4) == 0
        assert shm.is_tainted(0xD000, 4) is False

    def test_clear_partial_range_leaves_neighbours(self) -> None:
        """Clearing bytes 1-2 of a 4-byte tainted region leaves 0 and 3."""
        shm = BitPreciseShadowMemory()
        shm.write_mask(0xE000, 0xAABBCCDD, 4)  # bytes [0xDD, 0xCC, 0xBB, 0xAA]
        shm.clear(0xE001, 2)
        assert shm.read_mask(0xE000, 1) == 0xDD
        assert shm.read_mask(0xE001, 1) == 0x00
        assert shm.read_mask(0xE002, 1) == 0x00
        assert shm.read_mask(0xE003, 1) == 0xAA


# ---------------------------------------------------------------------------
# Property tests — bit-precision under stress
# ---------------------------------------------------------------------------


class TestPropertyStyle:
    @pytest.mark.parametrize('taint_byte', [0x01, 0x02, 0x10, 0x80, 0x55, 0xAA, 0xC3, 0x69])
    def test_per_byte_distinct_patterns(self, taint_byte: int) -> None:
        """For every interesting per-bit pattern, write/read must round-trip."""
        shm = BitPreciseShadowMemory()
        shm.write_mask(0xF000, taint_byte, 1)
        assert shm.read_mask(0xF000, 1) == taint_byte

    @pytest.mark.parametrize('size', [1, 2, 4, 8])
    @pytest.mark.parametrize('addr', [0x10000, 0x10003, 0x10FFE, 0x11FFF])
    def test_aligned_and_unaligned_writes(self, size: int, addr: int) -> None:
        """Write/read is correct at any alignment, including across pages."""
        shm = BitPreciseShadowMemory()
        # Per-byte mask that's distinct for every byte
        mask = 0
        for i in range(size):
            mask |= ((0x69 + i) & 0xFF) << (i * 8)
        shm.write_mask(addr, mask, size)
        got = shm.read_mask(addr, size)
        assert got == mask, f'addr={addr:#x} size={size}: expected {mask:#x}, got {got:#x}'
