"""
Tests for BitPreciseShadowMemory.

These tests are purely in-process — no Qiling, no compilation.
They validate the byte-indexed taint model that the user specified:
  - One taint byte per memory byte, indexed by address alone
  - Size is a count of bytes, never part of the key or the taint encoding
  - write_bytes / read_bytes are the canonical API
  - write_mask / read_mask are compatibility shims (packed little-endian int)
  - Writes of zero explicitly clear taint (stale-slot safety)
  - Poison tracking is entirely separate from taint
  - Cross-page reads and writes work correctly
"""

from __future__ import annotations

import pytest

from microtaint.emulator.shadow import BitPreciseShadowMemory


@pytest.fixture
def shadow() -> BitPreciseShadowMemory:
    return BitPreciseShadowMemory()


# ===========================================================================
# Core byte API
# ===========================================================================


class TestWriteReadBytes:
    def test_clean_on_creation(self, shadow: BitPreciseShadowMemory) -> None:
        result = shadow.read_bytes(0x1000, 8)
        assert result == bytearray(8)

    def test_write_single_byte_fully_tainted(self, shadow: BitPreciseShadowMemory) -> None:
        shadow.write_bytes(0x1000, bytes([0xFF]))
        assert shadow.read_bytes(0x1000, 1) == bytearray([0xFF])

    def test_write_single_byte_partially_tainted(self, shadow: BitPreciseShadowMemory) -> None:
        shadow.write_bytes(0x1000, bytes([0x0F]))  # low nibble tainted
        assert shadow.read_bytes(0x1000, 1) == bytearray([0x0F])

    def test_write_multi_byte_independent(self, shadow: BitPreciseShadowMemory) -> None:
        taint = bytes([0xFF, 0x00, 0xAA, 0x55])
        shadow.write_bytes(0x2000, taint)
        result = shadow.read_bytes(0x2000, 4)
        assert result == bytearray(taint)

    def test_write_zero_clears_stale_taint(self, shadow: BitPreciseShadowMemory) -> None:
        """Writing 0x00 to a byte must clear any previous taint at that address."""
        shadow.write_bytes(0x3000, bytes([0xFF] * 8))
        # Now overwrite with zeros — simulates storing an untainted value
        shadow.write_bytes(0x3000, bytes(8))
        assert shadow.read_bytes(0x3000, 8) == bytearray(8)

    def test_partial_overwrite_preserves_surrounding_bytes(self, shadow: BitPreciseShadowMemory) -> None:
        """Writing 4 bytes must not disturb adjacent bytes."""
        shadow.write_bytes(0x4000, bytes([0xFF] * 8))
        shadow.write_bytes(0x4004, bytes(4))  # clear second half
        result = shadow.read_bytes(0x4000, 8)
        assert result[:4] == bytearray([0xFF] * 4)  # first half intact
        assert result[4:] == bytearray(4)  # second half cleared

    def test_adjacent_bytes_do_not_alias(self, shadow: BitPreciseShadowMemory) -> None:
        shadow.write_bytes(0x5000, bytes([0xFF]))
        shadow.write_bytes(0x5001, bytes([0x00]))
        assert shadow.read_bytes(0x5000, 1) == bytearray([0xFF])
        assert shadow.read_bytes(0x5001, 1) == bytearray([0x00])

    def test_read_subset_of_written_range(self, shadow: BitPreciseShadowMemory) -> None:
        """Reading fewer bytes than were written returns the correct subset."""
        shadow.write_bytes(0x6000, bytes([0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]))
        assert shadow.read_bytes(0x6002, 3) == bytearray([0x33, 0x44, 0x55])

    def test_read_superset_returns_zeros_for_unwritten(self, shadow: BitPreciseShadowMemory) -> None:
        """Reading beyond written range returns 0x00 for uninitialised bytes."""
        shadow.write_bytes(0x7000, bytes([0xFF, 0xFF]))
        result = shadow.read_bytes(0x7000, 4)
        assert result == bytearray([0xFF, 0xFF, 0x00, 0x00])

    def test_size_independence(self, shadow: BitPreciseShadowMemory) -> None:
        """
        Write 4 bytes of taint, read back as 8 bytes.
        The second 4 bytes must be clean (zero), not garbage.
        Write 8 bytes, read back as 4 — must match the first 4.
        This verifies size is just a count, not part of the key.
        """
        shadow.write_bytes(0x8000, bytes([0xAB] * 4))
        result8 = shadow.read_bytes(0x8000, 8)
        assert result8[:4] == bytearray([0xAB] * 4)
        assert result8[4:] == bytearray(4)

        shadow.write_bytes(0x9000, bytes([0xCD] * 8))
        result4 = shadow.read_bytes(0x9000, 4)
        assert result4 == bytearray([0xCD] * 4)


# ===========================================================================
# Packed-integer shim API (write_mask / read_mask)
# ===========================================================================


class TestMaskShim:
    def test_write_mask_fully_tainted_8_bytes(self, shadow: BitPreciseShadowMemory) -> None:
        mask = (1 << 64) - 1  # all 64 bits set
        shadow.write_mask(0x1000, mask, 8)
        assert shadow.read_bytes(0x1000, 8) == bytearray([0xFF] * 8)

    def test_write_mask_zero_clears(self, shadow: BitPreciseShadowMemory) -> None:
        shadow.write_mask(0x2000, (1 << 64) - 1, 8)
        shadow.write_mask(0x2000, 0, 8)
        assert shadow.read_bytes(0x2000, 8) == bytearray(8)

    def test_write_mask_partial_bits(self, shadow: BitPreciseShadowMemory) -> None:
        # Only byte 0 fully tainted, byte 1 half tainted (low nibble)
        mask = 0xFF | (0x0F << 8)
        shadow.write_mask(0x3000, mask, 2)
        result = shadow.read_bytes(0x3000, 2)
        assert result[0] == 0xFF
        assert result[1] == 0x0F

    def test_read_mask_round_trips(self, shadow: BitPreciseShadowMemory) -> None:
        mask = 0xDEADBEEFCAFEBABE
        shadow.write_mask(0x4000, mask, 8)
        assert shadow.read_mask(0x4000, 8) == mask

    def test_read_mask_size_independent(self, shadow: BitPreciseShadowMemory) -> None:
        """
        Write 8 bytes via mask, read back as 4 bytes.
        Result must equal the low 32 bits of the original mask.
        """
        mask = 0xAABBCCDD11223344
        shadow.write_mask(0x5000, mask, 8)
        low32_mask = mask & 0xFFFFFFFF
        assert shadow.read_mask(0x5000, 4) == low32_mask

    def test_write_different_sizes_same_address(self, shadow: BitPreciseShadowMemory) -> None:
        """
        Write 8-byte mask, then overwrite with 4-byte mask.
        Only the first 4 bytes change; bytes 4-7 retain old taint.
        """
        shadow.write_mask(0x6000, 0xFFFFFFFFFFFFFFFF, 8)
        shadow.write_mask(0x6000, 0x00000000, 4)  # clear first 4 bytes
        result = shadow.read_bytes(0x6000, 8)
        assert result[:4] == bytearray(4)  # cleared
        assert result[4:] == bytearray([0xFF] * 4)  # still tainted


# ===========================================================================
# Cross-page reads and writes
# ===========================================================================


class TestCrossPage:
    def test_write_spanning_page_boundary(self, shadow: BitPreciseShadowMemory) -> None:
        page_size = shadow.PAGE_SIZE
        addr = page_size - 2  # 2 bytes before end of page 0
        shadow.write_bytes(addr, bytes([0xAA, 0xBB, 0xCC, 0xDD]))
        result = shadow.read_bytes(addr, 4)
        assert result == bytearray([0xAA, 0xBB, 0xCC, 0xDD])

    def test_read_spanning_page_boundary(self, shadow: BitPreciseShadowMemory) -> None:
        page_size = shadow.PAGE_SIZE
        addr = page_size - 1
        shadow.write_bytes(addr, bytes([0x12]))  # last byte of page 0
        shadow.write_bytes(addr + 1, bytes([0x34]))  # first byte of page 1
        result = shadow.read_bytes(addr, 2)
        assert result == bytearray([0x12, 0x34])

    def test_mask_spanning_page_boundary(self, shadow: BitPreciseShadowMemory) -> None:
        page_size = shadow.PAGE_SIZE
        addr = page_size - 4
        mask = 0xFFFF_FFFF_FFFF_FFFF  # 8 bytes, all tainted
        shadow.write_mask(addr, mask, 8)
        assert shadow.read_mask(addr, 8) == mask

    def test_uninitialised_page_returns_zero(self, shadow: BitPreciseShadowMemory) -> None:
        result = shadow.read_bytes(0xDEAD0000, 16)
        assert result == bytearray(16)


# ===========================================================================
# is_tainted helper
# ===========================================================================


class TestIsTainted:
    def test_clean_memory_not_tainted(self, shadow: BitPreciseShadowMemory) -> None:
        assert not shadow.is_tainted(0x1000, 8)

    def test_fully_tainted_range(self, shadow: BitPreciseShadowMemory) -> None:
        shadow.write_bytes(0x1000, bytes([0xFF] * 8))
        assert shadow.is_tainted(0x1000, 8)

    def test_partially_tainted_in_range(self, shadow: BitPreciseShadowMemory) -> None:
        # Only one byte at offset 3 is tainted
        shadow.write_bytes(0x1003, bytes([0x01]))
        assert shadow.is_tainted(0x1000, 8)

    def test_taint_outside_range_not_detected(self, shadow: BitPreciseShadowMemory) -> None:
        shadow.write_bytes(0x2000, bytes([0xFF]))
        assert not shadow.is_tainted(0x2001, 4)

    def test_single_bit_tainted(self, shadow: BitPreciseShadowMemory) -> None:
        shadow.write_bytes(0x3000, bytes([0x01]))  # only bit 0 tainted
        assert shadow.is_tainted(0x3000, 1)

    def test_clear_removes_taint(self, shadow: BitPreciseShadowMemory) -> None:
        shadow.write_bytes(0x4000, bytes([0xFF] * 4))
        shadow.clear(0x4000, 4)
        assert not shadow.is_tainted(0x4000, 4)


# ===========================================================================
# Poison / UAF tracking (separate from taint)
# ===========================================================================


class TestPoison:
    def test_fresh_memory_not_poisoned(self, shadow: BitPreciseShadowMemory) -> None:
        assert not shadow.is_poisoned(0x1000, 8)

    def test_poison_marks_region(self, shadow: BitPreciseShadowMemory) -> None:
        shadow.poison(0x2000, 16)
        assert shadow.is_poisoned(0x2000, 16)

    def test_poison_single_byte_detected_in_range(self, shadow: BitPreciseShadowMemory) -> None:
        shadow.poison(0x3004, 1)
        assert shadow.is_poisoned(0x3000, 8)

    def test_poison_does_not_affect_taint(self, shadow: BitPreciseShadowMemory) -> None:
        shadow.poison(0x4000, 8)
        assert not shadow.is_tainted(0x4000, 8)
        # Taint independently
        shadow.write_bytes(0x4000, bytes([0xFF]))
        assert shadow.is_tainted(0x4000, 1)
        assert shadow.is_poisoned(0x4000, 8)

    def test_taint_does_not_affect_poison(self, shadow: BitPreciseShadowMemory) -> None:
        shadow.write_bytes(0x5000, bytes([0xFF] * 8))
        assert not shadow.is_poisoned(0x5000, 8)

    def test_unpoison_clears_region(self, shadow: BitPreciseShadowMemory) -> None:
        shadow.poison(0x6000, 8)
        shadow.unpoison(0x6000, 8)
        assert not shadow.is_poisoned(0x6000, 8)

    def test_unpoison_partial(self, shadow: BitPreciseShadowMemory) -> None:
        shadow.poison(0x7000, 8)
        shadow.unpoison(0x7000, 4)
        assert not shadow.is_poisoned(0x7000, 4)
        assert shadow.is_poisoned(0x7004, 4)

    def test_poison_cross_page(self, shadow: BitPreciseShadowMemory) -> None:
        addr = shadow.PAGE_SIZE - 2
        shadow.poison(addr, 4)
        assert shadow.is_poisoned(addr, 4)
        assert not shadow.is_poisoned(addr - 1, 1)
        assert not shadow.is_poisoned(addr + 4, 1)
