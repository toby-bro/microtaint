"""
test_shadow_memory.py
=====================
Direct unit tests for BitPreciseShadowMemory.

These tests do NOT involve Qiling, the wrapper, or any binary execution.
They verify the documented semantics of write_mask / read_mask / clear /
poison / unpoison directly.

Why
---
If the avalanche evaluation is wrong, we need to be sure the shadow layer
itself is bulletproof first.  Any bug here cascades upward.

Usage
-----
    uv run pytest test_shadow_memory.py -v
"""

# mypy: disable-error-code="no-untyped-def,no-untyped-call,type-arg,arg-type"

from __future__ import annotations

import pytest

from microtaint.emulator.shadow import BitPreciseShadowMemory

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def shadow() -> BitPreciseShadowMemory:
    return BitPreciseShadowMemory()


# ===========================================================================
# Round-trip tests for write_mask / read_mask
# ===========================================================================


class TestRoundTrip:
    """write_mask then read_mask at the same address must return the same value."""

    @pytest.mark.parametrize('size', [1, 2, 3, 4, 5, 6, 7, 8])
    def test_full_mask_round_trip(self, shadow, size):
        """A mask of all 1s for `size` bytes round-trips correctly."""
        addr = 0x1000
        mask = (1 << (size * 8)) - 1
        shadow.write_mask(addr, mask, size)
        assert shadow.read_mask(addr, size) == mask, f'size={size}: full mask {mask:#x} did not round-trip'

    def test_single_bit_round_trip_per_byte_position(self, shadow):
        """
        Per-byte tainting: shadow[addr+i] should hold the byte at bit-position
        i*8 in the packed mask integer.  Verifies the LE byte packing.
        """
        addr = 0x2000
        # Set distinct value in each shadow byte: byte i = (i+1) << 0
        mask = 0
        for i in range(8):
            mask |= (i + 1) << (i * 8)
        shadow.write_mask(addr, mask, 8)
        for i in range(8):
            byte_shadow = shadow.read_mask(addr + i, 1)
            expected = (i + 1) & 0xFF
            assert byte_shadow == expected, f'byte {i}: expected {expected:#x}, got {byte_shadow:#x}'
        # Reading all 8 bytes back as a packed mask should give us the original
        assert shadow.read_mask(addr, 8) == mask

    def test_zero_mask_clears(self, shadow):
        """write_mask(addr, 0, n) must clear n bytes of pre-existing taint."""
        addr = 0x3000
        shadow.write_mask(addr, 0xFFFFFFFFFFFFFFFF, 8)
        assert shadow.read_mask(addr, 8) == 0xFFFFFFFFFFFFFFFF
        shadow.write_mask(addr, 0, 8)
        assert shadow.read_mask(addr, 8) == 0, 'Zero mask did not clear taint'


# ===========================================================================
# Page boundary semantics
# ===========================================================================


class TestPageBoundary:
    """Shadow uses 4096-byte pages internally — taint across boundaries must work."""

    def test_taint_spanning_page_boundary(self, shadow):
        """
        Write 8 bytes starting at 0xFFC: 4 bytes in page 0x000, 4 in page 0x1000.
        Read them back.
        """
        addr = 0xFFC  # last 4 bytes of first page
        mask = 0x1122334455667788
        shadow.write_mask(addr, mask, 8)
        assert shadow.read_mask(addr, 8) == mask, 'Cross-page round-trip failed'

    def test_high_address_far_from_zero(self, shadow):
        """Writing at a high stack-like address (Qiling's stack base) must work."""
        addr = 0x80000000DD30  # the actual addr we saw in diag2
        shadow.write_mask(addr, 0x01, 1)
        assert shadow.read_mask(addr, 1) == 0x01

    def test_two_distant_addresses_independent(self, shadow):
        """Writes at distant addresses (different pages) don't affect each other."""
        shadow.write_mask(0x1000, 0xAB, 1)
        shadow.write_mask(0x80000000, 0xCD, 1)
        assert shadow.read_mask(0x1000, 1) == 0xAB
        assert shadow.read_mask(0x80000000, 1) == 0xCD


# ===========================================================================
# is_tainted boolean
# ===========================================================================


class TestIsTainted:
    def test_untainted_returns_false(self, shadow):
        assert not shadow.is_tainted(0x1000, 8)

    def test_any_byte_tainted_returns_true(self, shadow):
        shadow.write_mask(0x1004, 0x01, 1)  # taint one byte in the middle
        assert shadow.is_tainted(0x1000, 8)
        assert shadow.is_tainted(0x1004, 1)
        assert not shadow.is_tainted(0x1000, 4)  # range before the taint
        assert not shadow.is_tainted(0x1005, 3)  # range after

    def test_partial_overlap_with_tainted_region(self, shadow):
        shadow.write_mask(0x1010, 0xFFFFFFFF, 4)
        # Range [0x100E, 0x1014) overlaps tainted region [0x1010, 0x1014)
        assert shadow.is_tainted(0x100E, 6)


# ===========================================================================
# clear()
# ===========================================================================


class TestClear:
    def test_clear_removes_taint(self, shadow):
        shadow.write_mask(0x1000, 0xAB, 1)
        assert shadow.read_mask(0x1000, 1) == 0xAB
        shadow.clear(0x1000, 1)
        assert shadow.read_mask(0x1000, 1) == 0

    def test_clear_only_affects_specified_range(self, shadow):
        shadow.write_mask(0x1000, 0xFFFFFFFFFFFFFFFF, 8)
        shadow.clear(0x1002, 4)  # clear middle 4 bytes
        assert shadow.read_mask(0x1000, 1) == 0xFF
        assert shadow.read_mask(0x1001, 1) == 0xFF
        assert shadow.read_mask(0x1002, 1) == 0
        assert shadow.read_mask(0x1005, 1) == 0
        assert shadow.read_mask(0x1006, 1) == 0xFF
        assert shadow.read_mask(0x1007, 1) == 0xFF


# ===========================================================================
# write_bytes / read_bytes (per-byte-array form)
# ===========================================================================


class TestWriteBytesReadBytes:
    def test_write_bytes_round_trip(self, shadow):
        addr = 0x1000
        payload = bytes([0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80])
        shadow.write_bytes(addr, payload)
        assert bytes(shadow.read_bytes(addr, 8)) == payload

    def test_write_bytes_zero_clears(self, shadow):
        shadow.write_mask(0x1000, 0xFFFFFFFFFFFFFFFF, 8)
        shadow.write_bytes(0x1000, bytes(8))  # all zeros
        assert shadow.read_mask(0x1000, 8) == 0


# ===========================================================================
# Poison API (UAF detection — must NOT interfere with taint pages)
# ===========================================================================


class TestPoisonIsolation:
    def test_poison_does_not_affect_taint(self, shadow):
        shadow.write_mask(0x1000, 0xAB, 1)
        shadow.poison(0x1000, 1)
        # Taint is preserved
        assert shadow.read_mask(0x1000, 1) == 0xAB
        # And we know it's poisoned
        assert shadow.is_poisoned(0x1000, 1)

    def test_taint_does_not_affect_poison(self, shadow):
        shadow.poison(0x2000, 8)
        shadow.write_mask(0x2000, 0xFF, 1)
        assert shadow.is_poisoned(0x2000, 8)

    def test_unpoison_clears_poison_only(self, shadow):
        shadow.write_mask(0x3000, 0xFF, 1)
        shadow.poison(0x3000, 1)
        shadow.unpoison(0x3000, 1)
        assert not shadow.is_poisoned(0x3000, 1)
        # Taint is unaffected
        assert shadow.read_mask(0x3000, 1) == 0xFF

    def test_is_poisoned_default_false(self, shadow):
        assert not shadow.is_poisoned(0x4000, 8)
