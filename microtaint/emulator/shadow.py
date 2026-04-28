from __future__ import annotations


class BitPreciseShadowMemory:
    """
    Byte-granular shadow memory for taint and allocation-state tracking.

    Design
    ------
    One shadow byte per memory byte, stored in 4096-byte pages.

    Taint byte semantics
    --------------------
    Each shadow byte is an 8-bit mask where bit i means bit i of the
    corresponding memory byte is tainted.  A value of 0x00 means the
    memory byte is entirely clean; 0xFF means every bit is tainted.

    This is the same representation used by CELLIFT and the classic
    byte-granular shadow memory in AddressSanitizer.

    API contract
    ------------
    All public methods take (address, count) — never a packed integer
    mask with a size suffix.  The caller says "read/write N bytes
    starting at address"; the shadow memory handles page splits
    transparently.

    Poison (UAF) tracking uses a completely separate page dictionary so
    that it never interferes with taint reads/writes.
    """

    STATE_POISONED = 0xFF  # sentinel value in state_pages

    def __init__(self) -> None:
        self.PAGE_SIZE = 4096
        # taint_pages[page_base] = bytearray(PAGE_SIZE)
        # Each element is the taint mask for one memory byte.
        self.taint_pages: dict[int, bytearray] = {}
        # state_pages[page_base] = bytearray(PAGE_SIZE)
        # Non-zero means the byte has been freed/poisoned.
        self.state_pages: dict[int, bytearray] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _page(self, pages: dict[int, bytearray], page_base: int) -> bytearray:
        if page_base not in pages:
            pages[page_base] = bytearray(self.PAGE_SIZE)
        return pages[page_base]

    def _page_base(self, address: int) -> int:
        return address & ~(self.PAGE_SIZE - 1)

    def _offset(self, address: int) -> int:
        return address & (self.PAGE_SIZE - 1)

    # ------------------------------------------------------------------
    # Taint API
    # ------------------------------------------------------------------

    def write_bytes(self, address: int, taint: bytes | bytearray) -> None:
        """
        Write taint into shadow memory, one shadow byte per memory byte.

        *taint* must have exactly as many bytes as the memory region being
        tainted.  taint[i] is the 8-bit taint mask for memory byte at
        address+i.  Writing 0x00 bytes explicitly clears taint.
        """
        for i, tb in enumerate(taint):
            addr = address + i
            page = self._page(self.taint_pages, self._page_base(addr))
            page[self._offset(addr)] = tb

    def read_bytes(self, address: int, count: int) -> bytearray:
        """
        Read *count* taint bytes from shadow memory.

        Returns a bytearray of length *count* where each element is the
        taint mask for the corresponding memory byte.  Uninitialized
        (never-written) addresses return 0x00 (clean).
        """
        result = bytearray(count)
        for i in range(count):
            addr = address + i
            pb = self._page_base(addr)
            if pb in self.taint_pages:
                result[i] = self.taint_pages[pb][self._offset(addr)]
            # else: already 0 from bytearray initialisation
        return result

    def write_mask(self, address: int, mask: int, size: int) -> None:
        """
        Compatibility shim: convert a packed little-endian integer mask to
        per-byte taint and write it.

        *mask* is a (size*8)-bit integer; bit group [i*8 .. i*8+7] is the
        taint mask for memory byte at address+i.
        *size* is the number of bytes to write.

        Calling write_mask(addr, 0, n) explicitly clears n bytes of taint.
        """
        taint = bytearray(size)
        for i in range(size):
            taint[i] = (mask >> (i * 8)) & 0xFF
        self.write_bytes(address, taint)

    def read_mask(self, address: int, size: int) -> int:
        """
        Compatibility shim: read *size* taint bytes and pack them into a
        little-endian integer.

        The result's bit group [i*8 .. i*8+7] is the taint mask for memory
        byte at address+i.  Returns 0 if no bytes in the range are tainted.
        """
        taint = self.read_bytes(address, size)
        result = 0
        for i, tb in enumerate(taint):
            result |= tb << (i * 8)
        return result

    def is_tainted(self, address: int, size: int) -> bool:
        """True if any byte in [address, address+size) carries any taint."""
        for i in range(size):
            addr = address + i
            pb = self._page_base(addr)
            if pb in self.taint_pages and self.taint_pages[pb][self._offset(addr)]:
                return True
        return False

    def clear(self, address: int, size: int) -> None:
        """Explicitly clear taint for *size* bytes starting at *address*."""
        self.write_bytes(address, bytes(size))

    # ------------------------------------------------------------------
    # Poison (UAF) API — separate from taint
    # ------------------------------------------------------------------

    def poison(self, address: int, size: int) -> None:
        """Mark *size* bytes as freed/poisoned (for UAF detection)."""
        for i in range(size):
            addr = address + i
            page = self._page(self.state_pages, self._page_base(addr))
            page[self._offset(addr)] = self.STATE_POISONED

    def unpoison(self, address: int, size: int) -> None:
        """Un-poison *size* bytes (e.g. when a region is re-allocated)."""
        for i in range(size):
            addr = address + i
            pb = self._page_base(addr)
            if pb in self.state_pages:
                self.state_pages[pb][self._offset(addr)] = 0

    def is_poisoned(self, address: int, size: int) -> bool:
        """True if any byte in [address, address+size) is poisoned."""
        for i in range(size):
            addr = address + i
            pb = self._page_base(addr)
            if pb in self.state_pages and self.state_pages[pb][self._offset(addr)]:
                return True
        return False
