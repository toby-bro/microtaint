# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: infer_types=True

from libc.stdint cimport uint64_t, uint8_t

# ---------------------------------------------------------------------------
# BitPreciseShadowMemory
#
# Byte-granular shadow memory for taint and allocation-state tracking.
# Identical semantics to the original shadow.py but all hot methods are
# cpdef so Cython-to-Cython calls never touch the Python object model.
#
# One shadow byte per memory byte, stored in 4096-byte pages (bytearray).
# Taint byte: bit i set means bit i of the corresponding memory byte is tainted.
# Poison pages are separate from taint pages — UAF tracking never interferes
# with taint reads/writes.
# ---------------------------------------------------------------------------

cdef int PAGE_SIZE = 4096
cdef int PAGE_MASK = PAGE_SIZE - 1
cdef uint8_t STATE_POISONED = 0xFF


cdef class BitPreciseShadowMemory:
    """
    Fast Cython implementation of bit-precise shadow memory.

    All public methods are cpdef — callable from both Python and Cython.
    Hot paths (read_mask, write_mask, is_tainted, is_poisoned) use typed
    C locals and never allocate Python integers in their inner loops.
    """

    cdef dict taint_pages   # page_base (int) -> bytearray(PAGE_SIZE)
    cdef dict state_pages   # page_base (int) -> bytearray(PAGE_SIZE), UAF poison

    def __init__(self):
        self.taint_pages = {}
        self.state_pages = {}

    # ------------------------------------------------------------------
    # Internal helpers — cdef, never visible to Python
    # ------------------------------------------------------------------

    cdef inline bytearray _get_taint_page(self, uint64_t page_base):
        cdef bytearray page
        try:
            return <bytearray>self.taint_pages[page_base]
        except KeyError:
            page = bytearray(PAGE_SIZE)
            self.taint_pages[page_base] = page
            return page

    cdef inline bytearray _get_state_page(self, uint64_t page_base):
        cdef bytearray page
        try:
            return <bytearray>self.state_pages[page_base]
        except KeyError:
            page = bytearray(PAGE_SIZE)
            self.state_pages[page_base] = page
            return page

    cdef inline uint64_t _page_base(self, uint64_t address):
        return address & ~<uint64_t>PAGE_MASK

    cdef inline int _offset(self, uint64_t address):
        return <int>(address & <uint64_t>PAGE_MASK)

    # ------------------------------------------------------------------
    # Taint API — cpdef so both Python and Cython can call without boxing
    # ------------------------------------------------------------------

    cpdef void write_bytes(self, uint64_t address, object taint):
        """
        Write taint into shadow memory, one shadow byte per memory byte.
        taint[i] is the 8-bit mask for memory byte at address+i.
        Writing 0x00 bytes explicitly clears taint.
        """
        cdef int i, length
        cdef uint64_t addr
        cdef bytearray page
        cdef uint8_t tb

        length = len(taint)
        for i in range(length):
            addr = address + <uint64_t>i
            tb   = <uint8_t>taint[i]
            page = self._get_taint_page(self._page_base(addr))
            page[self._offset(addr)] = tb

    cpdef bytearray read_bytes(self, uint64_t address, int count):
        """
        Read count taint bytes. Returns bytearray(count); uninitialized = 0x00.
        """
        cdef bytearray result = bytearray(count)
        cdef int i
        cdef uint64_t addr, pb
        cdef bytearray page

        for i in range(count):
            addr = address + <uint64_t>i
            pb   = self._page_base(addr)
            if pb in self.taint_pages:
                page      = <bytearray>self.taint_pages[pb]
                result[i] = page[self._offset(addr)]
        return result

    cpdef void write_mask(self, uint64_t address, uint64_t mask, int size):
        """
        Convert a packed little-endian integer mask to per-byte taint and write.
        mask bit group [i*8 .. i*8+7] is the taint mask for memory byte address+i.
        Calling write_mask(addr, 0, n) explicitly clears n bytes of taint.
        """
        cdef int i
        cdef uint64_t addr
        cdef uint8_t byte_taint
        cdef bytearray page

        for i in range(size):
            addr       = address + <uint64_t>i
            byte_taint = <uint8_t>((mask >> (i * 8)) & 0xFF)
            page       = self._get_taint_page(self._page_base(addr))
            page[self._offset(addr)] = byte_taint

    cpdef uint64_t read_mask(self, uint64_t address, int size):
        """
        Read size taint bytes and pack into a little-endian integer.
        Bit group [i*8 .. i*8+7] is the taint mask for memory byte address+i.
        Returns 0 if no bytes in the range are tainted.
        """
        cdef uint64_t result = 0
        cdef int i
        cdef uint64_t addr, pb
        cdef bytearray page
        cdef uint8_t tb

        for i in range(size):
            addr = address + <uint64_t>i
            pb   = self._page_base(addr)
            if pb in self.taint_pages:
                page = <bytearray>self.taint_pages[pb]
                tb   = <uint8_t>page[self._offset(addr)]
                if tb:
                    result |= (<uint64_t>tb) << (i * 8)
        return result

    cpdef bint is_tainted(self, uint64_t address, int size):
        """True if any byte in [address, address+size) carries any taint."""
        cdef int i
        cdef uint64_t addr, pb
        cdef bytearray page

        for i in range(size):
            addr = address + <uint64_t>i
            pb   = self._page_base(addr)
            if pb in self.taint_pages:
                page = <bytearray>self.taint_pages[pb]
                if page[self._offset(addr)]:
                    return True
        return False

    cpdef void clear(self, uint64_t address, int size):
        """Explicitly clear taint for size bytes starting at address."""
        cdef int i
        cdef uint64_t addr
        cdef bytearray page

        for i in range(size):
            addr = address + <uint64_t>i
            pb   = self._page_base(addr)
            if pb in self.taint_pages:
                page = <bytearray>self.taint_pages[pb]
                page[self._offset(addr)] = 0

    # ------------------------------------------------------------------
    # Poison (UAF) API — separate page dict, never interferes with taint
    # ------------------------------------------------------------------

    cpdef void poison(self, uint64_t address, int size):
        """Mark size bytes as freed/poisoned (for UAF detection)."""
        cdef int i
        cdef uint64_t addr
        cdef bytearray page

        for i in range(size):
            addr = address + <uint64_t>i
            page = self._get_state_page(self._page_base(addr))
            page[self._offset(addr)] = STATE_POISONED

    cpdef void unpoison(self, uint64_t address, int size):
        """Un-poison size bytes (e.g. when a region is re-allocated)."""
        cdef int i
        cdef uint64_t addr, pb
        cdef bytearray page

        for i in range(size):
            addr = address + <uint64_t>i
            pb   = self._page_base(addr)
            if pb in self.state_pages:
                page = <bytearray>self.state_pages[pb]
                page[self._offset(addr)] = 0

    cpdef bint is_poisoned(self, uint64_t address, int size):
        """True if any byte in [address, address+size) is poisoned."""
        cdef int i
        cdef uint64_t addr, pb
        cdef bytearray page

        for i in range(size):
            addr = address + <uint64_t>i
            pb   = self._page_base(addr)
            if pb in self.state_pages:
                page = <bytearray>self.state_pages[pb]
                if page[self._offset(addr)]:
                    return True
        return False

    # ------------------------------------------------------------------
    # Python-visible constants (for external code that checks STATE_POISONED)
    # ------------------------------------------------------------------

    @property
    def PAGE_SIZE(self) -> int:
        return PAGE_SIZE

    @property
    def STATE_POISONED(self) -> int:
        return STATE_POISONED
