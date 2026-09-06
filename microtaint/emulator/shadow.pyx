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

    cdef attribute declarations and cpdef method signatures are in shadow.pxd
    so that other Cython modules can `cimport BitPreciseShadowMemory` and
    dispatch its cpdef methods at C level (skipping the Python attribute
    lookup + bound-method-call path).
    """

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
        cdef uint64_t addr, pb
        cdef uint64_t cur_pb = 0
        cdef bint have = False
        cdef bytearray page = None
        cdef uint8_t tb

        length = len(taint)
        for i in range(length):
            addr = address + <uint64_t>i
            tb   = <uint8_t>taint[i]
            pb   = self._page_base(addr)
            if not have or pb != cur_pb:
                page   = self._get_taint_page(pb)
                cur_pb = pb
                have   = True
            page[self._offset(addr)] = tb

    cpdef bytearray read_bytes(self, uint64_t address, int count):
        """
        Read count taint bytes. Returns bytearray(count); uninitialized = 0x00.
        """
        cdef bytearray result = bytearray(count)
        cdef int i
        cdef uint64_t addr, pb
        cdef uint64_t cur_pb = 0
        cdef bint have = False
        cdef bytearray page = None

        for i in range(count):
            addr = address + <uint64_t>i
            pb   = self._page_base(addr)
            if not have or pb != cur_pb:
                page   = <bytearray>self.taint_pages.get(pb)
                cur_pb = pb
                have   = True
            if page is not None:
                result[i] = page[self._offset(addr)]
        return result

    cpdef void write_mask(self, uint64_t address, uint64_t mask, int size):
        """
        Convert a packed little-endian integer mask to per-byte taint and write.
        mask bit group [i*8 .. i*8+7] is the taint mask for memory byte address+i.
        Calling write_mask(addr, 0, n) explicitly clears n bytes of taint.
        """
        cdef int i
        cdef uint64_t addr, pb
        cdef uint64_t cur_pb = 0
        cdef bint have = False
        cdef uint8_t byte_taint
        cdef bytearray page = None

        for i in range(size):
            addr       = address + <uint64_t>i
            byte_taint = <uint8_t>((mask >> (i * 8)) & 0xFF)
            pb         = self._page_base(addr)
            if not have or pb != cur_pb:
                page   = self._get_taint_page(pb)
                cur_pb = pb
                have   = True
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
        cdef uint64_t cur_pb = 0
        cdef bint have = False
        cdef bytearray page = None
        cdef uint8_t tb

        # Cache the current page across the byte loop: a read almost always
        # lies within one 4096-byte page, so this turns the old per-byte
        # `contains`+subscript double dict lookup into ONE `.get()` per page.
        # Bit-exact: same bytes read, same little-endian packing.
        for i in range(size):
            addr = address + <uint64_t>i
            pb   = self._page_base(addr)
            if not have or pb != cur_pb:
                page   = <bytearray>self.taint_pages.get(pb)
                cur_pb = pb
                have   = True
            if page is not None:
                tb = <uint8_t>page[self._offset(addr)]
                if tb:
                    result |= (<uint64_t>tb) << (i * 8)
        return result

    cpdef bint is_tainted(self, uint64_t address, int size):
        """True if any byte in [address, address+size) carries any taint."""
        cdef int i
        cdef uint64_t addr, pb
        cdef uint64_t cur_pb = 0
        cdef bint have = False
        cdef bytearray page = None

        for i in range(size):
            addr = address + <uint64_t>i
            pb   = self._page_base(addr)
            if not have or pb != cur_pb:
                page   = <bytearray>self.taint_pages.get(pb)
                cur_pb = pb
                have   = True
            if page is not None and page[self._offset(addr)]:
                return True
        return False

    cpdef void clear(self, uint64_t address, int size):
        """Explicitly clear taint for size bytes starting at address."""
        cdef int i
        cdef uint64_t addr, pb
        cdef uint64_t cur_pb = 0
        cdef bint have = False
        cdef bytearray page = None

        for i in range(size):
            addr = address + <uint64_t>i
            pb   = self._page_base(addr)
            if not have or pb != cur_pb:
                page   = <bytearray>self.taint_pages.get(pb)
                cur_pb = pb
                have   = True
            if page is not None:
                page[self._offset(addr)] = 0

    # ------------------------------------------------------------------
    # Poison (UAF) API — separate page dict, never interferes with taint
    # ------------------------------------------------------------------

    cpdef void poison(self, uint64_t address, int size):
        """Mark size bytes as freed/poisoned (for UAF detection)."""
        cdef int i
        cdef uint64_t addr, pb
        cdef uint64_t cur_pb = 0
        cdef bint have = False
        cdef bytearray page = None

        for i in range(size):
            addr = address + <uint64_t>i
            pb   = self._page_base(addr)
            if not have or pb != cur_pb:
                page   = self._get_state_page(pb)
                cur_pb = pb
                have   = True
            page[self._offset(addr)] = STATE_POISONED

    cpdef void unpoison(self, uint64_t address, int size):
        """Un-poison size bytes (e.g. when a region is re-allocated)."""
        cdef int i
        cdef uint64_t addr, pb
        cdef uint64_t cur_pb = 0
        cdef bint have = False
        cdef bytearray page = None

        for i in range(size):
            addr = address + <uint64_t>i
            pb   = self._page_base(addr)
            if not have or pb != cur_pb:
                page   = <bytearray>self.state_pages.get(pb)
                cur_pb = pb
                have   = True
            if page is not None:
                page[self._offset(addr)] = 0

    cpdef bint is_poisoned(self, uint64_t address, int size):
        """True if any byte in [address, address+size) is poisoned."""
        cdef int i
        cdef uint64_t addr, pb
        cdef uint64_t cur_pb = 0
        cdef bint have = False
        cdef bytearray page = None

        for i in range(size):
            addr = address + <uint64_t>i
            pb   = self._page_base(addr)
            if not have or pb != cur_pb:
                page   = <bytearray>self.state_pages.get(pb)
                cur_pb = pb
                have   = True
            if page is not None and page[self._offset(addr)]:
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


# ---------------------------------------------------------------------------
# C-API capsule: lets a hand-written C consumer (circuit_c) call read_mask /
# write_mask at the C level, with no PyObject_CallMethod attribute-resolution
# churn.  Mirrors the cell_c._cell_capi pattern.  The function pointers take the
# shadow as a void* that is really the BitPreciseShadowMemory PyObject* the C
# caller already holds a (borrowed) reference to; the cast performs no refcount
# change, so the caller must keep the object alive for the call's duration (it
# does — the shadow lives for the whole run).  Only the two hot memory taint
# ops are exported; everything else stays on the normal cpdef path.
# ---------------------------------------------------------------------------
from cpython.pycapsule cimport PyCapsule_New


ctypedef uint64_t (*read_mask_fn)(object shadow, uint64_t address, int size) noexcept
ctypedef void (*write_mask_fn)(object shadow, uint64_t address, uint64_t mask, int size) noexcept


cdef struct _ShadowCAPI:
    read_mask_fn  read_mask
    write_mask_fn write_mask


cdef uint64_t _capi_read_mask(object shadow, uint64_t address, int size) noexcept:
    # `shadow` is a borrowed ref the C caller already holds; the cast is
    # unchecked (the caller only ever passes a BitPreciseShadowMemory).
    return (<BitPreciseShadowMemory>shadow).read_mask(address, size)


cdef void _capi_write_mask(object shadow, uint64_t address, uint64_t mask, int size) noexcept:
    (<BitPreciseShadowMemory>shadow).write_mask(address, mask, size)


cdef _ShadowCAPI _shadow_capi_struct
_shadow_capi_struct.read_mask = _capi_read_mask
_shadow_capi_struct.write_mask = _capi_write_mask

# Module attribute imported by circuit_c via the shadow module + getattr.
_shadow_capi = PyCapsule_New(<void *>&_shadow_capi_struct,
                             b"microtaint.emulator.shadow._shadow_capi", NULL)
