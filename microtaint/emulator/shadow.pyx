# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: infer_types=True

from libc.stdint cimport uint64_t, uint8_t
from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memset

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




# ---------------------------------------------------------------------------
# C page table.  Replaces the Python dict of bytearrays: a dict keyed by a
# Python int allocated a PyLong per lookup and ran a hashed compare, which
# showed up in the taint hot path as PyDict_GetItemWithError +
# PyLong_FromUnsignedLong + RichCompareBool.  Open addressing, power-of-two
# capacity, linear probing.  EMPTY_KEY can never collide with a real page base
# because page bases are 4096-aligned.
# ---------------------------------------------------------------------------
cdef uint64_t EMPTY_KEY = <uint64_t>0xFFFFFFFFFFFFFFFF


cdef int pm_init(PageMap *m, Py_ssize_t cap) noexcept nogil:
    cdef Py_ssize_t i
    m.keys = <uint64_t *>malloc(<size_t>cap * sizeof(uint64_t))
    if m.keys == NULL:
        return -1
    m.vals = <unsigned char **>calloc(<size_t>cap, sizeof(unsigned char *))
    if m.vals == NULL:
        free(m.keys); m.keys = NULL
        return -1
    for i in range(cap):
        m.keys[i] = EMPTY_KEY
    m.cap = cap
    m.n = 0
    return 0


cdef void pm_free(PageMap *m) noexcept nogil:
    cdef Py_ssize_t i
    if m.vals != NULL:
        for i in range(m.cap):
            if m.vals[i] != NULL:
                free(m.vals[i])
        free(m.vals); m.vals = NULL
    if m.keys != NULL:
        free(m.keys); m.keys = NULL
    m.cap = 0
    m.n = 0


cdef inline Py_ssize_t pm_slot(PageMap *m, uint64_t key) noexcept nogil:
    cdef uint64_t h = key * <uint64_t>0x9E3779B97F4A7C15
    h ^= h >> 29
    cdef Py_ssize_t i = <Py_ssize_t>(h & <uint64_t>(m.cap - 1))
    while m.keys[i] != EMPTY_KEY and m.keys[i] != key:
        i = (i + 1) & (m.cap - 1)
    return i


cdef int pm_grow(PageMap *m) noexcept nogil:
    cdef PageMap nm
    cdef Py_ssize_t i, j
    if pm_init(&nm, m.cap * 2) != 0:
        return -1
    for i in range(m.cap):
        if m.keys[i] != EMPTY_KEY:
            j = pm_slot(&nm, m.keys[i])
            nm.keys[j] = m.keys[i]
            nm.vals[j] = m.vals[i]
            nm.n += 1
    # free the old spine only; the pages themselves moved to the new table
    free(m.keys)
    free(m.vals)
    m.keys = nm.keys
    m.vals = nm.vals
    m.cap = nm.cap
    m.n = nm.n
    return 0


cdef inline unsigned char *pm_get(PageMap *m, uint64_t key) noexcept nogil:
    """Page for key, or NULL.  Never allocates."""
    cdef Py_ssize_t i
    if m.cap == 0:
        return NULL
    i = pm_slot(m, key)
    if m.keys[i] == key:
        return m.vals[i]
    return NULL


cdef inline unsigned char *pm_get_or_create(PageMap *m, uint64_t key, int page_size) noexcept nogil:
    """Page for key, allocating a zeroed one if absent.  NULL only on OOM."""
    cdef Py_ssize_t i
    cdef unsigned char *p
    if m.cap == 0:
        if pm_init(m, 64) != 0:
            return NULL
    if (m.n + 1) * 10 >= m.cap * 7:      # keep load factor under ~0.7
        if pm_grow(m) != 0:
            return NULL
    i = pm_slot(m, key)
    if m.keys[i] == key:
        return m.vals[i]
    p = <unsigned char *>calloc(<size_t>page_size, 1)
    if p == NULL:
        return NULL
    m.keys[i] = key
    m.vals[i] = p
    m.n += 1
    return p


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
        # Lazily sized; pm_get_or_create initialises on first write.
        self.taint_map.keys = NULL
        self.taint_map.vals = NULL
        self.taint_map.cap = 0
        self.taint_map.n = 0
        self.state_map.keys = NULL
        self.state_map.vals = NULL
        self.state_map.cap = 0
        self.state_map.n = 0

    def __dealloc__(self):
        pm_free(&self.taint_map)
        pm_free(&self.state_map)

    # ------------------------------------------------------------------
    # Internal helpers — pure C, no Python objects, callable without the GIL
    # ------------------------------------------------------------------

    cdef inline uint64_t _page_base(self, uint64_t address) noexcept nogil:
        return address & ~<uint64_t>PAGE_MASK

    cdef inline int _offset(self, uint64_t address) noexcept nogil:
        return <int>(address & <uint64_t>PAGE_MASK)

    cdef uint64_t read_mask_c(self, uint64_t address, int size) noexcept nogil:
        """Pure-C core of read_mask.  Same little-endian packing, same page
        caching across the byte loop; just no Python anywhere."""
        cdef uint64_t result = 0
        cdef int i
        cdef uint64_t addr, pb
        cdef uint64_t cur_pb = 0
        cdef bint have = False
        cdef unsigned char *page = NULL
        cdef unsigned char tb
        for i in range(size):
            addr = address + <uint64_t>i
            pb = addr & ~<uint64_t>PAGE_MASK
            if not have or pb != cur_pb:
                page = pm_get(&self.taint_map, pb)
                cur_pb = pb
                have = True
            if page != NULL:
                tb = page[<int>(addr & <uint64_t>PAGE_MASK)]
                if tb:
                    result |= (<uint64_t>tb) << (i * 8)
        return result

    cdef void write_mask_c(self, uint64_t address, uint64_t mask, int size) noexcept nogil:
        """Pure-C core of write_mask.  Writing 0 explicitly clears, exactly as
        before, so an all-zero mask still allocates the page and zeroes it."""
        cdef int i
        cdef uint64_t addr, pb
        cdef uint64_t cur_pb = 0
        cdef bint have = False
        cdef unsigned char *page = NULL
        for i in range(size):
            addr = address + <uint64_t>i
            pb = addr & ~<uint64_t>PAGE_MASK
            if not have or pb != cur_pb:
                page = pm_get_or_create(&self.taint_map, pb, PAGE_SIZE)
                cur_pb = pb
                have = True
            if page != NULL:
                page[<int>(addr & <uint64_t>PAGE_MASK)] = <unsigned char>((mask >> (i * 8)) & 0xFF)

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
        cdef unsigned char *page = NULL
        cdef unsigned char tb

        length = len(taint)
        for i in range(length):
            addr = address + <uint64_t>i
            tb = <unsigned char>taint[i]
            pb = self._page_base(addr)
            if not have or pb != cur_pb:
                page = pm_get_or_create(&self.taint_map, pb, PAGE_SIZE)
                cur_pb = pb
                have = True
            if page != NULL:
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
        cdef unsigned char *page = NULL

        for i in range(count):
            addr = address + <uint64_t>i
            pb = self._page_base(addr)
            if not have or pb != cur_pb:
                page = pm_get(&self.taint_map, pb)
                cur_pb = pb
                have = True
            if page != NULL:
                result[i] = page[self._offset(addr)]
        return result

    cpdef void write_mask(self, uint64_t address, uint64_t mask, int size):
        """
        Convert a packed little-endian integer mask to per-byte taint and write.
        mask bit group [i*8 .. i*8+7] is the taint mask for memory byte address+i.
        Calling write_mask(addr, 0, n) explicitly clears n bytes of taint.
        """
        self.write_mask_c(address, mask, size)

    cpdef uint64_t read_mask(self, uint64_t address, int size):
        """
        Read size taint bytes and pack into a little-endian integer.
        Bit group [i*8 .. i*8+7] is the taint mask for memory byte address+i.
        Returns 0 if no bytes in the range are tainted.
        """
        return self.read_mask_c(address, size)

    cpdef bint is_tainted(self, uint64_t address, int size):
        """True if any byte in [address, address+size) carries any taint."""
        cdef int i
        cdef uint64_t addr, pb
        cdef uint64_t cur_pb = 0
        cdef bint have = False
        cdef unsigned char *page = NULL

        for i in range(size):
            addr = address + <uint64_t>i
            pb = self._page_base(addr)
            if not have or pb != cur_pb:
                page = pm_get(&self.taint_map, pb)
                cur_pb = pb
                have = True
            if page != NULL and page[self._offset(addr)]:
                return True
        return False

    cpdef void clear(self, uint64_t address, int size):
        """Explicitly clear taint for size bytes starting at address."""
        cdef int i
        cdef uint64_t addr, pb
        cdef uint64_t cur_pb = 0
        cdef bint have = False
        cdef unsigned char *page = NULL

        for i in range(size):
            addr = address + <uint64_t>i
            pb = self._page_base(addr)
            if not have or pb != cur_pb:
                page = pm_get(&self.taint_map, pb)
                cur_pb = pb
                have = True
            if page != NULL:
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
        cdef unsigned char *page = NULL

        for i in range(size):
            addr = address + <uint64_t>i
            pb = self._page_base(addr)
            if not have or pb != cur_pb:
                page = pm_get_or_create(&self.state_map, pb, PAGE_SIZE)
                cur_pb = pb
                have = True
            if page != NULL:
                page[self._offset(addr)] = STATE_POISONED

    cpdef void unpoison(self, uint64_t address, int size):
        """Un-poison size bytes (e.g. when a region is re-allocated)."""
        cdef int i
        cdef uint64_t addr, pb
        cdef uint64_t cur_pb = 0
        cdef bint have = False
        cdef unsigned char *page = NULL

        for i in range(size):
            addr = address + <uint64_t>i
            pb = self._page_base(addr)
            if not have or pb != cur_pb:
                page = pm_get(&self.state_map, pb)
                cur_pb = pb
                have = True
            if page != NULL:
                page[self._offset(addr)] = 0

    cpdef bint is_poisoned(self, uint64_t address, int size):
        """True if any byte in [address, address+size) is poisoned."""
        cdef int i
        cdef uint64_t addr, pb
        cdef uint64_t cur_pb = 0
        cdef bint have = False
        cdef unsigned char *page = NULL

        for i in range(size):
            addr = address + <uint64_t>i
            pb = self._page_base(addr)
            if not have or pb != cur_pb:
                page = pm_get(&self.state_map, pb)
                cur_pb = pb
                have = True
            if page != NULL and page[self._offset(addr)]:
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
    # Calls the pure-C core, not the cpdef wrapper: the wrapper is a Python-
    # callable and Cython emits a PyErr_Occurred check after it, which perf
    # attributed ~2% of the taint phase to on this path.
    return (<BitPreciseShadowMemory>shadow).read_mask_c(address, size)


cdef void _capi_write_mask(object shadow, uint64_t address, uint64_t mask, int size) noexcept:
    (<BitPreciseShadowMemory>shadow).write_mask_c(address, mask, size)


cdef _ShadowCAPI _shadow_capi_struct
_shadow_capi_struct.read_mask = _capi_read_mask
_shadow_capi_struct.write_mask = _capi_write_mask

# Module attribute imported by circuit_c via the shadow module + getattr.
_shadow_capi = PyCapsule_New(<void *>&_shadow_capi_struct,
                             b"microtaint.emulator.shadow._shadow_capi", NULL)
