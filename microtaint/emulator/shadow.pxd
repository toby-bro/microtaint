# cython: language_level=3
"""
Cython declarations for BitPreciseShadowMemory.

Other Cython modules can `cimport BitPreciseShadowMemory` from here to
get C-level dispatch on its cpdef methods.  Without this .pxd, calls
into shadow_mem from another Cython module go through the Python
attribute-lookup + bound-method call path, defeating the purpose of
the cpdef declaration.

Storage is a pure C open-addressing page table (page_base -> malloc'd page),
NOT a Python dict.  The dict version allocated a PyLong for the page-base key
and ran a hashed compare on every shadow access, which perf attributed to
PyDict_GetItemWithError / PyLong_FromUnsignedLong / RichCompareBool inside the
taint hot path.  The C table also lets the hot cores be `noexcept nogil`, so the
capsule used by circuit_c no longer pays a PyErr_Occurred check per access and
the fast path holds no Python state at all.
"""
from libc.stdint cimport uint64_t, uint8_t


# Open-addressing map: page_base -> page.  EMPTY is 0xFFFF...FF, which can never
# be a real page base (page bases are 4096-aligned, so their low bits are zero).
cdef struct PageMap:
    uint64_t       *keys
    unsigned char **vals
    Py_ssize_t      cap
    Py_ssize_t      n


cdef class BitPreciseShadowMemory:
    cdef PageMap taint_map
    cdef PageMap state_map
    # Fired once, on the first poison, so the wrapper can arm the UAF read
    # callback only when a use-after-free has become possible.
    cdef public object on_first_poison
    cdef bint _poison_seen

    cdef inline uint64_t _page_base(self, uint64_t address) noexcept nogil
    cdef inline int _offset(self, uint64_t address) noexcept nogil

    # Pure-C cores: no Python objects, no exceptions, callable without the GIL.
    cdef uint64_t read_mask_c(self, uint64_t address, int size) noexcept nogil
    cdef void write_mask_c(self, uint64_t address, uint64_t mask, int size) noexcept nogil
    cdef void clear_c(self, uint64_t address, int size) noexcept nogil
    cdef bint is_poisoned_c(self, uint64_t address, int size) noexcept nogil

    cpdef void write_bytes(self, uint64_t address, object taint)
    cpdef bytearray read_bytes(self, uint64_t address, int count)
    cpdef void write_mask(self, uint64_t address, uint64_t mask, int size)
    cpdef uint64_t read_mask(self, uint64_t address, int size)
    cpdef bint is_tainted(self, uint64_t address, int size)
    cpdef void clear(self, uint64_t address, int size)
    cpdef clear_all(self)
    cpdef void poison(self, uint64_t address, int size)
    cdef void _poison(self, uint64_t address, int size)
    cpdef void unpoison(self, uint64_t address, int size)
    cpdef bint is_poisoned(self, uint64_t address, int size)
