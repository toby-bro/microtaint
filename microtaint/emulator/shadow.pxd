# cython: language_level=3
"""
Cython declarations for BitPreciseShadowMemory.

Other Cython modules can `cimport BitPreciseShadowMemory` from here to
get C-level dispatch on its cpdef methods.  Without this .pxd, calls
into shadow_mem from another Cython module go through the Python
attribute-lookup + bound-method call path, defeating the purpose of
the cpdef declaration.
"""
from libc.stdint cimport uint64_t


cdef class BitPreciseShadowMemory:
    cdef dict taint_pages
    cdef dict state_pages

    cdef inline bytearray _get_taint_page(self, uint64_t page_base)
    cdef inline bytearray _get_state_page(self, uint64_t page_base)
    cdef inline uint64_t _page_base(self, uint64_t address)
    cdef inline int _offset(self, uint64_t address)

    cpdef void write_bytes(self, uint64_t address, object taint)
    cpdef bytearray read_bytes(self, uint64_t address, int count)
    cpdef void write_mask(self, uint64_t address, uint64_t mask, int size)
    cpdef uint64_t read_mask(self, uint64_t address, int size)
    cpdef bint is_tainted(self, uint64_t address, int size)
    cpdef void clear(self, uint64_t address, int size)
    cpdef void poison(self, uint64_t address, int size)
    cpdef void unpoison(self, uint64_t address, int size)
    cpdef bint is_poisoned(self, uint64_t address, int size)
