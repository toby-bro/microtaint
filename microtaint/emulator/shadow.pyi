# shadow.pyi

class BitPreciseShadowMemory:
    """
    Fast Cython implementation of bit-precise shadow memory.
    """

    # Internal state managed by Cython, but visible to Python as dicts
    taint_pages: dict[int, bytearray]
    state_pages: dict[int, bytearray]

    def __init__(self) -> None: ...

    # Taint API
    def write_bytes(self, address: int, taint: bytes | bytearray) -> None:
        """
        Write taint into shadow memory, one shadow byte per memory byte.
        taint[i] is the 8-bit mask for memory byte at address+i. [cite: 10, 11]
        """

    def read_bytes(self, address: int, count: int) -> bytearray:
        """
        Read count taint bytes. Returns bytearray(count); uninitialized = 0x00. [cite: 13, 14]
        """

    def write_mask(self, address: int, mask: int, size: int) -> None:
        """
        Convert a packed little-endian integer mask to per-byte taint and write. [cite: 15, 16]
        """

    def read_mask(self, address: int, size: int) -> int:
        """
        Read size taint bytes and pack into a little-endian integer. [cite: 20, 22]
        """

    def is_tainted(self, address: int, size: int) -> bool:
        """True if any byte in [address, address+size) carries any taint. [cite: 24, 25]"""

    def clear(self, address: int, size: int) -> None:
        """Explicitly clear taint for size bytes starting at address. [cite: 26]"""
    # Poison (UAF) API
    def poison(self, address: int, size: int) -> None:
        """Mark size bytes as freed/poisoned (for UAF detection). [cite: 28]"""

    def unpoison(self, address: int, size: int) -> None:
        """Un-poison size bytes (e.g. when a region is re-allocated). [cite: 29]"""

    def is_poisoned(self, address: int, size: int) -> bool:
        """True if any byte in [address, address+size) is poisoned. [cite: 31, 32]"""
    # Properties
    @property
    def PAGE_SIZE(self) -> int: ...
    @property
    def STATE_POISONED(self) -> int: ...
