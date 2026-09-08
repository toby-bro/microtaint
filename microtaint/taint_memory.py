"""Guest memory for the taint API: the bytes, and the taint on them.

The taint side is the engine's own `BitPreciseShadowMemory` -- the same C page
table the emulator's fast path uses -- rather than a second implementation, so
the API and the emulator answer a memory question identically and this inherits
the fast structure instead of racing it.  The value side is what the API adds:
the emulator reads concrete bytes out of Unicorn, and an API caller working on a
sequence of instructions has no Unicorn, so it needs somewhere for a `push` to
put the word a `pop` will read back.

Sparse by page, because a sequence that touches a stack pointer near the top of
the address space should not allocate everything below it.
"""

from __future__ import annotations

from microtaint.emulator.shadow import BitPreciseShadowMemory

_PAGE_BITS = 12
_PAGE_SIZE = 1 << _PAGE_BITS
_PAGE_MASK = _PAGE_SIZE - 1


class TaintMemory:
    """Concrete bytes plus their taint, addressed like the guest addresses it.

    `read`/`write` move values; `read_mask`/`write_mask` move taint and are the
    two methods the engine itself calls, so an instance is usable directly as
    the `shadow_memory` of an `EvalContext` or as the compiled path's shadow.
    """

    __slots__ = ('_pages', '_reads', '_writes', 'little_endian', 'shadow')

    def __init__(self, *, little_endian: bool = True) -> None:
        self.shadow = BitPreciseShadowMemory()
        self._pages: dict[int, bytearray] = {}
        self._writes: list[tuple[int, int]] = []
        self._reads: list[tuple[int, int]] = []
        self.little_endian = little_endian

    # -- values ---------------------------------------------------------
    def read(self, address: int, size: int) -> int:
        """`size` bytes at `address` as an integer.  Never-written bytes read
        as zero, which is what an untouched page holds anyway."""
        return int.from_bytes(self.read_bytes(address, size),
                              'little' if self.little_endian else 'big')

    def write(self, address: int, value: int, size: int) -> None:
        self.write_bytes(address, (value & ((1 << (size * 8)) - 1)).to_bytes(
            size, 'little' if self.little_endian else 'big'))

    def read_bytes(self, address: int, count: int) -> bytes:
        out = bytearray(count)
        for i in range(count):
            page = self._pages.get((address + i) >> _PAGE_BITS)
            if page is not None:
                out[i] = page[(address + i) & _PAGE_MASK]
        return bytes(out)

    def write_bytes(self, address: int, data: bytes) -> None:
        for i, b in enumerate(data):
            key = (address + i) >> _PAGE_BITS
            page = self._pages.get(key)
            if page is None:
                page = self._pages[key] = bytearray(_PAGE_SIZE)
            page[(address + i) & _PAGE_MASK] = b

    # -- taint ----------------------------------------------------------
    # Delegated rather than reimplemented: these are the methods the engine
    # calls, and they have to mean exactly what they mean to the emulator.
    def read_mask(self, address: int, size: int) -> int:
        self._reads.append((address, size))
        return self.shadow.read_mask(address, size)

    def take_reads(self) -> list[tuple[int, int]]:
        """(address, size) of the loads since the last call, and forget them.

        A sequence needs these to hand the cell the words an instruction loads:
        the taint pass has already resolved the addresses, so nothing has to
        compute them twice."""
        reads, self._reads = self._reads, []
        return reads

    def write_mask(self, address: int, mask: int, size: int) -> None:
        self.shadow.write_mask(address, mask, size)
        self._writes.append((address, size))

    def take_writes(self) -> list[tuple[int, int]]:
        """(address, size) of the stores since the last call, and forget them.

        A sequence needs these to ask the cell for the concrete word a store
        put there: the taint pass has already resolved the address, so this
        saves computing it a second time."""
        writes, self._writes = self._writes, []
        return writes

    def is_tainted(self, address: int, size: int) -> bool:
        return bool(self.shadow.is_tainted(address, size))

    def taint(self, address: int, size: int, mask: int | None = None) -> None:
        """Mark `size` bytes tainted -- the API's equivalent of a source."""
        self.write_mask(address, (1 << (size * 8)) - 1 if mask is None else mask, size)

    def clear(self, address: int, size: int) -> None:
        self.shadow.clear(address, size)

    # -- for a caller that wants to see what happened --------------------
    def tainted_spans(self, address: int, size: int) -> dict[int, int]:
        """{address: mask} for the bytes in the range that carry any taint."""
        return {address + i: m
                for i in range(size)
                if (m := self.shadow.read_mask(address + i, 1))}

    def __repr__(self) -> str:
        return f'<TaintMemory {len(self._pages)} page(s)>'
