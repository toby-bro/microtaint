from __future__ import annotations

class BitPreciseShadowMemory:
    """Tracks taint and allocation state precisely down to the byte level."""
    
    STATE_CLEAN = 0x00
    STATE_POISONED = 0xAA # Used for UAF detection
    
    def __init__(self) -> None:
        self.PAGE_SIZE = 4096
        self.taint_pages: dict[int, bytearray] = {}
        self.state_pages: dict[int, bytearray] = {}

    def _get_or_create_page(self, page_dict: dict[int, bytearray], page_addr: int) -> bytearray:
        if page_addr not in page_dict:
            page_dict[page_addr] = bytearray(self.PAGE_SIZE)
        return page_dict[page_addr]

    def write_mask(self, address: int, mask: int, size: int) -> None:
        """Writes a bitmask to memory. Handles unaligned and cross-page writes."""
        mask_bytes = mask.to_bytes(size, 'little')
        for i in range(size):
            addr = address + i
            page_addr = addr & ~(self.PAGE_SIZE - 1)
            offset = addr & (self.PAGE_SIZE - 1)
            
            page = self._get_or_create_page(self.taint_pages, page_addr)
            page[offset] = mask_bytes[i]

    def read_mask(self, address: int, size: int) -> int:
        """Reads the exact taint bitmask for a given memory range."""
        result = 0
        for i in range(size):
            addr = address + i
            page_addr = addr & ~(self.PAGE_SIZE - 1)
            if page_addr in self.taint_pages:
                val = self.taint_pages[page_addr][addr & (self.PAGE_SIZE - 1)]
                result |= (val << (i * 8))
        return result

    def poison(self, address: int, size: int) -> None:
        """Marks a memory region as freed/poisoned for UAF detection."""
        for i in range(size):
            addr = address + i
            page_addr = addr & ~(self.PAGE_SIZE - 1)
            offset = addr & (self.PAGE_SIZE - 1)
            page = self._get_or_create_page(self.state_pages, page_addr)
            page[offset] = self.STATE_POISONED

    def is_poisoned(self, address: int, size: int) -> bool:
        """Checks if any byte in the range is poisoned."""
        for i in range(size):
            addr = address + i
            page_addr = addr & ~(self.PAGE_SIZE - 1)
            if page_addr in self.state_pages:
                if self.state_pages[page_addr][addr & (self.PAGE_SIZE - 1)] == self.STATE_POISONED:
                    return True
        return False
