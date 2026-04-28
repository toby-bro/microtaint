# ruff: noqa: ARG002
# mypy: disable-error-code="no-any-return"
from __future__ import annotations

import logging
from typing import Callable

from qiling import Qiling
from qiling.const import QL_INTERCEPT

from microtaint.emulator.shadow import BitPreciseShadowMemory

logger = logging.getLogger(__name__)

# Qiling's internal hook signature passes the Qiling instance as the first arg.
# malloc/calloc/realloc/free are hooked via ql.os.set_api() which fires on
# function *entry* by default; we also hook the *exit* of allocators to capture
# their return values (the freshly allocated pointer in RAX/X0).


class HeapTracker:
    """
    Hooks the C runtime heap API (malloc / calloc / realloc / free) via
    Qiling's symbol-level API interception so that freed memory is poisoned
    in BitPreciseShadowMemory for UAF detection.

    Architecture-independent: reads the return-value register based on the
    arch string reported by the Qiling instance.
    """

    def __init__(self, ql: Qiling, shadow: BitPreciseShadowMemory) -> None:
        self.ql = ql
        self.shadow = shadow
        self._pending_size: int = 0  # saved across malloc entry → exit
        self._pending_realloc_ptr: int = 0
        self._live_allocs: dict[int, int] = {}  # ptr → size for all live allocations

    # ------------------------------------------------------------------
    # Public install method
    # ------------------------------------------------------------------

    def install(self) -> None:
        """Register all hooks. Call once after creating the Qiling instance."""
        try:
            self.ql.os.set_api(
                'malloc',
                self._malloc_enter,
                self.ql.os.CALL_INTERCEPT if hasattr(self.ql.os, 'CALL_INTERCEPT') else None,
            )
        except Exception as e:
            logger.debug(f'[HeapTracker] Could not hook malloc: {e}')

        # Preferred approach: intercept via set_api for libc symbols.
        # Falls back gracefully if the symbol isn't found (static binary, musl, etc.)
        for name, enter, exit_ in [
            ('malloc', self._malloc_enter, self._malloc_exit),
            ('calloc', self._calloc_enter, self._malloc_exit),  # exit identical: result in ret reg
            ('realloc', self._realloc_enter, self._realloc_exit),
            ('free', self._free_enter, None),
        ]:
            self._try_hook(name, enter, exit_)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _try_hook(self, sym: str, on_enter: Callable[[Qiling], None], on_exit: Callable[[Qiling], None] | None) -> None:
        try:

            self.ql.os.set_api(sym, on_enter, QL_INTERCEPT.ENTER)
            if on_exit is not None:
                self.ql.os.set_api(sym, on_exit, QL_INTERCEPT.EXIT)
            logger.debug(f'[HeapTracker] Hooked {sym}')
        except Exception as exc:
            logger.debug(f'[HeapTracker] Could not hook {sym}: {exc}')

    def _ret_reg(self) -> int:
        """Read the architecture-appropriate return-value register."""
        try:
            arch = str(self.ql.arch.type).upper()
        except Exception:
            arch = 'AMD64'

        if 'X86' in arch or 'AMD64' in arch:
            return self.ql.arch.regs.read('RAX')
        if 'ARM64' in arch or 'AARCH64' in arch:
            return self.ql.arch.regs.read('X0')
        if 'ARM' in arch:
            return self.ql.arch.regs.read('R0')
        if 'MIPS' in arch:
            return self.ql.arch.regs.read('V0')
        return 0

    def _arg(self, n: int) -> int:
        """Read the nth function argument (0-indexed) for the calling convention."""
        try:
            arch = str(self.ql.arch.type).upper()
        except Exception:
            arch = 'AMD64'

        if 'AMD64' in arch:
            regs: tuple[str, ...] = ('RDI', 'RSI', 'RDX', 'RCX', 'R8', 'R9')
            if n < len(regs):
                return self.ql.arch.regs.read(regs[n])
            # stack args: RSP + 8*(n - len(regs) + 1)
            sp = self.ql.arch.regs.read('RSP')
            return int.from_bytes(self.ql.mem.read(sp + 8 * (n - len(regs) + 1), 8), 'little')

        if 'X86' in arch:
            sp = self.ql.arch.regs.read('ESP')
            return int.from_bytes(self.ql.mem.read(sp + 4 * (n + 1), 4), 'little')

        if 'ARM64' in arch or 'AARCH64' in arch:
            regs = tuple(f'X{i}' for i in range(8))
            if n < len(regs):
                return self.ql.arch.regs.read(regs[n])

        if 'ARM' in arch:
            regs = ('R0', 'R1', 'R2', 'R3')
            if n < len(regs):
                return self.ql.arch.regs.read(regs[n])

        return 0

    # ------------------------------------------------------------------
    # Hooks — malloc
    # ------------------------------------------------------------------

    def _malloc_enter(self, _ql: Qiling) -> None:
        self._pending_size = self._arg(0)

    def _malloc_exit(self, _ql: Qiling) -> None:
        ptr = self._ret_reg()
        size = self._pending_size
        if ptr and size:
            self._live_allocs[ptr] = size
            logger.debug(f'[HeapTracker] malloc({size}) → 0x{ptr:x}')

    # ------------------------------------------------------------------
    # Hooks — calloc  (nmemb, size)
    # ------------------------------------------------------------------

    def _calloc_enter(self, ql: Qiling) -> None:
        nmemb = self._arg(0)
        sz = self._arg(1)
        self._pending_size = nmemb * sz

    # calloc exit re-uses _malloc_exit

    # ------------------------------------------------------------------
    # Hooks — realloc  (ptr, size)
    # ------------------------------------------------------------------

    def _realloc_enter(self, ql: Qiling) -> None:
        self._pending_realloc_ptr = self._arg(0)
        self._pending_size = self._arg(1)

    def _realloc_exit(self, ql: Qiling) -> None:
        old_ptr = self._pending_realloc_ptr
        new_ptr = self._ret_reg()
        size = self._pending_size

        # If the block moved, poison the old location
        if old_ptr and old_ptr != new_ptr:
            old_size = self._live_allocs.pop(old_ptr, 0)
            if old_size:
                self.shadow.poison(old_ptr, old_size)
                logger.error(
                    f'[HeapTracker] realloc moved 0x{old_ptr:x} ({old_size}B) → 0x{new_ptr:x}; poisoned old region',
                )

        if new_ptr and size:
            self._live_allocs[new_ptr] = size
            logger.debug(f'[HeapTracker] realloc → 0x{new_ptr:x} ({size}B)')

    # ------------------------------------------------------------------
    # Hooks — free
    # ------------------------------------------------------------------

    def _free_enter(self, _ql: Qiling) -> None:
        ptr = self._arg(0)
        if not ptr:
            return  # free(NULL) is a no-op

        size = self._live_allocs.pop(ptr, 0)
        if size:
            self.shadow.poison(ptr, size)
            logger.error(f'[HeapTracker] free(0x{ptr:x}) — poisoned {size}B')
        else:
            # Unknown allocation — we saw the free but missed the malloc.
            # Poison a conservative 64-byte window to catch obvious UAF.
            self.shadow.poison(ptr, 64)
            logger.debug(f'[HeapTracker] free(0x{ptr:x}) — unknown size, poisoned 64B')
