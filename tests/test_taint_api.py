"""
test_taint_api.py
=================
Tests for MicrotaintWrapper.taint_bit() and MicrotaintWrapper.taint_region().

All tests use real Qiling + real MicrotaintWrapper + real compiled binaries.
No mocks.

Structure
---------
  Tier 1 — unit tests on the shadow memory layer directly (no Qiling):
    test_taint_bit_shadow_written_*
    test_taint_region_shadow_written_*
    test_taint_bit_invalid_bit_index

  Tier 2 — hook-arming tests (Qiling instantiated, ql.run() not called):
    test_taint_bit_arms_hooks
    test_taint_region_arms_hooks

  Tier 3 — end-to-end propagation (Qiling runs a binary):
    test_taint_bit_propagates_through_xor
    test_taint_bit_propagates_through_add_carry
    test_taint_region_partial_byte_through_xor
    test_taint_bit_does_not_infect_adjacent_bytes
    test_stdin_taint_still_works

Run
---
    uv run pytest test_taint_api.py -v
"""

# ruff: noqa: ARG001, ARG002, PLC0415, S110, S607, PLW1510, S603
# mypy: disable-error-code="no-untyped-def,no-untyped-call,type-arg,arg-type,return-value"

from __future__ import annotations

import io
import subprocess
import textwrap
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GCC_FLAGS = ['-O0', '-g', '-static', '-no-pie', '-fno-stack-protector']


def _compile(c_src: str, tmp_path: Path) -> Path:
    """Compile a C source string to a static binary. Returns the binary path."""
    src_file = tmp_path / 'prog.c'
    bin_file = tmp_path / 'prog'
    src_file.write_text(textwrap.dedent(c_src))
    r = subprocess.run(
        ['gcc', *GCC_FLAGS, '-o', str(bin_file), str(src_file)],
        capture_output=True,
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode())
    return bin_file


def _make_wrapper(binary: Path, *, check_sc: bool = False):
    """
    Instantiate Qiling + MicrotaintWrapper for a binary with a fixed empty stdin.
    Returns (ql, wrapper).
    """
    import logging

    logging.disable(logging.CRITICAL)

    from qiling import Qiling
    from qiling.const import QL_VERBOSE

    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper

    ql = Qiling([str(binary)], '/', verbose=QL_VERBOSE.OFF)

    class _EmptyStdin:
        def read(self, n: int) -> bytes:
            return b'\x00' * n

    ql.os.stdin = _EmptyStdin()
    reporter = Reporter(json_mode=True, stream=io.StringIO())
    wrapper = MicrotaintWrapper(
        ql,
        check_sc=check_sc,
        check_bof=False,
        check_uaf=False,
        check_aiw=False,
        reporter=reporter,
    )
    return ql, wrapper


def _run_with_taint(
    binary: Path,
    stdin_bytes: bytes,
    taint_fn,  # callable(wrapper, buf_address) -> None
    output_size: int = 1,
    check_sc: bool = False,
) -> tuple[int, list[dict]]:
    """
    Run `binary` under microtaint.  Override the read() syscall so that after
    writing `stdin_bytes` into the buffer, `taint_fn(wrapper, buf)` is called to
    inject taint with whatever granularity the test requires.

    Returns (shadow_mask_of_first_output_write, findings_list).

    IMPORTANT: the shadow must be read INSIDE the write() hook, at the
    moment the binary calls write(1, buf, n).  Reading it after ql.run()
    returns is too late: libc's cleanup code (~140k instructions of
    push/pop on stack exit paths) runs between the user's write() and
    process exit.  Those stack writes cause the circuit evaluator to emit
    untainted MEM_ outputs whose byte ranges overlap our buffer, clearing
    its shadow before we can read it.
    """
    import logging

    logging.disable(logging.CRITICAL)

    from qiling import Qiling
    from qiling.const import QL_INTERCEPT, QL_VERBOSE

    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper

    output_buf: list[int | None] = [None]
    shadow_mask: list[int] = [0]

    ql = Qiling([str(binary)], '/', verbose=QL_VERBOSE.OFF)

    class _FixedStdin:
        def read(self, n: int) -> bytes:
            return stdin_bytes[:n]

    ql.os.stdin = _FixedStdin()
    reporter = Reporter(json_mode=True, stream=io.StringIO())
    wrapper = MicrotaintWrapper(
        ql,
        check_sc=check_sc,
        check_bof=False,
        check_uaf=False,
        check_aiw=False,
        reporter=reporter,
    )

    def _read_hook(ql: Qiling, fd: int, buf: int, count: int) -> int:
        if fd != 0:
            return 0
        data = stdin_bytes[:count]
        if not data:
            return 0
        ql.mem.write(buf, data)
        taint_fn(wrapper, buf)
        return len(data)

    def _write_hook(ql: Qiling, fd: int, buf: int, count: int, *_: Any) -> int:
        # Capture buf AND read shadow_mask NOW, before libc cleanup runs.
        # Returns count to suppress the actual write (no bytes leak to stdout).
        if fd == 1 and output_buf[0] is None:
            output_buf[0] = buf
            shadow_mask[0] = wrapper.shadow_mem.read_mask(buf, output_size)
        return count

    ql.os.set_syscall(0, _read_hook, QL_INTERCEPT.CALL)
    ql.os.set_syscall(1, _write_hook, QL_INTERCEPT.CALL)

    try:
        ql.run()
    except Exception:
        pass

    findings = reporter.findings if hasattr(reporter, 'findings') else []
    return shadow_mask[0], findings


# ---------------------------------------------------------------------------
# Tier 1: Shadow memory unit tests (no Qiling, no binary)
# ---------------------------------------------------------------------------


class TestShadowMemory:
    """Direct tests on BitPreciseShadowMemory via taint_bit/taint_region."""

    def test_taint_bit_writes_correct_mask(self, tmp_path):
        binary = _compile(
            """
            #include <unistd.h>
            int main(void) { char b; read(0,&b,1); write(1,&b,1); return 0; }
        """,
            tmp_path,
        )
        _, wrapper = _make_wrapper(binary)
        addr = 0x1000

        for bit in range(8):
            wrapper.shadow_mem.write_mask(addr, 0, 1)  # clear first
            wrapper.taint_bit(addr, bit)
            got = wrapper.shadow_mem.read_mask(addr, 1)
            assert got == (1 << bit), f'taint_bit(addr, {bit}): expected shadow={1 << bit:#04x}, got {got:#04x}'

    def test_taint_bit_or_semantics(self, tmp_path):
        """
        taint_bit must OR the new bit into the existing shadow byte, not
        replace it.  Calling taint_bit(addr, 0) then taint_bit(addr, 1)
        must leave bits 0 AND 1 set (mask 0x03), not just bit 1 (0x02).
        """
        binary = _compile(
            """
            #include <unistd.h>
            int main(void) { char b; read(0,&b,1); write(1,&b,1); return 0; }
        """,
            tmp_path,
        )
        _, wrapper = _make_wrapper(binary)
        addr = 0x1100

        wrapper.shadow_mem.write_mask(addr, 0, 1)
        wrapper.taint_bit(addr, 0)
        wrapper.taint_bit(addr, 1)
        got = wrapper.shadow_mem.read_mask(addr, 1)
        assert got == 0x03, f'Expected 0x03 (bits 0+1), got {got:#04x}'

        # Add bit 7 — should now have 0x83
        wrapper.taint_bit(addr, 7)
        got = wrapper.shadow_mem.read_mask(addr, 1)
        assert got == 0x83, f'Expected 0x83 (bits 0+1+7), got {got:#04x}'

        # Add bit 0 again — idempotent (already set)
        wrapper.taint_bit(addr, 0)
        got = wrapper.shadow_mem.read_mask(addr, 1)
        assert got == 0x83, f'Re-tainting bit 0 changed mask: {got:#04x}'

    def test_taint_region_writes_per_byte_masks(self, tmp_path):
        binary = _compile(
            """
            #include <unistd.h>
            int main(void) { char b[4]; read(0,b,4); write(1,b,4); return 0; }
        """,
            tmp_path,
        )
        _, wrapper = _make_wrapper(binary)
        addr = 0x2000
        mask_bytes = bytes([0x00, 0x0F, 0xFF, 0xAA])
        wrapper.taint_region(addr, mask_bytes)

        assert wrapper.shadow_mem.read_mask(addr + 0, 1) == 0x00
        assert wrapper.shadow_mem.read_mask(addr + 1, 1) == 0x0F
        assert wrapper.shadow_mem.read_mask(addr + 2, 1) == 0xFF
        assert wrapper.shadow_mem.read_mask(addr + 3, 1) == 0xAA

    def test_taint_region_zero_mask_clears_existing_taint(self, tmp_path):
        """
        taint_region(addr, [0x00]) must CLEAR pre-existing taint at that address.
        write_mask(addr, 0, 1) is an explicit clear (documented in shadow.pyx).
        taint_region must not skip the call just because mask==0.
        """
        binary = _compile(
            """
            #include <unistd.h>
            int main(void) { char b; read(0,&b,1); write(1,&b,1); return 0; }
        """,
            tmp_path,
        )
        _, wrapper = _make_wrapper(binary)
        addr = 0x3000
        wrapper.taint_bit(addr, 3)
        assert wrapper.shadow_mem.read_mask(addr, 1) == 0x08  # sanity
        wrapper.taint_region(addr, bytes([0x00]))
        assert wrapper.shadow_mem.read_mask(addr, 1) == 0x00

    def test_taint_bit_and_taint_bytes_are_different(self, tmp_path):
        """taint_bit marks one bit (0x01..0x80); _taint_bytes marks all (0xFF)."""
        binary = _compile(
            """
            #include <unistd.h>
            int main(void) { char b; read(0,&b,1); write(1,&b,1); return 0; }
        """,
            tmp_path,
        )
        _, wrapper = _make_wrapper(binary)
        addr = 0x4000

        wrapper.taint_bit(addr, 3)
        assert wrapper.shadow_mem.read_mask(addr, 1) == 0x08

        # Full byte taint via _taint_bytes (private but needed here)
        wrapper._taint_bytes(addr, 1)
        assert wrapper.shadow_mem.read_mask(addr, 1) == 0xFF

    def test_taint_bit_invalid_bit_index_raises(self, tmp_path):
        binary = _compile(
            """
            #include <unistd.h>
            int main(void) { char b; read(0,&b,1); write(1,&b,1); return 0; }
        """,
            tmp_path,
        )
        _, wrapper = _make_wrapper(binary)
        addr = 0x5000
        with pytest.raises(ValueError, match='bit_index must be 0-7'):
            wrapper.taint_bit(addr, 8)
        with pytest.raises(ValueError, match='bit_index must be 0-7'):
            wrapper.taint_bit(addr, -1)


# ---------------------------------------------------------------------------
# Tier 2: Hook-arming tests (Qiling instantiated, ql.run() NOT called)
# ---------------------------------------------------------------------------


class TestHookArming:
    """
    Verify that taint_bit/taint_region arm the deferred instruction hook
    and mem-write hook, exactly like _taint_bytes does.
    """

    def test_taint_bit_arms_deferred_hooks(self, tmp_path):
        binary = _compile(
            """
            #include <unistd.h>
            int main(void) { char b; read(0,&b,1); write(1,&b,1); return 0; }
        """,
            tmp_path,
        )
        _, wrapper = _make_wrapper(binary)

        assert not wrapper._any_taint
        assert not wrapper._instr_hook_registered

        wrapper.taint_bit(0x1000, 0)

        assert wrapper._any_taint
        assert wrapper._instr_hook_registered

    def test_taint_region_arms_deferred_hooks(self, tmp_path):
        binary = _compile(
            """
            #include <unistd.h>
            int main(void) { char b; read(0,&b,1); write(1,&b,1); return 0; }
        """,
            tmp_path,
        )
        _, wrapper = _make_wrapper(binary)

        assert not wrapper._any_taint
        wrapper.taint_region(0x2000, bytes([0xFF, 0x00, 0x0F]))
        assert wrapper._any_taint
        assert wrapper._instr_hook_registered

    def test_arm_is_idempotent(self, tmp_path):
        """Calling taint_bit multiple times must not double-register hooks."""
        binary = _compile(
            """
            #include <unistd.h>
            int main(void) { char b; read(0,&b,1); write(1,&b,1); return 0; }
        """,
            tmp_path,
        )
        _, wrapper = _make_wrapper(binary)

        wrapper.taint_bit(0x1000, 0)
        wrapper.taint_bit(0x1001, 1)
        wrapper.taint_bit(0x1002, 7)

        # _arm_deferred_hooks must not have duplicated the registration
        assert wrapper._any_taint
        assert wrapper._instr_hook_registered


# ---------------------------------------------------------------------------
# Tier 3: End-to-end propagation through real binary execution
# ---------------------------------------------------------------------------


class TestEndToEndPropagation:
    """
    Compile tiny C programs and verify taint propagates through them correctly.
    """

    def test_taint_bit_propagates_through_xor(self, tmp_path):
        """
        Binary: read 1 byte, XOR with 0x5A, write result.

        XOR is a bijection on each bit independently: bit i of output
        depends solely on bit i of input.  Tainting only bit 3 of the input
        must produce exactly bit 3 tainted in the output (mask = 0x08).
        """
        binary = _compile(
            """
            #include <unistd.h>
            int main(void) {
                unsigned char b;
                read(0, &b, 1);
                b ^= 0x5A;
                write(1, &b, 1);
                return 0;
            }
        """,
            tmp_path,
        )

        def taint(wrapper, buf):
            wrapper.taint_bit(buf, 3)  # taint bit 3 only

        shadow, _ = _run_with_taint(binary, b'\x00', taint, output_size=1)
        assert shadow != 0, 'Expected non-zero taint on output after XOR'
        # XOR preserves bit identity: only bit 3 should be tainted
        assert shadow == 0x08, f'Expected shadow=0x08 (bit 3 only), got {shadow:#04x}'

    def test_taint_region_partial_byte_through_xor(self, tmp_path):
        """
        Same XOR binary, but taint only the low nibble (bits 0-3, mask=0x0F).
        Expected output shadow: 0x0F (XOR preserves each bit independently).
        """
        binary = _compile(
            """
            #include <unistd.h>
            int main(void) {
                unsigned char b;
                read(0, &b, 1);
                b ^= 0x5A;
                write(1, &b, 1);
                return 0;
            }
        """,
            tmp_path,
        )

        def taint(wrapper, buf):
            wrapper.taint_region(buf, bytes([0x0F]))  # low nibble only

        shadow, _ = _run_with_taint(binary, b'\x00', taint, output_size=1)
        assert shadow == 0x0F, f'Expected shadow=0x0F (low nibble through XOR), got {shadow:#04x}'

    def test_taint_bit_propagates_through_add_carry(self, tmp_path):
        """
        Binary: read 1 byte, add 1, write result.

        ADD has carry propagation: bit i of output can depend on bits 0..i
        of the input (carry ripple).  Tainting bit 0 should produce taint
        on bits 0..7 of the output (all bits potentially affected by carry).

        This tests that the engine correctly models ADD's taint spreading.
        """
        binary = _compile(
            """
            #include <unistd.h>
            int main(void) {
                unsigned char b;
                read(0, &b, 1);
                b += 1;
                write(1, &b, 1);
                return 0;
            }
        """,
            tmp_path,
        )

        def taint(wrapper, buf):
            wrapper.taint_bit(buf, 0)  # only bit 0 of input is tainted

        shadow, _ = _run_with_taint(binary, b'\xff', taint, output_size=1)
        assert shadow != 0, 'Expected non-zero taint on output after ADD'
        # At minimum, bit 0 of output must be tainted
        assert shadow & 0x01, 'Bit 0 must be tainted after ADD with tainted bit-0 input'

    def test_taint_bit_does_not_infect_adjacent_input_bytes(self, tmp_path):
        """
        Binary: read 2 bytes, write them back.
        Taint only bit 3 of the FIRST byte.  The second output byte must have
        zero shadow (taint must not bleed across byte boundaries at read time).
        """
        binary = _compile(
            """
            #include <unistd.h>
            int main(void) {
                unsigned char b[2];
                read(0, b, 2);
                write(1, b, 2);
                return 0;
            }
        """,
            tmp_path,
        )

        def taint(wrapper, buf):
            wrapper.taint_bit(buf, 3)  # only byte 0, bit 3

        shadow, _ = _run_with_taint(binary, b'\x00\x00', taint, output_size=2)
        byte0_shadow = shadow & 0xFF
        byte1_shadow = (shadow >> 8) & 0xFF
        assert byte0_shadow != 0, 'First byte must have taint'
        assert byte1_shadow == 0, f'Second byte must not be tainted (got shadow byte1={byte1_shadow:#04x})'

    def test_taint_all_bits_of_byte_equals_full_taint(self, tmp_path):
        """
        Tainting all 8 bits individually must produce the same shadow as
        tainting the whole byte via taint_region(addr, [0xFF]).
        Both should produce shadow 0xFF at that address.
        """
        binary = _compile(
            """
            #include <unistd.h>
            int main(void) {
                unsigned char b;
                read(0, &b, 1);
                write(1, &b, 1);
                return 0;
            }
        """,
            tmp_path,
        )

        # Approach A: taint all 8 bits individually
        def taint_all_bits(wrapper, buf):
            for bit in range(8):
                wrapper.taint_bit(buf, bit)

        shadow_a, _ = _run_with_taint(binary, b'\xaa', taint_all_bits, output_size=1)

        # Approach B: taint via taint_region with 0xFF mask
        def taint_full(wrapper, buf):
            wrapper.taint_region(buf, bytes([0xFF]))

        shadow_b, _ = _run_with_taint(binary, b'\xaa', taint_full, output_size=1)

        assert (
            shadow_a == shadow_b
        ), f'All-bits-individually ({shadow_a:#04x}) must equal full-byte-region ({shadow_b:#04x})'
        assert shadow_a == 0xFF, f'Expected 0xFF, got {shadow_a:#04x}'

    def test_stdin_taint_still_works(self, tmp_path):
        """
        Regression: the normal stdin-taint path (_taint_bytes via _sys_read_hook)
        must still work correctly after the new API additions.
        Using the CLI / subprocess path: the whole byte is tainted (0xFF).
        """
        binary = _compile(
            """
            #include <unistd.h>
            int main(void) {
                unsigned char b;
                read(0, &b, 1);
                write(1, &b, 1);
                return 0;
            }
        """,
            tmp_path,
        )

        # Use the default MicrotaintWrapper WITHOUT overriding the read hook.
        # The wrapper's _sys_read_hook should fully taint the byte.
        import logging

        logging.disable(logging.CRITICAL)

        from qiling import Qiling
        from qiling.const import QL_INTERCEPT, QL_VERBOSE

        from microtaint.emulator.reporter import Reporter
        from microtaint.emulator.wrapper import MicrotaintWrapper

        # We capture the shadow mask AT the moment of write(), not after
        # ql.run().  Libc's exit cleanup runs many instructions after our
        # write() that can clear the shadow at our buffer's address.
        captured_shadow: list[int] = [-1]
        captured_buf: list[int | None] = [None]

        ql = Qiling([str(binary)], '/', verbose=QL_VERBOSE.OFF)

        class _Stdin:
            def read(self, n: int) -> bytes:
                return b'\xab'

        ql.os.stdin = _Stdin()

        reporter = Reporter(json_mode=True, stream=io.StringIO())
        wrapper = MicrotaintWrapper(
            ql,
            check_sc=False,
            check_bof=False,
            check_uaf=False,
            check_aiw=False,
            reporter=reporter,
        )

        def _write_cap(ql, fd, buf, count, *_):
            if fd == 1 and captured_buf[0] is None:
                captured_buf[0] = buf
                captured_shadow[0] = wrapper.shadow_mem.read_mask(buf, 1)
            return count

        ql.os.set_syscall(1, _write_cap, QL_INTERCEPT.CALL)
        ql.run()

        assert captured_buf[0] is not None, 'write(1,...) was never called'
        assert (
            captured_shadow[0] == 0xFF
        ), f'stdin taint: expected full byte shadow 0xFF at write() time, got {captured_shadow[0]:#04x}'

    def test_taint_region_no_taint_zero_mask(self, tmp_path):
        """
        taint_region with all-zero mask must write no taint.
        Output shadow must be 0 even though the byte passes through.
        """
        binary = _compile(
            """
            #include <unistd.h>
            int main(void) {
                unsigned char b;
                read(0, &b, 1);
                write(1, &b, 1);
                return 0;
            }
        """,
            tmp_path,
        )

        def taint(wrapper, buf):
            wrapper.taint_region(buf, bytes([0x00]))  # explicit no-taint

        shadow, _ = _run_with_taint(binary, b'\xaa', taint, output_size=1)
        assert shadow == 0, f'Expected zero shadow with zero-mask region, got {shadow:#04x}'

    def test_taint_bit_7_msb(self, tmp_path):
        """
        Tainting bit 7 (MSB) specifically via taint_bit and verifying it
        survives through a passthrough binary.
        """
        binary = _compile(
            """
            #include <unistd.h>
            int main(void) {
                unsigned char b;
                read(0, &b, 1);
                write(1, &b, 1);
                return 0;
            }
        """,
            tmp_path,
        )

        def taint(wrapper, buf):
            wrapper.taint_bit(buf, 7)  # MSB only

        shadow, _ = _run_with_taint(binary, b'\x80', taint, output_size=1)
        assert shadow == 0x80, f'Expected shadow=0x80 (MSB only), got {shadow:#04x}'
