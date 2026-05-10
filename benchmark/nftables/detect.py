#!/usr/bin/env python3
"""
nft_byteorder_eval alignment-bug detection via two-color taint analysis.

Premise
-------
The nftables nft_byteorder expression has a stride/element-size mismatch
when priv->size == 2. CVE-2023-35001, found and exploited at Pwn2Own
Vancouver 2023 by Tanguy Dubroca (Synacktiv); writeup:
https://www.synacktiv.com/publications/old-bug-shallow-bug-exploiting-ubuntu-at-pwn2own-vancouver-2023
The configuration validator checks priv->len bytes of destination
space, but the evaluator iterates priv->len/2 times with a 4-byte
stride (because the destination pointer's static type is a union whose
sizeof is 4). The result: writes overshoot the validated region by
~len/2 bytes.

The detection idea you specified:

  * One shadow memory per "color".
  * Color A = the wrapper's existing shadow; carries dynamic taint that
    propagates with every instruction. Bytes coming in from stdin are
    fully color-A tainted at the byte level (wrapper.taint_region uses
    write_mask 0xFF for a fully-tainted byte; bit-precision is
    preserved because the engine still tracks per-bit independence in
    the rest of the run).
  * Color B = a SECOND, parallel BitPreciseShadowMemory we own. We
    statically mark the trailing canary region in color B as "must not
    be written to". Color B never propagates; it is a tripwire.

A guest write whose destination range overlaps a color-B byte is the
bug. If, in addition, the destination ALSO acquires color-A taint
through the same instruction, that is the precise "the two colors
mixed" event you described — attacker-controlled data physically
reached sentinel territory.

Both shadows are full bit-precise BitPreciseShadowMemory instances.
The bit-precision contract from shadow.pyx — "taint byte: bit i set
means bit i of the corresponding memory byte is tainted" — is
preserved; we use 0xFF in this PoC because the bug is about whole-byte
reach, but you could just as well use 0x01 to track only bit 0, etc.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# ---- microtaint / qiling imports ---------------------------------------------

from qiling import Qiling
from qiling.const import QL_INTERCEPT, QL_VERBOSE
from unicorn import UC_HOOK_MEM_WRITE

from microtaint.emulator.shadow import BitPreciseShadowMemory
from microtaint.emulator.reporter import Reporter
from microtaint.emulator.wrapper import MicrotaintWrapper

# ---- harness binary and layout constants -------------------------------------

HARNESS = Path(__file__).parent / 'harness'

# struct layout (must match harness.c):
#   canary_before : 256 B
#   regs          :  80 B   (NFT_REG32_NUM=20 * 4)
#   canary_after  : 256 B
CANARY_BEFORE_SIZE = 256
REGS_SIZE = 80
CANARY_AFTER_SIZE = 256

# Triggering parameters: priv->len=80, priv->size=2, op=NTOH(0).
# 80 is the maximum value the kernel's configuration validator accepts
# for dreg=0 (since dreg + len <= sizeof(nft_regs) = 80). The bug
# overshoots regardless: stride 4, len/2 = 40 iterations, last write
# at offset 4*39 = 156 — 76 bytes past the end of the register file.
CTL_BYTES = bytes([80, 2, 0])
REGS_PAYLOAD = bytes([0x42] * REGS_SIZE)


def find_layout_address(binary: Path) -> int:
    """Return the link-time address of the global `L` struct.

    The harness is built static and -no-pie, so this address is also
    the runtime address Qiling loads it at. We pull it from `nm`.
    """
    out = subprocess.check_output(['nm', str(binary)], text=True)
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == 'L':
            return int(parts[0], 16)
    raise RuntimeError('symbol `L` not found in harness binary')


def main() -> int:
    if not HARNESS.exists():
        print(f'[!] Harness binary missing: {HARNESS}', file=sys.stderr)
        print('    Build it first: gcc -O0 -static -no-pie -fno-stack-protector '
              '-fno-common -o harness harness.c', file=sys.stderr)
        return 2

    layout_addr = find_layout_address(HARNESS)
    canary_before_addr = layout_addr
    regs_addr          = canary_before_addr + CANARY_BEFORE_SIZE
    canary_after_addr  = regs_addr + REGS_SIZE

    print(f'[*] Layout (from ELF symbol table):')
    print(f'      canary_before : {canary_before_addr:#x} .. '
          f'{canary_before_addr + CANARY_BEFORE_SIZE:#x}')
    print(f'      regs_storage  : {regs_addr:#x} .. '
          f'{regs_addr + REGS_SIZE:#x}')
    print(f'      canary_after  : {canary_after_addr:#x} .. '
          f'{canary_after_addr + CANARY_AFTER_SIZE:#x}')

    # -------------------------------------------------------------------------
    # Boot Qiling and feed the binary its payload via stdin.
    # -------------------------------------------------------------------------
    payload = CTL_BYTES + REGS_PAYLOAD

    class StdinStream:
        """Drains a fixed byte buffer one read() call at a time."""
        def __init__(self, data: bytes):
            self._buf = bytearray(data)
        def read(self, count: int) -> bytes:
            chunk = bytes(self._buf[:count])
            del self._buf[:count]
            return chunk

    ql = Qiling([str(HARNESS)], '/', verbose=QL_VERBOSE.OFF)
    ql.os.stdin = StdinStream(payload)

    # -------------------------------------------------------------------------
    # microtaint wrapper provides shadow_mem (= color A).
    #
    # We keep all detection modes off — we are NOT using microtaint's
    # built-in BOF/UAF/SC/AIW classifiers. We are using its taint
    # propagation engine and its shadow as one of two parallel taint
    # maps. Disabling the classifiers also stops the wrapper from
    # calling ql.emu_stop() on first finding, which would interrupt
    # the very write we want to inspect.
    # -------------------------------------------------------------------------
    reporter = Reporter()
    wrapper = MicrotaintWrapper(
        ql,
        check_bof=False,
        check_uaf=False,
        check_sc=False,
        check_aiw=False,
        reporter=reporter,
    )

    color_a: BitPreciseShadowMemory = wrapper.shadow_mem  # alias for clarity

    # -------------------------------------------------------------------------
    # Color B — our second, independent shadow. Bit-precise, just like A.
    # Pre-mark the canary regions byte-by-byte with mask 0xFF.
    # We mark BOTH canaries so any direction of overshoot is caught;
    # for this specific bug only canary_after will matter.
    # -------------------------------------------------------------------------
    color_b = BitPreciseShadowMemory()
    for off in range(CANARY_BEFORE_SIZE):
        color_b.write_mask(canary_before_addr + off, 0xFF, 1)
    for off in range(CANARY_AFTER_SIZE):
        color_b.write_mask(canary_after_addr + off, 0xFF, 1)

    # Sanity-check the marking before the run.
    assert color_b.is_tainted(canary_after_addr, CANARY_AFTER_SIZE)
    assert not color_b.is_tainted(regs_addr, REGS_SIZE), (
        'register file must not carry color B — only canary territory does'
    )

    # -------------------------------------------------------------------------
    # Detection hook: on every guest memory write, ask
    #   1. does this write land in color-B sentinel territory?
    #   2. did color-A taint also reach this destination?
    # The second answer is read from color A AFTER the write —
    # microtaint's instruction hook updates shadow_mem as part of the
    # same instruction dispatch, so by the time any mem-write hook
    # runs, color A reflects the post-write state.
    # -------------------------------------------------------------------------
    findings: list[dict] = []

    def on_write(ql, access, address, size, value):
        # Cheap reject: anything outside the canary range can't trip B.
        if not color_b.is_tainted(address, size):
            return

        # The write hit a color-B byte. Record the event.
        # Read the per-byte color-A mask at this destination — non-zero
        # means tainted-source bytes physically landed here.
        ca_mask = color_a.read_mask(address, min(size, 8))
        # Also read color B at this destination — for completeness we
        # report which canary bytes were stomped.
        cb_mask = color_b.read_mask(address, min(size, 8))

        findings.append({
            'address': address,
            'size': size,
            'value': value & ((1 << (size * 8)) - 1),
            'color_a_mask': ca_mask,
            'color_b_mask': cb_mask,
            'pc': ql.arch.regs.read('rip'),
        })

    ql.hook_mem_write(on_write)

    # -------------------------------------------------------------------------
    # Run.
    # -------------------------------------------------------------------------
    print(f'[*] Running harness with priv->len={CTL_BYTES[0]}, '
          f'priv->size={CTL_BYTES[1]}, op={CTL_BYTES[2]}')
    try:
        ql.run()
    except Exception as exc:
        print(f'[!] Qiling raised: {exc}', file=sys.stderr)

    # -------------------------------------------------------------------------
    # Report.
    # -------------------------------------------------------------------------
    if not findings:
        print('[✗] NO color-B overlap detected — this is unexpected for '
              'the unpatched code. Check the layout or the trigger '
              'parameters.')
        return 1

    # Two populations:
    #   - color B alone: every guest write that touched canary territory.
    #     This includes the harness's OWN loop that initialised the
    #     canary pattern; those writes are legitimate (the harness, not
    #     the buggy function, did them) and are NOT colored A because
    #     the source operand was a hardcoded immediate, not stdin data.
    #   - color A ∩ color B: writes that BOTH crossed into sentinel
    #     territory AND carried attacker-tainted bytes. This is the
    #     precise bug signature — exactly the writes nft_byteorder_eval
    #     made past dreg + len with bytes derived from stdin.
    mixed = [f for f in findings if f['color_a_mask']]

    print(f'\n[*] Total writes touching color-B sentinel : {len(findings)}')
    print(f'    (this includes the harness\'s own canary-init loop,')
    print(f'     which writes hardcoded constants — not from stdin —')
    print(f'     and so does NOT carry color A)')

    print(f'\n[*] Writes where COLOR-A ∩ COLOR-B is non-empty : '
          f'{len(mixed)}')
    print(f'    (these are the ones whose source bytes came from stdin')
    print(f'     and whose destination is sentinel territory — the')
    print(f'     precise memory-corruption signal)\n')

    if mixed:
        print(f'    {"#":>3}  {"PC":>10}  {"DST":>12}  {"SIZE":>4}  '
              f'{"VALUE":>6}  {"COLOR-A":>10}  {"COLOR-B":>10}  NOTES')
        for i, f in enumerate(mixed):
            offset_in_canary = f['address'] - canary_after_addr
            notes = []
            if 0 <= offset_in_canary < CANARY_AFTER_SIZE:
                notes.append(f'canary_after[+{offset_in_canary:d}]')
            print(f'    {i:>3}  {f["pc"]:#010x}  {f["address"]:#012x}  '
                  f'{f["size"]:>4}  {f["value"]:#06x}  '
                  f'{f["color_a_mask"]:#010x}  {f["color_b_mask"]:#010x}  '
                  f'{", ".join(notes)}')

    print(f'\n[*] Distinct PCs touching color B : '
          f'{len(set(f["pc"] for f in findings))}')
    print(f'[*] Distinct PCs in A∩B mix       : '
          f'{len(set(f["pc"] for f in mixed))}')

    if mixed:
        print(f'\n[✓] Color A ∩ Color B is non-empty — attacker-tainted '
              f'data\n    physically reached sentinel territory. This is '
              f'the memory-corruption\n    signature of the '
              f'nft_byteorder bug.')
        return 0

    print(f'\n[~] Color B was hit but color A was not seen at the same '
          f'destination.\n    The geometry of the bug was caught; the '
          f'attacker-data flow was not.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
