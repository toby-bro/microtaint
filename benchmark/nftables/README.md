# Two-color taint detection of the `nft_byteorder` alignment bug (CVE-2023-35001)

A small proof-of-concept that uses **microtaint** with **two parallel
`BitPreciseShadowMemory` instances** to detect the stride/element-size
mismatch in the unpatched `nft_byteorder_eval`. The bug is in the
`case 2` branch, where a `union { u32; u16; }*` is indexed with stride
4 (the union's `sizeof`) while only 2 bytes are written per iteration,
allowing out-of-bounds writes past the validated destination region.

The vulnerability and full Pwn2Own Vancouver 2023 LPE exploit chain
are described in Tanguy Dubroca's writeup at Synacktiv:
<https://www.synacktiv.com/publications/old-bug-shallow-bug-exploiting-ubuntu-at-pwn2own-vancouver-2023>.
Exploit code: <https://github.com/Synacktiv/CVE-2023-35001>. Patch:
<https://lore.kernel.org/netfilter-devel/20230705210535.943194-1-cascardo@canonical.com/>.

This PoC is concerned only with **detecting the underlying memory
corruption** — not reproducing the exploit chain. The detector flags
the stride mismatch the moment the buggy loop writes a single byte
past the validated region.

## Files

- `harness.c` — userland driver. Embeds `nft_byteorder_eval` verbatim
  from the unpatched kernel source, plus minimal stubs to compile
  outside the kernel. Lays out a contiguous struct
  `[canary_before : 256B][regs : 80B][canary_after : 256B]` so the
  buggy stride is guaranteed to overshoot into known-address sentinel
  bytes.
- `harness_fixed.c` — same harness with the upstream-style fix applied
  to `case 2` (use `__be16 *` instead of `union*`).
- `detect.py` — the microtaint driver. Runs the harness under Qiling +
  `MicrotaintWrapper`, maintains a second independent
  `BitPreciseShadowMemory` (color B), and reports writes where the two
  shadows intersect.
- `Makefile` — `make` builds both.

## Methodology (two colors, two shadows)

| color | shadow                              | meaning                                              | propagates? |
|-------|-------------------------------------|------------------------------------------------------|-------------|
| A     | `wrapper.shadow_mem` (microtaint's) | "this byte is derived from attacker stdin"           | yes         |
| B     | a 2nd `BitPreciseShadowMemory` we own | "this byte is sentinel territory; nothing must write to it" | no, static  |

Both are real `BitPreciseShadowMemory` instances. Bit precision is
preserved end-to-end: every byte of every shadow carries an
independent 8-bit mask, exactly as documented in
`shadow.pyx:17` ("bit i set means bit i of the corresponding memory
byte is tainted").

A single `UC_HOOK_MEM_WRITE` callback on every guest write checks:

1. `color_b.is_tainted(address, size)` — did this write land in
   sentinel territory?
2. `color_a.read_mask(address, size)` — does the byte that just landed
   here also carry attacker taint?

`color_a ∩ color_b ≠ ∅` is the precise memory-corruption signal.

## Why bit precision matters here

The PoC marks color B as `0xFF` per byte and color A is also `0xFF`
per byte (because the wrapper's `_taint_bytes` is a full-byte source).
But the framework would happily distinguish, say, color B set only on
bit 0 of each canary byte vs color A on bit 1 — i.e. up to 8
independent dimensions per byte. That granularity isn't needed for
this particular bug (it's about whole-byte reach), but it's available
for bugs that are about partial-byte corruption (e.g. nibble overlays
in tagged-pointer schemes, parity bits, etc.).

## Running

```
make
uv run detect.py            # vulnerable run  → A∩B = 20 writes, 1 PC
# To verify the patched version:
uv run python3 -c "import detect; from pathlib import Path; \
            detect.HARNESS = Path('./harness_fixed'); \
            import sys; sys.exit(detect.main())"
                              # patched run    → A∩B = 0
```

## Expected output (vulnerable)

```
[*] Total writes touching color-B sentinel : 532
    (this includes the harness's own canary-init loop, which writes
     hardcoded constants — not from stdin — and so does NOT carry color A)

[*] Writes where COLOR-A ∩ COLOR-B is non-empty : 20
    (these are the ones whose source bytes came from stdin
     and whose destination is sentinel territory — the
     precise memory-corruption signal)

      #          PC           DST  SIZE   VALUE     COLOR-A     COLOR-B  NOTES
      0  0x00401dce  0x00004acc90     2  0xefcd  0x0000ffff  0x0000ffff  canary_after[+0]
      1  0x00401dce  0x00004acc94     2  0xefcd  0x0000ffff  0x0000ffff  canary_after[+4]
      ...
     19  0x00401dce  0x00004accdc     2  0xefcd  0x0000ffff  0x0000ffff  canary_after[+76]

[*] Distinct PCs touching color B : 3
[*] Distinct PCs in A∩B mix       : 1

[✓] Color A ∩ Color B is non-empty — attacker-tainted data
    physically reached sentinel territory. This is the memory-corruption
    signature of the nft_byteorder bug.
```

The 20 hits at stride 4 (offsets +0, +4, ..., +76 in `canary_after`)
are exactly the out-of-bounds writes the buggy loop performs:
`priv->len/2 = 40` iterations, the first 20 land in-bounds, the last
20 overshoot — all from the same `mov` instruction at PC `0x401dce`.

## Notes

- The detector keeps all of microtaint's built-in classifiers
  (`check_bof`, `check_uaf`, `check_sc`, `check_aiw`) **off**. We are
  not using its findings reporter; we are using its taint-propagation
  engine and its shadow as one of two parallel taint maps. Disabling
  the classifiers also stops the wrapper from calling `ql.emu_stop()`
  on the first finding, which would interrupt the very write we want
  to inspect.
- `color_b` is populated **before** `ql.run()` and never touched
  during the run — it's a static label, not a propagating taint. The
  wrapper's `MemWriteClearHook` does not touch it (it operates on
  `wrapper.shadow_mem` only).
- The harness is built `-no-pie` and `static` so symbol addresses are
  link-time-fixed and `nm` gives us the runtime address of the
  `[before, regs, after]` struct directly.

