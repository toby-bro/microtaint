# Multi-tool taint evaluation of the `nft_byteorder` bug (CVE-2023-35001)

This directory runs **seven dynamic taint engines** against the same reproduction
of the nftables `nft_byteorder` alignment/stride bug and reports, per engine, the
**exact over-taint and under-taint** at the point of memory corruption. The bug
was found and exploited at Pwn2Own Vancouver 2023 by Tanguy Dubroca (Synacktiv):
<https://www.synacktiv.com/publications/old-bug-shallow-bug-exploiting-ubuntu-at-pwn2own-vancouver-2023>.

The engines: **microtaint** (this project), **libdft64**, **taintgrind**,
**PANDA** (whole-program / DBI taint), and **angr**, **Triton**, **Maat**
(symbolic / emulation).

## The bug, reproduced faithfully

`harness.c` embeds `nft_byteorder_eval` **verbatim** from the unpatched kernel.
With `priv->size == 2` the evaluator iterates `priv->len / 2` times at **stride 4**
(the `sizeof` of a `union { u32; u16; }`) while writing only **2 bytes** per
iteration. The validator accepts `dreg + len <= 80` (the register file size), but
the loop reaches byte `2*len - 3`, overshooting the 80-byte register file.

The reproduction fixes `sreg = dreg = 0, len = 80, size = 2, op = NTOH`, so the
loop is an **in-place** byteswap of 40 elements:

```c
for i in 0 .. 39:                 # len/2 = 40 iterations, stride 4
    d[i].u16 = ntohs(s[i].u16)    # s == d == regs base
```

Memory layout (`struct layout`, contiguous):

```text
[ canary_before : 256 ][ regs : 80  (attacker-tainted from stdin) ][ victim : 256 ]
                        ^regs base                                  ^overshoot lands here
```

### The stride: written vs unwritten bytes

Each iteration writes the **low `u16`** of a 4-byte element and skips the high
`u16`. So the overshoot corrupts the victim in a characteristic comb pattern
(offsets into the victim region):

```text
WRITTEN   (stomped):   0,1  4,5  8,9  ... 76,77      (40 bytes)
UNWRITTEN (skipped):   2,3  6,7 10,11 ... 78,79      (40 bytes)
```

`ground_truth.py` works this map out directly from the loop, without running any tool.

### What is actually attacker-controlled

Because the operation is in place, iteration `i` reads its source from the *same*
offset it writes:

* `i = 0..19` (`4*i < 80`): source in the register file, dest **in bounds** ->
  the written value **is attacker-tainted**.
* `i = 20..39` (`4*i >= 80`): source in the victim region, dest **out of bounds**
  -> the written value is a byteswap of an **untainted constant**.

So **zero attacker-derived bytes reach the out-of-bounds sink by data flow.** The
overshoot is real memory corruption, but the attacker's control over it is through
the **tainted loop bound `len`** (a control dependence), not the written data. This
is the ground truth every engine is scored against.

## Method

Every driver `detect_<tool>.py` imports `common.py` so the experiment is identical
across engines:

* **Source (color A):** the 83 stdin bytes (`common.PAYLOAD`) are the taint source.
* **Sink (color B):** the `victim` region; addresses are deterministic
  (`-static -no-pie`, recovered from the `L` symbol via `nm`).
* **Verdict (`common.Verdict`):** for the vulnerable and the patched
  (`harness_fixed`) build, each engine records the out-of-bounds stride writes it
  sees, how many carry an attacker-tainted **value**, and how many of the 20
  in-bounds byteswap stores it taints.
* **Scoring (`ground_truth.py` + `report.py`):**
  * **over-taint** = OOB writes an engine calls value-tainted that the ground
    truth says are not (false positives).
  * **under-taint** = in-bounds attacker-tainted writes an engine misses (false
    negatives).

For the symbolic engines the 3 control bytes are concretized to the trigger
(`len=80, size=2, op=NTOH`) so the loop geometry matches the DBI runs; the 80
register bytes are symbolic. "Value-tainted" then means the written value is
input-dependent.

## Results

<!-- regenerate with: python report.py --md -->

| Tool | Family | OOB writes seen | OOB value-tainted | Over-taint (FP) | Under-taint (FN) | Control (patched) clean |
|---|---|---|---|---|---|---|
| `microtaint` | dbi | 20 | 20 | **20** | 0 | yes |
| `libdft64` | dbi | 20 | 0 | 0 | 0 | yes |
| `taintgrind` | dbi | 0¹ | 0 | 0 | 0 | yes |
| `panda` | dbi | 20 | 0 | 0 | 0 | yes |
| `angr` | symbolic | 20 | 0 | 0 | 0 | yes |
| `triton` | symbolic | 20 | 0 | 0 | 0 | yes |
| `maat` | symbolic | 20 | 0 | 0 | 0 | yes |

¹ `taintgrind` logs only *tainted* operations, so it emits 0 out-of-bounds
stores (none are tainted); it observes the 20 in-bounds tainted stores and the
tainted loop-bound test. Over/under-taint are still exact.

Every engine correctly taints all 20 **in-bounds** byteswap stores (no value
under-taints), and every engine's patched-harness control is clean.

## Findings

1. **The bug is a tainted-length-driven spatial overshoot, not an attacker-data
   flow.** Five independent precise engines (`libdft64`, `taintgrind`, `angr`,
   `Triton`, `Maat`) agree exactly: 20 out-of-bounds stride writes occur, and
   **none** carries an attacker-tainted value. The overshoot byteswaps the victim
   in place; the values written there are constants, not stdin.

2. **`taintgrind` alone reports the control dependence.** It flags the tainted
   loop-bound test (`i < len/2` as a tainted `IfGoto`) driving the overshoot,
   which the pure data-flow engines do not model.

3. **`microtaint` is the only engine that over-taints, and the taint it reports
   is the `len` control byte, not attacker register data.** Clearing only `len`
   at function entry drops the OOB value-taint from 20 to 0; clearing all 80
   attacker register bytes leaves it at 20. The over-taint is laundered by two
   compounding imprecisions: (1) `mov reg <- mem` does not clear stale register
   taint, so `len`'s taint from the loop-bound computation (`i < len/2`) survives
   into the *untainted* loop-index register; (2) the store/load address is then
   spuriously tainted, and loading through it taints the loaded value even though
   the source memory is the untainted sentinel (**pointer-avalanche**). So
   microtaint reaches the right conclusion (the overshoot is `len`-driven) for the
   wrong reason, and reports it in the value-data-flow column instead of as a
   control dependence. It stays **sound** (never under-taints) but is imprecise
   here; forcing a correct strong-update on the index `mov` collapses the 20 OOB
   value-taints to 0. The original "attacker data reached the sentinel" reading was
   resting entirely on this artifact.

## Reproduce

```sh
make                                   # build harness + harness_fixed
python report.py                       # aggregate results/*.json into the ledger

# per engine (each writes results/<tool>.json):
.venv/bin/python detect_microtaint.py                                   # this project's venv
<benchmark>/.venv_angr/bin/python   detect_angr.py
<benchmark>/.venv_triton/bin/python detect_triton.py
<benchmark>/.venv_maat/bin/python   detect_maat.py
<benchmark>/.venv_panda/bin/python  detect_panda.py
python detect_libdft64.py              # builds the Pin tool under libdft64_tool/
python detect_taintgrind.py            # builds the tg harness variant, runs docker taintgrind
```

`<benchmark>` = `.../PRIM/benchmark` (holds the tool installs under `external/`
and the per-tool venvs).

## Files

| file | role |
|---|---|
| `common.py` | shared source/sink/layout + `Verdict` schema |
| `ground_truth.py` | exact written/unwritten byte map + over/under scoring |
| `report.py` | aggregate `results/*.json` -> ledger (`--md` for Markdown) |
| `harness.c` / `harness_fixed.c` | vulnerable / patched reproduction |
| `detect_<tool>.py` | one driver per engine, all emitting `Verdict` |
| `<tool>_tool/` | tool-specific scratch (pintool, tg harness, loaders) |
| `results/<tool>.json` | per-engine verdict |
