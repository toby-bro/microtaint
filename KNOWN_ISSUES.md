# Known issues / modelling gaps

Tracked soundness-relevant divergences that are understood but not yet fixed.

---

## 1. `OF` after multi-bit shift/rotate: SLEIGH preserves, silicon recomputes

**Status:** open, deliberately deferred (2026-07-16). Narrow; no known corpus/fuzzer case hits it.

### What the ISA says
Intel SDM:
* `SHL/SHR/SAL/SAR` — *"The OF flag is affected only on 1-bit shifts; otherwise it is
  undefined."*
* `ROL/ROR` — *"The OF flag is affected only for single-bit rotates; it is undefined for
  multi-bit rotates."*

So for `shl rax, 8` / `rol rax, 8` the architectural value of `OF` is **undefined**: any
model is "legal" per the manual.

### What the two backends actually do
SLEIGH lifts the OF update as (see `ctx.translate` of `48c1e004`):

```
INT_EQUAL  u1 <- [count, 1]              # count == 1 ?
INT_XOR    u2 <- [CF, MSB(result)]       # the 1-bit formula
INT_AND    u3 <- [!(count==1), OF]       # <-- PRESERVES the OLD OF
INT_AND    u4 <- [(count==1), u2]
INT_OR     OF <- [u3, u4]                # OF = (count==1) ? CF^MSB : OLD_OF
```

i.e. **SLEIGH models "undefined" as "keep the previous OF"** -> `T_OF = T_OF_old = 0`.

Unicorn/QEMU instead **recomputes** OF. Measured (`OF_in` does not affect `OF_out`):

| instr (RAX=0x00FF00FF00FF00FF) | OF_in=1 -> OF_out | OF_in=0 -> OF_out | verdict |
| ------------------------------ | ----------------- | ----------------- | ------- |
| `shl rax, 8`                   | 1                 | 1                 | recomputes (ignores old) |
| `rol rax, 8`                   | 1                 | 1                 | recomputes (ignores old) |

With `res = 0xff00ff00ff00ff00` (MSB=1) and `CF=0`, `OF = MSB ^ CF = 1` — the 1-bit
formula applied to the *final* result, which is what real Intel/AMD silicon does.
Hardware genuinely *writes* OF; "undefined" only means software must not rely on it.

**This is NOT a Unicorn bug.** Unicorn matches silicon; SLEIGH's "preserve" is the
simplification that diverges from hardware. Neither violates the SDM.

### Why it matters (the soundness direction)
The engine runs the **p-code** backend by default (`CellSimulator.use_unicorn == False`),
so it reports `T_OF = 0` after a multi-bit shift. On **real hardware** OF is recomputed
*from the tainted result*, so:

```
shl rax, 8      ; silicon: OF := MSB(result) ^ CF   <- derived from tainted RAX
seto dl         ; DL carries tainted data on real hardware
```

the engine calls `DL` clean => **under-taint on real hardware**. It is a SLEIGH modelling
artifact, not a rule-family failure.

Latent today only because nothing generates `seto`/`jo` directly after a multi-bit shift
(the fuzzer's `seto` cases follow `sub`). Compilers never emit it — reading OF here is
reliance on undefined behaviour — but hand-written/obfuscated asm could.

### How it surfaced
Routing COND_TRANSPORTABLE flags through the 2-replica differential (which fixed the
`shl`->CF under-taint) made the two backends disagree, failing
`tests/test_cell_benchmark.py::test_pcode_matches_unicorn` for `SHL r64,imm8`,
`SHL r64,cl`, `ROL r64,imm8`:

```
mismatch on keys: ['EFLAGS', 'OF']
  OF: unicorn=0x1  pcode=0x0
```

The differential did not *create* this — it **exposed** a pre-existing p-code-vs-silicon
gap that the (now-deleted) per-shape `_is_bit_extract_*` gates were hiding.

### Options when we pick this up
1. **Conservatively taint OF for shift/rotate with count != 1** (preferred). Matches both
   silicon and Unicorn, closes the latent under-taint; costs ~nothing since no correct
   program reads OF here.
2. Exempt the undefined OF in the p-code-vs-unicorn gate (the test has a
   `PCODE_PRECISE_EXEMPTIONS` hook). Honest about it being unspecified, but leaves the
   hardware under-taint in place.
3. Fix the SLEIGH semantics to recompute OF (upstream-ish; largest blast radius).
