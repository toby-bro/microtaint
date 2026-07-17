# Known issues / modelling gaps

Tracked soundness-relevant divergences that are understood but not yet fixed.

---

## 0. The native C p-code cell evaluator is LITTLE-ENDIAN only

**Status:** WORKED AROUND (2026-07-17). Big-endian architectures (MIPS64BE,
PPC32BE, SPARC32BE) now evaluate through Unicorn instead of the native p-code
kernel (`CellSimulator` forces `use_unicorn=True` when the arch name ends `BE`);
the cross-ISA oracle then reports 0 under-taints on all five ISAs. The native
kernel itself is still LE-only -- making it endianness-aware (option B below) is
the proper fix and remains open. Does NOT affect x86-64/ARM64/RISCV64 (LE).

One honesty caveat for any BE soundness claim: on x86 soundness rests on TWO
independent implementations agreeing -- the native C p-code evaluator
cross-checked against Unicorn. Under this workaround the BE engine *uses*
Unicorn, and the oracle also uses Unicorn, so there is no independent second
opinion: the BE claim is "sound relative to Unicorn", a weaker guarantee. That
is the reason option B (native, independently checkable) is the real endgame.

### Symptom
With `use_unicorn=False` (the default, the fast C p-code path) the cell evaluator
returns **0** for a big-endian target instead of executing the instruction. It does
not raise -- it silently yields zero:

```
MIPS64BE  addu $2,$4,$5   A0=A1=0x0F0F0F0F
  C p-code : f(V|T)=0x0         f(V&~T)=0x0         D=0x0    <-- WRONG
  Unicorn  : f(V|T)=0x1e1e1e1e  f(V&~T)=0x1e1e1e1c  D=0x2    <-- correct
```

### Why it under-taints
Every differential-based rule computes `D = f(V|T) XOR f(V&~T)`. If both runs
return 0 then `D = 0` and the differential contributes nothing:

* TRANSPORTABLE (`D v T`) collapses to `T` -- the operand taint still shows, but
  the CARRY is lost. `addu` with `T=0b1` reports `0x1` instead of `0x3`.
* MAPPED / MONOTONIC, whose rule *is* the differential, collapse to **0** --
  i.e. total under-taint.

Measured with the per-ISA single-bit-flip oracle
(`benchmark/generalization/multiarch_oracle.py`):

    ISA          cases  under-taints  exact
    AMD64          162             0    91%     little-endian
    ARM64          288             0    76%     little-endian
    MIPS64BE       234            57    58%     BIG-endian
    PPC32BE        252            12    63%     BIG-endian
    SPARC32BE      252            27    66%     BIG-endian

Every under-taint is on a big-endian target; both little-endian ISAs are clean.
This is an EVALUATOR limitation, not a rule/category/polarity failure: the rules
are fed zeros.

### Note on an earlier, WRONG measurement
Commit 8821591 reported "no under-tainting on any ISA; MIPS64BE and PPC32BE are
100% EXACT". That was an artifact of a vacuous test: the oracle tainted a
POSITIONAL slice of the register list (`regs[1:3]`) while the MIPS programs read
`$4/$5` and the PPC programs read `R4/R5`. No taint flowed through the
instruction at all, so nothing could be got wrong. The oracle now declares each
program's real source operands and taints those.

### Options
1. Force `use_unicorn=True` for big-endian architectures. Correct, and known to
   work (see the table above), but gives up the native p-code fast path.
2. Make the C p-code evaluator endianness-aware (register packing / state build).
   The real fix; largest blast radius.
3. Keep the BE ports as lift+synthesis only and do not claim soundness for them.

Related, already fixed: `StateMapper.map_to_state` did byte->bit arithmetic
little-endian-only, so any SUB-register read on a BE target mapped to the wrong
bits (SPARC's `sll %g1,%g2,%g3` reads `register:0xb:1` = g2's least-significant
byte, reported as bits 24..31 instead of 0..7). See `_sub_reg_bit_start`.

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
