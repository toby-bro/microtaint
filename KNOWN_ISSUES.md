# Known issues / modelling gaps

Tracked soundness-relevant divergences that are understood but not yet fixed.

---

## -1. New ISAs under-taint on flag-extract and borrow-chain sequences

**Status:** open (2026-07-17). Affects ARM64, PPC32BE (and by inspection
SPARC32BE); MIPS64BE has no condition flags so is not affected.

The deterministic cross-ISA oracle (multiarch_oracle.py) reports 0 under-taints
on all five ISAs -- but its corpus is REGISTER-ARITHMETIC ONLY (no flags, no
conditional/carry sequences). The randomised fuzzer (multiarch_fuzz.py) adds
flag/carry/conditional SEQUENCES with random states and correlated taint masks,
and immediately finds under-taints:

    ARM64   cmp x1,x2; cset x0, lt        missed the flag bit (0x1)
    ARM64   subs x3,x1,x2; sbc x0,x1,x2   missed borrow-chain bits
    PPC32BE subfc 6,4,5; subfe 3,4,5      missed borrow-chain bits
    PPC32BE cmpw 0,4,5; mfcr 3            missed the CR0 field (0xe0000000)

These are the SAME classes fixed on x86 (flag-extract, borrow chains,
conditional select). The x86 fixes do NOT generalise: they key on x86's flat
1-bit flags (CF/OF/SF/ZF/PF/AF at fixed SLEIGH offsets) and on x86-specific
lifted patterns. ARM64 keeps its condition codes in NZCV, PPC in XER/CR fields,
SPARC in icc -- lifted to different p-code the current rules don't recognise.

So the defensible claim for the new ISAs is: rule SYNTHESIS is ISA-independent
and register-arithmetic is sound on all four, but flag/condition and
multi-instruction carry handling is NOT yet ported. It is a coverage gap in the
flag machinery, not a failure of the method (the SAME categories, applied to the
SAME classes, are already sound on x86).

**ROOT CAUSE (measured 2026-07-18) — corrects the earlier "general fix" below.**
These under-taints are in flag *consumption*, not production. Measurement: ARM64
`cmp x1,x2` already produces sound N/Z flag taint (0/3000 under-taints vs 2^k exact
GT, borrow-chain included). The fuzzer cases (`cmp;cset`, `cmp;csel`, `subs;sbc`)
under-taint because the flag-CONSUMING instruction under-taints:

  cset x0, lt  ->  X0 <- masked-cell(N,V) AND AVALANCHE(T_N|T_V)

The COND_TRANSPORTABLE / MONOTONIC 1-bit-flag **floor** was gated on
`out_bit_end - out_bit_start <= 7` (engine.py ~1803), sized for x86 `setcc` BYTE
outputs. ARM64 `cset`/`csel` write the flag into a **64-bit** GPR, so the floor
never fired; with the tainted flag masked to 0, `N!=V` collapses to 0 and X0 was
reported clean.

**PARTIAL FIX landed (a9b88bf).** Both COND flag floors (the 2-replica
differential and the FullMaskAvalanche-per-flag) now fire when a flag is consumed
into a wide register. x86 `setcc` unchanged (still <=8 bits); full x86 soundness
suite passes. ARM64 cross-ISA fuzzer under-taints drop **45 -> 20**.

**Remaining ARM64/PPC under-taints (separate, still open):**
* `csel x0,x1,x2,cc` — conditional SELECT: x0 = cond ? x1 : x2.  This is a
  CMOV-style data select; the missed bits are DATA bits of x1/x2, not the flag —
  a select data-flow / gated-passthrough issue, not the flag floor.
* `subs;sbc`, `adds;adc` — multi-instruction borrow/carry chains (the carry-in
  taint across the two instructions).
* A few `cmp;cset` cases still miss (cmp produces all flags soundly — verified
  0/2000 N/Z/C/V vs exact GT — so it is a `cset`/ChainedCircuit interaction on a
  specific threaded flag pattern, not flag production).

**Note on intermediate-taint materialization ("splitting"):** it was prototyped
(M1–M4) and measured to be soundness-NEUTRAL — over identical cases it changes the
taint output in ZERO cases on every ISA (amd64/arm64/mips/ppc), because it
re-organises *where* taint is computed but reuses the same 2-corner differential,
re-deriving the identical result. It is production-side; these under-taints are
consumption-side. It was reverted from main (commit `revert: remove
intermediate-taint materialization`) and preserved on branch
`intermediate_splitting`. The real levers are the flag floors and the fix below.

---

## 0. The native C p-code cell evaluator is LITTLE-ENDIAN only

**Status:** WORKED AROUND (2026-07-17); native kernel BE-aware + partially routed
(2026-07-18). Big-endian architectures (MIPS64BE, PPC32BE, SPARC32BE) evaluate
through Unicorn instead of the native p-code kernel (`CellSimulator` forces
`use_unicorn=True` when the arch name ends `BE`). Three increments have since
chipped at option B: (1) *maximal-register* BE instructions route to the native
Cython kernel (the PPC carry fix); (2) the Cython kernel's register-file and memory
**byte math is now endianness-aware**, validated byte-for-byte against Unicorn;
(3) the routing now also sends **sub-register ops of <=4-byte parents** (PPC
`extsb`/`srawi`, SPARC-class 32-bit sub-register ops) to the native kernel -- ~15x
faster than Unicorn per instruction, no regression. Still on Unicorn: MIPS64 32-bit
ops (64-bit-GPR sub-window; the canonical-input caveat), memory, and the
`use_unicorn` force itself. Payoff so far is the native fast path + independent
cross-check, not under-taint reduction. Full plan, formulas, validation, and the
caveat: `docs/design/native-big-endian-cython.md`. The C kernel (`cell_c`) stays
LE-only -- it is selected only for LE targets. Does NOT affect x86-64/ARM64/RISCV64 (LE).

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

### Partial native-BE routing (2026-07-18): carry/borrow chains

The Unicorn workaround has a hole option 2 does not: **Unicorn PPC neither seeds
nor exposes the XER carry varnode** (`xer_ca`). So a carry-consuming instruction
(`adde` = `rA + rB + xer_ca`) reads carry-in `0` in both differential replicas
and its differential collapses -- `addc;adde` and `subfc;subfe` under-tainted the
high bits of the extended result (the PPC entries above).

Fix (`_native_be_safe`, `CellSimulator._use_native_be` / `_read_reg_concrete`,
`_run_concrete_step`): on a BE target, evaluate an instruction with the native
kernel **when it is native-BE-safe** -- no memory access and every register-space
varnode is a MAXIMAL named register (not a byte-window strictly contained in a
wider register). Reading a maximal register moves its whole integer value, so the
kernel's internal byte layout is unobservable and the result round-trips
regardless of endianness; and the native kernel models `xer_ca` as a first-class
varnode, so the carry is seeded (carry-in) and read back (carry-out) correctly.
This is a scoped slice of option 2: it makes the native kernel the concrete engine
for exactly the instructions where it is provably byte-order-independent.

The MAXIMAL requirement (not merely "named") is load-bearing: MIPS64 32-bit ops
read/write `register[GPR+4:4]`, a byte-window alias of the 64-bit GPR (the LOW
word under BE, which the LE kernel reads as the HIGH word). Naming alone would
admit them and mis-read the wrong half; requiring maximal registers keeps them on
Unicorn while still routing PPC GPR arithmetic (PPC GPRs have no wider container)
and MIPS 64-bit ops (verified byte-identical to Unicorn). `extsb` reads
`register[rA+3:1]` (an unnamed slice) and is likewise excluded.
`tests/test_ppc_carry_chain.py` covers the predicate (incl. the MIPS byte-alias
exclusion), the carry/borrow chains, and the `extsb` non-regression.

Honesty caveat improves here: for native-BE-safe instructions the differential is
again computed by the native kernel, independent of the Unicorn oracle -- so the
two-implementation soundness argument is restored for that subset. The rest of the
BE surface (sub-register slices, memory) is still Unicorn-only ("sound relative to
Unicorn"). **Still open on BE:** `cmpw;mfcr` (condition-register-into-GPR flag
consumption, the ARM64 `cset` analog -- a flag-floor issue, not endianness) and a
1-2 bit differential-precision residual on `subfc;subfe` when both operands are
densely tainted (the non-monotone 2-corner limitation, shared with the flag
floors). Full option 2 (endianness-aware register packing so the native kernel
handles sub-register slices and memory too) remains the endgame.

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
