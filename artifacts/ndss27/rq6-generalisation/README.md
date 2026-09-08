# RQ6 — Generalisation across ISAs

**Claim (§6.6).** The synthesis generalises **unchanged** to four ISAs it was
not tuned on — ARM64, MIPS64BE, PPC32BE and RV64GC — with no per-architecture
taint rules and no engine edits.

This is the claim that most needs reproducing, because it is the one a reader is
most entitled to doubt: it asserts an absence of per-ISA special-casing.

## Run

```sh
cd ../../../benchmark/generalization
uv run python campaign.py                # ~35 min, all four ISAs
uv run python multiarch_fuzz.py          # ~10 min, randomised cross-ISA
```

## What to look at

Per ISA, the campaign reports instructions covered, cases evaluated, and
under-taints against per-bit ground truth. **The number that carries the claim
is zero under-taints on every ISA.**

Precision (exact vs over-tainted) is expected to vary between ISAs and that is
not a defect: an ISA whose lifter models fewer flags gives the differential less
to be precise about.

## Reading the flag exclusions

Where an ISA leaves a flag *architecturally undefined* — x86 OF after a rotate
by anything other than one — SLEIGH and QEMU model it differently and both are
entitled to. Since the ground-truth oracle is QEMU, comparing there measures
which vendor guessed what. Those outputs are detected (by running both models
over the same states) and excluded from the verdict, and **reported**, in two
kinds: `isa-undefined` and `lifter-gap`. On this corpus the exclusions are
confined to x86; ARM64, MIPS64BE, PPC32BE and RV64GC have none.

## Tolerance

Exact. The corpus and the taint masks are fixed and the ground truth is an
enumeration, so a rerun that reports a different under-taint count is a real
disagreement and should be reported as such.
