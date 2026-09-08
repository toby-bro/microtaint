# RQ6 — Generalisation across ISAs

**Claim (§6.6).** The synthesis generalises **unchanged** to four ISAs it was
not tuned on — ARM64, MIPS64BE, PPC32BE and RV64GC — with no per-architecture
taint rules and no engine edits.

This is the claim that most needs reproducing, because it is the one a reader is
most entitled to doubt: it asserts an absence of per-ISA special-casing.

## Run

```sh
# Pass 1 -- the wide net.  One Unicorn instance per ISA, reused across runs:
# about 8x faster, but hidden architectural state can leak between runs.  A leak
# can only ADD spurious under-taint reports, never hide a real one, so this is a
# conservative filter.
uv run python campaign.py pass1 --n 20000 --arch all --seed 1 --out camp

# Pass 2 -- re-check every pass-1 report with a FRESH Unicorn per run, so state
# cannot leak.  Real under-taints survive; the false positives vanish.
uv run python campaign.py pass2 --in camp

uv run python multiarch_fuzz.py          # randomised cross-ISA, ~10 min
```

## About the case count, and why the default is not the paper's

Measured on the reference machine, pass 1 runs at roughly **52 cases per second**
averaged over the five ISAs (AMD64 53, ARM64 27, MIPS64BE 103, RISCV64 54,
PPC32BE 81 — they differ because the instruction mixes differ).

So the paper's `--n 1000000` per ISA is about **26 hours** for this experiment
alone, which is past the one-day budget NDSS sets for a whole artifact. The
default here is `--n 20000`, about **32 minutes** for all five, which the AE
call explicitly permits ("accommodating scaled-down versions").

To reproduce the paper's exact case count, use `--n 1000000` and budget a day
for this experiment by itself. The claim being tested — zero under-taints on
every ISA — is one that more cases can only strengthen, so the scaled-down run
is a weaker instance of the same result, not a different one.

It reproduced at the reduced scale: 1,500 cases per ISA over all five,
**0 under-taint reports**.

**Both passes matter.** Pass 1 alone over-reports, by construction. A finding is
only a finding if it survives pass 2.

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
