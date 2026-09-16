# Table 5: cross-ISA soundness and precision

The campaign behind Table 5. It runs the full per-ISA corpus (about 1,500
instruction forms) against a `2^k` noninterference oracle, in practice a
one-bit-flip differential for about 90% of cases (see below), and reports,
per ISA, how many cases were checked, how many under-tainted, how often the
engine's mask was bit-exact, and how much it over-tainted.

This is a different experiment from the one in the parent directory.
`../campaign.py` is a fast soundness smoke test over about 80 forms; it answers
"does anything under-taint", cheaply, and it is what `run-all.sh` runs. This
directory answers Table 5, which needs the full corpus and the precision
columns, and it takes hours.

```sh
./run_table5.sh 12      # the paper's scale, about 55M cases across five ISAs
./run_table5.sh 0.2     # smoke run: proves the harness works, nothing else

python table5.py --tex table5.tex --json table5.json
```

Pin the engine under test with `$MT_ENGINE_ROOT`; without it the campaign
measures this repository. The paper's numbers come from a frozen tag.

## Why there is a validator

A soundness campaign reports "0 under-taints", and that sentence is only worth
something if the harness could have said otherwise. This one has been unable to
say otherwise twice, in two different ways, and both times it exited 0 with a
well-formed report:

* **The oracle invented taint.** `GTSim` reused one Unicorn across the `2^k`
  polarity runs of a case and initialised only four GPRs, so untracked
  registers carried run *i-1* into run *i*. `shrd rbp, rsp` has RBP as both
  source and destination and untracked, so the ground truth claimed PF depended
  on a bit that is not even in the shift count. That one form produced **2,810
  of the 2,875** under-taints in the 12h campaign of 2026-07-24.

* **The oracle measured nothing.** The fix for the above verified completion by
  reading PC. Unicorn does not update MIPS's PC after `emu_start` (it reads back
  the start address), so every MIPS run raised; the parent swallowed the raise
  in a per-run `except Exception: continue`; `outs` came back empty; and the
  empty accumulator was returned **marked exact**. An all-zero ground truth
  makes `gt & ~mt` unconditionally zero, so no under-taint can ever be
  reported. **19,085,696 MIPS cases** were scored that way on 2026-09-12, which
  is 36% of that campaign.

So `validate_oracle.py` checks the properties that make a zero mean something,
and exits non-zero if any fails:

```sh
python validate_oracle.py --cases 4 --forms 8
```

| property | what it proves |
| --- | --- |
| non-vacuity | the ground truth actually finds taint on this ISA |
| mutation | drop one truly-tainted bit from the engine's answer and the campaign's own test catches it |
| invalidation | a case whose runs cannot complete is counted as skipped, not scored as an empty ground truth |
| no-phantom | `shrd rbp, rsp` reports no taint for RCX bit 21, which is not in CL and cannot matter |

Completion is verified by counting executed instructions with a code hook
rather than by reading PC, because the hook is ISA-independent and costs
nothing measurable.

`table5.py` applies the same discipline at reporting time: it refuses to
certify a table in which any ISA has zero checked cases, or checked cases with
zero ground-truth bits.

## Instructions the oracle cannot execute

Some corpus entries are counted as `skipped`, never as passing. The distinction
matters: a skipped form is one the campaign could say nothing about, and folding
it into the answer as a pass is how a harness reports coverage it does not have.

**x86-64: `adcx` and `adox` (10 forms) are never scored at all.** Unicorn does
not implement the ADX extension, so every run raises
`UC_ERR_INSN_INVALID` and every case is invalidated: 162,960 attempted, 0
checked. So roughly 1.9% of the x86 corpus is **unmeasurable**, not
measured-and-sound, and this campaign says nothing about how the engine handles
those two mnemonics.

This is also where the old oracle went wrong in the flattering direction. It
scored `adcx`/`adox` at 28,890 cases each and 0% exact, on runs that never
executed, so failures were folded in as real comparisons and quietly depressed
the x86 exactness figure. Counting them as skipped is both more honest and more
accurate.

**MIPS64: two different causes, both partial.**

* `add` and `sub` are the TRAPPING variants and raise on signed overflow, so
  roughly 14-17% of random operand pairs legitimately have no ground truth.
  Confirmed by contrast: `addu` completes on every case.
* `clz`, `clo`, `dclz`, `dclo`, `movn` and `movz` raise `UC_ERR_EXCEPTION` in
  this Unicorn build on about half of states. State-dependent rather than a flat
  "unsupported"; the precise trigger is not pinned down.

Together these are 757,874 skipped cases, leaving MIPS64 at 96.9% of attempted
cases actually checked. Every other ISA is at 100% except x86-64 at 98.5%.

Every skipped case in this campaign comes from a run that could not complete.
The oracle also has a `k <= 13` budget beyond which it would fall back to a
non-exact lower bound, but the generator caps k at 8, so that path is never
taken here and nothing is skipped for being too heavily tainted.

## Over-taint is measured against a one-bit-flip oracle, so some of it is ours

The over-taint numbers must not be read as "the engine is this imprecise". Part
of the gap belongs to the oracle, by construction.

**The scored population is about 90% SINGLE-BIT-FLIP cases.** The generator
sweeps every bit of every source register one at a time (k = 1) and adds only
`MULTI_PER_RND = 6` multi-bit cases per base state with k drawn from 2 to 8.
Measured per form per round: 92.0% of x86-64 cases are k = 1, 94.4% on ARM64,
92.8% MIPS64, 86.6% PPC32, 93.7% RV64GC. So while the oracle is a `2^k`
enumeration in form, in practice k is nearly always 1 and it is a one-bit-flip
differential.

For such a case the ground truth is: flip this ONE input bit at ONE concrete
state, and record which output bits change. That is exact for that state and
that bit, and it is a LOWER BOUND on semantic dependence in two separate ways.
An output can depend on an input bit only in combination with other bits, which
a single flip never exhibits (carry interactions are the obvious family). And an
output can depend on an input bit at some other state while being insensitive at
this one.

A sound engine has to taint an output bit if it can depend on the input at any
reachable state. The oracle only ever witnesses one. So a bit the engine taints
that does not move at this particular state is counted as over-taint even when
the engine is right and the oracle simply did not sample the state that would
have shown it.

Two consequences for how the column should be quoted:

* the measured over-taint is an **upper bound** on the engine's true
  imprecision, not an estimate of it;
* the same applies in reverse to soundness, but harmlessly. A too-small ground
  truth makes the under-taint test `gt & ~mt` more lenient, never stricter, so
  a reported ZERO stays trustworthy while a reported over-taint may not be the
  engine's fault.

Cases with genuinely state-independent structure, like the parity flag reading
only the result's low byte, are unaffected by this and are real over-taints.
The caveat bites hardest on flags whose dependence is value-conditioned.

## The over-taint column

Table 5's over-taint figure is the **mean of the per-case** tainted-to-minimum
bit ratio. That is not the same as the ratio of summed bits, which weights
large-taint cases more heavily and gives a materially different number (2.80x
against 1.68x on AMD64), and it cannot be recovered from stored sums after the
fact. `run_campaign.py` accumulates it per case; `table5.json` records both
aggregations so the caption can state which one it means.

A mean is a poor summary of this distribution and the run shows why. Avalanche
cases are legitimate: one tainted bit entering a multiply taints all 64 outputs,
and no engine can do better. Those cases sit at 64-128x and drag the mean up, so
it reports how many multiplies the corpus contains as much as it reports the
engine. The campaign therefore also keeps a log2 histogram of the per-case
ratio, from which `table5.py` reports the median bucket. Measured over 68M
cases, **every ISA has a median case of ratio <= 1**, i.e. the typical case is
exact, while the means run 1.35x to 5.53x. Quote the mean and the median
together, or the column says something it does not mean.

## Quarantine

Some forms are excluded from scoring, and the campaign says which and why:

* **Unicorn defects**, e.g. `bzhi`: Unicorn flips CF at N=63 where Intel says
  64, and keeps masking bit 63 for every N >= 64, so both the flag and the
  result are wrong. Comparing against a wrong oracle measures the oracle.
* **Forms reading state the oracle does not model**, found by a sampled closure
  probe with no hardcoded list. It catches `shrd rbp, rsp` on its own.

## Files

Everything here except `run_table5.sh`, `table5.py` and `validate_oracle.py` is
vendored from the campaign harness that produced the published numbers, so the
artifact reproduces the paper rather than a re-derivation of it. The files carry
lint-suppression headers for that reason: restyling proven code is a behaviour
risk, and correctness is gated by `validate_oracle.py` instead.
