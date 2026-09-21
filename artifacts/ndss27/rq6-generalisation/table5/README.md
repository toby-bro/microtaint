# The cross-ISA table (Table IV): soundness and precision

The campaign behind Table IV. It runs the full per-ISA corpus against a bit-flip
noninterference oracle and reports, per ISA, how many cases were checked, how
many under-tainted, how often the engine's mask was bit-exact, and how much it
over-tainted.

**No measured value appears in this file.** The paper is the authority on every
figure; this README describes what the experiment does and how to run it, so the
two can never drift apart. Run the campaign and read the table it prints.

The oracle taints an arbitrary mask and flips each tainted bit once, from BOTH
polarities: once from the given state and once from the state with every tainted
bit inverted, unioning the two. That costs `2(k+1)` runs for `k` tainted bits,
linear rather than the `2^k` an exact enumeration needs, and it is still a lower
bound on semantic dependence -- a single flip cannot witness a dependence that
needs two inputs to move together. The second polarity exists because for the
common families that blind spot sits at a CORNER: `a AND b` hides where both
bits are 0, `a OR b` where both are 1. Measured against the exact enumeration on
overlapping masks, the second polarity recovers nearly all of the dependences a
single polarity misses on `and`, `or`, `add` and `imul`. The residual runs the
safe way for soundness, since a too-small ground truth makes `gt & ~mt` more
lenient, never stricter.

This is a different experiment from the one in the parent directory.
`../campaign.py` is a fast soundness smoke test over a small corpus; it answers
"does anything under-taint", cheaply, and it is what `run-all.sh` runs. This
directory answers Table IV, which needs the full corpus and the precision
columns, and it takes hours.

```sh
./run_table5.sh 12      # a long run across five ISAs
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
  on a bit that is not even in the shift count. That ONE form produced nearly
  every under-taint reported by the 2026-07-24 campaign.

* **The oracle measured nothing.** The fix for the above verified completion by
  reading PC. Unicorn does not update MIPS's PC after `emu_start` (it reads back
  the start address), so every MIPS run raised; the parent swallowed the raise
  in a per-run `except Exception: continue`; `outs` came back empty; and the
  empty accumulator was returned **marked exact**. An all-zero ground truth
  makes `gt & ~mt` unconditionally zero, so no under-taint can ever be
  reported. Most of one ISA's cases were scored that way on 2026-09-12, better
  than a third of that whole campaign.

So `validate_table5_oracle.py` checks the properties that make a zero mean something,
and exits non-zero if any fails:

```sh
python validate_table5_oracle.py --cases 4 --forms 8
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

## What the oracle watches, and why that is the whole ballgame

The oracle sets up and inspects a fixed register set per ISA, and pins
everything else to zero. Whatever is not in that set is not measured: a corpus
entry naming it is not tested, it is tested AT ZERO. `a('mul r8', ['RAX'])`
names `r8` deliberately, to exercise a register the engine has to get right, and
against a four-register model it multiplied by zero. Dozens of x86 forms and
dozens of PPC ones had an empty ground truth over their entire run and scored
bit-exact.

The sharper problem is that the state handed to the ENGINE is built from the
same list. The oracle knew `r8` was zero; the engine was never told `r8` exists.
The two sides were being asked about different machines, and the engine, which
correctly assumed an unknown multiplier, was charged every one of those cases
as over-taint for it.

So the model covers every general-purpose register the corpus can reach: 15 on
x86-64, 31 on PPC32 plus all eight condition fields, and HI/LO on MIPS64 so that
`mult` and `div` have a scored destination at all. ARM64 and RISCV are
deliberately NOT widened: no corpus entry there leaves the first four registers,
so it would cost measurably more to measure nothing new, and the gap on those
two is
the corpus rather than the model.

Every added name is round-tripped through Unicorn AND the engine's register map
before it is used. That check is not ceremony: MIPS `T0`-`T3` failed it, because
Unicorn calls `$8`-`$11` `T0`-`T3` while the engine calls them `A4`-`A7` under
the N64 ABI. Adding them would have varied a register whose taint is silently
discarded, replacing a visible vacuity (pinned to zero, empty ground truth) with
an invisible one.

## Half of a ground truth is the input passing through

The scored set includes registers an instruction only READS. For those the
ground truth is tautologically the input mask -- a no-op of the same length
produces it identically -- because flipping a bit of a register the instruction
does not write changes exactly that bit of its output.

Measured corpus-wide that is the majority of every ground-truth bit, so a
"bit-exact %" over all cases is in substantial part a test that a copy survives,
which any engine passes. `table5.py` therefore reports both: exactness over all
scored cases, and exactness over cases whose ground truth differs from the input
mask somewhere, alongside the fraction of ground-truth bits that are signal
rather than passthrough.

## Instructions the oracle cannot execute

Some corpus entries are counted as `skipped`, never as passing. The distinction
matters: a skipped form is one the campaign could say nothing about, and folding
it into the answer as a pass is how a harness reports coverage it does not have.

**x86-64: `adcx` and `adox` are never scored at all.** Unicorn does
not implement the ADX extension, so every run raises `UC_ERR_INSN_INVALID`.
Preflight now detects this by RUNNING each form at sampled states and
quarantining any that never completes, so these are excluded before scoring
rather than accumulating as invalidated cases, and the next such family is
caught without anyone adding it to a list. A small part of the x86 corpus is
therefore **unmeasurable**, not measured-and-sound, and this campaign says
nothing about how the engine handles those two mnemonics.

This is also where the old oracle went wrong in the flattering direction. It
scored `adcx`/`adox` on runs that never executed, so failures were folded in as
real comparisons and quietly depressed the x86 exactness figure. Counting them
as skipped is both more honest and more accurate.

**MIPS64: two different causes, both partial.**

* `add` and `sub` are the TRAPPING variants and raise on signed overflow, so
  a sizeable share of random operand pairs legitimately has no ground truth.
  Confirmed by contrast: `addu` completes on every case.
* `clz`, `clo`, `dclz`, `dclo`, `movn` and `movz` raise `UC_ERR_EXCEPTION` in
  this Unicorn build on about half of states. State-dependent rather than a flat
  "unsupported"; the precise trigger is not pinned down. These previously
  reported checked cases at perfect bit-exactness for instructions that never
  ran: a
  case whose taint mask canonicalisation had emptied returned an empty ground
  truth MARKED VALID without invoking Unicorn at all, and an empty ground truth
  scores bit-exact against anything. Such a case is now a skip.

MIPS64 therefore checks a noticeably smaller share of its attempted cases than
the other ISAs; x86-64 loses a little to the quarantined mnemonics above, and
the rest lose nothing. The campaign prints the per-ISA figures.

Every skipped case in this campaign comes from a run that could not complete,
or from a taint mask that canonicalisation emptied. There is no enumeration
budget: the oracle is linear in `k`, so a case tainting every bit of every
source register is affordable and nothing is skipped for being too heavily
tainted.

## Over-taint is measured against a lower bound, so some of it is ours

The over-taint numbers must not be read as "the engine is this imprecise". Part
of the gap belongs to the oracle, by construction.

The ground truth for a case is: flip each tainted input bit once, from each of
two polarities, at ONE concrete state, and record which output bits change. That
is a LOWER BOUND on semantic dependence in two separate ways. An output can
depend on an input bit only in combination with others, and while two polarities
catch the corner cases where AND and OR hide, they do not catch everything: a
zero flag depends on every result bit but changes only when the result crosses
zero, which no single flip from either corner exhibits. And an output can depend
on an input bit at some other state while being insensitive at this one.

A sound engine has to taint an output bit if it can depend on the input at any
reachable state. The oracle witnesses two corners of one state. So a bit the
engine taints that does not move here is counted as over-taint even when the
engine is right and the oracle simply did not sample the state that would have
shown it.

Two consequences for how the column should be quoted:

* the measured over-taint is an **upper bound** on the engine's true
  imprecision, not an estimate of it;
* the same applies in reverse to soundness, but harmlessly. A too-small ground
  truth makes the under-taint test `gt & ~mt` more lenient, never stricter, so
  a reported ZERO stays trustworthy while a reported over-taint may not be the
  engine's fault.

Cases with genuinely state-independent structure, like the parity flag reading
only the result's low byte, are unaffected by this and are real over-taints.
The caveat bites hardest on flags whose dependence is value-conditioned, and the
zero flag is the extreme: measured on the earlier one-polarity campaign, ARM64's
`ZR` and x86's `ZF` moved in only a small fraction of cases, so nearly all of the
over-taint charged against them is dependence the oracle cannot witness rather
than imprecision the engine could remove.

## The over-taint column

Table IV's over-taint figure is the **mean of the per-case** tainted-to-minimum
bit ratio. That is not the same as the ratio of summed bits, which weights
large-taint cases more heavily and gives a materially different number, and it
cannot be recovered from stored sums after the fact. `run_campaign.py` accumulates it per case; `table5.json` records both
aggregations so the caption can state which one it means.

A mean is a poor summary of this distribution and the run shows why. Avalanche
cases are legitimate: one tainted bit entering a multiply taints all 64 outputs,
and no engine can do better. Those cases sit far out in the tail and drag the
mean up, so
it reports how many multiplies the corpus contains as much as it reports the
engine. The campaign therefore also keeps a log2 histogram of the per-case
ratio, from which `table5.py` reports the median bucket. Measured over 68M
cases, **every ISA has a median case of ratio <= 1**, i.e. the typical case is
exact, while the means are substantially higher. Quote the mean and the median
together, or the column says something it does not mean.

## Quarantine

Some forms are excluded from scoring, and the campaign says which and why:

* **Unicorn defects**, e.g. `bzhi`: Unicorn flips CF at N=63 where Intel says
  64, and keeps masking bit 63 for every N >= 64, so both the flag and the
  result are wrong. Comparing against a wrong oracle measures the oracle.
* **Forms reading state the oracle does not model**, found by a sampled closure
  probe with no hardcoded list. It catches `shrd rbp, rsp` on its own.

## Files

Everything here except `run_table5.sh`, `table5.py` and `validate_table5_oracle.py` is
vendored from the campaign harness that produced the published numbers, so the
artifact reproduces the paper rather than a re-derivation of it. The files carry
lint-suppression headers for that reason: restyling proven code is a behaviour
risk, and correctness is gated by `validate_table5_oracle.py` instead.
