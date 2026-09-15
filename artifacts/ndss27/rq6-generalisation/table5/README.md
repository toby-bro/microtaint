# Table 5: cross-ISA soundness and precision

The campaign behind Table 5. It runs the full per-ISA corpus (about 1,500
instruction forms) against an exact `2^k` noninterference oracle and reports,
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

## The over-taint column

Table 5's over-taint figure is the **mean of the per-case** tainted-to-minimum
bit ratio. That is not the same as the ratio of summed bits, which weights
large-taint cases more heavily and gives a materially different number (2.80x
against 1.68x on AMD64), and it cannot be recovered from stored sums after the
fact. `run_campaign.py` accumulates it per case; `table5.json` records both
aggregations so the caption can state which one it means.

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
