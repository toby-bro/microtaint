# Machine-checked soundness of the rule family (Appendix A)

Z3 discharges, per category, the claim that a rule never leaves a truly tainted
output bit clear. Each P-code operation `f` is encoded as a bit-vector function
and the category's closed form `R` alongside it, then Z3 is asked for an
under-taint witness:

```
a, a'  agree on every untainted bit       (a & ~T == a' & ~T)
flip = f(a) ^ f(a')                       # a real, witnessed dependency
R    = rule(f, V, T)                      # the engine's rule at this state
witness  <=>  (flip & ~R) != 0            # a tainted bit the rule misses
```

`UNSAT` means no input escapes the rule, i.e. `GT ⊆ R` at that width. The query
is symbolic over all `2^(k·w)` values at each width, so within a width nothing
is sampled: that part is a proof and not a bounded check.

The widths themselves are a sample. `prove_soundness.py` sweeps
`[1, 2, 3, 4, 5, 8, 16, 32, 64]`, which covers the boolean case, the machine
widths the engine evaluates at, and the small widths where carry and borrow
behaviour is easiest to get wrong. It does not cover every `w` up to 64: a
3-byte or 5-byte varnode (`w = 24`, `w = 40`) is not among them. These rules are
uniform in the width, so a counterexample at an untested width would be
surprising, but it is a sample and not a sweep.

## What each rule is, and what it is not

The rules here are written in Z3, next to the operations they cover. They are
not extracted from the engine, so what is proved is that **the category
mathematics is sound**, not that microtaint implements that mathematics. The
second half is what the soundness campaigns and the differential oracle are
for; these proofs are what makes the campaigns' zero mean something rather than
being the only evidence.

```sh
uv run --with z3-solver python prove_soundness.py
```

For **that** script, exit 0 means every category was proved sound and both
negative controls fired. It is not a general rule for this directory: the
variable-operand scripts below exit 0 while reporting `UNKNOWN` entries, which
means Z3 gave up rather than that anything was proved. Read their tables, not
their exit codes.

The avalanche/MULT check is capped at a small width, since a 64-bit multiplier
bit-blasts slowly and avalanche soundness is width-independent.

| category | rule `R` | |
|---|---|---|
| mapped | `L(T_d)`, the differential on the single dynamic input | sound, and exact |
| weldable | `⋁_j T_j` | sound |
| avalanche | `Aval(⋁_j T_j, w)` | sound for any `f` |
| monotonic | `D` | re-checked (CellIFT) |
| transportable-add | `D^{++} ∨ T^{sx}` | re-checked (CellIFT) |
| transportable-sub | `D^{+-} ∨ T^{sx}` | re-checked (CellIFT) |

The three inherited categories were already proved at gate level by CellIFT; the
three software categories are what this is for.

Two deliberately broken rules must produce a witness, and do. ADD without the
union floor (`D` alone, dropping `∨ T`) is unsound because both polarised
replicas can miss an upward carry, witness at `w=1`. XOR mis-routed as *mapped*
(differential over one input, the other tainted input treated as constant) is
unsound because it misses the second operand, witness at `w=1`. They are why the
additive floor and the routing-opcode gate are load-bearing.

The other `prove_*.py` files check the individual terms the same way:
comparison, equality, signed overflow with and without carry-in, variable bit
select, variable multiply and variable shift.

## What actually completes

Measured on sixteen cores, each script run on its own. The distinction that
matters is between an entry Z3 discharged and one it gave up on: `UNKNOWN` is
the solver hitting its limits, and it is neither a proof nor a counterexample.
The scripts report it as such rather than folding it into a pass.

| script | time | result |
| --- | --- | --- |
| `prove_comparison_taint.py` | 10 s | all proved |
| `prove_equality_taint.py` | 12 s | all proved |
| `prove_signed_overflow.py` | 20 s | no counterexample at any tested width |
| `prove_signed_overflow_carryin.py` | 5 min | no counterexample; some entries UNKNOWN |
| `prove_variable_multiply.py` | 7 min | no counterexample; 3 entries UNKNOWN |
| `prove_soundness.py` | 44 min | **every category proved, no UNKNOWN, both controls fire** |
| `prove_variable_shift.py` | > 28 min | did not finish; of what it reached, soundness 17 proved and 4 UNKNOWN |
| `prove_variable_bit_select.py` | > 28 min | did not finish |

`prove_soundness.py` is the one the appendix rests on, and it is the one that
completes cleanly. The four variable-operand terms are the hard ones for a
bit-blasting solver: a shift or a bit-select by a symbolic amount forces a case
split over every possible amount, and at `w = 64` that is where Z3 stops.
Treat those four as checked at the widths they reached, not proved at all of
them.

## Why the negative controls matter

A harness that cannot fail proves nothing, so two rules that are known to be
wrong are checked alongside the real ones, and both must produce a witness.
They do, at `w = 1`. Without them, a bug that made every query trivially UNSAT
would read exactly like a clean sweep.
