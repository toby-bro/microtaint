# Machine-checked soundness of the rule family (Appendix A)

Z3 discharges, per category, the claim that a rule never leaves a truly tainted
output bit clear. Each P-code operation `f` is encoded as a bit-vector function
and the category's closed form `R` exactly as the engine evaluates it, then Z3
is asked for an under-taint witness:

```
a, a'  agree on every untainted bit       (a & ~T == a' & ~T)
flip = f(a) ^ f(a')                       # a real, witnessed dependency
R    = rule(f, V, T)                      # the engine's rule at this state
witness  <=>  (flip & ~R) != 0            # a tainted bit the rule misses
```

`UNSAT` means no input escapes the rule, i.e. `GT ⊆ R` at that width. The query
is symbolic over all `2^(k·w)` values at each width, and the engine only ever
evaluates rules at `w <= 64` (SIMD lanes are split before evaluation), so
sweeping `w` in 1..64 covers the whole input space the engine can present. This
is a proof, not a bounded check.

```sh
uv run --with z3-solver python prove_soundness.py
```

Exit 0 iff every category is proved sound and both negative controls fire. The
avalanche/MULT check is capped at a small width, since a 64-bit multiplier
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

The other `prove_*.py` files prove the individual terms the same way:
comparison, equality, signed overflow with and without carry-in, variable bit
select, variable multiply and variable shift.
