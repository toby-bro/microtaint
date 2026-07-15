# Machine-checked soundness of the taint-propagation rules

This experiment discharges, with the Z3 SMT solver, the claim that every one of
MicroTaint's per-instruction taint-propagation rules is **sound** — it never
leaves a truly-tainted output bit clear (never *under-taints*).

## What it proves

For each taint category we encode the P-code operation `f` as a bit-vector
function and the category's closed-form rule `R` exactly as the engine evaluates
it, then ask Z3 for an *under-taint witness*: an input on which a genuine
noninterference flip escapes the rule.

```
a, a'  agree on every untainted bit        (a & ~T == a' & ~T)
flip = f(a) ^ f(a')                         # a real, witnessed dependency
R    = rule(f, V, T)                        # the engine's rule at this state
witness  <=>  (flip & ~R) != 0              # a truly-tainted bit the rule misses
```

`UNSAT` ⇒ no input escapes the rule ⇒ **`GT ⊆ R`** (sound) at that width.

**Why the width sweep is a complete proof, not a bounded approximation.**
MicroTaint only ever evaluates rules at operand widths `w ≤ 64` (SIMD lanes are
split to ≤64 bits before evaluation). The query is symbolic over *all* `2^(k·w)`
input values at each width, so sweeping `w ∈ {1..64}` discharges soundness over
the entire input space the engine can present.

## Categories covered

| category | rule `R` | status |
|----------|----------|--------|
| **mapped** (new) | `L(T_d)` = differential (single dynamic input) | proved sound (and exact) |
| **weldable** (new) | `⋁_j T_j` | proved sound |
| **avalanche** (new) | `Aval(⋁_j T_j, w)` | proved sound (trivially, any `f`) |
| monotonic (CellIFT) | `D` | re-checked sound |
| transportable-add (CellIFT) | `D^{++} ∨ T^{sx}` | re-checked sound |
| transportable-sub (CellIFT) | `D^{+-} ∨ T^{sx}` | re-checked sound |

The three inherited categories (monotonic, transportable, translatable) were
already proved sound at the gate level by CellIFT; the three **new** software
categories (mapped, weldable, avalanche) are what this harness is really for.

## Negative controls (the floor and the gate are load-bearing)

Two deliberately-broken rules must produce an under-taint witness — and do:

- **ADD without the union floor** (`D` alone, dropping `∨ T`): unsound — the
  two polarised replicas can both miss an upward carry. Witness at `w=1`.
- **XOR mis-routed as *mapped*** (differential over one input only, treating the
  other tainted input as a constant): unsound — misses the second operand's
  taint. Witness at `w=1`.

These show why the additive `∨ T` floor and the routing-opcode gate that keeps
two-dynamic-input XOR out of *mapped* are necessary for soundness.

## Run

```sh
uv run --with z3-solver python prove_soundness.py
```

Exit code 0 iff every category is proved sound and both negative controls fire.
The avalanche/MULT check is capped at a small width (a 64-bit multiplier
bit-blasts slowly and avalanche soundness is width-independent anyway).

## Relation to the paper

This is the mechanised counterpart to the algebraic soundness argument
(§ "Taint rule format"). Combined with the empirical evidence — 0 under-taints
on the exhaustive ≤15-bit ground truth plus the single-bit-flip check over the
full corpus — it lets the soundness claim rest on *proof for the rule family*
plus *exhaustive testing of the pipeline*, rather than testing alone.
