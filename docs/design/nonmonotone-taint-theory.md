# From floors to theorems: a soundness/precision theory for the non-monotone cases

Status: design / theory. Goal: replace the four ad-hoc avalanche *floors* added on
`soundness-fixes` (commits 7c5d17b, 3c79783, 30d3f68, 52b53d2) with mechanisms that are
**as sound but strictly more precise**, and that are *general* (triggered by the p-code
structure, not by an ISA/output-width signature). Every claim here is grounded in the
current code and in measured polarities (see §3).

---

## 1. The single formal object

For one architectural output of one instruction, MicroTaint computes taint with a
**2-corner differential**. Let `f` be the concrete semantics of that output (a function of
the input state), `V` the concrete input values, `T` the input taint mask. The engine
emits, per output bit `j`:

```
D(V,T)_j  =  f(replica_+)_j  XOR  f(replica_-)_j
```

where the two replicas are built by `build_polarized_reg` (engine.py:2205):

```
replica_+ :  positive-polarity bits -> V|T (max),   negative-polarity bits -> V&~T (min)
replica_- :  positive-polarity bits -> V&~T (min),  negative-polarity bits -> V|T (max)
```

Polarity per operand comes from `compute_polarity` (polarity.py:8), refined for the sign
bit of signed comparisons by `apply_sless_msb_split` (engine.py:392).

**Truth.** Fix `(V,T)`. Restrict `f` to the *tainted sub-cube* `{x : x&~T = V&~T}`; call the
restriction `g : {0,1}^k -> {0,1}^m`, `k = popcount(T)`. The *true taint* of output bit `j`
is

```
Tr_j = 1   iff   g_j is non-constant on the cube   (some tainted input flips output bit j).
```

**Two facts that frame everything.**

* **Soundness one-liner.** `D_j = 1` exhibits two cube points with different bit `j`, so
  `D_j <= Tr_j` *always* — the differential never over-taints. The danger is only the other
  direction.
* **Exactness one-liner.** `D_j = Tr_j` **iff** the two replicas bracket the sensitive
  variation of `g_j`, i.e. iff `g_j` is monotone *in the polarity-signed literals*. If
  `g_j` is non-monotone in the chosen orientation, both replicas can land on the same value
  while an interior point differs — `D_j = 0 < Tr_j = 1`: **under-taint = unsound.**

> **Theorem (why the floors exist).** The differential is exact exactly when
> `compute_polarity` orients every operand along the monotone direction of `f`. Every
> under-taint we patched is a slice where no single global orientation does that. So the
> question is never "2 corners are too few"; it is "were the *right* 2 corners chosen?"

The floors restore soundness by `OR`-ing an avalanche term onto `D`. That is sound
(`D OR floor >= Tr`) but blunt (`>= Tr`, often `> Tr`): it re-introduces the byte-level
over-taint the paper claims to beat. We want the tight value in `[Tr, floor]` — ideally
`Tr` itself.

---

## 2. Two exact mechanisms already in the codebase

We do **not** need to invent machinery. Two exact, precise mechanisms already exist; the
floors are just the cases that fell through both.

### Mechanism A — polarity-aware corner selection (`compute_polarity` + `build_polarized_reg`)

Choosing each operand's corner by its monotone direction. This is *free* (the differential
already runs twice) and — crucially — **keeps input correlation**, because it evaluates the
real `f`: `sub rax,rax` and `xor r,r` cancel correctly, `a - a = 0` taints nothing. It is
exact for every slice that is monotone after correct orientation: add/sub chains, carries,
borrows, single-direction comparisons, mixed-polarity boolean conditions. Its one
limitation is structural: `compute_polarity` returns **one** polarity per varnode, so a
varnode that must be oriented *both* ways in the same slice cannot be served (see `cmpw`,
§3).

### Mechanism B — exact closed-form per-op taint terms, Z3-proved

For the residue where Mechanism A cannot orient consistently, the codebase already ships
**exact** replacement terms instead of floors:

* `SignedOverflowTaintExpr` — `_build_signed_overflow_taint` (engine.py:837); proof
  `benchmark/soundness/prove_signed_overflow.py` (identity + no-under-taint for w=2..64,
  no-over-taint for w<=6).
* `VariableBitSelectExpr` — `_build_variable_bit_select_taint`; proof
  `benchmark/soundness/prove_variable_bit_select.py`.

These are the template. Each is a small `Expr` computing the *exact* sensitivity of one
non-monotone primitive from `(V,T)` in closed form, gated to fire only when the slice is
exactly that primitive, and machine-checked in Z3. **This is what "real good" means here:
a proved theorem, not a fuzz-survived patch.**

The rest of this document shows every floor is either Mechanism A (a polarity bug) or a
missing Mechanism-B term, derives the term, and states its proof obligation.

---

## 3. The four floors, reclassified by *measured* polarity

Liftings via `sleigh-translate.py`; polarities via `compute_polarity` on the real slice.

### 3a. `cset w,hi` = `ZEXT(BOOL_AND(CY, BOOL_NEGATE(ZR)))` — floor 7c5d17b is redundant

Measured polarity: **`CY +1`, `ZR -0`** (opposite corners; `ZR` was flipped by the
`BOOL_NEGATE` rule from eb34323). Then

```
replica_+ : CY@max & ~(ZR@min) = 1 & ~0 = 1
replica_- : CY@min & ~(ZR@max) = 0 & ~1 = 0     ->  D = 1   (exact)
```

The differential is **already exact** for `hi`. Floor 7c5d17b (`OR` FullMask-avalanche of
each flag) now only *adds* over-taint: with `ZR=1` concrete and `CY` tainted, true taint is
0 (result is `CY & 0 = 0`), the differential correctly gives 0, but the floor still raises
bit 0. **Predicted precision win: delete the floor for the mixed-polarity-AND conditions.**

*Caveat that keeps us honest:* `cset` also has equality-type conditions. For 1-bit flags
these degenerate to XOR (`ne` = `N XOR V`, taint `= T_N | T_V`, linear/exact), so no floor
is needed there either — but this must be verified per condition, not assumed (§5, equality).

### 3b. `slt v0,a0,a1` = `ZEXT(INT_SLESS(a0,a1))` — floor 3c79783 is a polarity bug

Measured polarity: **`a0 +1`, `a1 +1`** — *same corner*. Both replicas evaluate the
comparison with both operands saturated the same way, landing in the equal regime, `D = 0`.
This is not "too few corners"; it is the **wrong** two corners.

*Correct orientation.* `a < b  <=>  a - b` borrows: **antitone in `a`, monotone in `b`**
(signed: in the sign-adjusted representation `apply_sless_msb_split` already provides).
So comparison ops must orient their LHS *opposite* to their RHS — the mirror of `INT_SUB`,
which already flips its RHS. With `a0 -> -1`, `a1 -> +1`:

```
replica_+ : [ a0@min  <  a1@max ]   = "can a<b be true?"
replica_- : [ a0@max  <  a1@min ]   = "is a<b always true?"
D = replica_+ XOR replica_-         = exact taint of the comparison
```

**Fix = one rule in `compute_polarity`:** `INT_LESS / INT_SLESS / INT_LESSEQUAL /
INT_SLESSEQUAL` invert LHS polarity, propagate RHS (dual of the existing `INT_SUB`
branch). This is exact and *correlation-safe* (Mechanism A). It dissolves floor 3c79783
and is why x86 `cmp` was never affected: x86 lifts comparisons *through* `INT_SUB` /
`INT_SBORROW`, which already set opposite polarities; MIPS `slt` emits a **bare**
`INT_SLESS` with no subtraction, so it never got them.

### 3c. `subfe` — already Mechanism A, keep as the template

Measured polarity: **`xer_ca +1`** through the double `BOOL_NEGATE` (`... - (r4 + !carry)
= ... - r4 - 1 + carry`). eb34323 is a pure, exact polarity correction — no floor. This is
the shape every case should converge to.

### 3d. `cmpw` — the genuinely irreducible conflict; needs Mechanism B

Lifting packs two comparisons into CR0:

```
LT bit :  INT_SLESS(r4, r5)      -> wants r4 antitone (-1), r5 monotone (+1)
GT bit :  INT_SLESS(r5, r4)      -> wants r5 antitone (-1), r4 monotone (+1)
```

`r4` must be `-1` (for LT) **and** `+1` (for GT) in the *same* slice. Measured: r4 is `+1`
on both paths (last-write-wins in the single-polarity map). No global orientation and no
single pair of replicas can serve both — this is the structural wall of Mechanism A.

**Resolution = Mechanism B: an exact per-comparison term** (below), OR'd per output bit.
Because each comparison computes its own `min/max` pairing, LT and GT are handled
independently with no shared orientation. This replaces floor 52b53d2's blanket
`0xC000_0000` avalanche of the LT/GT field with the *exact* LT and GT taint bits.

### 3e. `csel x0,x1,x2,hi` — `CBRANCH`, a value-select; needs Mechanism B

Lifting is a control branch (`CBRANCH`) selecting `x1` vs `x2`. This is **not** a corner
bug: when the condition is tainted the output flips between two *values*, so the sensitive
quantity is the value difference `x1 XOR x2`, not any operand taint. Floor 30d3f68 grafted
`(x1 XOR x2)` onto the cmov passthrough — right idea, but coarse (it also unions both
operands' taint regardless of which branch a concrete condition takes). The exact term
(below) selects correctly.

---

## 4. Summary map

| floor | instr | root cause | correct mechanism | result |
|---|---|---|---|---|
| 7c5d17b | `cset hi` | none — polarity already right (eb34323) | A: delete floor | exact |
| 3c79783 | `slt`/`sltu` | comparison LHS not inverted | A: comparison-polarity rule | exact |
| 52b53d2 | `cmpw;mfcr` | one varnode, conflicting orientations | B: `ComparisonTaintExpr` | exact per bit |
| 30d3f68 | `csel` | control-dependent value select | B: `SelectTaintExpr` | exact |
| (eb34323) | `subfe`/`sbc` | `BOOL_NEGATE` polarity | A (done) | exact |

Two floors are *polarity bugs* (Mechanism A, free, correlation-safe). Two are *missing
exact terms* (Mechanism B, Z3-proved, cheap). **None of the four requires an avalanche.**

---

## 5. The exact closed forms (Mechanism B) and their proof obligations

Notation: `min(x) = V_x & ~T_x`, `max(x) = V_x | T_x` (signed variants add the sign bias
`2^{w-1}`, i.e. compare in the sign-flipped domain, matching `apply_sless_msb_split`).

### `ComparisonTaintExpr` — `[a < b]` (one bit)

```
Tr = [ min(a) < max(b) ]  XOR  [ max(a) < min(b) ]
   = "can be true"  XOR  "is always true"
```

Exact: equals 1 iff the predicate is non-constant on the cube. Cost: two width-`w`
compares. **Correlation caveat:** loose when `a` and `b` share tainted symbolic bits
(e.g. `[a < a]` -> `min<max` = 1 but `Tr = 0`). Gate the term on *operand independence*
(distinct varnode ids / disjoint tainted-bit provenance); when they are correlated the
whole-`f` differential is already exact, so fall back to it. This gate is the same def-use
information `_slice_has_constant_dominator` already computes.

### `EqualityTaintExpr` — `[a == b]` (one bit)

Equality is the one comparison with *no* monotone orientation (symmetric, non-monotone
both ways), so it can never be a Mechanism-A case:

```
equal-achievable   :  (a ^ b) & ~(T_a | T_b) == 0     # all *fixed* bits already agree
unequal-achievable :  (T_a | T_b) != 0                # some bit is free to break equality
Tr = equal-achievable AND unequal-achievable
```

Exact and cheap. (For 1-bit operands this reduces to `T_a | T_b` = XOR taint, consistent
with §3a.)

### `SelectTaintExpr` — `c ? x : y` (width `w`)

```
c tainted  (T_c != 0) :  Tr = (x ^ y) | T_x | T_y          # both branches reachable
c clean               :  Tr = c ? T_x : T_y                # only the taken branch
```

Exact; generalises uniformly to `csel`/`cmov`/`csinc`/… and dissolves the ISA-specific
passthrough special-casing.

### Proof obligation (per term)

Mirror `prove_signed_overflow.py`: in Z3 prove, over symbolic `(V,T)` and width `w`,
1. **identity** — term evaluates to the differential where the differential is exact;
2. **no under-taint** — `term >= Tr` for all `w` (soundness), by relating `term` to the
   existential "two cube points differ";
3. **no over-taint** — `term <= Tr` for small `w` (exactness), by enumeration.
Only a term passing (1)+(2) may replace a floor; (3) is the precision guarantee.

---

## 6. The irreducible boundary (where avalanche is *proved* necessary, not guessed)

After A+B, the only remaining sound-but-imprecise cases are the ones with no cheap exact
term:

* `OPAQUE_OPCODES` (aes/pshufb/CALLOTHER…): non-affine, no closed form — AVALANCHE stays.
* Comparisons/equality of **non-affine** functions (e.g. `popcount(x) < k`, CRC into a
  flag): the operand of the comparison is itself avalanched upstream; the comparison term's
  independence gate fails; avalanche is correct.
* Genuinely correlated multi-op reconvergence the differential cannot see and no per-op
  term covers.

The point: this boundary becomes a *characterised, documented* set — we avalanche because
we can show no exact term is cheap, not because a fuzzer happened to find nothing.

---

## 7. Execution plan (Z3-first, each step measured)

Invariant every step: `ruff`/`mypy` clean; full `pytest`; x86 differential **digest byte-
identical** (no x86 behaviour change); cross-ISA fuzzer under-taints stay 0; record exact%
delta (the precision we are buying back).

1. **Comparison polarity (Mechanism A).** Add the LHS-inverting rule for
   `INT_LESS/SLESS/LESSEQUAL/SLESSEQUAL` to `compute_polarity`; delete floors 3c79783 and
   7c5d17b. *Predict:* under-taints stay 0, exact% up on MIPS `slt`/`sltu` and ARM64
   `cset`. Cheapest, highest-value, correlation-safe. Verify the `cset` equality
   sub-conditions are exact (XOR) before removing 7c5d17b.
2. **`ComparisonTaintExpr` + `EqualityTaintExpr` (Mechanism B).** Prove in Z3, wire with
   the independence gate, delete floor 52b53d2. *Predict:* `cmpw;mfcr` LT/GT bits exact.
3. **`SelectTaintExpr` (Mechanism B).** Prove in Z3, replace floor 30d3f68. *Predict:*
   `csel`/cmov value-select exact; ISA passthrough special-cases shrink.
4. **Document the irreducible boundary (§6)** in `KNOWN_ISSUES.md`; it is the honest
   statement of the avalanche limitation the paper already claims.

Net: four floors -> two free polarity rules + two Z3-proved exact terms + a characterised
avalanche boundary. Same soundness, strictly better precision, no ISA/width-specific gates.
