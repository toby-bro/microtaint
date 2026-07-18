# Design note: intermediate-taint materialization (segmented circuits)

**Status:** proposal / long-term. Not scheduled against any deadline.
**Author:** design discussion, 2026-07-17.
**Motivation:** the fuzzer-surfaced flag-extract / borrow-chain / conditional-select
under-taints on ARM64/PPC/SPARC (`KNOWN_ISSUES.md` §-1) are *instances* of a single
structural limitation, not four ISA bugs. This note describes the general fix.

---

## 1. The limitation, in the code's own terms

The unit of taint classification today is **one architectural output varnode → its
backward slice → exactly one category**:

- `_cached_generate_static_rule` (`microtaint/sleigh/engine.py`) loops over the
  instruction's written registers/memory (`unique_outputs`).
- For each output `out_vn` it calls `slice_backward(translation.ops, out_vn)`
  (`microtaint/sleigh/slicer.py`), which **flattens every `unique` temporary** back to
  the source registers/memory.
- `determine_category(slice_ops, …)` (`microtaint/sleigh/mapper.py`) then picks **one**
  category for the whole flattened slice via a priority ladder, and one rule is emitted.

Consequences:

1. **Intermediate `unique` values carry no taint of their own.** Taint is materialized
   only on architectural state. The only taint interface *between* instructions is the
   taint state; there is no interface *within* an instruction.

2. **A slice that mixes categories is forced into one.** For x86 `sub`:

   ```
   INT_SUB    t  <- a, b          # arithmetic core     (TRANSPORTABLE)
   INT_EQUAL  ZF <- t, 0          # zero test           (COND_TRANSPORTABLE)
   INT_SLESS  SF <- t, 0 ...      # sign               (routing off MSB)
   INT_SBORROW OF <- a, b         # signed overflow    (joint in a,b)
   ```

   `slice_backward(ZF)` returns `{INT_SUB, INT_EQUAL}` and `determine_category` picks the
   single dominant category. The rule is right for part of the slice and fragile for the
   rest. The same shape is `cmp;cset` (ARM64), `cmpw;mfcr` (PPC), `subs;sbc` (borrow
   chain) — the flag is a function of a *reused arithmetic result*, but the engine
   re-derives it from the source registers under one category.

3. **The arithmetic core is re-evaluated once per flag.** `sub` writes RAX + 6 flags = 7
   outputs; `slice_backward` re-includes `INT_SUB` in all 7 slices, so the runtime
   differential runs the subtraction ~7× per executed `sub`.

The current flag handling papers over (2) with ISA-specific special cases — see the
literal `_X86_FLAG_OFFSETS = {0x200, 0x20B, …}` in `is_mapped_permutation`
(`mapper.py`), and the dedicated builders `_build_signed_overflow_taint`,
`_build_variable_bit_select_taint`. These do not generalize because ARM64 keeps flags in
NZCV, PPC in XER/CR, SPARC in icc, lifted to different p-code.

---

## 2. The proposal

**Materialize taint on the intermediate value and let downstream circuits consume it,**
instead of re-slicing each output all the way back to the source registers.

This is common-subexpression elimination + def-use chaining on the *taint computation*.
The monolithic per-output slice is replaced by a small chain of single-category
**segments** joined by materialized taint masks on intermediate varnodes.

For `sub`, `T(t)` (taint of the subtraction result) is computed **once**, and then:

- `SF = MSB(t)`  → project bit 63 of `T(t)`             (exact)
- `ZF = (t==0)`  → COND_TRANSPORTABLE circuit over `t`  (exact given `T(t)`)
- `PF = parity(t_low)` → routing off `T(t)`             (exact)
- `RAX = t`      → `T(t)`                                (exact)
- `CF`, `OF`, `AF` → **not** functions of `t`; unchanged, keep their exact rules over `{a,b}`

Downstream reuse across instructions (`setcc`/`cset`/`csel`/`mfcr`, next instr reading
`RAX`) already works via the taint state; the win is that *within* the instruction the
result-derived flags get `T(t)` exactly instead of a mixed-category whole-slice rule.

---

## 3. Where to cut — the `partition_slice` algorithm

A new pass sits **between** `slice_backward` and `determine_category`. It does not change
either; it feeds `determine_category` smaller, single-category slices.

### 3.1 Cut predicate

Cut at (materialize taint on) a varnode `v` iff **either**:

- **(C1) heterogeneous fan-out** — `v` is consumed by ≥2 ops whose categories differ, or
- **(C2) result→flag boundary** — `v` is written by an arithmetic/routing op *and* read
  by a condition/flag op (`INT_EQUAL`, `INT_NOTEQUAL`, `INT_SLESS*`, `INT_LESS*`,
  `POPCOUNT`, bit-extract), *and* `v` is also either an architectural output or has
  fan-out ≥2.

Do **not** cut anywhere else. In particular **never cut inside a maximal monotone
arithmetic region** (a connected sub-DAG of `INT_ADD/SUB/2COMP/MULT/AND/OR/NEGATE/shift`
with matching polarity). That whole-region differential is *exact precisely because it
preserves the value-correlations*; cutting it would regress to CellIFT-style per-op union
and destroy the bit-precision that is the paper's contribution. The cut is a scalpel for
the core→flag seam, not a general SSA shredder.

### 3.2 Segment construction (sketch)

```text
def partition_slice(ops, out_vn):
    slice_ops = slice_backward(ops, out_vn)          # unchanged
    cutset    = {v : v satisfies C1 or C2 over slice_ops}   # materialization points
    # A segment is the maximal sub-DAG whose only tainted *inputs* are
    #   (a) architectural sources, or (b) varnodes in cutset,
    # and whose output is either out_vn or a varnode in cutset.
    segments = []
    for boundary in topo_order(cutset ∪ {out_vn}):
        seg_ops = { op reachable-backward from `boundary`,
                    stopping at any varnode in cutset }   # do not cross a cut
        segments.append(Segment(output=boundary, ops=seg_ops))
    return segments   # topologically ordered; last output == out_vn
```

Properties:
- Each `Segment.ops` is a connected sub-DAG with no internal cut varnode, so
  `determine_category(seg.ops)` sees a homogeneous slice — one category, correctly.
- `cutset = {}` reproduces today's behavior exactly (one segment == the whole slice), so
  the change is a strict generalization and can be gated per-instruction.
- The analysis is **cached per unique instruction encoding** (via
  `_cached_generate_static_rule`), so it is paid once per distinct instruction, never per
  execution.

---

## 4. The pseudo-input differential

Today `make_differential()` (`engine.py`) computes `D = f(V|T) XOR f(V&~T)` by running the
*actual* p-code with masked **values**; it never needed per-op taint. To evaluate a
segment whose tainted inputs include a materialized intermediate `r`, the differential
must treat `r` as a **pseudo-input: value known, taint given as a mask**.

### 4.1 Signature

```text
differential(
    seg_ops,                 # the segment's p-code
    entry_values: dict[VarnodeId, int],   # concrete values seeding intermediates,
                                          #   e.g. {t: value(a-b)}; produced once by
                                          #   evaluating the upstream segment's core
    taint:        dict[VarnodeId, int],   # per-input taint masks, now including {t: T(t)}
) -> Expr                                # taint of seg's output
```

The two replicas become `seg(entry_values with r|T(r), others|T_others)` XOR
`seg(entry_values with r&~T(r), others&~T_others)`. The intermediate is just one more
entry in the `(value, taintmask)` input vector; the joint differential over *all* tainted
inputs (both `r` and the segment's other real registers) is sound and, for a monotone
segment with matching polarity, exact.

### 4.2 The one genuinely new capability

The evaluator must **seed an intermediate varnode's value and run only the segment's
ops**, rather than always running the full slice from architectural inputs. Concretely:
the cell/AST evaluator gains an "entry state" that pre-populates the `unique`/register
value map with `value(r)` before executing `seg_ops`. p-code ops are position-independent
given their inputs' values, so this is well-defined. This is the load-bearing change and
the main implementation risk; §3 (partitioning) and §5 (driver) are mechanical.

---

## 5. Driver change

`_cached_generate_static_rule` moves from *"loop over architectural outputs, slice each"*
to *"propagate taint over all defs (arch **and** materialized `unique`) in topological
order"*:

```text
taint_env = { arch input varnodes : their taint state }   # symbolic
for seg in partition_slice(ops, out_vn):        # topo order
    cat  = determine_category(seg.ops, out_width_bits=width(seg.output))
    expr = build_rule(cat, seg, taint_env)      # existing rule builders, per segment
    taint_env[seg.output] = expr                # materialize; downstream segs read it
emit assignment: T(out_vn) <- taint_env[out_vn]
```

`build_rule` is the existing category dispatch (differential / signed-overflow /
variable-bit-select / weldable / …) applied to a segment. The `ast.pyx` expression graph
is already a shared DAG of `Expr` nodes, so `taint_env[seg.output]` is emitted once and
referenced by every downstream segment — the CSE is structural and free at runtime.

---

## 6. Soundness

Taint over-approximation **composes** (monotone substitution): if `T(r) ⊇ true taint of
r` and the downstream segment rule is sound when `r` is treated as an independent tainted
input with mask `T(r)`, the composite is sound. Formally, for `out = g(r, x)` with
`r = h(a,b)`:

- the true taint of `out` w.r.t. `{a,b,x}` is a subset of the taint of `g(r,x)` computed
  with `r` carrying `T(r) = ` (true taint of `h`) — because widening `r` to an
  independent variable with mask `T(r)` can only *add* reachable configurations.

So the cut **never under-taints**. Its only effect is possible **over-taint**: cutting
discards the value-correlation between `r` and `{a,b}` (e.g. that `r`'s sign is a
deterministic function of `a,b`), and feeding them as independent tainted inputs can only
widen. Over-taint is the safe direction. This is why the cut must be minimal (§3.1):
every unnecessary cut is precision thrown away for nothing.

**Where it is exact vs. merely sound:**
- *Result-only flags* (`SF=MSB(t)`, `ZF=(t==0)`, `PF`, and all downstream reuse):
  functions of `t` alone → cutting at `t` is **exact** given `T(t)`. This is the family
  the fuzzer flagged.
- *Joint flags* (`CF`, `OF`, `AF`): **not** functions of `t`; the cut does not apply, and
  they keep their existing exact rules (`_build_signed_overflow_taint`, borrow chains).

---

## 7. Performance

Expected **net win**, contrary to the naive "more passes = slower" worry:

- Today `sub` runs ~7 differentials, each re-including `INT_SUB`. Materializing `T(t)`
  once and projecting SF/ZF/PF collapses the duplicated subtraction to a single shared
  DAG node. Fewer, smaller runtime evaluations.
- Partitioning (§3) is analysis, cached per encoding → one-time, not per-execution.
- Larger cached AST by one shared node per cut → negligible.

The only way this regresses runtime is if segmentation forced *more* evaluator passes;
the shared-`Expr` DAG (§5) ensures the materialized taint is computed once and referenced,
not recomputed per consumer.

---

## 8. Rollout

1. Implement §4.2 (evaluator entry-state / seeded intermediate) — the hard part. Unit-test
   it against the full-slice evaluator on a monotone core (they must agree when `cutset`
   is a single interior varnode).
2. Implement `partition_slice` (§3) with `cutset = {}` as the default → byte-for-byte
   identical output to today. Land behind a flag; run the whole x86 differential suite
   and the cross-ISA oracle to confirm zero diff.
3. Turn on C2 (result→flag) cuts for x86 `sub`/`cmp` only, where exact ground truth
   exists (2^k enumeration). Verify `T(t)` projection == exact GT for SF/ZF/PF, and that
   `CF/OF/AF` are byte-identical (untouched).
4. Extend to ARM64 `subs`/`cmp`+`cset`/`csel`, PPC `cmpw`+`mfcr`, SPARC `subcc`. Because
   flags are now `t`-derived circuits like everyone else's, the `_X86_FLAG_OFFSETS`
   special-casing in `is_mapped_permutation` should become deletable — that deletion is
   the acceptance test that the generalization actually happened.
5. Re-run `benchmark/generalization/multiarch_fuzz.py`; the §-1 under-taints should close
   without per-ISA flag hacks.

## 9. Risks / open questions

- **Evaluator seeding (§4.2)** is the real work; everything else is dataflow plumbing.
- **Cut predicate tuning**: C1/C2 must be tight. A too-aggressive cutset silently
  over-taints (sound but imprecise) and would show up as regressions in the exact-match
  rate, not as test failures — so precision regression tests (exact-GT match %) gate this,
  not just soundness tests.
- **Memory/pointer intermediates**: a materialized `r` that is a pointer (feeds LOAD/STORE
  addresses) interacts with `resolve_ptr_with_offset`; the cut must not sever an address
  computation the pointer resolver still needs to walk. Start by excluding address-flowing
  varnodes from `cutset`.
- **Interaction with polarity** (`microtaint/sleigh/polarity.py`): polarity is propagated
  backward over the whole slice today. Per-segment polarity must be seeded at each cut
  boundary from the materialized value's polarity label; verify this composes.

---

## 10. Bring-up findings (M1–M4 groundwork)

Empirical results from building the primitives (M1–M3 landed; M4 in progress).

### 10.1 The 2-corner segment differential is UNSOUND for flags — floors are essential
Materializing a cut result as its two differential corners `(t_a, t_b)` and seeding a
downstream flag's segment differential from them does **not** suffice. Measured on x86
`cmp` against exact 2^k ground truth: the raw 2-corner differential under-taints SF/ZF in
6–8 / 8000 cases with the *polarity-correct* `D^{+-}` corners (8 with `D^{++}`). Reason:
SF/ZF are **non-monotone** in the result, so any 2-point evaluation misses interior
toggles. The engine is sound today only because each flag category adds a **floor**
(`FullMaskAvalanche` / pairwise `Avalanche` / the exact `SignedOverflow` / `VariableBitSelect`
rules). Those floors live inside `generate_taint_assignments`.

**Consequence for M4:** the downstream segment rule cannot be a bolt-on Expr. It must reuse
the engine's per-category rule machinery (differential **plus** the category floor). So M4
is a *targeted extension* of `generate_taint_assignments`, not a parallel rule generator —
reimplementing the floors would both duplicate ~1000 lines and risk divergence.

### 10.2 Each downstream category needs different info from the intermediate
- Monotone / transportable downstream: the polarised **corners** `(t_a, t_b)` (a 2-replica
  seeded differential). `(value, taint)` alone is insufficient — it cannot reconstruct the
  polarised corners a mixed-polarity core like `a−b` needs (the borrow-polarity issue,
  KNOWN_ISSUES #2).
- Condition / avalanche downstream: `(value(t), T(t))` fed to the masked-single-replica /
  avalanche rule.
So the intermediate should materialise enough to serve both: the two corners `t_a`, `t_b`
(from which `T(t) = t_a ^ t_b`, and `value` on untainted bits) cover every case.

### 10.3 The ARM64/PPC bug IS intra-instruction
`cmp`/`subs` on every ISA writes each flag as its own architectural output whose slice
mixes the subtraction with a flag-extract — the exact x86 `cmp` shape. x86 is sound only
via ISA-specific tuning (`_X86_FLAG_OFFSETS`, the sign split); ARM64's NZCV / PPC's CR lift
to different p-code the tuned rules miss, so they under-taint. Materialisation fixes this by
making the flag a result-derived circuit classified by ordinary dataflow — ISA-independent.

### 10.4 M4 shape (targeted extensions, all behind `MICROTAINT_SEGMENTED`)
1. **Intermediate core**: reuse `make_differential` unchanged, targeting a `UNIQ_<off>`
   output. *Enabled now*: the Cython concrete evaluator reads UNIQ outputs
   (`evaluate_concrete`/`_state`/`_differential` route through `_read_output_any`), so the
   existing polarity-aware differential materialises `T(t)` with no new rule code.
2. **`extract_dependencies`**: return each cut `UNIQ_<off>` as a leaf dep (a `UniqMapping`
   with `name='UNIQ_<off>'`), so the existing floors — which iterate deps — apply to it.
3. **`make_differential` / `InstructionCellExpr`**: seed cut-UNIQ deps (from the
   materialised corners) and run from `start_pc`, via the M1 primitive.
4. **`generate_output_target`**: accept a UNIQ output target for intermediate segments.
5. **Driver**: `partition_slice(CONSERVATIVE)` per output; emit intermediate assignments
   then downstream, assemble the two-phase `LogicCircuit` (M2). `compute_polarity` per
   segment (each segment keeps its own terminal, so it composes — agent-verified).

Validation gate before flipping the default: `NONE` policy byte-identical to today across
the x86 corpus; `CONSERVATIVE` on `sub`/`cmp` matches exact-GT for SF/ZF/PF and is
byte-identical for CF/OF/AF (operand-derived, not cut).
