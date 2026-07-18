# Native big-endian support in the Cython p-code kernel

**Status:** kernel done + validated; routing relaxation pending (2026-07-18).

- The first provably-safe routing slice landed in `fix(sim): sound PPC
  carry/borrow-chain taint via native-BE-safe routing` (`_native_be_safe` routes
  *maximal-register* BE instructions to the native Cython kernel).
- The Cython register file and memory byte math is now **endianness-aware**
  (`_PCodeFrame._is_big_endian`; changes 1-5 below).  Validated: the native kernel
  agrees byte-for-byte with Unicorn on BE sub-register instructions
  (`tests/test_native_be_kernel.py`; PPC differential 0/840, MIPS differential
  0/1040 on canonical inputs, sub-register concrete 0/500), and the little-endian
  path is byte-identical (`tests/test_cell_benchmark.py` unchanged).
- **Not yet done:** relaxing `_native_be_safe` to route sub-register / memory BE
  instructions to the (now-correct) native kernel, and removing the
  `use_unicorn=True` force.  Blocked on the canonical-input caveat below; until
  then the BE byte math is correct but reachable only through the maximal-register
  routing (which never exercises a sub-register byte offset), i.e. validated
  infrastructure ahead of its routing.

## Why

`CellSimulator` forces `use_unicorn=True` for any `*BE` arch (MIPS64BE, PPC32BE,
SPARC32BE) and leaves the native evaluator disabled, because the native register
file is byte-offset indexed assuming little-endian (KNOWN_ISSUES §0).  The BE
engine therefore rests on Unicorn alone — no independent second implementation to
cross-check (a weaker soundness guarantee) and no native fast path.

## Scope: only `cell.pyx` needs to change

BE instructions route to the **Cython** `PCodeCellEvaluator` (`CellSimulator._native_be`),
never to the C kernel `PCodeCellEvaluatorC` (which is selected only for LE via
`use_c`).  So the endianness-aware byte math is needed **only in `cell.pyx`**; the
C kernel (`cell_c/cell_core.h`, `cell_c.c`) stays LE-only and untouched.  This
halves the surface the exploration mapped.

SLEIGH already emits BE-correct varnode `(space, offset, size)` (genuine
`PowerPC:BE:32` / `MIPS:BE:64` / `sparc:BE:32` specs) — confirmed by
`engine.py::_sub_reg_bit_start`.  So the p-code *arithmetic* ops
(`INT_*`, `SUBPIECE`, `PIECE`, `ZEXT`, `SEXT`, shifts) are already
endianness-independent: they work on integer values.  The **only** LE assumptions
are (a) how a sub-register byte offset maps to a bit position in the register
file, and (b) memory byte assembly.

## Register-file model

`_PCodeFrame` stores each register as its integer value in `regs_arr[off]` (a flat
`uint64_t[]` indexed by SLEIGH byte offset), with `regs_sz[off]` its width.  Byte
`d` of an N-byte register currently sits at bit `d*8` — pure little-endian.

For a **big-endian** register based at file offset `k` with width `P = regs_sz[k]`,
byte at file offset `k` is the MOST significant.  A sub-access `(off, sz)` with
`k <= off` and `off+sz <= k+P`, at `d = off-k` bytes from the MSB, has value:

```
LE:  (regs_arr[k] >> (d*8))            & mask(sz)
BE:  (regs_arr[k] >> ((P - d - sz)*8)) & mask(sz)
```

The BE shift counts from the high end and therefore needs both the parent width
`P` and the access size `sz` (LE needs neither).

## Exact changes in `cell.pyx` (all gated on `frame._is_big_endian`; LE path byte-identical)

Thread endianness into the frame: add `cdef bint _is_big_endian` to `_PCodeFrame`,
set it in `_load`/`_load_state` from the evaluator's arch
(`str(arch).upper().endswith('BE')`, computed once in `PCodeCellEvaluator.__init__`).

1. **`_read_reg` Step 1 — parent extraction (line ~622).**
   `base = regs_arr[k] >> (byte_off*8)` → BE: `>> ((regs_sz[k] - byte_off - sz)*8)`.

2. **`_read_reg` Step 2 — sub-write merge (lines ~651-658).**
   A written sub-slot `k'` (with `byte_off' = k'-off`, width `k_sz'`) overlaid into a
   read of size `sz` at `off`:
   `shift = byte_off'*8` → BE: `shift = (sz - byte_off' - k_sz')*8`;
   `lane_mask = sub_mask << shift`, `base = (base & ~lane_mask) | (sub_val << shift)`.

3. **`_read_reg` cold dict path (lines ~675-677).** Same inversion as Step 1.

4. **`_write_reg` same-offset narrower overlay (lines ~575-584).**
   A narrower write sharing the base offset lands on the HIGH bytes under BE:
   `hi_shift = (regs_sz[off] - sz)*8`; overlay `masked` at `<< hi_shift` instead of
   the low `sz` bytes.  (Rare on BE — sub-registers usually arrive at a different
   offset — but required for correctness.)

5. **`_write_mem` / `_read_mem` (lines ~683-693).**
   Byte `i` of a `size`-byte access is the `i`-th from the MSB under BE:
   `mem[addr+i] = (val >> ((size-1-i)*8)) & 0xFF` and
   `result |= mem[addr+i] << ((size-1-i)*8)`.

The 128-bit wide-op low/high split (`uniq_hi`) is x86 SSE/CQO only — never a BE
target — so it stays LE and needs no change.  The `_read_output` / `_read_uniq`
`>> bit_start` masking already consumes BE-correct bit positions from
`StateMapper`, so it is correct once the register file (1-4) is fixed.

## Routing relaxation (the remaining step) and the canonical-input caveat

Once the Cython kernel is BE-correct, `_native_be_safe` can in principle drop the
*maximal* restriction (sub-register windows become valid) and, with BE-aware
memory, the LOAD/STORE exclusion too, then remove the `use_unicorn=True` force at
`simulator.py:310-311` — making the native kernel the BE concrete engine and
restoring the two-implementation soundness cross-check + native fast path.

**Caveat found during bring-up (why this is not yet done):** a sub-register op
places a *precondition* on the parent register that the native kernel and Unicorn
resolve identically only for *canonical* inputs.  MIPS64 32-bit ops (`sll`, `sra`,
`addu`, …) require the 64-bit GPR to hold a sign-extended 32-bit value; on a
non-canonical value (arbitrary high 32 bits) the result is architecturally
undefined and native vs Unicorn legitimately diverge (verified: random 64-bit
inputs give 80/1040 differential mismatches; canonicalised inputs give 0/1040).
The cross-ISA fuzzer seeds registers with *random* (non-canonical) values and
derives its taint ground truth from Unicorn, so routing MIPS 32-bit ops to the
native kernel could make microtaint's differential disagree with that ground truth
— a potential under-taint.  So relaxing the routing must either (a) be restricted
to arches with no aliased-parent precondition (PPC32 / SPARC32 32-bit GPRs — safe,
validated 0/840), or (b) canonicalise sub-32 register inputs before native
evaluation on 64-bit-alias arches.  This is the crux of the remaining work and is
**not** worth rushing: the under-taint payoff is nil (see below), so correctness of
the routing decision dominates.

## Validation plan (must pass before relaxing routing or the use_unicorn force)

1. **LE non-regression (highest priority):** the full pytest suite
   (`tests/test_cell_benchmark.py` native-vs-Unicorn parity + all others) must be
   byte-identical — the BE branches are gated, so LE must be untouched.  amd64
   fuzzer stays 0 under-taints.
2. **BE correctness:** for each BE arch, run the native Cython
   `evaluate_concrete` / `evaluate_differential` against Unicorn over the
   `multiarch_oracle` program corpus **and** random states, asserting equality —
   including sub-register (`extsb`, MIPS 32-bit halves) and memory (`lwz`/`stw`)
   instructions.  Any mismatch is a byte-math bug.
3. **Under-taint payoff is secondary.** Honest expectation: this feature is mostly
   a *perf + soundness-argument* win, not an under-taint reduction.  The remaining
   BE under-taints (`sra` sign-extension, `slt`/`sltu` / `cmpw;mfcr` comparisons,
   dense `subfc;subfe`) are **2-corner differential-precision** limits that are
   backend-independent (native == Unicorn), so a BE-correct native kernel does not
   reduce them — those belong to the flag-floor / materialization work (M6).

## Risk

The register file is the hottest, most subtle code in the engine and is
LE-critical for every architecture.  The merge-on-read logic (Step 2) is
particularly delicate.  Every BE branch MUST be gated so the LE path is provably
unchanged, and the change must not land without the full LE suite passing
byte-identically.  Because the under-taint payoff is low, there is no reason to
rush it: correctness of the LE path dominates.
