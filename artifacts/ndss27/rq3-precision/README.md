# RQ3 — Bit-level precision

**Claim (§6.3).** Among the engines that are sound, microtaint reaches the
highest bit-level precision — it marks fewest bits tainted while never missing
one that ground truth says moves.

## Run

This shares RQ2's run; there is no separate command. `benchmark.py` scores
soundness and precision in the same pass, over the same 3,263 enumerated cases.

If you have already run RQ2, read its report. If not, run RQ2 first.

## What to look at

Precision is only meaningful **conditional on soundness**: an engine that taints
nothing is maximally "precise" and useless. The table therefore reports, per
engine, unsound cases *and* over-tainted bits, and the claim is about the second
among engines with zero of the first.

Expect the byte-granular engines to be sound and heavily over-tainting (they
promote a tainted bit to its whole byte, by construction), and the symbolic
engines to be precise where they answer at all.

## Where microtaint is deliberately imprecise

Two places, both sound and both documented in the paper:

- **Opaque operations.** Where p-code does not model an instruction (a
  `CALLOTHER` — AES rounds, some AVX forms), the rule is the avalanche floor:
  every output bit tainted if any input bit is. Measured on the AVX forms, one
  tainted source bit taints 513 output bits, against 1-6 for the SSE encodings
  of the same operations, because Ghidra models the SSE forms and not the VEX
  ones.
- **Reconvergence.** Where an instruction's outputs share intermediates, the
  per-output differential can over-approximate; §5 gives the window that bounds
  this.

## Tolerance

Exact, for the same reason as RQ2.
