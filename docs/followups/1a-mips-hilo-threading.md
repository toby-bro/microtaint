# RESOLVED: cross-instruction intermediate-register threading (MIPS HI/LO)

**Status: closed 2026-09-11. The defect no longer exists and the held patch is
obsolete. Kept as a record rather than deleted, because the way it closed is
worth knowing.**

## What it was

Multi-instruction sequences were split into per-instruction sub-circuits whose
taint was threaded through the caller's `state_format` **plus x86 flags only**.
Any other register carrying taint between steps was silently dropped, notably
MIPS **HI/LO** after `mult;mflo` / `div;mfhi`: a product lands in a register
that neither instruction names in its operands, so every multiply and divide
result read back through `mflo`/`mfhi` came back clean. About 800 under-tainted
bits per form in the July 23 2026 campaign.

## Why it was held

A fix (`_cross_instruction_intermediate_regs`, threading every register written
by one sub-instruction and read by a later one) was written and validated, but
with it applied the full suite segfaulted non-deterministically in `circuit_c`
evaluating a ChainedCircuit. `MICROTAINT_DISABLE_COMPILED_CIRCUIT=1` made it
complete, so it was a latent C-VM bug the extra threaded registers exposed.
Rather than ship a crash, the patch was parked here.

## How it actually closed

Not by applying the patch. Re-measured on 2026-09-11 against Unicorn brute-force
ground truth, on **both** the circuit path and the taint-IR path: `mult;mflo`,
`mult;mfhi`, `multu;mflo`, `div;mflo` and `div;mfhi` all come back
**over-tainted, never under-tainted**. The general work done since (carry
threading, the per-op model, the rebuilt evaluator) subsumed the special case,
and the x86-only list it was working around is gone.

Checked for vacuity before believing it: with no tainted input the same
sequences come back clean, so the engine is propagating rather than blanket
tainting.

`1a-mips-hilo-threading.patch` is left beside this file for archaeology. **Do
not apply it.** It is written against a `generate_static_rule` that no longer
has that shape, and the behaviour it adds is already present.

## What replaced it

`tests/test_unsoundness_fixes_20260723b.py` now carries the five HI/LO
sequences as ordinary regression cases, against the exhaustive 2^k ground truth
the rest of that module uses.

They had to be written by hand: **the instruction bank holds no
multi-instruction HI/LO sequence at all**, which is exactly why the defect went
unnoticed for as long as it did, and why a campaign over single instructions
would never have found it.

That file also gained a vacuity gate, because the first version of the
`div;mflo` case used operands whose quotient is zero however the tainted bits
move. It passed, and proved nothing. The gate requires each case's ground truth
to carry taint on some register the case did not itself seed, so a case that
would agree with an engine propagating nothing now fails instead.
