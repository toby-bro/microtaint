# Held fix: cross-instruction intermediate-register threading (MIPS HI/LO)

## What it fixes
Multi-instruction sequences are split into per-instruction sub-circuits whose
taint is threaded through the caller's `state_format` **plus x86 flags only**
(the former `_X86_FLAG_REGISTERS` list).  Any other register carrying taint
between steps is silently dropped -- notably MIPS **HI/LO** after `mult;mflo` /
`div;mfhi`, which zeroed every multiply/divide result read through mflo/mfhi
(~800 under-tainted bits per form in the July 23 campaign).

The fix (`_cross_instruction_intermediate_regs` + the wiring in
`generate_static_rule`) threads *every* register written by one sub-instruction
and read by a later one, generalising across ISAs and dissolving the x86-only
hack.  It is soundness-validated: all 7 HI/LO forms go from ~800 under-taints to
0, x86 carry chains stay sound, and 30 regression tests (incl. the 4 MIPS cases)
pass individually.

## Why it is held
With this change the **full pytest suite segfaults non-deterministically**
(RC=139, crash point varies 10%..99%) in the native compiled-circuit path:

- baseline (committed): 2/2 clean full runs;
- 4 floor fixes only, no 1a: 2/2 clean;
- + 1a: 3+ segfaults.

`MICROTAINT_DISABLE_COMPILED_CIRCUIT=1` makes it complete, so the corruption is in
`circuit_c` evaluating the ChainedCircuit built for a split sequence once extra
intermediate registers are threaded into the sub-circuits' state_format.  It is
NOT the 8-bit-flag width issue (excluding byte-flags via the `size >= 2` filter in
the patch did not stop the crash), and single-sequence stress (mult;mflo,
mul;mov, add;adc at thousands of evals) does not reproduce it -- pointing to a
latent refcount / buffer bug in the C-VM's chained-circuit handling that the extra
registers expose.

## Next step
Root-cause the C-VM corruption (valgrind / ASAN on the pytest run with the patch
applied; inspect refcounting in `emit_call_cell` / `ChainedCircuit` threading in
`circuit_c.c`), or gate the compiled circuit off for chained circuits that carry
scanned intermediates (Python-eval fallback, small perf cost).  Then re-apply
`1a-mips-hilo-threading.patch`, restore the 4 MIPS cases in
`tests/test_unsoundness_fixes_20260723b.py`, and confirm the full suite is stable.
