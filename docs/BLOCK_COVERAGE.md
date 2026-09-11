# What block mode can and cannot lower

Block mode computes the taint of a whole basic block as one compiled program.
A block the lowering **refuses** is not analysed at all: the C hook counts it
and returns, so nothing computes its taint and a tainted value that moves
through one comes out clean. That is an under-taint, so the refusal set is a
soundness property, not a coverage statistic.

This file records what the refusals were, how each was removed, and what is
still declined and why. Measurements are on a **static-glibc x86-64 binary**
(`fgets`, `strlen`, `malloc`, `memcpy`, a byte loop, `snprintf`), because the
hand-written `-nostdlib` guests in `benchmark/taint_density/` refuse nothing at
all and hid every one of these.

The gate is `tests/test_block_skips_nothing.py`, which asserts
`hook_stats()['unhandled'] == 0` on that binary and separately asserts the
trace is big enough to mean something.

## Where it started

| | distinct blocks refused | block executions skipped |
|---|---|---|
| before | 49 of 381 (12.9%) | 71 of 642 (11.1%) |
| now | 0 | 0 |

## The causes, and what each needed

**Thread-local storage — 25 blocks.** SLEIGH names the x86-64 segment bases
`FS_OFFSET` / `GS_OFFSET`; Unicorn names them `FS_BASE` / `GS_BASE`; the alias
table connecting the two spellings had entries only for PowerPC. The engine
therefore could not read the FS base at all, which is worse than not tracking
it: an unreadable register reads back as **zero**, so every `%fs:`-relative
address was computed from the wrong base. glibc reaches thread-local storage
that way constantly (stack canary, `errno`, malloc arena). Fixed by adding the
aliases; block mode then has a slot for it and the value is read from Unicorn.

**Wide (SIMD) loads and stores — 7 blocks.** The lowering already split wide
*register* varnodes into 8-byte lanes and refused wide *memory*, which is what
every `movdqu` is. Now split the same way, one access per lane at
`address + lane`, which is what the runtime resolves and what the shadow is
addressed in.

**Wide cross-lane shifts — 8 blocks.** `pslldq` / `psrldq` lift to a single
`INT_LEFT` / `INT_RIGHT` on a 16-byte varnode. Not bit-parallel across lanes,
so the lane path declined. Still **exact**, not approximated, because the
distance is an immediate: lane *i* of the result is a funnel shift of two input
lanes, and taint follows the identical routing, a shift being a pure
permutation of bit positions.

**Predicated stores — 4 blocks.** A predicated register write had always been
handled; the store was declined. It is the same question about a different
place: read the old contents, join them with the new under the predicate, store
the result. Both now share one `_join_under_predicate`, so they cannot drift,
and the join is exact including the implicit-flow term when the predicate
itself is tainted.

**Access-limit overflow — 2 blocks.** Exceeding the runtime's per-program
access limit refused the whole block. That is the wrong shape of answer, and
splitting wide accesses into lanes made it more likely. The lowering is now
told the limit, declines at the instruction that overflowed, and the planner
**cuts the region** there instead.

**A program the emitter declined — 1 block.** The host emitter declines
division and count-leading-zeros on purpose (both want fixed registers or a CPU
feature check out of proportion to how rarely a taint rule reaches them) and
the contract has always been that the caller keeps the **interpreter**. The
block path had no interpreter, so it turned "run this interpreted" into "skip
this block". `taint_ir_c` now publishes the interpreter through a capsule and a
region carries its program alongside its emitted function. This is structural:
it covers every opcode the emitter lacks today or tomorrow.

That block also had to lower in the first place, and does because **a wide
divide whose high lane has a provably zero VALUE is the narrow one** — exactly
how a compiler sets up a 64-bit division (`xor %edx,%edx` folds the value,
while the XOR taint rule keeps an over-approximation, so the value is what
decides).

**p-code loops — 3 blocks.** SLEIGH models some instructions as loops rather
than opcodes. Every instance here was `bsf` inside memchr and strlen: a bit
scan is a walk over bit positions, so it arrives as a backward BRANCH to the
instruction's own IMARK, and the taint IR is straight-line by construction.
See below.

## p-code loops: unrolled, floored, or declined

The loop is **unrolled** up to `_UNROLL_LIMIT` (64, which covers every operand
width there is). Unrolling is exact while it lasts, because the predication
machinery already turns each exit test into a select: iteration *k*'s writes
land under "still looping after *k*", which is precisely the loop's meaning.

When the unrollings run out, the residual predicate says whether the loop could
still be running:

- **It folds to a constant zero** — the unrolling was complete and there is
  nothing left to do. This is the common case and it is exact.
- **It does not fold, and the body writes only registers** — everything the
  body writes is marked fully tainted *under that predicate*. The VALUE may be
  wrong there, but it is then a tainted value, so a load through it avalanches
  and a store through it is reported: the engine's existing policies carry it,
  and the answer stays sound rather than silently wrong.
- **It does not fold, and the body writes MEMORY** — **declined**. Flooring
  says "everything this wrote is unknown"; for a register that is a mask we can
  widen, for memory it would be every address the remaining iterations might
  touch, which is unbounded. A `rep movsb` with a count above the limit would
  otherwise have its later stores simply not modelled. Declining is worse than
  handling it and far better than being wrong.

So `rep`-prefixed string operations still decline. **No binary tested so far
executes one**, but that is luck rather than design, and the honest way to
handle them is not a bigger unroll.

### How `rep` should be handled, when it is needed

Not by unrolling. A CPU retires `rep movsb` as one instruction, and so does
Unicorn: the block hook fires once and the whole operation runs to completion.
Its taint effect is a **contiguous shadow range copy** whose length is a
runtime value the block program already threads. So it wants one new region
kind whose semantics the C runtime implements as a range copy, not *N*
iterations of the IR. That is smaller work than unrolling, not larger.

**A secret trip count is implicit flow and needs a report, not a mask.** If
`RCX` is tainted, the *length* of the copy is secret, and therefore so is the
*set of addresses written*. The tainted-pointer *load* analogy does not apply:
there the over-approximation is bounded (one word), so avalanche is a complete
answer. Here it is unbounded, which is exactly the argument the engine already
makes for a tainted store address, where it raises an AIW finding rather than
pretending a mask describes the obligation. The sound handling is all three
together: exact taint for the bytes actually copied, taint on the outputs that
genuinely carry it (`RSI`/`RDI` become `base + n` with *n* secret), and a report
that climbs back up.

## What this costs

Unrolling is not free. A 32-bit bit scan lowers to roughly five thousand
operations where an ordinary instruction takes tens. It is paid once per
distinct block, and only by blocks that contain one: measured on the three
density benchmarks, which contain none, end-to-end block-mode time moved by
less than 4%. `tests/test_pcode_loop_lowering.py` pins the figure so a change
to the limit is visible rather than silent.

A closed form would be far cheaper — `ctz(x) = popcnt((x & -x) - 1)`, and the
IR already has POPCNT — and would fit the codebase's existing closed-form
recognisers. It is not built, because correctness came first and the cost lands
only where a bit scan does.

## Measuring it yourself

`hook_stats()` reports `unhandled`, and why: `no_plan`, `no_regs`,
`cache_full`. Any non-zero `unhandled` is unanalysed code. To find out which
blocks and why, wrap `microtaint.taint_ir.blockcompile.compile_block`, keep the
ones that return `None`, and re-plan them — but note that re-planning with a
*synthetic* slot map answers a different question and will mislead you; the
engine's own map is what trips the biggest gate.

## Two traps worth knowing

**`gc.get_objects()` cannot see nanobind instances.** They are not GC-tracked,
so it reports zero live `PcodeOp`s while ten are held. Anything measuring
pypcode retention has to bisect caches instead.

**Scoring block mode against the per-instruction path reports precision gains
as losses.** Block mode is legitimately tighter in places: it leaves SF clean
after `and $0x1,%edi`, and Unicorn agrees that flipping every bit of RDI moves
PF and ZF and never SF. Score against ground truth, and exclude the flags the
ISA leaves undefined (`benchmark/ISA_UNDEFINED_FLAGS.md`) — a bit scan leaves
CF, OF, SF, AF and PF undefined, the hardware moves them, and no engine reading
that p-code can know what it left.

## Still open

Block mode and the per-instruction path disagree on `RAX` and four flags at the
end of the static-glibc run, with **zero blocks skipped** and memory agreeing
exactly. So it is not the refusals, and it predates this work. It has not been
scored against ground truth yet, and until it is, it is not known whether block
mode is losing taint there or is simply tighter.
