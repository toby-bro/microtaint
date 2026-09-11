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

The loop is **unrolled**, up to the smaller of `_UNROLL_LIMIT` (64) and the
widest varnode the body touches, in bits. A loop over bit positions cannot run
more times than its widest operand has bits, and the bound is read off varnode
SIZES rather than off opcodes, so it is a fact about the p-code and not a rule
about one instruction set. Unrolling is exact while it lasts, because the
predication machinery already turns each exit test into a select: iteration
*k*'s writes land under "still looping after *k*", which is precisely the
loop's meaning.

When the unrollings run out, the residual predicate says whether the loop could
still be running:

- **It folds to a constant zero** — the unrolling was complete and there is
  nothing left to do. **No bit scan takes this exit**; see below.
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

### Bit scans floor; the floor costs nothing at runtime

Measured rather than assumed: **every bit scan takes the FLOOR exit**, not the
fold. SLEIGH's loop exits on the operand's own bits, so the predicate is a
function of a runtime value and there is nothing for the constant folder to do.

That is not the imprecision it looks like, because **the floor is a runtime
guard, not a compile-time widening**. It contributes `splat(NEZ(pred))`, and
with concrete values a scan has provably stopped long before the limit, so
`pred` is zero and the floor adds nothing. Scored against Unicorn per-bit truth
over 168 outputs each: `bsf` over-taints its destination in 18 and 13 cases
(32-bit and 64-bit), `bsr` in none, flags are exact everywhere, and **none of
the three ever under-taints**.

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

Unrolling is not free, and the figure is worth having exactly, because the
block compiler is most of block mode's wall clock (87-93%) at around 8 ms per
distinct block. Lowering one instruction, measured:

| | ms | IR ops |
|---|---|---|
| an ordinary ALU instruction | 2.0 | 85 |
| `bsf eax,eax`, flat limit of 64 | 54.5 | 5121 |
| `bsf eax,eax`, bounded by the operand | 26.5 | 2562 |
| `bsf rax,rdx` (64-bit, so unchanged) | 48.9 | 4694 |

So a bit scan is the **p100 of block compilation**: one instruction costing more
than six ordinary blocks. Each iteration is around 80 IR operations, because
every write in the body becomes a select.

Bounding the unrolling by the operand width halves the 32-bit case **with no
change in precision at all** (the same 18 over-taints before and after, still
zero under-taints), because the iterations it drops are ones that can never run.
The 64-bit case is unchanged, correctly. `tests/test_pcode_loop_lowering.py`
pins the ratio rather than the two numbers, so it keeps its meaning when the
lowering gets cheaper for other reasons.

A closed form — `ctz(x) = popcnt((x & -x) - 1)`, and the IR already has POPCNT
— would take this to a handful of operations and would also remove the residual
over-taint. It is **not built**, and the reason is a real tension rather than a
shortage of time: recognising "this p-code loop is a bit scan" means matching a
shape that SLEIGH chose for one instruction set, which is exactly the kind of
x86 specialisation the rest of the lowering avoids. The width bound above is
the part of the win that can be had from varnode sizes alone. Taking the rest
is a deliberate decision about generality, not a cleanup.

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

## The leak block mode found and did not report

"Nothing is skipped" was not "block mode is sound end to end", and the thing it
was hiding was not a lost bit of taint. It was the **finding**.

The runtime has always detected a secret-dependent program counter: it tests
the PC slot's taint after every region and files a report. Two things then went
wrong, independently, and each one alone was enough to lose every finding:

- **The PC had no slot.** `rip_slot` was resolved lazily by the
  per-instruction hook, which is exactly the path block mode replaces, so it
  stayed at its initial `-1` for the whole run and the runtime's
  `pc_slot >= 0` test was false every time. Nothing was ever detected.
- **Nobody read the reports.** The reports live in the pending block, released
  by the same commit that releases its taint, because a block that faults
  partway through never happened. The test RUNNER had drained them since the
  day it was written. The live hook never did, so on the next commit they were
  overwritten.

That is why no test caught it: the tests drove the runner and the binaries
drove the hook, and only one of the two had a reader. **Block mode ran a real
binary, computed that a branch depended on stdin, and reported nothing.**

On the static-glibc guest the per-instruction path reports one side channel (it
calls `emu_stop` on the first, which is also why its trace is shorter and why an
entry-by-entry register diff shows the tail as a "loss" that is nothing of the
kind). Block mode now reports **137**, including that one: the delimiter test in
`_IO_getline_info`, the EOF test beside it, the guest's own byte loop and hash
loop, and `_itoa_word`'s digit loop inside `snprintf`. All of them are genuine
secret-dependent branches, and all but the first were invisible before, on
either path.

Block mode deliberately does **not** stop the run. The report is already a block
late, so stopping prevents nothing, and these milestones are analyses rather
than mitigations: continuing is what turns one finding into the whole set. That
is the one respect in which the two paths are meant to disagree, so the gate
(`tests/test_block_reports_leaks.py`) is one-directional: every address the
per-instruction path reports must be in block mode's answer, and block mode may
report more.

A finding names the **last instruction of the region** that computed it. The
runtime learns the counter is secret-dependent only once a whole region has run,
so that is the finest address available without spending ops on a per-
instruction check in the hot path; a block ends at its branch, so for the case
this catches it is the branch, and the same address the per-instruction path
gives.

## Still open

The per-instruction path reports the delimiter test at `0x407017` but not the
EOF test at `0x40700e`, which executes first and branches on the same tainted
`EAX`. Block mode reports both. Either the per-instruction path under-reports
there or the two disagree about when that taint arrives; it has not been scored
against ground truth, and until it is, which one is right is not known.

The register-level `RAX` difference that started this investigation is
explained: it was the shorter trace, caused by `emu_stop` on the first finding.
