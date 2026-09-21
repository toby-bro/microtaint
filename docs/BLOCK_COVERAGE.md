# What block mode can and cannot lower

Block mode computes the taint of a whole basic block as one compiled program.
A block the lowering **refuses** is not analysed at all: the C hook counts it
and returns, so nothing computes its taint and a tainted value that moves
through one comes out clean. That is an under-taint, so the refusal set is a
soundness property, not a coverage statistic.

This file records what the refusals were, how each was removed, and what is
still declined and why. Measurements are on a **static-glibc x86-64 binary**
(`fgets`, `strlen`, `malloc`, `memcpy`, a byte loop, `snprintf`), because the
hand-written `-nostdlib` guests we measured first refuse nothing at all and hid
every one of these.

The gate is `tests/test_block_skips_nothing.py`, which asserts
`hook_stats()['unhandled'] == 0` on that binary and separately asserts the
trace is big enough to mean something.

## Where it started

| | distinct blocks refused | block executions skipped |
| --- | --- | --- |
| before | 49 of 381 (12.9%) | 71 of 642 (11.1%) |
| now | 0 | 0 |

## The causes, and what each needed

**Thread-local storage, 25 blocks.** SLEIGH names the x86-64 segment bases
`FS_OFFSET` / `GS_OFFSET`; Unicorn names them `FS_BASE` / `GS_BASE`; the alias
table connecting the two spellings had entries only for PowerPC. The engine
therefore could not read the FS base at all, which is worse than not tracking
it: an unreadable register reads back as **zero**, so every `%fs:`-relative
address was computed from the wrong base. glibc reaches thread-local storage
that way constantly (stack canary, `errno`, malloc arena). Fixed by adding the
aliases; block mode then has a slot for it and the value is read from Unicorn.

**Wide (SIMD) loads and stores, 7 blocks.** The lowering already split wide
*register* varnodes into 8-byte lanes and refused wide *memory*, which is what
every `movdqu` is. Now split the same way, one access per lane at
`address + lane`, which is what the runtime resolves and what the shadow is
addressed in.

**Wide cross-lane shifts, 8 blocks.** `pslldq` / `psrldq` lift to a single
`INT_LEFT` / `INT_RIGHT` on a 16-byte varnode. Not bit-parallel across lanes,
so the lane path declined. Still **exact**, not approximated, because the
distance is an immediate: lane *i* of the result is a funnel shift of two input
lanes, and taint follows the identical routing, a shift being a pure
permutation of bit positions.

**Predicated stores, 4 blocks.** A predicated register write had always been
handled; the store was declined. It is the same question about a different
place: read the old contents, join them with the new under the predicate, store
the result. Both now share one `_join_under_predicate`, so they cannot drift,
and the join is exact including the implicit-flow term when the predicate
itself is tainted.

**Access-limit overflow, 2 blocks.** Exceeding the runtime's per-program
access limit refused the whole block. That is the wrong shape of answer, and
splitting wide accesses into lanes made it more likely. The lowering is now
told the limit, declines at the instruction that overflowed, and the planner
**cuts the region** there instead.

**A program the emitter declined, 1 block.** The host emitter declines
division and count-leading-zeros on purpose (both want fixed registers or a CPU
feature check out of proportion to how rarely a taint rule reaches them) and
the contract has always been that the caller keeps the **interpreter**. The
block path had no interpreter, so it turned "run this interpreted" into "skip
this block". `taint_ir_c` now publishes the interpreter through a capsule and a
region carries its program alongside its emitted function. This is structural:
it covers every opcode the emitter lacks today or tomorrow.

That block also had to lower in the first place, and does because **a wide
divide whose high lane has a provably zero VALUE is the narrow one**, exactly
how a compiler sets up a 64-bit division (`xor %edx,%edx` folds the value,
while the XOR taint rule keeps an over-approximation, so the value is what
decides).

**p-code loops, 3 blocks.** SLEIGH models some instructions as loops rather
than opcodes. Every instance here was `bsf` inside memchr and strlen: a bit
scan is a walk over bit positions, so it arrives as a backward BRANCH to the
instruction's own IMARK, and the taint IR is straight-line by construction.
See below.

## p-code loops: every one of them, and what happens to it

A p-code loop is the one shape the taint IR cannot lower directly, being
straight-line by construction. There are **two kinds**, and they are easy to
conflate because only one of them looks like a loop:

- **relative**: a BRANCH or CBRANCH to a negative offset in `const` space,
  which jumps within the instruction's own p-code. The bit scans and the bit
  deposit/extract instructions.
- **self-address**: a BRANCH to the instruction's own address in `ram` space,
  so the instruction re-executes. The `rep`-prefixed string operations. A
  detector that looks only at `const` space misses these entirely, which is
  worth knowing because the first survey written for this document did exactly
  that and reported `rep movsb` as having no loop.

Surveyed across the whole instruction bank plus the forms it does not carry,
**every looping instruction on every supported ISA is AMD64**. Fifteen forms:

| form | kind | handling | IR ops |
| --- | --- | --- | --- |
| `bsf`, `bsr`, `tzcnt` (7 forms) | relative | recognised, closed form | 37-59 |
| `pext`, `pdep` (3 forms) | relative | unrolled, then floored | 2259-5575 |
| `rep movsb`/`stosb`/`movsq` | self-address | **declined** | - |
| `rep cmpsb`, `repne scasb` | self-address | **declined** | - |

Every other instruction set's count lifts to the `LZCOUNT` opcode instead, so
nothing there loops at all: AArch64 `clz`/`cls`, MIPS `clz`/`clo`/`dclz`/`dclo`
and PowerPC `cntlzw` are all two to five p-code operations.

### Recognised: run the loop and ask what it computed

A loop is first offered to `taint_ir/loopform.py`, which **runs it concretely**
on about forty chosen inputs and compares the answers against the counting forms
it knows (trailing zeros, leading zeros, highest-set-bit index, population
count). Agreement on every probe means the closed form is used; disagreement on
any means the caller unrolls exactly as before.

This is deliberately not pattern matching. A pattern is a fact about one
instruction set's specification, it breaks when Ghidra rewrites the spec, and it
cannot say whether the match was right. Running the loop asks what it DOES, so
recognition is semantic, self-checking, and mentions no architecture.

Why a COUNT and nothing else: each recognised form is monotone or antitone in
every input bit, so the two corners of the taint cube (every tainted bit
cleared, and every one set) bracket the whole reachable range. That is the
same rule applied to the `POPCOUNT` and `LZCOUNT` opcodes elsewhere in this
lowering, asked of a loop instead. A loop computing something else has no such
argument, which is why `pext` and `pdep` are refused.

`bsf` at zero is the one place the argument fails: SLEIGH leaves 0 there where
the trailing-zero count is the operand width, so setting a bit RAISES the answer
where everywhere else it lowers it. Zero is reachable exactly when clearing
every tainted bit leaves nothing, and the emission widens to the full span
there. `tzcnt` has no such point and pays nothing for it.

### Unrolled and floored: the fallback

An unrecognised loop is **unrolled**, up to the smaller of `_UNROLL_LIMIT` (64)
and the widest varnode the body touches in bits, since a loop over bit positions
cannot run more times than its widest operand has bits. Unrolling is exact while
it lasts, because the predication machinery turns each exit test into a select:
iteration *k*'s writes land under "still looping after *k*", which is precisely
the loop's meaning.

When the unrollings run out the residual predicate says whether the loop could
still be running. If it folds to constant zero the unrolling was complete. If it
does not fold and the body writes only REGISTERS, everything the body writes is
marked fully tainted under that predicate. The value may be wrong there, but it
is then a tainted value, so a load through it avalanches and a store through it
is reported. If it does not fold and the body writes MEMORY, the lowering
**declines**: flooring would have to name every address the remaining iterations
might touch, which is unbounded.

The floor is a RUNTIME guard rather than a compile-time widening. It contributes
`splat(NEZ(pred))`, and with concrete values a scan has provably stopped long
before the limit, so the guard is zero and the floor adds nothing.

### The epilogue has to survive the loop

A backward CBRANCH means "go round again", and the lowering ANDed that into the
predicate without taking it back out. The unrolling stops EXACTLY when "go round
again" folds to false, so everything after the loop was then lowered under a
false predicate and discarded.

`pext eax,ebx,ecx` came out carrying nothing but the previous taint of its own
destination: read off the lowered program, the answer was
`(T_RAX & 0xffffffff00000000) | (T_RAX & 0xffffffff)`, the old RAX and nothing
else. Checked against Unicorn by hand, eight tainted bits of EBX move eight bits
of the result and the engine reported zero. The value predicate is now restored
to what it was when the loop was entered; `pred_t` is not, because it has
accumulated the implicit flow of every exit test and the rest of the instruction
inherits that.

`bsf` never showed it: its back-edge is an unconditional BRANCH, which does not
touch the predicate. Only the CBRANCH-backed loops were affected, and the two of
them were reporting nothing at all.

**`pext` and `pdep` are not scored against Unicorn**, which mis-executes these
VEX forms: it does not zero the destination's upper half for a 32-bit `pdep`,
so a per-bit comparison reports under-taints that are the emulator's. They are
gated on DEPENDENCE instead: a tainted source must reach the destination, which
needs no oracle and is exactly what was broken.

## What this costs

Lowering one instruction, measured:

| | ms | IR ops | over-taints vs hardware |
| --- | --- | --- | --- |
| an ordinary ALU instruction | 1.6 | 94 | - |
| `bsf eax,eax`, unrolled | 26.5 | 2562 | 18 of 168 |
| `bsf eax,eax`, recognised | **7.4** | **64** | **5 of 168** |
| `bsf rax,rdx`, unrolled | 48.9 | 4694 | 13 of 168 |
| `bsf rax,rdx`, recognised | **16.9** | **53** | **1 of 168** |

Faster to compile as well as to run, and tighter, with zero under-taints
throughout. The compile-time win is not obvious and took a measurement: the
probe was 391 ms for a 64-bit scan until it was compiled to plain tuples once,
because reading a varnode's space through the lifter binding on every step was
most of the cost. The result is memoised on the p-code itself, since the planner
lowers the same instruction many times while it searches for region boundaries.

Those IR operations are paid on every EXECUTION of a block containing the
instruction, not once at compile time, which is what makes the difference
between 2562 and 64 matter more than the milliseconds.

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
| --- | --- | --- |
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
The 64-bit case is unchanged, correctly.

Over the whole instruction bank that is **-12.6% of `AMD64_total_ops`**, 30,575
to 26,732 across 394 lowered forms, with every other ISA byte-identical and no
change to what lowers or declines. Ten of those 394 forms are bit scans, which
is how one bound moves an eighth of the total. `tests/test_pcode_loop_lowering.py`
pins the ratio rather than the two numbers, so it keeps its meaning when the
lowering gets cheaper for other reasons.

### The closed form is cheaper AND tighter, once the rule is right

A closed form would take this to a handful of operations. The obvious objection
was that it would cost precision, and the codebase could answer that from
inside, because **Ghidra already models the two directions differently**: the
leading-zero count lifts to a p-code OPCODE (`LZCOUNT`) and the trailing-zero
scan lifts to a LOOP. So both answers to the same question were already
running, and could be scored against each other.

Measured on `eax, ebx` forms, 256 outputs each, against Unicorn per-bit truth:

| | IR ops | over-tainted | under |
| --- | --- | --- | --- |
| `lzcnt` (closed form, old avalanche taint rule) | 31 | 62 | 0 |
| `tzcnt` (p-code loop, unrolled) | 2589 | 27 | 0 |
| `lzcnt` (closed form, corner taint rule) | **40** | **0** | 0 |

The objection was real but it was an artefact of the TAINT rule, not of the
closed form. `POPCOUNT` and `LZCOUNT` both carried
`span & splat(NEZ(input_taint))`: one tainted input bit marked every bit of the
result. That is very loose for a counter, in two specific ways. A leading-zero
count does not depend on the bits BELOW the highest set one at all, and a
population count that can move by one moves only its bottom bit.

Both are fixed by the codebase's own corner trick, the one `_corners` already
plays for a ripple carry: evaluate the count with every tainted bit CLEARED and
again with every tainted bit SET. Those bracket the reachable range, because
popcount is monotone in each bit and a leading-zero count is antitone in the
value, so the bits that can differ are the bits below where the two endpoints
first differ. Nine extra operations, and the cheap form becomes **exact** and
tighter than the 2589-operation unroll.

This is not a rare path. `POPCOUNT` is how SLEIGH computes x86's PARITY flag,
so it sits under every arithmetic and logic instruction there, and `LZCOUNT`
carries `lzcnt`, AArch64 `clz`/`cls`, MIPS `clz`/`clo`/`dclz`/`dclo` and
PowerPC `cntlzw`. Measured over the banks: every leading-zero form on AMD64 and
ARM64 went from 24 and 12 over-taints to **zero**, `popcnt` from 12 to 1-2, and
bank-wide over-tainted bits fell by 94 on AMD64 and 114 on ARM64 with no
under-taint anywhere. It costs 0.2-4.5% more IR operations.

`tests/test_bit_count_taint.py` gates it, including on AArch64, which the rule
was never written against: it is expressed over p-code's `LZCOUNT` rather than
over an instruction, so an untested ISA gets it for free, and that is the check
that says so.

**Recognising the loop is now the remaining half**, and the trade has reversed:
a matched `ctz` would be both ~64x cheaper and no less precise. What it still
costs is matching a shape SLEIGH chose for one instruction set. That is the
decision left open, and it is now a much easier one than it looked.

## Measuring it yourself

`hook_stats()` reports `unhandled`, and why: `no_plan`, `no_regs`,
`cache_full`. Any non-zero `unhandled` is unanalysed code. To find out which
blocks and why, wrap `microtaint.taint_ir.blockcompile.compile_block`, keep the
ones that return `None`, and re-plan them, but note that re-planning with a
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
ISA leaves undefined. A bit scan leaves
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

## Where the two paths still differ, and which is right

Both differences that turned up here are now settled, and neither is an
under-taint.

**The register-level `RAX` difference** that started the investigation was an
artifact: the per-instruction path calls `emu_stop` on its first finding, so its
trace is shorter, and an entry-by-entry diff reads the truncated tail as loss.

**Block mode reports a branch the per-instruction path does not**, at
`0x40700e`: `cmp $-1,%eax; je`, glibc's EOF test, which runs *before* the
delimiter test both paths report. Scored against hardware rather than against
each other, **the per-instruction path is right and block mode over-taints**.
`EAX` there holds a byte `__uflow` returned, so bits 8-31 are known zero and the
comparison can never be equal: ZF is constant however the tainted low byte
moves, and Unicorn confirms it does not move over all eight flips.

An earlier version of this section blamed the shape of the equality rule,
saying it was "any tainted input bit taints the boolean" and was not built. That
was wrong, and reading the code rather than assuming would have caught it: the
value-aware rule has been in `taint_ir` since the file was created, in both the
narrow and the wide form.

The real cause was the shape of the QUESTION. A comparison sets the zero flag,
and SLEIGH models that as `INT_EQUAL(INT_SUB(a, b), 0)`: subtract, then ask
whether the difference is zero. Asked about the difference, the rule can only
prove inequality when some bit of the difference is provably one. Asked about
the operands, it can prove it whenever they differ in a bit neither side can
change, which is a far weaker condition:

| | can the rule prove it | why |
| --- | --- | --- |
| operands `EAX` vs `-1` | yes | bits 8-31 are clean zero against set |
| difference `EAX + 1` | no | in [1, 0x100], no single bit provably one |

`(a - b) == 0` is exactly `a == b` in wrapping arithmetic, so the lowering now
remembers what a difference was a difference OF and answers equality on the
operands. The memory is keyed on the VALUE NODE rather than on the varnode:
a node id is a fixed expression, so it cannot go stale when a `unique` offset is
reused, which pypcode does freely. The substitution applies only against a
constant zero, and only when the recorded width matches, because neither holds
in general.

Measured on the static-glibc guest: 42 distinct sites down to 40, and roughly
72 ms down to 69 ms. Faster, not slower, because a branch whose flag is proved
clean does not taint the program counter, so less taint flows downstream. The
first timing said the opposite and was noise from too few iterations.

Over-reporting is the acceptable direction, and a leak reported at a branch that
cannot leak is a triage cost rather than a soundness failure. It is recorded
here because for the fuzzing and concolic milestones, triage cost is the thing
that decides whether the findings get used.

## Compiling a block is a one-time cost, not a per-run one

Block mode was faster per instruction from the day it was wired, and still lost
to the path it replaces on every run after the first. Three consecutive runs of
`bench_dense.elf` in one process, before any cache:

| | run 1 | run 2 | run 3 |
| --- | --- | --- | --- |
| block mode | 451.8 ms | 428.6 ms | 426.8 ms |
| *of which the compiler* | 360.2 | 354.7 | 353.3 |
| per instruction | 659.5 ms | 170.7 ms | 170.1 ms |

The per-instruction path has a process-wide rule cache, so its second run is
3.9x faster than its first. Block mode had none: its plan cache lived in the C
hook context, which is created per wrapper and dies with it, so all 38 blocks
were lifted, lowered and emitted again on every run. From run 2 onward the
slower path won by 2.5x, and run 2 onward is the only regime a fuzzer is ever
in.

`taint_ir/blockcompile.py` now keeps the compiled block for the process. The
split is what makes this safe: of `compile_block`'s 6.08 ms for a ten
instruction AMD64 block, 5.13 ms is the lift and lowering and 0.64 ms the emit,
while turning the result into a plan is 1.1 us. So the first two are cached and
the plan is still built per caller, because a plan carries the addresses of ONE
wrapper's Unicorn register-read buffers. Sharing those would have a second
emulator read its register file into the first one's scratch, and nothing would
raise.

Afterwards, same binary, same process: **771 ms, then 82 ms, then 82 ms**, with
0 compilations on runs 2 and 3. Against the per-instruction path's steady 171
ms, block mode is now 2.1x faster rather than 2.5x slower.

### What the key carries, and why each part is there

Each part earns its place by a way the cache would otherwise be silently wrong,
and `tests/test_block_plan_cache.py` has a behavioural test per part that fails
when it is dropped.

| part | what goes wrong without it |
| --- | --- |
| the base address | blocks are compiled with `abs_ram=True`, so a PC-relative operand's `ram` address is baked in; the same bytes at another address would read wherever the block first ran |
| the CODE, not its address | rewritten bytes would inherit the plan compiled for what used to be there |
| the slot map | emitted code addresses engine slots by index, so a program compiled against one layout writes its answer to the wrong register under another |
| the architecture | the same bytes decode differently on each |
| `publish_all_values` | a block that publishes nothing has had the code producing those values deleted, so a caller chaining regions by hand gets stale registers |

Keying on the bytes rather than on the address is also what makes
self-modifying and JIT'd code safe: rewritten bytes are a different key and
miss. A refusal is cached too, because learning that a block does not lower
costs a full lift and real binaries are mostly such blocks.

`MICROTAINT_BLOCK_PLAN_CACHE=0` bypasses the cache entirely, which is the switch
to reach for when a compiled block is suspected of being wrong.

### Two defects this surfaced, and one still open

Writing the tests for the cache turned up two bugs that had nothing to do with
it, both of which reproduce with the cache disabled.

**Lowering was not thread-safe.** `builder_for` hands every caller the same
stateful `Builder`, so two threads lowering at once walk over each other's
predicate stack. Six threads lowering nine short blocks raised `IndexError` out
of `prog.live[n]` on about one run in three. Lowering is now serialised under
`frompcode.BUILDER_LOCK`, and the cache lookup is inside that lock so two
threads asking for the same block produce one compilation and one hit. It costs
nothing: this runs once per distinct block per process.

**A write onto code threw away the held block's taint.** The deferred commit
holds a block until the next block proves it completed, and `hook_invalidate`
used to call `mt_blk_abandon` on every invalidation. That is right when the
write rewrote the held block's own instructions, and an UNDER-taint every other
time. The two are easy to confuse because the hook tracks the code it has
planned as a single `[lo, hi)` interval: one JIT page stretches that interval
across the whole image, and then every ordinary store to a global in between
looks like self-modifying code. Measured on a guest that mmaps an RWX page,
writes a four-byte stub and calls it, the value the stub returned came back
clean and the secret-dependent branch it fed was never reported. The
invalidation is now handed the write's own address and abandons the held block
only on a real overlap; `hook_stats` counts `abandoned` separately from
`invalidations` so the difference is visible.

**Fixed: `xor reg,reg` now clears taint.** The IR folded the VALUE of
`XOR(x, x)` to a constant, but the taint rule for a binary op is `t_a | t_b`,
which for two copies of the same node is `t | t = t`, so full taint walked
through a register that provably holds zero. The two halves were computed side
by side and never consulted each other.

The rule is asked once, where a result is written: if the value is a constant,
no input bit can move it, so its taint is zero. It covers `sub reg,reg`,
`and reg,0`, `or reg,-1` and the flags those set, and it is in the lowering
rather than an x86 table, so ARM64's `eor` and RISCV64's `xor` get it too.
Partial widths stay correct because the lowering already models them: `xor
eax,eax` clears all 64 bits, `xor ax,ax` leaves bits 63:16 tainted.

The first version of the rule was an UNDER-taint, and the shape of the mistake
is worth keeping. An operation p-code does not model writes an invented value:
`_emit_callother` writes a literal zero because it has to write something, while
keeping the taint at avalanche. So `const 0` in the IR means either "proved
zero" or "no idea", and clearing on the second loses real taint. Measured, it
lost bit 8 of `crc32 rax, cl`, and the per-bit ground-truth sweep caught it.

So the rule is opt-IN. `proved` defaults to false, and only callers that write
what p-code semantics actually computed pass it; `_emit_callother` does not.

Getting the polarity right was not enough on its own: the first shipped version
opted in too broadly and was an under-taint of its own.
`_invention_stays_opaque` deliberately lets an invented value travel through
COPY, INT_ZEXT, INT_SEXT, SUBPIECE and PIECE, because their taint rule reads
taint alone and moving a lie does not make it a worse lie. But their VALUE is
then the invented constant, so a movement op claiming `proved` clears the
avalanche it was carrying. Measured: `crc32 eax, bl` is a CALLOTHER into EAX
and an INT_ZEXT into RAX, and it reported RAX clean however the inputs were
tainted, while the 64-bit `crc32 rax, cl`, which has no zext, was correct.
Movement ops never claim `proved`, and nothing is lost by that: a zeroing idiom
has already had its taint cleared by the op that computed the zero, so the
movement copies that zero forward.

That bug was live in a gate reporting 15,022 passing tests, which is the more
useful half of the story. `test_ir_never_under_taints_vs_ground_truth` drew its
vectors from `fuzz_budget(2)`, and `fuzz_budget` takes a quarter outside the
slow tier with a floor of one, so the fast tier probed each instruction ONCE.
This under-taint needs two sparse vectors to surface. The gate now asks for
eight, so the fast tier gets two, and refuses to run at one rather than probing
once and reporting success. That polarity is the point: a caller that invents a value gets the
safe answer by saying nothing, and a future one that forgets loses an
optimisation rather than silently under-tainting. Two tests pin it, one
asserting the default and one reading `_emit_callother` to check it never
claims `proved`.

Measured on a static-glibc guest, which is where compiler output actually uses
the idiom:

| | time | side-channel findings | distinct sites |
| --- | --- | --- | --- |
| before | 79.6 ms | 110 | 47 |
| after | 72.4 ms | 105 | 42 |

Five fewer sites to triage, and 9% faster, because a taint of constant zero lets
dead-code elimination delete whatever was computing it: `xor rax,rax` goes from
40 IR nodes to 20 and `sub rax,rax` from 93 to 73. There is no change on
`bench_dense`, which is hand-written `-nostdlib` code that does not zero
registers this way, so the win is real but workload-shaped.

## Where a block's cost actually goes

Measured on `bench_dense.elf`, 379,795 guest instructions, steady state so
compilation is amortised. The rungs come from the staged bisect in
`blockpath_c.c` (`MICROTAINT_BLOCK_STOP`), read in the order 1, 2, 3, 7, 8, 5,
6, 4, plus the three skip probes 9, 10 and 11 inside the access loop.

| what it adds | ns/instr | share of 181 |
| --- | --- | --- |
| bare Qiling and Unicorn | 21.7 | 12% |
| an empty pure-C block hook | 0.9 | 0.5% |
| the engine attached, block body off | 51.4 | 28% |
| the plan-cache lookup | 8.1 | 4% |
| the register-file read | 8.0 | 4% |
| computing the block | 79 | 44% |
| the commit | 4.3 | 2% |

and inside "computing the block", before the dead seeding was removed:

| | ns/instr |
| --- | --- |
| per-region state copies | 12.1 |
| the copies pass 1 needed | 9.8 |
| pass 1, the address slice | 5.7 |
| resolving the loads it found | 31.5 |
| pass 2, the stores, threading | 29.8 |

and inside resolving the loads:

| | ns/instr |
| --- | --- |
| the guest memory read | 31.7 |
| the shadow read | 3.9 |
| the store overlay | 1.3 |

### What this says, and what it rules out

**The emulator floor is 12%, not 40%.** An empty pure-C block hook costs 0.9
ns/instr, so losing Unicorn's translation-block chaining is free, and the 51.4
that the engine adds before computing anything is OURS rather than Unicorn's.

**That 51.4 is not recoverable, and measuring it settled the question.** It is
the cost of having a `UC_HOOK_MEM_WRITE` registered at all, not of running it:
the hook body, the detectors and the deferred registration each cost nothing,
and restricting the hook's address range does not help because Unicorn filters
in the dispatcher after paying for the instrumentation. Removing the hook makes
the engine 3x slower in block mode and 23x on the per-instruction path, because
clearing taint is what keeps the shadow sparse and instructions on the cheap
untainted path. It also loses precision: block plans under-clear on their own,
and block mode reports 57 sites instead of 41 without it.

**The address slice works.** 1.93 addresses out of a program costing 5.7
ns/instr against the full program's 29.8, which is what it was built for. Its
setup cost 9.8, nearly twice the program, and that was dead: `so` is pass 1's
output array and the only thing read back out of it is each load's address.

**The guest memory read is the largest thing still under our control**, 31.7
ns/instr, roughly a fifth of block mode. 76% of loads genuinely need their
value, so dead-code elimination has already pruned what it can, and there are
only 1.2 loads per region so there is little to batch. `uc_mem_read` costs the
same for 8 bytes as for 64, so it is per-call overhead, and the only way past it
is to stop going through Unicorn's memory API: map guest memory with
`uc_mem_map_ptr` so the engine holds host pointers and a load becomes one
instruction. That is an architectural change, not a tweak.

**The remaining per-region copies are 12.1 ns/instr** and the same argument that
killed the pass-1 seeding may apply: the plan knows at compile time which value
slots each region publishes, so the two whole-register-file copies could become
a short list. It is worth 6-7%, and it is subtle enough to want its own careful
pass: getting the array and slot convention wrong there produces wrong VALUES,
which become wrong addresses, which become under-taints.


## What a fuzzer-shaped workload actually spends its time on

Everything above is measured on `bench_dense.elf`, which runs 38 distinct
blocks twenty thousand times. That is the right workload for the taint
computation and the wrong one for everything else: a fuzzer runs a fresh
emulator per input, so it meets each block once or twice and pays setup
repeatedly.

Profiled on a static-glibc guest driven that way (402 distinct blocks, 626
block executions, a fresh emulator per run), the shape is different enough to
change the priorities:

| | share of cycles |
|---|---|
| Python interpreter | 14.0% |
| the garbage collector | ~13.9% |
| Unicorn translating blocks (TCG) | ~8.4% |
| computing taint | does not reach the top 22 |

`mt_blk_compute` is the top item on `bench_dense` and invisible here. Neither
number is wrong; they are different questions, and a tool aimed at fuzzing has
to answer the second one.

### Identifying the slot map cost more than using it

The compiled-block cache is keyed partly on the slot map. Measured in the
engine rather than in a microbenchmark, per distinct block per run:

| | before | after |
|---|---|---|
| building the key (a frozenset of 210 pairs) | 12.5 us | 2.3 us |
| the dictionary hashing and comparing it | 31.2 us | 2.6 us |
| turning the compiled block into a plan | 24.6 us | ~25 us |

Over half the cost of a cache HIT was working out which slot map it was. A
`SlotMap` now answers with a token computed once at construction, from a
registry keyed on the map's content, so it is exact rather than a hash. On the
glibc guest that is 213.4 M instructions per run down to 183-194, about 12%,
with identical findings.

A microbenchmark had put the whole warm path at 24.5 us and it was 76-86.
Real blocks and a real 210-name slot map are nothing like a toy one, which is
the general lesson: measure the warm path where it runs.

### The collector is worth about as much as the interpreter

Most of the collector's work is traversing structures that exist for the life
of the process and can never become garbage: the compiled blocks, the emitted
code they hold alive, the slot map. `blockcompile.freeze_for_reuse()` takes
them out of its reach. Median run time over the same workload:

Measured as instructions retired, marginal per run, over three sets:

| | M instructions | vs on |
|---|---|---|
| collector on | 195.3 | |
| after `freeze_for_reuse()` | 187.7 | -3.9% |
| collector disabled | 182.2 | -6.7% |

It is opt-in and the engine never calls it, because `gc.freeze()` is
process-wide rather than ours alone, and because it has to be called once
rather than per run: calling it per run builds a permanent generation that only
grows. Disabling the collector outright is faster still and is the caller's
decision, not the engine's.


### Measure instructions, not seconds, on a machine you do not own

Two of the figures above were first taken as wall clock and were wrong by a
factor of three, because the machine was running something else at the time:
load average 12.7 on 16 cores, another agent's test suite, and a game. Wall
clock said `freeze_for_reuse()` was worth 13% and the slot-map token 27%;
instructions retired, which contention cannot move, say 4% and 12%.

The measurement that survives a busy machine is the MARGINAL instruction count:
run the workload N times and 5N times under `perf stat -e instructions`, and
take the difference divided by 4N. Startup cancels, and the result is
repeatable to about 3% even under load. Every performance figure in this
document from the block-cost ladder onward was taken that way or re-checked
that way.

Across the session's block work on the glibc guest, marginal instructions per
run: **240.9 M at the start, 183-194 M now, about 22%.**
