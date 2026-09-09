# RQ1: synthesis vs observation-based inference

**Claim (§6.1).** microtaint synthesises sound rules in regimes where
observation-based inference (TaintInduce) does not: carry chains, flags, and
wide operands, where the number of observations needed to pin the rule down
grows faster than it is practical to collect.

## First, a working TaintInduce

There is no runnable TaintInduce left to compare against. The original artifact
([github.com/melynx/taintinduce](https://github.com/melynx/taintinduce)) targets
Python 3.6, and its serialization layer imports `squirrelflowdb`, which has not
existed on PyPI for years. Other bugs surface once it starts.

We therefore maintain a repaired fork,
[github.com/toby-bro/taintinduce](https://github.com/toby-bro/taintinduce), and
that fork is what we evaluate. The repairs are dependency and porting work
(Python 3.12, a rewritten serialization layer, fixes to the x86/AMD64/ARM64
backends and to memory operands). The inference algorithm is the paper's.

`setup_taintinduce.sh` pins commit
[`64a703a`](https://github.com/toby-bro/taintinduce/commit/64a703a2ce539d725bc4dbf3d90394fb8c9443eb),
**not** the branch tip. Everything after that commit on `master` is the work that
grew into microtaint, and `master`'s entry point no longer runs the DNF inference
at all: it classifies the instruction and emits an instrumentation circuit
instead. Comparing that against microtaint would be comparing microtaint against
an early microtaint. `64a703a` is the last state of the tree that is still
TaintInduce.

## Run

```sh
./setup_taintinduce.sh          # clone, pin, install, smoke-test; ~2 min
./run_rq1.sh --quick            # the fast families, ~13 min
./run_rq1.sh                    # everything but the 64-bit probe, ~1 h
./run_rq1.sh --full             # everything, one hour of budget for the last row
```

Results land in `results/<timestamp>/`: one `.log` and one `.json` per case, the
inferred rules under `rules/`, and a `SUMMARY.md` holding the table.

Re-examining a result costs nothing, because inference is the expensive half:

```sh
uv run --project external/taintinduce python score_rule.py \
    --arch X86 --bytes 00d8 --name 'add al, bl' \
    --rule results/<stamp>/rules/00d8_X86_rule.json
```

`setup_taintinduce.sh` needs an **x86-64 host**. TaintInduce minimises its DNFs
with espresso, which the repository ships as a prebuilt x86-64 ELF; on any other
architecture the inference dies at its first minimisation call.

## The check: is every carry dataflow there, and does it fire?

For an addition the required dataflow structure is not a matter of opinion. A
carry out of bit *i* travels upward, so bit *i* of either operand can reach
output bit *j* for **every** *j* ≥ *i*. The rule therefore needs one dataflow per
`(i, j)` with `j >= i`: `w(w+1)/2` per source operand, so 1,056 of them for
`add eax, ebx`. For a bitwise instruction the required shape is the diagonal
instead, bit *i* to bit *i* and nothing else.

That shape is checkable directly, with no ground-truth corpus and no sampling.
`score_rule.py --carry-width W --carry-shape triangle|diagonal` asks it three
ways, in increasing order of what they prove:

**1. Is the `(i, j)` pair in the rule at all?** Purely structural: enumerate the
rule's condition-dataflow pairs and look for holes in the triangle. The answer
is that there are **none**, at 4, 8, 16 and 32 bits, for `add`, `sub` and `xor`
alike. Every required dataflow is present. So the failure is not a missing
dataflow, and this is worth knowing before drawing any other conclusion.

**2. Does it fire on a state that forces the carry?** Each pair carries a DNF
condition that decides whether it fires, and that is where the rules break. For
each `(i, j)` the harness *builds* a state that forces the dependency: a single 1
at bit *i* in one operand, a run of 1s across bits *i..j-1* in the other, so bit
*i* generates a carry, every intermediate bit propagates it, and it lands on bit
*j*. The state is executed on the real CPU and kept only if the dependency
actually shows up, then the rule is asked. On `sub eax, ebx`, **447 of the 1,056
constructed chains do not fire**. Nothing about that depends on a lucky draw:
the states are constructed, the check is deterministic, and it reruns
identically.

`add` scores 0 silent here and `sub` scores 447, and the asymmetry is in the
witnesses rather than in the rules. The constructed chains are built for
addition, so on `add` they land on exactly the shapes the inference's own
`Bitwalk` and `BitFill` seed strategies produce, and the rule has them. On
`sub` the same operand values exhibit a different set of `(i, j)` dependencies,
away from anything the seeds covered, and the rule goes quiet. Every silent
cell is still a state the CPU was observed to carry through, so each one is a
real under-taint.

**3. Does it fire on random states?** One random state exercises every *j* that
bit *i* reaches at once, which makes this cheap, and it is the realistic case.
For `add eax, ebx`, of the 1,056 required dataflows only 411 ever turn up in a
random draw at all; of those, 93 always fire, 182 fire only sometimes, and 136
never fire.

The three answers together are the whole story: TaintInduce recovers the
*structure* of addition and then guards each dataflow with a condition fitted to
the states it happened to sample.

## Reading the triangle

The `--carry-width` output draws the `(i, j)` grid directly. `add al, bl`:

```
  EAX[i] -> EAX[j]              EBX[i] -> EAX[j]
      j: 01234567                   j: 01234567
    i=0   ####oooo                i=0   #####ooo
    i=1    ####~~.                i=1    ######o
    i=2     ##~#~#                i=2     ###~..
    i=3      #~~~.                i=3      #~#~.
    i=4       #~~~                i=4       #~~~
    i=5        #~~                i=5        #~~
    i=6         #~                i=6         #~
    i=7          #                i=7          #

  legend: # always fires   ~ sometimes   . never, though witnessed
          o only a constructed witness, and it fires   ? unwitnessed
```

Short carries are learned (`#` down the diagonal), medium ones are unreliable
(`~`), longer ones are wrong (`.`), and the far corner is `o`: no random state
ever exercised those dataflows. That corner is the paper's Case 2 in visual
form. A random addition propagates a carry through bit *k* only when
`a[k] xor b[k]` is 1, so a carry of distance *d* appears with probability about
`2^-(d-1)`; past *d* ≈ 7 it does not appear in a sample of any practical size.
At 32 bits, 645 of the 1,056 cells are in that corner.

**The `o` cells should be read as the friendly case, not as a pass.** The
constructed carry chains are single set bits and runs of 1s, which is very close
in shape to TaintInduce's own `Bitwalk` and `BitFill` seed strategies. The
inference has almost certainly trained on states like them, so their firing is
what you would expect, and it is the `~` and `.` cells that carry the finding.

## The held-out scoring, and why the tool's own check does not do it

Alongside the structural check, every case is also scored on **held-out** states:
the rule is inferred from one seed draw and tested against a second, independent
one, using the one-bit noninterference oracle on the real CPU. This is what
covers the outputs that have no algebraic shape to check against, the flags in
particular.

This matters because TaintInduce ships a validator,
`validate_rule_explains_observations`, which passes on every case below,
including the ones reported here as unsound. It has to: it checks the rule
against the observations the rule was *fitted on*. Look for

```
Rule validation successful: all 74918 observation behaviors explained.
```

in `results/<stamp>/arithmetic_X86_00d8.log`, directly above the verdict
`UNSOUND`.

The held-out comparison also keeps the two failure directions apart, which the
tool's validator does not, since it scores a flow as explained only on set
*equality*:

- ground truth **minus** prediction non-empty → **under-taint**. Unsound.
- prediction **minus** ground truth non-empty → over-taint. Imprecise but safe.

Both register operands are taint sources, as §6.1 states and unlike the original
TaintInduce evaluation: the oracle flips every bit of every input register in
turn, so an under-taint reported on `EBX[2]` is a flow out of the second operand
that the rule does not have.

## The verdicts

| Verdict | Meaning |
| --- | --- |
| `correct` | every scored flow exact |
| `correct (over-taints flags)` | no dependency missed anywhere; the only over-approximation is on flag bits (see below) |
| `sound, over-taints data` | no dependency missed, but spurious flows onto data bits |
| `UNSOUND` | at least one real dependency missing |
| `did not converge (>Ns)` | the budget ran out during inference |
| `NO-DATA` | the oracle produced zero flows, so nothing was compared |

Two extra lines report the structural check: `STRUCTURE` for pair existence and
`FORCED-CARRY` for whether those pairs fire.

`correct (over-taints flags)` is Table 3's "correct". The over-taint it tolerates
is the paper's Case 3 (§3) happening in front of you: on `xor eax, ebx` every one
of the spurious flows is an input bit claiming to reach **ZF**.

```
spurious EBX[9] -> EFLAGS[6]=ZF  at state 0xd0f1f48c8da04a5fc6b2145e
```

ZF is set exactly when the 32-bit result is zero. As a DNF over the input bits
that predicate has no short form, so the inference cannot express *"and only when
the result happens to be zero"* and falls back to an unconditional flow. It
over-taints rather than under-taints, which is the safe direction, but it is the
representational ceiling §4 is built to get past.

## The one number that explains the rest of the table

TaintInduce enumerates the state space when it holds fewer than `2**14` states
and samples it otherwise (`observation.py`, `_gen_seeds`). Every case prints
which side of that line it is on:

```txt
state: 12 bits (R1:4 R2:4 NZCV:4) -> seeds are EXHAUSTIVE
state: 96 bits (EFLAGS:32 EAX:32 EBX:32) -> seeds are SAMPLED
```

Below the line the inference has seen every input and is right by construction.
Above it, it is generalising, and carry chains are what it fails to generalise: a
carry from bit *i* to bit *j > i* only shows up in a seed that configures the
intermediate bits of **both** operands so a carry passes through. Each `(i, j)`
pair is a separate dependency, the number of pairs grows with the square of the
operand width, and the seed budget does not.

The width sweep therefore takes its 4-bit point from **JN** ("Just Nibbles", the
fork's toy ISA: two 4-bit registers and a 4-bit flag register, 12 state bits,
below the line) and its wider points from real x86 registers: `add al, bl`,
`add ax, bx`, `add eax, ebx`, `add rax, rbx`. Note that the *state* is 96 bits
from `al` upward whatever the operand: the disassembler widens each operand to
its full architectural register. What changes along the sweep is the length of
the carry chain, which is the thing under test.

Flags are not a separate run. Every case scores its flag-register flows next to
its data flows and reports them on their own line, so the CF/ZF/SF/OF row of
Table 3 is read off the arithmetic cases.

## What to look at

`results/<stamp>/SUMMARY.md`. Its columns, left to right:

- **Dataflows**: how many the algebra requires, `w(w+1)/2` per source operand
  for an additive instruction, `w` for a bitwise one.
- **Silent on forced chains**: of those, how many do not fire on a state
  constructed to force them. Deterministic; this is the strongest column.
- **Random: always/partial/never**, of the dataflows a random state exercises at
  all, how many always fire, fire only sometimes, or never fire.
- **Never exercised**: dataflows no random draw reached. Large in the wide
  cases by construction, and that is the point.
- **Held-out under-taint**: the sampled cross-check, which also covers the flags.

The claim is that the verdict is `correct` on bit-moving, logic, control flow and
4-bit arithmetic, and `UNSOUND` from 8-bit arithmetic up. The logs name the
individual misses:

```txt
missed  EBX[4] -> EAX[5], EAX[6], EAX[7], EFLAGS[11]=OF, EFLAGS[7]=SF
```

with `predicted: ['EAX[4]', 'EBX[4]']` in the matching JSON entry. This is
`add al, bl` at `al = 0x9c`, `bl = 0xe3`: the sum is `0x7f`, and flipping `bl[4]`
makes it `0x9c + 0xf3 = 0x8f`. Output bits 4, 5, 6 and 7 all change, and the sign
and overflow flags change with them. The rule fires the dataflow into bit 4 and
not the ones into 5, 6 and 7, even though it *contains* all four: their
conditions do not match this state.

## Measured on the reference machine

One `./run_rq1.sh` (default tier), AMD Ryzen 7 5700U, 32 GiB, 59 minutes,
32 held-out states per case and 64 random witness states per input bit:

| Family | Instruction | W | Dataflows | Silent on forced chains | Random: always/partial/never | Never exercised | Held-out under-taint | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bit-moving | `bswap esi` | 32 | - | - | - | - | 0 | correct (5 s) |
| control-flow | `jz .+16` | 32 | - | - | - | - | 0 | correct (16 s) |
| logic | `xor eax, ebx` | 32 | 64 | 0 | 64/0/0 | 0 | 0 | correct, over-taints ZF (689 s) |
| logic | `and eax, ebx` | 32 | 64 | 0 | 64/0/0 | 0 | 0 | correct, over-taints ZF (615 s) |
| logic | `or eax, ebx` | 32 | 64 | 0 | 64/0/0 | 0 | 0 | correct (132 s) |
| arithmetic | `ADD R1, R2` (JN) | 4 | 20 | 0 | 20/0/0 | 0 | 0 | correct (9 s) |
| arithmetic | `add al, bl` | 8 | 72 | 0 | 37/22/5 | 8 | 74 (138 bits) | **UNSOUND** (40 s) |
| arithmetic | `add ax, bx` | 16 | 272 | 0 | 52/87/49 | 84 | 220 (410 bits) | **UNSOUND** (67 s) |
| arithmetic | `add eax, ebx` | 32 | 1056 | 0 | 93/182/136 | 645 | 562 (1127 bits) | **UNSOUND** (786 s) |
| arithmetic | `sub eax, ebx` | 32 | 1056 | 447 | 85/202/123 | 646 | 551 (1018 bits) | **UNSOUND** (1174 s) |
| arithmetic | `add rax, rbx` | 64 | - | - | - | - | - | did not converge (>900 s) |

The 64-bit row is from a separate probe with a 900 s budget; `--full` reruns it
with the paper's one hour. The four-figure times on the 32-bit rows are almost
all espresso: the observation phase costs seconds and the DNF minimisation costs
minutes.

Read down the width sweep. At 4 bits the seeds are exhaustive and every one of
the 20 required dataflows always fires. At 8 bits the seeds are samples and 27 of
the 64 witnessed dataflows already fail. At 32 bits, 318 of the 411 witnessed
dataflows fail and a further 645 are never witnessed at all: the required set
grows quadratically with the operand width, and the seed budget does not.

## Traps

**A vacuous oracle passes everything.** `jz .+2` branches to its own
fall-through, so TaintInduce's jump detection (execute 100 times, see whether the
PC ever lands anywhere but `start + len(insn)`) never fires, EIP never enters the
state format, and the case scores flag-to-flag identity flows while looking like
it tests control flow. The runner uses `jz .+16` for that reason, and
`score_rule.py` refuses to return a verdict on zero flows: it prints `NO-DATA`.

**The inference is not deterministic, the scoring is.** `--holdout-seed` fixes
the ground-truth set, which is drawn in this process, so two runs score against
identical states. The *training* set is not fixed: observation generation runs
across a process pool and each worker draws from an RNG state inherited at fork,
so the seeds depend on how the work was chunked. Two runs of `add al, bl` on this
machine inferred rules with 106 and 96 under-tainted flows. The verdicts did not
move. Treat a small difference in flow counts as noise, not as a disagreement.

The structural check inherits that: it depends on the rule it is given, and the
rule depends on the training draw. Given a fixed rule it is exact, both the
constructed witnesses and the random ones (fixed RNG seed), so re-scoring a saved
rule with `--rule` reproduces its numbers to the digit. That is the way to
re-examine a result you find surprising.

## Tolerance

The verdicts are the claim, and they are stable, as is the shape of the triangle:
a full diagonal, a fringe of unreliable short carries, and an unreached corner.
Exact counts are not, for the reason above and because they scale with
`--holdout-states` and with the number of random witness states. A rerun that
finds under-taints on bit-moving, logic or control flow, or none on 8-bit and
wider arithmetic, is a genuine disagreement with the paper and worth reporting.

Wall-clock times are machine-dependent, and the non-convergence rows are
budget-dependent by construction: `did not converge (>3600s)` asserts that an
hour was not enough, not that any particular number of seconds is the right cut.

## Not reproduced here

The microtaint side of Table 3, the synthesis timing (`<1 s` per opcode, 0.57 s
for the whole 522-opcode x86 corpus, cold), is not run from this directory yet.
