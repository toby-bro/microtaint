# Rule synthesis vs observation-based inference (RQ1, Table 3)

TaintInduce fuzzes I/O pairs and infers a DNF rule per bit-to-bit flow. The claim
is that this does not converge where microtaint's synthesis does: carry chains,
flags, and wide operands, where the observations needed to pin the rule down grow
faster than they can be collected.

## Which TaintInduce

There is none left. The original
([melynx/taintinduce](https://github.com/melynx/taintinduce)) targets Python 3.6
and imports `squirrelflowdb`, gone from PyPI for years; more bugs surface once it
starts. We evaluate a repaired fork,
[toby-bro/taintinduce](https://github.com/toby-bro/taintinduce): dependency and
porting work (Python 3.12, rewritten serialization, fixes to the x86/AMD64/ARM64
backends and to memory operands). The inference algorithm is the paper's.

`setup_taintinduce.sh` pins
[`64a703a`](https://github.com/toby-bro/taintinduce/commit/64a703a2ce539d725bc4dbf3d90394fb8c9443eb),
not the branch tip: everything after it is the work that grew into microtaint,
and `master`'s entry point no longer runs the DNF inference at all. `64a703a` is
the last state of the tree that is still TaintInduce.

### Disclaimer

While there were many logic bugs in the initial codebase we consider it fair to use our version with as many of these bugs patched.
Some of these bugs made the original submission of taintinduce questionnable here are a few:

- **They separated the conditions they generated on dataflows from the dataflows**: it is therefore impossible to claim they understand and model the conditionnal dataflows.
- **The boolean minimisation of the conditions only happened on a subset of the affected bits (e.g. flags)**: they can't capture real data dependencies if they are excluding most relevant bits from the condition.
- **They redefine what is an instruction**: they claim precision on millions of instructions, but they consider `and rax, 0x0`, `and rax, 0x1` to be two separate instructions, making it very easy to reach precision on more than $2^64$ instructions.
Instructions with only one register are, for the overwhelming majority, without any conditionnal flows.
Instructions with two registers (or memory values) on the other hand are way more complex, and on these taintinduce does not manage to converge, this is what we want to prove.

We did not want to shame taintinduce's authors in the main paper, so we decided to spare them and keep these doubts in this README in the artifacts.

These conclusions were found after months of work, whilst we honestly wanted to fix and improve taintinduce, and suprised us a lot.
Our efforts to make it work and the evaluated taintinduce presented in this paper is therefore far better than any implementation of taintinduce released before in terms of speed and correctness.

### Licensing

After having shown me where taintinduce could be found, the authors did not reply when asked about the licensing of their work. This puts our work in a legal grey-zone as most of the codebase has been rewritten (from git's perspective).
We nevertheless would like to insist once more on the importance of putting a LICENSE file in the artifacts, especially if they are to be published.
This repository does not distribute TaintInduce only provide scripts that can enable the user to install it, so is legal.

### Just Nibbles (JN) ISA

In order to understand what happend in TaintInduce easily, we introduced a very simple ISA with 4-bit general purpose registers, a few basic operands on these (ADD, AND, XOR...), and arm-style NZCV flags, it is used to produce very simple cases in the following evaluation, for which ground truth can be computed.

## Run

```sh
./setup_taintinduce.sh          # clone, pin, install, smoke-test; ~2 min
./run_rq1.sh --quick            # the fast families, ~13 min
./run_rq1.sh                    # everything but the 64-bit probe, ~1 h
./run_rq1.sh --full             # everything, an hour of budget for the last row
```

Needs an x86-64 host: TaintInduce minimises its DNFs with espresso, shipped as a
prebuilt x86-64 ELF.

Results land in `results/<stamp>/`: a `.log` and `.json` per case, the inferred
rules under `rules/`, and `SUMMARY.md` with the table. Inference is the expensive
half, so re-examining a saved rule is free and exact:

```sh
uv run --project external/taintinduce python score_rule.py \
    --arch X86 --bytes 00d8 --name 'add al, bl' \
    --rule results/<stamp>/rules/00d8_X86_rule.json
```

## The check

For an addition the required dataflow structure is not a matter of opinion: a
carry out of bit *i* travels upward, so bit *i* of either operand reaches output
bit *j* for every *j* >= *i*. That is `w(w+1)/2` dataflows per source operand,
a number that grows quadratically with the operand width. A bitwise
instruction needs the diagonal instead.
`score_rule.py --carry-width W --carry-shape triangle|diagonal` asks three
questions, in increasing order of what they prove:

1. **Is the `(i, j)` pair in the rule at all?** Purely structural. There are no
   holes, at every width swept, for `add`, `sub` and `xor` alike. The failure
   is not a missing dataflow, which is worth knowing before concluding anything
   else.
2. **Does it fire on a state that forces the carry?** Each pair carries a DNF
   condition, and that is what breaks. For each `(i, j)` the harness builds a
   state that forces the dependency (one 1 at bit *i*, a run of 1s across
   *i..j-1*), executes it on the real CPU, keeps it only if the dependency shows
   up, then asks the rule. On `sub eax, ebx` a substantial fraction of the
   constructed chains do
   not fire. Deterministic, and it reruns identically.
3. **Does it fire on random states?** The realistic case. For `add eax, ebx`,
   only a minority of the required dataflows ever turn up in a random draw at
   all, and of those only some always fire; the rest fire sometimes or never.

TaintInduce relies on fuzzing to detect dependencies between bits.
In our initial work to get TaintInduce working we realised that the cases/seeds it generated were far too few, and poor, so we added many more in the hope of managing to "fix" the additions, without succeeding.
This is why for additions the results are far better than substractions, even if still unsufficient.

`add` records no silent chains in (2) while `sub` records many, and the
asymmetry is in the
witnesses, not the rules: the constructed chains are built for addition, so on
`add` they land on the shapes TaintInduce's own `Bitwalk` and `BitFill` seed
strategies produce, and the rule has them. Read the `o` cells below as the
friendly case, not a pass.

Together: TaintInduce recovers the *structure* of addition, then guards each
dataflow with a condition fitted to the states it happened to sample.

## The triangle

`--carry-width` draws the `(i, j)` grid. `add al, bl`:

```txt
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

Short carries are learned, medium ones are unreliable, long ones are wrong, and
the far corner was never exercised at all. That corner is the paper's Case 2 in
visual form: a random addition propagates a carry through bit *k* only when
`a[k] xor b[k]` is 1, so a carry of distance *d* appears with probability about
`2^-(d-1)`, and past *d* ~ 7 it does not appear in a sample of any practical
size, and at the widest width swept it covers the majority of the cells.

## Held-out scoring, for the outputs with no shape to check

To not evaluate taintinduce on the cases it was fitted on, we make a second draw with an independant seed.

Verdicts are `correct`, `correct (over-taints flags)`, `sound, over-taints data`,
`UNSOUND`, `did not converge (>Ns)`, and `NO-DATA` when the oracle produced no
flows at all. Two extra lines report the structural check: `STRUCTURE` for pair
existence, `FORCED-CARRY` for whether they fire.

`correct (over-taints flags)` is Table 3's "correct", and the over-taint it
tolerates is the paper's Case 3 in front of you: on `xor eax, ebx` every spurious
flow is an input bit claiming to reach ZF. ZF is set exactly when the result is
zero, a predicate with no short DNF over the input bits, so the inference cannot
express "only when the result happens to be zero" and falls back to an
unconditional flow. Safe direction, but it is the representational ceiling.

## The one number that explains the rest

TaintInduce enumerates the state space below `2**14` states and samples above it
(`observation.py`, `_gen_seeds`). Every case prints its side of that line:

```txt
state: 12 bits (R1:4 R2:4 NZCV:4) -> seeds are EXHAUSTIVE      // JN
state: 96 bits (EFLAGS:32 EAX:32 EBX:32) -> seeds are SAMPLED  // X86
```

Below it the inference has seen every input and is right by construction. Above
it, it generalises, and carry chains are what it fails to generalise.

The sweep therefore takes its 4-bit point from JN ("Just Nibbles", the fork's toy
ISA: two 4-bit registers and a 4-bit flag register, 12 state bits) and its wider
points from real x86 registers. The state is 96 bits from `al` upward whatever
the operand, since the disassembler widens each operand to its architectural
register; what changes along the sweep is the length of the carry chain, which is
the thing under test. Flags are not a separate run: every case scores its flag
flows next to its data flows, so Table 3's CF/ZF/SF/OF row is read off the
arithmetic cases.

## Running the sweep

`./run_rq1.sh` (default tier) sweeps the families below. *Dataflows* is what the
algebra requires; *silent* is how many do not fire on a state built to force
them; *never exercised* is how many no random draw reached. The script prints
the table; **the paper is the authority on every value in it.**

| Family | Instruction | Width |
| --- | --- | --- |
| bit-moving | `bswap esi` | 32 |
| control-flow | `jz .+16` | 32 |
| logic | `xor eax, ebx` | 32 |
| logic | `and eax, ebx` | 32 |
| logic | `or eax, ebx` | 32 |
| arithmetic | `ADD R1, R2` (JN toy ISA) | 4 |
| arithmetic | `add al, bl` | 8 |
| arithmetic | `add ax, bx` | 16 |
| arithmetic | `add eax, ebx` | 32 |
| arithmetic | `sub eax, ebx` | 32 |
| arithmetic | `add rax, rbx` | 64 |

The 64-bit row is a separate probe under its own budget; `--full` reruns it with
the paper's hour. Run times are dominated by DNF minimisation, not observation.

What to read off the sweep, qualitatively: at the narrowest width the seeds are
EXHAUSTIVE and every required dataflow fires. As the width grows the seeds become
samples, a growing share of the witnessed dataflows already fails, and a growing
share is never witnessed at all. The required set grows quadratically with
operand width while the seed budget does not, which is the whole point of the
experiment: the bit-moving, control-flow and logic families stay correct at every
width, and only the arithmetic family degrades.

The logs name individual misses:

```txt
missed  EBX[4] -> EAX[5], EAX[6], EAX[7], EFLAGS[11]=OF, EFLAGS[7]=SF
```

That is `add al, bl` at `al = 0x9c`, `bl = 0xe3`: the sum is `0x7f`, flipping
`bl[4]` makes it `0x8f`, and bits 4-7 all change along with SF and OF. The rule
fires the dataflow into bit 4 and not the ones into 5, 6 and 7, although it
contains all four.

## Cost

Measured on sixteen cores with CPU boost disabled.

`./run_rq1.sh --quick` takes about ten minutes, which is what `run-all.sh
--quick` runs. `./run_rq1.sh` without arguments takes about an hour, and
`--full` adds the 64-bit non-convergence probe and can take an hour longer.

Almost all of that is espresso, the DNF minimiser: minimisation on a 96-bit x86
state costs minutes per instruction, while the observation phase costs seconds.
Setting up TaintInduce is a two-minute clone and install, mostly downloading
wheels, and needs an x86-64 host because espresso ships as a compiled binary.
