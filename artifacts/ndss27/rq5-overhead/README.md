# End-to-end overhead (RQ5, Figure 8)

What a real program costs under microtaint, emulator included. `bench.c` reads
tainted bytes from stdin, runs 100 rounds of mixed XOR/SBOX/ROL/ADD, then
overflows a 192-byte stack buffer. The workload is **3,768,369 guest
instructions** (count it yourself with `--instr-count`).

```sh
uv run python overhead_bench.py --gen-input 64 --runs 5 --instr-count \
    --native-timeout 10 --qiling-timeout 300 --microtaint-timeout 900 \
    --json overhead_results.json bench.elf
```

Add `--build-bench bench.c` instead of the positional `bench.elf` to rebuild the guest first.

## Detectors

The experiment here produces the table of the appendix, of which a reduced version is presented in the main content of the paper.

We evaluate many different steps as a form of ladder in which we add the different parts constitutive of microtaint, so as to show where are the actual bottlenecks of microtaint, and show precisely how much costs the taint propagation.

## What the numbers mean

`--instr-count` runs the guest once more under a counting hook (untimed, and
deliberately so: the hook costs ~2.2 us per instruction, about a hundred times
bare Qiling) so every `run_s` can be divided into ns per propagation step.

- **`x_qiling_run`** = `microtaint-*.run_s / qiling-only.run_s`, the cost of
  taint with the emulator's own cost divided out. This is the meaningful one,
  and it is `run_s` over `run_s`. Do not compute it from `wall_s`: wall carries
  0.2 to 0.4 s of Python import and Qiling init that have nothing to do with
  taint and dilute the ratio severalfold on the same data.
- **`ns_per_instr`** = `run_s` per guest instruction. The number to sanity-check
  against the bare-emulator floor, which the `qiling-only` rung measures.
- **`x_native_wall`** = `microtaint-all.wall_s / native.wall_s`.

All six detector configurations are measured, not just `microtaint-all`, so the
propagation cost and the detector cost CAN be separated in principle:
`microtaint-none` is propagation only. Whether they separate in PRACTICE depends
on the spread being larger than the run-to-run noise, which in the shipped run
it is not.

Absolute times are machine-dependent. Take them only from an otherwise idle
machine: this is one Python process and its numbers move by more than a factor
of two under load, in every phase including plain module import.

## Input size

`--gen-input 64` gives a clean exit and times pure propagation. `--gen-input
256` additionally triggers the deliberate stack overflow, which is what
`--check-bof` detects; expect the guest to fault or spin on the corrupted return
address afterwards, so give it a generous timeout. Either way the mix rounds run
in full, because `state` is 256 bytes regardless of how many were read.

## Whose overhead is it? (`overhead_ladder.py`)

`overhead_bench.py` says how much slower the whole thing is. It does not say
whose cost that is, and the single number invites the wrong reading. Build the
pure-C hooks once, then run the ladder:

```sh
gcc -O2 -fPIC -shared -o ladder_hooks.so ladder_hooks.c
uv run python overhead_ladder.py bench.elf --gen-input 64 --runs 3 \
    --json overhead_ladder.json
```

### The rungs, and what each one isolates

Each rung adds exactly one layer, so the DIFFERENCE between two adjacent rungs
is attributable to the thing that was added:

| layer | what it adds | attributable to |
|---|---|---|
| `native` | nothing; the guest run directly | baseline |
| `qiling-only` | the emulator | emulator |
| `c-blockhook` | a pure-C per-BLOCK hook, empty body | emulator |
| `c-codehook` | a pure-C per-INSTRUCTION hook, empty body | emulator |
| `c-codehook-regs` | the same, reading the four guest registers a taint engine needs | emulator |
| `microtaint-plumbing` | the real engine, armed, with every taint source stubbed out | **plumbing** |
| `microtaint-none` | propagation, no detectors | **propagation** |
| `microtaint-all` | the detectors | **detectors** |
| *`blockhook`* | an empty per-block PYTHON hook | *reference* |
| *`codehook`* | an empty per-instruction PYTHON hook | *reference* |
| *`codehook-regs`* | the same, reading four registers | *reference* |
| *`spawn`* | fork+exec of a do-nothing binary | *reference* |

**No figures are quoted in this file, deliberately.** Timing numbers belong with
the run that produced them, and the `overhead_ladder.json` and
`overhead_results.json` checked in beside this README predate the provenance
schema the scripts now write: neither carries an `engine` key or a `cpu` key, so
neither can be attributed to an engine version or to a machine with CPU boost
off. An earlier version of this README quoted a table that the shipped JSON
contradicts, including a marginal cost for the detectors that the data gives as
NEGATIVE. Rather than restate figures that cannot be checked, run the ladder and
read the table it prints; every current JSON records the engine commit, the
dirty flag and the boost state, so the numbers arrive attributable.

Both scripts REFUSE to run with CPU boost enabled (`ALLOW_CPU_BOOST=1`
overrides, and the state is recorded either way), because a boosted machine
makes the rungs incomparable.

**`c-blockhook` settles what block chaining costs.** Compare it against
`qiling-only`: losing Unicorn's translation-block chaining turns out to be close
to free, so whatever `c-codehook` adds on top is per-instruction DISPATCH and
nothing else. An earlier reading of this ladder blamed chaining for the empty
per-block PYTHON hook's cost; that was CPython, not chaining, and the C rung is
what distinguishes them.

The three Python rows are slow to measure and cannot move when the engine
changes, so `--only` re-runs just the chain after an optimisation and carries
the previous run's Python rows forward. When it does, the JSON records which run
each rung came from: a table mixing rungs from two runs is only meaningful if it
says so.

**`microtaint-plumbing` is the rung that makes the rest attributable.** It is
the real engine with the hook armed on every instruction, but with every taint
source stubbed out, so nothing is ever tainted and no propagation happens. What
remains is the hook entry, reading the operands the instruction actually uses,
the address-to-circuit cache, the prefilter test, and for a memory operand an
effective-address computation and a shadow lookup. That is work a byte-granular
taint engine owes per instruction however its propagation is written.

It reports `residual_register_taint` and fails the run if any register ended
tainted, because "nothing was tainted" is the rung's whole claim. It does NOT
gate on the prefilter share: a small fraction of instructions are
prefilter-ineligible by structure (a PC target needs the implicit-taint
decision, a memory write needs the store path) and evaluate even on a clean
machine, so a threshold there would be measuring circuit shape rather than
taint.

**Per-instruction hooking is not what costs.** Compare `c-codehook` against
`qiling-only`: a pure-C `UC_HOOK_CODE` with an empty body adds a modest amount
over bare emulation, and that already includes losing translation-block
chaining. `c-codehook-regs` then adds reading the four guest registers a taint
engine needs. Those two rungs bound what ANY per-instruction dynamic analysis
owes before it computes anything, and it is a small multiple of the emulator
rather than the order of magnitude the end-to-end number suggests.

**The rest splits into plumbing and propagation**, and the split is what this
ladder exists to measure: `microtaint-plumbing` minus `c-codehook-regs` is the
plumbing, `microtaint-none` minus `microtaint-plumbing` is the taint algebra.
Both halves are ours; they are just different work, and only the propagation
half is affected by making the taint formulas cheaper. Read the proportion off
your own run rather than from here, because it has already been quoted wrongly
once.

**The hosting language dominates the analysis.** Compare `codehook` against
`c-codehook`: the same empty hook written in Python costs more than an order of
magnitude more, and more than microtaint's entire engine. `codehook-regs`, which
merely reads four registers from Python, costs several times what microtaint
spends doing full bit-precise propagation. Any tool that hooks every instruction
from Python is slow for reasons unrelated to what it computes.

**The detectors' marginal cost is at or below this experiment's noise floor.**
In the shipped run the six detector configurations all fall within half a
percent of one another, and `microtaint-all` measures very slightly FASTER than
`microtaint-none`, which is not a real speed-up but a difference this setup
cannot resolve. The honest statement is that disabling the checks buys nothing
measurable here, not that they cost a specific percentage; `gen_paper_macros.py`
now refuses to emit that percentage rather than printing a negative one.

The Python rows are **reference points, not rungs**: microtaint uses a C hook
and does not stand on them, so subtracting them is meaningless. An earlier
version had them in the chain and the marginal column went negative, which is
how that mistake announces itself. `ladder_hooks.c` counts its own invocations
and the harness checks that count against the instruction count, because a C
hook that silently never fired would read as "free".

## Cost

Measured on sixteen cores with CPU boost disabled.

| step | time |
| --- | --- |
| the ladder, `--quick` | 58 min |
| the end-to-end benchmark, `--quick` | 4 min |

The ladder is the single longest step in a quick run, because it measures each
rung repeatedly to separate configurations whose costs are close together.
Memory is modest, a few hundred MB.

**Disable CPU boost before running this.** The rungs are compared against each
other, so they must all be measured at the same clock, and both scripts refuse
to run with boost enabled. Even with boost off, the reduced corpus cannot
separate the detectors from the noise: that figure needs the full corpus.
