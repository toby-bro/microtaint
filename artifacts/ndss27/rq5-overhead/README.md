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

Roughly 5 minutes. Add `--build-bench bench.c` instead of the positional
`bench.elf` to rebuild the guest first.

## Read this before trusting any number here

The published version of this experiment measured nothing. `bench.c` opens with

```c
long n = sys_read(0, state, INPUT_SIZE);
if (n <= 0) { sys_exit(1); }
```

so with no stdin the guest exits at the first check and the 3.77 M instructions
never run. The helpers wrap `ql.run()` in `except Exception: pass`, the recorded
return code is the *helper's* and not the guest's, and microtaint installs its
per-instruction hook lazily, only once taint exists. Every one of those failures
is silent, and all of them are *fast*: the result was three clean sub-100 ms
timings whose ratio, 1.58x, went into the paper as the end-to-end taint
overhead. It was the ratio of two Python startup paths.

The arithmetic that catches it: 67.7 ms over 3.77 M instructions is 18 ns per
instruction, and bare Unicorn alone costs about 22. A taint engine cannot run
below the emulator it runs inside.

Three checks now make that failure loud instead of silent, and the harness
**refuses to run** rather than report a number it cannot stand behind:

- **`guest_bytes`** — `bench.c` writes its 8-byte hash *after* the mix rounds
  and *before* the overflow, so a non-empty count proves the workload ran. Each
  helper redirects fd 1 for the duration of `ql.run()` and reports the count.
- **`instr_hook_registered`** — false means no taint was ever propagated.
- **empty stdin** — refused up front, with the explanation above.

Use `--allow-vacuous` only to debug the harness. Results produced with it are
not publishable, and `gen_paper_macros.py` rejects a JSON without `guest_bytes`.

Note that a *timing* threshold would never have caught any of this. The check
has to be on what the guest did, not on how long it took.

## What the numbers mean

`--instr-count` runs the guest once more under a counting hook (untimed, and
deliberately so: the hook costs ~2.2 us per instruction, about a hundred times
bare Qiling) so every `run_s` can be divided into ns per propagation step.

- **`x_qiling_run`** = `microtaint-*.run_s / qiling-only.run_s`, the cost of
  taint with the emulator's own cost divided out. This is the meaningful one,
  and it is `run_s` over `run_s`. Do not compute it from `wall_s`: wall carries
  0.2 to 0.4 s of Python import and Qiling init that have nothing to do with
  taint and dilute the ratio about fivefold (5.6x against 25.8x on the same
  data). `gen_paper_macros.py` used to divide the walls; it no longer does.
- **`ns_per_instr`** = `run_s` per guest instruction. The number to sanity-check
  against the ~22 ns/instr floor.
- **`x_native_wall`** = `microtaint-all.wall_s / native.wall_s`.

All six detector configurations are measured, not just `microtaint-all`, so the
propagation cost and the detector cost are separable: `microtaint-none` is
propagation only.

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

A representative result (3,768,369 guest instructions, idle machine, 5 runs):

| layer | ns/instr | marginal | x qiling | attributable to |
|---|---|---|---|---|
| native | 0.5 | | 0.0 | baseline |
| qiling-only | 24.2 | +23.7 | 1.0 | emulator |
| c-blockhook (pure-C per-BLOCK hook, empty) | 24.7 | +0.5 | 1.0 | emulator |
| c-codehook (pure-C per-INSTRUCTION hook, empty) | 45.6 | +20.9 | 1.9 | emulator |
| c-codehook-regs (+ read 4 guest registers) | 102.9 | +57.3 | 4.3 | emulator |
| microtaint-plumbing (engine armed, nothing tainted) | 495.2 | +392.3 | 20.5 | **plumbing** |
| microtaint-none | 1162.1 | +666.9 | 48.1 | **propagation** |
| microtaint-all | 1290.7 | +128.6 | 53.4 | **detectors** |
| *blockhook* (empty per-block Python hook) | *139.4* | | *5.8* | *reference* |
| *codehook* (empty per-instruction Python hook) | *2037.5* | | *84.2* | *reference* |
| *codehook-regs* (same, reading 4 registers) | *20370.9* | | *842.3* | *reference* |
| *spawn* (fork+exec of a do-nothing binary) | *0.2* | | *0.0* | *reference* |

**`c-blockhook` settles what block chaining costs: +0.5 ns/instr.** Losing
Unicorn's translation-block chaining is free in practice, so the +21 that
`c-codehook` adds is per-instruction DISPATCH and nothing else. An earlier
reading of this ladder blamed chaining for the empty per-block PYTHON hook's
139 ns/instr; that was CPython, not chaining, and the C rung is what
distinguishes them.

The three Python rows are slow to measure and cannot move when the engine
changes, so `--only` re-runs just the chain after an optimisation; their figures
above are from the 2026-09-11 full run.

**`microtaint-plumbing` is the rung that makes the rest attributable.** It is
the real engine with the hook armed on every instruction, but with every taint
source stubbed out, so nothing is ever tainted and no propagation happens. What
remains is the hook entry, reading the operands the instruction actually uses,
the address-to-circuit cache, the prefilter test, and for a memory operand an
effective-address computation and a shadow lookup. That is work a byte-granular
taint engine owes per instruction however its propagation is written.

It reports `residual_register_taint` and fails the run if any register ended
tainted, because "nothing was tainted" is the rung's whole claim. It does NOT
gate on the prefilter share: about 5% of instructions are prefilter-ineligible
by structure (a PC target needs the implicit-taint decision, a memory write
needs the store path) and evaluate even on a clean machine, so a threshold there
would be measuring circuit shape rather than taint.

**Per-instruction hooking is not what costs.** A pure-C `UC_HOOK_CODE` with an
empty body is 48 ns/instr, only +23 over bare emulation, and that already
includes losing Unicorn's translation-block chaining. Reading the four guest
registers a taint engine needs brings it to 104. So the plumbing a
per-instruction dynamic analysis cannot avoid is about 4x the emulator, not 60x.

**The rest splits roughly in half.** Of microtaint-none's 1188 ns/instr, 510 is
plumbing and 574 is propagation, so the taint algebra is about 48% of the
engine's cost rather than all of it. Both halves are ours; they are just
different work, and only the propagation half is affected by making the taint
formulas cheaper.

**The hosting language dominates the analysis.** The same empty hook written in
Python costs 1963 ns/instr, 41x the C one, and more than microtaint's entire
engine. A Python hook that merely reads four registers costs 19,705 ns/instr,
12x microtaint doing full bit-precise propagation. Any tool that hooks every
instruction from Python is slow for reasons unrelated to what it computes.

**The detectors add 8%,** so disabling checks is not a speed-up worth having.

The Python rows are **reference points, not rungs**: microtaint uses a C hook
and does not stand on them, so subtracting them is meaningless. An earlier
version had them in the chain and the marginal column went negative, which is
how that mistake announces itself. `ladder_hooks.c` counts its own invocations
and the harness checks that count against the instruction count, because a C
hook that silently never fired would read as "free".
