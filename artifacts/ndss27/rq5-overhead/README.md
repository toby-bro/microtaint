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
whose cost that is, and the single number invites the wrong reading. Run

```sh
uv run python overhead_ladder.py bench.elf --gen-input 64 --runs 3 \
    --json overhead_ladder.json
```

which runs the same workload under a ladder that each time adds exactly one
thing, and measures two reference points besides. A representative result
(3,768,369 guest instructions):

| layer | ns/instr | marginal | x qiling | attributable to |
|---|---|---|---|---|
| native | 0.5 | | 0.0 | baseline |
| qiling-only | 23.5 | +23.0 | 1.0 | emulator |
| blockhook (empty per-block hook) | 140.1 | +116.7 | 6.0 | emulator |
| microtaint-none | 1559.8 | +1419.7 | 66.5 | microtaint |
| microtaint-all | 1677.0 | +117.2 | 71.5 | microtaint |
| *codehook* (empty per-instruction Python hook) | *1966.9* | | *83.8* | *reference* |
| *codehook-regs* (same, reading 4 registers) | *19820.1* | | *844.7* | *reference* |

Three things the ladder separates that the single ratio does not:

- **The emulator is 23 ns/instr**, and merely *registering* a hook costs
  another 117 before any callback body runs, because Unicorn stops chaining
  translation blocks once one is installed. Neither is attributable to taint
  analysis; any tool in this stack pays both.
- **An empty per-instruction Python hook costs 1967 ns/instr.** microtaint,
  doing full bit-precise propagation, costs 1560: *less than being called and
  returning immediately*. A Python hook that merely reads four registers costs
  12.7x microtaint's entire engine. The cost of per-instruction control in this
  stack dominates the cost of the analysis done with it.
- **The four detectors add 8%.** Almost all the cost is propagation, so
  disabling checks is not a speed-up worth having.

The last two rows are **reference points, not rungs**: microtaint uses a C hook
and does not stand on the Python ones. Subtracting a reference row from a
microtaint row is meaningless, and the script keeps them out of the marginal
column for that reason (when they were in it, the marginal went negative, which
is how the mistake announces itself).

What this does *not* excuse: 1560 ns/instr is still 66x the emulator, and the
tail belongs to microtaint. The ladder locates the cost, it does not dissolve it.
