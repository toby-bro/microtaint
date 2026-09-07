# Taint-density benchmarks

`bench_r10` taints essentially every value it touches. That is the worst case,
and on its own it is misleading: it says nothing about the case that actually
dominates real targets, where a program reads a little untrusted input and then
spends the overwhelming majority of its instructions on untainted work (loop
counters, table setup, pointer arithmetic on clean data).

These two fill that gap, so a change can be judged on both ends of the range:

| binary              | tainted work                             |
|---------------------|------------------------------------------|
| `bench_dense.c`     | essentially every value is tainted        |
| `bench_sparse.c`    | ~1.5% of mixing iterations read taint     |
| `bench_untainted.c` | taint is injected and then never used     |

`bench_sparse` and `bench_untainted` do the same total amount of guest work;
only the taint density differs. `bench_dense` is the heavy-propagation case
(the old `bench_r10`), kept so both ends of the range are measured together: a
change that helps one end can easily cost the other.

## Build and run

    make                                   # builds all three .elf
    python run_bench.py bench_dense.elf 64 5   # best-of-5 wall clock
    python taint_stats.py bench_dense.elf      # where the work went

`run_bench.py` says whether a change was faster. `taint_stats.py` says whether
it was faster for the reason you thought: fast-path coverage, how many
instructions were dismissed as untainted, cache hit rate, cell re-executions,
and the breakdown of why instructions left the C path. Watch both -- a change
that barely moves the clock but drops fast-path coverage has quietly pushed
work back onto the slow path.

## Why they exist

They are what justified the untainted-input fast exit in
`microtaint/emulator/fastpath.h`. Before it, the fully untainted program still
cost 645 ms and ran 124,154 SLEIGH cell re-executions to compute taint that was
provably zero. After it: 248 ms and 41,340.

They also drove the implicit-taint check into C: `taint_stats.py` showed that
99% of Python fallbacks were PC-writing instructions (branches) taking the slow
path purely for a check that had nothing to decide, because the PC taint was
zero. Fixing that took the untainted case from 233ms to 163ms.

The residue is memory instructions, which the untainted exit cannot yet dismiss
because deciding whether a load's source is tainted needs the effective address.
That is the next thing these benchmarks are for.

Soundness of the exit is not measured here; it is gated by
`tests/test_untainted_exit.py`, which asserts over the whole instruction bank
that no form invents taint from a clean input state.
