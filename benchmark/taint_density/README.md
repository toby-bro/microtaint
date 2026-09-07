# Taint-density benchmarks

`bench_r10` taints essentially every value it touches. That is the worst case,
and on its own it is misleading: it says nothing about the case that actually
dominates real targets, where a program reads a little untrusted input and then
spends the overwhelming majority of its instructions on untainted work (loop
counters, table setup, pointer arithmetic on clean data).

These two fill that gap, so a change can be judged on both ends of the range:

| binary             | tainted work                          |
|--------------------|---------------------------------------|
| `bench_sparse.c`   | ~1.5% of mixing iterations read taint |
| `bench_untainted.c`| taint is injected and then never used |

Both do the same total amount of guest work; only the taint density differs.

## Build

    gcc -O1 -static -nostdlib -fno-stack-protector -o bench_sparse.elf bench_sparse.c
    gcc -O1 -static -nostdlib -fno-stack-protector -o bench_untainted.elf bench_untainted.c

## Run

    python run_bench.py bench_untainted.elf 64 5    # best-of-5 wall clock

## Why they exist

They are what justified the untainted-input fast exit in
`microtaint/emulator/fastpath.h`. Before it, the fully untainted program still
cost 645 ms and ran 124,154 SLEIGH cell re-executions to compute taint that was
provably zero. After it: 248 ms and 41,340.

The residue is memory instructions, which the exit cannot yet dismiss because
deciding whether a load's source is tainted needs the effective address. That is
the next thing these benchmarks are for.

Soundness of the exit is not measured here; it is gated by
`tests/test_untainted_exit.py`, which asserts over the whole instruction bank
that no form invents taint from a clean input state.
