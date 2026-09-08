# Guest binaries for the taint-density benchmarks.
#
# Freestanding on purpose: pure syscalls, no libc, so nearly every executed
# instruction belongs to the binary itself and the engine's ranged hook sees the
# whole propagation chain rather than loader and libc noise.
#
# The optimisation level is part of each benchmark's definition, not a detail:
# it changes the instruction mix, and therefore what is being measured. Do not
# "tidy" these into one flag set -- numbers taken before and after would stop
# being comparable, silently.
#
#   bench_dense     -O0  keeps the mixing loops memory-heavy (the historical
#                        heavy-taint workload; -O0 reproduces its exact cell
#                        count, 268938, so older numbers remain comparable).
#   bench_sparse    -O1  as measured when the untainted-input exit was added.
#   bench_untainted -O1  likewise.
CC       ?= gcc
BASE     := -static -nostdlib -fno-stack-protector

BINS := bench_dense.elf bench_sparse.elf bench_untainted.elf

all: $(BINS)

bench_dense.elf: bench_dense.c
	$(CC) -O0 $(BASE) -o $@ $<

bench_sparse.elf: bench_sparse.c
	$(CC) -O1 $(BASE) -o $@ $<

bench_untainted.elf: bench_untainted.c
	$(CC) -O1 $(BASE) -o $@ $<

# The same workload on a second ISA, so a per-architecture number compares the
# engine rather than two different programs.  Not in $(BINS): it needs a cross
# compiler, and everything else here builds with the host one.  Build it with
# `make bench_untainted_arm64.elf` when a non-x86 measurement is wanted.
ARM64_CC ?= aarch64-linux-gnu-gcc

bench_untainted_arm64.elf: bench_untainted_arm64.c
	$(ARM64_CC) -O1 $(BASE) -o $@ $<

# The pure-C control: same emulation, hooks that are genuinely empty C
# functions, no Python anywhere.  It is what separates Unicorn's own
# per-instruction dispatch cost from the cost of the engine's callback glue --
# a distinction run_overhead.py alone cannot make, because every configuration
# it times has a Python interpreter in the process.  Links the venv's Unicorn so
# it measures the same library the engine runs against.
UC ?= ../../.venv/lib/python3.13/site-packages/unicorn

hookcost: hookcost.c
	$(CC) -O2 -o $@ $< -I$(UC)/include -L$(UC)/lib -l:libunicorn.so.2 \
		-Wl,-rpath,$(UC)/lib

clean:
	rm -f $(BINS)

.PHONY: all clean
