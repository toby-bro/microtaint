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

clean:
	rm -f $(BINS)

.PHONY: all clean
