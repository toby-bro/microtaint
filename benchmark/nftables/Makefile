# Build the two harness binaries used by the python detector.
#
#   make            — builds the unpatched (vulnerable) harness
#   make harness_fixed — builds the patched harness (no bug)
#   make all        — builds both
#   make clean      — removes both
#
# The detector script resolves symbol addresses from `nm`, so the binary
# must be static and -no-pie. -O0 is used to keep the loop body close to
# the C source (helps when reading the disassembly at the reported PC).

CC      ?= gcc
CFLAGS  := -O0 -static -no-pie -fno-stack-protector -fno-common
TARGETS := harness harness_fixed

all: $(TARGETS)

harness: harness.c
	$(CC) $(CFLAGS) -o $@ $<

harness_fixed: harness_fixed.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean
