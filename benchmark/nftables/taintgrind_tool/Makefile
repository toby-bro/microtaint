# Build the taintgrind-instrumented harness variants.
#
# These are byte-for-byte identical to ../harness.c and ../harness_fixed.c
# EXCEPT they #include "taintgrind.h" and call TNT_TAINT() on the 83 stdin
# bytes right after fread(), defining that memory as the attacker taint
# source. nft_byteorder_eval and the [before|regs|after] layout are verbatim.
#
# TNT_TAINT expands to a VALGRIND_DO_CLIENT_REQUEST_STMT, which is pure inline
# asm (a special no-op instruction sequence) -- no linking against valgrind is
# needed, only the valgrind.h / taintgrind.h headers at compile time. The host
# valgrind headers (3.25) are ABI-compatible with the docker image (3.25.1);
# the client-request magic sequence is identical across valgrind 3.x.
#
#   make            -- builds both harness_tg.bin and harness_tg_fixed.bin
#   make clean      -- removes them

CC        ?= gcc
# Same flags as ../Makefile: static + no-pie so `nm` gives runtime addresses.
CFLAGS    := -O0 -static -no-pie -fno-stack-protector -fno-common
# taintgrind.h lives here; it #includes "valgrind.h" from the host valgrind.
INCLUDES  := -I. -I/usr/include/valgrind
TARGETS   := harness_tg.bin harness_tg_fixed.bin

all: $(TARGETS)

harness_tg.bin: harness_tg.c taintgrind.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $<

harness_tg_fixed.bin: harness_tg_fixed.c taintgrind.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean
