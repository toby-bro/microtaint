# Builds the nft_byteorder harness used as one of the three avalanche workloads.
# Static and -no-pie so every instruction the engine sees is in this binary,
# -O0 so the disassembly at a reported PC still matches the C source.
CFLAGS := -O0 -static -no-pie -fno-stack-protector -fno-common

nftables_harness: harness.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f nftables_harness
