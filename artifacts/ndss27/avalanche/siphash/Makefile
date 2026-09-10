# SipHash-2-4 over 16 tainted bytes of stdin: the third avalanche workload, and
# the target of eval_siphash_avalanche.py. Static and -no-pie so the whole hash
# runs inside the hooked binary.
CFLAGS := -O0 -static -no-pie -fno-stack-protector

siphash_bin: test_avalanche_siphash.c siphash_ref.c
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -f siphash_bin
