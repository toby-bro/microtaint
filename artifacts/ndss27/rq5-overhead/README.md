# End-to-end overhead (RQ5, Figure 8)

What a real program costs under microtaint, emulator included: `bench.c` reads
256 tainted bytes from stdin, runs 100 rounds of mixed XOR/SBOX/ROL/ADD (over
1 M instructions), then overflows a 192-byte stack buffer. It is run 100 times
in three configurations: native, Qiling with no taint, and Qiling with all four
detectors on.

```sh
uv run python overhead_bench.py --build-bench bench.c --gen-input 256 --runs 100 \
    --only native --only qiling-only --only microtaint-all \
    --native-timeout 5 --qiling-timeout 120 --microtaint-timeout 1800 \
    --json overhead_results.json
```

About 15 minutes. Use that invocation as written: the positional `binary`
argument is `argparse.REMAINDER`, so `overhead_bench.py bench.elf --gen-input
256` hands those flags to the guest program instead, runs once with no tainted
input, and reports a time about eighty times too low.

`overhead_results.json` gets three entries, and the claim is two ratios:

- `microtaint-all.extra.run_s / qiling-only.extra.run_s`, the cost of taint with
  the emulator's own cost divided out. This is the meaningful one.
- `microtaint-all.wall_s / native.wall_s`, the overhead over native.

`run_s` is the `ql.run()` phase alone. `wall_s` on this benchmark is dominated
by Python startup and module import, which has nothing to do with taint.

Absolute times are machine-dependent; the ratio against `qiling-only` is the
figure the paper's claim is stated in.
