# End-to-end applications (RQ7)

Three scenarios. The first is the memory-safety battery every production taint
engine is expected to support, where microtaint matches them. The other two are
where bit precision does something the coarser engines cannot: attributing a
constant-time leak to one key bit, and separating two fields packed in one byte.

- [`memory-safety/`](./memory-safety/) buffer overflow, use after free, side
  channel, arbitrary indexed write.
- `crypto/square_and_multiply/` constant-time verification (Table 7).
- `dns/` DNS bit-field side channel (Table 8).
- [`other-engines/`](./other-engines/) the same two analyses against angr, Maat,
  Triton, libdft64, TaintGrind and PANDA, which is what turns "coarser engines
  cannot" into a measurement.

```sh
cd crypto/square_and_multiply
uv run python check_side_channel.py      # ~5 min, is there a leak
uv run python localise_side_channel.py   # ~5 min, which key bit

cd ../../dns
uv run python dns_experiment.py          # ~8 min
```

Both engines agree with every baseline on the *binary* verdict, so that is not
the claim. On square-and-multiply the exponent is consumed one bit per
iteration, so tainting bit `k` alone should flag the branch at step `k+1` and
nowhere else; a byte- or register-granular engine taints the whole exponent and
flags all 32 steps. On DNS, `AND AL,0x78` provably masks `QR` away before the
shift, so tainting `QR` must give a clean verdict and tainting `OPCODE` a leak;
an engine with no sub-byte source reports a leak either way, which is a false
positive.

These are findings, not measurements: the leaking step and the tainted field are
either identified or they are not.

## Outputs

Each script takes an optional flag that writes its verdict as JSON, so these
numbers reach the paper by processing rather than by retyping:

```sh
uv run python check_side_channel.py    --json ct_check.json     # both variants
uv run python localise_side_channel.py --json ct_localise.json  # per-bit steps
uv run python dns_experiment.py        --json dns.json          # per-run verdicts
uv run microtaint --json --check-bof --input input.bin -- ./bof.elf > bof.json
```

Every document carries the engine's version, commit and dirty flag, so a result
names what produced it. The memory-safety detectors put findings on stdout and
progress on stderr, so redirecting stdout alone gives a parseable document
without losing the narrative.

**The detectors' exit code is a result, not a status.** The CLI returns 0 for no
security findings, 1 for at least one, and 2 for a usage error. Each of the four
targets contains exactly one planted bug, so 1 is the expected outcome and 0
means the detector missed it. A harness that reads non-zero as failure files
every successful detection as a failure, and one that reads 0 as success goes
green precisely when the detector has stopped working. `run-all.sh` checks for
the expected code rather than for zero.
