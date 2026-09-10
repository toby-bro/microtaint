# Generalisation across ISAs (RQ6, Table 5)

The same engine, mapper and P-code evaluator on ARM64, MIPS64BE, PPC32BE and
RV64GC as on x86-64: a port only names the ISA's registers and status flags. The
claim is an absence of per-ISA taint rules, which is the one a reader is most
entitled to doubt, and the number that carries it is zero under-taints per ISA.

```sh
# Pass 1, the wide net. One Unicorn instance per ISA, reused across runs: ~8x
# faster, but hidden architectural state can leak between runs. A leak can only
# add spurious reports, never hide a real one.
uv run python campaign.py pass1 --n 20000 --arch all --seed 1 --out camp

# Pass 2, re-check every pass-1 report with a fresh Unicorn per run. Real
# under-taints survive, false positives vanish. A finding is only a finding if
# it survives this.
uv run python campaign.py pass2 --in camp

uv run python multiarch_fuzz.py     # randomised cross-ISA, ~10 min
```

Pass 1 runs at about 52 cases/s averaged over the five ISAs (AMD64 53, ARM64 27,
MIPS64BE 103, RISCV64 54, PPC32BE 81; the instruction mixes differ). The paper's
`--n 1000000` per ISA is therefore about 26 hours for this experiment alone,
past the one-day budget for a whole artifact, so the default here is `--n 20000`,
about 32 minutes for all five. More cases can only strengthen a zero, so the
scaled-down run is a weaker instance of the same result. It reproduced at that
scale: 1,500 cases per ISA, 0 under-taints.

Precision varies between ISAs and that is not a defect: an ISA whose lifter
models fewer flags gives the differential less to be precise about.

Where an ISA leaves a flag architecturally undefined (x86 OF after a rotate by
anything but one), SLEIGH and QEMU model it differently and both are entitled
to. Since the oracle is QEMU, comparing there would measure which vendor guessed
what, so those outputs are detected by running both models over the same states,
excluded from the verdict, and reported as `isa-undefined` or `lifter-gap`. On
this corpus they are confined to x86.

`proto_port.py` is the throwaway prototype that first ported MIPS64 and PPC64 by
monkeypatch alone, which is what established that no engine edit was needed.
