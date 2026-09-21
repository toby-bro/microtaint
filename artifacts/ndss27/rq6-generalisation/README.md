# Generalisation across ISAs (RQ6, Table IV)

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

Throughput differs per ISA because the instruction mixes differ.  The paper's
per-ISA case count takes far longer than the one-day budget for a whole
artifact, so the default here is a reduced `--n`,
about eight minutes for all five. More cases can only strengthen a zero, so the
scaled-down run is a weaker instance of the same result. It reproduced at that
scale, and reports no under-taints.

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

## The cross-ISA table is a different, longer experiment

The campaign above is a fast soundness smoke test over about 80 instruction
forms: it answers "does anything under-taint", cheaply, and it is what
`run-all.sh` runs. Table IV needs the full per-ISA corpus and
the precision columns, and it takes hours. That harness lives in
[`table5/`](./table5/) with its own README, launcher and validator.

## Outputs

Everything is written next to `--out`, so `--out camp` produces:

| file | what it holds |
| --- | --- |
| `camp_run.json` | the run manifest: engine version, commit, dirty flag, `n`, seed, ISA list, start time. Written before the first case, so an interrupted run still names the engine that produced it. |
| `camp_<arch>.jsonl` | one JSON object per raw pass-1 report, one file per ISA. **An empty file is the expected result**, meaning zero under-taints. |
| `camp_coverage.json` | per-ISA coverage counters: `compared`, `effective`, `skipped_oracle`, `skipped_mt`, `no_gt_taint`. |
| `camp_<arch>_REAL.jsonl` | pass 2 only: the reports that survived re-checking with a fresh Unicorn. |

Read `camp_coverage.json` before quoting a zero. `--n` counts loop iterations,
not comparisons: a case whose oracle or engine raises is skipped, and a case
with no ground-truth taint compares nothing, so the nominal `--n` overstates
what was actually checked. The gap is not uniform across ISAs, and MIPS64BE has
the lowest effective fraction on this corpus. A zero means "no under-taint in
the cases that were genuinely compared", and the second half of that sentence is
in the coverage file.

## Cost

Measured on sixteen cores with CPU boost disabled.

`run-all.sh --quick` runs 20000 cases per ISA and takes about eight minutes.
The full corpus is 1000000 cases per ISA, fifty times as many.

The standalone cross-ISA campaign in `table5/` is open-ended instead: it takes a
number of hours as an argument and runs the five ISAs in parallel until that
deadline. For scale, a 24 hour run on a six-core machine covered 89.8 million
cases. Memory is modest, a few hundred MB per ISA worker, and the only
significant disk use is the per-ISA report.

## Regenerating the cross-ISA table

`table5/table5.py` builds the paper's cross-ISA table from the campaign shards:

```sh
cd table5 && uv run python table5.py --tex table5.tex --json table5.json
```

It reads `campaign_<isa>.json` from its own directory, so it describes the
`run_table5.sh` campaign and not the RQ6 pass in `run-all.sh`, which uses a
different harness and a different file format. `--dir` points it elsewhere,
which is how our own 24 hour run is read back:

```sh
uv run python table5.py --dir ../../reference-runs/rq6-campaign-ryzen5-3600-24h
```

It refuses to emit a table in which any ISA measured nothing, and deletes the
output files rather than leaving a paste-ready table behind on a refusal.

`run-all.sh` calls it when the campaign has been run, and records a skip
otherwise.
