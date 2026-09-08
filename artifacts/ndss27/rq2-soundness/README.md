# RQ2 — Soundness against runtime engines

**Claim (§6.2).** microtaint is sound on every x86-64 test case in the suite —
no untainted output bit can be flipped by flipping a tainted input bit — and it
is the only engine of the seven for which that holds.

This is the paper's central claim. It is also the most expensive to reproduce,
because it needs all six baselines.

## Run

```sh
cd ../../../benchmark/precision_soundess
./setup_envs.sh                 # once; see INSTALL.md step 4
uv run python benchmark.py      # ~3 h, dominated by container life-cycle
```

`benchmark.py` orchestrates one worker per engine (`worker_angr.py`,
`worker_maat.py`, `worker_triton.py`, `worker_panda.py`, `worker_microtaint.py`)
over the shared corpus, and scores every engine against the same ground truth.

An engine whose environment is missing is **skipped and recorded as skipped**,
so a partial install yields a partial table rather than a crash. Check the
summary's engine list before concluding anything about a comparison.

## The corpus and the ground truth

9,858 tests over 127 mnemonics: random single instructions, short sequences,
systematic sweeps with five canonical taint masks, and hand-curated cases
(`imul` edge cases, real-world idioms, bug seeds, a path-explosion stress).

For the 3,263 tests with at most 15 tainted bits — about 33% — every input
combination is enumerated on Unicorn to obtain **exact** taint. That subset is
where soundness and precision are scored, because only there is the truth known
rather than approximated.

Byte-granular engines (TaintGrind, libdft64, PANDA) promote tainted bits to
whole bytes; this is accounted for in scoring and is why they are sound-but-
imprecise rather than unsound.

## What to look at

Per engine: cases scored, unsound cases, and the specific failures. The claim is
**zero unsound cases for microtaint** and non-zero for every other engine.

## Tolerance

Exact. Fixed corpus, fixed masks, enumerated ground truth. A rerun that finds an
unsound microtaint case is a genuine disagreement with the paper and worth
reporting.

## Regenerating the paper's numbers

```sh
uv run python gen_paper_macros.py REPORT.json --overhead ../overhead/overhead_results.json
```

writes the LaTeX macros the paper's tables read, so the numbers in the text and
the numbers in the run cannot drift apart silently.
