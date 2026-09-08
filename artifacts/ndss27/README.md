# microtaint — NDSS 2027 artifact

This artifact reproduces the evaluation of *microtaint*, a taint-rule synthesis
framework that turns machine-readable ISA semantics into per-instruction
**taint-propagation circuits** and evaluates them at run time without a solver.

Everything here runs against **tag `v0.6.15`**, which is the engine the paper
describes: one taint implementation, reached through `generate_static_rule` and
`circuit.evaluate`. Later versions of microtaint add a second, faster
implementation with *different precision*; it is not what the paper evaluates,
and using it would reproduce different numbers. `INSTALL.md` pins the tag.

---

## Claims this artifact supports

Each claim maps to one experiment directory. The paper's research questions are
numbered as in §6.

| # | Claim (paper §) | Experiment | Needs | Runtime |
|---|---|---|---|---|
| RQ1 | Synthesis produces sound rules where observation-based inference does not converge (§6.1) | `rq1-synthesis-vs-inference/` | TaintInduce | ~30 min |
| RQ2 | The synthesised rules are sound on every x86-64 test case, where the six baseline engines are not (§6.2) | `rq2-soundness/` | all baselines | ~3 h |
| RQ3 | Highest bit-level precision among sound engines (§6.3) | `rq3-precision/` | all baselines | shares RQ2's run |
| RQ4 | Per-step propagation cost is competitive, and independent of operand width (§6.4) | `rq4-per-step-cost/` | microtaint only | ~20 min |
| RQ5 | End-to-end overhead on real programs under full-system emulation (§6.5) | `rq5-overhead/` | microtaint only | ~15 min |
| RQ6 | The synthesis generalises unchanged to ARM64, MIPS64, PPC32 and RV64GC (§6.6) | `rq6-generalisation/` | microtaint only | ~45 min |
| RQ7 | Enables analyses coarser engines cannot do: constant-time checking and bit-field side channels (§6.7) | `rq7-applications/` | microtaint only | ~20 min |

**Total, all experiments: about five hours** on the reference machine, within
the one-day budget. `RQ4`–`RQ7` need nothing but microtaint and take about
100 minutes together; start there if you are checking *Functional* rather than
*Reproduced*.

Each experiment directory contains a `README.md` with: the exact command, what
it writes, the number in the paper it corresponds to, and the tolerance within
which a rerun should agree.

## What is and is not deterministic

Soundness and precision counts are **exact**: the corpus is fixed, the taint
masks are fixed, and the ground truth is an exhaustive enumeration. A rerun that
disagrees on those is a real disagreement, not noise.

Timings are **not** exact. They depend on the machine, and the paper's are from
an AMD Ryzen 7 5700U at ~3 GHz. Ratios between engines are far more stable than
absolute numbers, and the per-experiment READMEs say which ratio each claim
rests on.

## Layout

    README.md            this file
    INSTALL.md           getting to a working environment
    REQUIREMENTS.md      hardware, software, and what each baseline needs
    LICENSE
    run-all.sh           every experiment, in order, into results/
    rq1-.../ ... rq7-.../ one directory per claim
    results/             where runs land (created on first run)
