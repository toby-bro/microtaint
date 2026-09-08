# RQ1 — Synthesis vs observation-based inference

**Claim (§6.1).** microtaint synthesises sound rules in regimes where
observation-based inference (TaintInduce) does not converge — carry chains,
flags, and the zero-mask idiom, where the number of observations needed to
infer the rule grows faster than it is practical to collect.

## Status of this experiment in the artifact

**Not yet wired up.** TaintInduce needs its own environment and a comparison
harness that is not in this repository, and it is the one experiment here whose
setup has not been reproduced end to end. It is being prepared separately.

Everything else in this artifact is runnable; this directory is a placeholder so
the mapping from claims to experiments stays complete rather than silently
dropping RQ1.

## What it will need

- TaintInduce from source, in its own virtualenv
- the shared instruction corpus (already in `benchmark/instruction_bank/`)
- a comparison over the regimes named above, scoring convergence and soundness

Reviewers assessing *Functional* should not be blocked by this: RQ2-RQ7 exercise
every component of the system. Reviewers assessing *Reproduced* for RQ1
specifically should contact the authors.
