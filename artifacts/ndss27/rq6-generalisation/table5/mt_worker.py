#!/usr/bin/env python3
# ruff: noqa: W505, E501, RUF100, I001
#   RUF100 and I001 are config differences, not defects: the source repo
#   selects BLE001/E402/C901 (so those noqa ARE used there) and sorts
#   `microtaint` as third-party, while it is first-party here.  No single
#   spelling satisfies both repos, and the body must stay byte-identical
#   to the harness that produced the published numbers.
#   Style only, and suppressed rather than rewritten: this harness is
#   vendored from the campaign that produced the published numbers, and
#   e.g. adding zip(strict=True) would change behaviour where the
#   original silently truncated.  Correctness is gated by
#   validate_table5_oracle.py, not by restyling proven code.
# Vendored verbatim from the soundness-campaign harness; see README.md.
# Experiment script, not library code: see artifacts/ndss27/README.md,
# "Lint and type checking", for why annotations are not required here.
# mypy: disable-error-code="no-untyped-def, no-untyped-call, type-arg, no-any-return, var-annotated, assignment, index, arg-type, union-attr, operator, attr-defined, misc, call-overload, return-value, unreachable"
"""Batched MicroTaint worker for mt_multiarch.py.

stdin  <- {"mt_arch": "ARM64", "regs": [["x0",64],...,["NG",1],...], "cases": [{bytes,state,taint}]}
stdout -> [{reg: taint_mask}, ...]   aligned with cases

Runs microtaint-only (no Unicorn), so it is stable at scale in its own process.
"""
import json
import sys

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, ImplicitTaintPolicy, Register


def main():
    payload = json.load(sys.stdin)
    arch = getattr(Architecture, payload['mt_arch'])
    fmt = [Register(n, b) for n, b in payload['regs']]
    names = [n for n, _ in payload['regs']]
    # The stack pointer is part of the state (so STORE/LOAD addresses resolve) but is
    # never a checked output and never carries taint.
    # NOTE: no shadow_memory is passed.  STORE writes taint into the state dict as
    # MEM_<hex>_<size> (LogicCircuit.evaluate), while MemoryOperand READS from
    # shadow_memory whenever one is present and only falls back to that dict
    # otherwise.  Supplying an empty shadow therefore hides the stored taint.
    # ChainedCircuit already threads MEM_ keys between steps, so the dict path is
    # the correct mechanism for single-instruction and short-chain propagation;
    # shadow memory belongs to whole-system emulation, where a real image exists.
    sp = payload.get('sp')
    if sp:
        fmt.append(Register(sp[0], sp[1]))
    sim = CellSimulator(arch)
    zero = dict.fromkeys(names, 0)
    out = []
    rule_cache = {}
    for c in payload['cases']:
        code = bytes.fromhex(c['bytes'])
        h = c['bytes']
        rule = rule_cache.get(h)
        if rule is None:
            rule = generate_static_rule(arch, code, fmt)
            rule_cache[h] = rule
        vals = {**zero, **{k: int(v) for k, v in c['state'].items()}}
        if sp:
            vals[sp[0]] = sp[2]
        ctx = EvalContext(
            input_taint={**zero, **{k: int(v) for k, v in c['taint'].items()}},
            input_values=vals,
            simulator=sim,
            implicit_policy=ImplicitTaintPolicy.IGNORE,
        )
        res = rule.evaluate(ctx)
        out.append({n: int(res.get(n, 0) or 0) for n in names})
    json.dump(out, sys.stdout)


if __name__ == '__main__':
    main()
