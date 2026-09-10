"""Drive the C one-pass per-op taint composer over the instruction bank.

Two jobs, both permanent:

  * VALIDATION -- every instruction the pass answers must agree with the engine
    oracle, and must never under-taint against it.  A decline is a legitimate
    answer ("fall back to the monolithic differential"); a wrong answer is not.

  * MEASUREMENT -- how many primitive operations does propagating taint through
    a whole instruction actually cost, flags included?  That number is the thing
    being optimised, and the end goal (compiling the taint circuit to native
    code) is bounded by it, so it is measured per ISA over the whole corpus
    rather than estimated from a few examples.

Run directly for a report:

    .venv/bin/python -m tests.perop_c_bank --isas AMD64 ARM64

The pytest gate lives in tests/test_perop_c_soundness.py; this module is the
harness both it and the ratchet import.
"""
from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Callable

from microtaint.instrumentation.ast import LogicCircuit
from microtaint.instrumentation.cell_c.cell_c import PCodeCellEvaluatorC
from microtaint.types import Architecture, Register

if TYPE_CHECKING:                    # imported inside the run for import cost
    from tests.oracle_harness import UcDesc, Verdict

#: A per-output mask keyed by register or flag name.
TaintState = dict[str, int]

class Ref(StrEnum):
    """Which reference a sweep judges the engine against.

    A StrEnum, like the rest: a caller may still pass the spelling and a
    report may still print it, while the sweep compares members.
    """

    #: The whole-instruction differential.  A bit-exact gate.
    DIFFERENTIAL = 'differential'
    #: Unicorn per-bit sensitivity.  Soundness and precision, and the
    #: only correct gate where the two paths may legitimately differ.
    GROUND_TRUTH = 'ground_truth'

#: What one pass over an instruction costs, by category.  Keys are fixed by
#: `OpStats.add`: ops, pcode_ops, n_route, n_diff, n_floor, n_cube, reads,
#: writes, forks.
Cost = dict[str, int]

#: The signature `run_bank_perop_c(step=...)` drives.  Returning three things
#: rather than taint alone is what lets the bank check the VALUES a pass
#: computes as well as the taint it routes.
Step = Callable[[Architecture, bytes, list[Register], TaintState, TaintState],
                tuple[TaintState, TaintState, Cost]]


class Declined(Exception):
    """The per-op pass does not model this instruction's p-code shape."""


_EVALS: dict[str, PCodeCellEvaluatorC] = {}


def _evaluator(arch: Architecture) -> PCodeCellEvaluatorC:
    key = arch.value if hasattr(arch, 'value') else str(arch)
    ev = _EVALS.get(key)
    if ev is None:
        ev = PCodeCellEvaluatorC(arch)
        _EVALS[key] = ev
    return ev


# Bank/human register spellings that differ from the engine's SLEIGH geometry.
# AArch64 condition flags are the case that matters: the bank writes N/Z/C/V,
# Ghidra's register file calls them NG/ZR/CY/OV.  Feeding the bank spelling
# straight through drops the flag VALUES, so a predicated instruction (csel,
# csinc) silently takes the same branch whatever the caller set -- which reads
# as an engine under-taint when it is only a name that did not resolve.
_NAME_ALIASES = {'N': 'NG', 'Z': 'ZR', 'C': 'CY', 'V': 'OV'}


def _engine_names(arch: Architecture, names: Iterable[str]) -> dict[str, str]:
    """bank name -> engine geometry name, for the names that differ."""
    key = arch.value if hasattr(arch, 'value') else str(arch)
    cached = _NAME_MAPS.get(key)
    if cached is None:
        cached = {}
        _NAME_MAPS[key] = cached
    missing = [n for n in names if n not in cached]
    if missing:
        # Extend rather than rebuild: two bank sections can share an
        # architecture and spell the same registers differently (ARM64 uses
        # X0/N/Z/C/V, ARM64_SIMD uses x0/NG/ZR), and a cache keyed only by
        # architecture would hand the second section the first one's names.
        from microtaint.instrumentation.cell import _build_reg_maps
        offsets = _build_reg_maps(arch)[0]
        for n in missing:
            if n in offsets:
                cached[n] = n
            elif _NAME_ALIASES.get(n) in offsets:
                cached[n] = _NAME_ALIASES[n]
            elif n.upper() in offsets:
                cached[n] = n.upper()
            else:
                cached[n] = n          # unresolved: taint_step ignores it
    return cached


_NAME_MAPS: dict[str, dict[str, str]] = {}


def perop_c_step(arch: Architecture, code: bytes, regs: list[Register],
                 in_taint: TaintState, in_values: TaintState,
                 ) -> tuple[TaintState, TaintState, Cost]:
    """One pass: returns (taint_by_reg, value_by_reg, cost).  Raises Declined."""
    names = [r.name for r in regs]
    alias = _engine_names(arch, names)
    ev = _evaluator(arch)
    res = ev.taint_step(code,
                        {alias[n]: in_values.get(n, 0) for n in names},
                        {alias[n]: in_taint.get(n, 0) for n in names},
                        [alias[n] for n in names])
    if res is None:
        raise Declined(code.hex())
    td, vd, cost = res
    return ({n: td.get(alias[n], 0) for n in names},
            {n: vd.get(alias[n], 0) for n in names}, cost)


def perop_c_taint(arch: Architecture, code: bytes, regs: list[Register],
                  in_taint: TaintState, in_values: TaintState,
                  *, circuit: LogicCircuit = None) -> TaintState:
    """Oracle-harness adapter: taint only."""
    del circuit
    return perop_c_step(arch, code, regs, in_taint, in_values)[0]


@dataclass
class OpStats:
    """Op-count distribution over the instructions the pass answers."""

    counts: list[int] = field(default_factory=list)      # primitive taint ops
    pcode: list[int] = field(default_factory=list)       # p-code ops in the program
    by_class: dict[str, int] = field(default_factory=lambda: {'route': 0, 'diff': 0,
                                                    'floor': 0, 'cube': 0})
    worst: list[tuple[int, str]] = field(default_factory=list)   # (ops, label)

    def add(self, label: str, cost: Cost) -> None:
        self.counts.append(cost['ops'])
        self.pcode.append(cost['pcode_ops'])
        self.by_class['route'] += cost['n_route']
        self.by_class['diff'] += cost['n_diff']
        self.by_class['floor'] += cost['n_floor']
        self.by_class['cube'] += cost['n_cube']
        self.worst.append((cost['ops'], label))

    def quantile(self, q: float) -> int:
        if not self.counts:
            return 0
        s = sorted(self.counts)
        return s[min(len(s) - 1, int(q * len(s)))]

    def summary(self) -> str:
        if not self.counts:
            return 'no answered instructions'
        n = len(self.counts)
        mean = sum(self.counts) / n
        tot = sum(self.by_class.values()) or 1
        return (f'n={n} ops mean={mean:.1f} p50={self.quantile(0.5)} '
                f'p95={self.quantile(0.95)} p100={max(self.counts)} | '
                f'rules route={100*self.by_class["route"]/tot:.0f}% '
                f'diff={100*self.by_class["diff"]/tot:.0f}% '
                f'floor={100*self.by_class["floor"]/tot:.0f}% '
                f'cube={100*self.by_class["cube"]/tot:.0f}%')


@dataclass
class BankReport:
    n_instrs: int = 0
    n_answered: int = 0
    n_declined: int = 0
    n_cases: int = 0
    n_exact: int = 0
    n_over: int = 0
    n_under: int = 0
    # Under-taint the CURRENT engine does not also have.  This is the gate:
    # under(per-op, truth) must be a subset of under(differential, truth), so
    # anything here is a regression this work introduced, while shared
    # under-taint is a pre-existing lifter/ISA gap (x86 leaves AF after cmp and
    # OF after a multi-bit shift undefined; SLEIGH models them as unchanged and
    # Unicorn computes something, so both engines "miss" the same bits).
    n_under_new: int = 0
    new_under_examples: list[tuple[str, TaintState, TaintState, TaintState]] = (
        field(default_factory=list))
    # vs the whole-instruction differential (the engine's current answer).
    n_tighter: int = 0     # per-op reports LESS taint and ground truth agrees
    n_looser: int = 0      # per-op reports MORE taint than the oracle
    n_same: int = 0
    under_examples: list[tuple[str, TaintState, TaintState, Verdict]] = (
        field(default_factory=list))
    over_examples: list[tuple[str, Verdict]] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)
    #: instruction -> the outputs left out of its verdict because the ISA does
    #: not define them.  Reported, never silently dropped: an exclusion nobody
    #: sees is indistinguishable from a bug nobody found.
    undefined_outputs: dict[str, list[str]] = field(default_factory=dict)
    #: instruction -> flags the LIFTER never writes, so the engine cannot taint
    #: them however sound it is.  Excluded for the same reason and reported
    #: apart, because unlike the above this one IS a gap somebody could close.
    unmodelled_outputs: dict[str, list[str]] = field(default_factory=dict)
    stats: OpStats = field(default_factory=OpStats)
    declined_labels: list[str] = field(default_factory=list)

    def summary(self) -> str:
        cov = 100 * self.n_answered / self.n_instrs if self.n_instrs else 0
        return (f'instrs={self.n_instrs} answered={self.n_answered} ({cov:.1f}%) '
                f'declined={self.n_declined} | cases={self.n_cases} '
                f'exact={self.n_exact} over={self.n_over} '
                f'under={self.n_under} UNDER-NEW={self.n_under_new} '
                f'isa-undefined={len(self.undefined_outputs)} '
                f'lifter-gap={len(self.unmodelled_outputs)} '
                f'errors={len(self.errors)}')

    def vs_oracle(self) -> str:
        return (f'vs whole-instruction differential: same={self.n_same} '
                f'tighter={self.n_tighter} looser={self.n_looser}')


_UC_DESC = {'AMD64': '_uc_desc_amd64', 'ARM64': '_uc_desc_arm64',
            'RISCV64': '_uc_desc_riscv64'}

MASK64 = 0xFFFFFFFFFFFFFFFF


_REG_MAPS: dict[str, dict[int, str]] = {}


def _offset_to_name(arch: Architecture) -> dict[int, str]:
    """Invert the engine's register geometry: frame byte offset -> name."""
    key = arch.value if hasattr(arch, 'value') else str(arch)
    inv = _REG_MAPS.get(key)
    if inv is None:
        from microtaint.instrumentation.cell import _build_reg_maps
        offsets, _sizes = _build_reg_maps(arch)[:2]
        inv = {}
        for name, off in offsets.items():
            inv.setdefault(off, name)
        _REG_MAPS[key] = inv
    return inv


def written_registers(arch: Architecture, code: bytes) -> set[str]:
    """Names of the registers this instruction's p-code actually writes.

    The comparison against Unicorn has to be restricted to these.  SLEIGH does
    not model every flag every instruction touches -- x86 `cmp` never computes
    AF, `imul` never computes SF/PF (the ISA leaves them undefined) -- so the
    real CPU moves bits the p-code has no expression for.  Scoring those as
    under-taint would blame the taint rules for a gap in the lifter, and would
    do it identically for the whole-instruction differential, which reads the
    same p-code.  Comparing only what the p-code writes keeps the gate on the
    thing under test.
    """
    from microtaint.sleigh.lifter import get_context
    key = arch.value if hasattr(arch, 'value') else str(arch)
    inv = _offset_to_name(arch)
    rev_alias = {v: k for k, v in _NAME_ALIASES.items()}
    out: set[str] = set()
    for op in get_context(key).translate(code, 0x1000).ops:
        vn = op.output
        if vn is not None and vn.space.name == 'register':
            for b in range(vn.size):
                n = inv.get(vn.offset + b)
                if n:
                    out.add(n)
                    if n in rev_alias:
                        out.add(rev_alias[n])
    return out


def uc_initial_state(desc: UcDesc) -> TaintState:
    """The register state Unicorn starts an instruction from, for everything the
    ground-truth driver does NOT write.

    It only writes `desc.gp`, so the flags (and anything else) keep Unicorn's
    reset values.  The per-op pass, by contrast, starts from whatever the caller
    hands it -- so handing it random flag values makes a predicated instruction
    (cmov, csel) take a different branch than the reference did, and the
    disagreement is the harness's, not the engine's.  Read the real reset state
    once and seed both sides from it.
    """
    import unicorn
    uc = unicorn.Uc(desc.uc_arch, desc.uc_mode)
    out: TaintState = {}
    if desc.eflags_reg is not None:
        ef = uc.reg_read(desc.eflags_reg)
        out['EFLAGS'] = ef
        for fname, bit in desc.flags.items():
            out[fname] = (ef >> bit) & 1
    return out


def gt_vectors(desc: UcDesc, reg_names: list[str], rng: random.Random,
               n: int) -> Iterator[tuple[TaintState, TaintState]]:
    """Input vectors the per-bit ground truth can actually enumerate.

    Two constraints the generic fuzzer does not meet.  Unicorn ground truth
    flips one tainted bit at a time, so the total tainted-bit count sets the
    number of emulations -- keep it small.  And it can only VARY the registers
    in `desc.gp`, so taint anywhere else would be invisible to it and would
    score as spurious over-taint; restrict the taint to those registers.

    Values are drawn for every register so the value-aware rules (AND/OR
    masking, the carry corners) are exercised, not just the taint routing.
    """
    gp = list(desc.gp)
    init = uc_initial_state(desc)
    for i in range(n):
        t = dict.fromkeys(reg_names, 0)
        # 1..3 tainted bits spread over 1..2 of the tracked registers, plus one
        # whole-byte case so byte-granular routing (SUBPIECE/PIECE) is covered.
        if i % 4 == 3:
            t[rng.choice(gp)] = 0xFF << (8 * rng.randint(0, 3))
        else:
            for _ in range(rng.randint(1, 3)):
                t[rng.choice(gp)] |= 1 << rng.randint(0, 63)
        vals = {r: rng.randint(1, MASK64) for r in reg_names}
        vals.update(init)
        yield t, vals


def run_bank_perop_c(*, isas: list[str] | None = None, n_dense: int = 3,
                     n_sparse: int = 5, seed: int = 1234,
                     max_examples: int = 12, skip_mem: bool = True,
                     ref: Ref = Ref.GROUND_TRUTH,
                     step: Step | None = None) -> BankReport:
    """Sweep the bank; validate and collect op counts.

    `ref=Ref.GROUND_TRUTH` compares against Unicorn per-bit sensitivity, which is
    the only correct soundness gate here: the per-op pass is deliberately
    TIGHTER than the whole-instruction differential in places (it knows an
    interior carry cannot reach a flag), so scoring it against that differential
    reports genuine precision gains as under-taint.  `ref=Ref.DIFFERENTIAL` keeps
    the old comparison for measuring how the two answers differ.
    """
    from benchmark.instruction_bank import load_bank
    from tests import oracle_harness as oh

    if step is None:
        step = perop_c_step
    rep = BankReport()
    specs = load_bank(isas=set(isas) if isas else None)
    for spec in specs.values():
        arch_key = spec.arch.value if hasattr(spec.arch, 'value') else str(spec.arch)
        reg_names = [r.name for r in spec.regs]
        desc: UcDesc | None = None
        if ref is Ref.GROUND_TRUTH:
            maker = _UC_DESC.get(arch_key)
            if maker is None:
                continue                     # no per-bit truth for this ISA yet
            made: UcDesc = getattr(oh, maker)()
            desc = made
            base_keys = list(made.gp) + list(made.flags)
        else:
            base_keys = reg_names

        for ins in spec.instructions:
            try:
                circuit = oh.build_circuit(spec.arch, ins.bytes, spec.regs)
            except Exception as e:
                rep.errors.append((ins.label, f'build: {e!r}'))
                continue
            if skip_mem and oh._compile_and_mem(circuit, spec.arch, spec.regs):
                continue
            rep.n_instrs += 1

            if ref is Ref.GROUND_TRUTH:
                # Set above, or the ISA was skipped before the loop started.
                assert desc is not None
                try:
                    written = written_registers(spec.arch, ins.bytes)
                except Exception:
                    written = None
                # GP registers stay in scope even when untouched: their taint
                # must pass through unchanged, which is a real claim to check.
                keys = ([k for k in base_keys
                         if k in desc.gp or written is None or k in written])
            else:
                keys = base_keys

            rng = random.Random(f'{seed}:{ins.label}')
            if ref is Ref.GROUND_TRUTH:
                assert desc is not None
                vectors = list(gt_vectors(desc, reg_names, rng, n_sparse))
            else:
                vectors = list(oh.fuzz_vectors(reg_names, rng, n_dense, n_sparse))

            # Where the ISA declines to define a flag -- x86 OF after a rotate
            # by anything but one, AF after most arithmetic -- SLEIGH and QEMU
            # model it differently and both are entitled to.  Comparing the
            # engine's taint against QEMU there measures which vendor guessed
            # what, so those outputs leave the verdict.  Detected from the very
            # vectors about to be judged, not from a table: a table cannot cover
            # an ISA nobody has written one for.
            if ref is Ref.GROUND_TRUTH:
                assert desc is not None
                undefined, unmodelled = oh.models_disagree(
                    desc, spec.arch, ins.bytes, [v for _t, v in vectors])
                if undefined:
                    rep.undefined_outputs[ins.label] = sorted(undefined)
                if unmodelled:
                    rep.unmodelled_outputs[ins.label] = sorted(unmodelled)
                keys = [k for k in keys if k not in undefined | unmodelled]
            answered = False
            for in_taint, in_values in vectors:
                try:
                    got, _vals, cost = step(spec.arch, ins.bytes, spec.regs,
                                            in_taint, in_values)
                except Declined:
                    break
                except Exception as e:
                    rep.errors.append((ins.label, f'eval: {e!r}'))
                    break
                if not answered:
                    answered = True
                    rep.stats.add(ins.label, cost)
                rep.n_cases += 1
                try:
                    if ref is Ref.GROUND_TRUTH:
                        assert desc is not None
                        refd = oh.ground_truth(desc, ins.bytes, in_taint, in_values)
                    else:
                        refd = oh.reference_taint(spec.arch, ins.bytes, spec.regs,
                                                  in_taint, in_values, circuit=circuit)
                    oracle = oh.reference_taint(spec.arch, ins.bytes, spec.regs,
                                                in_taint, in_values, circuit=circuit)
                except Exception as e:
                    rep.errors.append((ins.label, f'ref: {e!r}'))
                    continue
                v = oh.classify(got, refd, keys)
                if v.exact:
                    rep.n_exact += 1
                elif v.sound:
                    rep.n_over += 1
                    if len(rep.over_examples) < max_examples:
                        rep.over_examples.append((ins.label, v))
                else:
                    rep.n_under += 1
                    if len(rep.under_examples) < max_examples:
                        rep.under_examples.append((ins.label, in_taint, in_values, v))
                    # Does the current engine miss the same bits?
                    ov = oh.classify(oracle, refd, keys)
                    new = {k: bits & ~int(ov.under.get(k, 0))
                           for k, bits in v.under.items()}
                    new = {k: b for k, b in new.items() if b}
                    if new:
                        rep.n_under_new += 1
                        if len(rep.new_under_examples) < max_examples:
                            rep.new_under_examples.append(
                                (ins.label, in_taint, in_values, new))

                vo = oh.classify(got, oracle, keys)
                if vo.exact:
                    rep.n_same += 1
                elif vo.over:
                    rep.n_looser += 1
                else:
                    rep.n_tighter += 1
            if answered:
                rep.n_answered += 1
            else:
                rep.n_declined += 1
                if len(rep.declined_labels) < 400:
                    rep.declined_labels.append(ins.label)
    return rep


def main(argv: list[str] | None = None) -> int:
    import argparse
    from collections import Counter

    ap = argparse.ArgumentParser()
    ap.add_argument('--isas', nargs='*', default=None)
    ap.add_argument('--dense', type=int, default=3)
    ap.add_argument('--sparse', type=int, default=5)
    ap.add_argument('--per-isa', action='store_true')
    ap.add_argument('--show-declines', type=int, default=0)
    ap.add_argument('-v', '--verbose', action='store_true')
    ap.add_argument('--ref', default='ground_truth',
                    choices=['ground_truth', 'differential'])
    args = ap.parse_args(argv)

    groups: list[list[str] | None]
    if args.per_isa and args.isas:
        groups = [[i] for i in args.isas]
    else:
        groups = [args.isas] if args.isas else [None]

    for g in groups:
        rep = run_bank_perop_c(isas=g, n_dense=args.dense,
                               n_sparse=args.sparse, ref=args.ref)
        tag = ','.join(g) if g else 'ALL'
        print(f'[{tag}] {rep.summary()}')
        print(f'[{tag}] {rep.stats.summary()}')
        print(f'[{tag}] {rep.vs_oracle()}')
        if rep.new_under_examples:
            print(f'[{tag}] *** NEW UNDER-TAINT (regressions) ***')
            for label, it, _iv, new in rep.new_under_examples[:8]:
                shown = {k: hex(b) for k, b in new.items()}
                tshown = {k: hex(x) for k, x in it.items() if x}
                print(f'    {label}: {shown}')
                print(f'        taint={tshown}')
        if rep.under_examples and args.verbose:
            print(f'[{tag}] under-taint shared with the current engine:')
            for label, it, iv, v in rep.under_examples[:8]:
                print(f'    {label}: {v}')
                print(f'        taint={ {k: hex(x) for k, x in it.items() if x} }')
                print(f'        values={ {k: hex(x) for k, x in iv.items() if x} }')
        if args.show_declines and rep.declined_labels:
            c = Counter(lbl.split()[0] for lbl in rep.declined_labels)
            print(f'[{tag}] top declined mnemonics: {c.most_common(args.show_declines)}')
        if rep.stats.worst:
            worst = sorted(rep.stats.worst, reverse=True)[:8]
            print(f'[{tag}] most expensive: ' +
                  ', '.join(f'{lbl}={n}' for n, lbl in worst))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
