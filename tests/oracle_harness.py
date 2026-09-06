"""Oracle harness for the unify-taint-and-execution rework (Phase 0).

The safety net every rework phase is gated against. It provides TWO oracles and
a corpus driver over the multi-ISA instruction bank:

  1. reference_taint(...)  -- the CURRENT whole-instruction differential
     (circuit.evaluate).  This is the BIT-EXACT reference: interface/rep changes
     (Phases 1-2) must reproduce it exactly; the per-op compaction (Phase 3) is
     diffed against it to MEASURE precision drift.

  2. ground_truth(...)     -- per-bit Unicorn sensitivity (flip each tainted
     input bit, XOR outputs, OR).  The TRUE oracle: soundness = the engine mask
     must CONTAIN the ground-truth mask (no under-taint); over-taint is the
     precision cost, measured but allowed.

An `engine_fn` is any callable
    engine_fn(arch, code, regs, in_taint, in_values, *, circuit) -> dict
returning an output-taint dict keyed by register name.  `reference_taint` is
itself a valid engine_fn (used for harness self-test).  Future engines (array
rep, per-op compaction) plug in the same way.

This module is import-only (no test_ prefix); tests/test_oracle_harness.py runs a
bounded gate, and it can be driven standalone for full-bank sweeps:
    .venv/bin/python -m tests.oracle_harness --isas AMD64 --vectors 8
"""
# ruff: noqa: PLC0415
# mypy: disable-error-code="no-untyped-def,no-untyped-call,attr-defined,import-untyped,var-annotated"
from __future__ import annotations

import random
from dataclasses import dataclass, field

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, ImplicitTaintPolicy

MASK64 = 0xFFFFFFFFFFFFFFFF


# ---------------------------------------------------------------------------
# Oracle 1: the current whole-instruction differential (bit-exact reference).
# ---------------------------------------------------------------------------

def build_circuit(arch: Architecture, code: bytes, regs):
    """Generate the taint circuit for one instruction form (uncached, so a
    harness run never depends on LRU state)."""
    return generate_static_rule(arch, code, list(regs))


def reference_taint(arch, code, regs, in_taint, in_values, *, circuit=None):
    """Oracle 1: circuit.evaluate -- the current whole-instruction differential.
    Register-only (no shadow); memory forms are filtered by the corpus driver."""
    if circuit is None:
        circuit = build_circuit(arch, code, regs)
    sim = CellSimulator(arch)
    ctx = EvalContext(
        input_taint=dict(in_taint),
        input_values=dict(in_values),
        simulator=sim,
        implicit_policy=ImplicitTaintPolicy.IGNORE,
    )
    return circuit.evaluate(ctx)


# ---------------------------------------------------------------------------
# Engine adapters.  An engine_fn has the signature
#   (arch, code, regs, in_taint, in_values, *, circuit) -> out_taint dict
# reference_taint is one; this is the template for plugging in a new engine.
# ---------------------------------------------------------------------------

def engine_evaluate_c(arch, code, regs, in_taint, in_values, *, circuit=None):
    """The CURRENT C register fast path (CompiledCircuit.evaluate_c), falling
    back to the differential where it declines (None: mem / PC / wide).  Proves
    the harness detects a real (non-identity) engine matching the oracle, and is
    the shape every future engine adapter takes."""
    if circuit is None:
        circuit = build_circuit(arch, code, regs)
    sim = CellSimulator(arch)
    comp = getattr(circuit, '_compiled', None)
    if comp is None:
        # populate _compiled via one evaluate, then retry
        reference_taint(arch, code, regs, in_taint, in_values, circuit=circuit)
        comp = getattr(circuit, '_compiled', None)
    if comp is not None and comp is not False:
        out = comp.evaluate_c(dict(in_taint), dict(in_values), sim._pcode)
        if out is not None:
            return out
    return reference_taint(arch, code, regs, in_taint, in_values, circuit=circuit)


# ---------------------------------------------------------------------------
# Oracle 2: per-bit Unicorn sensitivity (true ground truth), per ISA.
# ---------------------------------------------------------------------------

@dataclass
class UcDesc:
    """Everything the Unicorn ground truth needs for one ISA."""
    uc_arch: int
    uc_mode: int
    code_addr: int
    gp: dict            # reg-name -> unicorn reg const (GP regs to track)
    flags: dict = field(default_factory=dict)  # flag-name -> bit index within eflags_reg
    eflags_reg: int | None = None              # unicorn const for the flags register
    mask: int = MASK64


def _uc_desc_amd64() -> UcDesc:
    import unicorn
    import unicorn.x86_const as ux
    return UcDesc(
        uc_arch=unicorn.UC_ARCH_X86, uc_mode=unicorn.UC_MODE_64, code_addr=0x1000,
        gp={'RAX': ux.UC_X86_REG_RAX, 'RBX': ux.UC_X86_REG_RBX,
            'RCX': ux.UC_X86_REG_RCX, 'RDX': ux.UC_X86_REG_RDX,
            'RSI': ux.UC_X86_REG_RSI, 'RDI': ux.UC_X86_REG_RDI},
        flags={'CF': 0, 'PF': 2, 'AF': 4, 'ZF': 6, 'SF': 7, 'OF': 11},
        eflags_reg=ux.UC_X86_REG_EFLAGS,
    )


UC_DESCS = {'AMD64': _uc_desc_amd64}


def _uc_run(desc: UcDesc, code: bytes, vals: dict) -> dict:
    import unicorn
    uc = unicorn.Uc(desc.uc_arch, desc.uc_mode)
    uc.mem_map(desc.code_addr, 0x2000)
    uc.mem_write(desc.code_addr, code)
    for name, const in desc.gp.items():
        uc.reg_write(const, vals.get(name, 0) & desc.mask)
    uc.emu_start(desc.code_addr, desc.code_addr + len(code))
    out = {name: uc.reg_read(const) & desc.mask for name, const in desc.gp.items()}
    if desc.eflags_reg is not None:
        ef = uc.reg_read(desc.eflags_reg)
        for fname, bit in desc.flags.items():
            out[fname] = (ef >> bit) & 1
    return out


def ground_truth(desc: UcDesc, code: bytes, in_taint: dict, in_values: dict) -> dict:
    """Oracle 2: per-bit sensitivity via Unicorn.  For every tainted input bit,
    flip it (from the clean base) and OR the output XOR into the result mask.
    Returns a per-output taint mask (GP regs + flags)."""
    base_vals = {n: (in_values.get(n, 0) & ~in_taint.get(n, 0) & desc.mask) for n in desc.gp}
    base_out = _uc_run(desc, code, base_vals)
    result: dict = {n: 0 for n in base_out}
    for src in desc.gp:
        tm = in_taint.get(src, 0) & desc.mask
        bit = 0
        while tm:
            if tm & 1:
                flipped = dict(base_vals)
                flipped[src] = (base_vals[src] | (1 << bit)) & desc.mask
                out = _uc_run(desc, code, flipped)
                for k in base_out:
                    result[k] |= base_out[k] ^ out[k]
            tm >>= 1
            bit += 1
    return result


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

@dataclass
class Verdict:
    exact: bool
    under: dict = field(default_factory=dict)   # reg -> bits present in ref/truth, missing in got
    over: dict = field(default_factory=dict)    # reg -> bits in got not in ref/truth

    @property
    def sound(self) -> bool:
        return not self.under

    @property
    def n_over(self) -> int:
        return sum(bin(v).count('1') for v in self.over.values())

    @property
    def n_under(self) -> int:
        return sum(bin(v).count('1') for v in self.under.values())


def classify(got: dict, ref: dict, keys) -> Verdict:
    """Compare an engine's output mask `got` against a reference `ref` over
    `keys` (register/flag names).  under = ref & ~got, over = got & ~ref."""
    under: dict = {}
    over: dict = {}
    for k in keys:
        g = int(got.get(k, 0) or 0)
        r = int(ref.get(k, 0) or 0)
        u = r & ~g
        o = g & ~r
        if u:
            under[k] = u
        if o:
            over[k] = o
    return Verdict(exact=(not under and not over), under=under, over=over)


# ---------------------------------------------------------------------------
# Input-vector fuzzing
# ---------------------------------------------------------------------------

def fuzz_vectors(reg_names, rng: random.Random, n_dense: int, n_sparse: int):
    """Yield (in_taint, in_values) pairs.  Dense masks stress the differential;
    sparse masks (few bits) are what the Unicorn ground truth can enumerate
    cheaply.  Values avoid 0 to dodge div-by-zero-class faults."""
    def rand_vals():
        return {r: rng.randint(1, MASK64) for r in reg_names}

    # edge taint patterns (dense)
    edges = [
        {r: MASK64 for r in reg_names},
        {reg_names[0]: MASK64} if reg_names else {},
        {r: 0xFF for r in reg_names},
        {r: 0xFFFFFFFF00000000 for r in reg_names},
        {r: 0xAAAAAAAAAAAAAAAA for r in reg_names},
    ]
    for t in edges[:n_dense]:
        yield dict(t), rand_vals()
    for _ in range(max(0, n_dense - len(edges))):
        yield {r: rng.randint(0, MASK64) for r in reg_names}, rand_vals()
    # sparse: 1-4 random bits per register (ground-truth-friendly)
    for _ in range(n_sparse):
        t = {}
        for r in reg_names:
            m = 0
            for _b in range(rng.choice([0, 1, 2, 3])):
                m |= 1 << rng.randint(0, 63)
            t[r] = m
        yield t, rand_vals()


# ---------------------------------------------------------------------------
# Corpus driver
# ---------------------------------------------------------------------------

@dataclass
class Report:
    n_cases: int = 0
    n_instrs: int = 0
    n_exact: int = 0
    n_over_only: int = 0
    n_under: int = 0
    over_bits_total: int = 0
    mismatches: list = field(default_factory=list)   # (label, kind, verdict)
    skipped_mem: int = 0
    errors: list = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"cases={self.n_cases} instrs={self.n_instrs} "
            f"exact={self.n_exact} over-only={self.n_over_only} UNDER={self.n_under} "
            f"over-bits={self.over_bits_total} mem-skipped={self.skipped_mem} "
            f"errors={len(self.errors)}"
        )


def _compile_and_mem(circuit, arch, regs) -> bool:
    """`circuit._compiled` is populated lazily on the first evaluate, so a probe
    evaluate (zero taint/values) forces compilation; then has_mem_ops is a
    reliable "reads/writes guest memory" signal (LEA stays False -> register
    corpus keeps it; real load/store/RMW become True -> skipped here)."""
    zero = {r.name: 0 for r in regs}
    sim = CellSimulator(arch)
    try:
        ctx = EvalContext(input_taint=zero, input_values=zero, simulator=sim,
                          implicit_policy=ImplicitTaintPolicy.IGNORE)
        circuit.evaluate(ctx)
    except Exception:  # noqa: BLE001
        pass
    c = getattr(circuit, '_compiled', None)
    return bool(c) and getattr(c, 'has_mem_ops', False)


def run_bank(engine_fn, *, isas=None, n_dense=5, n_sparse=8, seed=1234,
             ref='differential', uc_desc=None, skip_mem=True, max_mismatch=25):
    """Drive `engine_fn` over the instruction bank and compare vs a reference.

    ref='differential' -> compare vs reference_taint (bit-exact gate).
    ref='ground_truth'  -> compare vs Unicorn per-bit (needs uc_desc; soundness
                            + precision).  Uses only the sparse vectors (GT is
                            per-bit-expensive and exact only for few bits).
    """
    from benchmark.instruction_bank import load_bank

    rep = Report()
    specs = load_bank(isas=set(isas) if isas else None)
    for spec in specs.values():
        reg_names = [r.name for r in spec.regs]
        for ins in spec.instructions:
            try:
                circuit = build_circuit(spec.arch, ins.bytes, spec.regs)
            except Exception as e:  # noqa: BLE001
                rep.errors.append((ins.label, f'build: {e!r}'))
                continue
            if skip_mem and _compile_and_mem(circuit, spec.arch, spec.regs):
                rep.skipped_mem += 1
                continue
            rep.n_instrs += 1
            rng = random.Random(f'{seed}:{ins.label}')
            for in_taint, in_values in fuzz_vectors(reg_names, rng, n_dense, n_sparse):
                if ref == 'ground_truth':
                    # GT enumerates tainted bits: cap total to keep it cheap/exact.
                    total_bits = sum(bin(v).count('1') for v in in_taint.values())
                    if total_bits == 0 or total_bits > 20:
                        continue
                rep.n_cases += 1
                try:
                    got = engine_fn(spec.arch, ins.bytes, spec.regs,
                                    in_taint, in_values, circuit=circuit)
                    if ref == 'ground_truth':
                        keys = list(uc_desc.gp) + list(uc_desc.flags)
                        refd = ground_truth(uc_desc, ins.bytes, in_taint, in_values)
                    else:
                        keys = reg_names
                        refd = reference_taint(spec.arch, ins.bytes, spec.regs,
                                               in_taint, in_values, circuit=circuit)
                except Exception as e:  # noqa: BLE001
                    rep.errors.append((ins.label, f'eval: {e!r}'))
                    continue
                v = classify(got, refd, keys)
                if v.exact:
                    rep.n_exact += 1
                elif v.sound:
                    rep.n_over_only += 1
                    rep.over_bits_total += v.n_over
                    if len(rep.mismatches) < max_mismatch:
                        rep.mismatches.append((ins.label, 'OVER', v))
                else:
                    rep.n_under += 1
                    if len(rep.mismatches) < max_mismatch:
                        rep.mismatches.append((ins.label, 'UNDER', v))
    return rep


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--isas', nargs='*', default=None)
    ap.add_argument('--dense', type=int, default=5)
    ap.add_argument('--sparse', type=int, default=8)
    ap.add_argument('--ref', choices=['differential', 'ground_truth'], default='differential')
    args = ap.parse_args()
    ud = UC_DESCS['AMD64']() if args.ref == 'ground_truth' else None
    r = run_bank(reference_taint, isas=args.isas, n_dense=args.dense,
                 n_sparse=args.sparse, ref=args.ref, uc_desc=ud)
    print(r.summary())
    for label, kind, v in r.mismatches[:25]:
        print(f'  {kind} {label}: under={v.under} over={v.over}')
    for label, err in r.errors[:10]:
        print(f'  ERR {label}: {err}')
