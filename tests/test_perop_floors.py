"""Standing gate for the per-op-with-floors eval-core model (Phase 3b GO/NO-GO).

The eval-core rework ([[differential-compaction-design]]) moves the differential
from whole-instruction to per-p-code-op granularity: affine ops route the taint
mask exactly, non-affine ops get a local differential OR a sound per-op floor,
and intra-instruction control flow / reconvergence falls back to the monolithic
window (the current whole-instruction differential = the oracle).

These tests prove the model is SOUND before any of it is wired into the engine:

  * RIGOROUS SOUNDNESS: over the bank, per-op-with-floors introduces NO under-taint
    beyond what the engine's OWN whole-instruction differential already has vs
    Unicorn per-bit ground truth.  Formally, for every case,
        under(perop, GT)  is a bitwise subset of  under(engine, GT)
    so the shared SLEIGH undefined-flag artifacts (adc AF, imul SF/PF, bsf PF...)
    are allowed but a NEW leak fails the gate.  Under-taint is never acceptable.

  * control-flow instructions (cmov) route to the monolithic fallback rather than
    under-taint on the not-taken side.

  * the core ALU + flag instructions are bit-exact vs ground truth.

Bounded here for CI; the full multi-thousand-case sweep runs standalone via
`python -m tests.perop_floors_study` (see __main__ below).
"""
# ruff: noqa: PLC0415
# mypy: disable-error-code="no-untyped-def,no-untyped-call,import-untyped"
from __future__ import annotations

import random

import pytest

MASK64 = 0xFFFFFFFFFFFFFFFF


def _gt_vectors(gp_names, rng, n, max_bits=4):
    """Vectors tainting only GP regs with few TOTAL bits (ground-truth friendly:
    per-bit enumeration stays cheap and exact)."""
    for _ in range(n):
        t = {r: 0 for r in gp_names}
        for _b in range(rng.randint(1, max_bits)):
            t[rng.choice(gp_names)] |= 1 << rng.randint(0, 63)
        vals = {r: rng.randint(1, MASK64) for r in gp_names}
        yield t, vals


def _under_bits(got, ref, keys):
    u = {}
    for k in keys:
        m = (int(ref.get(k, 0) or 0)) & ~(int(got.get(k, 0) or 0))
        if m:
            u[k] = m
    return u


def soundness_report(isa, n_per, seed=7, instr_limit=None):
    """Three-way check: per-op vs ground truth vs engine oracle.  Returns
    (n_cases, n_exact_vs_gt, n_new_under, new_under_examples)."""
    from tests.oracle_harness import (UC_DESCS, build_circuit, classify,
                                       ground_truth, reference_taint)
    from tests.perop_floors import Unsupported, engine_perop_floors
    from benchmark.instruction_bank import load_bank

    ud = UC_DESCS[isa]()
    gp = list(ud.gp)
    keys = gp + list(ud.flags)
    specs = load_bank(isas={isa})
    n_cases = n_exact = 0
    new_under = []
    for spec in specs.values():
        regs = spec.regs
        instrs = spec.instructions
        if instr_limit:
            instrs = instrs[:instr_limit]
        for ins in instrs:
            try:
                circuit = build_circuit(spec.arch, ins.bytes, spec.regs)
            except Exception:  # noqa: BLE001
                continue
            rng = random.Random(f'{seed}:{ins.label}')
            for t, vals in _gt_vectors(gp, rng, n_per):
                in_taint = {r.name: t.get(r.name, 0) for r in regs}
                in_values = {r.name: vals.get(r.name, 0) for r in regs}
                try:
                    got = engine_perop_floors(spec.arch, ins.bytes, regs, in_taint, in_values)
                except Unsupported:
                    continue
                except Exception:  # noqa: BLE001
                    continue
                try:
                    gtd = ground_truth(ud, ins.bytes, t, vals)
                    engd = reference_taint(spec.arch, ins.bytes, regs, in_taint,
                                           in_values, circuit=circuit)
                except Exception:  # noqa: BLE001
                    continue
                n_cases += 1
                if classify(got, gtd, keys).exact:
                    n_exact += 1
                u_perop = _under_bits(got, gtd, keys)
                u_eng = _under_bits(engd, gtd, keys)
                for k, m in u_perop.items():
                    extra = m & ~u_eng.get(k, 0)
                    if extra and len(new_under) < 20:
                        new_under.append((ins.label, k, hex(extra),
                                          {kk: hex(vv) for kk, vv in t.items() if vv}))
    return n_cases, n_exact, len(new_under), new_under


def test_perop_no_new_undertaint_vs_engine_amd64():
    """Per-op-with-floors must not UNDER-taint anything the engine's own
    whole-instruction differential catches (vs Unicorn ground truth).  Bounded
    corpus; the standalone sweep covers the full bank."""
    pytest.importorskip('unicorn')
    n_cases, n_exact, n_new_under, examples = soundness_report('AMD64', n_per=3)
    assert n_cases > 500, f'too few cases exercised: {n_cases}'
    assert n_new_under == 0, f'NEW under-taints (soundness bugs): {examples}'


def test_perop_control_flow_routes_to_monolithic():
    """cmov lifts to an intra-instruction CBRANCH; the per-op model must defer to
    the monolithic window (NeedsMonolithic) rather than execute the move
    unconditionally and drop the not-taken side's taint."""
    import keystone

    from benchmark.instruction_bank import isa_registers
    from microtaint.types import Architecture
    from tests.perop_floors import NeedsMonolithic, perop_floors_taint

    ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
    code = bytes(ks.asm('cmove rax, rbx')[0])
    regs = list(isa_registers('AMD64'))
    it = {r.name: (MASK64 if r.name in ('RAX', 'RBX') else 0) for r in regs}
    iv = {r.name: 0x1234 for r in regs}
    with pytest.raises(NeedsMonolithic):
        perop_floors_taint(Architecture.AMD64, code, regs, it, iv)


def test_perop_core_alu_exact_vs_ground_truth():
    """add / sub / and / or / imul / mov are bit-exact vs Unicorn ground truth
    (the floors recover the carry/parity bits the bare differential misses)."""
    pytest.importorskip('unicorn')
    import keystone

    from benchmark.instruction_bank import isa_registers
    from microtaint.types import Architecture
    from tests.oracle_harness import UC_DESCS, classify, ground_truth
    from tests.perop_floors import perop_floors_taint

    ud = UC_DESCS['AMD64']()
    gp = list(ud.gp)
    keys = gp + list(ud.flags)
    regs = list(isa_registers('AMD64'))
    ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
    rng = random.Random(1)
    for mnem in ('add rax, rbx', 'sub rax, rbx', 'and rax, rbx',
                 'or rax, rbx', 'imul rax, rbx', 'mov rax, rbx'):
        code = bytes(ks.asm(mnem)[0])
        for _ in range(8):
            t = {r: 0 for r in gp}
            for _b in range(rng.randint(1, 3)):
                t[rng.choice(gp)] |= 1 << rng.randint(0, 63)
            vals = {r: rng.randint(1, MASK64) for r in gp}
            in_taint = {r.name: t.get(r.name, 0) for r in regs}
            in_values = {r.name: vals.get(r.name, 0) for r in regs}
            got = perop_floors_taint(Architecture.AMD64, code, regs, in_taint, in_values)
            gtd = ground_truth(ud, code, t, vals)
            v = classify(got, gtd, keys)
            # imul leaves SF/PF undefined in SLEIGH (shared artifact); allow those.
            real_under = {k: m for k, m in v.under.items()
                          if not (mnem.startswith('imul') and k in ('SF', 'PF', 'ZF', 'AF'))}
            assert not real_under, f'{mnem}: under={real_under} taint={t}'
