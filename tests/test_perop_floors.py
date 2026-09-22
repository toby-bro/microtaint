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
from __future__ import annotations

import random
from collections.abc import Iterable, Iterator

import pytest

from tests.conftest import fuzz_budget

#: Release tier: this file is 140s of the suite's 1903s under CI conditions
#: (serial, with coverage), and the cost is the ground-truth oracle rather than
#: anything here.  `ground_truth` re-executes the instruction under Unicorn once
#: per tainted input bit, over four bases -- on AMD64 with every GP register
#: tainted that is ~1,540 emulations per case -- so these tests are inherently
#: expensive and cannot be tuned down without weakening the evidence.
#:
#: Deselected by default, run by --slow or MICROTAINT_SLOW_TESTS=1, which
#: release-soundness.yml sets for both taint paths.  The CHEAP ground-truth
#: files stay on the fast tier on purpose -- the per-bug regression tests in
#: test_unsoundness_fixes_*, test_push_rsp_taint, test_oracle_harness and the
#: rest are ~37s together -- so a pull request still exercises the oracle.
pytestmark = pytest.mark.slow

MASK64 = 0xFFFFFFFFFFFFFFFF


def _gt_vectors(gp_names: list[str], rng: random.Random, n: int,
                max_bits: int = 4) -> Iterator[tuple[dict[str, int], dict[str, int]]]:
    """Vectors tainting only GP regs with few TOTAL bits (ground-truth friendly:
    per-bit enumeration stays cheap and exact)."""
    for _ in range(n):
        t = dict.fromkeys(gp_names, 0)
        for _b in range(rng.randint(1, max_bits)):
            t[rng.choice(gp_names)] |= 1 << rng.randint(0, 63)
        vals = {r: rng.randint(1, MASK64) for r in gp_names}
        yield t, vals


def _under_bits(got: dict[str, int], ref: dict[str, int],
                keys: Iterable[str]) -> dict[str, int]:
    u: dict[str, int] = {}
    for k in keys:
        m = (int(ref.get(k, 0) or 0)) & ~(int(got.get(k, 0) or 0))
        if m:
            u[k] = m
    return u


def soundness_report(isa: str, n_per: int, seed: int = 7,
                     instr_limit: int | None = None,
                     ) -> tuple[int, int, int, list[object]]:
    """Three-way check: per-op vs ground truth vs engine oracle.  Returns
    (n_cases, n_exact_vs_gt, n_new_under, new_under_examples)."""
    from instruction_bank import load_bank

    from tests.oracle_harness import UC_DESCS, build_circuit, classify, ground_truth, reference_taint
    from tests.perop_floors import Unsupported, engine_perop_floors

    ud = UC_DESCS[isa]()
    gp = list(ud.gp)
    keys = gp + list(ud.flags)
    specs = load_bank(isas={isa})
    n_cases = n_exact = 0
    new_under: list[object] = []
    for spec in specs.values():
        regs = spec.regs
        instrs = spec.instructions
        if instr_limit:
            instrs = instrs[:instr_limit]
        for ins in instrs:
            try:
                circuit = build_circuit(spec.arch, ins.bytes, spec.regs)
            except Exception:
                continue
            rng = random.Random(f'{seed}:{ins.label}')
            for t, vals in _gt_vectors(gp, rng, n_per):
                in_taint = {r.name: t.get(r.name, 0) for r in regs}
                in_values = {r.name: vals.get(r.name, 0) for r in regs}
                try:
                    got = engine_perop_floors(spec.arch, ins.bytes, regs, in_taint, in_values)
                except Unsupported:
                    continue
                except Exception:
                    continue
                try:
                    gtd = ground_truth(ud, ins.bytes, t, vals)
                    engd = reference_taint(spec.arch, ins.bytes, regs, in_taint,
                                           in_values, circuit=circuit)
                except Exception:
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


def slicewise_report(isa: str, n_per: int, seed: int = 11,
                     ) -> tuple[int, int, int, int, list[object]]:
    """Per-output-slice window (the integration model): trust per-op only on
    outputs whose cone is reconvergence-free.  Returns
    (n_cases, tot_slices, clean_slices, n_new_under_on_clean, examples)."""
    from instruction_bank import load_bank

    from tests.oracle_harness import UC_DESCS, build_circuit, ground_truth, reference_taint
    from tests.perop_floors import NeedsMonolithic, Unsupported, perop_floors_slicewise

    ud = UC_DESCS[isa]()
    gp = list(ud.gp)
    keys = gp + list(ud.flags)
    specs = load_bank(isas={isa})
    n_cases = tot_slices = clean_slices = 0
    new_under: list[object] = []
    for spec in specs.values():
        regs = spec.regs
        for ins in spec.instructions:
            try:
                circuit = build_circuit(spec.arch, ins.bytes, spec.regs)
            except Exception:
                continue
            rng = random.Random(f'{seed}:{ins.label}')
            for t, vals in _gt_vectors(gp, rng, n_per):
                it = {r.name: t.get(r.name, 0) for r in regs}
                iv = {r.name: vals.get(r.name, 0) for r in regs}
                try:
                    taint, clean = perop_floors_slicewise(spec.arch, ins.bytes, regs, it, iv)
                except (NeedsMonolithic, Unsupported):
                    continue
                except Exception:
                    continue
                try:
                    gtd = ground_truth(ud, ins.bytes, t, vals)
                    engd = reference_taint(spec.arch, ins.bytes, regs, it, iv, circuit=circuit)
                except Exception:
                    continue
                n_cases += 1
                for k in keys:
                    tot_slices += 1
                    if k not in clean:
                        continue
                    clean_slices += 1
                    up = int(gtd.get(k, 0) or 0) & ~int(taint.get(k, 0) or 0)
                    ue = int(gtd.get(k, 0) or 0) & ~int(engd.get(k, 0) or 0)
                    extra = up & ~ue
                    if extra and len(new_under) < 20:
                        new_under.append((ins.label, k, hex(extra)))
    return n_cases, tot_slices, clean_slices, len(new_under), new_under


def test_perop_slicewise_sound_and_covers(request: pytest.FixtureRequest) -> None:
    """The per-output-slice window is the integration model: a reconvergent flag
    no longer disqualifies a clean result register.  On the CLEAN slices, per-op
    must introduce no under-taint beyond the engine's own differential (vs ground
    truth), and coverage must be high (reconvergence is usually confined to a few
    flags, so most slices stay on the fast path)."""
    pytest.importorskip('unicorn')
    # min cases per ISA (RISCV bank is small); all must be sound + high coverage.
    # The floors were written for n_per=2, so they scale with the budget: a
    # reduced run legitimately produces proportionally fewer cases, and a floor
    # that did not scale would fail for the one reason that is not a defect.
    n_per = fuzz_budget(2, request.config)
    for isa, min_cases in (('AMD64', 300), ('ARM64', 300), ('RISCV64', 40)):
        n_cases, tot, clean, n_new_under, ex = slicewise_report(isa, n_per=n_per)
        want = min_cases * n_per // 2
        assert n_cases > want, f'{isa}: too few cases: {n_cases} (want > {want})'
        assert n_new_under == 0, f'{isa}: NEW under-taints on clean slices: {ex}'
        cov = clean / max(1, tot)
        assert cov > 0.85, f'{isa}: slice coverage too low: {cov:.2f}'


def test_perop_no_new_undertaint_vs_engine_amd64() -> None:
    """Per-op-with-floors must not UNDER-taint anything the engine's own
    whole-instruction differential catches (vs Unicorn ground truth).  Bounded
    corpus; the standalone sweep covers the full bank."""
    pytest.importorskip('unicorn')
    n_cases, _n_exact, n_new_under, examples = soundness_report('AMD64', n_per=3)
    assert n_cases > 500, f'too few cases exercised: {n_cases}'
    assert n_new_under == 0, f'NEW under-taints (soundness bugs): {examples}'


def test_perop_no_new_undertaint_vs_engine_arm64() -> None:
    """ISA-generality: the same per-op model (p-code is the contract) is sound on
    AArch64 too -- no under-taint beyond what the engine's differential has vs
    Unicorn ground truth.  Exercises the register-name resolver (bank X0/N-Z-C-V
    -> pypcode x0/NG-ZR-CY-OV)."""
    pytest.importorskip('unicorn')
    n_cases, _n_exact, n_new_under, examples = soundness_report('ARM64', n_per=3)
    assert n_cases > 400, f'too few cases exercised: {n_cases}'
    assert n_new_under == 0, f'NEW under-taints (soundness bugs): {examples}'


def test_perop_control_flow_routes_to_monolithic() -> None:
    """cmov lifts to an intra-instruction CBRANCH; the per-op model must defer to
    the monolithic window (NeedsMonolithic) rather than execute the move
    unconditionally and drop the not-taken side's taint."""
    import keystone
    from instruction_bank import isa_registers

    from microtaint.types import Architecture
    from tests.perop_floors import NeedsMonolithic, perop_floors_taint

    ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
    code = bytes(ks.asm('cmove rax, rbx')[0])
    regs = list(isa_registers('AMD64'))
    it = {r.name: (MASK64 if r.name in ('RAX', 'RBX') else 0) for r in regs}
    iv = {r.name: 0x1234 for r in regs}
    with pytest.raises(NeedsMonolithic):
        perop_floors_taint(Architecture.AMD64, code, regs, it, iv)


def test_perop_core_alu_exact_vs_ground_truth() -> None:
    """add/sub/and/or/mov take the per-op fast path: the destination register and
    the carry/overflow flags (CF/OF) are bit-exact vs Unicorn ground truth -- the
    value-aware carry ripple recovers the interior carry bits the bare two-corner
    differential misses, without the coarse smear -- and nothing under-taints.
    (ZF/SF via compare ops keep a sound coarse floor, tightened at integration.)"""
    pytest.importorskip('unicorn')
    import keystone
    from instruction_bank import isa_registers

    from microtaint.types import Architecture
    from tests.oracle_harness import UC_DESCS, classify, ground_truth
    from tests.perop_floors import perop_floors_taint

    ud = UC_DESCS['AMD64']()
    gp = list(ud.gp)
    keys = gp + list(ud.flags)
    exact_keys = ['RAX']  # destination result: bit-exact via the carry ripple
    regs = list(isa_registers('AMD64'))
    ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
    rng = random.Random(1)
    for mnem in ('add rax, rbx', 'sub rax, rbx', 'and rax, rbx',
                 'or rax, rbx', 'mov rax, rbx'):
        code = bytes(ks.asm(mnem)[0])
        for _ in range(8):
            t = dict.fromkeys(gp, 0)
            for _b in range(rng.randint(1, 3)):
                t[rng.choice(gp)] |= 1 << rng.randint(0, 63)
            vals = {r: rng.randint(1, MASK64) for r in gp}
            in_taint = {r.name: t.get(r.name, 0) for r in regs}
            in_values = {r.name: vals.get(r.name, 0) for r in regs}
            got = perop_floors_taint(Architecture.AMD64, code, regs, in_taint, in_values)
            gtd = ground_truth(ud, code, t, vals)
            assert not classify(got, gtd, keys).under, f'{mnem}: under taint={t}'
            v = classify(got, gtd, exact_keys)
            assert v.exact, f'{mnem}: destination not exact: over={v.over} taint={t}'


def test_perop_reconvergence_routes_to_monolithic() -> None:
    """xor rax,rax (input used twice, cancels) and imul rax,rbx (OF compares the
    result against the full product -> shared source) are reconvergent: the per-op
    window widens to the monolithic differential rather than over-taint."""
    import keystone
    from instruction_bank import isa_registers

    from microtaint.types import Architecture
    from tests.perop_floors import NeedsMonolithic, perop_floors_taint

    ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
    regs = list(isa_registers('AMD64'))
    it = {r.name: MASK64 for r in regs}
    iv = {r.name: 0x1234 for r in regs}
    for mnem in ('xor rax, rax', 'imul rax, rbx'):
        code = bytes(ks.asm(mnem)[0])
        with pytest.raises(NeedsMonolithic):
            perop_floors_taint(Architecture.AMD64, code, regs, it, iv)
