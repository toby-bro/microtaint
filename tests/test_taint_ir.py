"""Gates for the lowered taint IR and its backends.

Three claims, each of which has already been wrong at least once:

  * the IR agrees with Unicorn per-bit ground truth, and never under-taints
    where the current engine does not;
  * `finalize` -- which expands one-bit cones into cheapest expressions and
    compacts the program -- preserves meaning exactly;
  * the two backends compute the same thing as the interpreter.  Three
    register-clobbering bugs in the emitter were caught only by comparing
    against it on many random states, so that comparison is a permanent test
    rather than something run once.
"""
# ruff: noqa: PLC0415
import random

import pytest

from microtaint.taint_ir.ir import IRProg
from tests.perop_c_bank import Ref, TaintState

MASK64 = 0xFFFFFFFFFFFFFFFF

_CASES = [
    ('AMD64', 'mov rax, rbx', '4889d8'),
    ('AMD64', 'add rax, rbx', '4801d8'),
    ('AMD64', 'sub rax, rbx', '4829d8'),
    ('AMD64', 'and rax, rbx', '4821d8'),
    ('AMD64', 'xor rax, rbx', '4831d8'),
    ('AMD64', 'cmp rax, rbx', '4839d8'),
    ('AMD64', 'shl rax, 4', '48c1e004'),
    ('AMD64', 'shl rax, cl', '48d3e0'),
    ('AMD64', 'shrd rax, rbx, 7', '480facd807'),
    ('AMD64', 'adc rax, rbx', '4811d8'),
    ('AMD64', 'cmove rax, rbx', '480f44c3'),
    ('AMD64', 'mov ah, bh', '88fc'),
    ('AMD64', 'inc rax', '48ffc0'),
    ('AMD64', 'neg rax', '48f7d8'),
    ('ARM64', 'add x0, x1, x2', '2000028b'),
    ('ARM64', 'subs x0, x1, x2', '200002eb'),
    ('ARM64', 'csel x0, x1, x2, eq', '2000829a'),
    ('ARM64', 'eor x0, x1, x2', '200002ca'),
]


def _arch(name: str):
    from microtaint.types import Architecture
    return getattr(Architecture, name)


def _prog(isa: str, hexcode):
    from microtaint.taint_ir.frompcode import build_ir
    return build_ir(_arch(isa), bytes.fromhex(hexcode))


def _states(prog: IRProg, n: int,
            seed: str | int) -> list[tuple[TaintState, TaintState]]:
    rng = random.Random(seed)
    keys = sorted({k for (_kind, k) in prog.inputs})
    return [({k: rng.getrandbits(64) for k in keys},
             {k: rng.getrandbits(64) for k in keys}) for _ in range(n)]


@pytest.mark.parametrize('isa,label,code', _CASES,
                         ids=[f'{i}:{l}' for i, l, _c in _CASES])
def test_finalize_preserves_meaning(isa: str, label: str, code: bytes) -> None:
    """Expanding the one-bit cones and compacting must not change any answer."""
    prog = _prog(isa, code)
    fin = prog.finalize()
    for vals, tnts in _states(prog, 120, f'fin:{label}'):
        assert prog.run(vals, tnts) == fin.run(vals, tnts), label


@pytest.mark.parametrize('isa,label,code', _CASES,
                         ids=[f'{i}:{l}' for i, l, _c in _CASES])
def test_backends_agree(isa: str, label: str, code: bytes) -> None:
    """The C interpreter, the host emitter and the Python reference agree."""
    from microtaint.instrumentation.cell_c import taint_ir_c
    from microtaint.taint_ir.exec import compile_program

    prog = _prog(isa, code)
    keys = sorted({k for (_kind, k) in prog.inputs} | {k for k, _n in prog.outputs})
    slot = {k: i for i, k in enumerate(keys)}
    cap, _d = compile_program(prog, lambda n: slot.get(n))

    rng = random.Random(f'be:{label}')
    for _ in range(120):
        vals = [rng.getrandbits(64) for _ in keys]
        tnts = [rng.getrandbits(64) for _ in keys]
        ref = prog.run({k: vals[slot[k]] for k in keys},
                       {k: tnts[slot[k]] for k in keys})
        interp = taint_ir_c.run(cap, vals, tnts)
        for k, want in ref.items():
            assert interp[slot[k]] == want, f'{label}: interpreter differs on {k}'
    if not taint_ir_c.jit(cap):
        pytest.skip('host emitter declined this program')
    for _ in range(200):
        vals = [rng.getrandbits(64) for _ in keys]
        tnts = [rng.getrandbits(64) for _ in keys]
        ref = prog.run({k: vals[slot[k]] for k in keys},
                       {k: tnts[slot[k]] for k in keys})
        native = taint_ir_c.run(cap, vals, tnts)
        for k, want in ref.items():
            assert native[slot[k]] == want, f'{label}: emitted code differs on {k}'


@pytest.mark.parametrize('isa', ['AMD64', 'ARM64', 'RISCV64'])
def test_ir_never_under_taints_vs_ground_truth(isa: str, request: pytest.FixtureRequest) -> None:
    from tests.conftest import fuzz_budget
    from tests.perop_c_bank import run_bank_perop_c
    from tests.taint_ir_bank import ir_step

    rep = run_bank_perop_c(isas=[isa], n_sparse=fuzz_budget(2, request.config),
                           ref=Ref.GROUND_TRUTH, step=ir_step)
    assert rep.n_cases > 0
    if rep.n_under_new:
        detail = '\n'.join(
            f'  {label}: missing {[(k, hex(b)) for k, b in new.items()]}'
            for label, _it, _iv, new in rep.new_under_examples[:6])
        pytest.fail(f'{isa}: {rep.n_under_new} case(s) under-taint where the '
                    f'current engine does not:\n{detail}')


@pytest.mark.parametrize('policy', ['concrete', 'avalanche'])
@pytest.mark.parametrize('isa', ['AMD64', 'ARM64'])
def test_memory_taint_matches_ground_truth(isa: str, policy):
    """Loads and stores, against Unicorn per-bit truth over a real data page.

    Both pointer policies are checked.  'avalanche' is the sound one and what
    the engine now runs: its vectors taint the pointer's low bits, so the rule
    that a load through a tainted address taints the whole loaded word is
    exercised rather than assumed.  'concrete' makes no claim about a tainted
    address, so those vectors are not generated for it.

    That exclusion is why this bank did not catch the under-taint fixed in
    tests/test_tainted_pointer_load.py: the hot path ran 'concrete', and the
    only policy checked against tainted pointers was the one it did not run.
    Keep the shipped default and the vector generation in step.
    """
    from tests.taint_ir_mem import run_mem_bank

    rep = run_mem_bank(isa, n_vec=3, policy=policy)
    assert rep.n > 0, f'{isa}/{policy}: no memory cases evaluated'
    if rep.under:
        detail = '\n'.join(
            f'  {label}: {[(k, hex(v)) for k, v in list(u.items())[:6]]}'
            for label, u in rep.under_examples[:5])
        pytest.fail(f'{isa}/{policy}: {rep.under} memory case(s) under-taint:'
                    f'\n{detail}')


@pytest.mark.parametrize('isa', ['AMD64', 'ARM64'])
def test_uc_desc_carries_its_own_isa_tag(isa: str) -> None:
    """`ground_truth_mem` reads `desc.tag` to find the pointer register.

    Nothing declared that attribute: `run_mem_bank` grafted it on after
    building the desc, so the ground truth worked only for a desc that had been
    through that one function, and raised AttributeError for any other caller.
    The field is now on UcDesc and `_uc_desc` fills it, so a desc is usable as
    soon as it exists.
    """
    import dataclasses

    from tests.oracle_harness import UcDesc
    from tests.taint_ir_mem import _uc_desc, build_cases, ground_truth_mem

    # Declared, not grafted: a non-frozen dataclass accepts an attribute set
    # from outside, so only the field list distinguishes the two.
    assert 'tag' in {f.name for f in dataclasses.fields(UcDesc)}

    desc = _uc_desc(isa)
    assert desc.tag == isa, desc.tag

    # And it is enough on its own: no external graft, straight into the truth.
    cases = build_cases(isa)
    assert cases, f'{isa}: no memory cases to drive the ground truth'
    truth = ground_truth_mem(desc, cases[0].code,
                             {}, {}, [0] * 64, [0] * 64)
    assert '@mem' in truth, sorted(truth)


def test_vector_lanes_match_ground_truth() -> None:
    """Lane splitting, against per-bit truth read straight out of XMM.

    The bank's Unicorn descriptors cover general-purpose registers only, so
    without this a wrong lane order would show up merely as a disagreement with
    the engine's own differential -- which cannot distinguish a bug from the
    precision gain lane splitting is supposed to produce.
    """
    from tests.taint_ir_simd import run_simd_bank

    n, under, skipped = run_simd_bank(n_vec=3)
    assert n > 0, 'no SIMD cases evaluated'
    # A form Unicorn does not honour is not ground truth and is skipped, but a
    # probe that started skipping everything would pass this test while
    # measuring nothing, so what it kept is asserted too.
    assert n > 4 * len(skipped), (
        f'only {n} cases against {len(skipped)} forms skipped as not ground '
        f'truth: the encoding probe is refusing too much')
    if under:
        detail = '\n'.join(f'  {lbl}: {items[0]}' for lbl, items in
                           list(under.items())[:5])
        pytest.fail(f'{len(under)} vector instruction(s) under-taint:\n{detail}')
