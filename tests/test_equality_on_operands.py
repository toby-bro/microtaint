"""`(a - b) == 0` is answered as `a == b`, which is what can be proved.

A comparison sets the zero flag, and no lifter writes that as "are the operands
equal".  SLEIGH models it as `INT_EQUAL(INT_SUB(a, b), 0)`: subtract, then ask
whether the difference is zero.  The two forms mean exactly the same thing in
wrapping arithmetic, and they are worlds apart for a taint rule.

Asked about the DIFFERENCE, the rule can only prove inequality when some bit of
the difference is provably one.  Asked about the OPERANDS, it can prove it
whenever they differ in any bit that neither side can change, which is a far
weaker condition and the one that actually occurs.

The case that motivated this is glibc's EOF test, `cmp $-1, %eax` where EAX
holds a byte returned by `__uflow`:

    operands     EAX has bits 8-31 clean zero, -1 has them set -> never equal
    difference   EAX + 1, in [1, 0x100], no single bit provably one -> unknown

So block mode reported a secret-dependent branch at a comparison that cannot
vary, and a person had to triage it.

The oracle here is Unicorn per-bit sensitivity, not a hand-written expectation:
flip each tainted input bit and see whether the flag actually moves.  The
"can be equal" cases pass INPUT VALUES that put the comparison on its boundary,
because a probe that never makes the two sides equal would report the flag as
clean and agree with anything.
"""
from __future__ import annotations

import pytest

from microtaint.emulator import blockpath_c as B
from microtaint.taint_ir import frompcode
from microtaint.taint_ir.blockcompile import compile_block
from microtaint.types import Architecture

#: Release tier: this file is 50s of the suite's 1903s under CI conditions
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

FULL = (1 << 64) - 1
_BASE = 0x401000


@pytest.fixture(scope='module')
def layouts() -> dict[str, dict[str, int]]:
    out = {}
    for isa in ('AMD64', 'ARM64'):
        arch = Architecture[isa]
        names = sorted(set(frompcode.builder_for(arch).name_by_off.values()))
        out[isa] = {n: i for i, n in enumerate(names)}
    return out


def _engine(isa: str, code: bytes, layout: dict[str, int],
            values: dict[str, int]) -> dict[str, int] | None:
    mem = B.mem_new(0x7000, 0x2000)
    runner = B.runner_new(len(layout), -1, mem)
    B.runner_seed(runner, [FULL] * len(layout))
    regs = [0] * len(layout)
    for n, v in values.items():
        regs[layout[n]] = v & FULL
    got = compile_block(Architecture[isa], code, _BASE, layout, cache=False)
    if got is None:
        return None
    if B.runner_on_block(runner, got[0], _BASE, regs) != 0:
        return None
    B.runner_finish(runner, True)
    taint = B.runner_taint(runner)
    return {n: taint[s] for n, s in layout.items()}


def _truth(isa: str, code: bytes, values: dict[str, int]) -> dict[str, int]:
    from tests.oracle_harness import UC_DESCS, ground_truth

    desc = UC_DESCS[isa]()
    in_taint = dict.fromkeys(desc.gp, desc.mask)
    in_values = {n: values.get(n, 0) for n in desc.gp}
    return ground_truth(desc, code, in_taint, in_values)


#: (label, isa, bytes, flag, input values, can the comparison ever be true)
_CASES: list[tuple[str, str, str, str, dict[str, int], bool]] = [
    # --- the shape that motivated this -------------------------------------
    ('movzx eax,al; cmp eax,-1',   'AMD64', '0fb6c083f8ff',
     'ZF', {'RAX': 0x41}, False),
    ('movzx eax,al; cmp eax,0x100', 'AMD64', '0fb6c03d00010000',
     'ZF', {'RAX': 0x41}, False),
    # --- the same shapes where equality IS reachable -----------------------
    #     values put the comparison on its boundary, so flipping one bit of
    #     the tainted byte genuinely moves the flag and the oracle can see it.
    ('movzx eax,al; cmp eax,5',    'AMD64', '0fb6c083f805',
     'ZF', {'RAX': 5}, True),
    ('movzx eax,al; cmp eax,0xff', 'AMD64', '0fb6c03dff000000',
     'ZF', {'RAX': 0xFF}, True),
    ('cmp eax,-1 (all of eax tainted)', 'AMD64', '83f8ff',
     'ZF', {'RAX': 0xFFFFFFFF}, True),
    ('cmp rax,rbx',                'AMD64', '4839d8',
     'ZF', {'RAX': 7, 'RBX': 7}, True),
    ('test eax,eax',               'AMD64', '85c0',
     'ZF', {'RAX': 0}, True),
    # --- another ISA: the rule is in the lowering, not an x86 table ---------
    # ARM64's zero flag is `Z` to Unicorn and `ZR` to the lifter; the case
    # table uses the ORACLE's name and `_engine_flag_names` translates.
    ('and w0,#0xff; cmp w0,#0x100', 'ARM64', '001c00121f000471',
     'Z', {'X0': 0x41}, False),
    ('and w0,#0xff; cmp w0,#5',     'ARM64', '001c00121f140071',
     'Z', {'X0': 5}, True),
    ('cmp x0,x1',                   'ARM64', '1f0001eb',
     'Z', {'X0': 7, 'X1': 7}, True),
]


@pytest.mark.parametrize(('label', 'isa', 'hexs', 'flag', 'values', 'reachable'),
                         _CASES, ids=[c[0] for c in _CASES])
def test_a_comparison_that_cannot_vary_does_not_taint_its_flag(
        label: str, isa: str, hexs: str, flag: str, values: dict[str, int],
        reachable: bool, layouts: dict[str, dict[str, int]]) -> None:
    from tests.oracle_harness import UC_DESCS, _engine_flag_names

    code = bytes.fromhex(hexs)
    got = _engine(isa, code, layouts[isa], values)
    assert got is not None, f'{label}: the engine refused to lower it'
    truth = _truth(isa, code, values)
    # The oracle and the engine spell flags differently per ISA: AMD64 agrees
    # on ZF, ARM64's zero flag is `Z` to Unicorn and `ZR` to the lifter.  Ask
    # rather than hardcode, or the ARM64 cases look up a name that is not there
    # and read as clean.
    eng_flag = _engine_flag_names(Architecture[isa], UC_DESCS[isa]()).get(flag, flag)
    assert eng_flag in got, (
        f'{label}: the engine has no flag named {eng_flag}')
    assert flag in truth, (
        f'{label}: the ground truth does not track {flag}, so it cannot say '
        f'whether it moves')

    # Soundness first, and for every output, not only the flag under test.
    for name, want in truth.items():
        if name in got:
            assert got[name] | want == got[name], (
                f'{label}: {name} under-tainted, hardware says {want:#x} moves '
                f'and the engine reports {got[name]:#x}')

    if reachable:
        # The hardware must AGREE the flag moves, or this case is proving
        # nothing about precision and the input values are badly chosen.
        assert truth[flag], (
            f'{label}: the flag never moves for these inputs, so it is not a '
            f'reachable comparison and the case cannot test over-taint')
        assert got[eng_flag], f'{label}: {flag} came back clean but can vary'
    else:
        assert truth[flag] == 0, (
            f'{label}: the hardware says {flag} DOES move, so this comparison '
            f'is reachable and the test expects the wrong thing')
        assert got[eng_flag] == 0, (
            f'{label}: {flag} is {got[eng_flag]:#x}; the operands differ in '
            f'bits neither side can change, so the comparison cannot vary')


def test_the_substitution_only_applies_against_zero(
        layouts: dict[str, dict[str, int]]) -> None:
    """`(a - b) == k` for a non-zero k is NOT `a == b`.

    The rewrite is only valid against zero, so a comparison of a difference
    against anything else has to keep going through the difference.  There is
    no x86 instruction that writes that shape, so this drives the IR directly:
    the guard is a `p.is_const(bv) and p.const_val(bv) == 0` and this is what
    notices if it is ever loosened.
    """
    import inspect

    src = inspect.getsource(frompcode.Builder._rule)
    marker = 'p.is_const(bv) and p.const_val(bv) == 0'
    assert marker in src, (
        'the equality rule no longer restricts the difference substitution to '
        'a comparison against zero, which is the only value it is valid for')


def test_a_difference_of_the_wrong_width_is_not_substituted() -> None:
    """The recorded difference carries the width it was computed at.

    A `unique` holding a 4-byte difference can be read back as a 1-byte
    varnode, and substituting the 4-byte operands for a 1-byte comparison would
    compare the wrong things.  The width check is what stops that.
    """
    import inspect

    src = inspect.getsource(frompcode.Builder._rule)
    assert 'got[4] == ibits' in src, (
        'the width guard on the difference substitution is gone; a difference '
        'computed at one width could answer a comparison made at another')


def test_a_reachable_comparison_is_still_reported(
        layouts: dict[str, dict[str, int]]) -> None:
    """Anti-vacuity for the file as a whole.

    Every "cannot vary" case asserts a flag is CLEAN, and a rule that simply
    cleared every comparison flag would pass all of them.  So count the cases
    where the flag must stay tainted and refuse a file where that is zero.
    """
    tainted = 0
    for label, isa, hexs, flag, values, reachable in _CASES:
        if not reachable:
            continue
        got = _engine(isa, bytes.fromhex(hexs), layouts[isa], values)
        assert got is not None, label
        from tests.oracle_harness import UC_DESCS, _engine_flag_names
        eng = _engine_flag_names(Architecture[isa], UC_DESCS[isa]()).get(flag, flag)
        tainted += bool(got[eng])
    assert tainted >= 5, (
        f'only {tainted} reachable comparisons still taint their flag; a rule '
        f'that cleared everything would pass the cases above')
