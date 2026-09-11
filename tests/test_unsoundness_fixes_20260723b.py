"""Multi-ISA soundness regressions for the July 23 2026 exhaustive campaign.

Continuation of ``test_unsoundness_fixes_20260723`` -- reuses its exact 2^k
Unicorn ground-truth harness -- covering the generalising floor fixes made after
enumerating ~all integer/scalar instructions of every keystone-supported ISA:

  * (2) the equality-to-zero (ZF) floor is category-independent and resolves a
        UNIQUE compared value to its aliasing register -- fixes ``xadd``/``cmpxchg``;
  * (1) a shift composed with a variable-width mask (``bextr``) gets a
        reachable-window avalanche when the exact subcube term declines;
  * (3) negate-through-a-shift (``neg``/``negs``, both ``-x`` and ``0 - x`` lifts)
        gets a borrow-smear at the operand's post-shift positions;
  * (4) the sign/MSB flag floor extends to LOGICAL results (``ands``/``tst`` NG).

The MIPS HI/LO multiply/divide threading was held here in July: taint crossing
HI/LO between ``mult`` and ``mflo``/``mfhi`` was dropped, zeroing every product
read back, and the fix for it destabilised the compiled chained-circuit
evaluator.  **It is no longer under-tainted**, on either the circuit path or the
taint-IR one, so the held patch is obsolete and the defect it addressed is now
covered by cases here rather than by a patch file.  It went unnoticed for as
long as it did because the instruction bank holds no multi-instruction HI/LO
sequence at all, which is exactly why these are written out by hand.

Every case is a (state, taint) that under-tainted BEFORE the corresponding fix,
with ground truth carrying real taint on the checked register.
"""


from __future__ import annotations

import pytest

from tests.test_unsoundness_fixes_20260723 import _assert_sound

# (arch, code_hex, state, taint, label) -- same schema as the sibling module.
_CASES = [
    # -- (2): equality-to-zero floor, category-independent + unique-aliased ------
    ('AMD64', '0fc0d8', {'RAX': 0x5E, 'RBX': 0x86}, {'RAX': 0xA0, 'RBX': 0x95},
     'xadd al,bl ZF (compared sum kept in a UNIQUE)'),
    ('AMD64', '0fb0cb', {'RAX': 0x5E, 'RBX': 0x86, 'RCX': 0x40},
     {'RAX': 0xA0, 'RBX': 0x95}, 'cmpxchg bl,cl ZF non-monotone'),
    # -- (1): bextr variable-width mask (shift composed with mask) ---------------
    ('AMD64', 'c4e2f0f7c3',
     {'RBX': 0x8DED3C9691EB79FA, 'RCX': 0xF0E642F43328AD08},
     {'RCX': 0x4420000000000001},
     'bextr rax,rbx,rcx data-dependent extract window'),
    # -- (3): negate-through-a-shift borrow (both -x and 0-x lifts) --------------
    ('ARM64', 'e10f02cb', {'x2': 0x180021}, {'x2': 0x180021},
     'neg x1,x2,lsl #3 borrow (INT_2COMP)'),
    ('ARM64', 'e10f02eb', {'x2': 0x180021}, {'x2': 0x180021},
     'negs x1,x2,lsl #3 borrow (0 - x)'),
    # -- (4): sign/MSB flag floor for a LOGICAL result --------------------------
    ('ARM64', '200c02ea',
     {'x1': 0x8000000000000040, 'x2': 0x1400010000000000},
     {'x1': 0x8000000000000040, 'x2': 0x1400010000000000},
     'ands x0,x1,x2,lsl #3 NG (sign of a shifted AND)'),
    # -- HI/LO: taint has to cross the multiply's implicit destination ----------
    # A product lands in HI/LO and is read back a whole instruction later, so
    # the taint has to survive a register that neither instruction names in its
    # operands.  When it did not, every multiply and divide result came back
    # clean -- about 800 under-tainted bits per form in the July campaign.
    ('MIPS64BE', '0085001800001012',
     {'A0': 0x11223344, 'A1': 0x55667788}, {'A0': 0x81, 'A1': 0x3},
     'mult $a0,$a1 ; mflo $v0 threads the product through LO'),
    ('MIPS64BE', '0085001800001010',
     {'A0': 0x11223344, 'A1': 0x55667788}, {'A0': 0x81, 'A1': 0x3},
     'mult $a0,$a1 ; mfhi $v0 threads the product through HI'),
    ('MIPS64BE', '0085001900001012',
     {'A0': 0x11223344, 'A1': 0x55667788}, {'A0': 0x81, 'A1': 0x3},
     'multu $a0,$a1 ; mflo $v0 (unsigned form)'),
    ('MIPS64BE', '0085001a00001012',
     {'A0': 0x11223344, 'A1': 0x7}, {'A0': 0x81, 'A1': 0x0},
     'div $a0,$a1 ; mflo $v0 threads the quotient through LO'),
    ('MIPS64BE', '0085001a00001010',
     {'A0': 0x11223344, 'A1': 0x55667788}, {'A0': 0x81, 'A1': 0x3},
     'div $a0,$a1 ; mfhi $v0 threads the remainder through HI'),
]


@pytest.mark.parametrize(('arch', 'code', 'state', 'taint', 'label'), _CASES,
                         ids=[c[4].split()[0] + '-' + c[0] for c in _CASES])
def test_no_under_taint(arch: str, code: str, state: dict[str, int],
                        taint: dict[str, int], label: str) -> None:
    _assert_sound(arch, code, state, taint, label)


@pytest.mark.parametrize(('arch', 'code', 'state', 'taint', 'label'), _CASES,
                         ids=[c[4].split()[0] + '-' + c[0] for c in _CASES])
def test_the_case_carries_taint_somewhere_it_was_not_given(
        arch: str, code: str, state: dict[str, int],
        taint: dict[str, int], label: str) -> None:
    """Each case must have a ground truth that actually moves.

    `_assert_sound` compares `gt & ~mt` over whatever the ground truth holds,
    so a case whose ground truth carries taint only on the registers it SEEDED
    passes against an engine that computed nothing at all -- the taint it finds
    is the taint it was handed.  That is not hypothetical: one of the HI/LO
    cases here was first written with operands whose quotient is zero whatever
    the tainted bits do, and it passed while proving nothing.

    So: some register the case did NOT taint must come back tainted.
    """
    from tests.test_unsoundness_fixes_20260723 import (
        _ARCHES,
        _SP_VALUE,
        _brute_gt,
        _canon,
    )

    spec = _ARCHES[arch]
    st, tn = _canon(spec, dict(state), dict(taint))
    full = dict(st)
    if spec.sp is not None:
        full.setdefault(spec.sp[0], _SP_VALUE)
    gt = _brute_gt(spec, bytes.fromhex(code), full, tn)
    seeded = {k for k, v in tn.items() if v}
    produced = {k: v for k, v in gt.items() if v and k not in seeded}
    assert produced, (
        f'[{label}] the ground truth carries taint only on the registers the '
        f'case seeded ({sorted(seeded)}), so it would pass against an engine '
        f'that propagated nothing')
