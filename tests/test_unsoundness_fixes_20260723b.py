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

(The MIPS HI/LO multiply/divide intermediate-register threading fix is tracked
separately -- it destabilises the compiled chained-circuit evaluator and is held
pending a C-VM fix.)

Every case is a (state, taint) that under-tainted BEFORE the corresponding fix,
with ground truth carrying real taint on the checked register.
"""

# mypy: disable-error-code="no-untyped-def,no-untyped-call,attr-defined,index"

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
]


@pytest.mark.parametrize(('arch', 'code', 'state', 'taint', 'label'), _CASES,
                         ids=[c[4].split()[0] + '-' + c[0] for c in _CASES])
def test_no_under_taint(arch, code, state, taint, label):
    _assert_sound(arch, code, state, taint, label)
