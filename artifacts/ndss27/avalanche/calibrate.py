"""Executable calibration gate for the avalanche attribution.

The paper states the attribution "matches the ground truth on calibration
instructions (add/xor/mov/and -> 0%, imul -> 100%, shl rax,cl with tainted
amount -> 86%, shl rax,4 -> 0%)".  Grepping the artifact, `calibrat` appears
ONLY in README prose: no code ever produced those four numbers.  This is that
claim, executed.

It matters because `imul -> 100%` is a direct probe of whether avalanche is
visible at all, and imul's taint runs through `VariableMultiplyTaintExpr` --
one of the seven node types the old attribution treated as an opaque leaf with
"no avalanche inside".  Had this gate existed, it would have failed the day that
node type landed instead of quietly lowering Table 6.

Run:  uv run --project <engine> python calibrate.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('MICROTAINT_TAINT_IR', '0')
os.environ.setdefault('MICROTAINT_BLOCK', '0')

import exprwalk as W  # noqa: E402
from microtaint.instrumentation.ast import EvalContext  # noqa: E402
from microtaint.simulator import CellSimulator  # noqa: E402
from microtaint.sleigh.engine import generate_static_rule  # noqa: E402
from microtaint.types import (  # noqa: E402
    Architecture, ImplicitTaintPolicy, Register,
)

NAMES = ['RAX', 'RBX', 'RCX', 'RDX', 'CF', 'PF', 'ZF', 'SF', 'OF']
FMT = ([Register(n, 64) for n in NAMES[:4]]
       + [Register(f, 1) for f in NAMES[4:]] + [Register('RSP', 64)])
SIM = CellSimulator(Architecture.AMD64)

#: (asm, tainted input, expected avalanche share of DATA bits, tolerance)
CASES = [
    ('add rax, rbx',  {'RAX': 0xFFFF_FFFF_FFFF_FFFF},   0.0, 0.0),
    ('xor rax, rbx',  {'RAX': 0xFFFF_FFFF_FFFF_FFFF},   0.0, 0.0),
    ('mov rax, rbx',  {'RBX': 0xFFFF_FFFF_FFFF_FFFF},   0.0, 0.0),
    ('and rax, rbx',  {'RAX': 0xFFFF_FFFF_FFFF_FFFF},   0.0, 0.0),
    ('shl rax, 4',    {'RAX': 0xFFFF_FFFF_FFFF_FFFF},   0.0, 0.0),
    # imul: the multiply term calls itself "a sound fill, not an avalanche", so
    # its bits ARE approximation and must be counted.  The paper's 100% dates
    # from when multiply fell back to AvalancheExpr outright.
    ('imul rax, rbx', {'RAX': 0xFFFF_FFFF_FFFF_FFFF}, 100.0, 0.0),
    # shl by CL: the paper's 86% also dates from the avalanche fallback.  The
    # engine has since gained VariableShiftTaintExpr, which claims EXACT, so the
    # honest expectation today is 0% -- a real precision gain, not a blind spot.
    # Flagged rather than silently re-based: see APPROXIMATING_TYPES.
    ('shl rax, cl',   {'RCX': 0xFF},                     0.0, 0.0),
]


def _ctx(taint):
    vals = dict.fromkeys(NAMES, 0)
    vals.update({'RAX': 0x0123_4567_89AB_CDEF, 'RBX': 0xFEDC_BA98_7654_3210,
                 'RCX': 5, 'RDX': 0x1111, 'RSP': 0x204000})
    tnt = dict.fromkeys(NAMES, 0)
    tnt.update(taint)
    return EvalContext(input_taint=tnt, input_values=vals, simulator=SIM,
                       implicit_policy=ImplicitTaintPolicy.IGNORE)


def data_avalanche_share(asm, taint):
    """(% of tainted DATA bits owed to avalanche, n_data_bits) for one instruction."""
    import keystone
    ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
    code = bytes(ks.asm(asm)[0])
    rule = generate_static_rule(Architecture.AMD64, code, FMT)
    full_bits = aval_bits = 0
    for a in rule.assignments:
        e = a.expression
        if e is None:
            continue
        tgt = a.target
        width = (tgt.size * 8 if hasattr(tgt, 'address_expr')
                 else tgt.bit_end - tgt.bit_start + 1)
        if width < 8:            # data registers only; flags are counted apart
            continue
        mask = (1 << width) - 1
        full = e.evaluate(_ctx(taint)) & mask
        if not full:
            continue
        # every APPROXIMATING node -> 0, root included; the ENGINE evaluates
        precise = W.evaluate_precise(e, _ctx(taint)) & mask
        full_bits += bin(full).count('1')
        aval_bits += bin(full & ~precise & mask).count('1')
    return (100.0 * aval_bits / full_bits if full_bits else 0.0), full_bits


def main() -> int:
    W.check_engine_node_types()
    print(f'{"instruction":16s} {"expected":>9s} {"measured":>9s} {"bits":>6s}   verdict')
    bad = 0
    for asm, taint, expect, tol in CASES:
        got, nbits = data_avalanche_share(asm, taint)
        ok = abs(got - expect) <= tol
        bad += not ok
        print(f'{asm:16s} {expect:8.1f}% {got:8.1f}% {nbits:6d}   {"OK" if ok else "MISMATCH"}')
    if bad:
        print(f'\nCALIBRATION FAILED on {bad} case(s): the attribution does not '
              'reproduce answers that are known independently, so any Table 6 it '
              'produces is unverified.')
        return 1
    print('\nCALIBRATION PASSED: the attribution reproduces every known answer.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
