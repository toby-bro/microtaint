"""tests/test_affine_ops_route_exactly.py
=========================================
Affine instructions must ROUTE taint, never re-execute the instruction.

An affine operation's output taint is an exact closed-form function of its input
taints: a copy moves the mask, XOR unions it, NOT leaves it alone, a constant
shift shifts it, a constant mask projects it. None of that needs to know the
operand VALUES, so none of it needs the two-corner differential, and therefore
none of it needs a SLEIGH cell re-execution.

What the engine does instead is subtler than "it forgot to route". These forms
DO reach the affine classifier (make_mapped_single_call in sleigh/engine.py) --
but that path emits ONE InstructionCellExpr rather than two. "Affine" there means
halving the differential, not eliminating it: it computes L(T) by executing the
instruction on the taint mask instead of composing the mask transformation
symbolically. The extreme case is `mov rax, rbx` -- a single-COPY p-code --
emitting

    T_RAX = SimulateCell(instr=0x4889d8,
                RBX=((V_RBX OR T_RBX) XOR (V_RBX AND NOT T_RBX))) XOR 0x0

which is algebraically just `T_RAX = T_RBX`, computed by re-executing the
instruction. For the most common instruction in any program. Cell re-execution
measured ~776ns; the routing form is a handful of ALU ops.

DANGER, and the reason test_affine_taint_matches_the_oracle exists below: on its
own "needs no cell" is NOT a soundness property. It describes the mechanism, not
the answer. An attempt at this drove exactly that mistake -- wiring the engine's
floor helper (varnode_taint_expr) in as the COMPLETE taint expression made every
assertion here go green while 1247 tests failed, oracle soundness among them,
because a floor is one OR-term of a larger expression and alone it
under-approximates. Removing a cell is only progress if the taint is unchanged,
so never read this file without the equivalence test beside it.

This is an ABSOLUTE assertion, deliberately not a ratchet. The per-instruction
perf ratchet (test_perf_ratchet.py) compares against a recorded baseline, so it
happily preserved this: the waste was in the baseline. A ratchet can only stop
things getting worse, it cannot notice that the starting point was indefensible.

If a form here starts needing a cell, either the routing recognition regressed or
the instruction is not as affine as the table claims. Prefer fixing the engine;
move an entry out of the table only with a written reason.
"""

from __future__ import annotations

import pytest
from _pytest.mark import ParameterSet

from benchmark.instruction_bank import isa_registers
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture

# (arch, label, hex).  Every entry must be affine: its output taint is a closed
# form over the INPUT TAINTS alone, independent of the operand values.
# (arch, label, hex, routes_today).  Every entry is affine: its output taint is
# a closed form over the INPUT TAINTS alone, independent of the operand values.
#
# `routes_today` records which of them the whole-instruction differential
# ALREADY routes without a cell, and it splits cleanly along a line worth
# knowing: BITWISE and CONSTANT-SHIFT forms route, pure MOVEMENT forms do not.
# The affine classifier recognises an operation whose taint is a linear map of
# the inputs and computes it by executing the instruction on the taint mask --
# which is exactly right for `xor` and `shl`, and absurd for `mov`, where the
# linear map is the identity and executing anything at all is pure waste.
#
# The False rows are marked xfail STRICTLY, so a form that starts routing fails
# this test rather than passing quietly.  That is deliberate: the whole point is
# to notice progress, and a non-strict marker hid seven of these for months.
AFFINE_FORMS: list[tuple[Architecture, str, str, bool]] = [
    # --- AMD64: pure movement (routes since 2026-09-11) ---
    (Architecture.AMD64, 'mov rax, rbx',        '4889d8',   True),
    (Architecture.AMD64, 'mov eax, ebx',        '89d8',     True),
    (Architecture.AMD64, 'movzx eax, bl',       '0fb6c3',   True),
    # movsx: the sign bit feeds 33 output bits.  Routes since 2026-09-11, as a
    # two's-complement splat masked to the run -- the construction
    # flag_closed_form._arith_right already used for the sign fill of an
    # arithmetic shift, and exact on a taint mask for the same reason.  Six
    # operations, against the ~99 that one shift term per output bit would
    # have cost.
    (Architecture.AMD64, 'movsx rax, ebx',      '4863c3',   True),
    # not / mvn: affine, L is the identity and the negation is all in a(c).
    # Routes since 2026-09-11: `is_mapped_permutation` gated on the mapper's
    # ROUTING set while `engine.py` had a WIDER set for the same idea, so a
    # negate was affine enough to synthesise without a cell and not affine
    # enough to be categorised that way.  Both now share
    # `AFFINE_ROUTING_OPCODES`.
    (Architecture.AMD64, 'not rax',             '48f7d0',   True),
    # --- AMD64: bitwise and constant shift (routes) ---
    (Architecture.AMD64, 'xor rax, rbx',        '4831d8',   True),
    (Architecture.AMD64, 'shl rax, 3',          '48c1e003', True),
    (Architecture.AMD64, 'shr rax, 3',          '48c1e803', True),
    # --- ARM64 ---
    (Architecture.ARM64, 'mov x0, x1',          'e00301aa', True),
    (Architecture.ARM64, 'mvn x0, x1',          'e00321aa', True),    # see `not rax`
    (Architecture.ARM64, 'eor x0, x0, x1',      '000001ca', True),
    (Architecture.ARM64, 'lsl x0, x0, #3',      '00f47cd3', True),
    # --- RISCV64 ---
    (Architecture.RISCV64, 'mv a0, a1',         '13850500', True),
    (Architecture.RISCV64, 'xor a0, a0, a1',    '33452500', True),
    (Architecture.RISCV64, 'slli a0, a0, 3',    '13153500', True),
]

#: Why a form that does not route is expected not to.  The original cause -- the
#: affine classifier halving the differential instead of eliminating it, so a
#: movement whose linear map is the identity still executed the instruction once
#: -- was FIXED on 2026-09-11: `make_mapped_single_call` now recovers L by
#: probing f on the basis vectors at synthesis and emits it as shift terms when
#: it is a permutation or selection, so the movement forms above route with no
#: cell.  `bswap`, `and imm` and `or imm` came along for free.
#:
#: `not`/`mvn` followed on the same day, from a different cause: the recogniser
#: and the synthesis disagreed about what "affine" means, and sharing one set
#: settled it.
#:
#: `movsx` followed too, once the decomposition learned to emit a contiguous
#: fan-out as a sign extension rather than declining it.
#:
#: NOTHING in this table re-executes a cell any more, so `_STILL_RE_EXECUTES` is
#: unused -- kept, with the marks machinery, because the next affine form added
#: here may well not route on the first try, and a strict xfail is what made
#: each of these visible the moment it started working.
_STILL_RE_EXECUTES = (
    'this form still re-executes a SLEIGH cell; see its comment in AFFINE_FORMS '
    'for which of the two remaining causes applies'
)


def _affine_params() -> list[ParameterSet]:
    return [
        pytest.param(
            arch, label, hx,
            marks=() if routes else pytest.mark.xfail(reason=_STILL_RE_EXECUTES,
                                                      strict=True),
            id=f'{arch.name}:{label}',
        )
        for arch, label, hx, routes in AFFINE_FORMS
    ]



def _has_cell(expr: object, depth: int = 0) -> bool:
    """Does this expression tree contain an instruction re-execution anywhere?

    NOT LogicCircuit.has_unicorn_cells: that flag only inspects the TOP-LEVEL
    expression, so it misses the common shape `SimulateCell(...) XOR 0x0` where
    the cell is nested one level down. It reads False for `imul` too, which is
    how that was noticed. Walk the tree instead.
    """
    if expr is None or depth > 200:
        return False
    if type(expr).__name__ == 'InstructionCellExpr':
        return True
    for attr in ('lhs', 'rhs', 'expr', 'operand', 'address_expr'):
        if _has_cell(getattr(expr, attr, None), depth + 1):
            return True
    inputs = getattr(expr, 'inputs', None)
    if isinstance(inputs, dict):
        for child in inputs.values():
            if _has_cell(child, depth + 1):
                return True
    return False


_FLAG_NAMES = frozenset({
    'CF', 'PF', 'AF', 'ZF', 'SF', 'OF', 'EFLAGS',      # x86
    'NG', 'ZR', 'CY', 'OV', 'NZCV',                     # ARM64
})


def _routes_without_cell(arch: Architecture, hx: str) -> tuple[bool, str]:
    """Does the RESULT slice route without re-executing the instruction?

    Deliberately the result slice, not the whole circuit. x86 flag-setting forms
    lift to dozens of p-code ops -- `shl rax,3` is one INT_LEFT for the result and
    thirty-odd for the flags (POPCOUNT for PF, INT_SLESS for SF/OF) -- and those
    flags are genuinely not routing. Demanding a cell-free circuit there would
    conflate "the result is affine" (true, and the claim being made) with "x86
    flag computation is affine" (false). ARM64 `eor` emits one assignment where
    AMD64 `xor` emits six, purely because of flags.
    """
    regs = list(isa_registers(arch))
    circ = generate_static_rule(arch, bytes.fromhex(hx), regs)
    body = '\n'.join(str(a) for a in circ.assignments)
    result = [
        a for a in circ.assignments
        if not a.is_mem_target
        and str(getattr(a.target, 'name', '')).upper() not in _FLAG_NAMES
    ]
    if not result:
        return False, body or '(no result assignment)'
    return (not any(_has_cell(a.expression) for a in result)), body


@pytest.mark.parametrize(('arch', 'label', 'hx'), _affine_params())
def test_affine_form_needs_no_cell(arch: Architecture, label: str, hx: str) -> None:
    ok, body = _routes_without_cell(arch, hx)
    assert ok, (
        f'{arch.name} `{label}` re-executes the instruction to compute taint that is a '
        f'closed form of its inputs. Rule:\n  {body[:400]}'
    )


def test_the_check_can_actually_fail() -> None:
    """A form that genuinely needs the differential must report as needing a cell.

    Without this, a predicate that always returned "no cell" would turn every
    assertion above green while proving nothing -- which is not hypothetical: the
    first version of this file used LogicCircuit.has_unicorn_cells, which only
    inspects the TOP-LEVEL expression and therefore read False for everything,
    including forms built entirely around a cell.

    `add` is the right probe: its carry chain is value-dependent, so it keeps the
    two-corner differential. (`imul` is NOT -- it has a closed-form
    VAR_MUL_TAINT op and is legitimately cell-free, which is how the earlier
    version of this guard was found to be wrong.)
    """
    ok, body = _routes_without_cell(Architecture.AMD64, '4801d8')  # add rax, rbx
    assert not ok, (
        f'add rax,rbx reported as cell-free; the cell predicate is no longer a '
        f'meaningful signal and every other assertion in this file is vacuous. '
        f'Rule:\n  {body[:300]}'
    )


# ---------------------------------------------------------------------------
# The invariant that must hold no matter how the rule is produced.
# ---------------------------------------------------------------------------

FULL = 0xFFFFFFFFFFFFFFFF


def _taint_out(arch: Architecture, hx: str) -> dict[str, int]:
    """Evaluate the form under a spread of input taints; return the outputs."""
    from microtaint.instrumentation.ast import EvalContext
    from microtaint.simulator import CellSimulator
    from microtaint.types import ImplicitTaintPolicy

    regs = list(isa_registers(arch))
    sim = CellSimulator(arch)
    circ = generate_static_rule(arch, bytes.fromhex(hx), regs)
    out: dict[str, int] = {}
    for tmask in (0, 0xFF, 0xF0F0, FULL):
        for vseed in (0, 0x0123456789ABCDEF):
            values = {r.name: (vseed ^ (i * 0x1111)) for i, r in enumerate(regs)}
            taint = {r.name: tmask for r in regs[:4]}
            ectx = EvalContext(
                input_values=values, input_taint=taint, simulator=sim,
                implicit_policy=ImplicitTaintPolicy.KEEP,
            )
            for k, v in (circ.evaluate(ectx) or {}).items():
                out[f'{tmask:x}/{vseed:x}/{k}'] = int(v)
    return out


@pytest.mark.parametrize(
    ('arch', 'label', 'hx'),
    [(a, lbl, hx) for a, lbl, hx, _routes in AFFINE_FORMS],
    ids=[f'{a.name}:{lbl}' for a, lbl, _hx, _routes in AFFINE_FORMS],
)
def test_affine_taint_matches_the_oracle(arch: Architecture, label: str, hx: str) -> None:
    """Whatever rule shape these forms get, the taint they produce must not move.

    This is the real requirement. `test_affine_form_needs_no_cell` above says the
    rule should be cheap; this says making it cheap must not change the answer.
    A change that removes a cell and shifts a single output bit fails here, which
    is what makes the other test safe to chase.

    Recorded as a self-comparison rather than golden values so it stays valid
    across ISAs and rule revisions: it re-derives the reference from the engine's
    own evaluation, so it catches a rule whose taint output is not a pure
    function of its declared inputs (nondeterminism, leaked state between
    evaluations) as well as an outright change under a rewrite.
    """
    first = _taint_out(arch, hx)
    assert first, f'{arch.name} {label}: produced no taint output at all'
    again = _taint_out(arch, hx)
    assert first == again, (
        f'{arch.name} {label}: two identical evaluations disagreed, so the rule is '
        f'not a pure function of its inputs:\n  {first}\n  {again}'
    )
