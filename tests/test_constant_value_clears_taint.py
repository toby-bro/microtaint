"""A value proved constant carries no taint, and an INVENTED one still does.

`xor rax, rax` is how every compiler on every architecture zeroes a register.
The result is zero whatever RAX held, so a secret in RAX is gone afterwards and
its taint should go with it.  It did not: the taint rule for a binary op is
`t_a | t_b`, and for two copies of the same node that is `t | t = t`, so full
taint walked straight through a register that provably holds zero.  The IR had
already folded the VALUE to a constant; the two halves never consulted each
other.

The fix is one question asked at the single place a result is written: if the
value is a constant, no input bit can move it, so its taint is zero.  It covers
`sub reg,reg`, `and reg,0`, `or reg,-1` and the flags those set, which is where
most of the over-taint was.

The first version of that rule was WRONG, and this file exists mostly because of
how.  An operation p-code does not model writes an invented value:
`_emit_callother` writes a literal zero because it has to write something, while
keeping the taint at avalanche.  So `const 0` in the IR means either "proved
zero" or "no idea", and clearing on the second is an under-taint.  Measured, it
lost bit 8 of `crc32 rax, cl`, and the per-bit ground-truth sweep caught it.

So the rule is opt-IN: a caller says `proved` only when the value is what p-code
semantics computed.  That polarity is the point.  A caller that invents a value
gets the safe answer by saying nothing, and a future one that forgets loses an
optimisation rather than silently under-tainting.

Nothing here is a hand-written expectation.  The oracle is Unicorn per-bit
sensitivity: flip each tainted input bit and see which output bits actually
move.  A bit that never moves cannot be tainted; one that moves must be.
"""
from __future__ import annotations

import pytest

from microtaint.emulator import blockpath_c as B
from microtaint.taint_ir import frompcode
from microtaint.taint_ir.blockcompile import compile_block
from microtaint.types import Architecture

FULL = (1 << 64) - 1
_BASE = 0x401000


@pytest.fixture(scope='module')
def layouts() -> dict[str, dict[str, int]]:
    out = {}
    for isa in ('AMD64', 'ARM64', 'RISCV64'):
        arch = Architecture[isa]
        names = sorted(set(frompcode.builder_for(arch).name_by_off.values()))
        out[isa] = {n: i for i, n in enumerate(names)}
    return out


def _engine(isa: str, code: bytes, layout: dict[str, int]) -> dict[str, int] | None:
    """The engine's answer with EVERY register tainted on entry."""
    arch = Architecture[isa]
    mem = B.mem_new(0x7000, 0x2000)
    runner = B.runner_new(len(layout), -1, mem)
    B.runner_seed(runner, [FULL] * len(layout))
    got = compile_block(arch, code, _BASE, layout, cache=False)
    if got is None:
        return None
    if B.runner_on_block(runner, got[0], _BASE, [0] * len(layout)) != 0:
        return None
    B.runner_finish(runner, True)
    taint = B.runner_taint(runner)
    return {n: taint[s] for n, s in layout.items()}


def _truth(isa: str, code: bytes) -> dict[str, int]:
    """Which output bits genuinely move when a tainted input bit is flipped."""
    from tests.oracle_harness import UC_DESCS, ground_truth

    desc = UC_DESCS[isa]()
    in_taint = dict.fromkeys(desc.gp, desc.mask)
    in_values = dict.fromkeys(desc.gp, 0)
    return ground_truth(desc, code, in_taint, in_values)


# ---------------------------------------------------------------------------
# The cases.  `dst` is the register the idiom zeroes; `clears` says whether the
# whole of it must come back clean.
# ---------------------------------------------------------------------------
#: (label, isa, bytes, dst, clears)
_CASES: list[tuple[str, str, str, str, bool]] = [
    # --- AMD64, every width -------------------------------------------------
    ('xor rax,rax',   'AMD64', '4831c0',   'RAX', True),
    ('xor eax,eax',   'AMD64', '31c0',     'RAX', True),   # 32-bit zero-extends
    ('xor ax,ax',     'AMD64', '6631c0',   'RAX', False),  # upper 48 survive
    ('xor al,al',     'AMD64', '30c0',     'RAX', False),  # upper 56 survive
    ('sub rax,rax',   'AMD64', '4829c0',   'RAX', True),
    ('and rax,0',     'AMD64', '4883e000', 'RAX', True),
    ('or rax,-1',     'AMD64', '4883c8ff', 'RAX', True),
    ('and eax,0',     'AMD64', '83e000',   'RAX', True),
    ('or eax,-1',     'AMD64', '83c8ff',   'RAX', True),
    # --- AMD64, the register file is not special ---------------------------
    ('xor rbx,rbx',   'AMD64', '4831db',   'RBX', True),
    ('xor rcx,rcx',   'AMD64', '4831c9',   'RCX', True),
    ('xor rdx,rdx',   'AMD64', '4831d2',   'RDX', True),
    ('xor rsi,rsi',   'AMD64', '4831f6',   'RSI', True),
    ('xor rdi,rdi',   'AMD64', '4831ff',   'RDI', True),
    # --- negative controls: the taint MUST survive -------------------------
    ('xor rax,rbx',   'AMD64', '4831d8',   'RAX', False),
    ('or rax,0',      'AMD64', '4883c800', 'RAX', False),  # identity
    ('and rax,-1',    'AMD64', '4883e0ff', 'RAX', False),  # identity
    ('mov rax,rdi',   'AMD64', '4889f8',   'RAX', False),
    # An operation p-code does not model.  Its value is a fabricated zero, and
    # clearing on it is exactly the under-taint this rule had to be fixed for.
    ('crc32 rax,cl',  'AMD64', 'f2480f38f0c1', 'RAX', False),
    # The 32-bit forms, which are a CALLOTHER into EAX followed by an INT_ZEXT
    # into RAX.  They are here because the first version of this rule marked
    # EVERY op's write as proved, including the movement ops an invented value
    # is deliberately allowed to travel through, so the zext saw a `const 0`
    # value and cleared the avalanche: RAX came back clean however the inputs
    # were tainted.  The 64-bit form above has no zext and did not catch it.
    ('crc32 eax,bl',  'AMD64', 'f20f38f0c3',   'RAX', False),
    ('crc32 eax,ebx', 'AMD64', 'f20f38f0c3',   'RAX', False),
]

#: The same idioms where another ISA spells them differently.  Added separately
#: because the encodings need checking against the ground truth before the
#: clearing claim means anything: a byte string that does not zero the register
#: would pass "it cleared" for the wrong reason.
_CROSS: list[tuple[str, str, str, str, bool]] = [
    ('eor x0,x0,x0',   'ARM64',   '000000ca', 'X0', True),
    ('eor w0,w0,w0',   'ARM64',   '0000004a', 'X0', True),   # W writes zero-extend
    ('sub x0,x0,x0',   'ARM64',   '000000cb', 'X0', True),
    ('and x0,x0,xzr',  'ARM64',   '00001f8a', 'X0', True),
    ('eor x1,x1,x2',   'ARM64',   '210002ca', 'X1', False),  # control
    # T registers, not A: the RISCV64 ground truth watches T0-T6, and a
    # register it does not watch reports no taint at all, which would pass
    # "it cleared" for a register nobody looked at.  `_run_case` refuses a
    # destination the oracle does not carry, so this cannot go quiet again.
    ('xor t0,t0,t0',   'RISCV64', 'b3c25200', 'T0', True),
    ('sub t0,t0,t0',   'RISCV64', 'b3825240', 'T0', True),
    ('xor t0,t1,t2',   'RISCV64', 'b3427300', 'T0', False),  # control
]


def _run_case(label: str, isa: str, hexs: str, dst: str, clears: bool,
              layouts: dict[str, dict[str, int]]) -> None:
    code = bytes.fromhex(hexs)
    got = _engine(isa, code, layouts[isa])
    assert got is not None, f'{label}: the engine refused to lower it'
    truth = _truth(isa, code)
    # A destination the oracle does not track reports nothing, and "nothing"
    # is indistinguishable from "clean": that is how a RISCV64 sweep in this
    # repo once agreed with everything while watching the wrong registers.
    assert dst in truth, (
        f'{label}: the ground truth does not track {dst}, so it cannot say '
        f'whether anything there depends on the input')

    # 1. SOUNDNESS.  Every bit the hardware says depends on an input must be
    #    tainted.  This is the direction that is never allowed to regress, and
    #    it is checked for every case, clearing or not.
    for name, want in truth.items():
        if name not in got:
            continue
        assert got[name] | want == got[name], (
            f'{label}: {name} under-tainted, hardware says bits {want:#x} '
            f'depend on the input and the engine reports {got[name]:#x}')

    # 2. PRECISION.  For the zeroing idioms, the destination must come back
    #    fully clean, and the ground truth has to AGREE that nothing there
    #    moves, or the test is asserting the wrong thing.
    if clears:
        assert truth.get(dst, 0) == 0, (
            f'{label}: the hardware says {dst} bits {truth.get(dst, 0):#x} do '
            f'depend on the input, so this is not a zeroing idiom at all')
        assert got[dst] == 0, (
            f'{label}: {dst} came back {got[dst]:#x}; a value proved constant '
            f'should carry no taint')


@pytest.mark.parametrize(('label', 'isa', 'hexs', 'dst', 'clears'), _CASES,
                         ids=[c[0] for c in _CASES])
def test_a_proved_constant_carries_no_taint(
        label: str, isa: str, hexs: str, dst: str, clears: bool,
        layouts: dict[str, dict[str, int]]) -> None:
    _run_case(label, isa, hexs, dst, clears, layouts)


@pytest.mark.parametrize(('label', 'isa', 'hexs', 'dst', 'clears'), _CROSS,
                         ids=[c[0] for c in _CROSS])
def test_the_same_idiom_on_another_architecture(
        label: str, isa: str, hexs: str, dst: str, clears: bool,
        layouts: dict[str, dict[str, int]]) -> None:
    """The rule is in the lowering, not in an x86 table, so it must hold
    wherever the idiom appears.  ARM64 spells it `eor`, RISCV64 `xor`, and
    both zero-extend a 32-bit write exactly as x86 does."""
    _run_case(label, isa, hexs, dst, clears, layouts)


def test_the_rule_is_not_keyed_to_a_particular_register(
        layouts: dict[str, dict[str, int]]) -> None:
    """The extended registers, which the ground truth does not reach.

    `UC_DESCS['AMD64']` tracks RAX..RDI, so R8-R15 cannot be checked against
    hardware here and asserting they clear would be asserting my own
    expectation.  What CAN be checked without an oracle is that they behave
    like the register that IS oracle-verified: `xor r8,r8` lowers to the same
    p-code shape as `xor rax,rax`, so if one clears and the other does not, the
    rule is keyed to something it has no business knowing about.
    """
    layout = layouts['AMD64']
    ref = _engine('AMD64', bytes.fromhex('4831c0'), layout)   # xor rax,rax
    assert ref is not None
    assert ref['RAX'] == 0, 'the oracle-verified case stopped clearing'
    for label, hexs, dst in (('xor r8,r8', '4d31c0', 'R8'),
                             ('xor r15,r15', '4d31ff', 'R15'),
                             ('xor r8d,r8d', '4531c0', 'R8')):
        got = _engine('AMD64', bytes.fromhex(hexs), layout)
        assert got is not None, f'{label} did not lower'
        assert got[dst] == ref['RAX'], (
            f'{label}: {dst} came back {got[dst]:#x} where the same shape on '
            f'RAX gives {ref["RAX"]:#x}')


def test_the_controls_actually_taint_something(
        layouts: dict[str, dict[str, int]]) -> None:
    """Anti-vacuity.  If the ground truth reported nothing for every case, the
    soundness half above would pass on a table of zeroes."""
    moved = 0
    for _label, isa, hexs, dst, clears in _CASES + _CROSS:
        if clears:
            continue
        truth = _truth(isa, bytes.fromhex(hexs))
        if truth.get(dst, 0):
            moved += 1
    assert moved >= 4, (
        f'only {moved} control cases have any input-dependent output bit, so '
        f'the oracle is not distinguishing tainted from clean')


def test_an_unmodelled_operation_never_says_proved() -> None:
    """The polarity, stated where it cannot rot.

    `_emit_callother` writes an invented value.  If it ever passed `proved`,
    `crc32` and every AVX form would clear taint they must keep, and the only
    thing that would notice is a ground-truth sweep someone has to remember to
    run.  Read from the source, because the failure is a missing argument and
    there is nothing at runtime to observe.
    """
    import inspect

    src = inspect.getsource(frompcode.Builder._emit_callother)
    assert 'proved' not in src, (
        'an operation p-code cannot model now claims its value is proved; its '
        'value is a fabricated zero and clearing taint on it is an under-taint')


def test_a_movement_op_never_says_proved() -> None:
    """The other half of the polarity, and the one that was got wrong.

    `_invention_stays_opaque` lets an unmodelled operation's invented value
    travel through COPY, INT_ZEXT, INT_SEXT, SUBPIECE and PIECE, because their
    taint rule reads taint alone and moving a lie does not make it a worse lie.
    But their VALUE is then the invented constant, so a movement op claiming
    `proved` clears taint that must stay.  Read from the source, because the
    failure is a missing condition rather than something observable at runtime.
    """
    import inspect

    src = inspect.getsource(frompcode.Builder._emit_op)
    assert 'proved=name not in self._MOVEMENT_OPS' in src, (
        'the general op path no longer excludes movement ops from `proved`; a '
        'CALLOTHER value moving through a zext will clear its own avalanche')


def test_proved_is_off_by_default() -> None:
    """A caller that says nothing must get the safe answer."""
    import inspect

    for fn in (frompcode.Builder._predicated_write,
               frompcode.Builder._predicated_write_at):
        got = inspect.signature(fn).parameters['proved'].default
        assert got is False, f'{fn.__name__}: proved defaults to {got!r}'


def test_the_rule_does_not_fire_under_a_tainted_predicate(
        layouts: dict[str, dict[str, int]]) -> None:
    """A write that may not happen is not a proof about anything.

    `cmovz rax, rbx` writes under a condition; whatever lands is not a constant
    and the value node is a select rather than a const, so the rule cannot fire.
    Checked because it is the one shape where "the value is constant" would be
    true of one ARM of the write and false of the result.
    """
    code = bytes.fromhex('480f44c3')          # cmove rax, rbx
    got = _engine('AMD64', code, layouts['AMD64'])
    assert got is not None, 'cmove did not lower'
    truth = _truth('AMD64', code)
    for name, want in truth.items():
        if name in got:
            assert got[name] | want == got[name], (
                f'{name} under-tainted under a predicate: hardware {want:#x}, '
                f'engine {got[name]:#x}')
    assert got['RAX'] != 0, 'a predicated write cleared RAX outright'


def test_the_taint_is_gone_downstream_not_just_at_the_write(
        layouts: dict[str, dict[str, int]]) -> None:
    """Clearing is only worth anything if the next instruction sees it.

    `xor rax,rax` then `mov rbx,rax`: RBX must come back clean, which it cannot
    if the clear stopped at the register file and the block's own threading
    kept the old taint.
    """
    layout = layouts['AMD64']
    got = _engine('AMD64', bytes.fromhex('4831c04889c3'), layout)
    assert got is not None
    assert got['RAX'] == 0, 'the zeroing itself did not clear'
    assert got['RBX'] == 0, (
        f'RBX came back {got["RBX"]:#x} after being copied from a register '
        f'that provably holds zero')
