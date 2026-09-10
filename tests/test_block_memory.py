"""Memory across a block region: a store must not be invisible to a later load.

The two-pass memory protocol resolves EVERY load against guest memory and the
shadow before the program runs, and commits every store after it.  Within one
instruction that ordering is exact.  Across a region spanning several
instructions it is not, and the failure is the worst kind:

    mov [rbp+8], rax        ; rax is tainted
    mov rax, [rbp+8]        ; reads the shadow as it was BEFORE the store

lowered as ONE region answered RAX = 0x00 where the same two instructions run
separately answer 0xff.  Storing a value and reading it straight back is what
every compiler does across a spill, so this is not an exotic shape.

The lowering now declines a load that follows a store in the same program, and
the planner cuts there instead.  These tests run the real two-pass protocol
against a byte-addressed memory and shadow, so they check the ANSWER rather than
just the decline: a rule that declined for the wrong reason, or a protocol that
stopped committing stores at all, would still pass a decline-only test.
"""
from __future__ import annotations

from collections.abc import Callable

import pytest
from pypcode import PcodeOp

from microtaint.taint_ir import frompcode
from microtaint.taint_ir.blocks import plan_block
from microtaint.taint_ir.frompcode import Emit, Unsupported
from microtaint.taint_ir.ir import IRKey, IRProg
from microtaint.types import Architecture

_ARCH, _KEY = Architecture.AMD64, 'AMD64'
_MEMBASE = 512
_PER_ACC = 5
_NS = _MEMBASE + _PER_ACC * 8
#: per access k: +0 loaded word / its shadow, +1 address, +2 address taint,
#: +3 store mask, +4 stored value.  The first four mirror
#: fastpath.h::mt_ir_mem_step; the fifth exists only for block lowering, which
#: is the only caller that can load back what it just stored.
_OFF = {'mem': 0, 'addr': 1, 'addrt': 2, 'sttaint': 3, 'stval': 4}


class _Machine:
    """Byte-addressed memory and shadow, so a store is visible to a later load."""

    def __init__(self) -> None:
        self.shadow: dict[int, int] = {}
        self.mem: dict[int, int] = {}

    def read(self, addr: int, size: int) -> tuple[int, int]:
        v = t = 0
        for i in range(size):
            v |= self.mem.get(addr + i, 0) << (8 * i)
            t |= self.shadow.get(addr + i, 0) << (8 * i)
        return v, t

    def write_taint(self, addr: int, mask: int, size: int) -> None:
        for i in range(size):
            self.shadow[addr + i] = (mask >> (8 * i)) & 0xFF

    def write_value(self, addr: int, value: int, size: int) -> None:
        """A store's VALUE, committed alongside its taint.

        Committing only the taint is how the engine's own block mode lost a
        byte: a later load of the same address then gets the right taint and a
        stale value, and any address derived from that value points somewhere
        else.  A harness that did the same could not see it.
        """
        for i in range(size):
            self.mem[addr + i] = (value >> (8 * i)) & 0xFF


#: A region runner: (taint out, values out) for one lowered region, given
#: the machine it reads and writes memory through.
RunRegion = Callable[[IRProg, dict[str, int], dict[str, int], _Machine],
                     tuple[dict[str, int], dict[str, int]]]


@pytest.fixture(scope='module')
def kit() -> tuple[frompcode.Builder, RunRegion]:
    from microtaint.instrumentation.cell_c import taint_ir_c
    from microtaint.taint_ir.exec import compile_program

    builder = frompcode.Builder(_ARCH, False)
    names = sorted(set(builder.name_by_off.values()))
    layout = {n: i for i, n in enumerate(names)}

    def slot_of(key: IRKey) -> int:
        # this layout only places the tuple keys; a bare register name
        # never reaches it
        if not isinstance(key, tuple):
            raise KeyError(key)
        if key[0] in ('reg', 'regv'):
            name = builder.name_by_off.get(key[1])
            if name is None or name not in layout:
                raise KeyError(key)
            return layout[name] + (256 if key[0] == 'regv' else 0)
        if key[0] in _OFF:
            return _MEMBASE + _PER_ACC * key[1] + _OFF[key[0]]
        raise KeyError(key)

    def run(prog: IRProg, values: dict[str, int], taints: dict[str, int],
            machine: _Machine) -> tuple[dict[str, int], dict[str, int]]:
        """The two-pass protocol, as fastpath.h::mt_ir_mem_step performs it.

        Returns (taint, values): a region hands the NEXT region its register
        values, since the emulator cannot be asked for them part way through a
        block.  The value slot is seeded with the register's current value, so
        a register the program does not write reads back unchanged and one it
        writes to zero is not mistaken for one it never touched.
        """
        capsule, _ = compile_program(prog, slot_of)
        sv = [0] * _NS
        st = [0] * _NS
        for name, slot in layout.items():
            sv[slot] = values.get(name, 0) & 0xFFFFFFFFFFFFFFFF
            st[slot] = taints.get(name, 0)
            st[slot + 256] = sv[slot]
        first = taint_ir_c.run(capsule, sv, st)          # pass 1: the addresses
        for k, acc in enumerate(prog.accesses):
            if acc['kind'] != 'load':
                continue
            addr = first[_MEMBASE + _PER_ACC * k + _OFF['addr']]
            value, taint = machine.read(addr, acc['size'])
            sv[_MEMBASE + _PER_ACC * k] = value
            st[_MEMBASE + _PER_ACC * k] = taint
        out = taint_ir_c.run(capsule, sv, st)            # pass 2: the taint
        for k, acc in enumerate(prog.accesses):
            if acc['kind'] == 'load':
                continue
            base = _MEMBASE + _PER_ACC * k
            machine.write_taint(out[base + _OFF['addr']],
                                out[base + _OFF['sttaint']], acc['size'])
            machine.write_value(out[base + _OFF['addr']],
                                out[base + _OFF['stval']], acc['size'])
        return ({n: out[s] for n, s in layout.items()},
                {n: out[s + 256] for n, s in layout.items()})

    return builder, run


def _pcode(code: bytes, base: int) -> list[PcodeOp]:
    from microtaint.sleigh.lifter import get_context
    return get_context(_KEY).translate(code, base).ops


_BASE = 0x20000
_SEED = {'RAX': 0xAABBCCDD11223344, 'RBX': 5, 'RDX': 0x99,
         'RBP': _BASE, 'RSP': _BASE + 0x100}

_CASES = {
    #: The spill: store a tainted register, read it straight back.
    'store_then_load_same': ['48894508', '488b4508'],
    'push_then_pop': ['50', '5b'],
    'store_load_store': ['48894508', '488b5508', '48895510'],
    #: Different addresses: must still agree, and must not be over-cut for free.
    'store_then_load_other': ['48894508', '488b4510'],
}


@pytest.mark.parametrize('name', sorted(_CASES))
@pytest.mark.parametrize('tainted', ['RAX', 'RDX'])
def test_regions_agree_with_the_sequence_through_memory(kit: tuple[frompcode.Builder, RunRegion],
                                                        name: str,
                                                        tainted: str) -> None:
    builder, run = kit
    seq = [bytes.fromhex(h) for h in _CASES[name]]
    seed_taint = {tainted: 0xFF}

    # Reference: one instruction at a time, stores committed between them.
    ref_machine = _Machine()
    taint, values = dict(seed_taint), dict(_SEED)
    base = frompcode.LIFT_BASE
    for code in seq:
        try:
            prog = builder.build(_pcode(code, base), base + len(code), emit=Emit.BOTH)
        except Unsupported:
            pytest.skip(f'{code.hex()} does not lower on its own')
        taint, values = run(prog, values, taint, ref_machine)
        base += len(code)

    # Candidate: the planned regions.
    got_machine = _Machine()
    got, got_values = dict(seed_taint), dict(_SEED)
    for region in plan_block(_ARCH, b''.join(seq), builder=builder):
        if region.prog is None:
            pytest.skip(f'{name}: a region did not lower')
        got, got_values = run(region.prog, got_values, got, got_machine)

    lost = {k: (taint[k], got[k]) for k in taint if taint[k] & ~got[k]}
    assert not lost, (
        f'{name} with {tainted} tainted: the regions LOST taint the sequence '
        f'holds: ' + ', '.join(f'{k} sequence {a:#x}, regions {b:#x}'
                               for k, (a, b) in sorted(lost.items())))
    differing = {k: (taint[k], got[k]) for k in taint if taint[k] != got[k]}
    assert not differing, (
        f'{name} with {tainted} tainted: regions disagree with the sequence: '
        + ', '.join(f'{k} sequence {a:#x}, regions {b:#x}'
                    for k, (a, b) in sorted(differing.items())))


def test_a_load_after_a_store_declines_in_block_mode() -> None:
    """The rule itself, at the lowering, so no caller can bypass it."""
    builder = frompcode.Builder(_ARCH, False)
    ops, base = [], frompcode.LIFT_BASE
    for code in (bytes.fromhex('48894508'), bytes.fromhex('488b4508')):
        ops.extend(_pcode(code, base))
        base += len(code)
    with pytest.raises(Unsupported, match='load after store'):
        builder.build(ops, base, emit=Emit.BOTH, block=True)


def test_a_single_instruction_that_stores_and_loads_is_unaffected() -> None:
    """The rule must fire only on an ordering several instructions create.  A
    read-modify-write is one instruction and the protocol handles it exactly."""
    builder = frompcode.Builder(_ARCH, False)
    for code in (bytes.fromhex('48014508'),      # add [rbp+8], rax
                 bytes.fromhex('48294508'),      # sub [rbp+8], rax
                 bytes.fromhex('48ff4508')):     # inc qword [rbp+8]
        ops = _pcode(code, frompcode.LIFT_BASE)
        end = frompcode.LIFT_BASE + len(code)
        plain = builder.build(ops, end, emit=Emit.BOTH)
        as_block = builder.build(ops, end, emit=Emit.BOTH, block=True)
        assert [k for k, _ in plain.outputs] == [k for k, _ in as_block.outputs], code.hex()


def test_the_decline_says_which_instruction_caused_it() -> None:
    """A decline carries the position, so the planner cuts instead of searching.

    Without it, splitting a block means lowering candidate lengths until one
    works, and a lowering is a SLEIGH translation plus an IR build: measured at
    up to 201 ms for a single block, which is paid on that block's first
    execution.  With it, the boundary is read off the exception.
    """
    builder = frompcode.Builder(_ARCH, False)
    seq = [bytes.fromhex('4883c001'),      # add rax, 1
           bytes.fromhex('48894508'),      # mov [rbp+8], rax      <- the store
           bytes.fromhex('488b4508'),      # mov rax, [rbp+8]      <- the load
           bytes.fromhex('4801d8')]        # add rax, rbx
    ops, base = [], frompcode.LIFT_BASE
    for code in seq:
        ops.extend(_pcode(code, base))
        base += len(code)
    with pytest.raises(Unsupported) as caught:
        builder.build(ops, base, emit=Emit.BOTH, block=True)
    assert caught.value.cut_at == 2, (
        f'the load is the third instruction (ordinal 2) and the decline '
        f'reported {caught.value.cut_at}; the planner would cut in the wrong '
        f'place, or fall back to searching')


def test_the_planner_cuts_where_the_decline_said() -> None:
    """And the cut it reports is a boundary that actually lowers."""
    builder = frompcode.Builder(_ARCH, False)
    code = (bytes.fromhex('4883c001') + bytes.fromhex('48894508')
            + bytes.fromhex('488b4508') + bytes.fromhex('4801d8'))
    regions = plan_block(_ARCH, code, builder=builder)
    assert sum(r.count for r in regions) == 4, 'the block was not fully covered'
    assert regions[0].count == 2, (
        f'the first region should hold the two instructions before the load, '
        f'got {regions[0].count}')
    assert all(r.prog is not None for r in regions), 'a region did not lower'


#: Store a byte, load it back, and USE IT AS AN ADDRESS.  The taint of the
#: final load then depends on the VALUE that went through memory, which is the
#: one thing a taint-only commit cannot carry.
_ADDR_SEQ = ['8855ef',      # mov  [rbp-0x11], dl
             '0fb645ef',    # movzx eax, byte [rbp-0x11]
             '0fb60c03']    # movzx ecx, byte [rbx+rax]


def test_a_stored_value_reaches_a_later_load_that_uses_it_as_an_address(
        kit: tuple[frompcode.Builder, RunRegion]) -> None:
    """The shape that cost a byte on bench_dense.

    The swap loop in that guest stores `idx`, reloads it two instructions later
    and indexes `state` with it.  Committing the store's TAINT but not its
    VALUE gives the reload the right taint and a stale value, so `state[idx]`
    reads and writes the PREVIOUS iteration's index.  It mostly self-heals,
    because the wrong byte is usually tainted too, which is exactly why it
    surfaced as one lost byte rather than as obvious chaos.

    Here the index is CLEAN, so the avalanche pointer policy cannot taint the
    result by itself and the answer can only come from the right address.
    """
    builder, run = kit
    seq = [bytes.fromhex(h) for h in _ADDR_SEQ]
    # RDX is clean and holds 0x99, so the reloaded index is 0x99 and the final
    # load must land on RBX + 0x99.  Only that byte is tainted.
    want_addr = _SEED['RBX'] + (_SEED['RDX'] & 0xFF)

    def machine() -> _Machine:
        m = _Machine()
        m.shadow[want_addr] = 0xFF
        m.mem[want_addr] = 0x5A
        return m

    ref_machine = machine()
    ref: dict[str, int] = {}
    ref_values = dict(_SEED)
    base = frompcode.LIFT_BASE
    for code in seq:
        prog = builder.build(_pcode(code, base), base + len(code), emit=Emit.BOTH)
        ref, ref_values = run(prog, ref_values, ref, ref_machine)
        base += len(code)
    assert ref.get('RCX', 0), (
        'the per-instruction sequence did not taint RCX either, so this test '
        'cannot tell a stale value from a correct one')

    got_machine = machine()
    got: dict[str, int] = {}
    got_values = dict(_SEED)
    for region in plan_block(_ARCH, b''.join(seq), builder=builder):
        assert region.prog is not None, 'a region did not lower'
        got, got_values = run(region.prog, got_values, got, got_machine)

    assert got.get('RCX', 0) == ref.get('RCX', 0), (
        f'the regions read from the wrong address: RCX taint '
        f'{got.get("RCX", 0):#x} against {ref.get("RCX", 0):#x} for the same '
        f"instructions run one at a time.  A store's value did not reach the "
        f'load that indexes with it.')
