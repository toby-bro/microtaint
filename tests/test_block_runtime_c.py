"""The C block runtime: regions must answer what the instructions answer.

Block tainting is a performance feature, so its runtime is C -- `blockpath.h`,
with no PyObject and no GIL anywhere in it.  These tests drive that C directly,
through the thin surface in `blockpath_c.c`, against a byte-addressed memory the
test can write to.  No emulator is involved: what is being checked is the
runtime's own three subtleties, and each of them is easier to attack here than
in a live guest.

    the two-pass protocol   a load's address is computed BY the program, so the
                            program runs once for addresses and again for the
                            answer
    the overlay             a block can store to an address and load it back,
                            which no single instruction does, so both the taint
                            AND the value of that store must reach the load
    the deferred commit     the hook fires BEFORE the block runs, so a block
                            that faults part way must not have committed

The comparison is a differential inside the same C runtime: the same
instructions, once cut into the regions the planner chose and once with every
instruction its own region, must reach the same taint and leave the same
shadow.  Running both through the same code means a bug in the runner cancels
out and only a bug in the REGIONING shows -- and the store/load cases below are
exactly where regioning is dangerous.
"""
from __future__ import annotations

from typing import Any

import pytest

from microtaint.emulator import blockpath_c as B
from microtaint.taint_ir import frompcode
from microtaint.taint_ir.blockcompile import compile_block
from microtaint.types import Architecture

_ARCH = Architecture.AMD64
_BASE = 0x401000
_ARENA = 0x7000
_ARENA_LEN = 0x2000
FULL = (1 << 64) - 1


@pytest.fixture(scope='module')
def layout() -> dict[str, int]:
    builder = frompcode.builder_for(_ARCH)
    names = sorted(set(builder.name_by_off.values()))
    return {n: i for i, n in enumerate(names)}


def _arena() -> Any:
    m = B.mem_new(_ARENA, _ARENA_LEN)
    # Distinct bytes everywhere, so a load from the wrong address is visible in
    # the VALUE as well as in the taint.
    B.mem_poke(m, _ARENA, bytes((i * 7 + 3) & 0xFF for i in range(_ARENA_LEN)))
    return m


def _regs(layout: dict[str, int], values: dict[str, int]) -> list[int]:
    out = [0] * len(layout)
    for name, v in values.items():
        out[layout[name]] = v & FULL
    return out


def _run(layout: dict[str, int], seq: list[bytes], values: dict[str, int],
         taint: dict[str, int], *, as_block: bool) -> tuple[dict[str, int], Any]:
    """Run `seq` (a list of instruction byte strings) through the C runtime.

    `as_block` runs it as ONE block cut into planned regions; otherwise each
    instruction is its own block, which is the reference.  Returns
    (taint by name, memory shadow reader).
    """
    mem = _arena()
    runner = B.runner_new(len(layout), -1, mem)
    B.runner_seed(runner, _regs(layout, taint))
    regs = _regs(layout, values)

    chunks = [b''.join(seq)] if as_block else list(seq)
    addr = _BASE
    for code in chunks:
        got = compile_block(_ARCH, code, addr, layout, publish_all_values=True)
        assert got is not None, f'{code.hex()} at {addr:#x} did not compile'
        rc = B.runner_on_block(runner, got[0], addr, regs)
        assert rc == 0, f'the runtime declined {code.hex()} at {addr:#x}'
        regs = B.runner_values(runner)
        addr += len(code)
    B.runner_finish(runner, True)

    slot_taint = B.runner_taint(runner)
    return {n: slot_taint[s] for n, s in layout.items()}, mem


#: (label, instructions, seeded taint).  Every one of these puts a store and a
#: later load of the SAME address in one block, which is the ordering the
#: two-pass protocol cannot express and the planner has to cut.
_CASES = {
    'spill_and_reload': (['48894508', '488b4508'], {'RAX': FULL}),          # mov [rbp+8],rax ; mov rax,[rbp+8]
    'push_then_pop': (['50', '5b'], {'RAX': FULL}),
    'store_load_store': (['48894508', '488b5508', '48895510'], {'RAX': FULL}),
    'store_then_load_other': (['48894508', '488b4510'], {'RAX': FULL}),
    'byte_spill': (['8855ef', '0fb645ef'], {'RDX': 0xFF}),                  # mov [rbp-0x11],dl ; movzx eax,[rbp-0x11]
}

_VALUES = {'RAX': 0xAABBCCDD11223344, 'RBX': _ARENA + 0x40, 'RCX': 5,
           'RDX': 0x99, 'RBP': _ARENA + 0x400, 'RSP': _ARENA + 0x800}


@pytest.mark.parametrize('name', sorted(_CASES))
def test_regions_answer_what_the_instructions_answer(
        layout: dict[str, int], name: str) -> None:
    seq = [bytes.fromhex(h) for h in _CASES[name][0]]
    taint = _CASES[name][1]
    ref, ref_mem = _run(layout, seq, _VALUES, taint, as_block=False)
    got, got_mem = _run(layout, seq, _VALUES, taint, as_block=True)

    lost = {k: (ref[k], got[k]) for k in ref if ref[k] & ~got[k]}
    assert not lost, (
        f'{name}: the regions LOST taint the instruction sequence holds: '
        + ', '.join(f'{k} {a:#x} -> {b:#x}' for k, (a, b) in sorted(lost.items())))
    differing = {k: (ref[k], got[k]) for k in ref if ref[k] != got[k]}
    assert not differing, (
        f'{name}: regions disagree with the sequence: '
        + ', '.join(f'{k} sequence {a:#x}, regions {b:#x}'
                    for k, (a, b) in sorted(differing.items())))

    for a in range(_ARENA, _ARENA + _ARENA_LEN, 8):
        r, g = B.mem_mask(ref_mem, a, 8), B.mem_mask(got_mem, a, 8)
        assert r == g, f'{name}: shadow differs at {a:#x}: sequence {r:#x}, regions {g:#x}'


def test_a_stored_value_reaches_a_load_that_indexes_with_it(
        layout: dict[str, int]) -> None:
    """The shape that cost a byte on bench_dense.

    Store a byte, load it back, and use it as an index.  The taint of the final
    load depends on the VALUE that went through memory, so committing a store's
    taint without its value gives the right mask and the wrong address.  The
    index is CLEAN, so the avalanche pointer policy cannot supply the answer by
    itself and it can only come from reading the right byte.
    """
    seq = [bytes.fromhex(h) for h in ('8855ef',      # mov  [rbp-0x11], dl
                                      '0fb645ef',    # movzx eax, byte [rbp-0x11]
                                      '0fb60c03')]   # movzx ecx, byte [rbx+rax]
    values = dict(_VALUES)
    want = values['RBX'] + (values['RDX'] & 0xFF)

    def prepared() -> Any:
        mem = _arena()
        B.mem_taint(mem, want, 0xFF, 1)
        return mem

    # Reference: one instruction per block.
    outs: dict[bool, int] = {}
    for as_block in (False, True):
        mem = prepared()
        runner = B.runner_new(len(layout), -1, mem)
        regs = _regs(layout, values)
        chunks = [b''.join(seq)] if as_block else list(seq)
        addr = _BASE
        for code in chunks:
            built = compile_block(_ARCH, code, addr, layout, publish_all_values=True)
            assert built is not None, f'{code.hex()} did not compile'
            assert B.runner_on_block(runner, built[0], addr, regs) == 0
            regs = B.runner_values(runner)
            addr += len(code)
        B.runner_finish(runner, True)
        t = B.runner_taint(runner)
        outs[as_block] = t[layout['RCX']]

    assert outs[False], (
        'the instruction sequence did not taint RCX either, so this test '
        'cannot tell a stale value from a correct one')
    assert outs[True] == outs[False], (
        f'the block read from the wrong address: RCX taint {outs[True]:#x} '
        f'against {outs[False]:#x} for the same instructions one at a time. '
        f"A store's value did not reach the load that indexes with it.")


def test_a_register_the_block_zeroes_does_not_keep_its_old_value(
        layout: dict[str, int]) -> None:
    """`xor %eax,%eax` then index with RAX.

    A published value of zero must not be read as "this register was not
    written".  If it is, RAX keeps its entry value and the load lands somewhere
    else entirely.
    """
    seq = [bytes.fromhex(h) for h in ('31c0',        # xor eax, eax
                                      '0fb60c03')]   # movzx ecx, byte [rbx+rax]
    values = dict(_VALUES)
    B_ARENA = values['RBX']
    mem = _arena()
    B.mem_taint(mem, B_ARENA, 0xFF, 1)              # only [rbx+0] is tainted

    runner = B.runner_new(len(layout), -1, mem)
    built = compile_block(_ARCH, b''.join(seq), _BASE, layout,
                            publish_all_values=True)
    assert built is not None, 'the block did not compile'
    assert B.runner_on_block(runner, built[0], _BASE, _regs(layout, values)) == 0
    B.runner_finish(runner, True)
    t = B.runner_taint(runner)
    assert t[layout['RCX']], (
        'RAX was zeroed by the block, so the load must read [rbx+0], which is '
        'the one tainted byte.  A stale RAX sends it elsewhere and RCX comes '
        'back clean.')
