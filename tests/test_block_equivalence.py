# ruff: noqa: PLC0415, S112
"""A block program must compute what its instructions compute in sequence.

This is the correctness question block-level tainting rests on.  Lowering a
whole basic block as one program is a large win (7.5x fewer compiled calls,
roughly half the executed taint ops), but only if the answer is the same one the
per-instruction programs produce.  Cross-instruction dead-code elimination is
supposed to delete work that is genuinely dead -- a flag the next instruction
overwrites -- and never work that is merely hard to see.

The comparison is run on real values.  Unicorn executes the sequence one
instruction at a time and the register state before each is captured, so the
reference runs every per-instruction program against the values that instruction
would really have seen.  The block program gets only the state at block entry,
which is the whole point: it has to derive the rest itself.

Register-only sequences here.  A memory access needs the two-pass protocol --
addresses resolved against guest memory and the shadow -- and comparing that
belongs with the runtime that implements it.
"""
from __future__ import annotations

import pytest

from microtaint.taint_ir import frompcode
from microtaint.taint_ir.frompcode import Unsupported
from microtaint.types import Architecture

_ARCH, _KEY = Architecture.AMD64, 'AMD64'

#: Straight-line register-only runs.  Each is chosen so a later instruction
#: overwrites flags an earlier one wrote, which is exactly the work
#: cross-instruction dead-code elimination is allowed to delete.
_SEQUENCES = {
    'add_then_xor': [
        bytes.fromhex('4801d8'),      # add rax, rbx
        bytes.fromhex('4831c8'),      # xor rax, rcx
    ],
    'arith_chain': [
        bytes.fromhex('4801d8'),      # add rax, rbx
        bytes.fromhex('4829c8'),      # sub rax, rcx
        bytes.fromhex('4821d0'),      # and rax, rdx
    ],
    'move_and_flags': [
        bytes.fromhex('4889d8'),      # mov rax, rbx
        bytes.fromhex('4839c8'),      # cmp rax, rcx
        bytes.fromhex('4809d0'),      # or  rax, rdx
    ],
    'shift_then_add': [
        bytes.fromhex('48c1e004'),    # shl rax, 4
        bytes.fromhex('4801d8'),      # add rax, rbx
    ],
}

_SEEDS = (
    {'RAX': 0x0123456789ABCDEF, 'RBX': 0xFEDCBA9876543210,
     'RCX': 0x00FF00FF00FF00FF, 'RDX': 0x5555555555555555},
    {'RAX': 1, 'RBX': 0xFFFFFFFFFFFFFFFF, 'RCX': 0, 'RDX': 0x8000000000000000},
)
#: One tainted input per run, so a lost dependency shows up as a cleared bit
#: rather than being masked by everything else already being tainted.
_TAINTS = ({'RBX': 0xFF}, {'RCX': 0xFF00}, {'RAX': 0x1})


def _ops(code: bytes, base: int):
    from microtaint.sleigh.lifter import get_context
    return get_context(_KEY).translate(code, base).ops


@pytest.fixture(scope='module')
def kit():
    """Builder, register layout, slot mapping and a compiled-program runner."""
    from microtaint.instrumentation.cell_c import taint_ir_c
    from microtaint.taint_ir.exec import compile_program

    builder = frompcode.Builder(_ARCH, False, 'concrete')
    names = sorted(set(builder.name_by_off.values()))
    layout = {n: i for i, n in enumerate(names)}

    def slot_of(key):
        if key[0] not in ('reg', 'regv'):
            raise KeyError(key)
        name = builder.name_by_off.get(key[1])
        if name is None or name not in layout:
            raise KeyError(key)
        # Taint in the low half, published values in the high half.
        return layout[name] + (len(layout) if key[0] == 'regv' else 0)

    def run(prog, values, taints):
        capsule, _ = compile_program(prog, slot_of)
        n = 2 * len(layout)
        v = [0] * n
        t = [0] * n
        for name, slot in layout.items():
            v[slot] = values.get(name, 0)
            t[slot] = taints.get(name, 0)
        out = taint_ir_c.run(capsule, v, t)
        taint_out = {n_: out[s] for n_, s in layout.items()}
        value_out = {n_: out[len(layout) + s] for n_, s in layout.items()}
        return taint_out, value_out

    return builder, layout, run


def _unicorn_states(seq: list[bytes], seed: dict, names) -> list[dict] | None:
    """Register state before each instruction, from a real execution."""
    import unicorn
    import unicorn.x86_const as ux

    consts = {n: getattr(ux, f'UC_X86_REG_{n}', None) for n in names}
    consts = {n: c for n, c in consts.items() if c is not None}
    uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
    base = 0x1000
    uc.mem_map(base, 0x2000)
    uc.mem_write(base, b''.join(seq))
    for name, const in consts.items():
        uc.reg_write(const, seed.get(name, 0) & 0xFFFFFFFFFFFFFFFF)
    uc.reg_write(ux.UC_X86_REG_EFLAGS, 0)

    states, addr = [], base
    for code in seq:
        state = {n: uc.reg_read(c) for n, c in consts.items()}
        ef = uc.reg_read(ux.UC_X86_REG_EFLAGS)
        for fname, bit in (('CF', 0), ('PF', 2), ('AF', 4),
                           ('ZF', 6), ('SF', 7), ('OF', 11)):
            state[fname] = (ef >> bit) & 1
        states.append(state)
        try:
            uc.emu_start(addr, addr + len(code))
        except Exception:  # noqa: BLE001 - Unicorn declined this form
            return None
        addr += len(code)
    return states


@pytest.mark.parametrize('name', sorted(_SEQUENCES))
@pytest.mark.parametrize('seed_i', range(len(_SEEDS)))
@pytest.mark.parametrize('taint_i', range(len(_TAINTS)))
def test_block_program_matches_the_sequence(kit, name, seed_i, taint_i) -> None:
    builder, layout, run = kit
    seq = _SEQUENCES[name]
    seed, taint_in = _SEEDS[seed_i], _TAINTS[taint_i]

    states = _unicorn_states(seq, seed, layout)
    if states is None:
        pytest.skip('Unicorn declined this sequence')

    # Reference: each instruction's own program, against the values it really
    # saw, with the taint carried forward.
    taint = dict(taint_in)
    base = frompcode.LIFT_BASE
    for code, state in zip(seq, states, strict=True):
        try:
            prog = builder.build(_ops(code, base), base + len(code), emit='both')
        except Unsupported:
            pytest.skip(f'{code.hex()} does not lower on its own')
        out, _v = run(prog, state, taint)
        taint = out
        base += len(code)
    reference = taint

    # Candidate: one program for the whole run, given only the entry state.
    ops, base = [], frompcode.LIFT_BASE
    for code in seq:
        ops.extend(_ops(code, base))
        base += len(code)
    try:
        block = builder.build(ops, base, emit='both', block=True)
    except Unsupported as exc:
        pytest.skip(f'the block does not lower: {exc}')
    got, _v = run(block, states[0], taint_in)

    differing = {k: (reference[k], got[k]) for k in reference if reference[k] != got[k]}
    assert not differing, (
        f'{name}: the block program disagrees with the same instructions run in '
        f'sequence, on {sorted(differing)}:\n  '
        + '\n  '.join(f'{k}: sequence {a:#x}, block {b:#x}'
                      for k, (a, b) in sorted(differing.items())))


@pytest.mark.parametrize('name', sorted(_SEQUENCES))
def test_a_clean_input_stays_clean_through_a_block(name: str, kit) -> None:
    """The untainted-input exit's premise, at block scale: no block may
    manufacture taint out of nothing."""
    builder, layout, run = kit
    seq = _SEQUENCES[name]
    ops, base = [], frompcode.LIFT_BASE
    for code in seq:
        ops.extend(_ops(code, base))
        base += len(code)
    try:
        block = builder.build(ops, base, emit='both', block=True)
    except Unsupported as exc:
        pytest.skip(f'the block does not lower: {exc}')
    out, _v = run(block, _SEEDS[0], {})
    dirty = {k: v for k, v in out.items() if v}
    assert not dirty, f'{name}: taint appeared from a clean state: {dirty}'
