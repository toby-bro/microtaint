# ruff: noqa: PLC0415, S112
"""Block equivalence on the big-endian architectures, on the ANSWER.

`test_block_equivalence` proves a block program computes what its instructions
compute, on AMD64.  Byte order is where a lowering breaks silently and keeps
passing every little-endian check, so the same property is asserted here on
MIPS64BE and PPC32BE, against Unicorn rather than against the engine.

Two things this file goes out of its way to avoid, because both made an earlier
version of it report success while comparing nothing:

  * random 64-bit register seeds make every load fault, Unicorn refuses the
    instruction, and the sequence is SKIPPED rather than verified -- 69 of 80
    MIPS sequences went that way.  Registers are seeded to point inside a mapped
    data page instead.
  * a register-name mismatch between the lowering and Unicorn silently empties
    the comparison (see the note in `_uc_desc_ppc32be`).  So the test asserts a
    minimum number of register values were actually compared, not merely that
    nothing disagreed.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT / 'benchmark'), str(_ROOT / 'tests')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from microtaint.taint_ir import frompcode  # noqa: E402
from microtaint.taint_ir.frompcode import Unsupported  # noqa: E402
from microtaint.types import Architecture  # noqa: E402

_CODE = 0x1000
_DATA = 0x40000
_ARCHES = {'MIPS64BE': Architecture.MIPS64BE, 'PPC32BE': Architecture.PPC32BE}


def _kit(isa: str):
    from microtaint.instrumentation.cell_c import taint_ir_c
    from microtaint.taint_ir.exec import compile_program

    arch = _ARCHES[isa]
    builder = frompcode.Builder(arch, True, 'concrete')     # big-endian
    names = sorted(set(builder.name_by_off.values()))
    layout = {n: i for i, n in enumerate(names)}
    width = 2 * len(layout)

    def slot_of(key):
        if key[0] not in ('reg', 'regv'):
            raise KeyError(key)
        name = builder.name_by_off.get(key[1])
        if name is None or name not in layout:
            raise KeyError(key)
        return layout[name] + (len(layout) if key[0] == 'regv' else 0)

    def run(prog, values, taints):
        capsule, _ = compile_program(prog, slot_of)
        v = [0] * width
        t = [0] * width
        for name, slot in layout.items():
            v[slot] = values.get(name, 0) & 0xFFFFFFFFFFFFFFFF
            t[slot] = taints.get(name, 0)
        out = taint_ir_c.run(capsule, v, t)
        return {n: out[s] for n, s in layout.items()}

    return arch, builder, run


def _pcode(isa: str, code: bytes, base: int):
    from microtaint.sleigh.lifter import get_context
    return get_context(isa).translate(code, base).ops


@pytest.mark.parametrize('isa', sorted(_ARCHES))
def test_block_equivalence_big_endian(isa: str) -> None:
    import unicorn

    import oracle_harness as OH  # type: ignore[import-not-found]
    from instruction_bank import load_bank  # type: ignore[import-not-found]

    bank = load_bank()
    if isa not in bank:
        pytest.skip(f'{isa} is not in the bank')
    spec = bank[isa]
    desc = OH.UC_DESCS[isa]()
    _arch, builder, run = _kit(isa)

    uc = unicorn.Uc(desc.uc_arch, desc.uc_mode)
    uc.mem_map(_CODE, 0x2000)
    uc.mem_map(_DATA, 0x10000)          # so a load or store has somewhere to go

    def step(code: bytes, values: dict) -> dict:
        for name, const in desc.gp.items():
            uc.reg_write(const, values.get(name, 0) & desc.mask)
        uc.mem_write(_CODE, code)
        uc.emu_start(_CODE, _CODE + len(code))
        return {n: uc.reg_read(c) & desc.mask for n, c in desc.gp.items()}

    rnd = random.Random(11)
    agree = compared = 0
    lost_examples: list[str] = []
    differing_examples: list[str] = []
    forms = [i for i in spec.instructions if len(i.bytes) == 4][:60]

    for start, trial in ((s_, t_) for s_ in range(0, max(len(forms) - 1, 0), 2)
                         for t_ in (0, 1)):
        seq = [bytes(i.bytes) for i in forms[start:start + 3]]
        if len(seq) < 2:
            break
        # Addresses, not random words: a random pointer faults and the whole
        # sequence is skipped instead of checked.
        seed = {n: _DATA + 0x800 + 8 * rnd.getrandbits(6) for n in desc.gp}
        states, current, ok = [], dict(seed), True
        for code in seq:
            states.append(dict(current))
            try:
                current = step(code, current)
            except Exception:
                ok = False
                break
        if not ok:
            continue

        # Two shapes per sequence.  ONE tainted register makes a lost
        # dependency show as a cleared bit instead of being masked by everything
        # already being tainted; EVERY register tainted with a distinct pattern
        # makes the answer depend on the whole sequence, so dropping any
        # instruction changes it.  With only the first, a random pick often
        # names a register the sequence never touches and the comparison is
        # insensitive to what the block did.
        if trial == 0:
            tainted = rnd.choice(sorted(desc.gp))
            seed_taint = {tainted: 0xFF}
        else:
            tainted = 'all'
            seed_taint = {n: (0xFF << (3 * (i % 8))) & 0xFFFFFFFF
                          for i, n in enumerate(sorted(desc.gp))}
        taint, base = dict(seed_taint), frompcode.LIFT_BASE
        for code, state in zip(seq, states, strict=True):
            try:
                prog = builder.build(_pcode(isa, code, base), base + len(code), emit='both')
            except Unsupported:
                ok = False
                break
            if getattr(prog, 'accesses', None):
                ok = False        # memory: the two-pass harness covers that shape
                break
            taint = run(prog, state, taint)
            base += len(code)
        if not ok:
            continue

        ops, base = [], frompcode.LIFT_BASE
        for code in seq:
            ops.extend(_pcode(isa, code, base))
            base += len(code)
        try:
            block = builder.build(ops, base, emit='both', block=True)
        except Unsupported:
            continue
        got = run(block, states[0], seed_taint)

        watched = [k for k in taint if k in desc.gp]
        compared += len(watched)
        lost = {k: (taint[k], got[k]) for k in watched if taint[k] & ~got[k]}
        differing = {k: (taint[k], got[k]) for k in watched if taint[k] != got[k]}
        if lost and len(lost_examples) < 4:
            lost_examples.append(
                f'{tainted} tainted: '
                + ', '.join(f'{k} sequence {a:#x}, block {b:#x}'
                            for k, (a, b) in sorted(lost.items())[:2]))
        if differing:
            if len(differing_examples) < 4:
                differing_examples.append(
                    f'{[c.hex() for c in seq]} with {tainted} tainted: '
                    + ', '.join(f'{k} sequence {a:#x}, block {b:#x}'
                                for k, (a, b) in sorted(differing.items())[:2]))
        else:
            agree += 1

    assert compared > 200, (
        f'{isa}: only {compared} register values were compared against the CPU. '
        f'The lowering names ({sorted(set(builder.name_by_off.values()))[:5]}...) '
        f'and the descriptor names ({sorted(desc.gp)[:5]}...) may not line up, '
        f'or every sequence faulted and was skipped rather than checked.')
    assert not lost_examples, (
        f'{isa}: the block program LOST taint the sequence holds:\n  '
        + '\n  '.join(lost_examples))
    # Equality, not "enough agreed".  An over-taint is not a soundness failure
    # but it is still a disagreement, and a threshold lets a mutation that
    # breaks some sequences hide behind the ones it does not.
    assert not differing_examples, (
        f'{isa}: the block program disagrees with the sequence:\n  '
        + '\n  '.join(differing_examples))
    assert agree > 5, f'{isa}: only {agree} sequences compared cleanly'
