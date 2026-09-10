# ruff: noqa: PLC0415, S112, C901
"""The lowering can publish the CPU's register VALUES, not just their taint.

Block-level tainting needs the register values BETWEEN the instructions of a
basic block.  Asking the emulator for them per instruction is precisely the cost
block tainting exists to remove: `uc_reg_read_batch` is 4.8 ns plus 11.3 ns per
register, and a per-instruction code hook is ~18 ns of Unicorn dispatch on top,
against 0.04 ns/instruction for a block hook.

The values are already there.  `frompcode.Builder.build` runs a value frame
beside the taint frame, aliasing identically, and publishes only the taint one;
dead-code elimination then deletes every value no taint rule happens to read.
`emit='value'` publishes the other frame instead, which costs the shipped path
nothing (the default is unchanged and the two never run together) and needs no
backend change at all: a value program serialises, compiles and runs through
exactly the same pipeline.

What has to be true for that to be worth anything is that the values are the
ones the hardware produces, so this checks them against Unicorn rather than
against the engine's own opinion.

Two families are excluded, both for reasons that predate this:
  * flags the ISA leaves architecturally UNDEFINED (x86 OF after a shift or
    rotate by other than 1, the flags after MUL/DIV/BSF).  Unicorn reports
    whatever the host CPU left in them; the lifter models a definition.  This
    is the same exclusion the fuzzing campaigns make.
  * VEX-encoded forms, which Unicorn is separately known to mis-decode, so it
    is not a usable oracle for them.
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

#: Flags the ISA leaves undefined for a family of forms.
_UNDEFINED = {
    'shift': ('OF', 'AF'),
    'rotate': ('OF', 'AF'),
    'mul': ('OF', 'CF', 'SF', 'ZF', 'AF', 'PF'),
    'div': ('OF', 'CF', 'SF', 'ZF', 'AF', 'PF'),
    'bit': ('OF', 'SF', 'ZF', 'AF', 'PF', 'CF'),
}


def _family(label: str) -> str:
    u = label.upper()
    if u.startswith(('SHL', 'SHR', 'SAR', 'SHLD', 'SHRD')):
        return 'shift'
    if u.startswith(('ROL', 'ROR', 'RCL', 'RCR')):
        return 'rotate'
    if u.startswith(('MUL', 'IMUL')):
        return 'mul'
    if u.startswith(('DIV', 'IDIV')):
        return 'div'
    if u.startswith(('BSF', 'BSR', 'BT')):
        return 'bit'
    return ''


def _is_vex(code: bytes) -> bool:
    """VEX two- and three-byte prefixes.  Unicorn mis-decodes several of these."""
    return bool(code) and code[0] in (0xC4, 0xC5)


class _Cpu:
    """One Unicorn instance, reused for every vector.

    oracle_harness._uc_run builds a fresh instance per call, which is right for
    a handful of vectors and wrong here: a few hundred instances in one process
    is the instability that crashes an xdist worker in test_riscv_microtaint.
    One instance, registers rewritten per run, is equivalent for a
    single-instruction run and does not accumulate.
    """

    def __init__(self, desc) -> None:
        import unicorn
        self.desc = desc
        self.uc = unicorn.Uc(desc.uc_arch, desc.uc_mode)
        self.uc.mem_map(desc.code_addr, 0x2000)

    def run(self, code: bytes, values: dict[str, int]) -> dict[str, int]:
        d = self.desc
        self.uc.mem_write(d.code_addr, code)
        for name, const in d.gp.items():
            self.uc.reg_write(const, values.get(name, 0) & d.mask)
        if d.eflags_reg is not None:
            self.uc.reg_write(d.eflags_reg, 0)
        self.uc.emu_start(d.code_addr, d.code_addr + len(code))
        out = {n: self.uc.reg_read(c) & d.mask for n, c in d.gp.items()}
        if d.eflags_reg is not None:
            ef = self.uc.reg_read(d.eflags_reg)
            for fname, bit in d.flags.items():
                out[fname] = (ef >> bit) & 1
        return out


@pytest.mark.parametrize('isa', ['AMD64', 'RISCV64'])
def test_lowered_values_match_the_cpu(isa: str) -> None:
    import oracle_harness as OH  # type: ignore[import-not-found]
    from instruction_bank import load_bank  # type: ignore[import-not-found]

    from microtaint.instrumentation.cell_c import taint_ir_c
    from microtaint.taint_ir import frompcode
    from microtaint.taint_ir.exec import compile_program

    bank = load_bank()
    if isa not in bank:
        pytest.skip(f'{isa} is not in the bank')
    make_desc = OH.UC_DESCS.get(isa)
    if make_desc is None:
        pytest.skip(f'{isa} has no Unicorn descriptor')
    spec = bank[isa]
    desc = make_desc()
    # The layout has to cover every register the LOWERING can name, not just the
    # ones the bank's descriptor lists: a program that mentions the program
    # counter, or any register the bank leaves out, is otherwise unplaceable and
    # gets skipped -- which on RISCV64 is all of them.
    frompcode.build_ir(spec.arch, spec.instructions[0].bytes, emit='value')
    # Asked for by ISA, not `next(iter(...))`: the cache holds one builder per
    # ISA and the first one in it belongs to whichever ISA ran first in this
    # process.
    builder = frompcode.builder_for(spec.arch)
    names = sorted(set(builder.name_by_off.values()))
    layout = {n: i for i, n in enumerate(names)}
    rnd = random.Random(20260909)
    cpu = _Cpu(desc)

    lowered = agree = compared = 0
    mismatches: list[str] = []
    # Bounded, and deliberately small: every vector builds a fresh Unicorn
    # instance, and building many hundreds in one process is the instability
    # that already makes test_riscv_microtaint crash an xdist worker.  This is a
    # property check, not a sweep; the standalone driver sweeps the bank.
    for ins in spec.instructions[:150]:
        if _is_vex(ins.bytes):
            continue
        try:
            prog = frompcode.build_ir(spec.arch, ins.bytes, emit='value')
        except Exception:
            continue
        # A memory program's state slots live PAST the register file, so running
        # it against an array sized to the register file writes out of bounds.
        # Registers only here; block tainting will need the memory shape, and
        # that is the two-pass protocol's problem, not this property's.
        if getattr(prog, 'accesses', None):
            continue
        if any(k[0] != 'regv' for k, _ in prog.outputs):
            continue
        def slot_of(key, builder=builder):
            # 'reg' is the INPUT namespace for both frames (the serializer
            # decides which array to read from), 'regv' is a published value
            # output.  Both land in the same slot here, so the run returns the
            # post-state values in the register positions.
            #
            # Raise rather than return -1 for anything this layout cannot
            # place: compile_program is documented to reject those, and a -1
            # slot is an out-of-bounds index once the program runs.
            if key[0] not in ('reg', 'regv'):
                raise KeyError(key)
            name = builder.name_by_off.get(key[1])
            if name is None or name not in layout:
                raise KeyError(key)
            return layout[name]

        try:
            capsule, _ = compile_program(prog, slot_of)
        except KeyError:
            continue                       # a name this layout cannot place
        lowered += 1
        written = [builder.name_by_off.get(k[1]) for k, _ in prog.outputs if k[0] == 'regv']
        skip = _UNDEFINED.get(_family(ins.label), ())

        for _ in range(2):
            values = {n: rnd.getrandbits(64) & desc.mask for n in desc.gp}
            try:
                truth = cpu.run(ins.bytes, values)
            except Exception:
                break
            state = [0] * len(names)
            for name, slot in layout.items():
                state[slot] = values.get(name, 0)
            got = taint_ir_c.run(capsule, state, [0] * len(names))
            checkable = [n for n in written
                         if n and n in layout and n in truth and n not in skip]
            compared += len(checkable)
            bad = [
                f'{ins.label}: {n} got {got[layout[n]]:#x} want {truth[n]:#x}'
                for n in checkable if got[layout[n]] != truth[n]
            ]
            if bad:
                mismatches.extend(bad[:1])
            else:
                agree += 1

    assert lowered > 20, f'only {lowered} forms lowered a value program; the mode is not working'
    # Without this the test passes vacuously whenever the lowering's register
    # names and the Unicorn descriptor's do not intersect, which is exactly what
    # a name-mapping mistake looks like.
    assert compared > 40, (
        f'only {compared} register values were actually compared against the CPU; '
        f'the lowering names ({sorted(set(names))[:6]}...) and the descriptor '
        f'names ({sorted(desc.gp)[:6]}...) may not line up')
    assert not mismatches, (
        f'{isa}: the lowered values disagree with the CPU on '
        f'{len(mismatches)} of {agree + len(mismatches)} vectors:\n  '
        + '\n  '.join(mismatches[:8]))


def test_value_mode_does_not_change_the_taint_program() -> None:
    """The shipped path must be untouched: same instruction, same taint program."""
    from instruction_bank import load_bank  # type: ignore[import-not-found]

    from microtaint.taint_ir import frompcode

    spec = load_bank()['AMD64']
    for ins in spec.instructions[:40]:
        try:
            a = frompcode.build_ir(spec.arch, ins.bytes)
            b = frompcode.build_ir(spec.arch, ins.bytes, emit='taint')
        except Exception:
            continue
        assert [k for k, _ in a.outputs] == [k for k, _ in b.outputs], ins.label
        assert len(a.nodes) == len(b.nodes), ins.label


def test_both_mode_agrees_with_each_alone() -> None:
    """One program can carry taint AND values, and must not change either.

    This is the form block tainting runs: the values a program publishes are the
    inputs of the next instruction in the block, so the emulator is asked for
    registers once per block instead of once per instruction.  What it must not
    do is answer either question differently from the single-purpose program.
    """
    from instruction_bank import load_bank  # type: ignore[import-not-found]

    from microtaint.taint_ir import frompcode

    spec = load_bank()['AMD64']
    key = spec.arch.value if hasattr(spec.arch, 'value') else str(spec.arch)
    checked = 0
    for ins in spec.instructions[:120]:
        try:
            only_t = frompcode.build_ir(spec.arch, ins.bytes, emit='taint')
            only_v = frompcode.build_ir(spec.arch, ins.bytes, emit='value')
            both = frompcode.build_ir(spec.arch, ins.bytes, emit='both')
        except Exception:  # noqa: BLE001 - unliftable forms are not the subject
            continue
        # A memory form also publishes addr / addrt / sttaint, which are the
        # two-pass protocol's business and identical in every mode; compare the
        # register outputs, which are what the mode selects.
        t_keys = [k for k, _ in only_t.outputs if k[0] == 'reg']
        v_keys = [k for k, _ in only_v.outputs if k[0] == 'regv']
        assert [k for k, _ in both.outputs if k[0] == 'reg'] == t_keys, ins.label
        assert [k for k, _ in both.outputs if k[0] == 'regv'] == v_keys, ins.label
        checked += 1
    assert checked > 40, f'only {checked} forms compared'
    assert key  # the builder cache key, kept so a future ISA loop reads naturally


def test_publishing_values_is_nearly_free() -> None:
    """The combined program must not cost much more than taint alone.

    The whole case for block tainting rests on this: a block hook is 0.04
    ns/instruction against ~18 for a code hook, but only if carrying the values
    through the block is cheaper than the ~46 ns/instruction of being hooked
    that it buys back.  Measured over the AMD64 bank the median is 1.1x, because
    the taint rules already compute most of these values and the IR is
    hash-consed, so publishing them shares the work rather than repeating it.

    The bound is deliberately loose: this guards against a regression that makes
    the values a second, separate computation, not against ordinary drift.
    """
    import statistics

    from instruction_bank import load_bank  # type: ignore[import-not-found]

    from microtaint.taint_ir import frompcode
    from microtaint.taint_ir.exec import serialize_for_c

    spec = load_bank()['AMD64']
    ratios = []
    for ins in spec.instructions[:200]:
        counts = {}
        for mode in ('taint', 'both'):
            try:
                prog = frompcode.build_ir(spec.arch, ins.bytes, emit=mode)
            except Exception:  # noqa: BLE001
                counts = None
                break
            builder = frompcode.builder_for(spec.arch)
            names = sorted(set(builder.name_by_off.values()))
            layout = {n: i for i, n in enumerate(names)}

            def slot_of(k, layout=layout, builder=builder):
                if k[0] not in ('reg', 'regv'):
                    raise KeyError(k)
                nm = builder.name_by_off.get(k[1])
                if nm is None or nm not in layout:
                    raise KeyError(k)
                return layout[nm] + (len(layout) if k[0] == 'regv' else 0)

            try:
                counts[mode] = len(serialize_for_c(prog, slot_of)['op_ids'])
            except KeyError:
                counts = None
                break
        if counts and counts.get('taint'):
            ratios.append(counts['both'] / counts['taint'])

    assert len(ratios) > 60, f'only {len(ratios)} forms measured'
    median = statistics.median(ratios)
    assert median < 1.6, (
        f'publishing values alongside taint now costs {median:.2f}x the ops of '
        f'taint alone (median over {len(ratios)} forms); it was 1.10x. '
        f'The values are supposed to be shared with the taint computation, not '
        f'recomputed beside it.')
