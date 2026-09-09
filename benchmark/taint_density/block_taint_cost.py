"""What block-level tainting is worth, measured on the taint math alone.

    python block_taint_cost.py bench_untainted.elf

Runs the guest once to collect the basic blocks Unicorn actually executes, then
prices two ways of computing the same taint:

  per-instruction    one compiled program per instruction, which is what the
                     engine runs today
  block regions      the block planned into the largest regions that lower as
                     one program (microtaint/taint_ir/blocks.py)

Both go through the same compiler, the same host emitter and the same timer, so
the difference is the taint computation and nothing else: no hook, no emulator,
no memory.  Weighted by how often each block really executes, because a block
that runs once should not count as much as the loop body next to it.

This is deliberately only half the story.  The other half is the per-instruction
hook, which costs ~18 ns of Unicorn dispatch plus the engine's own trampoline
against 0.04 ns/instruction for a block hook (hookcost.c), and which a block
amortises over its length.  Read the two together: this says the arithmetic gets
cheaper, hookcost.c says the dispatch nearly disappears.
"""
# ruff: noqa: PLC0415, S110, T201
from __future__ import annotations

import collections
import io
import os
import sys


def main() -> int:
    guest = sys.argv[1]

    from qiling import Qiling
    from qiling.const import QL_VERBOSE
    from unicorn import UC_HOOK_BLOCK

    from microtaint.instrumentation.cell_c import taint_ir_c
    from microtaint.taint_ir import frompcode
    from microtaint.taint_ir.blocks import _translate, instruction_starts, plan_block
    from microtaint.taint_ir.exec import compile_program
    from microtaint.types import Architecture

    arch = Architecture.AMD64
    builder = frompcode.Builder(arch, False, 'concrete')
    names = sorted(set(builder.name_by_off.values()))
    layout = {n: i for i, n in enumerate(names)}
    kinds = ('addr', 'addrt', 'sttaint', 'mem')
    n_slots = 2 * len(layout) + 4 * 8      # taint | values | up to 8 accesses

    def slot_of(key):
        if key[0] in ('reg', 'regv'):
            name = builder.name_by_off.get(key[1])
            if name is None or name not in layout:
                raise KeyError(key)
            return layout[name] + (len(layout) if key[0] == 'regv' else 0)
        if key[0] in kinds:
            return 2 * len(layout) + 4 * key[1] + kinds.index(key[0])
        raise KeyError(key)

    def nanoseconds(prog) -> float:
        capsule, _ = compile_program(prog, slot_of)
        taint_ir_c.jit(capsule)            # the host emitter, where it takes it
        ns, _sink = taint_ir_c.bench(capsule, [0] * n_slots, [0] * n_slots)
        return ns

    # Collect the blocks this guest really runs.
    saved, devnull = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    ql = Qiling([guest], '/', verbose=QL_VERBOSE.OFF)
    ql.os.stdin = io.BytesIO(bytes((i * 7 + 13) & 0xFF for i in range(64)))
    blocks: collections.Counter = collections.Counter()
    ql.uc.hook_add(UC_HOOK_BLOCK, lambda uc, a, s, u: blocks.update({(a, s): 1}))
    os.dup2(devnull, 1)
    try:
        ql.run()
    except Exception:
        pass
    os.dup2(saved, 1)

    w_instr = w_per = w_blk = 0.0
    per_calls = blk_calls = 0
    for (addr, size), count in blocks.items():
        try:
            code = bytes(ql.mem.read(addr, size))
        except Exception:
            continue
        ops = _translate(arch, code, frompcode.LIFT_BASE)
        marks = instruction_starts(ops)
        if not marks:
            continue
        n = len(marks)
        per_ns = 0.0
        ok = True
        for i, (lo, _a, _sz) in enumerate(marks):
            hi = marks[i + 1][0] if i + 1 < n else len(ops)
            end = marks[i + 1][1] if i + 1 < n else frompcode.LIFT_BASE + len(code)
            try:
                per_ns += nanoseconds(builder.build(ops[lo:hi], end, emit='both'))
            except Exception:
                ok = False
                break
        if not ok:
            continue
        try:
            regions = plan_block(arch, code, builder=builder)
            if any(r.prog is None for r in regions):
                continue                    # not comparable: one lowers nowhere
            blk_ns = sum(nanoseconds(r.prog) for r in regions)
        except Exception:
            continue
        w_instr += n * count
        w_per += per_ns * count
        w_blk += blk_ns * count
        per_calls += n * count
        blk_calls += len(regions) * count

    if not w_instr:
        print(f'{os.path.basename(guest)}: nothing comparable')
        return 1
    print(f'{os.path.basename(guest)}  ({int(w_instr)} instructions, execution-weighted)')
    print(f'  per-instruction programs : {w_per / w_instr:7.2f} ns/instruction   '
          f'({per_calls} calls)')
    print(f'  block region programs    : {w_blk / w_instr:7.2f} ns/instruction   '
          f'({blk_calls} calls, {per_calls / max(blk_calls, 1):.1f}x fewer)')
    print(f'  taint math               : {w_blk / max(w_per, 1):.2f}x')
    return 0


if __name__ == '__main__':
    sys.exit(main())
