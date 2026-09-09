"""What fraction of execution could a block hook handle on its own?

    python block_coverage.py bench_untainted.elf


The block hook fires BEFORE the block runs, so it can never ask the emulator for
a register value part-way through: everything after the block's first
instruction has to be computed by the lowering.  A block is therefore fully
handleable only if EVERY one of its instructions lowers.  One that does not
forces a fallback, and a fallback needs per-instruction register state, which
only a code hook can supply -- and registering a code hook at all costs ~18 ns
per instruction, which is most of what block hooking is trying to save.

So: how much of a real run lives in blocks that need no fallback?  Measured
2026-09-09, this is 100% of block executions on bench_untainted and bench_dense
and 99.98% on bench_sparse (2 executions of 2 distinct blocks), which is what
says the fallback should be a code hook registered for just those blocks rather
than a reason to abandon block mode.
"""
# ruff: noqa: PLC0415, S110, T201
import io, os, sys, collections
from qiling import Qiling
from qiling.const import QL_VERBOSE
from unicorn import UC_HOOK_BLOCK
from microtaint.taint_ir.blocks import plan_block, instruction_starts, _translate
from microtaint.taint_ir.frompcode import Builder, LIFT_BASE
from microtaint.types import Architecture

GUEST = sys.argv[1]
ARCH = Architecture.AMD64
saved, dn = os.dup(1), os.open(os.devnull, os.O_WRONLY)
ql = Qiling([GUEST], '/', verbose=QL_VERBOSE.OFF)
ql.os.stdin = io.BytesIO(bytes((i*7+13) & 0xFF for i in range(64)))
blocks = collections.Counter()
ql.uc.hook_add(UC_HOOK_BLOCK, lambda uc, a, s, u: blocks.update({(a, s): 1}))
os.dup2(dn, 1)
try: ql.run()
except Exception: pass
os.dup2(saved, 1)

b = Builder(ARCH, False, 'concrete')
tot_instr = ok_instr = tot_blocks = ok_blocks = 0
bad_blocks = collections.Counter()
regions_in_ok = 0
for (addr, size), count in blocks.items():
    try: code = bytes(ql.mem.read(addr, size))
    except Exception: continue
    n = len(instruction_starts(_translate(ARCH, code, LIFT_BASE)))
    if not n: continue
    regs = plan_block(ARCH, code, builder=b)
    covered = sum(r.count for r in regs)
    full = covered == n and all(r.prog is not None for r in regs)
    tot_instr += n * count; tot_blocks += count
    if full:
        ok_instr += n * count; ok_blocks += count
        regions_in_ok += len(regs) * count
    else:
        bad_blocks[addr] += count

print(f'{os.path.basename(GUEST)}')
print(f'  block executions fully handleable : {ok_blocks}/{tot_blocks} '
      f'({100*ok_blocks/max(tot_blocks,1):.2f}%)')
print(f'  INSTRUCTIONS in those blocks      : {ok_instr}/{tot_instr} '
      f'({100*ok_instr/max(tot_instr,1):.3f}%)')
if ok_blocks:
    print(f'  regions per handleable block      : {regions_in_ok/ok_blocks:.2f}'
          f'   ({ok_instr/max(regions_in_ok,1):.2f} instrs/region)')
if bad_blocks:
    print(f'  blocks needing a fallback         : {len(bad_blocks)} distinct, '
          f'{sum(bad_blocks.values())} executions')
