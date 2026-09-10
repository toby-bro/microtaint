"""Memory-operand flags must never UNDER-taint (soundness regression guard).

A flag derived from a MEMORY value operand through a non-monotone op (signed
compare, overflow, equality, or a shift's sign/zero flag) previously under-tainted:
the 2-corner differential missed it and the register-centric floors did not cover
the memory operand.  Three gaps, all fixed and guarded here:

  * the memory operand was not sign-split in the differential (a lossy D^{++});
  * the non-monotone flag floors iterated only register deps (memory skipped);
  * the COND_TRANSPORTABLE masked-replica cell used a bare ``MEM_<reg>`` key that
    did not round-trip the masked value, plus an RMW flag could re-LOAD the stored
    result.

For every instruction, output and (V, T) the emitted taint must CONTAIN the
brute-forced TRUE taint (non-constancy over the full register+memory taint cube).
Over-taint is allowed; under-taint is the bug.  MachineState.mem is keyed by
INTEGER address; the circuit reads memory via the MEM_<hex>_<size> string key.
"""
import itertools
import random

import pytest

from microtaint.instrumentation.ast import EvalContext, InstructionCellExpr
from microtaint.simulator import CellSimulator, MachineState
from microtaint.sleigh.engine import _cached_generate_static_rule, generate_static_rule
from microtaint.types import Architecture, ImplicitTaintPolicy, Register

ARCH = Architecture.AMD64
RN = ('RAX', 'RBX', 'RCX', 'RDX', 'RSP', 'RBP', 'RSI', 'RDI')
FL = ('CF', 'OF', 'SF', 'ZF', 'PF', 'AF')
REGS = [Register(n, 64) for n in RN] + [Register(n, 1) for n in FL]
M64 = (1 << 64) - 1
RSP = 0x7000

# label -> (bytes, mem_size_bytes)
CASES = {
    'cmp q[rsp],rbx': ('48391c24', 8), 'sub q[rsp],rbx': ('48291c24', 8),
    'add q[rsp],rbx': ('48011c24', 8), 'adc q[rsp],rbx': ('48111c24', 8),
    'sbb q[rsp],rbx': ('48191c24', 8), 'and q[rsp],rbx': ('48211c24', 8),
    'or q[rsp],rbx':  ('48091c24', 8), 'xor q[rsp],rbx': ('48311c24', 8),
    'test q[rsp],rbx': ('48851c24', 8), 'neg q[rsp]': ('48f71c24', 8),
    'shl q[rsp],7': ('48c1242407', 8), 'sar q[rsp],5': ('48c12c2405', 8),
    'shr q[rsp],3': ('48c12c2403', 8), 'inc q[rsp]': ('48ff0424', 8),
    'dec q[rsp]': ('48ff0c24', 8),
    'cmp b[rsp],bl': ('381c24', 1), 'sub b[rsp],bl': ('281c24', 1),
    'add b[rsp],bl': ('001c24', 1), 'sar b[rsp],3': ('c02c2403', 1),
    'cmp w[rsp],bx': ('66391c24', 2), 'cmp d[rsp],ebx': ('391c24', 4),
}


@pytest.fixture(scope='module')
def sim() -> CellSimulator:
    return CellSimulator(ARCH)


def _reg_targets(circ):
    return [a for a in circ.assignments
            if getattr(a.target, 'name', None) is not None and not hasattr(a.target, 'address_expr')]


def _true_taint(sim: CellSimulator, hexs: str, name: str, be, size: int, base, taint: dict[str, int]):
    ice = InstructionCellExpr(ARCH, hexs, name, 0, be, {})
    mk = f'MEM_{hex(RSP)}_{size}'
    pos = ([('r', r, b) for r in RN for b in range(64) if (taint.get(r, 0) >> b) & 1]
           + [('m', mk, b) for b in range(size * 8) if (taint.get(mk, 0) >> b) & 1])
    outs = []
    for combo in itertools.product((0, 1), repeat=len(pos)):
        v = dict(base)
        for (_k, key, b), bit in zip(pos, combo, strict=True):
            v[key] = (v[key] | (1 << b)) if bit else (v[key] & ~(1 << b) & M64)
        outs.append(sim.evaluate_concrete(ice, MachineState(regs={r: v[r] & M64 for r in RN}, mem={RSP: v[mk]})))
    acc = 0
    for o in outs[1:]:
        acc |= o ^ outs[0]
    return acc


@pytest.mark.parametrize(('label', 'hexs', 'size'),
                         [(k, v[0], v[1]) for k, v in CASES.items()], ids=list(CASES))
def test_memory_flags_never_under_taint(label: str, hexs: str, size: int, sim: CellSimulator) -> None:
    _cached_generate_static_rule.cache_clear()
    circ = generate_static_rule(ARCH, bytes.fromhex(hexs), REGS)
    targets = _reg_targets(circ)
    mk = f'MEM_{hex(RSP)}_{size}'
    rng = random.Random(hash(hexs) & 0xFFFF)
    for _ in range(40):
        rv = {r: rng.getrandbits(64) for r in RN}
        rv['RSP'] = RSP
        mv = rng.getrandbits(size * 8)
        taint = dict.fromkeys(RN, 0)
        for _ in range(rng.randint(0, 3)):
            taint[rng.choice(('RBX', 'RCX'))] |= (1 << rng.randrange(64))
        mt = 0
        for _ in range(rng.randint(0, 5)):
            mt |= (1 << rng.randrange(size * 8))
        taint[mk] = mt
        base = {**{r: rv[r] & ~taint[r] & M64 for r in RN}, mk: mv & ~mt}
        got = circ.evaluate(EvalContext(input_values={**base}, input_taint=taint, simulator=sim,
                                        implicit_policy=ImplicitTaintPolicy.IGNORE))
        for a in targets:
            nm = a.target.name
            be = a.target.bit_end - a.target.bit_start
            mask = (1 << (be + 1)) - 1
            true = _true_taint(sim, hexs, nm, be, size, base, taint) & mask
            g = got.get(nm, 0) & mask
            assert (true & ~g) == 0, f'{label} {nm} UNDER-taint: true={true:#x} got={g:#x} (mem_t={mt:#x})'
