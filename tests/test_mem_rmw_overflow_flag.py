"""`add [mem], reg` under-taints OF when the memory operand is clean.

SLEIGH lifts the read-modify-write form by loading the memory operand once per
flag (dumping `001c24`, `add byte ptr [rsp], bl`):

    unique = *[ram]RSP ; CF = carry(unique, BL)
    unique = *[ram]RSP ; OF = scarry(unique, BL)
    unique = *[ram]RSP ; unique = unique + BL ; *[ram]RSP = unique
    unique = *[ram]RSP ; SF = unique s< 0        <- reloads the STORED result
    ...

so OF is `scarry(mem_old, BL)` -- architecturally defined for `add`, and
non-monotone in either operand, which is exactly what a 2-corner differential
cannot see.

The rule covers OF with the differential plus a set of floors, and every floor is
conditioned on something that a clean memory operand switches off:

    AVALANCHE(T_MEM)                      memory taint
    AVALANCHE(T_MEM & 0x7f) AND ...       memory taint
    FULLMASK_AVAL(T_RBX[7:0])             BL tainted in FULL

With memory clean and BL tainted in a few bits, none of them fire and OF is left
to the differential alone. Measured at 201 under-taints in 4,000 random states
(5%); the reason it looks rare in the wider suite is that most generated cases
taint memory too, which switches the floors back on.

For a register-only `add rax, rbx` the exact signed-overflow term applies and OF
is exact. The gap is that the term is not reached when one addend arrives via a
LOAD.
"""

from __future__ import annotations

import itertools
import random

from microtaint.instrumentation.ast import EvalContext, InstructionCellExpr, LogicCircuit
from microtaint.simulator import CellSimulator, MachineState
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, ImplicitTaintPolicy, Register

_ARCH = Architecture.AMD64
_RN = ('RAX', 'RBX', 'RCX', 'RDX', 'RSP', 'RBP', 'RSI', 'RDI')
_FL = ('CF', 'OF', 'SF', 'ZF', 'PF', 'AF')
_REGS = [Register(n, 64) for n in _RN] + [Register(n, 1) for n in _FL]
_M64 = (1 << 64) - 1
_RSP = 0x7000
_HEXS = '001c24'          # add byte ptr [rsp], bl
_MK = f'MEM_{hex(_RSP)}_1'


def _true_of(sim: CellSimulator, base: dict[str, int], taint: dict[str, int]) -> int:
    """1 iff some assignment of the tainted bits changes OF."""
    ice = InstructionCellExpr(_ARCH, _HEXS, 'OF', 0, 0, {})
    pos = [(r, b) for r in _RN for b in range(64) if (taint.get(r, 0) >> b) & 1]
    outs = []
    for combo in itertools.product((0, 1), repeat=len(pos)):
        v = dict(base)
        for (r, b), bit in zip(pos, combo, strict=True):
            v[r] = (v[r] | (1 << b)) if bit else (v[r] & ~(1 << b) & _M64)
        outs.append(sim.evaluate_concrete(
            ice, MachineState(regs={r: v[r] & _M64 for r in _RN}, mem={_RSP: v[_MK]})))
    return int(any(o != outs[0] for o in outs[1:]))


def _got_of(sim: CellSimulator, circ: LogicCircuit, base: dict[str, int], taint: dict[str, int]) -> int:
    out = circ.evaluate(EvalContext(input_values=dict(base), input_taint=taint,
                                    simulator=sim, implicit_policy=ImplicitTaintPolicy.IGNORE))
    return int(out.get('OF', 0)) & 1


def test_mem_rmw_overflow_flag_one_state() -> None:
    """The first state the search finds: mem=0x60, BL=0x18, tainted bits 0xa4."""
    sim = CellSimulator(_ARCH)
    circ = generate_static_rule(_ARCH, bytes.fromhex(_HEXS), _REGS)
    taint = dict.fromkeys(_RN, 0)
    taint['RBX'] = 0xA4
    taint[_MK] = 0
    base = dict.fromkeys(_RN, 0)
    base['RSP'] = _RSP
    base['RBX'] = 0x18
    base[_MK] = 0x60
    assert _true_of(sim, base, taint) == 1, 'the state must actually move OF'
    assert _got_of(sim, circ, base, taint) == 1, 'OF under-tainted'


def test_mem_rmw_overflow_flag_never_under_taints() -> None:
    """Over a fixed sweep, not one state where OF moves may be reported clean."""
    sim = CellSimulator(_ARCH)
    circ = generate_static_rule(_ARCH, bytes.fromhex(_HEXS), _REGS)
    rng = random.Random(0)
    under = []
    for _ in range(400):
        taint = dict.fromkeys(_RN, 0)
        taint['RBX'] = sum(1 << b for b in rng.sample(range(8), rng.randint(1, 3)))
        taint[_MK] = 0                                   # memory clean
        base = {r: rng.getrandbits(64) & ~taint.get(r, 0) & _M64 for r in _RN}
        base['RSP'] = _RSP
        base[_MK] = rng.getrandbits(8)
        if _true_of(sim, base, taint) and not _got_of(sim, circ, base, taint):
            under.append((hex(base[_MK]), hex(base['RBX'] & 0xFF), hex(taint['RBX'])))
    assert not under, f'{len(under)}/400 states under-taint OF, e.g. {under[:3]}'
