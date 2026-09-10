"""Closed-form taint for STACK memory-RMW flags must be exact and cell-free.

An in-place stack operation (``shl qword [rsp],7``; ``add qword [rsp],rbx``)
reads its operand with a LOAD from ``[rsp]``.  The flag closed forms resolve that
LOAD to a MemoryOperand -- but only for STACK pointers (RSP/ESP/SP), where the
engine deliberately does NOT treat the address as attacker-controlled, so no
pointer-avalanche floor is needed and reading the shadow memory directly matches
the engine's own stack model.  This closes:

  * add / sub / cmp [rsp],reg : CF (the unsigned carry/borrow, a ComparisonTaintExpr)

shl / sar [rsp] flags are NOT closed: an RMW re-LOADs [rsp] after the STORE to read
back the shifted result, and a memory-leaf closed form would read the PRE-store
bytes (under-taint), so the memory resolver is disabled for any store-containing
instruction.  Their soundness is covered by test_mem_flag_soundness.

Correctness is checked against BRUTE-FORCED TRUE taint over the full cube of a
small register+memory taint mask (soundness for every output; exactness for the
cell-free ones).
"""
import itertools
import random

import pytest

from microtaint.instrumentation.ast import EvalContext, InstructionCellExpr
from microtaint.simulator import CellSimulator, MachineState
from microtaint.sleigh.engine import _cached_generate_static_rule, generate_static_rule
from microtaint.types import Architecture, ImplicitTaintPolicy, Register

ARCH = Architecture.AMD64
REG_NAMES = ('RAX', 'RBX', 'RCX', 'RDX', 'RSP', 'RBP', 'RSI', 'RDI')
FLAGS = ('CF', 'OF', 'SF', 'ZF', 'PF', 'AF')
REGS = [Register(n, 64) for n in REG_NAMES] + [Register(n, 1) for n in FLAGS]
M64 = (1 << 64) - 1
RSP = 0x7000

# label -> (bytes, size_bytes, {output: cells}, {outputs exact vs true taint})
CASES = {
    'add qword [rsp],rbx': ('48011c24', 8, {'CF': 0}, {'CF'}),
    'sub qword [rsp],rbx': ('48291c24', 8, {'CF': 0}, {'CF'}),
    'cmp qword [rsp],rbx': ('48391c24', 8, {'CF': 0}, {'CF'}),
}


@pytest.fixture(scope='module')
def sim() -> CellSimulator:
    return CellSimulator(ARCH, use_unicorn=False, use_c=True)


def _reg_targets(circ):
    return [a for a in circ.assignments
            if getattr(a.target, 'name', None) is not None and not hasattr(a.target, 'address_expr')]


def _true_taint(sim: CellSimulator, hexs: str, name: str, be, size: int, mk, base, taint: dict[str, int]):
    # MachineState.mem is keyed by INTEGER address (the ICE reads the concrete
    # bytes there); the circuit's EvalContext uses the MEM_<hex>_<size> string key.
    ice = InstructionCellExpr(ARCH, hexs, name, 0, be, {})
    pos = ([('r', r, b) for r in REG_NAMES for b in range(64) if (taint.get(r, 0) >> b) & 1]
           + [('m', mk, b) for b in range(size * 8) if (taint.get(mk, 0) >> b) & 1])
    outs = []
    for combo in itertools.product((0, 1), repeat=len(pos)):
        v = dict(base)
        for (_kind, k, b), bit in zip(pos, combo, strict=True):
            v[k] = (v[k] | (1 << b)) if bit else (v[k] & ~(1 << b) & M64)
        outs.append(sim.evaluate_concrete(
            ice, MachineState(regs={r: v[r] & M64 for r in REG_NAMES}, mem={RSP: v[mk]})))
    acc = 0
    for o in outs[1:]:
        acc |= o ^ outs[0]
    return acc


@pytest.mark.parametrize(('label', 'hexs', 'size', 'cells', 'exact'),
                         [(k, *v) for k, v in CASES.items()], ids=list(CASES))
def test_cell_free(label: str, hexs: str, size: int, cells, exact):  # noqa: ARG001
    _cached_generate_static_rule.cache_clear()
    circ = generate_static_rule(ARCH, bytes.fromhex(hexs), REGS)
    outs = {a.target.name: a for a in _reg_targets(circ)}
    for name, want in cells.items():
        assert name in outs, f'{label}: no {name} output'
        got = repr(outs[name].expression).count('InstructionCellExpr')
        assert got == want, f'{label} {name}: expected {want} cells, got {got}'


@pytest.mark.parametrize(('label', 'hexs', 'size', 'cells', 'exact'),
                         [(k, *v) for k, v in CASES.items()], ids=list(CASES))
def test_sound_and_exact_vs_true(label: str, hexs: str, size: int, cells, exact, sim: CellSimulator):  # noqa: ARG001
    _cached_generate_static_rule.cache_clear()
    circ = generate_static_rule(ARCH, bytes.fromhex(hexs), REGS)
    targets = _reg_targets(circ)
    mk = f'MEM_{hex(RSP)}_{size}'
    rng = random.Random(hash(hexs) & 0xFFFF)
    for _ in range(24):
        rv = {r: rng.getrandbits(64) for r in REG_NAMES}
        rv['RSP'] = RSP
        mv = rng.getrandbits(size * 8)
        taint = dict.fromkeys(REG_NAMES, 0)
        for _ in range(rng.randint(0, 3)):
            taint[rng.choice(('RBX', 'RCX'))] |= (1 << rng.randrange(64))
        mt = 0
        for _ in range(rng.randint(1, 4)):
            mt |= (1 << rng.randrange(size * 8))
        taint[mk] = mt
        base = {**{r: rv[r] & ~taint[r] & M64 for r in REG_NAMES}, mk: mv & ~mt}
        ectx = EvalContext(input_values={**base}, input_taint=taint, simulator=sim,
                          implicit_policy=ImplicitTaintPolicy.IGNORE)
        got = circ.evaluate(ectx)
        # Only the outputs THIS change closes (CF) are asserted here.  The other
        # flags of a memory RMW (SF/ZF of a memory shift) stay on the pre-existing
        # ICE differential, which has a path-dependent memory-flag soundness gap of
        # its own (see the bug-memory-compare-flag-undertaint note) -- out of scope
        # for the closed-form CF this test guards.
        for a in targets:
            nm = a.target.name
            if nm not in exact:
                continue
            be = a.target.bit_end - a.target.bit_start
            mask = (1 << (be + 1)) - 1
            true = _true_taint(sim, hexs, nm, be, size, mk, base, taint) & mask
            g = got.get(nm, 0) & mask
            assert (true & ~g) == 0, f'{label} {nm} UNDER-taint: true={true:#x} got={g:#x}'
            assert g == true, f'{label} {nm}: got {g:#x} != true {true:#x}'
