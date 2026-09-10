"""Closed-form taint for arithmetic-flag shapes must be exact and cell-free.

Two families gained closed forms (no InstructionCellExpr re-execution):

  * arithmetic RIGHT shift (``sar``/``asr``/``srawi``): the RESULT plus the CF, SF
    and ZF flags.  Every output bit of an arithmetic shift is exactly one input bit
    (bit i+k, or the replicated sign bit), so value and taint transform identically
    -- exact.  OF stays ICE-floored (undefined for count != 1).
  * unsigned CARRY / BORROW flag (x86 ``CF`` of add/sub/cmp): a monotone predicate
    routed to the Z3-proved ComparisonTaintExpr (``sub`` -> ``[a<u b]``; ``add`` ->
    ``[~b <u a]``).  Declines (keeps the differential) for dependent operands.

Both checks use BRUTE-FORCED TRUE taint (non-constancy over the full taint cube),
computed with a small taint mask so the cube is cheap:
  * SOUNDNESS -- the emitted taint contains the true taint for EVERY output: it
    never under-taints.
  * EXACTNESS -- for the monotone / pure-bit-routing outputs (CF here, and the
    result / CF / SF of an arithmetic shift) the emitted taint EQUALS the true
    taint.  ZF (equality) and OF (signed overflow) are non-monotone; PF is
    avalanche-modelled: only soundness is asserted for those.
"""
import itertools
import random
from collections.abc import Iterator

import pytest

from microtaint.instrumentation.ast import EvalContext, InstructionCellExpr, LogicCircuit, TaintAssignment
from microtaint.simulator import CellSimulator, MachineState
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, Register

ARCH = Architecture.AMD64
REG_NAMES = ('RAX', 'RBX', 'RCX', 'RDX')
FLAGS = ('CF', 'OF', 'SF', 'ZF', 'PF', 'AF')
REGS = [Register(n, 64) for n in REG_NAMES] + [Register(n, 1) for n in FLAGS]
M64 = (1 << 64) - 1

# label -> (bytes, {output: expected_cells}, {outputs that must EQUAL the differential})
CLOSED = {
    'sar eax,5':   ('c1f805',   {'RAX': 0, 'CF': 0, 'SF': 0, 'ZF': 0}, {'RAX', 'CF', 'SF'}),
    'sar rax,7':   ('48c1f807', {'RAX': 0, 'CF': 0, 'SF': 0, 'ZF': 0}, {'RAX', 'CF', 'SF'}),
    'sar al,2':    ('c0f802',   {'CF': 0, 'SF': 0, 'ZF': 0},           {'CF', 'SF'}),
    'add eax,ebx': ('01d8',     {'CF': 0},                             {'CF'}),
    'sub eax,ebx': ('29d8',     {'CF': 0},                             {'CF'}),
    'cmp eax,ebx': ('39d8',     {'CF': 0},                             {'CF'}),
    'add rax,rbx': ('4801d8',   {'CF': 0},                             {'CF'}),
    'add al,bl':   ('00d8',     {'CF': 0},                             {'CF'}),
    # immediate second operand (an untainted constant): CF still closes.
    'cmp eax,0x10': ('83f810',   {'CF': 0},                            {'CF'}),
    'cmp rax,0x10': ('4883f810', {'CF': 0},                            {'CF'}),
    'sub eax,0x10': ('83e810',   {'CF': 0},                            {'CF'}),
    'add eax,0x10': ('83c010',   {'CF': 0},                            {'CF'}),
}

# CF must DECLINE (stay on the differential) when the two operands are dependent.
CF_DECLINES = {
    'add eax,eax': '01c0',
    'sub eax,eax': '29c0',
}


@pytest.fixture(scope='module')
def sim() -> CellSimulator:
    return CellSimulator(ARCH, use_unicorn=False, use_c=True)


def _assignments(hexs: str,
                 ) -> tuple[LogicCircuit, dict[str, TaintAssignment]]:
    circ = generate_static_rule(ARCH, bytes.fromhex(hexs), REGS)
    out: dict[str, TaintAssignment] = {}
    for a in circ.assignments:
        nm = getattr(a.target, 'name', None)
        if nm is not None and not hasattr(a.target, 'address_expr'):
            out[nm] = a
    return circ, out


def _cells(expr: object) -> int:
    return repr(expr).count('InstructionCellExpr')


def _target_bits(a: TaintAssignment) -> int:
    """The width of the slice an assignment writes, as a top bit index.

    `TaintAssignment.target` is a register OR a memory operand, and only the
    register form carries a bit range; every assignment collected here came
    through the register filter in `_assignments`.
    """
    return int(a.target.bit_end) - int(a.target.bit_start)   # type: ignore[union-attr]


def _true_taint(sim: CellSimulator, hexs: str, name: str, bit_end: int,
                base_vals: dict[str, int], taint: dict[str, int]) -> int:
    """Non-constancy of the output over the FULL taint cube: OR of per-bit XORs of
    the output as the tainted input bits range over all 2^n assignments.  `base_vals`
    already has the tainted bits cleared.  Small taint keeps the cube cheap."""
    ice = InstructionCellExpr(ARCH, hexs, name, 0, bit_end, {})
    positions = [(r, b) for r in REG_NAMES for b in range(64) if (taint[r] >> b) & 1]
    outs = []
    for combo in itertools.product((0, 1), repeat=len(positions)):
        vals = dict(base_vals)
        for (r, b), bit in zip(positions, combo, strict=True):
            vals[r] = (vals[r] | (1 << b)) if bit else (vals[r] & ~(1 << b) & M64)
        outs.append(sim.evaluate_concrete(ice, MachineState(regs={r: vals[r] & M64 for r in REG_NAMES}, mem={})))
    acc = 0
    for o in outs[1:]:
        acc |= o ^ outs[0]
    return acc


def _small_taints(rng: random.Random) -> Iterator[dict[str, int]]:
    for _ in range(24):
        taint = dict.fromkeys(REG_NAMES, 0)
        for _ in range(rng.randint(1, 6)):
            taint[rng.choice(REG_NAMES)] |= (1 << rng.randrange(64))
        yield taint


@pytest.mark.parametrize(('label', 'hexs', 'expected'),
                         [(k, v[0], v[1]) for k, v in CLOSED.items()], ids=list(CLOSED))
def test_outputs_are_cell_free(label: str, hexs: str,
                               expected: dict[str, int]) -> None:
    _circ, outs = _assignments(hexs)
    for name, want in expected.items():
        assert name in outs, f'{label}: no {name} output'
        got = _cells(outs[name].expression)
        assert got == want, f'{label} {name}: expected {want} cells, got {got} ({repr(outs[name].expression)[:100]})'


@pytest.mark.parametrize(('label', 'hexs'), CF_DECLINES.items(), ids=list(CF_DECLINES))
def test_cf_declines_on_dependent_operands(label: str, hexs: str) -> None:
    _circ, outs = _assignments(hexs)
    assert 'CF' in outs
    assert 'ComparisonTaintExpr' not in repr(outs['CF'].expression), f'{label}: CF must decline to the differential'


@pytest.mark.parametrize(('label', 'hexs', 'exact'),
                         [(k, v[0], v[2]) for k, v in CLOSED.items()], ids=list(CLOSED))
def test_sound_and_exact(label: str, hexs: str, exact: dict[str, int],
                         sim: CellSimulator) -> None:
    circ, outs = _assignments(hexs)
    rng = random.Random(hash(hexs) & 0xFFFF)
    for taint in _small_taints(rng):
        values = {r: rng.getrandbits(64) for r in REG_NAMES}
        base = {r: values[r] & ~taint[r] & M64 for r in REG_NAMES}
        got = circ.evaluate(EvalContext(input_taint=taint, input_values=base, simulator=sim))
        for name, a in outs.items():
            bit_end = _target_bits(a)
            mask = (1 << (bit_end + 1)) - 1
            true = _true_taint(sim, hexs, name, bit_end, base, taint) & mask
            g = got.get(name, 0) & mask
            # Soundness: the emitted taint must contain the true taint (never under-taint).
            assert (true & ~g) == 0, f'{label} {name} UNDER-taint: true={true:#x} got={g:#x}'
            # Exactness for the monotone / bit-routing outputs.
            if name in exact:
                assert g == true, f'{label} {name}: got {g:#x} != true {true:#x} (taint RAX={taint["RAX"]:#x})'
