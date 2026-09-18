"""The zero flag of a BITWISE instruction must not under-taint.

``ZF = (result == 0)`` is non-monotone, so the two-corner differential can miss
it: both polarity corners can leave the result non-zero while an interior
assignment of the tainted bits drives it to zero.  ``add``/``sub``/``cmp`` are
protected from this by the equality-to-zero flag floor in
``generate_taint_assignments``; the bitwise family (``xor``/``and``/``or``) was
not, so it under-tainted ZF.

The witness comes from the RQ6 cross-ISA campaign, which reported it on AMD64
after 45.8M cases:

    xor ax, 0x7f       RAX = 0x69e9c457d669087f   taint(RAX) = 0xbaf66939dfd67bb5

Minimised, two tainted bits suffice.  With AX = 0x087f and bits 5 and 11
tainted the result is 0x0800 at one corner and 0x0020 at the other, while
clearing bit 11 alone gives 0x0000, so ZF is reachable and must stay tainted.

Ground truth here is non-constancy of the flag over the FULL taint cube,
brute-forced through the cell simulator, so the test cannot agree with a wrong
engine the way a hand-written expectation could.
"""
import itertools
import random
import zlib

import pytest

from microtaint.instrumentation.ast import EvalContext, InstructionCellExpr, LogicCircuit
from microtaint.simulator import CellSimulator, MachineState
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, Register

ARCH = Architecture.AMD64
REG_NAMES = ('RAX', 'RBX', 'RCX', 'RDX')
FLAGS = ('CF', 'OF', 'SF', 'ZF', 'PF', 'AF')
REGS = [Register(n, 64) for n in REG_NAMES] + [Register(n, 1) for n in FLAGS]
M64 = (1 << 64) - 1

# label -> instruction bytes.  Both operand widths the campaign reached, plus the
# register-operand and wider forms, so a fix that only special-cases `xor r16,imm`
# does not pass.
BITWISE = {
    'xor ax,0x7f':   '6683f07f',
    'xor al,0x7f':   '347f',
    'xor eax,0x7f':  '83f07f',
    'xor rax,0x7f':  '4883f07f',
    'xor ax,bx':     '6631d8',
    'and ax,0x7f':   '6683e07f',
    'and ax,bx':     '6621d8',
    'or ax,bx':      '6609d8',
    'test ax,0x7f':  '66a97f00',
}

# The exact campaign witness, and the two-bit minimisation of it.
WITNESSES = (
    ('campaign', '6683f07f', 0x69E9C457D669087F, 0xBAF66939DFD67BB5),
    ('minimal',  '6683f07f', 0x000000000000087F, 0x0000000000000820),
)


@pytest.fixture(scope='module')
def sim() -> CellSimulator:
    return CellSimulator(ARCH, use_unicorn=True)


def _zf_assignment(hexs: str) -> tuple[LogicCircuit, object]:
    circ = generate_static_rule(ARCH, bytes.fromhex(hexs), REGS)
    for a in circ.assignments:
        if getattr(a.target, 'name', None) == 'ZF' and not hasattr(a.target, 'address_expr'):
            return circ, a
    raise AssertionError(f'{hexs}: the lift produced no ZF assignment')


def _true_flag_taint(sim: CellSimulator, hexs: str, name: str,
                     base_vals: dict[str, int], taint: dict[str, int]) -> int:
    """1 iff the flag is non-constant over the full taint cube.

    `base_vals` already has the tainted bits cleared.  Keep the taint small so
    the 2^n cube stays cheap.
    """
    ice = InstructionCellExpr(ARCH, hexs, name, 0, 0, {})
    positions = [(r, b) for r in REG_NAMES for b in range(64) if (taint[r] >> b) & 1]
    outs = []
    for combo in itertools.product((0, 1), repeat=len(positions)):
        vals = dict(base_vals)
        for (r, b), bit in zip(positions, combo, strict=True):
            vals[r] = (vals[r] | (1 << b)) if bit else (vals[r] & ~(1 << b) & M64)
        outs.append(sim.evaluate_concrete(
            ice, MachineState(regs={r: vals[r] & M64 for r in REG_NAMES}, mem={})))
    return int(any(o != outs[0] for o in outs[1:]))


@pytest.mark.parametrize(('label', 'hexs', 'rax', 'taint_mask'), WITNESSES,
                         ids=[w[0] for w in WITNESSES])
def test_campaign_witness_taints_zf(label: str, hexs: str, rax: int,
                                    taint_mask: int, sim: CellSimulator) -> None:
    """The reported case itself, asserted against a brute-forced ground truth.

    The campaign mask is too wide to enumerate, so the cube is taken over the
    bits that can reach the 16-bit result; the upper bits cannot change AX.
    """
    cube_mask = taint_mask & 0xFFFF
    values = dict.fromkeys(REG_NAMES, 0)
    values['RAX'] = rax
    base = {r: values[r] & ~taint_mask & M64 for r in REG_NAMES}
    taint = dict.fromkeys(REG_NAMES, 0)
    taint['RAX'] = taint_mask

    cube_taint = dict.fromkeys(REG_NAMES, 0)
    cube_taint['RAX'] = cube_mask
    true = _true_flag_taint(sim, hexs, 'ZF', base, cube_taint)
    assert true == 1, f'{label}: the witness no longer reaches ZF, so it proves nothing'

    circ, _ = _zf_assignment(hexs)
    got = circ.evaluate(EvalContext(input_taint=taint, input_values=base, simulator=sim))
    assert (got.get('ZF', 0) or 0) & 1, (
        f'{label}: ZF UNDER-tainted for {hexs} at RAX={rax:#x} taint={taint_mask:#x}'
    )


@pytest.mark.parametrize(('label', 'hexs'), BITWISE.items(), ids=list(BITWISE))
def test_bitwise_zf_is_sound(label: str, hexs: str, sim: CellSimulator) -> None:
    """Randomised search for a corner-blind ZF across the bitwise family.

    Both the values and the taint are confined to the low byte.  A result can
    only be zero if its untainted bits are all zero, and a mask scattered over
    64 bits of a 16-bit form reaches that essentially never, which would leave
    the search vacuous rather than passing.  The run stops at 40 witnessed
    samples so a cheap form does not dominate the suite's runtime.
    """
    circ, _ = _zf_assignment(hexs)
    # zlib.crc32, not hash(): str hashing is salted per process, so a
    # hash()-derived seed draws a different sample in every xdist worker and
    # the test passes or fails by luck of the run.
    rng = random.Random(zlib.crc32(hexs.encode()))
    witnessed = 0
    for _ in range(3000):
        taint = dict.fromkeys(REG_NAMES, 0)
        for r in ('RAX', 'RBX'):
            for _ in range(rng.randint(1, 3)):
                taint[r] |= 1 << rng.randrange(8)
        values = {r: rng.getrandbits(8) for r in REG_NAMES}
        base = {r: values[r] & ~taint[r] & M64 for r in REG_NAMES}

        true = _true_flag_taint(sim, hexs, 'ZF', base, taint)
        if not true:
            continue
        witnessed += 1
        got = circ.evaluate(EvalContext(input_taint=taint, input_values=base, simulator=sim))
        assert (got.get('ZF', 0) or 0) & 1, (
            f'{label}: ZF UNDER-tainted at '
            f'RAX={base["RAX"]:#x} RBX={base["RBX"]:#x} '
            f'taint RAX={taint["RAX"]:#x} RBX={taint["RBX"]:#x}'
        )
        if witnessed >= 40:
            break
    assert witnessed, f'{label}: no sample reached ZF, so this parametrisation proves nothing'
