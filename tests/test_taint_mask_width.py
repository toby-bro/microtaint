"""A taint mask wider than its register must not crash the process.

`gen_taint` in the multi-ISA campaign builds a mask as a SUM of random bit
positions, `sum(1 << rng.randrange(bits) for _ in range(1, 3))`. Draw the same
position twice and the sum carries: two draws of bit 63 make bit 64, one bit
above a 64-bit register. That is a harness bug, but what it exposed is an engine
one -- `circuit.evaluate` segfaults on it, killing the run.

That is why `campaign.py pass1` dies every ~10k cases. Nothing about it is
specific to long runs; the reproduction is three lines and instant. Measured
identical on v0.6.9, v0.6.10, v0.6.11, v0.6.13 and v0.6.15, so it is not a
regression -- the input has never been validated.

The crash is a missing NULL check, not an overflow: gdb puts it in
`PyNumber_And` inside `do_evaluate`, which runs a long chain of Python C-API
calls without testing any of them, so the first one to fail hands NULL to the
next. A Python-level API may raise on an input it cannot represent; it may not
segfault.

Each case runs in a subprocess, because the failure mode under test takes the
interpreter down with it and would otherwise kill the whole session rather than
report.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

_CHILD = r"""
import sys
from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, Register

REGS = [Register(n, 64) for n in ('RAX', 'RBX', 'RCX', 'RDX')]
MASK64 = (1 << 64) - 1
code, taint = sys.argv[1], int(sys.argv[2])
circuit = generate_static_rule(Architecture.AMD64, bytes.fromhex(code), REGS)
vals = {r.name: 0x1234 for r in REGS}
taints = {r.name: 0 for r in REGS}
taints['RAX'] = taint
ctx = EvalContext(input_values=vals, input_taint=taints, simulator=CellSimulator(Architecture.AMD64))
try:
    out = circuit.evaluate(ctx)
except (ValueError, OverflowError) as e:
    print('RAISED', type(e).__name__)   # refusing the input is an acceptable answer
    sys.exit(0)
over = {k: hex(v) for k, v in out.items() if v > MASK64}
print('WIDE' if over else 'OK', over or '')
sys.exit(2 if over else 0)
"""


def _run(code: str, taint: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, '-c', _CHILD, code, str(taint)],
        capture_output=True, text=True, check=False, timeout=300,
    )


@pytest.mark.parametrize('code', ['4809d8', '4801d8', '4831d8'])  # or / add / xor rax, rbx
def test_taint_mask_wider_than_register(code: str) -> None:
    """One bit above the register width is not a bit of that register."""
    proc = _run(code, 1 << 64)
    assert proc.returncode >= 0, f'crashed with signal {-proc.returncode}: {proc.stdout}{proc.stderr[-400:]}'
    assert proc.returncode == 0, f'{proc.stdout}{proc.stderr[-400:]}'


def test_taint_mask_from_summed_bit_positions() -> None:
    """The exact shape the campaign generates: two draws of the same position."""
    proc = _run('4809d8', (1 << 63) + (1 << 63))
    assert proc.returncode >= 0, f'crashed with signal {-proc.returncode}: {proc.stdout}{proc.stderr[-400:]}'
    assert proc.returncode == 0, f'{proc.stdout}{proc.stderr[-400:]}'
