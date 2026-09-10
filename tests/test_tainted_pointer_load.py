"""A load through a TAINTED address has to taint what it loaded.

`state[i] = SBOX[state[i]]` with a secret `i` is the whole reason a taint engine
is interesting: the table is public, the index is not, and the byte that comes
back is a function of the secret.  An engine that answers "clean" there loses
the secret at the first table lookup and every conclusion after it is worthless.

The two implementations disagreed about this.  The whole-instruction
differential tainted the loaded value (perturbing the address reads a different
byte, so the output moves); the compiled path lowered the load under the
'concrete' pointer policy, which resolves the access at the address the
instruction computes and makes NO claim about a tainted address, and answered
clean.  'concrete' was what the hot path ran, so the shipped engine
under-tainted every load through a tainted pointer.

Measured on `benchmark/taint_density/bench_dense.elf`: the whole 256-byte state
buffer went clean at the first S-box round under the compiled path and stayed
tainted under the differential.

The existing ground-truth bank could not catch it.  `test_taint_ir.py` checks
both policies against Unicorn, but it generates the tainted-pointer vectors only
for 'avalanche' -- the policy the engine did not run.
"""
from __future__ import annotations

import pytest

from microtaint.taint_api import Path, explain
from microtaint.taint_memory import TaintMemory
from microtaint.types import Architecture

#: The two implementations the API exposes, typed so a loop over them
#: carries the API's own Literal rather than widening to str.
PATHS: list[Path] = ['compiled', 'differential']
TABLE = 0x402000
FULL = (1 << 64) - 1

#: (label, arch, code, taint in, values in, destination)
SHAPES = [
    ('AMD64  movzbl (%rax,%rcx,1),%eax', Architecture.AMD64,
     bytes.fromhex('0fb60408'), {'RAX': 0xFF}, {'RAX': 0x19, 'RCX': TABLE}, 'RAX'),
    ('AMD64  mov (%rax),%rbx', Architecture.AMD64,
     bytes.fromhex('488b18'), {'RAX': 0xFF}, {'RAX': TABLE + 0x40}, 'RBX'),
    ('ARM64  ldr x1, [x0]', Architecture.ARM64,
     bytes.fromhex('010040f9'), {'X0': 0xFF}, {'X0': TABLE + 0x40}, 'X1'),
]


def _table() -> TaintMemory:
    """A public table: every byte written, not one of them tainted."""
    m = TaintMemory()
    for i in range(0x2000):
        m.write(TABLE + i, (i * 7 + 3) & 0xFF, 1)
    assert m.read_mask(TABLE, 8) == 0, 'the table must be clean, or this proves nothing'
    return m


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize(('label', 'arch', 'code', 'taint', 'values', 'dest'), SHAPES,
                         ids=[s[0] for s in SHAPES])
def test_a_load_through_a_tainted_address_taints_the_value(
        label: str, arch: Architecture, code: bytes, taint: dict[str, int], values: dict[str, int], dest: str,
        path: Path) -> None:
    out, used = explain(arch, code, taint, values, memory=_table(), path=path)
    assert used == path, f'{label}: asked for {path}, answered by {used}'
    assert out.get(dest, 0), (
        f'{label} [{path}]: loaded through an address tainted '
        f'{taint} and {dest} came back clean')


@pytest.mark.parametrize(('label', 'arch', 'code', 'taint', 'values', 'dest'), SHAPES,
                         ids=[s[0] for s in SHAPES])
def test_the_compiled_path_does_not_under_taint_the_differential(
        label: str, arch: Architecture, code: bytes, taint: dict[str, int], values: dict[str, int], dest: str) -> None:
    """Whatever the differential can move, the compiled path must move too.

    Stated as containment rather than equality: the differential takes an
    avalanche floor over the whole destination register where the compiled path
    keeps the loaded word's width, so the two are allowed to differ in that
    direction and only in that direction.
    """
    answers = {}
    for path in PATHS:
        out, used = explain(arch, code, taint, values, memory=_table(), path=path)
        assert used == path, f'{label}: asked for {path}, answered by {used}'
        answers[path] = out.get(dest, 0)
    missing = answers['differential'] & ~answers['compiled']
    # The compiled path is allowed to be TIGHTER only above the loaded word.
    assert answers['compiled'], f'{label}: compiled says clean, differential says {answers["differential"]:#x}'
    assert missing == 0 or (missing & 0xFF) == 0, (
        f'{label}: compiled path drops bits the differential keeps: '
        f'differential={answers["differential"]:#x} compiled={answers["compiled"]:#x}')


def test_a_clean_address_into_a_clean_table_stays_clean() -> None:
    """The control.  Without it, an engine that tainted every load would pass
    every other test in this file."""
    for path in PATHS:
        out, used = explain(Architecture.AMD64, bytes.fromhex('488b18'), {},
                            {'RAX': TABLE + 0x40}, memory=_table(), path=path)
        assert used == path
        assert out.get('RBX', 0) == 0, (
            f'[{path}] a clean pointer into a clean table invented taint: {out}')
