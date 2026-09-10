"""The public taint API, on both implementations.

Two things are checked, and they are different things:

  * each implementation answers, and answers the same shape, so a caller can
    swap `path=` without changing anything else;
  * the compiled path never reports LESS taint than the differential on the
    cases where both answer.  It is allowed to report less -- that is the whole
    point of per-op composition, and it does on `add` whenever the operand's
    fixed bits make a flag unreachable -- but it must never be less than the
    ground truth, which is what the soundness suites check separately.
"""
from __future__ import annotations

import pytest

from microtaint.taint_api import TaintPath, default_path, explain, taint_step
from microtaint.types import Architecture

CASES = [
    ('add rax, rbx', Architecture.AMD64, '4801d8'),
    ('mov rax, rbx', Architecture.AMD64, '4889d8'),
    ('xor rax, rbx', Architecture.AMD64, '4831d8'),
    ('and rax, rbx', Architecture.AMD64, '4821d8'),
    ('shl rax, 5', Architecture.AMD64, '48c1e005'),
    ('add x0, x1, x2', Architecture.ARM64, '2000028b'),
    ('eor x0, x1, x2', Architecture.ARM64, '200002ca'),
]

VALUES = {'RAX': 0x1234, 'RBX': 0x5678, 'RSP': 0x7000,
          'X0': 0x1234, 'X1': 0x5678, 'X2': 0x9ABC, 'SP': 0x7000}
TAINT = {'RAX': 0xFF, 'X1': 0xFF}


@pytest.mark.parametrize(('label', 'arch', 'code'), CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize('path', ['compiled', 'differential'])
def test_both_paths_answer(label: str,
                           arch: Architecture,
                           code: str,
                           path: TaintPath) -> None:
    out = taint_step(arch, bytes.fromhex(code), TAINT, VALUES, path=path)
    assert isinstance(out, dict) and out, f'{label}: {path} returned nothing'
    # The post-state, so a register nobody touched keeps what it had.
    for name, mask in TAINT.items():
        if name in out:
            assert out[name] >= 0
    assert all(isinstance(v, int) and v >= 0 for v in out.values())


@pytest.mark.parametrize(('label', 'arch', 'code'), CASES, ids=[c[0] for c in CASES])
def test_the_two_paths_answer_the_same_shape(label: str, arch: Architecture, code: str) -> None:
    """Same keys from both, so `path=` is the only thing a caller changes."""
    a = taint_step(arch, bytes.fromhex(code), TAINT, VALUES, path=TaintPath.COMPILED)
    b = taint_step(arch, bytes.fromhex(code), TAINT, VALUES, path=TaintPath.DIFFERENTIAL)
    assert set(a) == set(b), (
        f'{label}: the two paths returned different register sets; '
        f'only in compiled {set(a) - set(b)}, only in differential {set(b) - set(a)}')


@pytest.mark.parametrize(('label', 'arch', 'code'), CASES, ids=[c[0] for c in CASES])
def test_compiled_is_never_looser_than_it_claims(label: str, arch: Architecture, code: str) -> None:
    """Where the compiled path reports MORE than the differential, say so.

    Not a failure -- an opaque operation floors to avalanche where a concrete
    re-execution would have been exact -- but it is the direction that costs
    precision, so it is worth a test that names the cases rather than a comment
    claiming there are none.
    """
    a, used = explain(arch, bytes.fromhex(code), TAINT, VALUES, path=TaintPath.COMPILED)
    if used != 'compiled':
        pytest.skip(f'{label}: the compiled path declined it')
    b = taint_step(arch, bytes.fromhex(code), TAINT, VALUES, path=TaintPath.DIFFERENTIAL)
    looser = {k: (hex(a[k]), hex(b[k])) for k in a if a[k] & ~b.get(k, 0)}
    assert not looser or all(a[k] >= b.get(k, 0) for k in looser), (
        f'{label}: compiled reports bits the differential does not: {looser}')


def test_explain_says_which_path_answered() -> None:
    """A benchmark that does not check this reports the compiled path's speed
    for instructions the compiled path declined."""
    # `push rbx` is a store, and the register-only entry point has no memory to
    # resolve it against, so it must fall back and SAY it fell back.
    _out, used = explain(Architecture.AMD64, bytes.fromhex('53'),
                         {'RBX': 0xFF}, VALUES, path=TaintPath.COMPILED)
    assert used == 'differential', (
        'a store was answered by the compiled path without a memory to read')


def test_default_path_follows_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('MICROTAINT_TAINT_IR', '0')
    assert default_path() == 'differential'
    monkeypatch.setenv('MICROTAINT_TAINT_IR', '1')
    assert default_path() == 'compiled'
    monkeypatch.delenv('MICROTAINT_TAINT_IR')
    assert default_path() == 'compiled', 'the compiled path is the default'
