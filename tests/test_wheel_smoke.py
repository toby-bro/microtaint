"""Does the built wheel actually work, on the machine that will install it?

A wheel that imports is not a wheel that works, and the publish pipeline ran
no tests at all: it built three wheels, created a release and uploaded to PyPI
without once asking whether any of them produced a taint answer.  Everything
here is chosen so it can run against an INSTALLED wheel on a bare runner, with
no repository checkout beyond this file and no fixtures.

Three things are checked, in increasing order of what they need:

  * the native extensions LOADED.  A wheel whose .so failed to build imports
    fine and answers fine, just slowly and from the .py, and nothing else in
    this file would notice.
  * both taint implementations answer, and agree, over a sequence with a
    spill and a reload.  This needs no operating system, no rootfs and no
    guest compiler, so it is the part that runs on all three platforms.
  * a real binary, emulated end to end, on both implementations.  This runs
    on all three platforms too: Qiling EMULATES its guest, so a Linux ELF is
    emulated just as well from Windows or macOS, and the guest is shipped
    pre-built because those runners have no compiler that emits one.

`MICROTAINT_TAINT_IR=0` selects the differential and anything else selects the
compiled path, but `path=` overrides it per call, so one process covers both
without being re-run.
"""
from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import tempfile

import pytest

from microtaint.taint_api import TaintPath, TaintSequence, taint_step
from microtaint.types import Architecture

pytestmark = pytest.mark.smoke

#: x86-64.  Deliberately not a straight line of register arithmetic: the store
#: and the reload are what make the sequence exercise TaintMemory, and a spill
#: that came back clean was a real bug once (a block region hid the store from
#: the later load), so it is the shape worth smoke-testing.
_SEQUENCE = [
    ('mov rbx, rax',        '4889c3'),
    ('add rbx, rbx',        '4801db'),
    ('mov [rsp-8], rbx',    '48895c24f8'),   # spill the tainted value
    ('xor rbx, rbx',        '4831db'),       # and lose it from the register
    ('mov rcx, [rsp-8]',    '488b4c24f8'),   # reload it into another one
]

_VALUES = {'RAX': 0x1234_5678, 'RBX': 0, 'RCX': 0, 'RSP': 0x7000}
_TAINT = {'RAX': 0xFF}


def test_the_native_extensions_are_the_ones_that_loaded() -> None:
    """Otherwise the wheel is a pure-Python one wearing a platform tag.

    Every other check in this file passes on the interpreted fallback, so
    without this one a wheel whose extensions silently failed to build would
    ship and simply be slow.
    """
    missing = []
    for name in ('microtaint.instrumentation.cell_c.cell_c',
                 'microtaint.instrumentation.cell_c.circuit_c'):
        mod = importlib.import_module(name)
        if not getattr(mod, '__file__', '').endswith(('.so', '.pyd')):
            missing.append(f'{name} -> {getattr(mod, "__file__", "?")}')
    assert not missing, 'compiled extensions did not load: ' + '; '.join(missing)


@pytest.mark.parametrize('path', [TaintPath.COMPILED, TaintPath.DIFFERENTIAL])
def test_one_instruction_answers_on_this_path(path: TaintPath) -> None:
    out = taint_step(Architecture.AMD64, bytes.fromhex('4801d8'),
                     _TAINT, _VALUES, path=path)
    assert out, f'{path} returned nothing'
    assert out.get('RAX', 0) & 0xFF, f'{path} lost the taint it was given'


@pytest.mark.parametrize('path', [TaintPath.COMPILED, TaintPath.DIFFERENTIAL])
def test_a_spill_and_reload_keeps_its_taint(path: TaintPath) -> None:
    """The taint has to survive going to memory and coming back.

    Asserted on RCX rather than on "some register is tainted": the value is
    deliberately cleared out of RBX in between, so RCX can only be tainted by
    way of the store and the load.
    """
    seq = TaintSequence(Architecture.AMD64, values=dict(_VALUES),
                        taint=dict(_TAINT), path=path)
    for _label, code in _SEQUENCE:
        seq.step(bytes.fromhex(code))
    assert seq.taint.get('RCX', 0) & 0xFF, (
        f'{path}: the reloaded value came back clean, taint={seq.taint}')
    assert not seq.taint.get('RBX', 0), (
        f'{path}: RBX was zeroed by xor and must be clean, taint={seq.taint}')


def test_both_implementations_agree_on_the_sequence() -> None:
    """They may differ in precision, but not about what is tainted at all.

    The compiled path is allowed to report LESS than the differential, which
    is the point of per-op composition. What would be a defect is one of them
    answering and the other not, which is what a broken wheel looks like.
    """
    seen = {}
    for path in (TaintPath.COMPILED, TaintPath.DIFFERENTIAL):
        seq = TaintSequence(Architecture.AMD64, values=dict(_VALUES),
                            taint=dict(_TAINT), path=path)
        for _label, code in _SEQUENCE:
            seq.step(bytes.fromhex(code))
        seen[path] = seq.taint
    compiled, differential = seen[TaintPath.COMPILED], seen[TaintPath.DIFFERENTIAL]
    for reg in ('RCX',):
        assert bool(compiled.get(reg, 0)) == bool(differential.get(reg, 0)), (
            f'the two paths disagree on whether {reg} is tainted at all: '
            f'compiled={compiled}, differential={differential}')


# ---------------------------------------------------------------------------
# A real binary, emulated, on EVERY platform.
#
# Qiling emulates its guest rather than running it, so the guest's operating
# system has nothing to do with the host's: a Linux ELF is emulated just as
# well from Windows or macOS.  Two things had to be true for that to be usable
# here, and both were checked rather than assumed:
#
#   * the guest is `-static -nostdlib`, so it needs no loader and no shared
#     library, and it loads against a rootfs that is an EMPTY directory.  That
#     is what makes it portable; a dynamically linked guest would need a Linux
#     root to resolve against and could only run on Linux.
#   * it is COMMITTED pre-built, as tests/guests/smoke_amd64.guest, because
#     the Windows and macOS runners have no compiler that emits a Linux ELF.
#     Shipping the binary is what lets all three platforms emulate the exact
#     same guest, which is also what makes their results comparable.
#
# tests/guests/smoke_amd64.c is that binary's source, kept beside it so the
# blob is readable.  Where a toolchain CAN emit a Linux ELF, the tests below
# also build the guest here and emulate that: it keeps the blob from drifting
# from its source, and a second compiler lowers the same program differently,
# so it covers instructions the shipped blob does not contain.
# ---------------------------------------------------------------------------

_GUESTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'guests')
_GUEST_BIN = os.path.join(_GUESTS, 'smoke_amd64.guest')
_GUEST_SRC = os.path.join(_GUESTS, 'smoke_amd64.c')

#: What the guest is fed. Fixed, so every platform emulates the same run.
_STDIN = bytes((i * 7 + 13) & 0xFF for i in range(16))


def _emulate(guest: str, taint_ir: str) -> dict[str, int]:
    """Run `guest` to completion under the engine; returns the register taint."""
    import io

    from qiling import Qiling
    from qiling.const import QL_VERBOSE

    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper

    prev = os.environ.get('MICROTAINT_TAINT_IR')
    os.environ['MICROTAINT_TAINT_IR'] = taint_ir
    # Qiling narrates the load on stdout, which is noise in a release log.
    saved, dn = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    try:
        with tempfile.TemporaryDirectory() as rootfs:
            ql = Qiling([guest], rootfs, verbose=QL_VERBOSE.OFF)
            ql.os.stdin = io.BytesIO(_STDIN)
            wrapper = MicrotaintWrapper(
                ql, reporter=Reporter(json_mode=False, stream=io.StringIO()))
            os.dup2(dn, 1)
            ql.run()
            os.dup2(saved, 1)
            return {k: v for k, v in wrapper.register_taint.items() if v}
    finally:
        os.dup2(saved, 1)
        os.close(saved)
        os.close(dn)
        if prev is None:
            os.environ.pop('MICROTAINT_TAINT_IR', None)
        else:
            os.environ['MICROTAINT_TAINT_IR'] = prev


@pytest.mark.parametrize('taint_ir', ['0', '1'], ids=['differential', 'compiled'])
def test_a_real_binary_runs_under_the_emulator(taint_ir: str) -> None:
    """End to end on both implementations: load, lift, emulate, propagate.

    `MICROTAINT_TAINT_IR` is the switch the emulator HOOK reads, so this is
    the only check here that covers the hook rather than the API: '0' selects
    the differential and '1' the compiled path.
    """
    assert os.path.exists(_GUEST_BIN), f'the shipped guest is missing: {_GUEST_BIN}'

    tainted = _emulate(_GUEST_BIN, taint_ir)

    # The guest reads 16 bytes, folds them into one value and branches on it.
    # So a run that propagated nothing is a broken engine, not a quiet guest,
    # and asserting merely "it did not crash" would pass on one.
    assert tainted, (
        f'taint_ir={taint_ir}: the guest branched on bytes it read and no '
        f'register came back tainted, so nothing propagated')
    # ZF is the branch's own flag: reaching it means taint crossed the fold
    # and the comparison, not merely that the read was marked.
    assert 'ZF' in tainted, (
        f'taint_ir={taint_ir}: taint never reached the branch flag; '
        f'tainted={sorted(tainted)}')


def test_both_implementations_carry_taint_through_the_binary() -> None:
    """Neither may be the only one that works.

    They are allowed to differ in precision, so this compares what each
    reached rather than demanding identical masks.
    """
    differential = _emulate(_GUEST_BIN, '0')
    compiled = _emulate(_GUEST_BIN, '1')
    both = f'differential={sorted(differential)}, compiled={sorted(compiled)}'
    assert 'ZF' in differential, f'the differential path did not reach it: {both}'
    assert 'ZF' in compiled, f'the compiled path did not reach it: {both}'


#: How this worker might build the guest, best first.  The source includes no
#: header at all -- inline syscalls and builtins only -- so a build needs a
#: compiler and an ELF-capable linker and NO sysroot, which is what makes this
#: worth attempting away from Linux at all.  On a Linux worker the system
#: linker already emits ELF; elsewhere it takes lld, which the Windows LLVM
#: install has and Apple's toolchain does not.
_TOOLCHAINS: list[tuple[str, list[str]]] = [
    ('gcc', ['-O0']),
    ('cc', ['-O0']),
    ('clang', ['-O0']),
    ('clang', ['-O0', '--target=x86_64-unknown-linux-gnu', '-fuse-ld=lld']),
]


def _build_guest_here(tmp: str) -> tuple[str, str] | None:
    """Build the guest with whatever this worker has; (path, how) or None."""
    for name, extra in _TOOLCHAINS:
        cc = shutil.which(name)
        if not cc:
            continue
        out = os.path.join(tmp, 'guest_' + str(abs(hash((name, tuple(extra))))))
        proc = subprocess.run(
            [cc, *extra, '-static', '-nostdlib', '-fno-stack-protector',
             '-o', out, _GUEST_SRC],
            capture_output=True, text=True, check=False)
        if proc.returncode == 0 and os.path.exists(out):
            return out, ' '.join([name, *extra])
    return None


@pytest.mark.parametrize('taint_ir', ['0', '1'], ids=['differential', 'compiled'])
def test_a_guest_this_worker_built_runs_too(taint_ir: str) -> None:
    """The committed blob is one compiler's idea of this program.

    Another toolchain lowers the same source differently, and the difference
    is not cosmetic: gcc and clang builds of this guest leave taint in
    different registers (RDX against RCX), because they allocate the fold
    differently.  So a guest built HERE, by whatever the worker has, emulates
    instructions the shipped blob does not contain, and the release gate
    covers this platform's toolchain as well as its wheel.

    Skipped where nothing can emit a Linux ELF, which is the case on a stock
    macOS runner.  That costs only this test: the committed blob is what the
    gate rests on, and it runs everywhere.
    """
    with tempfile.TemporaryDirectory() as tmp:
        built = _build_guest_here(tmp)
        if built is None:
            pytest.skip('no toolchain here emits a Linux ELF')
        guest, how = built
        # Which toolchain a release runner actually had is the thing worth
        # reading back out of its log.  Not via capsys: asking for that
        # fixture switches pytest to sys-level capture, and Qiling's logger
        # needs a stream with a real fileno.
        print(f'guest built by: {how}')
        tainted = _emulate(guest, taint_ir)
        assert 'ZF' in tainted, (
            f'taint_ir={taint_ir}: a guest built here by {how} did not carry '
            f'taint to the branch; tainted={sorted(tainted)}')


def test_the_shipped_guest_still_matches_its_source() -> None:
    """A committed binary nobody can rebuild is a binary nobody can review.

    Not a byte comparison: the compiler version decides those bytes, and
    pinning them would fail on every toolchain but one.  What must hold is
    that the source beside the blob still describes a guest the engine sees
    taint in, so the two cannot drift apart unnoticed.
    """
    with tempfile.TemporaryDirectory() as tmp:
        built = _build_guest_here(tmp)
        if built is None:
            pytest.skip('no toolchain here can rebuild the guest')
        guest, how = built
        assert 'ZF' in _emulate(guest, '1'), (
            f'the committed guest and its source no longer agree: a build by '
            f'{how} from the .c beside it does not carry taint to the branch')
