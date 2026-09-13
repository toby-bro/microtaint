"""Block mode must say which input-dependent memory operations an input reached.

A taint engine that only reports VIOLATIONS can find a bug solely by executing
it: you have to fuzz until an input actually drives the index out of bounds.
Reporting the SINK instead -- "this store's address depends on the input, here
are the bits, here is the address it used this time" -- is a far weaker
requirement, because reaching a `memcpy` is enormously easier than reaching it
with an overflowing length.  A solver can then answer whether a bad value is
possible, without the fuzzer ever having produced one.

The per-instruction path has done half of this for a while: `FindingKind.AIW`
fires on any store through a tainted pointer, violation or not.  Block mode,
which is the path a campaign actually runs, recorded nothing at all -- the
address taint was computed into `MT_BLK_A_ADDRT` on every access and then
dropped on the floor.

These tests pin the property on BOTH paths, because the whole point is that the
fast path can answer it.
"""
from __future__ import annotations

import io
import os
import platform
from collections.abc import Iterator

import pytest

from microtaint.emulator.reporter import Finding, FindingKind
from tests.test_detectors_on_later_visits import _SYSCALLS, _compile_freestanding

pytestmark = pytest.mark.skipif(
    platform.system() != 'Linux', reason='emulator tests require Linux',
)

#: A store whose ADDRESS comes from the input, and a load likewise.  Neither
#: goes out of bounds: the index is masked into the table, so nothing is
#: violated and a violation-only detector must stay silent.  That is exactly
#: the case this feature exists for.
#:
#: Built through the repo's own `_compile_freestanding` rather than a fresh
#: gcc line: the detectors are sensitive to the address form the compiler
#: picks, and re-deriving flags here produced a guest whose store the
#: per-instruction path did not report either, which would have blamed block
#: mode for a harness difference.
_GUEST = _SYSCALLS + r"""
static char table[4096];
static char sink;
void _start(void){
    unsigned long idx = 0, jdx = 0;
    sys_read(0, &idx, 8);
    sys_read(0, &jdx, 8);
    table[idx & 0xfff] = (char)idx;     /* STORE through an input-dependent address */
    sink = table[jdx & 0xfff];          /* LOAD  through an input-dependent address */
    sys_exit((int)sink);
}
"""


@pytest.fixture(scope='module')
def guest() -> Iterator[str]:
    path = _compile_freestanding(_GUEST)
    yield path
    os.unlink(path)


def _run(guest: str, *, block: bool) -> tuple[list[Finding], dict[str, int] | None]:
    """(findings, block counters)."""
    from qiling import Qiling
    from qiling.const import QL_VERBOSE

    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper

    prev = os.environ.get('MICROTAINT_BLOCK')
    os.environ['MICROTAINT_BLOCK'] = '1' if block else '0'
    saved, dn = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    try:
        rep = Reporter(json_mode=False, stream=io.StringIO())
        ql = Qiling([guest], '/', verbose=QL_VERBOSE.OFF)
        ql.os.stdin = io.BytesIO(bytes(range(1, 33)))
        w = MicrotaintWrapper(ql, reporter=rep)
        os.dup2(dn, 1)
        ok = True
        try:
            ql.run()
        except Exception:
            ok = False
        os.dup2(saved, 1)
        w.block_mode_finish(ok)
        # AIW is recorded in C, one entry per SITE, and handed over when the
        # report closes -- so a caller that only reads `findings` sees none of
        # it.  Closing the report is part of collecting the answer.
        stats = w.block_mode_stats()
        out = io.StringIO()
        rep.stream = out
        rep.finalize()
        return list(rep.findings), stats
    finally:
        os.dup2(saved, 1)
        os.close(dn)
        os.close(saved)
        if prev is None:
            os.environ.pop('MICROTAINT_BLOCK', None)
        else:
            os.environ['MICROTAINT_BLOCK'] = prev


def _sinks(findings: list[Finding]) -> list[Finding]:
    kinds = {FindingKind.AIW, getattr(FindingKind, 'AIR', FindingKind.AIW)}
    return [f for f in findings if f.kind in kinds]


def test_the_instruction_path_reports_the_store_sink(guest: str) -> None:
    """The premise and the oracle: the slow path already does this, so the
    guest really does contain an input-dependent memory access."""
    findings, _ = _run(guest, block=False)
    got = _sinks(findings)
    assert got, (
        'the per-instruction path reported no input-dependent memory access '
        'in a guest written to contain two, so this comparison cannot show '
        'anything')


def test_block_mode_reports_the_same_sinks(guest: str) -> None:
    """The property.  Block mode is the path a campaign runs, so a sink it
    cannot see is a sink the pipeline never hears about."""
    instr, _ = _run(guest, block=False)
    blk, stats = _run(guest, block=True)
    assert stats is not None, 'block mode did not install'
    assert stats['blocks'] > 0, 'the block hook never fired'
    assert _sinks(blk), (
        f'block mode reported no input-dependent memory access, while the '
        f'per-instruction path reported {len(_sinks(instr))}; the address '
        f'taint is computed into MT_BLK_A_ADDRT on every access and then '
        f'dropped')


def test_a_sink_carries_the_bits_that_control_the_address(guest: str) -> None:
    """A sink without its taint mask cannot be turned into a solver query,
    which is the entire purpose of recording it."""
    blk, _ = _run(guest, block=True)
    got = _sinks(blk)
    assert got, 'no sink to inspect'
    masked = [f for f in got
              if int(str(f.extra.get('pointer_taint', '0x0')), 16)]
    assert masked, (
        f'every sink reported a zero address-taint mask, so none of them says '
        f'which input bits control the address: '
        f'{[getattr(f, "extra", {}) for f in got]}')


def test_nothing_was_violated(guest: str) -> None:
    """The case this feature exists for: the guest is in bounds throughout, so
    a violation-only detector is silent and only a SINK report carries the
    information."""
    blk, _ = _run(guest, block=True)
    bof = [f for f in blk if f.kind == FindingKind.BOF]
    assert not bof, (
        f'the guest masks its index into the table and cannot overflow, so a '
        f'buffer-overflow finding here is a false positive: {bof}')


def test_both_a_store_and_a_load_sink_are_reported(guest: str) -> None:
    """A LOAD through an input-dependent address is a sink too.

    Pinned separately because the other assertions here are satisfied by the
    store alone: a mutation that skipped loads entirely left them all green.
    An out-of-bounds READ is an information leak, and the whole point of
    recording sinks is that the solver, not the emulator, decides whether the
    address can leave the object.
    """
    blk, _ = _run(guest, block=True)
    stores = [f for f in blk if f.kind == FindingKind.AIW]
    loads = [f for f in blk if f.kind == FindingKind.AIR]
    assert stores, f'no store sink: {[f.kind.value for f in blk]}'
    assert loads, (
        f'no LOAD sink, though the guest reads table[jdx & 0xfff] through an '
        f'input-dependent address: {[f.kind.value for f in blk]}')


def test_a_sink_says_where_the_access_landed(guest: str) -> None:
    """The concrete address is half the solver query: "it is here now, the
    input owns these bits, can it be driven out of the object?"."""
    blk, _ = _run(guest, block=True)
    loads = [f for f in blk if f.kind == FindingKind.AIR]
    assert loads, 'no load sink to inspect'
    assert any(int(str(f.extra.get('address', '0x0')), 16) for f in loads), (
        f'no load sink recorded the address it actually used: '
        f'{[f.extra for f in loads]}')


def test_the_counters_agree_with_the_findings(guest: str) -> None:
    """`sinks` counts every occurrence, the findings are one per SITE, and
    nothing may be left undrained."""
    blk, stats = _run(guest, block=True)
    assert stats is not None
    got = [f for f in blk if f.kind in (FindingKind.AIW, FindingKind.AIR)]
    assert stats['sinks'] >= len(got) > 0, (
        f"{stats['sinks']} occurrences counted but {len(got)} described")
    assert stats['sinks_pending'] == 0, (
        f"{stats['sinks_pending']} sinks were never handed to the reporter")
