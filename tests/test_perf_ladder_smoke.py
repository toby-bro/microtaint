"""`scripts/perf_ladder.py` still runs, and its guards still refuse.

The ladder is the script anyone reaches for before quoting an end-to-end
number, so the way it fails is to rot quietly: an API it calls moves, nobody
runs it for a month, and the next person to want a figure writes a fresh ad-hoc
script instead -- which is how this project accumulated a drawer of one-off
benchmarks that each got something wrong.

These are smoke tests, not measurements.  They assert that the harness
executes, that it reports the work alongside the cost, and above all that its
guards still FIRE: a harness whose refusals have quietly stopped working is
worse than no harness, because it will hand back a number for a run that
measured nothing.
"""
from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != 'Linux', reason='the ladder emulates Linux guests',
)

_SCRIPT = Path(__file__).resolve().parent.parent / 'scripts' / 'perf_ladder.py'

#: A perf CSV exactly as the harness receives it, including the two events
#: whose names are prefixes of one another.
_CSV = (
    '1234,,instructions:u,999,100.00,,\n'
    '5678,,instructions,999,100.00,,\n'
    '2.50,msec,task-clock,999,100.00,,\n'
    '42,,page-faults,999,100.00,,\n'
)


def test_counter_names_do_not_match_each_others_prefixes() -> None:
    """`instructions` must not match the prefix of `instructions:u`.

    It did.  The kernel column is computed as total minus user, so a parser
    that returned the user count for both made it structurally zero: a column
    that could never be non-zero, printed as though it were a measurement.
    """
    sys.path.insert(0, str(_SCRIPT.parent))
    # scripts/ is not a package and the path is added at run time, so the
    # checker cannot resolve this the way the interpreter does.
    import perf_ladder  # type: ignore[import-not-found]

    assert perf_ladder._count(_CSV, 'instructions:u') == 1234
    assert perf_ladder._count(_CSV, 'instructions') == 5678, (
        'asking for `instructions` returned the `instructions:u` value, so '
        'kernel = total - user can only ever be 0')
    assert perf_ladder._count(_CSV, 'task-clock') == 2.50
    assert perf_ladder._count(_CSV, 'page-faults') == 42
    assert perf_ladder._count(_CSV, 'cycles') is None


def _run(*args: str, timeout: int = 900) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(_SCRIPT), *args],
                          capture_output=True, text=True, check=False,
                          timeout=timeout)


def test_the_script_is_there_and_self_describes() -> None:
    assert _SCRIPT.exists(), f'{_SCRIPT} is gone'
    done = _run('--help', timeout=120)
    assert done.returncode == 0, done.stderr[-500:]
    assert 'instructions:u' in done.stdout, (
        'the help no longer explains the one metric choice that has been got '
        'wrong most often')


def test_a_rung_runs_and_reports_work_beside_cost() -> None:
    """The cheapest rung end to end.  `native` needs no emulator, so this stays
    fast and still exercises the build, the symbol lookup, the guest-work
    count, the marginal method and the table."""
    done = _run('--rungs', 'native', '--repeats', '1')
    assert done.returncode == 0, done.stderr[-900:]
    out = done.stdout
    assert 'guest instructions' in out, 'the work was not reported'
    assert 'one-time startup' in out, (
        'the startup share is what stops a checkpointed iteration being '
        'divided by a full launch; it must stay in the output')
    assert 'marginal cost of ONE more input' in out


def test_the_spread_guard_refuses() -> None:
    """Repeats that disagree must be an error, not a median.

    Forced with --max-spread 0 so the test does not depend on the machine
    actually being noisy.
    """
    done = _run('--rungs', 'native', '--repeats', '2', '--max-spread', '0')
    assert done.returncode != 0, (
        'the harness reported a figure even though the repeats were declared '
        'to disagree; its central refusal has stopped working')
    assert 'disagree' in (done.stdout + done.stderr)


def test_a_guest_it_cannot_measure_is_an_error() -> None:
    """Not a zero, not an empty table: an error."""
    done = _run('--rungs', 'native', '--repeats', '1',
                '--guest', '/bin/false', '--main-symbol', 'main', timeout=300)
    assert done.returncode != 0, (
        'the harness accepted a guest whose entry point it could not find')


def test_x_native_is_withheld_where_the_work_differs() -> None:
    """Rule 7, the error that produced "6.6x native" for something ~1,760x.

    A checkpointed iteration resumes at main; a native launch runs the whole
    program, 97% of which is one-time startup.  The table must not offer a
    ratio between them.
    """
    src = _SCRIPT.read_text()
    assert "'qiling-checkpoint': (100, False)" in src, (
        'the checkpoint rungs must stay flagged as NOT full-launch, or the '
        'table will print an x-native ratio between a partial run and a whole '
        'one')
    assert "'microtaint-checkpoint': (100, False)" in src
    assert 'r.full_launch' in src, (
        'the full_launch flag is no longer consulted when forming x native')
