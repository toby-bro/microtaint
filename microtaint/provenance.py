"""Which engine produced a result.

An artifact result that does not say which engine measured it cannot be checked
later, and the question does get asked: the published RQ2/3/4 macros could not be
matched to any report in the tree, because the report's metadata carried a
timestamp, a seed and a host, and nothing at all about the engine.  A re-run
fixes the numbers; it does not stop the same question being unanswerable next
time.

So the engine reports its own identity, and every runner stamps it into what it
writes.  It is deliberately cheap and deliberately unable to fail: a wheel
install has no git repository, a tarball has no `.git`, and neither is a reason
for an experiment to stop.  What it cannot answer, it says `None` to.

`dirty` matters as much as the commit.  A measurement taken with uncommitted
edits is not reproducible from the commit alone, and that is worth knowing when
a table looks wrong six months later.
"""
from __future__ import annotations

import subprocess
from functools import lru_cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from pathlib import Path

__all__ = ['engine_provenance']

#: Every field is a string or absent; `dirty` is the one boolean.
ProvenanceValue = str | bool | None

#: The repository root, if this is a checkout rather than an installed wheel.
_REPO = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str | None:
    """A git query that answers None rather than raising.

    Every failure mode is ordinary: no git binary, not a checkout, a repository
    that git refuses to read because it is owned by another user.
    """
    try:
        # `git` by name, on purpose: the tool must come from the PATH of
        # whoever runs the experiment, and the arguments are literals.
        out = subprocess.run(  # noqa: S603
            ['git', '-C', str(_REPO), *args],  # noqa: S607
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    # An EMPTY answer is a real answer -- `git status --porcelain` says "clean"
    # by printing nothing -- so only a failed command is None.  Conflating the
    # two reported `dirty: null` on a clean tree, which reads as "unknown".
    return out.stdout.strip()


@lru_cache(maxsize=1)
def engine_provenance() -> dict[str, ProvenanceValue]:
    """Identify the engine that is about to measure something.

    Cached: a campaign may write thousands of records and the answer cannot
    change inside one process.

    Returns keys that are always present, whose values may be None:

        version      the installed package version
        commit       full hash of HEAD
        commit_short abbreviated hash, for printing
        branch       the checked-out branch, or None when detached
        dirty        True when tracked files differ from HEAD
        describe     `git describe --tags --always --dirty`, when tags exist
    """
    try:
        version: str | None = _dist_version('microtaint')
    except PackageNotFoundError:               # pragma: no cover - packaging
        version = None

    commit = _git('rev-parse', 'HEAD')
    status = _git('status', '--porcelain', '--untracked-files=no')
    branch = _git('rev-parse', '--abbrev-ref', 'HEAD')

    describe = _git('describe', '--tags', '--always', '--dirty')
    return {
        'version': version,
        'commit': commit or None,
        'commit_short': commit[:12] if commit else None,
        # Detached HEAD reports the literal string 'HEAD', which names nothing.
        'branch': branch if branch and branch != 'HEAD' else None,
        # None only when git could not answer; '' is a clean tree.
        'dirty': None if status is None else status != '',
        'describe': describe or None,
    }
