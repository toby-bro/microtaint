"""Give pip the same answer `override-dependencies` gives uv.

Qiling declares `python-fx` and never imports it: there is no `import fx`
anywhere in the package, dynamic or otherwise (fx is a standalone JSON-viewer
CLI). It drags in seventeen packages, Pillow among them, and Pillow is the one
that breaks builds: python-fx pins `pillow<11`, the newest release satisfying
that is 10.4.0, and 10.4.0 ships no wheel past cp313. On 3.14 pip therefore
builds Pillow from source and wants zlib and libjpeg headers, which a
manylinux image and a Windows runner do not have.

`pyproject.toml` already answers this for uv:

    override-dependencies = ["python-fx ; sys_platform == 'never'"]

but that is uv-only, and cibuildwheel installs the wheel under test with pip,
which has no override mechanism at all. `pip install --no-deps python-fx` does
not help either: the resolver re-reads the installed distribution's metadata
and pulls Pillow anyway, which is measurable -- it still plans 35 packages.

So do to the metadata what the override does to the graph. Install python-fx
without its dependencies, then delete the `Requires-Dist` lines from its
installed METADATA. The node stays, so Qiling's requirement is satisfied; the
edges go, so nothing downstream is dragged in. Measured on a clean
environment: 36 packages before, 20 after, with Pillow and asciimatics gone.

This removes an imaging library, a terminal-animation library and their system
headers from a taint engine's dependency tree. It is a workaround for someone
else's metadata and should be deleted the moment python-fx relaxes the pin or
Qiling drops a dependency it does not use.
"""
from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
import sysconfig

#: The distribution Qiling asks for and never imports.
_DIST = 'python-fx'
_DIST_GLOB = 'python_fx-*.dist-info'


def _install_without_dependencies() -> None:
    # S603 wants proof the arguments are not untrusted input.  They are this
    # interpreter and four literals, with no shell and no caller-supplied
    # value anywhere in the list.
    subprocess.run(  # noqa: S603
        [sys.executable, '-m', 'pip', 'install', '--no-deps', _DIST],
        check=True)


def _strip_requires(purelib: str) -> int:
    """Delete every Requires-Dist line from python-fx's METADATA.

    Returns the number of edges removed, summed over the dist-info directories
    found, so the caller can tell "nothing to do" from "nothing found".
    """
    removed = 0
    found = False
    for meta in glob.glob(os.path.join(purelib, _DIST_GLOB, 'METADATA')):
        found = True
        text = open(meta, encoding='utf-8').read()
        edges = len(re.findall(r'(?mi)^Requires-Dist:', text))
        if edges:
            open(meta, 'w', encoding='utf-8').write(
                re.sub(r'(?mi)^Requires-Dist:.*\n', '', text))
        removed += edges
    if not found:
        raise SystemExit(
            f'{_DIST} is not installed in {purelib}: nothing was pruned, and '
            f'pip would go on to pull Pillow. Refusing to continue quietly.')
    return removed


def main() -> int:
    _install_without_dependencies()
    purelib = sysconfig.get_paths()['purelib']
    removed = _strip_requires(purelib)
    # Prove it rather than assume it: a second read must find no edges left.
    left = sum(
        len(re.findall(r'(?mi)^Requires-Dist:', open(m, encoding='utf-8').read()))
        for m in glob.glob(os.path.join(purelib, _DIST_GLOB, 'METADATA')))
    if left:
        raise SystemExit(f'{left} Requires-Dist lines survived the prune')
    print(f'{_DIST}: pruned {removed} dependency edges, Pillow among them'
          if removed else
          f'{_DIST}: already pruned, no dependency edges left')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
