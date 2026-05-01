"""
Hatchling build hook for microtaint's hand-written C extensions.

Compiles:
  microtaint/instrumentation/cell_c/cell_c.c    → cell_c.<EXT>
  microtaint/instrumentation/cell_c/circuit_c.c → circuit_c.<EXT>

These are pure CPython C extensions (no Cython preprocessing). They
provide the bit-precise taint-propagation kernel and the AST-bytecode
evaluator used by the Cython instrumentation layer at run-time.

The Cython modules (.pyx) are handled separately by the hatch-cython hook
configured in pyproject.toml. This hook only handles the .c sources.

Build is enabled by default; disable with HATCH_BUILD_HOOKS_ENABLE=false
(applies to all Hatch hooks) or by removing this section from
pyproject.toml.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import sysconfig
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

# --------------------------------------------------------------------------
# C extension manifest
#
# Each entry: (source path relative to project root, module name as it will
# appear at import time). The module name determines the .so filename:
#   cell_c    →  cell_c.cpython-3xx-<arch>.so
#   circuit_c →  circuit_c.cpython-3xx-<arch>.so
#
# Both modules live in microtaint/instrumentation/cell_c/ alongside the
# Cython modules so the runtime can find them via a single sys.path entry.
# --------------------------------------------------------------------------
C_EXTENSIONS: list[tuple[str, str]] = [
    ('microtaint/instrumentation/cell_c/cell_c.c', 'cell_c'),
    ('microtaint/instrumentation/cell_c/circuit_c.c', 'circuit_c'),
]


def _compiler_command() -> list[str]:
    """
    Return the C compiler command and flags to use.

    Honours the standard distutils/sysconfig env vars:
      CC      - compiler executable (default: from sysconfig)
      CFLAGS  - additional compile flags (appended)
    """
    cc = os.environ.get('CC') or sysconfig.get_config_var('CC') or 'cc'
    # CC from sysconfig may be 'gcc -pthread' etc; split on whitespace.
    cmd = shlex.split(cc)
    base_flags = [
        '-O3',
        '-march=native',
        '-ffast-math',
        '-shared',
        '-fPIC',
        '-Wall',
    ]
    if sys.platform == 'darwin':
        base_flags.extend(['-undefined', 'dynamic_lookup'])

    extra = shlex.split(os.environ.get('CFLAGS', ''))
    return cmd + base_flags + extra


class MicrotaintCExtBuildHook(BuildHookInterface):
    """Hatchling build hook that compiles the cell_c and circuit_c modules."""

    PLUGIN_NAME = 'microtaint-c-ext'

    def initialize(self, version: str, build_data: dict) -> None:
        """
        Called before the wheel is built. Compiles each C extension to a
        .so next to its source, and registers the .so as a force-included
        file so it ends up in the wheel.
        """
        if self.target_name != 'wheel':
            # sdist doesn't need the compiled artifacts; the .c source is
            # included via tool.hatch.build.targets.sdist.include below.
            return

        ext_suffix = sysconfig.get_config_var('EXT_SUFFIX') or '.so'
        py_include = sysconfig.get_paths()['include']
        cc_cmd = _compiler_command()

        force_include = build_data.setdefault('force_include', {})
        artifacts = build_data.setdefault('artifacts', [])

        for source_rel, module_name in C_EXTENSIONS:
            source = Path(self.root) / source_rel
            if not source.is_file():
                # Tolerate missing source on weird editable layouts; do
                # not fail the whole build for a missing optional ext.
                self.app.display_warning(
                    f'[microtaint-c-ext] {source_rel} not found, skipping',
                )
                continue

            so_path = source.with_name(module_name + ext_suffix)
            self._compile(cc_cmd, py_include, source, so_path)

            # Tell hatchling to include this .so in the wheel under the
            # same package path (relative to project root).
            rel_so = so_path.relative_to(self.root).as_posix()
            force_include[str(so_path)] = rel_so
            artifacts.append(rel_so)

    def _compile(
        self,
        cc_cmd: list[str],
        py_include: str,
        source: Path,
        output: Path,
    ) -> None:
        """Run the compiler and surface a clear error on failure."""
        # Include path: Python headers + the source's own directory (for
        # cell_c_api.h, circuit_bytecode.h, cell_core.h shared headers).
        include_dir = str(source.parent)

        cmd = [
            *cc_cmd,
            f'-I{py_include}',
            f'-I{include_dir}',
        ]

        # Windows specifically requires linking against the python library
        if sys.platform == 'win32':
            # Use sys.base_prefix to reliably find the Python root, then /libs
            py_libdir = Path(sys.base_prefix) / 'libs'

            py_version = sysconfig.get_config_var('VERSION')
            if not py_version:
                py_version = f'{sys.version_info.major}{sys.version_info.minor}'

            # Instead of -L and -l, we can just pass the direct path to the .lib file
            # which works flawlessly with both MinGW (GCC) and MSVC on Windows.
            lib_path = py_libdir / f'python{py_version}.lib'
            cmd.append(str(lib_path))

        cmd.extend(
            [
                str(source),
                '-o',
                str(output),
            ],
        )

        self.app.display_info(f'[microtaint-c-ext] {source.name} → {output.name}')
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f'[microtaint-c-ext] C compiler not found: {cc_cmd[0]!r}. Set $CC to an available compiler.',
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f'[microtaint-c-ext] failed to compile {source.name} (exit code {exc.returncode}): {" ".join(cmd)}',
            ) from exc


def get_build_hook() -> type[BuildHookInterface]:
    """Entry point used by hatchling to discover the hook."""
    return MicrotaintCExtBuildHook


# Hatchling discovers hooks declared in pyproject.toml via the
# `hatch-build.hooks.<name>` table; the actual class is found by importing
# this module. Module-level alias so the simple form works:
hatch_register_build_hook = MicrotaintCExtBuildHook
