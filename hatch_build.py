"""
Hatchling build hook for microtaint's hand-written C extensions.

Compiles:
  microtaint/instrumentation/cell_c/cell_c.c    → cell_c.<EXT>
  microtaint/instrumentation/cell_c/circuit_c.c → circuit_c.<EXT>
  microtaint/instrumentation/cell_c/taint_ir_c.c → taint_ir_c.<EXT>

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
import platform
import shlex
import subprocess
import sys
import sysconfig
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

# --------------------------------------------------------------------------
# Native re-execution (4th concrete-execution path).  A hand-written
# trampoline, so it is compiled INTO cell_c only where it can be; elsewhere
# cell_c is built without it and falls back to SLEIGH.  When present,
# MICROTAINT_HAVE_REEXEC is defined so cell_c.c gates every reexec use on it.
#
# The machine is not the only requirement, and treating it as one broke both
# non-Linux wheels:
#
#   * reexec.c includes <sys/mman.h> for the executable mapping, which Windows
#     does not have -- `fatal error: sys/mman.h: No such file or directory`.
#   * the .S files end in `.section .note.GNU-stack,"",@progbits`, which marks
#     the stack non-executable and is ELF syntax.  macOS is Mach-O and its
#     assembler rejects it -- `unexpected token in '.section' directive`.
#     Their `.globl` names would not match Mach-O's leading-underscore
#     convention either, so the directive is only the first thing that fails.
#
# So: Linux, on a machine with a trampoline.  Porting the stubs to Mach-O or
# Windows is a real piece of work and nobody has done it; claiming the feature
# by architecture alone only moved the failure into the wheel build.
# --------------------------------------------------------------------------
_REEXEC_DIR = 'microtaint/reexec'
_REEXEC_ASM = {
    'x86_64': 'microtaint/reexec/reexec_amd64.S',
    'AMD64': 'microtaint/reexec/reexec_amd64.S',
    'aarch64': 'microtaint/reexec/reexec_arm64.S',
    'arm64': 'microtaint/reexec/reexec_arm64.S',
}.get(platform.machine()) if platform.system() == 'Linux' else None
_REEXEC_AVAILABLE = _REEXEC_ASM is not None
_REEXEC_SOURCES = ['microtaint/reexec/reexec.c', _REEXEC_ASM] if _REEXEC_AVAILABLE else []

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
#: Where taint_ir_c's public header lives; blockpath_c.c includes it.
_TAINT_IR_C_DIR = 'microtaint/instrumentation/cell_c'

C_EXTENSIONS: list[tuple[str, str]] = [
    ('microtaint/instrumentation/cell_c/cell_c.c', 'cell_c'),
    ('microtaint/instrumentation/cell_c/circuit_c.c', 'circuit_c'),
    ('microtaint/instrumentation/cell_c/taint_ir_c.c', 'taint_ir_c'),
    ('microtaint/emulator/blockpath_c.c', 'blockpath_c'),
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
            # cell_c gains the native re-exec path on x86_64 hosts.
            extra_sources: list[Path] = []
            extra_flags: list[str] = []
            if module_name == 'circuit_c':
                # The circuit interpreter keeps its operand stack in a local
                # array, and GCC's SLP vectoriser turns every two-operand op
                # into a 16-byte load of stack[sp-2..sp-1], a lane shift and a
                # vector add:
                #
                #     vmovdqu 0x190(%rsp,%rax,8),%xmm0
                #     vpsrldq $0x8,%xmm0,%xmm1
                #     vpaddq  %xmm1,%xmm0,%xmm0
                #     vmovq   %xmm0,0x190(%rsp,%rax,8)
                #
                # That 16-byte load overlaps the two 8-byte stores that just
                # pushed those operands, so it cannot store-forward and stalls.
                # Measured with perf annotate, those three instructions were
                # 25% of eval_program, itself 27.65% of the run; turning the
                # vectoriser off on THIS FILE took the RQ5 workload from 5.87s
                # to 4.89s (-17%) with no other change.
                #
                # Scoped to circuit_c deliberately: vectorisation is worth
                # having in the cell kernel, which does real bulk work. The
                # problem here is a stack machine whose "vectors" are two
                # adjacent operands it just wrote.
                extra_flags = ['-fno-tree-slp-vectorize']
            if module_name == 'blockpath_c':
                # The block runtime runs a taint-IR program through the
                # interpreter the emitter's declines fall back to, so it needs
                # taint_ir_c's public header -- and that lives beside
                # taint_ir_c.c, not beside blockpath_c.c.  Only the source's
                # own directory is on the include path by default, so without
                # this a clean build fails outright.
                extra_flags = [f'-I{Path(self.root) / _TAINT_IR_C_DIR}']
            if module_name == 'cell_c' and _REEXEC_AVAILABLE:
                extra_sources = [Path(self.root) / s for s in _REEXEC_SOURCES]
                extra_flags = ['-DMICROTAINT_HAVE_REEXEC=1', f'-I{Path(self.root) / _REEXEC_DIR}']
                if all(p.is_file() for p in extra_sources):
                    self.app.display_info('[microtaint-c-ext] cell_c += native re-exec (x86_64)')
                else:
                    extra_sources, extra_flags = [], []
            self._compile(cc_cmd, py_include, source, so_path, extra_sources, extra_flags)

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
        extra_sources: list[Path] | None = None,
        extra_flags: list[str] | None = None,
    ) -> None:
        """Run the compiler and surface a clear error on failure."""
        # Include path: Python headers + the source's own directory (for
        # cell_c_api.h, circuit_bytecode.h, cell_core.h shared headers).
        include_dir = str(source.parent)

        cmd = [
            *cc_cmd,
            f'-I{py_include}',
            f'-I{include_dir}',
            *(extra_flags or []),
            str(source),
            *[str(s) for s in (extra_sources or [])],
            '-o',
            str(output),
        ]

        # Windows specifically requires linking against the python library
        if sys.platform == 'win32':
            py_libdir = Path(sys.base_prefix) / 'libs'
            py_version = sysconfig.get_config_var('VERSION')
            if not py_version:
                py_version = f'{sys.version_info.major}{sys.version_info.minor}'

            lib_path = py_libdir / f'python{py_version}.lib'
            cmd.append(str(lib_path))

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
