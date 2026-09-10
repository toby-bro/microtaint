# ruff: noqa: S603, S607, PLC0415
"""The Cython fast-path instruction hook must actually build and arm.

The emulator's hot path is the Cython ``InstructionHook``.  When its
construction raises, ``MicrotaintWrapper._make_cython_hook`` swallows the
exception and ``_arm_deferred_hooks`` silently falls back to the pure-Python
``_instruction_evaluator_raw`` bound method (wrapper.py: ``self._make_cython_hook()
or self._instruction_evaluator_raw``).  Taint results stay correct, so NO
correctness test catches the regression -- the only symptom is a large,
invisible slowdown (real case: a bad ctypes arg pre-wrap made every
InstructionHook.__init__ raise, dropping the whole run onto the slow path).

This test pins that the fast hook builds AND is the object actually armed, so a
future change that breaks the Cython hook fails loudly instead of degrading
silently.
"""
from __future__ import annotations

import os
import platform
import subprocess
import tempfile
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from microtaint.emulator.wrapper import MicrotaintWrapper

pytestmark = pytest.mark.skipif(
    platform.system() != 'Linux' or os.environ.get('MICROTAINT_DISABLE_CYTHON_HOOK') == '1',
    reason='emulator hook tests require Linux + gcc, and the Cython hook must not be '
    'intentionally disabled (MICROTAINT_DISABLE_CYTHON_HOOK=1)',
)

# Minimal static, no-libc ELF: a valid target so Qiling can construct.
_SRC = r"""
void _start(){ __asm__ volatile("syscall"::"a"(60),"D"(0):"rcx","r11"); }
"""


def _compile() -> str:
    fd, path = tempfile.mkstemp(suffix='.elf')
    os.close(fd)
    subprocess.run(
        ['gcc', '-nostdlib', '-static', '-no-pie', '-fno-pie', '-O0',
         '-fno-stack-protector', '-o', path, '-x', 'c', '-'],
        input=_SRC.encode(), check=True,
    )
    return path


def _wrapper() -> tuple[MicrotaintWrapper, str]:
    from qiling import Qiling
    from qiling.const import QL_VERBOSE

    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper

    binary = _compile()
    ql = Qiling([binary], '/', verbose=QL_VERBOSE.OFF)
    reporter = Reporter(json_mode=False)
    return MicrotaintWrapper(ql, reporter=reporter), binary


def test_make_cython_hook_builds() -> None:
    from microtaint.emulator.hook_core import InstructionHook

    wrapper, binary = _wrapper()
    try:
        hook = wrapper._make_cython_hook()
        assert hook is not None, (
            '_make_cython_hook returned None -- the Cython hot-path hook failed '
            'to construct (the emulator would silently use the slow Python hook)'
        )
        assert isinstance(hook, InstructionHook)
    finally:
        os.unlink(binary)


def test_armed_hook_is_the_cython_hook_not_the_python_fallback() -> None:
    from microtaint.emulator.hook_core import InstructionHook

    wrapper, binary = _wrapper()
    try:
        # Introduce taint through the public API -> _arm_deferred_hooks picks the
        # hook object that the emulator will actually run per instruction.
        wrapper.taint_region(0x100000, b'\xff' * 8)
        assert isinstance(wrapper._instr_hook_obj, InstructionHook), (
            'the armed instruction hook is the slow Python fallback, not the '
            'Cython InstructionHook -- the fast path silently degraded'
        )
    finally:
        os.unlink(binary)
