"""Native re-execution: the 4th concrete-execution path.

When the host ISA equals the target ISA, the concrete state a value-aware /
differential taint rule needs can be obtained by executing the real instruction
bytes on the host CPU (~30 ns, signal-guarded) instead of interpreting the
p-code via SLEIGH (~3.3 us). This module wraps the AMD64 in-process trampoline
(``reexec.c`` + ``reexec_amd64.S``); see ``README.md`` for the design, the
measured ~110x speedup, and the correctness result (bit-identical to SLEIGH on
every defined output; undefined flags such as OF after a count!=1 shift are
avalanche-floored by the taint model anyway).

Usage::

    from microtaint.reexec import NativeReExec, AVAILABLE
    if AVAILABLE:
        rx = NativeReExec()
        out_gpr, out_rflags = rx.run(b'\\x48\\x01\\xd8',  # add rax, rbx
                                     {'RAX': 5, 'RBX': 7}, 0x202)

Instructions that dereference memory, change control flow, or are
non-deterministic/privileged (syscall/cpuid/rdtsc) must NOT be run here; the
caller gates on the instruction's operands and falls back to SLEIGH.
"""
from __future__ import annotations

import ctypes
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

# cpu_state_t.gpr[16] index order (x86-64 encoding order).
REG_ORDER = ('RAX', 'RCX', 'RDX', 'RBX', 'RSP', 'RBP', 'RSI', 'RDI',
             'R8', 'R9', 'R10', 'R11', 'R12', 'R13', 'R14', 'R15')
_REG_IDX = {n: i for i, n in enumerate(REG_ORDER)}

# RFLAGS bit positions for the arithmetic flags.
FLAG_BITS = {'CF': 0x1, 'PF': 0x4, 'AF': 0x10, 'ZF': 0x40, 'SF': 0x80, 'OF': 0x800}

_PKG_DIR = Path(__file__).resolve().parent
_M64 = (1 << 64) - 1

# Per-host-arch trampoline assembly (the harness reexec.c is arch-general).
_ARCH_ASM = {
    'x86_64': 'reexec_amd64.S', 'AMD64': 'reexec_amd64.S',
    'aarch64': 'reexec_arm64.S', 'arm64': 'reexec_arm64.S',
}
_HOST_ASM = _ARCH_ASM.get(platform.machine())
AVAILABLE = _HOST_ASM is not None and shutil.which('cc') is not None


class _CpuState(ctypes.Structure):
    _fields_ = [('gpr', ctypes.c_uint64 * 16), ('rflags', ctypes.c_uint64)]


class NativeReExecUnavailable(RuntimeError):
    pass


_LIB_CACHE: ctypes.CDLL | None = None


def _build_and_load() -> ctypes.CDLL:
    """Compile reexec.c + reexec_amd64.S into a shared library and load it.

    Built once per process into a temp dir (the installed package dir is not
    assumed writable).  Cheap (~50 ms) and only when native re-exec is used.
    """
    global _LIB_CACHE  # noqa: PLW0603
    if _LIB_CACHE is not None:
        return _LIB_CACHE
    if not AVAILABLE:
        raise NativeReExecUnavailable('native re-exec needs an x86_64 host with cc')
    out = Path(tempfile.mkdtemp(prefix='mt_reexec_')) / 'reexec.so'
    cc = shutil.which('cc') or 'cc'
    subprocess.run(  # noqa: S603
        [cc, '-O2', '-shared', '-fPIC', '-o', str(out),
         str(_PKG_DIR / 'reexec.c'), str(_PKG_DIR / _HOST_ASM)],
        check=True,
    )
    lib = ctypes.CDLL(str(out))
    lib.reexec_init.restype = ctypes.c_int
    lib.reexec_run_one.restype = ctypes.c_int
    lib.reexec_run_one.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    lib.reexec_arm.restype = ctypes.c_int
    lib.reexec_set_instr.restype = ctypes.c_int
    lib.reexec_set_instr.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.reexec_call.restype = ctypes.c_int
    lib.reexec_call.argtypes = [ctypes.c_void_p]
    if lib.reexec_init() != 0:
        raise NativeReExecUnavailable('reexec_init failed')
    _LIB_CACHE = lib
    return lib


class NativeReExec:
    """Execute one target instruction on the host CPU and read back the state.

    AMD64 only for now (the trampoline is x86-64); the caller must ensure the
    instruction is register-only, not control-flow, and deterministic.
    """

    def __init__(self) -> None:
        self._lib = _build_and_load()

    def run(self, code: bytes, gpr: dict[str, int], rflags: int = 0x202):
        """Run ``code`` with the given register values + RFLAGS.

        Returns ``(out_gpr: dict[str,int], out_rflags: int)`` on success, or
        ``None`` if the instruction faulted (caller falls back to SLEIGH).
        """
        st = _CpuState()
        for name, val in gpr.items():
            idx = _REG_IDX.get(name)
            if idx is not None:
                st.gpr[idx] = val & _M64
        st.rflags = rflags & _M64
        rc = self._lib.reexec_run_one(ctypes.byref(st), code, len(code))
        if rc != 0:
            return None
        return ({REG_ORDER[i]: st.gpr[i] for i in range(16)}, st.rflags)

    @staticmethod
    def flag(rflags: int, name: str) -> int:
        return 1 if (rflags & FLAG_BITS[name]) else 0
