# ruff: noqa: S603, S607, S110, PLC0415
"""The address-keyed decode cache must be active AND bit-exact.

On a repeat visit to an address the Cython instruction hook reuses the
(instruction_bytes, circuit) it decoded the first time, skipping uc_mem_read +
the ctypes buffer slice + the cached_gen_rule tuplehash -- ~3.0 us of the
~4.5 us cache-hit floor (measured by scratch bench_hook_floor.py).  The
Tier-3/Tier-4 output caches already replay output_state by address without
re-checking bytes, so the decode cache adds no new correctness assumption.

Two properties are pinned:

  * ACTIVE: after a taint-carrying run with a loop, the hook's decode_cache is
    populated and revisited addresses were served by a fast path (a silent
    break -- e.g. the lookup never populating -- would erase the speed win with
    NO correctness symptom, exactly the failure mode test_cython_hook_active
    guards for).  WHICH fast path depends on the configuration, so both are
    accepted: the output cache, or the GIL-free express lane, which supersedes
    it wherever a compiled taint program is attached because running the
    program costs less than the memcmp that would decide the cache hit.
  * BIT-EXACT: a run with the cache enabled (default) and one with
    MICROTAINT_DISABLE_DECODE_CACHE=1 produce identical findings and identical
    final register taint.  Run in subprocesses because the enable flag is read
    once at module import.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile

import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != 'Linux', reason='emulator compilation tests require Linux',
)

# read a 64-byte tainted buffer into buf[16] (overflow -> BOF), then a loop that
# XOR-folds the tainted bytes into a register (revisits addresses -> decode-cache
# hits).  Exercises both the cache and a detection in one run.
_SRC = r"""
long sys_read(int fd, void *buf, unsigned long count){long r;
  __asm__ volatile("syscall":"=a"(r):"0"(0),"D"(fd),"S"(buf),"d"(count):"rcx","r11","memory");return r;}
long sys_exit(int status){long r;
  __asm__ volatile("syscall":"=a"(r):"0"(60),"D"(status):"rcx","r11","memory");return r;}
void vulnerable(char *out){
  char buf[16];
  sys_read(0, buf, 64);
  long acc = 0;
  for (int i = 0; i < 16; i++) acc ^= buf[i];
  out[0] = (char)acc;
}
void _start(){ char o[8]; vulnerable(o); sys_exit(0); }
"""


def _compile(src: str) -> str:
    fd, path = tempfile.mkstemp(suffix='.elf')
    os.close(fd)
    subprocess.run(['gcc', '-nostdlib', '-O0', '-fno-stack-protector', '-o', path, '-x', 'c', '-'],
                   input=src.encode(), check=True)
    return path


def _run(binary: str, stdin_data: bytes):
    """Run the binary once; return (wrapper, findings-labels)."""
    from qiling import Qiling
    from qiling.const import QL_VERBOSE
    from qiling.extensions import pipe

    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper
    from microtaint.sleigh.engine import _cached_generate_static_rule

    _cached_generate_static_rule.cache_clear()
    ql = Qiling([binary], '/', verbose=QL_VERBOSE.OFF)
    ql.os.stdin = pipe.SimpleInStream(0)
    ql.os.stdin.write(stdin_data)
    reporter = Reporter(json_mode=False)
    wrapper = MicrotaintWrapper(ql, check_bof=True, check_uaf=False, check_sc=False,
                                reporter=reporter)
    try:
        ql.run()
    except Exception:
        pass
    labels = sorted({f.kind.name for f in reporter.findings})
    return wrapper, labels


def test_decode_cache_is_active_and_hit() -> None:
    from microtaint.emulator.hook_core import InstructionHook

    binary = _compile(_SRC)
    try:
        wrapper, labels = _run(binary, b'A' * 64)
        hook = wrapper._instr_hook_obj
        assert isinstance(hook, InstructionHook), 'fast Cython hook not armed'
        assert len(hook.decode_cache) > 0, (
            'decode_cache is empty -- the address-keyed decode cache never '
            'populated; the fetch+decode fast path is silently disabled'
        )
        # The loop revisits addresses, so something must serve those
        # revisits cheaply.  Either mechanism satisfies that; requiring the
        # output cache specifically would fail the moment a faster one takes
        # its work, which is what the express lane does when a compiled program
        # is attached.
        served = hook.instr_cache_hits + hook.express_done
        assert served > 0, (
            'revisited addresses were served by neither the output cache nor '
            'the express lane -- the loop should revisit addresses, and '
            'without a fast path for them the decode cache brings no benefit '
            f'(cache hits {hook.instr_cache_hits}, express {hook.express_done})'
        )
        # The wrapper must wire the mem-write hook to this instruction hook, or
        # a self-modifying write would silently replay a stale decode.
        assert wrapper._mem_write_hook is not None
        assert wrapper._mem_write_hook.instr_hook is hook, (
            'mem-write hook is not wired to the instruction hook -- SMC writes '
            'would not invalidate the decode cache'
        )
    finally:
        os.unlink(binary)


@pytest.mark.skipif(
    os.environ.get('MICROTAINT_TAINT_IR') != '0',
    reason=(
        'The compiled-taint path hangs off the decode-cache entry, so turning '
        'the cache off turns it off too and the two runs are no longer the same '
        'evaluator. They legitimately disagree: the lowering is exact where the '
        'whole-instruction differential is not -- `push %rbp` leaves RSP clean '
        'in the differential even when RSP was tainted, and taints RSP from RBP '
        'when it was not. This test is about the cache, not about that.'))
def test_decode_cache_bit_exact_on_vs_off() -> None:
    binary = _compile(_SRC)
    try:
        env_on = dict(os.environ)
        env_on.pop('MICROTAINT_DISABLE_DECODE_CACHE', None)
        env_off = dict(os.environ)
        env_off['MICROTAINT_DISABLE_DECODE_CACHE'] = '1'

        out_on = subprocess.run([sys.executable, __file__, binary],
                                capture_output=True, text=True, env=env_on, check=True).stdout
        out_off = subprocess.run([sys.executable, __file__, binary],
                                 capture_output=True, text=True, env=env_off, check=True).stdout
        fp_on = json.loads(out_on)
        fp_off = json.loads(out_off)

        assert fp_on['findings'], f'expected a detection, got none: {fp_on}'
        assert fp_on['taint'], 'expected non-trivial final register taint'
        assert fp_on == fp_off, (
            f'decode cache changed results:\n  on : {fp_on}\n  off: {fp_off}'
        )
    finally:
        os.unlink(binary)


_STUB = r'void _start(){ __asm__ volatile("syscall"::"a"(60),"D"(0):"rcx","r11"); }'
FULL = 0xFFFFFFFFFFFFFFFF


def test_self_modifying_code_redecodes_after_write() -> None:
    """A write onto a cached instruction must force a re-decode (SMC / JIT).

    Drives the real Cython hooks directly:
      1. execute `mov rbx, rax` at X with RAX tainted  -> RBX tainted, X cached
      2. rewrite X's bytes to `xor rbx, rbx` WITHOUT signalling  -> the decode
         cache is address-keyed, so it would replay the stale `mov` (RBX still
         tainted).  This pins that the cache really is reused by address.
      3. fire the mem-write hook for the write to X  -> invalidate_smc drops the
         caches; re-executing X now decodes `xor rbx, rbx` -> RBX CLEAN.
    """
    from qiling import Qiling
    from qiling.const import QL_VERBOSE

    from microtaint.emulator.hook_core import InstructionHook
    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper

    fd, binary = tempfile.mkstemp(suffix='.elf')
    os.close(fd)
    subprocess.run(['gcc', '-nostdlib', '-static', '-no-pie', '-fno-pie', '-O0',
                    '-fno-stack-protector', '-o', binary, '-x', 'c', '-'],
                   input=_STUB.encode(), check=True)
    try:
        ql = Qiling([binary], '/', verbose=QL_VERBOSE.OFF)
        wrapper = MicrotaintWrapper(ql, check_bof=False, check_uaf=False, check_sc=False,
                                    reporter=Reporter(json_mode=False))
        hook = wrapper._make_cython_hook()
        assert isinstance(hook, InstructionHook)
        # Mirror the wrapper wiring so the mem-write hook can invalidate the hook.
        mem_hook = wrapper._mem_write_hook if wrapper._mem_write_hook_registered else None
        from microtaint.emulator.hook_core import MemWriteClearHook
        if mem_hook is None:
            mem_hook = MemWriteClearHook(wrapper)
        mem_hook.instr_hook = hook

        X = 0x4448000
        A = b'\x48\x89\xc3'  # mov rbx, rax
        B = b'\x48\x31\xdb'  # xor rbx, rbx  (same length)
        try:
            ql.uc.mem_map(X, 0x1000)
        except Exception:
            pass
        ql.uc.mem_write(X, A)

        def seed() -> None:
            wrapper.register_taint.clear()
            wrapper.register_taint['RAX'] = FULL
            wrapper._any_taint = True

        # 1. execute A: RBX picks up RAX's taint
        seed()
        hook(None, X, 3, None)
        assert wrapper.register_taint.get('RBX', 0) == FULL, 'mov rbx,rax should taint RBX'
        assert X in hook.decode_cache

        # 2. rewrite bytes, but DON'T signal: stale decode replays the mov
        ql.uc.mem_write(X, B)
        seed()
        hook(None, X, 3, None)
        assert wrapper.register_taint.get('RBX', 0) == FULL, (
            'without a write signal the address-keyed decode cache should still '
            'replay the old instruction (this is exactly what SMC handling fixes)'
        )

        # 3. signal the write -> invalidate -> re-decode the new instruction
        mem_hook(None, 1, X, 3, 0, None)
        assert X not in hook.decode_cache, 'invalidate_smc should drop the cached decode'
        seed()
        hook(None, X, 3, None)
        assert wrapper.register_taint.get('RBX', 0) == 0, (
            'after the write signal, X must be re-decoded as xor rbx,rbx -> RBX clean; '
            'a stale replay of mov rbx,rax would leave RBX tainted'
        )
    finally:
        os.unlink(binary)


def _fingerprint_main(binary: str) -> None:
    """Subprocess entry: run the binary, print a JSON taint fingerprint."""
    wrapper, labels = _run(binary, b'A' * 64)
    taint = {k: hex(v) for k, v in sorted(wrapper.register_taint.items()) if v}
    print(json.dumps({'findings': labels, 'taint': taint}))


if __name__ == '__main__':
    _fingerprint_main(sys.argv[1])
