#!/usr/bin/env python3
"""
overhead_bench.py — measure CPU and memory overhead of running bench.elf
under several configurations:

  1. native        — direct execution of bench.elf
  2. qiling-only   — bench.elf under qiling emulation, NO microtaint instrumentation
  3. microtaint *  — microtaint with each combination of detection modes

For each microtaint configuration, the script also decomposes total wall
time into:
    qiling_init_s   — time spent constructing the Qiling object and the
                      MicrotaintWrapper (one-shot setup cost)
    qiling_run_s    — time spent inside ql.run() (the actual taint
                      propagation phase)

The decomposition uses an in-process measurement (the script imports
microtaint and times around ql.run() directly).  All other measurements
use a child process so peak-RSS / CPU times reflect the same accounting
that a user would see from `/usr/bin/time`.

Usage:
    # build bench.elf from bench.c automatically and run with a generated input
    python3 overhead_bench.py --build-bench bench.c --gen-input 256

    # use an existing binary and stdin file
    python3 overhead_bench.py /path/to/bench.elf --stdin-file input.bin

    # 100 runs per config, save full results
    python3 overhead_bench.py --build-bench bench.c --gen-input 256 \
        --runs 100 --json overhead.json

Notes on bench.elf with --gen-input 256
---------------------------------------
The benchmark deliberately triggers a stack BOF when fed 256 bytes (the
``unsafe_copy`` overflow in the C source).  Native execution will SEGV
or hang as a result; the script handles non-zero exit codes / timeouts
gracefully.  microtaint's --check-bof should detect this before the
binary crashes.  If you want a clean exit (no BOF) for pure
mix-propagation timing, use ``--gen-input 64`` instead.
"""

# Experiment script, not library code: see artifacts/ndss27/README.md,
# "Lint and type checking", for why annotations are not required here.
# mypy: disable-error-code="no-untyped-def, no-untyped-call, type-arg, explicit-any"

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
from functools import lru_cache as _lru_cache
from typing import Any


def cpu_boost_state(sysfs: str = '/sys/devices/system/cpu') -> tuple[bool | None, str]:
    """Is CPU boost/turbo on?  (state, how we know) -- None when unknowable.

    Boost is not "the same numbers, faster": it lets the clock run until the
    package heats, then throttles, so two measurements of the same ladder are
    taken at different frequencies and their DIFFERENCE -- which is the whole
    point -- stops meaning anything.

    Two controls, with opposite polarity, and the driver-specific one wins:

        intel_pstate/no_turbo   0 means turbo is ON   (Intel, active mode)
        cpufreq/boost           1 means boost is ON   (amd-pstate, acpi-cpufreq,
                                                       and intel_pstate passive)

    `sysfs` is injectable so both branches can be tested on a machine that only
    has one of them -- this one is amd-pstate-epp, so the Intel path would
    otherwise never execute until it mattered.
    """
    for rel, on_when in (('intel_pstate/no_turbo', '0'), ('cpufreq/boost', '1')):
        path = f'{sysfs}/{rel}'
        try:
            with open(path) as fh:
                raw = fh.read().strip()
        except OSError:
            continue
        return raw == on_when, f'{path}={raw}'
    return None, f'no boost/turbo control under {sysfs}'


def cpu_scaling_driver(sysfs: str = '/sys/devices/system/cpu') -> str | None:
    """Which cpufreq driver is in charge, recorded for context."""
    try:
        with open(f'{sysfs}/cpu0/cpufreq/scaling_driver') as fh:
            return fh.read().strip()
    except OSError:
        return None


@_lru_cache(maxsize=1)
def require_no_cpu_boost() -> dict:
    """Refuse to time anything while boost is on, and say so.

    Override with ALLOW_CPU_BOOST=1 when you knowingly want the numbers anyway.
    The state is returned either way so it lands in the result file: a timing
    figure whose machine state is unrecorded cannot be compared with another.
    """
    on, how = cpu_boost_state()
    state = {'cpu_boost': on, 'detected_via': how,
             'scaling_driver': cpu_scaling_driver()}
    if on and os.environ.get('ALLOW_CPU_BOOST') != '1':
        raise SystemExit(
            f'CPU boost is ENABLED ({how}).\n'
            'These rungs are compared against each other, so they must all be '
            'measured at the same clock.  Boost runs until the package heats and '
            'then throttles, which silently changes the clock between rungs.\n'
            '  disable: echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost\n'
            '  or set ALLOW_CPU_BOOST=1 to measure anyway (the state is recorded '
            'in the output either way).')
    if on:
        print(f'[!] CPU boost is ON ({how}) and ALLOW_CPU_BOOST=1: '
              'these timings are not comparable across rungs.', flush=True)
    return state


def _engine_provenance() -> dict:
    """Which engine produced this result: commit, version, dirty flag.

    Stamped into every result file so a number can always be traced back to the
    engine that measured it.  The published RQ2/3/4 macros could not be matched
    to any report in the tree because nothing recorded this.

    Never raises.  An installed wheel has no git repository and a tarball has no
    `.git`; neither is a reason for an experiment to stop, so an unanswerable
    question writes an empty dict rather than ending the run.
    """
    try:
        from microtaint.provenance import engine_provenance
        return engine_provenance()
    except Exception:
        return {}


def _rusage_self() -> tuple[float, float, int]:
    """Return (user_cpu_s, sys_cpu_s, max_rss_bytes) for the current process.
    On Linux, ru_maxrss is in kilobytes.  We normalise to bytes."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime, ru.ru_stime, ru.ru_maxrss * 1024


def _peak_rss_kb_via_proc(pid: int) -> int:
    """Read VmPeak from /proc/<pid>/status.  Returns 0 if unavailable
    (e.g. process already exited and /proc entry was reaped).
    VmHWM is "high water mark" of resident set size in KiB."""
    try:
        with open(f'/proc/{pid}/status') as f:
            for line in f:
                if line.startswith('VmHWM:'):
                    return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return 0
    return 0


def _bytes_to_mib(b: int) -> float:
    return b / (1024 * 1024)


# ---------------------------------------------------------------------------
# Bench source build + input generation helpers
# ---------------------------------------------------------------------------


def build_bench(
    src_path: str, out_path: str | None = None, cc: str = 'gcc', extra_flags: list[str] | None = None,
) -> str:
    """
    Compile bench.c into a static, no-libc x86-64 ELF.

    The benchmark uses inline-syscall wrappers and ``_start`` directly
    (no main, no crt) so it must be linked with ``-nostdlib`` and
    ``-static -no-pie -fno-pie`` for predictable addresses (matters for
    the BOF demonstration: the saved RIP location is deterministic).

    Returns the absolute path of the produced ELF.
    """
    if out_path is None:
        # Default to CWD/bench.elf rather than alongside the source — the
        # source might live in a read-only location (e.g. /mnt/uploads).
        out_path = os.path.join(os.getcwd(), 'bench.elf')
    out_path = os.path.abspath(out_path)
    cmd = [
        cc,
        '-O0',
        '-static',
        '-nostdlib',
        '-fno-pie',
        '-no-pie',
        # Some distros (Ubuntu, Debian, Arch) enable -fstack-protector-strong
        # by default at the gcc spec level, which makes the compiler emit
        # __stack_chk_fail calls in any function with a stack array.  With
        # -nostdlib there's no libc to satisfy that symbol → link error.
        # The benchmark deliberately overflows a stack buffer in
        # unsafe_copy(), so the canary would defeat the whole point of the
        # BOF demonstration anyway.
        '-fno-stack-protector',
        # Same story for FORTIFY_SOURCE: enabled by default on some
        # distros, requires libc, and would replace the unbounded copy
        # loop with a bounded one.
        '-U_FORTIFY_SOURCE',
        '-D_FORTIFY_SOURCE=0',
        # Ensure no-pie binary even when the spec file forces -pie.
        '-fno-stack-clash-protection',
        os.path.abspath(src_path),
        '-o',
        out_path,
    ]
    if extra_flags:
        cmd.extend(extra_flags)
    print(f'# building: {" ".join(cmd)}')
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        raise RuntimeError(f'compilation failed (exit {r.returncode})')
    return out_path


def gen_input(size: int, seed: int = 0xC0FFEE, out_path: str | None = None) -> str:
    """
    Generate a deterministic ``size``-byte stdin file.

    Deterministic so timing comparisons across runs are reproducible —
    the same input bytes mean the same propagation graph, the same
    cache hit rate, the same SBOX lookup pattern.
    """
    import random

    if out_path is None:
        out_path = f'/tmp/bench_input_{size}_{seed:08x}.bin'
    rng = random.Random(seed)
    data = bytes(rng.randrange(256) for _ in range(size))
    with open(out_path, 'wb') as f:
        f.write(data)
    print(f'# wrote {size} byte deterministic input → {out_path}')
    return out_path


@dataclass
class Measurement:
    """One run's measurement result."""

    label: str
    wall_s: float
    user_cpu_s: float
    sys_cpu_s: float
    peak_rss_mib: float
    extra: dict[str, Any] = field(default_factory=dict)


def _run_subprocess(
    label: str, argv: list[str], env: dict | None = None, stdin_data: bytes | None = None, timeout: float = 600.0,
) -> Measurement:
    """Run a child process and measure wall time, CPU time, and peak RSS.

    Strategy: use ``os.wait4()`` to obtain a ``rusage`` struct for THIS
    specific child at exit time.  ``rusage.ru_maxrss`` is the peak
    resident set size of that child only (Linux: in KiB; Mac: in bytes
    — we assume Linux here).

    For very short-lived children we also poll ``/proc/<pid>/status`` at
    5 ms intervals as a fallback.  The two values are reconciled by
    taking the max.
    """
    import threading

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    polled_peak_kb = [0]
    poller_stop = threading.Event()

    def _poll():
        while not poller_stop.is_set():
            cur = _peak_rss_kb_via_proc(proc.pid)
            if cur > polled_peak_kb[0]:
                polled_peak_kb[0] = cur
            time.sleep(0.005)

    poller = threading.Thread(target=_poll, daemon=True)
    poller.start()

    # Drive stdin/stdout pipes ourselves so we can wait4() on the same
    # child PID afterwards.  Popen.communicate() reaps the child via
    # wait(), losing the per-child rusage information.
    out_chunks: list[bytes] = []
    err_chunks: list[bytes] = []
    if stdin_data is not None:
        assert proc.stdin is not None  # opened with stdin=PIPE just above
        try:
            proc.stdin.write(stdin_data)
        except BrokenPipeError:
            pass
        proc.stdin.close()

    # Read stdout/stderr in threads so we don't deadlock on full pipes
    def _drain(stream, into):
        try:
            into.append(stream.read())
        except Exception:
            pass
        finally:
            try:
                stream.close()
            except Exception:
                pass

    t_out = threading.Thread(target=_drain, args=(proc.stdout, out_chunks), daemon=True)
    t_err = threading.Thread(target=_drain, args=(proc.stderr, err_chunks), daemon=True)
    t_out.start()
    t_err.start()

    deadline = time.monotonic() + timeout
    rusage_child = None
    try:
        # Use os.wait4 so we can capture per-child rusage.
        while True:
            pid, status, rusage = os.wait4(proc.pid, os.WNOHANG)
            if pid != 0:
                rusage_child = rusage
                proc.returncode = os.waitstatus_to_exitcode(status)
                break
            if time.monotonic() > deadline:
                proc.kill()
                # final wait — block this time
                pid, status, rusage = os.wait4(proc.pid, 0)
                rusage_child = rusage
                proc.returncode = os.waitstatus_to_exitcode(status)
                poller_stop.set()
                poller.join(timeout=1)
                raise RuntimeError(f'{label}: timeout after {timeout}s')
            time.sleep(0.001)
    finally:
        poller_stop.set()
        poller.join(timeout=1)
        t_out.join(timeout=2)
        t_err.join(timeout=2)

    t1 = time.perf_counter()
    out = b''.join(out_chunks)
    err = b''.join(err_chunks)

    user_cpu = rusage_child.ru_utime if rusage_child else 0.0
    sys_cpu = rusage_child.ru_stime if rusage_child else 0.0
    rusage_peak_kb = rusage_child.ru_maxrss if rusage_child else 0
    peak_kb = max(rusage_peak_kb, polled_peak_kb[0])

    return Measurement(
        label=label,
        wall_s=t1 - t0,
        user_cpu_s=user_cpu,
        sys_cpu_s=sys_cpu,
        peak_rss_mib=peak_kb / 1024.0,
        extra={
            'returncode': proc.returncode,
            'stdout_size': len(out),
            'stderr_size': len(err),
            # For the native run the child IS the guest, so its stdout is the
            # guest's: same workload-ran proof the emulated helpers report.
            'guest_bytes': len(out),
            'stdin_bytes': len(stdin_data) if stdin_data is not None else 0,
        },
    )


#: The engine configuration the ARTIFACT is about, pinned for every microtaint run.
#:
#: Both of these default to ON in the engine, and both replace the per-instruction
#: LogicCircuit evaluation this paper describes:
#:
#:   * MICROTAINT_TAINT_IR -- `taint_ir/engine_glue.program_for` compiles each
#:     instruction to a native taint program and the hot path calls that instead
#:     of evaluating the circuit.  `engine_glue` says so plainly: "This is the hot
#:     path's default."  Measured without this pin, every number here describes
#:     taint_ir rather than the artifact.
#:   * MICROTAINT_BLOCK -- block-at-a-time taint, which also goes through taint_ir
#:     (`taint_ir/blockcompile.compile_block`).  Opt-in already, pinned anyway so
#:     an inherited environment cannot switch it on.
#:
#: taint_ir is later work and is deliberately OUT of scope for these artifacts, so
#: it is turned off here rather than left to whatever the shell happens to export.
ARTIFACT_ENGINE_ENV = {
    'MICROTAINT_TAINT_IR': '0',
    'MICROTAINT_BLOCK': '0',
}


def _run_helper_subprocess(
    label: str, argv: list[str], stdin_data: bytes | None, timeout: float = 1800.0,
    env: dict | None = None,
) -> Measurement:
    """Run a helper subprocess that emits a JSON timing line on stdout.
    Same accounting as _run_subprocess plus parses the breakdown JSON."""
    import threading

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, **env} if env else None,
    )
    polled_peak_kb = [0]
    poller_stop = threading.Event()

    def _poll():
        while not poller_stop.is_set():
            cur = _peak_rss_kb_via_proc(proc.pid)
            if cur > polled_peak_kb[0]:
                polled_peak_kb[0] = cur
            time.sleep(0.005)

    poller = threading.Thread(target=_poll, daemon=True)
    poller.start()

    out_chunks: list[bytes] = []
    err_chunks: list[bytes] = []
    if stdin_data is not None:
        assert proc.stdin is not None  # opened with stdin=PIPE just above
        try:
            proc.stdin.write(stdin_data)
        except BrokenPipeError:
            pass
        proc.stdin.close()

    def _drain(stream, into):
        try:
            into.append(stream.read())
        except Exception:
            pass
        finally:
            try:
                stream.close()
            except Exception:
                pass

    t_out = threading.Thread(target=_drain, args=(proc.stdout, out_chunks), daemon=True)
    t_err = threading.Thread(target=_drain, args=(proc.stderr, err_chunks), daemon=True)
    t_out.start()
    t_err.start()

    deadline = time.monotonic() + timeout
    rusage_child = None
    try:
        while True:
            pid, status, rusage = os.wait4(proc.pid, os.WNOHANG)
            if pid != 0:
                rusage_child = rusage
                proc.returncode = os.waitstatus_to_exitcode(status)
                break
            if time.monotonic() > deadline:
                proc.kill()
                pid, status, rusage = os.wait4(proc.pid, 0)
                rusage_child = rusage
                proc.returncode = os.waitstatus_to_exitcode(status)
                poller_stop.set()
                poller.join(timeout=1)
                raise RuntimeError(f'{label}: timeout after {timeout}s')
            time.sleep(0.001)
    finally:
        poller_stop.set()
        poller.join(timeout=1)
        t_out.join(timeout=2)
        t_err.join(timeout=2)

    t1 = time.perf_counter()
    out = b''.join(out_chunks)
    err = b''.join(err_chunks)

    user_cpu = rusage_child.ru_utime if rusage_child else 0.0
    sys_cpu = rusage_child.ru_stime if rusage_child else 0.0
    rusage_peak_kb = rusage_child.ru_maxrss if rusage_child else 0
    peak_kb = max(rusage_peak_kb, polled_peak_kb[0])

    breakdown: dict = {}
    # The bench may emit binary data on stdout (the bench.elf writes 8 raw
    # hash bytes), so simple splitlines() can fail to isolate the helper's
    # JSON line.  We look for the LAST balanced `{...}` JSON object in the
    # text, scanning backwards from the end.
    text = out.decode('utf-8', errors='replace')
    end_idx = text.rfind('}')
    while end_idx > 0:
        # find the matching opening brace by depth-tracking from end_idx
        depth = 0
        start_idx = -1
        for i in range(end_idx, -1, -1):
            c = text[i]
            if c == '}':
                depth += 1
            elif c == '{':
                depth -= 1
                if depth == 0:
                    start_idx = i
                    break
        if start_idx < 0:
            break
        candidate = text[start_idx : end_idx + 1]
        try:
            breakdown = json.loads(candidate)
            break
        except json.JSONDecodeError:
            # try a shorter window (move end_idx backwards past this `}`)
            end_idx = text.rfind('}', 0, end_idx)

    return Measurement(
        label=label,
        wall_s=t1 - t0,
        user_cpu_s=user_cpu,
        sys_cpu_s=sys_cpu,
        peak_rss_mib=peak_kb / 1024.0,
        extra={
            'returncode': proc.returncode,
            'import_s': breakdown.get('import_s'),
            'init_s': breakdown.get('init_s'),
            'run_s': breakdown.get('run_s'),
            'guest_bytes': breakdown.get('guest_bytes'),
            'stdin_bytes': breakdown.get('stdin_bytes'),
            'taint_ir_modules': breakdown.get('taint_ir_modules'),
            'env_taint_ir': breakdown.get('env_taint_ir'),
            'env_block': breakdown.get('env_block'),
            'wrapper_stats': breakdown.get('wrapper_stats', {}),
            # Kept even on success: the instruction-count helper reports through
            # stderr precisely so the guest's own stdout cannot corrupt its JSON.
            'stderr_tail': err.decode('utf-8', errors='replace')[-400:],
        },
    )


# ---------------------------------------------------------------------------
# Measurement: native baseline
# ---------------------------------------------------------------------------


def measure_native(
    binary: str, binary_args: list[str], stdin_data: bytes | None, timeout: float = 600.0,
) -> Measurement:
    """Run the binary directly with no instrumentation."""
    return _run_subprocess(
        label='native',
        argv=[binary, *binary_args],
        stdin_data=stdin_data,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Measurement: qiling-only (no microtaint instrumentation)
# ---------------------------------------------------------------------------

QILING_ONLY_HELPER = r"""
import json, os, sys, time, io
# --- guest stdout capture -------------------------------------------------
# The guest writes to fd 1 directly, so its bytes land in OUR stdout next to the
# JSON below.  Redirect fd 1 to a temp file for the duration of ql.run() and
# report the byte count: `bench.c` writes its 8-byte hash AFTER the mix rounds
# and BEFORE the overflow, so a non-empty count is exact proof the workload ran.
# Without this the harness could not tell a real run from a guest that exited at
# its first read, which is how the published RQ5 numbers came to describe two
# process-startup paths instead of taint propagation.
import tempfile
def _capture_fd1():
    tf = tempfile.TemporaryFile()
    saved = os.dup(1)
    os.dup2(tf.fileno(), 1)
    return tf, saved
def _release_fd1(tf, saved):
    os.dup2(saved, 1)
    os.close(saved)
    tf.seek(0, 2)
    n = tf.tell()
    tf.close()
    return n
binary = sys.argv[1]
rootfs = sys.argv[2]
binary_args = sys.argv[3:]

t_import0 = time.perf_counter()
from qiling import Qiling
from qiling.const import QL_VERBOSE
t_import1 = time.perf_counter()

t_init0 = time.perf_counter()
ql = Qiling([binary, *binary_args], rootfs, verbose=QL_VERBOSE.OFF)

# Wire up our pipe stdin to the emulated process's stdin.
# The bench reads with sys_read(0, buf, 256); without this, qiling's
# stdin abstraction would be empty and the bench would exit early.
stdin_data = sys.stdin.buffer.read() if not sys.stdin.isatty() else b""
if stdin_data:
    try:
        ql.os.stdin = io.BytesIO(stdin_data)
    except Exception:
        pass
t_init1 = time.perf_counter()

_tf, _saved = _capture_fd1()
t_run0 = time.perf_counter()
try:
    ql.run()
except Exception:
    pass
t_run1 = time.perf_counter()
guest_bytes = _release_fd1(_tf, _saved)

print(json.dumps({
    "import_s": t_import1 - t_import0,
    "init_s":   t_init1 - t_init0,
    "run_s":    t_run1 - t_run0,
    "guest_bytes": guest_bytes,
    "stdin_bytes": len(stdin_data),
}))
"""


def measure_qiling_only(
    binary: str, binary_args: list[str], rootfs: str, stdin_data: bytes | None, timeout: float = 600.0,
) -> Measurement:
    """Run the binary inside a fresh Qiling with NO microtaint hooks."""
    helper_path = '/tmp/_overhead_qiling_only.py'
    with open(helper_path, 'w') as f:
        f.write(QILING_ONLY_HELPER)
    return _run_helper_subprocess(
        label='qiling-only',
        argv=[sys.executable, helper_path, binary, rootfs, *binary_args],
        stdin_data=stdin_data,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Measurement: microtaint with given check flags
# ---------------------------------------------------------------------------

MICROTAINT_HELPER = r"""
import json, os, sys, time, io
# --- guest stdout capture -------------------------------------------------
# The guest writes to fd 1 directly, so its bytes land in OUR stdout next to the
# JSON below.  Redirect fd 1 to a temp file for the duration of ql.run() and
# report the byte count: `bench.c` writes its 8-byte hash AFTER the mix rounds
# and BEFORE the overflow, so a non-empty count is exact proof the workload ran.
# Without this the harness could not tell a real run from a guest that exited at
# its first read, which is how the published RQ5 numbers came to describe two
# process-startup paths instead of taint propagation.
import tempfile
def _capture_fd1():
    tf = tempfile.TemporaryFile()
    saved = os.dup(1)
    os.dup2(tf.fileno(), 1)
    return tf, saved
def _release_fd1(tf, saved):
    os.dup2(saved, 1)
    os.close(saved)
    tf.seek(0, 2)
    n = tf.tell()
    tf.close()
    return n


binary = sys.argv[1]
rootfs = sys.argv[2]
flag_string = sys.argv[3]   # comma-separated: bof,uaf,sc,aiw  (empty="")
binary_args = sys.argv[4:]

flags = set(f for f in flag_string.split(",") if f)

t_import0 = time.perf_counter()
from qiling import Qiling
from qiling.const import QL_VERBOSE
from microtaint.emulator.wrapper import MicrotaintWrapper
from microtaint.emulator.heap import HeapTracker
from microtaint.emulator.reporter import Reporter
t_import1 = time.perf_counter()

# Read the parent's stdin pipe BEFORE qiling init so we can wire it up
# directly.  This mirrors what microtaint's CLI --input flag does
# internally (reads a file and replaces ql.os.stdin with a BytesIO).
stdin_data = sys.stdin.buffer.read() if not sys.stdin.isatty() else b""

t_init0 = time.perf_counter()
ql = Qiling([binary, *binary_args], rootfs, verbose=QL_VERBOSE.OFF)
if stdin_data:
    try:
        ql.os.stdin = io.BytesIO(stdin_data)
    except Exception:
        pass
reporter = Reporter(json_mode=False, stream=sys.stderr)
wrapper = MicrotaintWrapper(
    ql,
    check_bof=("bof" in flags),
    check_uaf=("uaf" in flags),
    check_sc=("sc" in flags),
    check_aiw=("aiw" in flags),
    reporter=reporter,
)
# microtaint marks stdin bytes as tainted via the BytesIO wrapper; in the
# CLI this is done by _make_stdin_stream returning a tainting stream.
# For simplicity we just rely on the read syscall hook to taint everything
# that comes back from fd 0 — that's the default behaviour when no
# --input flag is provided either.

if "uaf" in flags:
    HeapTracker(ql, wrapper.shadow_mem).install()
t_init1 = time.perf_counter()

_tf, _saved = _capture_fd1()
t_run0 = time.perf_counter()
try:
    ql.run()
except Exception:
    pass
t_run1 = time.perf_counter()
guest_bytes = _release_fd1(_tf, _saved)

extra = {}
# _instr_hook_registered is the second half of the vacuous-run check: the
# wrapper installs the per-instruction hook LAZILY, only once taint exists, so
# a run with no tainted input reports False and costs nothing.  A "fast"
# microtaint number with this False is not a fast engine, it is no engine.
for attr in ("_instr_cache_hits", "_instr_cache_misses", "_instr_hook_registered"):
    if hasattr(wrapper, attr):
        extra[attr.lstrip("_")] = getattr(wrapper, attr)

# Which engine path actually ran.  taint_ir is later work and is out of scope
# for these artifacts, but it is the engine's DEFAULT, so "we did not ask for it"
# is not evidence -- report what was imported and let the harness check.
taint_ir_used = sorted(m for m in sys.modules if m.startswith("microtaint.taint_ir"))

print(json.dumps({
    "import_s": t_import1 - t_import0,
    "init_s":   t_init1 - t_init0,
    "run_s":    t_run1 - t_run0,
    "guest_bytes": guest_bytes,
    "stdin_bytes": len(stdin_data),
    "taint_ir_modules": taint_ir_used,
    "env_taint_ir": os.environ.get("MICROTAINT_TAINT_IR", "<unset>"),
    "env_block": os.environ.get("MICROTAINT_BLOCK", "<unset>"),
    "wrapper_stats": extra,
}))
"""


def measure_microtaint(
    label: str,
    binary: str,
    binary_args: list[str],
    rootfs: str,
    flags: set[str],
    stdin_data: bytes | None,
    timeout: float = 1800.0,
) -> Measurement:
    """Run microtaint with given detection flags and time each phase."""
    helper_path = '/tmp/_overhead_microtaint.py'
    with open(helper_path, 'w') as f:
        f.write(MICROTAINT_HELPER)
    flag_string = ','.join(sorted(flags))
    return _run_helper_subprocess(
        label=label,
        argv=[sys.executable, helper_path, binary, rootfs, flag_string, *binary_args],
        stdin_data=stdin_data,
        timeout=timeout,
        env=ARTIFACT_ENGINE_ENV,
    )



# ---------------------------------------------------------------------------
# Guest instruction count (untimed)
# ---------------------------------------------------------------------------

COUNT_HELPER = r"""
import json, sys, time, io
binary = sys.argv[1]
rootfs = sys.argv[2]
binary_args = sys.argv[3:]
from qiling import Qiling
from qiling.const import QL_VERBOSE
stdin_data = sys.stdin.buffer.read() if not sys.stdin.isatty() else b""
ql = Qiling([binary, *binary_args], rootfs, verbose=QL_VERBOSE.OFF)
if stdin_data:
    try:
        ql.os.stdin = io.BytesIO(stdin_data)
    except Exception:
        pass
n = [0]
ql.hook_code(lambda *a: n.__setitem__(0, n[0] + 1))
try:
    ql.run()
except Exception:
    pass
sys.stderr.write(json.dumps({"instrs": n[0]}))
"""


def count_guest_instructions(
    binary: str, binary_args: list[str], rootfs: str, stdin_data: bytes | None,
    timeout: float = 1800.0,
) -> int | None:
    """Guest instructions executed, counted once under a Python code hook.

    Deliberately NOT part of any timed configuration: the hook itself costs
    ~2.2 us per instruction (about a hundred times bare Qiling), so counting and
    timing in the same run would measure the counter. It is reported separately
    so every run_s can be divided into ns/instr, which is what made the old
    numbers falsifiable: `microtaint-all` at 67.7 ms over 3.77 M instructions is
    18 ns/instr, below bare Unicorn's own per-instruction cost, so it could not
    have been running a hook at all.

    Writes to stderr, so the guest's own stdout cannot corrupt the JSON.
    """
    helper_path = '/tmp/_overhead_count.py'
    with open(helper_path, 'w') as f:
        f.write(COUNT_HELPER)
    m = _run_helper_subprocess(
        label='instr-count',
        argv=[sys.executable, helper_path, binary, rootfs, *binary_args],
        stdin_data=stdin_data,
        timeout=timeout,
    )
    tail = m.extra.get('stderr_tail') or ''
    idx = tail.rfind('{')
    if idx < 0:
        return None
    try:
        return int(json.loads(tail[idx:])['instrs'])
    except (ValueError, KeyError):
        return None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(measurements: list[Measurement]) -> Measurement:
    """Combine N measurements of the same configuration into a median row."""
    if not measurements:
        raise ValueError('no measurements to aggregate')
    if len(measurements) == 1:
        return measurements[0]
    label = measurements[0].label
    return Measurement(
        label=label,
        wall_s=statistics.median(m.wall_s for m in measurements),
        user_cpu_s=statistics.median(m.user_cpu_s for m in measurements),
        sys_cpu_s=statistics.median(m.sys_cpu_s for m in measurements),
        peak_rss_mib=statistics.median(m.peak_rss_mib for m in measurements),
        extra={
            'returncode': measurements[0].extra.get('returncode'),
            'n_runs': len(measurements),
            'wall_s_min': min(m.wall_s for m in measurements),
            'wall_s_max': max(m.wall_s for m in measurements),
            'import_s': _median_or_none(m.extra.get('import_s') for m in measurements),
            'init_s': _median_or_none(m.extra.get('init_s') for m in measurements),
            'run_s': _median_or_none(m.extra.get('run_s') for m in measurements),
            'run_s_min': _min_or_none(m.extra.get('run_s') for m in measurements),
            'run_s_max': _max_or_none(m.extra.get('run_s') for m in measurements),
            # Workload-ran proof, kept per aggregate so the JSON carries it too.
            'guest_bytes': _min_or_none(m.extra.get('guest_bytes') for m in measurements),
            'stdin_bytes': measurements[0].extra.get('stdin_bytes'),
            'wrapper_stats': measurements[0].extra.get('wrapper_stats', {}),
            # Which engine path ran.  Carried into the aggregate deliberately: the
            # per-run validation already refused a taint_ir run, but a reader of the
            # JSON months later has only the JSON, and "it must have been fine
            # because the harness checked" is exactly the reasoning that let the
            # 2026-05 numbers stand.
            'env_taint_ir': measurements[0].extra.get('env_taint_ir'),
            'env_block': measurements[0].extra.get('env_block'),
            'taint_ir_modules': measurements[0].extra.get('taint_ir_modules'),
        },
    )


def _median_or_none(it):
    vals = [v for v in it if v is not None]
    return statistics.median(vals) if vals else None


def _min_or_none(it):
    vals = [v for v in it if v is not None]
    return min(vals) if vals else None


def _max_or_none(it):
    vals = [v for v in it if v is not None]
    return max(vals) if vals else None


class VacuousRunError(RuntimeError):
    """The guest did not execute the workload, so the timing means nothing.

    This exists because its absence cost a published result.  `bench.c` opens
    with ``n = sys_read(0, state, 256); if (n <= 0) sys_exit(1);`` and the
    helpers wrap ``ql.run()`` in ``except Exception: pass``, so with no stdin the
    guest quit at the first check, every config recorded a clean sub-100 ms
    ``run_s``, and the ratio of two process-startup paths was published as the
    end-to-end taint overhead.  Nothing in the harness objected.

    The check is the guest's own output: `bench.c` writes its 8-byte hash after
    the mix rounds and before the overflow, so a non-empty count proves the
    workload ran.  Byte counts, not wall times -- a vacuous run is FAST, which
    is exactly why a timing threshold would never have caught it.
    """


class WrongEnginePathError(RuntimeError):
    """The run exercised taint_ir rather than the per-instruction circuit path.

    taint_ir is later work and out of scope for these artifacts, but it is the
    engine's default (`taint_ir/engine_glue`: "This is the hot path's default"),
    so simply not asking for it measures it anyway.  The first corrected RQ5 run
    on 2026-09-11 did exactly that, and only reading engine_glue.enabled() caught
    it: nothing in the output looked wrong.
    """


def validate_not_vacuous(m: Measurement, min_guest_bytes: int) -> None:
    """Raise unless this measurement shows the guest actually ran the workload."""
    got = m.extra.get('guest_bytes')
    if got is None:
        raise VacuousRunError(
            f'{m.label}: helper reported no guest_bytes; cannot prove the workload ran')
    if got < min_guest_bytes:
        stdin_n = m.extra.get('stdin_bytes')
        raise VacuousRunError(
            f'{m.label}: guest wrote {got} bytes, expected at least {min_guest_bytes}. '
            f'The workload did not run (stdin was {stdin_n} bytes). '
            f'Pass --gen-input/--stdin-file, or lower --min-guest-bytes if this '
            f'binary legitimately writes less.')
    stats = m.extra.get('wrapper_stats') or {}
    if 'instr_hook_registered' in stats and not stats['instr_hook_registered']:
        raise VacuousRunError(
            f'{m.label}: the per-instruction hook was never registered, so no taint '
            f'was propagated. microtaint installs it lazily, only once taint exists, '
            f'so this timing measures an uninstrumented run.')
    env_ir = m.extra.get('env_taint_ir')
    if env_ir is not None and env_ir not in ('0', '<unset>'):
        raise WrongEnginePathError(
            f'{m.label}: MICROTAINT_TAINT_IR={env_ir}. taint_ir replaces the '
            f'per-instruction circuit evaluation this artifact measures, and it is the '
            f'engine DEFAULT, so a run that does not pin it off describes later work '
            f'instead of this paper.')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CONFIGS = [
    ('microtaint-none', set()),  # taint propagation only, no checks
    ('microtaint-bof', {'bof'}),
    ('microtaint-uaf', {'uaf'}),
    ('microtaint-sc', {'sc'}),
    ('microtaint-aiw', {'aiw'}),
    ('microtaint-all', {'bof', 'uaf', 'sc', 'aiw'}),
]



def _run_rounds_sweep(args, parser) -> int:
    """Run the whole configuration set once per -DROUNDS value.

    A single workload size gives one point and no way to tell a fixed cost from a
    per-instruction one: startup, JIT warm-up and rule synthesis are all paid once,
    while propagation scales with instructions executed.  Sweeping ROUNDS separates
    them, and the instruction count measured at each size is the x-axis.

    Each size is a full independent run, including its own instruction count and
    its own workload-ran check, and is written to `<json>` keyed by size.
    """
    sizes = [int(x) for x in str(args.rounds_sweep).split(',') if x.strip()]
    if not sizes:
        parser.error('--rounds-sweep needs at least one integer')
    out_path = args.json
    series: dict[str, object] = {}
    for n in sizes:
        sys.stderr.write(f'\n===== ROUNDS={n} =====\n')
        binary = build_bench(args.build_bench, out_path=f'bench_r{n}.elf',
                             extra_flags=[f'-DROUNDS={n}'])
        sub = argparse.Namespace(**vars(args))
        sub.rounds_sweep = None
        sub.build_bench = None
        sub.binary = binary
        sub.instr_count = True          # the sweep is meaningless without the x-axis
        sub.json = f'{out_path}.rounds{n}.json' if out_path else None
        rc = _main_single(sub, parser)
        if rc != 0:
            return rc
        if sub.json:
            with open(sub.json) as f:
                series[str(n)] = json.load(f)
    if out_path:
        with open(out_path, 'w') as f:
            json.dump({'rounds_sweep': series}, f, indent=2)
        print(f'\nSweep written to {out_path}')
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description='Measure CPU/memory overhead of bench.elf under qiling and microtaint',
        usage='%(prog)s [OPTIONS] [BINARY] [-- BINARY_ARGS...]',
    )
    p.add_argument('--runs', type=int, default=1, help='Number of runs per configuration (median reported)')
    p.add_argument('--rootfs', default='/', help='Qiling rootfs directory (default: /)')
    p.add_argument('--stdin-file', default=None, help="File whose contents are piped to the binary's stdin")
    p.add_argument(
        '--skip', action='append', default=[], help='Skip a configuration (e.g. --skip native, --skip microtaint-all)',
    )
    p.add_argument('--only', action='append', default=[], help='Run ONLY these configurations (repeatable)')
    p.add_argument('--json', default=None, help='Write full results as JSON to this path')
    p.add_argument(
        '--native-timeout',
        type=float,
        default=10.0,
        help='Timeout (s) for native runs — bench.elf with 256B input crashes/hangs after BOF, so cap it',
    )
    p.add_argument('--qiling-timeout', type=float, default=300.0, help='Timeout (s) for qiling-only runs')
    p.add_argument(
        '--microtaint-timeout', type=float, default=1800.0, help='Timeout (s) for microtaint runs (default: 30 min)',
    )
    p.add_argument('--build-bench', metavar='BENCH_C', help='Compile bench.c into bench.elf and use it as the target')
    p.add_argument(
        '--gen-input',
        type=int,
        metavar='SIZE',
        help='Generate a deterministic SIZE-byte stdin input (use 256 to trigger the BOF, 64 for clean exit)',
    )
    p.add_argument('--input-seed', type=int, default=0xC0FFEE, help='Seed for --gen-input (default: 0xC0FFEE)')
    p.add_argument(
        '--min-guest-bytes', type=int, default=1,
        help='Fail a configuration whose guest wrote fewer bytes than this. bench.c writes its '
             '8-byte hash after the mix rounds, so any non-zero count proves the workload ran; '
             '0 disables the check (NOT recommended, see VacuousRunError).',
    )
    p.add_argument(
        '--allow-vacuous', action='store_true',
        help='Record a run that cannot be shown to have executed the workload instead of failing. '
             'Only for debugging the harness itself; results are not publishable.',
    )
    p.add_argument(
        '--rounds-sweep', default=None, metavar='N,N,...',
        help='Rebuild bench.c at each -DROUNDS value and run the whole config set for each, '
             'giving cost as a function of workload size rather than a single point. '
             'Requires --build-bench. Implies --instr-count, since the whole purpose is to '
             'relate time to instructions executed.',
    )
    p.add_argument(
        '--instr-count', action='store_true',
        help='Additionally run the guest once under a counting hook and report guest instructions '
             'executed, so every timing can be expressed per propagation step (ns/instr).',
    )
    p.add_argument('binary', nargs='?', help='Path to bench.elf (omit if --build-bench is given)')
    p.add_argument('binary_args', nargs=argparse.REMAINDER, help='Arguments to pass to bench.elf (separate with --)')
    args = p.parse_args()
    # Every rung is compared against the others, so they must share a clock.
    _cpu_state = require_no_cpu_boost()

    # Resolve binary path: either use --build-bench result or the positional arg
    if args.rounds_sweep and not args.build_bench:
        p.error('--rounds-sweep needs --build-bench: it recompiles bench.c at each -DROUNDS')

    if args.rounds_sweep:
        return _run_rounds_sweep(args, p)

    return _main_single(args, p)


def _main_single(args, p) -> int:
    """One complete measurement: build/resolve the binary, run every configuration,
    print the table, write the JSON.  Factored out of main() so --rounds-sweep can
    call it once per workload size."""
    if args.build_bench:
        binary = build_bench(args.build_bench)
    elif args.binary:
        binary = os.path.abspath(args.binary)
    else:
        p.error('either BINARY or --build-bench must be provided')
    if not os.path.isfile(binary):
        print(f'error: binary not found: {binary}', file=sys.stderr)
        return 2

    binary_args = list(args.binary_args)
    if binary_args and binary_args[0] == '--':
        binary_args = binary_args[1:]

    # Resolve stdin: explicit --stdin-file wins, else --gen-input creates one
    stdin_path: str | None = args.stdin_file
    if stdin_path is None and args.gen_input is not None:
        stdin_path = gen_input(args.gen_input, args.input_seed)
    stdin_data = None
    if stdin_path:
        with open(stdin_path, 'rb') as stdin_f:
            stdin_data = stdin_f.read()
        print(f'# stdin source: {stdin_path} ({len(stdin_data)} bytes)')

    # Build config list
    all_labels = ['native', 'qiling-only'] + [c[0] for c in CONFIGS]
    if args.only:
        labels_to_run = [label for label in all_labels if label in args.only]
    else:
        labels_to_run = [label for label in all_labels if label not in args.skip]

    if not stdin_data and args.min_guest_bytes > 0 and not args.allow_vacuous:
        raise SystemExit(
            'refusing to run with no stdin: bench.c exits at its first read when stdin is '
            'empty, which produces a fast, clean, and completely meaningless measurement '
            '(that is how the published RQ5 numbers were made). Pass --gen-input 64 for a '
            'clean-exit workload, --gen-input 256 to also trigger the BOF, or --stdin-file. '
            'Use --allow-vacuous only to debug the harness.',
        )

    n_instrs: int | None = None
    if args.instr_count:
        sys.stderr.write('[instr-count] counting guest instructions (untimed)…\n')
        n_instrs = count_guest_instructions(
            binary, binary_args, args.rootfs, stdin_data, timeout=args.microtaint_timeout)
        print(f'# Guest instructions executed: {n_instrs if n_instrs is not None else "unknown"}')

    print(f'# Bench: {binary} {" ".join(binary_args)}')
    print(f'# Configurations: {labels_to_run}')
    print(f'# Runs per config: {args.runs}')
    print(f'# Rootfs: {args.rootfs}')
    print()

    all_results: dict[str, Measurement] = {}

    for label in labels_to_run:
        runs: list[Measurement] = []
        for i in range(args.runs):
            sys.stderr.write(f'[{label} {i + 1}/{args.runs}] running…\n')
            sys.stderr.flush()
            try:
                if label == 'native':
                    m = measure_native(binary, binary_args, stdin_data, timeout=args.native_timeout)
                elif label == 'qiling-only':
                    m = measure_qiling_only(binary, binary_args, args.rootfs, stdin_data, timeout=args.qiling_timeout)
                else:
                    cfg = next(c for c in CONFIGS if c[0] == label)
                    m = measure_microtaint(
                        label, binary, binary_args, args.rootfs, cfg[1], stdin_data, timeout=args.microtaint_timeout,
                    )
                if args.min_guest_bytes > 0:
                    try:
                        validate_not_vacuous(m, args.min_guest_bytes)
                    except VacuousRunError as exc:
                        if not args.allow_vacuous:
                            raise
                        sys.stderr.write(f'  ! VACUOUS (recorded anyway): {exc}\n')
                        m.extra['vacuous'] = str(exc)
                runs.append(m)
            except (VacuousRunError, WrongEnginePathError):
                # --allow-vacuous does NOT cover WrongEnginePathError: that one is
                # not about whether the workload ran but about which engine ran,
                # and there is no debugging reason to publish the wrong one.
                raise
            except RuntimeError as exc:
                # Timeout from a single run — log and continue
                sys.stderr.write(f'  ! {exc}  (counted as one timed-out run)\n')
                continue
            except Exception as exc:
                sys.stderr.write(f'  ! failed: {exc}\n')
                continue
        if runs:
            all_results[label] = aggregate(runs)
        else:
            sys.stderr.write(f'  (all runs of {label} failed)\n')

    # ----- Print table ---------------------------------------------------
    print()
    print('=' * 116)
    print(
        f'{"Configuration":<22} {"wall (s)":>9} {"peak RSS":>10} {"qil init":>9} '
        f'{"ql.run":>9} {"ns/instr":>10} {"×qiling":>9} {"×native":>9} {"guest B":>8}',
    )
    print('-' * 116)

    native_wall = all_results.get('native', Measurement('native', 0, 0, 0, 0)).wall_s
    _qil = all_results.get('qiling-only')
    qiling_run = _qil.extra.get('run_s') if _qil else None

    for label in labels_to_run:
        found = all_results.get(label)
        if not found:
            continue
        m = found
        init_s = m.extra.get('init_s')
        run_s = m.extra.get('run_s')
        gb = m.extra.get('guest_bytes')
        ratio_n = (m.wall_s / native_wall) if native_wall > 0 else None
        # vs qiling-only on ql.run(), NOT on wall: wall is dominated by Python
        # import and Qiling init, which have nothing to do with taint and dilute
        # the ratio by ~5x. This is the number the README always meant.
        ratio_q = (run_s / qiling_run) if (run_s and qiling_run) else None
        ns_per = (run_s * 1e9 / n_instrs) if (run_s and n_instrs) else None
        def _f(v, w, prec=1, suffix=''):
            return f'{v:>{w}.{prec}f}{suffix}' if v is not None else f'{"—":>{w}}'
        print(f'{label:<22} {m.wall_s:>9.3f} {m.peak_rss_mib:>8.1f} M {_f(init_s, 9, 3)} '
              f'{_f(run_s, 9, 3)} {_f(ns_per, 10)} {_f(ratio_q, 9)} {_f(ratio_n, 9)} '
              f'{(gb if gb is not None else "—"):>8}')

    print('=' * 116)
    print()
    print('Columns:')
    print('  ns/instr  : ql.run() divided by guest instructions (needs --instr-count)')
    print('  ×qiling   : ql.run() over qiling-only ql.run() — the taint cost, emulator divided out')
    print('  guest B   : bytes the GUEST wrote; proves the workload ran (see VacuousRunError)')
    print('  wall (s)  : total wall-clock time of the subprocess')
    print('  CPU  (s)  : user_cpu + sys_cpu of the subprocess (children rusage)')
    print('  peak RSS  : max resident set size of the subprocess, MiB')
    print('  qil init  : qiling/wrapper construction time (in-process measurement)')
    print('  ql.run    : ql.run() time, i.e. taint-propagation phase')
    print('  ×native   : ratio of wall time to the native run')
    print()
    print('Note:  qil init + ql.run < wall  →  the difference is python startup,')
    print('       module imports, and shutdown / report finalisation.')

    # ----- Save JSON if requested ---------------------------------------
    if args.json:
        out_dict = {label: asdict(m) for label, m in all_results.items()}
        # `_meta` carries what a plot needs and what a reader needs to trust the
        # numbers: the workload size (so any run_s can be checked against the
        # ~22 ns/instr floor of bare Unicorn), the stdin that produced it, and
        # the derived ratios, computed once here rather than re-derived by every
        # consumer with a different idea of which denominator to use.
        out_dict['_meta'] = {
            'binary': binary,
            'binary_args': binary_args,
            'guest_instructions': n_instrs,
            'stdin_bytes': len(stdin_data) if stdin_data else 0,
            'stdin_source': stdin_path,
            'engine_path': 'per-instruction LogicCircuit (taint_ir and block mode pinned off)',
            'artifact_engine_env': ARTIFACT_ENGINE_ENV,
            'runs_per_config': args.runs,
            'rootfs': args.rootfs,
            'min_guest_bytes': args.min_guest_bytes,
            'derived': {
                label: {
                    'run_s': m.extra.get('run_s'),
                    'ns_per_instr': (m.extra['run_s'] * 1e9 / n_instrs)
                    if (m.extra.get('run_s') and n_instrs) else None,
                    'x_qiling_run': (m.extra['run_s'] / qiling_run)
                    if (m.extra.get('run_s') and qiling_run) else None,
                    'x_native_wall': (m.wall_s / native_wall) if native_wall > 0 else None,
                    'guest_bytes': m.extra.get('guest_bytes'),
                }
                for label, m in all_results.items()
            },
        }
        with open(args.json, 'w') as f:
            json.dump({'engine': _engine_provenance(), 'cpu': require_no_cpu_boost(),
                       **out_dict}, f, indent=2)
        print(f'\nFull results written to {args.json}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
