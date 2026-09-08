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
from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return 0
    return 0


def _bytes_to_mib(b: int) -> float:
    return b / (1024 * 1024)


# ---------------------------------------------------------------------------
# Bench source build + input generation helpers
# ---------------------------------------------------------------------------

def build_bench(src_path: str, out_path: str | None = None,
                cc: str = "gcc",
                extra_flags: list[str] | None = None) -> str:
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
        out_path = os.path.join(os.getcwd(), "bench.elf")
    out_path = os.path.abspath(out_path)
    cmd = [
        cc,
        "-O0",
        "-static",
        "-nostdlib",
        "-fno-pie",
        "-no-pie",
        # Some distros (Ubuntu, Debian, Arch) enable -fstack-protector-strong
        # by default at the gcc spec level, which makes the compiler emit
        # __stack_chk_fail calls in any function with a stack array.  With
        # -nostdlib there's no libc to satisfy that symbol → link error.
        # The benchmark deliberately overflows a stack buffer in
        # unsafe_copy(), so the canary would defeat the whole point of the
        # BOF demonstration anyway.
        "-fno-stack-protector",
        # Same story for FORTIFY_SOURCE: enabled by default on some
        # distros, requires libc, and would replace the unbounded copy
        # loop with a bounded one.
        "-U_FORTIFY_SOURCE",
        "-D_FORTIFY_SOURCE=0",
        # Ensure no-pie binary even when the spec file forces -pie.
        "-fno-stack-clash-protection",
        os.path.abspath(src_path),
        "-o",
        out_path,
    ]
    if extra_flags:
        cmd.extend(extra_flags)
    print(f"# building: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        raise RuntimeError(f"compilation failed (exit {r.returncode})")
    return out_path


def gen_input(size: int, seed: int = 0xC0FFEE,
              out_path: str | None = None) -> str:
    """
    Generate a deterministic ``size``-byte stdin file.

    Deterministic so timing comparisons across runs are reproducible —
    the same input bytes mean the same propagation graph, the same
    cache hit rate, the same SBOX lookup pattern.
    """
    import random
    if out_path is None:
        out_path = f"/tmp/bench_input_{size}_{seed:08x}.bin"
    rng = random.Random(seed)
    data = bytes(rng.randrange(256) for _ in range(size))
    with open(out_path, "wb") as f:
        f.write(data)
    print(f"# wrote {size} byte deterministic input → {out_path}")
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


def _run_subprocess(label: str, argv: list[str], env: dict | None = None,
                    stdin_data: bytes | None = None,
                    timeout: float = 600.0) -> Measurement:
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
            try: stream.close()
            except Exception: pass
    t_out = threading.Thread(target=_drain, args=(proc.stdout, out_chunks), daemon=True)
    t_err = threading.Thread(target=_drain, args=(proc.stderr, err_chunks), daemon=True)
    t_out.start(); t_err.start()

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
                raise RuntimeError(f"{label}: timeout after {timeout}s")
            time.sleep(0.001)
    finally:
        poller_stop.set()
        poller.join(timeout=1)
        t_out.join(timeout=2)
        t_err.join(timeout=2)

    t1 = time.perf_counter()
    out = b"".join(out_chunks)
    err = b"".join(err_chunks)

    user_cpu = rusage_child.ru_utime if rusage_child else 0.0
    sys_cpu = rusage_child.ru_stime if rusage_child else 0.0
    rusage_peak_kb = (rusage_child.ru_maxrss if rusage_child else 0)
    peak_kb = max(rusage_peak_kb, polled_peak_kb[0])

    return Measurement(
        label=label,
        wall_s=t1 - t0,
        user_cpu_s=user_cpu,
        sys_cpu_s=sys_cpu,
        peak_rss_mib=peak_kb / 1024.0,
        extra={
            "returncode": proc.returncode,
            "stdout_size": len(out),
            "stderr_size": len(err),
        },
    )


def _run_helper_subprocess(label: str, argv: list[str], stdin_data: bytes | None,
                            timeout: float = 1800.0) -> Measurement:
    """Run a helper subprocess that emits a JSON timing line on stdout.
    Same accounting as _run_subprocess plus parses the breakdown JSON."""
    import threading

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
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
        try: proc.stdin.write(stdin_data)
        except BrokenPipeError: pass
        proc.stdin.close()

    def _drain(stream, into):
        try: into.append(stream.read())
        except Exception: pass
        finally:
            try: stream.close()
            except Exception: pass
    t_out = threading.Thread(target=_drain, args=(proc.stdout, out_chunks), daemon=True)
    t_err = threading.Thread(target=_drain, args=(proc.stderr, err_chunks), daemon=True)
    t_out.start(); t_err.start()

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
                raise RuntimeError(f"{label}: timeout after {timeout}s")
            time.sleep(0.001)
    finally:
        poller_stop.set()
        poller.join(timeout=1)
        t_out.join(timeout=2)
        t_err.join(timeout=2)

    t1 = time.perf_counter()
    out = b"".join(out_chunks)
    err = b"".join(err_chunks)

    user_cpu = rusage_child.ru_utime if rusage_child else 0.0
    sys_cpu = rusage_child.ru_stime if rusage_child else 0.0
    rusage_peak_kb = (rusage_child.ru_maxrss if rusage_child else 0)
    peak_kb = max(rusage_peak_kb, polled_peak_kb[0])

    breakdown: dict = {}
    # The bench may emit binary data on stdout (the bench.elf writes 8 raw
    # hash bytes), so simple splitlines() can fail to isolate the helper's
    # JSON line.  We look for the LAST balanced `{...}` JSON object in the
    # text, scanning backwards from the end.
    text = out.decode("utf-8", errors="replace")
    end_idx = text.rfind("}")
    while end_idx > 0:
        # find the matching opening brace by depth-tracking from end_idx
        depth = 0
        start_idx = -1
        for i in range(end_idx, -1, -1):
            c = text[i]
            if c == "}":
                depth += 1
            elif c == "{":
                depth -= 1
                if depth == 0:
                    start_idx = i
                    break
        if start_idx < 0:
            break
        candidate = text[start_idx:end_idx + 1]
        try:
            breakdown = json.loads(candidate)
            break
        except json.JSONDecodeError:
            # try a shorter window (move end_idx backwards past this `}`)
            end_idx = text.rfind("}", 0, end_idx)

    return Measurement(
        label=label,
        wall_s=t1 - t0,
        user_cpu_s=user_cpu,
        sys_cpu_s=sys_cpu,
        peak_rss_mib=peak_kb / 1024.0,
        extra={
            "returncode": proc.returncode,
            "import_s": breakdown.get("import_s"),
            "init_s": breakdown.get("init_s"),
            "run_s": breakdown.get("run_s"),
            "wrapper_stats": breakdown.get("wrapper_stats", {}),
            "stderr_tail": err.decode("utf-8", errors="replace")[-400:] if proc.returncode else None,
        },
    )


# ---------------------------------------------------------------------------
# Measurement: native baseline
# ---------------------------------------------------------------------------

def measure_native(binary: str, binary_args: list[str], stdin_data: bytes | None,
                    timeout: float = 600.0) -> Measurement:
    """Run the binary directly with no instrumentation."""
    return _run_subprocess(
        label="native",
        argv=[binary, *binary_args],
        stdin_data=stdin_data,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Measurement: qiling-only (no microtaint instrumentation)
# ---------------------------------------------------------------------------

QILING_ONLY_HELPER = r'''
import json, sys, time, io
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

t_run0 = time.perf_counter()
try:
    ql.run()
except Exception:
    pass
t_run1 = time.perf_counter()

print(json.dumps({
    "import_s": t_import1 - t_import0,
    "init_s":   t_init1 - t_init0,
    "run_s":    t_run1 - t_run0,
}))
'''


def measure_qiling_only(binary: str, binary_args: list[str], rootfs: str,
                         stdin_data: bytes | None,
                         timeout: float = 600.0) -> Measurement:
    """Run the binary inside a fresh Qiling with NO microtaint hooks."""
    helper_path = "/tmp/_overhead_qiling_only.py"
    with open(helper_path, "w") as f:
        f.write(QILING_ONLY_HELPER)
    return _run_helper_subprocess(
        label="qiling-only",
        argv=[sys.executable, helper_path, binary, rootfs, *binary_args],
        stdin_data=stdin_data,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Measurement: microtaint with given check flags
# ---------------------------------------------------------------------------

MICROTAINT_HELPER = r'''
import json, os, sys, time, io

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

t_run0 = time.perf_counter()
try:
    ql.run()
except Exception:
    pass
t_run1 = time.perf_counter()

extra = {}
for attr in ("_instr_cache_hits", "_instr_cache_misses"):
    if hasattr(wrapper, attr):
        extra[attr.lstrip("_")] = getattr(wrapper, attr)

print(json.dumps({
    "import_s": t_import1 - t_import0,
    "init_s":   t_init1 - t_init0,
    "run_s":    t_run1 - t_run0,
    "wrapper_stats": extra,
}))
'''


def measure_microtaint(label: str, binary: str, binary_args: list[str], rootfs: str,
                        flags: set[str], stdin_data: bytes | None,
                        timeout: float = 1800.0) -> Measurement:
    """Run microtaint with given detection flags and time each phase."""
    helper_path = "/tmp/_overhead_microtaint.py"
    with open(helper_path, "w") as f:
        f.write(MICROTAINT_HELPER)
    flag_string = ",".join(sorted(flags))
    return _run_helper_subprocess(
        label=label,
        argv=[sys.executable, helper_path, binary, rootfs, flag_string, *binary_args],
        stdin_data=stdin_data,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(measurements: list[Measurement]) -> Measurement:
    """Combine N measurements of the same configuration into a median row."""
    if not measurements:
        raise ValueError("no measurements to aggregate")
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
            "returncode": measurements[0].extra.get("returncode"),
            "n_runs": len(measurements),
            "wall_s_min": min(m.wall_s for m in measurements),
            "wall_s_max": max(m.wall_s for m in measurements),
            "import_s": _median_or_none(m.extra.get("import_s") for m in measurements),
            "init_s": _median_or_none(m.extra.get("init_s") for m in measurements),
            "run_s": _median_or_none(m.extra.get("run_s") for m in measurements),
        },
    )


def _median_or_none(it):
    vals = [v for v in it if v is not None]
    return statistics.median(vals) if vals else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CONFIGS = [
    ("microtaint-none",  set()),                # taint propagation only, no checks
    ("microtaint-bof",   {"bof"}),
    ("microtaint-uaf",   {"uaf"}),
    ("microtaint-sc",    {"sc"}),
    ("microtaint-aiw",   {"aiw"}),
    ("microtaint-all",   {"bof", "uaf", "sc", "aiw"}),
]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Measure CPU/memory overhead of bench.elf under qiling and microtaint",
        usage="%(prog)s [OPTIONS] [BINARY] [-- BINARY_ARGS...]")
    p.add_argument("--runs", type=int, default=1,
                   help="Number of runs per configuration (median reported)")
    p.add_argument("--rootfs", default="/",
                   help="Qiling rootfs directory (default: /)")
    p.add_argument("--stdin-file", default=None,
                   help="File whose contents are piped to the binary's stdin")
    p.add_argument("--skip", action="append", default=[],
                   help="Skip a configuration (e.g. --skip native, --skip microtaint-all)")
    p.add_argument("--only", action="append", default=[],
                   help="Run ONLY these configurations (repeatable)")
    p.add_argument("--json", default=None,
                   help="Write full results as JSON to this path")
    p.add_argument("--native-timeout", type=float, default=10.0,
                   help="Timeout (s) for native runs — bench.elf with 256B "
                        "input crashes/hangs after BOF, so cap it")
    p.add_argument("--qiling-timeout", type=float, default=300.0,
                   help="Timeout (s) for qiling-only runs")
    p.add_argument("--microtaint-timeout", type=float, default=1800.0,
                   help="Timeout (s) for microtaint runs (default: 30 min)")
    p.add_argument("--build-bench", metavar="BENCH_C",
                   help="Compile bench.c into bench.elf and use it as the target")
    p.add_argument("--gen-input", type=int, metavar="SIZE",
                   help="Generate a deterministic SIZE-byte stdin input "
                        "(use 256 to trigger the BOF, 64 for clean exit)")
    p.add_argument("--input-seed", type=int, default=0xC0FFEE,
                   help="Seed for --gen-input (default: 0xC0FFEE)")
    p.add_argument("binary", nargs="?",
                   help="Path to bench.elf (omit if --build-bench is given)")
    p.add_argument("binary_args", nargs=argparse.REMAINDER,
                   help="Arguments to pass to bench.elf (separate with --)")
    args = p.parse_args()

    # Resolve binary path: either use --build-bench result or the positional arg
    if args.build_bench:
        binary = build_bench(args.build_bench)
    elif args.binary:
        binary = os.path.abspath(args.binary)
    else:
        p.error("either BINARY or --build-bench must be provided")
    if not os.path.isfile(binary):
        print(f"error: binary not found: {binary}", file=sys.stderr)
        return 2

    binary_args = list(args.binary_args)
    if binary_args and binary_args[0] == "--":
        binary_args = binary_args[1:]

    # Resolve stdin: explicit --stdin-file wins, else --gen-input creates one
    stdin_path: str | None = args.stdin_file
    if stdin_path is None and args.gen_input is not None:
        stdin_path = gen_input(args.gen_input, args.input_seed)
    stdin_data = None
    if stdin_path:
        with open(stdin_path, "rb") as f:
            stdin_data = f.read()
        print(f"# stdin source: {stdin_path} ({len(stdin_data)} bytes)")

    # Build config list
    all_labels = ["native", "qiling-only"] + [c[0] for c in CONFIGS]
    if args.only:
        labels_to_run = [l for l in all_labels if l in args.only]
    else:
        labels_to_run = [l for l in all_labels if l not in args.skip]

    print(f"# Bench: {binary} {' '.join(binary_args)}")
    print(f"# Configurations: {labels_to_run}")
    print(f"# Runs per config: {args.runs}")
    print(f"# Rootfs: {args.rootfs}")
    print()

    all_results: dict[str, Measurement] = {}

    for label in labels_to_run:
        runs: list[Measurement] = []
        for i in range(args.runs):
            sys.stderr.write(f"[{label} {i+1}/{args.runs}] running…\n")
            sys.stderr.flush()
            try:
                if label == "native":
                    m = measure_native(binary, binary_args, stdin_data,
                                        timeout=args.native_timeout)
                elif label == "qiling-only":
                    m = measure_qiling_only(binary, binary_args, args.rootfs,
                                             stdin_data, timeout=args.qiling_timeout)
                else:
                    cfg = next(c for c in CONFIGS if c[0] == label)
                    m = measure_microtaint(label, binary, binary_args, args.rootfs,
                                            cfg[1], stdin_data,
                                            timeout=args.microtaint_timeout)
                runs.append(m)
            except RuntimeError as exc:
                # Timeout from a single run — log and continue
                sys.stderr.write(f"  ! {exc}  (counted as one timed-out run)\n")
                continue
            except Exception as exc:
                sys.stderr.write(f"  ! failed: {exc}\n")
                continue
        if runs:
            all_results[label] = aggregate(runs)
        else:
            sys.stderr.write(f"  (all runs of {label} failed)\n")

    # ----- Print table ---------------------------------------------------
    print()
    print("=" * 96)
    print(f"{'Configuration':<22} {'wall (s)':>10} {'CPU (s)':>10} {'peak RSS':>10} "
          f"{'qil init':>10} {'ql.run':>10} {'×native':>8}")
    print("-" * 96)

    native_wall = all_results.get("native", Measurement("native", 0, 0, 0, 0)).wall_s

    for label in labels_to_run:
        m = all_results.get(label)
        if not m:
            continue
        cpu = m.user_cpu_s + m.sys_cpu_s
        init_s = m.extra.get("init_s")
        run_s = m.extra.get("run_s")
        ratio = (m.wall_s / native_wall) if native_wall > 0 else 0
        init_str = f"{init_s:>10.3f}" if init_s is not None else f"{'—':>10}"
        run_str = f"{run_s:>10.3f}" if run_s is not None else f"{'—':>10}"
        ratio_str = f"{ratio:>8.1f}" if ratio else f"{'—':>8}"
        print(f"{label:<22} {m.wall_s:>10.3f} {cpu:>10.3f} "
              f"{m.peak_rss_mib:>8.1f} M {init_str} {run_str} {ratio_str}")

    print("=" * 96)
    print()
    print("Columns:")
    print("  wall (s)  : total wall-clock time of the subprocess")
    print("  CPU  (s)  : user_cpu + sys_cpu of the subprocess (children rusage)")
    print("  peak RSS  : max resident set size of the subprocess, MiB")
    print("  qil init  : qiling/wrapper construction time (in-process measurement)")
    print("  ql.run    : ql.run() time, i.e. taint-propagation phase")
    print("  ×native   : ratio of wall time to the native run")
    print()
    print("Note:  qil init + ql.run < wall  →  the difference is python startup,")
    print("       module imports, and shutdown / report finalisation.")

    # ----- Save JSON if requested ---------------------------------------
    if args.json:
        out_dict = {label: asdict(m) for label, m in all_results.items()}
        with open(args.json, "w") as f:
            json.dump(out_dict, f, indent=2)
        print(f"\nFull results written to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
