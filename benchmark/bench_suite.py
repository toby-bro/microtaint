"""The performance suite: one run, one JSON, meant to survive rewrites.

    python benchmark/bench_suite.py                  # everything
    python benchmark/bench_suite.py --quick          # the PR budget
    python benchmark/bench_suite.py --json out.json

Two kinds of number come out of here and they are used differently.

**Gates** are deterministic: counters, coverage percentages, operation counts.
They do not vary between two runs on the same commit, on any machine, so they
can FAIL a pull request.  **Timings** vary with whatever else the machine is
doing, so every timing is also reported as a RATIO against a baseline measured
in the same process, which cancels the machine out; timings are advisory.

What is deliberately NOT measured: anything named after the engine's interior.
`generate_static_rule` (which the previous CI benchmark timed), cell counts,
assignment counts -- all of them are artefacts of one implementation, two are
already close to dead, and a benchmark built on them stops meaning anything the
moment the interior changes.  What is measured instead is the contract that
survives a rewrite: nanoseconds per GUEST INSTRUCTION, executions per second,
and the ratio of those to bare emulation.

The tiers answer different questions:

  A  end to end     what a real run costs, split bare -> hooked -> full, so a
                    regression says WHICH layer moved
  B  throughput     executions per second, the unit a fuzzer counts in, measured
                    both with a fresh process per execution (the true shape) and
                    reusing one (the gap between them IS the setup cost)
  C  per instruction taint propagation over the bank, p50/p99/p100 -- the tail is
                    what hurts once an instruction sits in a hot loop
  D  primitives     pure C, no Python: what being hooked costs at all.  The most
                    stable numbers here, and what explains a regression in A
  T  fallback tail  the differential is still the oracle the compiled path is
                    checked against, still the fallback for what does not lower,
                    and still what absorbs a new ISA.  This measures how bad the
                    tail is when we fall back, not how fast the differential is
"""
# ruff: noqa: PLC0415, S603, T201, S110
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_DENSITY = _HERE / 'taint_density'

#: Checked-in guests, so the workload is pinned and not at the mercy of whatever
#: compiler the runner happens to have.
_WORKLOADS = ('bench_untainted', 'bench_sparse', 'bench_dense')
_STDIN = bytes((i * 7 + 13) & 0xFF for i in range(64))


# ---------------------------------------------------------------------------
# The worker: one configuration, one process.  The engine reads its switches at
# import, so a configuration cannot be changed in flight.
# ---------------------------------------------------------------------------

def _worker(mode: str, guest: str, iters: int) -> int:
    import io

    from qiling import Qiling
    from qiling.const import QL_VERBOSE

    saved, devnull = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    best, instrs = float('inf'), 0
    for _ in range(iters):
        ql = Qiling([guest], '/', verbose=QL_VERBOSE.OFF)
        ql.os.stdin = io.BytesIO(_STDIN)
        count = [0]
        if mode != 'bare':
            # 'hooked' is the ENGINE'S OWN hook returning immediately
            # (MICROTAINT_NULL_HOOK=1, set by the caller), not a Python
            # callback: a Python callback costs about a microsecond and would
            # make this layer look like the dominant cost when it is not.  What
            # this measures is the cost of being hooked at all, by the
            # trampoline that actually ships.
            from microtaint.emulator.reporter import Reporter
            from microtaint.emulator.wrapper import MicrotaintWrapper
            MicrotaintWrapper(ql, reporter=Reporter(json_mode=False, stream=io.StringIO()))
        os.dup2(devnull, 1)
        t0 = time.perf_counter()
        try:
            ql.run()
        except Exception:
            pass
        dt = time.perf_counter() - t0
        os.dup2(saved, 1)
        best = min(best, dt)
        instrs = count[0] or instrs
    print(json.dumps({'seconds': best}))
    return 0


def _run_worker(mode: str, guest: str, iters: int, env: dict | None = None) -> float:
    out = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), '--worker', mode,
         '--guest', guest, '--iters', str(iters)],
        capture_output=True, env={**os.environ, **(env or {})}, timeout=1800)
    for line in out.stdout.decode().splitlines()[::-1]:
        if line.startswith('{'):
            return json.loads(line)['seconds']
    raise RuntimeError(f'worker {mode} failed: {out.stderr.decode()[-500:]}')


def _instruction_count(guest: str) -> int:
    """How many guest instructions the workload executes.  Deterministic, so it
    doubles as a check that the workload itself has not drifted."""
    import io

    from qiling import Qiling
    from qiling.const import QL_VERBOSE
    from unicorn import UC_HOOK_CODE

    saved, devnull = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    ql = Qiling([guest], '/', verbose=QL_VERBOSE.OFF)
    ql.os.stdin = io.BytesIO(_STDIN)
    n = [0]

    def tick(uc, a, s, u):
        n[0] += 1

    ql.uc.hook_add(UC_HOOK_CODE, tick)
    os.dup2(devnull, 1)
    try:
        ql.run()
    except Exception:
        pass
    os.dup2(saved, 1)
    return n[0]


# ---------------------------------------------------------------------------
# A -- end to end
# ---------------------------------------------------------------------------

def tier_a(quick: bool) -> dict:
    out: dict = {}
    guests = _WORKLOADS[:2] if quick else _WORKLOADS
    iters = 2 if quick else 3
    for name in guests:
        guest = str(_DENSITY / f'{name}.elf')
        if not Path(guest).exists():
            continue
        n = _instruction_count(guest)
        bare = _run_worker('bare', guest, iters)
        hooked = _run_worker('hooked', guest, iters, {'MICROTAINT_NULL_HOOK': '1'})
        full = _run_worker('full', guest, iters)
        out[name] = {
            'instructions': n,
            'bare_ns': 1e9 * bare / n,
            'hooked_ns': 1e9 * hooked / n,
            'full_ns': 1e9 * full / n,
            # The ratio is the number that survives a change of machine.
            'x_bare': full / bare,
            'x_bare_hook_only': hooked / bare,
        }
    return out


# ---------------------------------------------------------------------------
# B -- throughput, the unit a fuzzer counts in
# ---------------------------------------------------------------------------

def _exec_once_in_process(guest: str, payload: bytes) -> None:
    import io

    from qiling import Qiling
    from qiling.const import QL_VERBOSE

    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper

    ql = Qiling([guest], '/', verbose=QL_VERBOSE.OFF)
    ql.os.stdin = io.BytesIO(payload)
    MicrotaintWrapper(ql, reporter=Reporter(json_mode=False, stream=io.StringIO()))
    try:
        ql.run()
    except Exception:
        pass


def _b_worker(guest: str, n: int) -> int:
    saved, devnull = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    t0 = time.perf_counter()
    for i in range(n):
        _exec_once_in_process(guest, bytes(((i * 31 + j) & 0xFF) for j in range(64)))
    dt = time.perf_counter() - t0
    os.dup2(saved, 1)
    print(json.dumps({'seconds': dt, 'n': n}))
    return 0


def tier_b(quick: bool) -> dict:
    """Executions per second, both shapes.

    A fuzzer forks per input, so a fresh process per execution is the honest
    number; reusing one process is the ceiling if the harness is made
    persistent.  The GAP between them is the per-execution setup cost --
    interpreter start, Qiling init, hook registration, and any planning the
    engine does on first sight of a block -- and nothing else in this suite can
    see it.
    """
    guest = str(_DENSITY / 'bench_sparse.elf')
    if not Path(guest).exists():
        return {}
    n = 6 if quick else 20

    # Reusing one process.
    out = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), '--b-worker',
         '--guest', guest, '--iters', str(n)],
        capture_output=True, timeout=1800)
    reused = None
    for line in out.stdout.decode().splitlines()[::-1]:
        if line.startswith('{'):
            reused = json.loads(line)
            break
    if reused is None:
        return {}

    # A fresh process per execution.
    t0 = time.perf_counter()
    per_proc = max(2, n // 3)
    for _ in range(per_proc):
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), '--b-worker',
             '--guest', guest, '--iters', '1'],
            capture_output=True, timeout=1800)
    fresh_dt = time.perf_counter() - t0

    reused_per = reused['seconds'] / reused['n']
    fresh_per = fresh_dt / per_proc
    return {
        'exec_per_sec_reusing_process': 1.0 / reused_per,
        'exec_per_sec_fresh_process': 1.0 / fresh_per,
        'setup_cost_seconds': fresh_per - reused_per,
    }


# ---------------------------------------------------------------------------
# D -- primitives, pure C
# ---------------------------------------------------------------------------

def tier_d() -> dict:
    exe = _DENSITY / 'hookcost'
    if not exe.exists():
        build = subprocess.run(['make', 'hookcost'], cwd=str(_DENSITY),
                               capture_output=True, timeout=1800)
        if build.returncode != 0 or not exe.exists():
            return {'skipped': 'hookcost did not build'}
    run = subprocess.run([str(exe)], capture_output=True, cwd=str(_DENSITY), timeout=1800)
    out: dict = {}
    pattern = re.compile(r'^\s+(\S.*?)\s{2,}[\d.]+ ms\s+([\d.]+) ns/instruction')
    for line in run.stdout.decode().splitlines():
        m = pattern.match(line)
        if m:
            out[m.group(1).strip()] = float(m.group(2))
    return out


# ---------------------------------------------------------------------------
# C -- per instruction, and T -- what the fallback tail costs
# ---------------------------------------------------------------------------

def _bank(quick: bool):
    sys.path.insert(0, str(_HERE))
    from instruction_bank import load_bank  # type: ignore[import-not-found]
    bank = load_bank()
    limit = 60 if quick else 400
    isas = ('AMD64', 'ARM64') if quick else sorted(bank)
    return [(i, bank[i], limit) for i in isas if i in bank]


def tier_c_and_tail(quick: bool) -> tuple[dict, dict, dict]:
    """Per-instruction propagation, the fallback tail, and the gates from both.

    The compiled program and the differential are timed on the SAME forms, so
    the tail number answers the question that matters when a form does not
    lower: how much worse is the run if this instruction sits in a loop.
    """
    from microtaint.instrumentation.ast import EvalContext
    from microtaint.instrumentation.cell_c import taint_ir_c
    from microtaint.simulator import CellSimulator
    from microtaint.sleigh.engine import generate_static_rule
    from microtaint.taint_ir import frompcode
    from microtaint.taint_ir.exec import compile_program
    from microtaint.types import ImplicitTaintPolicy

    per_instr: dict = {}
    tail: dict = {}
    gates: dict = {}
    for isa, spec, limit in _bank(quick):
        key = spec.arch.value if hasattr(spec.arch, 'value') else str(spec.arch)
        builder = frompcode.Builder(spec.arch, key.endswith('BE'), 'concrete')
        names = sorted(set(builder.name_by_off.values()))
        layout = {n: i for i, n in enumerate(names)}
        # Derive every base from the layout SIZE.  Hardcoding them (values at
        # 256, memory at 512) is fine until an architecture has more registers
        # than that -- PPC's geometry names 2240 -- and then the value slots run
        # into the memory slots and the run writes out of bounds.  It does not
        # fail, it corrupts and eventually dumps core.
        n_reg = len(layout)
        mem_base = 2 * n_reg
        width = mem_base + 4 * 8
        offs = {'mem': 0, 'addr': 1, 'addrt': 2, 'sttaint': 3}

        def slot_of(k, layout=layout, builder=builder, n_reg=n_reg,
                    mem_base=mem_base, offs=offs):
            if k[0] in ('reg', 'regv'):
                nm = builder.name_by_off.get(k[1])
                if nm is None or nm not in layout:
                    raise KeyError(k)
                return layout[nm] + (n_reg if k[0] == 'regv' else 0)
            if k[0] in offs:
                return mem_base + 4 * k[1] + offs[k[0]]
            raise KeyError(k)

        times, ops, lowered, declined, declined_forms = [], 0, 0, 0, []
        for ins in spec.instructions[:limit]:
            try:
                prog = frompcode.build_ir(spec.arch, bytes(ins.bytes))
                capsule, ser = compile_program(prog, slot_of)
            except Exception:
                declined += 1
                declined_forms.append(ins)
                continue
            lowered += 1
            ops += len(ser['op_ids'])
            try:
                taint_ir_c.jit(capsule)
                ns, _sink = taint_ir_c.bench(capsule, [0] * width, [0] * width)
                times.append(ns)
            except Exception:
                pass
        if times:
            times.sort()
            per_instr[isa] = {
                'p50': statistics.median(times),
                'p99': times[min(len(times) - 1, int(0.99 * len(times)))],
                'p100': times[-1],
            }
        # Gates: deterministic, identical on any machine for a given commit.
        gates[f'{isa}_lowered'] = lowered
        gates[f'{isa}_declined'] = declined
        gates[f'{isa}_total_ops'] = ops

        # The tail: the differential on forms the compiled path REFUSED, which
        # is exactly what the engine falls back to for them.
        sim = CellSimulator(spec.arch)
        vals = {r.name: (0x1234567 + 7 * i) for i, r in enumerate(spec.regs)}
        tt = []
        for ins in declined_forms[: (4 if quick else 12)]:
            try:
                circ = generate_static_rule(spec.arch, bytes(ins.bytes), spec.regs)
                ctx = EvalContext(input_values=vals, input_taint={}, simulator=sim,
                                  implicit_policy=ImplicitTaintPolicy.KEEP)
                t0 = time.perf_counter()
                circ.evaluate(ctx)
                tt.append(1e9 * (time.perf_counter() - t0))
            except Exception:
                continue
        if tt:
            tt.sort()
            tail[isa] = {'declined_forms_timed': len(tt), 'p50': statistics.median(tt),
                         'p100': tt[-1]}
    return per_instr, tail, gates


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--quick', action='store_true', help='the pull-request budget')
    ap.add_argument('--json', help='write the results here')
    ap.add_argument('--worker', help=argparse.SUPPRESS)
    ap.add_argument('--b-worker', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--guest', help=argparse.SUPPRESS)
    ap.add_argument('--iters', type=int, default=3, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.worker:
        return _worker(args.worker, args.guest, args.iters)
    if args.b_worker:
        return _b_worker(args.guest, args.iters)

    started = time.perf_counter()
    per_instr, tail, gates = tier_c_and_tail(args.quick)
    results = {
        'gates': gates,                       # deterministic: may FAIL a PR
        'a_end_to_end': tier_a(args.quick),   # everything below is advisory
        'b_throughput': tier_b(args.quick),
        'c_per_instruction': per_instr,
        'd_primitives': tier_d(),
        't_fallback_tail': tail,
    }
    results['wall_seconds'] = time.perf_counter() - started

    print(json.dumps(results, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
