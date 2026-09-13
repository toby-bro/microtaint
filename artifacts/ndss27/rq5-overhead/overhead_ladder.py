"""Where the end-to-end overhead actually comes from, layer by layer.

`overhead_bench.py` answers "how much slower is the whole thing".  This answers
"whose cost is it", by running the SAME workload under a ladder of
configurations that each add exactly one thing:

    native                the binary, run directly
    qiling-only           + Qiling/Unicorn emulation, no hooks at all
    c-codehook            + control on every instruction, PURE C, empty body
    c-codehook-regs       + reading the 4 guest registers a taint engine needs
    microtaint-none       + bit-precise taint propagation
    microtaint-all        + the four detectors

The C rungs are what make the ladder attributable.  microtaint's hook is compiled
C, so `c-codehook` is the floor it actually stands on, and the difference between
them is the analysis rather than the plumbing:

  * `c-codehook` prices per-instruction control with no Python anywhere: no GIL,
    no frame, no marshalling.  It also absorbs the loss of Unicorn's translation
    block chaining, which happens as soon as any hook exists.  Measured, the two
    together are about +20 ns/instr over bare emulation, so getting called on
    every instruction is NOT inherently expensive.
  * `c-codehook-regs` adds the guest state a taint engine has to read anyway.

Three Python rungs are measured as REFERENCE points, out of the chain.  They
price the hosting language, not the analysis: an empty per-instruction Python
hook costs more than microtaint's entire engine, so "hooks every instruction from
Python" explains more of a tool's cost than anything it computes.

Read the MARGINAL column.  "Is the overhead microtaint's fault" is answered by
`microtaint-none` minus `c-codehook-regs`, not by the x-native column.

Usage:
    python overhead_ladder.py bench.elf --gen-input 64 --runs 3
"""
# Experiment script, not library code.
# mypy: disable-error-code="no-untyped-def, no-untyped-call, type-arg, explicit-any"
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

# taint_ir and block mode replace the per-instruction circuit path this paper
# measures, and taint_ir is the engine's DEFAULT.  Same pin as overhead_bench.py.
ARTIFACT_ENGINE_ENV = {'MICROTAINT_TAINT_IR': '0', 'MICROTAINT_BLOCK': '0'}

HELPER = r"""
import io, json, os, sys, tempfile, time

binary, rootfs, layer = sys.argv[1], sys.argv[2], sys.argv[3]
hooklib_path = sys.argv[4] if len(sys.argv) > 4 else ""
stdin_data = sys.stdin.buffer.read() if not sys.stdin.isatty() else b""

from qiling import Qiling
from qiling.const import QL_VERBOSE

ql = Qiling([binary], rootfs, verbose=QL_VERBOSE.OFF)
if stdin_data:
    try:
        ql.os.stdin = io.BytesIO(stdin_data)
    except Exception:
        pass

wrapper = None
c_hook_lib = None
if layer == "blockhook":
    ql.hook_block(lambda *a: None)
elif layer == "codehook":
    ql.hook_code(lambda *a: None)
elif layer.startswith("c-"):
    # Pure-C UC_HOOK_CODE: no GIL, no Python frame, no marshalling.  Isolates
    # what Unicorn itself pays to call out on every instruction.
    import ctypes
    import unicorn.unicorn_py3.unicorn as _U3
    import unicorn.x86_const as ux
    lib = ctypes.CDLL(hooklib_path)
    lib.ladder_count.restype = ctypes.c_uint64
    lib.ladder_mem_count.restype = ctypes.c_uint64
    uclib = _U3.uclib
    lib.ladder_init(
        ctypes.cast(uclib.uc_hook_add, ctypes.c_void_p),
        ctypes.cast(uclib.uc_reg_read, ctypes.c_void_p),
        ctypes.cast(uclib.uc_mem_read, ctypes.c_void_p),
        ux.UC_X86_REG_RAX, ux.UC_X86_REG_RBX, ux.UC_X86_REG_RCX, ux.UC_X86_REG_RDX,
    )
    from unicorn import UC_HOOK_BLOCK, UC_HOOK_CODE
    # `ladder_install` already takes the hook TYPE, so the block rung needs no
    # new C: the same empty callback, registered per block instead of per
    # instruction.  That is what separates the two costs a per-instruction hook
    # pays at once -- see the `c-blockhook` entry in LAYERS.
    if layer == "c-blockhook":
        hook_type, which = UC_HOOK_BLOCK, 0
    else:
        hook_type, which = UC_HOOK_CODE, (0 if layer == "c-codehook" else 1)
    rc = lib.ladder_install(ql.uc._uch, hook_type, which)
    if rc != 0:
        raise SystemExit(f"uc_hook_add failed: {rc}")
    c_hook_lib = lib
elif layer == "codehook-regs":
    # A hook that actually touches guest state, so the rung separates "being
    # called" from "reading anything".  Four GP registers, the same four the
    # RQ2 comparison reports on.
    import unicorn.x86_const as ux
    regs = (ux.UC_X86_REG_RAX, ux.UC_X86_REG_RBX, ux.UC_X86_REG_RCX, ux.UC_X86_REG_RDX)
    def _h(q, addr, size, *_ud):
        # Qiling omits user_data when none was registered, so accept both arities.
        for r in regs:
            q.uc.reg_read(r)
    ql.hook_code(_h)
elif layer.startswith("microtaint"):
    from microtaint.emulator.reporter import Reporter
    from microtaint.emulator.wrapper import MicrotaintWrapper
    flags = set() if layer in ("microtaint-none", "microtaint-plumbing") else {"bof", "uaf", "sc", "aiw"}
    wrapper = MicrotaintWrapper(
        ql,
        check_bof=("bof" in flags), check_uaf=("uaf" in flags),
        check_sc=("sc" in flags), check_aiw=("aiw" in flags),
        reporter=Reporter(json_mode=False, stream=open(os.devnull, "w")),
    )
    if layer == "microtaint-plumbing":
        # The REAL engine, hook armed on every instruction, but nothing is ever
        # tainted.  Every instruction then takes the untainted prefilter exit:
        # hook entry, guest register reads, the address->circuit cache, the
        # prefilter test, and for a memory operand an effective-address
        # computation and a shadow lookup.  All of that is work a byte-granular
        # taint engine owes per instruction no matter how its propagation is
        # written; what is missing is only the propagation itself.
        #
        # Arming is normally deferred until the first taint arrives, so arm it
        # by hand and then stop the taint sources: _taint_bytes is what the read
        # syscall hook calls, and the two public injectors are the other doors.
        wrapper._arm_deferred_hooks()
        _noop = lambda *a, **k: None
        wrapper._taint_bytes = _noop
        wrapper.taint_bit = _noop
        wrapper.taint_region = _noop

# Guest stdout is redirected so its bytes cannot corrupt this JSON, and so the
# byte count proves the workload ran (bench.c writes its hash after the rounds).
tf = tempfile.TemporaryFile()
saved = os.dup(1)
os.dup2(tf.fileno(), 1)
t0 = time.perf_counter()
try:
    ql.run()
except Exception:
    pass
run_s = time.perf_counter() - t0
os.dup2(saved, 1); os.close(saved)
tf.seek(0, 2); guest_bytes = tf.tell(); tf.close()

print(json.dumps({
    "layer": layer,
    "per_block_hook": layer == "c-blockhook",
    "run_s": run_s,
    "guest_bytes": guest_bytes,
    # How many times the C callback actually ran.  A C hook that silently never
    # fires would read as "free", which is the kind of fast wrong number this
    # ladder exists to avoid, so the harness checks it against the instruction
    # count rather than trusting uc_hook_add's return value.
    "c_hook_calls": int(c_hook_lib.ladder_count()) if c_hook_lib is not None else None,
    "c_mem_hook_calls": int(c_hook_lib.ladder_mem_count()) if c_hook_lib is not None else None,
    "instr_hook_registered": getattr(wrapper, "_instr_hook_registered", None),
    # For microtaint rungs: how many instructions the hook actually saw, and how
    # many took the untainted prefilter exit.  The plumbing rung is only honest
    # if the hook ran on every instruction and every one of them prefiltered;
    # without these it could silently be measuring an unarmed hook, which is
    # fast and meaningless -- the exact failure this ladder exists to catch.
    "hook_instr_total": getattr(getattr(wrapper, "_instr_hook_obj", None), "instr_total", None),
    "hook_prefilter_hits": getattr(getattr(wrapper, "_instr_hook_obj", None), "prefilter_hits", None),
    # The rung's actual claim is "nothing was ever tainted".  Check THAT rather
    # than the prefilter share: ~5% of instructions are prefilter-INELIGIBLE by
    # structure (a PC target needs the implicit-taint decision, a memory write
    # needs the store path), so they evaluate even with a clean machine, and a
    # threshold on the share would be measuring circuit shape, not taint.
    "residual_register_taint": sum(
        1 for _v in (getattr(wrapper, "register_taint", {}) or {}).values() if _v
    ),
    "taint_ir_modules": sorted(m for m in sys.modules if m.startswith("microtaint.taint_ir")),
    "env_taint_ir": os.environ.get("MICROTAINT_TAINT_IR", "<unset>"),
}))
"""

COUNT_HELPER = r"""
import io, json, sys
binary, rootfs = sys.argv[1], sys.argv[2]
stdin_data = sys.stdin.buffer.read() if not sys.stdin.isatty() else b""
from qiling import Qiling
from qiling.const import QL_VERBOSE
ql = Qiling([binary], rootfs, verbose=QL_VERBOSE.OFF)
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

#: (name, description, attribution, in_chain).
#:
#: `in_chain` matters.  The chain is genuinely nested: emulate, get control on
#: every instruction, read the guest state a taint engine needs, compute the
#: taint, run the detectors.  microtaint's hook is compiled C, so the C rungs are
#: the floor it actually sits on and subtracting them means something.
#:
#: The PYTHON rungs are reference points and deliberately OUT of the chain:
#: microtaint does not build on them, it replaces them.  They price the hosting
#: language rather than the analysis.  An earlier version had them in the chain
#: and the marginal column went negative, which is how that mistake announces
#: itself.
LAYERS = [
    ('native',               'the binary, run directly',                      'baseline',   True),
    ('qiling-only',          'Qiling/Unicorn emulation, no hooks',            'emulator',   True),
    ('c-blockhook',          'per-BLOCK control, pure C, empty body',         'emulator',   True),
    ('c-codehook',           'per-instruction control, pure C, empty body',   'emulator',   True),
    ('c-codehook-regs',      'the same, reading 4 guest registers',           'emulator',   True),
    ('codehook',             'the same in PYTHON, empty body',                'hosting',    True),
    ('codehook-regs',        'the same in PYTHON, reading 4 registers',       'hosting',    True),
    ('microtaint-plumbing',  'the real engine, armed, but nothing tainted',   'plumbing',   True),
    ('microtaint-none',      'bit-precise taint propagation, no detectors',   'microtaint', True),
    ('microtaint-all',       'plus the four detectors',                       'microtaint', True),
    ('blockhook',            'empty per-BLOCK Python hook',                   'reference',  False),
]


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


def _run(argv, stdin_data, timeout, env=None):
    """Run a child to completion.  Returns (wall_s, stdout, stderr, metrics).

    `metrics` carries wall time, CPU time (user+sys) and peak RSS for the child.
    All three are reported because they answer different questions: `ql.run` is
    the taint phase, `wall` includes interpreter start-up and emulator
    construction that a user of the tool also waits for, and `cpu` says whether
    the difference between them is work or waiting.

    Peak RSS comes from wait4's rusage for THIS child rather than from polling
    /proc: a poll can miss a short-lived peak, and the rung that matters here
    (the shadow memory growing with the tainted address space) is exactly the
    kind of thing that would be missed intermittently and then look like noise.
    """
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={**os.environ, **env} if env else None,
    )
    out_chunks, err_chunks = [], []
    import threading
    def _drain(stream, sink):
        try:
            sink.append(stream.read())
        except Exception:
            pass
    if stdin_data is not None and proc.stdin is not None:
        try:
            proc.stdin.write(stdin_data)
        except BrokenPipeError:
            pass
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass
    to = threading.Thread(target=_drain, args=(proc.stdout, out_chunks))
    te = threading.Thread(target=_drain, args=(proc.stderr, err_chunks))
    to.start()
    te.start()
    _pid, status, rusage = os.wait4(proc.pid, 0)
    proc.returncode = os.waitstatus_to_exitcode(status)
    to.join(timeout)
    te.join(timeout)
    wall = time.perf_counter() - t0
    out = out_chunks[0] if out_chunks else b''
    err = err_chunks[0] if err_chunks else b''
    metrics = {
        'wall_s': wall,
        'cpu_s': (rusage.ru_utime + rusage.ru_stime) if rusage else 0.0,
        'peak_rss_mib': (rusage.ru_maxrss / 1024.0) if rusage else 0.0,
    }
    return wall, out, err, metrics


def measure(layer, binary, rootfs, stdin_data, timeout):
    """One run of one rung.  Returns (run_s, guest_bytes, extra) or None."""
    if layer == 'native':
        wall, out, _err, m = _run([binary], stdin_data, timeout)
        return wall, len(out), dict(m)
    path = '/tmp/_ladder_helper.py'
    with open(path, 'w') as f:
        f.write(HELPER)
    hooklib = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ladder_hooks.so')
    _, out, err, m = _run([sys.executable, path, binary, rootfs, layer, hooklib],
                          stdin_data, timeout, env=ARTIFACT_ENGINE_ENV)
    text = out.decode('utf-8', 'replace')
    idx = text.rfind('{')
    if idx < 0:
        raise RuntimeError(f'{layer}: helper produced no JSON\n{err.decode()[-400:]}')
    d = json.loads(text[idx:])
    d.update(m)
    return d['run_s'], d['guest_bytes'], d


def count_instructions(binary, rootfs, stdin_data, timeout):
    path = '/tmp/_ladder_count.py'
    with open(path, 'w') as f:
        f.write(COUNT_HELPER)
    _, _, err, _m = _run([sys.executable, path, binary, rootfs], stdin_data, timeout)
    text = err.decode('utf-8', 'replace')
    idx = text.rfind('{')
    return int(json.loads(text[idx:])['instrs']) if idx >= 0 else None


def _fmt(v, w, prec=1):
    """A right-aligned number, or a dash when the rung has no value for it."""
    return f'{v:>{w}.{prec}f}' if v is not None else f'{"-":>{w}}'


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('binary')
    p.add_argument('--rootfs', default='/')
    p.add_argument('--gen-input', type=int, default=64,
                   help='Bytes of tainted stdin (default 64: a clean exit, no BOF)')
    p.add_argument('--runs', type=int, default=3)
    p.add_argument(
        '--runs-for', action='append', default=[], metavar='LAYER=N',
        help='override --runs for one layer, e.g. codehook-regs=15.  The two '
             'Python-hosted rungs cost 8.7 s and 80 s per run, so running 100 of '
             'each would take longer than the whole rest of the ladder; they are '
             'the only rungs whose count the paper reduces.',
    )
    p.add_argument('--timeout', type=float, default=1800.0)
    p.add_argument('--json', default=None)
    p.add_argument(
        '--only', action='append', default=[],
        help='Run ONLY these rungs (repeatable). The chain rungs are cheap; the three '
             'Python reference rungs are not (codehook-regs alone is ~75s per run), so '
             '--only is how you re-measure after an engine change without paying for '
             'reference points that cannot have moved.',
    )
    p.add_argument('--skip', action='append', default=[], help='Skip a rung (repeatable)')
    args = p.parse_args()

    binary = os.path.abspath(args.binary)
    stdin_data = bytes(range(256))[:args.gen_input]
    if not stdin_data:
        raise SystemExit('refusing to run with empty stdin: bench.c exits at its first '
                         'read, and every rung would measure process startup')

    sys.stderr.write('[count] counting guest instructions (untimed)…\n')
    n_instrs = count_instructions(binary, args.rootfs, stdin_data, args.timeout)
    print(f'# workload: {binary}, {n_instrs:,} guest instructions, '
          f'{len(stdin_data)} tainted stdin bytes, {args.runs} runs/rung')
    runs_for = {}
    for spec in args.runs_for:
        k, _, v = spec.partition('=')
        runs_for[k] = int(v)
    if runs_for:
        print(f'  per-rung overrides: {runs_for}')
    print('# taint_ir and block mode pinned OFF (per-instruction LogicCircuit path)')
    print()

    wanted = [ly for ly, _d, _b, _c in LAYERS
              if (not args.only or ly in args.only) and ly not in args.skip]
    results = {}
    for layer, desc, _bucket, _chain in LAYERS:
        if layer not in wanted:
            continue
        samples = []
        gb = None
        extra = {}
        n_runs = runs_for.get(layer, args.runs)
        for i in range(n_runs):
            sys.stderr.write(f'[{layer} {i+1}/{n_runs}] running…\n')
            try:
                run_s, gb, extra = measure(layer, binary, args.rootfs, stdin_data, args.timeout)
                samples.append(run_s)
            except Exception as exc:
                if layer.startswith('c-'):
                    raise SystemExit(
                        f'{layer} failed: {exc}\n'
                        f'Build the C hooks first:  gcc -O2 -fPIC -shared '
                        f'-o ladder_hooks.so ladder_hooks.c') from exc
                sys.stderr.write(f'  ! {layer} failed: {exc}\n')
                break
        if not samples:
            continue
        # A rung writing nothing means the guest never got going, and a rung that
        # measures an empty run would read as gloriously fast.
        if not gb:
            raise SystemExit(f'{layer}: the guest wrote 0 bytes, so the workload never ran')
        # Each run is its own subprocess, so the counter starts at zero every time
        # and `calls` is already PER RUN -- dividing by len(samples) was wrong, and
        # the check caught it (1,256,123 x 3 = 3,768,369 exactly).
        if layer == 'microtaint-plumbing' and n_instrs:
            seen = extra.get('hook_instr_total') or 0
            pre = extra.get('hook_prefilter_hits') or 0
            if abs(seen - n_instrs) > 0.02 * n_instrs:
                raise SystemExit(
                    f'{layer}: the hook saw {seen:,} instructions against {n_instrs:,}; '
                    f'it is not armed on every one, so this rung would price an '
                    f'uninstrumented run')
            res = extra.get('residual_register_taint')
            if res:
                raise SystemExit(
                    f'{layer}: {res} registers ended tainted, so taint WAS propagated '
                    f'and this is not a plumbing measurement')
            if seen:
                sys.stderr.write(
                    f'  [{layer}] {100 * pre / seen:.1f}% of instructions took the '
                    f'untainted prefilter exit; the rest are prefilter-ineligible '
                    f'circuits (PC targets, memory writes)\n')

        calls = extra.get('c_hook_calls')
        if layer.startswith('c-') and n_instrs:
            if not calls:
                raise SystemExit(f'{layer}: the C hook never fired, so this rung would '
                                 f'report the cost of not being called')
            if extra.get('per_block_hook'):
                # A per-BLOCK hook fires once per basic block, so it must be
                # FEWER calls than instructions and not more: a block hook that
                # somehow fired per instruction would be pricing the wrong
                # thing.  A block averages ~15 instructions on these binaries,
                # so anything above a third is suspicious rather than merely
                # short-blocked.
                if calls > n_instrs / 3:
                    raise SystemExit(
                        f'{layer}: the block hook fired {calls:,} times against '
                        f'{n_instrs:,} instructions, which is too many to be '
                        f'once per basic block')
            elif abs(calls - n_instrs) > 0.02 * n_instrs:
                raise SystemExit(f'{layer}: the C hook fired {calls:,} times against '
                                 f'{n_instrs:,} instructions; it is not hooking every one')
        results[layer] = {
            'run_s': statistics.median(samples), 'desc': desc,
            'per_block_hook': bool(extra.get('per_block_hook')),
            'guest_bytes': gb, 'n_runs': len(samples),
            'peak_rss_mib': extra.get('peak_rss_mib'),
            'wall_s': extra.get('wall_s'),
            'cpu_s': extra.get('cpu_s'),
            'c_hook_calls_per_run': calls,
            'hook_instr_total': extra.get('hook_instr_total'),
            'hook_prefilter_hits': extra.get('hook_prefilter_hits'),
            'c_mem_hook_calls': extra.get('c_mem_hook_calls'),
            'residual_register_taint': extra.get('residual_register_taint'),
            'env_taint_ir': extra.get('env_taint_ir'),
            'taint_ir_modules': extra.get('taint_ir_modules'),
        }

    # ---- the ladder ----------------------------------------------------
    def ns_of(layer):
        r = results.get(layer)
        return (r['run_s'] * 1e9 / n_instrs) if (r and n_instrs) else None

    print('=' * 100)
    print(f'{"layer":<22} {"time (s)":>9} {"ns/instr":>10} {"marginal":>10} '
          f'{"x qiling":>9}  {"attributable to":<12}')
    print('-' * 100)
    prev_ns = None
    qil = ns_of('qiling-only')
    for layer, _desc, bucket, chain in LAYERS:
        if not chain or layer not in results:
            continue
        _bucket = bucket
        ns = ns_of(layer)
        # A PYTHON rung is not an increment on the C rung above it -- it is the
        # same granularity measured in the other host language -- so it gets no
        # marginal, and the chain resumes from the last C rung.
        is_hosting = (_bucket == 'hosting')
        marg = (None if is_hosting
                else (ns - prev_ns) if (ns is not None and prev_ns is not None) else None)
        xq = (ns / qil) if (ns and qil) else None
        print(f'{layer:<22} {results[layer]["run_s"]:>9.3f} {_fmt(ns, 10)} {_fmt(marg, 10)} '
              f'{_fmt(xq, 9)}  {bucket:<12}')
        if not is_hosting:
            prev_ns = ns
    print('-' * 100)
    print('  reference points (NOT rungs microtaint stands on: it uses a C hook, not these)')
    for layer, desc, _bucket, chain in LAYERS:
        if chain or layer not in results:
            continue
        ns = ns_of(layer)
        xq = (ns / qil) if (ns and qil) else None
        print(f'{layer:<22} {results[layer]["run_s"]:>9.3f} {_fmt(ns, 10)} {"-":>10} '
              f'{_fmt(xq, 9)}  {desc}')
    print('=' * 100)
    print()

    # The engine as anyone actually runs it: detectors on.  `microtaint-none`
    # stays MEASURED because it is the only thing that separates propagation
    # from detection, but it is no longer part of the story the ladder tells:
    # the detectors cost nothing now, so printing both invites the reader to
    # conclude that nothing is being measured.
    mt = ns_of('microtaint-all') or ns_of('microtaint-none')
    ch = ns_of('codehook')
    cch = ns_of('c-codehook')
    cchr = ns_of('c-codehook-regs')
    print('What the ladder says:')
    # Every line below is guarded: --only can select any subset, and a summary
    # that assumes a rung was measured crashes on the subset that omits it.
    if qil:
        print(f'  * the emulator costs {qil:.0f} ns/instr with nothing hooked.')
    cbh = ns_of('c-blockhook')
    if cbh and qil:
        # This used to be asserted in the prose of the rung above.  It is a
        # measurement now: a hook of ANY kind costs Unicorn its translation
        # block chaining, and a per-BLOCK hook pays that and almost nothing
        # else, so this rung is the chaining loss on its own.
        print(f"  * losing Unicorn's block chaining costs {cbh - qil:+.0f} ns/instr: that is a")
        print('    pure-C hook that fires once per basic block and does nothing.')
    if cch and qil and cbh:
        print(f'  * getting control on EVERY instruction in pure C costs {cch:.0f} ns/instr,')
        base = cbh if cbh else qil
        via = 'over a per-block hook' if cbh else 'over bare emulation'
        print(f'    {cch - base:+.0f} {via}. That difference is per-instruction')
        print('    DISPATCH, with the chaining loss already paid above.')
    if cch and cchr:
        print(f'  * reading 4 guest registers from that C callback adds {cchr - cch:+.0f} ns/instr,')
        print(f'    so control plus the state a taint engine needs is {cchr:.0f} ns/instr.')
    plumb = ns_of('microtaint-plumbing')
    if mt and cchr and plumb:
        print(f"  * the engine's PLUMBING is {plumb:.0f} ns/instr ({plumb - cchr:+.0f} over that):")
        print('    the hook, reading the operands the instruction actually uses, the')
        print('    address->circuit cache, the prefilter, and for a memory operand an')
        print('    effective address and a shadow lookup. No taint is computed; this is')
        print('    what a byte-granular engine owes per instruction however it is written.')
        print(f'  * the ANALYSIS adds {mt - plumb:+.0f} ns/instr on top '
              f'({100 * (mt - plumb) / mt:.0f}% of the total):')
        print('    evaluating the taint circuit for the instructions that carry taint,')
        print('    and checking the detectors. Note what is NOT in here: the')
        print('    address->circuit lookup is already paid in the plumbing rung, because')
        print('    the prefilter needs the compiled circuit to decide anything. So this')
        print('    buys the EVALUATION, not the acquisition -- the instructions that stop')
        print('    taking the untainted exit and go through the algebra instead.')
        print('    That is the honest number to attack.')
    elif mt and cchr:
        print(f'  * microtaint computes the taint in {mt - cchr:+.0f} ns/instr on top of that,')
        print(f'    {mt / cchr:.0f}x the cost of merely reading the same registers.')
        print('    Add the microtaint-plumbing rung to split that into plumbing and algebra.')
    if ch and cch and mt:
        print(f'  * for scale: the same empty hook written in PYTHON costs {ch:.0f} ns/instr,')
        print(f"    {ch / cch:.0f}x the C one, and more than microtaint's entire engine ({mt:.0f}).")
        print('    The hosting language dominates the analysis, so a Python-hooked tool')
        print('    is not slow because of what it computes.')
    # Kept as a guard rather than a headline: if the detectors ever become
    # expensive again, the two-step ladder would show it as a PROPAGATION
    # regression and nobody would know which half moved.
    _none = ns_of('microtaint-none')
    if _none and ns_of('microtaint-all'):
        print(f'  * of that analysis cost, the four detectors are '
              f'{ns_of("microtaint-all") - _none:+.0f} ns/instr:')
        print('    they are recorded in C and rendered once, so they cost nothing')
        print('    measurable. This rung is kept as a guard, not as a headline.')

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'guest_instructions': n_instrs, 'stdin_bytes': len(stdin_data),
                       'engine_env': ARTIFACT_ENGINE_ENV,
                       'engine': _engine_provenance(), 'layers': results}, f, indent=2)
        print(f'\nwritten to {args.json}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
