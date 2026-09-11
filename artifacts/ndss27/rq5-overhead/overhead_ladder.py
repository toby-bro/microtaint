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
elif layer.startswith("c-codehook"):
    # Pure-C UC_HOOK_CODE: no GIL, no Python frame, no marshalling.  Isolates
    # what Unicorn itself pays to call out on every instruction.
    import ctypes
    import unicorn.unicorn_py3.unicorn as _U3
    import unicorn.x86_const as ux
    lib = ctypes.CDLL(hooklib_path)
    lib.ladder_count.restype = ctypes.c_uint64
    uclib = _U3.uclib
    lib.ladder_init(
        ctypes.cast(uclib.uc_hook_add, ctypes.c_void_p),
        ctypes.cast(uclib.uc_reg_read, ctypes.c_void_p),
        ux.UC_X86_REG_RAX, ux.UC_X86_REG_RBX, ux.UC_X86_REG_RCX, ux.UC_X86_REG_RDX,
    )
    from unicorn import UC_HOOK_CODE
    which = 0 if layer == "c-codehook" else 1
    rc = lib.ladder_install(ql.uc._uch, UC_HOOK_CODE, which)
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
    flags = set() if layer == "microtaint-none" else {"bof", "uaf", "sc", "aiw"}
    wrapper = MicrotaintWrapper(
        ql,
        check_bof=("bof" in flags), check_uaf=("uaf" in flags),
        check_sc=("sc" in flags), check_aiw=("aiw" in flags),
        reporter=Reporter(json_mode=False, stream=open(os.devnull, "w")),
    )

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
    "run_s": run_s,
    "guest_bytes": guest_bytes,
    # How many times the C callback actually ran.  A C hook that silently never
    # fires would read as "free", which is the kind of fast wrong number this
    # ladder exists to avoid, so the harness checks it against the instruction
    # count rather than trusting uc_hook_add's return value.
    "c_hook_calls": int(c_hook_lib.ladder_count()) if c_hook_lib is not None else None,
    "instr_hook_registered": getattr(wrapper, "_instr_hook_registered", None),
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
    ('c-codehook',           'per-instruction control, pure C, empty body',   'emulator',   True),
    ('c-codehook-regs',      'the same, reading 4 guest registers',           'emulator',   True),
    ('microtaint-none',      'bit-precise taint propagation, no detectors',   'microtaint', True),
    ('microtaint-all',       'plus the four detectors',                       'microtaint', True),
    ('blockhook',            'empty per-BLOCK Python hook',                   'reference',  False),
    ('codehook',             'empty per-INSTRUCTION Python hook',             'reference',  False),
    ('codehook-regs',        'per-instruction Python hook reading 4 regs',    'reference',  False),
]


def _run(argv, stdin_data, timeout, env=None):
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={**os.environ, **env} if env else None,
    )
    out, err = proc.communicate(stdin_data, timeout=timeout)
    return time.perf_counter() - t0, out, err


def measure(layer, binary, rootfs, stdin_data, timeout):
    """One run of one rung.  Returns (run_s, guest_bytes, extra) or None."""
    if layer == 'native':
        wall, out, _ = _run([binary], stdin_data, timeout)
        return wall, len(out), {}
    path = '/tmp/_ladder_helper.py'
    with open(path, 'w') as f:
        f.write(HELPER)
    hooklib = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ladder_hooks.so')
    _, out, err = _run([sys.executable, path, binary, rootfs, layer, hooklib],
                       stdin_data, timeout, env=ARTIFACT_ENGINE_ENV)
    text = out.decode('utf-8', 'replace')
    idx = text.rfind('{')
    if idx < 0:
        raise RuntimeError(f'{layer}: helper produced no JSON\n{err.decode()[-400:]}')
    d = json.loads(text[idx:])
    return d['run_s'], d['guest_bytes'], d


def count_instructions(binary, rootfs, stdin_data, timeout):
    path = '/tmp/_ladder_count.py'
    with open(path, 'w') as f:
        f.write(COUNT_HELPER)
    _, _, err = _run([sys.executable, path, binary, rootfs], stdin_data, timeout)
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
    p.add_argument('--timeout', type=float, default=1800.0)
    p.add_argument('--json', default=None)
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
    print('# taint_ir and block mode pinned OFF (per-instruction LogicCircuit path)')
    print()

    results = {}
    for layer, desc, _bucket, _chain in LAYERS:
        samples = []
        gb = None
        extra = {}
        for i in range(args.runs):
            sys.stderr.write(f'[{layer} {i+1}/{args.runs}] running…\n')
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
        if not gb:
            raise SystemExit(f'{layer}: the guest wrote 0 bytes, so the workload never ran')
        # Each run is its own subprocess, so the counter starts at zero every time
        # and `calls` is already PER RUN -- dividing by len(samples) was wrong, and
        # the check caught it (1,256,123 x 3 = 3,768,369 exactly).
        calls = extra.get('c_hook_calls')
        if layer.startswith('c-') and n_instrs:
            if not calls:
                raise SystemExit(f'{layer}: the C hook never fired, so this rung would '
                                 f'report the cost of not being called')
            if abs(calls - n_instrs) > 0.02 * n_instrs:
                raise SystemExit(f'{layer}: the C hook fired {calls:,} times against '
                                 f'{n_instrs:,} instructions; it is not hooking every one')
        results[layer] = {
            'run_s': statistics.median(samples), 'desc': desc,
            'guest_bytes': gb, 'n_runs': len(samples),
            'c_hook_calls_per_run': calls,
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
        ns = ns_of(layer)
        marg = (ns - prev_ns) if (ns is not None and prev_ns is not None) else None
        xq = (ns / qil) if (ns and qil) else None
        print(f'{layer:<22} {results[layer]["run_s"]:>9.3f} {_fmt(ns, 10)} {_fmt(marg, 10)} '
              f'{_fmt(xq, 9)}  {bucket:<12}')
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

    mt = ns_of('microtaint-none')
    ch = ns_of('codehook')
    cch = ns_of('c-codehook')
    cchr = ns_of('c-codehook-regs')
    print('What the ladder says:')
    print(f'  * the emulator costs {qil:.0f} ns/instr with nothing hooked.')
    if cch and qil:
        print(f'  * getting control on EVERY instruction in pure C costs {cch:.0f} ns/instr,')
        print(f'    only {cch - qil:+.0f} over bare emulation. Per-instruction hooking is not')
        print("    inherently expensive, and losing Unicorn's block chaining is cheap.")
    if cch and cchr:
        print(f'  * reading 4 guest registers from that C callback adds {cchr - cch:+.0f} ns/instr,')
        print(f'    so control plus the state a taint engine needs is {cchr:.0f} ns/instr.')
    if mt and cchr:
        print(f'  * microtaint computes the taint in {mt - cchr:+.0f} ns/instr on top of that,')
        print(f'    {mt / cchr:.0f}x the cost of merely reading the same registers.')
        print('    THAT is the part that is ours, and the honest number to attack.')
    if ch and cch and mt:
        print(f'  * for scale: the same empty hook written in PYTHON costs {ch:.0f} ns/instr,')
        print(f"    {ch / cch:.0f}x the C one, and more than microtaint's entire engine ({mt:.0f}).")
        print('    The hosting language dominates the analysis, so a Python-hooked tool')
        print('    is not slow because of what it computes.')
    if ns_of('microtaint-all') and mt:
        print(f'  * the four detectors add {ns_of("microtaint-all") - mt:+.0f} ns/instr '
              f'({100 * (ns_of("microtaint-all") / mt - 1):.0f}%).')

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'guest_instructions': n_instrs, 'stdin_bytes': len(stdin_data),
                       'engine_env': ARTIFACT_ENGINE_ENV, 'layers': results}, f, indent=2)
        print(f'\nwritten to {args.json}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
