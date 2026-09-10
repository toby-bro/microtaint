"""How fast can taint be propagated, per instruction, on this machine.

Every instruction in the bank is lowered to a taint program, compiled twice --
once for the flat-array C interpreter, once through clang -O3 -- and timed.  The
number reported is the whole propagation for one instruction: every output the
instruction writes, flags included, from register state in and taint state out.

Both backends run the same program, so they also check each other: a compiled
result that disagrees with the interpreted one is a code-generation bug, and
that diff is reported rather than averaged away.

    .venv/bin/python -m tests.taint_ir_perf                 # all ISAs
    .venv/bin/python -m tests.taint_ir_perf --isas AMD64 --top 15
    .venv/bin/python -m tests.taint_ir_perf --update        # rewrite baseline
"""
# ruff: noqa: PLC0415
from __future__ import annotations

import ctypes
import json
import random
from pathlib import Path
from typing import Any

from benchmark.instruction_bank import ISASpec
from microtaint.types import Architecture

BASELINE = Path(__file__).parent / 'taint_ir_perf_baseline.json'
MASK64 = 0xFFFFFFFFFFFFFFFF


def _slot_map(spec: ISASpec) -> tuple[list[str], dict[str, int]]:
    """(ordered register names, name -> slot) over the whole architecture.

    Every register the architecture declares gets a slot, not just the ones the
    bank lists: a lifted instruction routinely reads a register outside the
    bank's set, and a program with an unplaceable input cannot be run at all.
    """
    from microtaint.taint_ir.frompcode import builder_for
    b = builder_for(spec.arch)
    names = sorted({nm for nm, _sz in b.declared.values()})
    return names, {n: i for i, n in enumerate(names)}


def _state_for(names: list[str], spec: ISASpec, seed: str | int,
               alias: dict[str, str]) -> tuple[list[int], list[int]]:
    """A pinned register state: values everywhere, taint on the operands the
    bank's instructions actually use.

    Values matter -- several rules take a cheaper path when an operand is
    provably clean or a mask provably zero -- so they are drawn once and pinned
    rather than left to whatever a run happens to produce.
    """
    from tests.taint_ir_bank import name_offset
    rng = random.Random(f'irperf:{seed}')
    sizes = _sizes(spec.arch)
    # Taint follows the OFFSET, not the name: the bank and the register file
    # spell the same lane differently.
    tainted_offs = {name_offset(spec.arch, alias.get(r.name, r.name))
                    for r in spec.regs[:4]}
    vals: list[int] = []
    tnts: list[int] = []
    for n in names:
        m = (1 << (sizes.get(n, 8) * 8)) - 1
        vals.append(rng.randint(1, MASK64) & m)
        hot = name_offset(spec.arch, n) in tainted_offs
        tnts.append((MASK64 if hot else 0) & m)
    return vals, tnts


_SIZE_CACHE: dict[str, dict[str, int]] = {}


def _sizes(arch: Architecture) -> dict[str, int]:
    key = arch.value if hasattr(arch, 'value') else str(arch)
    s = _SIZE_CACHE.get(key)
    if s is None:
        from microtaint.instrumentation.cell import _build_reg_maps
        _offsets, sz = _build_reg_maps(arch)[:2]
        s = dict(sz)
        _SIZE_CACHE[key] = s
    return s


def measure_isa(isa: str, spec: ISASpec, *, iters: int = 200000,
                opt: str = '-O3') -> dict[str, Any] | None:
    import time

    from microtaint.instrumentation.cell_c import taint_ir_c
    from microtaint.taint_ir.cbackend import bind_driver, compile_batch, emit_c, emit_driver
    from microtaint.taint_ir.exec import compile_program
    from microtaint.taint_ir.frompcode import Unsupported, build_ir
    from tests.perop_c_bank import _engine_names
    from tests.taint_ir_bank import slot_resolver

    # Slots must cover every register the ARCHITECTURE declares, not just the
    # ones the bank happens to list: a lifted instruction routinely reads a
    # register outside the bank's set (MIPS has dozens), and a program with an
    # unplaceable input is a program that cannot be run at all.
    names, slot = _slot_map(spec)
    alias = _engine_names(spec.arch, [r.name for r in spec.regs])
    slot_of = slot_resolver(spec.arch, slot)

    progs: list[Any] = []
    srcs: list[str] = []
    fnames: list[str] = []
    labels: list[str] = []
    declined = 0
    for ins in spec.instructions:
        try:
            p = build_ir(spec.arch, ins.bytes)
            src = emit_c(p, slot_of, f'tf_{len(progs)}')
        except (Unsupported, ValueError, KeyError):
            declined += 1
            continue
        except Exception:
            declined += 1
            continue
        fnames.append(f'tf_{len(progs)}')
        progs.append(p)
        srcs.append(src)
        labels.append(ins.label)
    if not progs:
        return None
    srcs.append(emit_driver(fnames))
    lib, compile_s = compile_batch(srcs, opt=opt)
    bench, call = bind_driver(lib)

    vals, tnts = _state_for(names, spec, isa, alias)
    # Memory accesses live in slots past the register file, so the state arrays
    # have to be long enough for the worst program in the batch -- a program
    # writing an address into a slot the caller never allocated would run off
    # the end of the array.
    from microtaint.taint_ir.frompcode import MEM_SLOTS_PER_ACCESS
    max_acc = max((len(p.accesses) for p in progs), default=0)
    total = len(names) + MEM_SLOTS_PER_ACCESS * (max_acc + 1)
    vals = vals + [0] * (total - len(vals))
    tnts = tnts + [0] * (total - len(tnts))
    V = (ctypes.c_uint64 * total)(*vals)
    T = (ctypes.c_uint64 * total)(*tnts)
    O = (ctypes.c_uint64 * total)()

    rows, mismatches = [], []
    n_jit, jit_compile_total = [0], [0.0]
    for i, (label, p) in enumerate(zip(labels, progs)):
        try:
            cap, _d = compile_program(p, slot_of)
        except KeyError:
            continue
        got_i = taint_ir_c.run(cap, list(vals), list(tnts))
        for k in range(total):
            O[k] = tnts[k]
        call(i, V, T, O)
        bad = []
        for k, _n in p.outputs:
            s = slot_of(k)
            if s is not None and O[s] != got_i[s]:
                bad.append(k)
        if bad:
            mismatches.append((label, bad))
        ns_c = bench(i, V, T, O, iters)
        ns_i, _sink = taint_ir_c.bench(cap, list(vals), list(tnts), max(2000, iters // 20))
        # The host emitter is checked against the interpreter on the same state
        # before it is timed: a code-generation bug that only shows on some
        # inputs is worth catching here rather than in a taint result.
        t0 = time.perf_counter()
        took = taint_ir_c.jit(cap)
        jit_compile = time.perf_counter() - t0
        ns_j = 0.0
        if took:
            if taint_ir_c.run(cap, list(vals), list(tnts)) != got_i:
                mismatches.append((label, ['jit']))
            ns_j, _s2 = taint_ir_c.bench(cap, list(vals), list(tnts), iters)
            n_jit[0] += 1
            jit_compile_total[0] += jit_compile
        rows.append((label, p.cost(), ns_c, ns_i, ns_j))
    return {'rows': rows, 'declined': declined, 'compile_s': compile_s,
            'n_progs': len(progs), 'mismatches': mismatches,
            'n_jit': n_jit[0], 'jit_compile_s': jit_compile_total[0]}


def _q(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def summarize(isa: str, res: dict[str, Any]) -> str:
    rows = res['rows']
    ops = [r[1] for r in rows]
    cns = [r[2] for r in rows]
    ins = [r[3] for r in rows]
    jns = [r[4] for r in rows if r[4] > 0]
    n = len(rows)
    jit = (f'jit mean={sum(jns)/len(jns):6.2f} p50={_q(jns,0.5):6.2f} ns '
           f'({res["n_jit"]}/{n}) | ' if jns else '')
    return (f'{isa:12s} n={n:4d} declined={res["declined"]:3d} | '
            f'ops mean={sum(ops)/n:5.1f} | '
            f'clang mean={sum(cns)/n:6.2f} p50={_q(cns,0.5):6.2f} '
            f'p95={_q(cns,0.95):6.2f} ns | '
            f'{jit}'
            f'interp mean={sum(ins)/n:7.1f} ns')


def main(argv: list[str] | None = None) -> int:
    import argparse

    from benchmark.instruction_bank import load_bank

    ap = argparse.ArgumentParser()
    ap.add_argument('--isas', nargs='*', default=None)
    ap.add_argument('--iters', type=int, default=200000)
    ap.add_argument('--top', type=int, default=0)
    ap.add_argument('--opt', default='-O3')
    ap.add_argument('--update', action='store_true')
    args = ap.parse_args(argv)

    specs = load_bank(isas=set(args.isas) if args.isas else None)
    out, total_c, total_i, total_n = {}, 0.0, 0.0, 0
    total_j, total_jn = 0.0, 0
    print('taint propagation per instruction, whole instruction, flags included')
    for isa, spec in sorted(specs.items()):
        res = measure_isa(isa, spec, iters=args.iters, opt=args.opt)
        if res is None:
            print(f'{isa:12s} no programs')
            continue
        print(summarize(isa, res))
        print(f'{"":12s} clang {args.opt}: {res["compile_s"]:.1f} s for '
              f'{res["n_progs"]} programs '
              f'({1000*res["compile_s"]/res["n_progs"]:.1f} ms each)')
        if res['mismatches']:
            print(f'{"":12s} *** {len(res["mismatches"])} compiled/interpreted '
                  f'MISMATCH: {res["mismatches"][:4]}')
        if args.top:
            worst = sorted(res['rows'], key=lambda r: -r[2])[:args.top]
            print(f'{"":12s} slowest: ' +
                  ', '.join(f'{r[0]}={r[2]:.1f}ns' for r in worst))
        print(f'{"":12s} host jit: {1e6*res["jit_compile_s"]/max(1, res["n_jit"]):.0f} us '
              f'per program (clang: '
              f'{1000*res["compile_s"]/res["n_progs"]:.1f} ms)')
        out[isa] = {lbl: [ops, round(ns_c, 3), round(ns_j, 3)]
                    for lbl, ops, ns_c, _ni, ns_j in res['rows']}
        total_c += sum(r[2] for r in res['rows'])
        total_i += sum(r[3] for r in res['rows'])
        total_j += sum(r[4] for r in res['rows'] if r[4] > 0)
        total_jn += sum(1 for r in res['rows'] if r[4] > 0)
        total_n += len(res['rows'])
    if total_n:
        jm = (total_j / total_jn) if total_jn else 0.0
        print(f'{"TOTAL":12s} n={total_n:4d} clang mean={total_c/total_n:.2f} ns '
              f'| host jit mean={jm:.2f} ns ({total_jn} programs) '
              f'| interpreted mean={total_i/total_n:.1f} ns')
    if args.update:
        BASELINE.write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
        print(f'baseline written: {BASELINE}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
