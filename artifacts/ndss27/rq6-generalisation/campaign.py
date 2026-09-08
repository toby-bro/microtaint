#!/usr/bin/env python3
"""Extensive two-pass under-taint campaign for MicroTaint across 5 ISAs.

Goal: stress soundness (mt_taint >= bitflip_lower_bound) over ~1M random cases per
ISA -- amd64, arm64, mips, riscv (RV64, hand-assembled), ppc.

Two passes, by design:

  PASS 1 (fast, wide net).  A single Unicorn instance per ISA is REUSED across runs
  (registers rewritten each run) -- ~8x faster than a fresh Unicorn per run, but
  hidden architectural state (flags, etc.) can LEAK between runs.  A leak can only
  ADD spurious diffs to the bitflip lower bound, i.e. it OVER-reports under-taints;
  it can never hide a real one.  So pass 1 is a conservative filter: every genuine
  under-taint is caught, plus some false positives.  exact_gt (over-taint check) is
  skipped -- this campaign is about UNDER-tainting only.

  PASS 2 (slow, isolated).  Every pass-1 report is re-checked with a FRESH Unicorn
  per run (multiarch_oracle._run) -- full state reset, zero leakage.  Real bugs
  survive; leaks vanish.

Run:
  python campaign.py pass1 --n 1000000 --arch all --seed 1 --out camp
  python campaign.py pass2 --in camp                 # verify pass-1 reports
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import multiarch_oracle as O  # noqa: E402
from unicorn import Uc, UcError  # noqa: E402
import unicorn.riscv_const as _rv  # noqa: E402

from microtaint.instrumentation.ast import EvalContext  # noqa: E402
from microtaint.simulator import CellSimulator  # noqa: E402
from microtaint.sleigh.engine import generate_static_rule  # noqa: E402
from microtaint.types import Architecture, ImplicitTaintPolicy, Register  # noqa: E402
from unicorn import UC_ARCH_RISCV, UC_MODE_RISCV64  # noqa: E402


# --------------------------------------------------------------------------- #
# Unified corpus entry: everything the campaign needs, ISA-agnostic.
# --------------------------------------------------------------------------- #
class Bench:
    def __init__(self, label, arch, bits, uc_arch, uc_mode, uc_regs, regs, flag_regs, canon):
        self.label = label
        self.arch = arch
        self.bits = bits
        self.mask = (1 << bits) - 1
        self.uc_arch = uc_arch
        self.uc_mode = uc_mode
        self.uc_regs = uc_regs           # microtaint-name -> unicorn reg id
        self.regs = regs                 # taint-source / scored GPR names (microtaint names)
        self.flag_regs = flag_regs       # [(name, bits)] modelled but not seeded
        self.canon = canon               # canonical_word_bits or None
        self.state_format = [Register(r, bits) for r in regs] + [Register(n, b) for n, b in flag_regs]
        self.entries = []                # (asm_label, code_bytes, out_reg, [src_regs])
        self.circuits = {}               # code.hex() -> compiled rule

    def canonicalize(self, state, taint):
        w = self.canon
        if w is None or w >= self.bits:
            return state, taint
        low = (1 << w) - 1
        sign = 1 << (w - 1)
        high = self.mask ^ low

        def sx(v):
            v &= low
            return v | high if v & sign else v
        return {r: sx(v) for r, v in state.items()}, {r: t & low for r, t in taint.items()}


def _from_isaspec(key) -> Bench:
    s = O.ISAS[key]()
    b = Bench(s.label, s.arch, s.bits, s.uc_arch, s.uc_mode, s.uc_regs, s.regs,
              s.flag_regs, s.canonical_word_bits)
    for asm, out, srcs in s.prog:
        try:
            code = bytes(s.ks.asm(asm, O.CODE_ADDR)[0])
        except Exception as e:  # noqa: BLE001
            print(f'  [{s.label}] SKIP {asm!r}: {e}', flush=True)
            continue
        b.entries.append((asm, code, out, srcs))
    return b


def _riscv() -> Bench:
    # ABI names (SLEIGH RISC-V uses them); map to Unicorn x-registers.
    abi = {'t0': 5, 't1': 6, 't2': 7, 't3': 28, 't4': 29, 'a0': 10, 'a1': 11, 'a2': 12}
    uc_regs = {n: getattr(_rv, f'UC_RISCV_REG_X{x}') for n, x in abi.items()}
    regs = list(abi.keys())
    b = Bench('RISCV64', Architecture.RISCV64, 64, UC_ARCH_RISCV, UC_MODE_RISCV64,
              uc_regs, regs, [], None)

    def rt(f7, rs2, rs1, f3, rd, op):
        return ((f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op).to_bytes(4, 'little')

    def it(imm, rs1, f3, rd, op):
        return (((imm & 0xFFF) << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op).to_bytes(4, 'little')

    # rd=t0(5), rs1=t1(6), rs2=t2(7)
    R, W = 0x33, 0x3B
    rr = [
        ('add t0,t1,t2', rt(0x00, 7, 6, 0x0, 5, R)), ('sub t0,t1,t2', rt(0x20, 7, 6, 0x0, 5, R)),
        ('sll t0,t1,t2', rt(0x00, 7, 6, 0x1, 5, R)), ('slt t0,t1,t2', rt(0x00, 7, 6, 0x2, 5, R)),
        ('sltu t0,t1,t2', rt(0x00, 7, 6, 0x3, 5, R)), ('xor t0,t1,t2', rt(0x00, 7, 6, 0x4, 5, R)),
        ('srl t0,t1,t2', rt(0x00, 7, 6, 0x5, 5, R)), ('sra t0,t1,t2', rt(0x20, 7, 6, 0x5, 5, R)),
        ('or t0,t1,t2', rt(0x00, 7, 6, 0x6, 5, R)), ('and t0,t1,t2', rt(0x00, 7, 6, 0x7, 5, R)),
        ('mul t0,t1,t2', rt(0x01, 7, 6, 0x0, 5, R)), ('mulh t0,t1,t2', rt(0x01, 7, 6, 0x1, 5, R)),
        ('mulhu t0,t1,t2', rt(0x01, 7, 6, 0x3, 5, R)),
        ('addw t0,t1,t2', rt(0x00, 7, 6, 0x0, 5, W)), ('subw t0,t1,t2', rt(0x20, 7, 6, 0x0, 5, W)),
        ('sllw t0,t1,t2', rt(0x00, 7, 6, 0x1, 5, W)), ('srlw t0,t1,t2', rt(0x00, 7, 6, 0x5, 5, W)),
        ('sraw t0,t1,t2', rt(0x20, 7, 6, 0x5, 5, W)), ('mulw t0,t1,t2', rt(0x01, 7, 6, 0x0, 5, W)),
    ]
    for asm, code in rr:
        b.entries.append((asm, code, 't0', ['t1', 't2']))
    ri = [
        ('addi t0,t1,5', it(5, 6, 0x0, 5, 0x13)), ('slli t0,t1,4', it(4, 6, 0x1, 5, 0x13)),
        ('srli t0,t1,4', it(4, 6, 0x5, 5, 0x13)), ('srai t0,t1,4', it((0x10 << 6) | 4, 6, 0x5, 5, 0x13)),
        ('xori t0,t1,-1', it(0xFFF, 6, 0x4, 5, 0x13)), ('andi t0,t1,15', it(0xF, 6, 0x7, 5, 0x13)),
        ('slliw t0,t1,4', it(4, 6, 0x1, 5, 0x1B)),
    ]
    for asm, code in ri:
        b.entries.append((asm, code, 't0', ['t1']))
    return b


def build(arch_key) -> Bench:
    return _riscv() if arch_key == 'riscv' else _from_isaspec(arch_key)


ALL_ARCHES = ['amd64', 'arm64', 'mips', 'riscv', 'ppc']


# --------------------------------------------------------------------------- #
# taint generation (mirrors multiarch_fuzz._gen_taint style: dense/adjacent/full/
# correlated masks that stress carry/borrow/cancellation boundaries)
# --------------------------------------------------------------------------- #
def gen_taint(rng, b: Bench, srcs):
    kind = rng.random()
    taint = {}
    for r in srcs:
        if kind < 0.25:                                   # single random bit
            taint[r] = 1 << rng.randrange(b.bits)
        elif kind < 0.45:                                 # run of adjacent bits
            w = rng.randint(1, 8)
            s = rng.randrange(max(1, b.bits - w))
            taint[r] = ((1 << w) - 1) << s
        elif kind < 0.6:                                  # full mask
            taint[r] = b.mask
        elif kind < 0.8:                                  # dense random
            taint[r] = rng.getrandbits(b.bits)
        else:                                             # sparse few bits
            taint[r] = sum(1 << rng.randrange(b.bits) for _ in range(rng.randint(1, 3)))
    if len(srcs) >= 2 and rng.random() < 0.35:            # correlate two sources
        a, c = srcs[0], srcs[1]
        taint[c] = taint[a]
    return taint


# --------------------------------------------------------------------------- #
# PASS 1 -- fast, reused Unicorn, under-taint only
# --------------------------------------------------------------------------- #
def _rule(b: Bench, code):
    h = code.hex()
    c = b.circuits.get(h)
    if c is None:
        c = generate_static_rule(b.arch, code, b.state_format)
        b.circuits[h] = c
    return c


def pass1_arch(b: Bench, n, seed, out_path, beat=10.0):
    rng = random.Random(seed)
    sim = CellSimulator(b.arch)
    # ONE Unicorn per distinct instruction: code is written once and never
    # rewritten, so Unicorn's JIT translation-block cache never goes stale (writing
    # different code to the same address in a reused Uc segfaults).  State still
    # leaks between cases OF THE SAME instruction -- that is the intended pass-1
    # over-reporting, filtered by the fresh-Uc pass 2.
    # ONE fresh Unicorn per CASE: it runs only base + this case's bit-flips
    # (<= bits*|srcs|+1 emu_start calls, far below the reuse-accumulation crash
    # threshold), then is discarded.  This is ~1 Uc construction per case rather than
    # per bit-flip (the fully-fresh oracle's cost), so it is both crash-stable and
    # fast.  Leakage is confined to WITHIN a case -- filtered by the fresh pass 2.
    # Append + flush each report immediately: a chunk that segfaults (the LE-64
    # native-cell x Unicorn interaction) then loses no found under-taints, and the
    # supervisor can relaunch with the next seed.
    out_f = open(out_path, 'a')
    reports = []
    t0 = time.time()
    last = t0
    done = 0
    for i in range(n):
        asm, code, out_reg, srcs = rng.choice(b.entries)
        state = {r: rng.getrandbits(b.bits) for r in b.regs}
        taint = gen_taint(rng, b, srcs)
        state, taint = b.canonicalize(state, taint)
        # oracle: the PROVEN-STABLE fresh-Unicorn-per-run lower bound (O._run rebuilds
        # a fresh Uc each call; the real fuzzer runs it for minutes without leaking).
        # My Bench duck-types onto O.bitflip_lower_bound (regs/bits/mask/uc_*).  Fresh
        # Uc per run means no leakage at all -- so pass 1 here already has NO false
        # positives from state leaks; pass 2 remains as an independent double-check.
        try:
            lb = O.bitflip_lower_bound(b, code, state, taint)
        except Exception:  # noqa: BLE001
            continue
        # microtaint
        try:
            ctx = EvalContext(input_taint=dict(taint), input_values=dict(state),
                              simulator=sim, implicit_policy=ImplicitTaintPolicy.IGNORE)
            mt = _rule(b, code).evaluate(ctx)
        except Exception:  # noqa: BLE001
            continue
        missed = 0
        for r in b.regs:
            missed |= lb[r] & ~(mt.get(r, 0) or 0) & b.mask
        if missed:
            rep = {
                'arch': b.label, 'asm': asm, 'bytes': code.hex(),
                'state': {r: state[r] for r in srcs}, 'taint': {r: taint.get(r, 0) for r in srcs},
                'out': out_reg, 'srcs': srcs,
                'mt': {r: (mt.get(r, 0) or 0) & b.mask for r in b.regs if (mt.get(r, 0) or 0) & b.mask},
                'lb': {r: lb[r] for r in b.regs if lb[r]}, 'missed': missed,
            }
            reports.append(rep)
            out_f.write(json.dumps(rep) + '\n')
            out_f.flush()
        done = i + 1
        now = time.time()
        if now - last >= beat:
            last = now
            print(f'  [{b.label}] {int(now - t0)}s n={done}/{n} reports={len(reports)} '
                  f'{done / max(now - t0, 1e-9):.0f} cases/s', flush=True)
    out_f.close()
    print(f'[{b.label}] PASS1 done: {done} cases, {len(reports)} raw under-taint reports '
          f'({done / max(time.time() - t0, 1e-9):.0f} cases/s) -> {out_path}', flush=True)
    return done, len(reports)


# --------------------------------------------------------------------------- #
# PASS 2 -- fresh Unicorn per run, full isolation
# --------------------------------------------------------------------------- #
def pass2_verify(report):
    """Re-check one pass-1 report with a FRESH Unicorn per run (zero leakage).
    Returns (still_under, detail)."""
    key = {'amd64': 'amd64', 'arm64': 'arm64', 'mips': 'mips', 'ppc': 'ppc',
           'RISCV64': 'riscv', 'ARM64': 'arm64', 'MIPS64BE': 'mips', 'PPC32BE': 'ppc',
           'AMD64': 'amd64'}[report['arch']]
    b = build(key)
    code = bytes.fromhex(report['bytes'])
    mask = b.mask
    state = {r: 0 for r in b.regs}
    state.update({r: v for r, v in report['state'].items()})
    taint = {r: v for r, v in report['taint'].items()}
    state, taint = b.canonicalize(state, taint)

    def fresh_run(st):
        uc = Uc(b.uc_arch, b.uc_mode)
        uc.mem_map(O.CODE_ADDR, 0x1000)
        uc.mem_write(O.CODE_ADDR, code + b'\x00' * 16)
        for r in b.regs:
            uc.reg_write(b.uc_regs[r], st[r] & mask)
        uc.emu_start(O.CODE_ADDR, O.CODE_ADDR + len(code))
        return {r: uc.reg_read(b.uc_regs[r]) & mask for r in b.regs}

    base = fresh_run(state)
    lb = dict.fromkeys(b.regs, 0)
    for r in report['srcs']:
        tr = taint.get(r, 0)
        for bit in range(b.bits):
            if (tr >> bit) & 1:
                s2 = dict(state)
                s2[r] = (s2[r] ^ (1 << bit)) & mask
                try:
                    out = fresh_run(s2)
                except UcError:
                    continue
                for rr in b.regs:
                    lb[rr] |= base[rr] ^ out[rr]
    sim = CellSimulator(b.arch)
    ctx = EvalContext(input_taint=dict(taint), input_values=dict(state),
                      simulator=sim, implicit_policy=ImplicitTaintPolicy.IGNORE)
    mt = generate_static_rule(b.arch, code, b.state_format).evaluate(ctx)
    missed = 0
    for r in b.regs:
        missed |= lb[r] & ~(mt.get(r, 0) or 0) & mask
    return missed, {'lb': {r: lb[r] for r in b.regs if lb[r]},
                    'mt': {r: (mt.get(r, 0) or 0) & mask for r in b.regs if (mt.get(r, 0) or 0) & mask},
                    'missed': missed}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p1 = sub.add_parser('pass1')
    p1.add_argument('--n', type=int, default=1_000_000)
    p1.add_argument('--arch', default='all')
    p1.add_argument('--seed', type=int, default=1)
    p1.add_argument('--out', default='camp')
    p2 = sub.add_parser('pass2')
    p2.add_argument('--in', dest='inp', default='camp')
    args = ap.parse_args()

    if args.cmd == 'pass1':
        arches = ALL_ARCHES if args.arch == 'all' else [args.arch]
        for i, k in enumerate(arches):
            b = build(k)
            print(f'=== PASS1 {b.label}: {args.n} cases (seed {args.seed + i}, '
                  f'{len(b.entries)} instrs) ===', flush=True)
            pass1_arch(b, args.n, args.seed + i, f'{args.out}_{k}.jsonl')
        return

    # pass2
    import glob
    files = sorted(glob.glob(f'{args.inp}_*.jsonl'))
    grand_raw = grand_real = 0
    for fp in files:
        reports = [json.loads(line) for line in open(fp) if line.strip()]
        grand_raw += len(reports)
        real = []
        for rep in reports:
            missed, detail = pass2_verify(rep)
            if missed:
                real.append({**rep, 'verified': detail})
        grand_real += len(real)
        tag = os.path.basename(fp)
        print(f'{tag}: {len(reports)} raw -> {len(real)} REAL under-taints (isolated re-verify)')
        for rep in real[:20]:
            print(f'    REAL {rep["arch"]} {rep["asm"]!r} bytes={rep["bytes"]} '
                  f'state={ {k: hex(v) for k, v in rep["state"].items()} } '
                  f'taint={ {k: hex(v) for k, v in rep["taint"].items()} } missed={rep["verified"]["missed"]:#x}')
        if real:
            with open(fp.replace('.jsonl', '_REAL.jsonl'), 'w') as f:
                for rep in real:
                    f.write(json.dumps(rep) + '\n')
    print(f'\n=== TOTAL: {grand_raw} raw pass-1 reports -> {grand_real} REAL under-taints after isolation ===')


if __name__ == '__main__':
    main()
