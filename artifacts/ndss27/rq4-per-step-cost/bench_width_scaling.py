#!/usr/bin/env python3
"""
Operand-width-scaling experiment for MicroTaint (Reviewer D4).

Claim under test: MicroTaint's per-step taint-propagation cost (one
circuit.evaluate call, the quantity Appendix D times) is INDEPENDENT of the
operand bit-width, because the CellIFT differential runs the native
instruction twice (on V|T and V&~T) and XORs the results -- an O(1)-in-width
operation -- rather than iterating over input/output bit pairs the way
TaintInduce's learned boolean rules do (O(n^2) in width).

We assemble the SAME opcode family (ADD, AND, XOR) at operand widths
8/16/32/64 bit (scalar GPR), 128 bit (SSE2 packed), and 256 bit (AVX2 packed),
synthesise each static rule ONCE, then time circuit.evaluate over many
iterations with a held-constant taint/value pattern, using
CLOCK_PROCESS_CPUTIME_ID. We report the median per-step latency per width and
flag which instructions stay on the native P-code differential vs. fall back
to Unicorn (or produce no rule at all).

Run with the microtaint venv:
    .venv/bin/python bench_width_scaling.py
Writes width_scaling_data.json next to itself.
"""
import json
import os
import statistics
import time

from keystone import Ks, KS_ARCH_X86, KS_MODE_64
from microtaint.types import Architecture, Register
from microtaint.simulator import CellSimulator
from microtaint.instrumentation.ast import EvalContext
from microtaint.sleigh.engine import generate_static_rule

HERE = os.path.dirname(os.path.abspath(__file__))
CLOCK = time.CLOCK_PROCESS_CPUTIME_ID

# --- timing parameters ---------------------------------------------------
TOTAL_ITERS = 100_000     # per (opcode,width) case
BATCHES = 25              # split into batches -> median per-batch per-call
WARMUP = 2_000

ks = Ks(KS_ARCH_X86, KS_MODE_64)


def asm(text: str) -> bytes:
    enc, _ = ks.asm(text, 0x1000)
    return bytes(enc)


arch = Architecture.AMD64
# use_unicorn=False -> native P-code differential path (the default, fastest).
sim = CellSimulator(arch)

GPR_SF = [Register('RAX', 64), Register('RBX', 64)]
XMM_SF = [Register('XMM0_LO', 64), Register('XMM0_HI', 64),
          Register('XMM1_LO', 64), Register('XMM1_HI', 64)]
YMM_SF = [Register('YMM0_LO', 64), Register('YMM0_HI', 64),
          Register('YMM1_LO', 64), Register('YMM1_HI', 64)]

# Held-constant taint/value pattern: destination fully tainted low-bit
# pattern, source untainted. Identical logical pattern at every width so the
# only thing that changes between rows is the operand width.
GPR_VALS = {'RAX': 0x0123456789ABCDEF, 'RBX': 0x0000000000000001}
GPR_TAINT = {'RAX': 0x0000000000000001, 'RBX': 0x0000000000000000}
XMM_VALS = {'XMM0_LO': 0x0123456789ABCDEF, 'XMM0_HI': 0x0123456789ABCDEF,
            'XMM1_LO': 0x0101010101010101, 'XMM1_HI': 0x0101010101010101}
XMM_TAINT = {'XMM0_LO': 0x1, 'XMM0_HI': 0x0, 'XMM1_LO': 0x0, 'XMM1_HI': 0x0}
YMM_VALS = {k.replace('XMM', 'YMM'): v for k, v in XMM_VALS.items()}
YMM_TAINT = {k.replace('XMM', 'YMM'): v for k, v in XMM_TAINT.items()}

# (family, width_bits, asm_text, state_format, values, taint)
CASES = [
    ('ADD', 8,   'add al, bl',              GPR_SF, GPR_VALS, GPR_TAINT),
    ('ADD', 16,  'add ax, bx',              GPR_SF, GPR_VALS, GPR_TAINT),
    ('ADD', 32,  'add eax, ebx',            GPR_SF, GPR_VALS, GPR_TAINT),
    ('ADD', 64,  'add rax, rbx',            GPR_SF, GPR_VALS, GPR_TAINT),
    ('ADD', 128, 'paddb xmm0, xmm1',        XMM_SF, XMM_VALS, XMM_TAINT),
    ('ADD', 256, 'vpaddb ymm0, ymm0, ymm1', YMM_SF, YMM_VALS, YMM_TAINT),

    ('AND', 8,   'and al, bl',              GPR_SF, GPR_VALS, GPR_TAINT),
    ('AND', 16,  'and ax, bx',              GPR_SF, GPR_VALS, GPR_TAINT),
    ('AND', 32,  'and eax, ebx',            GPR_SF, GPR_VALS, GPR_TAINT),
    ('AND', 64,  'and rax, rbx',            GPR_SF, GPR_VALS, GPR_TAINT),
    ('AND', 128, 'pand xmm0, xmm1',         XMM_SF, XMM_VALS, XMM_TAINT),
    ('AND', 256, 'vpand ymm0, ymm0, ymm1',  YMM_SF, YMM_VALS, YMM_TAINT),

    ('XOR', 8,   'xor al, bl',              GPR_SF, GPR_VALS, GPR_TAINT),
    ('XOR', 16,  'xor ax, bx',              GPR_SF, GPR_VALS, GPR_TAINT),
    ('XOR', 32,  'xor eax, ebx',            GPR_SF, GPR_VALS, GPR_TAINT),
    ('XOR', 64,  'xor rax, rbx',            GPR_SF, GPR_VALS, GPR_TAINT),
    ('XOR', 128, 'pxor xmm0, xmm1',         XMM_SF, XMM_VALS, XMM_TAINT),
    ('XOR', 256, 'vpxor ymm0, ymm0, ymm1',  YMM_SF, YMM_VALS, YMM_TAINT),
]


# Supplementary: fixed 128-bit width, varying SIMD lane count. Same total
# vector width (128 bit) in every row; only the number of independent
# element-add lanes changes (2/4/8/16). If per-step latency tracks lane count
# rather than the constant 128-bit width, that isolates lane-count -- not
# operand width -- as the driver of the packed-op cost.
LANE_CASES = [
    ('ADD-lanes', 128, 'paddq xmm0, xmm1', 2,  XMM_SF, XMM_VALS, XMM_TAINT),   # 2 x 64-bit
    ('ADD-lanes', 128, 'paddd xmm0, xmm1', 4,  XMM_SF, XMM_VALS, XMM_TAINT),   # 4 x 32-bit
    ('ADD-lanes', 128, 'paddw xmm0, xmm1', 8,  XMM_SF, XMM_VALS, XMM_TAINT),   # 8 x 16-bit
    ('ADD-lanes', 128, 'paddb xmm0, xmm1', 16, XMM_SF, XMM_VALS, XMM_TAINT),   # 16 x 8-bit
]


def fb_count() -> int:
    p = getattr(sim, '_pcode', None)
    return int(getattr(p, 'fallback_calls', 0)) if p is not None else 0


def time_case(circuit, ctx) -> dict:
    # warmup (also JIT-warms any per-circuit caching)
    for _ in range(WARMUP):
        circuit.evaluate(ctx)

    per_call = []
    batch = TOTAL_ITERS // BATCHES
    for _ in range(BATCHES):
        t0 = time.clock_gettime(CLOCK)
        for _ in range(batch):
            circuit.evaluate(ctx)
        t1 = time.clock_gettime(CLOCK)
        per_call.append((t1 - t0) / batch)  # seconds per evaluate
    per_call.sort()
    us = [x * 1e6 for x in per_call]
    return {
        'median_us': statistics.median(us),
        'min_us': min(us),
        'max_us': max(us),
        'mean_us': statistics.mean(us),
    }


def main():
    results = []
    for family, width, text, sf, vals, taint in CASES:
        row = {'family': family, 'width': width, 'asm': text}
        try:
            bs = asm(text)
            row['bytes'] = bs.hex()
        except Exception as e:
            row['status'] = f'ASM_FAIL: {e}'
            results.append(row)
            print(f"{family:3s} {width:4d}b  {text:26s}  ASM_FAIL {e}")
            continue
        try:
            circuit = generate_static_rule(arch, bs, sf)
        except Exception as e:
            row['status'] = f'RULE_FAIL: {type(e).__name__}: {e}'
            results.append(row)
            print(f"{family:3s} {width:4d}b  {text:26s}  RULE_FAIL {e}")
            continue

        n_assign = len(circuit.assignments)
        row['n_assignments'] = n_assign
        ctx = EvalContext(input_values=vals, input_taint=taint, simulator=sim)

        # correctness sample + fallback detection
        fb0 = fb_count()
        out = circuit.evaluate(ctx)
        fb1 = fb_count()
        fell_back = fb1 > fb0
        row['fallback'] = bool(fell_back)
        row['native'] = bool(n_assign > 0 and not fell_back)
        row['out_taint'] = {k: hex(v) for k, v in out.items()}

        if n_assign == 0:
            row['status'] = 'EMPTY_RULE (opcode not represented in state format -> unsupported)'
            row['median_us'] = None
            results.append(row)
            print(f"{family:3s} {width:4d}b  {text:26s}  EMPTY_RULE (unsupported, 0 assignments)")
            continue

        timing = time_case(circuit, ctx)
        row.update(timing)
        row['status'] = 'OK'
        results.append(row)
        tag = 'native' if row['native'] else 'FALLBACK->unicorn'
        print(f"{family:3s} {width:4d}b  {text:26s}  assigns={n_assign:2d} "
              f"{tag:18s}  median={timing['median_us']:.4f} us "
              f"(min={timing['min_us']:.4f} max={timing['max_us']:.4f})")

    lane_results = []
    print("\n--- Supplementary: fixed 128-bit width, varying lane count ---")
    for family, width, text, lanes, sf, vals, taint in LANE_CASES:
        row = {'family': family, 'width': width, 'asm': text, 'lanes': lanes}
        bs = asm(text)
        row['bytes'] = bs.hex()
        circuit = generate_static_rule(arch, bs, sf)
        n_assign = len(circuit.assignments)
        row['n_assignments'] = n_assign
        ctx = EvalContext(input_values=vals, input_taint=taint, simulator=sim)
        fb0 = fb_count()
        circuit.evaluate(ctx)
        fb1 = fb_count()
        row['fallback'] = bool(fb1 > fb0)
        row['native'] = bool(n_assign > 0 and fb1 == fb0)
        timing = time_case(circuit, ctx)
        row.update(timing)
        lane_results.append(row)
        print(f"{text:20s} lanes={lanes:2d}  assigns={n_assign:2d}  "
              f"median={timing['median_us']:.4f} us  "
              f"per-lane={timing['median_us']/lanes:.4f} us")

    meta = {
        'total_iters': TOTAL_ITERS,
        'batches': BATCHES,
        'clock': 'CLOCK_PROCESS_CPUTIME_ID',
        'use_unicorn_default': sim.use_unicorn,
        'arch': 'AMD64',
    }
    out_path = os.path.join(HERE, 'width_scaling_data.json')
    with open(out_path, 'w') as f:
        json.dump({'meta': meta, 'results': results, 'lane_results': lane_results}, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
