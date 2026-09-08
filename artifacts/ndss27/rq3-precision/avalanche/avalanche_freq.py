#!/usr/bin/env python3
"""
avalanche_freq.py  --  SCRATCH instrumentation harness (does NOT modify the
tracked engine).  Measures, for a real program run under MicroTaint's Qiling
emulation:

  1. the DYNAMIC frequency of each InstructionCategory the classifier assigns
     to executed *tainted* instructions, and
  2. the fraction of tainted OUTPUT bits that are attributable to the
     AVALANCHE fallback (i.e. bits that would NOT be tainted if the avalanche
     nodes were removed from the propagation rule).

Injection points (all monkeypatches, no edits to the engine source):

  * microtaint.sleigh.engine.generate_taint_assignments
        -- wrapped so that, for every TaintAssignment the engine builds, we
           record the InstructionCategory the classifier (determine_category)
           assigned to it.  This is the exact call the engine itself makes at
           the top of the category dispatch.  Keyed by id(assignment); rules
           are cached per instruction-bytes so this runs once per unique insn.

  * microtaint.emulator.wrapper._cached_generate_static_rule
        -- wrapped to return a thin CircuitProxy.  The wrapper evaluates
           circuit.evaluate(ctx) once per executed instruction; the proxy
           intercepts that call so we get (circuit, EvalContext, output) and
           can (a) tally which category fired and (b) re-evaluate each
           assignment's expression tree with the avalanche nodes forced to
           zero, giving a bit-exact avalanche-vs-precise attribution.

The Cython hot-path hook and the per-address memo cache are disabled via the
engine's own env-var switches so that every executed instruction flows through
the introspectable Python evaluate() path.
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from collections import defaultdict

# Force the introspectable, non-cached Python evaluation path.  These are the
# engine's own documented debug switches (wrapper.py __init__).
os.environ['MICROTAINT_DISABLE_CYTHON_HOOK'] = '1'
os.environ['MICROTAINT_DISABLE_INSTR_CACHE'] = '1'

from qiling import Qiling
from qiling.const import QL_VERBOSE

from microtaint.emulator.reporter import Reporter
from microtaint.emulator.wrapper import MicrotaintWrapper
from microtaint.sleigh import engine as _engine
from microtaint.sleigh.mapper import determine_category

# ---------------------------------------------------------------------------
# Global tallies
# ---------------------------------------------------------------------------
CAT_BY_ID: dict[int, str] = {}          # id(TaintAssignment) -> category label

# per-category, counted per tainted OUTPUT assignment (one tainted reg/mem write)
asg_count: dict[str, int] = defaultdict(int)
asg_out_bits: dict[str, int] = defaultdict(int)      # total tainted output bits
asg_aval_bits: dict[str, int] = defaultdict(int)     # avalanche-exclusive output bits

# per-category, counted per executed tainted INSTRUCTION (dominant category)
insn_count: dict[str, int] = defaultdict(int)

STATS = {
    'insns_hooked': 0,          # main-binary instructions evaluated (post taint-arm)
    'insns_tainted': 0,         # instructions producing >=1 tainted output bit
    'total_out_bits': 0,        # sum over tainted assignments of output-bit popcount
    'total_aval_bits': 0,       # sum of avalanche-exclusive output bits
    # data-register-only view (output width >= 8 bits; excludes 1-bit CPU flags)
    'data_out_bits': 0,
    'data_aval_bits': 0,
    'flag_out_bits': 0,
    'flag_aval_bits': 0,
    'budget_hit': False,
}

_BUDGET = [0]          # max tainted instructions (0 = unlimited)
_WRAPPER = [None]      # set to the MicrotaintWrapper so we can emu_stop


# ---------------------------------------------------------------------------
# 1. Category recording:  wrap generate_taint_assignments
# ---------------------------------------------------------------------------
_orig_gta = _engine.generate_taint_assignments


def _gta_wrapper(arch, bytestring, assignments, slice_ops, dep_set, out_target,
                 out_name, out_bit_start, out_bit_end, mapper, mapping=None,
                 has_cbranch=False, cbranch_flag_deps=None, is_bit_count=False,
                 is_software_loop=False):
    old_len = len(assignments)
    width = out_bit_end - out_bit_start + 1
    # Reproduce the classifier verdict exactly as the engine's dispatch does.
    is_store = hasattr(out_target, 'address_expr')
    try:
        cat = determine_category(slice_ops, out_width_bits=width)
        label = str(cat)
    except Exception:
        label = None
    if is_software_loop:
        label = 'Avalanche'          # engine forces AVALANCHE for BMI2 loops
    elif label is None:
        label = 'Mapped' if is_store else 'Unknown'

    _orig_gta(arch, bytestring, assignments, slice_ops, dep_set, out_target,
              out_name, out_bit_start, out_bit_end, mapper, mapping,
              has_cbranch=has_cbranch, cbranch_flag_deps=cbranch_flag_deps,
              is_bit_count=is_bit_count, is_software_loop=is_software_loop)

    for a in assignments[old_len:]:
        CAT_BY_ID[id(a)] = label


_engine.generate_taint_assignments = _gta_wrapper


# ---------------------------------------------------------------------------
# 2. Avalanche bit attribution:  precise-only re-evaluation of an expr tree
# ---------------------------------------------------------------------------
# The engine builds each assignment's taint expression as a tree of BinaryExpr
# / UnaryExpr over leaves (TaintOperand, Constant, InstructionCellExpr,
# MemoryDifferentialExpr, MemoryOperand) plus AvalancheExpr / FullMaskAvalanche
# fallback nodes.  We mirror the (bitwise) semantics but return 0 for every
# avalanche node.  The result is the taint mask the rule would produce WITHOUT
# any avalanche contribution.  full & ~precise == the bits that exist only
# because of the avalanche fallback.

def _eval_precise(e, ctx):
    tn = type(e).__name__
    if tn in ('AvalancheExpr', 'FullMaskAvalancheExpr'):
        return 0
    if tn == 'BinaryExpr':
        op = e.op
        opn = op.name if hasattr(op, 'name') else str(op)
        l = _eval_precise(e.lhs, ctx)
        r = _eval_precise(e.rhs, ctx)
        if opn == 'AND':
            return l & r
        if opn == 'OR':
            return l | r
        if opn == 'XOR':
            return l ^ r
        if opn == 'NOT':
            return ~l
        if opn == 'LEFT':
            return (l << r) if r >= 0 else 0
        if opn in ('ADD', 'SUB'):
            # memory-offset arithmetic, not taint logic; fall back to concrete
            return e.evaluate(ctx)
        return e.evaluate(ctx)
    if tn == 'UnaryExpr':
        op = e.op
        opn = op.name if hasattr(op, 'name') else str(op)
        v = _eval_precise(e.expr, ctx)
        if opn == 'NOT':
            return ~v
        return e.evaluate(ctx)
    # opaque leaf: no avalanche inside, evaluate concretely
    return e.evaluate(ctx)


def _tally(circuit, ctx, _out):
    insn_cats = []   # (cat, out_bits, aval_bits) for tainted assignments
    for a in circuit.assignments:
        expr = a.expression
        if expr is None:
            continue
        tgt = a.target
        if hasattr(tgt, 'address_expr'):
            width = tgt.size * 8
        else:
            width = tgt.bit_end - tgt.bit_start + 1
        mask = (1 << width) - 1
        try:
            full = expr.evaluate(ctx) & mask
        except Exception:
            continue
        if full == 0:
            continue
        try:
            precise = _eval_precise(expr, ctx) & mask
        except Exception:
            precise = full  # be conservative: attribute nothing to avalanche
        aval = full & (~precise & mask)
        out_bits = full.bit_count()
        aval_bits = aval.bit_count()
        cat = CAT_BY_ID.get(id(a), 'Unknown')
        asg_count[cat] += 1
        asg_out_bits[cat] += out_bits
        asg_aval_bits[cat] += aval_bits
        STATS['total_out_bits'] += out_bits
        STATS['total_aval_bits'] += aval_bits
        if width >= 8:
            STATS['data_out_bits'] += out_bits
            STATS['data_aval_bits'] += aval_bits
        else:
            STATS['flag_out_bits'] += out_bits
            STATS['flag_aval_bits'] += aval_bits
        insn_cats.append((cat, out_bits, aval_bits))

    if insn_cats:
        STATS['insns_tainted'] += 1
        dom = max(insn_cats, key=lambda t: t[1])[0]
        insn_count[dom] += 1
        if _BUDGET[0] and STATS['insns_tainted'] >= _BUDGET[0]:
            STATS['budget_hit'] = True
            w = _WRAPPER[0]
            if w is not None:
                try:
                    w.ql.emu_stop()
                except Exception:
                    pass


class CircuitProxy:
    __slots__ = ('_c',)

    def __init__(self, c):
        self._c = c

    @property
    def _compiled(self):
        return self._c._compiled

    @property
    def assignments(self):
        return self._c.assignments

    def evaluate(self, ctx):
        STATS['insns_hooked'] += 1
        out = self._c.evaluate(ctx)
        try:
            _tally(self._c, ctx, out)
        except Exception:
            pass
        return out


_orig_cached = _engine._cached_generate_static_rule


def _cached_proxy(arch, bytestring, state_format_tuple):
    return CircuitProxy(_orig_cached(arch, bytestring, state_format_tuple))


# Patch the name the wrapper actually calls (imported into wrapper's namespace)
import microtaint.emulator.wrapper as _wrap_mod
_wrap_mod._cached_generate_static_rule = _cached_proxy
# also clear any pre-existing cache so category recording sees every rule
try:
    _orig_cached.cache_clear()
except Exception:
    pass


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
class PreloadStdin:
    """Minimal stdin stream: preloaded, tainted-on-read, tolerant of the
    fd lifecycle calls Qiling makes (close/fileno/seek/flush)."""

    def __init__(self, data: bytes):
        self._buf = bytearray(data)

    def read(self, count: int) -> bytes:
        chunk = bytes(self._buf[:count])
        del self._buf[:count]
        return chunk

    def write(self, data: bytes) -> None:
        self._buf.extend(data)

    def close(self):
        return None

    def fileno(self):
        return 0

    def seek(self, *a, **k):
        return 0

    def tell(self):
        return 0

    def flush(self):
        return None

    def readline(self, *a, **k):
        nl = self._buf.find(b'\n')
        if nl < 0:
            chunk = bytes(self._buf); del self._buf[:]; return chunk
        chunk = bytes(self._buf[:nl + 1]); del self._buf[:nl + 1]; return chunk


def run(binary, binary_args, stdin_data, rootfs='/', budget=0, taint_file=None):
    _BUDGET[0] = budget
    reporter = Reporter(json_mode=False, stream=sys.stderr)
    ql = Qiling([binary, *binary_args], rootfs, verbose=QL_VERBOSE.OFF)
    ql.os.stdin = PreloadStdin(stdin_data)
    wrapper = MicrotaintWrapper(ql, check_bof=True, check_uaf=False,
                                check_sc=False, check_aiw=False, reporter=reporter)
    _WRAPPER[0] = wrapper
    # sanity: ensure Python path really is in force
    assert wrapper._disable_cython_hook, 'cython hook not disabled!'
    assert not wrapper._instr_cache_enabled, 'instr cache not disabled!'

    if taint_file is not None:
        # taint a file's contents instead of stdin: preload as an fd-backed read
        pass
    try:
        ql.run()
    except Exception as exc:
        print(f'[!] execution halted: {exc}', file=sys.stderr)
    return wrapper


def _print_report(title):
    print('\n' + '=' * 72)
    print(title)
    print('=' * 72)
    print(f"main-binary instructions evaluated (after taint armed): {STATS['insns_hooked']:,}")
    print(f"executed TAINTED instructions:                          {STATS['insns_tainted']:,}")
    if STATS['budget_hit']:
        print('   (stopped early: tainted-instruction budget reached)')
    print(f"total tainted output bits:                              {STATS['total_out_bits']:,}")
    print(f"avalanche-exclusive output bits:                        {STATS['total_aval_bits']:,}")
    if STATS['total_out_bits']:
        share = 100.0 * STATS['total_aval_bits'] / STATS['total_out_bits']
        print(f"AVALANCHE share of ALL tainted output bits:             {share:.2f}%")
    if STATS['data_out_bits']:
        ds = 100.0 * STATS['data_aval_bits'] / STATS['data_out_bits']
        print(f"  data registers only (>=8-bit outputs):  {STATS['data_aval_bits']:,}/{STATS['data_out_bits']:,}  ({ds:.2f}%)")
    if STATS['flag_out_bits']:
        fs = 100.0 * STATS['flag_aval_bits'] / STATS['flag_out_bits']
        print(f"  CPU flag bits only (1-bit outputs):     {STATS['flag_aval_bits']:,}/{STATS['flag_out_bits']:,}  ({fs:.2f}%)")

    print('\nPer-category DYNAMIC frequency (by executed tainted instruction, dominant category):')
    tot_i = sum(insn_count.values()) or 1
    for cat, n in sorted(insn_count.items(), key=lambda kv: -kv[1]):
        print(f'   {cat:<30} {n:>10,}  {100.0*n/tot_i:6.2f}%')

    print('\nPer-category (by tainted OUTPUT assignment) + avalanche bit share:')
    print(f"   {'category':<30} {'assigns':>9} {'freq%':>7} {'outbits':>10} {'avalbits':>9} {'aval%':>7}")
    tot_a = sum(asg_count.values()) or 1
    for cat in sorted(asg_count, key=lambda c: -asg_count[c]):
        n = asg_count[cat]
        ob = asg_out_bits[cat]
        ab = asg_aval_bits[cat]
        avp = (100.0 * ab / ob) if ob else 0.0
        print(f'   {cat:<30} {n:>9,} {100.0*n/tot_a:6.2f}% {ob:>10,} {ab:>9,} {avp:6.2f}%')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=int, default=0,
                    help='stop after N tainted instructions (0=run to completion)')
    ap.add_argument('--stdin', default='hello world\n',
                    help='literal stdin string to taint')
    ap.add_argument('--stdin-bytes', default=None,
                    help='hex string of stdin bytes to taint (overrides --stdin)')
    ap.add_argument('--rootfs', default='/')
    ap.add_argument('--title', default='RESULT')
    ap.add_argument('--json-out', default=None)
    ap.add_argument('binary')
    ap.add_argument('binary_args', nargs=argparse.REMAINDER)
    args = ap.parse_args()

    if args.stdin_bytes:
        data = bytes.fromhex(args.stdin_bytes)
    else:
        data = args.stdin.encode()

    ba = args.binary_args
    if ba and ba[0] == '--':
        ba = ba[1:]

    run(args.binary, ba, data, rootfs=args.rootfs, budget=args.budget)
    _print_report(args.title)

    if args.json_out:
        blob = {
            'title': args.title,
            'binary': args.binary,
            'stats': STATS,
            'insn_count': dict(insn_count),
            'asg_count': dict(asg_count),
            'asg_out_bits': dict(asg_out_bits),
            'asg_aval_bits': dict(asg_aval_bits),
        }
        with open(args.json_out, 'w') as fh:
            json.dump(blob, fh, indent=2)
        print(f'\n[+] wrote {args.json_out}')


if __name__ == '__main__':
    main()
