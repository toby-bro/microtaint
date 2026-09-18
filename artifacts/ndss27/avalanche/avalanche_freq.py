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

# Experiment script, not library code: see artifacts/ndss27/README.md,
# "Lint and type checking", for why annotations are not required here.
# mypy: disable-error-code="no-untyped-def, no-untyped-call, type-arg, explicit-any"

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import defaultdict
from typing import Any

# Force the introspectable, non-cached Python evaluation path.  These are the
# engine's own documented debug switches (wrapper.py __init__).
os.environ['MICROTAINT_DISABLE_CYTHON_HOOK'] = '1'
os.environ['MICROTAINT_DISABLE_INSTR_CACHE'] = '1'

import exprwalk as W  # type: ignore[import-not-found]  # sibling script, resolved at run time
from qiling import Qiling
from qiling.const import QL_VERBOSE

from microtaint.emulator.reporter import Reporter
from microtaint.emulator.wrapper import MicrotaintWrapper
from microtaint.sleigh import engine as _engine
from microtaint.sleigh.mapper import determine_category

# ---------------------------------------------------------------------------
# Global tallies
# ---------------------------------------------------------------------------
# (instruction bytes, index within the circuit) -> classifier label.
# The old key was id(TaintAssignment).  A cdef object's id is its address, which
# is reused after a free, so a later assignment could inherit a dead one's label
# -- a silent misattribution, strictly worse than the visible 'Unknown' it also
# produced.  Bytes+index is deterministic and cannot collide.
CAT_BY_KEY: dict[tuple[str, int], str] = {}

# Of the bits owed to approximation, which mechanism produced them.  The old
# harness could only answer "AvalancheExpr or not", which stopped being the whole
# story once multiply and variable-shift moved into their own terms.
mech_bits: dict[str, int] = defaultdict(int)

# Instructions whose classifier verdict and whose emitted tree disagree about
# whether they approximate.  This is the base64 contradiction (Avalanche on 0% of
# instructions, yet a large share of bits) made into a detectable event.
DISAGREE: dict[str, int] = defaultdict(int)
_DUMPED: list = []

# The engine's enum and the paper's taxonomy are the same six categories under
# two different names.  mapper.py's own comment says a slice combining with XOR
# "must be WELDABLE (the OR of input taints)" and then returns ORABLE; and
# COND_TRANSPORTABLE holds exactly the equality opcodes the paper lists under
# "Transportable (Eq)".  Report the paper's names so the artifact and the table
# cannot be read as describing different taxonomies.
PAPER_NAME = {
    'Orable': 'Weldable',
    'Conditionally Transportable': 'Transportable (Eq)',
}


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


def _paper(cat: str) -> str:
    return PAPER_NAME.get(cat, cat)

# per-category, counted per tainted OUTPUT assignment (one tainted reg/mem write)
asg_count: dict[str, int] = defaultdict(int)
asg_out_bits: dict[str, int] = defaultdict(int)  # total tainted output bits
asg_aval_bits: dict[str, int] = defaultdict(int)  # avalanche-exclusive output bits
# Split by output width, because the paper's claim is about the DATA path: the
# 1-bit CPU flags are bit-irreducible by construction and are argued separately.
asg_data_out: dict[str, int] = defaultdict(int)
asg_data_aval: dict[str, int] = defaultdict(int)
asg_flag_out: dict[str, int] = defaultdict(int)
asg_flag_aval: dict[str, int] = defaultdict(int)

# per-category, counted per executed tainted INSTRUCTION (dominant category)
insn_count: dict[str, int] = defaultdict(int)

STATS = {
    'insns_hooked': 0,  # main-binary instructions evaluated (post taint-arm)
    'insns_tainted': 0,  # instructions producing >=1 tainted output bit
    'total_out_bits': 0,  # sum over tainted assignments of output-bit popcount
    'total_aval_bits': 0,  # sum of avalanche-exclusive output bits
    # data-register-only view (output width >= 8 bits; excludes 1-bit CPU flags)
    'data_out_bits': 0,
    'data_aval_bits': 0,
    'flag_out_bits': 0,
    'flag_aval_bits': 0,
    'budget_hit': False,
}

# ---------------------------------------------------------------------------
# Failure recording.  Every one of these paths used to be `except Exception:
# pass`, so a broken measurement was indistinguishable from a clean one.  The
# worst was the per-instruction one: it dropped an ENTIRE instruction from both
# halves of the table while `insns_hooked` still counted it, which makes
# "carried no taint" and "the tally crashed" look identical.
#
# Raising here is not an option: an exception at the Unicorn callback boundary
# is swallowed by Unicorn and turns into a silent zero-tainted run.  So record,
# print the first of each kind, and fail the process at the end instead.
# ---------------------------------------------------------------------------
import traceback as _tb
from collections import Counter as _Counter

FAILURES: _Counter = _Counter()
_FAIL_SEEN: set = set()


def _fail(where, exc, ctx=''):
    key = f'{where} | {type(exc).__name__}: {exc}' if exc is not None else where
    FAILURES[key] += 1
    if key not in _FAIL_SEEN:
        _FAIL_SEEN.add(key)
        sys.stderr.write(f'\n[avalanche-harness] FAILURE ({where})\n')
        if ctx:
            sys.stderr.write(f'  context: {ctx}\n')
        if exc is not None:
            sys.stderr.write('  ' + _tb.format_exc().strip().replace('\n', '\n  ')[:1200] + '\n')


_BUDGET = [0]  # max tainted instructions (0 = unlimited)
_WRAPPER: list[Any] = [None]  # set to the MicrotaintWrapper so we can emu_stop


# ---------------------------------------------------------------------------
# 1. Category recording:  wrap generate_taint_assignments
# ---------------------------------------------------------------------------
_orig_gta = _engine.generate_taint_assignments


def _gta_wrapper(
    arch,
    bytestring,
    assignments,
    slice_ops,
    dep_set,
    out_target,
    out_name,
    out_bit_start,
    out_bit_end,
    mapper,
    mapping=None,
    has_cbranch=False,
    cbranch_flag_deps=None,
    is_bit_count=False,
    is_software_loop=False,
    # The engine has grown keyword arguments since this harness was written
    # (cbranch_cond_internal, ...).  Swallow and forward whatever it passes:
    # a TypeError here is caught by the Unicorn callback boundary and turns
    # into a silent zero-tainted-instruction run.
    **extra,
):
    old_len = len(assignments)
    width = out_bit_end - out_bit_start + 1
    # Reproduce the classifier verdict exactly as the engine's dispatch does.
    is_store = hasattr(out_target, 'address_expr')
    try:
        cat = determine_category(slice_ops, out_width_bits=width)
        label = str(cat)
    except Exception as _e:
        _ops = ' '.join(o.opcode.name for o in slice_ops) if slice_ops else '<empty slice>'
        _fail('1. determine_category raised -> category becomes Unknown', _e,
              f'bytes={bytestring.hex()} out={out_name}[{out_bit_start}:{out_bit_end}] ops={_ops[:100]}')
        label = None
    if is_software_loop:
        label = 'Avalanche'  # engine forces AVALANCHE for BMI2 loops
    elif label is None:
        label = 'Mapped' if is_store else 'Unknown'

    _orig_gta(
        arch,
        bytestring,
        assignments,
        slice_ops,
        dep_set,
        out_target,
        out_name,
        out_bit_start,
        out_bit_end,
        mapper,
        mapping,
        has_cbranch=has_cbranch,
        cbranch_flag_deps=cbranch_flag_deps,
        is_bit_count=is_bit_count,
        is_software_loop=is_software_loop,
        **extra,
    )

    for _i in range(old_len, len(assignments)):
        CAT_BY_KEY[(bytestring.hex(), _i)] = label


_engine.generate_taint_assignments = _gta_wrapper  # type: ignore[assignment]


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
        lhs = _eval_precise(e.lhs, ctx)
        rhs = _eval_precise(e.rhs, ctx)
        if opn == 'AND':
            return lhs & rhs
        if opn == 'OR':
            return lhs | rhs
        if opn == 'XOR':
            return lhs ^ rhs
        if opn == 'NOT':
            return ~lhs
        if opn == 'LEFT':
            return (lhs << rhs) if rhs >= 0 else 0
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


def _tally(circuit, ctx, _out, bytes_hex=''):
    insn_cats = []  # (cat, out_bits, aval_bits) for tainted assignments
    for idx, a in enumerate(circuit.assignments):
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
        except Exception as _e:
            _fail('3. expr.evaluate raised -> assignment DROPPED from the table', _e,
                  f'target={getattr(a.target, "name", "MEM")} expr={type(expr).__name__}')
            continue
        if full == 0:
            continue
        # The ENGINE evaluates the neutralised tree; nothing here reimplements
        # taint semantics, so a node type added tomorrow cannot silently make
        # approximation invisible.
        try:
            precise = W.evaluate_precise(expr, ctx) & mask
        except Exception as _e:
            _fail('4. evaluate_precise raised -> 0 bits attributed to approximation', _e,
                  f'target={getattr(a.target, "name", "MEM")} expr={type(expr).__name__}')
            precise = full
        aval = full & (~precise & mask)
        out_bits = full.bit_count()
        aval_bits = aval.bit_count()

        # Optional: dump the assignments in a given category that approximate,
        # so "category X over-approximates" can be answered with instructions
        # rather than adjectives.
        if aval_bits and os.environ.get('AVAL_DUMP_CAT'):
            _want = os.environ['AVAL_DUMP_CAT']
            _c = CAT_BY_KEY.get((bytes_hex, idx))
            if _c == _want and len(_DUMPED) < 40:
                _mech = W.approximating_nodes(expr)
                _tn = getattr(a.target, 'name', 'MEM')
                _DUMPED.append((bytes_hex, _tn, width, out_bits, aval_bits, tuple(_mech)))

        # Which mechanism owns those bits: neutralise one type at a time.
        if aval_bits:
            for mech in W.approximating_nodes(expr):
                try:
                    one = W.evaluate_precise(expr, ctx, frozenset({mech})) & mask
                except Exception:
                    continue
                mech_bits[mech] += (full & ~one & mask).bit_count()

        cat = CAT_BY_KEY.get((bytes_hex, idx))
        if cat is None:
            _fail('2. no classifier label for this assignment -> Unknown', None,
                  f'bytes={bytes_hex} idx={idx} '
                  f'target={getattr(a.target, "name", "MEM")}')
            cat = 'Unknown'
        # The classifier's verdict and the emitted tree must agree about whether
        # this assignment approximates.  They diverged silently for years.
        tree_approx = W.contains_approximation(expr)
        if (cat == 'Avalanche') != tree_approx:
            DISAGREE[f'{cat} vs tree_approximates={tree_approx}'] += 1

        cat = _paper(cat)
        asg_count[cat] += 1
        asg_out_bits[cat] += out_bits
        asg_aval_bits[cat] += aval_bits
        STATS['total_out_bits'] += out_bits
        STATS['total_aval_bits'] += aval_bits
        if width >= 8:
            STATS['data_out_bits'] += out_bits
            STATS['data_aval_bits'] += aval_bits
            asg_data_out[cat] += out_bits
            asg_data_aval[cat] += aval_bits
        else:
            STATS['flag_out_bits'] += out_bits
            STATS['flag_aval_bits'] += aval_bits
            asg_flag_out[cat] += out_bits
            asg_flag_aval[cat] += aval_bits
        insn_cats.append((cat, out_bits, aval_bits))

    if not insn_cats:
        return
    STATS['insns_tainted'] += 1
    dom = max(insn_cats, key=lambda t: t[1])[0]
    insn_count[dom] += 1
    if _BUDGET[0] and STATS['insns_tainted'] >= _BUDGET[0]:
        STATS['budget_hit'] = True
        w = _WRAPPER[0]
        if w is not None:
            try:
                w.ql.emu_stop()
            except Exception as _e:
                _fail('emu_stop failed after budget', _e)


class CircuitProxy:
    __slots__ = ('_bytes', '_c')

    def __init__(self, c, bytes_hex=''):
        self._c = c
        self._bytes = bytes_hex

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
            _tally(self._c, ctx, out, self._bytes)
        except Exception as _e:
            _fail('5. _tally raised -> the WHOLE INSTRUCTION is missing from both '
                  'halves of the table', _e, f'insn #{STATS["insns_hooked"]}')
        return out


# ---------------------------------------------------------------------------
# Coverage gate: which code paths construct a TaintAssignment?
#
# The harness hooks generate_taint_assignments, so anything that builds an
# assignment elsewhere is invisible to it and used to surface as 'Unknown' with
# no explanation.  Measured, there is exactly one such path:
# _exact_store_lane_targets, called from map_outputs_to_targets, which splits a
# wide SIMD copy into 8-byte lane STOREs (`movups [rdi], xmm0`).  Its own
# docstring calls those lanes EXACT, so their category is Mapped.
#
# If the engine grows another bypass, this fails instead of quietly mislabelling.
# ---------------------------------------------------------------------------
_KNOWN_BYPASS = {'_exact_store_lane_targets'}


def _check_assignment_construction_sites():
    import inspect
    import re as _re
    src = inspect.getsource(_engine).split('\n')
    fn = None
    outside: set[str] = set()
    for line in src:
        m = _re.match(r'^def (\w+)', line)
        if m:
            fn = m.group(1)
        if fn is not None and 'TaintAssignment(' in line and fn != 'generate_taint_assignments':
            outside.add(fn)
    unexpected = outside - _KNOWN_BYPASS
    if unexpected:
        raise SystemExit(
            f'avalanche harness: {sorted(unexpected)} now build TaintAssignments '
            'outside generate_taint_assignments, so the harness cannot label them. '
            'Hook them, or the table will carry Unknown rows with no cause.')


_check_assignment_construction_sites()


_orig_cached = _engine._cached_generate_static_rule


def _cached_proxy(arch, bytestring, state_format_tuple):
    c = _orig_cached(arch, bytestring, state_format_tuple)
    bh = bytestring.hex()
    # Fill in the one documented bypass rather than letting it read as Unknown.
    for i, a in enumerate(c.assignments):
        if (bh, i) in CAT_BY_KEY:
            continue
        if getattr(a, 'is_mem_target', False):
            CAT_BY_KEY[(bh, i)] = 'Mapped'   # exact 8-byte lane STORE
        else:
            _fail('2b. assignment from an UNKNOWN construction path (not a store '
                  'lane) -> cannot be labelled', None,
                  f'bytes={bh} idx={i} target={getattr(a.target, "name", "?")}')
    return CircuitProxy(c, bh)


# Patch the name the wrapper actually calls (imported into wrapper's namespace)
import microtaint.emulator.wrapper as _wrap_mod

_wrap_mod._cached_generate_static_rule = _cached_proxy  # type: ignore[attr-defined,assignment]
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
    fd lifecycle calls Qiling makes (close/fileno/seek/flush/name)."""

    # Qiling resolves `fstat(0)` -- which the guest issues as
    # `newfstatat(0, "", AT_EMPTY_PATH)` -- through
    # `transform_path`, whose AT_EMPTY_PATH branch returns `ql.os.fd[dirfd].name`
    # and then stats that HOST path.  Every real Qiling fd is a `ql_file`, whose
    # `.name` is the path it was opened from; this stand-in had none, so the
    # syscall raised AttributeError and execution halted.
    #
    # Whether the guest reaches that branch depends on the HOST's libc, not on
    # the guest binary: Qiling runs with rootfs=/, so a dynamic guest loads the
    # host's ld.so and libc.  Debian 12 (glibc 2.36) takes it and Arch (2.42)
    # does not, so the same base64 binary crashed on one and not the other.
    # `/dev/stdin` is the honest answer: it is what fd 0 means, and it stats.
    name = '/dev/stdin'

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
            chunk = bytes(self._buf)
            del self._buf[:]
            return chunk
        chunk = bytes(self._buf[: nl + 1])
        del self._buf[: nl + 1]
        return chunk


def run(binary, binary_args, stdin_data, rootfs='/', budget=0, taint_file=None):
    _BUDGET[0] = budget
    reporter = Reporter(json_mode=False, stream=sys.stderr)
    ql = Qiling([binary, *binary_args], rootfs, verbose=QL_VERBOSE.OFF)
    ql.os.stdin = PreloadStdin(stdin_data)
    wrapper = MicrotaintWrapper(ql, check_bof=True, check_uaf=False, check_sc=False, check_aiw=False, reporter=reporter)
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
    print(f'main-binary instructions evaluated (after taint armed): {STATS["insns_hooked"]:,}')
    print(f'executed TAINTED instructions:                          {STATS["insns_tainted"]:,}')
    if STATS['budget_hit']:
        print('   (stopped early: tainted-instruction budget reached)')
    print(f'total tainted output bits:                              {STATS["total_out_bits"]:,}')
    print(f'avalanche-exclusive output bits:                        {STATS["total_aval_bits"]:,}')
    if STATS['total_out_bits']:
        share = 100.0 * STATS['total_aval_bits'] / STATS['total_out_bits']
        print(f'AVALANCHE share of ALL tainted output bits:             {share:.2f}%')
    if STATS['data_out_bits']:
        ds = 100.0 * STATS['data_aval_bits'] / STATS['data_out_bits']
        print(
            f'  data registers only (>=8-bit outputs):  '
            f'{STATS["data_aval_bits"]:,}/{STATS["data_out_bits"]:,}  ({ds:.2f}%)',
        )
    if STATS['flag_out_bits']:
        fs = 100.0 * STATS['flag_aval_bits'] / STATS['flag_out_bits']
        print(
            f'  CPU flag bits only (1-bit outputs):     '
            f'{STATS["flag_aval_bits"]:,}/{STATS["flag_out_bits"]:,}  ({fs:.2f}%)',
        )

    print('\nPer-category DYNAMIC frequency (by executed tainted instruction, dominant category):')
    tot_i = sum(insn_count.values()) or 1
    for cat, n in sorted(insn_count.items(), key=lambda kv: -kv[1]):
        print(f'   {cat:<30} {n:>10,}  {100.0 * n / tot_i:6.2f}%')

    print('\nPer-category (by tainted OUTPUT assignment) + avalanche bit share:')
    print(f'   {"category":<30} {"assigns":>9} {"freq%":>7} {"outbits":>10} {"avalbits":>9} {"aval%":>7}')
    tot_a = sum(asg_count.values()) or 1
    for cat in sorted(asg_count, key=lambda c: -asg_count[c]):
        n = asg_count[cat]
        ob = asg_out_bits[cat]
        ab = asg_aval_bits[cat]
        avp = (100.0 * ab / ob) if ob else 0.0
        print(f'   {cat:<30} {n:>9,} {100.0 * n / tot_a:6.2f}% {ob:>10,} {ab:>9,} {avp:6.2f}%')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=int, default=0, help='stop after N tainted instructions (0=run to completion)')
    ap.add_argument('--stdin', default='hello world\n', help='literal stdin string to taint')
    ap.add_argument('--stdin-bytes', default=None, help='hex string of stdin bytes to taint (overrides --stdin)')
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

    EXACT_CATS = ('Mapped', 'Transportable', 'Translatable', 'Monotonic')
    print('\nDATA bits only (>=8-bit outputs), per category -- the paper claims the')
    print('exactly-handled categories propagate "with no over-approximation":')
    print(f'   {"category":32s} {"data bits":>10s} {"approx":>8s} {"approx%":>8s}')
    _eo = _ea = 0
    for cat in sorted(set(asg_data_out) | set(asg_data_aval),
                      key=lambda c: -asg_data_out[c]):
        o, a = asg_data_out[cat], asg_data_aval[cat]
        if cat in EXACT_CATS:
            _eo += o
            _ea += a
        print(f'   {cat:32s} {o:10,d} {a:8,d} {100*a/o if o else 0:7.2f}%')
    print(f'   {"-> exactly-handled categories":32s} {_eo:10,d} {_ea:8,d} '
          f'{100*_ea/_eo if _eo else 0:7.2f}%')
    if _ea:
        print(f'   NOT free of over-approximation: {_ea:,} of {_eo:,} data bits in the')
        print('   exactly-handled categories exist only because of an approximation.')

    if _DUMPED:
        import collections as _col
        print(f'\nSample of approximating assignments in category '
              f'{os.environ.get("AVAL_DUMP_CAT")!r}:')
        _agg: _col.Counter[tuple[str, str, int, tuple[str, ...]]] = _col.Counter()
        for bh, tn, w, _ob, _ab, mech in _DUMPED:
            _agg[(bh, tn, w, mech)] += 1
        for (bh, tn, w, mech), n in _agg.most_common(12):
            print(f'   x{n:<4d} bytes={bh:<14s} target={tn:<10s} width={w:<3d} via={list(mech)}')

    if mech_bits:
        print('\nApproximation bits BY MECHANISM (a bit can be owed to more than one):')
        for mech, n in sorted(mech_bits.items(), key=lambda kv: -kv[1]):
            print(f'   {mech:32s} {n:10,d}')
    if DISAGREE:
        print('\nCLASSIFIER vs EMITTED TREE disagreements:')
        for k, n in sorted(DISAGREE.items(), key=lambda kv: -kv[1]):
            print(f'   {n:8,d}  {k}')
        print('   (the classifier calls an instruction Avalanche while its tree')
        print('    emits no approximating node, or vice versa -- the two halves of')
        print('    Table 6 are answering different questions for these)')

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
        blob['failures'] = dict(FAILURES)
        blob['mech_bits'] = dict(mech_bits)
        blob['asg_data_out'] = dict(asg_data_out)
        blob['asg_data_aval'] = dict(asg_data_aval)
        blob['asg_flag_out'] = dict(asg_flag_out)
        blob['asg_flag_aval'] = dict(asg_flag_aval)
        blob['classifier_tree_disagreements'] = dict(DISAGREE)
        blob['approximating_types'] = sorted(W.APPROXIMATING_TYPES)
        blob['engine'] = _engine_provenance()
        with open(args.json_out, 'w') as fh:
            json.dump(blob, fh, indent=2)
        print(f'\n[+] wrote {args.json_out}')

    # A table built on a run that dropped instructions is not a measurement.
    # Exit non-zero so a caller cannot mistake it for one.
    if FAILURES:
        print('\n' + '=' * 72)
        print(f'REFUSING TO CERTIFY: {sum(FAILURES.values())} silent-path failure(s)')
        print('=' * 72)
        for key, n in FAILURES.most_common():
            print(f'  {n:7d}  {key}')
        print('\nThe table above is computed over whatever survived these, so the '
              'percentages are\nnot over the population they claim.')
        return 1
    # A run in which NOTHING was tainted has no failures either, and printed
    # exactly this line.  Table 6's producer must not certify a table computed
    # over an empty population.
    if not STATS['insns_tainted'] or not STATS['total_out_bits']:
        print('\nREFUSING TO CERTIFY: the run tainted '
              f'{STATS["insns_tainted"]:,} instructions and '
              f'{STATS["total_out_bits"]:,} output bits, so the table above is '
              'computed over nothing and cannot fail')
        return 1
    print('\n[ok] no failure on any recorded path: every hooked instruction '
          'reached the table')
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
