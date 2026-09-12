"""Why does the avalanche harness ever say 'Unknown'?

avalanche_freq.py swallows FOUR exceptions, each of which silently distorts
Table 6:

  1. determine_category() raising           -> label 'Unknown'
  2. CAT_BY_ID.get(id(a)) missing           -> label 'Unknown'
  3. expr.evaluate(ctx) raising             -> the assignment is DROPPED
  4. _eval_precise(expr, ctx) raising       -> 0 bits attributed to avalanche

(2) is worse than it looks: CAT_BY_ID is keyed on id() of a cdef-class object.
If an assignment is freed its address can be reused, so a later assignment can
inherit a STALE label -- a silent misattribution rather than a visible Unknown.

This script re-runs the same workload with every one of those paths recorded
instead of swallowed, and prints what actually fires.
"""
from __future__ import annotations

import os
import sys
import traceback
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('MICROTAINT_TAINT_IR', '0')
os.environ.setdefault('MICROTAINT_BLOCK', '0')

import avalanche_freq as A  # noqa: E402  (installs the monkeypatches on import)
from microtaint.sleigh.mapper import determine_category  # noqa: E402

FAILS: dict[str, Counter] = defaultdict(Counter)
EXAMPLES: dict[str, list] = defaultdict(list)
SEEN_IDS: set[int] = set()
ID_REUSE = Counter()


def _record(kind, exc, ctxinfo=''):
    key = f'{type(exc).__name__}: {exc}' if exc is not None else 'no-exception'
    FAILS[kind][key] += 1
    if len(EXAMPLES[kind]) < 5:
        EXAMPLES[kind].append((key, ctxinfo, traceback.format_exc(limit=4)
                               if exc is not None else ''))


# ---- 1 + 2: re-wrap the wrapper so we see the classifier failure -------------
_wrapped = A._gta_wrapper


def _diag_gta(arch, bytestring, assignments, slice_ops, dep_set, out_target,
              out_name, out_bit_start, out_bit_end, mapper, mapping=None, **kw):
    width = out_bit_end - out_bit_start + 1
    try:
        determine_category(slice_ops, out_width_bits=width)
    except Exception as exc:  # noqa: BLE001 -- that is the point
        ops = ' '.join(o.opcode.name for o in slice_ops) if slice_ops else '<empty slice>'
        _record('1. determine_category raised', exc,
                f'bytes={bytestring.hex()} out={out_name}[{out_bit_start}:{out_bit_end}] ops={ops[:120]}')
    old = len(assignments)
    r = _wrapped(arch, bytestring, assignments, slice_ops, dep_set, out_target,
                 out_name, out_bit_start, out_bit_end, mapper, mapping, **kw)
    for a in assignments[old:]:
        if id(a) in SEEN_IDS:
            ID_REUSE['id() collided with an earlier assignment'] += 1
        SEEN_IDS.add(id(a))
    return r


A._engine.generate_taint_assignments = _diag_gta
import microtaint.sleigh.engine as _eng  # noqa: E402
_eng.generate_taint_assignments = _diag_gta

# ---- 3 + 4: re-wrap _tally's inner evaluations -------------------------------
_orig_tally = A._tally


def _diag_tally(circuit, ctx, _out):
    for a in circuit.assignments:
        if a.expression is None:
            continue
        if id(a) not in A.CAT_BY_ID:
            tgt = a.target
            nm = getattr(tgt, 'name', None) or f'MEM@{getattr(tgt, "size", "?")}'
            _record('2. CAT_BY_ID miss (never seen at generation)', None, f'target={nm}')
        try:
            a.expression.evaluate(ctx)
        except Exception as exc:  # noqa: BLE001
            _record('3. expr.evaluate raised (assignment DROPPED)', exc,
                    f'target={getattr(a.target, "name", "MEM")}')
            continue
        try:
            A._eval_precise(a.expression, ctx)
        except Exception as exc:  # noqa: BLE001
            _record('4. _eval_precise raised (0 bits -> avalanche)', exc,
                    f'target={getattr(a.target, "name", "MEM")}')
    return _orig_tally(circuit, ctx, _out)


A._tally = _diag_tally

if __name__ == '__main__':
    rc = A.main() if hasattr(A, 'main') else 0
    print('\n' + '=' * 72)
    print('SWALLOWED-EXCEPTION REPORT')
    print('=' * 72)
    if not FAILS and not ID_REUSE:
        print('  nothing fired: no Unknown is reachable on this workload')
    for kind in sorted(FAILS):
        total = sum(FAILS[kind].values())
        print(f'\n{kind}  ({total} occurrences)')
        for key, n in FAILS[kind].most_common(5):
            print(f'    {n:6d}  {key}')
        for key, info, tb in EXAMPLES[kind][:2]:
            print(f'    e.g. {info}')
            if tb.strip():
                print('        ' + tb.strip().replace('\n', '\n        ')[:600])
    for k, n in ID_REUSE.items():
        print(f'\nid() REUSE: {k}: {n}')
    sys.exit(rc)
