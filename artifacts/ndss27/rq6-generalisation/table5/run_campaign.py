# ruff: noqa: W505, E501, B905, B007
#   Style only, suppressed rather than rewritten: vendored from the campaign
#   that produced the published numbers.  Correctness is gated by
#   validate_oracle.py, not by restyling proven code.
# Experiment script, not library code: see artifacts/ndss27/README.md,
# "Lint and type checking", for why annotations are not required here.
# mypy: disable-error-code="no-untyped-def, no-untyped-call, type-arg, no-any-return, var-annotated, assignment, index, arg-type, union-attr, operator, attr-defined, misc, call-overload, return-value, unreachable"
"""Soundness + precision campaign for the AE, with an oracle that cannot invent taint.

Usage:  run_campaign.py <isa_key> <deadline_epoch> [seed]

Differences from campaign12h.py, all of them about not lying:

  * HardenedGTSim, so the 2^k polarity runs of a case are independent.  The
    shipped oracle leaked untracked registers between runs and produced 2,810 of
    its 2,875 "under-taints" from one form, `shrd rbp, rsp`.
  * A pre-flight per corpus entry.  `model_closure_sampled` quarantines entries
    whose result depends on state the oracle does not model, and
    `decode_agreement` quarantines entries where Unicorn steps through something
    other than what the bytes encode.  Quarantined entries are reported, never
    counted as under-taints.
  * EVERY under-taint witness is written to under_<isa>.jsonl as it is found,
    not just the first per form.
  * Provenance: engine commit, package versions and the pinned engine env go into
    the result file, so it can never be unclear which engine produced it.

MICROTAINT_TAINT_IR=0 pins the LogicCircuit path.  mt_worker.py calls
generate_static_rule()+evaluate(), which cannot reach taint_ir anyway
(check_engine_path.py proves it), but the pin is recorded regardless.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

os.environ['MICROTAINT_TAINT_IR'] = '0'
os.environ['MICROTAINT_BLOCK'] = '0'

from corpora import CORPORA
from mt_multiarch import ENGINE_ROOT, GTSim, _build_isas, mt_batch, written_flags
from oracle import HardenedGTSim
from probes import decode_agreement, model_closure_sampled
from unicorn_bugs import quarantined as unicorn_quarantined
from unicorn_bugs import suppressed_flags as unicorn_suppressed

ISA_KEY = sys.argv[1]
DEADLINE = float(sys.argv[2])
SEED = int(sys.argv[3], 0) if len(sys.argv) > 3 else 0xC0FFEE

OUT = os.path.join(HERE, f'campaign_{ISA_KEY}.json')
WITNESS = os.path.join(HERE, f'under_{ISA_KEY}.jsonl')
SINGLE_STATES, MULTI_PER_RND, K_MIN, K_MAX = 2, 6, 2, 8
ENTRIES_PER_BATCH = 40   # one `uv run` spawn per 40 entries, not per entry
CLOSURE_STATES = 40
WITNESS_CAP = 500          # per instruction form; counts stay exact past the cap


def popcount(x):
    return bin(x & ((1 << 128) - 1)).count('1')


def apply_cons(st, cons):
    for r, c in cons.items():
        if r == 'exclude_flags':
            continue
        if c == 'nonzero':
            if st.get(r, 0) == 0:
                st[r] = 0x3
        elif isinstance(c, int):
            st[r] = c
    return st


def provenance():
    def git(*a):
        try:
            return subprocess.run(['git', '-C', ENGINE_ROOT, *a], capture_output=True,
                                  text=True).stdout.strip()
        except Exception:
            return None
    import unicorn
    try:
        import pypcode
        pv = getattr(pypcode, '__version__', None)
    except Exception:
        pv = None
    return {
        'engine_root': ENGINE_ROOT,
        'engine_commit': git('rev-parse', 'HEAD'),
        'engine_branch': git('rev-parse', '--abbrev-ref', 'HEAD'),
        'engine_describe': git('describe', '--tags', '--always', '--dirty'),
        'engine_dirty': bool(git('status', '--porcelain')),
        'engine_env': {'MICROTAINT_TAINT_IR': '0', 'MICROTAINT_BLOCK': '0'},
        'engine_path': 'generate_static_rule + LogicCircuit/ChainedCircuit (not taint_ir)',
        'unicorn': unicorn.__version__, 'pypcode': pv,
        'oracle': 'HardenedGTSim (context-restore per polarity run, memory reset, PC completion check)',
        'started': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }


def preflight(isa, gt, entries, rng):
    """Quarantine entries the oracle cannot legitimately judge."""
    fl = {f.name: 0 for f in isa.flags}
    states = []
    for _ in range(CLOSURE_STATES):
        st = {n: rng.getrandbits(isa.bits) for n, _ in isa.gprs}
        if isa.sp:
            st[isa.sp[0]] = GTSim.SP_VALUE
        states.append(st)
    quarantine = {}
    for asm, code, srcs, cons in entries:
        mc = model_closure_sampled(gt, code, states, fl)
        da = decode_agreement(gt, ISA_KEY, code, states[0], fl)
        reasons = []
        ub = unicorn_quarantined(ISA_KEY, asm)
        if ub:
            reasons.append(f'known Unicorn defect: {ub}')
        if mc.get('checked') and not mc.get('closed'):
            reasons.append(f"reads state the oracle does not model (leaks into {mc['leaked']})")
        if mc.get('checked') and not mc.get('deterministic'):
            reasons.append('non-deterministic under identical inputs')
        if da.get('checked') and da.get('agree') is False:
            reasons.append('Unicorn executes a different decode than the bytes encode')
        if reasons:
            quarantine[asm] = {'reasons': reasons, 'model_closure': mc, 'decode': da}
    return quarantine


def main():
    isa = _build_isas(ISA_KEY)[ISA_KEY]
    gt = HardenedGTSim(isa)
    fset = {f.name for f in isa.flags}
    # Stable bit index per flag, recorded in the shard so a mask can be decoded.
    flag_bit = {n: i for i, n in enumerate(sorted(fset))}
    gpr_names = [n for n, _ in isa.gprs]

    entries = []
    for label, asm, srcs, cons in CORPORA[ISA_KEY](isa):
        try:
            code = isa.assemble(asm)
        except Exception:
            try:
                code = bytes.fromhex(asm)
            except Exception:
                continue
        entries.append((asm, code, srcs, cons))

    rng = random.Random(SEED ^ hash(ISA_KEY) & 0xFFFFFFFF)
    prov = provenance()
    quarantine = preflight(isa, gt, entries, random.Random(SEED))
    print(f'{isa.label}: {len(entries)} entries, {len(quarantine)} quarantined', flush=True)
    for a, q in quarantine.items():
        print(f'  QUARANTINE {a}: {"; ".join(q["reasons"])}', flush=True)

    wf = open(WITNESS, 'a')
    metrics, rounds, start = {}, 0, time.time()

    # Resume: the supervisor in launch.sh restarts a crashed ISA, and the old
    # campaign segfaulted roughly every 10k cases.  Witnesses are appended, so
    # only the counters need carrying over.
    if os.path.exists(OUT):
        try:
            prev = json.load(open(OUT))
            metrics = prev.get('metrics', {})
            rounds = prev.get('rounds', 0)
            print(f'resumed at round {rounds}, '
                  f'{sum(v.get("checked", 0) for v in metrics.values()):,} cases', flush=True)
        except Exception:
            pass

    def checkpoint():
        """Atomic, so a reader (analyse.py) never sees a half-written file."""
        tmp = OUT + '.tmp'
        json.dump({
            'isa': isa.label, 'isa_key': ISA_KEY, 'provenance': prov,
            'rounds': rounds, 'elapsed_h': round((time.time() - start) / 3600, 3),
            'total_checked': sum(v.get('checked', 0) for v in metrics.values()),
            'total_under': sum(v.get('under', 0) for v in metrics.values()),
            'total_skipped': sum(v.get('skipped', 0) for v in metrics.values()),
            'oracle_skipped_cases': gt.skipped,
            # Bit index per flag, so flag_fail_hist masks can be decoded.  A
            # histogram whose key encoding is not recorded is unreadable.
            'flag_order': flag_bit,
            'unsound_instrs': [a for a, v in metrics.items() if v.get('under')],
            'quarantine': quarantine,
            'metrics': metrics,
        }, open(tmp, 'w'), indent=1)
        os.replace(tmp, OUT)

    while time.time() < DEADLINE:
        rounds += 1
        live = [e for e in entries if e[0] not in quarantine and e[2]]
        # One `uv run` spawn costs ~0.44 s regardless of how many cases it carries,
        # so cases from many entries share a call.  mt_worker keys its rule cache on
        # the per-case bytes, so a mixed batch is fine.
        for chunk_start in range(0, len(live), ENTRIES_PER_BATCH):
            if time.time() >= DEADLINE:
                break
            chunk = live[chunk_start:chunk_start + ENTRIES_PER_BATCH]
            built, batch = [], []
            for asm, code, srcs, cons in chunk:
                excl = set(cons.get('exclude_flags', ())) | unicorn_suppressed(ISA_KEY, asm)
                chk = gpr_names + sorted(written_flags(isa, code) - excl)
                gsrc = [s2 for s2 in srcs if s2 not in fset] or srcs

                bases = []
                for _ in range(SINGLE_STATES):
                    st = {n: rng.getrandbits(isa.bits) for n, _ in isa.gprs}
                    for f in isa.flags:
                        st[f.name] = rng.getrandbits(1)
                    if isa.sp:
                        st[isa.sp[0]] = GTSim.SP_VALUE
                    apply_cons(st, cons)
                    bases.append(st)

                cases = []
                for st in bases:
                    for s2 in srcs:
                        w = 1 if s2 in fset else isa.bits
                        for b in range(w):
                            cases.append(isa.canonicalize(dict(st), {s2: 1 << b}))
                for st in bases:
                    for _ in range(MULTI_PER_RND):
                        t = {}
                        for _ in range(rng.randint(K_MIN, K_MAX)):
                            s2 = rng.choice(gsrc)
                            w = 1 if s2 in fset else isa.bits
                            t[s2] = t.get(s2, 0) | (1 << rng.randrange(w))
                        cases.append(isa.canonicalize(dict(st), t))

                built.append((asm, code, chk, cases))
                batch.extend({'label': asm, 'asm': asm, 'bytes': code.hex(),
                              'srcs': list(t), 'state': st, 'taint': t} for st, t in cases)

            try:
                mts = mt_batch(isa, batch)
            except Exception as e:
                print(f'  mt_batch failed on a chunk of {len(chunk)}: {e}', flush=True)
                for asm, _c, _k, _cs in built:
                    m = metrics.setdefault(asm, {})
                    m['mt_errors'] = m.get('mt_errors', 0) + 1
                continue

            off = 0
            for asm, code, chk, cases in built:
                mts_e = mts[off:off + len(cases)]
                off += len(cases)
                m = metrics.setdefault(asm, {})
                for k in ('checked', 'under', 'over', 'exact', 'skipped',
                          'gt_bits', 'mt_bits', 'over_bits', 'witnesses_written',
                          'ratio_n',
                          # Split by register KIND.  A case is scored exact only
                          # if every scored register matched, so one coarse flag
                          # sinks the whole case: `imul rax, 1` is an identity on
                          # the result yet scores 0% exact because CF/OF take the
                          # multiply floor, and PF is over-tainted by any tainted
                          # high bit because it reads only the low byte.  Without
                          # this split, "55% bit-exact" reads as a data-path
                          # result when it is almost entirely a flag result.
                          'val_exact', 'val_over', 'val_under',
                          'flag_exact', 'flag_over', 'flag_under',
                          'val_checked', 'flag_checked',
                          'val_gt_bits', 'val_mt_bits', 'val_over_bits',
                          'flag_gt_bits', 'flag_mt_bits', 'flag_over_bits'):
                    m.setdefault(k, 0)
                m.setdefault('ratio_sum', 0.0)
                for (st, t), mt in zip(cases, mts_e):
                    g, ex = gt.taint(code, st, t)
                    if not ex:
                        m['skipped'] += 1
                        continue
                    m['checked'] += 1
                    under = mtb = gtb = ovb = 0
                    fail_mask = 0
                    v_u = v_o = f_u = f_o = 0
                    v_gt = v_mt = v_ov = f_gt = f_mt = f_ov = 0
                    n_val = n_flag = 0
                    for r in chk:
                        gv, mv = g.get(r, 0), mt.get(r, 0)
                        gtb += popcount(gv)
                        mtb += popcount(mv)
                        under |= gv & ~mv
                        ovb += popcount(mv & ~gv)
                        if r in fset:
                            n_flag += 1
                            f_gt += popcount(gv)
                            f_mt += popcount(mv)
                            f_ov += popcount(mv & ~gv)
                            f_u |= gv & ~mv
                            f_o |= mv & ~gv
                            # Per-flag totals: which flag is coarse, and by how
                            # much.  PF reads only the result's low byte, so any
                            # tainted high bit over-taints it; CF on `neg` is a
                            # predicate over all 64 bits.  Keeping them apart
                            # lets a decision about including PF be made from
                            # data instead of re-run.
                            pf = m.setdefault('per_flag', {}).setdefault(
                                r, {'checked': 0, 'exact': 0, 'over': 0,
                                    'under': 0, 'gt_bits': 0, 'mt_bits': 0,
                                    'over_bits': 0})
                            pf['checked'] += 1
                            pf['gt_bits'] += popcount(gv)
                            pf['mt_bits'] += popcount(mv)
                            pf['over_bits'] += popcount(mv & ~gv)
                            if gv & ~mv:
                                pf['under'] += 1
                            elif mv & ~gv:
                                pf['over'] += 1
                            else:
                                pf['exact'] += 1
                            if (gv & ~mv) or (mv & ~gv):
                                fail_mask |= 1 << flag_bit[r]
                        else:
                            n_val += 1
                            v_gt += popcount(gv)
                            v_mt += popcount(mv)
                            v_ov += popcount(mv & ~gv)
                            v_u |= gv & ~mv
                            v_o |= mv & ~gv
                    if n_val:
                        m['val_checked'] += 1
                        m['val_gt_bits'] += v_gt
                        m['val_mt_bits'] += v_mt
                        m['val_over_bits'] += v_ov
                        m['val_under' if v_u else ('val_over' if v_o else 'val_exact')] += 1
                    if n_flag:
                        m['flag_checked'] += 1
                        m['flag_gt_bits'] += f_gt
                        m['flag_mt_bits'] += f_mt
                        m['flag_over_bits'] += f_ov
                        m['flag_under' if f_u else ('flag_over' if f_o else 'flag_exact')] += 1
                        # Case-level failure mask.  The conjunction "exact on
                        # every flag EXCEPT PF" cannot be recomputed from
                        # per-flag totals, because a case may fail on PF alone
                        # or on PF and CF together.  The mask histogram makes
                        # ANY exclusion set exactly recoverable afterwards, for
                        # the price of one counter per distinct mask.
                        fh = m.setdefault('flag_fail_hist', {})
                        k3 = str(fail_mask)
                        fh[k3] = fh.get(k3, 0) + 1
                    m['gt_bits'] += gtb
                    m['mt_bits'] += mtb
                    m['over_bits'] += ovb
                    # Table 5's over-taint column is the MEAN tainted-to-minimum
                    # bit ratio, which is a mean of PER-CASE ratios and cannot be
                    # recovered from the summed bits afterwards: sum(mt)/sum(gt)
                    # weights big-taint cases more heavily and gives a different
                    # number (2.80x vs 1.68x on AMD64).  Accumulate it here.
                    if gtb:
                        m['ratio_sum'] += mtb / gtb
                        m['ratio_n'] += 1
                        # A MEAN over-taint ratio is dominated by avalanche
                        # cases, where one tainted bit legitimately taints 64,
                        # so it says more about how many multiplies are in the
                        # corpus than about the engine.  Keep a log2 histogram
                        # as well: it costs one increment and makes the median
                        # and the tail recoverable afterwards, which a running
                        # mean can never be.
                        h = m.setdefault('ratio_hist', {})
                        b = (mtb / gtb)
                        k2 = '0' if b <= 1 else str(min(int(b).bit_length(), 12))
                        h[k2] = h.get(k2, 0) + 1
                    if under:
                        m['under'] += 1
                        if m['witnesses_written'] < WITNESS_CAP:
                            m['witnesses_written'] += 1
                            wf.write(json.dumps({
                                'isa': ISA_KEY, 'asm': asm, 'bytes': code.hex(),
                                'round': rounds,
                                'state': {k2: hex(v) for k2, v in st.items()},
                                'taint': {k2: hex(v) for k2, v in t.items()},
                                'miss': {r: hex(g.get(r, 0) & ~mt.get(r, 0))
                                         for r in chk if g.get(r, 0) & ~mt.get(r, 0)},
                                'gt': {r: hex(g.get(r, 0)) for r in chk if g.get(r, 0)},
                                'mt': {r: hex(mt.get(r, 0)) for r in chk if mt.get(r, 0)},
                            }) + '\n')
                            wf.flush()
                    elif ovb:
                        m['over'] += 1
                    else:
                        m['exact'] += 1

            checkpoint()

    checkpoint()
    wf.close()
    print(f'{isa.label} DONE: {rounds} rounds, '
          f'{sum(v.get("checked", 0) for v in metrics.values())} cases, '
          f'{sum(v.get("under", 0) for v in metrics.values())} under-taints', flush=True)


if __name__ == '__main__':
    main()
