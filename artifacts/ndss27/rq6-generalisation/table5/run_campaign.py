# ruff: noqa: W505, E501, B905, B007, RUF100, I001
#   RUF100 and I001 are config differences, not defects: the source repo
#   selects BLE001/E402/C901 (so those noqa ARE used there) and sorts
#   `microtaint` as third-party, while it is first-party here.  No single
#   spelling satisfies both repos, and the body must stay byte-identical
#   to the harness that produced the published numbers.
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
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

os.environ['MICROTAINT_TAINT_IR'] = '0'
os.environ['MICROTAINT_BLOCK'] = '0'

from oracle import HardenedGTSim  # noqa: E402
from probes import (  # noqa: E402
    PRESERVED_STATES,
    decode_agreement,
    executability,
    model_closure_sampled,
    preserved_flags,
    undeclared_sources,
)
from unicorn_bugs import quarantined as unicorn_quarantined  # noqa: E402
from unicorn_bugs import suppressed_flags as unicorn_suppressed  # noqa: E402

from corpora import CORPORA  # noqa: E402
from mt_multiarch import (  # noqa: E402
    ENGINE_ROOT,
    GTSim,
    _build_isas,
    mt_batch,
    written_flags,
)

ISA_KEY = sys.argv[1]
DEADLINE = float(sys.argv[2])
SEED = int(sys.argv[3], 0) if len(sys.argv) > 3 else 0xC0FFEE

OUT = os.path.join(HERE, f'campaign_{ISA_KEY}.json')
WITNESS = os.path.join(HERE, f'under_{ISA_KEY}.jsonl')
# Cases generated per instruction form per ROUND.  This is batching
# granularity, not the sample size: the round loop runs until the deadline, so
# the wall-clock budget decides how many cases each form actually gets.  There
# is no cap on how many BITS a case may taint, and no enumeration budget,
# because the oracle costs k+1 runs rather than 2^k.
CASES_PER_FORM = 16
ENTRIES_PER_BATCH = 40   # one `uv run` spawn per 40 entries, not per entry
CLOSURE_STATES = 40
#: Consecutive failed engine batches before the campaign gives up.  One bad
#: chunk is survivable; an unusable engine must not look like a clean run.
MAX_CONSECUTIVE_BATCH_FAILURES = 5
WITNESS_CAP = 500          # per instruction form; counts stay exact past the cap


def popcount(x):
    return bin(x & ((1 << 128) - 1)).count('1')


def constrained(cons):
    """Registers an entry pins, which therefore must NOT be tainted.

    A constraint exists to hold a register inside the instruction's defined
    domain across the WHOLE oracle, not just at the base state: a divisor pinned
    non-zero, or a 16-bit shift count pinned below the operand width.  Tainting
    it defeats that, because the oracle flips each tainted bit and a flip takes
    the register straight back out of the domain.  `mt_multiarch.gen_cases`
    pops them; this path did not.
    """
    return {r for r in cons if r != 'exclude_flags'}


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


def provenance(isa):
    def git(*a, repo=ENGINE_ROOT):
        try:
            return subprocess.run(['git', '-C', repo, *a], capture_output=True,
                                  text=True).stdout.strip()
        except Exception:  # noqa: BLE001
            return None
    import unicorn
    try:
        import pypcode
        pv = getattr(pypcode, '__version__', None)
    except Exception:  # noqa: BLE001
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
        # This string is how a reader tells the fixed oracle from the broken
        # one, so it must name what actually ran.  It said "PC completion
        # check" -- the very mechanism that silently discarded 19M MIPS
        # cases, replaced by an instruction-count hook precisely because of
        # that -- and it said so in shards produced today.
        'oracle': 'HardenedGTSim (two-polarity flip union, context-restore '
                  'per run, memory reset, instruction-count completion hook)',
        'started': time.strftime('%Y-%m-%dT%H:%M:%S'),
        # The HARNESS, not just the engine.  Every number in Table 5 is defined
        # by this code, and recording only what was measured while saying
        # nothing about what did the measuring is how a campaign becomes
        # unreproducible after the fact.
        'harness_root': HARNESS_ROOT,
        'harness_commit': git('rev-parse', 'HEAD', repo=HARNESS_ROOT),
        # Scoped to the files that DEFINE the campaign.  A repo-wide status is
        # dirty for reasons that have nothing to do with the measurement (other
        # work in the tree, untracked scratch), so it would report "dirty" on
        # every run and mean nothing.  These three paths are the harness.
        'harness_dirty': bool(git('status', '--porcelain', '--', HERE,
                                  repo=HARNESS_ROOT)),
        'harness_paths': [os.path.relpath(HERE, HARNESS_ROOT)],
        'seed': SEED,
        'oracle_polarities': 2,
        'modelled_regs': [n for n, _ in isa.gprs],
        'modelled_flags': [f.name for f in isa.flags],
    }


#: States used by the cheaper per-form probes below.  Fewer than CLOSURE_STATES
#: because these cost one run per modelled register rather than one per state.
PROBE_STATES = 4




def preflight(isa, gt, entries, rng):
    """Quarantine entries the oracle cannot legitimately judge."""
    fl = {f.name: 0 for f in isa.flags}
    states = []
    for _ in range(CLOSURE_STATES):
        st = {n: rng.getrandbits(isa.bits) for n, _ in isa.gprs}
        if isa.sp:
            st[isa.sp[0]] = GTSim.SP_VALUE
        states.append(st)
    # Flags are SAMPLED here, not pinned to zero: a predicated form (`cmove`,
    # `setcc`) never fires under an all-zero flag state, so an all-zero probe
    # only ever exercises one branch of the condition.
    probe_cases = []
    for _ in range(PROBE_STATES):
        st = {n: rng.getrandbits(isa.bits) for n, _ in isa.gprs}
        if isa.sp:
            st[isa.sp[0]] = GTSim.SP_VALUE
        probe_cases.append((st, {f.name: rng.getrandbits(f.width) for f in isa.flags}))
    preserved_cases = []
    for _ in range(PRESERVED_STATES):
        st = {n: rng.getrandbits(isa.bits) for n, _ in isa.gprs}
        if isa.sp:
            st[isa.sp[0]] = GTSim.SP_VALUE
        preserved_cases.append(
            (st, {f.name: rng.getrandbits(f.width) for f in isa.flags}))
    quarantine = {}
    undeclared = {}
    preserved = {}
    for label, asm, code, srcs, cons in entries:
        mc = model_closure_sampled(gt, code, states, fl)
        da = decode_agreement(gt, ISA_KEY, code, states[0], fl)
        ex = executability(gt, code, probe_cases)
        reasons = []
        ub = unicorn_quarantined(ISA_KEY, asm)
        if ub:
            reasons.append(f'known Unicorn defect: {ub}')
        if not ex['ran']:
            reasons.append(f"Unicorn cannot execute it: 0 of {ex['of']} sampled "
                           f'states completed, so there is nothing to compare')
        if mc.get('checked') and not mc.get('closed'):
            reasons.append(f"reads state the oracle does not model (leaks into {mc['leaked']})")
        if mc.get('checked') and not mc.get('deterministic'):
            reasons.append('non-deterministic under identical inputs')
        if da.get('checked') and da.get('agree') is False:
            reasons.append('Unicorn executes a different decode than the bytes encode')
        if reasons:
            quarantine[label] = {'reasons': reasons, 'model_closure': mc,
                                 'decode': da, 'executability': ex}
            continue
        # Reported, NOT quarantined: an undeclared source makes the test weaker
        # (that operand is never tainted), not wrong, so dropping the form would
        # lose a real measurement.  It is recorded so it can be fixed.
        miss = undeclared_sources(gt, isa, code, srcs, probe_cases)
        if miss:
            undeclared[label] = {'declared': list(srcs), 'also_influences': miss}
        preserved[label] = sorted(
            preserved_flags(gt, isa, code, preserved_cases, srcs))
    return quarantine, undeclared, preserved


def main():  # noqa: C901
    isa = _build_isas(ISA_KEY)[ISA_KEY]
    gt = HardenedGTSim(isa)
    fset = {f.name for f in isa.flags}
    fwidth = {f.name: f.width for f in isa.flags}
    # Stable bit index per flag, recorded in the shard so a mask can be decoded.
    flag_bit = {n: i for i, n in enumerate(sorted(fset))}
    gpr_names = [n for n, _ in isa.gprs]

    entries = []
    for label, asm, srcs, cons in CORPORA[ISA_KEY](isa):
        try:
            code = isa.assemble(asm)
        except Exception:  # noqa: BLE001
            try:
                code = bytes.fromhex(asm)
            except Exception:  # noqa: BLE001
                continue
        # The LABEL, not just the asm.  Two corpus entries can share the same
        # instruction text and differ in constraints (`shl rax, cl` free vs
        # `shl rax, cl [cl=1]` with RCX pinned): keying metrics on the asm
        # merged two different experiments into one row and double-weighted it.
        entries.append((label, asm, code, srcs, cons))

    # zlib.crc32, not hash(): str hashing is PYTHONHASHSEED-salted, so the
    # case stream differed on every run and the campaign could not be
    # reproduced from its recorded seed.
    rng = random.Random(SEED ^ zlib.crc32(ISA_KEY.encode()))
    prov = provenance(isa)
    quarantine, undeclared, preserved = preflight(isa, gt, entries,
                                                  random.Random(SEED))
    print(f'{isa.label}: {len(entries)} entries, {len(quarantine)} quarantined, '
          f'{len(undeclared)} with an undeclared source', flush=True)
    for a, q in quarantine.items():
        print(f'  QUARANTINE {a}: {"; ".join(q["reasons"])}', flush=True)
    # The probe is authoritative about what can influence the result, so ADOPT
    # what it found rather than filing a defect against a hand-written list that
    # will drift again.  Registers the entry CONSTRAINS are excluded: a divisor
    # pinned non-zero must not be tainted, because a flip could take it to zero
    # mid-oracle and the case would trap instead of being measured.
    adopted = {}
    for i, (label, asm, code, srcs, cons) in enumerate(entries):
        u = undeclared.get(label)
        if not u:
            continue
        add = [r for r in u['also_influences'] if r not in cons]
        if not add:
            continue
        entries[i] = (label, asm, code, sorted(set(srcs) | set(add)), cons)
        adopted[label] = add
        print(f'  ADOPTED SOURCE {label}: declares {u["declared"]}, '
              f'{add} also change the result', flush=True)
    print(f'{isa.label}: adopted undeclared sources on {len(adopted)} entries',
          flush=True)

    consecutive_batch_failures = 0
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
        except Exception:  # noqa: BLE001 -- a checkpoint caught mid-write
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
            'undeclared_sources': undeclared,
            'adopted_sources': adopted,
            'preserved_flags': preserved,
            'metrics': metrics,
        }, open(tmp, 'w'), indent=1)
        os.replace(tmp, OUT)

    while time.time() < DEADLINE:
        rounds += 1
        live = [e for e in entries if e[0] not in quarantine and e[3]]
        # One `uv run` spawn costs ~0.44 s regardless of how many cases it carries,
        # so cases from many entries share a call.  mt_worker keys its rule cache on
        # the per-case bytes, so a mixed batch is fine.
        for chunk_start in range(0, len(live), ENTRIES_PER_BATCH):
            if time.time() >= DEADLINE:
                break
            chunk = live[chunk_start:chunk_start + ENTRIES_PER_BATCH]
            built, batch = [], []
            for label, asm, code, srcs, cons in chunk:
                excl = set(cons.get('exclude_flags', ())) | unicorn_suppressed(ISA_KEY, asm)
                # Flags the lift declares WRITTEN, plus flags the oracle measured
                # as PRESERVED.  A preserved flag's taint must survive the
                # instruction, and scoring only written flags never checked that:
                # 63 ARM64 forms (`adc`, `sbc`, `cset`) read a flag they do not
                # write, so the engine could have dropped its taint unnoticed.
                # `excl` still wins, so a flag the ISA leaves undefined stays out.
                chk = gpr_names + sorted(
                    (written_flags(isa, code) | set(preserved.get(label, ()))) - excl)
                # Never taint a register the entry pins: see constrained().
                tsrcs = [s2 for s2 in srcs if s2 not in constrained(cons)]
                if not tsrcs:
                    continue

                cases = []
                for _ in range(CASES_PER_FORM):
                    st = {n: rng.getrandbits(isa.bits) for n, _ in isa.gprs}
                    for f in isa.flags:
                        # f.width, not 1.  PPC's cr0 is a FOUR bit field and
                        # only bit 0 was ever set, so LT/GT/EQ sat permanently
                        # at an encoding no comparison can produce.
                        st[f.name] = rng.getrandbits(f.width)
                    if isa.sp:
                        st[isa.sp[0]] = GTSim.SP_VALUE
                    apply_cons(st, cons)
                    # An ARBITRARY taint mask over the operands this form reads.
                    # Density is drawn uniformly from 1 bit to the full width, so
                    # the sample covers sparse and dense taint alike instead of
                    # being ~90% single-bit as a one-bit-at-a-time sweep is.
                    # Canonicalise INSIDE the guard.  MIPS confines taint to
                    # the low 32 bits, so a mask approved here could be emptied
                    # afterwards, producing a case with nothing tainted that the
                    # oracle then scored as a free pass.
                    cst, ct = st, {}
                    while not any(ct.values()):
                        t = {}
                        for s2 in tsrcs:
                            w = fwidth.get(s2, isa.bits)
                            k = rng.randint(0, w)
                            m = 0
                            for b in rng.sample(range(w), k):
                                m |= 1 << b
                            if m:
                                t[s2] = m
                        cst, ct = isa.canonicalize(dict(st), t)
                    cases.append((cst, ct))

                built.append((label, asm, code, chk, cases))
                batch.extend({'label': label, 'asm': asm, 'bytes': code.hex(),
                              'srcs': list(t), 'state': st, 'taint': t} for st, t in cases)

            try:
                mts = mt_batch(isa, batch)
                consecutive_batch_failures = 0
            except Exception as e:  # noqa: BLE001 -- one bad chunk is survivable
                print(f'  mt_batch failed on a chunk of {len(chunk)}: {e}', flush=True)
                for label, _a, _c, _k, _cs in built:
                    m = metrics.setdefault(label, {})
                    m['mt_errors'] = m.get('mt_errors', 0) + 1
                # Surviving ONE bad chunk keeps a long campaign alive; surviving
                # every chunk turns a misconfiguration into a spin loop that
                # reports zero under-taints over zero cases and exits 0.  With
                # MT_ENGINE_ROOT pointing at a path that does not exist, this
                # produced 1080 rounds in 25 seconds and a well-formed report.
                consecutive_batch_failures += 1
                if consecutive_batch_failures >= MAX_CONSECUTIVE_BATCH_FAILURES:
                    raise SystemExit(
                        f'{consecutive_batch_failures} consecutive mt_batch '
                        f'failures: the engine at {ENGINE_ROOT!r} is not usable, '
                        f'so this campaign is measuring nothing.  Last error: {e}',
                    ) from e
                continue

            off = 0
            for label, asm, code, chk, cases in built:
                mts_e = mts[off:off + len(cases)]
                off += len(cases)
                m = metrics.setdefault(label, {})
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
                          'flag_gt_bits', 'flag_mt_bits', 'flag_over_bits',
                          # How much of the ground truth is real signal rather
                          # than input taint passing through.  `chk` scores ALL
                          # modelled GPRs including the ones the instruction only
                          # READS, and for those GT[r] is tautologically
                          # taint[r]: a no-op of the same length would produce it
                          # identically.  Measured corpus-wide that is ~62% of
                          # every gt_bits figure, so "bit-exact %" is in large
                          # part a test that a copy survives.  A case where NO
                          # scored register differs from passthrough witnesses
                          # nothing at all.
                          'info_bits', 'informative_checked', 'informative_exact'):
                    m.setdefault(k, 0)
                m.setdefault('ratio_sum', 0.0)
                for (st, t), mt in zip(cases, mts_e):
                    g, ex = gt.taint_flip_union(code, st, t)
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
                    # A no-op of the same length yields GT[r] == taint[r] for
                    # every r, so bits where the two AGREE witness nothing.
                    info = 0
                    for r in chk:
                        w = fwidth.get(r)
                        rmask = ((1 << w) - 1) if w else isa.mask
                        info += popcount(g.get(r, 0) ^ (t.get(r, 0) & rmask))
                    m['info_bits'] += info
                    if info:
                        m['informative_checked'] += 1
                    if under:
                        m['under'] += 1
                        if m['witnesses_written'] < WITNESS_CAP:
                            m['witnesses_written'] += 1
                            wf.write(json.dumps({
                                'isa': ISA_KEY, 'label': label, 'asm': asm,
                                'bytes': code.hex(),
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
                        if info:
                            m['informative_exact'] += 1

            checkpoint()

    checkpoint()
    wf.close()
    print(f'{isa.label} DONE: {rounds} rounds, '
          f'{sum(v.get("checked", 0) for v in metrics.values())} cases, '
          f'{sum(v.get("under", 0) for v in metrics.values())} under-taints', flush=True)


if __name__ == '__main__':
    main()
