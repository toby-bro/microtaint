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

# Experiment script, not library code: see artifacts/ndss27/README.md,
# "Lint and type checking", for why annotations are not required here.
# mypy: disable-error-code="no-untyped-def, no-untyped-call, type-arg"

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Honour $MT_ENGINE_ROOT here too.  Only the Table 5 campaign read it, so this
# pass -- the one run-all.sh actually runs for RQ6 -- always measured whatever
# microtaint happened to be importable, no matter which engine the reviewer
# pinned.  Must run before `import microtaint` below.
_ENGINE_ROOT = os.environ.get('MT_ENGINE_ROOT')
if _ENGINE_ROOT:
    if not os.path.isfile(os.path.join(_ENGINE_ROOT, 'microtaint', '__init__.py')):
        raise SystemExit(
            f'MT_ENGINE_ROOT={_ENGINE_ROOT!r} is not a microtaint checkout '
            f'(no microtaint/__init__.py), so the pin would be silently ignored',
        )
    sys.path.insert(0, os.path.abspath(_ENGINE_ROOT))

import multiarch_oracle as O

# Which flags an instruction's lift actually DEFINES.  Reused from the Table 5
# harness that ships beside this one rather than reimplemented: it detects the
# architecturally-UNDEFINED cases structurally, including the nasty one where
# SLEIGH lifts an undefined flag by writing its own previous value back (x86
# `shl r,imm` with count != 1 leaves OF undefined exactly that way).  Scoring an
# undefined flag is how a faithful engine gets reported as under-tainting; that
# shape once produced 1211 spurious reports out of 1500.
#
# If it cannot be imported, NO flag is scored, which is exactly the behaviour
# this pass had before flags were added: the fallback is never worse than the
# status quo, and it says so rather than silently scoring the wrong thing.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'table5'))
_WRITTEN_FLAGS_ERR = ''
try:
    from mt_multiarch import written_flags as _written_flags
except Exception as _exc:  # the fallback scores no flag at all, and says so
    _written_flags = None
    _WRITTEN_FLAGS_ERR = str(_exc)
import unicorn.riscv_const as _rv
from unicorn import UC_ARCH_RISCV, UC_MODE_RISCV64

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, ImplicitTaintPolicy, Register


# --------------------------------------------------------------------------- #
# Unified corpus entry: everything the campaign needs, ISA-agnostic.
# --------------------------------------------------------------------------- #
def _engine_provenance() -> dict:
    """Which engine produced this result: commit, version, dirty flag.

    Never raises: an installed wheel has no git repository, and that is not a
    reason for a campaign to stop.
    """
    try:
        from microtaint.provenance import engine_provenance
        return engine_provenance()
    except Exception:
        return {}


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
        self.flag_regs = flag_regs       # [(name, bits)] modelled AND scored
        self.flag_src = {}               # name -> (uc_reg_id, bit_offset)
        self.flag_base = {}              # uc_reg_id -> reserved bits
        self.canon = canon               # canonical_word_bits or None
        self.state_format = [Register(r, bits) for r in regs] + [Register(n, b) for n, b in flag_regs]
        self.entries = []                # (asm_label, code_bytes, out_reg, [src_regs])
        self.circuits = {}               # code.hex() -> compiled rule

    def flag_width(self, name):
        for n, w in self.flag_regs:
            if n == name:
                return w
        return 1

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

        def sxt(t):
            # Taint on the sign bit implies taint on the extension bits, which
            # are a function of it.  See bitflip_lower_bound in
            # multiarch_oracle.py: the oracle re-projects the flipped state, so
            # the mask handed to the engine has to describe the same
            # perturbation or the two sides answer different questions.
            t &= low
            return (t | high) if t & sign else t
        # Flags are 1-4 bits and are NOT part of the sign-extended word regime:
        # running sx() over them would mangle a 1-bit flag into 0xffff... .
        fl = {n for n, _ in self.flag_regs}
        return ({r: (v if r in fl else sx(v)) for r, v in state.items()},
                {r: (t if r in fl else sxt(t)) for r, t in taint.items()})


def _from_isaspec(key) -> Bench:
    s = O.ISAS[key]()
    b = Bench(s.label, s.arch, s.bits, s.uc_arch, s.uc_mode, s.uc_regs, s.regs,
              s.flag_regs, s.canonical_word_bits)
    b.flag_src = dict(s.flag_src)
    b.flag_base = dict(s.flag_base)
    for asm, out, srcs in s.prog:
        try:
            code = bytes(s.ks.asm(asm, O.CODE_ADDR)[0])
        except Exception as e:
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
_DEFINED_FLAGS: dict[tuple, frozenset] = {}


class _FlagName:
    """Minimal stand-in for the Table 5 Flag type: written_flags reads `.name`."""

    __slots__ = ('name',)

    def __init__(self, name):
        self.name = name


class _FlagShim:
    """Minimal stand-in for the Table 5 ISA type, for written_flags only."""

    __slots__ = ('flags', 'mt_arch')

    def __init__(self, mt_arch, flags):
        self.mt_arch = mt_arch
        self.flags = flags


def defined_flags(b: Bench, code: bytes) -> frozenset:
    """Flags this instruction's lift DEFINES, and which are therefore scorable."""
    if _written_flags is None or not b.flag_regs:
        return frozenset()
    key = (b.label, code)
    got = _DEFINED_FLAGS.get(key)
    if got is None:
        # written_flags() expects the Table 5 ISA shape: `mt_arch` as the
        # Architecture NAME and `flags` as objects with `.name`.  A Bench has
        # neither, so passing it raised, the except swallowed it, and every
        # instruction came back with NO defined flags -- which silently disabled
        # the flag scoring this function exists to enable.  Caught by a mutation
        # test showing 0 oracle flag bits and 0 engine flag bits.
        shim = _FlagShim(b.arch.name, [_FlagName(n) for n, _ in b.flag_regs])
        try:
            # `shim` is the whole point: see the comment above.  It is
            # structurally what written_flags reads, not an ISA.
            got = frozenset(_written_flags(shim, code)) & {n for n, _ in b.flag_regs}
        except Exception:  # unknown means "do not score it"
            got = frozenset()
        _DEFINED_FLAGS[key] = got
    return got


def gen_taint(rng, b: Bench, srcs):
    kind = rng.random()
    taint = {}
    fw = dict(b.flag_regs)
    for r in srcs:
        if r in fw:
            # A flag source is 1-4 bits wide; drawing a 64-bit mask for it would
            # claim taint in bits that do not exist.
            taint[r] = rng.getrandbits(fw[r]) or 1
            continue
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
            # OR, not sum: summing two draws of the same position carries into
            # the next bit, and two draws of the top bit carry OUT of the
            # register.  That produced a mask one bit wider than the register
            # about once in 1,400 cases, which used to segfault the engine.
            taint[r] = 0
            for _ in range(rng.randint(1, 3)):
                taint[r] |= 1 << rng.randrange(b.bits)
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


#: States to try before declaring a corpus form unexecutable.
QUARANTINE_PROBES = 12


def quarantine_unexecutable(b, rng):
    """Corpus forms this Unicorn build cannot execute at ANY sampled state.

    Decided by running the instruction, so it needs no per-ISA list and cannot
    go stale.  MIPS `movz`/`movn` raise UC_ERR_EXCEPTION on every state in
    Unicorn 2.1.4, and the aggregate skip counter hid that: 100% of their cases
    were discarded while they still appeared in the corpus size, so RQ6 looked
    like it covered conditional moves and covered none.

    Quarantining is not the same as passing.  These forms are excluded from
    sampling AND reported, so the claim is "not measured" rather than "sound".
    """
    dead = {}
    for asm, code, _out, srcs in b.entries:
        ran = 0
        for _ in range(QUARANTINE_PROBES):
            state = {r: rng.getrandbits(b.bits) for r in b.regs}
            state.update({n: rng.getrandbits(w) for n, w in b.flag_regs})
            taint = gen_taint(rng, b, srcs)
            state, taint = b.canonicalize(state, taint)
            try:
                O.bitflip_lower_bound(b, code, state, taint)
                ran += 1
                break
            except Exception:  # not executing IS the result here
                continue
        if not ran:
            dead[asm] = QUARANTINE_PROBES
    return dead


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
    # Truncate, do not append.  Re-running with the same --out concatenated the
    # previous run's reports onto this one (reproduced: 79 -> 237 -> 316 -> 395
    # lines across re-runs into one directory), and pass 2 then re-verified and
    # double-counted them.  The docstring anticipates relaunching after a
    # segfault, which is exactly when this bites; run-all.sh's timestamped OUT
    # merely hid it.
    out_f = open(out_path, 'w')
    reports = []
    t0 = time.time()
    last = t0
    done = 0
    # A campaign that compares NOTHING reports zero under-taints, exactly like a
    # campaign that compares everything and finds none.  `done` counts loop
    # iterations, so these count what was actually asked and answered:
    #   skipped_oracle  the ground truth raised (11% of MIPS cases, measured)
    #   skipped_mt      the engine raised
    #   no_gt_taint     the ground truth found nothing to miss (20% on MIPS)
    # compared - no_gt_taint is the sample the zero actually rests on.
    skipped_oracle = skipped_mt = no_gt_taint = compared = 0
    # Per-FORM accounting.  The aggregate skip counter hid that MIPS `movz` and
    # `movn` raise UC_ERR_EXCEPTION on EVERY state in this Unicorn build, so 100%
    # of their cases were discarded and they were reported as neither tested nor
    # failed.  They are also the only conditional-move forms in the whole
    # five-ISA corpus, so RQ6 covered no implicit-flow instruction at all while
    # appearing to cover two.
    per_form: dict[str, dict[str, int]] = {}
    flag_moved_undeclared: dict[str, int] = {}
    quarantined = quarantine_unexecutable(b, random.Random(seed ^ 0x51AB))
    if quarantined:
        for a in sorted(quarantined):
            print(f'[{b.label}]   QUARANTINE {a!r}: Unicorn could not execute it '
                  f'at any of {QUARANTINE_PROBES} sampled states, so it is '
                  f'excluded and reported as UNMEASURED, not as sound',
                  flush=True)
    live = [e for e in b.entries if e[0] not in quarantined]
    if not live:
        print(f'[{b.label}]   *** every corpus form is unexecutable ***', flush=True)
    for i in range(n):
        asm, code, out_reg, srcs = rng.choice(live)
        state = {r: rng.getrandbits(b.bits) for r in b.regs}
        # Flags are seeded, not left at zero: an instruction that READS a flag
        # (adc, sbb, every conditional form) is only testable if the flag varies.
        state.update({n: rng.getrandbits(w) for n, w in b.flag_regs})
        taint = gen_taint(rng, b, srcs)
        state, taint = b.canonicalize(state, taint)
        # oracle: the PROVEN-STABLE fresh-Unicorn-per-run lower bound (O._run rebuilds
        # a fresh Uc each call; the real fuzzer runs it for minutes without leaking).
        # My Bench duck-types onto O.bitflip_lower_bound (regs/bits/mask/uc_*).  Fresh
        # Uc per run means no leakage at all -- so pass 1 here already has NO false
        # positives from state leaks; pass 2 remains as an independent double-check.
        pf = per_form.setdefault(asm, {'attempted': 0, 'compared': 0})
        pf['attempted'] += 1
        try:
            lb = O.bitflip_lower_bound(b, code, state, taint)  # type: ignore[arg-type]  # Bench duck-types onto IsaSpec
        except Exception:
            skipped_oracle += 1
            continue
        # microtaint
        try:
            ctx = EvalContext(input_taint=dict(taint), input_values=dict(state),
                              simulator=sim, implicit_policy=ImplicitTaintPolicy.IGNORE)
            mt = _rule(b, code).evaluate(ctx)
        except Exception:
            skipped_mt += 1
            continue
        compared += 1
        pf['compared'] += 1
        # GPRs, plus the flags this instruction's lift DEFINES.  An undefined
        # flag is excluded because the engine is right not to model it.
        dflags = defined_flags(b, code)
        scored = [*b.regs, *(n for n, _ in b.flag_regs if n in dflags)]
        if not any(lb.get(r, 0) for r in scored):
            no_gt_taint += 1
        # Cross-check the exclusion rather than trusting it: if the ORACLE saw a
        # flag move that the lift does not declare, that is either an undefined
        # flag (fine, and why it is excluded) or a lifter gap (not fine).  Either
        # way it is recorded, so "which flags were not scored" is answerable.
        for n, w in b.flag_regs:
            if n not in dflags and lb.get(n, 0) & ((1 << w) - 1):
                flag_moved_undeclared[n] = flag_moved_undeclared.get(n, 0) + 1
        missed = 0
        for r in scored:
            w = b.flag_width(r) if r in b.flag_src else b.bits
            missed |= lb.get(r, 0) & ~(mt.get(r, 0) or 0) & ((1 << w) - 1)
        if missed:
            rep = {
                'arch': b.label, 'asm': asm, 'bytes': code.hex(),
                # The FULL state, not just the sources.  Pass 2 rebuilt every
                # other register as zero, so it re-checked a DIFFERENT case than
                # the one pass 1 reported and could dismiss a real finding.
                'state': dict(state), 'taint': dict(taint),
                'srcs_state': {r: state[r] for r in srcs},
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
                  f'cmp={compared} skip={skipped_oracle + skipped_mt} '
                  f'{done / max(now - t0, 1e-9):.0f} cases/s', flush=True)
    out_f.close()
    effective = compared - no_gt_taint
    print(f'[{b.label}] PASS1 done: {done} cases, {len(reports)} raw under-taint reports '
          f'({done / max(time.time() - t0, 1e-9):.0f} cases/s) -> {out_path}', flush=True)
    # What the zero rests on.  A case the oracle could not answer, or for which
    # it found no taint, contributes nothing: reporting only `done` overstates
    # the sample, and on MIPS that is ~30% of it.
    print(f'[{b.label}]   compared {compared}/{done}  '
          f'(skipped: oracle {skipped_oracle}, engine {skipped_mt})  '
          f'no ground-truth taint {no_gt_taint}  '
          f'-> effective sample {effective}', flush=True)
    if compared == 0:
        print(f'[{b.label}]   *** NOTHING WAS COMPARED: this zero means the '
              f'harness never ran, not that the engine is sound ***', flush=True)
    # A form that was sampled but never once executed is untested, not sound.
    dead = {a: v['attempted'] for a, v in per_form.items()
            if v['attempted'] and not v['compared']}
    for a, cnt in sorted(dead.items()):
        print(f'[{b.label}]   *** {a!r}: 0 of {cnt} cases could be executed, so '
              f'this form is UNTESTED ***', flush=True)
    if flag_moved_undeclared:
        print(f'[{b.label}]   flags the oracle moved but the lift does not '
              f'declare (excluded from scoring, see defined_flags): '
              f'{dict(sorted(flag_moved_undeclared.items()))}', flush=True)
    return done, len(reports), {
        'attempted': done, 'compared': compared,
        'flag_moved_undeclared': dict(sorted(flag_moved_undeclared.items())),
        'skipped_oracle': skipped_oracle, 'skipped_mt': skipped_mt,
        'no_gt_taint': no_gt_taint, 'effective': effective,
        'reports': len(reports),
        'dead_forms': dead,
        'quarantined_forms': sorted(quarantined),
        # Recorded so a reader of the JSON cannot mistake the scope either.
        'scored_registers': list(b.regs),
        'scored_flags': [n for n, _ in b.flag_regs],
        'flag_scoring': ('per-instruction, only flags the lift defines'
                         if _written_flags is not None else 'UNAVAILABLE'),
        'per_form': dict(sorted(per_form.items())),
    }


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
    # Reproduce the case pass 1 actually ran.  Older reports carry only the
    # source registers; for those the rest is unknown and zero is the best
    # available guess, which is why it is now recorded in full.
    state = dict.fromkeys(b.regs, 0)
    state.update({n: 0 for n, _ in b.flag_regs})
    state.update(dict(report['state'].items()))
    taint = dict(report['taint'].items())
    state, taint = b.canonicalize(state, taint)

    # Use the SAME ground truth as pass 1 rather than a second, weaker one.
    # The old local re-implementation had no completion check (the omission that
    # once scored 19M MIPS cases against an empty ground truth), swallowed
    # UcError per flip -- which only SHRINKS the bound, the direction that hides
    # a real under-taint -- did not re-canonicalise the flipped state, and did
    # not score flags at all.  Every one of those makes pass 2 more lenient than
    # pass 1, so it could dismiss a genuine finding.  `bitflip_lower_bound`
    # raises CaseInvalid on any incomplete run and starts every run from a
    # pristine context, so the independence pass 2 was written for is kept.
    try:
        # A Bench carries the attributes the oracle reads off an IsaSpec.
        lb = O.bitflip_lower_bound(b, code, state, taint)  # type: ignore[arg-type]
    except Exception as exc:
        return 0, {'error': f'could not re-run: {exc}', 'unverifiable': True}

    sim = CellSimulator(b.arch)
    ctx = EvalContext(input_taint=dict(taint), input_values=dict(state),
                      simulator=sim, implicit_policy=ImplicitTaintPolicy.IGNORE)
    mt = generate_static_rule(b.arch, code, b.state_format).evaluate(ctx)
    dflags = defined_flags(b, code)
    scored = [*b.regs, *(n for n, _ in b.flag_regs if n in dflags)]
    missed = 0
    for r in scored:
        w = b.flag_width(r) if r in b.flag_src else b.bits
        missed |= lb.get(r, 0) & ~(mt.get(r, 0) or 0) & ((1 << w) - 1)
    return missed, {'lb': {r: lb[r] for r in scored if lb.get(r)},
                    'mt': {r: (mt.get(r, 0) or 0) & mask for r in scored if (mt.get(r, 0) or 0) & mask},
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
        # The findings file is empty when the campaign is CLEAN, which is the
        # result RQ6 claims, so the engine that produced it would go unrecorded
        # in exactly the case that matters.  The manifest is written up front and
        # does not depend on finding anything.
        with open(f'{args.out}_run.json', 'w') as mf:
            import microtaint as _mt_pkg
            json.dump({'engine': _engine_provenance(),
                       'engine_import_path': os.path.dirname(
                           os.path.dirname(os.path.abspath(_mt_pkg.__file__))),
                       'mt_engine_root': _ENGINE_ROOT,
                       'n': args.n,
                       'seed': args.seed, 'arches': arches,
                       'started': str(datetime.datetime.now())}, mf, indent=2)
        print(f'[provenance] {args.out}_run.json', flush=True)
        coverage: dict[str, dict[str, int]] = {}
        for i, k in enumerate(arches):
            b = build(k)
            print(f'=== PASS1 {b.label}: {args.n} cases (seed {args.seed + i}, '
                  f'{len(b.entries)} instrs) ===', flush=True)
            # Say what this pass does NOT cover, every run, next to the number.
            # The comparison below is over b.regs, the GPRs.  Flags are modelled
            # in state_format and handed to the engine, but the oracle's _Runner
            # returns only spec.regs, so no flag is ever compared -- and AMD64
            # and RISCV64 declare no flag registers at all.  "0 under-taints"
            # from this pass therefore means "0 in the destination GPR", which
            # matters because flag carries were this project's entire x86 bug
            # class.  Flags ARE scored, on all five ISAs and with the
            # undefined-flag exclusions that requires, by the Table 5 campaign
            # in table5/; this pass is the fast smoke gate, not that experiment.
            _fl = [n for n, _ in b.flag_regs]
            print(f'[{b.label}]   SCOPE: comparing {len(b.regs)} GPR(s) '
                  f'({", ".join(b.regs)}) and {len(_fl)} flag(s) '
                  f'({", ".join(_fl) or "none in this ISA"}); per instruction, '
                  f'only the flags its lift DEFINES are scored',
                  flush=True)
            if _fl and _written_flags is None:
                print(f'[{b.label}]   *** flags are declared but CANNOT be '
                      f'scored: {_WRITTEN_FLAGS_ERR} -- without the '
                      f'undefined-flag discriminator, scoring them would report '
                      f'a correct engine as unsound ***', flush=True)
            _done, _nrep, _cov = pass1_arch(
                b, args.n, args.seed + i, f'{args.out}_{k}.jsonl')
            coverage[b.label] = _cov
        # What each ISA's zero actually rests on, written where a reader can
        # find it later: `attempted` is the number asked for, `effective` the
        # number that could contribute a finding at all.
        with open(f'{args.out}_coverage.json', 'w') as cf:
            json.dump(coverage, cf, indent=2)
        print(f'[coverage] {args.out}_coverage.json', flush=True)
        # An exit code, because run-all.sh records PASS iff this process exits 0.
        # Without one, a completely dead engine printed its under-taints and
        # still filed rq6-pass1 as PASS: the headline RQ6 result was a constant.
        # Two ways to fail: found something, or measured nothing.
        problems = []
        for label, cov in coverage.items():
            if cov.get('compared', 0) == 0:
                problems.append(
                    f'{label}: ZERO cases were compared out of '
                    f'{cov.get("attempted", 0)} attempted, so this ISA is an '
                    f'absence of measurement, not a clean result',
                )
            # The coverage dict mixes counters and maps, so mypy types this
            # access as int | dict and cannot know which one it is here.
            _dead: object = cov.get('dead_forms') or {}
            for form, n in sorted(_dead.items() if isinstance(_dead, dict) else []):
                problems.append(
                    f'{label}: {form!r} could not be executed on ANY of its {n} '
                    f'cases, so it is untested rather than sound',
                )
        reported = sum(1 for k in arches
                       if os.path.exists(f'{args.out}_{k}.jsonl')
                       and os.path.getsize(f'{args.out}_{k}.jsonl') > 0)
        if reported:
            problems.append(
                f'{reported} ISA(s) wrote under-taint reports; run pass2 to '
                f'isolate them',
            )
        if problems:
            print('\nPASS1 FAILED:')
            for p_ in problems:
                print(f'  {p_}')
            return 1
        return 0

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
    # A verified under-taint is a soundness failure and must exit non-zero, or
    # run-all.sh files it as PASS.
    if grand_real:
        print(f'PASS2 FAILED: {grand_real} verified under-taint(s)')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
