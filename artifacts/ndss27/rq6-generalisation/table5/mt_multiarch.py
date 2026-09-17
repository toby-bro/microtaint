#!/usr/bin/env python3
# ruff: noqa: W505, E501, B905, RUF059, RUF100, I001
#   RUF100 and I001 are config differences, not defects: the source repo
#   selects BLE001/E402/C901 (so those noqa ARE used there) and sorts
#   `microtaint` as third-party, while it is first-party here.  No single
#   spelling satisfies both repos, and the body must stay byte-identical
#   to the harness that produced the published numbers.
#   Style only, and suppressed rather than rewritten: this harness is
#   vendored from the campaign that produced the published numbers, and
#   e.g. adding zip(strict=True) would change behaviour where the
#   original silently truncated.  Correctness is gated by
#   validate_oracle.py, not by restyling proven code.
# Vendored verbatim from the soundness-campaign harness; see README.md.
# Experiment script, not library code: see artifacts/ndss27/README.md,
# "Lint and type checking", for why annotations are not required here.
# mypy: disable-error-code="no-untyped-def, no-untyped-call, type-arg, no-any-return, var-annotated, assignment, index, arg-type, union-attr, operator, attr-defined, misc, call-overload, return-value, unreachable"
"""Multi-ISA soundness+precision benchmark for MicroTaint (microtaint-only).

Generalises benchmark.py's methodology to non-x86 ISAs, with FLAGS as first-class
taint sources AND checked outputs:

  * Ground truth (GTSim): exact 2^k noninterference enumeration over every tainted
    INPUT bit -- GPR bits AND flag bits -- recording which OUTPUT bits (GPR + flag)
    vary.  Arch-parametric (Unicorn arch/mode + reg/flag map per ISA).
  * MicroTaint: driven per-ISA with a state_format that includes the flags, so it
    must both PRODUCE flag taint (e.g. adds -> NZCV) and CONSUME it (e.g. adc <- C).
  * Coverage: a systematic single-bit sweep (taint each input bit of each source
    register/flag in turn -> "every input bit x every output bit"), plus multi-bit
    sparse cases for carry/borrow/compare interaction.

Soundness check:  mt_taint  ⊇  true_taint      (never miss)      -- the paper's claim.
Precision check:  mt_taint  ==  true_taint      (exact match rate) when GT is exact.

MicroTaint runs in a BATCHED SUBPROCESS (mt_worker), so its native x86 cell evaluation
never shares a process with GTSim's Unicorn -- that interaction segfaults LE-64 at
scale; subprocess isolation is how benchmark.py avoids it too.

Run:  python mt_multiarch.py --arch arm64 --per-instr 200
      python mt_multiarch.py --arch all
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field

MASK64 = (1 << 64) - 1


# --------------------------------------------------------------------------- #
# Per-ISA configuration
# --------------------------------------------------------------------------- #
@dataclass
class Flag:
    name: str            # microtaint / SLEIGH varnode name
    bit: int             # LSB position of the field inside its Unicorn register
    width: int = 1       # field width (PPC cr0 is a 4-bit condition field)
    reg: int | None = None   # its Unicorn register; None -> isa.uc_flags_reg

    @property
    def mask(self):
        return (1 << self.width) - 1


@dataclass
class ISA:
    key: str
    label: str
    mt_arch: str                 # microtaint Architecture enum name
    bits: int
    uc_arch: int
    uc_mode: int
    gprs: list[tuple[str, int]]  # (mt/sleigh name, unicorn reg id)
    flags: list[Flag]
    uc_flags_reg: int | None     # unicorn status register holding the flags
    _asm: object = None          # keystone Ks, or None for hand-assembled
    corpus: list = field(default_factory=list)  # (label, asm, src_regs)
    # Architectural canonical form: MIPS64 32-bit ops (sra/srav/addu/...) are only
    # DEFINED on sign-extended 32-bit operands; a random 64-bit seed is UNPREDICTABLE,
    # so Unicorn's arbitrary choice diverges from the engine and reports a spurious
    # under-taint.  Project seeds/taint into the defined regime when set.
    canon: int | None = None
    # Bits that must be set in the Unicorn flags register for a legal CPU state
    # (x86 EFLAGS bit 1 is reserved-and-always-1).
    uc_flags_base: int = 0
    # Stack pointer: (sleigh/microtaint name, unicorn reg id).  Pinned to a mapped
    # address and never tainted, so a push/pop round trip has a well-defined and
    # IDENTICAL address on both sides.  Memory whose ADDRESS is tainted is out of
    # scope -- there is no unified memory model for that here.
    sp: tuple[str, int] | None = None

    @property
    def mask(self):
        return (1 << self.bits) - 1

    @property
    def reg_names(self):
        return [g[0] for g in self.gprs] + [f.name for f in self.flags]

    def canonicalize(self, state, taint):
        """Project GPR seeds/taint into the ISA's architecturally-DEFINED regime
        (sign-extended `canon`-bit words) and confine taint to those low bits.
        Flags are 1-bit and untouched.  Identity when `canon` is unset."""
        w = self.canon
        if w is None or w >= self.bits:
            return state, taint
        low = (1 << w) - 1
        sign = 1 << (w - 1)
        high = self.mask ^ low
        gpr = {n for n, _ in self.gprs}

        def sx(v):
            v &= low
            return v | high if v & sign else v

        def sxt(t):
            # Taint on the SIGN bit implies taint on the extension bits.  In a
            # sign-extended 32-bit regime bits 32-63 are a function of bit 31,
            # so anything controlling bit 31 controls them too, and a mask that
            # claims otherwise describes no reachable state.  Getting this wrong
            # makes the oracle and the engine answer different questions: the
            # oracle re-extends after flipping bit 31 and sees the whole high
            # half move, while the engine was handed a mask that stopped at bit
            # 31, which reported 25,547 under-taints in 48,021 MIPS cases.
            t &= low
            return (t | high) if t & sign else t

        st = {k: (sx(v) if k in gpr else v) for k, v in state.items()}
        tt = {k: (sxt(t) if k in gpr else t) for k, t in taint.items()}
        return st, tt

    def assemble(self, asm: str) -> bytes:
        return bytes(self._asm.asm(asm, 0x1000)[0])


def _build_isas(which):
    import keystone as K
    import unicorn as U
    import unicorn.arm64_const as ua
    import unicorn.mips_const as um
    import unicorn.ppc_const as up
    import unicorn.riscv_const as ur
    import unicorn.x86_const as ux

    isas = {}

    if which in ('x86_64', 'all'):
        ks = K.Ks(K.KS_ARCH_X86, K.KS_MODE_64)
        isas['x86_64'] = ISA(
            'x86_64', 'AMD64', 'AMD64', 64, U.UC_ARCH_X86, U.UC_MODE_64,
            # EVERY general-purpose register, not four.  A register the oracle
            # does not model is pinned to zero, so a corpus entry naming it is
            # not measured but measured AT ZERO: `mul r8` was multiply-by-zero.
            # Worse, the state handed to the engine is built from this list, so
            # the oracle knew r8 was zero and the engine was never told r8
            # exists -- the two sides were asked about DIFFERENT machines, and
            # the engine was charged over-taint for assuming an unknown
            # multiplier.  Every name here round-trips through Unicorn AND the
            # engine's register map; see the widening validator.  RSP stays out
            # because it is the pinned memory anchor (`sp=` below).
            [('RAX', ux.UC_X86_REG_RAX), ('RBX', ux.UC_X86_REG_RBX),
             ('RCX', ux.UC_X86_REG_RCX), ('RDX', ux.UC_X86_REG_RDX),
             ('RSI', ux.UC_X86_REG_RSI), ('RDI', ux.UC_X86_REG_RDI),
             ('RBP', ux.UC_X86_REG_RBP)]
            + [(f'R{i}', getattr(ux, f'UC_X86_REG_R{i}')) for i in range(8, 16)],
            # NOTE: AF (auxiliary carry, EFLAGS bit 4) is deliberately EXCLUDED --
            # Ghidra's x86 SLEIGH does not model it, so microtaint can never produce
            # AF taint and every add/sub/adc would read as a spurious under-taint.
            # This is a declared modelling limitation, not an engine gap.
            [Flag('CF', 0), Flag('PF', 2), Flag('ZF', 6), Flag('SF', 7), Flag('OF', 11)],
            ux.UC_X86_REG_EFLAGS, ks, uc_flags_base=0x2,
            sp=('RSP', ux.UC_X86_REG_RSP))

    if which in ('arm64', 'all'):
        ks = K.Ks(K.KS_ARCH_ARM64, K.KS_MODE_LITTLE_ENDIAN)
        isas['arm64'] = ISA(
            'arm64', 'ARM64', 'ARM64', 64, U.UC_ARCH_ARM64, U.UC_MODE_ARM,
            [('x0', ua.UC_ARM64_REG_X0), ('x1', ua.UC_ARM64_REG_X1),
             ('x2', ua.UC_ARM64_REG_X2), ('x3', ua.UC_ARM64_REG_X3)],
            [Flag('NG', 31), Flag('ZR', 30), Flag('CY', 29), Flag('OV', 28)],
            ua.UC_ARM64_REG_NZCV, ks, sp=('sp', ua.UC_ARM64_REG_SP))

    if which in ('mips', 'all'):
        ks = K.Ks(K.KS_ARCH_MIPS, K.KS_MODE_MIPS64 | K.KS_MODE_BIG_ENDIAN)
        isas['mips'] = ISA(
            'mips', 'MIPS64BE', 'MIPS64BE', 64, U.UC_ARCH_MIPS,
            U.UC_MODE_MIPS64 | U.UC_MODE_BIG_ENDIAN,
            # HI/LO are included because `mult`, `multu`, `div` and `divu`
            # write ONLY there: without them those forms had no scored
            # destination at all and could not be wrong.  The GPR file is NOT
            # widened here: no MIPS corpus entry names a register outside these,
            # and Unicorn and the engine disagree on what $8-$11 are called
            # (Unicorn T0-T3, the engine A4-A7 under the N64 ABI), so widening
            # would risk varying a register whose taint is silently discarded.
            [('A0', um.UC_MIPS_REG_A0), ('A1', um.UC_MIPS_REG_A1),
             ('A2', um.UC_MIPS_REG_A2), ('V0', um.UC_MIPS_REG_V0),
             ('HI', um.UC_MIPS_REG_HI), ('LO', um.UC_MIPS_REG_LO)],
            [], None, ks, canon=32, sp=('sp', um.UC_MIPS_REG_SP))

    if which in ('ppc', 'all'):
        ks = K.Ks(K.KS_ARCH_PPC, K.KS_MODE_PPC32 | K.KS_MODE_BIG_ENDIAN)
        # PPC flags live in TWO different Unicorn registers: XER (carry/summary-overflow)
        # and the 4-bit CR0 condition field -- hence per-flag `reg` and `width`.
        isas['ppc'] = ISA(
            'ppc', 'PPC32BE', 'PPC32BE', 32, U.UC_ARCH_PPC,
            U.UC_MODE_PPC32 | U.UC_MODE_BIG_ENDIAN,
            # The whole register file: 83 PPC corpus entries read r0, r7 or r15,
            # which were pinned to zero, and 46 write r0, which nothing scored.
            # That was 937,888 cases, 26.8% of the PPC run, measuring nothing.
            # r1 is excluded because it is the stack pointer (`sp=` below).
            [('R3', up.UC_PPC_REG_3), ('R4', up.UC_PPC_REG_4),
             ('R5', up.UC_PPC_REG_5), ('R6', up.UC_PPC_REG_6)]
            + [(f'R{i}', getattr(up, f'UC_PPC_REG_{i}'))
               for i in [0, 2, *range(7, 32)]],
            # XER (xer_ca / xer_so) is EXCLUDED: Unicorn's PPC does not model it --
            # verified empirically, no XER bit affects `adde`'s result and carry-out is
            # never written back (0xffffffff+1 leaves XER=0).  The ORACLE therefore
            # cannot ground-truth PPC carry chains, so including it would manufacture
            # spurious under-taints against an engine that models carry correctly.
            # (Same limitation noted in tests/test_ppc_carry_chain.py.)  cr0 IS modelled.
            # All eight condition fields.  29 corpus entries compare into CR1-CR7
            # (`cmpw 1, 3, 4`), which nothing scored, so their entire result was
            # invisible; widening the GPRs alone would not have reached them.
            [Flag(f'cr{i}', 0, 4, getattr(up, f'UC_PPC_REG_CR{i}'))
             for i in range(8)],
            None, ks, sp=('r1', up.UC_PPC_REG_1))

    if which in ('riscv', 'all'):
        abi = {'t0': 5, 't1': 6, 't2': 7, 't3': 28}
        isas['riscv'] = ISA(
            'riscv', 'RISCV64', 'RISCV64', 64, U.UC_ARCH_RISCV, U.UC_MODE_RISCV64,
            [(n, getattr(ur, f'UC_RISCV_REG_X{x}')) for n, x in abi.items()],
            [], None, None, sp=('sp', ur.UC_RISCV_REG_X2))  # hand-assembled corpus

    return isas


def _identity_preserved(ops, out_vn) -> bool:
    """True if this lift assigns `out_vn` its OWN previous value, always.

    A flag an ISA leaves ARCHITECTURALLY UNDEFINED is sometimes lifted not by
    omitting the write but by writing the old value back.  x86 `shl r,imm` with a
    count != 1 leaves OF undefined, and SLEIGH emits::

        u_shamt = 0x7 & 0x3f            -> 7
        u5d5    = (u_shamt == 1)        -> 0
        u5d9    = !u5d5                 -> 1
        u5da    = u5d9 & OF             -> OF
        u5db    = u5d5 & <new OF>       -> 0
        OF      = u5da | u5db           -> OF        (identity)

    A name-based "does the lift write this register" test says OF IS written and
    lets the check through; Unicorn then models the real silicon (OF = MSB ^ CF),
    the 2^k oracle sees OF vary, and an engine that faithfully implements the
    model is reported as under-tainting.  That is an oracle artifact.

    Detected structurally rather than by naming `shl`, so it covers shr/sar/rol/ror
    and every other ISA's equivalent: fold the slice with constants known and
    register reads opaque, and see whether the result reduces to the output's own
    prior value.  Anything not provably identity is left CHECKED -- the failure
    mode of this function must be "still tests it", never "silently hides a bug".
    """
    KNOWN, SELF, OPAQUE = 'k', 's', 'o'

    def key(vn):
        return (vn.space.name, vn.offset, vn.size)

    env: dict = {}

    def val(vn):
        if vn.space.name == 'const':
            return (KNOWN, vn.offset)
        if key(vn) == key(out_vn):
            return (SELF, None)
        return env.get(key(vn), (OPAQUE, None))

    for op in ops:
        if op.output is None:
            continue
        n = op.opcode.name
        ins = [val(v) for v in op.inputs]
        r = (OPAQUE, None)
        full = (1 << (op.output.size * 8)) - 1
        if n == 'COPY':
            r = ins[0]
        elif all(i[0] == KNOWN for i in ins):
            a = ins[0][1]
            b = ins[1][1] if len(ins) > 1 else None
            if n == 'INT_AND':
                r = (KNOWN, a & b)
            elif n == 'INT_OR':
                r = (KNOWN, a | b)
            elif n == 'INT_XOR':
                r = (KNOWN, a ^ b)
            elif n == 'INT_EQUAL':
                r = (KNOWN, int(a == b))
            elif n == 'INT_NOTEQUAL':
                r = (KNOWN, int(a != b))
            elif n in ('BOOL_NEGATE', 'INT_NEGATE'):
                r = (KNOWN, (~a) & full)
            elif n == 'INT_SUB':
                r = (KNOWN, (a - b) & full)
            elif n == 'INT_ADD':
                r = (KNOWN, (a + b) & full)
        elif n in ('INT_AND', 'BOOL_AND') and len(ins) == 2:
            # 0 & x == 0 ;  all-ones & x == x
            for i, j in ((0, 1), (1, 0)):
                if ins[i][0] == KNOWN:
                    r = (KNOWN, 0) if ins[i][1] == 0 else (ins[j] if ins[i][1] == full else r)
        elif n in ('INT_OR', 'BOOL_OR') and len(ins) == 2:
            # all-ones | x == all-ones ;  0 | x == x
            for i, j in ((0, 1), (1, 0)):
                if ins[i][0] == KNOWN:
                    r = (KNOWN, full) if ins[i][1] == full else (ins[j] if ins[i][1] == 0 else r)
        env[key(op.output)] = r

    return env.get(key(out_vn), (OPAQUE, None))[0] == SELF


def written_flags(isa: ISA, code: bytes, report: dict | None = None) -> set:
    """Flags this instruction's SLEIGH lift actually DEFINES.

    Two exclusions, both of which would otherwise produce spurious under-taints
    against an engine that correctly refuses to model undefined state:

      * not written at all -- x86 leaves SF/ZF/PF/AF undefined after IMUL, and the
        lift simply omits them;
      * written back unchanged -- see _identity_preserved (x86 `shl` with count
        != 1 and friends).  A name-based test misses this one.

    `report`, if given, collects {flag: reason} for the excluded-but-named flags so
    the campaign can print exactly what it stopped checking.  Nothing is hidden
    silently.
    """
    try:
        from microtaint.sleigh.lifter import get_context
        from microtaint.types import Architecture
        tr = get_context(getattr(Architecture, isa.mt_arch)).translate(code, 0x1000)
        last = {}
        for op in tr.ops:
            if op.output is not None and op.output.space.name == 'register':
                last[str(op.output)] = op.output
        out = set()
        for f in isa.flags:
            vn = last.get(f.name)
            if vn is None:
                continue
            if _identity_preserved(tr.ops, vn):
                if report is not None:
                    report[f.name] = 'identity-preserved (architecturally undefined)'
                continue
            out.add(f.name)
        return out
    except Exception:  # noqa: BLE001 -- fall back to checking all flags
        return {f.name for f in isa.flags}


# --------------------------------------------------------------------------- #
# Ground truth: exact 2^k noninterference over GPR + flag input bits
# --------------------------------------------------------------------------- #
class GTSim:
    """Arch-parametric exact ground truth via Unicorn.  Fresh Uc per case (reused
    across the 2^k runs of that case).  Flags are read/written as bits of the ISA's
    Unicorn status register."""

    CODE = 0x1000
    STACK = 0x200000

    def __init__(self, isa: ISA):
        from unicorn import Uc
        self.isa = isa
        self._Uc = Uc

    def _fresh(self, code):
        uc = self._Uc(self.isa.uc_arch, self.isa.uc_mode)
        uc.mem_map(self.CODE, 0x10000)
        uc.mem_map(self.STACK, 0x10000)
        uc.mem_write(self.CODE, code + b'\x00' * 16)
        return uc

    SP_VALUE = 0x204000

    def _run(self, uc, code, gpr_vals, flag_bits):
        isa = self.isa
        if isa.sp is not None:
            uc.reg_write(isa.sp[1], self.SP_VALUE)
        for name, rid in isa.gprs:
            uc.reg_write(rid, gpr_vals[name] & isa.mask)
        if isa.flags:
            regvals = {}
            for f in isa.flags:
                rid = f.reg if f.reg is not None else isa.uc_flags_reg
                if rid is None:
                    continue
                base = isa.uc_flags_base if rid == isa.uc_flags_reg else 0
                regvals[rid] = regvals.get(rid, base) | ((flag_bits.get(f.name, 0) & f.mask) << f.bit)
            for rid, v in regvals.items():
                uc.reg_write(rid, v)
        uc.emu_start(self.CODE, self.CODE + len(code))
        out = {name: uc.reg_read(rid) & isa.mask for name, rid in isa.gprs}
        if isa.flags:
            cache = {}
            for f in isa.flags:
                rid = f.reg if f.reg is not None else isa.uc_flags_reg
                if rid is None:
                    out[f.name] = 0
                    continue
                if rid not in cache:
                    cache[rid] = uc.reg_read(rid)
                out[f.name] = (cache[rid] >> f.bit) & f.mask
        return out

    def taint(self, code, state, taint, budget=13):
        """Return ({reg: output_taint_mask}, exact) or (bitflip_lb, False) when
        k>budget.  For flags the mask is 1 bit."""
        isa = self.isa
        # collect tainted input bit positions across GPRs (0..bits-1) and flags (bit 0)
        positions = []  # (reg, bit)
        for name, _ in isa.gprs:
            tm = taint.get(name, 0)
            for b in range(isa.bits):
                if (tm >> b) & 1:
                    positions.append((name, b))
        for f in isa.flags:
            tm = taint.get(f.name, 0)
            for b in range(f.width):
                if (tm >> b) & 1:
                    positions.append((f.name, b))
        k = len(positions)
        flagset = {f.name for f in isa.flags}

        def split(assign_bits):
            g = {name: (state.get(name, 0) & ~taint.get(name, 0)) & isa.mask for name, _ in isa.gprs}
            fl = {f.name: (state.get(f.name, 0) & ~taint.get(f.name, 0)) & f.mask for f in isa.flags}
            for v, (reg, b) in zip(assign_bits, positions):
                if reg in flagset:
                    fl[reg] = fl.get(reg, 0) | (v << b)
                else:
                    g[reg] = (g[reg] | (v << b)) & isa.mask
            return g, fl

        if k == 0:
            return dict.fromkeys(isa.reg_names, 0), True

        uc = self._fresh(code)
        if k <= budget:  # EXACT
            outs = []
            for a in range(1 << k):
                bits = [(a >> i) & 1 for i in range(k)]
                g, fl = split(bits)
                try:
                    outs.append(self._run(uc, code, g, fl))
                except Exception:  # noqa: BLE001  (illegal/unmapped -> treated as no info)
                    continue
            if not outs:
                return dict.fromkeys(isa.reg_names, 0), True
            res = {}
            for r in isa.reg_names:
                orv = 0
                andv = None
                for o in outs:
                    orv |= o[r]
                    andv = o[r] if andv is None else (andv & o[r])
                res[r] = orv ^ (andv or 0)
            return res, True
        # k>budget: single-bit-flip LOWER bound (soundness only, not exact)
        base = None
        try:
            g0, fl0 = split([0] * k)
            base = self._run(uc, code, g0, fl0)
        except Exception:  # noqa: BLE001
            return dict.fromkeys(isa.reg_names, 0), False
        lb = dict.fromkeys(isa.reg_names, 0)
        for i in range(k):
            bits = [0] * k
            bits[i] = 1
            g, fl = split(bits)
            try:
                o = self._run(uc, code, g, fl)
            except Exception:  # noqa: BLE001
                continue
            for r in isa.reg_names:
                lb[r] |= base[r] ^ o[r]
        return lb, False


# --------------------------------------------------------------------------- #
# case generation:  systematic single-bit sweep (every input bit) + sparse combos
# --------------------------------------------------------------------------- #
def gen_cases(isa: ISA, label, code, srcs, rng, per_instr, constraints=None):
    asm = label
    constraints = constraints or {}
    src_gprs = [s for s in srcs if s not in {f.name for f in isa.flags}]
    src_flags = [s for s in srcs if s in {f.name for f in isa.flags}]
    base_state = {name: rng.getrandbits(isa.bits) for name, _ in isa.gprs}
    for f in isa.flags:
        base_state[f.name] = rng.getrandbits(1)
    cases = []

    def mk(taint):
        st = dict(base_state)
        # fresh random concrete state per case keeps coverage broad
        for name, _ in isa.gprs:
            st[name] = rng.getrandbits(isa.bits)
        for f in isa.flags:
            st[f.name] = rng.getrandbits(1)
        taint = dict(taint)
        # Constraints keep an instruction inside its defined domain across the WHOLE
        # 2^k enumeration, not just the base case: a divisor must be non-zero in
        # every point of the cube or Unicorn traps mid-oracle, so the register is
        # pinned to a non-zero value AND excluded from tainting.
        for reg, kind in constraints.items():
            if reg == 'exclude_flags':
                continue
            if isinstance(kind, int):
                st[reg] = kind
                taint.pop(reg, None)
                continue
            if kind == 'nonzero':
                st[reg] = (rng.getrandbits(isa.bits - 1) or 1) | 1
                taint.pop(reg, None)
        st2, tt2 = isa.canonicalize(st, dict(taint))
        return {'label': label, 'asm': asm, 'bytes': code.hex(), 'srcs': srcs,
                'state': st2, 'taint': tt2}

    # (a) single-bit sweep over every bit of every source GPR  -> every input bit
    for s in src_gprs:
        for b in range(isa.bits):
            cases.append(mk({s: 1 << b}))
    # (b) every source flag tainted alone
    for s in src_flags:
        cases.append(mk({s: 1}))
    # (c) sparse multi-bit (carry/borrow/compare interaction), k in 2..8
    n_extra = max(0, per_instr - len(cases))
    if not srcs:
        # Reads no register (movz / lui / auipc): there is nothing to taint, and the
        # only meaningful check is that a constant result is reported UNtainted.
        cases.append(mk({}))
        return cases
    for _ in range(n_extra):
        k = rng.randint(2, 8)
        t = dict.fromkeys(srcs, 0)
        picks = [(s, rng.randrange(isa.bits if s in src_gprs else 1)) for s in
                 [rng.choice(srcs) for _ in range(k)]]
        for s, b in picks:
            t[s] = t.get(s, 0) | (1 << b)
        if rng.random() < 0.3 and len(src_gprs) >= 2:      # correlated masks
            t[src_gprs[1]] = t.get(src_gprs[0], 0)
        cases.append(mk(t))
    return cases


# --------------------------------------------------------------------------- #
# microtaint via batched subprocess worker (isolation from GTSim's Unicorn)
# --------------------------------------------------------------------------- #
def _resolve_engine_root() -> str:
    """Locate the microtaint engine checkout to run under test.

    Honours $MT_ENGINE_ROOT; otherwise looks for a sibling checkout next to this
    benchmark repo.  Returned best-effort (never raises at import); mt_batch reports
    a clear error if the path is actually used and missing.
    """
    env = os.environ.get('MT_ENGINE_ROOT')
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in ('pcode-taint-engine', 'pcode-taint-engine-cvm'):
        p = os.path.normpath(os.path.join(here, os.pardir, cand))
        if os.path.isdir(p):
            return p
    return os.path.normpath(os.path.join(here, os.pardir, 'pcode-taint-engine'))


ENGINE_ROOT = _resolve_engine_root()


def mt_batch(isa: ISA, cases):
    """Run a batch of cases through microtaint in a fresh subprocess; return list of
    {reg: taint} aligned with `cases`."""
    if not os.path.isdir(ENGINE_ROOT):
        raise RuntimeError(
            f'microtaint engine not found at {ENGINE_ROOT!r}. '
            'Set MT_ENGINE_ROOT to your pcode-taint-engine checkout '
            '(or place it beside this benchmark repo).',
        )
    payload = {
        'mt_arch': isa.mt_arch,
        'sp': ([isa.sp[0], isa.bits, GTSim.SP_VALUE] if isa.sp else None),
        'regs': [[n, isa.bits] for n, _ in isa.gprs] + [[f.name, f.width] for f in isa.flags],
        'cases': [{'bytes': c['bytes'], 'state': c['state'], 'taint': c['taint']} for c in cases],
    }
    p = subprocess.run(
        ['uv', 'run', '--project', ENGINE_ROOT, 'python',
         os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mt_worker.py')],
        input=json.dumps(payload), capture_output=True, text=True,
        env={**os.environ, 'VIRTUAL_ENV': ''},
    )
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-2000:])
        raise RuntimeError(f'mt_worker failed rc={p.returncode}')
    return json.loads(p.stdout)


# --------------------------------------------------------------------------- #
# main run
# --------------------------------------------------------------------------- #
def run_isa(isa: ISA, per_instr, seed, batch=400):
    rng = random.Random(seed)
    gt = GTSim(isa)
    _w = {n: isa.mask for n, _ in isa.gprs}
    _w.update({f.name: f.mask for f in isa.flags})
    per = {}          # label -> [exact_ok, exact_n, under, checked]
    grand_under = []  # under-taint reports
    all_cases = []
    for entry in isa.corpus:
        label, code_spec, srcs = entry[0], entry[1], entry[2]
        cons = entry[3] if len(entry) > 3 else {}
        try:
            code = isa.assemble(code_spec) if isa._asm is not None else bytes.fromhex(code_spec)
            if not code:
                raise ValueError('assembler returned nothing')
        except Exception as e:  # noqa: BLE001 -- an unsupported mnemonic must not kill the run
            print(f'  [{isa.label}] SKIP {label!r}: {e}', flush=True)
            continue
        chk = [n for n, _ in isa.gprs] + sorted(
            written_flags(isa, code) - set(cons.get('exclude_flags', ())),
        )
        for c in gen_cases(isa, label, code, srcs, rng, per_instr, cons):
            c['chk'] = chk
            all_cases.append(c)
    # microtaint in batched subprocesses
    mt_all = []
    for i in range(0, len(all_cases), batch):
        mt_all.extend(mt_batch(isa, all_cases[i:i + batch]))
    # compare against GT
    for c, mt in zip(all_cases, mt_all):
        gtres, exact = gt.taint(bytes.fromhex(c['bytes']), c['state'], c['taint'])
        st = per.setdefault(c['label'], [0, 0, 0, 0])
        st[3] += 1
        chk = c.get('chk', isa.reg_names)
        missed = any((gtres.get(r, 0) & ~(mt.get(r, 0) or 0) & _w[r]) for r in chk)
        if missed:
            st[2] += 1
            grand_under.append({'label': c['label'], 'asm': c['asm'], 'bytes': c['bytes'],
                                'state': {k: hex(v) for k, v in c['state'].items() if v},
                                'taint': {k: hex(v) for k, v in c['taint'].items() if v},
                                'gt': {r: hex(gtres[r]) for r in isa.reg_names if gtres.get(r)},
                                'mt': {r: hex(mt.get(r, 0) or 0) for r in isa.reg_names if (mt.get(r, 0) or 0)}})
        if exact:
            st[1] += 1
            ok = all(((mt.get(r, 0) or 0) & _w[r]) == gtres.get(r, 0) for r in chk)
            if ok:
                st[0] += 1
    return per, grand_under, len(all_cases)


def main():
    from corpora import CORPORA  # per-ISA instruction corpora
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='arm64')
    ap.add_argument('--per-instr', type=int, default=200)
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()
    isas = _build_isas(args.arch if args.arch != 'all' else 'all')
    targets = list(isas) if args.arch == 'all' else [args.arch]
    for k in targets:
        isa = isas[k]
        isa.corpus = CORPORA[k](isa)
        t0 = time.time()
        per, under, ncases = run_isa(isa, args.per_instr, args.seed)
        print(f'\n===== {isa.label}: {ncases} cases, {len(isa.corpus)} instrs, {time.time() - t0:.0f}s =====')
        print(f'{"instr":28} {"exact%":>7} {"chk":>5} {"UNDER":>6}')
        tot_ok = tot_n = tot_u = 0
        for label in sorted(per):
            ok, n, u, chk = per[label]
            tot_ok += ok
            tot_n += n
            tot_u += u
            pct = f'{100 * ok / n:.0f}%' if n else 'n/a'
            flag = '  <-- UNDER' if u else ''
            print(f'  {label:26} {pct:>7} {n:5d} {u:6d}{flag}')
        agg = f'{100 * tot_ok / tot_n:.1f}%' if tot_n else 'n/a'
        print(f'  {"TOTAL":26} {agg:>7} {tot_n:5d} {tot_u:6d}')
        if under:
            with open(f'under_{k}.jsonl', 'w') as f:
                for r in under:
                    f.write(json.dumps(r) + '\n')
            print(f'  !!! {len(under)} UNDER-TAINT cases -> under_{k}.jsonl (first 3):')
            for r in under[:3]:
                print('   ', r)


if __name__ == '__main__':
    main()
