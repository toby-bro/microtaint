#!/usr/bin/env python3
"""Single-bit-flip noninterference oracle for every supported ISA.

Checks MicroTaint's synthesised rules against a ground-truth oracle on x86-64,
ARM64, MIPS64BE and PPC32BE, to substantiate that soundness is intrinsic to the
synthesis rather than x86 tuning (reviewer D2).

The oracle
----------
Output bit b is truly tainted at (V, T) iff flipping some tainted input bits can
change it.  Flipping ONE tainted bit at a time yields a LOWER BOUND on the true
taint -- a necessary condition for soundness:

    sound  =>  microtaint_taint  >=  bitflip_lower_bound        (no under-taint)

It cannot certify soundness (exhaustive ground truth is 2^k), only refute it.
Where k is small enough we additionally enumerate all 2^k assignments for the
EXACT ground truth, which also detects OVER-tainting.

Every run uses a FRESH Unicorn instance.  Reusing one leaks state between runs
and manufactures impossible lower bounds -- e.g. tainted high-32 bits for an
`add eax, ebx` whose zero-extension pins them to 0.  That is precisely the
artifact that made 8/566110 fuzzer reports false positives.

Why these ISAs
--------------
* MIPS64BE -- SLEIGH reports NO condition-flag registers: compares write GPRs
  (`slt`).  Every under-taint the x86 campaign fixed was flag- or select-related,
  so a flag-free RISC removes that entire bug class by construction.
* PPC32BE  -- carry lives in XER (xer_ca), conditions in cr0..cr7, rather than in
  a flat set of 1-bit flags.  Unicorn 2.1.4 executes PPC32 but NOT PPC64
  (UC_ERR_EXCEPTION), so PPC64 has no oracle and is out of scope.
* ARM64    -- NZCV condition flags (SLEIGH: ng/zr/cy/ov).

Run:  python multiarch_oracle.py [--arch all] [--verbose]
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass, field
from typing import Callable

import unicorn.arm64_const as uc_arm64
import unicorn.mips_const as uc_mips
import unicorn.ppc_const as uc_ppc
import unicorn.sparc_const as uc_sparc
import unicorn.x86_const as uc_x86
from keystone import (
    KS_ARCH_ARM64,
    KS_ARCH_MIPS,
    KS_ARCH_PPC,
    KS_ARCH_SPARC,
    KS_ARCH_X86,
    KS_MODE_64,
    KS_MODE_BIG_ENDIAN,
    KS_MODE_LITTLE_ENDIAN,
    KS_MODE_MIPS64,
    KS_MODE_PPC32,
    KS_MODE_SPARC32,
    Ks,
)
from unicorn import (
    UC_ARCH_ARM64,
    UC_ARCH_MIPS,
    UC_ARCH_PPC,
    UC_ARCH_SPARC,
    UC_ARCH_X86,
    UC_MODE_64,
    UC_MODE_ARM,
    UC_MODE_BIG_ENDIAN,
    UC_MODE_MIPS64,
    UC_MODE_PPC32,
    UC_MODE_SPARC32,
    Uc,
)

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, Register

CODE_ADDR = 0x1000
EXHAUSTIVE_MAX_K = 12  # 2^k enumeration budget for the EXACT (over-taint) check


@dataclass
class IsaSpec:
    """Everything needed to lift, assemble, execute and score one ISA."""

    label: str
    arch: Architecture
    bits: int
    ks: Ks
    uc_arch: int
    uc_mode: int
    # state-register names we seed/score, and their Unicorn ids
    regs: list[str]
    uc_regs: dict[str, int]
    # (assembly, scored-output-register, SOURCE registers the instruction
    # actually reads).  Sources must be declared per-program: tainting a
    # positional slice of `regs` silently taints registers the instruction
    # never reads, making the whole check vacuous.
    prog: list[tuple[str, str, list[str]]] = field(default_factory=list)

    @property
    def mask(self) -> int:
        return (1 << self.bits) - 1

    def state_format(self) -> list[Register]:
        return [Register(r, self.bits) for r in self.regs]


def _mips() -> IsaSpec:
    gpr = ['ZERO', 'AT', 'V0', 'V1', 'A0', 'A1', 'A2', 'A3', 'T0', 'T1']
    names = [
        'zero', 'at', 'v0', 'v1', 'a0', 'a1', 'a2', 'a3',
        't0', 't1', 't2', 't3', 't4', 't5', 't6', 't7',
        's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7',
        't8', 't9', 'k0', 'k1', 'gp', 'sp', 's8', 'ra',
    ]
    uc_regs = {n.upper(): getattr(uc_mips, f'UC_MIPS_REG_{i}') for i, n in enumerate(names)}
    return IsaSpec(
        label='MIPS64BE',
        arch=Architecture.MIPS64BE,
        bits=64,
        ks=Ks(KS_ARCH_MIPS, KS_MODE_MIPS64 | KS_MODE_BIG_ENDIAN),
        uc_arch=UC_ARCH_MIPS,
        uc_mode=UC_MODE_MIPS64 | UC_MODE_BIG_ENDIAN,
        regs=gpr,
        uc_regs=uc_regs,
        prog=[
            ('addu $2, $4, $5', 'V0', ['A0', 'A1']),
            ('subu $2, $4, $5', 'V0', ['A0', 'A1']),
            ('and $2, $4, $5', 'V0', ['A0', 'A1']),
            ('or $2, $4, $5', 'V0', ['A0', 'A1']),
            ('xor $2, $4, $5', 'V0', ['A0', 'A1']),
            ('nor $2, $4, $5', 'V0', ['A0', 'A1']),
            ('sll $2, $4, 4', 'V0', ['A0']),
            ('srl $2, $4, 4', 'V0', ['A0']),
            ('sra $2, $4, 4', 'V0', ['A0']),
            ('slt $2, $4, $5', 'V0', ['A0', 'A1']),   # compare -> GPR, no flags
            ('sltu $2, $4, $5', 'V0', ['A0', 'A1']),
            ('movz $2, $4, $5', 'V0', ['A0', 'A1']),  # conditional move
            ('movn $2, $4, $5', 'V0', ['A0', 'A1']),
            ('sllv $2, $4, $5', 'V0', ['A0', 'A1']),  # variable shift
            ('srlv $2, $4, $5', 'V0', ['A0', 'A1']),
        ],
    )


def _ppc() -> IsaSpec:
    gpr = [f'R{i}' for i in range(8)]
    uc_regs = {f'R{i}': getattr(uc_ppc, f'UC_PPC_REG_{i}') for i in range(32)}
    return IsaSpec(
        label='PPC32BE',
        arch=Architecture.PPC32BE,
        bits=32,
        ks=Ks(KS_ARCH_PPC, KS_MODE_PPC32 | KS_MODE_BIG_ENDIAN),
        uc_arch=UC_ARCH_PPC,
        uc_mode=UC_MODE_PPC32 | UC_MODE_BIG_ENDIAN,
        regs=gpr,
        uc_regs=uc_regs,
        prog=[
            ('add 3, 4, 5', 'R3', ['R4', 'R5']),
            ('subf 3, 4, 5', 'R3', ['R4', 'R5']),   # rD = rB - rA
            ('and 3, 4, 5', 'R3', ['R4', 'R5']),
            ('or 3, 4, 5', 'R3', ['R4', 'R5']),
            ('xor 3, 4, 5', 'R3', ['R4', 'R5']),
            ('nand 3, 4, 5', 'R3', ['R4', 'R5']),
            ('slw 3, 4, 5', 'R3', ['R4', 'R5']),    # variable shift
            ('srw 3, 4, 5', 'R3', ['R4', 'R5']),
            ('sraw 3, 4, 5', 'R3', ['R4', 'R5']),
            ('slwi 3, 4, 4', 'R3', ['R4']),
            ('mullw 3, 4, 5', 'R3', ['R4', 'R5']),
            ('neg 3, 4', 'R3', ['R4']),
            ('extsb 3, 4', 'R3', ['R4']),     # sign-extend byte
            ('cntlzw 3, 4', 'R3', ['R4']),
        ],
    )


def _arm64() -> IsaSpec:
    gpr = [f'X{i}' for i in range(8)]
    uc_regs = {f'X{i}': getattr(uc_arm64, f'UC_ARM64_REG_X{i}') for i in range(31)}
    return IsaSpec(
        label='ARM64',
        arch=Architecture.ARM64,
        bits=64,
        ks=Ks(KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN),
        uc_arch=UC_ARCH_ARM64,
        uc_mode=UC_MODE_ARM,
        regs=gpr,
        uc_regs=uc_regs,
        prog=[
            ('add x0, x1, x2', 'X0', ['X1', 'X2']),
            ('sub x0, x1, x2', 'X0', ['X1', 'X2']),
            ('and x0, x1, x2', 'X0', ['X1', 'X2']),
            ('orr x0, x1, x2', 'X0', ['X1', 'X2']),
            ('eor x0, x1, x2', 'X0', ['X1', 'X2']),
            ('bic x0, x1, x2', 'X0', ['X1', 'X2']),
            ('lsl x0, x1, #4', 'X0', ['X1']),
            ('lsr x0, x1, #4', 'X0', ['X1']),
            ('asr x0, x1, #4', 'X0', ['X1']),
            ('lslv x0, x1, x2', 'X0', ['X1', 'X2']),   # variable shift
            ('mul x0, x1, x2', 'X0', ['X1', 'X2']),
            ('madd x0, x1, x2, x3', 'X0', ['X1', 'X2', 'X3']),
            ('sxtb x0, w1', 'X0', ['X1']),
            ('clz x0, x1', 'X0', ['X1']),
            ('rbit x0, x1', 'X0', ['X1']),       # bit reverse -- pure routing
            ('rev x0, x1', 'X0', ['X1']),        # byte reverse
        ],
    )


def _sparc() -> IsaSpec:
    # %g0 is hardwired to zero -- never seed or score it, or every "flip" of it is
    # silently discarded by the hardware and the oracle compares nonsense.
    gpr = ['G1', 'G2', 'G3', 'G4', 'G5', 'O0', 'O1', 'O2']
    names = (
        [f'g{i}' for i in range(8)]
        + [f'o{i}' for i in range(8)]
        + [f'l{i}' for i in range(8)]
        + [f'i{i}' for i in range(8)]
    )
    uc_regs = {n.upper(): getattr(uc_sparc, f'UC_SPARC_REG_{n.upper()}') for n in names}
    return IsaSpec(
        label='SPARC32BE',
        arch=Architecture.SPARC32BE,
        bits=32,
        ks=Ks(KS_ARCH_SPARC, KS_MODE_SPARC32 | KS_MODE_BIG_ENDIAN),
        uc_arch=UC_ARCH_SPARC,
        uc_mode=UC_MODE_SPARC32 | UC_MODE_BIG_ENDIAN,
        regs=gpr,
        uc_regs=uc_regs,
        prog=[
            ('add %g1, %g2, %g3', 'G3', ['G1', 'G2']),
            ('sub %g1, %g2, %g3', 'G3', ['G1', 'G2']),
            ('and %g1, %g2, %g3', 'G3', ['G1', 'G2']),
            ('or %g1, %g2, %g3', 'G3', ['G1', 'G2']),
            ('xor %g1, %g2, %g3', 'G3', ['G1', 'G2']),
            ('andn %g1, %g2, %g3', 'G3', ['G1', 'G2']),
            ('xnor %g1, %g2, %g3', 'G3', ['G1', 'G2']),
            ('sll %g1, %g2, %g3', 'G3', ['G1', 'G2']),   # variable shift
            ('srl %g1, %g2, %g3', 'G3', ['G1', 'G2']),
            ('sra %g1, %g2, %g3', 'G3', ['G1', 'G2']),
            ('sll %g1, 4, %g3', 'G3', ['G1']),     # constant shift
            ('addcc %g1, %g2, %g3', 'G3', ['G1', 'G2']),  # writes the icc condition codes
            ('subcc %g1, %g2, %g3', 'G3', ['G1', 'G2']),
            ('umul %g1, %g2, %g3', 'G3', ['G1', 'G2']),
        ],
    )


def _amd64() -> IsaSpec:
    gpr = ['RAX', 'RBX', 'RCX', 'RDX']
    uc_regs = {
        'RAX': uc_x86.UC_X86_REG_RAX,
        'RBX': uc_x86.UC_X86_REG_RBX,
        'RCX': uc_x86.UC_X86_REG_RCX,
        'RDX': uc_x86.UC_X86_REG_RDX,
    }
    return IsaSpec(
        label='AMD64',
        arch=Architecture.AMD64,
        bits=64,
        ks=Ks(KS_ARCH_X86, KS_MODE_64),
        uc_arch=UC_ARCH_X86,
        uc_mode=UC_MODE_64,
        regs=gpr,
        uc_regs=uc_regs,
        prog=[
            ('add rax, rbx', 'RAX', ['RAX', 'RBX']),
            ('sub rax, rbx', 'RAX', ['RAX', 'RBX']),
            ('and rax, rbx', 'RAX', ['RAX', 'RBX']),
            ('or rax, rbx', 'RAX', ['RAX', 'RBX']),
            ('xor rax, rbx', 'RAX', ['RAX', 'RBX']),
            ('shl rax, 4', 'RAX', ['RAX']),
            ('shr rax, 4', 'RAX', ['RAX']),
            ('imul rax, rbx', 'RAX', ['RAX', 'RBX']),
            ('neg rax', 'RAX', ['RAX']),
        ],
    )


ISAS: dict[str, Callable[[], IsaSpec]] = {
    'amd64': _amd64,
    'arm64': _arm64,
    'mips': _mips,
    'ppc': _ppc,
    'sparc': _sparc,
}


def _run(spec: IsaSpec, code: bytes, state: dict[str, int]) -> dict[str, int]:
    """Execute `code` once on a FRESH Unicorn (no cross-run state leakage)."""
    uc = Uc(spec.uc_arch, spec.uc_mode)
    uc.mem_map(CODE_ADDR, 0x1000)
    uc.mem_write(CODE_ADDR, code + b'\x00' * 16)
    for r, v in state.items():
        uc.reg_write(spec.uc_regs[r], v & spec.mask)
    uc.emu_start(CODE_ADDR, CODE_ADDR + len(code))
    return {r: uc.reg_read(spec.uc_regs[r]) & spec.mask for r in spec.regs}


def bitflip_lower_bound(spec: IsaSpec, code: bytes, state: dict[str, int], taint: dict[str, int]) -> dict[str, int]:
    """OR of output diffs over every single tainted-bit flip -> a LOWER bound."""
    base = _run(spec, code, state)
    lb = dict.fromkeys(spec.regs, 0)
    for r in spec.regs:
        for b in range(spec.bits):
            if (taint.get(r, 0) >> b) & 1:
                s2 = dict(state)
                s2[r] = (s2[r] ^ (1 << b)) & spec.mask
                try:
                    out = _run(spec, code, s2)
                except Exception:
                    continue
                for rr in spec.regs:
                    lb[rr] |= base[rr] ^ out[rr]
    return lb


def exact_gt(spec: IsaSpec, code: bytes, state: dict[str, int], taint: dict[str, int]) -> dict[str, int] | None:
    """EXACT ground truth by 2^k enumeration, or None if k exceeds the budget."""
    pos = [(r, b) for r in spec.regs for b in range(spec.bits) if (taint.get(r, 0) >> b) & 1]
    if len(pos) > EXHAUSTIVE_MAX_K:
        return None
    base = None
    gt = dict.fromkeys(spec.regs, 0)
    for asg in itertools.product([0, 1], repeat=len(pos)):
        s = dict(state)
        for (r, b), v in zip(pos, asg, strict=True):
            if v:
                s[r] = (s[r] ^ (1 << b)) & spec.mask
        try:
            out = _run(spec, code, s)
        except Exception:
            continue
        if base is None:
            base = out
            continue
        for rr in spec.regs:
            gt[rr] |= base[rr] ^ out[rr]
    return gt


def microtaint(spec: IsaSpec, code: bytes, state: dict[str, int], taint: dict[str, int]) -> dict[str, int]:
    rule = generate_static_rule(spec.arch, code, spec.state_format())
    ctx = EvalContext(
        input_values=dict(state),
        input_taint=dict(taint),
        simulator=CellSimulator(spec.arch),
    )
    return rule.evaluate(ctx)


def main() -> int:  # noqa: C901
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arch', default='all', choices=[*ISAS.keys(), 'all'])
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    targets = list(ISAS) if args.arch == 'all' else [args.arch]
    rng_states = [
        (0x0F0F0F0F0F0F0F0F, 0x00FF00FF00FF00FF, 0x1234, 0x5),
        (0x8000000000000001, 0x7FFFFFFFFFFFFFFF, 0x3, 0x2),
        (0xDEADBEEFCAFEBABE, 0x0102030405060708, 0x11, 0x4),
    ]
    taint_masks = [0b1, 0b11, 0b1001, 0x10, 1 << 31, 0b101 << 8]

    grand_unsound = grand_cases = grand_exact = grand_exact_ok = 0
    rows: list[tuple[str, int, int, int, int]] = []

    for key in targets:
        spec = ISAS[key]()
        n = unsound = exact_n = exact_ok = 0
        for asm, out_reg, srcs in spec.prog:
            try:
                code = bytes(spec.ks.asm(asm, CODE_ADDR)[0])
            except Exception as e:
                print(f'  [{spec.label}] SKIP {asm!r}: assemble: {e}')
                continue
            for vals in rng_states:
                state = {r: (vals[i % len(vals)] & spec.mask) for i, r in enumerate(spec.regs)}
                for tm in taint_masks:
                    taint = dict.fromkeys(spec.regs, 0)
                    # taint the operands the instruction ACTUALLY reads
                    for r in srcs:
                        taint[r] = tm & spec.mask
                    try:
                        lb = bitflip_lower_bound(spec, code, state, taint)
                        mt = microtaint(spec, code, state, taint)
                    except Exception as e:
                        if args.verbose:
                            print(f'  [{spec.label}] ERR {asm!r}: {type(e).__name__}: {e}')
                        continue
                    n += 1
                    missed = 0
                    for r in spec.regs:
                        missed |= lb[r] & ~(mt.get(r, 0) or 0) & spec.mask
                    if missed:
                        unsound += 1
                        print(
                            f'  [{spec.label}] UNSOUND {asm!r} taint={tm:#x} '
                            f'missed={missed:#x} mt={ {r: hex(mt.get(r, 0)) for r in spec.regs} }',
                        )
                    gt = exact_gt(spec, code, state, taint)
                    if gt is not None:
                        exact_n += 1
                        if all((mt.get(r, 0) or 0) & spec.mask == gt[r] for r in spec.regs):
                            exact_ok += 1
        rows.append((spec.label, n, unsound, exact_n, exact_ok))
        grand_cases += n
        grand_unsound += unsound
        grand_exact += exact_n
        grand_exact_ok += exact_ok

    print()
    print('=' * 72)
    print('MicroTaint vs single-bit-flip oracle -- cross-ISA')
    print('=' * 72)
    print(f'{"ISA":10} {"cases":>7} {"under-taints":>13} {"exact-checked":>14} {"exact":>7}')
    print('-' * 72)
    for label, n, unsound, en, eok in rows:
        pct = f'{100.0 * eok / en:.0f}%' if en else 'n/a'
        print(f'{label:10} {n:7d} {unsound:13d} {en:14d} {pct:>7}')
    print('-' * 72)
    print(f'{"TOTAL":10} {grand_cases:7d} {grand_unsound:13d} {grand_exact:14d}')
    print()
    if grand_unsound == 0:
        print('NO under-tainting on any ISA (single-bit-flip lower bound).')
    else:
        print(f'{grand_unsound} UNDER-TAINT case(s) -- see the lines above.')
    print('=' * 72)
    return 1 if grand_unsound else 0


if __name__ == '__main__':
    sys.exit(main())
