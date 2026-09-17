"""
Throwaway prototype: port MicroTaint to MIPS64 (BE) and PowerPC64 (BE) via
monkeypatch ONLY (no tracked files touched), lift add/sub/and/or/xor with
generate_static_rule, and validate against a single-bit-flip Unicorn oracle.

Run with:
    uv run python proto_port.py
"""

# Experiment script, not library code: see artifacts/ndss27/README.md,
# "Lint and type checking", for why annotations are not required here.
# mypy: disable-error-code="no-untyped-def, no-untyped-call, type-arg"

from __future__ import annotations

import pathlib
import sys

# The engine repo is three levels up from artifacts/ndss27/rq6-generalisation/.
REPO = str(pathlib.Path(__file__).resolve().parents[3])
sys.path.insert(0, REPO)

import pypcode
import unicorn
import unicorn.mips_const as um
import unicorn.ppc_const as up
from keystone import (
    KS_ARCH_MIPS,
    KS_ARCH_PPC,
    KS_MODE_BIG_ENDIAN,
    KS_MODE_MIPS64,
    KS_MODE_PPC32,
    Ks,
)
from unicorn import (
    UC_ARCH_MIPS,
    UC_ARCH_PPC,
    UC_MODE_BIG_ENDIAN,
    UC_MODE_MIPS64,
    UC_MODE_PPC32,
)

import microtaint.simulator as sim_mod
import microtaint.sleigh.lifter as lifter
from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import ImplicitTaintPolicy, Register

MASK64 = (1 << 64) - 1

# ---------------------------------------------------------------------------
# ISA descriptor tables — exactly the data a real port would add.
# ---------------------------------------------------------------------------

# MIPS GPR ordering (ABI names -> register number), Ghidra SLEIGH names.
MIPS_GPR = [
    'zero',
    'at',
    'v0',
    'v1',
    'a0',
    'a1',
    'a2',
    'a3',
    't0',
    't1',
    't2',
    't3',
    't4',
    't5',
    't6',
    't7',
    's0',
    's1',
    's2',
    's3',
    's4',
    's5',
    's6',
    's7',
    't8',
    't9',
    'k0',
    'k1',
    'gp',
    'sp',
    's8',
    'ra',
]


def build_mips():
    lid = 'MIPS:BE:64:default'
    ctx = pypcode.Context(lid)
    # SLEIGH register names actually present
    names = [n for n in MIPS_GPR if n in ctx.registers] + ['pc']
    regs = [Register(name=n.upper(), bits=64) for n in names]
    # Unicorn reg-id map (state name UPPER -> uc id). Numeric UC_MIPS_REG_<i>.
    uc_regs = {}
    for i, n in enumerate(MIPS_GPR):
        uc_regs[n.upper()] = getattr(um, f'UC_MIPS_REG_{i}')
    uc_regs['PC'] = um.UC_MIPS_REG_PC
    return {
        'arch': 'MIPS64BE',
        'lid': lid,
        'ctx': ctx,
        'regs': regs,
        'uc_regs': uc_regs,
        'uc_arch': UC_ARCH_MIPS,
        'uc_mode': UC_MODE_MIPS64 | UC_MODE_BIG_ENDIAN,
        'uc_pc': um.UC_MIPS_REG_PC,
        'ks': Ks(KS_ARCH_MIPS, KS_MODE_MIPS64 | KS_MODE_BIG_ENDIAN),
        # (asm, dst, s1, s2) using SLEIGH state names
        'prog': [
            ('addu $2, $4, $5', 'V0', 'A0', 'A1'),
            ('subu $2, $4, $5', 'V0', 'A0', 'A1'),
            ('and $2, $4, $5', 'V0', 'A0', 'A1'),
            ('or $2, $4, $5', 'V0', 'A0', 'A1'),
            ('xor $2, $4, $5', 'V0', 'A0', 'A1'),
        ],
    }


def build_ppc():
    # PowerPC 32-bit: Unicorn 2.1.4 executes PPC32 but NOT PPC64 (UC_ERR_EXCEPTION),
    # so the ground-truth oracle must run on PPC32.
    lid = 'PowerPC:BE:32:default'
    ctx = pypcode.Context(lid)
    names = [f'r{i}' for i in range(32)] + ['pc']
    regs = [Register(name=n.upper(), bits=32) for n in names]
    uc_regs = {f'R{i}': getattr(up, f'UC_PPC_REG_{i}') for i in range(32)}
    uc_regs['PC'] = up.UC_PPC_REG_PC
    return {
        'arch': 'PPC32BE',
        'lid': lid,
        'ctx': ctx,
        'regs': regs,
        'uc_regs': uc_regs,
        'bits': 32,
        'uc_arch': UC_ARCH_PPC,
        'uc_mode': UC_MODE_PPC32 | UC_MODE_BIG_ENDIAN,
        'uc_pc': up.UC_PPC_REG_PC,
        'ks': Ks(KS_ARCH_PPC, KS_MODE_PPC32 | KS_MODE_BIG_ENDIAN),
        'prog': [
            ('add 3, 4, 5', 'R3', 'R4', 'R5'),  # rD = rA + rB
            ('subf 3, 4, 5', 'R3', 'R4', 'R5'),  # rD = rB - rA
            ('and 3, 4, 5', 'R3', 'R4', 'R5'),  # rA = rS & rB  (dst is first)
            ('or 3, 4, 5', 'R3', 'R4', 'R5'),
            ('xor 3, 4, 5', 'R3', 'R4', 'R5'),
        ],
    }


# ---------------------------------------------------------------------------
# THE PORT: three data-table injections (mirrors the RV64 3-file change).
# ---------------------------------------------------------------------------
def install(cfg):
    # (1) lifter: register arch -> SLEIGH language id. We pre-seed the module
    #     context cache, which is exactly equivalent to adding one arch_map row.
    lifter._pypcode_contexts[cfg['arch']] = cfg['ctx']
    # (2) simulator: arch -> (uc_arch, uc_mode)
    sim_mod._ARCH_MAP[cfg['arch']] = (cfg['uc_arch'], cfg['uc_mode'])
    # (3) simulator: arch -> {state-name: uc-reg-id}
    sim_mod._UC_REGS[cfg['arch']] = cfg['uc_regs']


def make_sim(cfg):
    sim = CellSimulator(cfg['arch'], use_unicorn=True)
    # Fix PC id (the RV64 port adds one elif for this; here we set it post-init).
    sim._pc_reg = cfg['uc_pc']
    return sim


# ---------------------------------------------------------------------------
# Raw-Unicorn single-bit-flip ground-truth oracle.
# ---------------------------------------------------------------------------
_CODE = 0x1000


def _run_oracle(cfg, code, regvals):
    uc = unicorn.Uc(cfg['uc_arch'], cfg['uc_mode'])
    uc.mem_map(_CODE, 0x1000)
    uc.mem_write(_CODE, code)
    for name, rid in cfg['uc_regs'].items():
        uc.reg_write(rid, regvals.get(name, 0) & MASK64)
    try:
        uc.emu_start(_CODE, _CODE + len(code))
    except unicorn.UcError as exc:
        # Swallowing this returned the register file UNCHANGED, so an
        # instruction that never executed produced an oracle of 0, a `missed` of
        # 0, and printed OK while being counted in lifted+sound.  Reproduced: a
        # real `addu` gives oracle 0xffffffffffffffff, while four bytes of 0xff
        # (which faults) gives 0x0 and scores sound.  A run that did not happen
        # is not evidence about the engine.
        raise RuntimeError(
            f'the oracle could not execute {code.hex()}: {exc}',
        ) from exc
    return {n: uc.reg_read(r) & MASK64 for n, r in cfg['uc_regs'].items()}


def true_taint(cfg, code, values, taint, out_reg):
    base = {n: values.get(n, 0) & ~taint.get(n, 0) & MASK64 for n in cfg['uc_regs']}
    base_out = _run_oracle(cfg, code, base)[out_reg]
    acc = 0
    for reg, t in taint.items():
        for bit in range(64):
            if not (t >> bit) & 1:
                continue
            mut = dict(base)
            mut[reg] = base[reg] ^ (1 << bit)
            acc |= _run_oracle(cfg, code, mut)[out_reg] ^ base_out
    return acc & MASK64


# ---------------------------------------------------------------------------
# End-to-end driver.
# ---------------------------------------------------------------------------
def run_isa(cfg):
    print(f'\n{"=" * 70}\n{cfg["arch"]}  (SLEIGH {cfg["lid"]})\n{"=" * 70}')
    install(cfg)
    try:
        sim = make_sim(cfg)
    except Exception as e:
        print(f'  [FATAL] CellSimulator init failed: {e!r}')
        return 0, 1, 0
    ok = unsound = err = 0
    for asm, dst, s1, s2 in cfg['prog']:
        try:
            code = bytes(cfg['ks'].asm(asm)[0])
        except Exception as e:
            print(f'  {asm:22s} keystone-fail {e!r}')
            err += 1
            continue
        try:
            circuit = generate_static_rule(cfg['arch'], code, cfg['regs'])
        except Exception as e:
            print(f'  {asm:22s} lift-fail {type(e).__name__}: {e}')
            err += 1
            continue

        # Concrete values + taint: taint all of s1, low byte of s2.
        bits = cfg.get('bits', 64)
        wmask = (1 << bits) - 1
        values = {s1: 0xCAFEBABE12345678 & wmask, s2: 0x00000000DEADBEEF & wmask}
        taint = {s1: wmask, s2: 0xFF}
        ctx = EvalContext(
            input_values=values, input_taint=taint, simulator=sim, implicit_policy=ImplicitTaintPolicy.KEEP,
        )
        try:
            mt = circuit.evaluate(ctx)
        except Exception as e:
            print(f'  {asm:22s} eval-fail {type(e).__name__}: {e}')
            err += 1
            continue
        mt_dst = mt.get(dst, 0) & MASK64
        try:
            oracle = true_taint(cfg, code, values, taint, dst)
        except RuntimeError as e:
            # Counted, never scored: a form the oracle cannot execute is
            # untested, and folding it in as a comparison is what let a faulting
            # instruction print OK.
            print(f'  {asm:22s} ORACLE-FAIL {e}')
            err += 1
            continue
        missed = oracle & ~mt_dst  # unsound under-taint
        status = 'OK  ' if missed == 0 else 'UNSOUND'
        if missed == 0:
            ok += 1
        else:
            unsound += 1
        print(
            f'  {asm:22s} [{len(code)}B] mt={mt_dst:#018x} oracle={oracle:#018x} over={(mt_dst & ~oracle):#x} {status}',
        )
    print(f'  --> lifted+sound={ok}  unsound={unsound}  errors={err}')
    return unsound, err, ok


if __name__ == '__main__':
    # An exit code, because a script whose whole purpose is to report whether a
    # port is sound must be able to say "no".  It had none: __main__ looped and
    # fell off the end, so a run full of UNSOUND lines exited 0.
    _unsound = _err = _ok = 0
    for builder in (build_mips, build_ppc):
        try:
            u, e, o = run_isa(builder())
            _unsound += u
            _err += e
            _ok += o
        except Exception:
            import traceback

            traceback.print_exc()
            _err += 1
    if _unsound:
        print(f'\nFAILED: {_unsound} unsound form(s)')
        sys.exit(1)
    if not _ok:
        print('\nFAILED: nothing was scored, so this run says nothing')
        sys.exit(1)
    if _err:
        print(f'\n{_err} form(s) could not be scored and are UNTESTED, not sound')
    sys.exit(0)
