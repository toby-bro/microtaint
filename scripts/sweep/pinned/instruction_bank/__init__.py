"""Unified multi-ISA instruction bank.

A single durable storage file (``instructions.jsonl``) holding the ~1,500
instruction forms accumulated across the ISAs (AMD64 / ARM64 / MIPS64BE /
PPC32BE / RISCV64), each with PRE-ASSEMBLED raw bytes so consumers need no
keystone and do no per-run assembly. Shared by the perf ratchet
(tests/test_perf_ratchet.py) and the benchmark scripts.

Two things live here:

  * ``ISA_FORMATS`` -- the canonical per-ISA register/state format
    (``Architecture`` + a ``[(name, bits), ...]`` register list). Held in code
    (not a *.json the root .gitignore would swallow), typed against
    ``microtaint.types``. This is what ``generate_static_rule`` receives.
  * ``load_bank()`` -- read ``instructions.jsonl`` into typed ``Instruction``
    records, grouped per ISA with the format attached.

Regenerate the data with ``build.py`` (that one DOES need keystone).

Usage::

    from instruction_bank import load_bank
    for spec in load_bank().values():          # spec per ISA
        for ins in spec.instructions:
            circ = generate_static_rule(spec.arch, ins.bytes, spec.regs)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from microtaint.types import Architecture, Register

_HERE = Path(__file__).resolve().parent
DATA = _HERE / 'instructions.jsonl'


# ---------------------------------------------------------------------------
# Canonical per-ISA state formats. Generous GP banks + stack pointer + the
# modelled flags, so every corpus form (incl. push/pop / load-store sequences)
# has every operand it touches. Flag names are the SLEIGH names the engine uses
# (x86 CF/PF/AF/ZF/SF/OF, ARM64 N/Z/C/V); MIPS/RISC-V model no flag registers;
# PPC conditions live in CR0..CR7 (XER carry is unmodelled, so omitted).
# ---------------------------------------------------------------------------

_X86 = ([('RAX', 64), ('RBX', 64), ('RCX', 64), ('RDX', 64), ('RSI', 64),
         ('RDI', 64), ('RBP', 64), ('RSP', 64)]
        + [(f'R{i}', 64) for i in range(8, 16)] + [('RIP', 64), ('EFLAGS', 32)]
        + [(f, 1) for f in ('CF', 'PF', 'AF', 'ZF', 'SF', 'OF')])

_ARM64 = ([(f'X{i}', 64) for i in range(9)] + [('SP', 64)]
          + [(f, 1) for f in ('N', 'Z', 'C', 'V')])

_MIPS = [(n, 64) for n in ('ZERO', 'AT', 'V0', 'V1', 'A0', 'A1', 'A2', 'A3',
                           'T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7',
                           'S0', 'S1', 'GP', 'SP', 'FP', 'RA')]

_PPC = [(f'R{i}', 32) for i in range(11)] + [(f'CR{i}', 4) for i in range(8)]

_RISCV_ABI = ('ZERO', 'RA', 'SP', 'GP', 'TP', 'T0', 'T1', 'T2', 'S0', 'S1',
              'A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'S2', 'S3', 'S4',
              'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'T3', 'T4', 'T5', 'T6')
_RISCV = [(n, 64) for n in _RISCV_ABI] + [('PC', 64)]

# Architecture name -> [(reg_name, bits), ...]
ISA_FORMATS: dict[str, list[tuple[str, int]]] = {
    'AMD64': _X86, 'ARM64': _ARM64, 'MIPS64BE': _MIPS,
    'PPC32BE': _PPC, 'RISCV64': _RISCV,
}

# ---------------------------------------------------------------------------
# Vector/SIMD state profiles. Kept SEPARATE from the scalar ISA_FORMATS above so
# the scalar corpus (and its committed perf baselines) stay byte-identical:
# adding vector lanes to a scalar format perturbs the frame-recycle corner
# sharing and would SHIFT the scalar metrics (verified: shl rax,cl nodes
# 1047->1053). A SIMD form therefore carries its own group key ('AMD64_SIMD' /
# 'ARM64_SIMD', a distinct ratchet/baseline group) while its real
# ``Architecture`` is stored in the jsonl 'arch' field. Each profile is
# (Architecture, gp_names, vec_names); the vector registers are expanded to the
# engine's geometry lanes lazily via the geometry-derived RegisterAliases helper
# -- no hand-coded VL_ lane list, and no reg_aliases import cost unless a SIMD
# group is actually loaded.
_SIMD_PROFILES: dict[str, tuple[Architecture, list[str], list[str]]] = {
    'AMD64_SIMD': (Architecture.AMD64,
                   ['RAX', 'RBX', 'RCX', 'RDX', 'RSP', 'RIP', 'EFLAGS',
                    'CF', 'PF', 'AF', 'ZF', 'SF', 'OF'],
                   [f'XMM{i}' for i in range(8)]),
    'ARM64_SIMD': (Architecture.ARM64,
                   ['X0', 'X1', 'X2', 'X3', 'SP', 'N', 'Z', 'C', 'V'],
                   [f'V{i}' for i in range(8)]),
}


# PERF SWEEP: ISA groups this engine revision cannot describe -> isa -> reason.
_UNSUPPORTED_ISAS: dict[str, str] = {}


@lru_cache(maxsize=None)
def _simd_registers(key: str) -> tuple[Register, ...]:
    """Expand a SIMD profile to geometry lanes. Lazy: the RegisterAliases import
    + SLEIGH-geometry build (~150ms/arch) is paid only when a SIMD group loads."""
    from microtaint.debug.reg_aliases import RegisterAliases
    arch, gp, vec = _SIMD_PROFILES[key]
    return tuple(RegisterAliases(arch).state_format([*gp, *vec]))


@lru_cache(maxsize=None)
def keystone_for(isa_key: str):
    """corpora.py key -> a cached keystone assembler (None for riscv, which is
    hand-assembled hex). Imported lazily -- keystone is a build-time dep only,
    never needed to LOAD the bank.
    """
    from keystone import (
        KS_ARCH_ARM64, KS_ARCH_MIPS, KS_ARCH_PPC, KS_ARCH_X86,
        KS_MODE_64, KS_MODE_BIG_ENDIAN, KS_MODE_LITTLE_ENDIAN, KS_MODE_MIPS64,
        KS_MODE_PPC32, Ks,
    )
    return {
        'x86_64': lambda: Ks(KS_ARCH_X86, KS_MODE_64),
        'arm64': lambda: Ks(KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN),
        'mips': lambda: Ks(KS_ARCH_MIPS, KS_MODE_MIPS64 | KS_MODE_BIG_ENDIAN),
        'ppc': lambda: Ks(KS_ARCH_PPC, KS_MODE_PPC32 | KS_MODE_BIG_ENDIAN),
        'riscv': lambda: None,
    }[isa_key]()


@lru_cache(maxsize=None)
def isa_registers(arch_name: str) -> tuple[Register, ...]:
    if arch_name in _SIMD_PROFILES:
        return _simd_registers(arch_name)
    return tuple(Register(name=n, bits=b) for n, b in ISA_FORMATS[arch_name])


@dataclass(frozen=True)
class Instruction:
    isa: str
    label: str
    bytes: bytes
    asm: str | None = None
    srcs: tuple[str, ...] = ()
    constraints: dict = field(default_factory=dict)
    categories: tuple[str, ...] = ()
    source: str = ''
    oracle: dict | None = None


@dataclass
class ISASpec:
    name: str
    arch: Architecture
    regs: list[Register]
    instructions: list[Instruction]


def load_bank(
    isas: set[str] | None = None,
    categories: set[str] | None = None,
) -> dict[str, ISASpec]:
    """Load the bank, grouped per ISA. Optionally filter by ISA and/or category."""
    if not DATA.exists():
        raise FileNotFoundError(
            f'{DATA} missing -- generate it with '
            f'`python {(_HERE / "build.py")}`')
    specs: dict[str, ISASpec] = {}
    with DATA.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            isa = d['isa']
            if isas is not None and isa not in isas:
                continue
            if categories is not None and not (set(d.get('categories', [])) & categories):
                continue
            if isa in _UNSUPPORTED_ISAS:
                continue
            spec = specs.get(isa)
            if spec is None:
                # 'isa' is the group key (== a scalar Architecture, or a SIMD
                # profile like 'AMD64_SIMD'); the real Architecture is 'arch'
                # (defaults to the group key for the scalar groups).
                try:
                    regs = list(isa_registers(isa))
                except Exception as exc:
                    # PERF SWEEP: an engine revision that predates this ISA's
                    # register geometry (e.g. microtaint.debug.reg_aliases, which
                    # the SIMD profiles need) simply has no such ISA to measure.
                    # Drop the group instead of failing the whole bank load.
                    _UNSUPPORTED_ISAS[isa] = repr(exc)[:160]
                    continue
                spec = specs[isa] = ISASpec(
                    name=isa, arch=Architecture(d.get('arch', isa)),
                    regs=regs, instructions=[])
            spec.instructions.append(Instruction(
                isa=isa, label=d['label'], bytes=bytes.fromhex(d['bytes']),
                asm=d.get('asm'), srcs=tuple(d.get('srcs', ())),
                constraints=d.get('constraints', {}),
                categories=tuple(d.get('categories', ())),
                source=d.get('source', ''), oracle=d.get('oracle')))
    return specs


def all_instructions(**kw) -> list[Instruction]:
    out: list[Instruction] = []
    for spec in load_bank(**kw).values():
        out.extend(spec.instructions)
    return out
