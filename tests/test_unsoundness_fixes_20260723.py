"""Multi-ISA soundness regression tests for the July 22-23 2026 fuzzing campaign.

Every test brute-forces exact bit-level ground truth via 2^k Unicorn enumeration
over the tainted INPUT bits (GPRs and flags) and asserts
``microtaint_output ⊇ ground_truth`` -- no under-tainted bit -- on the SAME
(state, taint) that the campaign or targeted fuzzing flagged.

The cases span the generalising fixes made over the two days:

  * 128-bit varnodes no longer truncate (imul/mul overflow flags);
  * the sign-extension floor fires on any slice intersecting the fill region
    (MIPS sw;lw;addu, order-independent);
  * equality-to-zero (ZF) of a non-monotone sum gets a carry-smear floor;
  * a data-dependent shift is not a mapped permutation (MIPS srlv $2,$4,$4);
  * shifted-/extended-register operands keep their carry floor (ARM64);
  * variable-amount shift, multiply, carry-in overflow exact/floor terms;
  * the NOT idiom inverts polarity (bic) and a wide XOR unions transformed taint
    (eor); the conditional-move implicit-else weld (x86 cmovl); the bit-set
    reachable-index term (x86 bts).
"""

# mypy: disable-error-code="no-untyped-def,no-untyped-call,attr-defined,index"

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import pytest
import unicorn as U
import unicorn.arm64_const as ua
import unicorn.mips_const as um
import unicorn.x86_const as ux
from unicorn.unicorn_py3 import Uc

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, Register

MASK64 = 0xFFFFFFFFFFFFFFFF
_SP_VALUE = 0x204000


@dataclass
class Flag:
    name: str
    bit: int
    reg: int


@dataclass
class ArchDesc:
    key: str                       # microtaint Architecture enum name
    uc_arch: int
    uc_mode: int
    gprs: list[tuple[str, int]]    # (mt name, unicorn reg id)
    flags: list[Flag]
    status_reg: int | None         # unicorn status register holding the flags
    status_base: int = 0
    canon: int | None = None       # sign-extend GPR seeds to `canon` bits (MIPS32)
    sp: tuple[str, int] | None = None
    bits: int = 64
    _fmt: list[Register] = field(default_factory=list)

    @property
    def fmt(self) -> list[Register]:
        if not self._fmt:
            self._fmt = [Register(n, self.bits) for n, _ in self.gprs]
            self._fmt += [Register(f.name, 1) for f in self.flags]
            if self.sp is not None:
                self._fmt.append(Register(self.sp[0], self.bits))
        return self._fmt


_AMD64 = ArchDesc(
    'AMD64', U.UC_ARCH_X86, U.UC_MODE_64,
    [('RAX', ux.UC_X86_REG_RAX), ('RBX', ux.UC_X86_REG_RBX),
     ('RCX', ux.UC_X86_REG_RCX), ('RDX', ux.UC_X86_REG_RDX)],
    [Flag('CF', 0, ux.UC_X86_REG_EFLAGS), Flag('PF', 2, ux.UC_X86_REG_EFLAGS),
     Flag('ZF', 6, ux.UC_X86_REG_EFLAGS), Flag('SF', 7, ux.UC_X86_REG_EFLAGS),
     Flag('OF', 11, ux.UC_X86_REG_EFLAGS)],
    ux.UC_X86_REG_EFLAGS, status_base=0x2, sp=('RSP', ux.UC_X86_REG_RSP),
)
_ARM64 = ArchDesc(
    'ARM64', U.UC_ARCH_ARM64, U.UC_MODE_ARM,
    [('x0', ua.UC_ARM64_REG_X0), ('x1', ua.UC_ARM64_REG_X1),
     ('x2', ua.UC_ARM64_REG_X2), ('x3', ua.UC_ARM64_REG_X3)],
    [Flag('NG', 31, ua.UC_ARM64_REG_NZCV), Flag('ZR', 30, ua.UC_ARM64_REG_NZCV),
     Flag('CY', 29, ua.UC_ARM64_REG_NZCV), Flag('OV', 28, ua.UC_ARM64_REG_NZCV)],
    ua.UC_ARM64_REG_NZCV, sp=('sp', ua.UC_ARM64_REG_SP),
)
_MIPS = ArchDesc(
    'MIPS64BE', U.UC_ARCH_MIPS, U.UC_MODE_MIPS64 | U.UC_MODE_BIG_ENDIAN,
    [('A0', um.UC_MIPS_REG_A0), ('A1', um.UC_MIPS_REG_A1),
     ('A2', um.UC_MIPS_REG_A2), ('V0', um.UC_MIPS_REG_V0)],
    [], None, canon=32, sp=('sp', um.UC_MIPS_REG_SP),
)
_ARCHES = {'AMD64': _AMD64, 'ARM64': _ARM64, 'MIPS64BE': _MIPS}

_STACK_PAGE = 0x200000  # covers sp-16 downward, like GTSim


def _run(arch: ArchDesc, code: bytes, gpr: dict[str, int], flag: dict[str, int]) -> dict[str, int]:
    uc = Uc(arch.uc_arch, arch.uc_mode)
    uc.mem_map(0x1000, 0x10000)
    uc.mem_map(_STACK_PAGE, 0x10000)
    uc.mem_write(0x1000, code + b'\x90' * 16)
    if arch.sp is not None:
        uc.reg_write(arch.sp[1], _SP_VALUE)
    for name, rid in arch.gprs:
        uc.reg_write(rid, gpr[name] & MASK64)
    if arch.flags and arch.status_reg is not None:
        v = arch.status_base
        for f in arch.flags:
            v |= (flag.get(f.name, 0) & 1) << f.bit
        uc.reg_write(arch.status_reg, v)
    uc.emu_start(0x1000, 0x1000 + len(code))
    out = {name: uc.reg_read(rid) & MASK64 for name, rid in arch.gprs}
    if arch.flags and arch.status_reg is not None:
        sv = uc.reg_read(arch.status_reg)
        for f in arch.flags:
            out[f.name] = (sv >> f.bit) & 1
    return out


def _brute_gt(arch: ArchDesc, code: bytes, state: dict[str, int], taint: dict[str, int]) -> dict[str, int]:  # noqa: C901
    flagset = {f.name for f in arch.flags}
    positions: list[tuple[str, int]] = []
    for reg, mask in taint.items():
        w = 1 if reg in flagset else arch.bits
        for b in range(w):
            if (mask >> b) & 1:
                positions.append((reg, b))
    if len(positions) > 12:
        pytest.skip(f'k={len(positions)} exceeds enumeration budget')

    def split(s: dict[str, int]) -> tuple[dict[str, int], dict[str, int]]:
        g = {n: s.get(n, 0) & MASK64 for n, _ in arch.gprs}
        fl = {f.name: s.get(f.name, 0) & 1 for f in arch.flags}
        return g, fl

    results = []
    for assign in itertools.product([0, 1], repeat=len(positions)):
        s = dict(state)
        for v, (reg, b) in zip(assign, positions):  # noqa: B905
            if v:
                s[reg] = s.get(reg, 0) ^ (1 << b)
        g, fl = split(s)
        results.append(_run(arch, code, g, fl))
    gt = {n: 0 for n, _ in arch.gprs}
    gt.update({f.name: 0 for f in arch.flags})
    base = results[0]
    for r in results[1:]:
        for k in gt:
            gt[k] |= base[k] ^ r[k]
    return gt


def _canon(arch: ArchDesc, state: dict[str, int], taint: dict[str, int]):
    if arch.canon is None:
        return state, taint
    w = arch.canon
    low = (1 << w) - 1
    sign = 1 << (w - 1)
    high = MASK64 ^ low
    gpr = {n for n, _ in arch.gprs}

    def sx(v):
        v &= low
        return v | high if v & sign else v
    st = {k: (sx(v) if k in gpr else v) for k, v in state.items()}
    tt = {k: ((t & low) if k in gpr else t) for k, t in taint.items()}
    return st, tt


_SIMS: dict[str, CellSimulator] = {}


def _assert_sound(arch_name: str, code_hex: str, state: dict[str, int],
                  taint: dict[str, int], label: str, skip: frozenset[str] = frozenset()) -> None:
    arch = _ARCHES[arch_name]
    code = bytes.fromhex(code_hex)
    state, taint = _canon(arch, state, taint)
    full_state = dict(state)
    if arch.sp is not None:
        full_state.setdefault(arch.sp[0], _SP_VALUE)
    gt = _brute_gt(arch, code, full_state, taint)
    if arch_name not in _SIMS:
        _SIMS[arch_name] = CellSimulator(getattr(Architecture, arch.key))
    rule = generate_static_rule(getattr(Architecture, arch.key), code, arch.fmt)
    ctx = EvalContext(input_values=full_state, input_taint=taint, simulator=_SIMS[arch_name])
    mt = rule.evaluate(ctx)
    for reg, gt_bits in gt.items():
        if reg in skip:  # architecturally-undefined flag: Unicorn invents a value
            continue
        missed = gt_bits & ~mt.get(reg, 0)
        assert missed == 0, (
            f'[{label}] under-taint in {reg}: gt={gt_bits:#x} mt={mt.get(reg, 0):#x} missed={missed:#x}'
        )


# --------------------------------------------------------------------------- #
# (arch, code_hex, state, taint, label) -- each a config that under-tainted
# before the corresponding fix.
# --------------------------------------------------------------------------- #
_CASES = [
    # --- fixed during the July 23 night run ---
    ('MIPS64BE', 'afa4fff08fa2fff000451021',
     {'A0': 0x11223344, 'A1': 0x55667788}, {'A0': 0x80000000, 'A1': 0x80000000},
     'sw;lw;addu sign-extension fill (both sign bits tainted)'),
    ('MIPS64BE', '00851021',
     {'A0': 0x11223344, 'A1': 0x55667788}, {'A0': 0x80000000, 'A1': 0x80000000},
     'addu sign-extension fill (plain)'),
    ('AMD64', '00d8', {'RAX': 0x40, 'RBX': 0x40}, {'RAX': 0x80, 'RBX': 0x80},
     'add al,bl ZF equality-to-zero non-monotone floor'),
    ('AMD64', '4869c355050000', {'RBX': 0x200658E019F2D271}, {'RBX': 1 << 61},
     'imul rax,rbx,imm CF/OF 128-bit product (no truncation)'),
    ('AMD64', '48f7e3', {'RAX': 0x8000000000000123, 'RBX': 0x3}, {'RAX': 1 << 63},
     'mul rbx high half RDX 128-bit'),
    # --- generalising fixes from the preceding days ---
    ('MIPS64BE', '00841006', {'A0': 0x30}, {'A0': 0x1F},
     'srlv $2,$4,$4 data-dependent shift is not a permutation'),
    ('ARM64', '200c028b', {'x1': 0x1, 'x2': 0x1}, {'x1': 0x3, 'x2': 0x3},
     'add x0,x1,x2,lsl #3 shifted-register carry floor'),
    ('ARM64', '2000228b', {'x1': 0x1, 'x2': 0x1}, {'x1': 0x3, 'x2': 0x3},
     'add x0,x1,w2,uxtb extended-register (overlap-aware waist)'),
    ('ARM64', '2020c29a', {'x1': 0x9, 'x2': 0x3}, {'x1': 0x9, 'x2': 0x3},
     'lslv variable-amount shift subcube'),
    ('ARM64', '207cc29b', {'x1': 1 << 40, 'x2': 1 << 40}, {'x1': 1 << 40, 'x2': 1 << 40},
     'umulh multiply fill'),
    ('ARM64', '200002ba', {'x1': 0x1, 'x2': 0x1, 'CY': 1}, {'x1': 0x1, 'x2': 0x1, 'CY': 1},
     'adcs carry-in signed overflow'),
    ('ARM64', '3f0c02eb', {'x1': 0x6A8F1DD4E13A0996, 'x2': 0x5484B3DBBA6BC77C},
     {'x1': 0xA000008020000000, 'x2': 0x400}, 'cmp x1,x2,lsl #3 shifted-operand overflow term'),
    ('ARM64', '2004228a', {'x1': 0x800008000400000, 'x2': 0x400000000240000},
     {'x1': 0x8000400000, 'x2': 0x400000000240000}, 'bic x0,x1,x2,lsl #1 NOT-idiom polarity'),
    ('ARM64', '202cc2ca', {'x1': 0x123456789ABCDEF, 'x2': 0xFEDCBA9876543210},
     {'x1': 0x1000002, 'x2': 0x4000180021000}, 'eor x0,x1,x2,ror #11 wide-XOR transformed union'),
    ('ARM64', '20e042fa', {'x1': 0x6A8F1DD4E13A0996, 'x2': 0x5484B3DBBA6BC77C},
     {'x1': 0xA000008020000000, 'x2': 0x400}, 'ccmp OV conditional-compare overflow floor'),
    ('AMD64', '480fabd8', {'RAX': 0, 'RBX': 0x40}, {'RAX': 0, 'RBX': 0x3F},
     'bts rax,rbx reachable bit-index set'),
    ('AMD64', '480f4cc3', {'RAX': 0x1111, 'RBX': 0x2222, 'OF': 1, 'SF': 0},
     {'RAX': 0x5, 'RBX': 0x50, 'OF': 1}, 'cmovl implicit-else weld'),
    ('AMD64', '488d448b08', {'RBX': 0x1, 'RCX': 0x1}, {'RBX': 0x3, 'RCX': 0x3},
     'lea scaled index affine, not avalanche'),
]


@pytest.mark.parametrize(('arch', 'code', 'state', 'taint', 'label'), _CASES,
                         ids=[c[4].split()[0] + '-' + c[0] for c in _CASES])
def test_no_under_taint(arch, code, state, taint, label):
    # SF/ZF/PF are architecturally UNDEFINED after IMUL/MUL (Intel SDM); Unicorn
    # invents a value, so they are excluded -- the same declared limitation the
    # benchmark's written_flags applies.
    skip = frozenset({'SF', 'ZF', 'PF'}) if code in ('4869c355050000', '48f7e3') else frozenset()
    _assert_sound(arch, code, state, taint, label, skip)
