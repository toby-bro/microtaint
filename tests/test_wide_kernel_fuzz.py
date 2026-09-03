"""Differential fuzzer for the width-native SIMD cell kernels.

For many random (instruction, value, taint) triples it drives BOTH kernels
(cell.pyx PCodeCellEvaluator and the production cell_c PCodeCellEvaluatorC) and,
where an exact oracle exists, the pure-Python ground truth.  It asserts:

  * PARITY   -- cell.pyx and cell_c produce byte-identical taint (they are the
    two references the rule engine will target once VL_ lane-splitting is
    deleted, so any divergence is a bug in one of them).
  * EXACTNESS -- for byte-parallel / value-aware ops (COPY, INT_XOR/AND/OR) both
    kernels equal the differential oracle f(V|T) XOR f(V&~T).
  * SOUNDNESS -- for every op, no missed taint vs the oracle where one is known.

Inputs use dense, correlated masks (random full/zero/striped/single-byte lanes
with random concrete values) -- the patterns that surface high-lane and
value-aware bugs a sparse oracle misses.
"""

from __future__ import annotations

import random
import types

import pytest
from keystone import (
    KS_ARCH_ARM64,
    KS_ARCH_PPC,
    KS_ARCH_X86,
    KS_MODE_64,
    KS_MODE_BIG_ENDIAN,
    KS_MODE_LITTLE_ENDIAN,
    KS_MODE_PPC32,
    Ks,
)

from microtaint.instrumentation.cell import PCodeCellEvaluator
from microtaint.instrumentation.cell_c.cell_c import PCodeCellEvaluatorC
from microtaint.types import Architecture

_FULL = 0xFFFFFFFFFFFFFFFF
_KS_X86 = Ks(KS_ARCH_X86, KS_MODE_64)
_KS_ARM = Ks(KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN)
_KS_PPC = Ks(KS_ARCH_PPC, KS_MODE_BIG_ENDIAN + KS_MODE_PPC32)

# Vector register base offsets (from pypcode geometry; see test_wide_kernel_diff).
_XMM0, _XMM1, _XMM2 = 0x1200, 0x1240, 0x1280
_YMM0, _YMM1, _YMM2 = 0x1200, 0x1240, 0x1280
_Q0, _Q1, _Q2 = 0x5000, 0x5020, 0x5040
_VR0, _VR1, _VR2 = 0x4200, 0x4210, 0x4220


def _asm(ks: Ks, text: str) -> str:
    enc, _ = ks.asm(text, 0x1000)
    return bytes(enc).hex()


# (ks, arch, asm, out_base, nbytes, input_bases, oracle_op)
# oracle_op in {'copy','xor','and','or', None}; None => parity-only (shift/load).
_CORPUS = [
    (_KS_X86, Architecture.AMD64, 'movdqa xmm0, xmm1', _XMM0, 16, [_XMM1], 'copy'),
    (_KS_X86, Architecture.AMD64, 'pxor xmm0, xmm1', _XMM0, 16, [_XMM0, _XMM1], 'xor'),
    (_KS_X86, Architecture.AMD64, 'pand xmm0, xmm1', _XMM0, 16, [_XMM0, _XMM1], 'and'),
    (_KS_X86, Architecture.AMD64, 'por xmm0, xmm1', _XMM0, 16, [_XMM0, _XMM1], 'or'),
    (_KS_X86, Architecture.AMD64, 'psrldq xmm0, 5', _XMM0, 16, [_XMM0], None),
    (_KS_X86, Architecture.AMD64, 'pslldq xmm0, 3', _XMM0, 16, [_XMM0], None),
    (_KS_X86, Architecture.AMD64, 'vmovdqa ymm0, ymm1', _YMM0, 32, [_YMM1], 'copy'),
    (_KS_X86, Architecture.AMD64, 'vpxor ymm0, ymm1, ymm2', _YMM0, 32, [_YMM1, _YMM2], 'xor'),
    (_KS_ARM, Architecture.ARM64, 'orr v0.16b, v1.16b, v1.16b', _Q0, 16, [_Q1], 'copy'),
    (_KS_ARM, Architecture.ARM64, 'eor v0.16b, v1.16b, v2.16b', _Q0, 16, [_Q1, _Q2], 'xor'),
    (_KS_PPC, Architecture.PPC32BE, 'vxor 0, 1, 2', _VR0, 16, [_VR1, _VR2], 'xor'),
    (_KS_PPC, Architecture.PPC32BE, 'vor 0, 1, 2', _VR0, 16, [_VR1, _VR2], 'or'),
]


def _rand_lane(rng: random.Random) -> int:
    """A dense, correlated 64-bit mask: full / zero / striped / single byte."""
    kind = rng.randrange(6)
    if kind == 0:
        return 0
    if kind == 1:
        return _FULL
    if kind == 2:
        return 0xFF << (rng.randrange(8) * 8)          # single byte
    if kind == 3:
        return 0xFF00FF00FF00FF00                      # stripe
    if kind == 4:
        return 0x00000000FFFFFFFF if rng.random() < 0.5 else 0xFFFFFFFF00000000
    return rng.getrandbits(64)                          # arbitrary


def _lane_inputs(rng: random.Random, bases: list[int], nbytes: int,
                 ) -> tuple[dict[str, int], dict[str, int]]:
    """Random per-lane (value, taint) for each input register."""
    values: dict[str, int] = {}
    taints: dict[str, int] = {}
    for base in bases:
        for k in range(0, nbytes, 8):
            key = f'VL_{base + k:#x}'
            values[key] = _rand_lane(rng)
            taints[key] = _rand_lane(rng)
    return values, taints


def _cell_bytes(evaluator, arch, instr, out_base, nbytes, or_in, and_in) -> set[int]:
    ev = evaluator(arch)
    tainted: set[int] = set()
    for k in range(0, nbytes, 8):
        cell = types.SimpleNamespace(instruction=instr, out_reg=f'VL_{out_base + k:#x}',
                                     out_bit_start=0, out_bit_end=63)
        lane = ev.evaluate_differential(cell, dict(or_in), dict(and_in))
        for i in range(8):
            if (lane >> (i * 8)) & 0xFF:
                tainted.add(k + i)
    return tainted


def _oracle_bytes(op, bases, values, taints, nbytes) -> set[int]:
    """Exact taint via f(V|T) XOR f(V&~T) on the wide per-lane operands."""
    def wide(base: int, src: dict[str, int]) -> int:
        v = 0
        for k in range(0, nbytes, 8):
            v |= (src.get(f'VL_{base + k:#x}', 0) & _FULL) << (k * 8)
        return v
    mask = (1 << (nbytes * 8)) - 1
    operands = []
    for base in bases:
        v = wide(base, values)
        t = wide(base, taints)
        operands.append(((v | t) & mask, (v & ~t) & mask))
    if op == 'copy':
        hi = operands[0][0]
        lo = operands[0][1]
    elif op == 'xor':
        hi = operands[0][0] ^ operands[1][0]
        lo = operands[0][1] ^ operands[1][1]
    elif op == 'and':
        hi = operands[0][0] & operands[1][0]
        lo = operands[0][1] & operands[1][1]
    elif op == 'or':
        hi = operands[0][0] | operands[1][0]
        lo = operands[0][1] | operands[1][1]
    else:
        raise AssertionError(op)
    diff = (hi ^ lo) & mask
    return {j for j in range(nbytes) if (diff >> (j * 8)) & 0xFF}


@pytest.mark.parametrize('seed', range(40))
def test_wide_kernels_agree_and_are_exact(seed: int) -> None:
    rng = random.Random(seed)
    for entry in _CORPUS:
        ks, arch, asm, out_base, nbytes, bases, op = entry
        instr = _asm(ks, asm)
        for _ in range(6):
            values, taints = _lane_inputs(rng, bases, nbytes)
            or_in = {k: (values.get(k, 0) | t) & _FULL for k, t in taints.items()}
            and_in = {k: (values.get(k, 0) & ~t) & _FULL for k, t in taints.items()}
            for k, v in values.items():
                or_in.setdefault(k, v & _FULL)
                and_in.setdefault(k, v & _FULL)
            py = _cell_bytes(PCodeCellEvaluator, arch, instr, out_base, nbytes, or_in, and_in)
            c = _cell_bytes(PCodeCellEvaluatorC, arch, instr, out_base, nbytes, or_in, and_in)
            ctx = f'{asm} seed={seed} taints={taints}'
            assert py == c, f'PARITY: cell.pyx {py} != cell_c {c} :: {ctx}'
            if op is not None:
                oracle = _oracle_bytes(op, bases, values, taints, nbytes)
                assert py == oracle, f'EXACT: {py} != oracle {oracle} :: {ctx}'
