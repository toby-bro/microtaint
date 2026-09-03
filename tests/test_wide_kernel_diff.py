"""Equivalence harness for the width-native SIMD cell kernel (Stage 1a gate).

We are folding every ISA's SIMD (all extensions: x86 SSE/AVX, ARM64 NEON/SVE,
PPC AltiVec, ...) into NATIVE wide-type support in the simulation cell kernel, so
the rule engine can treat a wide (>8-byte) vector varnode as ONE operand and the
per-lane splitting (``VL_<offset>`` synthetic lanes) becomes obsolete.  The kernel
change is risky, so before touching kernel code we lock down an INDEPENDENT ground
truth and a measurement harness that both the current path and the future
width-native path are checked against.

Three ingredients
-----------------
* ``_oracle_taint`` -- the ground truth.  microtaint's taint of an op is, by
  definition, ``f(V | T) XOR f(V & ~T)`` (run the concrete op on the
  taint-forced-high inputs and on the taint-forced-low inputs, XOR the results).
  For the wide movement/bitwise/load anchor set (COPY, INT_XOR, INT_AND, INT_OR,
  INT_NEGATE, LOAD) the concrete op is plain Python integer arithmetic at
  arbitrary width, so the oracle is trivially correct and independent of BOTH
  kernels.  These ops are byte-parallel (output byte j depends only on input
  byte j), hence endianness-agnostic; the position-sensitive integer ops
  (INT_ZEXT/SEXT, SUBPIECE, PIECE) are deferred to the kernel-build increment
  with their own big-endian anchors.

* ``_engine_wide_taint`` -- PATH A, the CURRENT behaviour.  Drives the full rule
  engine (``generate_static_rule`` + ``circuit.evaluate``) exactly like
  ``test_simd_cross_isa``.  On ARM64 (LE) and PPC (BE) with a GP-only
  state_format, wide vector ops are handled by the geometry-derived ``VL_`` lane
  splitting.  Movement/XOR/value-aware OR are exact; AND and packed arithmetic
  are sound (over-approximate).  This is the regression baseline and, on the
  exact ops, it VALIDATES the oracle against known-good output in both
  endiannesses.

* ``_cell_wide_taint`` -- PATH B, the FUTURE behaviour.  Drives the cell kernel
  (``PCodeCellEvaluator.evaluate_differential``) directly, addressing each 8-byte
  output lane by its absolute offset via a ``VL_<hex>`` output name, with the
  wide op fed as a single real instruction.  TODAY this truncates: every vector
  register lives in the cold ``self.regs`` dict (offset >= 1104) and the 64-bit
  cold-path read/write does not carry bytes past the low 8 of a lane's parent
  (an ``x >> 64`` shift), so a wide reg->reg op mis-propagates the high lanes.
  Those cases are marked ``xfail`` and are the Stage-1a checklist: when the
  width-native kernel lands they turn green and the markers come off.  The scalar
  (<=8-byte) path through the SAME driver already matches the oracle, proving the
  driver is sound and the fast path is untouched.
"""

from __future__ import annotations

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

from microtaint.emulator.shadow import BitPreciseShadowMemory
from microtaint.instrumentation.ast import EvalContext
from microtaint.instrumentation.cell import PCodeCellEvaluator
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.sleigh.lifter import get_context
from microtaint.types import Architecture, Register

_FULL = 0xFFFFFFFFFFFFFFFF

_KS_X86 = Ks(KS_ARCH_X86, KS_MODE_64)
_KS_ARM = Ks(KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN)
_KS_PPC = Ks(KS_ARCH_PPC, KS_MODE_BIG_ENDIAN + KS_MODE_PPC32)


def _asm(ks: Ks, text: str) -> bytes:
    enc, _ = ks.asm(text, 0x1000)
    return bytes(enc)


def _reg_off(arch: Architecture, name: str) -> int:
    """Absolute Sleigh byte offset of a register, straight from pypcode."""
    return int(get_context(str(arch).split('.')[-1]).registers[name].offset)


def _vlane(base: int, k: int) -> str:
    return f'VL_{base + k:#x}'


def _gp_regs(names: list[str]) -> list[Register]:
    # GP-only state_format: NO vector register is enumerated, so the vector
    # lanes are synthesised from register geometry alone (the ISA-general path).
    return [Register(name=n, bits=64) for n in names]


def _tainted_bytes_from_lanes(out: dict[str, int], base: int, nbytes: int) -> set[int]:
    """Tainted byte indices (0..nbytes-1) read from a register's VL_ lanes."""
    tainted: set[int] = set()
    for k in range(0, nbytes, 8):
        mask = out.get(_vlane(base, k), 0)
        for i in range(min(8, nbytes - k)):
            if (mask >> (i * 8)) & 0xFF:
                tainted.add(k + i)
    return tainted


def _tainted_bytes_from_mask(mask: int, nbytes: int) -> set[int]:
    return {j for j in range(nbytes) if (mask >> (j * 8)) & 0xFF}


# ---------------------------------------------------------------------------
# The oracle: exact taint of a wide op via f(V|T) XOR f(V&~T), pure Python.
# ---------------------------------------------------------------------------

def _concrete(op: str, vals: list[int], nbytes: int) -> int:
    mask = (1 << (nbytes * 8)) - 1
    if op == 'copy':
        return vals[0] & mask
    if op == 'xor':
        return (vals[0] ^ vals[1]) & mask
    if op == 'and':
        return (vals[0] & vals[1]) & mask
    if op == 'or':
        return (vals[0] | vals[1]) & mask
    if op == 'negate':
        return (~vals[0]) & mask
    raise AssertionError(f'unmodelled op {op!r}')


def _oracle_taint(op: str, operands: list[tuple[int, int]], nbytes: int) -> set[int]:
    """Ground-truth tainted byte set for a wide op.

    ``operands`` is a list of ``(value, taint)`` wide ints (byte 0 = LSB).  The
    result is the differential ``f(V|T) XOR f(V&~T)`` reduced to the set of output
    bytes that carry any taint.
    """
    mask = (1 << (nbytes * 8)) - 1
    or_vals = [(v | t) & mask for v, t in operands]
    and_vals = [(v & ~t) & mask for v, t in operands]
    diff = _concrete(op, or_vals, nbytes) ^ _concrete(op, and_vals, nbytes)
    return _tainted_bytes_from_mask(diff, nbytes)


def _lanes_from_wide(value: int, taint: int, base: int, nbytes: int) -> tuple[dict, dict]:
    """Split a wide (value, taint) into per-8-byte VL_ input dicts (value, taint)."""
    values: dict[str, int] = {}
    taints: dict[str, int] = {}
    for k in range(0, nbytes, 8):
        values[_vlane(base, k)] = (value >> (k * 8)) & _FULL
        taints[_vlane(base, k)] = (taint >> (k * 8)) & _FULL
    return values, taints


# ---------------------------------------------------------------------------
# Path A -- the current engine (rule gen + VL_ lane splitting)
# ---------------------------------------------------------------------------

def _engine_wide_taint(
    arch: Architecture,
    code: bytes,
    gp: list[str],
    out_base: int,
    nbytes: int,
    input_taint: dict[str, int],
    input_values: dict[str, int] | None = None,
    shadow: BitPreciseShadowMemory | None = None,
) -> set[int]:
    circuit = generate_static_rule(arch, code, _gp_regs(gp))
    out = circuit.evaluate(EvalContext(
        input_values=input_values or {},
        input_taint=input_taint,
        simulator=CellSimulator(arch),
        shadow_memory=shadow,
    ))
    return _tainted_bytes_from_lanes(out, out_base, nbytes)


# ---------------------------------------------------------------------------
# Path B -- the cell kernel directly, one 8-byte output lane at a time
# ---------------------------------------------------------------------------

def _cell_wide_taint(
    arch: Architecture,
    instr_hex: str,
    out_base: int,
    nbytes: int,
    in_values: dict[str, int],
    in_taints: dict[str, int],
) -> set[int]:
    """Drive PCodeCellEvaluator.evaluate_differential once per 8-byte output lane.

    Inputs are VL_-addressed (value, taint) dicts; the differential is fed
    ``V|T`` and ``V&~T`` exactly as the engine's InstructionCellExpr does.
    """
    ev = PCodeCellEvaluator(arch)
    or_in = {k: (in_values.get(k, 0) | t) & _FULL for k, t in in_taints.items()}
    and_in = {k: (in_values.get(k, 0) & ~t) & _FULL for k, t in in_taints.items()}
    # value-only inputs (no taint) still need to be present in both runs
    for k, v in in_values.items():
        or_in.setdefault(k, v & _FULL)
        and_in.setdefault(k, v & _FULL)
    tainted: set[int] = set()
    for k in range(0, nbytes, 8):
        cell = types.SimpleNamespace(
            instruction=instr_hex, out_reg=_vlane(out_base, k),
            out_bit_start=0, out_bit_end=63,
        )
        lane = ev.evaluate_differential(cell, dict(or_in), dict(and_in))
        for i in range(min(8, nbytes - k)):
            if (lane >> (i * 8)) & 0xFF:
                tainted.add(k + i)
    return tainted


# ===========================================================================
# 1. Oracle self-checks (hand-verified) -- the ground truth is trustworthy.
# ===========================================================================

def test_oracle_copy_is_identity() -> None:
    # byte 0 and byte 15 tainted -> those exact bytes tainted.
    t = (0xFF) | (0xFF << (15 * 8))
    assert _oracle_taint('copy', [(0, t)], 16) == {0, 15}


def test_oracle_xor_is_union() -> None:
    a = 0x00000000FFFFFFFF  # bytes 0-3
    b = 0xFFFFFFFF00000000  # bytes 4-7
    assert _oracle_taint('xor', [(0, a), (0, b)], 8) == set(range(8))


def test_oracle_or_clears_taint_masked_by_concrete_ones() -> None:
    # a fully tainted; b a concrete ~0 (untainted). a|b == ~0 regardless of a,
    # so the output carries NO taint (value-aware OR).
    assert _oracle_taint('or', [(0, _FULL), (_FULL, 0)], 8) == set()
    # b concrete 0 passes a's taint through unchanged.
    assert _oracle_taint('or', [(0, _FULL), (0, 0)], 8) == set(range(8))


def test_oracle_and_clears_taint_masked_by_concrete_zeros() -> None:
    # a fully tainted; b a concrete 0. a&b == 0 regardless of a -> no taint.
    assert _oracle_taint('and', [(0, _FULL), (0, 0)], 8) == set()
    # b concrete ~0 passes a's taint through unchanged.
    assert _oracle_taint('and', [(0, _FULL), (_FULL, 0)], 8) == set(range(8))


def test_oracle_negate_preserves_taint() -> None:
    assert _oracle_taint('negate', [(0, 0xFF00)], 8) == {1}


# ===========================================================================
# 2. Path A (current engine) vs oracle -- LE (ARM64) and BE (PPC).
#    Validates the oracle on the exact ops; documents soundness on AND.
# ===========================================================================

def test_engine_arm64_mov_matches_oracle_exact() -> None:
    v1 = _reg_off(Architecture.ARM64, 'q1')
    q0 = _reg_off(Architecture.ARM64, 'q0')
    taint = 0xFF | (0xFF << (8 * 8))  # byte 0 and byte 8
    _, taints = _lanes_from_wide(0, taint, v1, 16)
    got = _engine_wide_taint(
        Architecture.ARM64, _asm(_KS_ARM, 'orr v0.16b, v1.16b, v1.16b'),
        ['x0', 'x1', 'x2'], q0, 16, taints,
    )
    assert got == _oracle_taint('copy', [(0, taint)], 16) == {0, 8}


def test_engine_ppc_vxor_matches_oracle_exact() -> None:
    vr0 = _reg_off(Architecture.PPC32BE, 'vs32')
    vr1 = _reg_off(Architecture.PPC32BE, 'vs33')
    vr2 = _reg_off(Architecture.PPC32BE, 'vs34')
    a = 0xFF00FF00FF00FF00
    b = 0x00FF00FF00FF00FF
    _, ta = _lanes_from_wide(0, a, vr1, 16)
    _, tb = _lanes_from_wide(0, b, vr2, 16)
    got = _engine_wide_taint(
        Architecture.PPC32BE, _asm(_KS_PPC, 'vxor 0, 1, 2'),
        ['r0', 'r1', 'r2'], vr0, 16, {**ta, **tb},
    )
    # low lane: every byte tainted by one operand or the other.
    assert got & set(range(8)) == _oracle_taint('xor', [(0, a), (0, b)], 8)


def test_engine_ppc_vor_value_aware_matches_oracle_exact() -> None:
    vr0 = _reg_off(Architecture.PPC32BE, 'vs32')
    vr1 = _reg_off(Architecture.PPC32BE, 'vs33')
    vr2 = _reg_off(Architecture.PPC32BE, 'vs34')
    # vr1 fully tainted; vr2 concrete: low lane ~0 (clears), high lane 0 (passes).
    input_taint = {_vlane(vr1, 0): _FULL, _vlane(vr1, 8): _FULL}
    input_values = {_vlane(vr2, 0): _FULL, _vlane(vr2, 8): 0}
    got = _engine_wide_taint(
        Architecture.PPC32BE, _asm(_KS_PPC, 'vor 0, 1, 2'),
        ['r0', 'r1', 'r2'], vr0, 16, input_taint, input_values,
    )
    operands_lo = [(0, _FULL), (_FULL, 0)]  # cleared
    operands_hi = [(0, _FULL), (0, 0)]      # passed
    expect = _oracle_taint('or', operands_lo, 8) | {
        8 + j for j in _oracle_taint('or', operands_hi, 8)
    }
    assert got == expect == set(range(8, 16))


def test_engine_ppc_vand_is_sound() -> None:
    # AND degrades to the sound union without values: >= oracle, no under-taint.
    vr0 = _reg_off(Architecture.PPC32BE, 'vs32')
    vr1 = _reg_off(Architecture.PPC32BE, 'vs33')
    input_taint = {_vlane(vr1, 0): _FULL, _vlane(vr1, 8): _FULL}
    got = _engine_wide_taint(
        Architecture.PPC32BE, _asm(_KS_PPC, 'vand 0, 1, 2'),
        ['r0', 'r1', 'r2'], vr0, 16, input_taint,
    )
    # oracle with both operands' concrete values unknown -> full union.
    oracle = _oracle_taint('and', [(0, _FULL), (0, _FULL)], 16)
    assert oracle.issubset(got)  # sound


def test_engine_arm64_ldr_q_matches_oracle_exact() -> None:
    q0 = _reg_off(Architecture.ARM64, 'q0')
    src = 0x5000
    shadow = BitPreciseShadowMemory()
    tainted_src = {0, 1, 7, 8, 15}
    for i in tainted_src:
        shadow.write_mask(src + i, 0xFF, 1)
    got = _engine_wide_taint(
        Architecture.ARM64, _asm(_KS_ARM, 'ldr q0, [x1]'),
        ['x0', 'x1', 'x2'], q0, 16, {}, {'x1': src}, shadow,
    )
    assert got == tainted_src  # LOAD oracle: register byte j = memory byte j


# ===========================================================================
# 3. Path B (cell kernel directly) -- scalar sanity now, wide is the Stage-1a
#    target (xfail until the width-native kernel lands).
# ===========================================================================

def test_cell_scalar_xor_matches_oracle() -> None:
    # 8-byte GP XOR through the SAME direct-cell driver: fast path, exact today.
    # eax ^= ebx ; taint eax bytes 0-1, ebx bytes 2-3.
    instr = _asm(_KS_X86, 'xor eax, ebx').hex()
    eax = _reg_off(Architecture.AMD64, 'EAX')
    ebx = _reg_off(Architecture.AMD64, 'EBX')
    ta = {f'VL_{eax:#x}': 0x0000FFFF, f'VL_{ebx:#x}': 0xFFFF0000}
    got = _cell_wide_taint(Architecture.AMD64, instr, eax, 4, {}, ta)
    assert got == _oracle_taint('xor', [(0, 0x0000FFFF), (0, 0xFFFF0000)], 4) == {0, 1, 2, 3}


_XMM0 = 0x1200
_XMM1 = 0x1240

# Characterisation (measured, cell.pyx use_c=False): every vector register sits
# in the cold ``self.regs`` dict (offset >= 1104), and its 64-bit cold-path
# read/write cannot carry bytes past the low 8 of a lane's parent.  The high lane
# (bytes 8-15) of a wide reg->reg op is therefore computed wrong, in BOTH
# directions depending on the opcode:
#   COPY : high lane mirrors the low lane (over-taints when only the low lane is
#          tainted; UNDER-TAINTS -- drops real taint -- when only the high lane is)
#   XOR  : high lane is dropped to 0 (UNDER-TAINTS whenever the high lane carries taint)
#   AND  : high lane ignores its operand's concrete value (value-aware clear fails)
# The two under-taints are soundness violations, reachable at the cell level; in
# production the rule engine hides them by splitting wide varnodes into VL_ lanes
# and never routing a wide reg op through the cell.  The width-native kernel makes
# the cell itself correct so the splitting can be deleted.  Each test below pins
# the EXACT taint a human expects (never the current engine output) on an input
# verified to expose the defect; strict=True flips XPASS->fail the instant the
# kernel is fixed, forcing removal of the marker (a red->green ratchet).


@pytest.mark.xfail(strict=True,
                   reason='Stage-1a: cold-dict wide reg->reg COPY drops high-lane '
                          'taint (soundness under-taint); width-native kernel pending')
def test_cell_wide_copy_high_lane_undertaint() -> None:
    # movdqa xmm0, xmm1 with ONLY xmm1's high lane (bytes 8-15) tainted.
    instr = _asm(_KS_X86, 'movdqa xmm0, xmm1').hex()
    taint = _FULL << 64  # high lane fully tainted, low lane clean
    _, taints = _lanes_from_wide(0, taint, _XMM1, 16)
    got = _cell_wide_taint(Architecture.AMD64, instr, _XMM0, 16, {}, taints)
    # A copy carries byte j -> byte j: the high lane's taint must survive.
    assert got == _oracle_taint('copy', [(0, taint)], 16) == set(range(8, 16))


@pytest.mark.xfail(strict=True,
                   reason='Stage-1a: cold-dict wide reg->reg COPY mirrors the low '
                          'lane into the high lane (over-taint); width-native kernel pending')
def test_cell_wide_copy_high_lane_overtaint() -> None:
    # movdqa xmm0, xmm1 with ONLY xmm1's low lane tainted.
    instr = _asm(_KS_X86, 'movdqa xmm0, xmm1').hex()
    taint = _FULL  # low lane fully tainted, high lane clean
    _, taints = _lanes_from_wide(0, taint, _XMM1, 16)
    got = _cell_wide_taint(Architecture.AMD64, instr, _XMM0, 16, {}, taints)
    # The clean high lane must stay clean; only bytes 0-7 are tainted.
    assert got == _oracle_taint('copy', [(0, taint)], 16) == set(range(8))


@pytest.mark.xfail(strict=True,
                   reason='Stage-1a: cold-dict wide INT_XOR drops the high lane to 0 '
                          '(soundness under-taint); width-native kernel pending')
def test_cell_wide_xor_high_lane_undertaint() -> None:
    # pxor xmm0, xmm1 with taint ONLY in the high lane (xmm1 bytes 8-15).
    instr = _asm(_KS_X86, 'pxor xmm0, xmm1').hex()
    b = _FULL << 64  # xmm1 high lane tainted
    _, tb = _lanes_from_wide(0, b, _XMM1, 16)
    got = _cell_wide_taint(Architecture.AMD64, instr, _XMM0, 16, {}, tb)
    # XOR taint is the byte union: the high lane's taint must reach the output.
    assert got == _oracle_taint('xor', [(0, 0), (0, b)], 16) == set(range(8, 16))


@pytest.mark.xfail(strict=True,
                   reason='Stage-1a: cold-dict wide INT_AND ignores the high lane '
                          "operand's concrete value (value-aware clear fails); kernel pending")
def test_cell_wide_and_high_lane_value_aware() -> None:
    # pand xmm0, xmm1 : xmm0 fully tainted; xmm1 low lane concrete ~0 (passes the
    # taint), high lane concrete 0 (masks it away).  Human-expected: low tainted,
    # high cleared.
    instr = _asm(_KS_X86, 'pand xmm0, xmm1').hex()
    v0 = {_vlane(_XMM0, 0): 0, _vlane(_XMM0, 8): 0}
    t0 = {_vlane(_XMM0, 0): _FULL, _vlane(_XMM0, 8): _FULL}
    v1 = {_vlane(_XMM1, 0): _FULL, _vlane(_XMM1, 8): 0}
    got = _cell_wide_taint(Architecture.AMD64, instr, _XMM0, 16, {**v0, **v1}, t0)
    operands_lo = [(0, _FULL), (_FULL, 0)]   # xmm1 lane0 == ~0 -> passed
    operands_hi = [(0, _FULL), (0, 0)]       # xmm1 lane1 == 0  -> cleared
    expect = _oracle_taint('and', operands_lo, 8) | {
        8 + j for j in _oracle_taint('and', operands_hi, 8)
    }
    assert got == expect == set(range(8))
