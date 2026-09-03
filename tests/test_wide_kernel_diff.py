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
  byte j), hence endianness-agnostic; the position-sensitive whole-register
  shifts (psrldq/pslldq -> INT_RIGHT/INT_LEFT) run on the target-endian value.

* ``_engine_wide_taint`` -- PATH A, the CURRENT behaviour.  Drives the full rule
  engine (``generate_static_rule`` + ``circuit.evaluate``) exactly like
  ``test_simd_cross_isa``.  On ARM64 (LE) and PPC (BE) with a GP-only
  state_format, wide vector ops are handled by the geometry-derived ``VL_`` lane
  splitting.  Movement/XOR/value-aware OR are exact; AND and packed arithmetic
  are sound (over-approximate).  This is the regression baseline and, on the
  exact ops, it VALIDATES the oracle against known-good output in both
  endiannesses.

* ``_cell_wide_taint`` -- PATH B, the width-native cell kernel.  Drives the cell
  kernel (``PCodeCellEvaluator.evaluate_differential``) directly, addressing each
  8-byte output lane by its absolute offset via a ``VL_<hex>`` output name, with
  the wide op fed as a single real instruction.  Every vector register lives in
  an endianness-neutral cold byte store (offset >= 1104); the wide-register
  handlers (Stage 1a) propagate movement, byte-parallel bitwise, LOAD/STORE and
  whole-register shifts at arbitrary width via Python-int values, so the high
  lanes are exact under both endiannesses.  Opaque data ops (CALLOTHER shuffles
  like pshufb, FLOAT SIMD) are AVALANCHED -- any tainted input taints the whole
  output -- with no Unicorn dependency.  Before Stage 1a the 64-bit cold path
  truncated / mirrored the high lanes (an ``x >> 64`` shift returning the low
  lane); the ``test_cell_wide_*`` tests pin those exact defects so they cannot
  regress.  The scalar (<=8-byte) path through the SAME driver matches the oracle
  too, proving the driver is sound and the fast path is untouched.

Coverage spans little-endian (x86 SSE, ARM64 NEON) and big-endian (PPC AltiVec)
vector files.  The handlers are ISA-agnostic (they key on space + size, not the
ISA), so wide register ops on any other ISA with a vector file (RISC-V V,
MIPS/SPARC) route through the same validated code.
"""

from __future__ import annotations

import types

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
from microtaint.instrumentation.cell_c.cell_c import PCodeCellEvaluatorC
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
    evaluator_cls: type = PCodeCellEvaluator,
) -> set[int]:
    """Drive <evaluator>.evaluate_differential once per 8-byte output lane.

    Inputs are VL_-addressed (value, taint) dicts; the differential is fed
    ``V|T`` and ``V&~T`` exactly as the engine's InstructionCellExpr does.
    ``evaluator_cls`` selects the kernel (cell.pyx or the C kernel).
    """
    ev = evaluator_cls(arch)
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


def _mk_diff_inputs(in_values: dict[str, int], in_taints: dict[str, int]) -> tuple[dict, dict]:
    or_in = {k: (in_values.get(k, 0) | t) & _FULL for k, t in in_taints.items()}
    and_in = {k: (in_values.get(k, 0) & ~t) & _FULL for k, t in in_taints.items()}
    for k, v in in_values.items():
        or_in.setdefault(k, v & _FULL)
        and_in.setdefault(k, v & _FULL)
    return or_in, and_in


def _cell_wide_taint_sleigh(
    arch: Architecture, instr_hex: str, out_base: int, nbytes: int,
    in_values: dict[str, int], in_taints: dict[str, int], be: bool,
) -> set[int]:
    """Like _cell_wide_taint but reports ARCHITECTURAL (sleigh) byte indices.

    An 8-byte VL_ lane is read as an integer whose byte i (bits 8i..) is sleigh
    byte off+i under LE but sleigh byte off+7-i under BE.  Un-reversing that gives
    'byte j' = the register's j-th byte in memory order, which is what a wide
    LOAD/STORE preserves against memory byte j regardless of target endianness.
    """
    ev = PCodeCellEvaluator(arch)
    or_in, and_in = _mk_diff_inputs(in_values, in_taints)
    tainted: set[int] = set()
    for k in range(0, nbytes, 8):
        cell = types.SimpleNamespace(
            instruction=instr_hex, out_reg=_vlane(out_base, k),
            out_bit_start=0, out_bit_end=63,
        )
        lane = ev.evaluate_differential(cell, dict(or_in), dict(and_in))
        for i in range(8):
            if (lane >> (i * 8)) & 0xFF:
                tainted.add(k + (7 - i if be else i))
    return tainted


def _cell_store_mem_taint(
    arch: Architecture, instr_hex: str, store_addr: int, nbytes: int,
    in_values: dict[str, int], in_taints: dict[str, int],
) -> set[int]:
    """Drive a wide STORE and read back each stored memory byte's taint (byte j =
    architectural memory offset store_addr+j), endianness-independent."""
    ev = PCodeCellEvaluator(arch)
    or_in, and_in = _mk_diff_inputs(in_values, in_taints)
    tainted: set[int] = set()
    for j in range(nbytes):
        cell = types.SimpleNamespace(
            instruction=instr_hex, out_reg=f'MEM_{store_addr + j:#x}_1',
            out_bit_start=0, out_bit_end=7,
        )
        if ev.evaluate_differential(cell, dict(or_in), dict(and_in)):
            tainted.add(j)
    return tainted


def _mem_byte_taints(addr: int, byte_indices: set[int]) -> dict[str, int]:
    """Per-byte memory taint via 1-byte MEM_ keys (size 1 has no endian ambiguity)."""
    return {f'MEM_{addr + j:#x}_1': 0xFF for j in byte_indices}


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

# These four cases pin defects that existed BEFORE the width-native kernel (and
# were measured on cell.pyx use_c=False): every vector register sits in the cold
# register store (offset >= 1104), and the old 64-bit cold path could not
# carry bytes past the low 8 of a lane's parent, so the high lane (bytes 8-15) of
# a wide reg->reg op was computed wrong in BOTH directions depending on opcode:
#   COPY : high lane mirrored the low lane (over-taint when only the low lane was
#          tainted; UNDER-TAINT -- dropped real taint -- when only the high lane was)
#   XOR  : high lane dropped to 0 (UNDER-TAINT whenever the high lane carried taint)
#   AND  : high lane ignored its operand's concrete value (value-aware clear failed)
# The two under-taints were soundness violations, reachable at the cell level.
# Each test asserts the EXACT taint a human expects (never engine output) on an
# input that exercises the high lane, so the width-native handlers cannot regress.


def test_cell_wide_copy_high_lane_undertaint() -> None:
    # movdqa xmm0, xmm1 with ONLY xmm1's high lane (bytes 8-15) tainted.
    instr = _asm(_KS_X86, 'movdqa xmm0, xmm1').hex()
    taint = _FULL << 64  # high lane fully tainted, low lane clean
    _, taints = _lanes_from_wide(0, taint, _XMM1, 16)
    got = _cell_wide_taint(Architecture.AMD64, instr, _XMM0, 16, {}, taints)
    # A copy carries byte j -> byte j: the high lane's taint must survive.
    assert got == _oracle_taint('copy', [(0, taint)], 16) == set(range(8, 16))


def test_cell_wide_copy_high_lane_overtaint() -> None:
    # movdqa xmm0, xmm1 with ONLY xmm1's low lane tainted.
    instr = _asm(_KS_X86, 'movdqa xmm0, xmm1').hex()
    taint = _FULL  # low lane fully tainted, high lane clean
    _, taints = _lanes_from_wide(0, taint, _XMM1, 16)
    got = _cell_wide_taint(Architecture.AMD64, instr, _XMM0, 16, {}, taints)
    # The clean high lane must stay clean; only bytes 0-7 are tainted.
    assert got == _oracle_taint('copy', [(0, taint)], 16) == set(range(8))


def test_cell_wide_xor_high_lane_undertaint() -> None:
    # pxor xmm0, xmm1 with taint ONLY in the high lane (xmm1 bytes 8-15).
    instr = _asm(_KS_X86, 'pxor xmm0, xmm1').hex()
    b = _FULL << 64  # xmm1 high lane tainted
    _, tb = _lanes_from_wide(0, b, _XMM1, 16)
    got = _cell_wide_taint(Architecture.AMD64, instr, _XMM0, 16, {}, tb)
    # XOR taint is the byte union: the high lane's taint must reach the output.
    assert got == _oracle_taint('xor', [(0, 0), (0, b)], 16) == set(range(8, 16))


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


# ===========================================================================
# 4. Big-endian wide register ops (PPC AltiVec) -- prove endianness handling.
#    The cold vector store is byte-addressable; the target-endian integer is
#    assembled/decomposed only at the byte<->int boundary, so byte-parallel ops
#    are exact under BE as well as LE.  Before the byte-store the LE-only wide
#    read/write silently swapped lanes on a BE target (a soundness under-taint).
# ===========================================================================

_VR0 = 0x4200  # PPC vs32 / vr0
_VR1 = 0x4210  # vs33 / vr1
_VR2 = 0x4220  # vs34 / vr2


def test_cell_wide_xor_be_low_lane() -> None:
    # vxor vr0, vr1, vr2 with ONLY vr1's low lane (bytes 0-7) tainted.
    instr = _asm(_KS_PPC, 'vxor 0, 1, 2').hex()
    a = _FULL  # low lane
    _, ta = _lanes_from_wide(0, a, _VR1, 16)
    got = _cell_wide_taint(Architecture.PPC32BE, instr, _VR0, 16, {}, ta)
    assert got == _oracle_taint('xor', [(0, a), (0, 0)], 16) == set(range(8))


def test_cell_wide_xor_be_high_lane() -> None:
    # vxor with ONLY vr2's high lane (bytes 8-15) tainted -- the lane the LE-only
    # code used to drop on a BE target (under-taint).
    instr = _asm(_KS_PPC, 'vxor 0, 1, 2').hex()
    b = _FULL << 64
    _, tb = _lanes_from_wide(0, b, _VR2, 16)
    got = _cell_wide_taint(Architecture.PPC32BE, instr, _VR0, 16, {}, tb)
    assert got == _oracle_taint('xor', [(0, 0), (0, b)], 16) == set(range(8, 16))


def test_cell_wide_or_be_value_aware() -> None:
    # vor vr0, vr1, vr2 : vr1 fully tainted; vr2 concrete low lane ~0 (OR-mask
    # clears the taint), high lane 0 (passes it).  Exact value-aware OR under BE.
    instr = _asm(_KS_PPC, 'vor 0, 1, 2').hex()
    v2 = {_vlane(_VR2, 0): _FULL, _vlane(_VR2, 8): 0}
    t1 = {_vlane(_VR1, 0): _FULL, _vlane(_VR1, 8): _FULL}
    got = _cell_wide_taint(Architecture.PPC32BE, instr, _VR0, 16, v2, t1)
    operands_lo = [(0, _FULL), (_FULL, 0)]   # vr2 lane0 == ~0 -> cleared
    operands_hi = [(0, _FULL), (0, 0)]       # vr2 lane1 == 0  -> passed
    expect = _oracle_taint('or', operands_lo, 8) | {
        8 + j for j in _oracle_taint('or', operands_hi, 8)
    }
    assert got == expect == set(range(8, 16))


# ===========================================================================
# 5. Wide vector LOAD / STORE at the cell level (LE + BE).  A wide load/store
#    carries memory byte j <-> register byte j (architectural / sleigh order),
#    which is the exact taint a human expects on both endiannesses.
# ===========================================================================


def test_cell_wide_load_arm_ldr_q_le() -> None:
    instr = _asm(_KS_ARM, 'ldr q0, [x1]').hex()
    q0 = _reg_off(Architecture.ARM64, 'q0')
    src, tb = 0x5000, {0, 1, 7, 8, 15}
    got = _cell_wide_taint_sleigh(
        Architecture.ARM64, instr, q0, 16, {'x1': src}, _mem_byte_taints(src, tb), be=False)
    assert got == tb  # memory byte j -> register byte j


def test_cell_wide_load_ppc_lvx_be() -> None:
    # lvx vr0, r1, r2 : address = (r1 + r2) & ~0xf.  r1 aligned, r2 = 0.
    instr = _asm(_KS_PPC, 'lvx 0, 1, 2').hex()
    vr0 = _reg_off(Architecture.PPC32BE, 'vs32')
    src, tb = 0x5000, {0, 1, 7, 8, 15}
    got = _cell_wide_taint_sleigh(
        Architecture.PPC32BE, instr, vr0, 16, {'R1': src, 'R2': 0},
        _mem_byte_taints(src, tb), be=True)
    assert got == tb  # memory byte j -> register byte j, under BE too


def test_cell_wide_store_x86_movdqu_le() -> None:
    instr = _asm(_KS_X86, 'movdqu [rdi], xmm0').hex()
    dst = 0x6000
    # xmm0 low lane fully tainted, high lane clean.
    taints = {_vlane(_XMM0, 0): _FULL, _vlane(_XMM0, 8): 0}
    got = _cell_store_mem_taint(Architecture.AMD64, instr, dst, 16, {'RDI': dst}, taints)
    assert got == set(range(8))  # register bytes 0-7 -> memory bytes 0-7


def test_cell_wide_store_ppc_stvx_be() -> None:
    instr = _asm(_KS_PPC, 'stvx 0, 1, 2').hex()
    vr0 = _reg_off(Architecture.PPC32BE, 'vs32')
    dst = 0x6000
    taints = {_vlane(vr0, 0): _FULL, _vlane(vr0, 8): 0}
    got = _cell_store_mem_taint(
        Architecture.PPC32BE, instr, dst, 16, {'R1': dst, 'R2': 0}, taints)
    assert got == set(range(8))  # register bytes 0-7 -> memory bytes 0-7, under BE


# ===========================================================================
# 6. Whole-register byte shifts (x86 psrldq/pslldq -> a single >8-byte
#    INT_RIGHT/INT_LEFT).  Position-sensitive: output byte j takes input byte
#    j +/- shift; bytes shifted past either end are dropped (clean).
# ===========================================================================


def test_cell_wide_psrldq_byte_shift() -> None:
    # psrldq xmm0, 4 : logical right shift by 4 bytes -> output byte j = input j+4.
    instr = _asm(_KS_X86, 'psrldq xmm0, 4').hex()
    taint = (0xFF << (4 * 8)) | (0xFF << (8 * 8)) | (0xFF << (15 * 8))  # in bytes 4,8,15
    _, taints = _lanes_from_wide(0, taint, _XMM0, 16)
    got = _cell_wide_taint(Architecture.AMD64, instr, _XMM0, 16, {}, taints)
    assert got == {0, 4, 11}  # 4->0, 8->4, 15->11 (bytes < shift are dropped)


def test_cell_wide_pslldq_byte_shift() -> None:
    # pslldq xmm0, 4 : left shift by 4 bytes -> output byte j = input byte j-4.
    instr = _asm(_KS_X86, 'pslldq xmm0, 4').hex()
    taint = 0xFF | (0xFF << (7 * 8)) | (0xFF << (11 * 8))  # in bytes 0,7,11
    _, taints = _lanes_from_wide(0, taint, _XMM0, 16)
    got = _cell_wide_taint(Architecture.AMD64, instr, _XMM0, 16, {}, taints)
    assert got == {4, 11, 15}  # 0->4, 7->11, 11->15 (bytes past the top are dropped)


# ===========================================================================
# 7. Opaque data ops (CALLOTHER shuffles / crypto, FLOAT SIMD) -> AVALANCHE.
#    The cell no longer runs these via Unicorn; any tainted input taints the
#    whole output slice (sound over-approximation, self-contained).
# ===========================================================================


def test_cell_wide_pshufb_avalanche() -> None:
    # pshufb xmm0, xmm1 lifts to a CALLOTHER.  Any tainted input byte must taint
    # the whole 16-byte output (a data-dependent shuffle can move it anywhere).
    instr = _asm(_KS_X86, 'pshufb xmm0, xmm1').hex()
    got = _cell_wide_taint(Architecture.AMD64, instr, _XMM0, 16, {}, {_vlane(_XMM1, 0): 0xFF})
    assert got == set(range(16))  # avalanche


def test_cell_wide_pshufb_no_taint_is_clean() -> None:
    # No tainted input -> no output taint (avalanche must not invent taint).
    instr = _asm(_KS_X86, 'pshufb xmm0, xmm1').hex()
    got = _cell_wide_taint(Architecture.AMD64, instr, _XMM0, 16, {}, {})
    assert got == set()


def test_cell_wide_ppc_vand_avalanche_be() -> None:
    # PPC vand lifts to a CALLOTHER too; avalanche under BE, no Unicorn.
    instr = _asm(_KS_PPC, 'vand 0, 1, 2').hex()
    got = _cell_wide_taint(Architecture.PPC32BE, instr, _VR0, 16, {}, {_vlane(_VR1, 8): 0xFF})
    assert got == set(range(16))  # avalanche


# ===========================================================================
# 8. 256-bit AVX (YMM) and the VEX upper-clear.  A VEX-encoded op writes its
#    result to a 16/32-byte UNIQUE temp, then INT_ZEXTs it into the 512-bit ZMM
#    (zeroing the upper bytes).  Exercises >128-bit unique storage (uniq_wide)
#    and the wide-register INT_ZEXT.  x86 is little-endian, the perf-critical ISA.
# ===========================================================================

_YMM0, _YMM1, _YMM2 = 0x1200, 0x1240, 0x1280


def test_cell_wide_vmovdqa_ymm_256() -> None:
    instr = _asm(_KS_X86, 'vmovdqa ymm0, ymm1').hex()
    taint = 0xFF | (0xFF << (16 * 8)) | (0xFF << (31 * 8))  # ymm1 bytes 0,16,31
    _, taints = _lanes_from_wide(0, taint, _YMM1, 32)
    got = _cell_wide_taint(Architecture.AMD64, instr, _YMM0, 32, {}, taints)
    assert got == {0, 16, 31}  # exact 32-byte copy


def test_cell_wide_vpxor_ymm_256() -> None:
    # vpxor lands in a 32-byte UNIQUE temp then ZEXTs to the ZMM: needs uniq_wide.
    instr = _asm(_KS_X86, 'vpxor ymm0, ymm1, ymm2').hex()
    a = 0xFF | (0xFF << (20 * 8))          # ymm1 bytes 0, 20
    b = (0xFF << (8 * 8)) | (0xFF << (31 * 8))  # ymm2 bytes 8, 31
    _, ta = _lanes_from_wide(0, a, _YMM1, 32)
    _, tb = _lanes_from_wide(0, b, _YMM2, 32)
    got = _cell_wide_taint(Architecture.AMD64, instr, _YMM0, 32, {}, {**ta, **tb})
    assert got == {0, 8, 20, 31}  # exact byte union across 32 bytes


def test_cell_wide_vpxor_xmm_avx128_upper_clear() -> None:
    # AVX-128 vpxor xmm still emits the ZEXT-to-ZMM upper-clear (uniq:16 -> reg:64).
    instr = _asm(_KS_X86, 'vpxor xmm0, xmm1, xmm2').hex()
    a = 0x00000000FFFFFFFF   # xmm1 bytes 0-3
    b = 0xFFFFFFFF00000000   # xmm2 bytes 4-7
    _, ta = _lanes_from_wide(0, a, _XMM1, 16)
    _, tb = _lanes_from_wide(0, b, 0x1280, 16)  # xmm2
    got = _cell_wide_taint(Architecture.AMD64, instr, _XMM0, 16, {}, {**ta, **tb})
    assert got == set(range(8))  # low lane union; high lane clean


def test_cell_wide_vpand_ymm_avalanche() -> None:
    instr = _asm(_KS_X86, 'vpand ymm0, ymm1, ymm2').hex()
    got = _cell_wide_taint(Architecture.AMD64, instr, _YMM0, 32, {}, {_vlane(_YMM1, 8): 0xFF})
    assert got == set(range(32))  # CALLOTHER -> avalanche across all 32 bytes


def test_cell_wide_vmovdqu_ymm_load_256() -> None:
    instr = _asm(_KS_X86, 'vmovdqu ymm0, [rdi]').hex()
    src, tb = 0x5000, {0, 1, 15, 16, 31}
    got = _cell_wide_taint_sleigh(
        Architecture.AMD64, instr, _YMM0, 32, {'RDI': src}, _mem_byte_taints(src, tb), be=False)
    assert got == tb  # memory byte j -> register byte j across 32 bytes


def test_cell_wide_vmovdqu_ymm_store_256() -> None:
    instr = _asm(_KS_X86, 'vmovdqu [rdi], ymm0').hex()
    dst = 0x6000
    # ymm0 low 16 bytes tainted (lanes 0,1), high 16 clean (lanes 2,3).
    taints = {_vlane(_YMM0, 0): _FULL, _vlane(_YMM0, 8): _FULL,
              _vlane(_YMM0, 16): 0, _vlane(_YMM0, 24): 0}
    got = _cell_store_mem_taint(Architecture.AMD64, instr, dst, 32, {'RDI': dst}, taints)
    assert got == set(range(16))  # register bytes 0-15 -> memory bytes 0-15


# ===========================================================================
# 9. cell_c (the production C kernel) parity for the avalanche floor.  Stage 1b
#    begins the port; opaque CALLOTHER/FLOAT ops must avalanche identically to
#    cell.pyx, with no Unicorn.  (The avalanche short-circuit runs before the
#    frame load, so it needs no vector backing store -- that lands next.)
# ===========================================================================


def test_cellc_pshufb_avalanche_matches_pyx() -> None:
    instr = _asm(_KS_X86, 'pshufb xmm0, xmm1').hex()
    taints = {_vlane(_XMM1, 0): 0xFF}
    py = _cell_wide_taint(Architecture.AMD64, instr, _XMM0, 16, {}, taints, PCodeCellEvaluator)
    c = _cell_wide_taint(Architecture.AMD64, instr, _XMM0, 16, {}, taints, PCodeCellEvaluatorC)
    assert py == c == set(range(16))  # both avalanche


def test_cellc_pshufb_no_taint_matches_pyx() -> None:
    instr = _asm(_KS_X86, 'pshufb xmm0, xmm1').hex()
    py = _cell_wide_taint(Architecture.AMD64, instr, _XMM0, 16, {}, {}, PCodeCellEvaluator)
    c = _cell_wide_taint(Architecture.AMD64, instr, _XMM0, 16, {}, {}, PCodeCellEvaluatorC)
    assert py == c == set()  # neither invents taint


def test_cellc_vpand_ymm_avalanche_matches_pyx() -> None:
    instr = _asm(_KS_X86, 'vpand ymm0, ymm1, ymm2').hex()
    taints = {_vlane(_YMM1, 8): 0xFF}
    py = _cell_wide_taint(Architecture.AMD64, instr, _YMM0, 32, {}, taints, PCodeCellEvaluator)
    c = _cell_wide_taint(Architecture.AMD64, instr, _YMM0, 32, {}, taints, PCodeCellEvaluatorC)
    assert py == c == set(range(32))  # both avalanche across 32 bytes


# ===========================================================================
# 10. cell_c parity for wide movement / bitwise, 128-bit and 256-bit, LE + BE.
#     The C kernel's lane-split handler now covers COPY/XOR/AND/OR/ZEXT/SEXT at
#     arbitrary width (register and unique), so it must agree byte-for-byte with
#     the cell.pyx reference.  (LOAD/STORE, shifts and ARM offsets >= 17000 are
#     still being ported and are not asserted here.)
# ===========================================================================


def _both_kernels(arch: Architecture, instr_hex: str, out_base: int, nbytes: int,
                  in_values: dict[str, int], in_taints: dict[str, int]) -> set[int]:
    py = _cell_wide_taint(arch, instr_hex, out_base, nbytes, in_values, in_taints,
                          PCodeCellEvaluator)
    c = _cell_wide_taint(arch, instr_hex, out_base, nbytes, in_values, in_taints,
                         PCodeCellEvaluatorC)
    assert py == c, f'cell.pyx {py} != cell_c {c}'
    return py


def test_cellc_movdqa_matches_pyx() -> None:
    instr = _asm(_KS_X86, 'movdqa xmm0, xmm1').hex()
    _, t = _lanes_from_wide(0, 0xFF | (0xFF << (15 * 8)), _XMM1, 16)
    assert _both_kernels(Architecture.AMD64, instr, _XMM0, 16, {}, t) == {0, 15}


def test_cellc_pxor_matches_pyx() -> None:
    instr = _asm(_KS_X86, 'pxor xmm0, xmm1').hex()
    _, ta = _lanes_from_wide(0, 0x00000000FFFFFFFF, _XMM0, 16)
    _, tb = _lanes_from_wide(0, 0xFFFFFFFF00000000, _XMM1, 16)
    assert _both_kernels(Architecture.AMD64, instr, _XMM0, 16, {}, {**ta, **tb}) == set(range(8))


def test_cellc_vmovdqa_ymm_matches_pyx() -> None:
    instr = _asm(_KS_X86, 'vmovdqa ymm0, ymm1').hex()
    _, t = _lanes_from_wide(0, 0xFF | (0xFF << (16 * 8)) | (0xFF << (31 * 8)), _YMM1, 32)
    assert _both_kernels(Architecture.AMD64, instr, _YMM0, 32, {}, t) == {0, 16, 31}


def test_cellc_vpxor_ymm_matches_pyx() -> None:
    instr = _asm(_KS_X86, 'vpxor ymm0, ymm1, ymm2').hex()
    _, ta = _lanes_from_wide(0, 0xFF, _YMM1, 32)
    _, tb = _lanes_from_wide(0, 0xFF << (31 * 8), _YMM2, 32)
    assert _both_kernels(Architecture.AMD64, instr, _YMM0, 32, {}, {**ta, **tb}) == {0, 31}


def test_cellc_ppc_vxor_be_matches_pyx() -> None:
    instr = _asm(_KS_PPC, 'vxor 0, 1, 2').hex()
    _, t = _lanes_from_wide(0, 0xFF | (0xFF << (15 * 8)), _VR1, 16)
    assert _both_kernels(Architecture.PPC32BE, instr, _VR0, 16, {}, t) == {0, 15}
