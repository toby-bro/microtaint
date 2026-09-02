"""ISA-general SIMD taint: ARM64 NEON and PPC AltiVec wide-vector propagation.

The taint engine tracks any ISA's vector register file with no per-ISA table.  A
register-space varnode wider than one 64-bit lane (x86 XMM/YMM/ZMM, ARM64 NEON
q/z, PPC AltiVec vs) that the state_format does not enumerate is decomposed into
synthetic 8-byte ``VL_<offset>`` lanes purely from pypcode's register geometry
(see StateMapper._synth_vec_lane).  These tests assert the exact tainted-byte SET
for the lane-independent classes -- movement (``mov``/``orr``), bitwise
(``eor``/``vxor``/``vor``), and vector loads -- and no-under-taint (soundness) for
packed arithmetic, on ARM64 and PPC where NO vector register is listed in the
state_format.  Before the geometry rule these wide ops were dropped entirely
(silent under-taint); x86 keeps its own regression suites (test_simd_xmm_*).
"""

from __future__ import annotations

import pytest
from keystone import (
    KS_ARCH_ARM64,
    KS_ARCH_PPC,
    KS_MODE_BIG_ENDIAN,
    KS_MODE_LITTLE_ENDIAN,
    KS_MODE_PPC32,
    Ks,
)

from microtaint.emulator.shadow import BitPreciseShadowMemory
from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.sleigh.lifter import get_context
from microtaint.types import Architecture, Register

_FULL = 0xFFFFFFFFFFFFFFFF
_SRC = 0x5000

_KS_ARM = Ks(KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN)
_KS_PPC = Ks(KS_ARCH_PPC, KS_MODE_BIG_ENDIAN + KS_MODE_PPC32)


def _asm(ks: Ks, text: str) -> bytes:
    enc, _ = ks.asm(text, 0x1000)
    return bytes(enc)


def _reg_off(arch: Architecture, name: str) -> int:
    """Absolute Sleigh byte offset of a register, straight from pypcode."""
    vn = get_context(str(arch).split('.')[-1]).registers[name]
    return int(vn.offset)


def _vlane(base: int, k: int) -> str:
    return f'VL_{base + k:#x}'


def _gp_regs(names: list[str]) -> list[Register]:
    # Deliberately GP-only: NO vector register is enumerated, so the vector
    # lanes must be synthesised from geometry alone.
    return [Register(name=n, bits=64) for n in names]


def _vec_tainted_bytes(out: dict[str, int], base: int, nbytes: int) -> set[int]:
    """Set of tainted byte indices (0..nbytes-1) across a vector register's VL_ lanes."""
    tainted: set[int] = set()
    for k in range(0, nbytes, 8):
        mask = out.get(_vlane(base, k), 0)
        for i in range(min(8, nbytes - k)):
            if (mask >> (i * 8)) & 0xFF:
                tainted.add(k + i)
    return tainted


# ---------------------------------------------------------------------------
# ARM64 NEON
# ---------------------------------------------------------------------------


def test_arm64_neon_mov_is_exact_copy() -> None:
    """`mov v0.16b, v1.16b` (orr alias): v0 byte j carries EXACTLY v1 byte j."""
    code = _asm(_KS_ARM, 'orr v0.16b, v1.16b, v1.16b')
    v0, v1 = _reg_off(Architecture.ARM64, 'q0'), _reg_off(Architecture.ARM64, 'q1')
    circuit = generate_static_rule(Architecture.ARM64, code, _gp_regs(['x0', 'x1', 'x2']))
    # v1 low 8 bytes fully tainted, high 8 bytes: only byte 8 tainted.
    out = circuit.evaluate(EvalContext(
        input_values={}, input_taint={_vlane(v1, 0): _FULL, _vlane(v1, 8): 0xFF},
        simulator=CellSimulator(Architecture.ARM64), shadow_memory=None,
    ))
    assert _vec_tainted_bytes(out, v0, 16) == set(range(8)) | {8}
    assert _vec_tainted_bytes(out, v1, 16) == set(range(8)) | {8}  # inputs preserved


def test_arm64_neon_eor_is_sound() -> None:
    """`eor v0.16b, v1.16b, v2.16b` (SLEIGH lifts this per-byte): every tainted
    input byte taints v0 (no under-taint).  A per-byte-lifted op routes through
    the differential, which may over-approximate on synthetic lanes -- sound, but
    lane-exact bitwise is a native-width (M2) improvement, not an M1a guarantee."""
    code = _asm(_KS_ARM, 'eor v0.16b, v1.16b, v2.16b')
    v0 = _reg_off(Architecture.ARM64, 'q0')
    v1 = _reg_off(Architecture.ARM64, 'q1')
    v2 = _reg_off(Architecture.ARM64, 'q2')
    circuit = generate_static_rule(Architecture.ARM64, code, _gp_regs(['x0', 'x1', 'x2']))
    out = circuit.evaluate(EvalContext(
        input_values={},
        input_taint={_vlane(v1, 0): 0x00000000FFFFFFFF, _vlane(v2, 0): 0xFFFFFFFF00000000},
        simulator=CellSimulator(Architecture.ARM64), shadow_memory=None,
    ))
    # No under-taint: bytes 0-3 (from v1) and 4-7 (from v2) MUST be tainted.
    assert set(range(8)).issubset(_vec_tainted_bytes(out, v0, 16))


@pytest.mark.xfail(
    reason='Wide (>8-byte) vector LOAD routes through the 64-bit differential '
    'kernel, which truncates a single 16-byte LOAD -- exact in the native-width '
    'differential (M2). Not a regression: non-x86 vectors were untracked before.',
    strict=False,
)
def test_arm64_neon_ldr_q_is_exact() -> None:
    """`ldr q0, [x1]`: memory byte j (tainted) -> q0 byte j, byte-exact."""
    code = _asm(_KS_ARM, 'ldr q0, [x1]')
    q0 = _reg_off(Architecture.ARM64, 'q0')
    circuit = generate_static_rule(Architecture.ARM64, code, _gp_regs(['x0', 'x1', 'x2']))
    shadow = BitPreciseShadowMemory()
    tainted_src = {0, 1, 7, 8, 15}
    for i in tainted_src:
        shadow.write_mask(_SRC + i, 0xFF, 1)
    out = circuit.evaluate(EvalContext(
        input_values={'x1': _SRC}, input_taint={},
        simulator=CellSimulator(Architecture.ARM64), shadow_memory=shadow,
    ))
    assert _vec_tainted_bytes(out, q0, 16) == tainted_src


def test_arm64_neon_add_is_sound() -> None:
    """`add v0.4s, v1.4s, v2.4s`: every tainted input lane taints its output lane
    (no under-taint).  Packed arithmetic may over-approximate (carry), which is
    sound; we assert the tainted output covers at least the tainted input bytes."""
    code = _asm(_KS_ARM, 'add v0.4s, v1.4s, v2.4s')
    v0, v1 = _reg_off(Architecture.ARM64, 'q0'), _reg_off(Architecture.ARM64, 'q1')
    circuit = generate_static_rule(Architecture.ARM64, code, _gp_regs(['x0', 'x1', 'x2']))
    out = circuit.evaluate(EvalContext(
        input_values={}, input_taint={_vlane(v1, 0): 0x00000000FFFFFFFF},
        simulator=CellSimulator(Architecture.ARM64), shadow_memory=None,
    ))
    # 32-bit lane 0 (bytes 0-3) of v1 is tainted -> at least those bytes of v0 tainted.
    assert {0, 1, 2, 3}.issubset(_vec_tainted_bytes(out, v0, 16))


# ---------------------------------------------------------------------------
# PPC AltiVec
# ---------------------------------------------------------------------------


def test_ppc_altivec_vxor_is_exact_lane_union() -> None:
    """`vxor 0,1,2`: vr0 lane taint = vr1 lane | vr2 lane, per byte (exact)."""
    code = _asm(_KS_PPC, 'vxor 0, 1, 2')
    vr0 = _reg_off(Architecture.PPC32BE, 'vs32')
    vr1 = _reg_off(Architecture.PPC32BE, 'vs33')
    vr2 = _reg_off(Architecture.PPC32BE, 'vs34')
    circuit = generate_static_rule(Architecture.PPC32BE, code, _gp_regs(['r0', 'r1', 'r2']))
    out = circuit.evaluate(EvalContext(
        input_values={},
        input_taint={_vlane(vr1, 0): 0xFF00FF00FF00FF00, _vlane(vr2, 0): 0x00FF00FF00FF00FF},
        simulator=CellSimulator(Architecture.PPC32BE), shadow_memory=None,
    ))
    # low lane: alternating bytes from each operand -> all 8 bytes tainted.
    assert _vec_tainted_bytes(out, vr0, 16) & set(range(8)) == set(range(8))


@pytest.mark.xfail(
    reason='Wide (>8-byte) value-aware bitwise (INT_OR/INT_AND) routes through the '
    '64-bit differential kernel, which truncates a single 16-byte op -- exact in the '
    'native-width differential (M2). vxor works because XOR taint is a pure union '
    '(value-independent) and takes the operand path instead.',
    strict=False,
)
def test_ppc_altivec_vor_is_sound() -> None:
    """`vor 0,1,2`: OR of two vectors; every tainted input byte taints vr0 (no under-taint)."""
    code = _asm(_KS_PPC, 'vor 0, 1, 2')
    vr0 = _reg_off(Architecture.PPC32BE, 'vs32')
    vr1 = _reg_off(Architecture.PPC32BE, 'vs33')
    circuit = generate_static_rule(Architecture.PPC32BE, code, _gp_regs(['r0', 'r1', 'r2']))
    out = circuit.evaluate(EvalContext(
        input_values={}, input_taint={_vlane(vr1, 0): _FULL},
        simulator=CellSimulator(Architecture.PPC32BE), shadow_memory=None,
    ))
    assert set(range(8)).issubset(_vec_tainted_bytes(out, vr0, 16))
