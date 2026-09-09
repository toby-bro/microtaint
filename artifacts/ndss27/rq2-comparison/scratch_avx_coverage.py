#!/usr/bin/env python3
"""
scratch_avx_coverage.py  —  SCRATCH validation harness (does NOT touch benchmark.py)

Goal
----
Probe wider-SSE and AVX/AVX2 integer op coverage for MicroTaint's native
P-code simulator, scored against a noninterference ground truth computed on
Unicorn.  MicroTaint ALONE (no cross-engine comparison — the other engines
don't uniformly support AVX).

For each candidate op we build a short movq-marshalled sequence:

    movq xmm0, rax        ; load tainted data into vector low lane
    movq xmm1, rbx        ; load second operand
    <OP under test>
    movq rax, xmm0        ; marshal result low-64 back to a GP reg GT can read

movq is native in MicroTaint (verified: a pure movq/paddb sequence reports
fallback_calls delta == 0), so any fallback_calls delta for the whole
sequence is cleanly attributable to <OP under test>.

Per op we report:
  assembles?            keystone .asm() succeeds
  runs on unicorn?      one concrete Unicorn emulation completes (needed for GT)
  microtaint native?    fallback_calls delta == 0  (else Unicorn fallback)
  sound?                microtaint_taint  ⊇  GT_taint  for every scenario
  exact / over-taint    per-bit comparison microtaint vs GT

Ground truth (noninterference, per-bit, on RAX):
  * exhaustive: enumerate all 2**k assignments of the k tainted input bits,
    a bit of the output is tainted iff it varies across the enumeration.
    Used when k <= GT_BIT_BUDGET (15).
  * single-bit-flip lower bound: flip each tainted bit alone from a base
    state, union the output bits that change.  A SUBSET of the true taint set
    (sound floor) — used to sanity check when k > budget.

Run with:  ./.venv_microtaint/bin/python scratch_avx_coverage.py
(.venv_microtaint has keystone + unicorn + microtaint together.)
"""

from __future__ import annotations

import itertools

import keystone
import unicorn
import unicorn.x86_const as ux
from keystone import KS_ARCH_X86, KS_MODE_64, Ks

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, Register

MASK64 = (1 << 64) - 1
GT_BIT_BUDGET = 15

KS = Ks(KS_ARCH_X86, KS_MODE_64)

# Same register model the benchmark's worker_microtaint.py uses.
_REGS = (
    [Register('RAX', 64), Register('RBX', 64), Register('RCX', 64), Register('RDX', 64)]
    + [Register('RSI', 64), Register('RDI', 64), Register('RSP', 64), Register('RBP', 64)]
    + [Register(f'R{n}', 64) for n in range(8, 16)]
    + [Register(f'XMM{n}_LO', 64) for n in range(8)]
    + [Register(f'XMM{n}_HI', 64) for n in range(8)]
)
GP4 = ('RAX', 'RBX', 'RCX', 'RDX')

SIM = CellSimulator(Architecture.AMD64, use_unicorn=False, use_c=True)
_DEFAULT_RSP = 0x80000000


# ---------------------------------------------------------------------------
# assembling
# ---------------------------------------------------------------------------
def assemble(lines: list[str]) -> bytes:
    out: list[int] = []
    for ln in lines:
        enc, _ = KS.asm(ln)
        out.extend(enc)
    return bytes(out)


# ---------------------------------------------------------------------------
# MicroTaint
# ---------------------------------------------------------------------------
def microtaint_eval(bs: bytes, state: dict[str, int], taint: dict[str, int]):
    """Return (output_taint over GP4, fallback_delta)."""
    circ = generate_static_rule(Architecture.AMD64, bs, _REGS)
    st = {r.name: state.get(r.name, 0) for r in _REGS}
    if st.get('RSP', 0) == 0:
        st['RSP'] = _DEFAULT_RSP
    tt = {r.name: taint.get(r.name, 0) for r in _REGS}
    ctx = EvalContext(input_values=st, input_taint=tt, simulator=SIM)
    fb0 = SIM._pcode.fallback_calls
    raw = circ.evaluate(ctx)
    fb1 = SIM._pcode.fallback_calls
    ot = {
        r: (v & MASK64 if isinstance(v, int) else 0)
        for r, v in raw.items()
        if r in GP4
    }
    return {r: ot.get(r, 0) for r in GP4}, (fb1 - fb0)


# ---------------------------------------------------------------------------
# Unicorn ground-truth engine (self-contained; mirrors benchmark GroundTruthSimulator)
# ---------------------------------------------------------------------------
_CODE_BASE = 0x1000
_STACK_BASE = 0x100000
_REG_MAP = {
    'RAX': ux.UC_X86_REG_RAX,
    'RBX': ux.UC_X86_REG_RBX,
    'RCX': ux.UC_X86_REG_RCX,
    'RDX': ux.UC_X86_REG_RDX,
}


_UC = None
_UC_LAST_BS: bytes | None = None
_STACK_ZERO = b'\x00' * 0x10000


def _reset_uc_for_case():
    """Fresh Uc per bytestring (matches benchmark GroundTruthSimulator)."""
    global _UC, _UC_LAST_BS
    _UC = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
    _UC.mem_map(_CODE_BASE, 0x10000)
    _UC.mem_map(_STACK_BASE, 0x10000)
    _UC_LAST_BS = None


def unicorn_run(bs: bytes, vals: dict[str, int]) -> dict[str, int] | None:
    """Single concrete run on the cached Uc; None if it traps."""
    global _UC_LAST_BS
    uc = _UC
    if bs != _UC_LAST_BS:
        uc.mem_write(_CODE_BASE, bs)
        _UC_LAST_BS = bs
    for i in range(16):
        try:
            uc.reg_write(getattr(ux, f'UC_X86_REG_XMM{i}'), 0)
        except Exception:
            pass
    uc.mem_write(_STACK_BASE, _STACK_ZERO)
    uc.reg_write(ux.UC_X86_REG_RSP, _STACK_BASE + 0x8000)
    uc.reg_write(ux.UC_X86_REG_EFLAGS, 0x2)
    for r, m in _REG_MAP.items():
        uc.reg_write(m, vals.get(r, 0) & MASK64)
    try:
        uc.emu_start(_CODE_BASE, _CODE_BASE + len(bs), timeout=10_000, count=1024)
    except unicorn.UcError:
        return None
    return {r: uc.reg_read(m) for r, m in _REG_MAP.items()}


def unicorn_ok(bs: bytes) -> bool:
    _reset_uc_for_case()
    return unicorn_run(bs, {'RAX': 1, 'RBX': 2, 'RCX': 3, 'RDX': 4}) is not None


def gt_exhaustive(bs: bytes, state: dict[str, int], taint: dict[str, int]):
    """Exact noninterference taint over GP4, or None if k>budget / all traps."""
    positions = [(r, b) for r in GP4 for b in range(64) if (taint.get(r, 0) >> b) & 1]
    k = len(positions)
    if k == 0:
        return {r: 0 for r in GP4}
    if k > GT_BIT_BUDGET:
        return None
    _reset_uc_for_case()
    base = {r: state.get(r, 0) & ~taint.get(r, 0) & MASK64 for r in GP4}
    successful = []
    for assignment in range(1 << k):
        vals = dict(base)
        for idx, (r, b) in enumerate(positions):
            if (assignment >> idx) & 1:
                vals[r] = (vals[r] | (1 << b)) & MASK64
        out = unicorn_run(bs, vals)
        if out is not None:
            successful.append(out)
    if not successful:
        return None
    gt = {}
    for r in GP4:
        agg_or, agg_and = 0, MASK64
        for o in successful:
            agg_or |= o[r]
            agg_and &= o[r]
        gt[r] = (agg_or ^ agg_and) & MASK64
    return gt


def gt_single_bit_flip(bs: bytes, state: dict[str, int], taint: dict[str, int]):
    """Sound lower bound: union of output bits that flip when each tainted
    input bit is toggled alone from the base (tainted-bits-zeroed) state."""
    positions = [(r, b) for r in GP4 for b in range(64) if (taint.get(r, 0) >> b) & 1]
    _reset_uc_for_case()
    base = {r: state.get(r, 0) & ~taint.get(r, 0) & MASK64 for r in GP4}
    ref = unicorn_run(bs, base)
    if ref is None:
        return None
    lb = {r: 0 for r in GP4}
    for (r, b) in positions:
        vals = dict(base)
        vals[r] = (vals[r] | (1 << b)) & MASK64
        out = unicorn_run(bs, vals)
        if out is None:
            continue
        for rr in GP4:
            lb[rr] |= (out[rr] ^ ref[rr]) & MASK64
    return lb


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------
# Distinct-byte operands so byte-shuffle / compare ops are meaningful.
# Low lane bytes: RAX = 07 06 05 04 03 02 01 00 (byte i == i)  [big->little]
RAX_VAL = 0x0706050403020100
# RBX low bytes 00..07 too => pshufb identity control for low 8 indices
RBX_VAL = 0x0706050403020100

SCENARIOS = [
    # (name, taint dict, description).  k<=15 => EXACT GT; k>15 => single-bit-flip LB.
    ('taint_byte0_rax', {'RAX': 0x00000000000000FF}, 'low byte of RAX (k=8, exhaustive)'),
    ('taint_bit0_rax', {'RAX': 0x1}, 'single bit of RAX (k=1, exhaustive)'),
    # dual-operand, WITHIN budget => exact GT.  Stresses the 2-point
    # differential's additive-carry / control-mask cancellation blind spot.
    ('taint_lo6_both', {'RAX': 0x3F, 'RBX': 0x3F}, 'low 6 bits of RAX+RBX (k=12, exhaustive)'),
    ('taint_lo7_both', {'RAX': 0x7F, 'RBX': 0x7F}, 'low 7 bits of RAX+RBX (k=14, exhaustive)'),
    # heavy dual-operand, BEYOND budget => single-bit-flip lower bound only.
    ('taint_lo16_both', {'RAX': 0xFFFF, 'RBX': 0xFFFF}, 'low 16 bits of RAX+RBX (k=32, LB)'),
]


# ---------------------------------------------------------------------------
# Op catalogue.  Each entry: (label, category, width, [op lines])
# marshalling is movq-only (native) so fallback delta attributes to the op.
# ---------------------------------------------------------------------------
def seq_bin(op: str) -> list[str]:
    """binary vector op: load rax->xmm0, rbx->xmm1, op, marshal xmm0->rax."""
    return ['movq xmm0, rax', 'movq xmm1, rbx', op, 'movq rax, xmm0']


def seq_bin_vex(op: str) -> list[str]:
    """3-operand AVX binary op writing xmm0."""
    return ['movq xmm0, rax', 'movq xmm1, rbx', op, 'movq rax, xmm0']


def seq_shift(op: str) -> list[str]:
    return ['movq xmm0, rax', op, 'movq rax, xmm0']


def seq_shift_vex(op: str) -> list[str]:
    return ['movq xmm0, rax', op, 'movq rax, xmm0']


def seq_mov(mov: str) -> list[str]:
    """marshalling move: rax->xmm0, mov xmm2<-xmm0, op result, rax<-xmm2."""
    return ['movq xmm0, rax', mov, 'movq rax, xmm2']


def seq_mov_mem(store: str, load: str) -> list[str]:
    """xmm through memory: rax->xmm0, store [rsp-32], load xmm2, rax<-xmm2."""
    return ['movq xmm0, rax', store, load, 'movq rax, xmm2']


CATALOGUE: list[tuple[str, str, str, list[str]]] = [
    # ---- SSE2 baseline subset already in benchmark (native reference) ----
    ('paddb', 'simd_sse_ref', 'xmm', seq_bin('paddb xmm0, xmm1')),
    ('paddd', 'simd_sse_ref', 'xmm', seq_bin('paddd xmm0, xmm1')),
    ('paddq', 'simd_sse_ref', 'xmm', seq_bin('paddq xmm0, xmm1')),
    ('psubb', 'simd_sse_ref', 'xmm', seq_bin('psubb xmm0, xmm1')),
    ('pand', 'simd_sse_ref', 'xmm', seq_bin('pand xmm0, xmm1')),
    ('por', 'simd_sse_ref', 'xmm', seq_bin('por xmm0, xmm1')),
    ('pxor', 'simd_sse_ref', 'xmm', seq_bin('pxor xmm0, xmm1')),
    ('pcmpeqb', 'simd_sse_ref', 'xmm', seq_bin('pcmpeqb xmm0, xmm1')),
    ('pshufb', 'simd_sse_ref', 'xmm', seq_bin('pshufb xmm0, xmm1')),
    ('psllq_imm', 'simd_sse_ref', 'xmm', seq_shift('psllq xmm0, 4')),
    ('psrlq_imm', 'simd_sse_ref', 'xmm', seq_shift('psrlq xmm0, 8')),

    # ---- wider SSE integer ops NOT currently in the benchmark subset ----
    ('paddw', 'simd_sse_wide', 'xmm', seq_bin('paddw xmm0, xmm1')),
    ('psubw', 'simd_sse_wide', 'xmm', seq_bin('psubw xmm0, xmm1')),
    ('psubd', 'simd_sse_wide', 'xmm', seq_bin('psubd xmm0, xmm1')),
    ('psubq', 'simd_sse_wide', 'xmm', seq_bin('psubq xmm0, xmm1')),
    ('pcmpeqd', 'simd_sse_wide', 'xmm', seq_bin('pcmpeqd xmm0, xmm1')),
    ('pcmpgtb', 'simd_sse_wide', 'xmm', seq_bin('pcmpgtb xmm0, xmm1')),
    ('pmaxub', 'simd_sse_wide', 'xmm', seq_bin('pmaxub xmm0, xmm1')),
    ('pminub', 'simd_sse_wide', 'xmm', seq_bin('pminub xmm0, xmm1')),
    ('punpcklbw', 'simd_sse_wide', 'xmm', seq_bin('punpcklbw xmm0, xmm1')),
    ('psadbw', 'simd_sse_wide', 'xmm', seq_bin('psadbw xmm0, xmm1')),
    ('pslld_imm', 'simd_sse_wide', 'xmm', seq_shift('pslld xmm0, 3')),
    ('pslldq_imm', 'simd_sse_wide', 'xmm', seq_shift('pslldq xmm0, 2')),
    ('psrldq_imm', 'simd_sse_wide', 'xmm', seq_shift('psrldq xmm0, 2')),
    ('psllq_reg', 'simd_sse_wide', 'xmm', seq_bin('psllq xmm0, xmm1')),

    # ---- AVX (VEX.128) integer ops on xmm ----
    ('vpaddb_x', 'simd_avx128', 'xmm', seq_bin_vex('vpaddb xmm0, xmm0, xmm1')),
    ('vpaddd_x', 'simd_avx128', 'xmm', seq_bin_vex('vpaddd xmm0, xmm0, xmm1')),
    ('vpaddq_x', 'simd_avx128', 'xmm', seq_bin_vex('vpaddq xmm0, xmm0, xmm1')),
    ('vpand_x', 'simd_avx128', 'xmm', seq_bin_vex('vpand xmm0, xmm0, xmm1')),
    ('vpor_x', 'simd_avx128', 'xmm', seq_bin_vex('vpor xmm0, xmm0, xmm1')),
    ('vpxor_x', 'simd_avx128', 'xmm', seq_bin_vex('vpxor xmm0, xmm0, xmm1')),
    ('vpshufb_x', 'simd_avx128', 'xmm', seq_bin_vex('vpshufb xmm0, xmm0, xmm1')),
    ('vpcmpeqb_x', 'simd_avx128', 'xmm', seq_bin_vex('vpcmpeqb xmm0, xmm0, xmm1')),
    ('vpsllq_x_imm', 'simd_avx128', 'xmm', seq_shift_vex('vpsllq xmm0, xmm0, 4')),
    ('vpsrlq_x_imm', 'simd_avx128', 'xmm', seq_shift_vex('vpsrlq xmm0, xmm0, 8')),

    # ---- AVX2 (VEX.256) integer ops on ymm ----
    ('vpaddb_y', 'simd_avx256', 'ymm', seq_bin_vex('vpaddb ymm0, ymm0, ymm1')),
    ('vpaddd_y', 'simd_avx256', 'ymm', seq_bin_vex('vpaddd ymm0, ymm0, ymm1')),
    ('vpaddq_y', 'simd_avx256', 'ymm', seq_bin_vex('vpaddq ymm0, ymm0, ymm1')),
    ('vpand_y', 'simd_avx256', 'ymm', seq_bin_vex('vpand ymm0, ymm0, ymm1')),
    ('vpor_y', 'simd_avx256', 'ymm', seq_bin_vex('vpor ymm0, ymm0, ymm1')),
    ('vpxor_y', 'simd_avx256', 'ymm', seq_bin_vex('vpxor ymm0, ymm0, ymm1')),
    ('vpshufb_y', 'simd_avx256', 'ymm', seq_bin_vex('vpshufb ymm0, ymm0, ymm1')),
    ('vpcmpeqb_y', 'simd_avx256', 'ymm', seq_bin_vex('vpcmpeqb ymm0, ymm0, ymm1')),
    ('vpsllq_y_imm', 'simd_avx256', 'ymm', seq_shift_vex('vpsllq ymm0, ymm0, 4')),
    ('vpsrlq_y_imm', 'simd_avx256', 'ymm', seq_shift_vex('vpsrlq ymm0, ymm0, 8')),

    # ---- wider MOVDQA/MOVDQU marshalling (reg-reg and through memory) ----
    ('movdqa_rr', 'simd_mov', 'xmm', seq_mov('movdqa xmm2, xmm0')),
    ('movdqu_rr', 'simd_mov', 'xmm', seq_mov('movdqu xmm2, xmm0')),
    ('movaps_rr', 'simd_mov', 'xmm', seq_mov('movaps xmm2, xmm0')),
    ('movdqa_mem', 'simd_mov', 'xmm',
     seq_mov_mem('movdqa [rsp - 32], xmm0', 'movdqa xmm2, [rsp - 32]')),
    ('movdqu_mem', 'simd_mov', 'xmm',
     seq_mov_mem('movdqu [rsp - 32], xmm0', 'movdqu xmm2, [rsp - 32]')),
    ('vmovdqa_rr', 'simd_mov', 'xmm', seq_mov('vmovdqa xmm2, xmm0')),
    ('vmovdqu_rr', 'simd_mov', 'xmm', seq_mov('vmovdqu xmm2, xmm0')),
    ('vmovdqu_mem_x', 'simd_mov', 'xmm',
     seq_mov_mem('vmovdqu [rsp - 32], xmm0', 'vmovdqu xmm2, [rsp - 32]')),
    ('vmovdqu_mem_y', 'simd_mov', 'ymm',
     seq_mov_mem('vmovdqu [rsp - 64], ymm0', 'vmovdqu ymm2, [rsp - 64]')),
]


def bitcount(x: int) -> int:
    return bin(x & MASK64).count('1')


def analyse():
    base_state = {'RAX': RAX_VAL, 'RBX': RBX_VAL, 'RCX': 0, 'RDX': 0}
    rows = []
    for label, cat, width, lines in CATALOGUE:
        asm_str = '; '.join(lines)
        row = {'label': label, 'cat': cat, 'width': width, 'asm': asm_str}
        # (a) assemble
        try:
            bs = assemble(lines)
            row['assembles'] = True
            row['bytes'] = bs.hex()
        except Exception as e:
            row['assembles'] = False
            row['err'] = f'asm: {e}'
            rows.append(row)
            continue
        # (b) runs on unicorn?
        row['unicorn'] = unicorn_ok(bs)
        # (c) microtaint native / fallback + soundness + exactness
        native = True
        fb_total = 0
        sound_exact = True   # authoritative: soundness under EXACT (<=budget) GT
        sound_lb = True      # informational: floor from single-bit-flip LB
        exact = True
        details = []
        mt_ran = True
        for sname, staint, sdesc in SCENARIOS:
            try:
                mt, fbdelta = microtaint_eval(bs, base_state, staint)
            except Exception as e:
                mt_ran = False
                details.append(f'{sname}: MT-error {type(e).__name__}')
                continue
            if fbdelta > 0:
                native = False
                fb_total += fbdelta
            # ground truth
            k = sum(bitcount(staint.get(r, 0)) for r in GP4)
            if k <= GT_BIT_BUDGET:
                gt = gt_exhaustive(bs, base_state, staint)
                gt_kind = 'exact'
            else:
                gt = gt_single_bit_flip(bs, base_state, staint)
                gt_kind = 'LB'
            if gt is None:
                details.append(f'{sname}: GT-unavailable(unicorn)')
                continue
            for r in GP4:
                miss = gt[r] & ~mt[r] & MASK64  # GT bit not covered => UNSOUND
                over = mt[r] & ~gt[r] & MASK64   # MT bit not in GT   => over-taint
                if miss:
                    exact = False
                    if gt_kind == 'exact':
                        sound_exact = False
                        details.append(f'{sname}/{r}: UNSOUND(exact) miss={miss:#018x}')
                    else:
                        sound_lb = False
                        details.append(f'{sname}/{r}: unsound(LB-floor) miss={miss:#018x}')
                elif over:
                    if gt_kind == 'exact':
                        exact = False
                        details.append(f'{sname}/{r}: over={bitcount(over)}b gt={gt[r]:#x} mt={mt[r]:#x}')
        row['mt_ran'] = mt_ran
        row['native'] = native if mt_ran else None
        row['fb_total'] = fb_total
        row['sound_exact'] = sound_exact if mt_ran else None
        row['sound_lb'] = sound_lb if mt_ran else None
        row['exact'] = exact if mt_ran else None
        row['details'] = details
        rows.append(row)
    return rows


def fmt(v):
    if v is True:
        return 'yes'
    if v is False:
        return 'no'
    if v is None:
        return '-'
    return str(v)


def main():
    rows = analyse()
    print('\n' + '=' * 120)
    print('MICROTAINT WIDER-SSE / AVX COVERAGE  (scored vs Unicorn noninterference GT; microtaint-only)')
    print('=' * 120)
    hdr = (f'{"op":14} {"category":16} {"w":4} {"asm?":5} {"uni?":5} '
           f'{"mt-path":10} {"sound*":7} {"exact?":7}')
    print(hdr)
    print('  (mt-path: native | fallback=N differential Unicorn evals;  '
          'sound* = under exact GT (k<=15), the only regime the benchmark scores)')
    print('-' * 120)
    for r in rows:
        if not r.get('assembles'):
            print(f'{r["label"]:14} {r["cat"]:16} {r["width"]:4} {"NO":5}  -> {r.get("err","")}')
            continue
        if r.get('native') is None:
            path = '-'
        elif r.get('native'):
            path = 'native'
        else:
            path = f'fallback:{r.get("fb_total")}'
        print(f'{r["label"]:14} {r["cat"]:16} {r["width"]:4} '
              f'{fmt(r["assembles"]):5} {fmt(r["unicorn"]):5} {path:10} '
              f'{fmt(r.get("sound_exact")):7} {fmt(r["exact"]):7}')
        for d in r.get('details', []):
            print(f'    · {d}')
    print('-' * 120)

    def cnt(pred):
        return sum(1 for r in rows if r.get('assembles') and pred(r))
    print(f'total ops                    : {len(rows)}')
    print(f'assemble ok                  : {cnt(lambda r: True)}')
    print(f'run on unicorn (GT possible) : {cnt(lambda r: r.get("unicorn"))}')
    print(f'microtaint native            : {cnt(lambda r: r.get("native") is True)}')
    print(f'microtaint fallback->unicorn : {cnt(lambda r: r.get("native") is False)}')
    print(f'sound under exact GT         : {cnt(lambda r: r.get("sound_exact") is True)}')
    print(f'UNSOUND under exact GT       : {cnt(lambda r: r.get("sound_exact") is False)}')
    print(f'bit-exact (all exact scen.)  : {cnt(lambda r: r.get("exact") is True)}')
    print(f'LB-floor violations (k=32)   : {cnt(lambda r: r.get("sound_lb") is False)}  '
          '(2-point differential cancellation; not scored by benchmark, but a real precision cliff)')
    print('=' * 120)


if __name__ == '__main__':
    main()
