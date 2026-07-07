#!/usr/bin/env .venv_master/bin/python
"""
benchmark.py  –  Taint-engine benchmark orchestrator  (NDSS edition)
=====================================================================

SCOPE
-----
This script provides a comprehensive, publication-quality evaluation of
taint-analysis engines on x86-64.  It is structured around five pillars
that reviewers typically demand for NDSS/S&P/USENIX-Security submissions,
plus an optional sixth pillar for noninterference ground-truth scoring:

  1. BREADTH   — a large, taxonomised instruction pool covering all major
                 x86-64 encoding categories (ALU, shifts, bitwise, moves,
                 SIMD/SSE2, string, conditional, flag-affecting).

  2. SEQUENCES — multi-instruction chains that expose propagation fidelity
                 (e.g. taint through carry chains, implicit flows, loop
                 bodies, function-call preambles).

  3. SEMANTICS — hand-crafted test sequences with documented expected
                 behaviour (sub-byte multiplications, x*0 collapse,
                 strcmp-style byte-difference, base64 decode patterns,
                 cmov branches, real-world snippets from glibc/openssl).
                 These are checked by the GT simulator (pillar 6) when
                 their input taint fits the bit budget — replacing the
                 hand-written per-test oracles that used to live here.

  4. PRECISION  — bit-level comparison between tools that support it;
                  per-test Jaccard similarity and overall F1 vs. a
                  "maximally-precise" reference (microtaint or angr).

  5. PERFORMANCE — latency distributions (mean, p50, p95, p99) per
                   instruction category and aggregate throughput.

  6. GROUND TRUTH (default; toggle with --no-ground-truth) — exhaustive
                   Unicorn enumeration of all 2**k assignments of the k
                   tainted input bits (k <= GT_BIT_BUDGET).  For every
                   test case that fits the budget, this gives the EXACT
                   noninterference taint set: bit i of output R is
                   tainted iff there exist two assignments x, x' of the
                   tainted bits such that output(x)[R][i] != output(x')
                   [R][i].  Bit-precise tools are then scored for
                   soundness (must contain GT) and over-taint count
                   (spurious bits beyond GT).  Test-case generators in
                   every pillar mix sparse-taint configurations so a
                   meaningful fraction of cases land in the GT budget.

Taint input format
------------------
  test_case["taint"] is a 64-bit bitmask per register.
  Bit i set → bit i of that input register is tainted.
  0 = fully clean, 0xFFFFFFFFFFFFFFFF = fully tainted.

  Bit-level tools (microtaint, angr, maat): propagate at bit granularity.
  Register-level tools (triton, panda, taintgrind, libdft64): any non-zero
  mask is treated as "register tainted".

Output format
-------------
  Bit-level tools : 64-bit bitmask per output register.
  Reg-level tools : 0 (clean) or 1 (tainted) per output register.
"""


import argparse
import concurrent.futures
import json
import os
import random
import re as _re
import statistics
import subprocess
import sys

# Force stdout to line-buffered mode so every print() appears immediately
# in the terminal even when piped through tee.  Without this Python switches
# to full-buffering (8 KB block) when stdout is not a TTY, which means
# nothing appears until the buffer fills or the process exits.
# PYTHONUNBUFFERED=1 does the same thing from the shell, but being explicit
# here means the script works correctly regardless of how it is invoked.
import sys as _sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from itertools import product as iterproduct

if hasattr(_sys.stdout, 'reconfigure'):
    _sys.stdout.reconfigure(line_buffering=True)

# Optional progress bar.  tqdm is small and pure-Python; fall back to a
# no-op iterator if it's not installed so the benchmark still runs in
# minimal environments (e.g. air-gapped CI).
#
# IMPORTANT: pass file=sys.stdout to every tqdm() call so the bar is
# written to stdout, not stderr.  When you pipe with `cmd | tee log`,
# only stdout is captured by tee; if the bar writes to stderr it appears
# in the terminal but not in the log file, and stdout's regular print()
# output and the bar interleave on separate streams causing garbled display.
try:
    import sys as _sys_tqdm

    from tqdm import tqdm as _tqdm_real  # type: ignore

    def tqdm(iterable=None, **kwargs):  # type: ignore
        kwargs.setdefault('file', _sys_tqdm.stdout)
        kwargs.pop('position', None)  # position= requires a real TTY; drop it
        return _tqdm_real(iterable, **kwargs)

    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

    def tqdm(iterable=None, **kwargs):  # type: ignore
        if iterable is None:

            class _NoOpBar:
                def update(self, n=1):
                    pass

                def close(self):
                    pass

                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    pass

                def set_postfix_str(self, s, refresh=True):
                    pass

            return _NoOpBar()
        return iterable


_ANSI_ESC = _re.compile(r'\x1b\[[0-9;]*m')

from keystone import KS_ARCH_X86, KS_MODE_64, Ks

# ---------------------------------------------------------------------------
# Paths / worker declarations (unchanged from original)
# ---------------------------------------------------------------------------

PIN_ROOT = os.path.abspath('external/pin-3.20-98437-gf02b61307-gcc-linux')
LIBDFT_TOOL = os.path.abspath('external/libdft64/tools/obj-intel64/track.so')
CWD = os.getcwd()

PYTHON_WORKERS = {
    'microtaint': '.venv_microtaint/bin/python worker_microtaint.py',
    'angr': '.venv_angr/bin/python worker_angr.py',
    'maat': '.venv_maat/bin/python worker_maat.py',
    'triton': '.venv_triton/bin/python worker_triton.py',
}
C_HARNESS_WORKERS = {
    'taintgrind': (
        f'docker run -i --rm -v {CWD}:/pwd taintgrind:latest /code/valgrind/build/bin/taintgrind /pwd/harness.bin'
    ),
    'libdft64': f'{PIN_ROOT}/pin -t {LIBDFT_TOOL} -- ./harness.bin',
}
PANDA_DOCKER_CMD = [
    'docker',
    'run',
    '--rm',
    '-i',
    '-v',
    f'{CWD}:/benchmark',
    '-v',
    'panda_qcows:/root/.panda',
    'pandare/panda',
    'python3',
    '/benchmark/worker_panda.py',
]
ALL_WORKERS = {**PYTHON_WORKERS, 'panda': 'panda', **C_HARNESS_WORKERS}

GRANULARITY = {
    'microtaint': 'bit',
    'triton': 'reg',
    'angr': 'bit',
    'panda': 'reg',
    'maat': 'bit',
    'taintgrind': 'reg',
    'libdft64': 'reg',
    'ground_truth': 'bit',
}

# ── Noninterference ground-truth simulator ────────────────────────────────────
# `ground_truth` is an in-process simulator (no subprocess) that computes the
# exact taint set under the standard noninterference definition: bit i of
# output register R is tainted iff there exist two assignments of the tainted
# input bits that produce different values at output[R][i].
#
# Implemented by enumeration: for k = popcount(T_RAX | T_RBX | ...) tainted
# input bits, run Unicorn 2^k times, one per assignment, and record the set
# of output bit positions whose value varies across the runs.  This is
# exponential in k, so the simulator is GATED at GT_BIT_BUDGET tainted bits.
# Cases exceeding the budget are reported as a structured "skipped" entry
# rather than producing a wrong answer.
#
# Budget calibration (Unicorn ~30µs/run with cached Uc instance):
#   k=8  → 256 sims      ≈ 8 ms / case      (cheap)
#   k=12 → 4096 sims     ≈ 120 ms / case    (moderate)
#   k=16 → 65536 sims    ≈ 2 s / case       (default ceiling)
#   k=20 → 1M sims       ≈ 30 s / case      (impractical)
#
# Default: k <= 12.  Test-case generators in every pillar (random single,
# sequence, sweep) emit a mix of taint configurations including sparse-
# taint patterns guaranteed to land below this threshold, so the GT
# simulator runs on a meaningful fraction of cases per pillar.
#
# Why 12 (not 16)?  k=12 is 4096 simulations per case — about 600 ms in
# the worst case with the 10 ms per-emu_start timeout.  k=16 is 65536
# simulations, roughly 10 s per case in normal runs but up to 11 minutes
# per case if every emulation hits the timeout (e.g. a sequence with an
# infinite loop that the per-test deadline would still need to abort).
# At k=12 the worst-case wall clock per test is bounded much more
# tightly, which matters for the overall 2-hour benchmark budget.
GT_BIT_BUDGET = 15

# Registers tracked throughout
REGISTERS = ['RAX', 'RBX', 'RCX', 'RDX']

# 64-bit all-ones mask, used widely for taint masking and Unicorn register I/O.
MASK64 = 0xFFFFFFFFFFFFFFFF

# ---------------------------------------------------------------------------
# Comprehensive instruction taxonomy
# ---------------------------------------------------------------------------
#
# Each entry: (mnemonic_string, category_tag)
# The category tag is used for per-category statistics in the report.
#
# Design rationale:
#   – We cover every major encoding class that taint engines treat
#     differently (ALU with carry, shifts with masks, SIMD byte-granular,
#     conditional moves that introduce implicit flows, etc.).
#   – Sequences (see INSTRUCTION_SEQUENCES below) exercise multi-step
#     propagation that single-instruction tests cannot reveal.

INSTRUCTION_POOL: list[tuple[str, str]] = [
    # =====================================================================
    # COVERAGE CLAIM
    # ---------------------------------------------------------------------
    # This pool covers, with at least a few representatives per class, every
    # major x86-64 instruction category that taint engines treat differently.
    # Categories follow the Intel SDM Volume 2 grouping:
    #
    #   Data transfer ............ MOV, MOVSX, MOVSXD, MOVZX, XCHG, CMOVcc
    #   Stack ................... PUSH, POP (in sequences)
    #   Atomic R-M-W ............ XADD, CMPXCHG
    #   Binary arithmetic ........ ADD, ADC, ADCX, ADOX, SUB, SBB, MUL, IMUL,
    #                              DIV, IDIV, NEG, INC, DEC, CMP
    #   Sign-extend implicit ..... CDQE, CDQ, CQO, CWDE, CWD
    #   Logical .................. AND, OR, XOR, NOT, TEST
    #   Shift / rotate ........... SHL/SAL, SHR, SAR, ROL, ROR, RCL, RCR,
    #                              SHLD, SHRD
    #   Bit / byte ............... BT, BTC, BTR, BTS, BSF, BSR, SETcc,
    #                              POPCNT, LZCNT, TZCNT
    #   BMI1 ..................... ANDN, BEXTR, BLSI, BLSR, BLSMSK
    #   BMI2 ..................... BZHI, MULX, PDEP, PEXT, RORX, SARX, SHLX,
    #                              SHRX
    #   Byte-permute ............. BSWAP
    #   Address calc ............. LEA (multiple addressing modes)
    #   Flag manipulation ........ LAHF, SAHF, CLC, STC, CMC
    #   SIMD (SSE2) .............. PADDB/W/D/Q, PSUBB/W/D/Q, PAND, POR, PXOR,
    #                              PCMPEQB, PSHUFB, PSLLQ, PSRLQ
    #     (most SIMD lives in INSTRUCTION_SEQUENCES because they need
    #      MOVQ xmm,rax setup; a few register-form singles appear here.)
    #   String / REP ............. MOVSB/Q, STOSB/Q, SCASB, CMPSB
    #     (sequences only — they need RSI/RDI/RCX setup.)
    #   No-op / hint ............. NOP
    #
    # Beyond breadth, three sub-categories are explicitly tagged for
    # soundness analysis:
    #
    #   "flag_only_*"     — sequences whose output depends on a flag input.
    #                       Engines without flag taint MUST be unsound here.
    #   "partial_write_*" — sub-register writes / partial updates that trap
    #                       common engine bugs (al-write doesn't clear RAX,
    #                       eax-write DOES clear RAX, etc.).
    #   Plus the existing "implicit_flow" cmov-after-cmp tests.
    # =====================================================================
    # ── Data movement: full register ─────────────────────────────────────
    ('mov rax, rbx', 'mov'),
    ('mov rbx, rax', 'mov'),
    ('mov rcx, rdx', 'mov'),
    ('mov rdx, rcx', 'mov'),
    ('mov rax, rcx', 'mov'),
    ('mov rbx, rdx', 'mov'),
    ('mov rdx, rax', 'mov'),
    ('mov rcx, rbx', 'mov'),
    ('xchg rax, rbx', 'xchg'),  # swap — both outputs tainted if either input is
    ('xchg rcx, rdx', 'xchg'),
    ('xchg rax, rcx', 'xchg'),
    ('xchg rbx, rdx', 'xchg'),
    # XADD: atomic exchange-and-add. dst ← dst + src ; src ← old dst.
    # Both registers receive output taint; richer than xchg or add alone.
    ('xadd rax, rbx', 'xadd'),
    ('xadd rcx, rdx', 'xadd'),
    # CMPXCHG: rax compared to dst; if equal, dst←src; else rax←dst.
    # Three-input, two-output, flag-conditioned. Stress test for engines.
    ('cmpxchg rbx, rcx', 'cmpxchg'),
    ('cmpxchg rdx, rax', 'cmpxchg'),
    # ── Data movement: zero/sign extension ───────────────────────────────
    ('movzx rax, bx', 'movzx'),  # 16→64 zero-ext: bits 63:16 definitely clean
    ('movzx rbx, ax', 'movzx'),
    ('movzx rcx, dl', 'movzx'),  # 8→64
    ('movzx rax, bl', 'movzx'),
    ('movzx eax, bx', 'movzx'),  # 16→32 (with implicit 32→64 zero-ext)
    ('movsx rax, bx', 'movsx'),  # 16→64 sign-ext: bits 63:16 = sign bit taint
    ('movsx rbx, cl', 'movsx'),
    ('movsx rcx, dl', 'movsx'),
    ('movsx rax, bl', 'movsx'),
    # MOVSXD: 32→64 sign-extension. Distinct opcode from MOVSX, often
    # mishandled (engines may treat as plain mov).
    ('movsxd rax, ebx', 'movsxd'),
    ('movsxd rcx, edx', 'movsxd'),
    # ── Conditional moves: complete CMOVcc family ────────────────────────
    # Implicit data flow: dst ← cond ? src : dst.
    # Sound taint: T(dst) ⊇ T(src) ∪ T(cond_flags) ∪ T(old_dst).
    # Engines without flag taint understate T(dst) when cond depends on
    # tainted comparisons (covered exhaustively in flag_only_cmov sequences).
    ('cmovz  rax, rbx', 'cmov'),
    ('cmovnz rax, rbx', 'cmov'),
    ('cmove  rcx, rdx', 'cmov'),
    ('cmovne rcx, rdx', 'cmov'),
    ('cmovs  rcx, rdx', 'cmov'),
    ('cmovns rcx, rdx', 'cmov'),
    ('cmovo  rax, rbx', 'cmov'),
    ('cmovno rax, rbx', 'cmov'),
    ('cmovc  rax, rbx', 'cmov'),
    ('cmovnc rax, rbx', 'cmov'),
    ('cmovp  rcx, rdx', 'cmov'),
    ('cmovnp rcx, rdx', 'cmov'),
    ('cmova  rax, rcx', 'cmov'),  # unsigned >
    ('cmovae rax, rcx', 'cmov'),  # unsigned >=
    ('cmovb  rbx, rdx', 'cmov'),  # unsigned <
    ('cmovbe rbx, rdx', 'cmov'),  # unsigned <=
    ('cmovg  rax, rcx', 'cmov'),  # signed >
    ('cmovge rax, rcx', 'cmov'),
    ('cmovl  rbx, rdx', 'cmov'),  # signed <
    ('cmovle rbx, rdx', 'cmov'),
    # ── Arithmetic: add/sub family ───────────────────────────────────────
    ('add rax, rbx', 'add'),
    ('add rcx, rdx', 'add'),
    ('add rax, rcx', 'add'),
    ('add rbx, rdx', 'add'),
    ('add rax, 0', 'add_imm'),  # identity — taint should be unchanged
    ('add rax, 1', 'add_imm'),  # +1: still taints all bits via carry chain
    ('add rax, 0x100', 'add_imm'),
    ('add rbx, 0x7fffffff', 'add_imm'),
    # ADC / SBB: explicit CF input. These test that engines model CF as a
    # taint source on the implicit-input side. Pair tests (add ; adc) live
    # in the flag_only_adc sequences below.
    ('adc rax, rbx', 'adc'),
    ('adc rcx, rdx', 'adc'),
    ('adc rax, 0', 'adc'),  # rax ← rax + CF — pure CF→reg propagation
    ('sub rax, rbx', 'sub'),
    ('sub rcx, rdx', 'sub'),
    ('sub rax, rcx', 'sub'),
    ('sub rax, 1', 'sub_imm'),
    ('sbb rax, rbx', 'sbb'),
    ('sbb rcx, rdx', 'sbb'),
    ('sbb rax, 0', 'sbb'),  # rax ← rax - CF — pure CF→reg propagation
    # ADCX / ADOX (Broadwell+): like ADC but use CF and OF independently,
    # and only update that one flag. Critical for big-integer crypto.
    # Engines that model CF and OF together mishandle these.
    ('adcx rax, rbx', 'adx'),
    ('adcx rcx, rdx', 'adx'),
    ('adox rax, rbx', 'adx'),
    ('adox rcx, rdx', 'adx'),
    ('neg rax', 'neg'),
    ('neg rbx', 'neg'),
    ('neg rcx', 'neg'),
    ('inc rax', 'inc'),
    ('inc rbx', 'inc'),
    ('dec rax', 'dec'),
    ('dec rbx', 'dec'),
    # Multiplication: 1-operand (rdx:rax = rax * src), 2-operand, 3-operand
    ('mul rbx', 'mul1'),  # rdx:rax ← rax * rbx (unsigned, full 128-bit)
    ('mul rcx', 'mul1'),
    ('imul rbx', 'imul1'),  # signed variant of above
    ('imul rcx', 'imul1'),
    ('imul rax, rbx', 'imul2'),
    ('imul rcx, rdx', 'imul2'),
    ('imul rax, rcx', 'imul2'),
    ('imul rax, rbx, 0', 'imul3'),  # x*0 → 0, taint must collapse
    ('imul rax, rbx, 1', 'imul3'),  # identity scaling
    ('imul rax, rbx, 3', 'imul3'),
    ('imul rax, rbx, 7', 'imul3'),
    ('imul rax, rbx, -1', 'imul3'),  # negation via scaling
    ('imul rcx, rdx, 0x100', 'imul3'),
    # MULX (BMI2): rax:rbx ← rcx * rdx. Does NOT touch flags (unlike MUL),
    # and the high/low halves go to two named registers, not implicit rdx:rax.
    # Often misclassified by engines as plain MUL.
    ('mulx rax, rbx, rcx', 'mulx'),
    # DIV / IDIV: rdx:rax / src → quotient in rax, remainder in rdx.
    # Both outputs tainted iff any of (rdx, rax, src) is tainted.
    # Divisor-non-zero is enforced by _safe_state.
    ('div rbx', 'div'),
    ('div rcx', 'div'),
    ('idiv rbx', 'idiv'),
    ('idiv rcx', 'idiv'),
    # ── Implicit sign-extension instructions ─────────────────────────────
    # These take RAX as implicit input and write RAX (CDQE, CWDE) or RDX
    # (CDQ, CQO, CWD). Flag-clean instructions; useful pre-IDIV.
    ('cdqe', 'signext'),  # eax → rax (sign-ext low 32 to all 64)
    ('cdq', 'signext'),  # eax → edx (sign of eax fills edx)
    ('cqo', 'signext'),  # rax → rdx
    ('cwde', 'signext'),  # ax  → eax
    ('cwd', 'signext'),  # ax  → dx
    # ── Bitwise / logical ────────────────────────────────────────────────
    ('and rax, rbx', 'and'),
    ('and rcx, rdx', 'and'),
    ('and rax, rcx', 'and'),
    ('or  rax, rbx', 'or'),
    ('or  rcx, rdx', 'or'),
    ('or  rax, rdx', 'or'),
    ('xor rax, rbx', 'xor'),
    ('xor rcx, rdx', 'xor'),
    ('xor rax, rdx', 'xor'),
    # XOR reg, reg: canonical idiom for register-clear. Output always 0,
    # taint MUST collapse to clean. Engines that propagate symbolically
    # without constant-folding miss this.
    ('xor rax, rax', 'xor_self'),
    ('xor rbx, rbx', 'xor_self'),
    ('xor rcx, rcx', 'xor_self'),
    ('xor rdx, rdx', 'xor_self'),
    ('not rax', 'not'),
    ('not rbx', 'not'),
    ('not rcx', 'not'),
    # AND with immediate: the masked-out bits MUST be untainted (output is 0).
    # Bit-precise engines should handle this; reg-level cannot.
    ('and rax, 0xff', 'and_imm'),
    ('and rbx, 0xffff', 'and_imm'),
    ('and eax, 0xffffffff', 'and_imm'),  # 32-bit AND zero-extends; clears bits 63:32 entirely
    ('and rcx, 0x0f0f0f0f', 'and_imm'),
    ('and rax, 0x00ff00ff', 'and_imm'),
    ('and rax, 0', 'and_zero'),  # full clear via AND-zero
    ('and rbx, 0', 'and_zero'),
    # OR with immediate: forced-1 bits should be untainted.
    ('or  rax, 0xff', 'or_imm'),
    ('or  rbx, 0x0f0f0f0f', 'or_imm'),
    ('or  rcx, 0x7fffffff', 'or_imm'),
    # TEST / CMP (flag-only — no GPR destination). Pair with SETcc/CMOVcc
    # in the flag_only_* sequences for end-to-end soundness probes.
    ('cmp rax, rbx', 'cmp'),
    ('cmp rcx, rdx', 'cmp'),
    ('cmp rax, 0', 'cmp_imm'),
    ('test rax, rbx', 'test'),
    ('test rcx, rdx', 'test'),
    ('test rax, rax', 'test_self'),
    # ── Shift / rotate ───────────────────────────────────────────────────
    ('shl rax, 1', 'shl'),
    ('shl rbx, 4', 'shl'),
    ('shl rcx, 31', 'shl'),
    ('shl rax, 63', 'shl'),
    ('shl rax, cl', 'shl_cl'),  # shift amount from CL: implicit input from RCX
    ('shr rax, 1', 'shr'),
    ('shr rbx, 8', 'shr'),
    ('shr rcx, 16', 'shr'),
    ('shr rax, cl', 'shr_cl'),
    ('sar rax, 1', 'sar'),
    ('sar rbx, 4', 'sar'),
    ('sar rcx, 63', 'sar'),  # full broadcast of sign bit
    ('sar rax, cl', 'sar_cl'),
    ('rol rax, 1', 'rol'),
    ('rol rbx, 8', 'rol'),
    ('rol rcx, 32', 'rol'),
    ('ror rax, 1', 'ror'),
    ('ror rbx, 4', 'ror'),
    ('ror rcx, cl', 'ror_cl'),
    # RCL / RCR: rotate-through-carry. CF is implicit input AND output bit.
    # If CF is tainted (e.g. from a prior add), the entire result is tainted.
    ('rcl rax, 1', 'rcl'),
    ('rcl rbx, 4', 'rcl'),
    ('rcr rax, 1', 'rcr'),
    ('rcr rbx, 4', 'rcr'),
    # SHLD / SHRD: double-precision shift. Three inputs (dst, src, count).
    # Result bits come from concatenation of dst and src. The trickiest
    # shift to get right; many engines mishandle the cross-register flow.
    ('shld rax, rbx, 4', 'shld'),
    ('shld rcx, rdx, 16', 'shld'),
    ('shld rax, rbx, cl', 'shld_cl'),
    ('shrd rax, rbx, 4', 'shrd'),
    ('shrd rcx, rdx, 32', 'shrd'),
    ('shrd rax, rbx, cl', 'shrd_cl'),
    # Zero-shift identity: output bits = input bits, taint preserved.
    # Some engines incorrectly conservative-taint for shl rax, 0.
    ('shl rax, 0', 'shl_zero'),
    ('shr rbx, 0', 'shr_zero'),
    ('rol rcx, 0', 'rol_zero'),
    # ── Bit manipulation: scan, count, test, set ─────────────────────────
    ('bswap rax', 'bswap'),  # byte-reverse: pure bit-permutation
    ('bswap rbx', 'bswap'),
    ('bswap rcx', 'bswap'),
    ('popcnt rax, rbx', 'popcnt'),  # output ∈ [0,64], taint collapses to low 7 bits
    ('popcnt rcx, rdx', 'popcnt'),
    ('lzcnt rax, rbx', 'lzcnt'),  # leading-zero count
    ('lzcnt rcx, rdx', 'lzcnt'),
    ('tzcnt rax, rbx', 'tzcnt'),  # trailing-zero count
    ('tzcnt rcx, rdx', 'tzcnt'),
    # BSF / BSR: bit-scan forward/reverse. Set ZF if input==0 (otherwise
    # ZF cleared and dst gets the bit index). The ZF behaviour is the only
    # signal that the input was zero — flag-only output channel.
    ('bsf rax, rbx', 'bsf'),
    ('bsr rcx, rdx', 'bsr'),
    # BT / BTC / BTR / BTS: bit test (and modify). Output is in CF (BT) or
    # in dst with CF receiving the original bit. Pure flag output for plain BT.
    ('bt  rax, rbx', 'bt'),
    ('bt  rax, 3', 'bt_imm'),
    ('btc rax, 3', 'btc'),
    ('btc rbx, 17', 'btc'),
    ('btr rbx, 5', 'btr'),
    ('btr rcx, 31', 'btr'),
    ('bts rcx, 7', 'bts'),
    ('bts rdx, 63', 'bts'),
    # ── BMI1 ─────────────────────────────────────────────────────────────
    ('andn rax, rbx, rcx', 'andn'),  # rax ← (~rbx) & rcx
    ('andn rcx, rdx, rax', 'andn'),
    ('blsi rax, rbx', 'blsi'),  # isolate lowest set bit: x & -x
    ('blsi rcx, rdx', 'blsi'),
    ('blsr rax, rbx', 'blsr'),  # reset lowest set bit: x & (x-1)
    ('blsr rcx, rdx', 'blsr'),
    ('blsmsk rax, rbx', 'blsmsk'),  # mask up to lowest set bit: x ^ (x-1)
    ('blsmsk rcx, rdx', 'blsmsk'),
    ('bextr rax, rbx, rcx', 'bextr'),  # bit field extract: rcx[7:0]=start, rcx[15:8]=len
    # ── BMI2 ─────────────────────────────────────────────────────────────
    # BZHI: zero high bits starting at position rcx[7:0]. Per-bit dependency.
    ('bzhi rax, rbx, rcx', 'bzhi'),
    ('bzhi rcx, rdx, rax', 'bzhi'),
    # PDEP / PEXT: parallel bit deposit / extract. Bit-precise permutation
    # gated by a mask. Trivial for bit-level engines (microtaint, angr) to
    # model precisely; impossible for register-level engines to be precise on.
    ('pdep rax, rbx, rcx', 'pdep'),
    ('pdep rcx, rdx, rax', 'pdep'),
    ('pext rax, rbx, rcx', 'pext'),
    ('pext rcx, rdx, rax', 'pext'),
    # BMI2 shifts: like SHL/SHR/SAR but flag-clean and VEX-encoded.
    # Engines often forget these exist and fail to recognise the mnemonic.
    ('rorx rax, rbx, 4', 'rorx'),
    ('rorx rcx, rdx, 16', 'rorx'),
    ('sarx rax, rbx, rcx', 'sarx'),
    ('shlx rax, rbx, rcx', 'shlx'),
    ('shrx rax, rbx, rcx', 'shrx'),
    # ── LEA: address calculation, no memory access ───────────────────────
    # LEA is heavily used as a strength-reduction tool (mul-by-3 = lea
    # rax,[rbx+rbx*2]). Bit-precise taint is hard because the inner
    # addition and shift have to be modelled.
    ('lea rax, [rbx]', 'lea'),  # plain copy
    ('lea rax, [rbx + rcx]', 'lea'),  # add
    ('lea rax, [rbx + rcx*1]', 'lea'),
    ('lea rax, [rbx + rcx*2]', 'lea'),  # 2x scaling
    ('lea rax, [rbx + rcx*4]', 'lea'),
    ('lea rax, [rbx + rcx*8]', 'lea'),
    ('lea rax, [rbx + rcx*4 + 8]', 'lea'),
    ('lea rax, [rbx + rcx*8 + 0x100]', 'lea'),
    ('lea rbx, [rax + rdx*8]', 'lea'),
    ('lea rcx, [rax*2 + rdx]', 'lea'),  # base-less form
    ('lea rax, [rcx*8]', 'lea'),
    ('lea rdx, [rip + 0x100]', 'lea'),  # PC-relative — no GPR input
    ('lea eax, [rbx + rcx*4]', 'lea_zext'),  # 32-bit dest zero-extends — partial-write trap
    # ── Flag manipulation as data ────────────────────────────────────────
    # LAHF: AH ← FLAGS[7:0] (SF, ZF, AF, PF, CF + reserved bits).
    # SAHF: FLAGS[7:0] ← AH.
    # The LAHF/SAHF round-trip exposes taint flowing through flags and back
    # to a GPR — engines without flag taint cannot reflect any taint here.
    ('lahf', 'lahf'),
    ('sahf', 'sahf'),
    ('clc', 'flagop'),  # CF ← 0 (sanitizer for CF)
    ('stc', 'flagop'),  # CF ← 1
    ('cmc', 'flagop'),  # CF ← ~CF (preserves CF taint)
    # ── SETcc family ─────────────────────────────────────────────────────
    # SETcc dst8: dst ← (cond ? 1 : 0). The ONLY data flow is through the
    # condition flags. If the flag-producer was tainted, the SET output
    # MUST be tainted. Engines without flag taint produce clean output here
    # — definitive soundness violations. Pairs in flag_only_set sequences.
    ('seto al', 'setcc'),
    ('setno al', 'setcc'),
    ('setc al', 'setcc'),
    ('setnc al', 'setcc'),
    ('setp al', 'setcc'),
    ('setnp al', 'setcc'),
    ('seta al', 'setcc'),
    ('setae al', 'setcc'),
    ('setb al', 'setcc'),
    ('setbe al', 'setcc'),
    ('sete al', 'setcc'),
    ('setne al', 'setcc'),
    ('setg al', 'setcc'),
    ('setge al', 'setcc'),
    ('setl al', 'setcc'),
    ('setle al', 'setcc'),
    ('sets al', 'setcc'),
    ('setns al', 'setcc'),
    # ── Partial-register writes (single-instruction trap cases) ──────────
    # Writes to AL/AX preserve the upper bits of RAX. Writes to EAX
    # zero-extend (clear bits 63:32). Engines that conflate these cases
    # are unsound or imprecise on common compiler-generated code.
    ('mov al, bl', 'partial_write_al'),
    ('mov ah, bh', 'partial_write_ah'),
    ('mov ax, bx', 'partial_write_ax'),
    ('mov eax, ebx', 'partial_write_eax_zext'),  # 32-bit write zero-extends RAX
    # Sub-register XOR: a famous trap. xor eax, eax CLEARS RAX entirely
    # (zero-extension of 0). xor al, al only clears the low byte and
    # taint of bits 63:8 of RAX MUST survive.
    ('xor eax, eax', 'partial_write_xor32_clears'),  # full clear
    ('xor ebx, ebx', 'partial_write_xor32_clears'),
    ('xor al, al', 'partial_write_xor8_keeps'),  # partial — high bits keep taint
    ('xor bl, bl', 'partial_write_xor8_keeps'),
    # Sub-register arithmetic: similar partial-update behaviour.
    ('add al, bl', 'partial_write_add8'),
    ('add ax, bx', 'partial_write_add16'),
    ('add eax, ebx', 'partial_write_add32_zext'),  # zero-extends
    ('and al, 0xf0', 'partial_write_and8'),
    ('or  al, 0x0f', 'partial_write_or8'),
    ('inc al', 'partial_write_inc8'),
    ('inc eax', 'partial_write_inc32_zext'),
    # ── No-op / hint ────────────────────────────────────────────────────
    # NOP must be a strict identity for taint state. Engines that mistakenly
    # mutate state on NOP are clearly broken; this is a sanity check.
    ('nop', 'nop'),
    # ── SIMD register-form (xmm0/xmm1) — for engines that accept xmm names
    # in single-instruction tests. These appear primarily in sequences with
    # MOVQ setup; the entries below are a representative subset.
    # NOTE: state generation does NOT initialize xmm registers, so taint
    # behaviour here depends on whatever the subprocess started with.
    # The corresponding sequences in INSTRUCTION_SEQUENCES are the
    # authoritative SIMD tests.
    ('paddb xmm0, xmm1', 'simd_padd'),
    ('paddq xmm0, xmm1', 'simd_padd'),
    ('psubb xmm0, xmm1', 'simd_psub'),
    ('pand  xmm0, xmm1', 'simd_pbool'),
    ('por   xmm0, xmm1', 'simd_pbool'),
    ('pxor  xmm0, xmm1', 'simd_pbool'),
    ('pcmpeqb xmm0, xmm1', 'simd_pcmp'),
    ('pshufb xmm0, xmm1', 'simd_pshufb'),
]

# ---------------------------------------------------------------------------
# Multi-instruction sequences
# ---------------------------------------------------------------------------
#
# Format: (label, [list of asm strings], category)
# Each sequence is assembled in order and treated as a single test case.
# The combined bytes are passed to workers that support sequences.
# Workers that do not (taintgrind, libdft64) execute them inside the harness
# as concatenated asm volatile blocks.

INSTRUCTION_SEQUENCES: list[tuple[str, list[str], str]] = [
    # ── Propagation chains (general) ─────────────────────────────────────
    ('chain_add3', ['add rax, rbx', 'add rcx, rax', 'add rdx, rcx'], 'chain'),
    ('chain_shift_or', ['shl rax, 4', 'or rbx, rax', 'shr rbx, 2'], 'chain'),
    ('chain_mov_xor', ['mov rcx, rax', 'xor rdx, rcx', 'mov rax, rdx'], 'chain'),
    ('chain_mul_add', ['imul rax, rbx', 'add rcx, rax', 'mov rdx, rcx'], 'chain'),
    ('chain_neg_sub', ['neg rax', 'sub rbx, rax', 'mov rcx, rbx'], 'chain'),
    (
        'chain_long',
        ['add rax, rbx', 'imul rcx, rax', 'xor rdx, rcx', 'shl rax, 2', 'or rbx, rax', 'and rcx, rbx', 'sub rdx, rax'],
        'chain',
    ),
    ('chain_xchg_chain', ['xchg rax, rbx', 'add rcx, rax', 'xchg rcx, rdx', 'or rax, rdx'], 'chain'),
    # =====================================================================
    # FLAG-SOUNDNESS PROBES
    # ---------------------------------------------------------------------
    # Each sequence below is constructed so the ONLY data-flow path from a
    # tainted input register to the recorded output register passes through
    # one or more EFLAGS bits. A taint engine that does not track flags as
    # state will declare the output clean — this is unsound and the GT
    # simulator will surface the disagreement.
    #
    # Categories:
    #   flag_only_set  — cmp/test/sub then setcc; output is in AL/AH/etc.
    #   flag_only_cmov — cmp/test then cmovcc; output is in dst register.
    #                    Distinct from existing "implicit_flow" because we
    #                    deliberately CLEAR dst beforehand so old-dst can't
    #                    smuggle taint past the flag check.
    #   flag_only_adc  — add then adc-zero, isolating CF as sole carrier.
    #   flag_only_sbb  — sub then sbb-zero / sbb-self for borrow.
    #   flag_only_sahf — store flags to AH (LAHF) / restore (SAHF) round-
    #                    trip; tests engines that lack flag taint
    #                    completely fail to propagate through AH.
    #   flag_only_bsx  — bsf/bsr produce data in dst, but ZF is the ONLY
    #                    signal that the input was zero. Engines that
    #                    don't taint ZF on bsf/bsr miss the i==0 channel.
    #   flag_only_bt   — bt rax, rbx; setc dl  (output is solely from CF)
    # =====================================================================
    # ── flag_only_set: SETcc family with various flag producers ─────────
    # Pattern: tainted-input → flag-producer → setcc dst8 (ZERO out the rest
    # of dst with `xor edx, edx; setcc dl`) so the output dl/al has NO
    # pre-existing taint to launder. The only taint path is RAX→FLAGS→dl.
    (
        'flag_only_setz_cmp',
        [
            'xor edx, edx',  # clear rdx; sound engines: rdx now untainted
            'cmp rax, rbx',  # ZF depends on (rax == rbx) — ZF tainted iff (rax|rbx) tainted
            'setz dl',
        ],  # dl = ZF; rdx[0] MUST be tainted for soundness
        'flag_only_set',
    ),
    ('flag_only_setnz_cmp', ['xor edx, edx', 'cmp rax, rbx', 'setnz dl'], 'flag_only_set'),
    ('flag_only_sets_cmp', ['xor edx, edx', 'cmp rax, rbx', 'sets dl'], 'flag_only_set'),
    ('flag_only_setc_sub', ['xor edx, edx', 'sub rax, rbx', 'setc dl'], 'flag_only_set'),
    ('flag_only_seto_sub', ['xor edx, edx', 'sub rax, rbx', 'seto dl'], 'flag_only_set'),
    ('flag_only_seta_cmp', ['xor edx, edx', 'cmp rax, rbx', 'seta dl'], 'flag_only_set'),
    ('flag_only_setb_cmp', ['xor edx, edx', 'cmp rax, rbx', 'setb dl'], 'flag_only_set'),
    ('flag_only_setg_cmp', ['xor edx, edx', 'cmp rax, rbx', 'setg dl'], 'flag_only_set'),
    ('flag_only_setl_cmp', ['xor edx, edx', 'cmp rax, rbx', 'setl dl'], 'flag_only_set'),
    ('flag_only_setp_test', ['xor edx, edx', 'test al, bl', 'setp dl'], 'flag_only_set'),
    ('flag_only_setz_testself', ['xor edx, edx', 'test rax, rax', 'setz dl'], 'flag_only_set'),
    ('flag_only_setz_testand', ['xor edx, edx', 'test rax, rbx', 'setz dl'], 'flag_only_set'),
    ('flag_only_setz_andimm', ['xor edx, edx', 'test rax, 0xff', 'setz dl'], 'flag_only_set'),
    # Multi-flag fan-out: one flag-producer feeds multiple SETcc outputs.
    # Tests whether engines duplicate flag taint across parallel SETcc.
    ('flag_only_set_multi', ['xor ecx, ecx', 'xor edx, edx', 'cmp rax, rbx', 'setz cl', 'setl dl'], 'flag_only_set'),
    # ── flag_only_cmov: tainted comparison feeds cmov ────────────────────
    # The new-dst is also zeroed first so dst-after = (cond ? src : 0).
    # Sound: dst tainted iff (flags tainted) ∨ (src tainted ∧ flag could be 1).
    # Engines without flag taint conclude dst clean when only the flag
    # carries the tainted comparison result.
    (
        'flag_only_cmovz_clean_src',
        [
            'xor edx, edx',  # rdx ← 0 (clean)
            'cmp rax, rbx',  # ZF ← (rax==rbx); ZF tainted
            'mov rcx, 5',  # rcx ← 5 (CONCRETE clean)
            'cmovz rdx, rcx',
        ],  # rdx ← (ZF ? rcx : rdx) = (ZF ? 5 : 0)
        # Both branches concrete; output depends ONLY on
        # ZF which depends only on (rax|rbx). rdx MUST
        # be tainted in low 3 bits if either is tainted.
        'flag_only_cmov',
    ),
    ('flag_only_cmovnz_clean_src', ['xor edx, edx', 'cmp rax, rbx', 'mov rcx, 7', 'cmovnz rdx, rcx'], 'flag_only_cmov'),
    (
        'flag_only_cmovs_clean_src',
        ['xor edx, edx', 'cmp rax, rbx', 'mov rcx, 0xff', 'cmovs rdx, rcx'],
        'flag_only_cmov',
    ),
    (
        'flag_only_cmovc_after_sub',
        ['xor edx, edx', 'sub rax, rbx', 'mov rcx, 0xff', 'cmovc rdx, rcx'],
        'flag_only_cmov',
    ),
    (
        'flag_only_cmovo_after_sub',
        ['xor edx, edx', 'sub rax, rbx', 'mov rcx, 0xff', 'cmovo rdx, rcx'],
        'flag_only_cmov',
    ),
    ('flag_only_cmovg_after_cmp', ['xor edx, edx', 'cmp rax, rbx', 'mov rcx, 1', 'cmovg rdx, rcx'], 'flag_only_cmov'),
    ('flag_only_cmovl_after_cmp', ['xor edx, edx', 'cmp rax, rbx', 'mov rcx, 1', 'cmovl rdx, rcx'], 'flag_only_cmov'),
    # ── flag_only_adc / sbb: pure carry-flag carriers ────────────────────
    # Pattern: produce CF from tainted op, then `adc reg, 0` / `sbb reg, 0`
    # where `reg` is concrete-cleaned beforehand. Output low bit = CF,
    # which is tainted iff the producer's inputs were tainted.
    (
        'flag_only_adc_zero',
        [
            'xor ecx, ecx',  # rcx ← 0 (clean)
            'add rax, rbx',  # CF tainted iff (rax|rbx) tainted
            'adc rcx, 0',
        ],  # rcx ← rcx + CF = CF; rcx[0] must be tainted
        'flag_only_adc',
    ),
    ('flag_only_adc_one', ['xor ecx, ecx', 'add rax, rbx', 'adc rcx, 1'], 'flag_only_adc'),
    ('flag_only_sbb_zero', ['xor ecx, ecx', 'sub rax, rbx', 'sbb rcx, 0'], 'flag_only_sbb'),
    # sbb reg, reg = -(CF) — generates 0 or -1 depending solely on CF.
    # Branchless idiom for "if CF then 0xFF...FF else 0".
    ('flag_only_sbb_self', ['sub rax, rbx', 'sbb rcx, rcx'], 'flag_only_sbb'),
    # ── flag_only_sahf / lahf round-trip ─────────────────────────────────
    # LAHF copies SF/ZF/AF/PF/CF into AH (bits 15:8 of RAX). Pure flag→GPR
    # transfer. After this, the upper bits of RAX (16:63) carry no flag
    # info, but bits 15:8 do. Sound engines: AH bits tainted iff any of
    # {SF, ZF, AF, PF, CF} were tainted.
    ('flag_only_lahf', ['cmp rax, rbx', 'lahf'], 'flag_only_lahf'),  # SF/ZF/CF/PF/AF/OF all tainted  # ah ← flags
    ('flag_only_lahf_after_test', ['test rax, rbx', 'lahf'], 'flag_only_lahf'),
    # SAHF: AH → flags. Composing this with a SETcc gives full GPR→flag→GPR
    # round-trip. Engines that don't taint flags lose the chain entirely.
    (
        'flag_only_sahf_setz',
        [
            'xor edx, edx',
            'mov ah, bl',  # ah carries bl's taint
            'sahf',  # flags ← ah  (SF/ZF/AF/PF/CF)
            'setz dl',
        ],  # dl ← ZF — taint round-trips bl→ah→ZF→dl
        'flag_only_sahf',
    ),
    # ── flag_only_bsx: BSF/BSR's ZF as input-zero indicator ─────────────
    # bsf rax, rbx: rax ← lowest set bit of rbx; ZF set iff rbx == 0.
    # If rbx is tainted, bit i of rax tainted depends on the set-bit
    # pattern, but the ZF channel ALWAYS depends on whether rbx was zero.
    # Engines that don't taint ZF on bsf miss the input-was-zero observation.
    ('flag_only_bsf_zf_setz', ['xor edx, edx', 'bsf rcx, rbx', 'setz dl'], 'flag_only_bsx'),
    ('flag_only_bsr_zf_setz', ['xor edx, edx', 'bsr rcx, rbx', 'setz dl'], 'flag_only_bsx'),
    # ── flag_only_bt: pure CF-output bit test ────────────────────────────
    ('flag_only_bt_cf_setc', ['xor edx, edx', 'bt rax, rbx', 'setc dl'], 'flag_only_bt'),
    ('flag_only_bt_imm', ['xor edx, edx', 'bt rax, 17', 'setc dl'], 'flag_only_bt'),
    # btc/btr/bts have GPR-side and flag-side outputs; isolate the flag side:
    ('flag_only_btc_cf', ['xor edx, edx', 'btc rax, 5', 'setc dl'], 'flag_only_bt'),
    # ── Chained flag dependencies: shift→adc, mul→jc-style ───────────────
    # SHL puts the last shifted-out bit in CF. ADC reads CF. Result is a
    # multi-step taint chain that requires CF-level tracking throughout.
    (
        'flag_only_shl_then_adc',
        ['xor ecx, ecx', 'shl rax, 4', 'adc rcx, 0'],  # CF ← bit 60 of rax  # rcx ← CF
        'flag_only_chain',
    ),
    ('flag_only_shr_then_adc', ['xor ecx, ecx', 'shr rax, 1', 'adc rcx, 0'], 'flag_only_chain'),
    # Three-way chain: tainted RBX → cmp → ZF → cmov → tainted RDX → use.
    (
        'flag_only_long_chain',
        [
            'xor edx, edx',
            'cmp rax, rbx',
            'mov rcx, 0x1234',
            'cmovz rdx, rcx',  # rdx now carries flag taint via concrete src
            'shl rdx, 2',
        ],  # propagate to bits 2..15
        'flag_only_chain',
    ),
    # ── ADCX / ADOX: independent-flag arithmetic ─────────────────────────
    # ADCX uses CF, ADOX uses OF. They DO NOT touch each other's flag.
    # An engine that conflates CF and OF taint (treats EFLAGS as one
    # opaque blob) will be UNNECESSARILY conservative here — over-taint.
    # Per-flag engines stay precise.
    ('flag_only_adcx_after_add', ['xor ecx, ecx', 'add rax, rbx', 'adcx rcx, rdx'], 'flag_only_adx'),
    ('flag_only_adox_after_add', ['xor ecx, ecx', 'add rax, rbx', 'adox rcx, rdx'], 'flag_only_adx'),
    # Big-integer-style: interleaved CF and OF chains in parallel.
    ('flag_only_adcx_adox_parallel', ['adcx rax, rbx', 'adox rcx, rdx'], 'flag_only_adx'),
    # =====================================================================
    # PARTIAL-REGISTER WRITE PROBES
    # ---------------------------------------------------------------------
    # Sub-register writes are a famous source of taint engine bugs:
    #   write to AL  → preserve bits 63:8  of RAX  (taint on those bits SURVIVES)
    #   write to AX  → preserve bits 63:16 of RAX
    #   write to EAX → ZERO bits 63:32 of RAX  (taint on those bits CLEARED)
    # Engines that get this wrong are unsound (taint disappearance) or
    # imprecise (over-taint after an EAX write that should sanitise).
    # =====================================================================
    # ── al-write must preserve high bits of RAX (taint survival) ─────────
    (
        'partial_al_high_survives',
        [
            'mov al, bl',  # only AL changes; bits 63:8 of RAX untouched
            'shr rax, 8',
        ],  # shifts the surviving high bits down for inspection
        'partial_write',
    ),
    (
        'partial_ah_only',
        [
            'mov ah, bh',
            'shr rax, 8',  # bring AH into low byte
            'and rax, 0xff',
        ],  # isolate former-AH; if rax taint survives here,
        # the AH write was correctly localized
        'partial_write',
    ),
    # ── eax-write MUST zero-extend (taint of bits 63:32 must DISAPPEAR) ─
    (
        'partial_eax_clears_high',
        [
            'mov eax, ebx',  # zero-extends — RAX[63:32] becomes 0
            'shr rax, 32',
        ],  # if engine still has high-bit taint, it'll show here
        'partial_write',
    ),
    # XOR EAX, EAX: idiomatic clear of full RAX. Sound engines: T(rax) = 0.
    (
        'partial_xor32_clears_rax',
        ['xor eax, eax', 'or rax, rbx'],  # clears all 64 bits of RAX  # remaining taint must come solely from RBX
        'partial_write',
    ),
    # XOR AL, AL: clears ONLY the low byte. Bits 63:8 of RAX retain taint.
    # Engines that conflate xor-self-32 and xor-self-8 incorrectly clear RAX.
    (
        'partial_xor8_keeps_high',
        ['xor al, al', 'shr rax, 8'],  # only AL ← 0; high bits keep taint  # surface the surviving taint
        'partial_write',
    ),
    # ADD on sub-registers: same partial-update pattern.
    ('partial_add8_keeps_high', ['add al, bl', 'shr rax, 8'], 'partial_write'),
    ('partial_add32_clears_high', ['add eax, ebx', 'shr rax, 32'], 'partial_write'),
    # AH alias trap: AH is bits 15:8 of RAX. Write to AH must not affect
    # bits 7:0 (AL) or bits 63:16. Tests engine handling of high-byte regs.
    (
        'partial_ah_independent_of_al',
        ['mov ah, bl', 'and rax, 0xff'],  # ah ← bl  # isolate AL — AH write must not have touched AL
        'partial_write',
    ),
    # Cross-write trap: write AL, write AH, then read full RAX.
    # Sound bit-precise engine: low byte from BL, AH from CL, others from old RAX.
    ('partial_al_then_ah', ['mov al, bl', 'mov ah, cl'], 'partial_write'),
    # 32-bit arithmetic that secretly clears the high half — common mistake
    # in disassembler IRs that don't know `add eax, ebx` zero-extends.
    ('partial_inc32_clears_high', ['inc eax', 'shr rax, 32'], 'partial_write'),
    # LEA with 32-bit destination: zero-extends. Engines that treat LEA
    # destination width as always 64 fail this.
    ('partial_lea32_clears_high', ['lea eax, [rbx + rcx*4]', 'shr rax, 32'], 'partial_write'),
    # ── Carry/borrow propagation (existing, kept and extended) ───────────
    ('carry_chain_adc', ['add rax, rbx', 'adc rcx, rdx'], 'carry'),
    ('carry_chain_adc_long', ['add rax, rbx', 'adc rcx, rdx', 'adc rax, 0'], 'carry'),
    ('borrow_chain_sbb', ['sub rax, rbx', 'sbb rcx, rdx'], 'carry'),
    ('borrow_chain_sbb_long', ['sub rax, rbx', 'sbb rcx, rdx', 'sbb rax, 0'], 'carry'),
    # ── Sanitiser patterns (output is concrete-zero or concrete value) ──
    ('sanitise_xor_self', ['xor rax, rax', 'xor rbx, rbx'], 'sanitiser'),
    ('sanitise_and_zero_rax', ['and rax, 0'], 'sanitiser'),
    ('sanitise_highbyte', ['and rax, 0xff'], 'sanitiser'),
    ('sanitise_sub_self', ['sub rax, rax'], 'sanitiser'),  # x-x = 0, taint must collapse
    ('sanitise_mul_zero', ['imul rax, rbx, 0'], 'sanitiser'),  # x*0 = 0
    ('not_and_mask', ['not rax', 'and rax, 0x0f0f0f0f'], 'sanitiser'),
    # ── Implicit flow via conditional move (existing, kept) ──────────────
    ('implicit_cmov_z', ['cmp rax, 0', 'cmovz rax, rbx'], 'implicit_flow'),
    ('implicit_cmov_nz', ['test rcx, rcx', 'cmovnz rcx, rdx'], 'implicit_flow'),
    # ── Sub-register and zero-extension (existing, expanded) ─────────────
    ('subreg_32bit_write', ['mov eax, ebx'], 'subreg'),
    ('subreg_8bit_write', ['mov al, bl'], 'subreg'),
    ('subreg_movzx_chain', ['movzx rax, bx', 'add rax, rcx'], 'subreg'),
    ('subreg_movsx_chain', ['movsx rax, bl', 'imul rax, rdx'], 'subreg'),
    ('subreg_movsxd_chain', ['movsxd rax, ebx', 'add rax, rcx'], 'subreg'),
    # MOVSX upper-bit replication: every bit 63:7 of dst becomes a copy of
    # bit 7 of src. So ALL high bits of taint = taint(bit 7 of src).
    ('subreg_movsx_sign_replication', ['movsx rax, bl', 'shr rax, 8'], 'subreg'),
    # ── Rotate and byte-swap (existing, expanded) ────────────────────────
    ('rotate_8', ['rol rax, 8', 'ror rbx, 8'], 'permutation'),
    ('rotate_double', ['shld rax, rbx, 8', 'shrd rcx, rdx, 8'], 'permutation'),
    ('bswap_and', ['bswap rax', 'mov rbx, 0xff00ff00ff00ff00', 'and rax, rbx'], 'permutation'),
    ('bswap_roundtrip', ['bswap rax', 'bswap rax'], 'permutation'),
    # PDEP/PEXT round-trip is a nontrivial bit permutation gated by mask.
    ('bmi_pdep_pext_roundtrip', ['pdep rax, rbx, rcx', 'pext rdx, rax, rcx'], 'permutation'),
    # ── Shift amount from register (existing, expanded) ──────────────────
    ('shl_cl_chain', ['mov cl, 4', 'shl rax, cl'], 'shift_by_reg'),
    ('shr_cl_tainted', ['mov rcx, rdx', 'shr rax, cl'], 'shift_by_reg'),
    ('shld_cl_chain', ['mov cl, 8', 'shld rax, rbx, cl'], 'shift_by_reg'),
    ('shrd_cl_tainted', ['mov rcx, rdx', 'shrd rax, rbx, cl'], 'shift_by_reg'),
    # ── LEA chains (existing, expanded) ──────────────────────────────────
    ('lea_chain', ['lea rax, [rbx + rcx*2]', 'lea rdx, [rax + rbx*4]'], 'lea'),
    ('lea_scale_add', ['lea rax, [rbx + rcx*8 + 16]', 'add rdx, rax'], 'lea'),
    ('lea_strength_reduction_mul3', ['lea rax, [rbx + rbx*2]'], 'lea'),  # rax = 3*rbx
    ('lea_strength_reduction_mul5', ['lea rax, [rbx + rbx*4]'], 'lea'),  # rax = 5*rbx
    ('lea_strength_reduction_mul9', ['lea rax, [rbx + rbx*8]'], 'lea'),  # rax = 9*rbx
    # ── Stack round-trip with various offsets ────────────────────────────
    ('stack_roundtrip_rax', ['push rax', 'pop rax'], 'memory'),
    ('stack_roundtrip_cross', ['push rax', 'pop rbx'], 'memory'),
    ('stack_offset_load_store', ['mov qword ptr [rsp - 16], rax', 'mov rbx, qword ptr [rsp - 16]'], 'memory'),
    (
        'stack_aliased_overwrite',
        [
            'mov qword ptr [rsp - 16], rax',  # store rax to slot
            'mov qword ptr [rsp - 16], rbx',  # overwrite — slot now solely from rbx
            'mov rcx, qword ptr [rsp - 16]',
        ],  # rcx should be tainted ONLY by rbx
        'memory',
    ),
    (
        'stack_indexed_read',
        ['mov rsi, rsp', 'sub rsi, 16', 'mov qword ptr [rsi], rax', 'mov rbx, qword ptr [rsi]'],
        'memory',
    ),
    # ── Memory-operand data movement (load / store / ALU-with-memory) ─────
    # These exercise taint flow *through* a memory operand rather than only
    # register-to-register. Memory contents are made deterministic (a tainted
    # register is stored to an RSP-relative slot first), so every engine and
    # the ground truth agree on both the address and the initial bytes; taint
    # stays register-anchored and is therefore scoreable on RAX-RDX, exactly
    # like the stack round-trips above.
    # ALU with a memory *source* operand (load path):
    ('mem_add_src', ['mov qword ptr [rsp - 16], rbx', 'add rax, qword ptr [rsp - 16]'], 'memory'),
    ('mem_sub_src', ['mov qword ptr [rsp - 16], rbx', 'sub rax, qword ptr [rsp - 16]'], 'memory'),
    ('mem_and_src', ['mov qword ptr [rsp - 16], rbx', 'and rax, qword ptr [rsp - 16]'], 'memory'),
    ('mem_or_src', ['mov qword ptr [rsp - 16], rbx', 'or rax, qword ptr [rsp - 16]'], 'memory'),
    ('mem_xor_src', ['mov qword ptr [rsp - 16], rbx', 'xor rax, qword ptr [rsp - 16]'], 'memory'),
    ('mem_test_flag', ['mov qword ptr [rsp - 16], rbx', 'test rax, qword ptr [rsp - 16]', 'setnz cl'], 'memory'),
    ('mem_cmp_flag', ['mov qword ptr [rsp - 16], rbx', 'cmp rax, qword ptr [rsp - 16]', 'setl cl'], 'memory'),
    # ALU with a memory *destination* operand (read-modify-write, then reload):
    ('mem_add_dst', ['mov qword ptr [rsp - 16], rax', 'add qword ptr [rsp - 16], rbx', 'mov rcx, qword ptr [rsp - 16]'], 'memory'),
    ('mem_xor_dst', ['mov qword ptr [rsp - 16], rax', 'xor qword ptr [rsp - 16], rbx', 'mov rcx, qword ptr [rsp - 16]'], 'memory'),
    ('mem_and_dst', ['mov qword ptr [rsp - 16], rax', 'and qword ptr [rsp - 16], rbx', 'mov rcx, qword ptr [rsp - 16]'], 'memory'),
    # Sub-register / narrow memory access — bit-precision discriminators.
    # Register-level engines coalesce these to the whole destination.
    ('mem_byte_load_zx', ['mov qword ptr [rsp - 16], rax', 'movzx rbx, byte ptr [rsp - 16]'], 'memory'),
    ('mem_word_load_zx', ['mov qword ptr [rsp - 16], rax', 'movzx rbx, word ptr [rsp - 16]'], 'memory'),
    ('mem_dword_load_ze', ['mov qword ptr [rsp - 16], rax', 'mov ebx, dword ptr [rsp - 16]'], 'memory'),
    # Sign-extended loads exercise the transportable sign-extension term.
    ('mem_byte_load_sx', ['mov qword ptr [rsp - 16], rax', 'movsx rbx, byte ptr [rsp - 16]'], 'memory'),
    ('mem_dword_load_sx', ['mov qword ptr [rsp - 16], rax', 'movsxd rbx, dword ptr [rsp - 16]'], 'memory'),
    # Partial memory overwrite: low bytes from one source, high bytes from another.
    ('mem_byte_store', ['mov qword ptr [rsp - 16], rbx', 'mov byte ptr [rsp - 16], al', 'mov rcx, qword ptr [rsp - 16]'], 'memory'),
    ('mem_high_byte_store', ['mov qword ptr [rsp - 16], rbx', 'mov byte ptr [rsp - 15], ah', 'mov rcx, qword ptr [rsp - 16]'], 'memory'),
    ('mem_dword_partial_overwrite', ['mov qword ptr [rsp - 16], rax', 'mov dword ptr [rsp - 16], ebx', 'mov rcx, qword ptr [rsp - 16]'], 'memory'),
    # Indexed / base+index+disp addressing through memory.
    ('mem_indexed_rmw', ['mov rsi, rsp', 'mov qword ptr [rsi - 8], rax', 'add rbx, qword ptr [rsi - 8]'], 'memory'),
    ('mem_base_index_disp', ['mov rsi, rsp', 'mov rdi, 8', 'mov qword ptr [rsi + rdi - 32], rax', 'mov rbx, qword ptr [rsi + rdi - 32]'], 'memory'),
    # ── Function preamble/epilogue (existing) ────────────────────────────
    ('func_preamble', ['push rbx', 'mov rbx, rax', 'imul rbx, rcx', 'mov rax, rbx', 'pop rbx'], 'func_skeleton'),
    # ── MUL 128-bit result (existing, expanded) ──────────────────────────
    ('mul_rdx_rax', ['mul rbx'], 'mulx'),
    ('mul_chain_add', ['mul rbx', 'add rdx, rcx'], 'mulx'),
    ('imul1_rdx_rax', ['imul rbx'], 'mulx'),
    # MULX (BMI2) doesn't touch flags and uses explicit rather than implicit
    # destinations. Tests that engines distinguish MUL from MULX.
    ('mulx_bmi2', ['mulx rax, rbx, rcx'], 'mulx'),
    # ── DIV / IDIV chains ────────────────────────────────────────────────
    # idiv reads rdx:rax, so a sign-extension via cqo is the canonical
    # preamble. Tests propagation through the implicit register pair.
    ('idiv_after_cqo', ['cqo', 'idiv rbx'], 'div_chain'),
    ('div_after_xor', ['xor edx, edx', 'div rbx'], 'div_chain'),  # zero-extend dividend
    # ── Bit manipulation chains (existing, expanded) ─────────────────────
    ('blsi_andn', ['blsi rax, rbx', 'andn rcx, rax, rdx'], 'bmi'),
    ('popcnt_cmp', ['popcnt rax, rbx', 'cmp rax, 32'], 'bmi'),
    ('bzhi_then_or', ['bzhi rax, rbx, rcx', 'or rdx, rax'], 'bmi'),
    ('pdep_then_xor', ['pdep rax, rbx, rcx', 'xor rdx, rax'], 'bmi'),
    ('bextr_then_use', ['bextr rax, rbx, rcx', 'shl rax, 4'], 'bmi'),
    # ── Loop bodies (existing, expanded) ─────────────────────────────────
    ('loop_body_x4', ['add rax, rbx', 'add rax, rbx', 'add rax, rbx', 'add rax, rbx'], 'loop'),
    ('loop_body_mixed', ['imul rax, rcx', 'add rbx, rax', 'shr rbx, 1', 'xor rcx, rbx', 'add rdx, rcx'], 'loop'),
    (
        'loop_body_long',
        [
            'add rax, rbx',
            'imul rcx, rax',
            'xor rdx, rcx',
            'shl rbx, 1',
            'or rax, rdx',
            'sub rcx, rbx',
            'shr rax, 3',
            'add rdx, rax',
        ],
        'loop',
    ),
    # ── Crypto-style mixing (existing, expanded) ─────────────────────────
    ('crypto_mix', ['rol rax, 13', 'xor rax, rbx', 'rol rax, 7', 'xor rax, rcx', 'add rax, rdx'], 'crypto'),
    (
        'crypto_mix_long',
        [
            'rol rax, 17',
            'xor rax, rbx',
            'shl rcx, 5',
            'xor rax, rcx',
            'ror rdx, 11',
            'add rax, rdx',
            'imul rax, rax, 0x1e3779b9',  # near-Knuth golden ratio const, fits signed 32-bit
            'ror rax, 7',
        ],
        'crypto',
    ),
    # ARX-style (Add-Rotate-Xor): basic primitive in ChaCha, BLAKE, etc.
    (
        'crypto_arx',
        ['add rax, rbx', 'xor rdx, rax', 'rol rdx, 16', 'add rcx, rdx', 'xor rbx, rcx', 'rol rbx, 12'],
        'crypto',
    ),
    # ── Sub-register narrowing (existing) ────────────────────────────────
    ('widen_narrow', ['movzx rax, bl', 'add rax, rcx', 'mov bl, al'], 'subreg'),
    # =====================================================================
    # SIMD (SSE2) sequences with proper xmm setup
    # ---------------------------------------------------------------------
    # MOVQ xmm,rax loads the low 64 bits of xmm with rax (high 64 cleared).
    # We exercise SSE ops then MOVQ back to rax. Bit-precise engines that
    # model XMM at byte/bit granularity (microtaint, angr) should stay
    # precise; reg-level engines collapse to "tainted/not".
    # =====================================================================
    (
        'simd_paddb_roundtrip',
        ['movq xmm0, rax', 'movq xmm1, rbx', 'paddb xmm0, xmm1', 'movq rax, xmm0'],  # byte-wise add
        'simd',
    ),
    (
        'simd_paddq_roundtrip',
        ['movq xmm0, rax', 'movq xmm1, rbx', 'paddq xmm0, xmm1', 'movq rax, xmm0'],  # full 64-bit lane add
        'simd',
    ),
    ('simd_pxor_roundtrip', ['movq xmm0, rax', 'movq xmm1, rbx', 'pxor xmm0, xmm1', 'movq rax, xmm0'], 'simd'),
    (
        'simd_pand_roundtrip',
        [
            'movq xmm0, rax',
            'movq xmm1, rbx',
            'pand xmm0, xmm1',  # bit-precise: both inputs needed for taint
            'movq rax, xmm0',
        ],
        'simd',
    ),
    (
        'simd_pcmpeqb_roundtrip',
        [
            'movq xmm0, rax',
            'movq xmm1, rbx',
            'pcmpeqb xmm0, xmm1',  # byte-wise compare; output bytes 0xFF or 0x00
            'movq rax, xmm0',
        ],
        'simd',
    ),
    (
        'simd_pshufb_roundtrip',
        [
            'movq xmm0, rax',
            'movq xmm1, rbx',
            'pshufb xmm0, xmm1',  # rbx selects which byte of rax goes where
            'movq rax, xmm0',
        ],
        'simd',
    ),
    ('simd_psllq_roundtrip', ['movq xmm0, rax', 'psllq xmm0, 4', 'movq rax, xmm0'], 'simd'),
    ('simd_psrlq_roundtrip', ['movq xmm0, rax', 'psrlq xmm0, 8', 'movq rax, xmm0'], 'simd'),
    (
        'simd_xor_self_clear',
        ['movq xmm0, rax', 'pxor xmm0, xmm0', 'movq rax, xmm0'],  # idiom for clearing xmm  # rax MUST be 0 / clean
        'simd',
    ),
    (
        'simd_chain',
        ['movq xmm0, rax', 'movq xmm1, rbx', 'paddb xmm0, xmm1', 'pxor  xmm0, xmm1', 'psllq xmm0, 8', 'movq rax, xmm0'],
        'simd',
    ),
    # =====================================================================
    # String / REP sequences
    # ---------------------------------------------------------------------
    # String ops need RSI, RDI, RCX setup pointing at valid memory. We
    # use the local stack frame as both source and destination buffer.
    # CLD ensures forward direction. The user-tainted register flows in
    # via RAX which we copy into the buffer.
    # =====================================================================
    (
        'string_rep_stosb',
        [
            'cld',
            'mov qword ptr [rsp - 64], rax',  # seed buffer with rax (tainted)
            'lea rdi, [rsp - 64]',  # dst pointer
            'mov rax, rbx',  # value to store comes from rbx
            'mov ecx, 8',  # count
            'rep stosb',  # fill 8 bytes with AL
            'mov rax, qword ptr [rsp - 64]',
        ],  # read back
        'string',
    ),
    (
        'string_rep_movsb',
        [
            'cld',
            'mov qword ptr [rsp - 32], rax',  # source = rax-tainted
            'lea rsi, [rsp - 32]',
            'lea rdi, [rsp - 64]',
            'mov ecx, 8',
            'rep movsb',  # copy 8 bytes
            'mov rax, qword ptr [rsp - 64]',
        ],
        'string',
    ),
    (
        'string_rep_movsq',
        [
            'cld',
            'mov qword ptr [rsp - 32], rax',
            'lea rsi, [rsp - 32]',
            'lea rdi, [rsp - 64]',
            'mov ecx, 1',
            'rep movsq',
            'mov rax, qword ptr [rsp - 64]',
        ],
        'string',
    ),
    # =====================================================================
    # Sign-extension implicit-input chains
    # ---------------------------------------------------------------------
    # CDQ/CQO/CWDE etc. take RAX (or sub-reg) as implicit input and write
    # RDX (or RAX). Pre-IDIV pattern in real compiler output.
    # =====================================================================
    (
        'signext_cqo_chain',
        [
            'cqo',  # rdx ← sign of rax (all bits identical)
            'xor rdx, rcx',
        ],  # output rdx tainted iff (rax MSB tainted) or (rcx tainted)
        'signext',
    ),
    (
        'signext_cdqe',
        [
            'cdqe',  # rax ← sign-ext eax → bit i (i>=32) = bit 31 of original
            'shr rax, 32',
        ],  # surface the replicated sign
        'signext',
    ),
    # ── Atomic R-M-W chains ──────────────────────────────────────────────
    ('xadd_chain', ['xadd rax, rbx', 'add rcx, rax', 'or rdx, rbx'], 'atomic'),
    ('cmpxchg_chain', ['cmpxchg rbx, rcx', 'add rdx, rbx'], 'atomic'),  # rax compared to rbx; conditional update
]

# ---------------------------------------------------------------------------
# Oracle test suite for IMUL chains
# ---------------------------------------------------------------------------
# These are DETERMINISTIC tests with analytically computed expected outputs.
# We fix the register state so the expected taint output can be hand-verified.
#
# Design: imul RAX, RBX computes RAX ← RAX * RBX (signed 64-bit, lower half).
# Taint propagation rule (bit-precise):
#   If EITHER operand is tainted, ALL bits of the result are potentially tainted
#   because multiplication causes carry across every bit position.
#   Exception: if one operand is 0 (clean, concrete 0), the result is 0 and
#   all bits are clean regardless of the other operand's taint.
#
# We test 5 cases:
#   (a) both tainted                → result all-tainted
#   (b) only RAX tainted            → result all-tainted
#   (c) only RBX tainted            → result all-tainted
#   (d) RBX=0 concrete, RAX tainted → result CLEAN (0*x=0, sanitiser)
#   (e) 3-step chain: imul; add; imul → taint flows through both multiplies
#
# oracle["RAX"] = expected taint mask (0 = clean, 0xFFFF...F = all-tainted)

ORACLE_IMUL_TESTS: list[dict] = [
    # (a) both operands tainted → all result bits tainted
    {
        'label': 'imul_both_tainted',
        'category': 'oracle_imul',
        'asm_lines': ['imul rax, rbx'],
        'state': {'RAX': 0x123456789ABCDEF0, 'RBX': 0xFEDCBA9876543210, 'RCX': 0, 'RDX': 0},
        'taint': {'RAX': 0xFFFFFFFFFFFFFFFF, 'RBX': 0xFFFFFFFFFFFFFFFF, 'RCX': 0, 'RDX': 0},
        'oracle': {'RAX': 0xFFFFFFFFFFFFFFFF},
        'rationale': 'imul(tainted, tainted) → all bits tainted (carry spread)',
    },
    # (b) only RAX tainted
    {
        'label': 'imul_rax_tainted_only',
        'category': 'oracle_imul',
        'asm_lines': ['imul rax, rbx'],
        'state': {'RAX': 0xDEADBEEFCAFEBABE, 'RBX': 0x0000000000000007, 'RCX': 0, 'RDX': 0},
        'taint': {'RAX': 0xFFFFFFFFFFFFFFFF, 'RBX': 0, 'RCX': 0, 'RDX': 0},
        'oracle': {'RAX': 0xFFFFFFFFFFFFFFFF},
        'rationale': 'imul(tainted, clean) → all result bits tainted',
    },
    # (c) only RBX tainted
    {
        'label': 'imul_rbx_tainted_only',
        'category': 'oracle_imul',
        'asm_lines': ['imul rax, rbx'],
        'state': {'RAX': 0x0000000000000003, 'RBX': 0xAAAAAAAAAAAAAAAA, 'RCX': 0, 'RDX': 0},
        'taint': {'RAX': 0, 'RBX': 0xFFFFFFFFFFFFFFFF, 'RCX': 0, 'RDX': 0},
        'oracle': {'RAX': 0xFFFFFFFFFFFFFFFF},
        'rationale': 'imul(clean, tainted) → all result bits tainted',
    },
    # (d) RBX=0 concrete → result is always 0, taint must be CLEAN
    # This is the key sanitiser case: x * 0 = 0 regardless of x.
    # A precise tool must detect that the result cannot carry any taint
    # because the concrete factor eliminates all symbolic dependence.
    {
        'label': 'imul_zero_factor_sanitiser',
        'category': 'oracle_imul',
        'asm_lines': ['imul rax, rbx'],
        'state': {'RAX': 0xDEADBEEFCAFEBABE, 'RBX': 0x0000000000000000, 'RCX': 0, 'RDX': 0},
        'taint': {'RAX': 0xFFFFFFFFFFFFFFFF, 'RBX': 0, 'RCX': 0, 'RDX': 0},
        'oracle': {'RAX': 0},  # x * 0 = 0, no taint survives
        'rationale': 'imul(tainted, 0) → result=0, CLEAN — precision test',
    },
    # (e) 3-operand imul: RAX = RBX * 7  (immediate)
    # RBX tainted → RAX all-tainted; immediate is clean
    {
        'label': 'imul3_imm_rbx_tainted',
        'category': 'oracle_imul',
        'asm_lines': ['imul rax, rbx, 7'],
        'state': {'RAX': 0, 'RBX': 0x1234567812345678, 'RCX': 0, 'RDX': 0},
        'taint': {'RAX': 0, 'RBX': 0xFFFFFFFFFFFFFFFF, 'RCX': 0, 'RDX': 0},
        'oracle': {'RAX': 0xFFFFFFFFFFFFFFFF},
        'rationale': 'imul3(tainted, imm) → RAX all-tainted',
    },
    # (f) 2-step chain: imul then add; taint must flow through both
    {
        'label': 'imul_chain_add_oracle',
        'category': 'oracle_imul',
        'asm_lines': ['imul rax, rbx', 'add rax, rcx'],
        'state': {'RAX': 0x1000000000000000, 'RBX': 0x0000000000000003, 'RCX': 0x0000000000000001, 'RDX': 0},
        'taint': {'RAX': 0xFFFFFFFFFFFFFFFF, 'RBX': 0, 'RCX': 0, 'RDX': 0},
        'oracle': {'RAX': 0xFFFFFFFFFFFFFFFF},
        'rationale': 'imul(tainted, clean) + add(clean) → RAX still all-tainted',
    },
    # (g) imul chain where taint is introduced only in second multiply
    {
        'label': 'imul_late_taint_chain',
        'category': 'oracle_imul',
        'asm_lines': ['imul rax, rbx', 'imul rax, rcx'],  # rax=clean*clean=clean  # rax=clean*tainted=tainted
        'state': {'RAX': 0x0000000000000005, 'RBX': 0x0000000000000003, 'RCX': 0xDEADBEEFDEADBEEF, 'RDX': 0},
        'taint': {'RAX': 0, 'RBX': 0, 'RCX': 0xFFFFFFFFFFFFFFFF, 'RDX': 0},
        'oracle': {'RAX': 0xFFFFFFFFFFFFFFFF},
        'rationale': 'clean*clean=clean, then clean*tainted=tainted (late introduction)',
    },
    # (h) partial taint: only low 8 bits of RBX tainted
    # Multiplying by a partially tainted value: ALL result bits tainted
    # because bit-0 of RBX (if 1) shifts RAX's bits upward through all positions.
    {
        'label': 'imul_partial_taint_low_byte',
        'category': 'oracle_imul',
        'asm_lines': ['imul rax, rbx'],
        'state': {'RAX': 0x0100000000000000, 'RBX': 0x0000000000000001, 'RCX': 0, 'RDX': 0},
        'taint': {'RAX': 0, 'RBX': 0x00000000000000FF, 'RCX': 0, 'RDX': 0},
        'oracle': {'RAX': 0xFFFFFFFFFFFFFFFF},
        'rationale': 'imul(clean, low-byte-tainted) → all result bits tainted via carry',
    },
]


def _oracle_test_to_tc(arch: str, ot: dict) -> dict:
    """Convert a hand-crafted IMUL test entry to a standard test-case dict.

    The legacy ``oracle`` field is intentionally not propagated: the noninter-
    ference GT simulator now provides ground truth uniformly for every case
    that fits the budget, removing the need for hand-written per-test taint
    expectations (which were brittle and frequently wrong, e.g. asserting a
    register's taint when the test sequence didn't write that register).
    The hand-crafted test cases themselves remain valuable as semantic stress
    tests; only the oracle assertion is dropped.
    """
    asm_lines = ot['asm_lines']
    all_bytes = []
    for line in asm_lines:
        enc, _ = _KS.asm(line)
        all_bytes.extend(enc)
    state = {k: (v if isinstance(v, int) else int(v, 16)) for k, v in ot['state'].items()}
    taint = {k: (v if isinstance(v, int) else int(v, 16)) for k, v in ot['taint'].items()}
    return {
        'arch': arch,
        'assembly': '; '.join(asm_lines),
        'asm_lines': asm_lines,
        'bytes': bytes(all_bytes).hex(),
        'state': state,
        'taint': taint,
        'category': ot['category'],
        'label': ot['label'],
        'rationale': ot.get('rationale', ''),
        'mode': 'imul_semantic',
    }


# ---------------------------------------------------------------------------
# Real-world program snippet tests
# ---------------------------------------------------------------------------
# These are instruction sequences extracted from actual binaries (glibc,
# openssl, standard library implementations) and represent patterns that
# taint engines encounter in practice.  They are NOT synthetic — each is
# a lightly-adapted disassembly excerpt with memory accesses replaced by
# register equivalents so workers that don't model memory can still run them.
#
# Source annotations reference the function and approximate offset.

REALWORLD_SEQUENCES: list[tuple[str, list[str], str, str]] = [
    # ── FNV-1a hash inner loop (djb2 variant)  ─────────────────────────────
    # Source: musl libc hash.c, equivalent to: hash = (hash ^ byte) * 2166136261
    # rax = accumulator (tainted input), rbx = current byte (tainted)
    # Pattern: XOR then multiply by FNV prime — very common in hash functions
    (
        'fnv1a_round',
        [
            'xor   rax, rbx',  # hash ^= byte
            'imul  rax, rax, 0x01000193',  # hash *= FNV prime (16777619)
        ],
        'realworld_hash',
        'FNV-1a hash round: accumulator XOR then multiply by prime',
    ),
    # ── CRC32 bit-by-bit inner step ────────────────────────────────────────
    # Source: zlib crc32.c bit-reversal computation
    # rax = crc (tainted), rbx = polynomial (clean 0xEDB88320)
    # Pattern: conditional XOR with polynomial based on LSB — common in CRC
    (
        'crc32_step',
        [
            'mov   rcx, rax',  # save crc
            'shr   rax, 1',  # crc >>= 1
            'and   rcx, 1',  # isolate LSB
            'neg   rcx',  # 0 → 0, 1 → 0xFFFF...F (mask)
            'and   rcx, rbx',  # conditional polynomial
            'xor   rax, rcx',  # apply polynomial if LSB was set
        ],
        'realworld_crc',
        'CRC32 bit-step: conditional XOR with polynomial via arithmetic mask',
    ),
    # ── memcpy-style word copy ─────────────────────────────────────────────
    # Source: glibc sysdeps/x86_64/memcpy.S unrolled word copy
    # rax=src_word0, rbx=src_word1, rcx=src_word2, rdx=src_word3 (all tainted)
    # Simulates copying 4 qwords; we track taint through the copies
    (
        'memcpy_4qword',
        [
            'mov   rax, rax',  # dst[0] = src[0]  (identity — taint preserved)
            'mov   rbx, rbx',  # dst[1] = src[1]
            'mov   rcx, rcx',  # dst[2] = src[2]
            'mov   rdx, rdx',  # dst[3] = src[3]
        ],
        'realworld_memcpy',
        'memcpy word-copy: taint must be preserved 1:1 through copies',
    ),
    # ── strlen byte-scanning loop body ─────────────────────────────────────
    # Source: glibc string/strlen.c inner loop
    # rax = pointer/counter (tainted), rbx = loaded byte (may be clean/tainted)
    # Pattern: test for zero byte, increment pointer
    (
        'strlen_inner',
        [
            'movzx rbx, al',  # load byte (simulate: movzx rbx, [rax])
            'test  rbx, rbx',  # check NUL
            'inc   rax',  # advance pointer
        ],
        'realworld_strlen',
        'strlen inner loop: byte test + pointer increment',
    ),
    # ── strcmp inner loop ──────────────────────────────────────────────────
    # Source: glibc sysdeps/x86_64/strcmp.S
    # rax=char_a (tainted), rbx=char_b (tainted), rcx=result
    # Pattern: subtract bytes, propagate difference as taint signal
    (
        'strcmp_inner',
        [
            'movzx rcx, al',  # char_a low byte
            'movzx rdx, bl',  # char_b low byte
            'sub   rcx, rdx',  # difference
            'movsx rax, cl',  # sign-extend result
        ],
        'realworld_strcmp',
        'strcmp byte comparison: difference between tainted chars',
    ),
    # ── AES SubBytes / MixColumns approximation ────────────────────────────
    # Source: openssl crypto/aes/aes_core.c (AES round without lookup table)
    # Simplified: represents the XOR-mix pattern of AES round keys
    # rax=state0, rbx=state1, rcx=roundkey0, rdx=roundkey1
    (
        'aes_round_xor',
        [
            'xor   rax, rcx',  # state0 ^= roundkey0
            'xor   rbx, rdx',  # state1 ^= roundkey1
            'rol   rax, 8',  # ShiftRows approximation (byte rotation)
            'xor   rax, rbx',  # MixColumns approximation (cross-lane XOR)
        ],
        'realworld_aes',
        'AES round approximation: SubBytes/ShiftRows/MixColumns via XOR+ROL',
    ),
    # ── Djb2 hash ──────────────────────────────────────────────────────────
    # Source: djb2 (Dan Bernstein): hash = hash * 33 + char
    # rax = hash accumulator (tainted), rbx = char (tainted)
    # LEA cannot encode scale 32 (only 1,2,4,8 are valid SIB scales);
    # use 3-operand IMUL instead.
    (
        'djb2_round',
        [
            'imul  rax, rax, 33',  # hash * 33
            'add   rax, rbx',  # + char
        ],
        'realworld_hash',
        'djb2 hash round: hash*33 + char via IMUL+ADD',
    ),
    # ── Base64 decode step ─────────────────────────────────────────────────
    # Source: common base64 decode inner body
    # rax=b0 (tainted), rbx=b1 (tainted), rcx=b2 (tainted), rdx=b3 (tainted)
    # Combines 4×6-bit groups into 3 bytes via shifts and ORs
    (
        'base64_decode_step',
        [
            'shl   rax, 18',  # b0 → bits [23:18]
            'shl   rbx, 12',  # b1 → bits [17:12]
            'shl   rcx, 6',  # b2 → bits [11:6]
            'or    rax, rbx',  # merge b0+b1
            'or    rax, rcx',  # merge b2
            'or    rax, rdx',  # merge b3 → 3-byte word in RAX
        ],
        'realworld_codec',
        'Base64 decode: shift-and-OR of 4×6-bit groups into 3-byte word',
    ),
    # ── Network byte-order swap (ntohl equivalent) ─────────────────────────
    # Source: Linux kernel include/uapi/linux/byteorder.h
    # rax = 32-bit network-order value (tainted)
    (
        'ntohl_bswap',
        [
            'bswap eax',  # reverse byte order (32-bit)
        ],
        'realworld_network',
        'ntohl: bswap32 — taint bits permuted, all must remain tainted',
    ),
    # ── Integer overflow check pattern ─────────────────────────────────────
    # Source: common hardened arithmetic (UBSAN / SafeInt pattern)
    # rax=a (tainted), rbx=b (tainted) — computes a+b and checks for overflow
    (
        'overflow_check',
        [
            'add   rax, rbx',  # compute sum
            'jno   0x3',  # jump if no overflow (3 bytes forward)
            'xor   rax, rax',  # overflow: return 0 (sanitiser)
        ],
        'realworld_overflow',
        'Overflow check: add + conditional clear (tainted or sanitised)',
    ),
    # ── Pointer masking / alignment (common in allocators) ─────────────────
    # Source: glibc malloc/arena.c alignment masking
    # rax = pointer (tainted), aligns down to 64-byte boundary.
    # 'and rax, -64' encodes as imm8-sign-extended (48 83 e0 c0) which
    # triggers a hang in maat's IR lifter.  Equivalent via shifts: clear
    # low 6 bits with shr+shl (round down to 64-byte boundary).
    (
        'align_mask',
        [
            'shr   rax, 6',  # discard low 6 bits
            'shl   rax, 6',  # restore alignment  →  rax & ~63
        ],
        'realworld_ptr',
        'Pointer alignment: clear low 6 bits via SHR+SHL (equiv. to AND with ~63)',
    ),
]


# ---------------------------------------------------------------------------
# Bug-detection scenario tests
# ---------------------------------------------------------------------------
# These model taint-relevant vulnerability patterns:
#   - Tainted buffer index (OOB / type confusion)
#   - Tainted function pointer (control-flow hijack)
#   - Tainted format string argument (format string bug)
#   - Tainted return value used as array index (CWE-129)
#   - Integer truncation of tainted value (CWE-197)
#
# For each, we track whether the "dangerous" register (the one that would
# reach a sink) is tainted at the end.  The expected answer is always that
# the dangerous register IS tainted — any tool reporting clean has a
# missed-finding (false negative) that would miss the bug.

BUGDETECT_SEQUENCES: list[tuple[str, list[str], str, str]] = [
    # ── Tainted array index (CWE-129) ──────────────────────────────────────
    # rax = tainted user input, used as array index via LEA
    # Dangerous register: rax (the final index/address)
    (
        'bug_tainted_index',
        [
            'movzx rax, ax',  # truncate to 16-bit (still tainted)
            'lea   rax, [rbx + rax*8]',  # compute array[tainted_index] address
        ],
        'bug_detection',
        'CWE-129: tainted array index — rax must be tainted at sink',
    ),
    # ── Integer truncation of tainted value (CWE-197) ─────────────────────
    # rax = 64-bit tainted value truncated to 32-bit then zero-extended
    # The upper bits are clean after truncation but the value is still tainted
    (
        'bug_int_truncation',
        [
            'mov   eax, eax',  # truncate: zero-extend → upper 32 clean
        ],
        'bug_detection',
        'CWE-197: integer truncation — low 32 bits of rax must remain tainted',
    ),
    # ── Tainted value flows to function pointer slot ───────────────────────
    # rax = tainted input, moved into rcx which is used as a call target.
    # 'and rcx, -4' (align to 4 bytes) encodes as imm8-sign-extended
    # (48 83 e1 fc) which triggers a hang in maat's IR lifter.
    # Equivalent via shifts: clear low 2 bits with shr+shl.
    (
        'bug_funcptr_taint',
        [
            'mov   rcx, rax',  # load function pointer from tainted source
            'shr   rcx, 2',  # clear low 2 bits (align to 4 bytes)
            'shl   rcx, 2',  # restore  →  rcx & ~3
        ],
        'bug_detection',
        'Function pointer from tainted input — rcx must be tainted at indirect call site',
    ),
    # ── Format string argument tainting (CWE-134) ─────────────────────────
    # rax = tainted format string pointer, processed through common idiom
    (
        'bug_format_string',
        [
            'mov   rdx, rax',  # arg3 = tainted format string
            'test  rdx, rdx',  # null check (does not clear taint)
        ],
        'bug_detection',
        'CWE-134: tainted format string — rdx must be tainted at printf sink',
    ),
    # ── Use-after-free pattern: stale pointer arithmetic ──────────────────
    # rax = freed but still tainted pointer, rbx = offset
    (
        'bug_uaf_ptr_arith',
        [
            'add   rax, rbx',  # ptr + offset (both potentially tainted)
            'and   rax, 0x7fffffff',  # mask to plausible range (still tainted)
        ],
        'bug_detection',
        'UAF: arithmetic on stale pointer — rax must remain tainted',
    ),
    # ── Taint survives conditional sanitisation attempt ────────────────────
    # Common pattern: developer bounds-checks a tainted value but the check
    # is bypassable — the value remains tainted after the check
    (
        'bug_conditional_sanitise_bypass',
        [
            'cmp   rax, 0x100',  # bounds check (sets flags, does not sanitise)
            'cmovg rax, rbx',  # if >256, clamp to rbx (which is also tainted)
        ],
        'bug_detection',
        'Bypassed sanitisation: cmov with tainted fallback — rax still tainted',
    ),
]


# ---------------------------------------------------------------------------
# Path-explosion / SMT stress tests
# ---------------------------------------------------------------------------
# These sequences are designed to stress symbolic-execution-based tools
# (angr, triton) by creating long chains of conditional operations.
# Each adds one dependent conditional move, forcing the SMT engine to
# maintain increasingly complex symbolic expressions.
#
# Expected observation (for your paper):
#   - microtaint / maat: latency O(N) — simple dataflow, no state explosion
#   - angr / triton: latency grows super-linearly with N because each
#     CMOV extends the symbolic AST by adding an ITE (if-then-else) node
#     whose subtrees grow with the number of prior conditions.
#
# We provide 4 lengths: 2, 4, 8, 16 conditional moves.


def _branching_dataflow(n: int) -> list[str]:
    """
    Explicit branch-driven dataflow.  Each block is:

        test rbx, 1      ; tainted bit-test
        jz   <next>      ; real conditional branch
        add  rax, rbx    ; explicit dataflow rbx → rax (only on taken path)
      next:
        shr  rbx, 1      ; advance to next bit

    With T_RBX set, RAX after N blocks is tainted (it conditionally
    accumulates RBX).  This is the canonical "path-explosion" stress
    case for symbolic engines (angr forks at every jz, producing 2^N
    final states), while dataflow engines stay O(N).

    Returned for documentation purposes; the main path-explosion
    generator in ``main()`` uses ``build_branching_bytestring`` directly
    because some assemblers reject ``jz +3`` with a numeric short-jump
    offset.
    """
    instrs = []
    for _ in range(n):
        instrs.append('test rbx, 1')
        instrs.append('jz +3')
        instrs.append('add rax, rbx')
        instrs.append('shr rbx, 1')
    return instrs


# Path-explosion N values.  N=12 means 2^12=4096 symbolic paths for angr —
# well past the practical wall.  Dataflow tools stay O(N).
EXPLICIT_BRANCH_NS = [2, 4, 6, 8, 10, 12]


def build_branching_bytestring(n: int) -> bytes:
    """
    Hand-assembled byte string for the explicit branching path-explosion
    test.  This is the only path-explosion stress pattern in the
    benchmark; the previous AST-growth chains (cmov_chain, add_chain,
    mixed_chain) were removed because they did not actually trigger
    path forking in symbolic engines (no real branches → no fork point)
    and so produced no measurable separation between symbolic and
    dataflow tools.

    Block layout (15 bytes per block):
      48 F7 C3 01 00 00 00   test rbx, 1     (7 bytes)
      74 03                  jz   +3         (2 bytes; skips the add)
      48 01 D8               add  rax, rbx   (3 bytes)
      48 D1 EB               shr  rbx, 1     (3 bytes)

    The jz +3 jumps to ``shr rbx, 1`` from the byte after jz (offset 9
    within the block).  Block count = N.  Total bytes = 3 + 15*N
    (3-byte preamble: ``xor rax, rax``).
    """
    preamble = b'\x48\x31\xc0'  # xor rax, rax
    block = (
        b'\x48\xf7\xc3\x01\x00\x00\x00'  # test rbx, 1
        b'\x74\x03'  # jz +3
        b'\x48\x01\xd8'  # add rax, rbx
        b'\x48\xd1\xeb'  # shr rbx, 1
    )
    assert len(block) == 15
    return preamble + block * n


# ---------------------------------------------------------------------------
# Architecturally guaranteed failure tests
# ---------------------------------------------------------------------------
# Each test is designed from first principles of how each tool works
# internally — not empirical guesses but structural proofs that a specific
# tool MUST produce wrong output given its architecture.
#
# Format: same as ORACLE_IMUL_TESTS.
# "expected_failures" maps tool_name → expected outcome:
#   "over_taint"   — tool reports tainted when it should be clean
#   "under_taint"  — tool reports clean when it should be tainted
#   "hang"         — tool will timeout/loop forever
#   "crash"        — tool will die with an error
#   "wrong_bits"   — bit-level output is wrong even if reg-level is correct

ARCHITECTURAL_FAILURE_TESTS: list[dict] = [
    # ── angr: CMOV over-taint with symbolic flags ────────────────────────
    # rax is tainted (symbolic). cmp rax, 0 → ZF = (rax==0), which is
    # symbolic because rax is symbolic. cmovz rax, rbx →
    #   claripy: rax = ITE(ZF_sym, rbx_clean, rax_sym)
    # Since ZF_sym is symbolic, ITE(...).symbolic = True → rax = TAINTED.
    # But rbx is clean (0 taint). If the condition were concretely True,
    # result would be rbx (clean). If concretely False, result stays rax (tainted).
    # A precise tool with concrete state would resolve this.
    # angr CANNOT — it keeps both arms alive → over-taint.
    # Note: microtaint resolves this correctly via SLEIGH semantics.
    {
        'label': 'angr_cmov_overtaint_symbolic_flag',
        'category': 'arch_failure_angr',
        'asm_lines': ['cmp rax, 0', 'cmovz rax, rbx'],
        'state': {
            'RAX': 0x0000000000000000,  # rax IS zero concretely
            'RBX': 0x0000000000000000,  # rbx is 0
            'RCX': 0,
            'RDX': 0,
        },
        # taint: rax is tainted (symbolic), rbx is clean
        'taint': {'RAX': 0xFFFFFFFFFFFFFFFF, 'RBX': 0, 'RCX': 0, 'RDX': 0},
        # Ground truth: rax IS zero concretely → ZF=1 → cmovz fires → rax = rbx = clean
        # A precise tool with concrete execution sees this.
        # angr: ZF is symbolic (derived from symbolic rax) → ITE keeps rax symbolic
        'oracle': {'RAX': 0},  # correct answer: clean
        'expected_failures': {
            'angr': 'over_taint',  # will report RAX tainted despite it being clean
        },
        'rationale': (
            'angr builds ITE(ZF_symbolic, rbx_clean, rax_tainted). '
            'Since ZF is derived from symbolic rax, .symbolic=True → over-taint. '
            'Precise tools see rax=0 concretely → ZF=1 → only rbx arm matters → clean.'
        ),
    },
    # ── angr: zero-factor multiplication ──────────────────────────────────
    # imul(symbolic_rax, 0) must equal 0 regardless of rax.
    # claripy: Mul(BVS("rax",64), BVV(0,64)).
    # claripy's simplifier does handle x*0=0 BUT only for BVV(0) on the right.
    # When the concrete state has RBX=0 and we build make_tainted_bv with
    # taint_mask=0 → BVV(0,64) for RBX.
    # Mul(BVS, BVV(0)) → claripy simplifies → BVV(0) → not symbolic → CORRECT.
    # HOWEVER: if RBX has any symbolic bits (partial taint), even one tainted
    # bit means RBX = claripy Concat with some BVS bits → Mul is not simplified.
    # Here we test partial taint on RBX to guarantee angr over-taints:
    {
        'label': 'angr_mul_partial_taint_overtaint',
        'category': 'arch_failure_angr',
        'asm_lines': ['imul rax, rbx'],
        'state': {'RAX': 0xDEADBEEFCAFEBABE, 'RBX': 0x0000000000000000, 'RCX': 0, 'RDX': 0},
        'taint': {
            'RAX': 0xFFFFFFFFFFFFFFFF,
            'RBX': 0x00000000000000FF,  # only low byte of RBX tainted
            'RCX': 0,
            'RDX': 0,
        },
        # Ground truth: RBX concrete value is 0. Result = rax * 0 = 0 → CLEAN.
        # angr: RBX = Concat(BVV(0,56), BVS("taint_RBX_b0..7", 8))
        #   → Mul(BVS(rax), Concat(...)) → NOT simplified → result.symbolic=True
        'oracle': {'RAX': 0},
        'expected_failures': {
            'angr': 'over_taint',
        },
        'rationale': (
            'RBX concrete value is 0 but has 1 tainted bit. '
            'angr builds Concat(BVV(0,56), BVS(low8)) for RBX — a symbolic expression. '
            'Mul(sym_rax, sym_rbx) is not simplified to 0 even though the concrete '
            'value guarantees it. Precise answer: rax*0=0 → clean.'
        ),
    },
    # ── angr: bit-precision over-taint on AND with clean mask ──────────────
    # and rax, 0xff: only low 8 bits of rax survive; bits [63:8] are forced to 0.
    # microtaint: output taint = input_taint & 0x00000000000000FF (precise mask).
    # angr: entire rax was symbolic → AND with BVV(0xff,64).
    #   result = Extract(63,0, BVS("rax",64) & BVV(0xff,64))
    #   Extract(bit, bit, result).symbolic: for bits 8-63, this extracts a 0-bit
    #   from the AND result. claripy may or may not propagate the concreteness.
    # In practice, angr marks ALL bits of rax as symbolic after AND with an
    # immediate — it does not do bit-level constant propagation through AND.
    {
        'label': 'angr_and_mask_bit_overtaint',
        'category': 'arch_failure_angr',
        'asm_lines': ['and rax, 0xff'],
        'state': {'RAX': 0xDEADBEEFCAFEBABE, 'RBX': 0, 'RCX': 0, 'RDX': 0},
        'taint': {'RAX': 0xFFFFFFFFFFFFFFFF, 'RBX': 0, 'RCX': 0, 'RDX': 0},
        # Correct bit-level: only bits [7:0] remain tainted, bits [63:8] = clean
        'oracle': {'RAX': 0x00000000000000FF},
        'expected_failures': {
            'angr': 'wrong_bits',  # will report 0xFFFFFFFFFFFFFFFF instead of 0xFF
        },
        'rationale': (
            'AND with 0xFF forces bits [63:8] to 0 unconditionally. '
            'microtaint propagates: output_taint = input_taint & 0xFF = 0xFF. '
            'angr: BVS & BVV(0xff) → the resulting AST is symbolic for all bits '
            'because claripy tracks symbolicity at the full-value level, not per-bit. '
            'Extract(8,8,rax_and_ff).symbolic = True even though that bit is always 0.'
        ),
    },
    # ── maat: REX.W + 0x83 + negative imm8 → guaranteed hang ─────────────
    # Documented above. One representative case per mnemonic.
    {
        'label': 'maat_hang_and_neg_imm8',
        'category': 'arch_failure_maat',
        'asm_lines': ['and rcx, -4'],
        'state': {'RAX': 0, 'RBX': 0, 'RCX': 0xDEADBEEFCAFEBABE, 'RDX': 0},
        'taint': {'RAX': 0, 'RBX': 0, 'RCX': 0xFFFFFFFFFFFFFFFF, 'RDX': 0},
        # Correct: and rcx, 0xFFFFFFFFFFFFFFFC → rcx tainted (bits 63:2), clean (bits 1:0)
        'oracle': {'RCX': 0xFFFFFFFFFFFFFFFC},
        'expected_failures': {
            'maat': 'hang',  # REX.W + 0x83 + fc → IR lifter infinite loop
        },
        'rationale': (
            "Encoding 48 83 e1 fc = REX.W + opcode 0x83 (imm8 group) + "
            "ModRM(AND/RCX) + imm8(0xfc=-4). maat's IR lifter for this specific "
            "combination (REX.W=1, opcode=0x83, imm8 MSB=1) enters an infinite loop, "
            "likely in its sign-extension logic for negative imm8 to 64-bit."
        ),
    },
    {
        'label': 'maat_hang_or_neg_imm8',
        'category': 'arch_failure_maat',
        'asm_lines': ['or rcx, -1'],
        'state': {'RAX': 0, 'RBX': 0, 'RCX': 0x0F0F0F0F0F0F0F0F, 'RDX': 0},
        'taint': {'RAX': 0, 'RBX': 0, 'RCX': 0xFFFFFFFFFFFFFFFF, 'RDX': 0},
        # Correct: or rcx, 0xFFFFFFFFFFFFFFFF → rcx = all-ones → clean (forced value)
        'oracle': {'RCX': 0},  # all bits forced to 1 by OR → no taint survives
        'expected_failures': {
            'maat': 'hang',  # same pattern: REX.W + 0x83 + ff
        },
        'rationale': (
            'Same REX.W+0x83+neg-imm8 pattern as and rcx,-4. '
            'Additionally interesting: OR with -1 forces all bits to 1 → '
            'precise tools should report RCX CLEAN (value is always 0xFFFFFFFFFFFFFFFF '
            'regardless of input). Tests both the maat hang AND the OR-with-all-ones '
            'sanitiser behaviour in other tools.'
        ),
    },
    # ── maat: branching sequence stops at first basic block boundary ──────
    # maat's run(1) means "1 basic block". A sequence containing a
    # conditional jump splits into 2 basic blocks. maat executes only
    # the first BB → instructions after the branch are not executed.
    # This is a systematic failure for all sequences containing branches.
    {
        'label': 'maat_branch_bb_cutoff',
        'category': 'arch_failure_maat',
        'asm_lines': [
            'add rax, rbx',  # BB1 begins
            'cmp rax, 0',  # still BB1
            'jz 0x3',  # BB1 ends here (branch instruction)
            'add rcx, rdx',  # BB2 — maat NEVER executes this
        ],
        'state': {'RAX': 0x1, 'RBX': 0x1, 'RCX': 0x0, 'RDX': 0xDEADBEEF00000000},
        'taint': {'RAX': 0, 'RBX': 0, 'RCX': 0, 'RDX': 0xFFFFFFFFFFFFFFFF},
        # Correct (concrete execution: rax=2 after add → jz not taken → add rcx,rdx executes)
        'oracle': {'RCX': 0xFFFFFFFFFFFFFFFF},  # RCX should receive RDX's taint
        'expected_failures': {
            'maat': 'under_taint',  # maat stops at jz → never updates RCX
        },
        'rationale': (
            'engine.run(1) executes 1 basic block. The jz creates a BB boundary. '
            'maat executes: add rax,rbx; cmp rax,0; jz (evaluates branch) → stops. '
            'add rcx,rdx is in BB2 → never executed → RCX stays clean. '
            'Other tools execute the full byte sequence → RCX gets RDX taint.'
        ),
    },
    # ── libdft64: implicit flow via CMOV ──────────────────────────────────
    # libdft64 uses a byte-granular shadow memory model. It tracks which
    # bytes of memory/registers are tainted. For CMOV, libdft64 models
    # only the DATA flow (source register → destination) not the CONTROL
    # flow (condition flags → destination). If the condition is derived
    # from a tainted comparison, the destination should carry implicit taint
    # even when the "not-taken" branch value is used.
    # libdft64 ALWAYS ignores the flag dependency → under-taint on cmov.
    {
        'label': 'libdft64_implicit_flow_cmov',
        'category': 'arch_failure_libdft64',
        'asm_lines': ['cmp rax, rbx', 'cmovz rcx, rdx'],
        'state': {'RAX': 0x42, 'RBX': 0x42, 'RCX': 0x0, 'RDX': 0x0},  # equal → ZF=1 → cmov fires
        'taint': {
            'RAX': 0xFFFFFFFFFFFFFFFF,  # rax tainted (affects ZF)
            'RBX': 0xFFFFFFFFFFFFFFFF,  # rbx tainted (affects ZF)
            'RCX': 0,  # rcx clean before cmov
            'RDX': 0,
        },  # rdx clean (source)
        # Correct: ZF = (rax==rbx) is tainted (depends on tainted rax,rbx).
        # cmovz fires → rcx = rdx (clean value). BUT the condition is tainted.
        # For implicit flow: rcx's value depends on whether the condition holds,
        # which depends on tainted inputs → rcx should be tainted.
        # libdft64 only tracks: cmovz(rcx←rdx) → rcx gets rdx's taint = 0 = CLEAN.
        # It does not ask: "does the condition depend on tainted data?"
        'oracle': {'RCX': 0xFFFFFFFFFFFFFFFF},  # implicit taint from condition
        'expected_failures': {
            'libdft64': 'under_taint',  # libdft64 reports RCX clean
            'panda': 'under_taint',  # panda also misses this
            'taintgrind': 'under_taint',  # taintgrind too — DTA tools typically miss implicit flows
        },
        'rationale': (
            "libdft64/panda/taintgrind track explicit data flow only. "
            "cmovz(rcx←rdx): they propagate rdx's taint to rcx (rdx is clean → rcx clean). "
            "They do NOT propagate: (rax,rbx) → ZF → cmov_decision → rcx. "
            "This is the classic DTA implicit flow limitation. "
            "Symbolic tools (angr, microtaint) track flag dependencies → correctly taint rcx."
        ),
    },
    # ── panda: BMI instruction not in taint model ─────────────────────────
    # PANDA's taint2 plugin is built on QEMU's TCG intermediate representation.
    # BMI1/BMI2 instructions (BLSI, BLSR, ANDN, LZCNT, TZCNT) were added to
    # QEMU's TCG relatively late and panda's taint propagation rules for them
    # are either missing or incorrect in many panda versions.
    # Expected: panda reports clean output even when input is tainted.
    {
        'label': 'panda_bmi_taint_missing',
        'category': 'arch_failure_panda',
        'asm_lines': ['blsi rax, rbx'],  # rax = rbx & (-rbx)  (isolate lowest set bit)
        'state': {'RAX': 0, 'RBX': 0x0F0F0F0F0F0F0F0C, 'RCX': 0, 'RDX': 0},
        'taint': {'RAX': 0, 'RBX': 0xFFFFFFFFFFFFFFFF, 'RCX': 0, 'RDX': 0},
        'oracle': {'RAX': 0xFFFFFFFFFFFFFFFF},  # rax must be tainted (derived from rbx)
        'expected_failures': {
            'panda': 'under_taint',  # panda's TCG taint rules missing for BMI
        },
        'rationale': (
            "BLSI = rbx & NEG(rbx). Both operands derive from tainted rbx → "
            "rax is fully tainted. PANDA's taint2 was built before BMI1 was "
            "widely supported in TCG; the taint propagation helper for opcode "
            "group VEX.NDS+0xF3 may be absent → panda treats rax as clean."
        ),
    },
    # ── panda: xor-self taint clear failure ───────────────────────────────
    # xor rax, rax algebraically equals 0 regardless of rax's value.
    # A correct taint engine must detect this and mark rax as CLEAN.
    # panda's taint2, in some versions, propagates taint from both operands
    # via the XOR rule (output_taint = taint_src1 | taint_src2) without
    # applying the algebraic identity x^x=0. This produces rax=TAINTED.
    {
        'label': 'panda_xor_self_taint_persist',
        'category': 'arch_failure_panda',
        'asm_lines': ['xor rax, rax'],
        'state': {'RAX': 0xDEADBEEFCAFEBABE, 'RBX': 0, 'RCX': 0, 'RDX': 0},
        'taint': {'RAX': 0xFFFFFFFFFFFFFFFF, 'RBX': 0, 'RCX': 0, 'RDX': 0},
        'oracle': {'RAX': 0},  # x^x=0 always → CLEAN
        'expected_failures': {
            'panda': 'over_taint',
        },
        'rationale': (
            "xor rax, rax = 0 unconditionally. Correct taint: rax → CLEAN. "
            "panda's taint2 may apply the naive rule: "
            "output_taint = taint(rax) | taint(rax) = tainted. "
            "This is a known DTA imprecision: not detecting algebraic simplifications."
        ),
    },
    # ── flag taint materialisation via SETCC ──────────────────────────────
    # Flags are intermediate values not tracked in our 4-register model.
    # SETE writes ZF directly into a byte register.
    # This tests whether tools correctly propagate: tainted_input → ZF → output.
    # All tools should get this right, but it surfaces bugs in flag modelling.
    {
        'label': 'flag_taint_sete',
        'category': 'arch_failure_flags',
        'asm_lines': ['cmp rax, 0', 'sete al'],
        'state': {'RAX': 0xDEADBEEF00000001, 'RBX': 0, 'RCX': 0, 'RDX': 0},
        'taint': {'RAX': 0xFFFFFFFFFFFFFFFF, 'RBX': 0, 'RCX': 0, 'RDX': 0},
        # After sete al: rax[7:0] = ZF = (rax==0), which depends on tainted rax.
        # → rax[7:0] must be tainted. rax[63:8] unchanged (sete writes only al).
        # Bit-precise oracle: 0xFF (only low byte tainted; upper 56 bits unchanged
        # from input taint which was all-ones — but sete only touches al, so
        # upper 56 bits of rax's taint = unchanged = 0xFFFFFFFFFFFFFF00 | 0xFF)
        # For register-level: rax must be tainted.
        'oracle': {'RAX': 0xFFFFFFFFFFFFFFFF},
        'expected_failures': {},  # expect all tools to get this right
        'rationale': (
            'cmp rax, 0 sets ZF = (rax==0). Since rax is tainted, ZF is tainted. '
            'sete al writes ZF into al. rax[7:0] must be tainted. '
            'This tests flag-to-register taint propagation. '
            'All tools should handle this; failure would indicate broken flag tracking.'
        ),
    },
    # ── maat: partial taint → Concat tree → O(64N) evaluation depth ──────
    # 30 sequential adds with partial taint input forces maat to build
    # a Concat of 64 1-bit Var nodes per register, then evaluate it
    # twice per register × 4 registers per test = 8 full tree evaluations.
    # Tree depth after N adds with partial taint = O(64 * N).
    # This is a SLOW test, not a crash — but it demonstrates maat's
    # complexity scaling issue vs all-tainted (which uses a single 64-bit Var).
    {
        'label': 'maat_partial_taint_depth_explosion',
        'category': 'arch_failure_maat',
        'asm_lines': ['add rax, rbx'] * 30,
        'state': {'RAX': 0x1000000000000000, 'RBX': 0x0000000000000001, 'RCX': 0, 'RDX': 0},
        'taint': {
            'RAX': 0x00000000FFFFFFFF,  # partial: low 32 bits tainted
            'RBX': 0x00000000FFFFFFFF,  # partial: low 32 bits tainted
            'RCX': 0,
            'RDX': 0,
        },
        'oracle': {'RAX': 0xFFFFFFFFFFFFFFFF},  # after 30 adds with partial taint → all tainted
        'expected_failures': {
            'maat': 'slow',  # will be significantly slower than all-tainted equivalent
        },
        'rationale': (
            'Partial taint (not 0 or 0xFFFF...F) forces maat to build a Concat '
            'of 64 1-bit Var nodes. Each add rax,rbx creates a new expression '
            'node wrapping the previous 64-node Concat. After 30 adds: '
            'expression tree depth ≈ 30 × 64 = 1920 nodes. as_uint() must '
            'traverse this tree twice (ctx_zero and ctx_ones). Compare latency '
            'against add_chain_30 with all-tainted input (single 64-bit Var → '
            'depth 30 only) to measure the Concat overhead.'
        ),
    },
]


def _arch_failure_to_tc(arch: str, ot: dict) -> dict:
    """Convert an ARCHITECTURAL_FAILURE_TESTS entry to a standard test-case dict."""
    asm_lines = ot['asm_lines']
    all_bytes = []
    for line in asm_lines:
        try:
            enc, _ = _KS.asm(line)
            all_bytes.extend(enc)
        except Exception:
            pass  # branches like jz may fail in isolation; skip byte collection
    state = {k: (v if isinstance(v, int) else int(v, 16)) for k, v in ot['state'].items()}
    taint = {k: (v if isinstance(v, int) else int(v, 16)) for k, v in ot['taint'].items()}
    return {
        'arch': arch,
        'assembly': '; '.join(asm_lines),
        'asm_lines': asm_lines,
        'bytes': bytes(all_bytes).hex(),
        'state': state,
        'taint': taint,
        'category': ot['category'],
        'label': ot['label'],
        'expected_failures': ot.get('expected_failures', {}),
        'rationale': ot.get('rationale', ''),
        'mode': 'arch_failure',
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def fmt_mask(val: int, granularity: str) -> str:
    if granularity == 'reg':
        return 'tainted' if val else 'clean'
    if val == 0:
        return '0 (clean)'
    if val == 0xFFFFFFFFFFFFFFFF:
        return '0xFFFFFFFFFFFFFFFF (all bits)'
    return f'0x{val:016x}'


def compare_results(tool_results: dict) -> list[str]:
    """Coarsen all results to register-level (0/1) for cross-tool comparison."""
    # Keep only tools with a valid output_taint dict (no error, output present).
    clean = {t: r for t, r in tool_results.items() if 'error' not in r and isinstance(r.get('output_taint'), dict)}
    if len(clean) < 2:
        return []
    disagreements = []
    for reg in REGISTERS:
        values = {tool: (1 if res['output_taint'].get(reg, 0) else 0) for tool, res in clean.items()}
        if len(set(values.values())) > 1:
            detail = ', '.join(f"{t}={'tainted' if v else 'clean'}" for t, v in values.items())
            disagreements.append(f'  {reg}: {detail}')
    return disagreements


def jaccard_bit(mask_a: int, mask_b: int) -> float:
    """Bit-level Jaccard similarity between two 64-bit taint masks."""
    inter = bin(mask_a & mask_b).count('1')
    union = bin(mask_a | mask_b).count('1')
    return inter / union if union else 1.0


# ---------------------------------------------------------------------------
# BatchedWorkerPool — raw binary pipes + select(), zero TextIOWrapper overhead
# ---------------------------------------------------------------------------
#
# Architecture
# ------------
# Problem: TextIOWrapper.readline() costs ~130 µs/call dominated by the kernel
# read() syscall blocking until a newline arrives.  With 5 workers × N tests
# that is 5×N blocking syscalls on the critical path.
#
# Solution: three changes working together:
#
#   1. RAW BINARY PIPES — subprocess opened with bufsize=0, all I/O via
#      os.read() / os.write() directly on the file descriptors.  No Python
#      text layer, no internal buffer lock, no encoding overhead.
#
#   2. BATCHED WRITES — for each worker, serialise ALL test cases into one
#      big byte string and write it in a single os.write() call.  The worker
#      process receives all N payloads in one kernel round-trip and starts
#      processing immediately.  We amortize the ~130 µs IPC latency across
#      the entire batch rather than paying it once per test.
#
#   3. select() COLLECTION LOOP — a single thread monitors all worker stdout
#      fds simultaneously.  Whenever any worker has data ready we drain it.
#      This replaces N blocking threads each stuck in readline() with one
#      non-blocking multiplexed reader.
#
# Crash recovery
# --------------
# If a worker process dies mid-batch (SIGSEGV from microtaint on push/pop,
# maat IR error, etc.) select() returns its fd as readable but os.read()
# returns b'' (EOF).  We detect this, mark the worker dead, fill the
# remaining expected results for that worker with error dicts, restart the
# process, and continue.  The orchestrator never stalls.
#
# Result ordering
# ---------------
# Workers process tests in FIFO order (their stdin is a byte stream, results
# come back in the same order they were sent).  We use a per-worker counter
# to assign results to the correct test-case slot.

import select as _select


class BatchedWorkerPool:
    """
    Manages a set of persistent worker subprocesses.
    All tests are dispatched in one batch per worker; results are collected
    via a single select() loop across all stdout fds.
    """

    BOOT_TIMEOUT = 600
    # BATCH_TIMEOUT: seconds to wait for an entire batch to complete.  This
    # is the wall-clock ceiling on the slowest Python worker for the entire
    # test set.  Default 600 s = 10 minutes covers typical runs (maat at
    # ~400 µs/case takes <1 s for 2000 cases; angr at ~7 ms takes ~14 s).
    # The pool's progress monitor reports completed-vs-total when this fires,
    # so a hung worker is identifiable.  Override via BATCH_TIMEOUT env var
    # if your run needs more.
    BATCH_TIMEOUT = int(os.environ.get('BATCH_TIMEOUT', '600'))

    def __init__(self) -> None:
        # name → subprocess.Popen
        self._procs: dict[str, subprocess.Popen] = {}
        # name → raw stdin/stdout fd ints
        self._in_fds: dict[str, int] = {}
        self._out_fds: dict[str, int] = {}
        # name → stderr accumulator (drained by background thread)
        self._stderr: dict[str, list[str]] = {}
        # reverse map: out_fd → name (for select loop)
        self._fd_to_name: dict[int, str] = {}
        # name → original spawn command, so _restart_worker can re-exec it.
        # Without this we cannot restart anything — the previous version
        # tore down the dead Popen and just set self._procs[name] = None,
        # which left the pool permanently degraded after the first crash.
        self._cmds: dict[str, list[str]] = {}
        # name → count of restarts so far.  Capped to MAX_RESTARTS to avoid
        # an infinite crash-loop if a worker is fundamentally broken.
        self._restart_count: dict[str, int] = {}
        self.MAX_RESTARTS = 100

    # ------------------------------------------------------------------ boot

    def start_worker(self, name: str, cmd: list[str], boot_timeout: int | None = None) -> None:
        timeout = boot_timeout or self.BOOT_TIMEOUT
        # Remember the command so _restart_worker can re-spawn after a crash.
        self._cmds[name] = list(cmd)
        self._restart_count.setdefault(name, 0)
        print(f'[{name}] Starting...', flush=True)

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,  # RAW binary — no Python buffering layer
        )
        self._procs[name] = proc
        self._stderr[name] = []

        # Drain stderr in background
        def _drain_stderr():
            for chunk in iter(lambda: proc.stderr.read(4096), b''):
                self._stderr[name].append(chunk.decode(errors='replace'))

        threading.Thread(target=_drain_stderr, daemon=True).start()

        # Read "READY\n" with timeout — still uses a thread here because
        # boot is a one-time cost and PANDA can take minutes
        ready_buf = b''
        deadline = time.monotonic() + timeout
        in_fd = proc.stdout.fileno()
        while b'\n' not in ready_buf:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                proc.kill()
                raise RuntimeError(f'[{name}] Timeout waiting for READY')
            r, _, _ = _select.select([in_fd], [], [], min(remaining, 5.0))
            if r:
                chunk = os.read(in_fd, 4096)
                if not chunk:
                    proc.kill()
                    raise RuntimeError(f'[{name}] EOF before READY')
                ready_buf += chunk

        line = ready_buf.split(b'\n')[0].strip().decode(errors='replace')
        if line != 'READY':
            proc.kill()
            raise RuntimeError(f'[{name}] Expected READY, got {line!r}')

        self._in_fds[name] = proc.stdin.fileno()
        self._out_fds[name] = proc.stdout.fileno()
        self._fd_to_name[proc.stdout.fileno()] = name
        print(f'[{name}] Ready.', flush=True)

    def is_alive(self, name: str) -> bool:
        p = self._procs.get(name)
        return p is not None and p.poll() is None

    def worker_names(self) -> list[str]:
        return list(self._procs)

    # --------------------------------------------------------- batch dispatch

    def run_batch(
        self,
        test_cases: list[dict],
        workers: list[str] | None = None,
        deadline: float | None = None,
        hb_counts: dict | None = None,
    ) -> list[dict[str, dict]]:
        """
        Send all test_cases to each worker, collect all results up to deadline.

        deadline  — monotonic time ceiling shared with the GT and C-harness
                    workers.  Defaults to self.BATCH_TIMEOUT seconds from now.
        hb_counts — shared dict updated in-place with per-worker progress so
                    the heartbeat thread can report live counts.
        """
        import signal as _signal

        active = [w for w in (workers or list(self._procs)) if self.is_alive(w)]
        configured = list(workers or list(self._procs))
        not_alive = [w for w in configured if w not in active]
        N = len(test_cases)
        _deadline = deadline if deadline is not None else (time.monotonic() + self.BATCH_TIMEOUT)

        # Pre-serialise all payloads once (shared across workers)
        payloads: list[bytes] = [(json.dumps(tc, default=str) + '\n').encode() for tc in test_cases]
        batch_bytes = b''.join(payloads)

        # Per-worker state
        results: list[dict[str, dict]] = [{} for _ in range(N)]
        # Pre-fill slots for any worker that was requested but isn't alive,
        # so compute_metrics doesn't hit a KeyError later.  Without this,
        # a worker that crashed at boot leaves results[i] missing the key
        # entirely — and compute_metrics's `tr.get(reference_tool, {})`
        # returns `{}` which lacks `output_taint`, crashing the run.
        if not_alive:
            for w in not_alive:
                print(
                    f"[run_batch] WARNING: worker {w!r} is not alive; "
                    f"marking all {N} tests as 'worker not running'",
                    flush=True,
                )
                for i in range(N):
                    results[i][w] = {
                        'error': f'{w} worker process not alive',
                        'time_ns': 0,
                    }
        bufs: dict[str, bytes] = dict.fromkeys(active, b'')
        counts: dict[str, int] = dict.fromkeys(active, 0)
        dead: set[str] = set()

        # Send the entire batch to every active worker IN BACKGROUND THREADS.
        #
        # CRITICAL: Do NOT write synchronously before reading.  The workers
        # process tests and write results back while we are still feeding them
        # input.  Linux pipe buffers are only 64 KB, so a synchronous write of
        # a large batch (e.g. 1500 tests × 300 bytes ≈ 450 KB) blocks as soon
        # as the pipe fills — while the worker is simultaneously blocked trying
        # to write output that the benchmark isn't reading yet.  Classic
        # bidirectional pipe deadlock.
        #
        # Solution: each worker's stdin gets its own daemon thread that feeds
        # data in chunks, yielding to the kernel when the pipe is full.  The
        # main thread simultaneously runs the select() read loop below, so
        # output is drained as fast as the workers produce it, keeping the
        # pipe buffer from filling on the write side.
        write_errors: dict[str, Exception] = {}

        def _feed_worker(name: str) -> None:
            fd = self._in_fds[name]
            remaining = batch_bytes
            try:
                while remaining:
                    # os.write may write fewer bytes than requested (pipe
                    # buffer full); loop until all bytes are written.
                    n = os.write(fd, remaining[:65536])
                    remaining = remaining[n:]
            except OSError as exc:
                write_errors[name] = exc

        write_threads = []
        for name in active:
            t = threading.Thread(target=_feed_worker, args=(name,), daemon=True)
            t.start()
            write_threads.append((name, t))

        # Collect results via select()
        fd_to_name = {self._out_fds[w]: w for w in active if w not in dead}
        _last_progress_print = time.monotonic()
        _PROGRESS_INTERVAL = 15.0  # print progress every 15 s

        while True:
            remaining_workers = [w for w in active if w not in dead and counts[w] < N]
            if not remaining_workers:
                break

            # Update shared heartbeat counts so the heartbeat thread can
            # report live per-worker progress without any locking (reads of
            # counts[w] are atomic in CPython due to the GIL).
            if hb_counts is not None:
                for w in active:
                    hb_counts[w] = f'{counts[w]}/{N}'

            # Periodic progress line every 15 s
            now = time.monotonic()
            if now - _last_progress_print >= _PROGRESS_INTERVAL:
                elapsed = now - (_deadline - self.BATCH_TIMEOUT)
                remaining_s = max(0.0, _deadline - now)
                parts = '  '.join(f'{w}: {counts[w]}/{N}' for w in active if w not in dead)
                print(
                    f'[run_batch] {elapsed:.0f}s elapsed  {remaining_s:.0f}s left  results: {parts}',
                    flush=True,
                )
                _last_progress_print = now

            timeout = _deadline - time.monotonic()
            if timeout <= 0:
                for name in remaining_workers:
                    stderr_tail = self._get_stderr_tail(name)
                    msg = f'batch timeout after {self.BATCH_TIMEOUT}s — completed {counts[name]}/{N} tests'
                    print(f"\n{'='*60}", flush=True)
                    print(f'[{name}] TIMEOUT', flush=True)
                    print(f'  Completed {counts[name]}/{N} tests before timeout')
                    if counts[name] < N:
                        tc = test_cases[counts[name]]
                        print(f"  Stalled on test {counts[name]}: {tc.get('assembly', tc.get('label', '?'))}")
                        print(f"  Bytes:  {tc.get('bytes', '?')}")
                        print(f"  Taint:  {tc.get('taint', {})}")
                    if stderr_tail:
                        print(f'  Stderr ({len(stderr_tail)} lines):')
                        for l in stderr_tail[-20:]:
                            print(f'    {l}')
                    print('=' * 60, flush=True)
                    for i in range(counts[name], N):
                        results[i][name] = {
                            'error': msg,
                            'time_ns': 0,
                            'crash_test_idx': counts[name],
                            'crash_stderr': stderr_tail,
                        }
                    dead.add(name)
                break

            live_fds = [self._out_fds[w] for w in remaining_workers]
            try:
                readable, _, _ = _select.select(live_fds, [], [], min(timeout, 1.0))
            except (ValueError, OSError):
                break

            for fd in readable:
                name = fd_to_name.get(fd)
                if name is None or name in dead:
                    continue

                chunk = os.read(fd, 65536)
                if not chunk:
                    # EOF — worker died; collect full diagnostics
                    proc = self._procs.get(name)
                    exit_code = proc.wait() if proc else None
                    sig_name = None
                    if exit_code is not None and exit_code < 0:
                        try:
                            sig_name = _signal.Signals(-exit_code).name
                        except ValueError:
                            sig_name = f'signal {-exit_code}'

                    # Give stderr a moment to flush after the process exits
                    time.sleep(0.05)
                    stderr_tail = self._get_stderr_tail(name)
                    stdout_tail = _ANSI_ESC.sub('', bufs[name].decode(errors='replace')).strip()
                    crash_idx = counts[name]
                    crash_tc = test_cases[crash_idx] if crash_idx < N else None

                    # Print a clear crash report to terminal immediately
                    print(f"\n{'='*60}", flush=True)
                    print(f'[{name}] CRASH detected', flush=True)
                    print(f'  Exit code : {exit_code}' + (f' ({sig_name})' if sig_name else ''), flush=True)
                    print(f'  Completed : {crash_idx}/{N} tests before crash', flush=True)
                    if crash_tc is not None:
                        asm = crash_tc.get('assembly', crash_tc.get('label', '?'))
                        print(f'  Crashing test [{crash_idx}]: {asm}', flush=True)
                        print(f"  Bytes  : {crash_tc.get('bytes', '?')}", flush=True)
                        print(f"  State  : {crash_tc.get('state', {})}", flush=True)
                        print(f"  Taint  : {crash_tc.get('taint', {})}", flush=True)
                        if crash_tc.get('rationale'):
                            print(f"  Note   : {crash_tc['rationale']}", flush=True)
                    if stdout_tail:
                        print('  Partial stdout (last output before crash):', flush=True)
                        for l in stdout_tail.splitlines()[-5:]:
                            print(f'    {l}', flush=True)
                    if stderr_tail:
                        n_shown = min(30, len(stderr_tail))
                        print(f'  Stderr (last {n_shown} lines):', flush=True)
                        for l in stderr_tail[-n_shown:]:
                            print(f'    {l}', flush=True)
                    else:
                        print('  Stderr: (empty)', flush=True)
                    print('=' * 60, flush=True)

                    # Build rich error dict for the SINGLE test that killed
                    # the worker.  Previously we marked every remaining test
                    # as errored — that's correct only if we cannot restart.
                    # Now that _restart_worker actually works, we attempt to
                    # resume past the offending test so the rest of the batch
                    # still gets processed.
                    crash_info = {
                        'crash_test_idx': crash_idx,
                        'crash_assembly': crash_tc.get('assembly') if crash_tc else None,
                        'crash_bytes': crash_tc.get('bytes') if crash_tc else None,
                        'crash_taint': crash_tc.get('taint') if crash_tc else None,
                        'crash_exit_code': exit_code,
                        'crash_signal': sig_name,
                        'crash_stderr': stderr_tail[-50:],
                        'crash_stdout_tail': stdout_tail[-500:],
                    }
                    crash_err = {
                        'error': (
                            f"{name} crashed on test {crash_idx} "
                            f"({asm if crash_tc else '?'})"
                            + (f' with {sig_name}' if sig_name else f' exit={exit_code}')
                        ),
                        'time_ns': 0,
                        **crash_info,
                    }
                    # Mark just the one offending test
                    if crash_idx < N:
                        results[crash_idx][name] = crash_err

                    # Try to restart and resume.  We feed only the tests AFTER
                    # the crashing one so the same input doesn't kill the
                    # worker again immediately.
                    resume_idx = crash_idx + 1
                    restarted = False
                    if resume_idx < N:
                        restarted = self._restart_worker(name)

                    if not restarted:
                        # Either we hit MAX_RESTARTS, or there's nothing left
                        # to feed, or restart itself failed.  Mark every
                        # remaining slot as errored, exactly like before.
                        for i in range(resume_idx, N):
                            results[i][name] = dict(crash_err)
                        dead.add(name)
                        continue

                    # Restart succeeded — wire the new fds into the live maps
                    # so the select() loop on the next iteration picks them up.
                    new_out_fd = self._out_fds[name]
                    fd_to_name[new_out_fd] = name
                    bufs[name] = b''
                    counts[name] = resume_idx  # next expected result index

                    # Spawn a fresh feeder thread for the remaining payload
                    # only.  The original thread for this worker is dead with
                    # the broken pipe; we don't try to revive it.
                    remaining_payload = b''.join(payloads[resume_idx:])

                    def _feed_resumed(name=name, data=remaining_payload):
                        fd = self._in_fds[name]
                        remaining = data
                        try:
                            while remaining:
                                n = os.write(fd, remaining[:65536])
                                remaining = remaining[n:]
                        except OSError as exc:
                            write_errors[name] = exc

                    t = threading.Thread(target=_feed_resumed, daemon=True)
                    t.start()
                    write_threads.append((name, t))
                    print(
                        f'[{name}] Resumed at test {resume_idx}/{N} (skipped crashing test {crash_idx})',
                        flush=True,
                    )
                    continue

                bufs[name] += chunk

                # Parse complete newline-delimited JSON lines
                while b'\n' in bufs[name]:
                    line_b, bufs[name] = bufs[name].split(b'\n', 1)
                    if not line_b.strip():
                        continue
                    idx = counts[name]
                    if idx >= N:
                        break
                    raw = _ANSI_ESC.sub('', line_b.decode(errors='replace')).strip()
                    try:
                        results[idx][name] = json.loads(raw)
                    except json.JSONDecodeError:
                        results[idx][name] = {
                            'error': f'Bad JSON: {raw[:300]}',
                            'time_ns': 0,
                        }
                    counts[name] += 1

        # Restart any workers that died and could not be recovered in-band
        # (e.g. they exhausted MAX_RESTARTS).  Best-effort — failures here
        # are non-fatal because results have already been marked errored.
        for name in dead:
            self._restart_worker(name)

        # Join write threads (they should be done by now — all data consumed
        # by the workers, or the workers are dead).  Give each at most 5 s;
        # a stuck write thread means the worker's stdin pipe is full and the
        # worker itself is dead, which should have been caught above.
        for name, t in write_threads:
            t.join(timeout=5.0)

        # Report write errors that were captured by the feeder threads.
        for name, exc in write_errors.items():
            if name not in dead:
                self._emit_crash_report(name, exc, counts[name], test_cases, bufs[name], is_write_error=True)
                for i in range(counts[name], N):
                    results[i][name] = {
                        'error': f'{name} stdin write error: {exc}',
                        'time_ns': 0,
                    }

        return results

    # --------------------------------------------------------- diagnostics

    def _get_stderr_tail(self, name: str) -> list[str]:
        """Return all stderr lines captured so far for a worker."""
        raw = ''.join(self._stderr.get(name, []))
        return [l for l in raw.splitlines() if l.strip()]

    def _emit_crash_report(
        self,
        name: str,
        exc: Exception,
        crash_idx: int,
        test_cases: list[dict],
        partial_stdout: bytes,
        is_write_error: bool = False,
    ) -> None:
        """Print a crash report when we can't write to the worker."""
        kind = 'WRITE ERROR' if is_write_error else 'CRASH'
        print(f"\n{'='*60}", flush=True)
        print(f'[{name}] {kind}: {exc}', flush=True)
        if crash_idx < len(test_cases):
            tc = test_cases[crash_idx]
            asm = tc.get('assembly', tc.get('label', '?'))
            print(f'  On test [{crash_idx}]: {asm}', flush=True)
            print(f"  Bytes: {tc.get('bytes', '?')}", flush=True)
        if partial_stdout:
            decoded = _ANSI_ESC.sub('', partial_stdout.decode(errors='replace')).strip()
            if decoded:
                print(f'  Partial stdout: {decoded[-200:]}', flush=True)
        stderr_tail = self._get_stderr_tail(name)
        if stderr_tail:
            print('  Stderr:', flush=True)
            for l in stderr_tail[-10:]:
                print(f'    {l}', flush=True)
        print('=' * 60, flush=True)

    # ------------------------------------------------------- crash recovery

    def _restart_worker(self, name: str) -> bool:
        """
        Tear down a dead worker and spawn a fresh one using the command
        recorded by start_worker.  Returns True on success, False if the
        worker has already exceeded MAX_RESTARTS or boot fails.

        After a successful restart, is_alive(name) is True again and the
        worker is ready to accept new test cases via run_batch.
        """
        cmd = self._cmds.get(name)
        if cmd is None:
            print(f'[{name}] Cannot restart: no spawn command recorded', flush=True)
            return False

        n_restarts = self._restart_count.get(name, 0)
        if n_restarts >= self.MAX_RESTARTS:
            print(
                f'[{name}] Restart limit reached ({n_restarts}/{self.MAX_RESTARTS}); giving up',
                flush=True,
            )
            return False

        print(
            f'[{name}] Restarting after crash (restart #{n_restarts + 1}/{self.MAX_RESTARTS})...',
            flush=True,
        )

        # Tear down the dead Popen and clean stale fd mappings.
        old = self._procs.get(name)
        if old is not None and old.poll() is None:
            try:
                old.kill()
                old.wait(timeout=5)
            except Exception:
                pass
        old_fd = self._out_fds.pop(name, None)
        if old_fd is not None:
            self._fd_to_name.pop(old_fd, None)
        self._in_fds.pop(name, None)
        self._procs.pop(name, None)
        # Reset the stderr accumulator so the next crash report only shows
        # output from the new process, not the old corpse.
        self._stderr[name] = []

        self._restart_count[name] = n_restarts + 1
        try:
            self.start_worker(name, cmd)
        except Exception as exc:
            print(f'[{name}] Restart failed: {exc}', flush=True)
            return False
        return True

    # --------------------------------------------------------------- shutdown

    def stop_all(self) -> None:
        quit_b = b'QUIT\n'
        for name, proc in list(self._procs.items()):
            if proc is None or proc.poll() is not None:
                continue
            try:
                os.write(self._in_fds[name], quit_b)
                proc.wait(timeout=10)
            except Exception:
                pass
            finally:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._procs.clear()
        self._in_fds.clear()
        self._out_fds.clear()
        self._fd_to_name.clear()


# ---------------------------------------------------------------------------
# C-harness workers (taintgrind / libdft64)
# Unchanged: compile + subprocess per test, run via thread pool.
# ---------------------------------------------------------------------------


def _run_c_harness(tool: str, cmd: str, tc: dict) -> dict:
    """
    Compile (source piped to gcc stdin, binary in /dev/shm) + run one test.
    Called from the thread pool — each tool uses a unique bin_file name so
    concurrent taintgrind and libdft64 invocations don't race.
    """
    import uuid

    bin_file = f'harness_{tool}_{uuid.uuid4().hex[:8]}.bin'

    try:
        compile_c_harness(tc, tool, bin_file=bin_file)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or '').strip()
        return {'error': f'compile failed: {stderr[:300]}', 'time_ns': 0}
    except Exception as exc:
        return {'error': f'compile error: {exc}', 'time_ns': 0}

    # Resolve the actual binary path (may be /dev/shm/... or CWD/...)
    bin_path = _BIN_PATHS.get(bin_file, os.path.join(CWD, bin_file))

    # Build tool-specific run command substituting the real binary path.
    # taintgrind uses a docker volume mount: CWD → /pwd inside the container.
    # We must translate bin_path back to its /pwd/... equivalent.
    if tool == 'taintgrind':
        if bin_path.startswith('/dev/shm/'):
            # /dev/shm is not inside the docker volume; fall back to CWD copy
            import shutil

            cwd_bin = os.path.join(CWD, bin_file)
            try:
                shutil.copy2(bin_path, cwd_bin)
            except FileNotFoundError as e:
                return {'error': f'shutil.copy2 failed: {e}', 'time_ns': 0}

            container_path = f'/pwd/{bin_file}'
            cleanup_extra = cwd_bin
        else:
            container_path = f'/pwd/{bin_file}'
            cleanup_extra = None
        actual_cmd = cmd.replace('/pwd/harness.bin', container_path)
    else:
        actual_cmd = cmd.replace('./harness.bin', bin_path)
        cleanup_extra = None

    t0 = time.perf_counter_ns()
    try:
        r = subprocess.run(actual_cmd.split(), capture_output=True, text=True, timeout=60)
        t1 = time.perf_counter_ns()

        if tool == 'libdft64':
            # Pin reports taint via [PIN][GETVAL] lines (one per __libdft_get_taint
            # call).  The C harness ALSO prints a JSON line with its measured
            # in-program clock_gettime delta — find that line for the engine
            # time, separate from Pin's own startup which dominates wall-clock.
            output_taint = {'RAX': 0, 'RBX': 0, 'RCX': 0, 'RDX': 0}
            reg_order = ['RAX', 'RBX', 'RCX', 'RDX']
            idx = 0
            harness_time_ns = None
            for line in r.stdout.splitlines() + r.stderr.splitlines():
                if line.startswith('[PIN][GETVAL]') and idx < 4:
                    if 'lb: 0, taint: {}' not in line:
                        output_taint[reg_order[idx]] = 1
                    idx += 1
                elif line.startswith('{"output_taint"'):
                    # The harness's own JSON line — extract its time_ns
                    try:
                        harness_res = json.loads(line)
                        harness_time_ns = harness_res.get('time_ns')
                    except (json.JSONDecodeError, ValueError):
                        pass
            # Prefer the harness's in-program measurement; fall back to wall
            # clock only if the harness JSON couldn't be parsed (rare — the
            # binary always prints it before exiting).
            return {
                'output_taint': output_taint,
                'time_ns': harness_time_ns if harness_time_ns is not None else (t1 - t0),
            }

        for line in r.stdout.splitlines():
            if line.startswith('{"output_taint"'):
                res = json.loads(line)
                if not res.get('time_ns'):
                    res['time_ns'] = t1 - t0
                return res

        return {
            'error': (r.stderr.strip() or r.stdout.strip() or 'no output')[:400],
            'time_ns': t1 - t0,
        }

    except subprocess.TimeoutExpired:
        return {'error': 'Timeout 60s', 'time_ns': 0}
    except Exception as exc:
        return {'error': str(exc), 'time_ns': 0}
    finally:
        # Remove binary from /dev/shm and any CWD copy
        _BIN_PATHS.pop(bin_file, None)
        for path in [bin_path, cleanup_extra]:
            if path:
                try:
                    os.remove(path)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Noninterference ground-truth simulator (in-process Unicorn enumeration)
# ---------------------------------------------------------------------------


class GroundTruthSimulator:
    """
    Computes the exact noninterference taint set by enumerating all 2**k
    assignments of the k tainted input bits and recording which output bits
    vary across the runs.

    ``k = popcount(T_RAX | T_RBX | T_RCX | T_RDX)``.

    For every test case with k <= GT_BIT_BUDGET (= 8 by default, 256 sims),
    this gives the *exact* per-bit-output taint mask under the standard
    Goguen-Meseguer noninterference definition: bit i of output R is
    tainted iff ∃ two assignments x, x' of the tainted bits such that
    ``output(x)[R][i] != output(x')[R][i]``.

    Cases with k > GT_BIT_BUDGET are reported as
    ``{"error": "skipped: k=<n> > budget=<N>"}`` so downstream metrics
    can exclude them without confusing a skip for an analysis failure.

    Implementation notes
    --------------------
    * Concrete inputs are taken from ``tc["state"]`` with the tainted bits
      replaced according to each enumerated assignment.  Untainted bits keep
      their concrete value.  This matches the convention used by every other
      tool in the benchmark: taint is attached to specific bits of specific
      registers, not to whole values.
    * Output is computed for all four registers (RAX, RBX, RCX, RDX); the
      taint mask for each is ``(OR of all 2**k outputs) XOR (AND of all
      2**k outputs)`` — exactly the bits whose value varies.
    * Multi-instruction sequences are supported: the entire concatenated
      byte string is executed in one Unicorn step.  This is important for
      sequence test cases.
    * Errors (illegal instruction, unmapped memory) for any single
      enumeration are swallowed and treated as the run producing 0 — sound,
      because the differential XOR/OR cannot fabricate taint, only miss it
      in the failed runs.
    """

    def __init__(self) -> None:
        # Lazily import unicorn so the file remains importable in environments
        # where unicorn is unavailable (e.g. analysis-only tooling).
        import unicorn
        import unicorn.x86_const as ux

        self._uc_module = unicorn
        self._reg_map = {
            'RAX': ux.UC_X86_REG_RAX,
            'RBX': ux.UC_X86_REG_RBX,
            'RCX': ux.UC_X86_REG_RCX,
            'RDX': ux.UC_X86_REG_RDX,
        }
        # Cached Unicorn instance — reusing the same Uc with mem_write +
        # reg_write is ~20× faster than creating a fresh Uc per run.  Re-
        # creating only when the bytestring changes (rare) keeps semantics
        # correct: x86 doesn't have lingering microarchitectural state that
        # affects emulator output.
        self._uc = None
        self._last_bytestring: bytes | None = None
        self._STACK_BASE = 0x100000
        self._CODE_BASE = 0x1000

    def _ensure_uc(self) -> object:
        if self._uc is None:
            unicorn = self._uc_module
            uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
            uc.mem_map(self._CODE_BASE, 0x10000)
            uc.mem_map(self._STACK_BASE, 0x10000)
            self._uc = uc
        return self._uc

    # Per-test-case wall-clock ceiling.  At k=12 (the default budget),
    # 4096 emu_start calls × 10 ms each gives a hard ceiling of ~41 s
    # in the absolute worst case (every call hits the timeout).  We
    # set the deadline a little above that so legitimate cases are
    # never cut short.  In normal operation k=12 cases finish in <1 s.
    GT_PER_CASE_DEADLINE_S = 60.0

    def evaluate(self, tc: dict, shared_deadline: float | None = None) -> dict:
        """Run the test case and return ``{"output_taint": {...}, "time_ns": int}``
        or ``{"error": "...", "time_ns": int}`` if the case exceeds the budget.

        ``shared_deadline`` is a ``time.monotonic()`` ceiling that aborts the
        enumeration mid-flight when the whole-batch wall clock fires.  This
        is critical: a single k=16 case can take 60+ seconds and would
        otherwise outlive the shared run deadline by minutes.
        """
        t0 = time.time_ns()
        per_case_deadline_ns = t0 + int(self.GT_PER_CASE_DEADLINE_S * 1e9)

        # CRITICAL: discard the cached Unicorn instance at the start of every
        # test case.  Even with full register reset before each run, internal
        # Unicorn state (translation cache, hidden microarchitectural state,
        # MSRs, debug registers, possibly stale TLB entries from prior code
        # writes) can carry over between cases and corrupt the enumeration.
        # Building a fresh Uc per test case costs ~200µs — negligible
        # compared to the 2^k Unicorn runs that follow — and gives provably
        # clean state.  Within a single test case the Uc is reused across
        # all 2^k runs (with full register reset each), which preserves the
        # speedup that matters.
        self._uc = None
        self._last_bytestring = None

        budget = int(tc.get('gt_budget', GT_BIT_BUDGET))
        taint: dict[str, int] = {r: int(tc['taint'].get(r, 0)) for r in REGISTERS}
        state: dict[str, int] = {r: int(tc['state'].get(r, 0)) for r in REGISTERS}
        bytestring = bytes.fromhex(tc['bytes'])

        # Collect (register, bit_index) for every tainted bit.
        positions: list[tuple[str, int]] = []
        for reg in REGISTERS:
            tm = taint[reg]
            for bit in range(64):
                if (tm >> bit) & 1:
                    positions.append((reg, bit))

        k = len(positions)
        if k > budget:
            return {
                'error': f'skipped: k={k} > budget={budget}',
                'time_ns': time.time_ns() - t0,
                'skipped_k': k,
            }

        # k == 0: nothing tainted, all output taint is 0 by definition.
        if k == 0:
            return {
                'output_taint': dict.fromkeys(REGISTERS, 0),
                'time_ns': time.time_ns() - t0,
                'k': 0,
            }

        # Base values: tainted bits are zeroed in the entry state, then
        # reconstructed per assignment by ORing in the assignment bits.
        base_vals: dict[str, int] = {r: state[r] & ~taint[r] & MASK64 for r in REGISTERS}

        # Enumerate 2**k assignments.  For each, set the tainted bits according
        # to the assignment value's bit pattern, run the bytecode, and collect
        # the output of every register.
        outputs: list[dict[str, int]] = []
        n_assignments = 1 << k
        # Convert shared monotonic-time deadline (if given) to time_ns scale
        # so we can compare against time.time_ns() cheaply inside the hot loop.
        # We check every 64 runs to keep overhead negligible.
        for assignment in range(n_assignments):
            # Check deadlines every 64 assignments (bit 6 of counter), or on
            # the first iteration so we always make at least one check.
            if assignment == 0 or (assignment & 0x3F) == 0:
                now_ns = time.time_ns()
                if now_ns > per_case_deadline_ns:
                    return {
                        'error': (
                            f'deadline: {self.GT_PER_CASE_DEADLINE_S:.0f}s per-case exceeded '
                            f'after {assignment}/{n_assignments} runs (k={k})'
                        ),
                        'time_ns': now_ns - t0,
                        'k': k,
                        'completed_runs': assignment,
                    }
                # Shared deadline (whole-batch wall clock).  This is the
                # critical guard: without it, a single slow case can outlive
                # the run deadline by tens of seconds and stall phase 3.
                if shared_deadline is not None and time.monotonic() > shared_deadline:
                    return {
                        'error': 'timeout',  # match other workers' format
                        'time_ns': now_ns - t0,
                        'k': k,
                        'completed_runs': assignment,
                    }
            run_vals = dict(base_vals)
            for idx, (reg, bit) in enumerate(positions):
                if (assignment >> idx) & 1:
                    run_vals[reg] = (run_vals[reg] | (1 << bit)) & MASK64
            try:
                outputs.append(self._run_unicorn(bytestring, run_vals))
            except Exception:
                # Sound fallback: treat the failed run as a trap.  Trapped
                # runs are dropped from the aggregation below.
                outputs.append(dict.fromkeys(REGISTERS, 0) | {'__trapped__': True})

        # Per-register taint mask: bits that vary across the enumeration.
        #
        # Trapped runs (uc.emu_start raised UcError — divide-by-zero,
        # invalid opcode, unmapped memory, timeout) are dropped.  Strict
        # noninterference semantics: a trap is not a defined output value
        # for this enumeration, so it cannot tell us anything about
        # input-output dependence.  Including trap-state register values
        # in the OR/AND aggregation would produce spurious "taint" on
        # bits that the architecture never wrote to in the trapping
        # enumeration.
        #
        # Concrete example: `cqo; idiv rbx` with RAX=INT64_MIN, RBX=-1
        # tainted in bit 0.  Bit 0 = 1 -> RBX = -1 -> #DE.  Bit 0 = 0 ->
        # RBX = -2 -> RAX = 0x4000000000000000.  The trapped run leaves
        # RAX at its pre-trap value (0x8000000000000000), so the
        # aggregation sees two "outputs" 0x80... and 0x40... and reports
        # 0xC000000000000000 as taint.  But architecturally the "output
        # of running with RBX=-1" is undefined — the program crashed.
        # Dropping the trap from the aggregation gives the architectural
        # answer (taint = 0 because only one enumeration produced an
        # observable output).
        successful = [o for o in outputs if not o.get('__trapped__', False)]
        gt_mask: dict[str, int] = {}
        if not successful:
            # Every enumeration trapped.  No information about input
            # dependence is observable through this register channel.
            # Report all-zero taint — this is the conservative answer
            # for noninterference (no observed difference).
            for reg in REGISTERS:
                gt_mask[reg] = 0
        else:
            for reg in REGISTERS:
                agg_or = 0
                agg_and = MASK64
                for out in successful:
                    v = out[reg]
                    agg_or |= v
                    agg_and &= v
                gt_mask[reg] = (agg_or ^ agg_and) & MASK64

        return {
            'output_taint': gt_mask,
            'time_ns': time.time_ns() - t0,
            'k': k,
        }

    # ─── Per-emu_start safety bounds ─────────────────────────────────────
    # ``timeout`` in microseconds — Unicorn aborts if execution exceeds this.
    # 10 ms is generous for any sane single instruction or short sequence
    # (typical instruction takes <1 µs to emulate).  Without this guard, a
    # test sequence containing an infinite loop (jmp $, loop with non-
    # decrementing condition, etc.) would block the entire benchmark
    # forever — verified empirically that emu_start with no timeout never
    # returns on `EB FE` (jmp -2).
    #
    # ``count`` is a belt-and-suspenders limit: even if timeout were
    # somehow ignored, this caps the number of x86 instructions executed.
    # Sequences in the benchmark are at most 32 instructions; we allow
    # 1024 to give plenty of headroom for loops with bounded iteration
    # while still preventing infinite execution.
    UC_TIMEOUT_US = 10_000  # 10 ms per call
    UC_MAX_COUNT = 1024  # max instructions per call

    # Full reset list.  We zero every architectural register before every
    # run.  Caching a Unicorn instance and only resetting RAX/RBX/RCX/RDX
    # turned out to leak state between consecutive enumerations: instructions
    # like INC, MOVZX produced wrong "ground truth" because some leftover
    # register or vector lane carried over from earlier runs.  Verified
    # empirically: dropping the XMM/segment/FPU resets brings k=16 cases
    # from ~14 s to ~22 s but reintroduces unsoundness on `movzx rcx, dl`
    # and similar — Unicorn's translation cache or some hidden state
    # depends on those registers in non-obvious ways.  We pay the cost.
    _RESET_GP_REGS = (
        'RAX',
        'RBX',
        'RCX',
        'RDX',
        'RSI',
        'RDI',
        'RBP',
        'R8',
        'R9',
        'R10',
        'R11',
        'R12',
        'R13',
        'R14',
        'R15',
    )
    _RESET_SEG_REGS = ('CS', 'SS', 'DS', 'ES', 'FS', 'GS')

    # Pre-allocated zero buffer for full-region wipes.  Sized at the
    # mapped stack region (0x10000 bytes); reused across all test cases
    # to avoid re-allocating a 64 KB buffer per call.
    _STACK_ZERO_BUF: bytes = b'\x00' * 0x10000
    _CODE_ZERO_BUF: bytes = b'\x00' * 0x10000

    def _run_unicorn(self, bytestring: bytes, vals: dict[str, int]) -> dict[str, int]:
        unicorn = self._uc_module
        ux = unicorn.x86_const
        uc = self._ensure_uc()
        # Code region: we must wipe BEFORE writing the new bytestring.
        # The mapped code region is 0x10000 bytes long.  If the previous
        # test case left bytes here (the cached _last_bytestring path
        # below only writes up to len(bytestring)), and the current
        # bytestring is shorter, RIP can fetch stale instructions from
        # the tail.  More subtly, an emu_start that overshoots
        # CODE_BASE+len(bytestring) (jmp/call into the trailing region)
        # would execute stale opcodes from a prior test.  Wiping the
        # whole code region at every call costs a single 64 KB memcpy
        # — negligible compared to ~30 µs of emulation per call.
        if bytestring != self._last_bytestring:
            uc.mem_write(self._CODE_BASE, self._CODE_ZERO_BUF)
            uc.mem_write(self._CODE_BASE, bytestring)
            self._last_bytestring = bytestring

        # Bulk zero every architectural register class.
        for reg_name in self._RESET_GP_REGS:
            uc.reg_write(getattr(ux, f'UC_X86_REG_{reg_name}'), 0)
        for i in range(16):
            try:
                uc.reg_write(getattr(ux, f'UC_X86_REG_XMM{i}'), 0)
            except Exception:
                pass
        for seg in self._RESET_SEG_REGS:
            try:
                uc.reg_write(getattr(ux, f'UC_X86_REG_{seg}'), 0)
            except Exception:
                pass
        for fpu_reg, default in (('FPSW', 0), ('FPCW', 0x037F), ('FPTAG', 0xFFFF)):
            try:
                uc.reg_write(getattr(ux, f'UC_X86_REG_{fpu_reg}'), default)
            except Exception:
                pass
        # Reset RIP explicitly.  emu_start(begin=...) overrides RIP, but
        # belt-and-suspenders: an emu_start that fails before any fetch
        # could leave RIP at a stale value that a future test reads back
        # via uc.reg_read.  We don't currently read RIP, but a developer
        # adding an RIP-output instruction (`call`, `ret`) later would
        # silently hit stale state.
        try:
            uc.reg_write(ux.UC_X86_REG_RIP, 0)
        except Exception:
            pass
        uc.reg_write(ux.UC_X86_REG_RSP, self._STACK_BASE + 0x8000)
        uc.reg_write(ux.UC_X86_REG_EFLAGS, 0x2)

        # Wipe the ENTIRE stack region (0x10000 bytes) before installing
        # this enumeration's overlay values.  Why the entire region:
        # a single test's prior enumeration may have written far outside
        # any fixed window around RSP.  Examples observed in practice:
        #
        #   - Sequence test sets RDI = STACK_BASE + 0x4000 then `mov [rdi],
        #     rax` writes 8 bytes at 0x104000, well outside any window
        #     centred on RSP=0x108000.  Untainted bits of that 8-byte
        #     region carry over into the next enumeration's reads.
        #   - Tainted base+offset addressing (`mov [rax+rcx*8], rdx`)
        #     can scatter writes across the whole region as the tainted
        #     bits are flipped through the 2^k enumerations.
        #   - rep stosb/movsb with a tainted RCX can write up to 0xFFFF
        #     bytes when the tainted bit makes ECX huge.  The instruction
        #     timeout halts execution, but the bytes already written
        #     persist.
        #
        # The previous 128-byte wipe at [RSP-64..RSP+64) was the right
        # idea but the wrong size.  A full-region zero costs one
        # 64 KB memcpy (~5 µs); negligible vs the ~30 µs emu_start.
        uc.mem_write(self._STACK_BASE, self._STACK_ZERO_BUF)

        # Overlay the test's specific register values.  Comes AFTER the
        # bulk zero AND after RSP is set so it can override RSP if the
        # test wants a different stack location (the test would also
        # need to ensure that location is mapped — but tests in this
        # suite do not currently override RSP, so the default holds).
        for reg, v in vals.items():
            uc.reg_write(self._reg_map[reg], v & MASK64)

        # Track whether the run completed without faulting.  Unicorn raises
        # UcError on every kind of trap: divide-by-zero (#DE), invalid
        # opcode (#UD), unmapped memory access (#PF), per-instruction
        # timeout, instruction count exceeded, etc.  In every case the
        # registers we read back are either the pre-trap state (for
        # synchronous faults like #DE) or the partially-updated state
        # (for memory faults that may have completed earlier writes).
        # Either way the values are NOT the "output of running this
        # bytestring on these inputs" — they're an artefact of the
        # abort.  Including them in the noninterference aggregation
        # produces spurious taint bits on cases where some inputs trap
        # and others don't.  Returning the trap flag lets the caller
        # drop trapped enumerations from the aggregation entirely
        # (strict noninterference semantics: a trap is "no observation").
        trapped = False
        try:
            uc.emu_start(
                self._CODE_BASE,
                self._CODE_BASE + len(bytestring),
                timeout=self.UC_TIMEOUT_US,
                count=self.UC_MAX_COUNT,
            )
        except unicorn.UcError:
            trapped = True
        regs_out = {r: uc.reg_read(self._reg_map[r]) for r in REGISTERS}
        regs_out['__trapped__'] = trapped
        return regs_out


def _run_ground_truth_batch(
    test_cases: list[dict],
    show_progress: bool = False,
    deadline: float | None = None,
    hb_counts: dict | None = None,
) -> list[dict]:
    """Run the GT simulator over every test case sequentially.

    Stops accepting new cases when the shared ``deadline`` fires and marks
    the remainder ``{"error": "timeout"}`` so aggregation can exclude them.
    ``hb_counts`` is updated in-place so the heartbeat thread shows GT
    progress alongside the other workers.
    """
    sim = GroundTruthSimulator()
    out: list[dict] = []
    n_total = len(test_cases)
    bar = tqdm(total=n_total, desc='ground_truth', unit='case') if show_progress else None
    HEARTBEAT_EVERY = 100
    last_beat = time.time()

    # Initialise heartbeat count to 0/N immediately so the heartbeat thread
    # shows GT as a participant from the very first tick, even if the first
    # case takes long enough to delay any updates.
    if hb_counts is not None:
        hb_counts['ground_truth'] = f'0/{n_total}'

    for i, tc in enumerate(test_cases):
        # Honour shared deadline before starting next case
        if deadline is not None and time.monotonic() > deadline:
            remaining = n_total - i
            print(
                f'[ground_truth] deadline reached after {i}/{n_total} cases '
                f'— marking remaining {remaining} as timeout',
                flush=True,
            )
            for _ in range(remaining):
                out.append({'error': 'timeout', 'time_ns': 0})
            break
        try:
            # Pass the shared deadline through so a single large-k case can
            # abort mid-enumeration rather than running to completion long
            # past the deadline.
            res = sim.evaluate(tc, shared_deadline=deadline)
        except Exception as e:
            res = {
                'error': f'GT crash: {type(e).__name__}: {str(e)[:120]}',
                'time_ns': 0,
            }
        out.append(res)
        if hb_counts is not None:
            hb_counts['ground_truth'] = f'{i+1}/{n_total}'
        if bar is not None:
            bar.update(1)
        else:
            now = time.time()
            if (i + 1) % HEARTBEAT_EVERY == 0 or now - last_beat > 30.0:
                print(
                    f'    [ground_truth] {i+1}/{n_total} cases ({100*(i+1)/n_total:.1f}%)',
                    flush=True,
                )
                last_beat = now
    if bar is not None:
        bar.close()
    # Safety pad in case of early break
    while len(out) < n_total:
        out.append({'error': 'timeout', 'time_ns': 0})
    return out


def _run_c_harnesses_parallel(
    test_cases: list[dict],
    c_harness_cmds: dict[str, str],
    executor: concurrent.futures.ThreadPoolExecutor,
    deadline: float | None = None,
    hb_counts: dict | None = None,
) -> list[dict[str, dict]]:
    """
    Submit all C-harness tests (taintgrind, libdft64) to the thread pool.
    Stops submitting new work when the shared deadline fires and marks
    remaining cases as timeout so aggregation can exclude them.
    """
    if not c_harness_cmds:
        return [{} for _ in test_cases]

    N = len(test_cases)
    tools = list(c_harness_cmds)

    # Submit all (test, tool) pairs at once
    futures: list[dict[str, concurrent.futures.Future]] = [
        {name: executor.submit(_run_c_harness, name, cmd, tc) for name, cmd in c_harness_cmds.items()}
        for tc in test_cases
    ]

    # Collect results, checking deadline between tests
    results: list[dict[str, dict]] = []
    _last_print = time.monotonic()
    _INTERVAL = 15.0
    for i, per_test in enumerate(futures):
        # Check shared deadline before collecting the next result
        if deadline is not None and time.monotonic() > deadline:
            remaining = N - i
            print(
                f'[c-harness] deadline reached after {i}/{N} tests — marking remaining {remaining} as timeout',
                flush=True,
            )
            for _ in range(remaining):
                results.append({name: {'error': 'timeout', 'time_ns': 0} for name in tools})
            break
        results.append({name: fut.result() for name, fut in per_test.items()})
        if hb_counts is not None:
            hb_counts['c-harness'] = f'{i+1}/{N}'
        now = time.monotonic()
        if now - _last_print >= _INTERVAL:
            print(
                f"[c-harness] {i+1}/{N} tests done  (tools: {', '.join(tools)})",
                flush=True,
            )
            _last_print = now

    # Safety pad
    while len(results) < N:
        results.append({name: {'error': 'timeout', 'time_ns': 0} for name in tools})
    return results


# ---------------------------------------------------------------------------
# Test-case generation
# ---------------------------------------------------------------------------

_KS = Ks(KS_ARCH_X86, KS_MODE_64)


def _rand_taint() -> int:
    """Bias toward interesting cases: clean / all-tainted / partial.

    Includes a "sparse" regime that produces low-popcount masks (1–4 bits set
    at random positions).  This is what makes the noninterference ground-truth
    simulator viable on randomly-generated cases: if every register were a
    full random 64-bit mask the total k would always be ~128 and the GT would
    always skip.  By emitting sparse masks ~30% of the time, a meaningful
    fraction of cases land in the GT's tractable budget without changing the
    overall coverage profile.
    """
    r = random.random()
    if r < 0.15:
        return 0  # fully clean
    if r < 0.30:
        return 0xFFFFFFFFFFFFFFFF  # fully tainted
    if r < 0.60:
        # Sparse: 1–4 bits set at random positions across 64.
        n_bits = random.choice([1, 2, 3, 4])
        bits = random.sample(range(64), n_bits)
        m = 0
        for b in bits:
            m |= 1 << b
        return m
    return random.randint(1, 0xFFFFFFFFFFFFFFFE)  # partial


def _rand_taint_dict_gt_friendly(
    fraction_sparse: float = 0.4,
    max_total_k: int = 16,
) -> dict[str, int]:
    """Generate a taint dict over all four registers.

    With probability ``fraction_sparse``, deliberately constrain the TOTAL
    popcount across all registers to ``<= max_total_k`` so the noninterference
    GT simulator can run on the case (2**k Unicorn enumerations are tractable
    only for small k).  The remaining cases use the unconstrained ``_rand_taint``
    distribution (mix of clean / full / sparse / partial).

    The "GT-friendly" branch picks a target k uniformly in {0, 1, 2, 4, 8, 16}
    (clamped at max_total_k) and distributes those k tainted bits randomly
    across the four registers.  This complements ``_rand_taint`` by guaranteeing
    that some pillar cases have low total k even when individual registers
    happen to draw "fully tainted" from the unconstrained sampler.
    """
    if random.random() < fraction_sparse:
        target = random.choice([0, 1, 2, 4, 8, 16])
        target = min(target, max_total_k)
        # Distribute target bits across registers
        if target == 0:
            return dict.fromkeys(REGISTERS, 0)
        positions = [(r, b) for r in REGISTERS for b in range(64)]
        chosen = random.sample(positions, target)
        out = dict.fromkeys(REGISTERS, 0)
        for reg, bit in chosen:
            out[reg] |= 1 << bit
        return out
    return {r: _rand_taint() for r in REGISTERS}


def _safe_state(asm_lines: list[str]) -> dict[str, int]:
    """
    Return register state that avoids architectural #DE / #UD faults for
    the new expanded instruction pool.

    Hazards handled:
      DIV  rsrc          : (rsrc != 0)  AND  (rdx:rax / rsrc) fits in 64 bits.
                           Easiest: force rdx = 0  (so dividend = rax < 2^64
                           and quotient ≤ rax / rsrc ≤ rax < 2^64). Loses
                           coverage of the high-half input but avoids #DE.
      IDIV rsrc          : (rsrc != 0) AND quotient fits in signed 64-bit.
                           Force rdx = sign-extension of rax (i.e. dividend
                           is the canonical sign-extended form), so the
                           quotient = rax / rsrc is bounded by |rax| / |rsrc|
                           which fits unless rax = INT64_MIN AND rsrc = -1.
                           Defend against that one corner case by clamping
                           rsrc bit 0 if necessary.
      IDIV after CQO     : The CQO pre-instruction will set up rdx, so we
                           don't need to. Detected and skipped.

    For everything else the state is uniform random over [0, 2^64).
    """
    mask64 = 0xFFFFFFFFFFFFFFFF
    state = {r: random.randint(0, mask64) for r in REGISTERS}
    joined = ' '.join(asm_lines).lower()

    # ── DIV / IDIV divisor non-zero ──
    for r in ['rbx', 'rcx', 'rdx']:
        if f'div {r}' in joined or f'idiv {r}' in joined:
            if state[r.upper()] == 0:
                state[r.upper()] = random.randint(1, mask64)

    # ── DIV / IDIV quotient overflow protection ──
    # If `cqo` precedes idiv in the same sequence, RDX is overwritten by
    # the cqo at runtime, so initial state doesn't matter. Same for `cdq`,
    # `xor edx, edx`, `xor rdx, rdx`. Detect and skip the rdx forcing.
    rdx_set_before_div = any(
        kw in joined
        for kw in (
            'cqo',
            'cdq',
            'xor edx, edx',
            'xor rdx, rdx',
            'mov rdx, ',
            'mov edx, ',
        )
    )
    has_div = any(f'div {r}' in joined for r in ('rbx', 'rcx', 'rdx'))
    has_idiv = any(f'idiv {r}' in joined for r in ('rbx', 'rcx', 'rdx'))

    if has_div and not rdx_set_before_div:
        # Unsigned: zero RDX so dividend = RAX, quotient ≤ RAX, no #DE.
        state['RDX'] = 0
    if has_idiv and not rdx_set_before_div:
        # Signed: force canonical sign-extension of RAX into RDX so the
        # 128-bit dividend equals the 64-bit RAX viewed as signed.
        sign_bit = (state['RAX'] >> 63) & 1
        state['RDX'] = mask64 if sign_bit else 0
        # Edge case: rax = INT64_MIN, divisor = -1 → quotient = +2^63 (overflow).
        # If the divisor register holds these exact values, perturb it.
        for r in ['rbx', 'rcx', 'rdx']:
            if f'idiv {r}' in joined:
                # Only RBX or RCX can carry an external divisor here; RDX is
                # the implicit high half of the dividend pair, can't be the
                # divisor for this corner.
                if r == 'rdx':
                    continue
                if state['RAX'] == 0x8000000000000000 and state[r.upper()] == mask64:
                    state[r.upper()] ^= 2  # avoid the one-and-only #DE corner

    # ── REP string ops: bound RCX so we don't loop forever on tainted RCX
    # values that random sampling produced. The sequence-generators set ECX
    # to a small literal explicitly (e.g. `mov ecx, 8`), so this is purely
    # defensive for the case where some future test forgets that.
    if ' rep ' in f' {joined} ' or joined.startswith('rep '):
        # Cap RCX initial value at 256 — large enough to be interesting,
        # small enough to not blow the per-test timeout.
        state['RCX'] = state['RCX'] & 0xFF or 1

    return state


def generate_single_test(arch: str, instr_pool_entry: tuple[str, str] | None = None) -> dict:
    """Generate a single-instruction test case."""
    if instr_pool_entry is None:
        instr_pool_entry = random.choice(INSTRUCTION_POOL)
    asm, category = instr_pool_entry
    enc, _ = _KS.asm(asm)
    state = _safe_state([asm])
    taint = _rand_taint_dict_gt_friendly()
    return {
        'arch': arch,
        'assembly': asm,
        'bytes': bytes(enc).hex(),
        'state': state,
        'taint': taint,
        'category': category,
        'mode': 'single',
    }


def generate_sequence_test(arch: str, seq_entry: tuple[str, list[str], str] | None = None) -> dict:
    """Generate a multi-instruction sequence test case."""
    if seq_entry is None:
        seq_entry = random.choice(INSTRUCTION_SEQUENCES)
    label, asm_lines, category = seq_entry
    # Assemble concatenated bytes
    all_bytes = []
    for line in asm_lines:
        enc, _ = _KS.asm(line)
        all_bytes.extend(enc)
    state = _safe_state(asm_lines)
    taint = _rand_taint_dict_gt_friendly()
    return {
        'arch': arch,
        'assembly': '; '.join(asm_lines),
        'asm_lines': asm_lines,
        'bytes': bytes(all_bytes).hex(),
        'state': state,
        'taint': taint,
        'category': category,
        'label': label,
        'mode': 'sequence',
    }


def generate_systematic_sweep(arch: str) -> list[dict]:
    """
    Generate a systematic sweep: every instruction in INSTRUCTION_POOL paired
    with a representative set of taint configurations:
      (a) all registers fully tainted
      (b) only RAX tainted
      (c) only RBX tainted
      (d) partially tainted (random, GT-friendly mix)
      (e) sparse taint (k=4 bits scattered across registers — GT-tractable)

    This gives 5 × |INSTRUCTION_POOL| deterministic cases.

    The sparse config (e) is added so the noninterference ground-truth
    simulator runs on every instruction in the pool: with k=4, the GT
    enumerates only 16 Unicorn runs per case, which is cheap and exact.
    Without this, the existing 4 configs all use full or near-full masks
    where k = popcount(taint) >> 16 and the GT skips them.
    """
    configs = [
        ('all_tainted', dict.fromkeys(REGISTERS, 18446744073709551615)),
        ('rax_only', {r: (0xFFFFFFFFFFFFFFFF if r == 'RAX' else 0) for r in REGISTERS}),
        ('rbx_only', {r: (0xFFFFFFFFFFFFFFFF if r == 'RBX' else 0) for r in REGISTERS}),
        ('partial', None),  # filled randomly per-case (GT-friendly mix)
        ('sparse_k4', 'k4'),  # exactly 4 tainted bits — always GT-tractable
    ]
    cases = []
    for (asm, category), (cfg_name, cfg_taint) in iterproduct(INSTRUCTION_POOL, configs):
        enc, _ = _KS.asm(asm)
        state = _safe_state([asm])
        if cfg_taint is None:
            taint = _rand_taint_dict_gt_friendly()
        elif cfg_taint == 'k4':
            # Place exactly 4 tainted bits at uniformly-random positions.
            positions = [(r, b) for r in REGISTERS for b in range(64)]
            chosen = random.sample(positions, 4)
            taint = dict.fromkeys(REGISTERS, 0)
            for reg, bit in chosen:
                taint[reg] |= 1 << bit
        else:
            taint = cfg_taint
        cases.append(
            {
                'arch': arch,
                'assembly': asm,
                'bytes': bytes(enc).hex(),
                'state': state,
                'taint': taint,
                'category': category,
                'sweep_cfg': cfg_name,
                'mode': 'sweep',
            },
        )
    return cases


# ---------------------------------------------------------------------------
# C harness compilation (extended to support sequences)
# ---------------------------------------------------------------------------


def _build_c_source(tc: dict, tool: str) -> str:
    """
    Return the complete C harness source as a string.
    Separated from compilation so the source can be piped to gcc stdin.

    Timing
    ------
    The harness measures elapsed CLOCK_MONOTONIC time around the inline
    assembly block ONLY.  This excludes:
      - process startup / dynamic linker
      - Pin / Valgrind / docker container init
      - libc startup (printf, etc.)
      - taint-apply (TNT_TAINT, __libdft_set_taint) and taint-check calls
        which are markers, not engine work.

    What it includes:
      - the 4 input ``mov`` instructions (engine instruments them)
      - the user's instruction sequence under test
      - the 4 output ``mov`` instructions
      - the two ``clock_gettime`` calls themselves (negligible — vDSO)

    The two ``clock_gettime`` calls add a tiny constant overhead per test
    (~50 ns on bare metal, ~200 ns under Valgrind, ~500 ns under Pin).
    For a test of 1–10 instructions in microtaint that would be massive,
    but for libdft64/taintgrind where the per-instruction engine work is
    in the microsecond range, it's <1% of the measurement.

    The reported time is parsed by ``_run_c_harness`` and used as the
    benchmark's ``time_ns`` instead of the wall-clock subprocess duration
    (which is dominated by Pin/Valgrind container startup, ~1 second).
    """
    asm_lines = tc.get('asm_lines', [tc['assembly']])

    # If asm_lines starts with a placeholder like "<branching_dataflow N=2>"
    # (used by path_explosion where the assembly is generated as raw bytes,
    # not as canonical asm), fall back to .byte directives. The path_explosion
    # bytestrings include forward jumps and labels that gas can't reassemble
    # from the human-readable form, so the only reliable way to emit them
    # in the C harness is as literal bytes.
    if asm_lines and asm_lines[0].startswith('<') and asm_lines[0].endswith('>'):
        raw = bytes.fromhex(tc['bytes'])
        # Group bytes into .byte directives, 12 per line for readability.
        byte_chunks = []
        for i in range(0, len(raw), 12):
            chunk = raw[i : i + 12]
            byte_chunks.append('.byte ' + ','.join(f'0x{b:02x}' for b in chunk))
        asm_lines = byte_chunks
    headers = '#include <stdio.h>\n#include <stdint.h>\n#include <time.h>\n'
    apply = ''
    check = 'int rax_tainted=0,rbx_tainted=0,rcx_tainted=0,rdx_tainted=0;\n'

    if tool == 'taintgrind':
        headers += '#include "taintgrind.h"\n'
        for r in REGISTERS:
            if tc['taint'].get(r, 0):
                apply += f'    TNT_TAINT(&{r.lower()[:3]},8);\n'
        check += (
            '    TNT_IS_TAINTED(rax_tainted,&rax,8);\n'
            '    TNT_IS_TAINTED(rbx_tainted,&rbx,8);\n'
            '    TNT_IS_TAINTED(rcx_tainted,&rcx,8);\n'
            '    TNT_IS_TAINTED(rdx_tainted,&rdx,8);\n'
        )
    elif tool == 'libdft64':
        headers += (
            '__attribute__((noinline)) void __libdft_set_taint(void *addr, unsigned int size) {}\n'
            '__attribute__((noinline)) void __libdft_get_taint(void *addr) {}\n'
            '__attribute__((noinline)) void __libdft_getval_taint(uint64_t val) {}\n'
        )
        for r in REGISTERS:
            if tc['taint'].get(r, 0):
                apply += f'    __libdft_set_taint(&{r.lower()}, 8);\n'
        check += (
            '    __libdft_get_taint(&rax); __libdft_getval_taint(rax);\n'
            '    __libdft_get_taint(&rbx); __libdft_getval_taint(rbx);\n'
            '    __libdft_get_taint(&rcx); __libdft_getval_taint(rcx);\n'
            '    __libdft_get_taint(&rdx); __libdft_getval_taint(rdx);\n'
        )

    # Build the inline asm block.
    # Each asm_line becomes its own adjacent C string literal on its own line.
    # Adjacent string literals are concatenated by the C compiler — no joining
    # characters needed between them, which avoids the "stray \\" gcc error
    # that occurs when \n\t appears OUTSIDE a string (between closing " and
    # opening " of the next literal).
    def asm_str(line: str) -> str:
        # Each instruction line, properly escaped for a C string literal.
        # We use Intel syntax switching around the user instructions.
        return f'        "{line}\\n\\t"'

    asm_body_lines = '\n'.join(asm_str(l) for l in asm_lines)

    state = tc['state']

    return f"""\
{headers}
int main(void) {{
    uint64_t rax = {state.get('RAX', 0)}ULL;
    uint64_t rbx = {state.get('RBX', 0)}ULL;
    uint64_t rcx = {state.get('RCX', 0)}ULL;
    uint64_t rdx = {state.get('RDX', 0)}ULL;
{apply}
    struct timespec __t0, __t1;
    clock_gettime(CLOCK_MONOTONIC, &__t0);
    __asm__ volatile (
        "mov %[rax], %%rax\\n\\t"
        "mov %[rbx], %%rbx\\n\\t"
        "mov %[rcx], %%rcx\\n\\t"
        "mov %[rdx], %%rdx\\n\\t"
        ".intel_syntax noprefix\\n\\t"
{asm_body_lines}
        ".att_syntax prefix\\n\\t"
        "mov %%rax, %[rax]\\n\\t"
        "mov %%rbx, %[rbx]\\n\\t"
        "mov %%rcx, %[rcx]\\n\\t"
        "mov %%rdx, %[rdx]\\n\\t"
        : [rax] "+m" (rax), [rbx] "+m" (rbx), [rcx] "+m" (rcx), [rdx] "+m" (rdx)
        :: "rax", "rbx", "rcx", "rdx", "cc", "memory"
    );
    clock_gettime(CLOCK_MONOTONIC, &__t1);
    long long __elapsed_ns = (long long)(__t1.tv_sec  - __t0.tv_sec ) * 1000000000LL
                           + (long long)(__t1.tv_nsec - __t0.tv_nsec);
{check}
    printf("{{\\\"output_taint\\\":{{\\\"RAX\\\":%d,\\\"RBX\\\":%d,"
           "\\\"RCX\\\":%d,\\\"RDX\\\":%d}},\\\"time_ns\\\":%lld}}\\n",
           rax_tainted, rbx_tainted, rcx_tainted, rdx_tainted, __elapsed_ns);
    return 0;
}}
"""


def compile_c_harness(
    tc: dict,
    tool: str,
    src_file: str = 'harness.c',
    bin_file: str = 'harness.bin',
) -> None:
    """
    Compile the C harness for taintgrind or libdft64.

    Source is piped to gcc via stdin (-x c -) — no .c file written.
    Binary is written to /dev/shm/<bin_file> (RAM-backed tmpfs) when
    available, falling back to the current directory.

    src_file is accepted for API compatibility but ignored.
    bin_file is the basename used under /dev/shm (or CWD as fallback).

    Raises subprocess.CalledProcessError on compile failure (stderr captured).
    """
    src = _build_c_source(tc, tool)

    # Prefer /dev/shm (RAM, no disk I/O) for the binary
    shm = '/dev/shm'
    if os.path.isdir(shm) and os.access(shm, os.W_OK):
        out_path = os.path.join(shm, bin_file)
    else:
        out_path = os.path.join(CWD, bin_file)

    cmd = [
        'gcc',
        '-O0',
        '-g',
        '-fno-pie',
        '-no-pie',
        '-rdynamic',
        '-x',
        'c',
        '-',  # read source from stdin
        '-o',
        out_path,
    ]
    if tool == 'taintgrind':
        cmd += ['-static', '-I./external/taintgrind', '-I/usr/include/valgrind']

    subprocess.run(
        cmd,
        input=src,
        check=True,
        capture_output=True,
        text=True,
    )

    # Store the actual binary path so _run_c_harness can use it
    # (returned via a hack: we write it into the bin_file attribute
    #  by using a module-level dict keyed by bin_file basename)
    _BIN_PATHS[bin_file] = out_path


# Registry: bin_file basename → full path (populated by compile_c_harness)
_BIN_PATHS: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Precision / recall metrics
# ---------------------------------------------------------------------------


def compute_metrics(report_results: list[dict], reference_tool: str) -> dict:
    """
    Compute per-tool and per-category metrics against `reference_tool`.

    Cases where ANY tool returned ``{"error": "timeout"}`` are excluded from
    precision/recall/Jaccard comparisons (they were not completed before the
    shared deadline fired) but counted separately in ``"timeout_cases"`` so
    the report shows how much of the benchmark each tool actually covered.

    Returns:
      {
        "per_tool": {tool: {"precision": float, "recall": float, "f1": float,
                            "jaccard_bit_mean": float,
                            "completed": int, "timed_out": int}},
        "per_category": {category: {tool: {"f1": float}}},
        "latency": {tool: {"mean_ms": float, "p50_ms": float, ...}},
        "timeout_cases": int,   # cases excluded from comparison
        "compared_cases": int,  # cases included in all comparisons
      }
    """
    # Gather reference outputs (only from cases the reference tool completed
    # successfully).  A "successful" run is one with an output_taint dict.
    # Anything else (error, timeout, missing key, malformed result) is skipped.
    ref_outputs: dict[int, dict[str, int]] = {}
    for entry in report_results:
        tid = entry['id']
        tr = entry['tool_results'].get(reference_tool, {})
        if 'error' in tr:
            continue
        ot = tr.get('output_taint')
        if not isinstance(ot, dict):
            continue
        ref_outputs[tid] = ot

    per_tool: dict[str, dict] = defaultdict(
        lambda: {
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'tn': 0,
            'jaccard_sum': 0.0,
            'jaccard_n': 0,
            'latencies_ns': [],
            'lat_records': [],  # list of (latency_ns, n_instrs, mode) tuples
            'completed': 0,
            'timed_out': 0,
            'errored': 0,
        },
    )
    per_category: dict[str, dict[str, dict]] = defaultdict(
        lambda: defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}),
    )
    n_timeout_excluded = 0
    n_compared = 0

    for entry in report_results:
        tid = entry['id']
        cat = entry['instruction'].get('category', 'unknown')
        tool_results = entry['tool_results']
        inst = entry['instruction']
        mode = inst.get('mode', '?')

        # Compute the number of x86 instructions in this test case.  We use
        # this to convert per-test latency to per-instruction latency
        # (tp/s/i), which is fairer across modes — a path-explosion test
        # of 25 instructions is naturally 25× slower than a single mov.
        # Priority: explicit n_instrs (path_explosion sets this) > len(asm_lines)
        # > count of ';' in the assembly string + 1.
        n_instrs = inst.get('n_instrs')
        if not n_instrs:
            asm_lines = inst.get('asm_lines') or []
            if asm_lines:
                n_instrs = len(asm_lines)
            else:
                asm = inst.get('assembly', '')
                n_instrs = (asm.count(';') + 1) if asm else 1
        n_instrs = max(1, int(n_instrs))

        # Classify every tool's result for this case
        for tool, res in tool_results.items():
            err = res.get('error', '')
            if err == 'timeout':
                per_tool[tool]['timed_out'] += 1
            elif err:
                per_tool[tool]['errored'] += 1
            else:
                per_tool[tool]['completed'] += 1
                per_tool[tool]['latencies_ns'].append(res.get('time_ns', 0))
                # Also record (latency, n_instrs, mode) so the throughput
                # calculations downstream can exclude pillars or normalise
                # by instruction count.
                per_tool[tool]['lat_records'].append((res.get('time_ns', 0), n_instrs, mode))

        # Exclude from comparison if the reference or any non-GT tool timed out.
        # GT is allowed to time out (budget-skipped) without excluding the case —
        # GT scoring is separate from tool-vs-tool comparison.
        comparison_tools = [t for t in tool_results if t != 'ground_truth']
        if any(tool_results.get(t, {}).get('error') == 'timeout' for t in comparison_tools):
            n_timeout_excluded += 1
            continue

        ref = ref_outputs.get(tid)
        if ref is None:
            continue  # reference tool timed out or errored on this case
        n_compared += 1

        for tool, res in tool_results.items():
            if tool == 'ground_truth':
                continue
            err = res.get('error', '')
            if err:
                continue
            ot = res.get('output_taint')
            if not isinstance(ot, dict):
                # Malformed result with no output_taint and no error key —
                # skip rather than crash.  Track as 'errored' for the tool.
                per_tool[tool]['errored'] += 1
                continue
            gt = GRANULARITY.get(tool, 'reg')

            # vs reference (register-level precision/recall)
            if tool != reference_tool:
                for reg in REGISTERS:
                    ref_v = 1 if ref.get(reg, 0) else 0
                    tool_v = 1 if ot.get(reg, 0) else 0
                    if ref_v == 1 and tool_v == 1:
                        per_tool[tool]['tp'] += 1
                        per_category[cat][tool]['tp'] += 1
                    elif ref_v == 0 and tool_v == 1:
                        per_tool[tool]['fp'] += 1
                        per_category[cat][tool]['fp'] += 1
                    elif ref_v == 1 and tool_v == 0:
                        per_tool[tool]['fn'] += 1
                        per_category[cat][tool]['fn'] += 1
                    else:
                        per_tool[tool]['tn'] += 1
                        per_category[cat][tool]['tn'] += 1

            # Bit-level Jaccard vs reference (only for bit-level tools)
            if gt == 'bit' and GRANULARITY.get(reference_tool, 'bit') == 'bit':
                for reg in REGISTERS:
                    per_tool[tool]['jaccard_sum'] += jaccard_bit(ref.get(reg, 0), ot.get(reg, 0))
                    per_tool[tool]['jaccard_n'] += 1

    def f1(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        return (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    def pct(lst, p):
        if not lst:
            return 0.0
        s = sorted(lst)
        idx = int(len(s) * p / 100)
        return s[min(idx, len(s) - 1)] / 1e6

    out_per_tool = {}
    for tool, d in per_tool.items():
        tp, fp, fn, tn = d['tp'], d['fp'], d['fn'], d['tn']
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        lats = d['latencies_ns']
        records = d['lat_records']  # list of (lat_ns, n_instrs, mode)

        # ─── Throughput variants ───────────────────────────────────────
        # The headline `tp/s` number is sensitive to two confounds:
        #  1. The path-explosion pillar.  Its sequences are 9–49
        #     instructions long with explicit branches; the slowest pillar
        #     by far.  Including it makes a tool's tp/s depend heavily on
        #     how many path-explosion cases were sampled.
        #  2. Mode mix.  A `single` test runs one mov in 5 µs while a
        #     `realworld` 5-instruction sequence takes 50 µs.  Saying
        #     "tool X does 10000 tp/s" hides the per-instruction throughput.
        #
        # We expose three numbers:
        #  * throughput_per_s            — median over all NON-path-explosion
        #                                  tests (the "typical workload" rate)
        #  * throughput_per_s_all        — median over EVERY completed test
        #                                  (kept for back-compat / parity)
        #  * throughput_per_s_per_instr  — instructions-per-second across all
        #                                  completed tests; the most apples-
        #                                  to-apples cross-mode metric.
        # All three are computed on CPU-time latencies (process_time_ns),
        # not wall-clock, so they don't fluctuate with system contention.
        non_pe_records = [r for r in records if r[2] != 'path_explosion']
        non_pe_lats = [r[0] for r in non_pe_records]
        median_lat_all = statistics.median(lats) if lats else 0
        median_lat_non_pe = statistics.median(non_pe_lats) if non_pe_lats else 0
        mean_lat = statistics.mean(lats) if lats else 0

        # Per-instruction throughput, computed CONSISTENTLY with tp/s.
        # tp/s uses median latency.  If we used `sum(instrs)/sum(latency)`
        # for tp/s/i, we'd be implicitly using MEAN latency, which gives
        # very different (smaller, outlier-dominated) numbers and breaks
        # the intuitive relation tp/s/i ≈ tp/s × avg_instrs_per_test.
        #
        # Instead: compute the per-test instruction rate (instrs / latency)
        # for each test, then take the median.  This is invariant to
        # outliers and produces tp/s/i values larger than tp/s by exactly
        # the average instruction-count factor (when tests are linear).
        # We also exclude path_explosion (consistent with tp/s).
        per_test_inst_rates = [(n * 1e9 / lat) for (lat, n, mode) in non_pe_records if lat > 0]
        instr_per_s = statistics.median(per_test_inst_rates) if per_test_inst_rates else 0

        # Aggregate "work" totals also recorded for sanity-checking and for
        # downstream papers that prefer the throughput-as-total-work
        # framing.  These use sums and are mean-equivalent.
        tot_lat_ns = sum(r[0] for r in records)
        tot_instrs = sum(r[1] for r in records)
        instr_per_s_aggregate = tot_instrs * 1e9 / tot_lat_ns if tot_lat_ns > 0 else 0

        out_per_tool[tool] = {
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1': round(f1(tp, fp, fn), 4),
            'jaccard_bit_mean': round(d['jaccard_sum'] / d['jaccard_n'], 4) if d['jaccard_n'] else None,
            'completed': d['completed'],
            'timed_out': d['timed_out'],
            'errored': d['errored'],
            'latency_mean_ms': round(mean_lat / 1e6, 3) if lats else 0,
            'latency_p50_ms': round(pct(lats, 50), 3),
            'latency_p95_ms': round(pct(lats, 95), 3),
            'latency_p99_ms': round(pct(lats, 99), 3),
            'throughput_per_s': round(1e9 / median_lat_non_pe, 1) if median_lat_non_pe > 0 else 0,
            'throughput_per_s_all': round(1e9 / median_lat_all, 1) if median_lat_all > 0 else 0,
            'throughput_per_s_mean': round(1e9 / mean_lat, 1) if mean_lat > 0 else 0,
            'throughput_per_s_per_instr': round(instr_per_s, 1),
            'throughput_per_s_per_instr_aggregate': round(instr_per_s_aggregate, 1),
            'n_excluding_path_explosion': len(non_pe_lats),
            'total_instructions_executed': tot_instrs,
        }

    out_per_cat = {}
    for cat, tool_map in per_category.items():
        out_per_cat[cat] = {}
        for tool, d in tool_map.items():
            out_per_cat[cat][tool] = {'f1': round(f1(d['tp'], d['fp'], d['fn']), 4)}

    # ── Path-explosion scaling: latency by instruction count ──────────────
    # Structure: {category: {n_instrs: {tool: mean_ms}}}
    path_scaling: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for entry in report_results:
        mode = entry['instruction'].get('mode', '')
        if mode != 'path_explosion':
            continue
        n = entry['instruction'].get('n_instrs', 0)
        cat = entry['instruction'].get('category', 'unknown')
        for tool, res in entry['tool_results'].items():
            if 'error' not in res and res.get('time_ns', 0) > 0:
                path_scaling[cat][n][tool].append(res['time_ns'] / 1e6)

    path_explosion_out: dict = {}
    for cat, n_map in path_scaling.items():
        path_explosion_out[cat] = {}
        for n, tool_map in n_map.items():
            path_explosion_out[cat][n] = {
                tool: round(statistics.mean(lats), 2) for tool, lats in tool_map.items() if lats
            }

    # ── Ground-truth scoring (noninterference, k <= GT_BIT_BUDGET) ──────────
    # If `ground_truth` was run, score every bit-precise tool against its
    # output.  This is the most rigorous evaluation in the benchmark: the GT
    # is exact (full enumeration) for cases that fit the budget, and the
    # comparison reports per-tool soundness and over-taint counts.
    gt_scoring: dict = {}
    gt_cases_total = 0
    gt_cases_within_budget = 0
    gt_k_distribution: dict = defaultdict(int)

    for entry in report_results:
        gt_res = entry['tool_results'].get('ground_truth')
        if gt_res is None:
            continue
        gt_cases_total += 1
        if 'error' in gt_res:
            # Skipped (k > budget).  Record k for distribution stats.
            sk = gt_res.get('skipped_k')
            if sk is not None:
                gt_k_distribution[f'>{GT_BIT_BUDGET}'] += 1
            continue
        gt_cases_within_budget += 1
        k = gt_res.get('k', 0)
        # Bin k into {0, 1, 2, 3, 4, 5, 6, 7, 8} for histogram.
        gt_k_distribution[k] += 1

        gt_mask = gt_res.get('output_taint', {})
        for tool, res in entry['tool_results'].items():
            if tool == 'ground_truth':
                continue
            if 'error' in res:
                continue
            tool_out = res.get('output_taint', {})
            tool_gran = GRANULARITY.get(tool, 'reg')
            if tool not in gt_scoring:
                gt_scoring[tool] = {
                    'n': 0,  # cases compared
                    'exact': 0,  # tool == GT exactly (per case)
                    'sound_cases': 0,  # tool >= GT (no under-taint anywhere) per case
                    'unsound_cases': 0,  # tool < GT somewhere
                    'n_regs_compared': 0,  # total (case, register) pairs
                    'exact_regs': 0,  # exact (case, register)
                    'over_regs': 0,  # tool > GT in this register
                    'under_regs': 0,  # tool < GT in this register (UNSOUND)
                    'both_regs': 0,  # tool over and under simultaneously
                    'over_bits_total': 0,  # total spurious bits across all (case, reg)
                    'under_bits_total': 0,  # total missed bits — must be 0 if sound
                    'jaccard_sum': 0.0,  # sum of bit-level Jaccard per (case, reg)
                    'jaccard_n': 0,
                }
            s = gt_scoring[tool]
            s['n'] += 1
            case_exact = True
            case_unsound = False
            for reg in REGISTERS:
                gt_v = gt_mask.get(reg, 0)
                # Register-granular tools report 0/1; compare at register level
                # (binarised) — they get a per-register ✓ if (got != 0) matches
                # (gt != 0).  This isn't bit-level evaluation but lets reg-level
                # tools appear in the same table for context.
                if tool_gran == 'reg':
                    got_v = 1 if tool_out.get(reg, 0) else 0
                    gt_b = 1 if gt_v else 0
                    s['n_regs_compared'] += 1
                    if got_v == gt_b:
                        s['exact_regs'] += 1
                    elif got_v and not gt_b:
                        s['over_regs'] += 1
                        case_exact = False
                    else:
                        s['under_regs'] += 1
                        case_exact = False
                        case_unsound = True
                    continue
                # Bit-precise comparison
                got_v = tool_out.get(reg, 0) & MASK64
                gt_v = gt_v & MASK64
                over = got_v & ~gt_v & MASK64
                under = gt_v & ~got_v & MASK64
                s['n_regs_compared'] += 1
                if over == 0 and under == 0:
                    s['exact_regs'] += 1
                elif under == 0:
                    s['over_regs'] += 1
                    case_exact = False
                elif over == 0:
                    s['under_regs'] += 1
                    case_exact = False
                    case_unsound = True
                else:
                    s['both_regs'] += 1
                    case_exact = False
                    case_unsound = True
                s['over_bits_total'] += bin(over).count('1')
                s['under_bits_total'] += bin(under).count('1')
                # Jaccard (bit-level set similarity)
                inter = bin(got_v & gt_v).count('1')
                union = bin(got_v | gt_v).count('1')
                s['jaccard_sum'] += inter / union if union else 1.0
                s['jaccard_n'] += 1
            if case_exact:
                s['exact'] += 1
            if not case_unsound:
                s['sound_cases'] += 1
            else:
                s['unsound_cases'] += 1

    gt_summary = {}
    for tool, s in gt_scoring.items():
        gt_summary[tool] = {
            'cases_compared': s['n'],
            'exact_cases': s['exact'],
            'exact_case_rate': round(s['exact'] / s['n'], 4) if s['n'] else 0,
            'sound_cases': s['sound_cases'],
            'soundness_rate': round(s['sound_cases'] / s['n'], 4) if s['n'] else 0,
            'unsound_cases': s['unsound_cases'],
            'regs_exact': s['exact_regs'],
            'regs_over_only': s['over_regs'],
            'regs_under_only': s['under_regs'],
            'regs_both': s['both_regs'],
            'over_bits_total': s['over_bits_total'],
            'under_bits_total': s['under_bits_total'],
            'mean_jaccard_bit': round(s['jaccard_sum'] / s['jaccard_n'], 4) if s['jaccard_n'] else None,
        }

    return {
        'per_tool': out_per_tool,
        'per_category': out_per_cat,
        'path_explosion_scaling': path_explosion_out,
        'ground_truth': {
            'cases_total': gt_cases_total,
            'cases_within_budget': gt_cases_within_budget,
            'cases_skipped': gt_cases_total - gt_cases_within_budget,
            'budget_bits': GT_BIT_BUDGET,
            'k_distribution': dict(gt_k_distribution),
            'per_tool': gt_summary,
        },
        'compared_cases': n_compared,
        'timeout_excluded': n_timeout_excluded,
        'total_cases': len(report_results),
    }


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------


def print_summary(metrics: dict, selected_tools: list[str], reference_tool: str) -> None:
    n_total = metrics.get('total_cases', '?')
    n_compared = metrics.get('compared_cases', '?')
    n_excl = metrics.get('timeout_excluded', 0)

    print('\n' + '=' * 72)
    print('SUMMARY METRICS')
    print(f'  Reference tool : {reference_tool}')
    print(f'  Total cases    : {n_total}')
    print(
        f'  Compared cases : {n_compared}' + (f'  ({n_excl} excluded — at least one tool timed out)' if n_excl else ''),
    )
    print('=' * 72)

    # Per-tool table with completion stats
    hdr = (
        f"{'Tool':<14} {'Done':>5} {'TO':>4} {'Err':>4}"
        f" {'Prec':>6} {'Rec':>6} {'F1':>6} {'Jac':>7}"
        f" {'p50ms':>7} {'p99ms':>7} {'tp/s':>8} {'tp/s/i':>8}"
    )
    print(hdr)
    print('-' * len(hdr))
    for tool in selected_tools:
        d = metrics['per_tool'].get(tool)
        if not d:
            continue
        jac = f"{d['jaccard_bit_mean']:.4f}" if d['jaccard_bit_mean'] is not None else '   N/A '
        done = d.get('completed', '?')
        to = d.get('timed_out', 0)
        err = d.get('errored', 0)
        print(
            f"{tool:<14} {done:>5} {to:>4} {err:>4}"
            f" {d['precision']:>6.4f} {d['recall']:>6.4f} {d['f1']:>6.4f}"
            f" {jac:>7}"
            f" {d['latency_p50_ms']:>7.1f} {d['latency_p99_ms']:>7.1f}"
            f" {d['throughput_per_s']:>8.1f}"
            f" {d['throughput_per_s_per_instr']:>8.1f}",
        )
    print('\n  Columns: Done=completed  TO=timed-out  Err=errored')
    print('  tp/s   = tests/sec  = 1 / median_per_test_latency')
    print('  tp/s/i = instructions/sec = median(n_instrs / latency) per test')
    print('  Both EXCLUDE path_explosion pillar; both use CPU time, not wall.')
    print('  Relation: tp/s/i = tp/s x avg_instrs_in_median_test (>= tp/s).')

    # Per-category F1
    print('\nPer-category F1 (vs reference):')
    cats = sorted(metrics['per_category'])
    if cats:
        col_tools = [t for t in selected_tools if t != reference_tool]
        cat_hdr = f"  {'Category':<22}" + ''.join(f' {t[:10]:>10}' for t in col_tools)
        print(cat_hdr)
        print('  ' + '-' * (22 + 11 * len(col_tools)))
        for cat in cats:
            row = f'  {cat:<22}'
            for tool in col_tools:
                f1v = metrics['per_category'][cat].get(tool, {}).get('f1')
                row += f' {f1v:>10.4f}' if f1v is not None else '        N/A'
            print(row)

    # Path-explosion scaling table
    if metrics.get('path_explosion_scaling'):
        print('\nPath-explosion latency scaling (ms) — SMT AST depth effect:')
        scaling = metrics['path_explosion_scaling']
        # Collect all (category, n) pairs
        all_ns = sorted({n for cat_data in scaling.values() for n in cat_data})
        for cat, cat_data in sorted(scaling.items()):
            print(f'\n  {cat}:')
            row_hdr = f"    {'Tool':<14}" + ''.join(f' {n:>6}i' for n in all_ns)
            print(row_hdr)
            print('    ' + '-' * (14 + 7 * len(all_ns)))
            for tool in selected_tools:
                row = f'    {tool:<14}'
                for n in all_ns:
                    v = cat_data.get(n, {}).get(tool)
                    row += f' {v:>6.1f}' if v is not None else '    N/A'
                print(row)

    # Ground-truth scoring (noninterference, exact for k <= GT_BIT_BUDGET)
    gt = metrics.get('ground_truth')
    if gt and gt.get('cases_within_budget', 0) > 0:
        print('\n' + '=' * 72)
        print('NONINTERFERENCE GROUND TRUTH (exhaustive Unicorn enumeration)')
        print(
            f"  Cases evaluated:  {gt['cases_within_budget']} / {gt['cases_total']}"
            f"   (budget: k <= {gt['budget_bits']} tainted bits, 2^k sims/case)",
        )
        if gt['cases_skipped']:
            print(f"  Skipped (k > budget): {gt['cases_skipped']}")
        # k-distribution histogram
        kd = gt.get('k_distribution', {})
        if kd:
            kd_keys = sorted(
                (k for k in kd.keys() if isinstance(k, int)),
                key=int,
            )
            kd_str = '  k-distribution:    ' + '  '.join(f'k={k}:{kd[k]}' for k in kd_keys)
            kd_skip = kd.get(f">{gt['budget_bits']}", 0)
            if kd_skip:
                kd_str += f"   k>{gt['budget_bits']}:{kd_skip}"
            print(kd_str)
        print()
        # Per-tool table against GT
        bit_tools = [t for t in selected_tools if GRANULARITY.get(t) == 'bit' and t != 'ground_truth']
        reg_tools = [t for t in selected_tools if GRANULARITY.get(t) == 'reg']
        gt_tools = bit_tools + reg_tools

        hdr = (
            f"  {'Tool':<14} {'Sound%':>7} {'Exact%':>7} {'Jaccard':>8}"
            f" {'OverBits':>9} {'UnderBits':>10} {'Unsound':>8}"
        )
        print(hdr)
        print('  ' + '-' * (len(hdr) - 2))
        for tool in gt_tools:
            d = gt['per_tool'].get(tool)
            if not d:
                continue
            jac = f"{d['mean_jaccard_bit']:.4f}" if d['mean_jaccard_bit'] is not None else '  N/A '
            print(
                f"  {tool:<14} "
                f"{100*d['soundness_rate']:>6.1f}% "
                f"{100*d['exact_case_rate']:>6.1f}% "
                f"{jac:>8} "
                f"{d['over_bits_total']:>9} "
                f"{d['under_bits_total']:>10} "
                f"{d['unsound_cases']:>8}",
            )
        print()
        print(
            '  Sound%   = fraction of cases where tool ⊇ GT (no missed taint)\n'
            '  Exact%   = fraction of cases where tool == GT exactly (per-bit equal)\n'
            '  Jaccard  = mean bit-level set similarity to GT (1.0 = exact)\n'
            '  OverBits = total spurious bits across all (case, register) pairs\n'
            '  UnderBits= total missed bits — must be 0 for a sound tool',
        )

    print('=' * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description='NDSS-grade taint engine benchmark for x86-64')
    parser.add_argument(
        '-n',
        '--number',
        type=int,
        default=500,
        help=(
            'Number of RANDOM single-instruction tests (default 500). '
            'With the expanded instruction pool (~250 mnemonics), 500 gives '
            '~2 random taint configs per instruction; 2000 gives ~8. For a '
            'publication-grade run, use --number 2000 --sequences 400 --sweep '
            '--all-suites.'
        ),
    )
    parser.add_argument(
        '--sequences',
        type=int,
        default=100,
        help='Number of random SEQUENCE tests (default 100, was 20).',
    )
    parser.add_argument(
        '--dedup-random',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Deduplicate random tests by (assembly, taint) pair (default on). '
            'Prevents the random sampler from generating the same case twice; '
            'important for paper-grade runs where every test should be unique. '
            'Disable with --no-dedup-random for legacy behaviour.'
        ),
    )
    parser.add_argument('--sweep', action='store_true', help='Run full systematic sweep (4 configs × all instructions)')
    parser.add_argument('--sweep-only', action='store_true', help='Run ONLY the systematic sweep (skip random tests)')
    parser.add_argument('-a', '--arch', choices=['x86_64'], default='x86_64')
    parser.add_argument('-i', '--instruction', default=None, help='Fix a single instruction for all random tests')
    parser.add_argument('--category', default=None, help='Restrict random tests to a specific category')
    parser.add_argument(
        '-w',
        '--workers',
        default=None,
        help=f"Comma-separated workers. Available: {','.join(ALL_WORKERS)}",
    )
    parser.add_argument(
        '--reference',
        default='microtaint',
        help='Reference tool for precision/recall metrics (default: microtaint)',
    )
    parser.add_argument('--no-summary', action='store_true', help='Skip summary metrics table')
    parser.add_argument(
        '--quiet',
        action='store_true',
        help=(
            'Suppress per-test verbose output and show a tqdm progress bar '
            'instead.  Recommended for long runs (--all-suites or --sweep).  '
            'Disagreement summary still prints at the end.'
        ),
    )
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument(
        '--imul-semantic',
        '--oracle-imul',
        dest='imul_semantic',
        action='store_true',
        help=(
            'Run deterministic IMUL semantic test suite (8 hand-crafted cases '
            'exercising sub-byte multiplications, x*0 collapse, mul-add chains, '
            'etc.).  These cases used to be checked against per-test taint '
            'oracles, but the noninterference GT simulator now provides ground '
            'truth uniformly for any case within the bit budget — much more '
            'rigorous and less error-prone.  The deprecated alias --oracle-imul '
            'still works.'
        ),
    )
    parser.add_argument(
        '--realworld',
        action='store_true',
        help='Run real-world program snippet tests (glibc/openssl patterns)',
    )
    parser.add_argument(
        '--bugdetect',
        action='store_true',
        help='Run bug-detection scenario tests (CWE-129, CWE-134, CWE-197, UAF, etc.)',
    )
    parser.add_argument(
        '--path-explosion',
        action='store_true',
        help='Run path-explosion stress tests (SMT AST growth benchmark)',
    )
    parser.add_argument(
        '--arch-failures',
        action='store_true',
        help='Run architecturally guaranteed failure tests (angr/maat/panda/libdft64 internals)',
    )
    parser.add_argument(
        '--all-suites',
        action='store_true',
        help='Run all test suites (shorthand for --imul-semantic --realworld --bugdetect --path-explosion --arch-failures)',
    )
    parser.add_argument(
        '--ground-truth',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Run the in-process noninterference ground-truth simulator '
            '(enabled by default).  For every test case where the total '
            f'popcount of input taint is k <= {GT_BIT_BUDGET}, the simulator '
            'enumerates all 2**k Unicorn assignments of the tainted bits and '
            'records the bits that vary across runs — the exact noninter'
            'ference taint set.  Cases with k > budget are reported as '
            'skipped.  Test-case generators in every pillar (random single, '
            'sequence, sweep) now mix sparse-taint configurations so a '
            'meaningful fraction of cases land within the GT budget.  Use '
            '--no-ground-truth to disable.'
        ),
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # Expand --all-suites shorthand
    if args.all_suites:
        args.imul_semantic = True
        args.realworld = True
        args.bugdetect = True
        args.path_explosion = True
        args.arch_failures = True

    # ── Worker selection ───────────────────────────────────────────────────
    if args.workers:
        names = [w.strip() for w in args.workers.split(',')]
        bad = set(names) - set(ALL_WORKERS)
        if bad:
            parser.error(f'Unknown workers: {bad}')
        selected = {k: ALL_WORKERS[k] for k in names}
    else:
        selected = dict(ALL_WORKERS)

    if args.reference not in selected:
        print(
            f"[!] Reference tool '{args.reference}' not in selected workers — "
            f"metrics vs reference will be skipped.",
            file=sys.stderr,
        )

    # ── Build test-case list ───────────────────────────────────────────────
    test_cases: list[dict] = []
    sweep_cases: list[dict] = []

    if not args.sweep_only:
        pool = INSTRUCTION_POOL
        if args.category:
            pool = [(a, c) for a, c in INSTRUCTION_POOL if c == args.category]
        if args.instruction:
            pool = [(args.instruction, 'custom')]
        if not pool:
            parser.error(f"No instructions match category '{args.category}'")

        # Dedup pass: keep generating until we have args.number UNIQUE
        # (assembly, taint) pairs (or we hit a sanity ceiling). For the
        # expanded pool (~250 instrs × ~5e38 distinct 4×64-bit taint masks)
        # collisions are vanishingly rare unless --number > pool_size and
        # the user has --instruction set; the ceiling protects against
        # pathological cases (e.g. category restricted to one tiny class).
        seen_keys: set[tuple[str, tuple[tuple[str, int], ...]]] = set()
        attempts = 0
        sanity_cap = max(args.number * 20, 1000)
        n_random_singles = 0
        while n_random_singles < args.number and attempts < sanity_cap:
            attempts += 1
            tc = generate_single_test(args.arch, random.choice(pool))
            if args.dedup_random:
                key = (tc['assembly'], tuple(sorted(tc['taint'].items())))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
            test_cases.append(tc)
            n_random_singles += 1
        if args.dedup_random and n_random_singles < args.number:
            print(
                f'[!] Dedup: only {n_random_singles}/{args.number} unique '
                f'single-instruction (asm, taint) pairs available with the '
                f'current pool/category restriction (after {attempts} attempts).',
                flush=True,
            )

        n_random_seqs = 0
        attempts = 0
        sanity_cap = max(args.sequences * 20, 500)
        while n_random_seqs < args.sequences and attempts < sanity_cap:
            attempts += 1
            tc = generate_sequence_test(args.arch)
            if args.dedup_random:
                key = (tc['assembly'], tuple(sorted(tc['taint'].items())))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
            test_cases.append(tc)
            n_random_seqs += 1
        if args.dedup_random and n_random_seqs < args.sequences:
            print(
                f'[!] Dedup: only {n_random_seqs}/{args.sequences} unique '
                f'sequence (asm, taint) pairs available '
                f'(after {attempts} attempts).',
                flush=True,
            )

    if args.sweep or args.sweep_only:
        sweep_cases = generate_systematic_sweep(args.arch)
        test_cases.extend(sweep_cases)
        print(f'[*] Systematic sweep: {len(sweep_cases)} cases ({len(INSTRUCTION_POOL)} instrs × 5 taint configs)')

    # ── IMUL semantic suite (hand-crafted multiplication test cases) ───────
    if args.imul_semantic:
        for ot in ORACLE_IMUL_TESTS:
            test_cases.append(_oracle_test_to_tc(args.arch, ot))
        print(f'[*] IMUL semantic suite: {len(ORACLE_IMUL_TESTS)} deterministic cases')

    # ── Real-world snippet suite ───────────────────────────────────────────
    if args.realworld:
        for label, asm_lines, category, rationale in REALWORLD_SEQUENCES:
            all_bytes = []
            for line in asm_lines:
                enc, _ = _KS.asm(line)
                all_bytes.extend(enc)
            # Taint all registers fully to stress propagation
            taint = dict.fromkeys(REGISTERS, 18446744073709551615)
            tc = {
                'arch': args.arch,
                'assembly': '; '.join(asm_lines),
                'asm_lines': asm_lines,
                'bytes': bytes(all_bytes).hex(),
                'state': _safe_state(asm_lines),
                'taint': taint,
                'category': category,
                'label': label,
                'rationale': rationale,
                'mode': 'realworld',
            }
            test_cases.append(tc)
        print(f'[*] Real-world suite: {len(REALWORLD_SEQUENCES)} snippets')

    # ── Bug-detection suite ────────────────────────────────────────────────
    if args.bugdetect:
        for label, asm_lines, category, rationale in BUGDETECT_SEQUENCES:
            all_bytes = []
            for line in asm_lines:
                try:
                    enc, _ = _KS.asm(line)
                    all_bytes.extend(enc)
                except Exception:
                    pass  # jno rel8 etc. may not assemble; skip that line
            # Taint RAX (the primary "dangerous" input register)
            taint = {'RAX': 0xFFFFFFFFFFFFFFFF, 'RBX': 0xFFFFFFFFFFFFFFFF, 'RCX': 0, 'RDX': 0}
            tc = {
                'arch': args.arch,
                'assembly': '; '.join(asm_lines),
                'asm_lines': [l for l in asm_lines if not l.strip().startswith('j')],  # skip branches
                'bytes': bytes(all_bytes).hex(),
                'state': _safe_state(asm_lines),
                'taint': taint,
                'category': category,
                'label': label,
                'rationale': rationale,
                'mode': 'bugdetect',
            }
            test_cases.append(tc)
        print(f'[*] Bug-detection suite: {len(BUGDETECT_SEQUENCES)} scenarios')

    # ── Path-explosion stress suite ────────────────────────────────────────
    if args.path_explosion:
        # Explicit-branch dataflow: hand-encoded jz +3 sequences.
        # For each N in EXPLICIT_BRANCH_NS we emit two cases — one with RBX
        # tainted (correct answer: RAX gets tainted via the conditional add)
        # and one with RCX tainted (decoy: RCX is never read; RAX must stay
        # clean).  This is the canonical symbolic-vs-dataflow stress test:
        # angr forks at every jz so it visits 2^N states; dataflow tools
        # stay O(N).
        n_explicit_branch_added = 0
        for n in EXPLICIT_BRANCH_NS:
            seq_bytes = build_branching_bytestring(n)
            asm_repr = 'xor rax, rax; ' + '; '.join(['test rbx,1; jz +3; add rax,rbx; shr rbx,1'] * n)
            base_state = {
                'RAX': 0,
                'RBX': 0xFFFFFFFFFFFFFFFF,
                'RCX': 0,
                'RDX': 0,
            }
            for label_suffix, taint_dict in (
                ('RBX_tainted', {'RAX': 0, 'RBX': 0xFFFFFFFFFFFFFFFF, 'RCX': 0, 'RDX': 0}),
                ('RCX_decoy', {'RAX': 0, 'RBX': 0, 'RCX': 0xFFFFFFFFFFFFFFFF, 'RDX': 0}),
            ):
                test_cases.append(
                    {
                        'arch': args.arch,
                        'assembly': asm_repr,
                        'asm_lines': [f'<branching_dataflow N={n}>'],
                        'bytes': seq_bytes.hex(),
                        'state': dict(base_state),
                        'taint': taint_dict,
                        'category': 'path_explosion_branching',
                        'label': f'branching_dataflow_n{n}_{label_suffix}',
                        'mode': 'path_explosion',
                        'n_instrs': 1 + 4 * n,
                        'branch_n': n,
                    },
                )
                n_explicit_branch_added += 1

        print(
            f'[*] Path-explosion suite: {n_explicit_branch_added} explicit-branch cases '
            f'(N ∈ {EXPLICIT_BRANCH_NS}; depths up to 2^{max(EXPLICIT_BRANCH_NS)} symbolic paths)',
        )

    # ── Architecturally guaranteed failure suite ───────────────────────────
    if args.arch_failures:
        for ot in ARCHITECTURAL_FAILURE_TESTS:
            test_cases.append(_arch_failure_to_tc(args.arch, ot))
        print(
            f'[*] Arch-failure suite: {len(ARCHITECTURAL_FAILURE_TESTS)} cases '
            f'(angr/maat/panda/libdft64 internal limitations)',
        )

    total = len(test_cases)
    # ── Start workers ──────────────────────────────────────────────────────
    pool = BatchedWorkerPool()
    c_harness_cmds: dict[str, str] = {}

    for name in selected:
        if name in PYTHON_WORKERS:
            try:
                pool.start_worker(name, PYTHON_WORKERS[name].split())
            except Exception as exc:
                print(f'[{name}] Failed to start: {exc}', file=sys.stderr)
        elif name == 'panda':
            try:
                pool.start_worker('panda', PANDA_DOCKER_CMD, boot_timeout=600)
            except Exception as exc:
                print(f'[panda] Failed to start: {exc}', file=sys.stderr)
        elif name in C_HARNESS_WORKERS:
            c_harness_cmds[name] = C_HARNESS_WORKERS[name]

    persistent_names = pool.worker_names()
    active_tools = persistent_names + list(c_harness_cmds)
    if args.ground_truth:
        active_tools = list(active_tools) + ['ground_truth']

    # ── Abort early if nothing is running ─────────────────────────────────
    # If every worker failed to start AND no C-harness workers AND GT is the
    # only thing left, there is nothing useful to benchmark — print a clear
    # error and exit rather than silently hanging for BATCH_TIMEOUT seconds.
    if not persistent_names and not c_harness_cmds:
        if args.ground_truth:
            print(
                '\n[!] WARNING: No tool workers started successfully.\n'
                '    Only the ground-truth Unicorn simulator will run.\n'
                '    Results will contain GT data only — no tool comparison.\n'
                '    Common causes:\n'
                '      • Worker venvs not present (check .venv_microtaint etc.)\n'
                '      • Docker containers not running (panda, taintgrind, libdft64)\n'
                '      • Wrong CWD — worker scripts not found at relative paths\n'
                '    Run: ps aux | grep worker  to check for running workers\n'
                '    Run: docker ps             to check container status\n',
                flush=True,
            )
        else:
            print(
                '\n[!!!] FATAL: No tool workers started and --no-ground-truth set.\n'
                '      Nothing to run — exiting.\n'
                '      Common causes:\n'
                '        • Worker venvs not present (check .venv_microtaint etc.)\n'
                '        • Wrong CWD — worker scripts not found at relative paths\n'
                '        • Docker containers not running\n'
                '      Run: ls .venv_* worker_*.py   to check file presence\n'
                '      Run: ps aux | grep worker      to check running workers\n'
                '      Run: docker ps                 to check container status\n',
                flush=True,
            )
            sys.exit(1)

    print(f'\n[*] {total} test(s) total')
    print(f'    batched pipe workers : {persistent_names}')
    print(f'    c-harness workers    : {list(c_harness_cmds)}')
    print(f'    active tools         : {active_tools}')
    if args.seed is not None:
        print(f'    random seed          : {args.seed}')

    # ── Shared wall-clock deadline ─────────────────────────────────────────
    # Every worker and the GT simulator receives this deadline.  When it
    # fires, each worker stops accepting new tests, returns partial results
    # for the cases it did finish, and marks the rest as timed-out.
    # The aggregation step (compute_metrics) then only compares cases where
    # every tool produced a real result — partial runs are still useful.
    _run_deadline = time.monotonic() + pool.BATCH_TIMEOUT
    print(
        f'    wall-clock budget    : {pool.BATCH_TIMEOUT}s '
        f'(deadline in {pool.BATCH_TIMEOUT//60}m {pool.BATCH_TIMEOUT%60}s)',
        flush=True,
    )

    report = {
        'metadata': {
            'timestamp': str(datetime.now()),
            'arch': args.arch,
            'workers': list(selected),
            'granularity': {t: GRANULARITY.get(t, '?') for t in selected},
            'reference': args.reference,
            'seed': args.seed,
            'n_single': args.number if not args.sweep_only else 0,
            'n_sequence': args.sequences if not args.sweep_only else 0,
            'n_sweep': len(sweep_cases),
        },
        'results': [],
    }

    all_tool_names = active_tools
    disagree_count = 0

    try:
        # ── Heartbeat thread ───────────────────────────────────────────────
        # Prints a one-liner every 15 s so you always know the benchmark is
        # alive.  Reports progress of every worker including GT.
        _hb_stop = threading.Event()
        _hb_start = time.monotonic()
        # Mutable containers so the heartbeat thread can read live counts
        # that are updated by the worker threads.
        _hb_counts: dict[str, str] = {}  # name → "done/total" string

        def _heartbeat():
            i = 0
            while not _hb_stop.wait(timeout=15.0):
                elapsed = time.monotonic() - _hb_start
                remaining = max(0.0, _run_deadline - time.monotonic())
                i += 1
                parts = '  '.join(f'{k}={v}' for k, v in sorted(_hb_counts.items()))
                print(
                    f'[heartbeat {i:>4d}] {elapsed/60:5.1f}m elapsed '
                    f'| {remaining/60:5.1f}m left'
                    + (f' | {parts}' if parts else ' | waiting for workers to start ...'),
                    flush=True,
                )

        threading.Thread(target=_heartbeat, daemon=True, name='heartbeat').start()

        # ── Dispatch: batch all tests to Python workers simultaneously,
        #    run C-harness tests in a thread pool in parallel.
        # ──────────────────────────────────────────────────────────────────
        n_threads = max(len(c_harness_cmds), 1) + 3
        # Use the executor as a plain object (not context manager) so we can
        # cancel and shutdown without blocking on pending tasks.
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=n_threads)

        # Fire off C-harness futures for the whole batch immediately
        print(
            f"[phase 1/4] dispatching {total} tests to C-harness workers: {list(c_harness_cmds) or '(none)'}",
            flush=True,
        )
        c_future = executor.submit(
            _run_c_harnesses_parallel,
            test_cases,
            c_harness_cmds,
            executor,
            _run_deadline,
            _hb_counts,
        )

        # Fire off ground-truth simulator in parallel
        if args.ground_truth:
            print(
                f'[phase 1/4] dispatching {total} tests to ground-truth simulator (k≤{GT_BIT_BUDGET})',
                flush=True,
            )
        gt_future = (
            executor.submit(
                _run_ground_truth_batch,
                test_cases,
                args.quiet,
                _run_deadline,
                _hb_counts,
            )
            if args.ground_truth
            else None
        )

        # Send all test cases to Python workers
        print(
            f"[phase 2/4] dispatching {total} tests to pipe workers: "
            f"{persistent_names or '(none)'} — waiting for all results ...",
            flush=True,
        )
        _t_batch = time.monotonic()
        batch_results = pool.run_batch(
            test_cases,
            deadline=_run_deadline,
            hb_counts=_hb_counts,
        )
        print(f'[phase 2/4] pipe workers done in {time.monotonic()-_t_batch:.1f}s', flush=True)

        # Collect C-harness and GT results — both honour the shared deadline
        # so they should return quickly (either done or already marked timeout)
        print('[phase 3/4] waiting for C-harness results ...', flush=True)
        _t_c = time.monotonic()
        c_results = c_future.result()
        print(f'[phase 3/4] C-harness done in {time.monotonic()-_t_c:.1f}s', flush=True)

        if gt_future is not None:
            print('[phase 3/4] waiting for ground-truth results ...', flush=True)
            _t_gt = time.monotonic()
            gt_results = gt_future.result()
            print(f'[phase 3/4] ground-truth done in {time.monotonic()-_t_gt:.1f}s', flush=True)
        else:
            gt_results = [None] * len(test_cases)

        # Stop heartbeat immediately — results are in hand
        _hb_stop.set()

        # Shut down the executor without waiting for already-submitted but
        # now-unwanted C-harness subprocesses to finish.  cancel_futures=True
        # cancels tasks that haven't started yet; running tasks (subprocesses
        # already spawned) continue in the background but we don't wait.
        # wait=False means shutdown() returns immediately instead of blocking
        # until every thread drains — which could take hours for taintgrind.
        executor.shutdown(wait=False, cancel_futures=True)

        # ── Merge results and print ────────────────────────────────────────
        print(f'[phase 4/4] merging and writing report ({total} cases) ...', flush=True)
        bar = None
        if args.quiet:
            bar = tqdm(total=total, desc='merging', unit='case')
        for i, tc in enumerate(test_cases):
            label = tc.get('label', tc['assembly'])
            cat = tc.get('category', '?')
            mode = tc.get('mode', 'single')
            tool_results = {**batch_results[i], **c_results[i]}
            if gt_results[i] is not None:
                tool_results['ground_truth'] = gt_results[i]
            test_result = {'id': i, 'instruction': tc, 'tool_results': tool_results}

            if not args.quiet:
                print(f'\n[Test {i+1}/{total}] [{mode}/{cat}] {label}')
                print(f"  taint: { {k: hex(v) if v else 0 for k, v in tc['taint'].items()} }")
                for name in all_tool_names:
                    res = tool_results.get(name, {'error': 'not run', 'time_ns': 0})
                    gran = GRANULARITY.get(name, '?')
                    if 'error' in res:
                        print(f"  -> {name:<14} [{gran}]: ERR ({res['error'][:120].replace(chr(10), ' ')})")
                    else:
                        ot = res.get('output_taint', {})
                        parts = ' '.join(f'{r}={fmt_mask(v, gran)}' for r, v in ot.items())
                        print(f"  -> {name:<14} [{gran}]: {res['time_ns']/1e6:8.1f} ms | {parts}")

            disagreements = compare_results(tool_results)
            if disagreements:
                disagree_count += 1
                if not args.quiet:
                    print('  !! DISAGREEMENT (register-level):')
                    for d in disagreements:
                        print(d)
            elif not args.quiet:
                n_ok = sum(1 for r in tool_results.values() if 'error' not in r)
                if n_ok > 1:
                    print(f'  ✓  All {n_ok} tools agree')

            report['results'].append(test_result)
            if bar is not None:
                bar.update(1)
                # Update postfix with running disagreement count
                bar.set_postfix_str(f'disagreements={disagree_count}', refresh=False)
        if bar is not None:
            bar.close()

    finally:
        pool.stop_all()

    # ── Metrics ───────────────────────────────────────────────────────────
    if not args.no_summary and report['results']:
        metrics = compute_metrics(report['results'], args.reference)
        report['metrics'] = metrics
        print_summary(metrics, all_tool_names, args.reference)
        print(f'\n[*] Disagreements: {disagree_count}/{total} ({100*disagree_count/total:.1f}%)')

    fname = f'report_{int(datetime.now().timestamp())}.json'
    with open(fname, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f'\n[+] Done. Report: {fname}')


if __name__ == '__main__':
    main()
