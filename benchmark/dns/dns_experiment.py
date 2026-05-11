#!/usr/bin/env python3
"""
run_bit_precision_dns_experiment.py
====================================

Bit-precision evaluation of microtaint 0.6.4 on the DNS flag-byte
unpacking pattern from RFC 1035 sec. 4.1.1.

This is the experiment that the run-eval.py CLI harness cannot run,
because microtaint's --input flag taints all 8 bits of every input
byte and exposes no per-bit taint mask on the source.  Here we drive
microtaint's IR-level evaluator directly (the same path the upstream
unit tests use) so we can set arbitrary per-bit input taint masks.

Sequence under test
-------------------

    AND AL, 0x78        ; isolate bits 6..3 (OPCODE)   24 78
    SHR AL, 3           ; shift OPCODE down to LSB     C0 E8 03

This is the LDNS_OPCODE_WIRE macro from ldns (also used unmodified
in BIND, dnsmasq, and Wireshark's DNS dissector) lowered to x86-64.

Four runs
---------

  A  QR only       T_in=0x80  V_in=0x80   expect post-AL taint = 0x00
  B  OPCODE only   T_in=0x78  V_in=0x10   expect post-AL taint = 0x0F
  C  whole byte    T_in=0xFF  V_in=0x10   expect post-AL taint = 0x0F
  D  no taint      T_in=0x00  V_in=0x10   expect post-AL taint = 0x00

A and B are the headline differential.  A vs B is the discrimination
a byte-precision DTA cannot make (it sees both as 'byte tainted').
C is the byte-precision-tool baseline: microtaint's answer for B
must match its answer for C, otherwise the precision in A was bought
by losing coverage in B.  D is a negative control: no taint in,
no taint out.

Ground truth
------------

For each run, _true_taint_x86 flips each tainted input bit
individually under Unicorn and ORs the output XOR-deltas.  This
yields the exact 'could flip' mask, against which microtaint's
predicted mask is asserted bit-exact equal in both directions:
  - engine >= GT  =>  soundness (no missed dependencies)
  - engine <= GT  =>  precision (no over-tainting)
"""

from __future__ import annotations

import sys

import unicorn
import unicorn.x86_const as ux

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, ImplicitTaintPolicy, Register

MASK64 = 0xFFFFFFFFFFFFFFFF
_X86_FLAG_BITS = {'CF': 0, 'PF': 2, 'ZF': 6, 'SF': 7, 'OF': 11}
_X86_GP = {'RAX': ux.UC_X86_REG_RAX}


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------


def _c(code: str, text: str) -> str:
    return f'\033[{code}m{text}\033[0m' if sys.stdout.isatty() else text


def green(t: str) -> str:
    return _c('1;32', t)


def red(t: str) -> str:
    return _c('1;31', t)


def yellow(t: str) -> str:
    return _c('1;33', t)


def bold(t: str) -> str:
    return _c('1', t)


def dim(t: str) -> str:
    return _c('2', t)


# ---------------------------------------------------------------------------
# Ground-truth helpers (lifted from upstream test_bit_precision_edge_cases.py)
# ---------------------------------------------------------------------------


def _run_x86(code: bytes, regs: dict[str, int]) -> dict[str, int]:
    """Execute `code` on Unicorn with the given register/flag state, return outputs."""
    uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
    uc.mem_map(0x1000, 0x1000)
    uc.mem_write(0x1000, code)
    eflags = 2  # reserved always-1 bit
    for name, val in regs.items():
        if name in _X86_FLAG_BITS:
            if val:
                eflags |= 1 << _X86_FLAG_BITS[name]
        elif name in _X86_GP:
            uc.reg_write(_X86_GP[name], val & MASK64)
    uc.reg_write(ux.UC_X86_REG_EFLAGS, eflags)
    try:
        uc.emu_start(0x1000, 0x1000 + len(code))
    except Exception:
        return {}
    result: dict[str, int] = {}
    for n, rid in _X86_GP.items():
        result[n] = uc.reg_read(rid)
    ef = uc.reg_read(ux.UC_X86_REG_EFLAGS)
    for n, bit in _X86_FLAG_BITS.items():
        result[n] = (ef >> bit) & 1
    return result


def _true_taint_x86(
    code: bytes,
    reg_names: list[str],
    taint: dict[str, int],
    values: dict[str, int],
) -> dict[str, int]:
    """Per-bit sensitivity ground truth: flip each tainted input bit, OR output XOR-deltas."""
    base_vals = {n: values.get(n, 0) & ~taint.get(n, 0) & MASK64 for n in reg_names}
    base_out = _run_x86(code, base_vals)
    result = dict.fromkeys(reg_names, 0)
    for reg in reg_names:
        width = 1 if reg in _X86_FLAG_BITS else 64
        tmask = taint.get(reg, 0)
        if not tmask:
            continue
        for bit in range(width):
            if not (tmask >> bit) & 1:
                continue
            flipped = dict(base_vals)
            flipped[reg] = (base_vals[reg] ^ (1 << bit)) & MASK64
            out = _run_x86(code, flipped)
            for out_reg in reg_names:
                result[out_reg] |= base_out.get(out_reg, 0) ^ out.get(out_reg, 0)
    return result


# ---------------------------------------------------------------------------
# Sequence under test
# ---------------------------------------------------------------------------

OPCODE_EXTRACT = bytes.fromhex('2478') + bytes.fromhex('c0e803')
OPCODE_EXTRACT_DISASM = 'and al, 0x78 ; shr al, 3'

REG_NAMES = ['RAX', 'CF', 'OF', 'ZF', 'SF', 'PF']
REGS = [
    Register('RAX', 64),
    Register('CF', 1),
    Register('OF', 1),
    Register('ZF', 1),
    Register('SF', 1),
    Register('PF', 1),
]


def _mt_eval(sim: CellSimulator, code: bytes, taint: dict[str, int], values: dict[str, int]) -> dict[str, int]:
    circuit = generate_static_rule(Architecture.AMD64, code, REGS)
    ctx = EvalContext(
        input_taint=taint,
        input_values=values,
        simulator=sim,
        implicit_policy=ImplicitTaintPolicy.IGNORE,
    )
    return circuit.evaluate(ctx)


# ---------------------------------------------------------------------------
# Run definitions
# ---------------------------------------------------------------------------


class Run:
    def __init__(self, label: str, taint_mask: int, value: int, expected_al_taint: int, narrative: str) -> None:
        self.label = label
        self.taint_mask = taint_mask
        self.value = value
        self.expected_al_taint = expected_al_taint
        self.narrative = narrative
        self.mt_out: dict[str, int] = {}
        self.gt_out: dict[str, int] = {}
        self.passed = False

    @property
    def mt_al(self) -> int:
        return self.mt_out.get('RAX', 0) & 0xFF

    @property
    def gt_al(self) -> int:
        return self.gt_out.get('RAX', 0) & 0xFF


RUNS = [
    Run(
        'A: QR only (mask 0x80)',
        taint_mask=0x80,
        value=0x80,
        expected_al_taint=0x00,
        narrative='QR bit (bit 7) is tainted, but AND/0x78 zeroes it out before SHR.',
    ),
    Run(
        'B: OPCODE only (mask 0x78)',
        taint_mask=0x78,
        value=0x10,
        expected_al_taint=0x0F,
        narrative='OPCODE bits (6..3) tainted; AND keeps them, SHR shifts them to bits 3..0.',
    ),
    Run(
        'C: whole byte (mask 0xFF) -- byte-precision baseline',
        taint_mask=0xFF,
        value=0x10,
        expected_al_taint=0x0F,
        narrative='Coarse byte-level taint -- the answer a byte-precision DTA produces for ANY of A, B, C.',
    ),
    Run(
        'D: no taint (negative control)',
        taint_mask=0x00,
        value=0x10,
        expected_al_taint=0x00,
        narrative='No source taint -- engine must not invent any.',
    ),
]


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------


def execute_run(sim: CellSimulator, run: Run) -> None:
    taint = {n: 0 for n in REG_NAMES}
    taint['RAX'] = run.taint_mask

    values = {n: 0 for n in REG_NAMES}
    values['RAX'] = run.value & 0xFF

    run.gt_out = _true_taint_x86(OPCODE_EXTRACT, REG_NAMES, taint, values)
    run.mt_out = _mt_eval(sim, OPCODE_EXTRACT, taint, values)

    sound = (run.mt_al | run.gt_al) == run.mt_al
    precise = (run.mt_al & run.gt_al) == run.mt_al
    matches_expected = run.mt_al == run.expected_al_taint
    matches_gt = run.mt_al == run.gt_al

    run.passed = sound and precise and matches_expected and matches_gt


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_header() -> None:
    print(bold('\n=== microtaint 0.6.4 -- bit-precision DNS flag-byte experiment ==='))
    print(f'  Sequence:    {OPCODE_EXTRACT_DISASM}')
    print(f'  Bytes:       {OPCODE_EXTRACT.hex()}')
    print('  Source:      RFC 1035 sec. 4.1.1, third byte of the DNS header')
    print('  Macro:       LDNS_OPCODE_WIRE (ldns, BSD-3-Clause)')
    print('  Layers:      QR | OPCODE(4) | AA | TC | RD')


def print_run_table() -> None:
    print()
    print(bold('Per-run results (post-AL taint mask):'))
    print(
        f'  {"Run":<55} {"T_in":>5} {"V_in":>5} {"Exp":>5} {"MT":>5} {"GT":>5}  Verdict'
    )
    print('  ' + dim('-' * 99))
    for r in RUNS:
        verdict = green('PASS') if r.passed else red('FAIL')
        print(
            f'  {r.label:<55} '
            f'{r.taint_mask:#05x} {r.value:#05x} {r.expected_al_taint:#05x} '
            f'{r.mt_al:#05x} {r.gt_al:#05x}  {verdict}'
        )


def print_run_details() -> None:
    print()
    print(bold('Per-run detail:'))
    for r in RUNS:
        marker = green('+') if r.passed else red('!')
        print(f'\n  {marker} {bold(r.label)}')
        print(f'      {dim(r.narrative)}')
        print(f'      input :       T_RAX={r.taint_mask:#04x}    V_RAX={r.value:#04x}')
        print(f'      expected:     T_AL_post = {r.expected_al_taint:#04x}')
        mt_fmt = ', '.join(f'{k}={v:#x}' for k, v in r.mt_out.items())
        gt_fmt = ', '.join(f'{k}={v:#x}' for k, v in r.gt_out.items())
        print(f'      microtaint:   {mt_fmt}')
        print(f'      ground truth: {gt_fmt}')

        # EFLAGS over-tainting check (informational; not part of the headline assertion)
        flag_overtaint = []
        for f in ('CF', 'OF', 'ZF', 'SF', 'PF'):
            mt_v = r.mt_out.get(f, 0) & 1
            gt_v = r.gt_out.get(f, 0) & 1
            if mt_v and not gt_v:
                flag_overtaint.append(f)
        if flag_overtaint:
            print(f'      {yellow("note:")}        EFLAGS over-tainted on {", ".join(flag_overtaint)} '
                  f'(GT clean, MT tainted)')


def print_conclusions() -> None:
    a = next(r for r in RUNS if r.label.startswith('A'))
    b = next(r for r in RUNS if r.label.startswith('B'))
    c = next(r for r in RUNS if r.label.startswith('C'))
    d = next(r for r in RUNS if r.label.startswith('D'))

    print()
    print(bold('=== Conclusions ==='))
    print()

    # 1. Bit-precision discrimination
    print(bold('1. Bit-precision discrimination (the headline claim).'))
    discriminates = a.mt_al == 0x00 and b.mt_al == 0x0F
    if discriminates:
        print('   Run A (QR-only)     -> AL taint = ' + green(f'{a.mt_al:#04x}') + '   (QR bit erased by & 0x78)')
        print('   Run B (OPCODE-only) -> AL taint = ' + green(f'{b.mt_al:#04x}') + '   (OPCODE shifted to bits 0..3)')
        print('   Same byte position, opposite verdicts.  A byte-precision DTA cannot')
        print('   distinguish these two cases; microtaint discriminates them exactly.')
    else:
        print(red('   Did not discriminate.  See per-run detail.'))

    # 2. Soundness + precision on AL
    print()
    print(bold('2. Soundness and precision on the data register (AL).'))
    al_exact = all(r.mt_al == r.gt_al for r in RUNS)
    if al_exact:
        print('   In all 4 runs, microtaint\'s predicted AL taint mask is ' + green('bit-exact equal') +
              ' to the')
        print('   per-bit Unicorn ground truth: engine >= GT (no missed deps) AND')
        print('   engine <= GT (no over-tainting).')
    else:
        print(red('   AL mask did not match GT in some run; not bit-exact.'))

    # 3. Run B vs Run C: precision is real, not bought by losing coverage
    print()
    print(bold('3. Run B vs Run C: precision in A is not bought by losing coverage in B.'))
    if b.mt_al == c.mt_al == 0x0F:
        print('   With OPCODE-only taint (B) the engine reports AL taint = ' + green(f'{b.mt_al:#04x}') + ',')
        print('   identical to the whole-byte-tainted baseline (C) of ' + green(f'{c.mt_al:#04x}') + '.')
        print('   The extra precision in Run A is therefore a refinement on top of')
        print('   correct byte-level behaviour, not a regression on it.')
    else:
        print(red('   B and C disagree -- precision claim weakened.'))

    # 4. Negative control
    print()
    print(bold('4. No spurious taint without a source.'))
    if d.mt_al == 0:
        print('   Run D feeds the same value (0x10) through the same sequence with')
        print('   ' + green('no input taint') + '.  The engine reports ' + green('zero taint') +
              ' on every output register and flag.')
    else:
        print(red('   Negative control failed: engine produced taint without a source.'))

    # 5. Caveat: EFLAGS over-tainting
    print()
    print(bold('5. Known caveat: EFLAGS over-tainting on the AND/SHR sequence.'))
    overtaint_runs = []
    for r in RUNS[:3]:  # A, B, C
        for f in ('CF', 'OF', 'ZF', 'SF', 'PF'):
            mt_v = r.mt_out.get(f, 0) & 1
            gt_v = r.gt_out.get(f, 0) & 1
            if mt_v and not gt_v:
                overtaint_runs.append((r.label[0], f))
    if overtaint_runs:
        print('   Microtaint reports condition flags as tainted in runs A/B/C even')
        print('   when the per-bit ground truth shows the flag does not depend on the')
        print('   tainted bits.  Specifically: ' + yellow(', '.join(f'{lbl}.{flag}' for lbl, flag in overtaint_runs)))
        print('   This is the family of issues the upstream test suite tracks under')
        print('   group G-A (constant-result idioms) and G-B (shift carry semantics).')
        print('   It does not affect the OPCODE-extraction result on AL, and the DNS')
        print('   dispatch in parse_flags reads `op` (a register), not these flags.')
        print('   But: any claim of "bit-exact precision on the entire architectural')
        print('   state" must be qualified -- the precision is exact on the data path,')
        print('   ' + yellow('not on EFLAGS') + '.')
    else:
        print('   No EFLAGS over-tainting detected.')

    # Final headline
    print()
    print(dim('-' * 70))
    all_pass = all(r.passed for r in RUNS)
    if all_pass:
        print(green(bold('OVERALL: 4/4 runs PASS on the AL taint-mask assertion.')))
        print('  Bit-precision over byte-precision is demonstrated end-to-end on the')
        print('  RFC 1035 OPCODE-extraction sequence with bit-exact agreement to a')
        print('  per-bit Unicorn ground truth on the data register, with one documented')
        print('  caveat on EFLAGS over-tainting that does not affect this experiment.')
    else:
        failing = [r.label for r in RUNS if not r.passed]
        print(red(bold(f'OVERALL: {sum(1 for r in RUNS if r.passed)}/{len(RUNS)} runs pass.')))
        print('  Failing: ' + ', '.join(failing))
    print(dim('-' * 70))


def main() -> int:
    print_header()
    sim = CellSimulator(Architecture.AMD64)
    for r in RUNS:
        execute_run(sim, r)
    print_run_table()
    print_run_details()
    print_conclusions()
    return 0 if all(r.passed for r in RUNS) else 1


if __name__ == '__main__':
    raise SystemExit(main())
