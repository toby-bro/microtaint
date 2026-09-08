#!/usr/bin/env python3
"""Re-check every [UNSOUND] case the fuzzer logged with a CLEAN bit-flip oracle
(fresh Unicorn per run, so no state leakage), to separate real MicroTaint
under-taints from oracle artifacts.

Usage: python triage_unsound.py fuzz_XXXX.log
"""
from __future__ import annotations

import ast
import re
import sys
from collections import Counter

from unicorn import UC_ARCH_X86, UC_MODE_64, Uc
from unicorn.x86_const import (
    UC_X86_REG_RAX,
    UC_X86_REG_RBX,
    UC_X86_REG_RCX,
    UC_X86_REG_RDX,
    UC_X86_REG_RSP,
)

import worker_microtaint as wm

REGS = {'RAX': UC_X86_REG_RAX, 'RBX': UC_X86_REG_RBX, 'RCX': UC_X86_REG_RCX, 'RDX': UC_X86_REG_RDX}
MASK64 = (1 << 64) - 1
CODE = 0x400000
# stack region straddling the worker's default RSP so [rsp+-off] is mapped and
# the concrete addresses match MicroTaint's evaluation.
RSP = wm._DEFAULT_RSP  # 0x80000000
STK_LO = (RSP - 0x8000) & ~0xFFF
STK_SPAN = 0x10000

_LINE = re.compile(r"bytes=(\S+) state=(\{[^}]*\}) taint=(\{[^}]*\})")


def _clean_run(bs: bytes, state: dict[str, int]) -> dict[str, int]:
    uc = Uc(UC_ARCH_X86, UC_MODE_64)
    uc.mem_map(CODE, 0x1000)
    uc.mem_map(STK_LO, STK_SPAN)
    uc.mem_write(CODE, bs + b'\x90' * 16)
    for r, i in REGS.items():
        uc.reg_write(i, state[r])
    uc.reg_write(UC_X86_REG_RSP, RSP)
    uc.emu_start(CODE, CODE + len(bs))
    return {r: uc.reg_read(i) for r, i in REGS.items()}


def _clean_lb(bs: bytes, state: dict[str, int], taint: dict[str, int]) -> dict[str, int] | None:
    try:
        base = _clean_run(bs, state)
    except Exception:  # noqa: BLE001 -- baseline trap: no reference, treat as uncovered
        return None
    lb = dict.fromkeys(REGS, 0)
    for r in REGS:
        for b in range(64):
            if (taint[r] >> b) & 1:
                s2 = dict(state)
                s2[r] ^= 1 << b
                try:
                    o = _clean_run(bs, s2)
                except Exception:  # noqa: BLE001 -- flipped trap only shrinks the LB
                    continue
                for rr in REGS:
                    lb[rr] |= base[rr] ^ o[rr]
    return lb


def main() -> int:
    log = sys.argv[1]
    real = 0
    artifact = 0
    uncovered = 0
    real_cats: Counter = Counter()
    real_examples: list[str] = []
    with open(log) as f:
        for line in f:
            if '[UNSOUND]' not in line:
                continue
            m = _LINE.search(line)
            if not m:
                continue
            bs = bytes.fromhex(m.group(1))
            state = ast.literal_eval(m.group(2))
            taint = ast.literal_eval(m.group(3))
            cat = (re.search(r'cat=(\S+)', line) or [None, '?'])[1]
            tc = {'bytes': m.group(1), 'state': state, 'taint': taint}
            mt = wm.run_one(tc).get('output_taint', {})
            lb = _clean_lb(bs, state, taint)
            if lb is None:
                uncovered += 1
                continue
            missed = 0
            for r in REGS:
                missed |= (lb[r] & MASK64) & ~((mt.get(r, 0) or 0) & MASK64)
            if missed:
                real += 1
                real_cats[cat] += 1
                if len(real_examples) < 15:
                    asm = (re.search(r"asm='([^']*)'", line) or [None, ''])[1]
                    real_examples.append(f'  [{cat}] {asm}  missed={missed:#x}')
            else:
                artifact += 1
    total = real + artifact + uncovered
    print('=' * 70)
    print(f'triaged {total} flagged cases:')
    print(f'  REAL under-taints (clean oracle): {real}')
    print(f'  oracle artifacts (sound w/ clean oracle): {artifact}')
    print(f'  uncovered (baseline trap): {uncovered}')
    if real_cats:
        print('  real by category:', dict(real_cats.most_common()))
        print('  examples:')
        print('\n'.join(real_examples))
    print('=' * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
