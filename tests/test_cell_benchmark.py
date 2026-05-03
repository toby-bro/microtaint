"""
test_cell_benchmark.py
======================
Pytest-benchmark suite comparing the Unicorn-backed CellSimulator against
the native P-code CellSimulator (use_unicorn=False).

Run:
    # Correctness only (no timing):
    pytest test_cell_benchmark.py -v

    # With benchmarks (requires pytest-benchmark):
    pytest test_cell_benchmark.py -v --benchmark-warmup=on --benchmark-min-rounds=50

    # Compare both backends and save a baseline:
    pytest test_cell_benchmark.py --benchmark-autosave

Structure
---------
1.  AMD64 instruction corpus  (~80 instructions across all major categories)
2.  Correctness tests          — assert that unicorn_output == pcode_output for
                                 every instruction x every taint scenario
3.  Single-instruction benchmarks — throughput for one instruction, 100 rounds
4.  Sequential-trace benchmarks   — throughput for the full corpus in order,
                                    simulating a real taint-analysis trace
"""

# mypy: disable-error-code="type-arg,no-any-return"
# ruff: noqa: ARG001

from __future__ import annotations

import itertools
import os
from typing import Any

import pytest

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, ImplicitTaintPolicy, Register

# ===========================================================================
# Shared register format (identical to reference test)
# ===========================================================================

AMD64_REGS: list[Register] = [
    Register(name='RAX', bits=64),
    Register(name='RBX', bits=64),
    Register(name='RCX', bits=64),
    Register(name='RDX', bits=64),
    Register(name='RSP', bits=64),
    Register(name='RBP', bits=64),
    Register(name='RSI', bits=64),
    Register(name='RDI', bits=64),
    Register(name='R8', bits=64),
    Register(name='R9', bits=64),
    Register(name='R10', bits=64),
    Register(name='R11', bits=64),
    Register(name='R12', bits=64),
    Register(name='R13', bits=64),
    Register(name='R14', bits=64),
    Register(name='R15', bits=64),
    Register(name='RIP', bits=64),
    Register(name='EFLAGS', bits=32),
    Register(name='ZF', bits=1),
    Register(name='CF', bits=1),
    Register(name='SF', bits=1),
    Register(name='OF', bits=1),
    Register(name='PF', bits=1),
    Register(name='AX', bits=16),
    Register(name='AL', bits=8),
    Register(name='AH', bits=8),
    Register(name='EAX', bits=32),
    Register(name='BX', bits=16),
    Register(name='BL', bits=8),
    Register(name='BH', bits=8),
    Register(name='EBX', bits=32),
]

FULL_TAINT_64 = 0xFFFFFFFFFFFFFFFF
FULL_TAINT_32 = 0xFFFFFFFF
FULL_TAINT_16 = 0xFFFF
FULL_TAINT_8 = 0xFF

# ===========================================================================
# AMD64 instruction corpus
# Each entry: (mnemonic, hex_bytes, input_values, input_taint,
#              expected_key, expected_taint, description)
#
# expected_key   : register name to check in output dict (or None to skip check)
# expected_taint : exact expected value (or None to only compare unicorn↔pcode)
# ===========================================================================

#: (mnemonic, hex, input_values, input_taint, check_reg, expected, description)
CORPUS: list[tuple[str, str, dict, dict, str | None, int | None, str]] = [
    # -----------------------------------------------------------------------
    # MOV variants
    # -----------------------------------------------------------------------
    (
        'MOV r64,r64',
        '4889C3',
        {'RAX': 0x1234567890ABCDEF, 'RBX': 0},
        {'RAX': FULL_TAINT_64},
        'RBX',
        FULL_TAINT_64,
        'MOV RBX,RAX — full taint propagation',
    ),
    (
        'MOV r64,imm64',
        '48B80100000000000000',
        {'RAX': 0xDEAD},
        {'RAX': FULL_TAINT_64},
        'RAX',
        0,
        'MOV RAX,1 — immediate clears taint',
    ),
    (
        'MOV r32,r32',
        '89C3',
        {'EAX': 0xCAFEBABE, 'EBX': 0},
        {'EAX': FULL_TAINT_32},
        'EBX',
        FULL_TAINT_32,
        'MOV EBX,EAX — 32-bit propagation',
    ),
    (
        'MOV r16,r16',
        '6689C3',
        {'AX': 0x1234, 'BX': 0},
        {'AX': FULL_TAINT_16},
        'BX',
        FULL_TAINT_16,
        'MOV BX,AX — 16-bit propagation',
    ),
    (
        'MOV r8,r8',
        '88C3',
        {'AL': 0xFF, 'BL': 0},
        {'AL': FULL_TAINT_8},
        'BL',
        FULL_TAINT_8,
        'MOV BL,AL — taint moves to destination BL',
    ),
    (
        'MOV r64,0',
        '4831C0',  # XOR RAX,RAX (canonical zero)
        {'RAX': 0xDEAD},
        {'RAX': FULL_TAINT_64},
        'RAX',
        0,
        'XOR RAX,RAX — zeroing idiom clears taint',
    ),
    (
        'MOVSX r64,r8',
        '480FBEC0',
        {'AL': 0x80},
        {'AL': FULL_TAINT_8},
        'RAX',
        FULL_TAINT_64,
        'MOVSX RAX,AL — sign-extend propagates taint',
    ),
    (
        'MOVZX r32,r8',
        '0FB6C0',
        {'AL': 0xFF},
        {'AL': FULL_TAINT_8},
        'EAX',
        FULL_TAINT_8,
        'MOVZX EAX,AL — zero-extend keeps low-byte taint',
    ),
    (
        'MOVZX r32,r16',
        '0FB7C3',
        {'BX': 0x1234},
        {'BX': FULL_TAINT_16},
        'EAX',
        FULL_TAINT_16,
        'MOVZX EAX,BX — zero-extend 16-bit',
    ),
    # -----------------------------------------------------------------------
    # ADD variants
    # -----------------------------------------------------------------------
    (
        'ADD r64,r64',
        '4801D8',
        {'RAX': 0, 'RBX': 0},
        {'RAX': 0xAAAAAAAAAAAAAAAA, 'RBX': 0x5555555555555555},
        'RAX',
        FULL_TAINT_64,
        'ADD RAX,RBX — both operands tainted',
    ),
    (
        'ADD r64,r64 flags',
        '4801D8',
        {'RAX': 0xFFFFFFFFFFFFFFFF, 'RBX': 1},
        {'RAX': 0x10, 'RBX': 0},
        'CF',
        1,
        'ADD RAX,RBX — tainted operand causes carry flag taint',
    ),
    (
        'ADD r32,r32',
        '01D8',
        {'EAX': 0x7FFFFFFF, 'EBX': 1},
        {'EAX': 0x10, 'EBX': 0},
        'OF',
        1,
        'ADD EAX,EBX — overflow flag tainted',
    ),
    (
        'ADD r64,imm32',
        '4883C001',
        {'RAX': 5},
        {'RAX': FULL_TAINT_64},
        'RAX',
        FULL_TAINT_64,
        'ADD RAX,1 — immediate add preserves taint',
    ),
    (
        'ADD r16,r16',
        '6601C3',
        {'AX': 0x1234, 'BX': 0x5678},
        {'AX': FULL_TAINT_16, 'BX': FULL_TAINT_16},
        'BX',
        FULL_TAINT_16,
        'ADD BX,AX — 16-bit add tainted',
    ),
    (
        'ADD r8,r8',
        '00D8',
        {'AL': 0x7F, 'BL': 1},
        {'AL': FULL_TAINT_8, 'BL': 0},
        'AL',
        FULL_TAINT_8,
        'ADD AL,BL — 8-bit add tainted',
    ),
    # -----------------------------------------------------------------------
    # SUB / NEG / INC / DEC
    # -----------------------------------------------------------------------
    (
        'SUB r64,r64',
        '4829D8',
        {'RAX': 10, 'RBX': 3},
        {'RAX': FULL_TAINT_64, 'RBX': FULL_TAINT_64},
        'RAX',
        FULL_TAINT_64,
        'SUB RAX,RBX — both tainted',
    ),
    (
        'SUB r64,imm8',
        '4883E801',
        {'RAX': 5},
        {'RAX': FULL_TAINT_64},
        'RAX',
        FULL_TAINT_64,
        'SUB RAX,1 — immediate sub preserves taint',
    ),
    (
        'NEG r64',
        '48F7D8',
        {'RAX': 5},
        {'RAX': FULL_TAINT_64},
        'RAX',
        FULL_TAINT_64,
        "NEG RAX — two's complement preserves taint",
    ),
    (
        'INC r32',
        'FFC0',
        {'EAX': 0},
        {'EAX': FULL_TAINT_32},
        'EAX',
        FULL_TAINT_32,
        'INC EAX — taint preserved',
    ),
    (
        'INC r64',
        '48FFC0',
        {'RAX': 0},
        {'RAX': FULL_TAINT_64},
        'RAX',
        FULL_TAINT_64,
        'INC RAX — 64-bit taint preserved',
    ),
    (
        'DEC r32',
        'FFC8',
        {'EAX': 5},
        {'EAX': FULL_TAINT_32},
        'EAX',
        FULL_TAINT_32,
        'DEC EAX — taint preserved',
    ),
    (
        'DEC r64',
        '48FFC8',
        {'RAX': 5},
        {'RAX': FULL_TAINT_64},
        'RAX',
        FULL_TAINT_64,
        'DEC RAX — 64-bit taint preserved',
    ),
    # -----------------------------------------------------------------------
    # AND / OR / XOR / NOT
    # -----------------------------------------------------------------------
    (
        'AND r64,r64',
        '4821D8',
        {'RAX': FULL_TAINT_64, 'RBX': FULL_TAINT_64},
        {'RAX': FULL_TAINT_64, 'RBX': FULL_TAINT_64},
        'RAX',
        FULL_TAINT_64,
        'AND RAX,RBX — both tainted',
    ),
    (
        'AND r32,imm32',
        '25FF000000',
        {'EAX': FULL_TAINT_32},
        {'EAX': FULL_TAINT_32},
        'EAX',
        FULL_TAINT_8,
        'AND EAX,0xFF — mask narrows taint',
    ),
    (
        'AND r16,r16',
        '6621C3',
        {'AX': 0xFFFF, 'BX': 0xFFFF},
        {'AX': FULL_TAINT_16, 'BX': FULL_TAINT_16},
        'BX',
        FULL_TAINT_16,
        'AND BX,AX — 16-bit both tainted',
    ),
    (
        'OR r64,r64',
        '4809D8',
        {'RAX': 0, 'RBX': 0},
        {'RAX': FULL_TAINT_64, 'RBX': 0},
        'RAX',
        FULL_TAINT_64,
        'OR RAX,RBX — first operand tainted',
    ),
    (
        'OR r8,r8',
        '08D8',
        {'AL': 0x00, 'BL': 0xFF},
        {'AL': 0, 'BL': FULL_TAINT_8},
        'AL',
        FULL_TAINT_8,
        'OR AL,BL — second operand tainted',
    ),
    (
        'XOR r64,r64',
        '4831D8',
        {'RAX': 0, 'RBX': 0},
        {'RAX': FULL_TAINT_64, 'RBX': FULL_TAINT_64},
        'RAX',
        FULL_TAINT_64,
        'XOR RAX,RBX — both tainted',
    ),
    (
        'XOR r64,r64 zero',
        '4831C0',
        {'RAX': 0xDEAD},
        {'RAX': FULL_TAINT_64},
        'RAX',
        0,
        'XOR RAX,RAX — zeroing idiom clears taint',
    ),
    (
        'XOR r32,imm32',
        '35FF000000',
        {'EAX': 0},
        {'EAX': FULL_TAINT_32},
        'EAX',
        FULL_TAINT_32,
        'XOR EAX,0xFF — partial immediate keeps taint',
    ),
    (
        'NOT r64',
        '48F7D0',
        {'RAX': FULL_TAINT_64},
        {'RAX': FULL_TAINT_64},
        'RAX',
        FULL_TAINT_64,
        'NOT RAX — bitwise NOT preserves taint',
    ),
    # -----------------------------------------------------------------------
    # SHL / SHR / SAR / ROL / ROR
    # -----------------------------------------------------------------------
    (
        'SHL r64,cl',
        '48D3E0',
        {'RAX': 0x00FF00FF00FF00FF, 'RCX': 8},
        {'RAX': 0x00FF00FF00FF00FF, 'RCX': 0},
        'RAX',
        None,  # compare backends only
        'SHL RAX,CL — shift by concrete amount',
    ),
    (
        'SHL r64,imm8',
        '48C1E008',
        {'RAX': 0x00FF00FF00FF00FF},
        {'RAX': 0x00FF00FF00FF00FF},
        'RAX',
        0xFF00FF00FF00FF00,
        'SHL RAX,8 — bit-precise left shift taint',
    ),
    (
        'SHR r64,imm8',
        '48C1E808',
        {'RAX': 0xFF00FF00FF00FF00},
        {'RAX': 0xFF00FF00FF00FF00},
        'RAX',
        0x00FF00FF00FF00FF,
        'SHR RAX,8 — bit-precise right shift taint',
    ),
    (
        'SAR r64,imm8',
        '48C1F808',
        {'RAX': 0xFF00000000000000},
        {'RAX': 0xFF00000000000000},
        'RAX',
        None,  # compare backends only
        'SAR RAX,8 — arithmetic right shift taint',
    ),
    (
        'ROL r64,imm8',
        '48C1C008',
        {'RAX': 0x00FF00FF00FF00FF},
        {'RAX': 0x00FF00FF00FF00FF},
        'RAX',
        0xFF00FF00FF00FF00,
        'ROL RAX,8 — rotate left taint',
    ),
    (
        'ROR r64,imm8',
        '48C1C808',
        {'RAX': 0xFF00FF00FF00FF00},
        {'RAX': 0xFF00FF00FF00FF00},
        'RAX',
        None,  # compare backends only
        'ROR RAX,8 — rotate right taint',
    ),
    (
        'SHL r32,imm8',
        'C1E004',
        {'EAX': 0x00FF00FF},
        {'EAX': 0x00FF00FF},
        'EAX',
        0x0FF00FF0,
        'SHL EAX,4 — 32-bit bit-precise left shift',
    ),
    (
        'SHR r32,imm8',
        'C1E804',
        {'EAX': 0x0FF00FF0},
        {'EAX': 0x0FF00FF0},
        'EAX',
        None,
        'SHR EAX,4 — 32-bit bit-precise right shift',
    ),
    # -----------------------------------------------------------------------
    # MUL / IMUL / DIV
    # -----------------------------------------------------------------------
    (
        'MUL r64',
        '48F7E3',
        {'RAX': 2, 'RBX': 3},
        {'RAX': FULL_TAINT_64, 'RBX': 0},
        'RAX',
        FULL_TAINT_64,
        'MUL RBX — tainted RAX propagates to RAX:RDX',
    ),
    (
        'MUL r64 rdx',
        '48F7E3',
        {'RAX': 2, 'RBX': 3},
        {'RAX': FULL_TAINT_64, 'RBX': 0},
        'RDX',
        FULL_TAINT_64,
        'MUL RBX — tainted RAX propagates to high half RDX',
    ),
    (
        'IMUL r64,r64',
        '480FAFC3',
        {'RAX': 2, 'RBX': 3},
        {'RAX': FULL_TAINT_64, 'RBX': 0},
        'RAX',
        FULL_TAINT_64,
        'IMUL RAX,RBX — 2-op taint to destination',
    ),
    (
        'IMUL r64,r64,imm8',
        '486BC302',
        {'RBX': 5},
        {'RBX': FULL_TAINT_64},
        'RAX',
        FULL_TAINT_64,
        'IMUL RAX,RBX,2 — 3-op imul propagates taint',
    ),
    (
        'IMUL r32,r32',
        '0FAFC3',
        {'EAX': 2, 'EBX': 3},
        {'EAX': FULL_TAINT_32, 'EBX': 0},
        'EAX',
        FULL_TAINT_32,
        'IMUL EAX,EBX — 32-bit 2-op taint',
    ),
    # -----------------------------------------------------------------------
    # CMP / TEST / SETcc
    # -----------------------------------------------------------------------
    (
        'CMP r64,r64 ZF',
        '4839D8',
        {'RAX': 5, 'RBX': 5},
        {'RAX': FULL_TAINT_64, 'RBX': 0},
        'ZF',
        1,
        'CMP RAX,RBX equal — tainted operand taints ZF',
    ),
    (
        'CMP r32,r32 CF',
        '39D8',
        {'EAX': 0, 'EBX': 1},
        {'EAX': FULL_TAINT_32, 'EBX': 0},
        'CF',
        1,
        'CMP EAX,EBX borrow — tainted operand taints CF',
    ),
    (
        'CMP r64,imm8',
        '4883F800',
        {'RAX': 0},
        {'RAX': FULL_TAINT_64},
        'ZF',
        1,
        'CMP RAX,0 — tainted register taints ZF',
    ),
    (
        'TEST r64,r64',
        '4885C0',
        {'RAX': 0},
        {'RAX': FULL_TAINT_64},
        'ZF',
        1,
        'TEST RAX,RAX — tainted register taints ZF',
    ),
    (
        'TEST r32,r32',
        '85D2',
        {'EDX': 0},
        {'EDX': FULL_TAINT_32},
        'ZF',
        1,
        'TEST EDX,EDX — 32-bit tainted operand taints ZF',
    ),
    (
        'SETZ r8',
        '0F94C0',
        {'EFLAGS': 0x40},
        {'ZF': 1},
        'AL',
        1,
        'SETZ AL — tainted ZF taints destination byte',
    ),
    (
        'SETNZ r8',
        '0F95C0',
        {'EFLAGS': 0},
        {'ZF': 1},
        'AL',
        1,
        'SETNZ AL — tainted ZF taints destination byte',
    ),
    (
        'SETL r8',
        '0F9CC0',
        {'EFLAGS': 0},
        {'SF': 1},
        'AL',
        None,
        'SETL AL — compare backends only (COND_TRANSPORTABLE: both give 0 with SF-only taint)',
    ),
    (
        'SETS r8',
        '0F98C0',
        {'EFLAGS': 0},
        {'SF': 1},
        'AL',
        1,
        'SETS AL — tainted SF taints destination',
    ),
    (
        'CMP r8,imm8',
        '3C58',
        {'RAX': 0x58},
        {'RAX': FULL_TAINT_8},
        'ZF',
        1,
        'CMP AL,0x58 — tainted AL can reach constant, ZF tainted',
    ),
    # -----------------------------------------------------------------------
    # LEA / XCHG
    # -----------------------------------------------------------------------
    (
        'LEA r64,[r64+r64]',
        '488D0418',
        {'RAX': 0x1000, 'RBX': 0x20},
        {'RAX': FULL_TAINT_64, 'RBX': 0},
        'RAX',
        FULL_TAINT_64,
        'LEA RAX,[RAX+RBX] — address computation with tainted base',
    ),
    (
        'LEA r64,[r64*4]',
        '488D048500000000',
        {'RAX': 4},
        {'RAX': FULL_TAINT_64},
        'RAX',
        FULL_TAINT_64,
        'LEA RAX,[RAX*4+0] — scaled address propagates taint',
    ),
    (
        'XCHG r64,r64',
        '4887C3',
        {'RAX': 0xAAAA, 'RBX': 0xBBBB},
        {'RAX': 0x10, 'RBX': 0x20},
        'RAX',
        0x20,
        'XCHG RAX,RBX — taint swap (RAX gets RBX taint)',
    ),
    (
        'XCHG r64,r64 B',
        '4887C3',
        {'RAX': 0xAAAA, 'RBX': 0xBBBB},
        {'RAX': 0x10, 'RBX': 0x20},
        'RBX',
        0x10,
        'XCHG RAX,RBX — taint swap (RBX gets RAX taint)',
    ),
    # -----------------------------------------------------------------------
    # BSWAP / POPCNT / BSF / BSR / LZCNT / TZCNT
    # -----------------------------------------------------------------------
    (
        'BSWAP r64',
        '480FC8',
        {'RAX': 0x0123456789ABCDEF},
        {'RAX': FULL_TAINT_64},
        'RAX',
        FULL_TAINT_64,
        'BSWAP RAX — byte reversal preserves full taint',
    ),
    (
        'BSWAP r64 partial',
        '480FC8',
        {'RAX': 0x0123456789ABCDEF},
        {'RAX': 0xFF},
        'RAX',
        None,  # compare backends only (byte 0 taint → byte 7 after swap)
        'BSWAP RAX — partial taint byte reposition',
    ),
    (
        'POPCNT r64,r64',
        'F3480FB8C0',
        {'RAX': 0xFF},
        {'RAX': FULL_TAINT_64},
        'RAX',
        0x7F,  # count fits in 7 bits (0..64) — bounded avalanche
        'POPCNT RAX,RAX — count bounded to 7 bits',
    ),
    (
        'BSF r64,r64',
        '480FBCC3',
        {'RBX': 0x40},
        {'RBX': FULL_TAINT_64},
        'RAX',
        0x7F,  # bit index 0..63 fits in 7 bits
        'BSF RAX,RBX — bit-index result bounded to 7 bits',
    ),
    (
        'BSR r64,r64',
        '480FBDC3',
        {'RBX': 0x40},
        {'RBX': FULL_TAINT_64},
        'RAX',
        0x7F,
        'BSR RAX,RBX — bit-index result bounded to 7 bits',
    ),
    # -----------------------------------------------------------------------
    # PUSH / POP (memory taint)
    # -----------------------------------------------------------------------
    (
        'PUSH r64',
        '50',
        {'RAX': 0xDEAD, 'RSP': 0x80000010},
        {'RAX': FULL_TAINT_64, 'RSP': 0},
        None,
        None,  # output: memory — compare backends via full dict
        'PUSH RAX — tainted value written to stack',
    ),
    (
        'PUSH imm32',
        '6800000000',
        {'RSP': 0x80000010},
        {'RSP': 0},
        None,
        None,
        'PUSH 0 — immediate push clears stack taint',
    ),
    # -----------------------------------------------------------------------
    # ADC / SBB (with carry)
    # -----------------------------------------------------------------------
    (
        'ADC r64,r64',
        '4811D8',
        {'RAX': 0, 'RBX': 0, 'EFLAGS': 1},  # CF=1
        {'RAX': FULL_TAINT_64, 'RBX': 0, 'CF': 0},
        'RAX',
        FULL_TAINT_64,
        'ADC RAX,RBX — tainted RAX + untainted CF',
    ),
    (
        'ADC r64,r64 cf',
        '4811D8',
        {'RAX': 0, 'RBX': 0, 'EFLAGS': 1},
        {'RAX': 0, 'RBX': 0, 'CF': 1},
        'RAX',
        1,
        'ADC RAX,RBX — tainted CF (1-bit) propagates as bit 0 to result',
    ),
    (
        'SBB r64,r64',
        '4819D8',
        {'RAX': 5, 'RBX': 2, 'EFLAGS': 0},
        {'RAX': FULL_TAINT_64, 'RBX': 0, 'CF': 0},
        'RAX',
        FULL_TAINT_64,
        'SBB RAX,RBX — tainted minuend',
    ),
    # -----------------------------------------------------------------------
    # CMOV variants
    # -----------------------------------------------------------------------
    (
        'CMOVZ r64,r64',
        '480F44C3',
        {'RAX': 0xAAAA, 'RBX': 0xBBBB, 'EFLAGS': 0x40},  # ZF=1
        {'RBX': FULL_TAINT_64, 'ZF': 1},
        'RAX',
        FULL_TAINT_64,
        'CMOVZ RAX,RBX — tainted ZF + tainted source: dest fully tainted',
    ),
    (
        'CMOVNZ r64,r64',
        '480F45C3',
        {'RAX': 0xAAAA, 'RBX': 0xBBBB, 'EFLAGS': 0x00},  # ZF=0
        {'RBX': FULL_TAINT_64, 'ZF': 1},
        'RAX',
        None,
        'CMOVNZ RAX,RBX — compare backends only (COND_TRANSPORTABLE behavior)',
    ),
    (
        'CMOVS r64,r64',
        '480F48C3',
        {'RAX': 0xAAAA, 'RBX': 0xBBBB, 'EFLAGS': 0x80},  # SF=1
        {'RBX': FULL_TAINT_64, 'SF': 1},
        'RAX',
        FULL_TAINT_64,
        'CMOVS RAX,RBX — tainted SF + tainted source: dest fully tainted',
    ),
    # -----------------------------------------------------------------------
    # IMUL 3-operand / signed arithmetic
    # -----------------------------------------------------------------------
    (
        'IMUL r64,r64,imm32',
        '4869C200000080',
        {'RDX': 1},
        {'RDX': FULL_TAINT_64},
        'RAX',
        FULL_TAINT_64,
        'IMUL RAX,RDX,0x80000000 — large immediate 3-op propagates taint',
    ),
    (
        'IMUL r32,r32,imm8',
        '6BC00A',
        {'EAX': 5},
        {'EAX': FULL_TAINT_32},
        'EAX',
        FULL_TAINT_32,
        'IMUL EAX,EAX,10 — 32-bit 3-op propagates taint',
    ),
    # -----------------------------------------------------------------------
    # BT / BTC / BTR / BTS
    # -----------------------------------------------------------------------
    (
        'BT r64,imm8',
        '480FBA200A',
        {'RAX': 0x400},
        {'RAX': FULL_TAINT_64},
        'CF',
        1,
        'BT RAX,10 — tainted source taints CF',
    ),
    (
        'BTS r64,r64',
        '480FABD8',
        {'RAX': 0, 'RBX': 3},
        {'RBX': FULL_TAINT_64},
        'RAX',
        None,  # compare backends only
        'BTS RAX,RBX — tainted bit index taints result',
    ),
    # -----------------------------------------------------------------------
    # RCL / RCR (rotate through carry)
    # -----------------------------------------------------------------------
    (
        'RCL r64,1',
        '48D1D0',
        {'RAX': 0x1, 'EFLAGS': 0x1},  # CF=1 (concrete, untainted)
        {'RAX': FULL_TAINT_64, 'CF': 0},
        'RAX',
        0xFFFFFFFFFFFFFFFE,
        'RCL RAX,1 — bits 63:1 tainted (shifted from RAX), bit 0 = old CF (concrete)',
    ),
    # -----------------------------------------------------------------------
    # MOVSX / MOVZX edge cases
    # -----------------------------------------------------------------------
    (
        'MOVSX r64,r16',
        '480FBFC0',
        {'AX': 0x8000},
        {'AX': FULL_TAINT_16},
        'RAX',
        FULL_TAINT_64,
        'MOVSX RAX,AX — sign-extend 16→64 propagates all bits',
    ),
    (
        'MOVZX r64,r8',
        '480FB6C0',
        {'AL': 0xFF},
        {'AL': FULL_TAINT_8},
        'RAX',
        FULL_TAINT_8,
        'MOVZX RAX,AL — zero-extend 8→64 keeps byte taint',
    ),
    # -----------------------------------------------------------------------
    # RET (stack memory taint to RIP)
    # -----------------------------------------------------------------------
    (
        'RET',
        'C3',
        {'RSP': 0x80000000},
        {'RSP': 0, f'MEM_{hex(0x80000000)}_8': FULL_TAINT_64},
        'RIP',
        FULL_TAINT_64,
        'RET — tainted stack memory propagates to RIP',
    ),
    # -----------------------------------------------------------------------
    # CALL (indirect — taints RIP)
    # -----------------------------------------------------------------------
    (
        'CALL r64',
        '48FFD0',
        {'RAX': 0x401000, 'RSP': 0x80000010},
        {'RAX': FULL_TAINT_64, 'RSP': 0},
        None,
        None,  # RIP is consumed as control flow — compare full dicts
        'CALL RAX — tainted target (compare backends)',
    ),
    # -----------------------------------------------------------------------
    # NOP (sanity — no taint change)
    # -----------------------------------------------------------------------
    (
        'NOP',
        '90',
        {'RAX': 0xDEAD},
        {'RAX': FULL_TAINT_64},
        'RAX',
        FULL_TAINT_64,
        'NOP — no-op does not clear taint',
    ),
    # -----------------------------------------------------------------------
    # Partial-register write zero-extension (REX.W absent)
    # -----------------------------------------------------------------------
    (
        'MOV r32,r32 zext',
        '89C8',  # MOV EAX,ECX — zero-extends into RAX
        {'ECX': 0xDEADBEEF, 'EAX': 0},
        {'ECX': FULL_TAINT_32},
        'EAX',
        FULL_TAINT_32,
        'MOV EAX,ECX — 32-bit write zero-extends; EAX tainted',
    ),
    # -----------------------------------------------------------------------
    # XORPS / PXOR (SSE zeroing — backends may differ, compare only)
    # -----------------------------------------------------------------------
    # Skip SSE for now — requires SSE state format; mark as compare-only
    # -----------------------------------------------------------------------
    # Conditional branch (only taint propagation to flags matters)
    # -----------------------------------------------------------------------
    (
        'CMP r64,0 SF',
        '4883F800',
        {'RAX': 0xFFFFFFFFFFFFFFFF},
        {'RAX': FULL_TAINT_64},
        'SF',
        1,
        'CMP RAX,0 — tainted RAX taints SF (sign flag)',
    ),
    (
        'CMP r64,0 OF',
        '4883F800',
        {'RAX': 0x8000000000000000},
        {'RAX': FULL_TAINT_64},
        'OF',
        None,
        'CMP RAX,0 — OF behaviour with tainted RAX (compare backends only)',
    ),
]

# Build the ID list once for parametrize
CORPUS_IDS = [f'{mnemonic}:{desc[:30]}' for mnemonic, _, _, _, _, _, desc in CORPUS]

# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(scope='session')
def sim_unicorn() -> CellSimulator:
    """Unicorn-backed CellSimulator — the reference backend."""
    return CellSimulator(Architecture.AMD64, use_unicorn=True)


@pytest.fixture(scope='session')
def sim_pcode() -> CellSimulator:
    """P-code native CellSimulator — the new backend."""
    return CellSimulator(Architecture.AMD64, use_unicorn=False)


@pytest.fixture(scope='session')
def prebuilt_circuits() -> dict[str, Any]:
    """
    Pre-compile every circuit in CORPUS once at session scope.
    Both simulators share these cached LogicCircuit objects — the
    generate_static_rule LRU cache is keyed on (arch, bytstring) so
    there is exactly one circuit per unique instruction.
    """
    circuits: dict[str, Any] = {}
    for _mnemonic, hex_bytes, _, _, _, _, _ in CORPUS:
        key = hex_bytes
        if key not in circuits:
            circuits[key] = generate_static_rule(
                Architecture.AMD64,
                bytes.fromhex(hex_bytes),
                AMD64_REGS,
            )
    return circuits


# ===========================================================================
# Helper
# ===========================================================================


def _run(
    sim: CellSimulator,
    circuit: Any,
    input_values: dict,
    input_taint: dict,
    implicit_policy: ImplicitTaintPolicy = ImplicitTaintPolicy.KEEP,
) -> dict:
    ctx = EvalContext(
        input_values=input_values,
        input_taint=input_taint,
        simulator=sim,
        implicit_policy=implicit_policy,
    )
    return circuit.evaluate(ctx)


def _extract_flag(output: dict, flag: str) -> int:
    if flag in output:
        return int(output[flag])
    eflags = output.get('EFLAGS', 0)
    match flag:
        case 'CF':
            return (eflags >> 0) & 1
        case 'PF':
            return (eflags >> 2) & 1
        case 'ZF':
            return (eflags >> 6) & 1
        case 'SF':
            return (eflags >> 7) & 1
        case 'OF':
            return (eflags >> 11) & 1
    return 0


def _get(output: dict, key: str) -> int:
    """Get a register value, handling flag extraction transparently."""
    if key in ('CF', 'PF', 'ZF', 'SF', 'OF'):
        return _extract_flag(output, key)
    return output.get(key, 0)


# ===========================================================================
# PART 1 — Correctness: unicorn output == pcode output for every instruction
# ===========================================================================


@pytest.mark.parametrize(
    ('mnemonic', 'hex_bytes', 'input_values', 'input_taint', 'check_reg', 'expected_taint', 'description'),
    CORPUS,
    ids=CORPUS_IDS,
)
def test_pcode_matches_unicorn(
    sim_unicorn: CellSimulator,
    sim_pcode: CellSimulator,
    prebuilt_circuits: dict,
    mnemonic: str,
    hex_bytes: str,
    input_values: dict,
    input_taint: dict,
    check_reg: str | None,
    expected_taint: int | None,
    description: str,
) -> None:
    """
    For every instruction x scenario, assert that the P-code backend
    produces the same taint output as the Unicorn backend.

    This is the primary correctness gate: if pcode != unicorn the test fails,
    regardless of what the 'expected' column says.
    """
    circuit = prebuilt_circuits[hex_bytes]

    out_unicorn = _run(sim_unicorn, circuit, input_values, input_taint)
    out_pcode = _run(sim_pcode, circuit, input_values, input_taint)

    # 1. Compare the register we care about (if specified)
    if check_reg is not None:
        u_val = _get(out_unicorn, check_reg)
        p_val = _get(out_pcode, check_reg)
        assert u_val == p_val, (
            f'[{mnemonic}] {description}\n'
            f'  Register  : {check_reg}\n'
            f'  Unicorn   : {hex(u_val)}\n'
            f'  P-code    : {hex(p_val)}\n'
            f'  InputValues : {input_values}\n'
            f'  InputTaint  : {input_taint}'
        )

    # 2. Full dict comparison for all keys that unicorn produced
    differing = {}
    all_keys = set(out_unicorn.keys()) | set(out_pcode.keys())
    for k in all_keys:
        u = out_unicorn.get(k, 0)
        p = out_pcode.get(k, 0)
        if u != p:
            differing[k] = (u, p)

    assert (
        not differing
    ), f'[{mnemonic}] {description}\n  Full-dict mismatch on keys: {list(differing.keys())}\n' + '\n'.join(
        f'    {k}: unicorn={hex(u)} pcode={hex(p)}' for k, (u, p) in differing.items()
    )


# ===========================================================================
# PART 2 — Absolute correctness: expected taint values (Unicorn as oracle)
# ===========================================================================


@pytest.mark.parametrize(
    ('mnemonic', 'hex_bytes', 'input_values', 'input_taint', 'check_reg', 'expected_taint', 'description'),
    [(m, h, iv, it, cr, et, d) for m, h, iv, it, cr, et, d in CORPUS if et is not None],
    ids=[cid for cid, (_, _, _, _, _, et, _) in zip(CORPUS_IDS, CORPUS, strict=False) if et is not None],
)
def test_expected_taint_unicorn(
    sim_unicorn: CellSimulator,
    prebuilt_circuits: dict,
    mnemonic: str,
    hex_bytes: str,
    input_values: dict,
    input_taint: dict,
    check_reg: str,
    expected_taint: int,
    description: str,
) -> None:
    """Unicorn backend produces the expected taint value for known-good cases."""
    circuit = prebuilt_circuits[hex_bytes]
    out = _run(sim_unicorn, circuit, input_values, input_taint)
    actual = _get(out, check_reg)
    assert (
        actual == expected_taint
    ), f'[Unicorn] [{mnemonic}] {description}\n  {check_reg}: expected {hex(expected_taint)}, got {hex(actual)}'


@pytest.mark.parametrize(
    ('mnemonic', 'hex_bytes', 'input_values', 'input_taint', 'check_reg', 'expected_taint', 'description'),
    [(m, h, iv, it, cr, et, d) for m, h, iv, it, cr, et, d in CORPUS if et is not None],
    ids=[cid for cid, (_, _, _, _, _, et, _) in zip(CORPUS_IDS, CORPUS, strict=False) if et is not None],
)
def test_expected_taint_pcode(
    sim_pcode: CellSimulator,
    prebuilt_circuits: dict,
    mnemonic: str,
    hex_bytes: str,
    input_values: dict,
    input_taint: dict,
    check_reg: str,
    expected_taint: int,
    description: str,
) -> None:
    """P-code backend produces the expected taint value for known-good cases."""
    circuit = prebuilt_circuits[hex_bytes]
    out = _run(sim_pcode, circuit, input_values, input_taint)
    actual = _get(out, check_reg)
    assert (
        actual == expected_taint
    ), f'[P-code] [{mnemonic}] {description}\n  {check_reg}: expected {hex(expected_taint)}, got {hex(actual)}'


# ===========================================================================
# PART 3 — Multi-taint-scenario sweep: stress every instruction with
#           several different taint masks to maximise differential coverage
# ===========================================================================

TAINT_SCENARIOS: list[dict] = [
    {'RAX': FULL_TAINT_64, 'RBX': 0},
    {'RAX': 0, 'RBX': FULL_TAINT_64},
    {'RAX': FULL_TAINT_64, 'RBX': FULL_TAINT_64},
    {'RAX': 0xAAAAAAAAAAAAAAAA, 'RBX': 0},
    {'RAX': 0x00000000FFFFFFFF, 'RBX': 0},
    {'RAX': 0xFF, 'RBX': 0xFF},
    {'RAX': 0x01, 'RBX': 0x01},
]


@pytest.mark.parametrize(
    ('hex_bytes', 'taint'),
    list(
        itertools.product(
            # Only register-only instructions (no memory operands in input) for the sweep
            [
                '4889C3',  # MOV RBX,RAX
                '4801D8',  # ADD RAX,RBX
                '4829D8',  # SUB RAX,RBX
                '4821D8',  # AND RAX,RBX
                '4809D8',  # OR  RAX,RBX
                '4831D8',  # XOR RAX,RBX
                '48F7D8',  # NEG RAX
                '48F7D0',  # NOT RAX
                '4839D8',  # CMP RAX,RBX (flags)
                '4885C0',  # TEST RAX,RAX
                '480FAFD8',  # IMUL RAX,RBX
                '48C1E008',  # SHL RAX,8
                '48C1E808',  # SHR RAX,8
                '48C1C008',  # ROL RAX,8
                '480FC8',  # BSWAP RAX
            ],
            TAINT_SCENARIOS,
        ),
    ),
    ids=[
        f'{h[:8]}:taint={list(t.values())[:2]}'
        for h, t in itertools.product(
            [
                'MOV',
                'ADD',
                'SUB',
                'AND',
                'OR',
                'XOR',
                'NEG',
                'NOT',
                'CMP',
                'TEST',
                'IMUL',
                'SHL',
                'SHR',
                'ROL',
                'BSWAP',
            ],
            TAINT_SCENARIOS,
        )
    ],
)
def test_multi_scenario_backends_agree(
    sim_unicorn: CellSimulator,
    sim_pcode: CellSimulator,
    prebuilt_circuits: dict,
    hex_bytes: str,
    taint: dict,
) -> None:
    """
    Exhaustive taint-scenario sweep: for each (instruction, taint_mask) pair
    the two backends must produce identical output dicts.
    """
    if hex_bytes not in prebuilt_circuits:
        prebuilt_circuits[hex_bytes] = generate_static_rule(
            Architecture.AMD64,
            bytes.fromhex(hex_bytes),
            AMD64_REGS,
        )
    circuit = prebuilt_circuits[hex_bytes]
    # Build neutral concrete values so the instruction doesn't fault
    values = {'RAX': 1, 'RBX': 1, 'RCX': 8, 'RSP': 0x80000010, 'RBP': 0x80000100}

    out_u = _run(sim_unicorn, circuit, values, taint)
    out_p = _run(sim_pcode, circuit, values, taint)

    differing = {
        k: (out_u.get(k, 0), out_p.get(k, 0)) for k in set(out_u) | set(out_p) if out_u.get(k, 0) != out_p.get(k, 0)
    }
    assert not differing, f'{hex_bytes} taint={taint}\n' + '\n'.join(
        f'  {k}: unicorn={hex(u)} pcode={hex(p)}' for k, (u, p) in differing.items()
    )


# ===========================================================================
# PART 4 — Throughput benchmarks: single instruction, 100 rounds
# ===========================================================================

# Pick a representative mix of instruction categories
BENCH_SINGLE: list[tuple[str, str, dict, dict]] = [
    ('MOV_r64_r64', '4889C3', {'RAX': 0x1234, 'RBX': 0}, {'RAX': FULL_TAINT_64}),
    ('ADD_r64_r64', '4801D8', {'RAX': 1, 'RBX': 1}, {'RAX': FULL_TAINT_64}),
    ('SUB_r64_r64', '4829D8', {'RAX': 5, 'RBX': 2}, {'RAX': FULL_TAINT_64}),
    ('AND_r64_r64', '4821D8', {'RAX': 0xFF, 'RBX': 0xFF}, {'RAX': FULL_TAINT_64}),
    ('XOR_r64_r64', '4831D8', {'RAX': 1, 'RBX': 2}, {'RAX': FULL_TAINT_64}),
    ('IMUL_r64_r64', '480FAFD8', {'RAX': 2, 'RBX': 3}, {'RAX': FULL_TAINT_64}),
    ('SHL_r64_imm8', '48C1E008', {'RAX': 0x00FF00FF00FF00FF}, {'RAX': 0x00FF00FF00FF00FF}),
    ('BSWAP_r64', '480FC8', {'RAX': 0x0123456789ABCDEF}, {'RAX': FULL_TAINT_64}),
    ('CMP_r64_r64', '4839D8', {'RAX': 5, 'RBX': 5}, {'RAX': FULL_TAINT_64}),
    ('TEST_r64_r64', '4885C0', {'RAX': 0}, {'RAX': FULL_TAINT_64}),
    ('POPCNT', 'F3480FB8C0', {'RAX': 0xFF}, {'RAX': FULL_TAINT_64}),
    ('NEG_r64', '48F7D8', {'RAX': 5}, {'RAX': FULL_TAINT_64}),
    ('NOT_r64', '48F7D0', {'RAX': 0xFFFF}, {'RAX': FULL_TAINT_64}),
    ('LEA_complex', '488D0418', {'RAX': 0x1000, 'RBX': 0x20}, {'RAX': FULL_TAINT_64}),
]


@pytest.mark.parametrize(
    ('mnemonic', 'hex_bytes', 'input_values', 'input_taint'),
    BENCH_SINGLE,
    ids=[m for m, _, _, _ in BENCH_SINGLE],
)
def test_bench_single_unicorn(
    benchmark: Any,
    sim_unicorn: CellSimulator,
    prebuilt_circuits: dict,
    mnemonic: str,
    hex_bytes: str,
    input_values: dict,
    input_taint: dict,
) -> None:
    """Unicorn backend — single instruction throughput."""
    if hex_bytes not in prebuilt_circuits:
        prebuilt_circuits[hex_bytes] = generate_static_rule(
            Architecture.AMD64,
            bytes.fromhex(hex_bytes),
            AMD64_REGS,
        )
    circuit = prebuilt_circuits[hex_bytes]

    def _run_once() -> dict:
        return _run(sim_unicorn, circuit, input_values, input_taint)

    result = benchmark.pedantic(_run_once, rounds=100, warmup_rounds=5)
    assert result is not None


@pytest.mark.parametrize(
    ('mnemonic', 'hex_bytes', 'input_values', 'input_taint'),
    BENCH_SINGLE,
    ids=[m for m, _, _, _ in BENCH_SINGLE],
)
def test_bench_single_pcode(
    benchmark: Any,
    sim_pcode: CellSimulator,
    prebuilt_circuits: dict,
    mnemonic: str,
    hex_bytes: str,
    input_values: dict,
    input_taint: dict,
) -> None:
    """P-code backend — single instruction throughput."""
    if hex_bytes not in prebuilt_circuits:
        prebuilt_circuits[hex_bytes] = generate_static_rule(
            Architecture.AMD64,
            bytes.fromhex(hex_bytes),
            AMD64_REGS,
        )
    circuit = prebuilt_circuits[hex_bytes]

    def _run_once() -> dict:
        return _run(sim_pcode, circuit, input_values, input_taint)

    result = benchmark.pedantic(_run_once, rounds=100, warmup_rounds=5)
    assert result is not None


# ===========================================================================
# PART 5 — Sequential trace benchmark: the full CORPUS in order,
#           simulating an actual taint-analysis trace through a binary
# ===========================================================================


def _run_full_corpus(sim: CellSimulator, circuits: dict) -> list[dict]:
    """Evaluate every instruction in CORPUS and return all output dicts."""
    outputs = []
    for _, hex_bytes, input_values, input_taint, _, _, _ in CORPUS:
        circuit = circuits[hex_bytes]
        outputs.append(_run(sim, circuit, input_values, input_taint))
    return outputs


def test_bench_trace_unicorn(
    benchmark: Any,
    sim_unicorn: CellSimulator,
    prebuilt_circuits: dict,
) -> None:
    """
    Unicorn backend — full CORPUS sequential trace.
    Measures total throughput when evaluating many different instructions
    back-to-back, as happens during a real binary analysis.
    """
    result = benchmark.pedantic(
        _run_full_corpus,
        args=(sim_unicorn, prebuilt_circuits),
        rounds=30,
        warmup_rounds=3,
    )
    assert len(result) == len(CORPUS)


def test_bench_trace_pcode(
    benchmark: Any,
    sim_pcode: CellSimulator,
    prebuilt_circuits: dict,
) -> None:
    """
    P-code backend — full CORPUS sequential trace.
    """
    result = benchmark.pedantic(
        _run_full_corpus,
        args=(sim_pcode, prebuilt_circuits),
        rounds=30,
        warmup_rounds=3,
    )
    assert len(result) == len(CORPUS)


# ===========================================================================
# PART 6 — Repeated single-instruction hot-path benchmark
#           (simulates the tightest loop: one hot instruction 10 000x)
# ===========================================================================

_HOT_HEX = '4801D8'  # ADD RAX,RBX — most common arithmetic pattern
_HOT_VALUES = {'RAX': 1, 'RBX': 1}
_HOT_TAINT = {'RAX': FULL_TAINT_64, 'RBX': 0x5555555555555555}


def _run_hot_loop(sim: CellSimulator, circuit: Any, n: int) -> None:
    for _ in range(n):
        _run(sim, circuit, _HOT_VALUES, _HOT_TAINT)


def test_bench_hot_loop_unicorn(
    benchmark: Any,
    sim_unicorn: CellSimulator,
    prebuilt_circuits: dict,
) -> None:
    """
    Unicorn backend — 1 000-iteration hot loop on ADD RAX,RBX.
    This measures the raw per-cell overhead stripped of circuit-build cost.
    """
    circuit = prebuilt_circuits[_HOT_HEX]
    benchmark.pedantic(
        _run_hot_loop,
        args=(sim_unicorn, circuit, 1000),
        rounds=20,
        warmup_rounds=2,
    )


def test_bench_hot_loop_pcode(
    benchmark: Any,
    sim_pcode: CellSimulator,
    prebuilt_circuits: dict,
) -> None:
    """
    P-code backend — 1 000-iteration hot loop on ADD RAX,RBX.
    """
    circuit = prebuilt_circuits[_HOT_HEX]
    benchmark.pedantic(
        _run_hot_loop,
        args=(sim_pcode, circuit, 1000),
        rounds=20,
        warmup_rounds=2,
    )


# ===========================================================================
# PART 7 — Fallback rate smoke-test
#           After running the full corpus, assert that the pcode backend
#           fell back to Unicorn for at most 10% of calls (float-only).
# ===========================================================================


def test_pcode_fallback_rate(
    sim_pcode: CellSimulator,
    prebuilt_circuits: dict,
) -> None:
    """
    After running the full corpus, the pcode backend's fallback rate must be
    below 10 %.  A high rate means new opcodes need to be added to cell.py.
    """
    # Run corpus to populate stats
    _run_full_corpus(sim_pcode, prebuilt_circuits)

    pcode_eval = sim_pcode._pcode
    if pcode_eval is None:
        pytest.skip('use_unicorn=True, no pcode evaluator present')

    rate = pcode_eval.fallback_rate
    assert rate < 0.10, f'P-code fallback rate {rate:.1%} exceeds 10% threshold.\nStats: {pcode_eval.stats()}'


# ===========================================================================
# PART 8 — Taint propagation chain test
#           Simulate a multi-instruction taint chain where the output of
#           one instruction feeds the input of the next, and verify that
#           both backends produce the same taint state at every step.
# ===========================================================================

# A realistic instruction trace: load taint into RAX, propagate through ops
TAINT_CHAIN: list[tuple[str, str]] = [
    ('MOV RBX,RAX', '4889C3'),  # RBX ← RAX (tainted)
    ('ADD RCX,RBX', '4801D9'),  # RCX ← RCX + RBX (spreads)
    ('SHL RCX,8', '48C1E108'),  # RCX ← RCX << 8
    ('AND RDX,RCX', '4821CA'),  # RDX ← RDX & RCX
    ('OR RAX,RDX', '4809D0'),  # RAX ← RAX | RDX
    ('NOT RAX', '48F7D0'),  # RAX ← ~RAX
    ('XOR RBX,RAX', '4831C3'),  # RBX ← RBX ^ RAX
    ('NEG RBX', '48F7DB'),  # RBX ← -RBX
    ('IMUL RBX,RCX', '480FAFD9'),  # RBX ← RBX * RCX
    ('CMP RAX,RBX', '4839D8'),  # flags from tainted RAX,RBX
]


# Pre-compile chain circuits (same session cache)
@pytest.fixture(scope='session')
def chain_circuits() -> dict:
    circuits = {}
    for _, hex_bytes in TAINT_CHAIN:
        if hex_bytes not in circuits:
            circuits[hex_bytes] = generate_static_rule(
                Architecture.AMD64,
                bytes.fromhex(hex_bytes),
                AMD64_REGS,
            )
    return circuits


def _run_chain(sim: CellSimulator, circuits: dict) -> dict:
    """
    Run the taint chain, feeding output taint of each step as input taint
    for the next.  Returns the final taint state.
    """
    # Initial state: RAX fully tainted, all other regs zero / untainted
    values = {
        'RAX': 0xCAFEBABE12345678,
        'RBX': 0x1111111111111111,
        'RCX': 0x2222222222222222,
        'RDX': 0x3333333333333333,
        'RSP': 0x80000010,
    }
    taint = {'RAX': FULL_TAINT_64}

    for _, hex_bytes in TAINT_CHAIN:
        circuit = circuits[hex_bytes]
        ctx = EvalContext(
            input_values=values,
            input_taint=taint,
            simulator=sim,
            implicit_policy=ImplicitTaintPolicy.KEEP,
        )
        taint = circuit.evaluate(ctx)
        # Keep concrete values unchanged (in a real analysis they'd update too,
        # but for taint-only comparison the concrete values just need to be valid)

    return taint


def test_chain_backends_agree(
    sim_unicorn: CellSimulator,
    sim_pcode: CellSimulator,
    chain_circuits: dict,
) -> None:
    """
    Full 10-instruction taint chain: final taint state must be identical
    between Unicorn and P-code backends.
    """
    final_unicorn = _run_chain(sim_unicorn, chain_circuits)
    final_pcode = _run_chain(sim_pcode, chain_circuits)

    differing = {
        k: (final_unicorn.get(k, 0), final_pcode.get(k, 0))
        for k in set(final_unicorn) | set(final_pcode)
        if final_unicorn.get(k, 0) != final_pcode.get(k, 0)
    }
    assert not differing, 'Taint chain final state differs between backends:\n' + '\n'.join(
        f'  {k}: unicorn={hex(u)}  pcode={hex(p)}' for k, (u, p) in differing.items()
    )


def test_bench_chain_unicorn(
    benchmark: Any,
    sim_unicorn: CellSimulator,
    chain_circuits: dict,
) -> None:
    """Unicorn backend — 10-instruction taint-propagation chain throughput."""
    benchmark.pedantic(
        _run_chain,
        args=(sim_unicorn, chain_circuits),
        rounds=50,
        warmup_rounds=5,
    )


def test_bench_chain_pcode(
    benchmark: Any,
    sim_pcode: CellSimulator,
    chain_circuits: dict,
) -> None:
    """P-code backend — 10-instruction taint-propagation chain throughput."""
    benchmark.pedantic(
        _run_chain,
        args=(sim_pcode, chain_circuits),
        rounds=50,
        warmup_rounds=5,
    )


# ===========================================================================
# PART 9 — Diagnostic: print full backend outputs for EVERY corpus entry.
#           This test always passes. Run with -s to see the output.
#           Use it to diagnose pcode_matches failures by seeing exact values.
# ===========================================================================


def test_diagnostic_print_all_diffs(
    sim_unicorn: CellSimulator,
    sim_pcode: CellSimulator,
    prebuilt_circuits: dict,
    capsys: Any,
) -> None:
    """
    Diagnostic helper: runs every corpus entry through both backends and
    prints a side-by-side diff of their outputs.  Always passes.
    Run with:  pytest -s -k test_diagnostic_print_all_diffs

    Output format per entry:
      [MATCH]  or  [DIFF]  mnemonic: description
        KEY : unicorn=0x...  pcode=0x...  (only for differing keys)
    """
    any_diff = False
    lines = []

    for mnemonic, hex_bytes, input_values, input_taint, _check_reg, _, description in CORPUS:
        circuit = prebuilt_circuits[hex_bytes]
        try:
            out_u = _run(sim_unicorn, circuit, input_values, input_taint)
            out_p = _run(sim_pcode, circuit, input_values, input_taint)
        except Exception as exc:
            any_diff = True
            lines.append(f'[ERR]   {mnemonic}: {description}')
            lines.append(f'        EXCEPTION: {type(exc).__name__}: {exc}')
            continue

        diffs = {
            k: (out_u.get(k, 0), out_p.get(k, 0)) for k in set(out_u) | set(out_p) if out_u.get(k, 0) != out_p.get(k, 0)
        }

        if diffs:
            any_diff = True
            lines.append(f'[DIFF]  {mnemonic}: {description}')
            lines.append(f'        input_values = {input_values}')
            lines.append(f'        input_taint  = {input_taint}')
            for k, (u, p) in sorted(diffs.items()):
                lines.append(f'          {k:12s}: unicorn={hex(u):<22} pcode={hex(p)}')
        else:
            lines.append(f'[MATCH] {mnemonic}: {description}')

    output = '\n'.join(lines)
    with capsys.disabled():
        if any_diff:
            print('\n\n=== BACKEND COMPARISON DIAGNOSTIC ===')
            print(output)
            print(f"\n*** {sum(1 for lll in lines if lll.startswith('[DIFF]'))} entries differ ***")

    # Always pass — this is a diagnostic tool only.
    assert not any_diff, 'Backends differ on one or more entries.  Run with -s to see details.'


def test_diagnostic_pcode_register_map(
    sim_pcode: CellSimulator,
    capsys: Any,
) -> None:
    """
    Prints the register name -> (offset, size) mapping that the pcode evaluator uses.
    Helps verify that register offsets (especially flags) are resolved correctly.
    Run with:  pytest -s -k test_diagnostic_pcode_register_map
    """
    pcode = sim_pcode._pcode
    if pcode is None:
        pytest.skip('use_unicorn=True')

    offsets = pcode._offsets
    sizes = pcode._sizes

    interesting = [
        'RAX',
        'RBX',
        'RCX',
        'RDX',
        'RSI',
        'RDI',
        'RBP',
        'RSP',
        'RIP',
        'EAX',
        'EBX',
        'ECX',
        'EDX',
        'AX',
        'BX',
        'AL',
        'BL',
        'AH',
        'BH',
        'CF',
        'PF',
        'ZF',
        'SF',
        'OF',
        'EFLAGS',
        'R8',
        'R9',
        'R10',
        'R11',
        'R12',
        'R13',
        'R14',
        'R15',
    ]

    if os.getenv('SHOW_PCODE_MAP', '0') == '1':
        with capsys.disabled():
            print('\n\n=== PCODE REGISTER OFFSET MAP ===')
            print(f"  {'Name':<12} {'Offset':>10}  {'Size':>6}")
            print(f"  {'-'*12} {'-'*10}  {'-'*6}")
            for name in interesting:
                off = offsets.get(name, 'MISSING')
                size = sizes.get(name, 'MISSING')
                print(f'  {name:<12} {off!s:>10}  {size!s:>6}')

            print('\n  All flag-related entries:')
            for name, off in sorted(offsets.items()):
                if any(
                    f in name for f in ['CF', 'PF', 'ZF', 'SF', 'OF', 'FLAGS', 'cf', 'pf', 'zf', 'sf', 'of', 'flags']
                ):
                    print(f"  {name:<12} offset={off}  size={sizes.get(name,'?')}")

    assert True
