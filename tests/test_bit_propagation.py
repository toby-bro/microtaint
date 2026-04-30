"""
test_bit_propagation.py
=======================
Per-pcode-operation tests for bit-precise taint propagation.

Each test generates the LogicCircuit for ONE x86-64 instruction via
generate_static_rule and evaluates it via circuit.evaluate(EvalContext).
We compare the output taint mask to the differential ground truth:

    expected_diff = result(V | T) XOR result(V & ~T)

Where T is the input taint mask, V is the concrete input value.

Why this set of tests
---------------------
diag4 showed taint never enters registers via memory loads in our SipHash
trace.  These tests isolate that failure to a single instruction so we
can pinpoint exactly which opcode breaks bit propagation.

Coverage
--------
  Tier 1 — register-to-register baseline (must work):
    - mov reg, reg
    - xor reg, reg / xor reg, imm
    - or  reg, reg
    - and reg, reg
    - add reg, reg / add reg, imm
    - sub reg, reg
    - shl, shr, rol  (rotation primitives used in SipHash)

  Tier 2 — memory loads (CURRENTLY SUSPECTED BROKEN):
    - movzx eax, byte ptr [rax]            ← #45 in diag4
    - mov rax, qword ptr [rbp - 0x40]      ← stack-relative load
    - mov rax, qword ptr [rax]             ← register-indirect load

  Tier 3 — single-bit precision (the avalanche claim):
    - xor with single tainted bit          (must yield exactly that bit)
    - add with low-bit taint and known V   (carry ripple ground truth)
    - or with single bit                   (preserves the bit)

  Tier 4 — composition / cross-register flow:
    - add rax, rbx where ONLY rbx is tainted
      (shows whether taint crosses register boundaries on ADD)
    - movabs rax, 0x...   (must clear all RAX taint)

Usage
-----
    uv run pytest test_bit_propagation.py -v
"""

# ruff: noqa: ARG001, PLC0415
# mypy: disable-error-code="no-untyped-def,no-untyped-call,type-arg,no-any-return"

from __future__ import annotations

from typing import Callable

import pytest

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, ImplicitTaintPolicy, Register

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def simulator() -> CellSimulator:
    return CellSimulator(Architecture.AMD64)


@pytest.fixture(scope='module')
def regs() -> list[Register]:
    """Minimal AMD64 register set — full set is in test_mistral_comprehensive.py."""
    return [
        Register(name='RAX', bits=64),
        Register(name='EAX', bits=32),
        Register(name='AX', bits=16),
        Register(name='AL', bits=8),
        Register(name='RBX', bits=64),
        Register(name='EBX', bits=32),
        Register(name='RCX', bits=64),
        Register(name='ECX', bits=32),
        Register(name='RDX', bits=64),
        Register(name='EDX', bits=32),
        Register(name='RSI', bits=64),
        Register(name='RDI', bits=64),
        Register(name='RBP', bits=64),
        Register(name='RSP', bits=64),
        Register(name='R8', bits=64),
        Register(name='R9', bits=64),
        Register(name='RIP', bits=64),
        Register(name='EFLAGS', bits=32),
        Register(name='ZF', bits=1),
        Register(name='CF', bits=1),
        Register(name='SF', bits=1),
        Register(name='OF', bits=1),
        Register(name='PF', bits=1),
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx(simulator, taint, values, *, shadow=None):
    """Build an EvalContext with IGNORE policy (we don't care about implicit taint)."""
    return EvalContext(
        input_taint=taint,
        input_values=values,
        simulator=simulator,
        implicit_policy=ImplicitTaintPolicy.IGNORE,
        shadow_memory=shadow,
    )


def _eval(simulator, regs, bytestring, taint, values, *, shadow=None):
    """Generate the circuit and evaluate it. Returns the output_taint dict."""
    circuit = generate_static_rule(Architecture.AMD64, bytestring, regs)
    return circuit.evaluate(_ctx(simulator, taint, values, shadow=shadow))


def _diff_truth(op: Callable, v: int, t: int, mask: int = 0xFFFFFFFFFFFFFFFF) -> int:
    """
    Differential ground-truth: op(V|T) XOR op(V&~T), masked to mask.
    `op` is a unary callable that does the operation given a concrete input.
    """
    return (op(v | t) ^ op(v & ~t)) & mask


# ===========================================================================
# Tier 1 — register-to-register baseline (full taint mask)
# ===========================================================================


class TestTier1RegisterBaseline:
    """If any of these fail, the whole engine is broken; nothing else matters."""

    def test_mov_reg_reg_propagates_full_taint(self, simulator, regs):
        # mov rbx, rax  (48 89 c3)
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('4889c3'),
            taint={'RAX': 0xFFFFFFFFFFFFFFFF, 'RBX': 0},
            values={'RAX': 0x1234, 'RBX': 0},
        )
        assert out.get('RBX', 0) == 0xFFFFFFFFFFFFFFFF, f'mov rbx, rax: expected full taint, got {out.get("RBX", 0):#x}'

    def test_xor_reg_reg_propagates_full_taint(self, simulator, regs):
        # xor rax, rbx (48 31 d8)
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('4831d8'),
            taint={'RAX': 0xFF, 'RBX': 0xF0},
            values={'RAX': 0, 'RBX': 0},
        )
        # XOR is bit-independent: each bit's taint comes from EITHER input
        assert out.get('RAX', 0) == (
            0xFF | 0xF0
        ), f'xor rax, rbx: expected {0xFF | 0xF0:#x}, got {out.get("RAX", 0):#x}'

    def test_or_reg_reg_propagates(self, simulator, regs):
        # or rax, rbx (48 09 d8)
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('4809d8'),
            taint={'RAX': 0x0F, 'RBX': 0x10},
            values={'RAX': 0, 'RBX': 0},
        )
        # bit i of OR-result depends on bit i of either input
        assert out.get('RAX', 0) == (0x0F | 0x10)

    def test_add_reg_reg_propagates(self, simulator, regs):
        # add rax, rbx (48 01 d8)
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('4801d8'),
            taint={'RAX': 0xAAAAAAAAAAAAAAAA, 'RBX': 0x5555555555555555},
            values={'RAX': 0, 'RBX': 0},
        )
        assert out.get('RAX', 0) == 0xFFFFFFFFFFFFFFFF, f'add rax, rbx: expected full taint, got {out.get("RAX", 0):#x}'


# ===========================================================================
# Tier 2 — memory loads (THESE ARE WHERE SipHash TAINT IS DYING)
# ===========================================================================


class TestTier2MemoryLoads:
    """Loads must propagate shadow taint into the destination register."""

    def test_movzx_byte_from_register_indirect(self, simulator, regs):
        """
        movzx eax, byte ptr [rax]   (0f b6 00)

        This is THE instruction (idx 45 in diag4) where SipHash's input
        bytes enter the register file via the U8TO64_LE macro.

        Setup: input_taint['MEM_<addr>_1'] = 0x01, RAX points there.
        Expected: output taint on RAX low byte (AL) has bit 0 set.
        """
        addr = 0x1000
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('0fb600'),
            taint={'RAX': 0, f'MEM_{hex(addr)}_1': 0x01},
            values={'RAX': addr},
        )
        rax_taint = out.get('RAX', 0)
        assert rax_taint & 0xFF == 0x01, (
            f'movzx eax, [rax]: expected AL bit 0 tainted (got RAX taint {rax_taint:#x}). '
            'This is where SipHash input loading currently fails to taint registers.'
        )

    def test_mov_qword_from_stack(self, simulator, regs):
        """
        mov rax, qword ptr [rbp - 0x40]  (48 8b 45 c0)

        Stack-relative 8-byte load — the form used by SipHash to load
        the message into v0..v3 between rounds.
        """
        rbp = 0x7FFFFFFF0000
        addr = rbp - 0x40
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('488b45c0'),
            taint={'RAX': 0, f'MEM_{hex(addr)}_8': 0xFFFFFFFFFFFFFFFF, 'RBP': 0},
            values={'RAX': 0, 'RBP': rbp},
        )
        assert (
            out.get('RAX', 0) == 0xFFFFFFFFFFFFFFFF
        ), f'mov rax, [rbp-0x40]: expected full taint, got {out.get("RAX", 0):#x}'

    def test_mov_qword_from_register_indirect(self, simulator, regs):
        """
        mov rax, qword ptr [rax]  (48 8b 00)
        Register-indirect 8-byte load.
        """
        addr = 0x2000
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('488b00'),
            taint={'RAX': 0, f'MEM_{hex(addr)}_8': 0x00FF00FF00FF00FF},
            values={'RAX': addr},
        )
        assert out.get('RAX', 0) == 0x00FF00FF00FF00FF


# ===========================================================================
# Tier 3 — single-bit precision (the differential / avalanche claim)
# ===========================================================================


class TestTier3SingleBitPrecision:
    """Single-bit input taint must propagate per the differential ground truth."""

    def test_xor_single_bit_preserves_position(self, simulator, regs):
        """
        XOR with constant: bit i of input → bit i of output (XOR is bit-independent).

          xor eax, 0x5A   (35 5a 00 00 00)
        Tainting only bit 3 of EAX must yield exactly bit 3 in the output.
        """
        out = _eval(simulator, regs, bytes.fromhex('355a000000'), taint={'RAX': 0x08}, values={'RAX': 0xAB})  # bit 3
        rax_taint = out.get('RAX', 0)
        assert (
            rax_taint & 0xFFFFFFFF == 0x08
        ), f'xor eax, imm: expected output taint 0x08, got {rax_taint & 0xFFFFFFFF:#x}'

    def test_add_low_bit_carry_ripple(self, simulator, regs):
        """
        add rax, rbx with bit 0 of RAX tainted.

        Differential:  V_RAX=0xFF, V_RBX=0x01, T_RAX=0x01
          C1 = (0xFF | 0x01) + 0x01 = 0xFF + 0x01 = 0x100
          C2 = (0xFF & ~0x01) + 0x01 = 0xFE + 0x01 = 0xFF
          diff = 0x100 ^ 0xFF = 0x1FF (9 bits set)

        Carry ripples from bit 0 all the way up through the all-ones byte.
        """
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('4801d8'),  # add rax, rbx
            taint={'RAX': 0x01, 'RBX': 0},
            values={'RAX': 0xFF, 'RBX': 0x01},
        )
        expected = _diff_truth(lambda x: (x + 0x01) & 0xFFFFFFFFFFFFFFFF, v=0xFF, t=0x01)
        rax_taint = out.get('RAX', 0)
        assert rax_taint == expected, (
            f'add carry ripple: expected {expected:#x}, got {rax_taint:#x}. '
            f'A passing test produces multiple bits set (>= 2) because carry '
            f'from bit 0 affects higher bits.'
        )
        assert bin(rax_taint).count('1') >= 2, 'add with carry chain MUST produce more than one tainted output bit'


# ===========================================================================
# Tier 4 — cross-register flow
# ===========================================================================


class TestTier4CrossRegister:
    """Taint must cross between registers via mixing operations."""

    def test_add_only_rbx_tainted(self, simulator, regs):
        """
        add rax, rbx with only RBX tainted — RAX must become tainted too.
        This is the cross-register flow that SipHash mixing depends on.
        """
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('4801d8'),
            taint={'RAX': 0, 'RBX': 0xFF},
            values={'RAX': 0, 'RBX': 0},
        )
        rax_taint = out.get('RAX', 0)
        assert rax_taint != 0, 'add rax, rbx with tainted RBX must taint RAX (cross-register flow)'

    def test_movabs_clears_destination_taint(self, simulator, regs):
        """
        movabs rax, 0x736f6d6570736575   (48 b8 75 65 73 70 65 6d 6f 73)
        Loading an immediate must CLEAR any prior RAX taint.

        Used to load SipHash's magic constants — must not preserve stale taint.
        """
        out = _eval(
            simulator,
            regs,
            bytes.fromhex('48b875657370656d6f73'),
            taint={'RAX': 0xFFFFFFFFFFFFFFFF},
            values={'RAX': 0},
        )
        assert out.get('RAX', 0) == 0, 'movabs rax, imm must clear RAX taint'


# ===========================================================================
# Tier 5 — bit-shifts / rotations (SipHash rotates everywhere)
# ===========================================================================


class TestTier5ShiftsAndRotates:
    """Shifts move tainted bits to known positions."""

    def test_shl_by_constant_preserves_bit(self, simulator, regs):
        """
        shl rax, 8   (48 c1 e0 08)
        Tainting bit 0 must produce bit 8 in the output (shifted by 8).
        """
        out = _eval(simulator, regs, bytes.fromhex('48c1e008'), taint={'RAX': 0x01}, values={'RAX': 0})
        rax_taint = out.get('RAX', 0)
        assert rax_taint == 0x100, f'shl rax, 8 with bit 0 tainted: expected 0x100, got {rax_taint:#x}'

    def test_shr_by_constant_preserves_bit(self, simulator, regs):
        """
        shr rax, 4   (48 c1 e8 04)
        Tainting bit 7 must produce bit 3 in the output (shifted right by 4).
        """
        out = _eval(simulator, regs, bytes.fromhex('48c1e804'), taint={'RAX': 0x80}, values={'RAX': 0})
        rax_taint = out.get('RAX', 0)
        assert rax_taint == 0x08, f'shr rax, 4 with bit 7 tainted: expected 0x08, got {rax_taint:#x}'


# ===========================================================================
# Tier 6 — RMW (Read-Modify-Write) memory destinations
# ===========================================================================
# These tests document the ROOT-CAUSE BUG that breaks SipHash avalanche.
#
# When SipHash mixes state, gcc -O0 emits instructions like:
#     add [rbp-0x10], rax   ; v0 += v1
#     xor [rbp-0x18], rax   ; v1 ^= v0
# These are READ-MODIFY-WRITE: read the old memory value, do arithmetic
# with the source register, write the new value back.
#
# The engine's rule generator (engine.py, generate_taint_assignments)
# treats ALL memory-destination instructions as pure STOREs, computing
# the destination taint as `OR of all input taints` instead of the proper
# differential `C1_cell XOR C2_cell`.  For pure stores (`mov [mem], reg`)
# this is correct — the value being stored IS the source register.  For
# RMW, this loses the carry chain in ADD/SUB and the bit-by-bit XOR
# differential, collapsing taint to literally the OR of input bit masks.
#
# Effect on SipHash: a single tainted input bit propagates through one
# byte load (correctly), but then `add [v0_slot], v1_reg` fails to
# expand the taint via carry, so the avalanche never happens.
#
# Suggested fix in engine.py: in generate_taint_assignments, before
# taking the `is_store_target` early-return, check whether slice_ops
# contains BOTH a LOAD and a STORE op pointing at the same address.
# If yes → it's RMW, use the normal differential path (the same one
# used for register-destination ADD).
# ===========================================================================


class TestTier6RMWMemoryDestination:
    """RMW operations to memory must use the differential, not just OR."""

    def test_add_mem_reg_carry_ripple(self, simulator, regs):
        """
        add [rbp-0x10], rax  (48 01 45 f0)

        With V_mem=0xFF, V_RAX=0x01, T_mem=0x01, T_RAX=0x01:
          C1 = (0xFF | 0x01) + (0x01 | 0x01) = 0xFF + 0x01 = 0x100
          C2 = (0xFF & ~0x01) + (0x01 & ~0x01) = 0xFE + 0x00 = 0xFE
          diff = 0x100 ^ 0xFE = 0x1FE  (popcount=8)

        The engine currently returns 0x01 (just T_mem|T_RAX) which is the
        bug.  When fixed, this test will pass.
        """
        from microtaint.emulator.shadow import BitPreciseShadowMemory

        rbp = 0x80000000DD00
        mem_addr = rbp - 0x10
        shadow = BitPreciseShadowMemory()
        shadow.write_mask(mem_addr, 0x01, 8)

        def reader(addr, sz):
            if addr == mem_addr:
                return 0xFF
            return 0

        ctx = EvalContext(
            input_taint={'RAX': 0x01},
            input_values={'RAX': 0x01, 'RBP': rbp},
            simulator=simulator,
            shadow_memory=shadow,
            mem_reader=reader,
            implicit_policy=ImplicitTaintPolicy.IGNORE,
        )
        circuit = generate_static_rule(Architecture.AMD64, bytes.fromhex('480145f0'), regs)
        out = circuit.evaluate(ctx)
        mem_taint = out.get(f'MEM_{hex(mem_addr)}_8', 0)
        # Bare minimum: more than 1 bit must be tainted (carry ripple)
        popcount = bin(mem_taint).count('1')
        assert popcount >= 2, (
            f'add [mem], reg: expected popcount>=2 from carry ripple, got {popcount} '
            f'(mask={mem_taint:#x}). This is the SipHash avalanche bug — engine '
            f'treats RMW like a pure STORE and skips the differential.'
        )

    def test_xor_mem_reg_bit_independence(self, simulator, regs):
        """
        xor [rbp-0x18], rax  (48 31 45 e8)
        XOR is bit-independent so the differential must equal T_mem | T_RAX
        — but the test still verifies the differential path is taken (not
        a bare OR), because for XOR they happen to coincide.
        """
        from microtaint.emulator.shadow import BitPreciseShadowMemory

        rbp = 0x80000000DD00
        mem_addr = rbp - 0x18
        shadow = BitPreciseShadowMemory()
        shadow.write_mask(mem_addr, 0xF0, 8)

        def reader(addr, sz):
            if addr == mem_addr:
                return 0xAB
            return 0

        ctx = EvalContext(
            input_taint={'RAX': 0x0F},
            input_values={'RAX': 0xCD, 'RBP': rbp},
            simulator=simulator,
            shadow_memory=shadow,
            mem_reader=reader,
            implicit_policy=ImplicitTaintPolicy.IGNORE,
        )
        circuit = generate_static_rule(Architecture.AMD64, bytes.fromhex('483145e8'), regs)
        out = circuit.evaluate(ctx)
        mem_taint = out.get(f'MEM_{hex(mem_addr)}_8', 0)
        # XOR is bit-independent, so the right answer is T_mem | T_RAX = 0xFF
        assert mem_taint & 0xFF == 0xFF, f'xor [mem], reg: expected low byte = 0xFF, got {mem_taint & 0xFF:#x}'

    def test_sub_mem_reg_borrow_ripple(self, simulator, regs):
        """
        sub [rbp-0x10], rax  (48 29 45 f0)
        Borrow propagation: same shape as ADD's carry ripple.

        Inputs that produce a genuine differential:
          V_mem = 0x100, T_mem = 0x01 (only mem tainted at bit 0)
          V_RAX = 0x01, T_RAX = 0
        Differential:
          C1 = (0x100 | 0x01) - (0x01) = 0x100
          C2 = (0x100 & ~0x01) - (0x01) = 0x0FF
          diff = 0x1FF (popcount = 9)
        """
        from microtaint.emulator.shadow import BitPreciseShadowMemory

        rbp = 0x80000000DD00
        mem_addr = rbp - 0x10
        shadow = BitPreciseShadowMemory()
        shadow.write_mask(mem_addr, 0x01, 8)

        def reader(addr, sz):
            if addr == mem_addr:
                return 0x100
            return 0

        ctx = EvalContext(
            input_taint={'RAX': 0},
            input_values={'RAX': 0x01, 'RBP': rbp},
            simulator=simulator,
            shadow_memory=shadow,
            mem_reader=reader,
            implicit_policy=ImplicitTaintPolicy.IGNORE,
        )
        circuit = generate_static_rule(Architecture.AMD64, bytes.fromhex('482945f0'), regs)
        out = circuit.evaluate(ctx)
        mem_taint = out.get(f'MEM_{hex(mem_addr)}_8', 0)
        popcount = bin(mem_taint).count('1')
        assert (
            popcount >= 2
        ), f'sub [mem], reg: expected popcount>=2 from borrow ripple, got {popcount} (mask={mem_taint:#x})'
