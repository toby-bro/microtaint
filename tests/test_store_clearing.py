"""
Tests isolating the computed-pointer STORE shadow-clearing bug.

These tests run entirely in-process — no Qiling, no subprocess, no compilation.
They call generate_static_rule + EvalContext directly to verify whether the
circuit produces MEM_ output keys for various store instruction forms.

The bug:
    STORE instructions whose pointer is a SLEIGH 'unique' varnode
    (computed addresses: call, mov [rbp-N], push via RSP-8 intermediate,
    str [reg, #off] on ARM) produce no MEM_ key in circuit output_state.
    Shadow memory at those addresses is therefore never updated by the circuit,
    leaving stale taint that causes false-positive BOF findings.

Expected behaviour after the fix:
    Every instruction that writes to memory must produce a MEM_ key in
    output_state with the correct taint value (0 for untainted stores,
    nonzero for tainted stores). The wrapper's _instruction_evaluator then
    calls shadow.write_mask(addr, val, size) which clears or sets taint correctly.
"""

from __future__ import annotations

import pytest

from microtaint.emulator.shadow import BitPreciseShadowMemory
from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, ImplicitTaintPolicy, Register

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

X64 = [
    Register('RAX', 64),
    Register('RBX', 64),
    Register('RCX', 64),
    Register('RDX', 64),
    Register('RSI', 64),
    Register('RDI', 64),
    Register('RBP', 64),
    Register('RSP', 64),
    Register('R8', 64),
    Register('R9', 64),
    Register('R10', 64),
    Register('R11', 64),
    Register('R12', 64),
    Register('R13', 64),
    Register('R14', 64),
    Register('R15', 64),
    Register('RIP', 64),
    Register('EFLAGS', 32),
    Register('ZF', 1),
    Register('CF', 1),
    Register('SF', 1),
    Register('OF', 1),
    Register('PF', 1),
]

STACK = 0x80000000  # safe stack base
RBP_VAL = STACK + 0x200


@pytest.fixture(scope='module')
def sim() -> CellSimulator:
    return CellSimulator(Architecture.AMD64)


def _eval(
    hex_bytes: str,
    reg_taint: dict,
    reg_values: dict,
    shadow: BitPreciseShadowMemory,
    sim: CellSimulator,
) -> dict:
    """Run circuit.evaluate() and return the full output_state dict."""
    bs = bytes.fromhex(hex_bytes)
    circuit = generate_static_rule(Architecture.AMD64, bs, X64)
    ctx = EvalContext(
        input_taint=reg_taint,
        input_values=reg_values,
        simulator=sim,
        implicit_policy=ImplicitTaintPolicy.KEEP,
        shadow_memory=shadow,
        mem_reader=lambda addr, sz: 0,
    )
    print(circuit)
    return circuit.evaluate(ctx)


def _mem_keys(output_state: dict) -> dict[str, int]:
    """Return only MEM_ keys from output_state."""
    return {k: v for k, v in output_state.items() if k.startswith('MEM_')}


def _shadow_at(shadow: BitPreciseShadowMemory, address: int, size: int = 8) -> int:
    return shadow.read_mask(address, size)


# ===========================================================================
# PUSH — direct register pointer (RSP used directly as pointer)
# ===========================================================================


class TestPushInstruction:
    """
    push rbp  (55)
    SLEIGH: RSP = RSP - 8; STORE [RSP], RBP
    The pointer MAY be in register space (RSP directly) or unique space (RSP-8).
    """

    def test_push_rbp_produces_mem_output(self, sim: CellSimulator) -> None:
        """push rbp must produce a MEM_ key so shadow at [RSP-8] is updated."""
        shadow = BitPreciseShadowMemory()
        # Pre-taint the slot that push will write to (RSP - 8)
        shadow.write_mask(STACK - 8, 0xFFFFFFFFFFFFFFFF, 8)

        output = _eval('55', {}, {'RSP': STACK, 'RBP': RBP_VAL}, shadow, sim)
        mem = _mem_keys(output)

        assert mem, (
            'push rbp produced no MEM_ output. '
            'Shadow at [RSP-8] will retain stale taint, causing false-positive BOF.'
        )

    def test_push_rbp_clears_shadow_when_rbp_untainted(self, sim: CellSimulator) -> None:
        """When RBP is untainted, push rbp must write MEM_=0, clearing stale taint."""
        shadow = BitPreciseShadowMemory()
        shadow.write_mask(STACK - 8, 0xFFFFFFFFFFFFFFFF, 8)  # stale taint

        output = _eval('55', {}, {'RSP': STACK, 'RBP': RBP_VAL}, shadow, sim)
        mem = _mem_keys(output)

        # The MEM_ key for the pushed slot must exist and be 0
        pushed_slot = STACK - 8
        found_clear = any(int(k[4 : k.rfind('_')], 16) == pushed_slot and v == 0 for k, v in mem.items())
        assert (
            found_clear
        ), f'push rbp with untainted RBP did not produce a zero MEM_ at {hex(pushed_slot)}. MEM_ outputs: {mem}'

    def test_push_rbp_propagates_taint_when_rbp_tainted(self, sim: CellSimulator) -> None:
        """When RBP is tainted, push rbp must write nonzero MEM_ taint."""
        shadow = BitPreciseShadowMemory()
        output = _eval(
            '55',
            {'RBP': 0xFFFFFFFFFFFFFFFF},  # RBP fully tainted
            {'RSP': STACK, 'RBP': RBP_VAL},
            shadow,
            sim,
        )
        mem = _mem_keys(output)
        tainted_writes = {k: v for k, v in mem.items() if v != 0}
        assert tainted_writes, 'push rbp with tainted RBP produced no nonzero MEM_ output.'


# ===========================================================================
# CALL — computed pointer (return address pushed via unique varnode RSP-8)
# ===========================================================================


class TestCallInstruction:
    """
    call rel32  (e8 00 00 00 00 = call +5, calls next instruction)
    SLEIGH: $u1 = RSP-8; STORE [$u1], RIP_next; RSP = $u1; BRANCH target
    The STORE pointer is a 'unique' varnode — this is the bug location.
    """

    def test_call_produces_mem_output_for_return_address(self, sim: CellSimulator) -> None:
        """
        'call' must produce a MEM_ key for the pushed return address slot.

        This is the primary regression test for the computed-pointer STORE bug.
        If this test fails, the engine does NOT produce MEM_ output for call's
        return-address write, and shadow memory at that slot will never be cleared.
        """
        shadow = BitPreciseShadowMemory()
        # Pre-taint the slot at [RSP-8] — simulates a previous function that
        # left taint there (e.g. buf[0] from sys_read)
        shadow.write_mask(STACK - 8, 0xFFFFFFFFFFFFFFFF, 8)

        output = _eval(
            'e800000000',  # call +5 (to next instruction)
            {},
            {'RSP': STACK, 'RIP': 0x1000},
            shadow,
            sim,
        )
        mem = _mem_keys(output)

        assert mem, (
            'call rel32 produced no MEM_ output. '
            'The return address slot [RSP-8] retains stale taint, '
            'causing false-positive BOF when the called function returns.'
        )

    def test_call_clears_return_address_slot_shadow(self, sim: CellSimulator) -> None:
        """
        The return address pushed by 'call' is a code constant — untainted.
        Shadow at [RSP-8] must be cleared (val=0) by the circuit.
        """
        shadow = BitPreciseShadowMemory()
        shadow.write_mask(STACK - 8, 0xFFFFFFFFFFFFFFFF, 8)

        output = _eval(
            'e800000000',
            {},
            {'RSP': STACK, 'RIP': 0x1000},
            shadow,
            sim,
        )
        mem = _mem_keys(output)

        pushed_slot = STACK - 8
        found_clear = any(int(k[4 : k.rfind('_')], 16) == pushed_slot and v == 0 for k, v in mem.items())
        assert found_clear, f'call rel32 did not produce a zero MEM_ at {hex(pushed_slot)}. MEM_ outputs: {mem}'

    def test_ret_with_clean_slot_does_not_taint_rip(self, sim: CellSimulator) -> None:
        """
        ret (c3) with a clean return address slot must not taint RIP.
        This is the false-positive symptom: if shadow[RSP] != 0, BOF fires.
        """
        shadow = BitPreciseShadowMemory()
        # Shadow at RSP is clean (as it should be after call cleared it)

        output = _eval('c3', {}, {'RSP': STACK}, shadow, sim)

        rip_taint = output.get('RIP', 0)
        assert rip_taint == 0, (
            f'ret with clean shadow[RSP] tainted RIP with {hex(rip_taint)}. ' 'This is a false-positive BOF.'
        )

    def test_ret_with_tainted_slot_does_taint_rip(self, sim: CellSimulator) -> None:
        """
        ret with a tainted return address slot MUST taint RIP.
        This is a real BOF — the saved return address was overwritten.
        """
        shadow = BitPreciseShadowMemory()
        shadow.write_mask(STACK, 0xFFFFFFFFFFFFFFFF, 8)  # saved RIP is tainted

        output = _eval('c3', {}, {'RSP': STACK}, shadow, sim)

        # With implicit_policy=KEEP, RIP taint is preserved in output
        rip_taint = output.get('RIP', 0)
        assert rip_taint != 0, 'ret with tainted shadow[RSP] did not taint RIP. Real BOF would not be detected.'


# ===========================================================================
# MOV [rbp-N], reg — computed pointer (RBP+offset via unique varnode)
# ===========================================================================


class TestMovToMemory:
    """
    mov [rbp-8], rax  (48 89 45 f8)
    SLEIGH: $u1 = RBP - 8; STORE [$u1], RAX
    Pointer is a unique varnode — same class of bug as CALL.
    """

    def test_mov_mem_rbp_produces_mem_output(self, sim: CellSimulator) -> None:
        """mov [rbp-8], rax must produce a MEM_ key in output_state."""
        shadow = BitPreciseShadowMemory()
        shadow.write_mask(RBP_VAL - 8, 0xFFFFFFFFFFFFFFFF, 8)

        output = _eval(
            '488945F8',  # mov [rbp-8], rax
            {},
            {'RSP': STACK, 'RBP': RBP_VAL, 'RAX': 0xDEAD},
            shadow,
            sim,
        )
        mem = _mem_keys(output)

        assert mem, 'mov [rbp-8], rax produced no MEM_ output. Stale taint at [rbp-8] is never cleared.'

    def test_mov_mem_rbp_clears_slot_when_rax_untainted(self, sim: CellSimulator) -> None:
        """Storing untainted RAX to [rbp-8] must clear shadow there."""
        shadow = BitPreciseShadowMemory()
        shadow.write_mask(RBP_VAL - 8, 0xFFFFFFFFFFFFFFFF, 8)

        output = _eval(
            '488945F8',
            {},  # no register taint
            {'RSP': STACK, 'RBP': RBP_VAL, 'RAX': 0xDEAD},
            shadow,
            sim,
        )
        mem = _mem_keys(output)

        slot = RBP_VAL - 8
        found_clear = any(int(k[4 : k.rfind('_')], 16) == slot and v == 0 for k, v in mem.items())
        assert (
            found_clear
        ), f'mov [rbp-8], rax with untainted RAX did not produce zero MEM_ at {hex(slot)}. MEM_ outputs: {mem}'

    def test_mov_mem_rbp_propagates_taint_when_rax_tainted(self, sim: CellSimulator) -> None:
        """Storing tainted RAX to [rbp-8] must write nonzero taint to shadow."""
        shadow = BitPreciseShadowMemory()

        output = _eval(
            '488945F8',
            {'RAX': 0xFFFFFFFFFFFFFFFF},
            {'RSP': STACK, 'RBP': RBP_VAL, 'RAX': 0xAAAA},
            shadow,
            sim,
        )
        mem = _mem_keys(output)

        tainted = {k: v for k, v in mem.items() if v != 0}
        assert tainted, 'mov [rbp-8], rax with fully-tainted RAX produced no nonzero MEM_ output.'

    def test_mov_mem_rsp_offset_produces_mem_output(self, sim: CellSimulator) -> None:
        """
        mov [rsp+8], rax  (48 89 44 24 08)
        Same unique-pointer pattern, different base register.
        """
        shadow = BitPreciseShadowMemory()
        shadow.write_mask(STACK + 8, 0xFFFFFFFFFFFFFFFF, 8)

        output = _eval(
            '4889442408',  # mov [rsp+8], rax
            {},
            {'RSP': STACK, 'RAX': 0xBEEF},
            shadow,
            sim,
        )
        mem = _mem_keys(output)

        assert mem, 'mov [rsp+8], rax produced no MEM_ output. Stale taint at [rsp+8] is never cleared.'


# ===========================================================================
# End-to-end: safe_read scenario — the failing test case
# ===========================================================================


class TestSafeReadScenario:
    """
    Simulates the exact sequence that causes the false-positive BOF:

    1. sys_read taints 8 bytes at buf (e.g. at RBP-0x108)
    2. safe_read returns: leave; ret
    3. After safe_read's ret, _start calls safe_read again (or another func)
       via 'call': pushes return address to [RSP-8]
    4. The new function's ret should fire without BOF

    The bug: step 2's 'call' does not clear shadow at [RSP-8].
    If [RSP-8] was previously tainted (from buf overlap or prior frame reuse),
    the 'ret' fires ImplicitTaintError incorrectly.
    """

    def test_call_clears_stale_buf_taint_from_prior_frame(self, sim: CellSimulator) -> None:
        """
        Simulates buf[256] taint followed by call to another function.

        The return address slot (RSP-8 before call) must be cleared by the
        call instruction, even if it was previously in the same memory region
        that sys_read tainted.
        """
        shadow = BitPreciseShadowMemory()

        # sys_read tainted 8 bytes at buf.
        # In a typical -O0 frame, RSP after 'sub rsp, 0x108' points to buf[0].
        # The return address that 'call' will push is at old_RSP - 8.
        # We simulate this by tainting the slot that 'call' will use.
        buf_addr = STACK  # buf starts at RSP (after sub rsp, N)
        ret_addr_slot = STACK + 0x110 - 8  # [old_RSP - 8] = return address slot

        # Taint buf[0..7]
        shadow.write_mask(buf_addr, 0xFFFFFFFFFFFFFFFF, 8)
        # Also taint ret_addr_slot if it overlaps with buf — the worst case.
        # In practice they don't overlap with buf[256], but with buf[16] they might.
        # We explicitly taint it to test the clearing mechanism.
        shadow.write_mask(ret_addr_slot, 0xFFFFFFFFFFFFFFFF, 8)

        # Now 'call' executes from old_RSP + 0x108 (after leave restores RSP).
        old_rsp = STACK + 0x110
        output = _eval(
            'e800000000',  # call rel32
            {},
            {'RSP': old_rsp, 'RIP': 0x1000},
            shadow,
            sim,
        )
        mem = _mem_keys(output)

        # The slot at old_RSP - 8 must have been cleared
        cleared = any(int(k[4 : k.rfind('_')], 16) == old_rsp - 8 and v == 0 for k, v in mem.items())
        assert cleared, (
            f'call did not clear return-address slot at {hex(old_rsp - 8)}. '
            f'MEM_ outputs: {mem}. '
            'This means the subsequent ret will see stale taint and fire a false BOF.'
        )

    def test_full_safe_read_sequence_no_false_bof(self, sim: CellSimulator) -> None:
        """
        Manually simulate the shadow state through a safe_read() execution.

        safe_read stack layout (buf[256], -O0):
          [rbp+8]   = saved RIP (return to _start)   <- RSP after leave
          [rbp]     = saved old_RBP                   <- pushed by push rbp
          [rbp-8]..[rbp-0x108] = buf[256]

        sys_read taints [rbp-0x108]..[rbp-0x101] (8 bytes = buf[0..7])

        After safe_read returns:
          - RSP points to [rbp+8] = saved RIP (which is a code address, untainted)
          - circuit for ret: shadow[RSP] must be 0
        """
        shadow = BitPreciseShadowMemory()

        # Lay out the stack
        saved_rip_addr = RBP_VAL + 8  # [rbp+8]
        saved_rbp_addr = RBP_VAL  # [rbp]
        buf_start = RBP_VAL - 0x108  # buf[0]

        # 1. 'call safe_read' from _start pushed return address
        # Simulate: shadow at saved_rip_addr is cleared by 'call'
        shadow.write_mask(saved_rip_addr, 0, 8)  # call cleared this

        # 2. push rbp inside safe_read
        # Simulate: shadow at saved_rbp_addr is cleared (rbp is untainted)
        shadow.write_mask(saved_rbp_addr, 0, 8)  # push rbp cleared this

        # 3. sys_read taints 8 bytes at buf
        shadow.write_mask(buf_start, 0xFFFFFFFFFFFFFFFF, 8)

        # 4. leave; ret from safe_read
        # leave: RSP = RBP + 8 (which is saved_rip_addr)
        # ret reads shadow[RSP] = shadow[saved_rip_addr]
        rsp_at_ret = saved_rip_addr
        shadow_at_ret = shadow.read_mask(rsp_at_ret, 8)

        # The critical assertion: saved_rip_addr must be clean
        assert shadow_at_ret == 0, (
            f'shadow[RSP] at ret = {hex(shadow_at_ret)}, expected 0. '
            f'buf is at {hex(buf_start)}, saved RIP at {hex(saved_rip_addr)} '
            f'(distance = {saved_rip_addr - buf_start} bytes). '
            'The return address slot is tainted — this would cause false BOF.'
        )

        # 5. Confirm ret circuit produces no RIP taint with clean shadow
        output = _eval('c3', {}, {'RSP': rsp_at_ret}, shadow, sim)
        rip_taint = output.get('RIP', 0)
        assert rip_taint == 0, f'ret with shadow[RSP]=0 still produced RIP taint {hex(rip_taint)}.'
