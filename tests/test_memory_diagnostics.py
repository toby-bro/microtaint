"""
Diagnostic test suite for microtaint store taint propagation and shadow memory.

Covers:
  1. MEM_ key format confusion (offset vs size in _read_reg)
  2. STORE value vs address dependency separation
  3. push/call/mov-to-mem taint propagation (tainted and untainted)
  4. RSP taint not bleeding into stored values
  5. ret BOF detection (tainted shadow → tainted RIP)
  6. ret false-positive suppression (clean shadow → clean RIP)
  7. Full safe_read stack sequence (no false BOF)
  8. Full unsafe_read stack sequence (real BOF detected)
  9. leave instruction taint propagation
 10. pop rbp from tainted memory
"""

# mypy: disable-error-code="index"

from __future__ import annotations

import pytest

from microtaint.emulator.shadow import BitPreciseShadowMemory
from microtaint.instrumentation.ast import EvalContext, Expr
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, ImplicitTaintPolicy, Register

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STACK = 0x80000000  # RSP base used in all tests
RBP_VAL = STACK + 0x200  # RBP value (above stack base)

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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def sim() -> CellSimulator:
    return CellSimulator(Architecture.AMD64)


def _eval(
    hex_bytes: str,
    reg_taint: dict[str, int],
    reg_values: dict[str, int],
    shadow: BitPreciseShadowMemory,
    sim: CellSimulator,
    *,
    print_circuit: bool = True,
) -> dict[str, int]:
    """
    Evaluate a circuit and return the full output_state dict.
    Always prints the circuit assignments for diagnostic clarity.
    """
    bs = bytes.fromhex(hex_bytes)
    circuit = generate_static_rule(Architecture.AMD64, bs, X64)

    if print_circuit:
        print(f'\n=== Circuit for {hex_bytes} ===')
        for a in circuit.assignments:
            print(f'  {a}')

    ctx = EvalContext(
        input_taint=reg_taint,
        input_values=reg_values,
        simulator=sim,
        implicit_policy=ImplicitTaintPolicy.KEEP,
        shadow_memory=shadow,
        mem_reader=lambda addr, sz: 0,
    )
    result = circuit.evaluate(ctx)

    if print_circuit:
        print(f'--- Output state for {hex_bytes} ---')
        for k, v in sorted(result.items()):
            if v != 0:
                print(f'  {k} = {hex(v)}')
        mem_keys = {k: v for k, v in result.items() if k.startswith('MEM_')}
        if mem_keys:
            print(f'  MEM_ keys: {mem_keys}')
        else:
            print('  (no MEM_ keys with value != 0)')

    return result


def _mem_keys(output_state: dict[str, int]) -> dict[str, int]:
    return {k: v for k, v in output_state.items() if k.startswith('MEM_')}


def _parse_mem_key(key: str) -> tuple[int, int] | None:
    """
    Mirror of wrapper._parse_mem_key.
    Expects: MEM_<hex_addr>_<decimal_size>
    """
    if not key.startswith('MEM_'):
        return None
    body = key[4:]
    last = body.rfind('_')
    if last < 0:
        return None
    try:
        addr = int(body[:last], 16)
        size = int(body[last + 1 :])
        return addr, size
    except ValueError:
        return None


# ===========================================================================
# Section 1: MEM_ key format — address must be a concrete hex address,
#             NOT a register name. Verify _parse_mem_key handles the output.
# ===========================================================================


class TestMemKeyFormat:
    """
    The circuit's output_state MEM_ keys must use concrete hex addresses
    (e.g. MEM_0x7ffff000_8), not register-name keys (e.g. MEM_RBP_8).
    The wrapper's _parse_mem_key only handles hex addresses.

    If the circuit emits MEM_RBP_8, _parse_mem_key returns None and the
    shadow is never updated — taint is silently lost.
    """

    def test_mov_mem_rbp_offset_key_is_hex_address(self, sim: CellSimulator) -> None:
        """
        mov [rbp-8], rax (488945F8): MEM_ key must be a concrete hex address.
        MEM_RBP_... or MEM_RBP_-8_... keys are unusable by the wrapper.
        """
        shadow = BitPreciseShadowMemory()
        output = _eval(
            '488945F8',
            {'RAX': 0xFFFFFFFFFFFFFFFF},
            {'RSP': STACK, 'RBP': RBP_VAL, 'RAX': 0xAAAA},
            shadow,
            sim,
        )
        mem = _mem_keys(output)
        print(f'\n[KEY FORMAT] MEM_ keys: {list(mem.keys())}')

        for key in mem:
            parsed = _parse_mem_key(key)
            assert parsed is not None, (
                f'MEM_ key {key!r} cannot be parsed by wrapper._parse_mem_key. '
                f'Key must be MEM_<hex_addr>_<decimal_size>, not a register name. '
                f'This means shadow_mem.write_mask is never called → taint silently lost.'
            )
            addr, size = parsed
            print(f'  Parsed {key!r} → addr={hex(addr)}, size={size}')
            assert (
                addr == RBP_VAL - 8
            ), f'Parsed address {hex(addr)} != expected {hex(RBP_VAL - 8)}. Offset -8 was lost in the key.'

    def test_push_rbp_mem_key_is_hex_address(self, sim: CellSimulator) -> None:
        """
        push rbp (55): MEM_ key must be concrete hex address = RSP - 8.
        """
        shadow = BitPreciseShadowMemory()
        output = _eval(
            '55',
            {'RBP': 0xFFFFFFFFFFFFFFFF},
            {'RSP': STACK, 'RBP': RBP_VAL},
            shadow,
            sim,
        )
        mem = _mem_keys(output)
        print(f'\n[KEY FORMAT] push rbp MEM_ keys: {list(mem.keys())}')

        expected_addr = STACK - 8
        for key in mem:
            parsed = _parse_mem_key(key)
            assert (
                parsed is not None
            ), f'push rbp MEM_ key {key!r} not parseable. Wrapper cannot update shadow → stale taint persists.'
            addr, size = parsed
            print(f'  Parsed {key!r} → addr={hex(addr)}, size={size}')
            assert (
                addr == expected_addr
            ), f'push rbp wrote to {hex(addr)}, expected {hex(expected_addr)} (RSP-8). Address offset lost.'

    def test_call_mem_key_is_hex_address(self, sim: CellSimulator) -> None:
        """
        call rel32 (e800000000): MEM_ key must be concrete hex address = RSP - 8.
        """
        shadow = BitPreciseShadowMemory()
        output = _eval(
            'e800000000',
            {},
            {'RSP': STACK, 'RIP': 0x1000},
            shadow,
            sim,
        )
        mem = _mem_keys(output)
        print(f'\n[KEY FORMAT] call rel32 MEM_ keys: {list(mem.keys())}')

        assert mem, 'call rel32 produced no MEM_ keys at all.'
        expected_addr = STACK - 8
        for key in mem:
            parsed = _parse_mem_key(key)
            assert parsed is not None, f'call rel32 MEM_ key {key!r} not parseable by wrapper.'
            addr, size = parsed
            print(f'  Parsed {key!r} → addr={hex(addr)}, size={size}')
            assert addr == expected_addr, f'call wrote to {hex(addr)}, expected {hex(expected_addr)}.'

    def test_no_register_name_in_mem_key(self, sim: CellSimulator) -> None:
        """
        No MEM_ key should contain a bare register name like MEM_RBP_8.
        Such keys are not parseable as hex addresses and are silently dropped.
        """
        instructions = {
            'mov [rbp-8], rax': ('488945F8', {'RSP': STACK, 'RBP': RBP_VAL, 'RAX': 0xDEAD}),
            'push rbp': ('55', {'RSP': STACK, 'RBP': RBP_VAL}),
            'call rel32': ('e800000000', {'RSP': STACK, 'RIP': 0x1000}),
            'mov [rsp+8], rax': ('4889442408', {'RSP': STACK, 'RAX': 0xBEEF}),
        }
        for name, (hexb, vals) in instructions.items():
            shadow = BitPreciseShadowMemory()
            output = _eval(hexb, {}, vals, shadow, sim)
            for key in _mem_keys(output):
                body = key[4:]  # strip MEM_
                # If the part before the last _ starts with a letter, it's a register name
                last = body.rfind('_')
                addr_part = body[:last] if last >= 0 else body
                assert addr_part.startswith('0x'), (
                    f'[{name}] MEM_ key {key!r} has non-hex address part {addr_part!r}. '
                    f'This is a register name — wrapper cannot parse it. '
                    f'Fix: generate_output_target must produce concrete hex addresses.'
                )


# ===========================================================================
# Section 2: STORE value vs address — RSP taint must NOT bleed into
#            the value being stored.
# ===========================================================================


class TestStoreValueVsAddressTaint:
    """
    For STORE instructions, the memory taint must equal the taint of the
    VALUE being stored — not the ADDRESS register used to compute the target.

    push rbp: stores RBP at [RSP-8].
      - Memory taint = T_RBP  (value)
      - NOT T_RSP              (address base — this is address, not value)

    mov [rbp-8], rax: stores RAX at [RBP-8].
      - Memory taint = T_RAX
      - NOT T_RBP

    If address-register taint bleeds into memory taint, then tainting RSP
    (e.g. via 'leave: mov rsp, rbp' after overflow) will cause ALL subsequent
    pushes to write tainted shadow → cascade of false positives.
    """

    def test_push_rbp_tainted_rsp_does_not_taint_memory(self, sim: CellSimulator) -> None:
        """
        push rbp with tainted RSP but clean RBP → memory taint must be 0.
        RSP is the address register, not the value. Tainted address should
        cause AIW (arbitrary indexed write), not taint the stored value.
        """
        shadow = BitPreciseShadowMemory()
        output = _eval(
            '55',
            {'RSP': 0xFFFFFFFFFFFFFFFF},  # RSP tainted, RBP clean
            {'RSP': STACK, 'RBP': RBP_VAL},
            shadow,
            sim,
        )
        mem = _mem_keys(output)
        print(f'\n[VALUE vs ADDR] push rbp, tainted RSP: MEM_={mem}')

        tainted_stores = {k: v for k, v in mem.items() if v != 0}
        assert not tainted_stores, (
            f'push rbp with clean RBP but tainted RSP produced nonzero MEM_ taint: {tainted_stores}. '
            f'RSP is the address register — its taint must NOT bleed into stored value taint. '
            f'This causes false-positive BOF: every subsequent push writes tainted shadow '
            f'after RSP gets tainted by leave (mov rsp, rbp after overflow).'
        )

    def test_push_rbp_tainted_rbp_does_taint_memory(self, sim: CellSimulator) -> None:
        """
        push rbp with tainted RBP → memory taint must be nonzero.
        RBP is the value being stored.
        """
        shadow = BitPreciseShadowMemory()
        output = _eval(
            '55',
            {'RBP': 0xFFFFFFFFFFFFFFFF},  # RBP tainted
            {'RSP': STACK, 'RBP': RBP_VAL},
            shadow,
            sim,
        )
        mem = _mem_keys(output)
        print(f'\n[VALUE vs ADDR] push rbp, tainted RBP: MEM_={mem}')

        tainted_stores = {k: v for k, v in mem.items() if v != 0}
        assert tainted_stores, (
            'push rbp with tainted RBP produced no MEM_ taint. '
            'The stored value (RBP) is tainted — shadow must reflect this.'
        )

    def test_mov_mem_rbp_tainted_rbp_does_not_taint_value(self, sim: CellSimulator) -> None:
        """
        mov [rbp-8], rax with tainted RBP but clean RAX → memory taint must be 0.
        RBP is the address register, not the stored value.
        """
        shadow = BitPreciseShadowMemory()
        output = _eval(
            '488945F8',
            {'RBP': 0xFFFFFFFFFFFFFFFF},  # RBP tainted (address reg), RAX clean
            {'RSP': STACK, 'RBP': RBP_VAL, 'RAX': 0x1234},
            shadow,
            sim,
        )
        mem = _mem_keys(output)
        print(f'\n[VALUE vs ADDR] mov [rbp-8], rax, tainted RBP: MEM_={mem}')

        # With clean RAX, value taint = 0. Tainted RBP is the address → AIW territory,
        # not a value taint. Memory shadow at the written slot must be 0.
        # (AIW detection is a separate concern in the wrapper, not shadow taint.)
        tainted_stores = {k: v for k, v in mem.items() if v != 0}
        assert not tainted_stores, (
            f'mov [rbp-8], rax with clean RAX but tainted RBP produced nonzero MEM_ taint: {tainted_stores}. '
            f'RBP is the address register — its taint must NOT pollute the stored value taint.'
        )

    def test_mov_mem_rbp_tainted_rax_does_taint_value(self, sim: CellSimulator) -> None:
        """
        mov [rbp-8], rax with tainted RAX → memory taint must be nonzero.
        """
        shadow = BitPreciseShadowMemory()
        output = _eval(
            '488945F8',
            {'RAX': 0xFFFFFFFFFFFFFFFF},  # RAX tainted (the value)
            {'RSP': STACK, 'RBP': RBP_VAL, 'RAX': 0xAAAA},
            shadow,
            sim,
        )
        mem = _mem_keys(output)
        print(f'\n[VALUE vs ADDR] mov [rbp-8], rax, tainted RAX: MEM_={mem}')

        tainted_stores = {k: v for k, v in mem.items() if v != 0}
        assert tainted_stores, (
            'mov [rbp-8], rax with tainted RAX produced no MEM_ taint. '
            'RAX is the stored value — shadow must be tainted.'
        )

    def test_both_tainted_rsp_and_rbp_push(self, sim: CellSimulator) -> None:
        """
        push rbp with both RSP and RBP tainted.
        Memory taint must equal T_RBP only — not OR of T_RSP and T_RBP.
        (RSP taint → AIW detection, separate from value taint.)
        """
        shadow = BitPreciseShadowMemory()
        output = _eval(
            '55',
            {'RSP': 0xFFFFFFFFFFFFFFFF, 'RBP': 0x00000000000000FF},
            {'RSP': STACK, 'RBP': RBP_VAL},
            shadow,
            sim,
        )
        mem = _mem_keys(output)
        print(f'\n[VALUE vs ADDR] push rbp, both tainted: MEM_={mem}')

        for key, val in mem.items():
            parsed = _parse_mem_key(key)
            if parsed is None:
                continue
            addr, size = parsed
            if addr == STACK - 8:
                # Expected: only low byte tainted (T_RBP = 0xFF)
                # Wrong:    all bytes tainted (T_RBP | T_RSP = 0xFF...FF)
                assert val == 0xFF, (
                    f'Memory taint at pushed slot = {hex(val)}, expected 0xFF (only T_RBP). '
                    f'If val == 0xFFFFFFFFFFFFFFFF, T_RSP is bleeding into value taint. '
                    f'RSP is the address — its taint must NOT appear in the stored value taint.'
                )


# ===========================================================================
# Section 3: Correct taint propagation for all store variants
# ===========================================================================


class TestStoreVariantsPropagation:
    """
    Verify each store instruction form propagates taint correctly.
    """

    def test_push_rbp_clean_clears_shadow(self, sim: CellSimulator) -> None:
        """push rbp with clean RBP clears stale shadow at [RSP-8]."""
        shadow = BitPreciseShadowMemory()
        shadow.write_mask(STACK - 8, 0xFFFFFFFFFFFFFFFF, 8)  # stale taint

        output = _eval('55', {}, {'RSP': STACK, 'RBP': RBP_VAL}, shadow, sim)
        mem = _mem_keys(output)

        slot = STACK - 8
        found = any(_parse_mem_key(k) is not None and _parse_mem_key(k)[0] == slot and v == 0 for k, v in mem.items())
        assert found, (
            f'push rbp (clean RBP) did not produce MEM_=0 at {hex(slot)}. '
            f'Stale shadow taint will not be cleared. MEM_ keys: {mem}'
        )

    def test_push_rbp_tainted_sets_shadow(self, sim: CellSimulator) -> None:
        """push rbp with tainted RBP sets shadow at [RSP-8]."""
        shadow = BitPreciseShadowMemory()
        output = _eval(
            '55',
            {'RBP': 0xFFFFFFFFFFFFFFFF},
            {'RSP': STACK, 'RBP': RBP_VAL},
            shadow,
            sim,
        )
        mem = _mem_keys(output)
        slot = STACK - 8
        found = any(_parse_mem_key(k) is not None and _parse_mem_key(k)[0] == slot and v != 0 for k, v in mem.items())
        assert found, f'push rbp (tainted) did not set MEM_ taint. MEM_: {mem}'

    def test_call_clears_return_address_slot(self, sim: CellSimulator) -> None:
        """
        call rel32 (e800000000): return address is a constant code pointer, untainted.
        Shadow at [RSP-8] must be cleared even if it was previously tainted.
        """
        shadow = BitPreciseShadowMemory()
        shadow.write_mask(STACK - 8, 0xFFFFFFFFFFFFFFFF, 8)  # stale taint

        output = _eval(
            'e800000000',
            {},
            {'RSP': STACK, 'RIP': 0x1000},
            shadow,
            sim,
        )
        mem = _mem_keys(output)
        slot = STACK - 8
        found = any(_parse_mem_key(k) is not None and _parse_mem_key(k)[0] == slot and v == 0 for k, v in mem.items())
        assert found, (
            f'call rel32 did not clear shadow at return address slot {hex(slot)}. '
            f'Stale taint causes false-positive BOF at ret. MEM_: {mem}'
        )

    def test_mov_mem_rsp_offset_clean(self, sim: CellSimulator) -> None:
        """mov [rsp+8], rax (4889442408) with clean RAX clears shadow."""
        shadow = BitPreciseShadowMemory()
        shadow.write_mask(STACK + 8, 0xFFFFFFFFFFFFFFFF, 8)

        output = _eval(
            '4889442408',
            {},
            {'RSP': STACK, 'RAX': 0xBEEF},
            shadow,
            sim,
        )
        mem = _mem_keys(output)
        slot = STACK + 8
        found = any(_parse_mem_key(k) is not None and _parse_mem_key(k)[0] == slot and v == 0 for k, v in mem.items())
        assert found, f'mov [rsp+8], rax (clean) did not clear shadow at {hex(slot)}. MEM_: {mem}'

    def test_mov_mem_rsp_offset_tainted(self, sim: CellSimulator) -> None:
        """mov [rsp+8], rax with tainted RAX sets shadow."""
        shadow = BitPreciseShadowMemory()
        output = _eval(
            '4889442408',
            {'RAX': 0xFFFFFFFFFFFFFFFF},
            {'RSP': STACK, 'RAX': 0xBEEF},
            shadow,
            sim,
        )
        mem = _mem_keys(output)
        slot = STACK + 8
        found = any(_parse_mem_key(k) is not None and _parse_mem_key(k)[0] == slot and v != 0 for k, v in mem.items())
        assert found, f'mov [rsp+8], rax (tainted) did not set MEM_ taint. MEM_: {mem}'

    def test_mov_mem_rbp_minus8_clean_clears(self, sim: CellSimulator) -> None:
        """mov [rbp-8], rax (488945F8) with clean RAX clears shadow."""
        shadow = BitPreciseShadowMemory()
        shadow.write_mask(RBP_VAL - 8, 0xFFFFFFFFFFFFFFFF, 8)

        output = _eval(
            '488945F8',
            {},
            {'RSP': STACK, 'RBP': RBP_VAL, 'RAX': 0},
            shadow,
            sim,
        )
        mem = _mem_keys(output)
        slot = RBP_VAL - 8
        found = any(_parse_mem_key(k) is not None and _parse_mem_key(k)[0] == slot and v == 0 for k, v in mem.items())
        assert found, f'mov [rbp-8], rax (clean) did not clear shadow at {hex(slot)}. MEM_: {mem}'


# ===========================================================================
# Section 4: ret BOF detection — shadow at [RSP] controls RIP taint
# ===========================================================================


class TestRetBOFDetection:
    """
    ret (c3) lifts as: RIP = LOAD [RSP]; RSP = RSP + 8.
    The RIP taint comes entirely from shadow memory at [RSP].
    """

    def test_ret_clean_shadow_no_rip_taint(self, sim: CellSimulator) -> None:
        """
        ret with clean shadow at [RSP] → RIP must be untainted.
        This is the false-positive test: safe_read must not trigger BOF.
        """
        shadow = BitPreciseShadowMemory()
        # Shadow at RSP is clean (call cleared it, return address is code)

        output = _eval('c3', {}, {'RSP': STACK}, shadow, sim)
        rip_taint = output.get('RIP', 0)
        print(f'\n[RET] clean shadow → RIP taint = {hex(rip_taint)}')

        assert rip_taint == 0, (
            f'ret with clean shadow[RSP] produced RIP taint {hex(rip_taint)}. '
            f'This is a false-positive BOF on safe functions.'
        )

    def test_ret_tainted_shadow_sets_rip_taint(self, sim: CellSimulator) -> None:
        """
        ret with tainted shadow at [RSP] → RIP must be tainted (real BOF).
        """
        shadow = BitPreciseShadowMemory()
        shadow.write_mask(STACK, 0xFFFFFFFFFFFFFFFF, 8)  # saved RIP is tainted

        output = _eval('c3', {}, {'RSP': STACK}, shadow, sim)
        rip_taint = output.get('RIP', 0)
        print(f'\n[RET] tainted shadow → RIP taint = {hex(rip_taint)}')

        assert (
            rip_taint != 0
        ), 'ret with tainted shadow[RSP] produced RIP taint = 0. Real BOF not detected — taint lost at ret.'

    def test_ret_partially_tainted_shadow(self, sim: CellSimulator) -> None:
        """
        ret with partially tainted shadow (only low byte tainted) → RIP tainted.
        Even one tainted byte in the return address means BOF.
        """
        shadow = BitPreciseShadowMemory()
        shadow.write_mask(STACK, 0xFF, 1)  # only byte 0 tainted

        output = _eval('c3', {}, {'RSP': STACK}, shadow, sim)
        rip_taint = output.get('RIP', 0)
        print(f'\n[RET] partial shadow taint → RIP taint = {hex(rip_taint)}')

        assert rip_taint != 0, (
            'ret with partially tainted shadow (byte 0) produced RIP taint = 0. '
            'BOF not detected for partial overflow.'
        )

    def test_ret_taint_is_load_like_not_differential(self, sim: CellSimulator) -> None:
        """
        ret's RIP output uses the load-like path (reads taint from shadow directly),
        not the differential (which would XOR two Unicorn runs — wrong for a load).
        Verify by checking that circuit has a MemoryOperand in RIP's expression.
        """
        from microtaint.instrumentation.ast import AvalancheExpr, BinaryExpr, MemoryOperand

        bs = bytes.fromhex('c3')
        circuit = generate_static_rule(Architecture.AMD64, bs, X64)

        print('\n[RET CIRCUIT] Full assignments:')
        for a in circuit.assignments:
            print(f'  {a}')

        rip_assignment = next(
            (a for a in circuit.assignments if hasattr(a.target, 'name') and a.target.name == 'RIP'),
            None,
        )
        assert rip_assignment is not None, 'No RIP assignment in ret circuit.'

        def has_memory_operand(expr: Expr) -> bool:
            if isinstance(expr, MemoryOperand):
                return True
            if isinstance(expr, (AvalancheExpr,)):
                return has_memory_operand(expr.expr)
            if isinstance(expr, BinaryExpr):
                return has_memory_operand(expr.lhs) or has_memory_operand(expr.rhs)
            return False

        print(f'  RIP expression: {rip_assignment.expression}')
        assert has_memory_operand(rip_assignment.expression), (  # type: ignore[arg-type]
            'ret RIP assignment does not contain a MemoryOperand. '
            'The load-like path was not taken — RIP taint reads from differential '
            'instead of shadow memory. Shadow taint will be ignored.'
        )


# ===========================================================================
# Section 5: leave instruction — mov rsp, rbp + pop rbp
# ===========================================================================


class TestLeaveInstruction:
    """
    leave (C9) = mov rsp, rbp; pop rbp.
    SLEIGH: RSP = RBP; RBP = LOAD[RSP]; RSP = RSP + 8.

    Critical: the RSP=RBP assignment propagates T_RBP → T_RSP.
    After overflow: T_RBP gets set from pop (LOAD of tainted saved_rbp).
    This then propagates T_RSP = T_RBP.
    Any subsequent push with tainted RSP must NOT taint stored values.
    """

    def test_leave_propagates_rbp_taint_to_rsp(self, sim: CellSimulator) -> None:
        """
        leave with tainted RBP → RSP must also be tainted (mov rsp, rbp).
        """
        shadow = BitPreciseShadowMemory()
        # saved_rbp (at [RBP]) is clean for this test
        shadow.write_mask(RBP_VAL, 0, 8)

        output = _eval(
            'c9',
            {'RBP': 0xFFFFFFFFFFFFFFFF},  # RBP tainted
            {'RSP': STACK, 'RBP': RBP_VAL},
            shadow,
            sim,
        )
        rsp_taint = output.get('RSP', 0)
        print(f'\n[LEAVE] tainted RBP → RSP taint = {hex(rsp_taint)}')

        assert (
            rsp_taint != 0
        ), 'leave with tainted RBP produced no RSP taint. mov rsp, rbp must propagate T_RBP → T_RSP.'

    def test_leave_pop_from_tainted_shadow_taints_rbp(self, sim: CellSimulator) -> None:
        """
        leave with tainted shadow at [RBP] (saved_rbp slot) → RBP must be tainted.
        This is 'pop rbp' reading from tainted memory.
        """
        shadow = BitPreciseShadowMemory()
        shadow.write_mask(RBP_VAL, 0xFFFFFFFFFFFFFFFF, 8)  # saved_rbp tainted

        output = _eval(
            'c9',
            {},  # no register taint
            {'RSP': STACK, 'RBP': RBP_VAL},
            shadow,
            sim,
        )
        rbp_taint = output.get('RBP', 0)
        print(f'\n[LEAVE] tainted shadow[RBP] → RBP taint = {hex(rbp_taint)}')

        assert rbp_taint != 0, (
            'leave with tainted shadow at [RBP] (saved_rbp) produced no RBP taint. '
            'pop rbp must read taint from shadow memory.'
        )

    def test_leave_clean_shadow_no_spurious_taint(self, sim: CellSimulator) -> None:
        """
        leave with completely clean state → no register taint output.
        """
        shadow = BitPreciseShadowMemory()

        output = _eval(
            'c9',
            {},
            {'RSP': STACK, 'RBP': RBP_VAL},
            shadow,
            sim,
        )
        non_mem = {k: v for k, v in output.items() if not k.startswith('MEM_') and v != 0}
        # RIP is always cleared by KEEP policy — ignore it
        non_mem.pop('RIP', None)
        print(f'\n[LEAVE] clean state → non-zero reg taints: {non_mem}')

        assert not non_mem, (
            f'leave with clean state produced spurious register taint: {non_mem}. '
            f'This indicates taint is being generated from nothing.'
        )


# ===========================================================================
# Section 6: Full stack simulation — safe_read (no BOF) and unsafe_read (BOF)
# ===========================================================================


class TestFullStackSequence:
    """
    Simulate the exact shadow state transitions for a function call sequence,
    verifying that:
      - safe reads (buf smaller than frame) don't trigger BOF
      - unsafe reads (overflow reaches saved RIP) do trigger BOF
    """

    def _simulate_frame(
        self,
        sim: CellSimulator,
        buf_size_bytes: int,
        read_size_bytes: int,
    ) -> tuple[int, BitPreciseShadowMemory]:
        """
        Simulate stack setup + sys_read + epilogue shadow state.

        Stack layout (RBP-relative, 0-based offsets from buf):
          [caller_RSP - 8] = saved_RIP   (pushed by call)
          [caller_RSP - 16] = saved_RBP  (pushed by push rbp)
          [caller_RSP - 16 - buf_size] = buf[0]  (after sub rsp, buf_size)

        Returns (rsp_at_ret, shadow) after leave.
        rsp_at_ret points to saved_RIP slot.
        """
        caller_rsp = STACK + 0x400

        # call: saves return address at [caller_RSP - 8]
        saved_rip_addr = caller_rsp - 8
        saved_rbp_addr = caller_rsp - 16
        buf_start = caller_rsp - 16 - buf_size_bytes

        rbp_in_frame = caller_rsp - 16  # RBP = RSP after push rbp + mov rbp, rsp

        shadow = BitPreciseShadowMemory()

        # Step 1: call — clears saved_rip slot (return addr is code, untainted)
        shadow.write_mask(saved_rip_addr, 0, 8)

        # Step 2: push rbp — clears saved_rbp slot (RBP is untainted before call)
        shadow.write_mask(saved_rbp_addr, 0, 8)

        # Step 3: sys_read taints read_size_bytes at buf
        if read_size_bytes > 0:
            taint_mask = (1 << (min(read_size_bytes, 8) * 8)) - 1
            for chunk_start in range(0, read_size_bytes, 8):
                chunk_size = min(8, read_size_bytes - chunk_start)
                chunk_taint = (1 << (chunk_size * 8)) - 1
                shadow.write_mask(buf_start + chunk_start, chunk_taint, chunk_size)

        # Step 4: leave → RSP = RBP; RBP = pop [RBP]; RSP += 8
        # After leave: RSP points to saved_rip_addr
        rsp_at_ret = saved_rip_addr

        print(f'\n[FRAME] buf_size={buf_size_bytes}, read={read_size_bytes}')
        print(f'  buf_start={hex(buf_start)}, saved_rbp={hex(saved_rbp_addr)}, saved_rip={hex(saved_rip_addr)}')
        print(f'  shadow[saved_rip] = {hex(shadow.read_mask(saved_rip_addr, 8))}')
        print(f'  shadow[saved_rbp] = {hex(shadow.read_mask(saved_rbp_addr, 8))}')
        overflow_bytes = max(0, read_size_bytes - buf_size_bytes)
        print(f'  overflow_bytes = {overflow_bytes}')
        if overflow_bytes > 0:
            print(
                f'  shadow[buf_start + {buf_size_bytes}..] (saved_rbp region) = '
                f'{hex(shadow.read_mask(saved_rbp_addr, 8))}',
            )

        return rsp_at_ret, shadow

    def test_safe_read_no_bof(self, sim: CellSimulator) -> None:
        """
        buf[256], read 8 bytes: saved_RIP is 264 bytes past buf[0].
        Shadow at saved_RIP must be 0 → ret sees clean RIP → no BOF.
        """
        rsp_at_ret, shadow = self._simulate_frame(sim, buf_size_bytes=256, read_size_bytes=8)

        # Check shadow at saved_rip is clean
        shadow_at_ret = shadow.read_mask(rsp_at_ret, 8)
        assert shadow_at_ret == 0, (
            f'Safe read (buf[256], read 8): shadow[saved_RIP] = {hex(shadow_at_ret)}, expected 0. '
            f'False positive BOF would fire.'
        )

        # Confirm ret sees no RIP taint
        output = _eval('c3', {}, {'RSP': rsp_at_ret}, shadow, sim)
        rip_taint = output.get('RIP', 0)
        assert rip_taint == 0, f'Safe read: ret produced RIP taint {hex(rip_taint)} with clean shadow. False positive.'

    def test_unsafe_read_bof_detected(self, sim: CellSimulator) -> None:
        """
        buf[16], read 32 bytes: overflow covers saved_RBP (bytes 16-23)
        and saved_RIP (bytes 24-31).
        Shadow at saved_RIP must be nonzero → ret sees tainted RIP → BOF.
        """
        rsp_at_ret, shadow = self._simulate_frame(sim, buf_size_bytes=16, read_size_bytes=32)

        shadow_at_ret = shadow.read_mask(rsp_at_ret, 8)
        assert shadow_at_ret != 0, (
            'Unsafe read (buf[16], read 32): shadow[saved_RIP] = 0. '
            'BOF would not be detected — overflow into saved_RIP not reflected in shadow.'
        )

        output = _eval('c3', {}, {'RSP': rsp_at_ret}, shadow, sim)
        rip_taint = output.get('RIP', 0)
        assert (
            rip_taint != 0
        ), 'Unsafe read: ret produced RIP taint = 0 despite tainted shadow[saved_RIP]. BOF not detected.'

    def test_exact_boundary_safe(self, sim: CellSimulator) -> None:
        """
        buf[16], read exactly 16 bytes: fills buf but does NOT overflow.
        saved_RIP must remain clean.
        """
        rsp_at_ret, shadow = self._simulate_frame(sim, buf_size_bytes=16, read_size_bytes=16)

        shadow_at_ret = shadow.read_mask(rsp_at_ret, 8)
        assert shadow_at_ret == 0, (
            f'Exact-boundary read (buf[16], read 16): shadow[saved_RIP] = {hex(shadow_at_ret)}. '
            f'No overflow occurred — saved_RIP must be clean.'
        )

    def test_one_byte_overflow_detected(self, sim: CellSimulator) -> None:
        """
        buf[16], read 17 bytes: one byte into saved_RBP region.
        saved_RBP gets tainted but saved_RIP may not — depends on exact layout.
        At minimum, saved_RBP must show taint.
        """
        caller_rsp = STACK + 0x400
        saved_rbp_addr = caller_rsp - 16
        buf_start = caller_rsp - 16 - 16

        shadow = BitPreciseShadowMemory()
        shadow.write_mask(caller_rsp - 8, 0, 8)  # call cleared saved_rip
        shadow.write_mask(saved_rbp_addr, 0, 8)  # push rbp cleared saved_rbp

        # sys_read taints 17 bytes: buf[0..15] + 1 byte into saved_rbp
        for i in range(17):
            shadow.write_mask(buf_start + i, 0xFF, 1)

        shadow_rbp = shadow.read_mask(saved_rbp_addr, 8)
        print(f'\n[BOUNDARY] 1-byte overflow: shadow[saved_rbp] = {hex(shadow_rbp)}')

        assert shadow_rbp != 0, (
            '1-byte overflow into saved_RBP region: shadow[saved_rbp] = 0. '
            'The overflow byte was not recorded in shadow.'
        )


# ===========================================================================
# Section 7: Cascading taint — RSP taint after leave must not cause false BOF
# ===========================================================================


class TestRSPTaintCascade:
    """
    After 'leave': RSP = RBP (inherits T_RBP).
    If T_RBP is nonzero (e.g. overflow reached saved_RBP), then T_RSP is nonzero.
    The NEXT instruction might be 'call' which does push [RSP-8].
    With T_RSP nonzero, if STORE shortcut includes T_RSP in value taint,
    then [RSP-8] gets tainted → the callee's ret sees tainted shadow → false BOF.

    This test verifies that T_RSP does NOT bleed into the stored value of push/call.
    """

    def test_push_with_tainted_rsp_does_not_poison_stack(self, sim: CellSimulator) -> None:
        """
        Simulate post-leave state where RSP is tainted (from tainted RBP).
        push rbp with tainted RSP but clean RBP → [RSP-8] must NOT be tainted.
        """
        shadow = BitPreciseShadowMemory()
        # RSP is tainted (e.g. leave propagated T_RBP to T_RSP)
        # RBP is clean (freshly restored from saved_rbp which was untainted)
        output = _eval(
            '55',
            {'RSP': 0xFFFFFFFFFFFFFFFF},  # T_RSP from cascaded leave
            {'RSP': STACK, 'RBP': 0},  # RBP restored to 0 (clean)
            shadow,
            sim,
        )
        mem = _mem_keys(output)
        tainted = {k: v for k, v in mem.items() if v != 0}
        print(f'\n[CASCADE] push rbp, tainted RSP, clean RBP → MEM_ tainted: {tainted}')

        assert not tainted, (
            f'push rbp with tainted RSP (from leave cascade) produced nonzero MEM_ taint: {tainted}. '
            f'RSP taint bleeds into stored-value taint → false BOF cascade: '
            f'every subsequent function call will leave tainted shadow on the stack.'
        )

    def test_call_with_tainted_rsp_does_not_poison_return_slot(self, sim: CellSimulator) -> None:
        """
        call rel32 with tainted RSP → return address slot must NOT be tainted.
        The return address is a code constant, always untainted.
        """
        shadow = BitPreciseShadowMemory()
        output = _eval(
            'e800000000',
            {'RSP': 0xFFFFFFFFFFFFFFFF},  # T_RSP
            {'RSP': STACK, 'RIP': 0x1000},
            shadow,
            sim,
        )
        mem = _mem_keys(output)
        tainted = {k: v for k, v in mem.items() if v != 0}
        print(f'\n[CASCADE] call rel32, tainted RSP → MEM_ tainted: {tainted}')

        # The value stored is inst_next (a constant) — always untainted
        # Even with tainted RSP (address), the VALUE taint must be 0
        assert not tainted, (
            f'call rel32 with tainted RSP produced nonzero MEM_ taint: {tainted}. '
            f'Return address is a code constant — its shadow must be 0 regardless of RSP taint.'
        )

    def test_ret_after_cascaded_taint_correct_result(self, sim: CellSimulator) -> None:
        """
        After RSP cascade: simulate that shadow at [RSP] is CLEAN
        (the return address was placed by call, which cleared it).
        ret must see clean shadow → no BOF.
        """
        shadow = BitPreciseShadowMemory()
        # Even with RSP register tainted, what matters is shadow memory content
        shadow.write_mask(STACK, 0, 8)  # clean return address

        output = _eval(
            'c3',
            {'RSP': 0xFFFFFFFFFFFFFFFF},  # T_RSP tainted (cascaded)
            {'RSP': STACK},
            shadow,
            sim,
        )
        rip_taint = output.get('RIP', 0)
        print(f'\n[CASCADE] ret, tainted RSP reg, clean shadow → RIP taint = {hex(rip_taint)}')

        # RIP taint should come from shadow[RSP], not from T_RSP register taint
        # If shadow is clean, RIP should be untainted
        assert rip_taint == 0, (
            f'ret with tainted RSP register but clean shadow[RSP] produced RIP taint {hex(rip_taint)}. '
            f'T_RSP (address taint) must not propagate to RIP — only shadow[RSP] matters.'
        )


# ===========================================================================
# Section 8: _read_reg format — verify simulator reads correct address
# ===========================================================================


class TestSimulatorMemKeyParsing:
    """
    Verify that the simulator correctly handles MEM_ key formats
    when reading back the differential output for STORE instructions.

    The key format MEM_<reg>_<offset> (new) must not be confused with
    MEM_<reg>_<size> (old). If -8 is treated as a size, we read 8 bytes
    at RSP instead of 8 bytes at RSP-8 — wrong address.

    We test this indirectly: if the circuit for 'push rbp' produces
    correct MEM_ taint when RBP is tainted but not when RSP is tainted,
    the simulator is reading the right address.
    """

    def test_differential_reads_correct_address_for_push(self, sim: CellSimulator) -> None:
        """
        For push rbp, the differential (if used) must read memory at RSP-8,
        NOT at RSP. Verify by checking the MEM_ key address in circuit output.
        """
        shadow = BitPreciseShadowMemory()
        rbp_taint = 0xFF  # only low byte tainted

        output = _eval(
            '55',
            {'RBP': rbp_taint},
            {'RSP': STACK, 'RBP': 0x1234567890ABCDEF},
            shadow,
            sim,
        )
        mem = _mem_keys(output)
        print(f'\n[ADDR CHECK] push rbp MEM_ keys: {mem}')

        # The written address must be RSP-8 = STACK-8, not RSP = STACK
        for key, val in mem.items():
            parsed = _parse_mem_key(key)
            if parsed is None:
                continue
            addr, size = parsed
            assert addr != STACK, (
                f'push rbp wrote to {hex(addr)} = RSP (STACK), not RSP-8. '
                f'Offset -8 was lost — simulator is reading at the base register address. '
                f'This is the MEM_RSP vs MEM_RSP_-8 confusion.'
            )
            assert addr == STACK - 8, f'push rbp wrote to {hex(addr)}, expected {hex(STACK-8)} (RSP-8).'

    def test_differential_reads_correct_address_for_mov_rbp_minus8(self, sim: CellSimulator) -> None:
        """
        For mov [rbp-8], rax, writes must go to RBP-8, not RBP.
        """
        shadow = BitPreciseShadowMemory()
        output = _eval(
            '488945F8',
            {'RAX': 0xFF},
            {'RSP': STACK, 'RBP': RBP_VAL, 'RAX': 0xAAAA},
            shadow,
            sim,
        )
        mem = _mem_keys(output)
        print(f'\n[ADDR CHECK] mov [rbp-8], rax MEM_ keys: {mem}')

        for key, val in mem.items():
            parsed = _parse_mem_key(key)
            if parsed is None:
                continue
            addr, size = parsed
            assert addr != RBP_VAL, (
                f'mov [rbp-8], rax wrote to {hex(addr)} = RBP, not RBP-8. '
                f'The -8 offset was dropped. MEM_RBP vs MEM_RBP_-8 confusion.'
            )
            assert addr == RBP_VAL - 8, f'mov [rbp-8], rax wrote to {hex(addr)}, expected {hex(RBP_VAL-8)}.'

    def test_mov_rsp_plus8_correct_address(self, sim: CellSimulator) -> None:
        """
        mov [rsp+8], rax writes to RSP+8, not RSP.
        Positive offset must not be confused with size either.
        """
        shadow = BitPreciseShadowMemory()
        output = _eval(
            '4889442408',
            {'RAX': 0xFF},
            {'RSP': STACK, 'RAX': 0xBEEF},
            shadow,
            sim,
        )
        mem = _mem_keys(output)
        print(f'\n[ADDR CHECK] mov [rsp+8], rax MEM_ keys: {mem}')

        for key, val in mem.items():
            parsed = _parse_mem_key(key)
            if parsed is None:
                continue
            addr, size = parsed
            assert addr != STACK, f'mov [rsp+8], rax wrote to {hex(addr)} = RSP, not RSP+8. +8 offset dropped.'
            assert addr == STACK + 8, f'mov [rsp+8], rax wrote to {hex(addr)}, expected {hex(STACK+8)}.'
