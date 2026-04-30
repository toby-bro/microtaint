"""
test_regressions.py
===================
Regression tests for the bugs uncovered while diagnosing the SipHash
avalanche failure.  Each test class is keyed to one specific bug and
exists to fail loudly if the underlying behaviour ever regresses.

Bug index
---------
1. RMW memory destinations skip the differential
   - Symptom: `add/sub/and/or/xor [mem], reg` produced popcount=1 (just
     OR-of-input-taints) instead of the carry/borrow ripple captured by
     the proper differential.
   - Root cause: `is_store_target` early-return in
     `generate_taint_assignments` treated all memory targets as pure
     stores.  Pure stores HAVE no LOAD in slice_ops; RMW does.
   - Fix: detect RMW (`is_store_target` AND any LOAD op in slice_ops),
     route through `MemoryDifferentialExpr`.

2. Memory inputs with non-zero offsets land at the wrong address
   - Symptom: `add rax, [rbp-0x10]` produced popcount=1 even though the
     memory at `[rbp-0x10]` was bit-tainted.
   - Root cause: `process_dependencies` built input keys as
     `MEM_<reg>` (no offset, no size).  At evaluation,
     `_build_machine_state` resolved the address to
     `input_values[<reg>]`, ignoring the offset entirely; the value was
     stored at `mem[reg]` while the simulator dereferenced
     `mem[reg + offset]` and got zero.
   - Fix: use `MEM_<reg>_<offset>_<size>` format end-to-end; both
     ast.pyx `_build_machine_state` and cell.pyx `_load`/`_read_output`
     now parse it natively.

3. Address-only registers missing from simulator state
   - Symptom: `add rdx, [rax]` (offset=0) where `RAX` is only an address
     also produced popcount=1.
   - Root cause: `process_dependencies` put address registers into
     `mem_groups` (key `MEM_<reg>`) but did NOT add a separate entry
     under bare `<reg>` for `state.regs`.  The simulator cleared all
     registers and ran the instruction with `RAX=0`, reading from
     `mem[0]` instead of `mem[<RAX>]`.
   - Fix: `MemoryDifferentialExpr` accepts an explicit
     `addr_only_regs` list and populates them as concrete (V, V) in
     both `or_inputs` and `and_inputs`.

4. cell.pyx _read_output cannot parse dynamic MEM keys
   - Symptom: even with engine emitting correct cell out_reg names
     like `MEM_RBP_-16_8`, the cell.pyx native path returned 0.
   - Root cause: `_read_output` attempted `int(body[:sep], 16)` on
     `'RBP_-16'`, raising ValueError, returning 0 silently.
   - Fix: extended `_read_output` to also parse
     `MEM_<reg>_<offset>_<size>`, resolving via `_read_reg`.

5. cell.pyx _load cannot parse dynamic MEM keys (input side)
   - Symptom: Memory inputs in dynamic format never reached the
     simulator's frame; they were silently dropped.
   - Root cause: same as bug 4 but on the input side of `_load`.
   - Fix: `_load` does a two-pass build (registers first, memory
     second) so dynamic MEM keys can resolve via the frame's
     already-loaded register state.

Run with:
    uv run pytest test_regressions.py -v
"""

# ruff: noqa: ARG001, ARG002, PLC0415, S110, S607, PLW1510, S603, ARG005
# mypy: disable-error-code="no-untyped-def,no-untyped-call,type-arg,arg-type"

from __future__ import annotations

import pytest

from microtaint.emulator.shadow import BitPreciseShadowMemory
from microtaint.instrumentation.ast import (
    EvalContext,
    MemoryDifferentialExpr,
)
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, ImplicitTaintPolicy, Register

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def simulator() -> CellSimulator:
    """Default simulator (use_unicorn=False so we exercise cell.pyx native)."""
    return CellSimulator(Architecture.AMD64, use_unicorn=False)


@pytest.fixture(scope='module')
def regs() -> list[Register]:
    return [
        Register(name='RAX', bits=64),
        Register(name='EAX', bits=32),
        Register(name='AL', bits=8),
        Register(name='RBX', bits=64),
        Register(name='RCX', bits=64),
        Register(name='RDX', bits=64),
        Register(name='RBP', bits=64),
        Register(name='RSP', bits=64),
        Register(name='RSI', bits=64),
        Register(name='RDI', bits=64),
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


# ===========================================================================
# Bug #1 — RMW memory destinations skip the differential
# ===========================================================================


class TestBug1RMWDifferential:
    """`add [mem], reg`, `sub [mem], reg` etc. must use the proper
    differential, not just OR-of-input-taints."""

    def test_add_mem_reg_carries_through_byte_boundary(self, simulator, regs):
        """add [rbp-0x10], rax with V_mem=0xFF, V_RAX=0x01, T_RAX=0x01.
        Sum = 0x100; differential reveals 9 affected bits.

        The pure differential is 0x1FE.  The engine ORs this with the
        transport-term fallback (T_RAX | T_mem = 0x01) so explicitly-
        tainted bits are never lost on simulator failure, giving 0x1FF.
        We assert the differential bits are all present.
        """
        rbp = 0x80000000DD00
        mem_addr = rbp - 0x10
        shadow = BitPreciseShadowMemory()
        shadow.write_mask(mem_addr, 0x01, 8)

        def reader(addr, sz):
            return 0xFF if addr == mem_addr else 0

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
        # The differential bits (0x1FE) MUST be set; the OR-fallback may
        # additionally set bit 0 (giving 0x1FF).  Either is correct.
        assert mem_taint & 0x1FE == 0x1FE, (
            f'add carry should include differential mask 0x1FE, got {mem_taint:#x}.  '
            'This indicates the RMW path is no longer using MemoryDifferentialExpr '
            'or the differential is not detecting the carry chain.'
        )
        # And the popcount must be at least 8 (the differential alone).
        assert bin(mem_taint).count('1') >= 8, f'add carry must produce popcount>=8, got {bin(mem_taint).count("1")}.'

    def test_sub_mem_reg_borrows_through_byte_boundary(self, simulator, regs):
        """sub [rbp-0x10], rax: borrow ripple from bit 0 cascades upward."""
        rbp = 0x80000000DD00
        mem_addr = rbp - 0x10
        shadow = BitPreciseShadowMemory()
        shadow.write_mask(mem_addr, 0x01, 8)

        def reader(addr, sz):
            return 0x100 if addr == mem_addr else 0

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
        assert (
            bin(mem_taint).count('1') >= 9
        ), f'sub borrow should produce popcount>=9, got {bin(mem_taint).count("1")} (mask={mem_taint:#x}).'

    def test_pure_store_uses_or_path(self, simulator, regs):
        """`mov [rbp-0x10], rax` is a PURE store — NOT RMW.  It must NOT
        get routed through MemoryDifferentialExpr.  Asserting the cheap
        path (OR-of-input-taints) is taken: a tainted RAX with full mask
        gives a fully-tainted memory output, but the expression class
        must NOT be MemoryDifferentialExpr."""
        rbp = 0x80000000DD00
        mem_addr = rbp - 0x10
        ctx = EvalContext(
            input_taint={'RAX': 0xFFFFFFFFFFFFFFFF},
            input_values={'RAX': 0xDEADBEEF, 'RBP': rbp},
            simulator=simulator,
            shadow_memory=BitPreciseShadowMemory(),
            mem_reader=lambda a, s: 0,
            implicit_policy=ImplicitTaintPolicy.IGNORE,
        )
        # mov [rbp-0x10], rax (48 89 45 f0)
        circuit = generate_static_rule(Architecture.AMD64, bytes.fromhex('488945f0'), regs)
        # Find the memory-target assignment and check its expression class.
        mem_assignments = [a for a in circuit.assignments if hasattr(a.target, 'address_expr')]
        assert len(mem_assignments) == 1
        kind = type(mem_assignments[0].expression).__name__
        assert kind != 'MemoryDifferentialExpr', (
            f'pure store routed through differential (got {kind}); '
            'the cheap OR-only path should be used for performance.'
        )
        out = circuit.evaluate(ctx)
        assert out.get(f'MEM_{hex(mem_addr)}_8', 0) == 0xFFFFFFFFFFFFFFFF


# ===========================================================================
# Bug #2 — Memory inputs with non-zero offsets land at the wrong address
# ===========================================================================


class TestBug2MemoryInputOffset:
    """Memory operands like `[rbp-0x10]` must dereference rbp+(-0x10), not rbp."""

    def test_add_rax_mem_with_negative_offset(self, simulator, regs):
        """add rax, [rbp-0x10] — REG-DEST with offset memory input.
        V_RAX=0x01, T_RAX=0; V_mem=0xFF, T_mem=0x01.
        Expected differential: (0x01 + 0xFF) XOR (0x01 + 0xFE) = 0x100 ^ 0xFF = 0x1FF.
        """
        rbp = 0x80000000DD00
        mem_addr = rbp - 0x10
        shadow = BitPreciseShadowMemory()
        shadow.write_mask(mem_addr, 0x01, 8)

        def reader(addr, sz):
            return 0xFF if addr == mem_addr else 0

        ctx = EvalContext(
            input_taint={'RAX': 0},
            input_values={'RAX': 0x01, 'RBP': rbp},
            simulator=simulator,
            shadow_memory=shadow,
            mem_reader=reader,
            implicit_policy=ImplicitTaintPolicy.IGNORE,
        )
        circuit = generate_static_rule(Architecture.AMD64, bytes.fromhex('480345f0'), regs)
        out = circuit.evaluate(ctx)
        rax_taint = out.get('RAX', 0)
        assert bin(rax_taint).count('1') >= 9, (
            f'add rax, [rbp-0x10] with tainted mem must produce carry-ripple '
            f'taint with popcount>=9; got {bin(rax_taint).count("1")} (mask={rax_taint:#x}). '
            'This indicates the MEM input offset is being dropped from the address.'
        )

    def test_load_with_offset_propagates_full_taint(self, simulator, regs):
        """`mov rax, [rbp-0x10]` (qword load) — already worked even before
        the fix because it goes through the LOAD-LIKE path that reads
        shadow directly.  Regression guard."""
        rbp = 0x80000000DD00
        mem_addr = rbp - 0x10
        shadow = BitPreciseShadowMemory()
        shadow.write_mask(mem_addr, 0xFFFFFFFFFFFFFFFF, 8)

        def reader(addr, sz):
            return 0xDEADBEEF if addr == mem_addr else 0

        ctx = EvalContext(
            input_taint={},
            input_values={'RBP': rbp},
            simulator=simulator,
            shadow_memory=shadow,
            mem_reader=reader,
            implicit_policy=ImplicitTaintPolicy.IGNORE,
        )
        circuit = generate_static_rule(Architecture.AMD64, bytes.fromhex('488b45f0'), regs)
        out = circuit.evaluate(ctx)
        assert out.get('RAX', 0) == 0xFFFFFFFFFFFFFFFF


# ===========================================================================
# Bug #3 — Address-only registers missing from simulator state
# ===========================================================================


class TestBug3AddressOnlyRegisters:
    """When a register is used ONLY as a memory address (e.g. `RAX` in
    `add rdx, [rax]`), its concrete value must reach the simulator so
    the dereference resolves correctly."""

    def test_add_reg_mem_with_address_only_register(self, simulator, regs):
        """add rdx, [rax] — RAX is an address-only register.
        V_RAX=0x3000, V_RDX=0xFF, V_mem=0x01, T_mem=0x01.
        Expected: differential popcount=9 (carry from bit 0 ripples)."""
        shadow = BitPreciseShadowMemory()
        shadow.write_mask(0x3000, 0x01, 8)

        def reader(addr, sz):
            return 0x01 if addr == 0x3000 else 0

        ctx = EvalContext(
            input_taint={'RDX': 0},
            input_values={'RAX': 0x3000, 'RDX': 0xFF},
            simulator=simulator,
            shadow_memory=shadow,
            mem_reader=reader,
            implicit_policy=ImplicitTaintPolicy.IGNORE,
        )
        circuit = generate_static_rule(Architecture.AMD64, bytes.fromhex('480310'), regs)
        out = circuit.evaluate(ctx)
        rdx_taint = out.get('RDX', 0)
        assert bin(rdx_taint).count('1') >= 9, (
            f'add rdx, [rax] (RAX address-only) must produce carry-ripple '
            f'popcount>=9; got {bin(rdx_taint).count("1")} (mask={rdx_taint:#x}). '
            'This indicates the address register is not reaching the simulator state.'
        )

    def test_rmw_with_address_only_register_for_destination(self, simulator, regs):
        """add [rax], rbx — destination's own RAX is address-only and
        must be in the simulator's regs so the write hits the right address."""
        shadow = BitPreciseShadowMemory()
        shadow.write_mask(0x4000, 0x01, 8)

        def reader(addr, sz):
            return 0xFF if addr == 0x4000 else 0

        ctx = EvalContext(
            input_taint={'RBX': 0x01},
            input_values={'RAX': 0x4000, 'RBX': 0x01},
            simulator=simulator,
            shadow_memory=shadow,
            mem_reader=reader,
            implicit_policy=ImplicitTaintPolicy.IGNORE,
        )
        # add [rax], rbx  (48 01 18)
        circuit = generate_static_rule(Architecture.AMD64, bytes.fromhex('480118'), regs)
        out = circuit.evaluate(ctx)
        mem_taint = out.get('MEM_0x4000_8', 0)
        assert bin(mem_taint).count('1') >= 8, (
            f'add [rax], rbx must produce carry-ripple popcount>=8; '
            f'got {bin(mem_taint).count("1")} (mask={mem_taint:#x}).'
        )


# ===========================================================================
# Bug #4 — cell.pyx _read_output dynamic MEM format
# ===========================================================================


class TestBug4ReadOutputDynamicMem:
    """cell.pyx _read_output must parse `MEM_<reg>_<offset>_<size>`
    by reading the address register from the executed frame."""

    def test_pcode_native_handles_dynamic_mem_output(self, regs):
        """Direct call into the native p-code evaluator with a dynamic
        MEM out_reg.  Earlier the parser fell into the hex-only path and
        returned 0 silently."""
        from microtaint.simulator import _get_pcode_evaluator_class

        ev = _get_pcode_evaluator_class()(Architecture.AMD64)

        class Cell:
            def __init__(self, instr, out_reg, b_start, b_end):
                self.instruction = instr
                self.out_reg = out_reg
                self.out_bit_start = b_start
                self.out_bit_end = b_end

        # add [rbp-0x10], rax with both inputs polarised high
        rbp = 0x80000000DD00
        or_inputs = {
            'RAX': 0x01,
            'RBP': rbp,
            'MEM_RBP_-16_8': 0xFF,
        }
        and_inputs = {
            'RAX': 0x00,
            'RBP': rbp,
            'MEM_RBP_-16_8': 0xFE,
        }
        cell = Cell('480145f0', 'MEM_RBP_-16_8', 0, 63)
        diff = ev.evaluate_differential(cell, or_inputs, and_inputs)
        assert diff == 0x1FE, (
            f'cell.pyx evaluate_differential with dynamic MEM out_reg returned '
            f'{diff:#x}, expected 0x1FE.  Indicates _read_output is no longer '
            'parsing MEM_<reg>_<offset>_<size>.'
        )


# ===========================================================================
# Bug #5 — cell.pyx _load dynamic MEM format
# ===========================================================================


class TestBug5LoadDynamicMem:
    """cell.pyx _load must parse `MEM_<reg>_<offset>_<size>` for inputs
    by looking up the address register from the frame's already-loaded
    register state (two-pass: regs first, mems second)."""

    def test_pcode_native_consumes_dynamic_mem_input(self, regs):
        """Single concrete (non-differential) execution with a memory
        input in dynamic format.  Demonstrates that _load resolves the
        address correctly: the loaded byte should match what the
        instruction would read from `[rbp-0x10]`."""
        from microtaint.simulator import _get_pcode_evaluator_class

        ev = _get_pcode_evaluator_class()(Architecture.AMD64)

        class Cell:
            def __init__(self, instr, out_reg, b_start, b_end):
                self.instruction = instr
                self.out_reg = out_reg
                self.out_bit_start = b_start
                self.out_bit_end = b_end

        # mov rax, [rbp-0x10]   (48 8b 45 f0)
        # We feed the register-relative MEM key holding 0xCAFEBABEDEADBEEF
        # and verify that's what RAX gets.
        rbp = 0x80000000DD00
        inputs = {
            'RBP': rbp,
            'MEM_RBP_-16_8': 0xCAFEBABEDEADBEEF,
        }
        cell = Cell('488b45f0', 'RAX', 0, 63)
        result = ev.evaluate_concrete(cell, inputs)
        assert result == 0xCAFEBABEDEADBEEF, (
            f'cell.pyx _load did not place the MEM_RBP_-16_8 value at the '
            f'right address.  RAX after load = {result:#x}, expected '
            '0xCAFEBABEDEADBEEF.'
        )

    def test_pcode_native_legacy_static_mem_still_works(self, regs):
        """The static `MEM_<hex>_<size>` format must continue to work
        for callers that build flat dicts directly with hex addresses."""
        from microtaint.simulator import _get_pcode_evaluator_class

        ev = _get_pcode_evaluator_class()(Architecture.AMD64)

        class Cell:
            def __init__(self, instr, out_reg, b_start, b_end):
                self.instruction = instr
                self.out_reg = out_reg
                self.out_bit_start = b_start
                self.out_bit_end = b_end

        # mov rax, [0x10000]   (48 a1 00 00 01 00 00 00 00 00)
        inputs = {
            'MEM_0x10000_8': 0xCAFEBABEDEADBEEF,
        }
        cell = Cell('48a10000010000000000', 'RAX', 0, 63)
        result = ev.evaluate_concrete(cell, inputs)
        assert result == 0xCAFEBABEDEADBEEF


# ===========================================================================
# Cross-cutting: SipHash avalanche regression
# ===========================================================================


class TestSipHashAvalanche:
    """End-to-end avalanche check.  Single-bit input taint must spread
    to every output bit through SipHash's mixing rounds.  This test is
    the canary for ALL of bugs 1-5: any regression in any of them
    immediately breaks the avalanche."""

    def test_single_bit_taint_avalanches_through_siphash(self, tmp_path):
        import io
        import logging
        import subprocess

        logging.disable(logging.CRITICAL)

        # Compile the SipHash test binary.  Test files referenced live
        # alongside this test file in the project's test dir.
        from pathlib import Path

        SCRIPT_DIR = Path(__file__).resolve().parent
        srcs = [SCRIPT_DIR / 'test_avalanche_siphash.c', SCRIPT_DIR / 'siphash_ref.c']
        if not all(s.exists() for s in srcs):
            pytest.skip(f'SipHash sources not present at {SCRIPT_DIR}')

        binary = tmp_path / 'sip'
        r = subprocess.run(
            [
                'gcc',
                '-O0',
                '-g',
                '-static',
                '-no-pie',
                '-fno-stack-protector',
                '-o',
                str(binary),
                *[str(s) for s in srcs],
            ],
            capture_output=True,
        )
        if r.returncode != 0:
            pytest.skip(f'Could not build SipHash binary: {r.stderr.decode()[:200]}')

        from qiling import Qiling
        from qiling.const import QL_INTERCEPT, QL_VERBOSE

        from microtaint.emulator.reporter import Reporter
        from microtaint.emulator.wrapper import MicrotaintWrapper

        ql = Qiling([str(binary)], '/', verbose=QL_VERBOSE.OFF)
        msg = bytes(range(16))

        class _S:
            def read(self, n):
                return msg[:n]

        ql.os.stdin = _S()

        reporter = Reporter(json_mode=True, stream=io.StringIO())
        wrapper = MicrotaintWrapper(
            ql,
            check_sc=False,
            check_bof=False,
            check_uaf=False,
            check_aiw=False,
            reporter=reporter,
        )

        captured = [0]

        def _read_hook(ql, fd, buf, count):
            if fd != 0:
                return 0
            data = msg[:count]
            if not data:
                return 0
            ql.mem.write(buf, data)
            wrapper.taint_bit(buf + 0, 0)
            return len(data)

        ql.os.set_syscall(0, _read_hook, QL_INTERCEPT.CALL)

        def _write_hook(ql, fd, buf, count, *_):
            if fd == 1 and count == 8 and captured[0] == 0:
                captured[0] = wrapper.shadow_mem.read_mask(buf, 8)
            return count

        ql.os.set_syscall(1, _write_hook, QL_INTERCEPT.CALL)

        try:
            ql.run()
        except Exception:
            pass

        popcount = bin(captured[0]).count('1')
        assert popcount >= 25, (
            f'SipHash avalanche FAILED: only {popcount}/64 output bits depend '
            f'on the input bit (mask={captured[0]:#018x}).  This regression '
            'almost certainly means one of bugs 1-5 has reappeared.'
        )


# ===========================================================================
# MemoryDifferentialExpr direct tests (Cython class)
# ===========================================================================


class TestMemoryDifferentialExpr:
    """Direct unit tests on the Cython class itself, without going
    through the rule generator."""

    def test_construction_caches_target_strings(self):
        """The constructor pre-computes _target_out_reg etc. so the
        hot path doesn't re-format strings on every evaluate()."""
        expr = MemoryDifferentialExpr(
            bytestring=b'\x48\x01\x45\xf0',
            target=('MEM', 'RBP', -0x10, 8),
            reg_inputs=[('RAX', 0, 63)],
            mem_inputs=[('RBP', -0x10, 8)],
            addr_only_regs=['RBP'],
        )
        assert expr.out_reg == 'MEM_RBP_-16_8'
        assert expr.out_bit_start == 0
        assert expr.out_bit_end == 63
        assert expr.instruction == '480145f0'

    def test_register_target_uses_same_format(self):
        expr = MemoryDifferentialExpr(
            bytestring=b'\x48\x01\xd8',
            target=('REG', 'RAX', 0, 63),
            reg_inputs=[('RAX', 0, 63), ('RBX', 0, 63)],
            mem_inputs=[],
            addr_only_regs=[],
        )
        assert expr.out_reg == 'RAX'
        assert expr.out_bit_start == 0
        assert expr.out_bit_end == 63

    def test_or_input_polarisation_full_register(self, simulator):
        """For full-register slices (b_start=0, b_end>=63), or_inputs[reg]
        must equal V|T and and_inputs[reg] must equal V&~T, NOT a partial mask."""
        # We can't introspect the inputs directly, but we can verify by
        # constructing a known-output instruction.
        # mov rbx, rax  (48 89 c3) — copies RAX into RBX.
        # If V_RAX=0xCAFE and T_RAX=0x000F, the differential of RBX is:
        #   C1 = (0xCAFE | 0x000F) = 0xCAFF
        #   C2 = (0xCAFE & ~0x000F) = 0xCAF0
        #   diff = 0x000F
        expr = MemoryDifferentialExpr(
            bytestring=bytes.fromhex('4889c3'),
            target=('REG', 'RBX', 0, 63),
            reg_inputs=[('RAX', 0, 63)],
            mem_inputs=[],
            addr_only_regs=[],
        )
        ctx = EvalContext(
            input_taint={'RAX': 0x000F},
            input_values={'RAX': 0xCAFE},
            simulator=simulator,
            shadow_memory=BitPreciseShadowMemory(),
            implicit_policy=ImplicitTaintPolicy.IGNORE,
        )
        result = expr.evaluate(ctx)
        assert result == 0x000F

    def test_simulator_failure_falls_back_to_or(self, simulator):
        """If the simulator raises (e.g. unsupported instruction), the
        Cython class must fall back to OR-of-input-taints rather than
        returning 0 silently."""
        # Use clearly-invalid instruction bytes; cell.pyx will raise.
        expr = MemoryDifferentialExpr(
            bytestring=b'\xff\xff\xff\xff\xff\xff\xff\xff',
            target=('REG', 'RAX', 0, 63),
            reg_inputs=[('RAX', 0, 63), ('RBX', 0, 63)],
            mem_inputs=[],
            addr_only_regs=[],
        )
        ctx = EvalContext(
            input_taint={'RAX': 0x0F, 'RBX': 0xF0},
            input_values={'RAX': 0, 'RBX': 0},
            simulator=simulator,
            shadow_memory=BitPreciseShadowMemory(),
            implicit_policy=ImplicitTaintPolicy.IGNORE,
        )
        result = expr.evaluate(ctx)
        # Fallback should give us at least the OR of input taints.
        assert result == 0xFF
