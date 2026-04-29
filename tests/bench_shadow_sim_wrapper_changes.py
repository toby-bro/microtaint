"""
bench_changes.py — Targeted benchmarks for the performance-focused changes.

Tests ONLY the three modified components in isolation:
  1. shadow.pyx  — BitPreciseShadowMemory hot methods
  2. simulator.py — setup_registers_and_memory (batch writes) + bytestring cache
  3. wrapper.py  — _get_live_registers (batched reads) + _is_main_binary cache

Run with:
    pytest tests/bench_changes.py --benchmark-only
    pytest tests/bench_changes.py --benchmark-only --benchmark-sort=mean

Pin CPU frequency before running for reliable results:
    sudo cpupower frequency-set -g performance
    echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost
"""

# mypy: disable-error-code="no-untyped-def"
from __future__ import annotations

import pytest

from microtaint.emulator.shadow import BitPreciseShadowMemory
from microtaint.simulator import CellSimulator, MachineState
from microtaint.types import Architecture

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def shadow() -> BitPreciseShadowMemory:
    return BitPreciseShadowMemory()


@pytest.fixture(scope='module')
def amd64_sim() -> CellSimulator:
    return CellSimulator(Architecture.AMD64)


@pytest.fixture(scope='module')
def arm64_sim() -> CellSimulator:
    return CellSimulator(Architecture.ARM64)


# ---------------------------------------------------------------------------
# 1. shadow.pyx — BitPreciseShadowMemory
#
# These methods are called on every instruction in the hot path.
# write_mask/read_mask are called for every memory taint update.
# is_tainted/is_poisoned are called on every memory access for UAF detection.
# ---------------------------------------------------------------------------


class TestShadowMemory:
    """
    Isolates BitPreciseShadowMemory hot methods.
    All page allocations happen on first write; subsequent calls hit the
    fast dict-lookup path — which is what matters at runtime.
    """

    def test_write_mask_8bytes(self, benchmark, shadow):
        """write_mask for a single 8-byte taint word — most common case (register spill)."""
        shadow.write_mask(0x1000, 0xFFFFFFFFFFFFFFFF, 8)  # warm up page
        benchmark(shadow.write_mask, 0x1000, 0xFFFFFFFFFFFFFFFF, 8)

    def test_write_mask_1byte(self, benchmark, shadow):
        """write_mask for 1 byte — flag or byte register write."""
        shadow.write_mask(0x2000, 0xFF, 1)
        benchmark(shadow.write_mask, 0x2000, 0xFF, 1)

    def test_write_mask_clear_8bytes(self, benchmark, shadow):
        """write_mask with mask=0 — clearing taint (most common store path)."""
        shadow.write_mask(0x3000, 0, 8)
        benchmark(shadow.write_mask, 0x3000, 0, 8)

    def test_read_mask_8bytes_tainted(self, benchmark, shadow):
        """read_mask on a fully-tainted 8-byte region."""
        shadow.write_mask(0x4000, 0xFFFFFFFFFFFFFFFF, 8)
        benchmark(shadow.read_mask, 0x4000, 8)

    def test_read_mask_8bytes_clean(self, benchmark, shadow):
        """read_mask on a clean (zero-taint) region — early-exit path."""
        shadow.write_mask(0x5000, 0, 8)
        benchmark(shadow.read_mask, 0x5000, 8)

    def test_read_mask_8bytes_cold(self, benchmark, shadow):
        """read_mask on a never-written page — tests dict miss path."""
        # Use a fresh address that was never written
        benchmark(shadow.read_mask, 0xDEAD0000, 8)

    def test_is_tainted_true(self, benchmark, shadow):
        """is_tainted on a tainted region — common in load taint propagation."""
        shadow.write_mask(0x6000, 0xFFFFFFFFFFFFFFFF, 8)
        benchmark(shadow.is_tainted, 0x6000, 8)

    def test_is_tainted_false(self, benchmark, shadow):
        """is_tainted on a clean region — most common outcome (untainted data)."""
        shadow.write_mask(0x7000, 0, 8)
        benchmark(shadow.is_tainted, 0x7000, 8)

    def test_is_poisoned_false(self, benchmark, shadow):
        """is_poisoned on non-freed memory — fires on every mem read in UAF mode."""
        benchmark(shadow.is_poisoned, 0x8000, 8)

    def test_is_poisoned_true(self, benchmark, shadow):
        """is_poisoned on freed memory — the detection hit case."""
        shadow.poison(0x9000, 64)
        benchmark(shadow.is_poisoned, 0x9000, 8)

    def test_write_bytes_8(self, benchmark, shadow):
        """write_bytes for 8 bytes — used by the chunk loop in _taint_bytes."""
        data = bytes([0xFF] * 8)
        shadow.write_mask(0xA000, 0, 8)  # warm page
        benchmark(shadow.write_bytes, 0xA000, data)

    def test_cross_page_write(self, benchmark, shadow):
        """write_mask straddling a 4096-byte page boundary — rare but tested."""
        addr = 0xB000 - 4  # 4 bytes before page boundary
        shadow.write_mask(addr, 0xFFFFFFFFFFFFFFFF, 8)  # warm both pages
        benchmark(shadow.write_mask, addr, 0xFFFFFFFFFFFFFFFF, 8)


# ---------------------------------------------------------------------------
# 2. simulator.py — CellSimulator hot paths
#
# setup_registers_and_memory is called twice per evaluate_cell_differential.
# The bytestring cache means the second _execute call skips mem_write.
# We test both small (2 regs, typical rule evaluation) and large (18 regs,
# wrapper full-state evaluation) register sets.
# ---------------------------------------------------------------------------


class TestSimulator:
    """
    Isolates CellSimulator.setup_registers_and_memory and evaluate_cell_differential.
    """

    # --- setup_registers_and_memory directly ---------------------------------

    def test_setup_regs_2(self, benchmark, amd64_sim):
        """2 registers — typical InstructionCellExpr fallback state size."""
        state = MachineState(regs={'RDI': 0xFFFFFFFFFFFFFFFF, 'RAX': 0})
        amd64_sim.clear_memory_and_registers()
        benchmark(amd64_sim.setup_registers_and_memory, state, None)

    def test_setup_regs_6(self, benchmark, amd64_sim):
        """6 registers — batch threshold boundary."""
        state = MachineState(regs={
            'RAX': 0x1, 'RBX': 0x2, 'RCX': 0x3,
            'RDX': 0x4, 'RSI': 0x5, 'RDI': 0x6,
        })
        amd64_sim.clear_memory_and_registers()
        benchmark(amd64_sim.setup_registers_and_memory, state, None)

    def test_setup_regs_18(self, benchmark, amd64_sim):
        """18 registers — full canonical AMD64 state (wrapper path)."""
        state = MachineState(regs={
            'RAX': 0x1, 'RBX': 0x2, 'RCX': 0x3, 'RDX': 0x4,
            'RSI': 0x5, 'RDI': 0x6, 'RBP': 0x7, 'RSP': 0x80000000,
            'R8':  0x8, 'R9':  0x9, 'R10': 0xA, 'R11': 0xB,
            'R12': 0xC, 'R13': 0xD, 'R14': 0xE, 'R15': 0xF,
            'RIP': 0x400000, 'EFLAGS': 0x202,
        })
        amd64_sim.clear_memory_and_registers()
        benchmark(amd64_sim.setup_registers_and_memory, state, None)

    # --- bytestring cache ----------------------------------------------------

    def test_execute_same_bytes_twice(self, benchmark, amd64_sim):
        """
        evaluate_cell_differential on the same instruction twice.
        Second call should hit the bytestring cache and skip mem_write.
        This is the exact pattern of the differential: OR run then AND run.
        """
        # ADD RAX, RBX
        bytestring = bytes.fromhex('4801D8')
        v_state = MachineState(regs={'RAX': 0xAAAA, 'RBX': 0x5555})
        t_state = MachineState(regs={'RAX': 0xAAAA, 'RBX': 0x5555})
        benchmark(amd64_sim.evaluate_cell_differential, bytestring, 'RAX', v_state, t_state)

    def test_execute_different_bytes(self, benchmark, amd64_sim):
        """
        evaluate_cell_differential with bytes that change each round.
        Cache miss every call — measures baseline without cache benefit.
        Implemented by alternating two instructions.
        """
        instrs = [bytes.fromhex('4801D8'), bytes.fromhex('4829D8')]
        v_state = MachineState(regs={'RAX': 0xAAAA, 'RBX': 0x5555})
        t_state = MachineState(regs={'RAX': 0xAAAA, 'RBX': 0x5555})
        state = {'i': 0}

        def run():
            bs = instrs[state['i'] & 1]
            state['i'] += 1
            return amd64_sim.evaluate_cell_differential(bs, 'RAX', v_state, t_state)

        benchmark(run)

    # --- full differential evaluation at different register counts -----------

    def test_differential_2regs(self, benchmark, amd64_sim):
        """Differential with 2 registers — path_explosion benchmark baseline."""
        bytestring = bytes.fromhex('4801F8')  # ADD RAX, RDI
        v_state = MachineState(regs={'RDI': 0xFFFFFFFFFFFFFFFF, 'RAX': 0})
        t_state = MachineState(regs={'RDI': 0xFFFFFFFFFFFFFFFF, 'RAX': 0})
        benchmark(amd64_sim.evaluate_cell_differential, bytestring, 'RAX', v_state, t_state)

    def test_differential_18regs(self, benchmark, amd64_sim):
        """Differential with 18 registers — full wrapper-style state."""
        bytestring = bytes.fromhex('4801D8')  # ADD RAX, RBX
        regs = {
            'RAX': 0x1, 'RBX': 0x2, 'RCX': 0x3, 'RDX': 0x4,
            'RSI': 0x5, 'RDI': 0x6, 'RBP': 0x7, 'RSP': 0x80000000,
            'R8':  0x8, 'R9':  0x9, 'R10': 0xA, 'R11': 0xB,
            'R12': 0xC, 'R13': 0xD, 'R14': 0xE, 'R15': 0xF,
            'RIP': 0x400000, 'EFLAGS': 0x202,
        }
        v_state = MachineState(regs=regs)
        t_state = MachineState(regs={**regs, 'RAX': 0x1, 'RBX': 0x2})
        benchmark(amd64_sim.evaluate_cell_differential, bytestring, 'RAX', v_state, t_state)

    def test_differential_with_memory(self, benchmark, amd64_sim):
        """Differential with memory operand — exercises load_memory_state."""
        bytestring = bytes.fromhex('488B07')  # MOV RAX, [RDI]
        v_state = MachineState(
            regs={'RDI': 0x500000, 'RAX': 0},
            mem={0x500000: 0xDEADBEEF},
        )
        t_state = MachineState(
            regs={'RDI': 0x500000, 'RAX': 0},
            mem={0x500000: 0xFFFFFFFF},
        )
        benchmark(amd64_sim.evaluate_cell_differential, bytestring, 'RAX', v_state, t_state)

    # --- ARM64 ---------------------------------------------------------------

    def test_differential_arm64_2regs(self, benchmark, arm64_sim):
        """ARM64 differential — ADD X0, X0, X1."""
        bytestring = bytes.fromhex('0000018B')
        v_state = MachineState(regs={'X0': 0x1, 'X1': 0x2})
        t_state = MachineState(regs={'X0': 0xFFFFFFFFFFFFFFFF, 'X1': 0})
        benchmark(arm64_sim.evaluate_cell_differential, bytestring, 'X0', v_state, t_state)


# ---------------------------------------------------------------------------
# 3. wrapper.py — _get_live_registers and _is_main_binary
#
# These cannot be benchmarked through Qiling without a full binary.
# Instead we benchmark the underlying operations they perform:
#   - Reading N registers from a dict (simulates the batched read result)
#   - Address range check (simulates _is_main_binary)
# This lets us compare old vs new implementation cost directly.
# ---------------------------------------------------------------------------


class TestWrapperPrimitives:
    """
    Benchmarks the primitives underlying wrapper.py hot paths,
    without needing a full Qiling instance.
    """

    # --- _is_main_binary patterns --------------------------------------------

    def test_address_check_single_range_hit(self, benchmark):
        """
        _is_main_binary fast path: single cached range, address inside.
        This is the hot case — every main-binary instruction.
        New: two int comparisons. Old: any() over list of tuples.
        """
        base, end = 0x400000, 0x410000
        address = 0x405000

        # New implementation — direct int comparison
        def check_new():
            return base <= address < end

        benchmark(check_new)

    def test_address_check_single_range_miss(self, benchmark):
        """_is_main_binary fast path: address outside range (library code)."""
        base, end = 0x400000, 0x410000
        address = 0x7FFFF7A00000  # libc address

        def check_new():
            return base <= address < end

        benchmark(check_new)

    def test_address_check_list_any_hit(self, benchmark):
        """
        Old _is_main_binary: any() over list of tuples.
        Compare against test_address_check_single_range_hit.
        """
        bounds = [(0x400000, 0x410000)]
        address = 0x405000

        def check_old():
            return any(s <= address < e for s, e in bounds)

        benchmark(check_old)

    # --- _get_live_registers patterns ----------------------------------------

    def test_read_18_keys_from_dict(self, benchmark):
        """
        Simulates new _get_live_registers: read 18 keys from a pre-built dict.
        Represents the result of a batch register read being unpacked.
        """
        _CANONICAL = [
            'RAX', 'RBX', 'RCX', 'RDX', 'RSI', 'RDI', 'RBP', 'RSP',
            'R8', 'R9', 'R10', 'R11', 'R12', 'R13', 'R14', 'R15',
            'RIP', 'EFLAGS',
        ]
        source = {k: i * 0x100 for i, k in enumerate(_CANONICAL)}

        def read_18():
            return {k: source[k] for k in _CANONICAL}

        benchmark(read_18)

    def test_read_23_keys_from_dict(self, benchmark):
        """
        Simulates old _get_live_registers: read all 23 X64_FORMAT entries
        including redundant sub-registers EAX, AX, AL, AH, ZF, CF, SF, OF, PF.
        Compare against test_read_18_keys_from_dict.
        """
        _ALL = [
            'RAX', 'RBX', 'RCX', 'RDX', 'RSI', 'RDI', 'RBP', 'RSP',
            'R8', 'R9', 'R10', 'R11', 'R12', 'R13', 'R14', 'R15',
            'RIP', 'EFLAGS', 'ZF', 'CF', 'SF', 'OF', 'PF',
        ]
        source = {k: i * 0x100 for i, k in enumerate(_ALL)}

        def read_23():
            return {k: source[k] for k in _ALL}

        benchmark(read_23)

    def test_eflags_unpack_bitshift(self, benchmark):
        """
        New: unpack 5 flags from EFLAGS by bit shifting — no extra reads.
        """
        eflags = 0x246  # ZF=1, PF=1, CF=0, SF=0, OF=0

        def unpack():
            return {
                'CF': (eflags >> 0)  & 1,
                'PF': (eflags >> 2)  & 1,
                'ZF': (eflags >> 6)  & 1,
                'SF': (eflags >> 7)  & 1,
                'OF': (eflags >> 11) & 1,
            }

        benchmark(unpack)

    def test_shallow_copy_dict_small(self, benchmark):
        """
        Cost of dict(self.register_taint) in _pre_taint snapshot.
        Small dict — typical taint state has few active registers.
        """
        d = {'RAX': 0xFFFFFFFFFFFFFFFF, 'RDI': 0x12345678}
        benchmark(dict, d)

    def test_shallow_copy_dict_large(self, benchmark):
        """
        Cost of dict(self.register_taint) worst case — all registers tainted.
        """
        d = {f'R{i}': i * 0xFF for i in range(18)}
        benchmark(dict, d)
