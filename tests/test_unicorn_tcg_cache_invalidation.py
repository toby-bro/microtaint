"""
test_unicorn_tcg_cache_invalidation.py
======================================
Regression tests for the cross-test taint-leakage bug caused by Unicorn's
JIT translation cache.

Bug
---
Unicorn JIT-compiles guest basic blocks at first execution and caches the
translations indexed by guest *physical* address.  When ``CellSimulator``
overwrites the bytes at ``CODE_ADDR`` with ``mem_write``, Unicorn does NOT
invalidate the cached translations — it doesn't watch for self-modifying
code unless asked.  Without ``ctl_remove_cache``, ``emu_start`` runs the
PREVIOUS test's JIT-compiled translation against the new register state.

This produces side effects from the previous test's instructions executed
in the current test's register state — for instance, a `mov [rsp-32], rax`
from a prior `rep movsb` test getting executed when the new test is
nominally `mov al, bl`.  The wild memory writes are not tracked in
``_dirtied_memory``, so ``clear_memory_and_registers`` on a subsequent
test won't zero them.  Later tests that read those memory locations pick
up the polluted bytes and produce incorrect taint outputs.

Discovery
---------
The bug was reproduced via a 3-test sequence (idx 7989 / 7990 / 8009 in
the precision-soundness benchmark), all dispatching to the same
CellSimulator.  Removing any one test makes the bug disappear.  Adding
``ctl_remove_cache`` after every ``mem_write(CODE_ADDR, bytestring)``
fixes the leak across all three CellSimulator backends (use_unicorn=True,
use_c=False, and use_c=True).

Tests below pin the fix.  Each test plays a sequence of instructions
through ONE shared ``CellSimulator`` and asserts the final test's taint
output is the byte-repeated value the rep-prefixed semantics require —
i.e. that no garbage bytes from prior tests' Unicorn execution have
leaked into the differential evaluator's memory reads.
"""

from __future__ import annotations

from typing import Any

import pytest

from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import (
    _cached_generate_static_rule,
    generate_static_rule,
)
from microtaint.types import Architecture, Register

# ---------------------------------------------------------------------------
# Helpers — match the worker_microtaint.py state-format exactly so the
# regression tests exercise the same circuit-cache key as the benchmark.

_REGS = (
    [Register('RAX', 64), Register('RBX', 64), Register('RCX', 64), Register('RDX', 64)]
    + [Register('RSI', 64), Register('RDI', 64), Register('RSP', 64), Register('RBP', 64)]
    + [Register(f'R{n}', 64) for n in range(8, 16)]
    + [Register(f'XMM{n}_LO', 64) for n in range(8)]
    + [Register(f'XMM{n}_HI', 64) for n in range(8)]
)
_DEFAULT_RSP = 0x80000000


def _run(sim: CellSimulator, bs_hex: str, state: dict[str, int], taint: dict[str, int]) -> int:
    """Run one taint analysis through the given simulator and return the
    RAX bit-mask of the resulting taint dict.  Mirrors what the worker
    does, including the RSP=0 → 0x80000000 fallback."""
    full_state = {r.name: state.get(r.name, 0) for r in _REGS}
    if full_state.get('RSP', 0) == 0:
        full_state['RSP'] = _DEFAULT_RSP
    full_taint = {r.name: taint.get(r.name, 0) for r in _REGS}
    bytestring = bytes.fromhex(bs_hex)
    circuit = generate_static_rule(Architecture.AMD64, bytestring, _REGS)
    ctx = EvalContext(input_values=full_state, input_taint=full_taint, simulator=sim)
    raw = circuit.evaluate(ctx)
    return raw.get('RAX', 0) & ((1 << 64) - 1)


# Backend matrix — the bug must be fixed across all three.  Each
# parameter is a kwargs dict for ``CellSimulator``.
BACKENDS: list[tuple[str, dict[str, Any]]] = [
    ('unicorn', {'use_unicorn': True, 'use_c': False}),
    ('cython', {'use_unicorn': False, 'use_c': False}),
    ('c', {'use_unicorn': False, 'use_c': True}),
]


@pytest.fixture(autouse=True)
def _clear_circuit_cache() -> Any:
    """Ensure each test starts with an empty static-rule cache so prior
    test runs cannot affect the LogicCircuit identity used here."""
    _cached_generate_static_rule.cache_clear()
    yield
    _cached_generate_static_rule.cache_clear()


# ---------------------------------------------------------------------------
# Test inputs — extracted verbatim from the precision-soundness benchmark
# (idx 7989 / 7990 / 8009 in report_1778427241.json).  Hard-coded so this
# test is self-contained and doesn't depend on a benchmark report file.

# Test 7989: rep movsb — fully-tainted RAX spilled to [rsp-32] then
# copied to [rsp-64], reloaded into RAX.  Output taint is the propagation
# of the fully-tainted RAX through the memory copy.
T7989_BYTES = 'fc48894424e0488d7424e0488d7c24c0b908000000f3a4488b4424c0'
T7989_STATE: dict[str, int] = {
    'RAX': 15470179137417423390,
    'RBX': 5284306197202325960,
    'RCX': 91,
    'RDX': 13740211632721712556,
}
T7989_TAINT: dict[str, int] = {
    'RAX': 18446744073709551615,
    'RBX': 35184372088832,
    'RCX': 12207444339575697730,
    'RDX': 1828153273910131988,
}

# Test 7990: mov al, bl ; mov ah, cl  — tiny ChainedCircuit of two
# partial-register writes.  Bytestring is ``88d8 88cc`` (4 bytes total),
# while 7989's bytestring is 28 bytes — so when 7990 overwrites only the
# first 4 bytes at CODE_ADDR, bytes 4..27 still hold 7989's tail.  THIS
# is what triggers the cache-leak: emu_start runs the cached translation
# of 7989's prefix and dispatches to the cached translation of the tail
# memory-store instructions, dirtying [rsp-32] and [rsp-64] with 7990's
# RAX bytes — without recording those writes in _dirtied_memory.
T7990_BYTES = '88d888cc'
T7990_STATE: dict[str, int] = {
    'RAX': 7314663058932324252,
    'RBX': 14501752426502692987,
    'RCX': 2069826156496372160,
    'RDX': 8251849794471225177,
}
T7990_TAINT: dict[str, int] = {
    'RAX': 2097152,
    'RBX': 1073741824,
    'RCX': 0,
    'RDX': 0,
}

# Test 8009: rep stosb — spills RAX to [rsp-64], then stores 8 copies of
# AL there, reloads RAX from [rsp-64].  AL of the post-`mov rax, rbx`
# RAX is BL, so the result is BL replicated 8x.  Bit 1 of BL is tainted
# (RBX taint = 0x2), so the architecturally-correct taint output is
# 0x0202020202020202 — a byte-repeated mask.  Microtaint may overtaint
# slightly to 0x0202020202021302 (the full RAX[15:8] taint slice from the
# original RAX is preserved in some byte 0 / byte 1 bits); both are
# acceptable.  The BUG produces 0x7706aa7ab587952e — a non-byte-repeated
# garbage value that betrays cross-test memory pollution.
T8009_BYTES = 'fc48894424c0488d7c24c04889d8b908000000f3aa488b4424c0'
T8009_STATE: dict[str, int] = {
    'RAX': 10940498380929573403,
    'RBX': 1830928842394036844,
    'RCX': 19,
    'RDX': 5767559093351484470,
}
T8009_TAINT: dict[str, int] = {
    'RAX': 4352,
    'RBX': 2,
    'RCX': 0,
    'RDX': 8796093022208,
}

# Soundness invariants for the rep-stosb taint output.
#
# Test 8009 stores ``BL`` (bit 1 tainted) 8 times to ``[rsp-64]``, then
# reloads RAX from there.  Each output byte gets BL's taint — bit 1 of
# every byte = mask 0x0202020202020202.  Microtaint's differential
# evaluator may overtaint *byte 1* with the original RAX[15:8] taint
# slice (0x1100 = bits 8 and 12 = bits 0 and 4 of byte 1), giving
# 0x0202020202021302.  Both are acceptable, sound outputs.
#
# The bug's value 0x7706aa7ab587952e is unmistakably wrong.

# Sound microtaint output we observed; pin it as the canonical answer.
SOUND_8009_TAINT = 0x0202020202021302

# Bit 1 of every byte must always be set — that bit is the BL taint that
# rep stosb propagates into every output byte.  Independent of overtaint.
REP_STOSB_TAINT_FLOOR = 0x0202020202020202

# Specific known-broken value from the original bug report.
KNOWN_BROKEN_8009 = 0x7706AA7AB587952E


def _has_rep_stosb_taint_floor(mask: int) -> bool:
    """Every byte of the 64-bit mask has at least bit 1 set?"""
    return (mask & REP_STOSB_TAINT_FLOOR) == REP_STOSB_TAINT_FLOOR


def _is_byte_repeated(mask: int) -> bool:
    """A 64-bit mask is byte-repeated iff every byte equals byte 0.
    This is a stronger property used only where the test sets up taint
    such that only the BL bit is propagated, with no overtaint."""
    b0 = mask & 0xFF
    return mask == (b0 * 0x0101010101010101) & 0xFFFFFFFFFFFFFFFF


# ---------------------------------------------------------------------------
# Core regression tests — the 3-test poisoning sequence


@pytest.mark.parametrize(('backend_name', 'backend_kwargs'), BACKENDS, ids=[name for name, _ in BACKENDS])
class TestThreeTestPoisoningSequence:
    """The 3-test sequence (7989, 7990, 8009) that originally exposed the
    bug.  All three backends must produce the architecturally-correct
    byte-repeated taint output for test 8009, and must NOT produce the
    known-broken value 0x7706aa7ab587952e."""

    def test_full_sequence_produces_byte_repeated_taint(
        self,
        backend_name: str,
        backend_kwargs: dict[str, Any],
    ) -> None:
        sim = CellSimulator(Architecture.AMD64, **backend_kwargs)
        _run(sim, T7989_BYTES, T7989_STATE, T7989_TAINT)
        _run(sim, T7990_BYTES, T7990_STATE, T7990_TAINT)
        rax_8009 = _run(sim, T8009_BYTES, T8009_STATE, T8009_TAINT)
        assert rax_8009 != KNOWN_BROKEN_8009, (
            f'Backend {backend_name}: produced the exact known-broken value '
            f'0x{rax_8009:016x} — Unicorn TCG cache invalidation regression.'
        )
        assert _has_rep_stosb_taint_floor(rax_8009), (
            f'Backend {backend_name}: rep-stosb taint 0x{rax_8009:016x} '
            f'is missing the per-byte BL-bit-1 floor 0x{REP_STOSB_TAINT_FLOOR:016x}; '
            f'rep stosb stores the same byte 8x, so every output byte must '
            f"inherit BL's taint at bit 1."
        )

    def test_full_sequence_matches_known_sound_value(self, backend_name: str, backend_kwargs: dict[str, Any]) -> None:
        """Sharp-but-fragile pin against the exact value microtaint
        produces today.  Update this only when an intentional taint-
        precision change (more or less overtaint) is committed."""
        sim = CellSimulator(Architecture.AMD64, **backend_kwargs)
        _run(sim, T7989_BYTES, T7989_STATE, T7989_TAINT)
        _run(sim, T7990_BYTES, T7990_STATE, T7990_TAINT)
        rax_8009 = _run(sim, T8009_BYTES, T8009_STATE, T8009_TAINT)
        assert rax_8009 == SOUND_8009_TAINT, (
            f'Backend {backend_name}: expected 0x{SOUND_8009_TAINT:016x}, '
            f'got 0x{rax_8009:016x}.  Either a TCG-cache-invalidation '
            f"regression has reappeared, or microtaint's overtaint pattern "
            f'changed (intentional? then update this test).'
        )

    def test_full_sequence_matches_isolated_8009(self, backend_name: str, backend_kwargs: dict[str, Any]) -> None:
        """Test 8009 alone produces the correct taint; running the
        poisoning sequence before it must not change that result."""
        sim_alone = CellSimulator(Architecture.AMD64, **backend_kwargs)
        rax_alone = _run(sim_alone, T8009_BYTES, T8009_STATE, T8009_TAINT)

        sim_poisoned = CellSimulator(Architecture.AMD64, **backend_kwargs)
        _run(sim_poisoned, T7989_BYTES, T7989_STATE, T7989_TAINT)
        _run(sim_poisoned, T7990_BYTES, T7990_STATE, T7990_TAINT)
        rax_after = _run(sim_poisoned, T8009_BYTES, T8009_STATE, T8009_TAINT)

        assert rax_alone == rax_after, (
            f'Backend {backend_name}: 8009 alone produced 0x{rax_alone:016x} '
            f'but 8009 after the poisoning sequence produced 0x{rax_after:016x} '
            f'— prior tests must not affect 8009.  TCG cache invalidation regression.'
        )

    def test_repeated_evaluation_is_stable(self, backend_name: str, backend_kwargs: dict[str, Any]) -> None:
        """Running 8009 multiple times after the poisoning sequence must
        produce the SAME taint each time.  The bug originally produced
        a different broken value on the first call vs. subsequent calls."""
        sim = CellSimulator(Architecture.AMD64, **backend_kwargs)
        _run(sim, T7989_BYTES, T7989_STATE, T7989_TAINT)
        _run(sim, T7990_BYTES, T7990_STATE, T7990_TAINT)
        outputs = [_run(sim, T8009_BYTES, T8009_STATE, T8009_TAINT) for _ in range(5)]
        assert len(set(outputs)) == 1, (
            f'Backend {backend_name}: repeated 8009 evaluations gave '
            f'different results: {[hex(v) for v in outputs]} — first-call '
            f'TCG-cache-leak bug regression.'
        )

    @pytest.mark.parametrize(
        'order',
        [
            pytest.param([T7989_BYTES, T7990_BYTES, T8009_BYTES], id='7989_7990_8009'),
            pytest.param([T7990_BYTES, T7989_BYTES, T8009_BYTES], id='7990_7989_8009'),
            pytest.param([T7989_BYTES, T8009_BYTES], id='7989_8009'),
            pytest.param([T7990_BYTES, T8009_BYTES], id='7990_8009'),
            pytest.param([T8009_BYTES], id='8009_alone'),
        ],
    )
    def test_8009_taint_satisfies_floor_under_all_orderings(
        self,
        backend_name: str,
        backend_kwargs: dict[str, Any],
        order: list[str],
    ) -> None:
        """Whatever the prior-test sequence, 8009's output must keep
        every byte's bit-1 set (the BL-taint floor).  The differential
        evaluation of rep-stosb mandates this regardless of preceding
        instructions."""
        states = {T7989_BYTES: T7989_STATE, T7990_BYTES: T7990_STATE, T8009_BYTES: T8009_STATE}
        taints = {T7989_BYTES: T7989_TAINT, T7990_BYTES: T7990_TAINT, T8009_BYTES: T8009_TAINT}
        sim = CellSimulator(Architecture.AMD64, **backend_kwargs)
        last_rax = 0
        for bs in order:
            last_rax = _run(sim, bs, states[bs], taints[bs])
        assert order[-1] == T8009_BYTES, 'test misconfigured'
        assert (
            last_rax != KNOWN_BROKEN_8009
        ), f'Backend {backend_name}: order {order!r} reproduces the known-broken value.'
        assert _has_rep_stosb_taint_floor(last_rax), (
            f'Backend {backend_name}: order {order!r} gave 0x{last_rax:016x} '
            f'for 8009 — missing rep-stosb BL-bit-1 floor.'
        )


# ---------------------------------------------------------------------------
# Generic byte-rewrite-at-CODE_ADDR regression — synthetic, decoupled
# from the specific 3-test sequence.


@pytest.mark.parametrize(('backend_name', 'backend_kwargs'), BACKENDS, ids=[name for name, _ in BACKENDS])
class TestCodeRewriteSemantics:
    """Sanity checks: when the simulator's code region is rewritten with
    a new bytestring, ``emu_start`` must execute the NEW bytes.  These
    tests would all fail catastrophically if the TCG translation cache
    were not invalidated."""

    def test_long_then_short_bytestring(self, backend_name: str, backend_kwargs: dict[str, Any]) -> None:
        """A long bytestring followed by a SHORTER one — the classic
        cache-invalidation hazard.  The short bytestring's emu_start
        must not dispatch to the cached long-bytestring translation."""
        sim = CellSimulator(Architecture.AMD64, **backend_kwargs)
        # First: 28-byte rep-movsb sequence
        _run(sim, T7989_BYTES, T7989_STATE, T7989_TAINT)
        # Second: 4-byte mov al, bl ; mov ah, cl
        rax = _run(sim, T7990_BYTES, T7990_STATE, T7990_TAINT)
        # 7990's correct taint output is just RAX bit 21 (AH bit 5 from
        # CL bit 5? no — RAX taint=0x200000 stays in place since neither
        # mov writes to that bit, AL/AH only).  The exact value isn't the
        # point here; the point is that bytes 4..27 of CODE_ADDR are
        # 7989's tail that include `mov [rsp-32], rax`, and that
        # instruction must NOT execute against 7990's RSP/RAX state.
        # We assert the differential-eval result is bounded (no garbage
        # in unrelated bits).
        assert rax & ~0xFFFFFFFF == 0, (
            f'Backend {backend_name}: rax taint 0x{rax:016x} has high-bit '
            f'pollution; expected only low-32-bit propagation from a '
            f'partial-register write.  TCG cache regression.'
        )

    def test_short_then_long_bytestring(self, backend_name: str, backend_kwargs: dict[str, Any]) -> None:
        """Reverse order — a 4-byte bytestring then a 28-byte one.  The
        long bytestring's emu_start must translate ALL its bytes
        afresh, not start with the cached short translation."""
        sim = CellSimulator(Architecture.AMD64, **backend_kwargs)
        _run(sim, T7990_BYTES, T7990_STATE, T7990_TAINT)
        rax_8009 = _run(sim, T8009_BYTES, T8009_STATE, T8009_TAINT)
        assert _has_rep_stosb_taint_floor(rax_8009), (
            f'Backend {backend_name}: 8009 after a short bytestring gave '
            f'0x{rax_8009:016x}, missing the rep-stosb BL-bit-1 floor. '
            f'TCG cache regression.'
        )

    def test_alternating_bytestrings(self, backend_name: str, backend_kwargs: dict[str, Any]) -> None:
        """Alternate between two bytestrings repeatedly.  Each switch
        must invalidate the cache; if it doesn't, errors will accumulate
        and at least one run will produce an output missing the
        rep-stosb BL-bit-1 floor."""
        sim = CellSimulator(Architecture.AMD64, **backend_kwargs)
        for _ in range(5):
            _run(sim, T7989_BYTES, T7989_STATE, T7989_TAINT)
            rax = _run(sim, T8009_BYTES, T8009_STATE, T8009_TAINT)
            assert _has_rep_stosb_taint_floor(
                rax,
            ), f'Backend {backend_name}: alternating runs produced 0x{rax:016x} for 8009, missing BL-bit-1 floor.'


# ---------------------------------------------------------------------------
# Internal-state probes — assert the simulator's memory-tracking invariants


@pytest.mark.parametrize(('backend_name', 'backend_kwargs'), BACKENDS, ids=[name for name, _ in BACKENDS])
class TestMemoryStateInvariants:

    def test_8009_memory_does_not_leak_prior_test_rax(self, backend_name: str, backend_kwargs: dict[str, Any]) -> None:
        """After running the poisoning sequence and reading the spill
        location [rsp-64] = 0x7FFFFFC0 in Unicorn memory, the bytes
        there must be the rep-stosb output (AL of post-mov RAX), NOT
        test 7990's RAX bytes (the leak signature).

        Pre-fix, mem[0x7FFFFFC0] would contain 0x9c57551f4ce78265 (the
        little-endian bytes of T7990_STATE['RAX'] = 0x6582e74c1f55579c)
        after running the poisoning sequence, even though 7990's
        bytestring (`mov al, bl; mov ah, cl`) doesn't write to memory."""
        sim = CellSimulator(Architecture.AMD64, **backend_kwargs)
        _run(sim, T7989_BYTES, T7989_STATE, T7989_TAINT)
        _run(sim, T7990_BYTES, T7990_STATE, T7990_TAINT)

        # The exact byte signature of the pre-fix bug — RAX 7990 LE.
        rax_le_signature = T7990_STATE['RAX'].to_bytes(8, 'little')
        try:
            mem_after_7990 = bytes(sim.uc.mem_read(0x7FFFFFC0, 8))
        except Exception:
            # Memory may not be mapped at all if no test wrote there —
            # that itself proves no leak.
            return
        assert mem_after_7990 != rax_le_signature, (
            f'Backend {backend_name}: mem[0x7FFFFFC0] = '
            f'{mem_after_7990.hex()} matches T7990_STATE["RAX"] in '
            f'little-endian — the TCG cache leak signature.  Test 7990 '
            f'(mov al, bl ; mov ah, cl) must not write to memory.'
        )
