# ruff: noqa: S101, PLC0415
"""evaluate_c_mem must be bit-identical to circuit.evaluate (do_evaluate).

Memory circuits (loads, stores, mem-ALU) can run the C-array interior via
CompiledCircuit.evaluate_c_mem: register taints/values come from uint64 arrays,
shadow taint is read/written at the C level (the shadow C-API capsule), and
concrete memory values go through mem_reader (GIL held).  It must return either
None (the caller falls back to do_evaluate) or a dict byte-identical to
do_evaluate's output.  This walks a spread of load/store/RMW instructions across
ISAs, seeds real shadow taint + concrete memory, and asserts equality (or an
acceptable bail) directly -- no Qiling, so no multi-instance segfault risk.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'benchmark'))
from instruction_bank import isa_registers  # type: ignore[import-not-found]

from microtaint.emulator.shadow import BitPreciseShadowMemory
from microtaint.instrumentation.ast import EvalContext
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, ImplicitTaintPolicy

FULL = 0xFFFFFFFFFFFFFFFF
BASE = 0x100000  # a base address the addressing modes below land on/near

# (arch, label, hex).  Chosen so the effective address stays near BASE.
CASES = [
    # --- AMD64 ---
    (Architecture.AMD64, 'mov rax, [rbx]',   '488b03'),   # load
    (Architecture.AMD64, 'mov [rbx], rax',   '488903'),   # store
    (Architecture.AMD64, 'add rax, [rbx]',   '480303'),   # load-ALU (cell + shadow taint)
    (Architecture.AMD64, 'add [rbx], rax',   '480103'),   # RMW (cell + mem value + shadow)
    (Architecture.AMD64, 'and [rbx], rax',   '482103'),   # RMW value-dependent
    (Architecture.AMD64, 'xor rax, [rbx]',   '483303'),   # load-XOR
    (Architecture.AMD64, 'mov eax, [rbx]',   '8b03'),      # 32-bit load (zero-extend)
    (Architecture.AMD64, 'movzx eax, byte [rbx]', '0fb603'),
    (Architecture.AMD64, 'mov al, [rbx]',    '8a03'),      # 8-bit load
    (Architecture.AMD64, 'mov [rbx], eax',   '8903'),      # 32-bit store
    (Architecture.AMD64, 'sub [rbx], rax',   '482903'),
]


def _mem_reader_factory(mem: dict[int, int]):
    def reader(addr: int, size: int) -> int:
        return sum((mem.get(addr + i, 0) & 0xFF) << (8 * i) for i in range(size))
    return reader


def _run_case(arch: Architecture, hx: str, *, rax_taint: int, mem_taint_bytes: dict[int, int]):
    regs = list(isa_registers(arch))
    sim = CellSimulator(arch)
    circ = generate_static_rule(arch, bytes.fromhex(hx), regs)

    # Register state: a pointer register -> BASE, plus a tainted data register.
    values = {r.name: 0 for r in regs}
    taint = {r.name: 0 for r in regs}
    # AMD64: RBX is the pointer, RAX the data.  (isa_registers uses parent names.)
    for name in ('RBX', 'X1', 'r3', '4'):
        if name in values:
            values[name] = BASE
    for name in ('RAX', 'X0', 'r4', '5'):
        if name in taint:
            taint[name] = rax_taint
            values[name] = 0xDEADBEEFCAFEBABE

    shadow = BitPreciseShadowMemory()
    for off, tb in mem_taint_bytes.items():
        shadow.write_mask(BASE + off, tb, 1)
    # Concrete memory values (for mem-value reads in RMW cells).
    concrete = {BASE + i: (0x11 * (i + 1)) & 0xFF for i in range(8)}
    mem_reader = _mem_reader_factory(concrete)

    ctx = EvalContext(
        input_values=dict(values), input_taint=dict(taint),
        simulator=sim, implicit_policy=ImplicitTaintPolicy.KEEP,
        shadow_memory=shadow, mem_reader=mem_reader,
    )
    ref = circ.evaluate(ctx)
    compiled = circ._compiled
    assert compiled not in (None, False), f'{hx} did not compile'

    cmem = compiled.evaluate_c_mem(dict(taint), dict(values), sim._pcode, shadow, mem_reader)
    return ref, cmem, compiled


@pytest.mark.parametrize('label,hx', [(c[1], c[2]) for c in CASES])
@pytest.mark.parametrize('rax_taint', [0x00, 0xFF, FULL, 0xF0F0])
@pytest.mark.parametrize('mem_taint', [
    {},                         # clean memory
    {0: 0xFF},                  # low byte tainted
    {0: 0xFF, 1: 0xFF, 2: 0xFF, 3: 0xFF, 4: 0xFF, 5: 0xFF, 6: 0xFF, 7: 0xFF},  # fully tainted
    {2: 0x0F, 5: 0xF0},         # scattered partial
])
def test_c_mem_matches_do_evaluate(label: str, hx: str, rax_taint: int, mem_taint) -> None:
    ref, cmem, _ = _run_case(Architecture.AMD64, hx, rax_taint=rax_taint, mem_taint_bytes=mem_taint)
    if cmem is None:
        pytest.skip(f'{label}: evaluate_c_mem bailed (falls back to do_evaluate)')
    assert cmem == ref, (
        f'{label} rax_taint={rax_taint:#x} mem_taint={mem_taint}:\n'
        f'  cmem={ {k: hex(v) for k, v in cmem.items()} }\n'
        f'  ref ={ {k: hex(v) for k, v in ref.items()} }'
    )


def _run_case_ptr(arch: Architecture, hx: str, *, rax_taint: int, mem_taint_bytes: dict[int, int]):
    """Mirror of _run_case for evaluate_c_mem_ptr: register taint/values live in
    raw uint64 C arrays (ctypes) indexed by slot; the eval writes reg targets to
    the taint array and mem targets to the shadow, returning the mem writes.
    Returns (ref, got_reg, got_mem, bailed) where got_reg is the reg-taint dict
    read back from the array and got_mem maps (addr,size)->taint from the writes."""
    import ctypes
    regs = list(isa_registers(arch))
    names = [r.name for r in regs]
    sim = CellSimulator(arch)
    circ = generate_static_rule(arch, bytes.fromhex(hx), regs)
    values = {r.name: 0 for r in regs}
    taint = {r.name: 0 for r in regs}
    for name in ('RBX', 'X1', 'r3', '4'):
        if name in values:
            values[name] = BASE
    for name in ('RAX', 'X0', 'r4', '5'):
        if name in taint:
            taint[name] = rax_taint
            values[name] = 0xDEADBEEFCAFEBABE
    concrete = {BASE + i: (0x11 * (i + 1)) & 0xFF for i in range(8)}
    mem_reader = _mem_reader_factory(concrete)

    # Reference on a fresh shadow.
    shadow_ref = BitPreciseShadowMemory()
    for off, tb in mem_taint_bytes.items():
        shadow_ref.write_mask(BASE + off, tb, 1)
    ctx = EvalContext(input_values=dict(values), input_taint=dict(taint),
                      simulator=sim, implicit_policy=ImplicitTaintPolicy.KEEP,
                      shadow_memory=shadow_ref, mem_reader=mem_reader)
    ref = circ.evaluate(ctx)
    compiled = circ._compiled
    assert compiled not in (None, False), f'{hx} did not compile'

    # Engine on an identically-seeded fresh shadow, via the pointer path.
    shadow = BitPreciseShadowMemory()
    for off, tb in mem_taint_bytes.items():
        shadow.write_mask(BASE + off, tb, 1)
    K = len(names)
    TArr = ctypes.c_uint64 * K
    t_arr = TArr(*[taint[n] & FULL for n in names])
    v_arr = TArr(*[values[n] & FULL for n in names])
    slot_map = {n: i for i, n in enumerate(names)}
    writes = compiled.evaluate_c_mem_ptr(ctypes.addressof(t_arr), ctypes.addressof(v_arr),
                                         K, sim._pcode, shadow, mem_reader, slot_map)
    if writes is None:
        return ref, None, None, True
    got_reg = {names[i]: int(t_arr[i]) for i in range(K)}
    got_mem = {(int(a), int(s)): int(t) for (a, s, t) in writes}
    return ref, got_reg, got_mem, False


@pytest.mark.parametrize('label,hx', [(c[1], c[2]) for c in CASES])
@pytest.mark.parametrize('rax_taint', [0x00, 0xFF, FULL, 0xF0F0])
@pytest.mark.parametrize('mem_taint', [
    {}, {0: 0xFF},
    {0: 0xFF, 1: 0xFF, 2: 0xFF, 3: 0xFF, 4: 0xFF, 5: 0xFF, 6: 0xFF, 7: 0xFF},
    {2: 0x0F, 5: 0xF0},
])
def test_c_mem_ptr_matches_do_evaluate(label: str, hx: str, rax_taint: int, mem_taint) -> None:
    """evaluate_c_mem_ptr's array+shadow output must equal the differential."""
    ref, got_reg, got_mem, bailed = _run_case_ptr(
        Architecture.AMD64, hx, rax_taint=rax_taint, mem_taint_bytes=mem_taint)
    if bailed:
        pytest.skip(f'{label}: evaluate_c_mem_ptr bailed (falls back)')
    # Split the reference into register + MEM_ parts.
    ref_reg, ref_mem = {}, {}
    for k, v in ref.items():
        if isinstance(k, str) and k.startswith('MEM_'):
            body = k[4:]
            cut = body.rfind('_')
            ref_mem[(int(body[:cut], 16), int(body[cut + 1:]))] = int(v)
        else:
            ref_reg[k] = int(v)
    # Registers: every reg the reference reports must match the array read-back.
    for name, rv in ref_reg.items():
        gv = got_reg.get(name, 0)
        assert gv == rv, (f'{label} rax_taint={rax_taint:#x} mem_taint={mem_taint} '
                          f'reg {name}: got {gv:#x} != ref {rv:#x}')
    # Memory writes: same set of (addr,size) with same taint.
    assert got_mem == ref_mem, (
        f'{label} rax_taint={rax_taint:#x} mem_taint={mem_taint}:\n'
        f'  got_mem={ {k: hex(v) for k, v in got_mem.items()} }\n'
        f'  ref_mem={ {k: hex(v) for k, v in ref_mem.items()} }')


def test_c_mem_arms_per_evaluate_frame_recycle() -> None:
    """Regression: evaluate_c_mem must arm the per-evaluate cell frame-recycle
    cache (reset_frame_cache), exactly like do_evaluate.

    The frame pool is shared across instructions on one kernel; do_evaluate resets
    it at the start of every evaluate so each instruction's cells re-execute.  If
    evaluate_c_mem skips that reset, the pool left by the previous instruction is
    reused (bit-exact, since the cache is keyed by (cell, inputs) and cells are
    pure -- so the TAINT stays correct), but the cell-execution cadence collapses:
    repeated identical calls hit the stale pool frame and execute ~0 cells instead
    of re-executing per call.  The per-case tests above each build a fresh kernel,
    so they can't see this; drive ONE persistent kernel and pin the cadence.

    This test fails (count_cmem ~= 0) if the reset_frame_cache arming block in
    CompiledCircuit_evaluate_c_mem is removed.
    """
    regs = list(isa_registers(Architecture.AMD64))
    sim = CellSimulator(Architecture.AMD64)
    circ = generate_static_rule(Architecture.AMD64, bytes.fromhex('480103'), regs)  # add [rbx],rax
    values = {r.name: 0 for r in regs}
    values['RBX'] = BASE
    values['RAX'] = 0x1234
    taint = {r.name: 0 for r in regs}
    taint['RAX'] = 0xFF
    shadow = BitPreciseShadowMemory()
    shadow.write_mask(BASE, 0xFF, 1)
    reader = _mem_reader_factory({BASE + i: 0x11 for i in range(8)})

    def make_ctx():
        return EvalContext(
            input_values=dict(values), input_taint=dict(taint),
            simulator=sim, implicit_policy=ImplicitTaintPolicy.KEEP,
            shadow_memory=shadow, mem_reader=reader,
        )

    circ.evaluate(make_ctx())  # warm / compile
    compiled = circ._compiled
    assert compiled not in (None, False)
    assert getattr(compiled, 'has_mem_ops', 0), 'expected a memory circuit with cells'

    kernel = sim._pcode
    if getattr(kernel, 'native_calls', None) is None:
        pytest.skip('kernel does not expose a cell-execution counter')

    N = 15
    # Reference cadence: do_evaluate resets the pool per call -> N re-executions.
    base = kernel.native_calls
    for _ in range(N):
        circ.evaluate(make_ctx())
    count_ref = kernel.native_calls - base
    assert count_ref > 0, 'do_evaluate executed no cells (circuit has no cell?)'

    # evaluate_c_mem must show the SAME cadence.  Without the arming it reuses the
    # pool frame left by the reference phase and executes ~0 cells.
    base = kernel.native_calls
    for _ in range(N):
        compiled.evaluate_c_mem(dict(taint), dict(values), kernel, shadow, reader)
    count_cmem = kernel.native_calls - base
    assert count_cmem == count_ref, (
        f'evaluate_c_mem cell-execution cadence {count_cmem} != do_evaluate {count_ref}: '
        'the per-evaluate frame-recycle reset (reset_frame_cache) is not armed, so '
        'the shared cell frame pool is reused across instructions instead of reset '
        'per call.'
    )


def test_c_mem_persistent_kernel_sequence() -> None:
    """evaluate_c_mem stays bit-identical to do_evaluate across an interleaved
    SEQUENCE of memory instructions sharing ONE kernel + shadow (so the cell
    frame-recycle pool persists), guarding the shared-pool path the fresh-sim
    per-case tests never exercise.
    """
    regs = list(isa_registers(Architecture.AMD64))
    sim = CellSimulator(Architecture.AMD64)
    shadow = BitPreciseShadowMemory()
    # Seed some memory taint + concrete values shared by both paths.
    for i in range(16):
        shadow.write_mask(BASE + i, (0xFF if i % 2 == 0 else 0x0F), 1)
    reader = _mem_reader_factory({BASE + i: (0x11 * (i + 1)) & 0xFF for i in range(16)})

    hexes = ['488b03', '488903', '480303', '480103', '482103', '483303', '482903']
    circuits = {hx: generate_static_rule(Architecture.AMD64, bytes.fromhex(hx), regs) for hx in hexes}

    for step in range(40):
        hx = hexes[step % len(hexes)]
        circ = circuits[hx]
        values = {r.name: 0 for r in regs}
        values['RBX'] = BASE + (step % 8)          # vary the effective address
        values['RAX'] = (0xDEAD * (step + 1)) & 0xFFFFFFFFFFFFFFFF
        taint = {r.name: 0 for r in regs}
        taint['RAX'] = (0xFF << (step % 5)) & 0xFFFFFFFFFFFFFFFF
        ctx = EvalContext(
            input_values=dict(values), input_taint=dict(taint),
            simulator=sim, implicit_policy=ImplicitTaintPolicy.KEEP,
            shadow_memory=shadow, mem_reader=reader,
        )
        ref = circ.evaluate(ctx)
        cmem = circ._compiled.evaluate_c_mem(dict(taint), dict(values), sim._pcode, shadow, reader)
        if cmem is None:
            continue
        assert cmem == ref, (
            f'step {step} {hx}: persistent-kernel divergence\n'
            f'  cmem={ {k: hex(v) for k, v in cmem.items()} }\n'
            f'  ref ={ {k: hex(v) for k, v in ref.items()} }'
        )


def test_c_mem_actually_used_for_loads_and_stores() -> None:
    # At least the plain load and store must take the C path (not always bail),
    # else the optimization is inert.
    took = []
    for hx in ('488b03', '488903', '480303'):
        _, cmem, _ = _run_case(Architecture.AMD64, hx, rax_taint=FULL,
                               mem_taint_bytes={0: 0xFF, 1: 0xFF})
        took.append(cmem is not None)
    assert any(took), 'evaluate_c_mem bailed on every load/store; C path is inert'
