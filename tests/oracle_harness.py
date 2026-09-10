"""Oracle harness for the unify-taint-and-execution rework (Phase 0).

The safety net every rework phase is gated against. It provides TWO oracles and
a corpus driver over the multi-ISA instruction bank:

  1. reference_taint(...)  -- the CURRENT whole-instruction differential
     (circuit.evaluate).  This is the BIT-EXACT reference: interface/rep changes
     (Phases 1-2) must reproduce it exactly; the per-op compaction (Phase 3) is
     diffed against it to MEASURE precision drift.

  2. ground_truth(...)     -- per-bit Unicorn sensitivity (flip each tainted
     input bit, XOR outputs, OR).  The TRUE oracle: soundness = the engine mask
     must CONTAIN the ground-truth mask (no under-taint); over-taint is the
     precision cost, measured but allowed.

An `engine_fn` is any callable
    engine_fn(arch, code, regs, in_taint, in_values, *, circuit) -> dict
returning an output-taint dict keyed by register name.  `reference_taint` is
itself a valid engine_fn (used for harness self-test).  Future engines (array
rep, per-op compaction) plug in the same way.

This module is import-only (no test_ prefix); tests/test_oracle_harness.py runs a
bounded gate, and it can be driven standalone for full-bank sweeps:
    .venv/bin/python -m tests.oracle_harness --isas AMD64 --vectors 8
"""
# ruff: noqa: PLC0415  (deferred imports: unicorn and the engine are
# optional at collection time and expensive to import eagerly)
from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from types import SimpleNamespace

from microtaint.instrumentation.ast import EvalContext, LogicCircuit
from microtaint.simulator import CellSimulator
from microtaint.sleigh.engine import generate_static_rule
from microtaint.types import Architecture, ImplicitTaintPolicy, Register

#: A per-output mask keyed by register or flag name.
TaintState = dict[str, int]

MASK64 = 0xFFFFFFFFFFFFFFFF


# ---------------------------------------------------------------------------
# Oracle 1: the current whole-instruction differential (bit-exact reference).
# ---------------------------------------------------------------------------

def build_circuit(arch: Architecture, code: bytes,
                  regs: list[Register]) -> LogicCircuit:
    """Generate the taint circuit for one instruction form (uncached, so a
    harness run never depends on LRU state)."""
    return generate_static_rule(arch, code, list(regs))


def reference_taint(arch: Architecture, code: bytes, regs: list[Register],
                    in_taint: TaintState, in_values: TaintState, *,
                    circuit: LogicCircuit = None) -> TaintState:
    """Oracle 1: circuit.evaluate -- the current whole-instruction differential.
    Register-only (no shadow); memory forms are filtered by the corpus driver."""
    if circuit is None:
        circuit = build_circuit(arch, code, regs)
    sim = CellSimulator(arch)
    ctx = EvalContext(
        input_taint=dict(in_taint),
        input_values=dict(in_values),
        simulator=sim,
        implicit_policy=ImplicitTaintPolicy.IGNORE,
    )
    out: TaintState = circuit.evaluate(ctx)
    return out


# ---------------------------------------------------------------------------
# Engine adapters.  An engine_fn has the signature
#   (arch, code, regs, in_taint, in_values, *, circuit) -> out_taint dict
# reference_taint is one; this is the template for plugging in a new engine.
# ---------------------------------------------------------------------------

def engine_evaluate_c(arch: Architecture, code: bytes, regs: list[Register],
                      in_taint: TaintState, in_values: TaintState, *,
                      circuit: LogicCircuit = None) -> TaintState:
    """The CURRENT C register fast path (CompiledCircuit.evaluate_c), falling
    back to the differential where it declines (None: mem / PC / wide).  Proves
    the harness detects a real (non-identity) engine matching the oracle, and is
    the shape every future engine adapter takes."""
    if circuit is None:
        circuit = build_circuit(arch, code, regs)
    sim = CellSimulator(arch)
    comp = getattr(circuit, '_compiled', None)
    if comp is None:
        # populate _compiled via one evaluate, then retry
        reference_taint(arch, code, regs, in_taint, in_values, circuit=circuit)
        comp = getattr(circuit, '_compiled', None)
    if comp is not None and comp is not False:
        out: TaintState | None = comp.evaluate_c(
            dict(in_taint), dict(in_values), sim._pcode)
        if out is not None:
            return out
    return reference_taint(arch, code, regs, in_taint, in_values, circuit=circuit)


def engine_evaluate_c_arr(arch: Architecture, code: bytes,
                          regs: list[Register], in_taint: TaintState,
                          in_values: TaintState, *, circuit: LogicCircuit = None) -> TaintState:
    """The array-gather register path (CompiledCircuit.evaluate_c_arr): register
    taint/values are passed as slot-indexed lists (slot = position in `regs`), so
    the per-op input fill is an array gather, not a dict hash lookup.  Returns
    only the computed targets; pass-through (untouched registers keep their input
    taint) is reconstructed here.  Falls back to the differential where the path
    declines (mem / PC / non-c_evaluable)."""
    if circuit is None:
        circuit = build_circuit(arch, code, regs)
    sim = CellSimulator(arch)
    comp = getattr(circuit, '_compiled', None)
    if comp is None:
        reference_taint(arch, code, regs, in_taint, in_values, circuit=circuit)
        comp = getattr(circuit, '_compiled', None)
    if comp is not None and comp is not False:
        names = [r.name for r in regs]
        slot_map = {n: i for i, n in enumerate(names)}
        taint_list = [int(in_taint.get(n, 0)) for n in names]
        val_list = [int(in_values.get(n, 0)) for n in names]
        targets = comp.evaluate_c_arr(taint_list, val_list, sim._pcode, slot_map)
        if targets is not None:
            out = {n: int(in_taint.get(n, 0)) for n in names}  # pass-through
            out.update(targets)                                # overlay computed targets
            return out
    return reference_taint(arch, code, regs, in_taint, in_values, circuit=circuit)


def engine_evaluate_c_arr_ptr(arch: Architecture, code: bytes,
                              regs: list[Register], in_taint: TaintState,
                              in_values: TaintState, *,
                              circuit: LogicCircuit = None) -> TaintState:
    """The live-usable pointer form (evaluate_c_arr_ptr): register taint/values
    live in raw uint64 C arrays (here ctypes arrays), indexed by slot; the eval
    writes target slots of the taint array in place (atomic).  Reads the array
    back to form the output.  Falls back where the path declines."""
    import ctypes
    if circuit is None:
        circuit = build_circuit(arch, code, regs)
    sim = CellSimulator(arch)
    comp = getattr(circuit, '_compiled', None)
    if comp is None:
        reference_taint(arch, code, regs, in_taint, in_values, circuit=circuit)
        comp = getattr(circuit, '_compiled', None)
    if comp is not None and comp is not False:
        names = [r.name for r in regs]
        slot_map = {n: i for i, n in enumerate(names)}
        K = len(names)
        TArr = ctypes.c_uint64 * K
        t_arr = TArr(*[int(in_taint.get(n, 0)) & MASK64 for n in names])
        v_arr = TArr(*[int(in_values.get(n, 0)) & MASK64 for n in names])
        rc = comp.evaluate_c_arr_ptr(ctypes.addressof(t_arr), ctypes.addressof(v_arr),
                                     K, sim._pcode, slot_map)
        if rc is not None:
            return {names[i]: int(t_arr[i]) for i in range(K)}  # array is the state
    return reference_taint(arch, code, regs, in_taint, in_values, circuit=circuit)


# ---------------------------------------------------------------------------
# Oracle 2: per-bit Unicorn sensitivity (true ground truth), per ISA.
# ---------------------------------------------------------------------------

@dataclass
class UcDesc:
    """Everything the Unicorn ground truth needs for one ISA."""
    uc_arch: int
    uc_mode: int
    code_addr: int
    gp: dict[str, int]  # reg-name -> unicorn reg const (GP regs to track)
    flags: dict[str, int] = field(default_factory=dict)  # flag-name -> bit index within eflags_reg
    eflags_reg: int | None = None              # unicorn const for the flags register
    mask: int = MASK64
    #: Which ISA this describes.  The memory bank sets it and reads it back
    #: to find the pointer register; it used to be grafted on from outside,
    #: so a desc that skipped the graft raised AttributeError instead.
    tag: str = ''


def _uc_desc_amd64() -> UcDesc:
    import unicorn
    import unicorn.x86_const as ux
    return UcDesc(
        uc_arch=unicorn.UC_ARCH_X86, uc_mode=unicorn.UC_MODE_64, code_addr=0x1000,
        gp={'RAX': ux.UC_X86_REG_RAX, 'RBX': ux.UC_X86_REG_RBX,
            'RCX': ux.UC_X86_REG_RCX, 'RDX': ux.UC_X86_REG_RDX,
            'RSI': ux.UC_X86_REG_RSI, 'RDI': ux.UC_X86_REG_RDI},
        flags={'CF': 0, 'PF': 2, 'AF': 4, 'ZF': 6, 'SF': 7, 'OF': 11},
        eflags_reg=ux.UC_X86_REG_EFLAGS,
    )


def _uc_desc_arm64() -> UcDesc:
    import unicorn
    import unicorn.arm64_const as ua
    # AArch64 PSTATE condition flags live in NZCV: N=31, Z=30, C=29, V=28.
    # Bank names them N/Z/C/V (pypcode/Ghidra: NG/ZR/CY/OV); the study's
    # register resolver bridges the spelling, so GT keys use the bank names.
    return UcDesc(
        uc_arch=unicorn.UC_ARCH_ARM64, uc_mode=unicorn.UC_MODE_ARM, code_addr=0x1000,
        gp={'X0': ua.UC_ARM64_REG_X0, 'X1': ua.UC_ARM64_REG_X1,
            'X2': ua.UC_ARM64_REG_X2, 'X3': ua.UC_ARM64_REG_X3,
            'X4': ua.UC_ARM64_REG_X4, 'X5': ua.UC_ARM64_REG_X5},
        flags={'N': 31, 'Z': 30, 'C': 29, 'V': 28},
        eflags_reg=ua.UC_ARM64_REG_NZCV,
    )


def _uc_desc_riscv64() -> UcDesc:
    import unicorn
    import unicorn.riscv_const as ur
    # RISC-V has no architectural condition-flag register (compares write GP
    # registers), so flags is empty.
    #
    # Track the temporaries and saved registers as well as the arguments.  a0-a5
    # alone looked like enough and is not: the RISCV64 instruction bank assembles
    # its forms around t0/t1/t2, so a ground truth that watched only a0-a5 could
    # not see the destination of a single one of them and agreed with anything.
    names = ('T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6',
             'S0', 'S1', 'S2', 'S3', 'S4', 'S5',
             'A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7',
             'RA', 'SP', 'GP', 'TP')
    return UcDesc(
        uc_arch=unicorn.UC_ARCH_RISCV, uc_mode=unicorn.UC_MODE_RISCV64, code_addr=0x1000,
        gp={n: getattr(ur, f'UC_RISCV_REG_{n}') for n in names
            if hasattr(ur, f'UC_RISCV_REG_{n}')},
        flags={},
        eflags_reg=None,
    )


def _uc_desc_mips64be() -> UcDesc:
    import unicorn
    import unicorn.mips_const as um
    # No architectural condition-flag register: MIPS comparisons write a GP
    # register, so flags is empty.  Track the temporaries, arguments, saved and
    # value registers, which is what the bank assembles its forms around.
    names = ('AT', 'V0', 'V1', 'A0', 'A1', 'A2', 'A3',
             'T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9',
             'S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7',
             'GP', 'SP', 'FP', 'RA')
    return UcDesc(
        uc_arch=unicorn.UC_ARCH_MIPS,
        uc_mode=unicorn.UC_MODE_MIPS64 | unicorn.UC_MODE_BIG_ENDIAN,
        code_addr=0x1000,
        gp={n: getattr(um, f'UC_MIPS_REG_{n}') for n in names
            if hasattr(um, f'UC_MIPS_REG_{n}')},
        flags={},
        eflags_reg=None,
    )


def _uc_desc_ppc32be() -> UcDesc:
    import unicorn
    import unicorn.ppc_const as up

    def const(name: str) -> int | None:
        """Unicorn spells the general registers UC_PPC_REG_0..31; the engine's
        geometry spells them R0..R31.  Matching on the literal name finds only
        CTR, LR, MSR, PC and XER, and a ground truth that watches five special
        registers agrees with anything -- which is exactly what the first run of
        this reported before the mapping was added."""
        c: int | None = getattr(up, f'UC_PPC_REG_{name}', None)
        if c is not None:
            return c
        if len(name) > 1 and name[0] == 'R' and name[1:].isdigit():
            alt: int | None = getattr(up, f'UC_PPC_REG_{name[1:]}', None)
            return alt
        return None

    names = [f'R{i}' for i in range(32)] + ['LR', 'CTR', 'XER']
    gp = {n: const(n) for n in names}
    return UcDesc(
        uc_arch=unicorn.UC_ARCH_PPC,
        uc_mode=unicorn.UC_MODE_PPC32 | unicorn.UC_MODE_BIG_ENDIAN,
        code_addr=0x1000,
        gp={n: c for n, c in gp.items() if c is not None},
        flags={},                 # CR fields are not modelled here
        eflags_reg=None,
        mask=0xFFFFFFFF,          # 32-bit registers
    )


UC_DESCS = {'AMD64': _uc_desc_amd64, 'ARM64': _uc_desc_arm64,
            'MIPS64BE': _uc_desc_mips64be, 'PPC32BE': _uc_desc_ppc32be,
            'RISCV64': _uc_desc_riscv64}


def _uc_run(desc: UcDesc, code: bytes, vals: dict[str, int], *,
            seed_flags: bool = False) -> dict[str, int]:
    import unicorn
    uc = unicorn.Uc(desc.uc_arch, desc.uc_mode)
    uc.mem_map(desc.code_addr, 0x2000)
    uc.mem_write(desc.code_addr, code)
    for name, const in desc.gp.items():
        uc.reg_write(const, vals.get(name, 0) & desc.mask)
    if seed_flags and desc.eflags_reg is not None:
        # Start the flags where the cell's frame starts them: at zero.  Only
        # `models_disagree` asks for this, and it has to -- comparing an
        # unseeded flag against the frame's zero says the two models differ for
        # every instruction that does not touch the flag, which is most of
        # AArch64.  `ground_truth` does not need it (it XORs two runs, so a
        # constant seed cancels) and does not get it, so its numbers do not move.
        uc.reg_write(desc.eflags_reg, 0)
    uc.emu_start(desc.code_addr, desc.code_addr + len(code))
    out = {name: uc.reg_read(const) & desc.mask for name, const in desc.gp.items()}
    if desc.eflags_reg is not None:
        ef = uc.reg_read(desc.eflags_reg)
        for fname, bit in desc.flags.items():
            out[fname] = (ef >> bit) & 1
    return out


def models_disagree(desc: UcDesc, arch: Architecture, code: bytes,
                    states: Iterable[TaintState]) -> tuple[set[str], set[str]]:
    """Outputs where SLEIGH and Unicorn model this instruction differently.

    Where an ISA leaves a flag ARCHITECTURALLY UNDEFINED -- x86 OF after a
    rotate or shift by anything other than one, AF after most arithmetic -- the
    two model it differently, and both are entitled to.  Ghidra computes one
    thing, QEMU another.  Neither is the specification, because the
    specification declines to say.

    That matters because the ground-truth oracle IS QEMU.  Asking whether the
    engine's taint covers a bit QEMU moves, on a flag the manual says has no
    defined value, measures the disagreement between two vendors rather than
    anything about the engine.  Measured on `rol rax, 31`: flipping bit 33 of
    RAX moves QEMU's OF, and the engine's SLEIGH-derived rule does not taint it,
    which the campaign then reports as an under-taint.

    So they are DETECTED rather than listed.  A hand-written table of undefined
    flags per instruction goes stale the moment the bank grows and cannot cover
    a new ISA at all; running both models over the states the campaign is about
    to use costs one extra concrete execution per state and is exact for
    whatever the two disagree on, on any architecture.

    Returns (undefined, unmodelled), and the difference is worth keeping:

      undefined  -- the lifter WRITES the flag and the two models disagree
                    about its value.  x86 OF after `rol rax, 31`.  The manual
                    declines to say, so neither is wrong and it is not the
                    engine's question to answer.
      unmodelled -- the lifter never writes the flag at all.  x86 AF after
                    `add`, which the ISA DOES define and SLEIGH simply does not
                    compute.  That is a real gap, and folding it into the first
                    would hide it behind a name that sounds like somebody
                    else's problem.

    Both leave the verdict, because the engine cannot answer for either.  Only
    one of them is nobody's fault.  Report them apart.
    """
    from microtaint.simulator import CellSimulator

    if not desc.flags:
        return set(), set()               # nothing undefined to find
    sim = CellSimulator(arch)
    # Only the C evaluator has `evaluate_concrete_flat`.  This used to be a bare
    # attribute reach inside `except Exception: continue`, so on the Cython
    # evaluator every flag was skipped, nothing was ever compared, and the
    # function returned two empty sets -- an exclusion list that excludes
    # nothing, indistinguishable from an instruction the two models agree on.
    flat = getattr(sim._pcode, 'evaluate_concrete_flat', None)
    if flat is None:
        raise RuntimeError(
            f'{type(sim._pcode).__name__} has no evaluate_concrete_flat, so '
            f'this function cannot compare the two models at all')
    hx = code.hex()
    written = _flags_the_lifter_writes(arch, code, desc)
    # And the cell has to be asked by the ENGINE's name too.  Asking it for `N`
    # when the geometry calls the register `NG` resolves to nothing, the read
    # comes back zero, and every flag-setting AArch64 instruction looks like a
    # disagreement -- which is exactly what it looked like.
    engine_name = _engine_flag_names(arch, desc)
    disagree: set[str] = set()
    for vals in states:
        try:
            truth = _uc_run(desc, code, vals, seed_flags=True)
        except Exception:
            continue
        for fname in desc.flags:
            if fname in disagree:
                continue
            # A duck-type on purpose: the evaluator reads only these four
            # fields, and building a real InstructionCellExpr here would tie
            # the harness to the lifter it is supposed to be checking.
            cell = SimpleNamespace(
                instruction=hx, out_reg=engine_name.get(fname, fname),
                out_bit_start=0, out_bit_end=0)
            try:
                mine = flat(cell, dict(vals))
            except Exception:
                # This ONE flag has no cell for this instruction, which is
                # ordinary: measured over the AMD64 bank, 43 of the first 400
                # instructions answer for no flag at all -- every `push`, and
                # `adcx`/`adox`, whose CF/OF the lifter writes but the cell
                # declines.  So a per-instruction "compared nothing" is not
                # evidence of anything.  What IS evidence is the evaluator
                # having no `evaluate_concrete_flat` at all, and that is
                # refused above rather than swallowed here.
                continue
            if (int(mine) & 1) != (truth.get(fname, 0) & 1):
                disagree.add(fname)
    return (disagree & written), (disagree - written)


def _engine_flag_names(arch: Architecture, desc: UcDesc) -> dict[str, str]:
    """Descriptor flag name -> the name this architecture's geometry uses."""
    from microtaint.debug.reg_aliases import RegisterAliases
    from microtaint.instrumentation.cell import _build_reg_maps

    offsets, _sizes = _build_reg_maps(arch)
    try:
        alias = RegisterAliases(arch)
    except Exception:
        alias = None
    out = {}
    for f in desc.flags:
        if f.upper() in offsets:
            out[f] = f.upper()
            continue
        if alias is not None:
            try:
                for n in alias.to_engine_names(f) or []:
                    if n.upper() in offsets:
                        out[f] = n.upper()
                        break
            except Exception:
                pass
    return out


def _flags_the_lifter_writes(arch: Architecture, code: bytes,
                             desc: UcDesc) -> set[str]:
    """Which of this architecture's flags the p-code for `code` assigns."""
    # The descriptor names flags the way a person does (AArch64 N/Z/C/V); the
    # geometry names them the way SLEIGH does (NG/ZR/CY/OV).  Resolving through
    # the alias helper rather than assuming they match is the difference between
    # "adds writes no flags" -- which is what a direct lookup concluded, for
    # every flag-setting AArch64 instruction in the bank -- and the truth.
    from microtaint.debug.reg_aliases import RegisterAliases
    from microtaint.instrumentation.cell import _build_reg_maps
    from microtaint.sleigh.lifter import get_context

    offsets, _sizes = _build_reg_maps(arch)
    try:
        alias = RegisterAliases(arch)
    except Exception:
        alias = None
    want: dict[int, str] = {}
    for f in desc.flags:
        names = [f]
        if alias is not None:
            try:
                names = alias.to_engine_names(f) or [f]
            except Exception:
                names = [f]
        for n in names:
            off = offsets.get(n.upper())
            if off is not None:
                want[off] = f
    try:
        tx = get_context(arch).translate(code, base_address=desc.code_addr)
    except Exception:
        return set(desc.flags)            # cannot tell: assume it writes them
    out = set()
    for op in tx.ops:
        o = op.output
        if o is not None and o.space.name == 'register' and o.offset in want:
            out.add(want[o.offset])
    return out


def ground_truth(desc: UcDesc, code: bytes, in_taint: dict[str, int],
                 in_values: dict[str, int]) -> dict[str, int]:
    """Oracle 2: per-bit sensitivity via Unicorn.  For every tainted input bit,
    flip it (from the clean base) and OR the output XOR into the result mask.
    Returns a per-output taint mask (GP regs + flags)."""
    base_vals = {n: (in_values.get(n, 0) & ~in_taint.get(n, 0) & desc.mask) for n in desc.gp}
    base_out = _uc_run(desc, code, base_vals)
    result: dict[str, int] = dict.fromkeys(base_out, 0)
    for src in desc.gp:
        tm = in_taint.get(src, 0) & desc.mask
        bit = 0
        while tm:
            if tm & 1:
                flipped = dict(base_vals)
                flipped[src] = (base_vals[src] | (1 << bit)) & desc.mask
                out = _uc_run(desc, code, flipped)
                for k in base_out:
                    result[k] |= base_out[k] ^ out[k]
            tm >>= 1
            bit += 1
    return result


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

@dataclass
class Verdict:
    exact: bool
    under: dict[str, int] = field(default_factory=dict)   # reg -> bits present in ref/truth, missing in got
    over: dict[str, int] = field(default_factory=dict)    # reg -> bits in got not in ref/truth

    @property
    def sound(self) -> bool:
        return not self.under

    @property
    def n_over(self) -> int:
        return sum(bin(v).count('1') for v in self.over.values())

    @property
    def n_under(self) -> int:
        return sum(bin(v).count('1') for v in self.under.values())


def classify(got: TaintState, ref: TaintState, keys: Iterable[str]) -> Verdict:
    """Compare an engine's output mask `got` against a reference `ref` over
    `keys` (register/flag names).  under = ref & ~got, over = got & ~ref."""
    under: dict[str, int] = {}
    over: dict[str, int] = {}
    for k in keys:
        g = int(got.get(k, 0) or 0)
        r = int(ref.get(k, 0) or 0)
        u = r & ~g
        o = g & ~r
        if u:
            under[k] = u
        if o:
            over[k] = o
    return Verdict(exact=(not under and not over), under=under, over=over)


# ---------------------------------------------------------------------------
# Input-vector fuzzing
# ---------------------------------------------------------------------------

def fuzz_vectors(reg_names: list[str], rng: random.Random, n_dense: int,
                 n_sparse: int) -> Iterator[tuple[TaintState, TaintState]]:
    """Yield (in_taint, in_values) pairs.  Dense masks stress the differential;
    sparse masks (few bits) are what the Unicorn ground truth can enumerate
    cheaply.  Values avoid 0 to dodge div-by-zero-class faults."""
    def rand_vals() -> TaintState:
        return {r: rng.randint(1, MASK64) for r in reg_names}

    # edge taint patterns (dense)
    edges = [
        dict.fromkeys(reg_names, MASK64),
        {reg_names[0]: MASK64} if reg_names else {},
        dict.fromkeys(reg_names, 255),
        dict.fromkeys(reg_names, 18446744069414584320),
        dict.fromkeys(reg_names, 12297829382473034410),
    ]
    for t in edges[:n_dense]:
        yield dict(t), rand_vals()
    for _ in range(max(0, n_dense - len(edges))):
        yield {r: rng.randint(0, MASK64) for r in reg_names}, rand_vals()
    # sparse: 1-4 random bits per register (ground-truth-friendly)
    for _ in range(n_sparse):
        t = {}
        for r in reg_names:
            m = 0
            for _b in range(rng.choice([0, 1, 2, 3])):
                m |= 1 << rng.randint(0, 63)
            t[r] = m
        yield t, rand_vals()


# ---------------------------------------------------------------------------
# Corpus driver
# ---------------------------------------------------------------------------

@dataclass
class Report:
    n_cases: int = 0
    n_instrs: int = 0
    n_exact: int = 0
    n_over_only: int = 0
    n_under: int = 0
    over_bits_total: int = 0
    mismatches: list[tuple[str, str, Verdict]] = field(default_factory=list)
    skipped_mem: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f'cases={self.n_cases} instrs={self.n_instrs} '
            f'exact={self.n_exact} over-only={self.n_over_only} UNDER={self.n_under} '
            f'over-bits={self.over_bits_total} mem-skipped={self.skipped_mem} '
            f'errors={len(self.errors)}'
        )


def _compile_and_mem(circuit: LogicCircuit, arch: Architecture,
                     regs: list[Register]) -> bool:
    """`circuit._compiled` is populated lazily on the first evaluate, so a probe
    evaluate (zero taint/values) forces compilation; then has_mem_ops is a
    reliable "reads/writes guest memory" signal (LEA stays False -> register
    corpus keeps it; real load/store/RMW become True -> skipped here)."""
    zero = {r.name: 0 for r in regs}
    sim = CellSimulator(arch)
    try:
        ctx = EvalContext(input_taint=zero, input_values=zero, simulator=sim,
                          implicit_policy=ImplicitTaintPolicy.IGNORE)
        circuit.evaluate(ctx)
    except Exception:
        pass
    c = getattr(circuit, '_compiled', None)
    return bool(c) and getattr(c, 'has_mem_ops', False)


#: `engine_fn(arch, code, regs, in_taint, in_values, *, circuit) -> dict`,
#: the one shape every engine under test is adapted to.
EngineFn = Callable[..., dict[str, int]]


def run_bank(engine_fn: EngineFn, *, isas: list[str] | None = None,
             n_dense: int = 5, n_sparse: int = 8, seed: int = 1234,
             ref: str = 'differential', uc_desc: UcDesc | None = None,
             skip_mem: bool = True, max_mismatch: int = 25) -> Report:
    """Drive `engine_fn` over the instruction bank and compare vs a reference.

    ref='differential' -> compare vs reference_taint (bit-exact gate).
    ref='ground_truth'  -> compare vs Unicorn per-bit (needs uc_desc; soundness
                            + precision).  Uses only the sparse vectors (GT is
                            per-bit-expensive and exact only for few bits).
    """
    from benchmark.instruction_bank import load_bank

    rep = Report()
    specs = load_bank(isas=set(isas) if isas else None)
    for spec in specs.values():
        reg_names = [r.name for r in spec.regs]
        for ins in spec.instructions:
            try:
                circuit = build_circuit(spec.arch, ins.bytes, spec.regs)
            except Exception as e:
                rep.errors.append((ins.label, f'build: {e!r}'))
                continue
            if skip_mem and _compile_and_mem(circuit, spec.arch, spec.regs):
                rep.skipped_mem += 1
                continue
            rep.n_instrs += 1
            rng = random.Random(f'{seed}:{ins.label}')
            for in_taint, in_values in fuzz_vectors(reg_names, rng, n_dense, n_sparse):
                if ref == 'ground_truth':
                    # GT enumerates tainted bits: cap total to keep it cheap/exact.
                    total_bits = sum(bin(v).count('1') for v in in_taint.values())
                    if total_bits == 0 or total_bits > 20:
                        continue
                rep.n_cases += 1
                try:
                    got = engine_fn(spec.arch, ins.bytes, spec.regs,
                                    in_taint, in_values, circuit=circuit)
                    if ref == 'ground_truth':
                        assert uc_desc is not None, \
                            "ref='ground_truth' needs a UcDesc"
                        keys = list(uc_desc.gp) + list(uc_desc.flags)
                        refd = ground_truth(uc_desc, ins.bytes, in_taint, in_values)
                    else:
                        keys = reg_names
                        refd = reference_taint(spec.arch, ins.bytes, spec.regs,
                                               in_taint, in_values, circuit=circuit)
                except Exception as e:
                    rep.errors.append((ins.label, f'eval: {e!r}'))
                    continue
                v = classify(got, refd, keys)
                if v.exact:
                    rep.n_exact += 1
                elif v.sound:
                    rep.n_over_only += 1
                    rep.over_bits_total += v.n_over
                    if len(rep.mismatches) < max_mismatch:
                        rep.mismatches.append((ins.label, 'OVER', v))
                else:
                    rep.n_under += 1
                    if len(rep.mismatches) < max_mismatch:
                        rep.mismatches.append((ins.label, 'UNDER', v))
    return rep


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--isas', nargs='*', default=None)
    ap.add_argument('--dense', type=int, default=5)
    ap.add_argument('--sparse', type=int, default=8)
    ap.add_argument('--ref', choices=['differential', 'ground_truth'], default='differential')
    args = ap.parse_args()
    ud = UC_DESCS['AMD64']() if args.ref == 'ground_truth' else None
    r = run_bank(reference_taint, isas=args.isas, n_dense=args.dense,
                 n_sparse=args.sparse, ref=args.ref, uc_desc=ud)
    print(r.summary())
    for label, kind, v in r.mismatches[:25]:
        print(f'  {kind} {label}: under={v.under} over={v.over}')
    for label, err in r.errors[:10]:
        print(f'  ERR {label}: {err}')
