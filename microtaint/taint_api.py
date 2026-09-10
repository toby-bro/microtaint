"""One instruction's taint, by whichever implementation you asked for.

The engine has two implementations of the same specification, and they are not
interchangeable, so this module makes you name the one you mean instead of
guessing.

    DIFFERENTIAL -- generate a LogicCircuit and evaluate it.  For each output
      (the result slice, and each flag separately) it runs the WHOLE instruction
      concretely twice, on `V|T` and on `V&~T`, and XORs the two.  A bit that
      differs is a bit the attacker can move.  Exact wherever the function is
      monotone across that cube; where it is not, both corners can agree while
      the output varies, which is why the overflow flag needs a table of its own.
      Costs microseconds: a flag-setting instruction is eight outputs and two
      corners, so sixteen whole-instruction executions.

    COMPILED -- lower the whole instruction to one straight-line SSA program in
      a 64-bit IR, constant-fold it, drop what nothing reads, and emit machine
      code for it.  Every output including flags falls out of a single pass, and
      the instruction is never executed at all: values enter as inputs.  Costs
      about eleven nanoseconds.

Both are derived from the same p-code from the same lifter, both compute the
same quantity, and both over-approximate rather than under.  They disagree in
about twelve percent of cases on AMD64, in both directions: the compiled path is
TIGHTER where per-op composition keeps structure the two corners destroy (`push`
keeps RSP's own taint; the whole-instruction differential clears it), and LOOSER
where an operation p-code does not model has to take the avalanche floor while
re-executing it concretely would have been exact.

That disagreement is the reason this is a choice and not a switch: the
differential is what every soundness test is checked against, so it has to stay
reachable and stay the reference, and the compiled path is what production runs.
Asking for one and silently getting the other would quietly turn the oracle into
a mirror.
"""

# ruff: noqa: PLC0415  (deferred imports: the engine modules cycle back here)
from __future__ import annotations

import os
import re
from enum import StrEnum
from typing import Any

from microtaint.types import Architecture, ImplicitTaintPolicy, Register

MASK64 = 0xFFFFFFFFFFFFFFFF
#: Must match MT_IR_MEM_BASE / MEM_SLOTS_PER_ACCESS in fastpath.h and frompcode.
MEM_SLOT_BASE = 512
MEM_SLOTS_PER_ACCESS = 4

#: `MEM_<address>_<size>`, the name a resolved store output carries.
_MEM_OUTPUT = re.compile(r'^MEM_(-?0x[0-9a-fA-F]+)_(\d+)$')

class Path(StrEnum):
    """Which implementation answered, or which one a caller is asking for.

    A StrEnum so a caller may still pass the spelling and a report may
    still print it, while the dispatch below compares members.
    """

    #: The lowered taint program.  What production runs, and what declines
    #: on a shape the lowering does not model.
    COMPILED = 'compiled'
    #: The whole-instruction differential.  Always reachable, and the
    #: reference the compiled path is judged against.
    DIFFERENTIAL = 'differential'

#: What `path=None` means.  The same variable the emulator hook reads, so one
#: run of the suite exercises one implementation from top to bottom.
_ENV = 'MICROTAINT_TAINT_IR'


def default_path() -> Path:
    """The implementation this process uses when the caller does not say.

    `MICROTAINT_TAINT_IR=0` selects the differential; anything else, including
    unset, selects the compiled path.  It sets the DEFAULT, and it is what
    switches the compiled path off inside the emulator -- it does not overrule a
    caller who names an implementation, because comparing the two in a run
    configured for one of them is exactly what the tests need to do.
    """
    return Path.DIFFERENTIAL if os.environ.get(_ENV) == '0' else Path.COMPILED


def taint_step(
    arch: Architecture,
    code: bytes,
    in_taint: dict[str, int],
    in_values: dict[str, int] | None = None,
    *,
    path: Path | None = None,
    state_format: list[Register] | None = None,
    implicit_policy: ImplicitTaintPolicy = ImplicitTaintPolicy.IGNORE,
    memory: object | None = None,
) -> dict[str, int]:
    """Taint after executing `code` once, as {register name: mask}.

    `in_taint` is the taint before, `in_values` the concrete register values it
    sits on; the differential needs those values, and the compiled path needs
    them for any rule that is value-dependent (an AND clears the taint of a bit
    the other operand zeroes, and it can only know that from the value).

    `path` picks the implementation and defaults to `default_path()`.  Asking
    for 'compiled' where the lowering declines -- an instruction p-code models
    as a loop, an emitter that has no rule for an opcode -- falls back to the
    differential rather than failing, which is what the emulator does too.  Ask
    `explain()` if you need to know which one answered.
    """
    return _dispatch(arch, code, in_taint, in_values or {}, path,
                     state_format, implicit_policy, memory)[0]


def explain(
    arch: Architecture,
    code: bytes,
    in_taint: dict[str, int],
    in_values: dict[str, int] | None = None,
    *,
    path: Path | None = None,
    state_format: list[Register] | None = None,
    implicit_policy: ImplicitTaintPolicy = ImplicitTaintPolicy.IGNORE,
    memory: object | None = None,
) -> tuple[dict[str, int], Path]:
    """`taint_step`, plus which implementation actually answered.

    A benchmark that does not check this can report the compiled path's speed
    for an instruction the compiled path declined.
    """
    return _dispatch(arch, code, in_taint, in_values or {}, path,
                     state_format, implicit_policy, memory)


# ---------------------------------------------------------------------------


def _dispatch(arch: Architecture, code: bytes, in_taint: dict[str, int],
              in_values: dict[str, int], path: Path | None,
              state_format: list[Register] | None,
              implicit_policy: ImplicitTaintPolicy,
              memory: Any) -> tuple[dict[str, int], Path]:
    if state_format is None:
        from microtaint.emulator import archregs
        state_format = archregs.state_format(arch)
    want = Path(path) if path is not None else default_path()
    if want is Path.COMPILED:
        got = _compiled(arch, code, in_taint, in_values, state_format, memory)
        if got is not None:
            return got, Path.COMPILED
    return _differential(arch, code, in_taint, in_values, state_format,
                         implicit_policy, memory), Path.DIFFERENTIAL


_LAYOUTS: dict[Any, dict[str, int]] = {}


def _slot_layout(arch: Architecture, state_format: list[Register],
                 in_taint: dict[str, int],
                 in_values: dict[str, int]) -> dict[str, int]:
    """A slot number for every register this architecture tracks.

    It has to cover the whole state, not just the registers the caller named:
    `add rax, rbx` writes five flags, and a layout without them leaves the
    program with outputs it has no slot for -- so it declines, and the caller
    silently gets the differential while believing otherwise.

    One layout per architecture, so a program is compiled once and reused.  The
    layout is part of the program's cache key, and an unstable one would
    recompile on every call and end up measuring the compiler.  Registers the
    caller names that the format does not list are appended in sorted order.
    """
    names = [r.name for r in state_format]
    extra = sorted((set(in_taint) | set(in_values)).difference(names))
    key = (arch, tuple(names), tuple(extra))
    layout = _LAYOUTS.get(key)
    if layout is None:
        layout = {n: i for i, n in enumerate(names + extra)}
        _LAYOUTS[key] = layout
    return layout


def _compiled(arch: Architecture, code: bytes, in_taint: dict[str, int],
              in_values: dict[str, int], state_format: list[Register],
              memory: Any) -> dict[str, int] | None:
    """-> the taint dict, or None if the compiled path declined this one."""
    from microtaint.instrumentation.cell_c import taint_ir_c
    from microtaint.taint_ir.engine_glue import program_for

    layout = _slot_layout(arch, state_format, in_taint, in_values)
    if not layout:
        return None
    got = program_for(arch, code, layout, force=True)
    if got is None:
        return None
    cap, _addr, accesses, _writes_pc, _live = got
    if accesses and memory is None:
        # A load or a store needs its address resolved and its shadow read.
        # Without a memory to ask, decline and let the differential answer --
        # which is what the emulator does when its own shadow is missing.
        return None
    n = MEM_SLOT_BASE + MEM_SLOTS_PER_ACCESS * len(accesses) if accesses else len(layout)
    if len(layout) > MEM_SLOT_BASE:
        return None                       # registers would collide with memory
    values = [0] * n
    taints = [0] * n
    for name, slot in layout.items():
        values[slot] = in_values.get(name, 0) & MASK64
        taints[slot] = in_taint.get(name, 0) & MASK64
    if not accesses:
        out = taint_ir_c.run(cap, values, taints)
        return {name: out[slot] for name, slot in layout.items()}
    return _compiled_with_memory(cap, accesses, layout, values, taints, memory)


def _compiled_with_memory(cap: Any, accesses: Any, layout: dict[str, int],
                          values: list[int], taints: list[int],
                          memory: Any) -> dict[str, int]:
    """The two-pass protocol, the same one `fastpath.h::mt_ir_mem_step` runs.

    A load's address is computed BY the program, so it cannot be known before
    running it, and the taint the load contributes cannot be known before the
    address.  So the program runs twice: once to learn the addresses, then again
    with the loaded words' taint filled in.  The second run is what counts; the
    first is discarded.
    """
    from microtaint.instrumentation.cell_c import taint_ir_c

    # Pass 1: the addresses.
    out = taint_ir_c.run(cap, values, list(taints))
    for k, acc in enumerate(accesses):
        kind, size, needval = acc
        if kind != 0:                                    # a store reads nothing
            continue
        base = MEM_SLOT_BASE + MEM_SLOTS_PER_ACCESS * k
        address = out[base + 1]
        if needval:
            values[base] = memory.read(address, size) & MASK64
        taints[base] = memory.read_mask(address, size) & MASK64

    # Pass 2: the taint, now that the loaded words are known.
    out = taint_ir_c.run(cap, values, taints)

    # A store publishes its mask; the caller's memory has to receive it, or a
    # `push` would leave nothing for the matching `pop` to find.
    for k, acc in enumerate(accesses):
        kind, size, _needval = acc
        if kind != 1:
            continue
        base = MEM_SLOT_BASE + MEM_SLOTS_PER_ACCESS * k
        memory.write_mask(out[base + 1], out[base + 3], size)
    return {name: out[slot] for name, slot in layout.items()}


def _differential(arch: Architecture, code: bytes, in_taint: dict[str, int],
                  in_values: dict[str, int], state_format: list[Register],
                  implicit_policy: ImplicitTaintPolicy,
                  memory: Any) -> dict[str, int]:
    from microtaint.instrumentation.ast import EvalContext
    from microtaint.simulator import CellSimulator
    from microtaint.sleigh.engine import generate_static_rule

    circuit = generate_static_rule(arch, code, list(state_format))
    ectx = EvalContext(
        input_taint=dict(in_taint),
        input_values=dict(in_values),
        simulator=CellSimulator(arch),
        implicit_policy=implicit_policy,
        shadow_memory=memory,
        mem_reader=memory.read if memory is not None else None,
    )
    # The POST-STATE, not just what the instruction wrote: an untouched
    # register keeps the taint it had.  `circuit.evaluate` reports only its
    # own assignments, so overlaying them on the input is what makes the two
    # paths answer the same question in the same shape -- without which
    # "run the suite against either implementation" is not a thing you can do.
    out = circuit.evaluate(ectx)
    # A store arrives as an output named `MEM_<address>_<size>`, with the
    # address already resolved.  `evaluate` does not commit it -- in the
    # emulator the hook does that -- so the API has to, or a `push` leaves
    # nothing for the matching `pop` to read back and the two paths disagree
    # about memory while agreeing about registers.
    if memory is not None:
        for key, mask in out.items():
            hit = _MEM_OUTPUT.match(key)
            if hit:
                memory.write_mask(int(hit.group(1), 16), mask, int(hit.group(2)))
    after = {name: in_taint.get(name, 0)
             for name in _slot_layout(arch, state_format, in_taint, in_values)}
    after.update({k: v for k, v in out.items() if not k.startswith('MEM_')})
    return after


class TaintSequence:
    """Taint through a run of instructions, with the state threaded for you.

    `taint_step` answers about one instruction and leaves the caller to carry
    the state forward.  For a sequence that is not enough: `push rbx` moves the
    stack pointer, and a `pop` that does not see the new RSP reads the wrong
    word.  So this threads three things between steps -- the register taint, the
    register VALUES, and a `TaintMemory` holding both the stored bytes and their
    taint -- which is what makes `push`/`pop` and a spill/reload actually work.

    The taint is computed by whichever implementation `path` selects and stays
    on the fast one.  The concrete values are the expensive half: they come from
    executing the instruction in the cell, once per register it writes, because
    nothing else in the engine knows what `push` does to RSP.  That cost is per
    STEP, not per instruction executed by a program, so a sequence of a few
    dozen instructions is still immediate -- but it is why this is a separate
    class and not the default behaviour of `taint_step`.
    """

    __slots__ = (
        '_paths_used',
        '_sim',
        'arch',
        'memory',
        'path',
        'state_format',
        'taint',
        'values',
    )

    def __init__(self, arch: Architecture, *,
                 values: dict[str, int] | None = None,
                 taint: dict[str, int] | None = None,
                 memory: Any = None,
                 path: Path | None = None,
                 state_format: list[Register] | None = None) -> None:
        from microtaint.emulator import archregs
        from microtaint.simulator import CellSimulator
        from microtaint.taint_memory import TaintMemory

        self.arch = arch
        self.state_format = state_format or archregs.state_format(arch)
        self.values = dict(values or {})
        self.taint = dict(taint or {})
        self.memory = memory if memory is not None else TaintMemory()
        self.path = path
        self._sim = CellSimulator(arch)
        self._paths_used: list[Path] = []

    @property
    def paths_used(self) -> list[Path]:
        """Which implementation answered each step, in order.

        Worth checking before quoting a speed: a step the compiled path
        declined was answered by the differential.
        """
        return list(self._paths_used)

    def step(self, code: bytes) -> dict[str, int]:
        """Advance one instruction; returns the register taint after it."""
        after, used = explain(self.arch, code, self.taint, self.values,
                              path=self.path, state_format=self.state_format,
                              memory=self.memory)
        self._paths_used.append(used)
        self.taint = after
        self._advance_values(code)
        return after

    def run(self, *codes: bytes) -> dict[str, int]:
        """Advance through several instructions; returns the taint after all."""
        after = self.taint
        for code in codes:
            after = self.step(code)
        return after

    def _advance_values(self, code: bytes) -> None:
        """Execute the instruction concretely and carry its writes forward.

        Only what the instruction WRITES is re-read.  The lowering already
        enumerates that, so a `push` costs two concrete executions (RSP and the
        pushed word) rather than one per register in the state format.
        """
        import types as _types

        # The words this instruction loads have to reach the cell as inputs, or
        # a `pop` executes against an empty memory and returns zero.  The taint
        # pass has already resolved those addresses, so they are read back here
        # rather than computed a second time.
        inputs = dict(self.values)
        for address, size in self.memory.take_reads():
            inputs[f'MEM_{address:#x}_{size}'] = self.memory.read(address, size)

        def concrete(out_reg: str, size: int) -> int | None:
            cell = _types.SimpleNamespace(
                instruction=code.hex(), out_reg=out_reg,
                out_bit_start=0, out_bit_end=size * 8 - 1)
            try:
                # The C evaluator has this entry point; the Python one does
                # not, and either may be in use.  A miss is a decline, which is
                # what the except already means.
                return self._sim._pcode.evaluate_concrete_flat(cell, inputs)  # type: ignore[union-attr,arg-type]
            except Exception:
                return None

        for name, size in self._written_registers(code):
            value = concrete(name, size)
            if value is not None:
                self.values[name] = value

        # The stores this step made: the taint pass resolved their addresses and
        # the memory recorded them, so the concrete word can be asked for by the
        # name the address gives it, without computing the address twice.
        for address, size in self.memory.take_writes():
            value = concrete(f'MEM_{address:#x}_{size}', size)
            if value is not None:
                self.memory.write(address, value, size)

    def _written_registers(self, code: bytes) -> list[tuple[str, int]]:
        """(name, size) for every register this instruction writes.

        From the lowering, which covers the result and every flag.  If it
        declines the instruction there is nothing to ask, so fall back to the
        whole state format -- correct, and slow only for the instructions the
        compiled path was never going to take anyway.
        """
        from microtaint.taint_ir.frompcode import Unsupported, build_ir
        from microtaint.taint_ir.regmap import name_offset

        sizes = {r.name: (min(r.bits, 64) + 7) // 8 for r in self.state_format}
        try:
            prog = build_ir(self.arch, code)
        except (Unsupported, Exception):
            return list(sizes.items())
        by_offset: dict[int, str] = {}
        for r in self.state_format:
            off = name_offset(self.arch, r.name)
            if off is not None:
                by_offset.setdefault(off, r.name)
        out = []
        for key, _node in prog.outputs:
            if isinstance(key, tuple) and key[0] == 'reg':
                name = by_offset.get(key[1])
                if name is not None:
                    out.append((name, sizes.get(name, 8)))
        return out
