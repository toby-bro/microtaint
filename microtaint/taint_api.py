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
from typing import Literal

from microtaint.types import Architecture, ImplicitTaintPolicy, Register

Path = Literal['compiled', 'differential']

#: What `path=None` means.  The same variable the emulator hook reads, so one
#: run of the suite exercises one implementation from top to bottom.
_ENV = 'MICROTAINT_TAINT_IR'


def default_path() -> Path:
    """The implementation this process uses when the caller does not say.

    `MICROTAINT_TAINT_IR=0` selects the differential; anything else, including
    unset, selects the compiled path.
    """
    return 'differential' if os.environ.get(_ENV) == '0' else 'compiled'


def taint_step(
    arch: Architecture,
    code: bytes,
    in_taint: dict[str, int],
    in_values: dict[str, int] | None = None,
    *,
    path: Path | None = None,
    state_format: list[Register] | None = None,
    implicit_policy: ImplicitTaintPolicy = ImplicitTaintPolicy.IGNORE,
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
                     state_format, implicit_policy)[0]


def explain(
    arch: Architecture,
    code: bytes,
    in_taint: dict[str, int],
    in_values: dict[str, int] | None = None,
    *,
    path: Path | None = None,
    state_format: list[Register] | None = None,
    implicit_policy: ImplicitTaintPolicy = ImplicitTaintPolicy.IGNORE,
) -> tuple[dict[str, int], Path]:
    """`taint_step`, plus which implementation actually answered.

    A benchmark that does not check this can report the compiled path's speed
    for an instruction the compiled path declined.
    """
    return _dispatch(arch, code, in_taint, in_values or {}, path,
                     state_format, implicit_policy)


# ---------------------------------------------------------------------------


def _dispatch(arch, code, in_taint, in_values, path, state_format, implicit_policy):
    if state_format is None:
        from microtaint.emulator import archregs
        state_format = archregs.state_format(arch)
    want = path or default_path()
    if want == 'compiled':
        got = _compiled(arch, code, in_taint, in_values, state_format)
        if got is not None:
            return got, 'compiled'
    return _differential(arch, code, in_taint, in_values, state_format,
                         implicit_policy), 'differential'


_LAYOUTS: dict = {}


def _slot_layout(arch, state_format, in_taint: dict, in_values: dict) -> dict[str, int]:
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


def _compiled(arch, code, in_taint, in_values, state_format):
    """-> the taint dict, or None if the compiled path declined this one."""
    from microtaint.instrumentation.cell_c import taint_ir_c
    from microtaint.taint_ir.engine_glue import enabled, program_for

    if not enabled():
        return None
    layout = _slot_layout(arch, state_format, in_taint, in_values)
    if not layout:
        return None
    got = program_for(arch, code, layout)
    if got is None:
        return None
    _cap, _addr, accesses, _writes_pc, _live = got
    if accesses:
        # A load or a store needs its address resolved and its shadow read,
        # which is a conversation with a memory the caller has not given us.
        # The emulator has one; this entry point does not.
        return None
    n = len(layout)
    values = [0] * n
    taints = [0] * n
    for name, slot in layout.items():
        values[slot] = in_values.get(name, 0) & 0xFFFFFFFFFFFFFFFF
        taints[slot] = in_taint.get(name, 0) & 0xFFFFFFFFFFFFFFFF
    out = taint_ir_c.run(got[0], values, taints)
    return {name: out[slot] for name, slot in layout.items()}


def _differential(arch, code, in_taint, in_values, state_format, implicit_policy):
    from microtaint.instrumentation.ast import EvalContext
    from microtaint.simulator import CellSimulator
    from microtaint.sleigh.engine import generate_static_rule

    circuit = generate_static_rule(arch, code, list(state_format))
    ctx = EvalContext(
        input_taint=dict(in_taint),
        input_values=dict(in_values),
        simulator=CellSimulator(arch),
        implicit_policy=implicit_policy,
    )
    # The POST-STATE, not just what the instruction wrote: an untouched
    # register keeps the taint it had.  `circuit.evaluate` reports only its
    # own assignments, so overlaying them on the input is what makes the two
    # paths answer the same question in the same shape -- without which
    # "run the suite against either implementation" is not a thing you can do.
    after = {name: in_taint.get(name, 0)
             for name in _slot_layout(arch, state_format, in_taint, in_values)}
    after.update(circuit.evaluate(ctx))
    return after
