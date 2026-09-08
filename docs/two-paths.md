# Two ways to answer the same question

microtaint has two implementations of one specification: *which bits of the
machine state can an attacker move by moving a tainted input bit*. They are not
interchangeable, they disagree in both directions, and which one you get is
something you should choose on purpose.

## The two

### The differential — the reference

```python
circuit = generate_static_rule(arch, code, state_format)
out     = circuit.evaluate(EvalContext(input_taint=..., input_values=..., ...))
```

For each output separately — the result slice, and each flag — it runs the
**whole instruction concretely twice**, on `V|T` (every tainted bit forced to
one) and on `V&~T` (forced to zero), and XORs the two. A bit that differs is a
bit the attacker can move.

Exact wherever the function is monotone across that cube. Where it is not, both
corners can agree while the output actually varies — which is why the overflow
flag needs a table of its own, and why `rol rax, 31` shows an under-taint
against per-bit ground truth that is the *oracle's* artifact, not the engine's.

Costs microseconds. A flag-setting instruction is eight outputs times two
corners: sixteen whole-instruction executions.

### The compiled program — what ships

```python
prog = build_ir(arch, code)          # one program, every output, flags included
cap  = compile_program(prog, slot_of)
taint_ir_c.jit(cap)                  # native code for this host
```

The whole instruction lowers to one straight-line SSA program in a 64-bit IR,
constant-folded, hash-consed, with everything nothing reads removed. Instead of
*running* the instruction on two corners it **composes the differential per
p-code op symbolically**: exact routing where an op only moves bits, an inlined
`(lo ^ hi) | ta | tb` where it is monotone, a sound floor where it is not. Every
output falls out of one pass, and the instruction is never executed — values
enter as ordinary inputs.

Costs about eleven nanoseconds. There are two host backends, `taint_jit_x64.h`
and `taint_jit_a64.h`, so every guest architecture reaches native taint code on
either host.

## What they share

This is why they are comparable at all, and why one is a good check on the
other:

* **The same source of truth.** Both derive from the same p-code from the same
  SLEIGH lifter. Neither has a hand-written per-instruction rule table, so both
  gain an architecture at the same time.
* **The same definition of taint**, answered two ways: one by running the
  corners, one by composing the same differential algebraically.
* **Every output**, flags included.
* **Soundness**: both over-approximate, never under.

## Where they differ, and why

Measured over the instruction bank against per-bit Unicorn ground truth:

| | same | compiled tighter | compiled looser |
|---|---|---|---|
| AMD64 | 1829 | 215 | 28 |
| ARM64 | 1373 | 137 | 14 |
| RISCV64 | 184 | 0 | 0 |

**Tighter** — per-op composition keeps intermediate structure that two
whole-instruction corners destroy. `add rax, rbx` with only RAX's low byte
tainted: the differential says ZF is tainted, the compiled path says it is not,
and the compiled path is right, because RAX's upper bits are fixed so the sum is
`0x68xx` and can never be zero. `push` is the other standard example: the
whole-instruction differential clears RSP's own taint, the lowering keeps it.

**Looser** — where p-code does not model an operation at all (a `CALLOTHER`),
the lowering has to take the avalanche floor, while a concrete re-execution
would simply have run it and got the exact answer.

Different failure modes, one specification, both sound. That is exactly why
`circuit.evaluate` stays as it is: it is not a slower version of the compiled
path, it is the **independent check on it**. Pointing the oracle tests at the
compiled path would turn the check into a mirror.

## Choosing one

```python
from microtaint.taint_api import taint_step, explain, TaintSequence
from microtaint.taint_memory import TaintMemory

after = taint_step(arch, code, in_taint, in_values, path='compiled')
after, used = explain(arch, code, in_taint, in_values)   # which one answered
```

* `path='compiled'` — the fast one. Honoured even when the environment selects
  the other as the default, because comparing the two in a run configured for
  one of them is what the tests need.
* `path='differential'` — the reference.
* `path=None` (the default) — whatever `MICROTAINT_TAINT_IR` says.

`explain()` also returns which implementation answered. **Check it in a
benchmark**: the compiled path declines some instructions and falls back, and a
benchmark that does not ask will report the compiled path's speed for an
instruction it never ran.

### `MICROTAINT_TAINT_IR`

| value | meaning |
|---|---|
| `0` | the differential is the default, and the emulator's compiled path is off |
| anything else, or unset | the compiled path is the default |

It sets the default and switches the emulator; it does not overrule an explicit
`path=`. Running the suite once with each value is how a release checks that
both implementations are still correct.

## Memory, and sequences

A register-only answer cannot say whether a value that went into memory comes
back out tainted, so the API takes a memory:

```python
memory = TaintMemory()
seq = TaintSequence(arch, values={'RBX': 0xCAFEBABE, 'RSP': 0x7000},
                    taint={'RBX': (1 << 64) - 1}, memory=memory)
after = seq.run(bytes.fromhex('53'), bytes.fromhex('58'))   # push rbx ; pop rax
assert after['RAX'] == (1 << 64) - 1
```

`TaintMemory` holds the bytes and their taint. The taint side delegates to the
engine's own `BitPreciseShadowMemory` — the same C page table the emulator's
fast path uses — so the API and the emulator answer a memory question
identically rather than approximately.

The compiled path runs the same two-pass protocol as
`fastpath.h::mt_ir_mem_step`: a load's address is computed *by* the program, so
the program runs once to learn the addresses, the shadow is read, and it runs
again with the loaded words' taint in place.

`TaintSequence` additionally threads the concrete register values, because a
`pop` that does not see what `push` did to RSP reads the wrong word. That half
is the expensive one — it executes the instruction in the cell, once per
register it writes — which is why it is a separate class and not the default
behaviour of `taint_step`.

## What this cost, historically

`v0.6.15` is the last release with one implementation. From `950e8c0` there were
two, and only the emulator could reach the second: `engine_glue` was imported by
exactly one file. So 88 of 109 test files, every benchmark outside
`taint_density`, and the perf ratchet all went on measuring the reference — the
ratchet's docstring still calls `circuit.evaluate` "the taint-propagation HOT
PATH", a path the engine now reaches 31 times in 155,184 instructions.

That is the gap this API closes, and the reason the version moves to 0.7.0
rather than a patch: callers have to say which one they mean.
