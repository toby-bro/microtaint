# taint_ir, compiling taint propagation

One instruction's whole taint propagation, flags included, lowered to a
branch-free program and then to machine code.

## Why

The engine's older model evaluates one compiled expression per output *slice*.
`add rax, rbx` is seven programs (RAX plus six flags), each re-deriving the
answer from RAX and RBX, and most of them getting there by re-executing the
instruction in SLEIGH twice to take a differential, at about 776 ns per
execution. Roughly 4 µs for one `add`.

Almost none of that work depends on runtime state. An instruction's p-code
shape and its constant operands are fixed the moment it is lifted, so the
decisions can be made once, at lift time, and what is left compiled.

## The pipeline

    p-code  ──►  taint IR  ──►  { C interpreter | host emitter | clang -O3 }
             frompcode.py     taint_ir_c.c   taint_jit_x64.h   cbackend.py

**`frompcode.py`** walks the p-code once, emitting both the value expression
and the taint expression for every op into one program. Every architectural
register's taint is whatever its slot holds when the pass ends, result
registers and flags alike, no per-output program, no second traversal.

**`ir.py`** is the IR: a small, machine-shaped instruction set over 64-bit
words, hash-consed and constant-folded as it is built, then dead-code
eliminated. Three structural guarantees make it compilable:

- *straight-line*. P-code's intra-instruction control flow becomes
  predication. A write under an unknown predicate is a select; under a
  **tainted** predicate it is implicit flow, and the join carries both sides'
  taint plus the bits on which they differ, which is exact. A branch that
  leaves the instruction is not control flow *within* it: it writes the program
  counter, and the rule is short: a fixed target taints nothing, an indirect
  branch inherits its operand's taint, a conditional jump makes the counter
  secret-dependent exactly when its condition is, and a return carries nothing
  new because the load that fetched the address already wrote the counter.
- *SSA*. Register aliasing (AL inside RAX, a flag byte inside a flag block) is
  resolved at lift time by a symbolic frame; none of it survives into the
  program.
- *64-bit only*. Widths are explicit masks, and a known-significant-bits bound
  deletes the redundant ones.

**`boolsynth.py`** collapses the one-bit flag algebra a lifter emits. Any
boolean function of at most three one-bit leaves is reduced to its truth table
and re-emitted as a cheapest expression; all 256 of them cost at most three
operations.

## The taint rules

Three kinds, and only the third costs anything:

| kind | ops | cost |
|---|---|---|
| routing (exact) | COPY, ZEXT, SEXT, TRUNC, SUBPIECE, PIECE, NEGATE, XOR, value-aware AND/OR, shifts by a known amount | 1-6 |
| inlined differential (exact) | ADD, SUB, 2COMP, INT_CARRY, the comparisons | ~8 |
| soundness floor | variable-count shifts, MULT/DIV/REM by a variable, POPCOUNT, CALLOTHER | ~2 |

The differential is `(lo ^ hi) | ta | tb` over the two extremal corners,
evaluated inline in machine arithmetic. It is exact (not a floor) wherever
the operation is monotone in every input bit, which covers the whole
carry-coupled family. The `| ta | tb` term is what repairs the classic
two-corner under-taint: a bit whose two inputs both flip cancels in the XOR,
and the carry it generates would otherwise be reported clean.

Signed overflow is the one flag a differential cannot answer, because
`OF = c_{w-1} XOR c_w` is an XOR of two monotone functions and is not monotone
itself: the corners can agree while OF varies, and differ while OF is pinned.
It reduces to `(a XNOR b) AND (a XOR c)` over three one-bit inputs, whose taint
is therefore a 64-entry table, one `uint64` constant, so `(K >> idx) & 1` is
exact and branch-free.

## Memory

A memory access becomes four state slots past the register file: the loaded
value and its shadow taint as inputs, and the address, that address's taint,
and the taint to store as outputs. Modelling it this way rather than as a
callback keeps the program straight-line, so every backend runs it unchanged.

The protocol is two passes (evaluate for the addresses, resolve them against
guest memory and the shadow, evaluate again for the taint), which is sound
exactly while no address depends on a value the same instruction loaded. The
builder verifies that and declines when it does not hold.

What a *tainted* address implies is a policy, chosen at lowering time and named
rather than assumed:

- `concrete` (the default, and what the engine runs) reads the shadow at the
  address the instruction computes. This is what the whole-instruction
  differential does, so a compiled program is a faithful replacement for it.
- `avalanche` taints the whole loaded word, because which bytes are read is then
  itself secret-dependent. This is the sound answer, and the memory oracle
  checks it with vectors that taint the pointer's low bits.

Either way the address's own taint is published as an output, so the policy can
change without changing the lowering. Adopting `avalanche` in the engine is a
decision for the implicit-taint policy: it cascades, since a tainted stack
pointer would make every local read fully tainted.

## What it costs

Whole instruction, all outputs including flags, over the 1569-instruction bank:

| backend | mean | compile |
|---|---|---|
| clang -O3 | 5.5 ns | 3.8-29 ms per program |
| host emitter (x86-64) | 10.8 ns | 21-44 µs per program |
| C interpreter | 184 ns | none |

There are two host emitters, `taint_jit_x64.h` and `taint_jit_a64.h`, selected
by `#if defined(__x86_64__)` / `__aarch64__` in `taint_ir_c.c`. They consume the
same program: the IR comes from p-code and is ISA-general, so a backend is about
the host the analysis runs on, not the guest it analyses, and every guest
architecture the engine lifts reaches native taint code on either host. Anywhere
else keeps the interpreter.

Per ISA, compiled: AMD64 7.0 ns, ARM64 5.4, PPC32BE 4.3, MIPS64BE 3.3,
RISCV64 3.9. Coverage is 98.2% of the bank (1540/1569); what is left is a CALLOTHER whose
result is read downstream or is wider than a word, a p-code loop (a backward
branch), and the wide add `adcx` lifts to. The host emitter takes all but
division and count-leading-zeros.

In the engine, against the same run with `MICROTAINT_TAINT_IR=0`:

| benchmark | bare Qiling | differential | compiled |
|---|---|---|---|
| bench_dense | 8.39 ms | 652.05 ms (x76) | 108.41 ms (x13) |
| bench_sparse | 2.52 ms | 54.72 ms (x20) | 21.39 ms (x8) |
| bench_untainted | 2.26 ms | 50.85 ms (x24) | 20.26 ms (x9) |

SLEIGH re-executions on the dense workload drop from 131231 to 182: the
differential is essentially never reached.

The hot path reads the program's LIVE input registers rather than the circuit's,
which is a smaller set: a taint rule routes masks around and mostly never looks
at what a register held. Over the bank that is 1.48 registers of 3.69 (37%
fewer), and on the benchmarks it removes 31-34% of the register reads an
instruction makes -- about 1% of wall clock at 11.3 ns per read, and more on the
architectures whose instructions name three operands (ARM64 39%, MIPS64BE 51%,
RISCV64 54%).

Both columns are lower than they were when this was first measured (dense
731/286, sparse 92/74, untainted 85/66), and the reason is not the taint
arithmetic. Reading the untainted row at the time made the point: with taint
work at essentially zero the overhead was still x29, so what remained was the
per-instruction hook. `benchmark/taint_density/hookcost.c` then priced that
against a pure-C Unicorn harness with no Python in the process at all:

| | ns/instruction |
|---|---|
| no hook | 0.64 |
| code hook | 18.29 |
| block hook | 0.67 |

So Unicorn's own per-instruction dispatch is 18 ns, and the ~200 ns the
engine's empty callback cost was almost entirely its own: the GIL acquire a
`with gil` callback makes on every guest instruction, plus the refcounts and
attribute reads before the C path is reached. The express lane in
`emulator/fastpath.h` runs the cases that need none of that (the
untainted-input exit and a compiled program, in its register, program-counter
and memory shapes) with no GIL at all, from constants cached on the address
entry; `emulator/memhook.h` does the same for the load and store callbacks.
Over 99% of both bench_untainted and bench_dense now finish without the GIL,
once the attacker-influenced-write check learned to defer rather than refuse
(it was keeping 98% of bench_dense's instructions on the GIL path for a check
that does nothing unless the instruction made tainted stores), the UAF read
callback was armed on the first poison rather than for the whole run, and the
guest read behind a load was skipped where the program never reads the value.

Three things follow for anyone picking this up. Block-level hooking is worth
the 18 ns and no more, which is much less than the 15x amortisation a naive
reading of "18 instructions per basic block" suggests. `uc_reg_read_batch`
costs 4.8 ns plus 11.3 ns *per register* and `uc_mem_read` about 81 ns for
eight bytes, so what the fast path asks Unicorn for matters more than what it
computes. And the answer to what it should ask for is in the program itself:
of 3.69 register values a program mentions only 1.48 survive dead-code
elimination, and only 32% of loads have their value live at all.

That last point has a sharper consequence than performance. The fast path
fills its value array from the CIRCUIT's declared inputs, not the program's,
and on an AArch64 guest the rule generator resolves a memory operand into a
named MEM_ input without listing the base register: the program then reads a
slot nobody filled, computes address 0, and the guest read fails. Every
AArch64 memory instruction falls back to the circuit, at 818 us against 130 ns
on x86-64. Filling from the program's live inputs fixes that and reads fewer
registers at the same time.

## How it is checked

Under-taint is never acceptable, so every claim is measured against Unicorn
per-bit ground truth rather than against the engine's own answer, because the IR is
deliberately *tighter* than the whole-instruction differential in places, and
scoring it against that would report precision gains as under-taint. The gate
is a subset property: `under(IR) ⊆ under(current engine)`, since both read the
same p-code and inherit its undefined-flag gaps.

- `tests/taint_ir_bank.py`, registers and flags, all ISAs with a Unicorn
  descriptor.
- `tests/taint_ir_mem.py`, loads and stores against a real mapped data page,
  including secret-dependent addresses.
- `tests/taint_ir_simd.py`, vector lanes read and written directly in XMM.
- `tests/test_taint_ir.py`, the pytest gates, plus a diff-test of all three
  backends against each other on many random states.
- `tests/taint_ir_perf.py`, the per-instruction timing baseline.

## p-code loops, the one structural gap left

A `rep`-prefixed string operation lifts to a loop whose backward branch targets
the instruction's own IMARK. The trip count is a runtime value, so it cannot be
unrolled at lift time, but the body is an ordinary straight-line program, so
the shape of the answer is: lower the body once and let the caller iterate it.

With an **untainted** loop condition the trip count is public, so running the
body once per iteration is exact: no implicit flow, nothing to
over-approximate. With a **tainted** one, how many times the body runs is itself
secret over an unbounded set, which has no cheap sound answer; decline.

Two things have to change for that, and neither is done: the body's *values*
must survive from one iteration to the next (today only taint is an output, and
the value computation is eliminated unless a taint rule reads it), and a loop
that stores has to commit shadow writes per iteration rather than accumulating
them, which costs the all-or-nothing property the other paths rely on. Until
then a backward branch declines and the circuit handles it.

## Using it in the engine

`engine_glue.program_for(arch, code, slot_map)` compiles a program against the
engine's slot layout, refusing unless every input register already has an
interned slot (they are interned lazily, so it retries), no offset it touches
carries two of the caller's names, and the host emitter took it: the
interpreter is not reachable from the hot path. It also returns which registers'
VALUES the program reads, so the hot path can read those instead of the
circuit's larger set.

The hot path calls it through `MtAddrEntry.ir_fn`. A memory program runs the
two-pass protocol in `fastpath.h::mt_ir_mem_step`, and a program that writes the
program counter runs into scratch so the implicit-taint policy can be applied
before anything is committed, since deciding after committing would mean reporting a
leak already written down. Under WARN or STOP the instruction goes back to the
circuit, which owns the reporting. Neither path commits anything until it has
succeeded, so a refusal simply leaves the instruction to the circuit.

This is the default path.  `MICROTAINT_TAINT_IR=0` turns it off and sends every
instruction back through the differential evaluator, which is what the parity
tests compare against.
