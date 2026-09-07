# taint_ir — compiling taint propagation

One instruction's whole taint propagation, flags included, lowered to a
branch-free program and then to machine code.

## Why

The engine's older model evaluates one compiled expression per output *slice*.
`add rax, rbx` is seven programs — RAX plus six flags — each re-deriving the
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
register's taint is whatever its slot holds when the pass ends — result
registers and flags alike, no per-output program, no second traversal.

**`ir.py`** is the IR: a small, machine-shaped instruction set over 64-bit
words, hash-consed and constant-folded as it is built, then dead-code
eliminated. Three structural guarantees make it compilable:

- *straight-line* — p-code's intra-instruction control flow becomes
  predication. A write under an unknown predicate is a select; under a
  **tainted** predicate it is implicit flow, and the join carries both sides'
  taint plus the bits on which they differ, which is exact.
- *SSA* — register aliasing (AL inside RAX, a flag byte inside a flag block) is
  resolved at lift time by a symbolic frame; none of it survives into the
  program.
- *64-bit only* — widths are explicit masks, and a known-significant-bits bound
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
evaluated inline in machine arithmetic. It is exact — not a floor — wherever
the operation is monotone in every input bit, which covers the whole
carry-coupled family. The `| ta | tb` term is what repairs the classic
two-corner under-taint: a bit whose two inputs both flip cancels in the XOR,
and the carry it generates would otherwise be reported clean.

Signed overflow is the one flag a differential cannot answer, because
`OF = c_{w-1} XOR c_w` is an XOR of two monotone functions and is not monotone
itself: the corners can agree while OF varies, and differ while OF is pinned.
It reduces to `(a XNOR b) AND (a XOR c)` over three one-bit inputs, whose taint
is therefore a 64-entry table — one `uint64` constant, so `(K >> idx) & 1` is
exact and branch-free.

## Memory

A memory access becomes four state slots past the register file: the loaded
value and its shadow taint as inputs, and the address, that address's taint,
and the taint to store as outputs. Modelling it this way rather than as a
callback keeps the program straight-line, so every backend runs it unchanged.

The protocol is two passes — evaluate for the addresses, resolve them against
guest memory and the shadow, evaluate again for the taint — which is sound
exactly while no address depends on a value the same instruction loaded. The
builder verifies that and declines when it does not hold.

A load through a *tainted* address taints the whole loaded word: which bytes
are read is then itself secret-dependent. A store through one is a much larger
obligation, so its address taint is reported and the caller hands the
instruction to the slow path rather than silently writing one location.

## What it costs

Whole instruction, all outputs including flags, over the 1569-instruction bank:

| backend | mean | compile |
|---|---|---|
| clang -O3 | 5.5 ns | 3.5-37 ms per program |
| host emitter (x86-64) | 10.8 ns | 21-57 µs per program |
| C interpreter | 187 ns | none |

Per ISA, compiled: AMD64 7.0 ns, ARM64 5.4, PPC32BE 4.1, MIPS64BE 3.5,
RISCV64 3.8. Coverage is 96.6%; the rest is a 128-bit multiply, a CALLOTHER
whose result is read downstream, and backward branches.

## How it is checked

Under-taint is never acceptable, so every claim is measured against Unicorn
per-bit ground truth rather than against the engine's own answer — the IR is
deliberately *tighter* than the whole-instruction differential in places, and
scoring it against that would report precision gains as under-taint. The gate
is a subset property: `under(IR) ⊆ under(current engine)`, since both read the
same p-code and inherit its undefined-flag gaps.

- `tests/taint_ir_bank.py` — registers and flags, all ISAs with a Unicorn
  descriptor.
- `tests/taint_ir_mem.py` — loads and stores against a real mapped data page,
  including secret-dependent addresses.
- `tests/taint_ir_simd.py` — vector lanes read and written directly in XMM.
- `tests/test_taint_ir.py` — the pytest gates, plus a diff-test of all three
  backends against each other on many random states.
- `tests/taint_ir_perf.py` — the per-instruction timing baseline.

## Using it in the engine

`engine_glue.program_for(arch, code, slot_map)` compiles a program against the
engine's slot layout, refusing unless it is register-only, every input register
already has an interned slot, and the host emitter took it. The hot path calls
it through `MtAddrEntry.ir_fn`. Opt in with `MICROTAINT_TAINT_IR=1`.
