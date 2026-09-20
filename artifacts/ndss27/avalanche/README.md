# Over-approximation's cost (§6.7, Table 6)

Two questions on real code: which of the categories the classifier assigns to
executed tainted instructions, and how many tainted output bits exist *only*
because the engine over-approximated.

```sh
make -C nftables && make -C siphash

# every run pins the engine path: taint_ir is the engine DEFAULT and is excluded
# from the artifacts, so an unpinned run measures the wrong engine.
export MICROTAINT_TAINT_IR=0 MICROTAINT_BLOCK=0

uv run python calibrate.py            # gate: must pass before any number counts

uv run python avalanche_freq.py --title base64 \
    --stdin 'The quick brown fox jumps over the lazy dog' \
    --json-out base64.json /usr/bin/base64

uv run python avalanche_freq.py --title nftables \
    --stdin-bytes "$(python3 -c 'print((bytes([80,2,0])+bytes(range(80))).hex())')" \
    --json-out nftables.json ./nftables/nftables_harness

uv run python avalanche_freq.py --title siphash \
    --stdin 'sixteen byte msg' \
    --json-out siphash.json ./siphash/siphash_bin

uv run python gen_avalanche_macros.py base64.json nftables.json siphash.json \
    --out ../../../paper/avalanche_numbers.tex
```

Every number in Table 6 and the three observations around it comes from that
last command. Nothing there is hand-typed.

## What the workloads are

`base64` encodes tainted stdin; its whole data path runs through a 64-entry
alphabet table indexed by the secret. SipHash-2-4 is pure add-rotate-xor over 16
tainted bytes. **nftables is a small extract** -- `nft_byteorder_eval`, verbatim
from the unpatched kernel (CVE-2023-35001) -- and is included because it is real
vulnerable kernel code, not because it is a large sample: it executes **35**
tainted instructions. Table 6 prints the sample size for exactly that reason.

The three all propagate taint end to end *inside the hooked binary*. That is why
coreutils `sha256sum`, say, is not here: it copies stdin through libc `memcpy`,
outside the hooked range, so nothing tainted reaches its transform.

## How the attribution works, and why it was rebuilt

A bit is attributable to over-approximation iff it is tainted in the real rule
and clean once every approximating node is forced to zero.

The previous harness answered that by **reimplementing** `Expr.evaluate` so it
could re-evaluate with the avalanche nodes zeroed. That copy had to grow a case
for every node type the engine ever added, and when it did not it degraded
silently: an unrecognised node was treated as an opaque leaf with "no avalanche
inside", so recursion stopped and anything beneath it vanished. Measured, it was
blind to **7 of the engine's 11 node types**, including `InstructionCellExpr`,
whose `inputs` dict holds whole sub-expressions.

That bias runs one way. Under-counting over-approximation makes bit precision
look *better*, so the instrument failed in the direction that flattered the claim
it exists to test.

The rebuild removes the copy:

- `exprwalk.py` finds a node's children by **reflection** over the public
  attributes every `Expr` exposes, so a node type added tomorrow is traversed
  today. It never evaluates anything.
- Neutralising is a temporary in-place swap with guaranteed restore, and the
  **engine's own `evaluate()`** runs on the modified tree. There is exactly one
  implementation of taint semantics.
- `evaluate_precise` handles the root case: a tree whose root IS the
  approximation has no parent to rewrite, which reported no avalanche share for
  `imul` -- an
  instruction that is entirely approximation -- until it was fixed.

## What counts as over-approximation

By the engine's own documentation, not by assumption:

| node | verdict |
| --- | --- |
| `AvalancheExpr`, `FullMaskAvalancheExpr` | the conservative fallback |
| `VariableMultiplyTaintExpr` | *"a sound fill, not an avalanche ... cannot be made exact cheaply"* |
| `VariableShift`, `VariableBitSelect`, `Comparison`, `Equality`, `SignedOverflow` | claim EXACT, so not counted |

Counting only the two avalanche nodes is what let Table 6 drift: commits
`4731f08` and `f3f52bb` moved multiply and variable-shift out of the blanket
fallback into dedicated terms, so their over-approximation stopped being counted
while `determine_category` still called them Avalanche.

## The gates

The old harness had five `except Exception:` paths, two of which produced a
visible `Unknown` and three of which were silent -- one dropped an *entire
instruction* from both halves of the table while `insns_hooked` still counted it,
making "carried no taint" and "the tally crashed" indistinguishable. All five now
record, print the first of each, and **exit non-zero**; `gen_avalanche_macros.py`
refuses to emit macros from a run that recorded any.

Four more gates, each closing a way the table used to go quietly wrong:

- **node-type drift** -- an unrecognised `Expr` type, or one with no
  exactness verdict, stops the run. A new node cannot join the "not counted"
  side by default.
- **construction-site** -- the harness hooks `generate_taint_assignments`, so
  anything building a `TaintAssignment` elsewhere is invisible to it. There is
  exactly one such path (`_exact_store_lane_targets`, the SIMD wide-store lane
  split behind `movups [rdi], xmm0`), and it is filled as Mapped; a second one
  fails the run. That, not an exception, was the real source of `Unknown`.
- **classifier vs tree** -- if `determine_category` says Avalanche while the
  emitted tree has no approximating node, or the reverse, the run reports it.
  The two halves of Table 6 disagreed for years without anyone noticing.
- **calibration** (`calibrate.py`) -- the paper states the attribution matches
  the ground truth: no avalanche share for the bitwise and movement forms, a
  full one for `imul`, and none for a constant shift. That claim previously
  appeared **only in prose**; no code in
  the artifact ever produced those numbers. It is now executable, and it caught
  both the original drift and a bug in its own replacement.

`shl rax,cl` is deliberately expected to show NO avalanche share, unlike the
figure the paper's prose carries for it: the engine
gained `VariableShiftTaintExpr`, which claims exact, and the measurement agrees.
That is a real precision gain, and the older figure describes an engine that no
longer
exists.

## Naming

The engine's enum and the paper's taxonomy are the same six categories under two
names. `mapper.py`'s own comment says a slice combining with XOR *"must be
WELDABLE (the OR of input taints)"* and then returns `ORABLE`; and
`COND_TRANSPORTABLE` holds exactly the equality opcodes the paper lists under
"Transportable (Eq)". The harness reports the paper's names so the artifact and
the table cannot be read as describing different taxonomies.

## Files

| file | |
| --- | --- |
| `avalanche_freq.py` | the measurement; `--json-out` feeds the macro generator |
| `exprwalk.py` | reflective traversal, neutralise/restore, the drift gates |
| `calibrate.py` | the known-answer gate; run it first |
| `gen_avalanche_macros.py` | emits every Table 6 number as a LaTeX macro |
| `diagnose_unknown.py` | re-runs with every swallowed path reported, for triage |
| `coreutils_unsound_freq.py` | unrelated: unsound-instruction frequency |

## Cost

Measured on sixteen cores with CPU boost disabled, as part of
`./run-all.sh --quick`.

| workload | time |
| --- | --- |
| calibration | a few seconds |
| each of the three `base64` builds | 4 to 8 s |
| nftables | 7 s |
| siphash | 5 s |

Under half a minute in total, and the same on the full corpus: this experiment
has no reduced mode, so `--quick` and the full run measure the same thing.
Memory is negligible. The first run fetches two pinned `base64` binaries, about
30 MB.
