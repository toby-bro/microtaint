# microtaint native acceleration layer (v6)

This directory contains hand-written C extensions and supporting glue that
accelerate microtaint's hot path. The full stack now consists of three
optimization tiers plus a Cython-implemented Unicorn hook:

| Layer | What | Where |
|---|---|---|
| Tier 1 | `cell_c` — pure-C P-code kernel | this directory |
| Tier 1 | `circuit_c` — bytecode-compiled LogicCircuit evaluator with CellCAPI | this directory |
| Tier 3 | per-instruction taint-state memoization | `wrapper.py.patch` |
| V5 | `hook_core` — Cython-compiled per-instruction Unicorn hook | `microtaint/emulator/hook_core.pyx` |

## Performance

End-to-end on the 256-byte stdin × 100-mix-round bench against a static x86-64
binary running ~1.19M tainted instructions:

| Configuration | Wall time | × Cython baseline |
|---|---:|---:|
| Native binary | 10 ms | — |
| Cython baseline (no extensions) | 38 s | 1× |
| + Tier 1 (cell_c + circuit_c with CellCAPI) | 32 s | 1.2× |
| + Tier 3 (instruction memoization, ~86% hit rate) | 16.5 s | 2.3× |
| + V5 (Cython hook) | **13.8 s** | **2.8×** |

## Files in this directory

```
cell_core.h          — Pure C: PCodeOp struct, MemMap, Frame, execute_decoded()
cell_c.c             — Python C extension wrapping cell_core; exports CellCAPI
cell_c_api.h         — Public C API header for circuit_c → cell_c calls
circuit_bytecode.h   — Bytecode opcode definitions for compiled circuits
circuit_c.c          — Bytecode compiler + evaluator for LogicCircuit ASTs
build_pyx.py         — Standalone build script (for development)

ast.pyx.patch        — Hooks compiled-circuit dispatch into LogicCircuit.evaluate
cell.pyx.patch       — Adds get_buf_bytes() helper to DecodedOps
simulator.py.patch   — Adds use_c=True flag to CellSimulator
wrapper.py.patch     — Tier 3 instruction cache + Cython-hook wiring
pyproject.toml.patch — Native compilation hooked into the wheel build
```

The companion Cython module is at `microtaint/emulator/hook_core.pyx`; the
custom Hatchling build hook is `hatch_build.py` at the project root.

## Installing

### From a wheel (recommended)

If a wheel is available for your platform, just `pip install` it. The
`.so` files for all six extensions ship inside the wheel:

```bash
pip install microtaint
python -c "from microtaint.instrumentation.cell_c import cell_c; print(cell_c.__file__)"
```

### From source

The native compilation runs automatically when building a wheel from the
source tree, via the custom Hatchling build hook in `hatch_build.py`:

```bash
pip install build
python -m build --wheel
pip install dist/microtaint-*.whl
```

The build hook compiles `cell_c.c` and `circuit_c.c` with
`-O3 -march=native -ffast-math` by default, honouring `$CC` and
`$CFLAGS`. The Cython modules (`ast.pyx`, `cell.pyx`, `shadow.pyx`,
`hook_core.pyx`) are compiled by the `hatch-cython` plugin in the same
build pass.

### Manual build (for development)

```bash
python build_pyx.py        # compile .pyx → .c → .so
cd microtaint/instrumentation/cell_c
EXT=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
INC=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])")
gcc -O3 -march=native -ffast-math -shared -fPIC -I"$INC" -I. cell_c.c    -o "cell_c${EXT}"
gcc -O3 -march=native -ffast-math -shared -fPIC -I"$INC" -I. circuit_c.c -o "circuit_c${EXT}"
```

## Runtime knobs

The optimizations layer cleanly: each can be disabled independently for
debugging or A/B benchmarking via environment variables.

| Variable | Effect |
|---|---|
| `MICROTAINT_USE_C=1` | Use the C kernel (`cell_c.PCodeCellEvaluatorC`) instead of Cython |
| `MICROTAINT_DISABLE_COMPILED_CIRCUIT=1` | Skip the compiled-bytecode evaluator (`circuit_c`) |
| `MICROTAINT_DISABLE_INSTR_CACHE=1` | Skip the Tier 3 per-address taint memoization |
| `MICROTAINT_DISABLE_CYTHON_HOOK=1` | Use the pure-Python hook instead of `hook_core` |
| `HATCH_BUILD_HOOKS_ENABLE=1` | Enable optional `mypyc` build hook (off by default) |

## Architecture

### Tier 1: CellCAPI

`cell_c` exports a `CellCAPI` PyCapsule containing function pointers to its
fast cell-evaluation routine. `circuit_c` imports the capsule at module
init via `PyCapsule_Import`. When the bytecode evaluator hits an
`OP_CALL_CELL`, it bypasses the Python boundary entirely and calls into
`cell_c` through the cached function pointer with a pre-resolved
`CellHandle` struct.

Microbench: ~20 µs → ~8.5 µs per circuit eval (2.4× kernel-side improvement).

### Tier 3: instruction-level memoization

The bench profile shows only **150 unique instruction addresses for 1.19M
callbacks** — average **7948 revisits per address**. Every revisit with
the same `register_taint` signature produces the same output (for
register-only instructions). The wrapper caches `(address, frozenset(taint.items())) → output_state`
and replays on hit, skipping `circuit.evaluate(ctx)` entirely.

`circuit_c` exposes a `has_mem_ops` flag on each `CompiledCircuit` so the
wrapper only caches memory-free instructions (where shadow memory state
can't affect the result).

Hit rate on the bench: **85.9%**.

### V5: Cython hook

The per-instruction Unicorn callback was the dominant cost after Tier 3.
`hook_core.pyx` reimplements the entire `_instruction_evaluator_raw` body
in Cython with typed locals and direct `PyDict_*` C-API calls. The
wrapper constructs an `InstructionHook` instance once, captures all
needed wrapper fields as cdef attributes, and registers the instance as
the Unicorn callback. Every dict op, register read, and cache lookup
runs at C speed inside `__call__`.

## Falling back

If any compiled module fails to load at import time, microtaint
gracefully falls back to the pure-Python / Cython implementation. The
`.so` files are functional accelerators, not required.

## Testing

```bash
PYTHONPATH=. pytest tests/                                          # 327 tests, Cython kernel
PYTHONPATH=. MICROTAINT_USE_C=1 pytest tests/                        # 327 tests, C kernel + extensions
```

Both modes pass identically.
