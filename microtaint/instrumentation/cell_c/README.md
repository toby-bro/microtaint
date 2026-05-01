# microtaint native acceleration layer (v9)

Hand-written C extensions and supporting glue that accelerate microtaint's
hot path. The full stack now consists of six optimization layers — and
**zero Python frames remain on the per-instruction hot path**.

| Layer | What | Where |
|---|---|---|
| Tier 1 | `cell_c` — pure-C P-code kernel | this directory |
| Tier 1 | `circuit_c` — bytecode-compiled LogicCircuit evaluator with CellCAPI | this directory |
| Tier 3 | per-instruction taint-state memoization | `wrapper.py.patch` |
| V5 | `hook_core` — Cython-compiled per-instruction Unicorn hook | `microtaint/emulator/hook_core.pyx` |
| V7 | direct `uc_hook_add` bypass + version-cache fast path | `wrapper.py` + `hook_core.pyx` |
| V8 | Cython memory hooks (read/write/UAF-unmapped) with `cimport`-typed shadow_mem | `hook_core.pyx` + `shadow.pxd` |
| V9 | `LiveMemReader` Cython class + direct C-kernel dispatch from `MemoryDifferentialExpr` | `hook_core.pyx` + `ast.pyx` |

## Performance

End-to-end on the 256-byte stdin × 100-mix-round bench against a static
x86-64 binary running ~1.19M tainted instructions:

| Configuration | Wall time | × Cython | × Native |
|---|---:|---:|---:|
| Native binary | 10 ms | — | 1× |
| Cython baseline (no extensions) | 38 s | 1× | 3800× |
| + Tier 1 (cell_c + circuit_c with CellCAPI) | 32 s | 1.2× | 3200× |
| + Tier 3 (instruction memoization, ~86% hit rate) | 16.5 s | 2.3× | 1650× |
| + V5 (Cython hook) | 13.8 s | 2.8× | 1380× |
| + V7 (uc_hook_add bypass + version cache) | 11.2 s | 3.4× | 1120× |
| + V8 (Cython memory hooks) | 6.7 s | 5.7× | 670× |
| **+ V9 (LiveMemReader, direct kernel dispatch)** | **6.5 s** | **5.8×** | **650×** |

## V9 — eliminating the last Python frames

After V8, profile audit (cProfile under bench) showed exactly two
Python-level frames called >1000 times on the hot path:

- `MicrotaintWrapper._read_live_memory` — 256k calls × 1.48 µs = **0.38 s**
  Called from `circuit_c.c`'s `OP_PUSH_MEM_VALUE` whenever a circuit reads
  guest memory.

- `CellSimulator.evaluate_differential` — 205k calls × 0.32 µs = **0.065 s**
  A thin Python wrapper around `self._pcode.evaluate_differential` (the
  C kernel) that catches `PCodeFallbackNeeded` and falls back to Unicorn.

V9 removes both:

- `LiveMemReader` (new cdef class in `hook_core.pyx`) is a drop-in
  replacement for `_read_live_memory`. Built once in `_setup_hooks()`,
  passed via `mem_reader=` to circuit_c. Same ctypes-wrapped `uc_mem_read`
  body, but no Python frame setup per call.

- `MemoryDifferentialExpr.evaluate` (in `ast.pyx`) now calls the C kernel's
  `evaluate_differential` directly, replicating the simulator middleman's
  `PCodeFallbackNeeded` handling locally so the simulator method is only
  hit when an actual Unicorn fallback is needed.

After V9, the hot-path profile shows **no Python frames called >1000
times** — every per-instruction operation goes through C, Cython, or
ctypes-wrapped libunicorn calls only.

## Default configuration

The runtime now defaults to the fastest available stack:

- C kernel (`PCodeCellEvaluatorC`) is used unless `MICROTAINT_DISABLE_C_KERNEL=1` is set
- Compiled-circuit fast path is used unless `MICROTAINT_DISABLE_COMPILED_CIRCUIT=1`
- Per-instruction memoization is used unless `MICROTAINT_DISABLE_INSTR_CACHE=1`
- Cython hook is used unless `MICROTAINT_DISABLE_CYTHON_HOOK=1`
- The `uc_hook_add` bypass is unconditional (with graceful fallback to `ql.uc.hook_add` on registration failure)

No environment variables or flags are needed for peak performance.

## Files in this directory

```
cell_core.h          — Pure C: PCodeOp struct, MemMap, Frame, execute_decoded()
cell_c.c             — Python C extension wrapping cell_core; exports CellCAPI
cell_c_api.h         — Public C API header for circuit_c -> cell_c calls
circuit_bytecode.h   — Bytecode opcode definitions for compiled circuits
circuit_c.c          — Bytecode compiler + evaluator for LogicCircuit ASTs

ast.pyx.patch        — Hooks compiled-circuit dispatch into LogicCircuit.evaluate
cell.pyx.patch       — Adds get_buf_bytes() helper to DecodedOps
simulator.py.patch   — Default use_c=True with MICROTAINT_DISABLE_C_KERNEL opt-out
wrapper.py.patch     — Tier 3 cache + Cython hook + uc_hook_add bypass + version cache
pyproject.toml.patch — Native compilation hooked into the wheel build
```

The companion Cython module is at `microtaint/emulator/hook_core.pyx`; the
custom Hatchling build hook is `hatch_build.py` at the project root.

## Installing

### Wheel (pip / uv)

```bash
pip install dist/microtaint-*.whl
# or:
uv sync
```

The Hatchling build hook compiles `cell_c.c`, `circuit_c.c`, and the four
`.pyx` files automatically. All six `.so` files end up in the installed
package.

### Editable install / iterative dev

```bash
uv sync --reinstall-package=microtaint   # full clean rebuild (~30 s)
```

This rebuilds every Cython module and the two hand-written C extensions
from scratch. Use whenever you've edited a `.pyx` or `.c` file.

## Architecture

### V7: uc_hook_add bypass

Unicorn's Python binding wraps every code-hook callback in two Python
frames: `__hook_code_cb` (which dispatches based on a callback table)
and `uccallback`'s `wrapper` (which adds try/except). Each frame costs
~3 us in CPython. With 1.19M callbacks per bench run, that's ~7 s of
pure binding overhead.

The wrapper now bypasses both by registering the Cython hook directly
through ctypes-wrapped `uc_hook_add`:

```python
self._instr_cfunc = _HOOK_CODE_CFUNC(instr_hook)  # CFUNCTYPE trampoline
_uc_hook_add(uc_handle, &handle, UC_HOOK_CODE,
             cast(self._instr_cfunc, c_void_p),
             None, begin, end)
```

This saves about 1 us per call by eliminating the inner `wrapper` frame
(the outer dispatch is still required by ctypes' `CFUNCTYPE`). On the
bench, that's ~1.2 s savings.

### V7: version-cache fast path

The Tier 3 instruction cache used `frozenset(register_taint.items())`
as its key — about 1.1 us per call to construct, on the cache-hit path
of every instruction. With ~1M cache hits per bench run, that's ~1.1 s
just hashing.

The new design replaces that with a monotonic `taint_version` counter
that increments only when `register_taint`'s content actually changes.
Each cache entry is `(input_version, output_version, output_state)`:

- On hit: compare `taint_version` (a single uint64 compare). Replay
  `output_state` and adopt `output_version`. No frozenset.
- On miss: compute `output_version = hash(frozenset(output_state.items()))`
  once. Store the cache entry. Adopt `output_version`.

Versions are derived deterministically from content, so two visits to
the same address with semantically-identical taint state always match.
The legacy frozenset-keyed cache is kept as a fallback for the rare
cross-version equivalence cases.

Saves ~1.1 s on the bench.

## Falling back

If any compiled module fails to load at import time, microtaint
gracefully falls back to the pure-Python / Cython implementation. The
`.so` files are functional accelerators, not required.

If `uc_hook_add` registration fails (very unlikely — would indicate a
fundamentally broken Unicorn install), the wrapper falls back to
`ql.uc.hook_add` automatically.

## Testing

```bash
PYTHONPATH=. pytest tests/                                          # 845 tests, default (C kernel)
PYTHONPATH=. MICROTAINT_DISABLE_C_KERNEL=1 pytest tests/            # 845 tests, Cython kernel
```

Both modes pass identically.

## Bottleneck picture (post-V7)

After V7, the profile shows:

| Component | Wall % | Notes |
|---|---:|---|
| `emu_start` (Unicorn C dispatch + our hook bodies) | 73% | Includes ~167k slow-path circuit evaluations |
| Memory hooks (`__hook_mem_access_cb` + Qiling layer) | 11% | 365k mem accesses through Python binding |
| `_read_live_memory` | 5% | 256k Python upcalls from circuit_c |
| Cell C kernel | 5% | 205k cell evaluations |
| Other (init, imports) | 6% | One-shot startup |

The next big win would be applying the same `uc_hook_add` bypass trick
to the memory hooks (~1 s reachable). After that, the floor is set by
the slow-path circuit evaluation work itself, which is already C.

For a true sub-100x regime, an architectural change (TCG plugin or
hardware-augmented IR) is required.
