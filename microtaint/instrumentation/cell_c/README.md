# microtaint pure-C cell evaluator

A drop-in C replacement for `microtaint.instrumentation.cell.PCodeCellEvaluator`
that bypasses the Cython-Python boundary and runs the P-code interpreter
entirely in C.

## What's in this directory

```
microtaint/instrumentation/cell_c/
├── cell_core.h    — Pure C: PCodeOp struct, MemMap, Frame, execute_decoded()
└── cell_c.c       — Python C extension wrapping cell_core for Python use
```

## Building

```bash
cd microtaint/instrumentation/cell_c
EXT=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
INC=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])")
gcc -O3 -march=native -ffast-math -shared -fPIC -I"$INC" -I. \
    cell_c.c -o "cell_c${EXT}"
```

This builds a single `.so` (e.g. `cell_c.cpython-312-x86_64-linux-gnu.so`).

## Source-tree changes required

Two surgical edits to existing files:

### 1. `microtaint/instrumentation/cell.pyx` — add `get_buf_bytes()`

Inside `cdef class DecodedOps`, add:

```cython
    def get_buf_bytes(self):
        """Return the raw PCodeOp struct array as a bytes object.
        Used by cell_c.c to copy the pre-decoded ops into its own DecodedBundle."""
        return bytes((<unsigned char*>self.buf)[:sizeof(PCodeOp) * MAX_PCODE_OPS])
```

This exposes the pre-decoded P-code op array so the C module can memcpy it
into its own bundle cache without re-doing the SLEIGH lift.

### 2. `microtaint/simulator.py` — add `use_c=True` flag

Add a `use_c: bool = False` parameter to `CellSimulator.__init__`, and
when `use_c=True`, instantiate `PCodeCellEvaluatorC` instead of the
Cython evaluator. Falls back gracefully to Cython if `cell_c` is not built.

See `simulator.py.patch` in this directory for the exact diff.

## Usage

```python
from microtaint.simulator import CellSimulator
from microtaint.types import Architecture

# Cython kernel (default, unchanged behaviour)
sim = CellSimulator(Architecture.AMD64)

# Pure-C kernel
sim = CellSimulator(Architecture.AMD64, use_c=True)
```

## Architecture support

Same as the Cython evaluator — register layout is built per-instance from
`_build_reg_maps(arch)`. Tested on:

- AMD64 (x86_64)
- x86 (32-bit)
- ARM64 (AArch64)

The C module bumps `REGS_ARR_SIZE` to 17 000 (vs Cython's 1 104) so all
ARM64 X-registers (offset ≥ 16384) hit the flat-array path instead of
falling back to a Python dict. Register names are looked up via a 4096-slot
FNV-1a open-addressing hash table.

## Drop-in correctness

All 327 existing pytests pass with the C evaluator forced on:

```bash
MICROTAINT_USE_C=1 pytest tests/ -q --no-header \
  --ignore=tests/bench_microtaint.py \
  --ignore=tests/bench_shadow_sim_wrapper_changes.py \
  --ignore=tests/test_cell_benchmark.py \
  --ignore=tests/test_cli_suite.py \
  --ignore=tests/test_e2e_simulate.py \
  --ignore=tests/test_emulator.py \
  --ignore=tests/test_heap.py \
  --ignore=tests/test_memory_diagnostics.py \
  --ignore=tests/test_shadow_memory-th.py
```

The `tests/conftest.py` shipped in this archive enables the `MICROTAINT_USE_C=1`
environment variable to monkey-patch every `CellSimulator()` call site so
the existing test suite is reused without modification.

## Performance

### Kernel-level (`evaluate_differential`, 200 000 iterations, AMD64)

| Instruction       | Cython  | C       | Speedup |
|-------------------|--------:|--------:|--------:|
| MOV RAX,RBX       | 1637 ns | 1215 ns | 1.35×   |
| AND EAX,EBX       | 2390 ns | 1406 ns | 1.70×   |
| OR  RAX,RBX       | 2292 ns | 1392 ns | 1.65×   |
| XOR EAX,EBX       | 2259 ns | 1373 ns | 1.65×   |
| ADD RAX,RBX       | 2276 ns | 1472 ns | 1.55×   |
| SUB RAX,RBX       | 2346 ns | 1445 ns | 1.62×   |
| SHL EAX,1         | 1893 ns | 1354 ns | 1.40×   |
| IMUL EAX,EBX      | 2174 ns | 1429 ns | 1.52×   |
| BSWAP EAX         | 1976 ns | 1379 ns | 1.43×   |
| **Geomean**       |         |         | **1.51×** |
| x86 ADD EAX,EBX   | 2373 ns | 1452 ns | 1.63×   |
| ARM64 ADD X0,X1,X2| 3967 ns | 1445 ns | 2.75×   |

### End-to-end on `bench.c` (microtaint --check-all, full BOF detection)

| | Cython | Pure C | Speedup |
|---|---:|---:|---:|
| Wall-clock (mean of 3 runs) | 43.2 s | 42.0 s | **1.03×** |

The end-to-end gain is small because the kernel itself is only ~10–15% of
the total pipeline cost. From cProfile of the bench run:

| Component                      | Tottime | % of total |
|--------------------------------|--------:|-----------:|
| `_instruction_evaluator_raw`   | 36.8 s  | 65 %       |
| Cython/C kernel (combined)     |  ~10 s  | 18 %       |
| Unicorn `emu_start`            |   2.2 s |  4 %       |
| Other (MachineState, Qiling…)  |   ~7 s  | 13 %       |

The kernel is already near hand-written C performance in Cython; the
larger remaining target for performance work is the Python-side
`_instruction_evaluator_raw` hook (uc_reg_read_batch, EvalContext build,
LogicCircuit.evaluate AST walk, shadow memory I/O).

## Implementation notes

- **PCodeOp struct layout** is identical to the Cython version: 15 fields
  with natural alignment, sizeof = 80 bytes on x86_64. The C module
  memcpys directly from `DecodedOps.get_buf_bytes()` into its own bundle.
- **Bundle cache** uses a Python dict keyed on the instruction byte string,
  with each value a `PyCapsule` holding a heap-allocated `DecodedBundle*`.
  Cached after first lookup; subsequent calls are O(1) dict get + capsule deref.
- **EFLAGS reconstruction** mirrors `cell.pyx _read_output` exactly: when
  reading EFLAGS@offset 640 size 4 returns 0, rebuild from individual flag
  registers (CF@512, PF@514, ZF@518, SF@519, DF@522, OF@523). Without this
  fix, x86 flag-taint tests fail.
- **Memory map** uses a 256-slot open-addressing hash table with Knuth
  multiplicative hashing. This replaces the per-call Python dict in
  `cell.pyx`'s memory frame and is one source of the 1.5× speedup.
- **Fallback path** raises `microtaint.instrumentation.cell.PCodeFallbackNeeded`
  on float ops, CALLOTHER with output, BRANCHIND/CALLIND, and unknown ops —
  identical fallback semantics to the Cython evaluator.
