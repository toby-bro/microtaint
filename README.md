# Microtaint

## Benchmarks and evaluation

The benchmark and evaluation scripts used for the submission are present in the [benchmark](./benchmark/) sub directory.
To know how to run each script a dedicated [README](./benchmark/README.md) is present in the subdir.

## Introduction

Microtaint is a strictly typed Python library and command-line engine for performing **bit-precise, dynamic Information Flow Tracking (IFT)** on compiled binaries.

Originally an abstract rule generator based on the **CELLIFT** paradigm, Microtaint has evolved into a complete, out-of-the-box dynamic taint analysis emulator. Built on top of [Qiling](https://github.com/qilingframework/qiling) and [Unicorn](https://github.com/unicorn-engine/unicorn), it dynamically monitors program execution, identifies complex exploitation primitives (Buffer Overflows, Use-After-Frees, Side Channels, and Arbitrary Indexed Writes) and logs them in real-time.

It retains its foundational mathematical precision: behind the scenes, Microtaint still lifts executed instructions using Ghidra's P-Code ([pypcode](https://github.com/angr/pypcode)) and models them as logical ASTs, computing taint propagation rigorously down to individual carry/zero flags and partial register mutations.

## Features

- **Out-of-the-box Vulnerability Hunting:** Pre-built command-line flags to instantaneously trace standard input flows and check for vulnerabilities:
  - **BOF (Buffer Overflow):** Detects when the instruction pointer (RIP/PC) becomes tainted.
  - **UAF (Use After Free):** Monitors heap operations via a built-in `HeapTracker` and alarms on poisoned mapping accesses.
  - **AIW (Arbitrary Indexed Write):** Detects store operations executing with tainted pointer addresses.
  - **SC (Side Channels):** Emits findings when critical conditional branching decisions depend on tainted input.
- **Qiling-Powered Emulation Wrapper:** Fully integrates with the Qiling Framework. Drop your ELF/PE/Mach-O binaries in with a custom rootfs, and Microtaint wraps the CPU states gracefully.
- **High-Performance Tracing:** Built-in Cython `BitPreciseShadowMemory`, direct Unicorn state hooks, and custom JIT caching ensure fast execution capabilities.
- **Bit-Precise Rule Generation:** Still capable of generating mathematical formulas statically (via `generate_static_rule`), treating raw assembly instructions as monolithic logical circuits evaluated using simulated differentials.

## Installation

Microtaint is available on the [pypi](https://pypi.org/project/microtaint/), so you can use uv/pip/your_favorite_tool to install it.

If you want to build it locally then once you cloned the repo you can use `uv` to build it.

```sh
uv sync --reinstall-package=microtaint
```

For performance optimized builds of the leftover python code...
_(I am not so sure this makes any difference since the Cython and C migration of the hotpath. But before this enabled quite a good improvement)_

```sh
HATCH_BUILD_HOOKS_ENABLE=1 MYPYC_OPT_LEVEL=3 uv sync --reinstall-package=microtaint
```

## Command Line Usage

Use the provided `microtaint` command to execute and dynamically analyze a binary. Provide flags before the `--` separator. Any arguments after `--` represent the execution format for your compiled target.

```bash
# Detect everything, feed stdin automatically from the terminal
uv run microtaint --check-all -- ./binary arg1 arg2

# Read binary taint source from a specific file instead of stdin
uv run microtaint --check-bof --input payload.bin -- ./binary

# Pipe raw data directly to the binary while applying the UAF trace
python -c "print('A'*64)" | uv run microtaint --check-uaf -- ./binary

# Execute quietly and emit structured JSON findings (useful for CI/fuzzers)
uv run microtaint --check-all --quiet --json -- ./binary 2>/dev/null
```

## Python API Integration

### 1. Qiling Emulator Integration (High-Level)

The `MicrotaintWrapper` can be integrated manually onto any existing Qiling instance. This provides fine-grained control to programmatically trace or assert bitwise taints seamlessly during full-system/binary emulation.

```python
from qiling import Qiling
from microtaint.emulator.wrapper import MicrotaintWrapper

# Setup standard Qiling Environment
ql = Qiling(["path/to/binary"], rootfs="/custom/rootfs")

# Mount Bit-Precise Taint Engine on top
wrapper = MicrotaintWrapper(ql)

# Enable active security modules
wrapper.check_bof = True  # Track instruction pointers
wrapper.check_aiw = True  # Track memory addresses
wrapper.check_uaf = True  # Monitor frees

# Taint specific memory regions (e.g. 12 bytes at 0x1000)
wrapper.taint_region(0x1000, 12, "my_custom_tag")

# Run Emulator
ql.run()

# Review findings identified by the Reporter
for finding in wrapper.reporter.findings:
    print(finding)
```

### 2. Stateless AST Generation (Low-Level)

For cases where you don't need full emulation but want to analyze the math and formulas of taint propagation for a specific instruction byte string, you can directly interface with the static generator and native evaluator:

```python
from microtaint.sleigh.engine import generate_static_rule
from microtaint.simulator import CellSimulator
from microtaint.instrumentation.ast import EvalContext
from microtaint.types import Architecture, Register

arch = Architecture.AMD64
simulator = CellSimulator(arch)

# 1. Provide an instruction (AND EAX, 0x0F0F)
bytestring = bytes.fromhex('250f0f0000')

# 2. Lift it into a stateless logical circuit (AST)
circuit = generate_static_rule(arch, bytestring, [Register('RAX', 64)])

# 3. Form a concrete runtime execution context
ctx = EvalContext(
    input_values={'RAX': 0xFFFF},
    input_taint={'RAX': 0xFFFF},
    simulator=simulator
)

# 4. Mathematically evaluate how the taint propagates bit-by-bit
output_taint = circuit.evaluate(ctx)
# output_taint['RAX'] bitmask mathematically evaluates to 0x0F0F
```

## Development & Testing

Run tests and check typings/formatting with:

```bash
uv run mypy .
uv run ruff check .
uv run pytest
```

If a C/Cython file has been modified it is necessary to force a rebuild of the .so shared libraries with a

```sh
uv sync --reinstall-package=microtaint
```
