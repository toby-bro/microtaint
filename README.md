# Microtaint

Microtaint is a strictly typed Python library for generating and evaluating **bit-precise, dynamic Information Flow Tracking (IFT) rules** directly from raw instruction bytestrings.

Inspired by the hardware-level methodologies of the **CELLIFT** paper, Microtaint elevates the concept of mathematical "cell properties" to software ISAs. By combining the static analysis power of Ghidra's P-Code with the concrete execution accuracy of the Unicorn Engine, Microtaint computes perfectly precise taint propagation—including complex edge cases like partial register zero-extensions, bitwise arithmetic ripples, and architecture-specific condition flags (like x86's `EFLAGS` and ARM64's `NZCV`).

Microtaint serves as a standalone abstract equation generator and evaluator, capable of seamlessly feeding dynamic taint analysis engines or symbolic execution frameworks without requiring manually written semantics for thousands of instructions.

## Features

- **Bit-Precise Taint Rules:** Stop relying on rough block-level or byte-level taints. Microtaint tracks dependencies precisely down to the exact bit, handling shifts, partial registers, and individual flag propagation flawlessly.
- **CELLIFT Software Paradigm:** Automatically classifies machine instructions into mathematical archetypes (Mapped, Monotonic, Transportable, Translatable, Avalanche, etc.) to apply optimized tracking formulas.
- **Dual-Engine Architecture:** - Uses [pypcode](https://github.com/angr/pypcode) to lift instructions, compute backwards slices, and extract architectural dependencies statically.
  - Uses [unicorn](https://github.com/unicorn-engine/unicorn) to natively simulate the generated logical differentials, bypassing the need to build massive shadow-logic ASTs.
- **Fast & Stateless ASTs:** Pass in instruction bytes and your CPU state format; get back a mathematical AST (`LogicCircuit`) that can be evaluated against any dynamic concrete state.

## Demo

The tool takes raw architecture bytestrings, lifts them, and maps the output back to your provided logical state (a list of tracked registers).

Check out the `demo.py` file to see it in action, or evaluate a circuit dynamically:

```python
from microtaint.sleigh.engine import generate_static_rule
from microtaint.simulator import CellSimulator
from microtaint.instrumentation.ast import EvalContext
from microtaint.types import Architecture, Register

arch = Architecture.AMD64
simulator = CellSimulator(arch)
bytestring = bytes.fromhex("4801D8") # ADD RAX, RBX

# 1. Generate the static Logic Circuit
circuit = generate_static_rule(arch, bytestring, [Register('RAX', 64), Register('RBX', 64)])

# 2. Evaluate dynamically against concrete Values (V) and Taints (T)
ctx = EvalContext(
    input_values={'RAX': 0x0, 'RBX': 0x0},
    input_taint={'RAX': 0x0, 'RBX': 0x10}, # Bit 4 of RBX is tainted
    simulator=simulator
)
output_taint = circuit.evaluate(ctx) 
# output_taint['RAX'] will mathematically evaluate to 0x10
```

## Development & Testing

```bash
# Run type checking
uv run mypy .

# Lint & Format
uv run ruff check .

# Run Tests
uv run pytest
```

## Understanding the Formulas

When you generate rules, you receive an abstract syntax tree representing how taints flow constraint-by-constraint. Because we treat each assembly instruction as a monolithic computational "Cell" ($C$), the formulas rely heavily on mathematical differentials.

An output formula assignment looks like this:

```txt
T_RAX[63:0] = (SimulateCell(instr=0x4801d8, out=RAX[63:0], RAX=(V_RAX[63:0] OR T_RAX[63:0]), RBX=(V_RBX[63:0] OR T_RBX[63:0])) 
               XOR 
               SimulateCell(instr=0x4801d8, out=RAX[63:0], RAX=(V_RAX[63:0] AND NOT(T_RAX[63:0])), RBX=(V_RBX[63:0] AND NOT(T_RBX[63:0])))) 
              OR 
              (T_RAX[63:0] OR T_RBX[63:0])
```

Here is how to read the components of Microtaint's engine:

- **`V_REG` and `T_REG`**: Denotes the actual concrete runtime **Value** ($V$) and the **Taint mask** ($T$) of the register at specific bits.
- **`SimulateCell(...)`**: This node takes the concrete instruction and natively executes it inside the Unicorn Engine using a specialized subset of the state. It acts as a perfect architectural oracle.
- **The Logical Differential (`XOR`)**:
  Instead of guessing how an `ADD` or `IMUL` mixes bits, we calculate the differential: $C(V \lor T) \oplus C(V \land \neg T)$.
  We execute the cell once with all tainted bits forced to `1` (High Replica), and once with all tainted bits forced to `0` (Low Replica). The `XOR` of these two simulations is a strict mathematical proof: if the output changes between the two replicas, *the taint successfully propagated to that specific output bit*.
- **Polarity ($p$)**: Some instructions (like `SUB`) are *bitwise non-increasing*—meaning forcing an input bit to `0` actually makes the result *higher*. Microtaint's Sleigh backend automatically detects operations that invert polarity and flips their replicas ($V \land \neg T$ becomes the High replica) to ensure the differential accurately captures borrows and underflows.
- **Transportability Term (`OR (T_RAX ... OR T_RBX)`)**: If Sleigh classifies an instruction as an arithmetic "Transportable" cell (like `ADD`), the differential is combined with the direct bitwise OR of the input taints, guaranteeing that information flowing perfectly column-by-column isn't masked by identical values.
