# Microtaint

Microtaint is a lightweight, strictly typed Python library for generating **bit-precise taint dataflow rules** directly from raw instruction bytestrings.

Using [pypcode](https://github.com/angr/pypcode) (the Ghidra P-Code lifter under the hood), Microtaint translates machine instructions (x86, AMD64, ARM64, etc.) into intermediate representation and then mathematically simplifies the side-effects into near boolean logic circuits. This creates an exact formula identifying which bits of the output registers/memory depend on which bits of the input state.

The circuits leverage the instruction they want to instrument in order to simulate precise taint propagation.

Microtaint serves purely as a standalone abstract equation generator, capable of seamlessly feeding taint analysis engines or symbolic execution frameworks.

## Features

- **Bit-precise Taint Rules:** Stop relying on rough block-level taints. Microtaint evaluates partial register uses, bitwise operations, and flags dependencies strictly.
- **Fast & Stateless:** Pure functional rule generator. You pass instruction bytes and context; you get back mathematical ASTs (`LogicCircuit`).
- **Ghidra/angr P-Code Backend:** Robust and multi-architecture instruction lifting via [pypcode](https://github.com/angr/pypcode).

## Demo

The tool takes raw architecture bytestrings, lifts them, and maps the output back to your provided logical state (a list of tracked registers).
Check out the `demo.py` file.

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

When you generate rules (as shown in the demo), you receive an abstract representation of how taints flow constraint-by-constraint.

An output formula looks like this:

```txt
((SimulateCell(instr=0x4801d8, out=RAX[63:0], RAX=(V_RAX[63:0] OR T_RAX[63:0]), RBX=(V_RBX[63:0] OR T_RBX[63:0])) XOR SimulateCell(instr=0x4801d8, out=RAX[63:0], RAX=(V_RAX[63:0] AND NOT(T_RAX[63:0])), RBX=(V_RBX[63:0] AND NOT(T_RBX[63:0])))) OR (T_RAX[63:0] OR T_RBX[63:0]))
```

Here's how to read the components:

- **`V_RAX[...]`**: Denotes the actual concrete runtime **Value** of the register `RAX` at specific bits.
- **`T_RAX[...]`**: Denotes the **Taint** label (boolean/bitmask array) of `RAX` at specific bits.
- **`SimulateCell(instr=0x..., out=REG, args...)`**: Defines a computational constraint wrapper. It acts as an evaluation node that simulates the target bits of the output (`out=REG`) when evaluated through exactly that instruction (`instr=0x...`). This bridges mathematical abstraction to real architectural execution semantics, feeding exactly into constraint solvers like Z3/Claripy.
- **`XOR` differential**: The formula often evaluates the `SimulateCell` twice—once with the taint labels masked in (`OR T_...`), and once with them masked out (`AND NOT(T_...)`). The `XOR` of these two simulations is a strict mathematical proof of whether those specific tainted bits modified the final output target. If the `XOR` is nonzero, the taint successfully propagated!
