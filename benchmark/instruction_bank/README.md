# Unified instruction bank

A single durable storage file for the machine instructions used to test and
benchmark microtaint across every supported ISA, consolidating the ~1,500
verified-sound instruction forms the soundness campaign accumulated (previously
stranded in the deletable scratchpad module `corpora.py`) plus the engine's own
AMD64 `test_cell_benchmark.CORPUS`.

## Layout

| File | What |
|---|---|
| `instructions.jsonl` | The bank. One JSON object per instruction, with **pre-assembled bytes** (so consumers need no keystone and do no per-run assembly). |
| `__init__.py` | Loader + the canonical per-ISA register/state formats (`ISA_FORMATS`, held in code so the root `*.json` ignore can't swallow them). |
| `build.py` | Regenerates `instructions.jsonl` from the source corpora (this one DOES need keystone). |

## Line schema

```json
{"isa": "AMD64", "label": "adc rax, rbx", "asm": "adc rax, rbx",
 "bytes": "4811d8", "srcs": ["RAX","RBX","CF"], "constraints": {},
 "categories": ["arith","flag"], "source": "corpora", "oracle": null}
```

`asm` is `null` for RISC-V (hand-assembled, only the mnemonic label is kept).
`oracle` carries concrete value/taint/expected vectors when the entry came from
the cell-benchmark corpus (`source: "cell_corpus"`), else `null`.

## Use

```python
import sys; sys.path.insert(0, "<engine>/benchmark")
from instruction_bank import load_bank, all_instructions

for spec in load_bank().values():                 # grouped per ISA
    for ins in spec.instructions:
        circ = generate_static_rule(spec.arch, ins.bytes, spec.regs)

shifts = all_instructions(categories={"shift"})    # filter by category
arm    = all_instructions(isas={"ARM64"})          # filter by ISA
```

Consumers: `tests/test_perf_ratchet.py` (the per-instruction performance
ratchet) and the benchmark drivers.

## Regenerate / extend

```sh
uv run --project . python benchmark/instruction_bank/build.py \
    [--corpora /path/to/corpora.py] [--no-cell-corpus]
```

Each form is validated by actually building its static rule with the canonical
format; a form whose rule cannot be generated is skipped and reported, so the
bank only ever contains instructions the engine can lift. Current contents:
AMD64 564, ARM64 398, MIPS64BE 219, PPC32BE 308, RISCV64 46 (1,535 total).
