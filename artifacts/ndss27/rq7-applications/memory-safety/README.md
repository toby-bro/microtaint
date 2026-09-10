# Memory safety detectors

The four detectors of §6.8, the ones every production taint engine is expected
to have: buffer overflow, use after free, side channel and arbitrary indexed
write. microtaint finds all four, and can run them in a single pass.

Each target is a freestanding x86-64 program (`-nostdlib`), so the only
instructions the engine executes are the ones written in the C file. `input.bin`
is the taint source: microtaint taints whatever the program reads on stdin.

```sh
make
uv run microtaint --check-bof --input input.bin -- ./bof.elf
uv run microtaint --check-uaf                   -- ./uaf.elf
uv run microtaint --check-sc  --input input.bin -- ./sc.elf
uv run microtaint --check-aiw --input input.bin -- ./aiw.elf
```

Under a second each. Every run must print exactly one finding of its kind:

| target    | detector | what it reports                                  |
| --------- | -------- | ------------------------------------------------ |
| `bof.elf` | `[BOF]`  | tainted data reaches RIP at the `ret`            |
| `uaf.elf` | `[UAF]`  | access to the poisoned range after `munmap`      |
| `sc.elf`  | `[SC]`   | the `jne` condition depends on a tainted bit     |
| `aiw.elf` | `[AIW]`  | the store destination address carries taint      |

`--check-all` runs the four detectors together, which is the multi-shadow claim:

```sh
uv run microtaint --check-all --input input.bin -- ./bof.elf
```

`aiw.elf` stores a tainted byte through the tainted index. The detector wants
both the address and the stored value tainted, so a constant stored through a
tainted pointer is not reported.
