# Requirements

## Hardware

A commodity x86-64 desktop, as NDSS specifies: **8 cores, 16 GB RAM**. The
paper's numbers are from an AMD Ryzen 7 5700U (~3 GHz, 32 GB). More RAM shortens
nothing; more cores shorten RQ2 roughly linearly.

`RQ6` cross-compiles and emulates ARM64, MIPS64, PPC32 and RV64GC guests. No
non-x86 hardware is needed: the guests run under Unicorn, and the cross
toolchains are only needed to rebuild the test binaries, which ship prebuilt.

## Software

**Required for everything:**

- Linux (the paper used Arch with kernel 7.0; any recent distribution works)
- Python 3.13
- `uv` for dependency resolution

microtaint's own dependencies are pinned in `uv.lock` and install with one
command (see `INSTALL.md`). The three that matter are pypcode 3.3 (the SLEIGH
bindings), Unicorn 2.1 (concrete execution and the ground-truth oracle) and
Qiling 1.4 (the full-system harness for RQ5 and RQ7).

**Required only for RQ1–RQ3**, which compare against other engines:

| Baseline | How it is obtained | Notes |
|---|---|---|
| angr 9.2 | pip, in its own virtualenv | |
| Triton 1.0 | `triton-library` on pip, with Pin 3.20 | |
| Maat 0.6 | `pymaat` on pip | |
| TaintGrind 3.25 | built against Valgrind | |
| libdft64 | built against Pin 3.20 | Pin needs a manual download |
| PANDA | the `panda-re/panda` Docker image | needs Docker |
| TaintInduce | source, for RQ1 only | |

Each lives in its own environment because several of them pin conflicting
Python versions. `INSTALL.md` sets them up one at a time, and every experiment
**skips a baseline that is absent and says so** rather than failing — so RQ4-RQ7
are runnable with none of them installed.

## Time and disk

- microtaint only (RQ4-RQ7): ~100 minutes, ~2 GB
- everything: ~5 hours, ~15 GB (most of it the PANDA image and Pin)
