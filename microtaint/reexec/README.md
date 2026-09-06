# microtaint.reexec — native re-execution (4th concrete-execution path)

microtaint recovers the concrete state a value-aware / differential taint rule
needs by **re-executing** the instruction. Today that is the SLEIGH p-code
interpreter (`cell_c` `execute_decoded`), ~3.3 us per cell.

When the host ISA equals the target ISA (instrumenting AMD64 on an AMD64 host,
the common deployment), we can instead execute the real instruction bytes **on
the host CPU** and read back the concrete registers + flags. This module is the
prototype of that path (alongside the Cython walker, the compiled bytecode + C
kernel, and Unicorn).

## Result (AMD64)

| path | ns / instruction |
|---|---|
| SLEIGH concrete cell (`cell_c`) | ~3555 |
| native re-exec, raw call | ~30 |
| native re-exec, signal-guarded | ~32 |

**~110x** faster, and **bit-identical to SLEIGH on every defined output**
(validated over 1400+ (instruction, input) pairs — see
`tests/test_reexec_amd64.py`). The only differences are the officially
**undefined** flags (e.g. OF after a shift/rotate with count != 1): silicon
returns a real value, SLEIGH floors it — taint-safe, since microtaint already
avalanche-floors undefined flags. A differential runs the two polarity corners,
so ~64 ns vs ~7100 ns for the two SLEIGH cells.

## Files

- `reexec_amd64.S` — the copyable trampoline template (loads GPRs+RFLAGS, runs
  ONE instruction patched into a 16-byte hole, saves GPRs+RFLAGS; all scratch in
  RIP-relative slots inside the blob so it survives `memcpy` to an RWX buffer).
- `reexec.c` — harness: mmap RWX, copy template, patch, signal-guard, call.
  Amortised API (`reexec_arm`/`reexec_set_instr`/`reexec_call`) + all-in-one
  (`reexec_run_one`). `-DREEXEC_SELFTEST` and `-DREEXEC_BENCH` build standalone
  self-test / benchmark mains.
- `__init__.py` — `NativeReExec` (lazy-builds + loads the lib via ctypes) and
  `AVAILABLE`.

## Three fixes that mattered

1. **`sigaltstack` + `SA_ONSTACK`** — a faulting instruction runs with the
   target's (garbage) RSP, so the signal frame needs a dedicated stack or the
   process double-faults.
2. **Page-align the scratch data off the code page** (`.p2align 12`) — writing
   scratch in the executing code's cache lines caused self-modifying-code
   pipeline flushes (387 ns → 25 ns raw).
3. **`sigsetjmp(env, 0)`** — `SA_NODEFER` leaves the mask unchanged, so skip the
   per-call `sigprocmask` (456 ns → 32 ns signal-guarded).

## Fall back to SLEIGH for

Memory operands, control-flow, non-deterministic / privileged instructions
(`syscall`, `cpuid`, `rdtsc`, …), and anything > 15 bytes.

## Next steps

- **ISA abstraction**: per-ISA trampoline chosen at compile time. ARM64: x0-x30
  + NZCV via `mrs`/`msr`, testable under `qemu-aarch64`.
- **ptrace backend**: an isolated-process variant (`PTRACE_SETREGS` /
  `PTRACE_SINGLESTEP` / `PTRACE_GETREGS`) for fault isolation over raw speed.
- **Build integration**: compile the lib as part of the package build instead of
  the lazy ctypes build.
- **Engine integration**: route the concrete-execution primitive to native
  re-exec when host ISA == target ISA, falling back to SLEIGH for the excluded
  classes. Complements the closed-form flag work (fewer cells) and the C-hook
  work (no per-instruction Python frame).
