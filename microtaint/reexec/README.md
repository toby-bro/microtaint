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

## Integration finding (important)

reexec is wired into the C kernel's `cell_eval_fast` (gated by
`MICROTAINT_REEXEC`, default OFF). It is **proven to fire**: over the AMD64 bank,
`stats()` reports `reexec_hits=565` of 869 concrete executions with reexec ON and
`0` with it OFF (the other 304 are excluded memory/branch/multi-instruction
cells).

But on the compiled **fast path** it gives **no speedup** (AMD64 p50 3.82 -> 4.13
us, p100 ~60 -> ~60 us). The ~110x above is versus the *slow* SLEIGH cell
(`evaluate_concrete`); the production `cell_eval_fast` -> `execute_decoded` C
interpreter is already fast (~hundreds of ns), and reexec's per-call marshaling
(16 GPRs + 7 flags read from the frame and written back) cancels the gain. Two
real levers: (1) marshal only the instruction's actual registers (2-4, not 23);
(2) the slow path / emulator, where the CAPI is not loaded and cells go through
the ~us `evaluate_concrete` path -- there reexec's 110x applies (the real use
case).

One taint difference when enabled: `imul r64,r64,imm32` CF/OF -- reexec is *more
precise* (the real CPU shows no overflow; SLEIGH over-taints imul flags), so it
is a precision gain, not unsound, but it breaks bit-exactness-with-SLEIGH.

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
