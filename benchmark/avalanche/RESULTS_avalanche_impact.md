# MicroTaint — dynamic category frequency & AVALANCHE real-world impact (Reviewer D)

**Method.** A scratch harness (`avalanche_freq.py`) runs a real program under the
*unmodified* MicroTaint engine (Qiling/Unicorn + P-code taint circuits) with two
monkeypatches: (1) it wraps `generate_taint_assignments` to record the
`InstructionCategory` the classifier (`determine_category`) assigns to every
`TaintAssignment` it builds — the exact call the engine's own dispatch makes;
(2) it wraps `_cached_generate_static_rule` with a `CircuitProxy` that intercepts
the per-instruction `circuit.evaluate(ctx)` the wrapper performs, so each executed
instruction is tallied by category, and every tainted output assignment is
re-evaluated with the avalanche nodes forced to 0. A bit is **attributable to the
avalanche fallback** iff it is tainted in the real rule but *untainted* once the
`AvalancheExpr`/`FullMaskAvalancheExpr` nodes are removed (`full & ~precise`).
The Cython hot-path hook and per-address memo cache are disabled via the engine's
own env switches so every dynamic instruction flows through the introspectable
Python path. **No tracked engine file was modified.**

The attribution was validated on known instructions:
ADD/XOR/MOV/AND → 0% avalanche (precise), IMUL → 100%, `SHL rax,cl` (tainted
amount) → 86%, `SHL rax,4` (constant amount) → 0%.

## Targets (chosen after OpenSSH proved infeasible — see notes)

| target | what it is | why realistic |
|---|---|---|
| GNU coreutils `base64` | encode tainted stdin | ships on every Linux box; parses untrusted input in-binary |
| `nft_byteorder_eval` harness | **CVE-2023-35001** kernel function, verbatim (Pwn2Own Vancouver 2023 LPE) | real vulnerable parser; taint drives the buggy stride |
| SipHash-2-4 | veorq reference ARX PRF over 16 tainted stdin bytes | canonical avalanche/crypto construction |

All three propagate taint end-to-end and produce correct output
(`VGhl…ZG9n`, correct byteswap, correct 64-bit MAC).

## Result 1 — dynamic per-category frequency (share of executed **tainted** instructions, dominant category)

| Category | base64 | nftables (CVE) | SipHash |
|---|---|---|---|
| Mapped | 52.2% | 50.0% | 23.0% |
| Monotonic | 16.1% | 4.2% | 4.3% |
| Transportable | 15.6% | 25.0% | 54.9% |
| Translatable | – | – | 17.5% |
| Cond. Transportable | – | – | 0.3% |
| **Avalanche** | **16.1%** | **20.8%** | **0.0%** |

(Counts are fully dynamic — a loop body executed N times counts N times.)

## Result 2 — AVALANCHE's share of tainted **output bits** (the impact metric)

| bits considered | base64 | nftables (CVE) | SipHash |
|---|---|---|---|
| **all output bits** | 46.6% (6710/14384) | 33.7% (486/1442) | 5.0% (906/18291) |
| **data registers only (≥8-bit)** | **42.3%** (5568/13158) | **33.5%** (480/1432) | **0.0%** (0/17217) |
| CPU flag bits only (1-bit) | 93.1% (1142/1226) | 60.0% (6/10) | 84.4% (906/1074) |

## Reading of the result (for the rebuttal)

- **Avalanche's real-world impact is program-dependent, not uniform.** On
  arithmetic-shaped code it is substantial: base64's 3-byte→4-char packing uses
  integer multiply/divide, and `nft_byteorder_eval`'s out-of-bounds stride is an
  index **multiply** — both lower to `INT_MULT` (AVALANCHE), so ~34–42% of tainted
  **data** bits exist only because of the fallback. On a pure ARX cipher
  (SipHash: add / rotate-by-constant / xor) the precise rules
  (Transportable / Mapped / Orable / Translatable-with-constant-amount) cover
  **100% of data-register bits** and avalanche contributes **zero** data taint.
- **Most "avalanche" firings on benign code are the parity flag.** SipHash's
  AVALANCHE-category assignments are all 1-bit outputs: the `PF` flag is computed
  via `POPCOUNT`, which the classifier lists as an AVALANCHE opcode. Separating
  1-bit flags from ≥8-bit data (rows above) is what exposes this — the headline
  "5% all-bits" for SipHash is almost entirely flag bits, and its data-taint is
  exact.
- **Net:** avalanche is the fallback that keeps table-/multiply-driven transforms
  *sound*, and it does real work there (~1/3 of data bits), but it does **not**
  dominate a well-behaved crypto data path.

## Honest notes on target selection

- **OpenSSH `sshd` does not run under Qiling.** A bare load
  (`sshd -T -f /dev/null`) aborts during startup at a `futex` syscall
  (`ql.os.thread_management` is `None`): Qiling's user-mode ELF emulation has no
  thread/priv-sep machinery, and sshd forks + initialises OpenSSL threading
  *before* it ever reads network input. Its real attack surface is a socket after
  privilege separation, not stdin, so it is doubly unsuitable here.
- **`sha256sum` runs and hashes correctly, but reports 0 tainted instructions.**
  MicroTaint hooks only the *main binary's* address range (a deliberate
  performance choice). coreutils copies the 3 stdin bytes into its aligned
  64-byte block buffer with **libc `memcpy`**, which is outside the hooked range,
  so taint never reaches the in-binary SHA-256 transform. This is an inherent
  limitation of main-binary-only instrumentation (taint is not propagated through
  uninstrumented libc), not a property of SHA-256. base64, the nftables harness,
  and the static SipHash binary all process their tainted input **in-binary**,
  which is why they measure cleanly.

## Files

- Harness: `avalanche_freq.py`
- Per-target raw tallies: `base64.json`, `nftables.json`, `siphash.json`
- SipHash target build: `gcc -O0 -static -no-pie -fno-stack-protector -o siphash_bin
  benchmark/crypto/siphash/test_avalanche_siphash.c benchmark/crypto/siphash/siphash_ref.c`
- nftables target: `avalanche/nftables_harness` (prebuilt static ELF)
</content>
