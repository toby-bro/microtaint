# Baseline engines on the constant-time and DNS applications

Per-tool baseline verdicts for two of the RQ7 end-to-end applications (§IX-B of
the paper), so the byte- and register-granularity DIFT baselines' configurations
and false-positive/false-negative decisions are demonstrated, not asserted.

Every engine gets the identical source and the identical detection question; the
only variable is the engine. `results/<tool>.json` holds each verdict; the raw
tool logs sit alongside.

## Workload 1 — constant-time (control-flow side channel)

`../crypto/square_and_multiply/test_constant_time.c`: two square-and-multiply
modular exponentiations sharing one arithmetic core, selected by argv:
`vuln` (`pow_branch`, `if (e & 1)` branches on the secret exponent bit) and
`ct` (`pow_ct`, branch-free mask-select). Secret = 4-byte LE exponent from
stdin, tainted at the source; base and modulus are public constants. A leak is a
conditional branch whose condition depends on the secret.

| Engine | leak (branch-on-secret) | leak (constant-time) | matches microtaint |
|---|---|---|---|
| microtaint | yes (≥1) | no (0) | — |
| libdft64   | yes (je 0x402fa8, ×32) | no (0) | yes |
| TaintGrind | yes (32 IfGoto) | no (0) | yes |
| Triton     | yes (tainted ZF, ×32) | no (0) | yes |
| angr       | yes (1 secret-dep branch) | no (0) | yes |
| Maat       | yes (1 site, ×32) | no (0) | yes |

**All engines agree.** The leak is a control-flow dependence, which byte- and
register-level engines track as precisely as a bit-level one. So this workload
certifies that microtaint does not *hallucinate* a leak on constant-time code
(no false positive); it is not a byte-vs-bit advantage. (libdft additionally
taints the `read()` return via its syscall hook, so both variants show 2 extra
length/error-check branches from shared libc plumbing; the algorithm-level
differential is a clean 32 vs 0.)

## Workload 2 — DNS bit-field (sub-byte discrimination)

The `LDNS_OPCODE_WIRE` extraction of the RFC 1035 flag byte: `AND AL, 0x78`
(`24 78`) then `SHR AL, 3` (`C0 E8 03`). The flag byte packs `QR` at bit 7
(benign) and `OPCODE` at bits 6..3 (security-relevant). Ground truth: tainting
`QR` (bit 7) leaves the output clean (the `AND 0x78` masks bit 7 away); tainting
`OPCODE` (bits 6..3) taints the output.

| Engine | granularity | can taint 1 bit | QR/OPCODE separated | verdict |
|---|---|---|---|---|
| microtaint | bit      | yes | yes | correct |
| angr       | bit      | yes | yes | correct |
| Maat       | bit      | yes | yes | correct |
| libdft64   | byte     | no  | no  | **false positive** |
| TaintGrind | byte     | no  | no  | **false positive** |
| Triton     | register | no  | no  | **false positive** |

**The engines split exactly by granularity.** The bit-granular engines taint
`QR` and `OPCODE` independently and clear `QR` after the mask. The byte- and
register-granular engines cannot mark a single bit: tainting the flag byte taints
both fields, so the extracted `OPCODE` is reported tainted whichever field is
under analysis, and the benign `QR` flow cannot be cleared — a tool-specific
false positive. Bit precision, not the engine identity, is the discriminator
(microtaint, angr and Maat all get it right).

## Reproduce

```
# constant-time
gcc -O0 -g -static -no-pie -fno-stack-protector -o ct_test ../crypto/square_and_multiply/test_constant_time.c
python detect_<tool>_ct.py     # or the tool's driver
# dns
python detect_<tool>_dns.py    # runs the AND 0x78 ; SHR 3 macro under each engine
python report.py               # aggregate results/*.json (if present)
```

Tool installs and venvs are under `../../../benchmark` (`external/`, `.venv_*`).
