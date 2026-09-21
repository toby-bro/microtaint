# The same two analyses on the baseline engines (§IX-H)

Each engine gets the identical source and the identical question; the only
variable is the engine. `results/<tool>.json` holds each verdict, with the raw
logs alongside. This is what turns "coarser engines cannot do this" from an
assertion into a measurement.

The environments are the ones `rq2-comparison/setup_envs.sh` builds.

## Constant time (Table VI)

`../crypto/square_and_multiply/test_constant_time.c`: two square-and-multiply
modular exponentiations over one arithmetic core, chosen by argv. `vuln`
(`pow_branch`) branches on the secret exponent bit, `ct` (`pow_ct`) mask-selects
branch-free. The secret is the 4-byte LE exponent read from stdin; base and
modulus are public constants. A leak is a conditional branch whose condition
depends on the secret.

| engine     | leak on `vuln`         | leak on `ct` | agrees |
| ---------- | ---------------------- | ------------ | ------ |
| microtaint | yes                    | no           | -      |
| libdft64   | yes (je 0x402fa8, x32) | no           | yes    |
| TaintGrind | yes (32 IfGoto)        | no           | yes    |
| Triton     | yes (tainted ZF, x32)  | no           | yes    |
| angr       | yes (1 branch)         | no           | yes    |
| Maat       | yes (1 site, x32)      | no           | yes    |
| PANDA      | yes (32 execs of 0x402fa8) | no       | yes    |

All seven agree, because the leak is a control-flow dependence that every
engine tracks. So this half certifies that microtaint does not hallucinate a leak on
constant-time code; the bit-vs-byte difference is in *attribution*, which is
what `localise_angr_ct.py` and `localise_maat_ct.py` measure: with only exponent
bit `k` tainted, the branch should be flagged at step `k+1` and nowhere else.
(libdft additionally taints the `read()` return through its syscall hook, so
both variants show two extra length-check branches from libc; the
algorithm-level differential is a clean 32 against 0.)

## DNS bit field (Table VII)

The `LDNS_OPCODE_WIRE` extraction of the RFC 1035 flag byte, `AND AL,0x78`
(`24 78`) then `SHR AL,3` (`c0 e8 03`). `QR` is bit 7 and benign, `OPCODE` is
bits 6..3. Ground truth: tainting `QR` leaves the output clean, since the mask
removes bit 7; tainting `OPCODE` taints it.

| engine     | granularity | can taint one bit | QR/OPCODE separated | verdict        |
| ---------- | ----------- | ----------------- | ------------------- | -------------- |
| microtaint | bit         | yes               | yes                 | correct        |
| angr       | bit         | yes               | yes                 | correct        |
| Maat       | bit         | yes               | yes                 | correct        |
| libdft64   | byte        | no                | no                  | false positive |
| TaintGrind | byte        | no                | no                  | false positive |
| Triton     | register    | no                | no                  | false positive |
| PANDA      | byte        | no                | no                  | false positive |

The split is exactly by granularity, not by engine identity: the three
bit-granular engines all get it right. The others cannot mark a single bit, so
tainting the flag byte taints both fields and the benign `QR` flow can never be
cleared.

## Run

`run-all.sh` runs all of this when the baselines are enabled. To do it by
hand, each driver writes one `results/<tool>.json` and builds whatever guests
it needs:

```sh
V=../../rq2-comparison

$V/.venv_angr/bin/python   detect_angr_apps.py    # -> results/angr.json
$V/.venv_angr/bin/python   localise_angr_ct.py    # per-bit attribution
$V/.venv_maat/bin/python   detect_maat.py         # -> results/maat.json
$V/.venv_maat/bin/python   localise_maat_ct.py
$V/.venv_triton/bin/python detect_triton.py       # -> results/triton.json
python3 detect_libdft64.py                        # -> results/libdft64.json
python3 detect_taintgrind.py                      # -> results/taintgrind.json
python3 detect_panda.py                           # -> results/panda.json
```

The last three fold what their tools print into the shared schema:

* `detect_triton.py` runs `detect_triton_ct.py` and `detect_triton_dns.py` and
  folds the two raw traces, as `detect_maat.py` does for Maat.
* `detect_libdft64.py` builds the two Pin tools and the two guests, runs them,
  and parses the `[cf_leak] cond_branches_executed=...` and
  `[dns_taint] SUMMARY ...` lines. The branches libdft taints in *both*
  variants are its `read()` hook tainting the syscall return, not the crypto
  side channel, so the sites common to both are subtracted and the
  algorithm-level differential is what the table reports.
* `detect_taintgrind.py` builds the two harnesses against valgrind's headers,
  runs them in the pinned container, and counts trace lines tagged `IfGoto`,
  an `Ist_Exit` whose guard is tainted. The trace carries the occasional
  non-UTF-8 byte, enough that a naive `grep -c` treats it as binary and
  reports nothing, so the parser decodes permissively.

Until these three existed, `triton.json`, `libdft64.json` and
`taintgrind.json` were typed by hand from tool output and no re-run could
check them. Each now reproduces its hand-written predecessor on every field.

## Cost

| engine | time |
| --- | --- |
| angr, Maat, Triton | seconds to a few minutes each, in process |
| libdft64, TaintGrind | a minute or so each, user-mode instrumentation |
| PANDA | about 40 min: three full-system guest boots of about 13 min each |

PANDA dominates because it is the only full-system engine here, so each of its
three workloads boots an entire Ubuntu guest under QEMU before any taint runs.
That is a fixed cost per workload and says nothing about its propagation speed,
which is the second fastest of the seven on the RQ2 corpus.

Measured on sixteen cores with CPU boost disabled, each driver from a cold
start including the guests it builds:

| driver | time |
| --- | --- |
| `detect_triton.py` | 1 s |
| `detect_libdft64.py` | 4 s |
| `detect_taintgrind.py` | 4 s |
| `detect_maat.py` | 8 s |
| `detect_angr_apps.py` | 9 s |
| `localise_angr_ct.py` | 53 s |
| `localise_maat_ct.py` | 109 s |
| `detect_panda.py` | about 40 min |
| **total** | **about 43 min** |
