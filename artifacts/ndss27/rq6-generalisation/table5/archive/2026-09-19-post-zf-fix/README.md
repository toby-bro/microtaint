# The campaign that verified the bitwise ZF fix

A full 24h cross-ISA soundness campaign at engine `2b5d385`, run after the
bitwise-ZF under-taint found by the previous campaign was fixed in `71a1c3e`.

**It is not an input to Table 5.** `table5.py` globs `campaign_*.json` in its
own directory only, so nothing here is picked up.

## Result

Engine `2b5d385`, clean tree, the full 24h on all five ISAs.

| ISA | checked | rounds | under-taints |
| --- | --- | --- | --- |
| MIPS64BE | 33,878,963 | 10,412 | 0 |
| RISCV64 | 23,206,592 | 32,964 | 0 |
| ARM64 | 13,404,944 | 2,090 | 0 |
| PPC32BE | 10,355,360 | 2,088 | 0 |
| AMD64 | 8,989,920 | 1,071 | 0 |

**89,835,779 cases, zero under-taints.** Every `under_*.jsonl` is empty.

## Why the AMD64 line is the one to read

The previous campaign (`../2026-09-18-pre-zf-fix/`) found two `xor ax, 0x7f`
witnesses at **5,162,080** AMD64 cases. This run reached **8,989,920** on the
same ISA, 1.7x as far, with nothing. Combined with the regression test in
`tests/test_zero_flag_bitwise_undertaint.py`, whose ground truth is
brute-forced over the full taint cube rather than hand-written, that is
evidence the fix holds rather than that the bug is merely rare.

This run also covers nearly twice the previous campaign's total (89.8M against
47.7M), because that one was stopped at 14h to pick up the fix.
