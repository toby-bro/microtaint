# Archived campaign: the run that found the bitwise ZF under-taint

This is the raw state of the 24h cross-ISA campaign as it stood when it was
stopped early, on 2026-09-18. It is kept because it is the evidence for a real
soundness bug, and because the engine it measured no longer exists.

**It is not an input to Table IV.** `table5.py` globs `campaign_*.json` in its own
directory only, so nothing here is picked up. Do not move these files up a level.

## What it measured

Engine `384a86b`, clean tree. Stopped at 14.06h of the scheduled 24, after
**47,694,567 cases** over five ISAs.

| ISA | checked | forms | under-taints |
| --- | --- | --- | --- |
| MIPS64BE | 17,096,183 | 208 | 0 |
| RISCV64 | 12,236,928 | 44 | 0 |
| ARM64 | 7,262,256 | 400 | 0 |
| PPC32BE | 5,937,120 | 308 | 0 |
| AMD64 | 5,162,080 | 518 | **2** |

Both AMD64 witnesses are the same form, `xor ax, 0x7f`, and both are in
`under_x86_64.jsonl` with the full state that produced them.

## What the two witnesses were

`ZF = (result == 0)` is non-monotone, so the engine's two-corner differential
can miss it: both polarity corners can leave the result non-zero while an
interior assignment of the tainted bits drives it to zero. The equality-to-zero
floor that exists to cover exactly that was gated on carry arithmetic, so
`add`/`sub`/`cmp` were protected and the whole bitwise family was not.

Fixed in `71a1c3e`, with a regression test that brute-forces ground truth over
the full taint cube (`tests/test_zero_flag_bitwise_undertaint.py`). The fix also
showed that `xor eax` and `xor rax` were affected, which this campaign never
witnessed: a 64-bit result essentially never crosses zero, so the oracle had no
way to see them.

## Why the run was stopped

It had about 10h left and was measuring the pre-fix engine, so every remaining
hour would have re-found the same class of defect against a commit that had
already been superseded. The replacement run starts from the fixed tag.
