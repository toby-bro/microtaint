# RQ1 - TaintInduce vs held-out ground truth (20260909-204037, tier=default)

| Family | Instruction | W | Dataflows | Silent on forced chains | Random: always/partial/never | Never exercised | Held-out under-taint | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bit-moving | `bswap esi` | 32 | - | - | - | - | 0 (0 bits) | correct (5s) |
| control-flow | `jz .+16` | 32 | - | - | - | - | 0 (0 bits) | correct (16s) |
| logic | `xor eax, ebx` | 32 | 64 | 0 | 64/0/0 | 0 | 0 (0 bits) | correct (over-taints flags) (689s) |
| logic | `and eax, ebx` | 32 | 64 | 0 | 64/0/0 | 0 | 0 (0 bits) | correct (over-taints flags) (615s) |
| logic | `or eax, ebx` | 32 | 64 | 0 | 64/0/0 | 0 | 0 (0 bits) | correct (132s) |
| arithmetic | `ADD R1, R2 (JN)` | 4 | 20 | 0 | 20/0/0 | 0 | 0 (0 bits) | correct (9s) |
| arithmetic | `add al, bl` | 8 | 72 | 0 | 37/22/5 | 8 | 74 (138 bits) | UNSOUND (40s) |
| arithmetic | `add ax, bx` | 16 | 272 | 0 | 52/87/49 | 84 | 220 (410 bits) | UNSOUND (67s) |
| arithmetic | `add eax, ebx` | 32 | 1056 | 0 | 93/182/136 | 645 | 562 (1127 bits) | UNSOUND (786s) |
| arithmetic | `sub eax, ebx` | 32 | 1056 | 447 | 85/202/123 | 646 | 551 (1018 bits) | UNSOUND (1174s) |
