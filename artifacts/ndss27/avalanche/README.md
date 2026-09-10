# Avalanche's cost (§6.7, Table 6)

Two questions on real code: which of the six categories the classifier actually
assigns to executed tainted instructions, and how many tainted output bits exist
*only* because of the conservative Avalanche fallback.

`avalanche_freq.py` answers both in one run. It monkeypatches the engine (it
never edits it): every taint assignment is tallied by category, and every
tainted output is re-evaluated with the avalanche nodes forced to zero. A bit is
attributable to avalanche iff it is tainted in the real rule and clean without
those nodes. The attribution is calibrated on instructions whose answer is known
(`add`/`xor`/`mov`/`and` give 0%, `imul` 100%, `shl rax,cl` with a tainted
amount 86%, `shl rax,4` 0%).

The three workloads all propagate taint end to end inside the hooked binary,
which is the reason for this particular choice: coreutils `sha256sum`, for
instance, copies stdin through libc `memcpy`, which is outside the hooked range,
so nothing tainted ever reaches its transform.

```sh
make -C nftables && make -C siphash

# GNU coreutils base64, encoding tainted stdin
uv run python avalanche_freq.py --title base64 \
    --stdin 'The quick brown fox jumps over the lazy dog' \
    --json-out base64.json /usr/bin/base64

# nft_byteorder_eval, verbatim from the unpatched kernel (CVE-2023-35001).
# stdin is len=80, size=2, op=0 followed by the 80-byte register file.
uv run python avalanche_freq.py --title nftables \
    --stdin-bytes "$(python3 -c 'print((bytes([80,2,0])+bytes(range(80))).hex())')" \
    --json-out nftables.json ./nftables/nftables_harness

# SipHash-2-4 over 16 tainted bytes
uv run python avalanche_freq.py --title siphash \
    --stdin 'sixteen byte msg' \
    --json-out siphash.json ./siphash/siphash_bin
```

Each prints the two halves of Table 6: the dynamic per-category frequency, then
avalanche's share of tainted output bits split into data registers (>=8 bit) and
CPU flags (1 bit). The split is the point. On SipHash avalanche decides 0% of
data bits and 84% of flag bits, so a single all-bits number would hide that its
data path is exact.

`--budget N` stops after N tainted instructions, for a quick look. The paper's
numbers are full runs, a few minutes each.

`RESULTS_avalanche_impact.md` holds the full measured tables and the notes on
why these three targets and not others.

Two supporting scripts, neither of which produces a paper table:
`coreutils_unsound_freq.py` counts how often the instruction families that make
other engines unsound (`cmov`, `setcc`, `adc`/`sbb`) appear across 28 coreutils
binaries, and `siphash/eval_siphash_avalanche.py` taints one input bit at a time
over SipHash and checks the engine's mask against the bits that empirically flip.
