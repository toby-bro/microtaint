# RQ7 — Analyses coarser engines cannot perform

**Claim (§6.7).** Bit-level precision enables two analyses that byte- and
register-level engines cannot do: constant-time checking of a square-and-multiply
implementation, and bit-field-precise side-channel detection in a DNS header
parser.

The point of both is *precision*, not speed. A byte-granular engine reports the
whole byte as tainted and cannot tell the secret-dependent bit from its
neighbours, so it either misses the finding or drowns it.

## Run

```sh
cd crypto/square_and_multiply
uv run python check_side_channel.py        # ~5 min
uv run python localise_side_channel.py     # ~5 min, locates the leaking branch

cd ../../dns
uv run python dns_experiment.py            # ~8 min
```

## What to look at

**Square-and-multiply.** `check_side_channel.py` reports whether a
secret-dependent branch was found; `localise_side_channel.py` reports *where*.
The claim is that the leaking branch is identified at bit granularity — the
specific exponent bit — not merely that a leak exists somewhere.

**DNS.** The parser packs several fields into one 16-bit header word. The claim
is that a taint on one field is reported on that field alone. A byte-level
engine cannot separate fields that share a byte, and the experiment prints the
per-field verdict so this is visible directly rather than inferred.

## Comparing against the other engines

`rq7-applications/other-engines/` holds the same two analyses written against
angr, Maat, Triton and PANDA (`detect_*`, `localise_*`). Those need the baseline
environments from `INSTALL.md` step 4. Without them, this experiment still
demonstrates what microtaint reports; it just cannot show the others failing to.

## Tolerance

Exact — these are findings, not measurements. The leaking branch and the tainted
DNS field are either identified or they are not.
