# RQ4 — Per-step propagation cost

**Claim (§6.4).** microtaint's per-step taint-propagation cost is competitive
with the fastest engines, and — unlike rule-inference approaches whose learned
boolean rules are O(n²) in operand width — is **independent of operand width**,
because the differential runs the instruction twice and XORs, which is O(1) in
width.

## Run

```sh
uv run python bench_width_scaling.py     # ~8 min, writes width_scaling_data.json
uv run python plot_width_scaling.py      # writes fig_width_scaling.pdf
```

## What to look at

`width_scaling_data.json` has one record per (family, width). The claim is about
the SHAPE across widths, not the absolute level:

- ADD/AND/XOR at 8, 16, 32 and 64 bits should differ by well under 2×. They are
  the same circuit evaluated on wider words.
- 128-bit and 256-bit **packed** entries are not a counter-example. `paddb` is
  sixteen per-byte lanes, so it costs roughly sixteen times one lane; the
  supplementary lane-count table in the same file isolates that (2, 4, 8 and 16
  lanes at a fixed 128-bit width), which is what shows the driver is lane count
  and not width.

## Tolerance

Absolute microseconds are machine-dependent and will not match the paper on
different hardware. What must hold is the shape: the scalar widths flat within
2×, and the packed cost tracking lane count rather than total width. A rerun on
the reference machine reproduced every point within ±4%.

## Note on which engine this measures

This times `circuit.evaluate` — the taint-propagation circuit the paper is
about. It is the right thing to measure for this claim, and deliberately not the
compiled backend that later versions add.
