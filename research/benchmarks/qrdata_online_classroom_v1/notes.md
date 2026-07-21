# qrdata_online_classroom_v1 — provenance & tolerance rationale

**Established:** 2026-07-14, cloud mode, recipe `causal_effect.observational.dowhy`
(re-layered v2.1.0 — objective-named roles authoritative, post-treatment exclusions honored).
**Source:** QRData benchmark (github.com/xxxiaol/QRData), adapted as a GADS spec: the
question is unchanged; roles are named in the objective per the recipe's contract.

## External anchor (the point of this benchmark family)

QRData's gold answer: **ATE = -4.91** (online classroom dataset, QRData gold; format_blended is an alternative-arm indicator to exclude). Unlike the AMLB benchmarks, the
"right answer" here is externally defined, not self-canonicalized — the reference run's
ATE of **-4.2125** is graded against its own reproducibility, and the distance to the
gold anchors methodological appropriateness.

## Reference run

| Mode | Project | ate | placebo (~0) | subset (~ate) |
|---|---|---|---|---|
| cloud | `31891ed3` | -4.212521 | 0.238462 | -4.313361 |
| local | `c7b0678e` | **FAIL 11/14** — no metrics.json | | |

**Local run FAILED (2026-07-15, `c7b0678e`, 3 workflow attempts):** the dataset has NaNs
in 76/323 rows of exactly the six demographic-dummy confounders; statsmodels raises
`exog contains inf or nans` inside the native node. The cloud engine preemptively cleaned
(`df.replace([inf,-inf],nan).dropna(subset=required_cols)`, run `31891ed3` line 137); the
local engine called the node on raw `df` and produced byte-identical code across all three
attempts despite the error being fed back — a genuine engine-boundary failure (error →
diagnosis → data-cleaning step is beyond the 12B realizer here). Note the scorer's
methodology tier still passed 8/8 on the failed run: regex checks can't distinguish
"right code that never produced numbers". Fix direction (deliberate, not yet applied):
harden `gads_causal_estimate_ate` to drop NaN/inf rows over the used columns internally —
would align with cloud behavior and not shift NaN-free benchmarks' canonicals.

## Tolerance rationale

Provisional tolerances (2% relative, 0.01 floor) pending repeat-run verification — the
native node (`gads_causal_estimate_ate`) is internally seeded, so these should tighten
to exact once a second run confirms, same policy as amlb_*_v1.
