# qrdata_hospital_treatment_v1 — provenance & tolerance rationale

**Established:** 2026-07-14, cloud mode, recipe `causal_effect.observational.dowhy`
(re-layered v2.1.0 — objective-named roles authoritative, post-treatment exclusions honored).
**Source:** QRData benchmark (github.com/xxxiaol/QRData), adapted as a GADS spec: the
question is unchanged; roles are named in the objective per the recipe's contract.

## External anchor (the point of this benchmark family)

QRData's gold answer: **ATE = -7.59** (hospital treatment dataset, QRData gold; severity is the essential confounder). Unlike the AMLB benchmarks, the
"right answer" here is externally defined, not self-canonicalized — the reference run's
ATE of **-7.5912** is graded against its own reproducibility, and the distance to the
gold anchors methodological appropriateness.

## Reference run

| Mode | Project | ate | placebo (~0) | subset (~ate) |
|---|---|---|---|---|
| cloud | `21d21c3f` | -7.591172 | 0.052836 | -7.689737 |
| local | `94f77f10` | -7.591172 | 0.052836 | -7.689737 |

**Cross-engine verification (2026-07-15, local run `94f77f10`, 16/16):** all three metrics
are **bitwise identical** to the cloud reference (both engines selected the objective-named
`['severity']` adjustment set; the D5 node is deterministic given the set). This is the
only benchmark in the family where cross-engine agreement is exact rather than last-ulp —
safe to tighten to `exact: true` once a same-engine repeat also confirms.

## Tolerance rationale

Provisional tolerances (2% relative, 0.01 floor) pending repeat-run verification — the
native node (`gads_causal_estimate_ate`) is internally seeded, so these should tighten
to exact once a second run confirms, same policy as amlb_*_v1.
