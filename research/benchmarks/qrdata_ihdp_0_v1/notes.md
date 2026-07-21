# qrdata_ihdp_0_v1 — provenance & tolerance rationale

**Established:** 2026-07-14, cloud mode, recipe `causal_effect.observational.dowhy`
(re-layered v2.1.0 — objective-named roles authoritative, post-treatment exclusions honored).
**Source:** QRData benchmark (github.com/xxxiaol/QRData), adapted as a GADS spec: the
question is unchanged; roles are named in the objective per the recipe's contract.

## External anchor (the point of this benchmark family)

QRData's gold answer: **ATE = 4.02** (IHDP replicate 0, QRData gold answer). Unlike the AMLB benchmarks, the
"right answer" here is externally defined, not self-canonicalized — the reference run's
ATE of **3.8965** is graded against its own reproducibility, and the distance to the
gold anchors methodological appropriateness.

## Reference run

| Mode | Project | ate | placebo (~0) | subset (~ate) |
|---|---|---|---|---|
| cloud | `3f6f83d5` | 3.896514 | -0.003578 | 3.902398 |
| local | `6e4591f6` | 3.896514 | -0.003578 | 3.902398 |

## Tolerance rationale

Provisional tolerances (2% relative, 0.01 floor) pending repeat-run verification — the
native node (`gads_causal_estimate_ate`) is internally seeded, so these should tighten
to exact once a second run confirms, same policy as amlb_*_v1.

**Cross-engine verification (2026-07-15, local run `6e4591f6`, 16/16):** ate and placebo
agree with the cloud reference to ~1e-15 relative (last-ulp float noise, NOT bitwise);
subset_new_effect is bitwise identical. Same-engine repeats may be exact, but cross-engine
`exact: true` would fail on the last ulp — keep a tiny tolerance (e.g. 1e-12 relative)
rather than `exact` if tightening. Two conditions produced the agreement: both engines
independently converged on the same top-10-by-|corr| confounder set (verified in both
workflow_execution.py files), and the D5 native node is deterministic given that set.
When the engines diverge on the adjustment set, agreement degrades to ~1% (see ihdp_1).
