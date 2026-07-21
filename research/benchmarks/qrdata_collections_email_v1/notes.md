# qrdata_collections_email_v1 — provenance & tolerance rationale

**Established:** 2026-07-14, cloud mode, recipe `causal_effect.observational.dowhy`
(re-layered v2.1.0 — objective-named roles authoritative, post-treatment exclusions honored).
**Source:** QRData benchmark (github.com/xxxiaol/QRData), adapted as a GADS spec: the
question is unchanged; roles are named in the objective per the recipe's contract.

## External anchor (the point of this benchmark family)

QRData's gold answer: **ATE = 4.43** (collections email dataset, QRData gold; opened/agreement are post-treatment traps the adjustment set must exclude). Unlike the AMLB benchmarks, the
"right answer" here is externally defined, not self-canonicalized — the reference run's
ATE of **4.4304** is graded against its own reproducibility, and the distance to the
gold anchors methodological appropriateness.

## Reference run

| Mode | Project | ate | placebo (~0) | subset (~ate) |
|---|---|---|---|---|
| cloud | `1e601199` | 4.430356 | 0.096387 | 4.387105 |
| local | `fc101698` | 4.430356 | 0.096387 | 4.387105 |

**Cross-engine verification (2026-07-15, local run `fc101698`, 16/16):** the mediator trap
held on the local engine too — it selected exactly `['credit_limit', 'risk_score']` and
named `opened`/`agreement` as post-treatment exclusions. With the identical adjustment
set, agreement with cloud is ~1e-13 relative (last-ulp, not bitwise): local ate
4.430355676952331 vs canonical …217. Keep tolerances ≥1e-12 relative if tightening.

## Tolerance rationale

Provisional tolerances (2% relative, 0.01 floor) pending repeat-run verification — the
native node (`gads_causal_estimate_ate`) is internally seeded, so these should tighten
to exact once a second run confirms, same policy as amlb_*_v1.
