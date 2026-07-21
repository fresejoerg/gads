# qrdata_ihdp_1_v1 — provenance & tolerance rationale

**Established:** 2026-07-14, cloud mode, recipe `causal_effect.observational.dowhy`
(re-layered v2.1.0 — objective-named roles authoritative, post-treatment exclusions honored).
**Source:** QRData benchmark (github.com/xxxiaol/QRData), adapted as a GADS spec: the
question is unchanged; roles are named in the objective per the recipe's contract.

## External anchor (the point of this benchmark family)

QRData's gold answer: **ATE = 4.05** (IHDP replicate 1, QRData gold answer). Unlike the AMLB benchmarks, the
"right answer" here is externally defined, not self-canonicalized — the reference run's
ATE of **3.8521** is graded against its own reproducibility, and the distance to the
gold anchors methodological appropriateness.

## Reference run

| Mode | Project | ate | placebo (~0) | subset (~ate) |
|---|---|---|---|---|
| cloud | `7d07a611` | 3.852051 | -0.024396 | 3.852237 |
| local | `3fbfec1a` | 3.899352 | -0.023568 | 3.897435 |

## Tolerance rationale

Provisional tolerances (2% relative, 0.01 floor) pending repeat-run verification — the
native node (`gads_causal_estimate_ate`) is internally seeded, so these should tighten
to exact once a second run confirms, same policy as amlb_*_v1.

**Cross-engine verification (2026-07-15, local run `3fbfec1a`, 16/16):** unlike ihdp_0
(cross-engine agreement to ~1e-15), local differs from cloud by 0.047 (~1.2%, inside the
2% band; local is 0.15 closer to the 4.05 gold). Verified cause (workflow_execution.py of
both runs): cloud capped confounders to top-10 by |corr with y|; local passed **all 25**,
skipping the task's "at most 10" cap — the same local engine *did* apply the cap in its
ihdp_0 run, so this is within-engine variance at the D4 role-assignment task, not a
stable engine trait. The D5 estimate node is deterministic given the set (sandbox
reproduction on ihdp_0: top-10 → 3.8965142215032174, all-25 → 3.9286717508727147).
Do NOT tighten these tolerances to exact; cross-engine agreement here is bounded by
adjustment-set choice, not float noise. Note the scorer's methodology tier cannot see
this difference — regex checks pass either way; adjustment-set identity needs the
decision-matching scorer (Phase 2/BLADE).
