# learning_mindset_achievement_v1 — provenance & tolerance rationale

**Established:** 2026-08-21. Sourced per approach_docs/026 (golden hypothesis-investigation
tasks). Unlike the D3-directed `qrdata_*` benchmarks already in this repository, this spec is
**D0 (undirected)**: `disable_recipes: true`, no method named anywhere in the objective, no
`recipe_id`. It measures the free-drafted lane, not a pinned recipe's realization.

**Source:** QRData learning_mindset.csv (Causal Inference for the Brave and True, ch.11), gold estimate. License: QRData is **CC BY-NC 4.0** (non-commercial) — retained for
research/internal use; do not redistribute commercially.

## Ground truth

The external anchor is QRData's own gold answer, established by the benchmark's authors from
the cited textbook/paper — not computed by us. **No GADS reference run has been made yet.**
The `tol` in `expected.json` is a provisional placeholder band, not derived from observed
run-to-run variance (see approach_docs/024's lesson on the same mistake for the Router: an
unvalidated tolerance is a claim about precision we don't have).

## Why `methodology` is empty and `recipe_id` is null

This is deliberate, not an oversight: a D0/drafted-lane spec has no fixed DAG or fixed code
pattern to check against — that's the entire point of the D0 rung (approach_docs/013). What
"methodologically appropriate" means here is closer to BLADE's decision-space grading (a
distribution of defensible choices) than to a `required_patterns` regex check, and that
scorer does not exist yet (flagged as open work in approach_docs/026 §5).

## Next step

Launch a reference run (any mode) against this spec, record the result here, and tighten (or
justify) the tolerance — the same discipline `research/benchmarks/README.md` already requires
for every other benchmark in this repo. Until that happens, treat this benchmark as
**sourced but not yet validated**.

## Reference run (2026-08-21, cloud, project `7e6dacef`) — verified cloud model throughout

Drafted lane fit propensity scores (logistic regression), then estimated the treatment
effect two independent ways: **AIPW/Doubly-Robust (ATE = 0.3925)** and **LinearDML
(ATE = 0.3955)** — both matching QRData's gold (0.39) almost exactly, and agreeing with
each other far more tightly than either matches "exactly 0.39," which is itself a good
sign (two different estimators converging is stronger evidence than one estimator
hitting a rounded target). Ran placebo refutation (effect ≈ 0.0064, correctly near
zero) and a data-subset refuter (effect = 0.3956, correctly stable) — full robustness
checks nobody asked for, plus a CATE/heterogeneity breakdown by baseline expectation and
school mindset. This is the strongest result in the batch: right estimate, right
uncertainty quantification, right robustness checks, entirely unprompted.

Only one reference run so far.

## Reference run — LOCAL mode (2026-08-22, project `5bd140fb`) — estimate produced, but under-adjusted

Produced an ATE (0.3347) via `gads_causal_estimate_ate`, but with only `success_expect`
named as a confounder — the dataset's ~9 available covariates include several
school-level variables (`school_mindset`, `school_achievement`, `school_urbanicity`,
`school_poverty`, `school_size`) and student demographics (`ethnicity`, `gender`,
`frst_in_family`) that the source study's own design treats as important adjustment
variables (the study is explicitly multi-level: student-level treatment nested in
schools with their own characteristics). Local used one of nine.

Result: 0.3347, noticeably below both cloud's AIPW/LinearDML estimates (0.3925/0.3955)
and QRData's gold (0.39) — not wildly wrong, but a visibly less-adjusted answer,
consistent with a smaller, less complete confounder set than cloud used unprompted.
A separate robustness/sensitivity step also failed (index out of bounds, size 0) and
was never recovered — the run reached a real estimate but delivered none of the
uncertainty/robustness apparatus cloud built for the same spec.

Not scored against a tight tolerance — recorded as a genuine but partial-capability
result: right ballpark, incomplete adjustment set, no working robustness checks.
