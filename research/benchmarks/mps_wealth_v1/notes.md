# mps_wealth_v1 — provenance & tolerance rationale

**Established:** 2026-08-21. Sourced per approach_docs/026 (golden hypothesis-investigation
tasks). Unlike the D3-directed `qrdata_*` benchmarks already in this repository, this spec is
**D0 (undirected)**: `disable_recipes: true`, no method named anywhere in the objective, no
`recipe_id`. It measures the free-drafted lane, not a pinned recipe's realization.

**Source:** QRData MPs.csv (Quantitative Social Science 4.3.4), gold estimate. License: QRData is **CC BY-NC 4.0** (non-commercial) — retained for
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

## Reference run (2026-08-21, cloud, project `104379eb`) — verified cloud model throughout

**Same identification strategy as the gold (sharp RDD on the vote-margin cutoff), a
scale mismatch rather than a method mismatch.** GADS's run fit a local-linear RDD
(bandwidth h=0.10) on `ln.net`, found +0.842 log points (~132% wealth increase, p=0.089),
and ran real falsification checks — pre-determined covariates (birth year, death year,
prior margin) shown smooth across the cutoff, confirming design validity — plus
robustness across bandwidths, polynomial order, and demographic controls. This is
methodologically the same approach the gold used.

The divergence is that QRData's gold (255051) is an **absolute wealth-level** change,
back-transformed from the same kind of log-discontinuity using an assumed baseline
wealth the original question's transformation specified; this run stayed in log points
and reported a percentage effect instead, never producing a level figure. Not a
disagreement about the causal story — both find a large, RDD-identified wealth premium
for winning — just a difference in which scale got reported, the same class of
divergence as `minwage_employment_v1`'s level-vs-proportion mismatch. Not scored against
a tolerance for this reason.

Only one reference run so far.

## Reference run — LOCAL mode (2026-08-22, project `7dcf695b`) — no estimate reached

Failed at a Contract Violation: step 7 required the `margin` column and it wasn't found
in the dataset schema at that point — most plausibly an earlier step transformed or
subset the dataframe and dropped it before this step needed it, a state-tracking bug
rather than a conceptual error. The treatment-indicator creation step (`margin > 0`)
also failed twice on plain syntax errors. Only an exploratory scatter chart (election
margin by region/win status) was produced; the causal estimation stage was never
reached, so there's nothing to compare against cloud's 0.842 log points.

No GADS-canonical local value exists for this benchmark.
