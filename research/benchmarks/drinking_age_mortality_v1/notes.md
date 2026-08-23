# drinking_age_mortality_v1 — provenance & tolerance rationale

**Established:** 2026-08-21. Sourced per approach_docs/026 (golden hypothesis-investigation
tasks). Unlike the D3-directed `qrdata_*` benchmarks already in this repository, this spec is
**D0 (undirected)**: `disable_recipes: true`, no method named anywhere in the objective, no
`recipe_id`. It measures the free-drafted lane, not a pinned recipe's realization.

**Source:** QRData drinking.csv (Causal Inference for the Brave and True, ch.16), gold estimate. License: QRData is **CC BY-NC 4.0** (non-commercial) — retained for
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

## Reference run (2026-08-21, cloud, project `641e9649`) — verified cloud model throughout

Drafted lane fit a sharp RDD (OLS with robust SE) and found **+7.66** (p=1.78e-09) — the
right DIRECTION (turning 21 increases mortality) but roughly **77x the gold's magnitude**
(0.10). Two likely, non-exclusive causes, neither verified by re-deriving the regression:

1. **Bandwidth**: the spec deliberately said "no older than 22, no younger than 20" in
   the QRData original question (a narrow local-linear RDD), but this run's report
   describes using "51 narrow age-bin records" without confirming it excluded bins
   outside 20-22 — the source file itself spans a much wider age range. A global/wider
   bandwidth RDD is a well-known source of larger, more biased discontinuity estimates
   than a narrow local one; this spec's objective (correctly, per the D0 phrasing rule)
   never told the model to restrict the bandwidth, so this is the kind of framing-driven
   divergence approach_docs/026 anticipated, not obviously a model error.
2. **Units**: the run's own report repeatedly states "deaths per 100,000," but the
   source column (`all`) is documented upstream as "deaths per 10,000 population" — a
   labeling inconsistency in the generated narrative. Even correcting for a 10x unit
   slip (7.66 → 0.766 per 10,000) the estimate is still ~7.7x the gold, so units alone
   don't explain the gap; bandwidth is the more likely primary driver.

Not scored against a tolerance — recorded as an open, unreconciled magnitude divergence
with a plausible but unverified explanation, per the same honesty standard as
`minwage_employment_v1` and `prop99_cigarette_sales_v1`.

Only one reference run so far.

## Reference run — LOCAL mode (2026-08-22, project `4e9e2cec`) — no estimate reached

Step 1 (data load) hit a syntax error, recovered on retry. The causal-estimation step
then failed with "infinite or NaN values in the exogenous variables" fitting the RDD
model — plausibly related to this dataset's unusual shape: `drinking.csv` ships several
pre-computed `*fitted` columns (`allfitted`, `internalfitted`, `alcoholfitted`, etc.)
alongside the raw outcomes, and a naive regression specification that pulls in the wrong
columns or divides by a running-variable transform near the exact age-21 cutoff could
plausibly produce non-finite values. Only exploratory age-distribution plots were
produced; no ATE, no comparable number to cloud's 7.66.

No GADS-canonical local value exists for this benchmark.
