# women_reservation_water_v1 — provenance & tolerance rationale

**Established:** 2026-08-21. Sourced per approach_docs/026 (golden hypothesis-investigation
tasks). Unlike the D3-directed `qrdata_*` benchmarks already in this repository, this spec is
**D0 (undirected)**: `disable_recipes: true`, no method named anywhere in the objective, no
`recipe_id`. It measures the free-drafted lane, not a pinned recipe's realization.

**Source:** QRData women.csv (Quantitative Social Science 4.3.1), gold estimate. License: QRData is **CC BY-NC 4.0** (non-commercial) — retained for
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

## Reference run (2026-08-21, cloud, project `5cc60665`)

Drafted lane recognized the randomized-assignment structure unprompted (nothing in the
spec says "randomized") and estimated an ITT effect: **+9.2524** drinking-water
facilities (p=0.0666, GP-clustered SEs) — matching QRData's gold (9.25) essentially
exactly. It then went further than the source question asks, fitting a 2SLS model using
`reserved` as an instrument for actual `female` leadership to estimate a structural LATE
of **+23.99** (p<0.0001), plus a placebo check on `irrigation` (correctly near-null,
-0.37, p=0.73) — a more complete causal analysis than the benchmark's own gold answer
represents. This is exactly the shape approach_docs/026 predicted for the D0 lane: a
capable engine's free plan can exceed a fixed external answer, which a tight exact-match
scorer would never reward and a "does it match this one number" framing undersells.

Only one reference run so far.

## Reference run — LOCAL mode (2026-08-22, project `550c5641`) — no estimate reached

Failed on execution mechanics, not on identification logic. Five of nine tasks failed:
`df.sample()` called with an invalid keyword argument, then two Plotly calls to
functions that don't exist (`plotly.express.boxplot`, `plotly.express.table` — plotly's
box/table chart constructors have different names). Only the initial data-load step
succeeded; the causal estimate was never reached, so there's nothing to compare against
cloud's 9.2524 or the gold's 9.25.

**Sharp contrast with `jobs_lalonde_training` and `billboard_deposits` in this same
local-mode pass**, where local either succeeded (lalonde) or crashed at the planning
level (billboard). Here the plan itself was presumably reasonable (the cloud run on the
identical spec found the right ITT effect via a very similar OLS approach) but local's
CODE GENERATION hallucinated a nonexistent Plotly API twice in a row on retries — a
different, narrower failure signature than either of the other two local runs so far.
No GADS-canonical local value exists for this benchmark.
