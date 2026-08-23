# ak91_schooling_wages_v1 — provenance & tolerance rationale

**Established:** 2026-08-21. Sourced per approach_docs/026 (golden hypothesis-investigation
tasks). Unlike the D3-directed `qrdata_*` benchmarks already in this repository, this spec is
**D0 (undirected)**: `disable_recipes: true`, no method named anywhere in the objective, no
`recipe_id`. It measures the free-drafted lane, not a pinned recipe's realization.

**Source:** QRData ak91.csv (Causal Inference for the Brave and True, ch.8), gold IV estimate. License: QRData is **CC BY-NC 4.0** (non-commercial) — retained for
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

## Reference run (2026-08-21, cloud, project `8ed41bd2`) — the predicted trap fired

This spec's own provenance note above predicted exactly this outcome, and it happened:
the run **did** recognize the identification problem unprompted — it imported
`linearmodels.iv.IV2SLS` and attempted a proper 2SLS design without being told an
instrument existed. But the 2SLS implementation failed repeatedly (3 failed tasks) on
this large dataset (329,509 rows, SampleBudget correctly injected a 50K training-row
cap) across the full model escalation ladder — gemini-3.7-flash → kimi-k2.7-code →
kimi-k3 → gemini-3.1-pro-preview, ~18 minutes on this one step, syntax errors and
timeouts on `linearmodels`'s API — and was ultimately abandoned. The final report
explicitly states **"Full causal verification via Two-Stage Least Squares (2SLS)
remains unexecuted in this pipeline run"** and instead reports the naive OLS estimates:
7.09% (univariate) and 6.73% (with birth-year/state fixed effects) — both confounded by
ability bias, both far from the IV gold of 8.53%.

**This is not a failure of the benchmark design — it is the benchmark working exactly as
intended.** The interesting question was never "does the run hit 8.53%," it was "does
the run recognize this needs an instrument, and does it actually deliver one." The
answer here is a clean, informative split: **recognition yes, execution no.** A reader
who only skims the headline "6.73% return to schooling" without reading the caveat
sentence would walk away with the confounded, wrong-for-the-reason-this-dataset-exists
number. That is a real and important failure mode for report consumers, distinct from
either the model "not knowing" causal inference or a plain code bug — an honest report
that hedges its own conclusion in prose is arguably worse for a downstream reader than
a hard failure, since it looks complete.

Not scored against the IV gold's tolerance — not comparable, different estimand
entirely. Only one reference run so far; a retry might succeed on the 2SLS step given
this was largely a `linearmodels` API-usage difficulty rather than a conceptual one, but
per the one-retry policy none was launched.

## Reference run — LOCAL mode (2026-08-22, project `b797c4b8`) — instrument/treatment conflation

**A third, and arguably the most concerning, failure mode on this spec.** Cloud
recognized it needed 2SLS, tried, failed to execute, and shipped a caveated fallback.
Local never got as far as attempting IV at all: it built "a binary treatment indicator
... 1 if born in Q4 ... 0 otherwise" and used that flag DIRECTLY as the `treatment_col`
in a native ATE call against `log_wage` — years_of_schooling never appears as a
regression variable anywhere in the run. This substitutes the *instrument* for the
*treatment*, which is not IV, not OLS, not any recognized estimator — it's a category
error in the identification strategy itself.

The result, 0.3920, is presented as "individuals who completed one extra year of
schooling ... experienced a 0.3920 increase in their log wage" — a ~39% return to one
year of education, an order of magnitude beyond any credible estimate in this literature
(returns to schooling are conventionally 5-15%; even the QRData gold's IV estimate is
8.53%). **Nothing in the report flags this as unusual or caveats the identification
strategy** — contrast with cloud's explicit "2SLS remains unexecuted" sentence. A reader
gets a specific, confident, wrong number and no warning sign, the same failure shape as
`prop99_cigarette_sales`'s local misidentification but on a spec that was specifically
sourced to test whether a run would even recognize the need for an instrument at all.

Ordering the three outcomes on this one spec by how dangerous they are to a downstream
reader: cloud's caveated-but-confounded OLS (visible warning, wrong number) < local's
uncaveated instrument/treatment conflation (no warning, wrong number, plausible-sounding
narrative) — the QRData gold's actual IV estimate is the only one of the three that
correctly executes the design this dataset exists to teach.
