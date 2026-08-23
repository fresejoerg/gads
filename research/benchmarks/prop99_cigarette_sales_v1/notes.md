# prop99_cigarette_sales_v1 — provenance & tolerance rationale

**Established:** 2026-08-21. Sourced per approach_docs/026 (golden hypothesis-investigation
tasks). Unlike the D3-directed `qrdata_*` benchmarks already in this repository, this spec is
**D0 (undirected)**: `disable_recipes: true`, no method named anywhere in the objective, no
`recipe_id`. It measures the free-drafted lane, not a pinned recipe's realization.

**Source:** QRData smoking2.csv (Causal Inference for the Brave and True, ch.25), gold DiD-ATT. License: QRData is **CC BY-NC 4.0** (non-commercial) — retained for
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

## Reference run (2026-08-21, cloud, project `93b4cea9`) — verified cloud model throughout

Drafted lane independently reached for BOTH a Synthetic Control estimate (-41.71 packs
per capita/year) and a Two-Way Fixed Effects DiD regression (+7.68, p=0.0007) — neither
was named in the spec. This is the richest and most puzzling result in the batch:

- **Synthetic Control matches the gold's direction**: -41.71 vs. QRData's -27.35, same
  sign (Prop 99 reduced sales), different magnitude. Synthetic control is also the more
  methodologically credible approach for this exact dataset in the wider literature
  (Abadie, Diamond & Hainmueller 2010 is the canonical Prop 99 synthetic-control paper),
  so this run reaching for it unprompted is a good sign about the drafted lane's method
  selection, not just its arithmetic.
- **The TWFE regression flips sign** (+7.68) — the run's own narrative doesn't
  acknowledge this contradiction; it treats the Synthetic Control number as the
  substantive answer and never reconciles the two. Working hypothesis (not verified):
  TWFE's added state/year fixed effects and economic covariates estimate a materially
  different quantity than QRData's gold, which is explicitly restricted to `cigsale`,
  `california`, `after_treatment` only — the same class of divergence seen in
  `minwage_employment_v1` (richer specification, different effective estimand). An
  unverified alternative: a genuine sign/coding bug in the generated TWFE code (e.g. an
  inverted interaction term) — cannot rule this out without re-deriving the regression
  by hand, which hasn't been done.

**This is exactly the shape of finding a D0 benchmark should surface and a D3-pinned one
never would**: the drafted lane isn't graded against one fixed script, so it's free to
produce two internally inconsistent answers in the same run — which is itself
informative. A future pass should verify the TWFE sign directly (rerun `workflow_execution.py`'s
regression cell by hand) before trusting either explanation above.

Only one reference run so far.

## Reference run — LOCAL mode (2026-08-22, project `e06bf428`) — the batch's most important finding

**Confirmed via the report's own text and its own error message**: the run invoked the
native `gads_causal_estimate_ate` node with `after_treatment` as the treatment column
and `cigsale` as the outcome. `california` — the variable that makes this a
difference-in-differences design at all, rather than a plain before/after study — is
never used as the treatment, an interaction term, or even mentioned as part of the
identification strategy anywhere in the report. Step 7's own failure message confirms
the call signature required a `confounder_cols` argument that a later step omitted;
nothing in any step supplied `california` as a role of any kind.

**What this actually estimates**: since cigarette consumption was declining nationwide
across all 39 states over 1970–2000 (visible in cloud's own Figure 1 for this same
dataset — "prior to 1988 sales were already trending downward due to general public
health awareness"), an ATE of `after_treatment` on `cigsale` pooled across all states
mostly captures that secular trend, not anything specific to California or Proposition
99. The reported ATE (-6.3108) is of the right sign by coincidence (both the secular
trend and the true policy effect point the same direction) but is not a defensible
answer to "did Prop 99 reduce sales" — it would produce nearly the same number if
California had never passed the law at all.

**Why this is worse than a crash.** The report presents this with full confidence: a
chart, a placebo refutation check (0.0961, correctly near zero) and a subset refutation
check (-6.2787, correctly stable) — both of which *pass*, because they test whether the
chosen specification is robust to resampling and placebo treatments, not whether the
specification correctly isolates the causal contrast the question is actually about. A
reader has no signal anywhere in this report that the analysis silently answered a
different, easier question than the one asked. Every other local-mode failure in this
pass (billboard's halt, women's Plotly hallucination, snow's encoder crash,
jobs_lalonde's earlier bug) is visible — the pipeline stops and says so. This one
completes cleanly and looks correct.

**Implication for this benchmark's design and for GADS more broadly**: `required_metrics`-style
contracts and refutation checks validate internal consistency of a chosen model, not
whether the model choice itself is right. Catching this class of error needs something
closer to BLADE's decision-space grading (does the *identification strategy* match the
question) rather than a scalar-metric or robustness-check gate — exactly the gap
approach_docs/026 §6 already flags as unbuilt tooling.
