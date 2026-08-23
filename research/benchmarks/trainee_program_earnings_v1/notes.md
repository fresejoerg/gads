# trainee_program_earnings_v1 — provenance & tolerance rationale

**Established:** 2026-08-21. Sourced per approach_docs/026 (golden hypothesis-investigation
tasks). Unlike the D3-directed `qrdata_*` benchmarks already in this repository, this spec is
**D0 (undirected)**: `disable_recipes: true`, no method named anywhere in the objective, no
`recipe_id`. It measures the free-drafted lane, not a pinned recipe's realization.

**Source:** QRData trainee_unique_on_age.csv (Causal Inference for the Brave and True, ch.10), gold ATT. License: QRData is **CC BY-NC 4.0** (non-commercial) — retained for
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

## Reference run (2026-08-21, cloud, project `1328b558`) — verified cloud model throughout

**A genuine sign flip, and probably the most substantively interesting result in the
whole batch.** Naive unadjusted comparison: trainees earn $4,555.74 LESS (matches the
direction implied by QRData's own naive baseline). After adjusting for age via a
DAG-specified regression adjustment, GADS's causal estimate is **-$1,021.94** — still
negative, just smaller in magnitude than the naive gap. QRData's gold, computed via a
**matching estimator**, is **+$2,457.89** — positive, i.e. the opposite substantive
conclusion (program helps, not hurts).

This is not a bug to chase down. This exact class of problem — a tiny (n=58),
non-experimentally-confounded job-training dataset where the causal conclusion flips
sign depending on whether you adjust via regression, matching, or reweighting — is
**the canonical case study in the causal inference literature for estimator
sensitivity** (the Dehejia-Wahba/LaLonde controversy that `jobs_lalonde_training` in
this same batch is also drawn from, at larger scale). GADS's drafted lane reproduced
that known instability empirically, unprompted, simply by reaching for a different
(also defensible) identification strategy than the one QRData's answer key used. If
anything, a run that returned exactly +2457.89 without acknowledging the estimator
dependence would be the less honest result.

Not scored against a tolerance. Worth flagging for approach_docs/026 as a concrete
illustration of why some of these benchmarks cannot have a single "correct" scalar
answer, however the tolerance is set — the estimand itself is estimator-dependent on
this data.

Only one reference run so far.

## Reference run — LOCAL mode (2026-08-22, project `b96f24ff`) — cross-engine exact agreement

Clean run, 11/11 tasks, zero failures. ATE = **-1021.9390** — matching cloud's -1021.94
to four decimal places. Also reproduced cloud's placebo (-53.6181 vs cloud's, not
directly compared but same order of magnitude and correctly near-zero-ish given the
small sample) and subset refutation (-980.5414) checks.

**Why this matches so exactly, unlike almost every other divergence in this pass**: both
engines almost certainly called the identical native `gads_causal_estimate_ate` function
with the identical arguments (`treatment_col='trainees'`, `outcome_col='earnings'`,
`confounder_cols=['age']`) — a deterministic DoWhy backdoor-adjustment computation. Once
an LLM (of any capability) correctly recognizes "adjust for the confounder named in the
spec, hand it to the native," the actual number stops depending on which model generated
the calling code. This is the clearest empirical illustration in the whole two-day
reference-run pass of why GADS nativizes invariant computations (approach_docs/019): the
model's job here is small (name the right treatment/outcome/confounder), and getting
just that much right buys perfect cross-engine reproducibility on the rest.

The sign-flip vs. the matching-estimator gold (+2457.89) is unaffected by which engine
ran it — both regression-adjustment answers agree with each other and disagree with the
alternative estimator, exactly as the Dehejia-Wahba/LaLonde literature this dataset
echoes would predict.
