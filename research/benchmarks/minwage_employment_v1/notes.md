# minwage_employment_v1 — provenance & tolerance rationale

**Established:** 2026-08-21. Sourced per approach_docs/026 (golden hypothesis-investigation
tasks). Unlike the D3-directed `qrdata_*` benchmarks already in this repository, this spec is
**D0 (undirected)**: `disable_recipes: true`, no method named anywhere in the objective, no
`recipe_id`. It measures the free-drafted lane, not a pinned recipe's realization.

**Source:** QRData minwage.csv (Quantitative Social Science 2.5.3), gold DiD estimate. License: QRData is **CC BY-NC 4.0** (non-commercial) — retained for
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

## Reference run (2026-08-21, cloud, project `fc9d384a`) — verified cloud model throughout

(This spec's first two launch attempts, project ids `3abd7c1d` and earlier, are VOID: they
ran on `local_model` because a backend restart mid-session silently reverted `routing_mode`
to the `.env` default and this wasn't caught until after those runs had already used the
wrong engine. Not written up. See JOURNAL.md 2026-08-21 for the full incident.)

Drafted lane fit an OLS DiD regression (with and without restaurant-chain fixed effects)
and found a **positive, non-significant** point estimate: **+2.93 full-time employees**
without chain controls, +2.82 with (95% CI roughly ±3.4 either way) — i.e. no evidence
that the minimum-wage increase reduced full-time employment.

**This does not match the external gold in the way earlier specs did, and the mismatch is
informative rather than a simple error.** QRData's gold (-0.062) is defined on the
**proportion** of full-time employees (`full / (full + part)`); GADS's drafted plan instead
modeled the **level/count** of full-time employees. These are different outcome
definitions, not directly comparable — a level effect and a share effect can disagree in
sign even when neither analysis is wrong. Nothing here contradicts the gold; nothing here
confirms it either. This is exactly the kind of divergence approach_docs/026 predicted
would surface once real reference runs existed, and it should not be quietly reconciled
away — record it as an open finding: **does the drafted lane reliably pick the same
outcome operationalization as the source question intends, when the spec describes raw
columns rather than a named derived metric?** This spec named `fullBefore`/`fullAfter`/
`partBefore`/`partAfter` as separate columns, never "proportion full-time" — so the
ambiguity was already present in how the D0 spec was written, not manufactured by the
model. Worth revisiting whether some D0 specs should describe the *outcome quantity* the
question is actually asking about, without naming a method, rather than leaving even the
outcome definition to be inferred from raw columns.

## Reference run — LOCAL mode (2026-08-22, project `22bf3670`) — stuck on one line, three times

Loaded the data and built the treatment indicator cleanly (steps 1–2), then failed on
the single-line computation `delta_full = fullAfter - fullBefore` with a Python syntax
error (mismatched parentheses) at step 3, retried, failed again identically at step 7,
retried again, failed a third time at step 10. Never reached causal estimation — no ATE,
no comparable number to cloud's 2.93 or QRData's gold. The recurring identical error
across retries (rather than distinct errors that would indicate genuine self-correction)
is exactly the same-reason pattern the executor's adaptive retry loop is designed to
detect and stop early on — worth checking whether this run's retries were being
correctly recognized as repeats or were each scored as a "new" attempt.

No GADS-canonical local value exists for this benchmark.
